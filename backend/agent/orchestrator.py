import json
import asyncio
import logging
import os
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy.orm import Session
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from backend.config import config
from backend.models import Conversation, Message, Memory
from backend.rag.retrieval import retrieve_and_rerank
from backend.agent.tools import build_agent_tools

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_TEMPLATE = """You are an advanced, intelligent AI Assistant with access to autonomous tools, long-term memory, and retrieved knowledge context.

Key Guidelines:
1. When the user asks you to remember or store facts about themselves (preferences, name, habits, setup), call the `add_to_memory` tool.
2. When the user asks you to forget or delete facts from long-term memory, call the `remove_from_memory` tool.
3. When the user asks you to clear, reset, or delete the current conversation history, call the `clear_conversation_history` tool.
4. When answering questions requiring up-to-date real-world facts, current news, or live data, call the `tavily_search` tool.
5. Be concise, precise, and polite. If knowledge context or memory is provided below, utilize it faithfully.
"""

def extract_thinking_and_content(ai_msg: AIMessage) -> Tuple[str, Optional[str]]:
    """
    Extracts main response text cleanly and separates any reasoning/thinking tokens.
    Guarantees that raw thinking dictionaries or metadata never leak into the response content.
    """
    content = ""
    thoughts = None

    if isinstance(ai_msg.content, list):
        text_parts = []
        thought_parts = []
        for part in ai_msg.content:
            if isinstance(part, dict):
                p_type = str(part.get("type", "")).lower()
                if p_type in ["thinking", "thought"] or "thinking" in part or "thought" in part:
                    thought_val = part.get("thinking") or part.get("thought") or ""
                    thought_parts.append(str(thought_val))
                elif p_type == "text" or "text" in part:
                    text_parts.append(str(part.get("text", "")))
                elif "content" in part:
                    text_parts.append(str(part["content"]))
            elif isinstance(part, str):
                text_parts.append(part)

        content = "\n".join(text_parts).strip()
        if thought_parts:
            thoughts = "\n".join(thought_parts).strip()

    elif isinstance(ai_msg.content, str):
        raw = ai_msg.content
        import re
        match = re.match(r"^\s*\{['\"]type['\"]\s*:\s*['\"]thinking['\"].*?\}\s*", raw, re.DOTALL)
        if match:
            raw = raw[match.end():].strip()
        content = raw.strip()

    if not thoughts and hasattr(ai_msg, "additional_kwargs"):
        if "thoughts" in ai_msg.additional_kwargs:
            thoughts = str(ai_msg.additional_kwargs["thoughts"])

    return content, thoughts

def extract_token_from_chunk(chunk: Any) -> str:
    """
    Extracts purely user-facing text tokens from an incoming stream chunk.
    Filters out any thinking, signatures, or tool parameters.
    """
    if not chunk or not hasattr(chunk, "content") or not chunk.content:
        return ""
    if isinstance(chunk.content, list):
        parts = []
        for p in chunk.content:
            if isinstance(p, dict):
                p_type = str(p.get("type", "")).lower()
                if p_type == "text" or ("text" in p and p_type not in ["thinking", "thought"]):
                    text_val = p.get("text", "")
                    if text_val:
                        parts.append(str(text_val))
            elif isinstance(p, str):
                parts.append(p)
        return "".join(parts)
    elif isinstance(chunk.content, str):
        return chunk.content
    return ""

def build_llm_instance(thinking_level: str) -> Optional[ChatGoogleGenerativeAI]:
    """
    Initializes ChatGoogleGenerativeAI with configured model, temperature,
    and selectable thinking budget / level.
    """
    api_key = config.GOOGLE_API_KEY or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return None

    level_norm = thinking_level.lower() if thinking_level else "med"
    thinking_level_enum = config.THINKING_LEVEL_MAP.get(level_norm, "MEDIUM")

    try:
        llm = ChatGoogleGenerativeAI(
            model=config.GEMINI_MODEL,
            google_api_key=api_key,
            temperature=config.TEMPERATURE,
            thinking_config={
                "thinking_level": thinking_level_enum
            }
        )
        return llm
    except Exception as e:
        logger.warning(f"Failed to initialize with thinking_config: {e}. Falling back to standard init.")
        return ChatGoogleGenerativeAI(
            model=config.GEMINI_MODEL,
            google_api_key=api_key,
            temperature=config.TEMPERATURE
        )

def run_agent_orchestrator(
    db: Session,
    conversation_id: str,
    user_prompt: str,
    rag_enabled: bool,
    memory_enabled: bool,
    thinking_level: str,
    linked_message_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Core agent orchestration function:
    1. Rebuilds conversation context from PostgreSQL.
    2. Handles Linear mode vs Isolated mode with linked exchange context.
    3. Injects persistent memory facts if memory_enabled is True.
    4. Injects hybrid retrieved + reranked context if rag_enabled is True.
    5. Dispatches autonomous tools (memory, history clear, Tavily search).
    6. Persists user message and assistant reply to database.
    """
    conversation = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if not conversation:
        raise ValueError(f"Conversation {conversation_id} not found.")

    # 1. Save the incoming user message to PostgreSQL
    user_msg_record = Message(
        conversation_id=conversation_id,
        role="user",
        content=user_prompt,
        rag_enabled=rag_enabled,
        memory_enabled=memory_enabled,
        thinking_level=thinking_level,
        linked_message_id=linked_message_id
    )
    db.add(user_msg_record)
    db.commit()
    db.refresh(user_msg_record)

    # 2. Check for Google API key
    llm = build_llm_instance(thinking_level)
    if not llm:
        fallback_msg = (
            "**Notice:** `GOOGLE_API_KEY` is not configured in the `.env` file.\n\n"
            "To connect to Gemini, please add your Google AI Studio API key to the `.env` file in the root directory:\n"
            "```env\nGOOGLE_API_KEY=AIzaSy...\n```\n"
            "The system is running and all database, RAG, and UI functions are active."
        )
        asst_record = Message(
            conversation_id=conversation_id,
            role="assistant",
            content=fallback_msg,
            rag_enabled=rag_enabled,
            memory_enabled=memory_enabled,
            thinking_level=thinking_level
        )
        db.add(asst_record)
        db.commit()
        db.refresh(asst_record)
        return {
            "user_message": user_msg_record,
            "assistant_message": asst_record,
            "rag_sources": [],
            "tool_calls": [],
            "thoughts": None
        }

    # 3. Build System Prompt components
    system_sections = [SYSTEM_PROMPT_TEMPLATE]

    # Persistent Memory Injection
    if memory_enabled:
        memories = db.query(Memory).order_by(Memory.created_at).all()
        if memories:
            mem_lines = "\n".join([f"- {m.content}" for m in memories])
            system_sections.append(
                f"## Persistent User Memory:\nThe following facts are remembered about the user across conversations:\n{mem_lines}"
            )
        else:
            system_sections.append("## Persistent User Memory:\nNo stored facts in memory yet.")

    # RAG Retrieval & Injection
    rag_sources = []
    if rag_enabled:
        rag_sources = retrieve_and_rerank(user_prompt, top_k=config.TOP_K, top_n=config.TOP_N)
        if rag_sources:
            context_blocks = []
            for i, src in enumerate(rag_sources, 1):
                context_blocks.append(
                    f"[Document Snippet {i} | Source: {src.get('source', 'Unknown')} | Score: {src.get('score', 0):.2f}]\n{src['content']}"
                )
            joined_context = "\n\n".join(context_blocks)
            system_sections.append(
                f"## Retrieved Document Knowledge Context:\nUse the following verified context passages to answer:\n{joined_context}"
            )

    system_message = SystemMessage(content="\n\n".join(system_sections))

    # 4. Rebuild conversation messages from DB
    messages_to_send = [system_message]

    if conversation.mode == "linear":
        # Full prior history ordered by created_at (excluding the newly created user_msg_record)
        prior_records = (
            db.query(Message)
            .filter(Message.conversation_id == conversation_id, Message.id != user_msg_record.id)
            .order_by(Message.created_at)
            .all()
        )
        for rec in prior_records:
            if rec.role == "user":
                messages_to_send.append(HumanMessage(content=rec.content))
            elif rec.role == "assistant":
                messages_to_send.append(AIMessage(content=rec.content))

        # Add the current user prompt
        messages_to_send.append(HumanMessage(content=user_prompt))

    else:
        # Isolated mode: NO automatic history
        # If user explicitly linked a prior message exchange (hk, ak):
        if linked_message_id:
            linked_user = db.query(Message).filter(Message.id == linked_message_id).first()
            if linked_user:
                # Find the assistant response that followed that user message
                linked_asst = (
                    db.query(Message)
                    .filter(
                        Message.conversation_id == conversation_id,
                        Message.role == "assistant",
                        Message.created_at > linked_user.created_at
                    )
                    .order_by(Message.created_at)
                    .first()
                )
                linked_context_text = f"Linked Prior Question: {linked_user.content}"
                if linked_asst:
                    linked_context_text += f"\nLinked Prior Answer: {linked_asst.content}"

                messages_to_send.append(
                    SystemMessage(content=f"## Explicitly Linked Prior Exchange:\n{linked_context_text}")
                )

        # Append current user prompt
        messages_to_send.append(HumanMessage(content=user_prompt))

    # 5. Bind agent tools & execute agent loop
    tool_events: List[Dict[str, Any]] = []
    tools = build_agent_tools(db, conversation_id, tool_events)
    tools_by_name = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools)

    final_content = ""
    final_thoughts = None
    current_messages = list(messages_to_send)

    # Max 5 tool call iterations to prevent infinite loops
    for step in range(5):
        try:
            ai_response: AIMessage = llm_with_tools.invoke(current_messages)
        except Exception as e:
            logger.error(f"Error invoking LLM with tools: {e}")
            final_content = f"An error occurred while generating a response: {str(e)}"
            break

        extracted_text, extracted_thoughts = extract_thinking_and_content(ai_response)
        if extracted_thoughts:
            final_thoughts = extracted_thoughts

        # Check if the LLM decided to call tools
        if hasattr(ai_response, "tool_calls") and ai_response.tool_calls:
            current_messages.append(ai_response)
            for tc in ai_response.tool_calls:
                tool_name = tc["name"]
                tool_args = tc["args"]
                tool_id = tc["id"]

                if tool_name in tools_by_name:
                    try:
                        tool_output = tools_by_name[tool_name].invoke(tool_args)
                    except Exception as err:
                        tool_output = f"Tool execution error: {str(err)}"
                else:
                    tool_output = f"Unknown tool: {tool_name}"

                current_messages.append(
                    ToolMessage(
                        content=str(tool_output),
                        tool_call_id=tool_id,
                        name=tool_name
                    )
                )
        else:
            # Final response reached without tool calls
            final_content = extracted_text
            break
    else:
        # Loop finished without breaking
        if not final_content:
            final_content = extracted_text or "Task completed."

    # 6. Save the assistant response to PostgreSQL
    # (Note: if clear_conversation_history was invoked, the conversation messages were cleared,
    # and this new assistant reply will start the fresh conversation)
    asst_record = Message(
        conversation_id=conversation_id,
        role="assistant",
        content=final_content,
        rag_enabled=rag_enabled,
        memory_enabled=memory_enabled,
        thinking_level=thinking_level
    )
    db.add(asst_record)
    db.commit()
    db.refresh(asst_record)

    return {
        "user_message": user_msg_record,
        "assistant_message": asst_record,
        "rag_sources": rag_sources,
        "tool_calls": tool_events,
        "thoughts": final_thoughts
    }

async def stream_agent_orchestrator(
    db: Session,
    conversation_id: str,
    user_prompt: str,
    rag_enabled: bool,
    memory_enabled: bool,
    thinking_level: str,
    linked_message_id: Optional[str] = None
):
    """
    Streaming agent orchestration yielding Server-Sent Events (SSE):
    - data: {"type": "token", "token": "..."}
    - data: {"type": "tool_call", "tool": "...", "args": {...}, "result": "..."}
    - data: {"type": "done", "user_message": {...}, "assistant_message": {...}, ...}
    - data: {"type": "error", "error": "..."}
    """
    conversation = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if not conversation:
        yield f"data: {json.dumps({'type': 'error', 'error': f'Conversation {conversation_id} not found.'})}\n\n"
        return

    # 1. Save incoming user message to PostgreSQL
    user_msg_record = Message(
        conversation_id=conversation_id,
        role="user",
        content=user_prompt,
        rag_enabled=rag_enabled,
        memory_enabled=memory_enabled,
        thinking_level=thinking_level,
        linked_message_id=linked_message_id
    )
    db.add(user_msg_record)
    db.commit()
    db.refresh(user_msg_record)

    # 2. Check LLM availability
    llm = build_llm_instance(thinking_level)
    if not llm:
        fallback_msg = (
            "**Notice:** `GOOGLE_API_KEY` is not configured in the `.env` file.\n\n"
            "To connect to Gemini, please add your Google AI Studio API key to the `.env` file in the root directory:\n"
            "```env\nGOOGLE_API_KEY=AIzaSy...\n```"
        )
        asst_record = Message(
            conversation_id=conversation_id,
            role="assistant",
            content=fallback_msg,
            rag_enabled=rag_enabled,
            memory_enabled=memory_enabled,
            thinking_level=thinking_level
        )
        db.add(asst_record)
        db.commit()
        db.refresh(asst_record)

        yield f"data: {json.dumps({'type': 'token', 'token': fallback_msg})}\n\n"
        yield f"data: {json.dumps({'type': 'done', 'user_message': {'id': user_msg_record.id, 'conversation_id': user_msg_record.conversation_id, 'role': 'user', 'content': user_prompt, 'created_at': user_msg_record.created_at.isoformat(), 'rag_enabled': rag_enabled, 'memory_enabled': memory_enabled, 'thinking_level': thinking_level, 'linked_message_id': linked_message_id}, 'assistant_message': {'id': asst_record.id, 'conversation_id': asst_record.conversation_id, 'role': 'assistant', 'content': fallback_msg, 'created_at': asst_record.created_at.isoformat(), 'rag_enabled': rag_enabled, 'memory_enabled': memory_enabled, 'thinking_level': thinking_level, 'linked_message_id': None}, 'rag_sources': [], 'tool_calls': []})}\n\n"
        return

    # 3. Build System Prompt components
    system_sections = [SYSTEM_PROMPT_TEMPLATE]

    if memory_enabled:
        memories = db.query(Memory).order_by(Memory.created_at).all()
        if memories:
            mem_lines = "\n".join([f"- {m.content}" for m in memories])
            system_sections.append(f"## Persistent User Memory:\nThe following facts are remembered about the user across conversations:\n{mem_lines}")
        else:
            system_sections.append("## Persistent User Memory:\nNo stored facts in memory yet.")

    rag_sources = []
    if rag_enabled:
        rag_sources = retrieve_and_rerank(user_prompt, top_k=config.TOP_K, top_n=config.TOP_N)
        if rag_sources:
            context_blocks = []
            for i, src in enumerate(rag_sources, 1):
                context_blocks.append(
                    f"[Document Snippet {i} | Source: {src.get('source', 'Unknown')} | Score: {src.get('score', 0):.2f}]\n{src['content']}"
                )
            joined_context = "\n\n".join(context_blocks)
            system_sections.append(f"## Retrieved Document Knowledge Context:\nUse the following verified context passages to answer:\n{joined_context}")

    system_message = SystemMessage(content="\n\n".join(system_sections))

    # 4. Rebuild conversation messages
    messages_to_send = [system_message]
    if conversation.mode == "linear":
        prior_records = (
            db.query(Message)
            .filter(Message.conversation_id == conversation_id, Message.id != user_msg_record.id)
            .order_by(Message.created_at)
            .all()
        )
        for rec in prior_records:
            if rec.role == "user":
                messages_to_send.append(HumanMessage(content=rec.content))
            elif rec.role == "assistant":
                messages_to_send.append(AIMessage(content=rec.content))
        messages_to_send.append(HumanMessage(content=user_prompt))
    else:
        if linked_message_id:
            linked_user = db.query(Message).filter(Message.id == linked_message_id).first()
            if linked_user:
                linked_asst = (
                    db.query(Message)
                    .filter(
                        Message.conversation_id == conversation_id,
                        Message.role == "assistant",
                        Message.created_at > linked_user.created_at
                    )
                    .order_by(Message.created_at)
                    .first()
                )
                linked_context_text = f"Linked Prior Question: {linked_user.content}"
                if linked_asst:
                    linked_context_text += f"\nLinked Prior Answer: {linked_asst.content}"
                messages_to_send.append(SystemMessage(content=f"## Explicitly Linked Prior Exchange:\n{linked_context_text}"))
        messages_to_send.append(HumanMessage(content=user_prompt))

    # 5. Bind agent tools & execute stream loop
    tool_events: List[Dict[str, Any]] = []
    tools = build_agent_tools(db, conversation_id, tool_events)
    tools_by_name = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools)

    current_messages = list(messages_to_send)
    final_content_parts = []

    for step in range(5):
        tool_calls_detected = []
        step_tokens = []

        try:
            for chunk in llm_with_tools.stream(current_messages):
                if hasattr(chunk, "tool_calls") and chunk.tool_calls:
                    tool_calls_detected.extend(chunk.tool_calls)

                token = extract_token_from_chunk(chunk)
                if token:
                    step_tokens.append(token)
                    yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"
                await asyncio.sleep(0)
        except Exception as e:
            logger.error(f"Error streaming response chunk: {e}")
            err_msg = f"\n[Error generating response: {str(e)}]"
            yield f"data: {json.dumps({'type': 'token', 'token': err_msg})}\n\n"
            step_tokens.append(err_msg)
            break

        if tool_calls_detected:
            # Append AIMessage with tool calls
            current_messages.append(AIMessage(content="".join(step_tokens), tool_calls=tool_calls_detected))
            for tc in tool_calls_detected:
                t_name = tc.get("name")
                t_args = tc.get("args", {})
                t_id = tc.get("id")

                if t_name in tools_by_name:
                    try:
                        t_output = tools_by_name[t_name].invoke(t_args)
                    except Exception as err:
                        t_output = f"Tool execution error: {str(err)}"
                else:
                    t_output = f"Unknown tool: {t_name}"

                tool_events.append({"tool": t_name, "args": t_args, "result": str(t_output)})
                yield f"data: {json.dumps({'type': 'tool_call', 'tool': t_name, 'args': t_args, 'result': str(t_output)})}\n\n"
                current_messages.append(ToolMessage(content=str(t_output), tool_call_id=t_id, name=t_name))
        else:
            final_content_parts.extend(step_tokens)
            break
    else:
        if not final_content_parts and step_tokens:
            final_content_parts.extend(step_tokens)

    final_content = "".join(final_content_parts).strip()
    if not final_content:
        final_content = "Response completed."

    # 6. Save assistant reply to PostgreSQL
    asst_record = Message(
        conversation_id=conversation_id,
        role="assistant",
        content=final_content,
        rag_enabled=rag_enabled,
        memory_enabled=memory_enabled,
        thinking_level=thinking_level
    )
    db.add(asst_record)
    db.commit()
    db.refresh(asst_record)

    # 7. Yield completion event
    yield f"data: {json.dumps({
        'type': 'done',
        'user_message': {
            'id': user_msg_record.id,
            'conversation_id': user_msg_record.conversation_id,
            'role': user_msg_record.role,
            'content': user_msg_record.content,
            'created_at': user_msg_record.created_at.isoformat(),
            'rag_enabled': user_msg_record.rag_enabled,
            'memory_enabled': user_msg_record.memory_enabled,
            'thinking_level': user_msg_record.thinking_level,
            'linked_message_id': user_msg_record.linked_message_id
        },
        'assistant_message': {
            'id': asst_record.id,
            'conversation_id': asst_record.conversation_id,
            'role': asst_record.role,
            'content': asst_record.content,
            'created_at': asst_record.created_at.isoformat(),
            'rag_enabled': asst_record.rag_enabled,
            'memory_enabled': asst_record.memory_enabled,
            'thinking_level': asst_record.thinking_level,
            'linked_message_id': asst_record.linked_message_id
        },
        'rag_sources': rag_sources,
        'tool_calls': tool_events
    })}\n\n"
