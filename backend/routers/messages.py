import logging
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from backend.database import get_db
from backend.models import Conversation, Message, User
from backend.routers.auth import get_current_user
from backend.schemas import MessageCreate, MessageEdit, AgentReply, MessageResponse
from backend.agent.orchestrator import run_agent_orchestrator, stream_agent_orchestrator
from backend.rag.vector_store import QdrantVectorStore

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["messages"])


def remove_indexed_replies(assistant_ids: list[str]) -> None:
    if not assistant_ids:
        return
    try:
        store = QdrantVectorStore.get_instance()
        for assistant_id in assistant_ids:
            store.delete_exchange(assistant_id)
    except Exception:
        logger.exception("Failed to remove indexed chat replies from Qdrant")

@router.post("/conversations/{conversation_id}/messages/stream")
def send_message_stream(
    conversation_id: str,
    payload: MessageCreate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """
    Stream tokens in real time via Server-Sent Events (SSE).
    """
    conv = db.query(Conversation).filter(Conversation.id == conversation_id, Conversation.user_id == user.id).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # If first message, auto-title conversation from first words
    if conv.title == "New Conversation":
        snippet = payload.content.strip().replace("\n", " ")
        if len(snippet) > 35:
            conv.title = snippet[:35] + "..."
        else:
            conv.title = snippet or "New Conversation"
        db.commit()

    return StreamingResponse(
        stream_agent_orchestrator(
            db=db,
            conversation_id=conversation_id,
            user_prompt=payload.content,
            rag_enabled=payload.rag_enabled,
            memory_enabled=payload.memory_enabled,
            thinking_level=payload.thinking_level,
            linked_message_id=payload.linked_message_id,
            user_id=user.id,
        ),
        media_type="text/event-stream"
    )

@router.post("/conversations/{conversation_id}/messages", response_model=AgentReply)
def send_message(
    conversation_id: str,
    payload: MessageCreate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """
    Post a user prompt to a conversation, rebuild context from database,
    orchestrate LLM deliberation and autonomous tool execution,
    and persist conversation messages.
    """
    conv = db.query(Conversation).filter(Conversation.id == conversation_id, Conversation.user_id == user.id).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # If first message, auto-title conversation from first words
    if conv.title == "New Conversation":
        snippet = payload.content.strip().replace("\n", " ")
        if len(snippet) > 35:
            conv.title = snippet[:35] + "..."
        else:
            conv.title = snippet or "New Conversation"
        db.commit()

    reply_data = run_agent_orchestrator(
        db=db,
        conversation_id=conversation_id,
        user_prompt=payload.content,
        rag_enabled=payload.rag_enabled,
        memory_enabled=payload.memory_enabled,
        thinking_level=payload.thinking_level,
        linked_message_id=payload.linked_message_id,
        user_id=user.id,
    )

    return reply_data

@router.put("/messages/{message_id}", response_model=AgentReply)
def edit_message(
    message_id: str,
    payload: MessageEdit,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """
    Edit a message's content in the database:
    1. Update the message content in PostgreSQL.
    2. Hard-delete every message in the conversation that came after it.
    3. Treat the edited message as the new end of the conversation.
    4. Re-run LLM orchestration to generate the new AI response.
    """
    target_msg = db.query(Message).join(Conversation).filter(Message.id == message_id, Conversation.user_id == user.id).first()
    if not target_msg:
        raise HTTPException(status_code=404, detail="Message not found")

    conv_id = target_msg.conversation_id
    msg_created_at = target_msg.created_at

    # Hard-delete all subsequent messages in this conversation
    subsequent_messages = (
        db.query(Message)
        .filter(Message.conversation_id == conv_id, Message.created_at > msg_created_at)
        .all()
    )
    assistant_ids = [msg.id for msg in subsequent_messages if msg.role == "assistant"]
    if target_msg.role == "assistant":
        assistant_ids.append(target_msg.id)
    remove_indexed_replies(assistant_ids)
    subsequent_deleted = (
        db.query(Message)
        .filter(Message.conversation_id == conv_id, Message.created_at > msg_created_at)
        .delete()
    )
    logger.info(f"Edited message {message_id}. Hard deleted {subsequent_deleted} subsequent messages.")

    # Update content
    target_msg.content = payload.content
    db.commit()
    db.refresh(target_msg)

    # If edited message was a user prompt, regenerate the assistant reply
    if target_msg.role == "user":
        # Delete the target message record temporarily or pass its parameters to orchestrator
        # To avoid duplicating target_msg, delete target_msg and re-run orchestrator with its exact parameters
        rag_enabled = target_msg.rag_enabled
        memory_enabled = target_msg.memory_enabled
        thinking_level = target_msg.thinking_level
        linked_id = target_msg.linked_message_id
        new_content = payload.content

        db.delete(target_msg)
        db.commit()

        reply_data = run_agent_orchestrator(
            db=db,
            conversation_id=conv_id,
            user_prompt=new_content,
            rag_enabled=rag_enabled,
            memory_enabled=memory_enabled,
            thinking_level=thinking_level,
            linked_message_id=linked_id,
            user_id=user.id,
        )
        return reply_data

    else:
        # If an assistant message was edited directly
        prior_user = (
            db.query(Message)
            .filter(Message.conversation_id == conv_id, Message.role == "user", Message.created_at < msg_created_at)
            .order_by(Message.created_at.desc())
            .first()
        )
        if prior_user:
            from backend.agent.orchestrator import index_exchange
            index_exchange(user.id, prior_user, target_msg)
        return {
            "user_message": target_msg,
            "assistant_message": target_msg,
            "rag_sources": [],
            "tool_calls": [],
            "thoughts": None
        }

@router.delete("/messages/{message_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_message(message_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    """
    Hard delete an individual message permanently from PostgreSQL.
    """
    target_msg = db.query(Message).join(Conversation).filter(Message.id == message_id, Conversation.user_id == user.id).first()
    if not target_msg:
        raise HTTPException(status_code=404, detail="Message not found")

    assistant_ids = [target_msg.id] if target_msg.role == "assistant" else []
    if target_msg.role == "user":
        next_assistant = (
            db.query(Message)
            .filter(Message.conversation_id == target_msg.conversation_id, Message.role == "assistant", Message.created_at > target_msg.created_at)
            .order_by(Message.created_at)
            .first()
        )
        if next_assistant:
            assistant_ids.append(next_assistant.id)
    remove_indexed_replies(assistant_ids)
    db.delete(target_msg)
    db.commit()
    return None
