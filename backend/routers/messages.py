import logging
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from backend.database import get_db
from backend.models import Conversation, Message
from backend.schemas import MessageCreate, MessageEdit, AgentReply, MessageResponse
from backend.agent.orchestrator import run_agent_orchestrator, stream_agent_orchestrator

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["messages"])

@router.post("/conversations/{conversation_id}/messages/stream")
def send_message_stream(
    conversation_id: str,
    payload: MessageCreate,
    db: Session = Depends(get_db)
):
    """
    Stream tokens in real time via Server-Sent Events (SSE).
    """
    conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
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
            linked_message_id=payload.linked_message_id
        ),
        media_type="text/event-stream"
    )

@router.post("/conversations/{conversation_id}/messages", response_model=AgentReply)
def send_message(
    conversation_id: str,
    payload: MessageCreate,
    db: Session = Depends(get_db)
):
    """
    Post a user prompt to a conversation, rebuild context from database,
    orchestrate LLM deliberation and autonomous tool execution,
    and persist conversation messages.
    """
    conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
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
        linked_message_id=payload.linked_message_id
    )

    return reply_data

@router.put("/messages/{message_id}", response_model=AgentReply)
def edit_message(
    message_id: str,
    payload: MessageEdit,
    db: Session = Depends(get_db)
):
    """
    Edit a message's content in the database:
    1. Update the message content in PostgreSQL.
    2. Hard-delete every message in the conversation that came after it.
    3. Treat the edited message as the new end of the conversation.
    4. Re-run LLM orchestration to generate the new AI response.
    """
    target_msg = db.query(Message).filter(Message.id == message_id).first()
    if not target_msg:
        raise HTTPException(status_code=404, detail="Message not found")

    conv_id = target_msg.conversation_id
    msg_created_at = target_msg.created_at

    # Hard-delete all subsequent messages in this conversation
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
            linked_message_id=linked_id
        )
        return reply_data

    else:
        # If an assistant message was edited directly
        return {
            "user_message": target_msg,
            "assistant_message": target_msg,
            "rag_sources": [],
            "tool_calls": [],
            "thoughts": None
        }

@router.delete("/messages/{message_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_message(message_id: str, db: Session = Depends(get_db)):
    """
    Hard delete an individual message permanently from PostgreSQL.
    """
    target_msg = db.query(Message).filter(Message.id == message_id).first()
    if not target_msg:
        raise HTTPException(status_code=404, detail="Message not found")

    db.delete(target_msg)
    db.commit()
    return None
