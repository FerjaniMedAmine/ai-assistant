from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import func
from backend.database import get_db
from backend.models import Conversation, Message, User
from backend.routers.auth import get_current_user
from backend.schemas import ConversationCreate, ConversationResponse

router = APIRouter(prefix="/api/conversations", tags=["conversations"])

@router.get("", response_model=List[ConversationResponse])
def list_conversations(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    """
    List all conversations that have messages (ChatGPT/Claude pattern),
    ordered by creation date descending, including their message counts.
    Conversations without messages are not displayed in the sidebar.
    """
    conversations = (
        db.query(
            Conversation,
            func.count(Message.id).label("message_count")
        )
        .join(Message, Message.conversation_id == Conversation.id)
        .filter(Conversation.user_id == user.id)
        .group_by(Conversation.id)
        .order_by(Conversation.created_at.desc())
        .all()
    )

    results = []
    for conv, count in conversations:
        results.append(
            ConversationResponse(
                id=conv.id,
                title=conv.title,
                mode=conv.mode,
                created_at=conv.created_at,
                message_count=count
            )
        )
    return results

@router.post("", response_model=ConversationResponse, status_code=status.HTTP_201_CREATED)
def create_conversation(payload: ConversationCreate, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    """
    Create a new conversation thread with specified mode ('linear' or 'isolated').
    """
    conv = Conversation(
        title=payload.title or "New Conversation",
        user_id=user.id,
        mode=payload.mode
    )
    db.add(conv)
    db.commit()
    db.refresh(conv)

    return ConversationResponse(
        id=conv.id,
        title=conv.title,
        mode=conv.mode,
        created_at=conv.created_at,
        message_count=0
    )

@router.get("/{conversation_id}")
def get_conversation(conversation_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    """
    Retrieve conversation metadata and all messages ordered chronologically.
    """
    conv = db.query(Conversation).filter(Conversation.id == conversation_id, Conversation.user_id == user.id).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    messages = (
        db.query(Message)
        .filter(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.asc())
        .all()
    )

    return {
        "id": conv.id,
        "title": conv.title,
        "mode": conv.mode,
        "created_at": conv.created_at,
        "messages": messages
    }

@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_conversation(conversation_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    """
    Hard delete a conversation and all its messages permanently from PostgreSQL,
    and remove any associated vectors from Qdrant.
    """
    conv = db.query(Conversation).filter(Conversation.id == conversation_id, Conversation.user_id == user.id).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Remove indexed exchanges while the conversation still exists in PostgreSQL.
    try:
        from backend.rag.vector_store import QdrantVectorStore
        QdrantVectorStore.get_instance().delete_by_conversation(conversation_id, user.id)
    except Exception:
        import logging
        logging.getLogger(__name__).exception("Failed to remove Qdrant points for conversation %s", conversation_id)

    db.delete(conv)
    db.commit()

    return None
