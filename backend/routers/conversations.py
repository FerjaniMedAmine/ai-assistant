from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import func
from backend.database import get_db
from backend.models import Conversation, Message
from backend.schemas import ConversationCreate, ConversationResponse

router = APIRouter(prefix="/api/conversations", tags=["conversations"])

@router.get("", response_model=List[ConversationResponse])
def list_conversations(db: Session = Depends(get_db)):
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
def create_conversation(payload: ConversationCreate, db: Session = Depends(get_db)):
    """
    Create a new conversation thread with specified mode ('linear' or 'isolated').
    """
    conv = Conversation(
        title=payload.title or "New Conversation",
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
def get_conversation(conversation_id: str, db: Session = Depends(get_db)):
    """
    Retrieve conversation metadata and all messages ordered chronologically.
    """
    conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
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
def delete_conversation(conversation_id: str, db: Session = Depends(get_db)):
    """
    Hard delete a conversation and all its messages permanently from PostgreSQL,
    and remove any associated vectors from Qdrant.
    """
    conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Hard delete from PostgreSQL (cascades to messages)
    db.delete(conv)
    db.commit()

    # Hard delete any associated vector embeddings from Qdrant
    try:
        from backend.rag.vector_store import QdrantVectorStore
        QdrantVectorStore.get_instance().delete_by_conversation(conversation_id)
    except Exception as e:
        pass

    return None
