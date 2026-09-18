import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Text, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from backend.database import Base

def generate_uuid() -> str:
    return str(uuid.uuid4())

def utc_now() -> datetime:
    return datetime.now(timezone.utc)

class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(String(36), primary_key=True, default=generate_uuid, index=True)
    title = Column(String(255), nullable=False)
    mode = Column(String(20), nullable=False, default="linear")  # 'linear' or 'isolated'
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)

    # Relationships
    messages = relationship(
        "Message",
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.created_at"
    )

class Message(Base):
    __tablename__ = "messages"

    id = Column(String(36), primary_key=True, default=generate_uuid, index=True)
    conversation_id = Column(
        String(36),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    role = Column(String(20), nullable=False)  # 'user', 'assistant', 'system'
    content = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False, index=True)
    rag_enabled = Column(Boolean, default=False, nullable=False)
    memory_enabled = Column(Boolean, default=False, nullable=False)
    thinking_level = Column(String(20), default="med", nullable=False)
    linked_message_id = Column(
        String(36),
        ForeignKey("messages.id", ondelete="SET NULL"),
        nullable=True
    )

    # Relationships
    conversation = relationship("Conversation", back_populates="messages")
    linked_message = relationship("Message", remote_side=[id], foreign_keys=[linked_message_id])

class Memory(Base):
    __tablename__ = "memory"

    id = Column(String(36), primary_key=True, default=generate_uuid, index=True)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)
