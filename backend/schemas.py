from datetime import datetime
from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field

# ----------------- Conversations -----------------

class ConversationCreate(BaseModel):
    title: Optional[str] = Field(default="New Conversation", description="Title of conversation")
    mode: Literal["linear", "isolated"] = Field(default="linear", description="linear or isolated mode")

class ConversationResponse(BaseModel):
    id: str
    title: str
    mode: str
    created_at: datetime
    message_count: Optional[int] = 0

    model_config = {"from_attributes": True}

# ----------------- Messages -----------------

class MessageCreate(BaseModel):
    content: str = Field(..., min_length=1, description="Text prompt from the user")
    rag_enabled: bool = Field(default=False, description="Enable RAG retrieved context")
    memory_enabled: bool = Field(default=False, description="Enable persistent user facts memory")
    thinking_level: Literal["low", "med", "high"] = Field(default="med", description="Thinking depth level")
    linked_message_id: Optional[str] = Field(default=None, description="Linked prior message ID for isolated mode")

class MessageEdit(BaseModel):
    content: str = Field(..., min_length=1, description="New content for the edited message")

class MessageResponse(BaseModel):
    id: str
    conversation_id: str
    role: str
    content: str
    created_at: datetime
    rag_enabled: bool
    memory_enabled: bool
    thinking_level: str
    linked_message_id: Optional[str] = None

    model_config = {"from_attributes": True}

class RAGSourceItem(BaseModel):
    content: str
    source: Optional[str] = "Unknown"
    score: float
    metadata: Optional[Dict[str, Any]] = None

class ToolCallEvent(BaseModel):
    tool: str
    args: Dict[str, Any]
    result: str

class AgentReply(BaseModel):
    user_message: MessageResponse
    assistant_message: MessageResponse
    rag_sources: List[RAGSourceItem] = []
    tool_calls: List[ToolCallEvent] = []
    thoughts: Optional[str] = None

# ----------------- Memory -----------------

class MemoryCreate(BaseModel):
    content: str = Field(..., min_length=1, description="Persistent fact about the user")

class MemoryResponse(BaseModel):
    id: str
    content: str
    created_at: datetime

    model_config = {"from_attributes": True}

# ----------------- Ingestion & Documents -----------------

class IngestResponse(BaseModel):
    status: str
    chunks_indexed: int
    collection_name: str
    source_name: str

class DocumentInfo(BaseModel):
    total_points: int
    collection_name: str
    status: str

# ----------------- Config -----------------

class ConfigResponse(BaseModel):
    gemini_model: str
    embedding_model: str
    reranker_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    top_n: int
    qdrant_collection: str
    has_google_key: bool
    has_tavily_key: bool
