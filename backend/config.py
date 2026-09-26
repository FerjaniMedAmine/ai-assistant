from typing import Dict, Optional
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, AliasChoices

# Base project root
BASE_DIR = Path(__file__).resolve().parent.parent

class AppConfig(BaseSettings):
    """
    Centralized configuration for the entire AI Assistant system.
    All tunable parameters (chunk size, overlap, top-k, top-n, model names,
    Qdrant settings, database connection, thinking budgets, etc.) are declared here.
    """
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Database
    DATABASE_URL: str = Field(description="Neon PostgreSQL connection string")

    # API Keys
    GOOGLE_API_KEY: Optional[str] = Field(default=None, description="Google Gemini API Key")
    TAVILY_API_KEY: Optional[str] = Field(default=None, description="Tavily Web Search API Key")
    GOOGLE_OAUTH_CLIENT_ID: Optional[str] = Field(default=None, validation_alias=AliasChoices("GOOGLE_OAUTH_CLIENT_ID", "CLIENT-ID"))
    GOOGLE_OAUTH_CLIENT_SECRET: Optional[str] = Field(default=None, validation_alias=AliasChoices("GOOGLE_OAUTH_CLIENT_SECRET", "CLIENT-SECRET"))
    GOOGLE_OAUTH_REDIRECT_URI: str = "http://localhost:5173/api/auth/callback"
    FRONTEND_ORIGIN: str = "http://localhost:5173"
    SESSION_SECRET: Optional[str] = None
    SESSION_HTTPS_ONLY: bool = False

    # LLM Settings
    GEMINI_MODEL: str = Field(default="gemini-3.5-flash", description="Gemini model identifier")
    TEMPERATURE: float = Field(default=0.7, description="Sampling temperature")
    THINKING_BUDGET_MAP: Dict[str, int] = Field(
        default_factory=lambda: {
            "low": 0,       # 0 disables thinking tokens for instantaneous responses
            "med": 1024,    # Fast thinking
            "high": 4096    # Deep deliberation
        },
        description="Thinking budget token allocations for low, med, high levels"
    )
    THINKING_LEVEL_MAP: Dict[str, str] = Field(
        default_factory=lambda: {
            "low": "LOW",
            "med": "MEDIUM",
            "high": "HIGH"
        },
        description="Gemini API Thinking Level enum mapping"
    )

    # Qdrant Vector DB Settings (Cloud or Remote)
    QDRANT_ENDPOINT: Optional[str] = Field(default=None, description="Qdrant Cloud endpoint URL")
    QDRANT_URL: Optional[str] = Field(default=None, description="Remote Qdrant server URL")
    QDRANT_API_KEY: Optional[str] = Field(default=None, description="Qdrant API Key (for Qdrant Cloud or protected cluster)")
    QDRANT_COLLECTION_NAME: str = Field(default="ai-assistant-collection", description="Vector collection name")
    DISTANCE_METRIC: str = Field(default="Cosine", description="Vector similarity metric: Cosine, Dot, or Euclid")

    @property
    def qdrant_connection_url(self) -> Optional[str]:
        return self.QDRANT_ENDPOINT or self.QDRANT_URL

    # Embeddings & Reranker
    EMBEDDING_MODEL_NAME: str = Field(default="gemini-embedding-2", description="Google Gemini Embedding model")
    RERANKER_MODEL_NAME: str = Field(default="BAAI/bge-reranker-v2-m3", description="FlagEmbedding reranker model")
    EMBEDDING_DIMENSION: int = Field(default=3072, description="gemini-embedding-2 dense dimension")
    QDRANT_DENSE_DATATYPE: str = Field(default="float32", description="Dense vector storage: float32 or turbo4 (Qdrant 1.19+)")
    RERANKER_4BIT: bool = True

    # RAG Chunking & Retrieval Parameters
    CHUNK_SIZE: int = Field(default=512, description="Chunk size in characters")
    CHUNK_OVERLAP: int = Field(default=64, description="Chunk overlap in characters")
    TOP_K: int = Field(default=10, description="Initial candidates retrieved via hybrid search")
    TOP_N: int = Field(default=3, description="Final candidates preserved after reranker")

    # Server Configuration
    HOST: str = "127.0.0.1"
    PORT: int = 8000

# Singleton configuration instance
config = AppConfig()
