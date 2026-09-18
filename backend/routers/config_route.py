import os
from fastapi import APIRouter
from backend.config import config
from backend.schemas import ConfigResponse

router = APIRouter(prefix="/api/config", tags=["config"])

@router.get("", response_model=ConfigResponse)
def get_public_config():
    """
    Returns non-secret runtime configuration and status flags to the frontend.
    """
    has_google = bool(config.GOOGLE_API_KEY or os.environ.get("GOOGLE_API_KEY"))
    has_tavily = bool(config.TAVILY_API_KEY or os.environ.get("TAVILY_API_KEY"))

    return ConfigResponse(
        gemini_model=config.GEMINI_MODEL,
        embedding_model=config.EMBEDDING_MODEL_NAME,
        reranker_model=config.RERANKER_MODEL_NAME,
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        top_k=config.TOP_K,
        top_n=config.TOP_N,
        qdrant_collection=config.QDRANT_COLLECTION_NAME,
        has_google_key=has_google,
        has_tavily_key=has_tavily
    )
