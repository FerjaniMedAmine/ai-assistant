import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.database import init_db
from backend.rag.vector_store import QdrantVectorStore
from backend.routers import conversations, messages, memory, rag, config_route

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("ai_assistant")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting AI Assistant backend services...")
    # Initialize PostgreSQL tables
    init_db()
    logger.info("PostgreSQL tables verified.")
    # Initialize Qdrant collection
    QdrantVectorStore.get_instance()
    logger.info("Qdrant Vector Store initialized.")

    # Pre-warm BM25, Gemini embeddings and reranker models in background thread
    import threading
    def warmup_models():
        try:
            from backend.rag.embeddings import GeminiEmbeddingService, BGERerankerService
            GeminiEmbeddingService.get_instance().warmup()
            BGERerankerService.get_instance().warmup()
            logger.info("Gemini dense, BM25 sparse, and BGE reranker models are warmed up.")
        except Exception as e:
            logger.warning(f"Model warmup notice: {e}")

    threading.Thread(target=warmup_models, daemon=True).start()
    yield
    logger.info("Shutting down AI Assistant backend services...")

app = FastAPI(
    title="AI Assistant Backend",
    description="Full-stack AI assistant with configurable RAG, hybrid search & conversation memory",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for local frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routers
app.include_router(conversations.router)
app.include_router(messages.router)
app.include_router(memory.router)
app.include_router(rag.router)
app.include_router(config_route.router)

@app.get("/api/health")
def health_check():
    return {"status": "ok", "service": "ai_assistant_backend"}

@app.get("/")
def root():
    return {
        "message": "AI Assistant Backend API is operational",
        "docs": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    from backend.config import config
    uvicorn.run("backend.main:app", host=config.HOST, port=config.PORT, reload=True)
