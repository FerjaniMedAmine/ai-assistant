import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from backend.config import config
from backend.database import init_db
from backend.rag.vector_store import QdrantVectorStore
from backend.routers import conversations, messages, memory, rag, config_route, auth

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

    yield
    logger.info("Shutting down AI Assistant backend services...")

app = FastAPI(
    title="AI Assistant Backend",
    description="Full-stack AI assistant with configurable RAG, hybrid search & conversation memory",
    version="1.0.0",
    lifespan=lifespan
)

if not config.SESSION_SECRET or len(config.SESSION_SECRET) < 32 or config.SESSION_SECRET.startswith("replace_with"):
    raise RuntimeError("Set a random SESSION_SECRET of at least 32 characters in .env")
app.add_middleware(SessionMiddleware, secret_key=config.SESSION_SECRET, same_site="lax", https_only=config.SESSION_HTTPS_ONLY)

# CORS middleware for local frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[config.FRONTEND_ORIGIN],
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
app.include_router(auth.router)

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
