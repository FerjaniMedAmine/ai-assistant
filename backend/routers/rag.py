import os
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from backend.models import User
from backend.routers.auth import get_current_user
from backend.rag.ingestion import ingest_text, ingest_file
from backend.rag.vector_store import QdrantVectorStore
from backend.schemas import IngestResponse, DocumentInfo
from backend.config import config

router = APIRouter(prefix="/api/documents", tags=["documents"])

@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(None),
    text: str = Form(None),
    source_name: str = Form(None),
    user: User = Depends(get_current_user),
):
    """
    Ingest a document file (.txt, .md, .pdf) or raw text snippet into Qdrant Cloud
    with hybrid Gemini (gemini-embedding-2) dense and FastEmbed BM25 sparse representations.
    """
    if file and file.filename:
        # Save to a temporary file and parse
        suffix = Path(file.filename).suffix
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = Path(tmp.name)

        try:
            count = ingest_file(tmp_path, user_id=user.id, source_name=Path(file.filename).name)
            doc_name = file.filename
        finally:
            if tmp_path.exists():
                os.unlink(tmp_path)

        return IngestResponse(
            status="success",
            chunks_indexed=count,
            collection_name=config.QDRANT_COLLECTION_NAME,
            source_name=doc_name
        )

    elif text and text.strip():
        name = source_name or "text_snippet"
        count = ingest_text(text, source_name=name, user_id=user.id)
        return IngestResponse(
            status="success",
            chunks_indexed=count,
            collection_name=config.QDRANT_COLLECTION_NAME,
            source_name=name
        )

    else:
        raise HTTPException(
            status_code=400,
            detail="Either an uploaded file or text content must be provided."
        )

@router.get("", response_model=DocumentInfo)
def get_documents_info(user: User = Depends(get_current_user)):
    """
    Returns current index statistics from Qdrant.
    """
    store = QdrantVectorStore.get_instance()
    stats = store.get_stats(user.id)
    return DocumentInfo(
        total_points=stats.get("total_points", 0),
        collection_name=stats.get("collection_name", config.QDRANT_COLLECTION_NAME),
        status=stats.get("status", "ready")
    )

@router.delete("", status_code=200)
def clear_all_documents(user: User = Depends(get_current_user)):
    """
    Purges all documents and indexed points from Qdrant.
    """
    store = QdrantVectorStore.get_instance()
    store.clear_all(user.id)
    return {"status": "success", "message": "Your documents have been removed."}


@router.get("/list")
def list_documents(user: User = Depends(get_current_user)):
    return QdrantVectorStore.get_instance().list_documents(user.id)


@router.delete("/source")
def delete_document(source: str, user: User = Depends(get_current_user)):
    QdrantVectorStore.get_instance().delete_document(user.id, source)
    return {"status": "success"}
