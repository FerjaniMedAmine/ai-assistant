import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from backend.config import config
from backend.rag.embeddings import GeminiEmbeddingService
from backend.rag.vector_store import QdrantVectorStore

logger = logging.getLogger(__name__)

def extract_text_from_file(file_path: Path) -> str:
    """
    Extracts raw text from .txt, .md, and .pdf files.
    """
    suffix = file_path.suffix.lower()
    if suffix in [".txt", ".md", ".json", ".csv", ".py", ".html"]:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    elif suffix == ".pdf":
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(str(file_path))
            text_parts = [page.get_text() for page in doc]
            return "\n\n".join(text_parts)
        except ImportError:
            import pypdf
            reader = pypdf.PdfReader(str(file_path))
            text_parts = [page.extract_text() or "" for page in reader.pages]
            return "\n\n".join(text_parts)
    else:
        # Fallback to binary/text read
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

def ingest_text(
    text: str,
    source_name: str = "raw_text",
    metadata: Optional[Dict[str, Any]] = None,
    user_id: str = "default_user"
) -> int:
    """
    Splits text into chunks, computes hybrid embeddings (Gemini dense 3072 + FastEmbed BM25),
    and indexes into Qdrant Cloud vector store with tenant user-id.
    """
    if not text.strip():
        return 0

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_text(text)
    if not chunks:
        return 0

    logger.info(f"Ingesting {len(chunks)} chunks from source '{source_name}' (user_id='{user_id}')...")
    embedder = GeminiEmbeddingService.get_instance()
    dense_vecs, sparse_vecs = embedder.encode(chunks)

    metadatas = []
    base_meta = metadata or {}
    for i, chunk in enumerate(chunks):
        chunk_meta = dict(base_meta)
        chunk_meta.update({
            "source": source_name,
            "chunk_index": i,
            "total_chunks": len(chunks),
            "user-id": user_id
        })
        metadatas.append(chunk_meta)

    vector_store = QdrantVectorStore.get_instance()
    count = vector_store.upsert_chunks(
        chunks=chunks,
        dense_vectors=dense_vecs,
        sparse_vectors=sparse_vecs,
        metadatas=metadatas,
        user_id=user_id
    )
    logger.info(f"Successfully indexed {count} chunks for '{source_name}'.")
    return count

def ingest_file(file_path: Path, user_id: str = "default_user") -> int:
    """
    Reads a file from disk and ingests it.
    """
    text = extract_text_from_file(file_path)
    return ingest_text(
        text,
        source_name=file_path.name,
        metadata={"file_path": str(file_path)},
        user_id=user_id
    )
