"""
Standalone CLI script for document ingestion into Qdrant hybrid vector store.
Supports individual files (.txt, .md, .pdf, etc.) and entire directories.

Usage:
    python ingest.py --file path/to/document.pdf
    python ingest.py --path path/to/docs_folder/
"""

import sys
import argparse
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend.config import config
from backend.rag.ingestion import ingest_file
from backend.rag.vector_store import QdrantVectorStore

def main():
    parser = argparse.ArgumentParser(description="Ingest documents into Qdrant Hybrid Vector Store (gemini-embedding-2 + BM25)")
    parser.add_argument("--file", type=str, help="Path to a single document file to ingest")
    parser.add_argument("--path", type=str, help="Path to a directory containing documents to ingest")
    parser.add_argument("--user-id", type=str, default="default_user", help="User ID for multi-user isolation")
    args = parser.parse_args()

    if not args.file and not args.path:
        parser.print_help()
        print("\nError: Please specify either --file or --path")
        sys.exit(1)

    print(f"============================================================")
    print(f"Document Ingestion Utility")
    print(f"Collection:     {config.QDRANT_COLLECTION_NAME}")
    print(f"Embedding:      {config.EMBEDDING_MODEL_NAME} (3072-dim) + BM25")
    print(f"User ID:        {args.user_id}")
    print(f"Chunk Size:     {config.CHUNK_SIZE}")
    print(f"Chunk Overlap:  {config.CHUNK_OVERLAP}")
    print(f"============================================================")

    total_chunks = 0

    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"Error: File not found: {file_path}")
            sys.exit(1)
        print(f"Processing file: {file_path}")
        chunks = ingest_file(file_path, user_id=args.user_id)
        total_chunks += chunks
        print(f"Ingested {chunks} chunks from {file_path.name}")

    if args.path:
        dir_path = Path(args.path)
        if not dir_path.exists() or not dir_path.is_dir():
            print(f"Error: Directory not found: {dir_path}")
            sys.exit(1)
        
        supported_exts = {".txt", ".md", ".pdf", ".py", ".json", ".html"}
        files = [p for p in dir_path.rglob("*") if p.is_file() and p.suffix.lower() in supported_exts]
        print(f"Found {len(files)} supported document(s) in {dir_path}")

        for p in files:
            print(f"Processing: {p.relative_to(dir_path)}")
            chunks = ingest_file(p, user_id=args.user_id)
            total_chunks += chunks
            print(f"  -> {chunks} chunks")

    stats = QdrantVectorStore.get_instance().get_stats()
    print(f"\nIngestion Complete!")
    print(f"Total chunks indexed this run: {total_chunks}")
    print(f"Total points now in Qdrant collection: {stats['total_points']}")

if __name__ == "__main__":
    main()
