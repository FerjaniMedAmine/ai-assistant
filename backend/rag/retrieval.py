import logging
from typing import List, Dict, Any
from backend.config import config
from backend.rag.embeddings import GeminiEmbeddingService, BGERerankerService
from backend.rag.vector_store import QdrantVectorStore

logger = logging.getLogger(__name__)

def retrieve_and_rerank(
    query: str,
    top_k: int = None,
    top_n: int = None,
    user_id: str = None
) -> List[Dict[str, Any]]:
    """
    RAG Retrieval Pipeline:
    1. Hybrid Search in Qdrant (dense with gemini-embedding-2 + sparse with FastEmbed BM25) retrieving top_k candidates.
    2. Rerank top_k candidates using BGE-Reranker-v2-M3.
    3. Filter and retain top_n highest scoring passages.
    """
    if not query.strip():
        return []

    effective_top_k = top_k or config.TOP_K
    effective_top_n = top_n or config.TOP_N

    embedder = GeminiEmbeddingService.get_instance()
    vector_store = QdrantVectorStore.get_instance()
    reranker = BGERerankerService.get_instance()

    # Step 1: Hybrid encoding of query (dense 3072 + BM25 sparse)
    dense_q, sparse_q = embedder.encode_query(query)

    # Step 2: Retrieve top-k candidates from Qdrant via RRF
    candidates = vector_store.hybrid_search(
        dense_query=dense_q,
        sparse_query=sparse_q,
        top_k=effective_top_k,
        user_id=user_id
    )

    if not candidates:
        logger.info(f"No candidates found in Qdrant for query: '{query}'")
        return []

    # Step 3: Rerank candidates with BGE-Reranker-v2-M3
    pairs = [[query, c["content"]] for c in candidates]
    rerank_scores = reranker.compute_scores(pairs)

    for i, score in enumerate(rerank_scores):
        candidates[i]["rerank_score"] = score
        candidates[i]["score"] = score  # Primary relevance score

    # Sort descending by rerank score
    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

    # Step 4: Keep top-n
    final_results = candidates[:effective_top_n]
    logger.info(
        f"Retrieved {len(candidates)} candidates, reranked, and selected top {len(final_results)} for RAG."
    )
    return final_results
