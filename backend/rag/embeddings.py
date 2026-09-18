import os
import logging
from typing import List, Tuple, Dict, Any, Optional
import torch
from FlagEmbedding import FlagReranker
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from fastembed import SparseTextEmbedding
from backend.config import config

logger = logging.getLogger(__name__)

class GeminiEmbeddingService:
    _instance: Optional["GeminiEmbeddingService"] = None

    def __init__(self):
        api_key = config.GOOGLE_API_KEY or os.environ.get("GOOGLE_API_KEY")
        model_name = config.EMBEDDING_MODEL_NAME
        if not model_name.startswith("models/"):
            model_name = f"models/{model_name}"

        logger.info(f"Initializing Gemini dense embedder ({model_name})...")
        self.dense_model: Optional[GoogleGenerativeAIEmbeddings] = None
        if api_key:
            self.dense_model = GoogleGenerativeAIEmbeddings(
                model=model_name,
                google_api_key=api_key
            )
        else:
            logger.warning("GOOGLE_API_KEY is not set. Gemini dense embeddings will be initialized when key is available.")

        logger.info("Initializing FastEmbed BM25 sparse embedder (Qdrant/bm25)...")
        self.sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
        logger.info("GeminiEmbeddingService initialized successfully.")

    @classmethod
    def get_instance(cls) -> "GeminiEmbeddingService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _ensure_dense_model(self):
        if self.dense_model is None:
            api_key = config.GOOGLE_API_KEY or os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY is not configured. Please set GOOGLE_API_KEY in your .env file.")
            model_name = config.EMBEDDING_MODEL_NAME
            if not model_name.startswith("models/"):
                model_name = f"models/{model_name}"
            self.dense_model = GoogleGenerativeAIEmbeddings(
                model=model_name,
                google_api_key=api_key
            )

    def encode(self, texts: List[str]) -> Tuple[List[List[float]], List[Dict[int, float]]]:
        """
        Encodes a list of passages or documents into:
        1. Dense vectors via Google gemini-embedding-2 (size 3072)
        2. Sparse lexical vectors via BM25 (FastEmbed Qdrant/bm25)
        """
        if not texts:
            return [], []

        self._ensure_dense_model()

        # Dense encoding via Gemini API
        dense_list = self.dense_model.embed_documents(texts)

        # Sparse encoding via BM25
        sparse_embeddings = list(self.sparse_model.passage_embed(texts))
        sparse_list: List[Dict[int, float]] = []
        for sp in sparse_embeddings:
            sparse_dict = {int(idx): float(val) for idx, val in zip(sp.indices, sp.values)}
            sparse_list.append(sparse_dict)

        return dense_list, sparse_list

    def encode_query(self, query: str) -> Tuple[List[float], Dict[int, float]]:
        """
        Encodes a single search query into dense (gemini-embedding-2) and sparse (BM25) representations.
        """
        self._ensure_dense_model()
        dense_vec = self.dense_model.embed_query(query)

        sparse_emb = list(self.sparse_model.query_embed(query))[0]
        sparse_dict = {int(idx): float(val) for idx, val in zip(sparse_emb.indices, sparse_emb.values)}

        return dense_vec, sparse_dict

    def warmup(self):
        """Pre-warms BM25 sparse embedder and Gemini dense embedder."""
        try:
            list(self.sparse_model.query_embed("warmup query"))
            api_key = config.GOOGLE_API_KEY or os.environ.get("GOOGLE_API_KEY")
            if api_key:
                self._ensure_dense_model()
                self.dense_model.embed_query("warmup query")
            logger.info("Gemini dense & FastEmbed BM25 embedder warmed up.")
        except Exception as e:
            logger.warning(f"Embedding warmup warning: {e}")

# Backward compatibility alias
BGEEmbeddingService = GeminiEmbeddingService


class BGERerankerService:
    _instance: Optional["BGERerankerService"] = None

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_fp16 = torch.cuda.is_available()
        logger.info(f"Loading BGE-Reranker model ({config.RERANKER_MODEL_NAME}) on {self.device}...")
        self.reranker = FlagReranker(
            config.RERANKER_MODEL_NAME,
            use_fp16=self.use_fp16,
            device=self.device
        )
        logger.info("BGE-Reranker model loaded successfully on GPU.")

    @classmethod
    def get_instance(cls) -> "BGERerankerService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def compute_scores(self, pairs: List[List[str]]) -> List[float]:
        """
        Computes relevance scores with GPU acceleration and inference mode.
        """
        if not pairs:
            return []
        with torch.inference_mode():
            scores = self.reranker.compute_score(pairs, max_length=384)
        if isinstance(scores, (float, int)):
            return [float(scores)]
        return [float(s) for s in scores]

    def warmup(self):
        """Pre-compiles CUDA kernels for instantaneous reranking."""
        try:
            self.compute_scores([["q", "p"]])
            logger.info("BGE-Reranker GPU kernels warmed up.")
        except Exception as e:
            logger.warning(f"Reranker warmup warning: {e}")
