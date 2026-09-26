import logging
import uuid
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from backend.config import config

logger = logging.getLogger(__name__)

class QdrantVectorStore:
    _instance: Optional["QdrantVectorStore"] = None

    def __init__(self):
        qdrant_url = config.qdrant_connection_url
        if not qdrant_url or not config.QDRANT_API_KEY:
            raise ValueError("Qdrant Cloud requires QDRANT_ENDPOINT and QDRANT_API_KEY in .env")
        logger.info("Connecting to Qdrant Cloud")
        self.client = QdrantClient(url=qdrant_url, api_key=config.QDRANT_API_KEY, timeout=30)

        self.collection_name = config.QDRANT_COLLECTION_NAME
        self._ensure_collection()

    @classmethod
    def get_instance(cls) -> "QdrantVectorStore":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _get_distance_metric(self) -> models.Distance:
        metric_str = config.DISTANCE_METRIC.lower()
        if metric_str == "dot":
            return models.Distance.DOT
        elif metric_str == "euclid":
            return models.Distance.EUCLID
        return models.Distance.COSINE

    def _ensure_collection(self):
        """
        Verifies or creates the Qdrant collection configured for hybrid search:
        - Dense vector 'dense' (size 3072, Cosine)
        - Sparse vector 'sparse' with modifier 'idf' (BM25)
        - Payload index on 'user-id' (tenant index)
        """
        datatype_name = config.QDRANT_DENSE_DATATYPE.lower()
        if datatype_name not in ("turbo4", "float32"):
            raise ValueError("QDRANT_DENSE_DATATYPE must be turbo4 or float32")
        datatype = models.Datatype.TURBO4 if datatype_name == "turbo4" else models.Datatype.FLOAT32
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
            logger.info("Creating Qdrant collection '%s' with %s dense and BM25 sparse vectors", self.collection_name, datatype_name)
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": models.VectorParams(
                        size=config.EMBEDDING_DIMENSION,
                        distance=self._get_distance_metric(),
                        on_disk=True,
                        datatype=datatype,
                    )
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(
                        index=models.SparseIndexParams(on_disk=True),
                        modifier=models.Modifier.IDF,
                    )
                },
            )
        info = self.client.get_collection(collection_name=self.collection_name)
        vectors = info.config.params.vectors
        dense = vectors.get("dense") if isinstance(vectors, dict) else None
        sparse = (info.config.params.sparse_vectors or {}).get("sparse")
        if not dense or dense.size != config.EMBEDDING_DIMENSION or dense.distance != self._get_distance_metric() or not sparse or sparse.modifier != models.Modifier.IDF:
            raise ValueError(f"Qdrant collection '{self.collection_name}' has an incompatible dense/BM25 schema. Set QDRANT_COLLECTION_NAME to a new name and reingest documents.")
        actual_datatype = dense.datatype or models.Datatype.FLOAT32
        if actual_datatype != datatype:
            raise ValueError(f"Qdrant collection '{self.collection_name}' uses {actual_datatype}, expected {datatype}. Set QDRANT_COLLECTION_NAME to a new name and reingest documents, or set QDRANT_DENSE_DATATYPE to match.")
        # Existing collections also need this index; do not hide failures.
        if "user-id" not in (info.payload_schema or {}):
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="user-id",
                field_schema=models.KeywordIndexParams(type="keyword", is_tenant=True),
                wait=True,
            )

    def upsert_chunks(
        self,
        chunks: List[str],
        dense_vectors: List[List[float]],
        sparse_vectors: List[Dict[int, float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        user_id: str = "default_user"
    ) -> int:
        """
        Upserts document chunks with dense (3072) and sparse (BM25) vectors into Qdrant.
        Includes 'user-id' in the payload for multi-user isolation.
        """
        if not chunks:
            return 0
        if len(chunks) != len(dense_vectors) or len(chunks) != len(sparse_vectors):
            raise ValueError("Every chunk must have one dense and one sparse vector")

        points = []
        for i, chunk in enumerate(chunks):
            point_id = str(uuid.uuid4())
            sparse_dict = sparse_vectors[i]

            chunk_meta = metadatas[i] if (metadatas and i < len(metadatas)) else {}
            chunk_user_id = chunk_meta.get("user-id") or chunk_meta.get("user_id") or user_id

            payload = {
                "content": chunk,
                "chunk_index": i,
                "user-id": chunk_user_id,
            }
            if chunk_meta:
                payload.update(chunk_meta)
            payload["user-id"] = chunk_user_id

            point = models.PointStruct(
                id=point_id,
                vector={
                    "dense": dense_vectors[i],
                    "sparse": models.SparseVector(
                        indices=[int(k) for k in sparse_dict.keys()],
                        values=[float(v) for v in sparse_dict.values()]
                    )
                },
                payload=payload
            )
            points.append(point)

        # Batch upsert
        batch_size = 64
        for idx in range(0, len(points), batch_size):
            batch = points[idx:idx + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
                wait=True
            )

        return len(points)

    def upsert_exchange(self, user_id: str, conversation_id: str, user_message_id: str,
                        assistant_message_id: str, user_text: str, assistant_text: str) -> None:
        """Index a completed chat exchange independently of the RAG retrieval switch."""
        from backend.rag.embeddings import GeminiEmbeddingService

        content = f"User: {user_text}\nAssistant: {assistant_text}"
        dense, sparse = GeminiEmbeddingService.get_instance().encode([content])
        self.client.upsert(
            collection_name=self.collection_name,
            points=[models.PointStruct(
                id=assistant_message_id,
                vector={
                    "dense": dense[0],
                    "sparse": models.SparseVector(
                        indices=[int(key) for key in sparse[0]],
                        values=[float(value) for value in sparse[0].values()],
                    ),
                },
                payload={
                    "kind": "conversation",
                    "content": content,
                    "source": "Conversation",
                    "user-id": user_id,
                    "conversation_id": conversation_id,
                    "user_message_id": user_message_id,
                    "assistant_message_id": assistant_message_id,
                },
            )],
            wait=True,
        )

    def delete_exchange(self, assistant_message_id: str) -> None:
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(points=[assistant_message_id]),
            wait=True,
        )

    def hybrid_search(
        self,
        dense_query: List[float],
        sparse_query: Dict[int, float],
        top_k: int,
        user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes hybrid search combining dense similarity (gemini-embedding-2)
        and sparse lexical matching (BM25) using Reciprocal Rank Fusion (RRF).
        Supports optional multi-user tenant filtering via 'user-id'.
        """
        if not user_id:
            raise ValueError("user_id is required for document retrieval")
        sparse_vector = models.SparseVector(
            indices=[int(k) for k in sparse_query.keys()],
            values=[float(v) for v in sparse_query.values()]
        )

        query_filter = self._user_filter(user_id)

        response = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(
                    query=sparse_vector,
                    using="sparse",
                    filter=query_filter,
                    limit=top_k
                ),
                models.Prefetch(
                    query=dense_query,
                    using="dense",
                    filter=query_filter,
                    limit=top_k
                )
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )

        results = []
        for point in response.points:
            results.append({
                "id": str(point.id),
                "score": float(point.score) if point.score is not None else 0.0,
                "content": point.payload.get("content", ""),
                "source": point.payload.get("source", "Unknown"),
                "user_id": point.payload.get("user-id", "default_user"),
                "metadata": point.payload
            })
        return results

    def delete_by_conversation(self, conversation_id: str, user_id: str):
        """
        Deletes all vector points associated with the given conversation_id from Qdrant.
        """
        matching_ids = self._scroll_matching_ids(user_id, lambda p: p.get("conversation_id") == conversation_id)
        self._delete_ids(matching_ids)

    def _scroll_matching_ids(self, user_id: str, predicate) -> List[Any]:
        offset = None
        matching_ids = []
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=self._user_filter(user_id),
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            matching_ids.extend(point.id for point in points if predicate(point.payload or {}))
            if offset is None:
                break
        return matching_ids

    def _delete_ids(self, point_ids: List[Any]) -> None:
        for start in range(0, len(point_ids), 256):
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(points=point_ids[start:start + 256]),
                wait=True,
            )

    def clear_all(self, user_id: Optional[str] = None):
        """
        Purges indexed points from Qdrant without dropping collection schemas or tenant indexes.
        If user_id is provided, only deletes points for that user.
        """
        if not user_id:
            raise ValueError("user_id is required when clearing documents")
        try:
            self._delete_ids(self._scroll_matching_ids(user_id, lambda p: p.get("kind") != "conversation"))
            logger.info(f"Purged vector points from '{self.collection_name}' (user_id={user_id}).")
            return True
        except Exception as e:
            logger.error(f"Failed to clear Qdrant collection points: {e}")
            raise e

    @staticmethod
    def _user_filter(user_id: str) -> models.Filter:
        conditions = [models.FieldCondition(key="user-id", match=models.MatchValue(value=user_id))]
        return models.Filter(must=conditions)

    def list_documents(self, user_id: str) -> List[Dict[str, Any]]:
        documents: Dict[str, int] = {}
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=self._user_filter(user_id),
                limit=256,
                offset=offset,
                with_payload=["source", "kind"],
                with_vectors=False,
            )
            for point in points:
                if (point.payload or {}).get("kind") == "conversation":
                    continue
                source = (point.payload or {}).get("source", "Unknown")
                documents[source] = documents.get(source, 0) + 1
            if offset is None:
                break
        return [{"source": source, "chunks": count} for source, count in sorted(documents.items())]

    def delete_document(self, user_id: str, source: str) -> None:
        matching_ids = self._scroll_matching_ids(user_id, lambda p: p.get("kind") != "conversation" and p.get("source") == source)
        self._delete_ids(matching_ids)

    def get_stats(self, user_id: str) -> Dict[str, Any]:
        """
        Returns stats about the Qdrant collection.
        """
        try:
            info = self.client.get_collection(collection_name=self.collection_name)
            point_count = sum(document["chunks"] for document in self.list_documents(user_id))
            return {
                "collection_name": self.collection_name,
                "total_points": point_count,
                "status": str(info.status)
            }
        except Exception as e:
            return {
                "collection_name": self.collection_name,
                "total_points": 0,
                "status": f"unreachable: {str(e)}"
            }
