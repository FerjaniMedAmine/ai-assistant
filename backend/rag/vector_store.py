import logging
import os
import uuid
from typing import List, Dict, Any, Optional
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.http import models
from backend.config import config

logger = logging.getLogger(__name__)

class QdrantVectorStore:
    _instance: Optional["QdrantVectorStore"] = None

    def __init__(self):
        qdrant_url = config.qdrant_connection_url
        if qdrant_url:
            logger.info(f"Connecting to remote/cloud Qdrant at {qdrant_url}")
            self.client = QdrantClient(
                url=qdrant_url,
                api_key=config.QDRANT_API_KEY
            )
        else:
            qdrant_dir = Path(config.QDRANT_PATH)
            qdrant_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Connecting to local embedded Qdrant at {qdrant_dir}")
            self.client = QdrantClient(path=str(qdrant_dir))

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
        try:
            collections = [c.name for c in self.client.get_collections().collections]
            if self.collection_name not in collections:
                logger.info(f"Creating Qdrant collection '{self.collection_name}' with dense (3072) and BM25 sparse index...")
                distance = self._get_distance_metric()
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config={
                        "dense": models.VectorParams(
                            size=config.EMBEDDING_DIMENSION,
                            distance=distance,
                            on_disk=True,
                            hnsw_config=models.HnswConfigDiff(
                                m=0,
                                payload_m=24,
                                ef_construct=256
                            ),
                            datatype=models.Datatype.FLOAT32
                        )
                    },
                    sparse_vectors_config={
                        "sparse": models.SparseVectorParams(
                            index=models.SparseIndexParams(on_disk=True),
                            modifier=models.Modifier.IDF
                        )
                    }
                )
                logger.info(f"Collection '{self.collection_name}' created successfully.")

                # Create tenant payload index on user-id
                try:
                    self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name="user-id",
                        field_schema=models.KeywordIndexParams(
                            type="keyword",
                            is_tenant=True,
                            on_disk=False
                        )
                    )
                    logger.info("Created tenant index on 'user-id'.")
                except Exception as ex_idx:
                    logger.warning(f"Payload index creation note: {ex_idx}")
            else:
                logger.info(f"Qdrant collection '{self.collection_name}' already exists.")
        except Exception as e:
            logger.warning(f"Note connecting to Qdrant collection '{self.collection_name}': {e}")

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
        sparse_vector = models.SparseVector(
            indices=[int(k) for k in sparse_query.keys()],
            values=[float(v) for v in sparse_query.values()]
        )

        query_filter = None
        if user_id:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="user-id",
                        match=models.MatchValue(value=user_id)
                    )
                ]
            )

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
            limit=top_k
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

    def delete_by_conversation(self, conversation_id: str):
        """
        Deletes all vector points associated with the given conversation_id from Qdrant.
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="conversation_id",
                                match=models.MatchValue(value=conversation_id)
                            )
                        ]
                    )
                )
            )
            logger.info(f"Removed vector points for conversation {conversation_id}")
        except Exception as e:
            logger.warning(f"Failed to delete Qdrant points for conversation {conversation_id}: {e}")

    def clear_all(self, user_id: Optional[str] = None):
        """
        Purges indexed points from Qdrant without dropping collection schemas or tenant indexes.
        If user_id is provided, only deletes points for that user.
        """
        try:
            if user_id:
                filter_cond = models.Filter(
                    must=[models.FieldCondition(key="user-id", match=models.MatchValue(value=user_id))]
                )
            else:
                filter_cond = models.Filter()

            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(filter=filter_cond)
            )
            logger.info(f"Purged vector points from '{self.collection_name}' (user_id={user_id}).")
            return True
        except Exception as e:
            logger.error(f"Failed to clear Qdrant collection points: {e}")
            raise e

    def get_stats(self) -> Dict[str, Any]:
        """
        Returns stats about the Qdrant collection.
        """
        try:
            info = self.client.get_collection(collection_name=self.collection_name)
            return {
                "collection_name": self.collection_name,
                "total_points": info.points_count or 0,
                "status": str(info.status)
            }
        except Exception as e:
            return {
                "collection_name": self.collection_name,
                "total_points": 0,
                "status": f"unreachable: {str(e)}"
            }

