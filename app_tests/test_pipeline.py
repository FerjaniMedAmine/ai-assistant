import unittest
import json
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from qdrant_client.http import models
from langchain_core.messages import AIMessageChunk
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from backend.config import config
from backend.database import Base
from backend.models import Conversation, User, Message
from backend.rag.vector_store import QdrantVectorStore


class PipelineTests(unittest.TestCase):
    def test_cloud_collection_uses_turbo4_and_bm25(self):
        client = MagicMock()
        client.get_collections.return_value.collections = []
        client.get_collection.return_value = SimpleNamespace(
            config=SimpleNamespace(params=SimpleNamespace(
                vectors={"dense": models.VectorParams(
                    size=config.EMBEDDING_DIMENSION,
                    distance=models.Distance.COSINE,
                    datatype=models.Datatype.TURBO4,
                )},
                sparse_vectors={"sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)},
            )),
            payload_schema={},
        )
        with patch("backend.rag.vector_store.QdrantClient", return_value=client), patch.object(config, "QDRANT_ENDPOINT", "https://example.cloud.qdrant.io:6333"), patch.object(config, "QDRANT_API_KEY", "test-key"), patch.object(config, "QDRANT_DENSE_DATATYPE", "turbo4"):
            store = QdrantVectorStore()
        self.assertIs(store.client, client)
        kwargs = client.create_collection.call_args.kwargs
        self.assertEqual(kwargs["vectors_config"]["dense"].datatype, models.Datatype.TURBO4)
        self.assertEqual(kwargs["sparse_vectors_config"]["sparse"].modifier, models.Modifier.IDF)
        client.create_payload_index.assert_called_once()

    def test_cloud_credentials_required(self):
        with patch.object(config, "QDRANT_ENDPOINT", None), patch.object(config, "QDRANT_URL", None):
            with self.assertRaisesRegex(ValueError, "QDRANT_ENDPOINT"):
                QdrantVectorStore()

    def test_hybrid_search_filters_tenant_and_returns_payload(self):
        store = QdrantVectorStore.__new__(QdrantVectorStore)
        store.collection_name = "test"
        store.client = MagicMock()
        store.client.query_points.return_value.points = [SimpleNamespace(
            id="point-1", score=0.8,
            payload={"content": "sample text", "source": "sample.txt", "user-id": "default_user"},
        )]
        results = store.hybrid_search([0.1, 0.2], {3: 1.0}, 3, "default_user")
        self.assertEqual(results[0]["source"], "sample.txt")
        kwargs = store.client.query_points.call_args.kwargs
        self.assertTrue(kwargs["with_payload"])
        self.assertEqual(kwargs["query"].fusion, models.Fusion.RRF)
        self.assertEqual(kwargs["prefetch"][0].using, "sparse")
        self.assertEqual(kwargs["prefetch"][0].filter.must[0].match.value, "default_user")

    def test_uploaded_source_name_is_preserved(self):
        from pathlib import Path
        from backend.rag.ingestion import ingest_file
        with patch("backend.rag.ingestion.extract_text_from_file", return_value="sample"), patch("backend.rag.ingestion.ingest_text", return_value=1) as ingest:
            self.assertEqual(ingest_file(Path("/tmp/random.pdf"), source_name="manual.pdf"), 1)
        self.assertEqual(ingest.call_args.kwargs["source_name"], "manual.pdf")

    def test_document_deletion_is_user_scoped(self):
        store = QdrantVectorStore.__new__(QdrantVectorStore)
        store.collection_name = "test"
        store.client = MagicMock()
        store.client.scroll.return_value = ([SimpleNamespace(id="owned-point", payload={"source": "manual.pdf"}),
                                             SimpleNamespace(id="other-source", payload={"source": "other.pdf"})], None)
        store.delete_document("alice", "manual.pdf")
        conditions = store.client.scroll.call_args.kwargs["scroll_filter"].must
        self.assertEqual({condition.key: condition.match.value for condition in conditions}, {"user-id": "alice"})
        self.assertEqual(store.client.delete.call_args.kwargs["points_selector"].points, ["owned-point"])
        with self.assertRaisesRegex(ValueError, "user_id"):
            store.clear_all()

    def test_tavily_search_returns_source_urls(self):
        from backend.agent.tools import build_agent_tools
        with patch.object(config, "TAVILY_API_KEY", "test-key"), patch("tavily.TavilyClient") as client_type:
            client_type.return_value.search.return_value = {"results": [
                {"title": "Source", "url": "https://example.com/story", "content": "Current fact"}
            ]}
            events = []
            search = next(tool for tool in build_agent_tools(None, "conversation", events, "test-user") if tool.name == "tavily_search")
            result = search.invoke({"query": "current fact"})
        self.assertIn("https://example.com/story", result)
        self.assertEqual(events[0]["result"], "Found 1 web results.")

    def test_completed_exchange_is_upserted_with_tenant_and_message_ids(self):
        store = QdrantVectorStore.__new__(QdrantVectorStore)
        store.collection_name = "test"
        store.client = MagicMock()
        assistant_id = str(uuid.uuid4())
        with patch("backend.rag.embeddings.GeminiEmbeddingService.get_instance") as embedder:
            embedder.return_value.encode.return_value = ([[0.1, 0.2]], [{4: 1.0}])
            store.upsert_exchange("alice", "conversation-1", "user-1", assistant_id, "Hello", "Hi")
        point = store.client.upsert.call_args.kwargs["points"][0]
        self.assertEqual(point.id, assistant_id)
        self.assertEqual(point.payload["user-id"], "alice")
        self.assertEqual(point.payload["kind"], "conversation")
        self.assertEqual(point.payload["conversation_id"], "conversation-1")
        self.assertIn("Assistant: Hi", point.payload["content"])

    def test_clearing_documents_keeps_indexed_conversations(self):
        store = QdrantVectorStore.__new__(QdrantVectorStore)
        store.collection_name = "test"
        store.client = MagicMock()
        store.client.scroll.return_value = ([
            SimpleNamespace(id="doc", payload={"source": "manual.pdf"}),
            SimpleNamespace(id="chat", payload={"kind": "conversation", "source": "Conversation"}),
        ], None)
        store.clear_all("alice")
        self.assertEqual(store.client.delete.call_args.kwargs["points_selector"].points, ["doc"])
        self.assertEqual(store.list_documents("alice"), [{"source": "manual.pdf", "chunks": 1}])


class StreamTests(unittest.IsolatedAsyncioTestCase):
    async def test_stream_saves_real_answer_and_indexes_when_rag_off(self):
        from backend.agent.orchestrator import stream_agent_orchestrator

        engine = create_engine("sqlite+pysqlite:///:memory:")
        Base.metadata.create_all(engine)
        user_id = str(uuid.uuid4())
        conversation_id = str(uuid.uuid4())
        llm = MagicMock()
        llm.bind_tools.return_value.stream.return_value = iter([
            AIMessageChunk(content="The weather is sunny."),
            AIMessageChunk(content=""),
        ])
        with Session(engine) as db:
            db.add(User(id=user_id, google_sub="stream-test", email="test@example.com", name="Test"))
            db.add(Conversation(id=conversation_id, user_id=user_id, title="Weather", mode="linear"))
            db.commit()
            with patch("backend.agent.orchestrator.build_llm_instance", return_value=llm), \
                 patch("backend.agent.orchestrator.build_agent_tools", return_value=[]), \
                 patch("backend.agent.orchestrator.retrieve_and_rerank") as retrieval, \
                 patch("backend.agent.orchestrator.index_exchange") as indexing:
                indexing.return_value = True
                events = [json.loads(event.removeprefix("data: ")) async for event in stream_agent_orchestrator(
                    db, conversation_id, "Weather in Tunisia?", False, False, "low", user_id=user_id,
                )]
            retrieval.assert_not_called()
            indexing.assert_called_once()
            self.assertEqual(events[-1]["type"], "done")
            self.assertEqual(events[-1]["assistant_message"]["content"], "The weather is sunny.")
            self.assertEqual(db.query(Message).filter(Message.role == "assistant").one().content, "The weather is sunny.")

    async def test_tool_stream_saves_answer_after_search(self):
        from backend.agent.orchestrator import stream_agent_orchestrator

        engine = create_engine("sqlite+pysqlite:///:memory:")
        Base.metadata.create_all(engine)
        user_id = str(uuid.uuid4())
        conversation_id = str(uuid.uuid4())
        llm = MagicMock()
        llm.bind_tools.return_value.stream.side_effect = [
            iter([AIMessageChunk(content="Checking weather...", tool_call_chunks=[{
                "name": "tavily_search", "args": '{"query":"weather Tunisia"}', "id": "call-1", "index": 0,
            }])]),
            iter([AIMessageChunk(content="It is sunny in Tunisia today.")]),
        ]
        search = MagicMock()
        search.name = "tavily_search"
        search.invoke.return_value = "Sunny, 25 C"
        with Session(engine) as db:
            db.add(User(id=user_id, google_sub="tool-test", email="tool@example.com", name="Test"))
            db.add(Conversation(id=conversation_id, user_id=user_id, title="Weather", mode="linear"))
            db.commit()
            with patch("backend.agent.orchestrator.build_llm_instance", return_value=llm), \
                 patch("backend.agent.orchestrator.build_agent_tools", return_value=[search]), \
                 patch("backend.agent.orchestrator.index_exchange", return_value=True) as indexing:
                events = [json.loads(event.removeprefix("data: ")) async for event in stream_agent_orchestrator(
                    db, conversation_id, "Weather in Tunisia?", False, False, "low", user_id=user_id,
                )]
            self.assertEqual(search.invoke.call_count, 1)
            self.assertEqual(indexing.call_count, 1)
            self.assertEqual(events[-1]["assistant_message"]["content"], "It is sunny in Tunisia today.")


if __name__ == "__main__":
    unittest.main()
