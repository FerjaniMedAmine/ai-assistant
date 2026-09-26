import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.database import Base, get_db
from backend.main import app
from backend.models import Conversation, Memory, User
from backend.routers.auth import get_current_user
from langchain_core.messages import AIMessage


class AuthIsolationTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
        Base.metadata.create_all(self.engine)
        self.session = sessionmaker(bind=self.engine)()
        self.alice = User(id="alice", google_sub="google-alice", email="alice@example.com", name="Alice")
        self.bob = User(id="bob", google_sub="google-bob", email="bob@example.com", name="Bob")
        self.session.add_all([self.alice, self.bob,
                              Conversation(id="private-chat", user_id="bob", title="Bob's chat"),
                              Memory(id="private-memory", user_id="bob", content="Bob's secret")])
        self.session.commit()
        app.dependency_overrides[get_db] = lambda: self.session
        app.dependency_overrides[get_current_user] = lambda: self.alice
        self.client = TestClient(app)

    def tearDown(self):
        app.dependency_overrides.clear()
        self.client.close()
        self.session.close()
        self.engine.dispose()

    def test_other_users_records_are_hidden(self):
        self.assertEqual(self.client.get("/api/conversations/private-chat").status_code, 404)
        self.assertEqual(self.client.delete("/api/conversations/private-chat").status_code, 404)
        self.assertEqual(self.client.get("/api/memory").json(), [])
        self.assertEqual(self.client.delete("/api/memory/private-memory").status_code, 404)
        created = self.client.post("/api/conversations", json={"title": "My chat", "mode": "linear"})
        self.assertEqual(created.status_code, 201)
        self.assertEqual(self.session.query(Conversation).filter_by(id=created.json()["id"]).one().user_id, "alice")

    def test_ingestion_works_without_rag_toggle(self):
        with patch("backend.routers.rag.ingest_text", return_value=2) as ingest:
            response = self.client.post("/api/documents/ingest", data={"text": "Knowledge is stored", "source_name": "note"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["chunks_indexed"], 2)
        self.assertEqual(ingest.call_args.kwargs["user_id"], "alice")

    def test_rag_off_skips_retrieval(self):
        from backend.agent.orchestrator import run_agent_orchestrator

        class FakeLLM:
            def bind_tools(self, tools):
                return self

            def invoke(self, messages):
                return AIMessage(content="Answer without retrieval")

        conversation = Conversation(id="alice-chat", user_id="alice", title="Test")
        self.session.add(conversation)
        self.session.commit()
        with patch("backend.agent.orchestrator.build_llm_instance", return_value=FakeLLM()), patch("backend.agent.orchestrator.retrieve_and_rerank") as retrieve, patch("backend.agent.orchestrator.index_exchange", return_value=True):
            response = run_agent_orchestrator(
                self.session, conversation.id, "Hello", rag_enabled=False,
                memory_enabled=False, thinking_level="low", user_id="alice",
            )
        retrieve.assert_not_called()
        self.assertEqual(response["rag_sources"], [])
        with patch("backend.agent.orchestrator.build_llm_instance", return_value=FakeLLM()), patch("backend.agent.orchestrator.retrieve_and_rerank", return_value=[]) as retrieve, patch("backend.agent.orchestrator.index_exchange", return_value=True):
            run_agent_orchestrator(
                self.session, conversation.id, "Hello again", rag_enabled=True,
                memory_enabled=False, thinking_level="low", user_id="alice",
            )
        self.assertEqual(retrieve.call_args.kwargs["user_id"], "alice")

    def test_unauthenticated_api_is_rejected(self):
        app.dependency_overrides.pop(get_current_user)
        self.assertEqual(self.client.get("/api/auth/me").status_code, 401)
        self.assertEqual(self.client.get("/api/memory").status_code, 401)

    def test_google_callback_creates_session_and_user(self):
        from backend.routers import auth
        app.dependency_overrides.pop(get_current_user)

        class GoogleClient:
            async def authorize_access_token(self, request):
                return {"userinfo": {"sub": "google-new", "email": "new@example.com", "email_verified": True, "name": "New User"}}

        with patch.object(auth.oauth, "google", GoogleClient()):
            response = self.client.get("/api/auth/callback", follow_redirects=False)
        self.assertEqual(response.status_code, 303)
        self.assertEqual(self.client.get("/api/auth/me").json()["email"], "new@example.com")
        self.assertEqual(self.session.query(User).filter_by(google_sub="google-new").count(), 1)


if __name__ == "__main__":
    unittest.main()
