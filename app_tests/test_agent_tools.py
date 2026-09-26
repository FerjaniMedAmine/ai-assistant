import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from backend.agent.context import build_history_context
from backend.agent.tools import build_agent_tools
from backend.agent.workspace import UserWorkspace
from backend.database import Base
from backend.models import Conversation, Message, User


class WorkspaceTests(unittest.TestCase):
    def test_file_and_folder_operations_are_scoped(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            alice = UserWorkspace("alice", root=base / "alice")
            bob = UserWorkspace("bob", root=base / "bob")
            alice.create_directory("notes")
            alice.write_file("notes/hello.txt", "héllo")
            self.assertEqual(alice.read_file("notes/hello.txt", start_char=1), "éllo")
            self.assertIn("hello.txt", alice.list_directory("notes"))
            with self.assertRaises(ValueError):
                bob.read_file("notes/hello.txt")
            with self.assertRaises(ValueError):
                alice.read_file("../bob/notes/hello.txt")
            with self.assertRaises(ValueError):
                alice.write_file("notes/hello.txt", "overwrite")
            alice.move_path("notes/hello.txt", "notes/renamed.txt")
            self.assertEqual(alice.read_file("notes/renamed.txt"), "héllo")
            alice.delete_path("notes/renamed.txt")
            alice.delete_path("notes")

    def test_symlink_cannot_escape_workspace(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            workspace = UserWorkspace("alice", root=base / "alice")
            outside = base / "outside.txt"
            outside.write_text("private")
            (workspace.root / "link.txt").symlink_to(outside)
            with self.assertRaises(ValueError):
                workspace.read_file("link.txt")


class ContextTests(unittest.TestCase):
    def test_large_user_and_assistant_messages_are_recallable_references(self):
        records = [
            SimpleNamespace(id="user-1", role="user", content="<html>" + "x" * 6000),
            SimpleNamespace(id="assistant-1", role="assistant", content="```json\n" + "y" * 6000),
        ]
        messages = build_history_context(records)
        self.assertEqual(len(messages), 2)
        self.assertIn("user-1", messages[0].content)
        self.assertIn("assistant-1", messages[1].content)
        self.assertNotIn("x" * 1000, messages[0].content)
        self.assertNotIn("y" * 1000, messages[1].content)

    def test_history_recall_stays_in_own_conversation(self):
        engine = create_engine("sqlite+pysqlite:///:memory:")
        Base.metadata.create_all(engine)
        with tempfile.TemporaryDirectory() as temp_dir, Session(engine) as db:
            db.add_all([
                User(id="alice", google_sub="google-alice", email="alice@example.com", name="Alice"),
                User(id="bob", google_sub="google-bob", email="bob@example.com", name="Bob"),
                Conversation(id="alice-chat", user_id="alice", title="Alice"),
                Conversation(id="bob-chat", user_id="bob", title="Bob"),
            ])
            db.commit()
            bob_message = Message(id=str(uuid.uuid4()), conversation_id="bob-chat", role="user", content="Bob's secret")
            alice_message = Message(id=str(uuid.uuid4()), conversation_id="alice-chat", role="user", content="Alice's note")
            db.add_all([bob_message, alice_message])
            db.commit()
            with patch("backend.agent.tools.UserWorkspace", return_value=UserWorkspace("alice", root=Path(temp_dir) / "alice")):
                tools = {item.name: item for item in build_agent_tools(db, "alice-chat", [], "alice")}
            self.assertIn("Alice's note", tools["recall_message"].invoke({"message_id": alice_message.id}))
            self.assertEqual(tools["recall_message"].invoke({"message_id": bob_message.id}), "Message not found in this conversation")
            self.assertIn(alice_message.id, tools["search_history"].invoke({"query": "Alice"}))
            self.assertNotIn(bob_message.id, tools["search_history"].invoke({"query": "secret"}))


if __name__ == "__main__":
    unittest.main()
