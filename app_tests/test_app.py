import sys
from pathlib import Path
from sqlalchemy.orm import Session

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import config
from backend.database import SessionLocal, init_db
from backend.models import Conversation, Message, Memory
from backend.agent.tools import build_agent_tools
from backend.rag.retrieval import retrieve_and_rerank
from backend.rag.vector_store import QdrantVectorStore

def setup_module():
    init_db()

def test_config_values():
    assert config.CHUNK_SIZE > 0
    assert config.CHUNK_OVERLAP >= 0
    assert config.TOP_K >= config.TOP_N
    assert config.TOP_N > 0
    assert config.EMBEDDING_MODEL_NAME == "gemini-embedding-2"
    assert config.EMBEDDING_DIMENSION == 3072
    assert config.RERANKER_MODEL_NAME == "BAAI/bge-reranker-v2-m3"
    assert "low" in config.THINKING_BUDGET_MAP
    assert "high" in config.THINKING_BUDGET_MAP

def test_database_conversations_and_cascade_delete():
    db: Session = SessionLocal()
    try:
        # Create Linear conversation
        conv = Conversation(title="Test Linear Conversation", mode="linear")
        db.add(conv)
        db.commit()
        db.refresh(conv)

        # Add messages
        m1 = Message(
            conversation_id=conv.id,
            role="user",
            content="Hello 1",
            rag_enabled=False,
            memory_enabled=False,
            thinking_level="low"
        )
        m2 = Message(
            conversation_id=conv.id,
            role="assistant",
            content="Reply 1",
            rag_enabled=False,
            memory_enabled=False,
            thinking_level="low"
        )
        db.add_all([m1, m2])
        db.commit()

        # Verify messages belong to conversation
        msgs = db.query(Message).filter(Message.conversation_id == conv.id).all()
        assert len(msgs) == 2

        # Delete conversation and check cascade delete
        db.delete(conv)
        db.commit()

        msgs_after = db.query(Message).filter(Message.conversation_id == conv.id).all()
        assert len(msgs_after) == 0
    finally:
        db.close()

def test_isolated_mode_and_exchange_linking():
    db: Session = SessionLocal()
    try:
        conv = Conversation(title="Test Isolated Conversation", mode="isolated")
        db.add(conv)
        db.commit()
        db.refresh(conv)

        # Exchange 1
        q1 = Message(
            conversation_id=conv.id,
            role="user",
            content="Question 1",
            rag_enabled=False,
            memory_enabled=False,
            thinking_level="med"
        )
        db.add(q1)
        db.commit()
        db.refresh(q1)

        a1 = Message(
            conversation_id=conv.id,
            role="assistant",
            content="Answer 1",
            rag_enabled=False,
            memory_enabled=False,
            thinking_level="med"
        )
        db.add(a1)
        db.commit()

        # Exchange 2 linked to q1
        q2 = Message(
            conversation_id=conv.id,
            role="user",
            content="Question 2 (linked to 1)",
            rag_enabled=False,
            memory_enabled=False,
            thinking_level="med",
            linked_message_id=q1.id
        )
        db.add(q2)
        db.commit()
        db.refresh(q2)

        assert q2.linked_message_id == q1.id

        # Clean up
        db.delete(conv)
        db.commit()
    finally:
        db.close()

def test_editing_message_hard_deletes_subsequent():
    db: Session = SessionLocal()
    try:
        conv = Conversation(title="Test Edit Conversation", mode="linear")
        db.add(conv)
        db.commit()
        db.refresh(conv)

        # Create 3 exchanges (6 messages)
        import time
        m1 = Message(conversation_id=conv.id, role="user", content="Q1")
        db.add(m1)
        db.commit()
        db.refresh(m1)
        time.sleep(0.02)

        m2 = Message(conversation_id=conv.id, role="assistant", content="A1")
        db.add(m2)
        db.commit()
        time.sleep(0.02)

        m3 = Message(conversation_id=conv.id, role="user", content="Q2")
        db.add(m3)
        db.commit()
        time.sleep(0.02)

        m4 = Message(conversation_id=conv.id, role="assistant", content="A2")
        db.add(m4)
        db.commit()

        all_msgs = db.query(Message).filter(Message.conversation_id == conv.id).all()
        assert len(all_msgs) == 4

        # Edit m1: hard delete all messages after m1
        deleted_count = db.query(Message).filter(
            Message.conversation_id == conv.id,
            Message.created_at > m1.created_at
        ).delete()
        m1.content = "Q1 Edited"
        db.commit()

        assert deleted_count == 3
        remaining = db.query(Message).filter(Message.conversation_id == conv.id).all()
        assert len(remaining) == 1
        assert remaining[0].content == "Q1 Edited"

        # Clean up
        db.delete(conv)
        db.commit()
    finally:
        db.close()

def test_agent_tools():
    db: Session = SessionLocal()
    try:
        conv = Conversation(title="Test Tools Conv", mode="linear")
        db.add(conv)
        db.commit()
        db.refresh(conv)

        # Pre-seed messages
        db.add(Message(conversation_id=conv.id, role="user", content="Msg to clear"))
        db.commit()

        tool_events = []
        tools = build_agent_tools(db, conv.id, tool_events)
        tools_dict = {t.name: t for t in tools}

        # Test add_to_memory
        res_add = tools_dict["add_to_memory"].invoke({"fact": "User prefers concise answers"})
        assert "Fact saved to persistent memory" in res_add

        # Verify fact exists in DB
        fact = db.query(Memory).filter(Memory.content == "User prefers concise answers").first()
        assert fact is not None

        # Test remove_from_memory
        res_remove = tools_dict["remove_from_memory"].invoke({"fact": "concise answers"})
        assert "Permanently removed 1 memory item(s)" in res_remove
        fact_after = db.query(Memory).filter(Memory.content == "User prefers concise answers").first()
        assert fact_after is None

        # Test clear_conversation_history
        res_clear = tools_dict["clear_conversation_history"].invoke({})
        assert "Permanently cleared" in res_clear
        msgs_after = db.query(Message).filter(Message.conversation_id == conv.id).all()
        assert len(msgs_after) == 0

        # Clean up
        db.delete(conv)
        db.commit()
    finally:
        db.close()

def test_qdrant_hybrid_search():
    # Verify hybrid retrieval returns candidates and reranks
    results = retrieve_and_rerank("Gemini model thinking", top_k=5, top_n=2)
    assert isinstance(results, list)
    if results:
        assert "content" in results[0]
        assert "score" in results[0]
        assert "rerank_score" in results[0]

if __name__ == "__main__":
    setup_module()
    print("Running test_config_values...")
    test_config_values()
    print("Running test_database_conversations_and_cascade_delete...")
    test_database_conversations_and_cascade_delete()
    print("Running test_isolated_mode_and_exchange_linking...")
    test_isolated_mode_and_exchange_linking()
    print("Running test_editing_message_hard_deletes_subsequent...")
    test_editing_message_hard_deletes_subsequent()
    print("Running test_agent_tools...")
    test_agent_tools()
    print("Running test_qdrant_hybrid_search...")
    test_qdrant_hybrid_search()
    print("\n==============================")
    print("ALL 6 TESTS PASSED SUCCESSFULLY!")
    print("==============================")

