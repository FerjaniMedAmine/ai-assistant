import logging
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from langchain_core.tools import tool
from backend.models import Memory, Message
from backend.config import config
from backend.agent.workspace import UserWorkspace

logger = logging.getLogger(__name__)

def build_agent_tools(
    db: Session,
    conversation_id: str,
    tool_events: List[Dict[str, Any]],
    user_id: str,
):
    """
    Factory function producing LangChain tools bound to the current database session,
    conversation ID, and tool execution event tracker.
    """
    workspace = UserWorkspace(user_id)

    def record_tool(name: str, args: Dict[str, Any], result: str) -> str:
        tool_events.append({"tool": name, "args": args, "result": result[:500]})
        return result

    @tool
    def list_directory(path: str = ".") -> str:
        """List files and folders in your private workspace. Paths are relative to its root."""
        return record_tool("list_directory", {"path": path}, workspace.list_directory(path))

    @tool
    def read_file(path: str, start_char: int = 0, max_chars: int = 12000) -> str:
        """Read a range from a UTF-8 text file in your private workspace."""
        result = workspace.read_file(path, start_char, max_chars)
        record_tool("read_file", {"path": path, "start_char": start_char}, f"Read {len(result)} characters")
        return result

    @tool
    def write_file(path: str, content: str, overwrite: bool = False) -> str:
        """Create a UTF-8 text file in your private workspace; set overwrite to replace one."""
        result = workspace.write_file(path, content, overwrite)
        return record_tool("write_file", {"path": path, "overwrite": overwrite}, result)

    @tool
    def create_directory(path: str) -> str:
        """Create a folder in your private workspace, including missing parents."""
        return record_tool("create_directory", {"path": path}, workspace.create_directory(path))

    @tool
    def move_path(source: str, destination: str) -> str:
        """Rename or move a file or folder within your private workspace."""
        return record_tool("move_path", {"source": source, "destination": destination}, workspace.move_path(source, destination))

    @tool
    def delete_path(path: str) -> str:
        """Delete a file or an empty folder in your private workspace."""
        return record_tool("delete_path", {"path": path}, workspace.delete_path(path))

    @tool
    def search_history(query: str) -> str:
        """Find older messages in this conversation and return IDs and short previews."""
        query = query.strip()
        if not query:
            return "Provide a search phrase"
        matches = (
            db.query(Message)
            .filter(Message.conversation_id == conversation_id, Message.content.ilike(f"%{query[:100]}%"))
            .order_by(Message.created_at.desc())
            .limit(10)
            .all()
        )
        lines = [f"{item.id} ({item.role}): {item.content[:180]}" for item in matches]
        return record_tool("search_history", {"query": query}, "\n".join(lines) or "No matching messages")

    @tool
    def recall_message(message_id: str, start_char: int = 0, max_chars: int = 6000) -> str:
        """Retrieve a precise range of a previous user or assistant message by its ID."""
        if start_char < 0 or not 1 <= max_chars <= 6000:
            return "Invalid range; max_chars must be 1 to 6000"
        item = db.query(Message).filter(Message.id == message_id, Message.conversation_id == conversation_id).first()
        if not item:
            return "Message not found in this conversation"
        result = item.content[start_char:start_char + max_chars]
        suffix = f"\n[Continue at character {start_char + len(result)}]" if start_char + len(result) < len(item.content) else ""
        record_tool("recall_message", {"message_id": message_id, "start_char": start_char}, f"Recalled {len(result)} characters")
        return result + suffix

    @tool
    def add_to_memory(fact: str) -> str:
        """
        Store a permanent fact about the user in long-term memory across all conversations.
        Use this tool when the user shares personal details, preferences, names, habits,
        equipment, or instructs you to remember something (e.g., 'remember that I prefer Python',
        'my dog is named Max').
        """
        fact_clean = fact.strip()
        if not fact_clean:
            return "Cannot store empty fact."

        memory_item = Memory(content=fact_clean, user_id=user_id)
        db.add(memory_item)
        db.commit()
        db.refresh(memory_item)

        msg = f"Fact saved to persistent memory: '{fact_clean}'"
        tool_events.append({
            "tool": "add_to_memory",
            "args": {"fact": fact_clean},
            "result": msg
        })
        logger.info(msg)
        return msg

    @tool
    def remove_from_memory(fact: str) -> str:
        """
        Remove or forget a specific fact or detail from long-term persistent memory.
        Use this tool when the user asks you to forget or delete something from memory
        (e.g., 'forget my dog's name', 'remove my old location from memory').
        """
        query_term = fact.strip()
        if not query_term:
            return "Cannot search for empty fact to remove."

        matches = db.query(Memory).filter(Memory.user_id == user_id, Memory.content.ilike(f"%{query_term}%")).all()
        if not matches:
            msg = f"No persistent memories matched '{query_term}'."
            tool_events.append({
                "tool": "remove_from_memory",
                "args": {"fact": query_term},
                "result": msg
            })
            return msg

        count = len(matches)
        for m in matches:
            db.delete(m)
        db.commit()

        msg = f"Permanently removed {count} memory item(s) matching '{query_term}'."
        tool_events.append({
            "tool": "remove_from_memory",
            "args": {"fact": query_term},
            "result": msg
        })
        logger.info(msg)
        return msg

    @tool
    def clear_conversation_history() -> str:
        """
        Permanently delete all previous messages in the current conversation thread.
        Use this tool when the user specifically instructs you to clear, erase, or forget
        this conversation's history (e.g., 'forget this conversation', 'clear chat history').
        """
        try:
            from backend.rag.vector_store import QdrantVectorStore
            QdrantVectorStore.get_instance().delete_by_conversation(conversation_id, user_id)
        except Exception:
            logger.exception("Failed to clear indexed conversation history from Qdrant")
        deleted_count = db.query(Message).filter(Message.conversation_id == conversation_id).delete()
        db.commit()

        msg = f"Permanently cleared {deleted_count} messages from this conversation."
        tool_events.append({
            "tool": "clear_conversation_history",
            "args": {},
            "result": msg
        })
        logger.info(msg)
        return msg

    @tool
    def tavily_search(query: str) -> str:
        """
        Search the live web for real-time, up-to-date information, current news, sports,
        weather, facts, or technical documentation that may not be in your static knowledge.
        Use autonomously whenever answering questions requiring current or external web information.
        """
        api_key = config.TAVILY_API_KEY
        if not api_key:
            msg = "Web search is disabled: TAVILY_API_KEY is not set in the environment."
            tool_events.append({
                "tool": "tavily_search",
                "args": {"query": query},
                "result": msg
            })
            return msg

        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=api_key)
            search_response = client.search(query=query, max_results=5)
            
            results_list = search_response.get("results", [])
            if not results_list:
                msg = f"No web search results found for query: '{query}'."
            else:
                formatted = []
                for r in results_list:
                    formatted.append(f"Title: {r.get('title')}\nURL: {r.get('url')}\nContent: {r.get('content')}")
                msg = "\n\n---\n\n".join(formatted)

            tool_events.append({
                "tool": "tavily_search",
                "args": {"query": query},
                "result": f"Found {len(results_list)} web results."
            })
            return msg
        except Exception as e:
            err_msg = f"Error performing Tavily web search: {str(e)}"
            logger.error(err_msg)
            tool_events.append({
                "tool": "tavily_search",
                "args": {"query": query},
                "result": err_msg
            })
            return err_msg

    return [
        add_to_memory, remove_from_memory, clear_conversation_history, tavily_search,
        list_directory, read_file, write_file, create_directory, move_path, delete_path,
        search_history, recall_message,
    ]
