import logging
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from langchain_core.tools import tool
from backend.models import Memory, Message
from backend.config import config

logger = logging.getLogger(__name__)

def build_agent_tools(
    db: Session,
    conversation_id: str,
    tool_events: List[Dict[str, Any]]
):
    """
    Factory function producing LangChain tools bound to the current database session,
    conversation ID, and tool execution event tracker.
    """

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

        memory_item = Memory(content=fact_clean)
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

        matches = db.query(Memory).filter(Memory.content.ilike(f"%{query_term}%")).all()
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

    return [add_to_memory, remove_from_memory, clear_conversation_history, tavily_search]
