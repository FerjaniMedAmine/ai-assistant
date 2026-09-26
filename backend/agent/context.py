"""Bounded model context with full conversation text retained in PostgreSQL."""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

MAX_INLINE_CHARS = 4_000
RECENT_MESSAGE_LIMIT = 12


def compact_text(message) -> str:
    content = message.content or ""
    if len(content) <= MAX_INLINE_CHARS:
        return content
    preview = "" if "data:image/" in content[:500] else content[:350].strip()
    return (
        f"[Large {message.role} message archived from model context. "
        f"Message ID: {message.id}; {len(content)} characters. "
        f"Use recall_message to retrieve a specific range.]"
        + (f"\nOpening excerpt: {preview}" if preview else "")
    )


def build_history_context(records):
    """Send only recent turns, compacting large bodies while retaining full DB data."""
    omitted = max(0, len(records) - RECENT_MESSAGE_LIMIT)
    messages = []
    if omitted:
        messages.append(SystemMessage(content=(
            f"{omitted} older chat messages are omitted to control context size. "
            "Use search_history to find their IDs, then recall_message to read them."
        )))
    for record in records[-RECENT_MESSAGE_LIMIT:]:
        content = compact_text(record)
        if record.role == "user":
            messages.append(HumanMessage(content=content))
        elif record.role == "assistant":
            messages.append(AIMessage(content=content))
    return messages
