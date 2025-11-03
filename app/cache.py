from aiocache import Cache

cache = Cache(Cache.MEMORY)

async def get_conversation_history(user_id: str) -> list[dict[str, str]]:
    """Retrieves the conversation memory for a given user_id."""
    history = await cache.get(user_id)
    return history if history is not None else []

async def add_message_to_history(user_id: str, role: str, content: str):
    """
    Adds a new message to a user's conversation memory.
    This performs the full "Read-Modify-Write" cycle.
    """
    history = await get_conversation_history(user_id)

    history.append({"role": role, "content": content})

    await cache.set(user_id, history)