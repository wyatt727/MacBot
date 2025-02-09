# agent/system_prompt.py
import os
import asyncio
import logging
from .config import SYSTEM_PROMPT_FILE

logger = logging.getLogger(__name__)

_cached_system_prompt = None
_cached_system_prompt_mtime = None

async def get_system_prompt(filepath: str = SYSTEM_PROMPT_FILE) -> str:
    """
    Asynchronously read and cache the system prompt.
    Re-read if the file's modification time changes.
    """
    global _cached_system_prompt, _cached_system_prompt_mtime
    try:
        mtime = os.path.getmtime(filepath)
    except Exception as e:
        logger.error(f"Error getting mtime for {filepath}: {e}")
        return ""
    if _cached_system_prompt is None or _cached_system_prompt_mtime != mtime:
        def read_file():
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read().strip()
        _cached_system_prompt = await asyncio.to_thread(read_file)
        _cached_system_prompt_mtime = mtime
        logger.info(f"System prompt loaded/updated (mtime: {mtime}).")
    return _cached_system_prompt
