# agent/system_prompt.py
import os
import asyncio
import logging
import aiofiles
from .config import SYSTEM_PROMPT_FILE

logger = logging.getLogger(__name__)

_cached_system_prompt = None
_cached_system_prompt_mtime = None

async def get_system_prompt(filepath: str = SYSTEM_PROMPT_FILE) -> str:
    """
    Asynchronously read and cache the system prompt.
    Uses aiofiles for non‑blocking file I/O and re‑reads if the file's modification time changes.
    """
    global _cached_system_prompt, _cached_system_prompt_mtime
    try:
        # Offload the stat call to a thread since os.path.getmtime is blocking.
        mtime = await asyncio.to_thread(os.path.getmtime, filepath)
    except Exception as e:
        logger.error(f"Error getting mtime for {filepath}: {e}")
        return ""
    if _cached_system_prompt is None or _cached_system_prompt_mtime != mtime:
        async with aiofiles.open(filepath, mode="r", encoding="utf-8") as f:
            content = await f.read()
        _cached_system_prompt = content.strip()
        _cached_system_prompt_mtime = mtime
        logger.info(f"System prompt loaded/updated (mtime: {mtime}).")
    return _cached_system_prompt
