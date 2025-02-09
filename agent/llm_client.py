# agent/llm_client.py
import aiohttp
import asyncio
import logging
from .config import LLM_API_URL

logger = logging.getLogger(__name__)

async def get_llm_response_async(context: list, model: str, session: aiohttp.ClientSession) -> str:
    """
    Asynchronously send the provided context to the LLM API and return its response.
    """
    payload = {
        "model": model,
        "messages": context,
        "stream": False
    }
    try:
        async with session.post(LLM_API_URL, json=payload, timeout=300) as resp:
            resp.raise_for_status()
            data = await resp.json()
            text = data.get("response") or data.get("message", {}).get("content", "")
            return text
    except Exception as e:
        logger.error(f"LLM API call failed: {e}")
        return f"[Error] LLM API call failed: {e}"
