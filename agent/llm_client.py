# agent/llm_client.py
import aiohttp
import asyncio
import logging
import json
from .config import LLM_API_URL

logger = logging.getLogger(__name__)

# In-memory cache for LLM responses.
_llm_cache = {}
_max_retries = 3

async def get_llm_response_async(context: list, model: str, session: aiohttp.ClientSession, stream: bool = False) -> str:
    """
    Asynchronously send the provided context to the LLM API and return its response.
    Implements caching, streaming responses, and retry logic with exponential backoff.
    """
    cache_key = f"{model}:{json.dumps(context, sort_keys=True)}"
    if cache_key in _llm_cache:
        return _llm_cache[cache_key]
    
    payload = {
        "model": model,
        "messages": context,
        "stream": stream
    }
    
    attempt = 0
    while attempt < _max_retries:
        try:
            async with session.post(LLM_API_URL, json=payload, timeout=300) as resp:
                resp.raise_for_status()
                if stream:
                    # Process streaming response.
                    chunks = []
                    async for chunk in resp.content.iter_any():
                        chunks.append(chunk.decode())
                    text = ''.join(chunks)
                else:
                    data = await resp.json()
                    text = data.get("response") or data.get("message", {}).get("content", "")
                _llm_cache[cache_key] = text
                return text
        except Exception as e:
            wait_time = 2 ** attempt
            logger.error(f"LLM API call failed on attempt {attempt+1}: {e}. Retrying in {wait_time} seconds...")
            await asyncio.sleep(wait_time)
            attempt += 1
    text = "[Error] LLM API call failed after multiple retries."
    _llm_cache[cache_key] = text
    return text
