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
    If stream is True, the function returns the complete response after streaming.
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
    response_text = ""
    while attempt < _max_retries:
        try:
            async with session.post(LLM_API_URL, json=payload, timeout=300) as resp:
                resp.raise_for_status()
                if stream:
                    # Instead of accumulating chunks silently, we yield every chunk as it's decoded.
                    async for chunk in resp.content.iter_any():
                        decoded = chunk.decode()
                        response_text += decoded
                        # Here you could, for example, call a callback or update the UI directly.
                        print(decoded, end="", flush=True)
                else:
                    data = await resp.json()
                    response_text = data.get("response") or data.get("message", {}).get("content", "")
                _llm_cache[cache_key] = response_text
                return response_text
        except Exception as e:
            wait_time = 2 ** attempt
            logger.error(f"LLM API call failed on attempt {attempt+1}: {e}. Retrying in {wait_time} seconds...")
            await asyncio.sleep(wait_time)
            attempt += 1
    response_text = "[Error] LLM API call failed after multiple retries."
    _llm_cache[cache_key] = response_text
    return response_text

async def stream_llm_response(context: list, model: str, session: aiohttp.ClientSession):
    """
    An asynchronous generator that yields response chunks as they are received from the LLM API.
    This allows the caller to update the UI incrementally.
    """
    payload = {
        "model": model,
        "messages": context,
        "stream": True
    }
    async with session.post(LLM_API_URL, json=payload, timeout=300) as resp:
        resp.raise_for_status()
        async for chunk in resp.content.iter_any():
            yield chunk.decode()
