# agent/llm_client.py
import aiohttp
import asyncio
import logging
import json
from .config import LLM_API_URL, CACHE_TTL
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# In-memory cache for LLM responses with timestamps
_llm_cache = {}

def _get_cache_entry(cache_key: str) -> Optional[Tuple[datetime, str]]:
    """Get a cache entry and check if it's still valid."""
    if cache_key in _llm_cache:
        timestamp, response = _llm_cache[cache_key]
        if datetime.now() - timestamp < timedelta(seconds=CACHE_TTL):
            logger.debug("Cache hit for query")
            return timestamp, response
    return None

async def get_llm_response_async(
    messages: List[Dict[str, str]], 
    model: str,
    session: aiohttp.ClientSession
) -> str:
    """
    Get a response from the LLM API asynchronously.
    Uses persistent caching with TTL and handles both streaming and non-streaming responses.
    Cache entries are kept even after expiration for potential reuse.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content'
        model: Name of the model to use
        session: aiohttp ClientSession with configured timeout
        
    Returns:
        The response text from the LLM
    """
    # Generate cache key from model and messages
    cache_key = f"{model}:{json.dumps(messages, sort_keys=True)}"
    
    # Check cache first
    cache_entry = _get_cache_entry(cache_key)
    if cache_entry:
        return cache_entry[1]
    
    # If not in cache or expired, make API request
    try:
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "raw": True  # Request raw output to bypass any API-level caching
        }
        
        async with session.post(
            LLM_API_URL,
            json=payload
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
            
            # Handle both standard and message-style responses
            response = data.get("message", {}).get("content", "") or data.get("response", "")
            if not response:
                # If we have an expired cache entry, use it as fallback
                if cache_key in _llm_cache:
                    logger.warning("Empty API response, using expired cache as fallback")
                    return _llm_cache[cache_key][1]
                raise ValueError("Empty response from LLM API")
            
            # Update cache with new response
            _llm_cache[cache_key] = (datetime.now(), response)
            return response
            
    except (aiohttp.ClientError, json.JSONDecodeError) as e:
        logger.error(f"API request failed: {e}")
        # On API errors, try to use expired cache as fallback
        if cache_key in _llm_cache:
            logger.warning(f"API error, using expired cache as fallback: {e}")
            return _llm_cache[cache_key][1]
        raise
    except Exception as e:
        logger.error(f"Unexpected error in LLM request: {e}")
        # For other errors, also try to use expired cache
        if cache_key in _llm_cache:
            logger.warning(f"Unexpected error, using expired cache as fallback: {e}")
            return _llm_cache[cache_key][1]
        raise

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
    async with session.post(LLM_API_URL, json=payload) as resp:
        resp.raise_for_status()
        async for chunk in resp.content.iter_any():
            yield chunk.decode()
