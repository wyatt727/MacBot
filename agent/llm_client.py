# agent/llm_client.py
import aiohttp
import asyncio
import logging
import json
from .config import OLLAMA_API_BASE
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

async def get_llm_response_async(
    messages: List[Dict[str, str]], 
    model: str, 
    session: aiohttp.ClientSession,
    num_thread: int = 4,
    num_gpu: int = 0,
    timeout: Optional[int] = None
) -> str:
    """
    Get a response from the LLM using Ollama's API with optimized settings.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content'
        model: Name of the Ollama model to use
        session: aiohttp ClientSession for making requests
        num_thread: Number of CPU threads to use (default: 4)
        num_gpu: Number of GPU layers to use (default: 0)
        timeout: Request timeout in seconds (default: None, uses session timeout)
    
    Returns:
        The model's response text
    
    Raises:
        TimeoutError: If the request times out
        ValueError: If the response is invalid
        RuntimeError: For other API errors
    """
    try:
        # Prepare the request with Ollama-specific optimizations
        request_data = {
            "model": model,
            "messages": messages,
            "options": {
                "num_thread": num_thread,
                "num_gpu": num_gpu
            },
            "stream": False  # Disable streaming for better performance
        }
        
        url = f"{OLLAMA_API_BASE}/api/chat"
        start_time = None
        
        try:
            # Create a new timeout for this specific request if specified
            request_timeout = aiohttp.ClientTimeout(total=timeout) if timeout is not None else None
            
            async with session.post(url, json=request_data, timeout=request_timeout) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(
                        f"API request failed with status {response.status}: {error_text}"
                    )
                
                try:
                    result = await response.json()
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON response: {e}")
                
                if not isinstance(result, dict) or 'message' not in result:
                    raise ValueError(f"Unexpected response format: {result}")
                
                return result['message']['content']
                
        except aiohttp.ClientError as e:
            if isinstance(e, aiohttp.ClientTimeout):
                raise TimeoutError(f"Request timed out: {e}")
            raise RuntimeError(f"Network error: {e}")
            
    except Exception as e:
        logger.error(f"Error in LLM request: {e}")
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
    url = f"{OLLAMA_API_BASE}/api/chat"
    async with session.post(url, json=payload) as resp:
        resp.raise_for_status()
        async for chunk in resp.content.iter_any():
            yield chunk.decode()
