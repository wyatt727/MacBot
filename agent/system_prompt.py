# agent/system_prompt.py
import os
import asyncio
import logging
import aiofiles
from typing import List, Dict, Optional
from datetime import datetime
from .config import SYSTEM_PROMPT_FILE, DEBUG_MODE, MAX_SIMILAR_EXAMPLES
from .db import ConversationDB

logger = logging.getLogger(__name__)

# Cache the base system prompt and its last modified time
_system_prompt_cache = {
    'content': None,
    'last_modified': None
}

async def get_base_system_prompt() -> str:
    """
    Read the base system prompt from file, with caching based on file modification time.
    """
    try:
        # Check if file has been modified
        current_mtime = os.path.getmtime(SYSTEM_PROMPT_FILE)
        if (_system_prompt_cache['content'] is not None and 
            _system_prompt_cache['last_modified'] == current_mtime):
            return _system_prompt_cache['content']

        # Read and cache the system prompt
        async with aiofiles.open(SYSTEM_PROMPT_FILE, 'r') as f:
            content = await f.read()
            _system_prompt_cache['content'] = content
            _system_prompt_cache['last_modified'] = current_mtime
            return content
    except Exception as e:
        logger.error(f"Error reading system prompt: {e}")
        # Return a minimal working prompt if file can't be read
        return "You are MacBot, an expert codeblock generator for MacOS that provides code in ```sh or ```python format without additional dialogue."

async def get_system_prompt(user_message: Optional[str] = None) -> str:
    """
    Get the system prompt with optional similar examples included.
    """
    start_time = datetime.now()
    
    # Load base prompt
    base_prompt = await get_base_system_prompt()
    if DEBUG_MODE:
        base_time = (datetime.now() - start_time).total_seconds()
        print("│  ⏱  Base: {:.2f}s".format(base_time))
    
    if not user_message or MAX_SIMILAR_EXAMPLES == 0:
        return base_prompt
        
    # Find similar examples
    db = ConversationDB()
    similar_start = datetime.now()
    similar_exchanges = db.find_successful_exchange(user_message)
    
    if DEBUG_MODE:
        similar_time = (datetime.now() - similar_start).total_seconds()
        print("│  ⏱  Similar: {:.2f}s".format(similar_time))
    
    if not similar_exchanges:
        return base_prompt
        
    # Format examples
    format_start = datetime.now()
    examples = []
    for i, (query, response, similarity) in enumerate(similar_exchanges[:MAX_SIMILAR_EXAMPLES], 1):
        if similarity >= 0.5:  # Only include somewhat relevant examples
            match_type = "Exact match" if similarity == 1.0 else (
                        "Good match" if similarity >= 0.8 else 
                        "Partial match")
            if DEBUG_MODE:
                print(f"│  • {match_type} ({similarity:.1%}): '{query}' -> '{response[:60]}...'")
            examples.append(f"""
Example (semantically similar user request (score: {similarity:.2%})):

User: {query}
Assistant: {response}""")
    
    if DEBUG_MODE:
        format_time = (datetime.now() - format_start).total_seconds()
        print("│  ⏱  Format: {:.2f}s".format(format_time))
    
    # Combine prompts
    combine_start = datetime.now()
    final_prompt = base_prompt + "\n".join(examples)
    
    if DEBUG_MODE:
        combine_time = (datetime.now() - combine_start).total_seconds()
        total_time = (datetime.now() - start_time).total_seconds()
        print("│  ⏱  Combine: {:.2f}s".format(combine_time))
        print("│  ⏱  Total: {:.2f}s".format(total_time))
    
    return final_prompt
