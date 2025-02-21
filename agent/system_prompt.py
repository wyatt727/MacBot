# agent/system_prompt.py
import os
import asyncio
import logging
import aiofiles
from typing import List, Dict, Optional
from datetime import datetime
from .config import SYSTEM_PROMPT_FILE
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
    Get the system prompt with dynamically included similar examples.
    Shows which examples are being used in the console.
    
    Args:
        user_message: Optional current user message to find similar examples for
    
    Returns:
        Complete system prompt including relevant examples
    """
    # Get the base system prompt
    base_prompt = await get_base_system_prompt()
    
    # If no user message, return just the base prompt
    if not user_message:
        return base_prompt
        
    try:
        # Get similar examples from the database
        db = ConversationDB()
        similar_exchanges = db.find_successful_exchange(user_message)
        
        if not similar_exchanges:
            print("\n[Context] Warning: No exchanges found in database. This should not happen if database is populated.")
            return base_prompt
            
        # Build example section and show what's being used
        print("\n[Context] Including similar examples:")
        examples = "\n\nHere are some example interactions to guide your responses:"
        for stored_prompt, stored_response, ratio in similar_exchanges:
            similarity_desc = "Excellent" if ratio >= 0.9 else "Good" if ratio >= 0.7 else "Partial" if ratio >= 0.5 else "Low"
            print(f"• {similarity_desc} match ({ratio:.2%}): '{stored_prompt}' -> '{stored_response}'")
            examples += f"\n\nUser: {stored_prompt}\nAssistant: {stored_response}"
        
        # Combine base prompt with examples
        complete_prompt = base_prompt + examples
        
        # Show the complete prompt that will be sent to the model
        print("\n[Context] Complete system prompt being sent to model:")
        print("-" * 80)
        print(complete_prompt)
        print("-" * 80)
        
        return complete_prompt
        
    except Exception as e:
        logger.error(f"Error adding similar examples to prompt: {e}")
        print("\n[Context] Error retrieving similar examples:", str(e))
        return base_prompt
    finally:
        if 'db' in locals():
            db.close()
