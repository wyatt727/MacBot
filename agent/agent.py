# agent/agent.py
import sys
import asyncio
import aiohttp
import subprocess
import logging
import readline
import os
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Dict
from .db import ConversationDB
from .system_prompt import get_system_prompt
from .llm_client import get_llm_response_async
from .code_executor import execute_code_async
from .config import (
    LLM_MODEL, MAX_CONCURRENT_LLM_CALLS, 
    MAX_CONCURRENT_CODE_EXECUTIONS, RESPONSE_TIMEOUT,
    SIMILARITY_THRESHOLD
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class CommandHistory:
    """Manages command history with persistence and navigation."""
    def __init__(self, history_file: str = os.path.expanduser("~/.macbot_history")):
        self.history_file = history_file
        self.current_index = 0
        self.temp_input = ""
        
        # Ensure the directory exists with correct permissions
        try:
            history_dir = os.path.dirname(self.history_file)
            if not os.path.exists(history_dir):
                os.makedirs(history_dir, mode=0o700)  # Only user can read/write
            # If file doesn't exist, create it with correct permissions
            if not os.path.exists(self.history_file):
                with open(self.history_file, 'w') as f:
                    pass
                os.chmod(self.history_file, 0o600)  # Only user can read/write
        except Exception as e:
            logger.warning(f"Failed to setup history file: {e}")
            # Fallback to temporary file in /tmp
            self.history_file = os.path.join('/tmp', '.macbot_history_' + str(os.getuid()))
        
        self.load_history()
        
        # Set up readline
        readline.set_history_length(1000)
        readline.parse_and_bind('"\e[A": previous-history')  # Up arrow
        readline.parse_and_bind('"\e[B": next-history')      # Down arrow
        readline.parse_and_bind('"\C-r": reverse-search-history')  # Ctrl+R
        readline.parse_and_bind('"\t": complete')  # Tab completion
        
    def load_history(self):
        """Load command history from file."""
        try:
            if os.path.exists(self.history_file):
                readline.read_history_file(self.history_file)
                logger.debug(f"Loaded command history from {self.history_file}")
        except Exception as e:
            logger.warning(f"Failed to load history file: {e}")
    
    def save_history(self):
        """Save command history to file."""
        try:
            os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
            readline.write_history_file(self.history_file)
            logger.debug(f"Saved command history to {self.history_file}")
        except Exception as e:
            logger.warning(f"Failed to save history file: {e}")
    
    def add_command(self, command: str):
        """Add a command to history if it's not empty and different from last command."""
        if command.strip():
            # Only add if different from the last command
            if readline.get_current_history_length() == 0 or command != readline.get_history_item(readline.get_current_history_length()):
                readline.add_history(command)
                self.save_history()

class MinimalAIAgent:
    """
    Minimal AI Agent that uses a local SQLite database for conversation history.
    The agent first checks if there is a cached successful exchange whose user prompt is
    very similar (≥ 95%) to the current query. If so, it immediately uses that cached response
    (thus bypassing new LLM text generation), but still processes any code execution.
    Otherwise, it builds context by combining:
      - The system prompt.
      - An example interaction from the success DB (if the best match is at least 80% similar).
      - The current user query.
    This context is then sent to the LLM.
    """
    def __init__(self, model: str = LLM_MODEL):
        self.model = model
        self.db = ConversationDB()
        self.last_user_query = ""
        self.aiohttp_session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=RESPONSE_TIMEOUT)
        )
        # Optimized semaphores
        self.llm_semaphore = asyncio.Semaphore(MAX_CONCURRENT_LLM_CALLS)
        self.code_semaphore = asyncio.Semaphore(MAX_CONCURRENT_CODE_EXECUTIONS)
        self.command_history = CommandHistory()
        self.session_start = datetime.now()
        self.in_comparison_mode = False
        
        # Response cache
        self._response_cache = {}
        self._cache_lock = asyncio.Lock()
        
        # Initialize command aliases and shortcuts
        self.command_aliases = {
            '/h': self.show_help,
            '/help': self.show_help,
            '/history': self.show_command_history,
            '/clear': self.clear_screen,
            '/stats': self.show_session_stats,
            '/repeat': self.repeat_last_command,
            '/success': self.process_success_command,
            '/compare': self.compare_models_command,
            '/nocache': lambda x: x  # Pass-through for nocache command
        }

    async def show_help(self, _: str):
        """Show help information about available commands."""
        help_text = """
Available Commands:
------------------
Up/Down Arrow : Navigate through command history
Ctrl+R       : Search command history
Tab          : Auto-complete commands
/h, /help    : Show this help message
/history     : Show command history
/clear       : Clear the screen
/stats       : Show session statistics
/repeat      : Repeat last command
/success     : Manage successful exchanges
/compare     : Compare responses across all available models
/bye, exit   : Exit the agent

Model Comparison:
---------------
Use /compare [prompt] to run a prompt against all available models and compare their:
- Response times
- Token counts
- Code execution results
- Success rates

If no prompt is provided, the last query will be used.

Tips:
-----
- Use arrow keys to navigate through previous commands
- Commands are automatically saved and persisted between sessions
- Use Tab for command auto-completion
- Ctrl+C to cancel current operation
"""
        print(help_text)

    async def show_command_history(self, _: str):
        """Show the command history with timestamps."""
        print("\nCommand History:")
        print("-" * 50)
        for i in range(1, readline.get_current_history_length() + 1):
            cmd = readline.get_history_item(i)
            print(f"{i:3d}. {cmd}")
        print("-" * 50)

    async def clear_screen(self, _: str):
        """Clear the terminal screen."""
        os.system('cls' if os.name == 'nt' else 'clear')

    async def show_session_stats(self, _: str):
        """Show statistics for the current session."""
        duration = datetime.now() - self.session_start
        total_commands = readline.get_current_history_length()
        print("\nSession Statistics:")
        print("-" * 50)
        print(f"Session Duration: {duration}")
        print(f"Commands Executed: {total_commands}")
        print(f"Successful Exchanges: {len(self.db.list_successful_exchanges())}")
        print("-" * 50)

    async def repeat_last_command(self, _: str):
        """Repeat the last command."""
        if readline.get_current_history_length() > 0:
            last_cmd = readline.get_history_item(readline.get_current_history_length())
            print(f"Repeating: {last_cmd}")
            return last_cmd
        return ""

    def _extract_command_and_args(self, message: str) -> Tuple[Optional[str], str]:
        """Extract command and arguments from a message."""
        if message.startswith('/'):
            parts = message.split(maxsplit=1)
            command = parts[0][1:]  # Remove the leading '/'
            args = parts[1] if len(parts) > 1 else ""
            return command, args
        return None, message

    async def _build_context(self, user_message: str, no_cache: bool = False) -> List[Dict[str, str]]:
        """
        Build the context for the conversation, including:
        1. System prompt (with dynamically included similar examples)
        2. Recent conversation history
        3. Similar successful exchanges (if caching is enabled)
        
        Args:
            user_message: The current user message
            no_cache: If True, skips retrieving similar examples from cache
        
        Returns:
            List of message dictionaries forming the conversation context
        """
        context = []
        
        # 1. Add system prompt with relevant examples
        system_prompt = await get_system_prompt(user_message if not no_cache else None)
        context.append({"role": "system", "content": system_prompt})
        
        # 2. Add recent conversation history
        context.extend(self.db.get_recent_messages())
        
        # 3. Add current user message
        context.append({"role": "user", "content": user_message})
        return context

    def extract_code_from_response(self, response: str):
        """Extract code blocks from a response."""
        from .code_executor import extract_code_blocks
        return extract_code_blocks(response)

    async def process_code_block(self, language: str, code: str) -> Tuple[int, str]:
        """
        Process a code block with auto-fix attempts.
        If execution fails, the agent sends a fix prompt to the LLM and retries (up to MAX_FIX_ATTEMPTS).
        """
        from .config import MAX_FIX_ATTEMPTS
        attempt = 0
        current_code = code.strip()
        last_error = ""
        while attempt < MAX_FIX_ATTEMPTS:
            ret, output = await execute_code_async(language, current_code)
            if ret == 0:
                return ret, output
            else:
                last_error = output
                fix_prompt = (
                    f"The following {language} code produced an error:\n\n"
                    f"{current_code}\n\n"
                    f"Error Output:\n{output}\n\n"
                    f"Please fix the code. Return only the corrected code in a single code block."
                )
                fix_context = [{"role": "user", "content": fix_prompt}]
                async with self.llm_semaphore:
                    fix_response = await get_llm_response_async(fix_context, self.model, self.aiohttp_session)
                blocks = self.extract_code_from_response(fix_response)
                if blocks:
                    current_code = blocks[0][1]
                    logger.info(f"Auto-fix attempt {attempt+1} applied.")
                else:
                    logger.error("LLM did not return a corrected code block.")
                    break
            attempt += 1
        return ret, last_error

    async def process_code_block_with_semaphore(self, language: str, code: str, idx: int):
        async with self.code_semaphore:
            return await self.process_code_block(language, code)

    async def _get_cached_response(self, prompt: str) -> Optional[str]:
        """Get response from cache if available and not expired."""
        async with self._cache_lock:
            if prompt in self._response_cache:
                timestamp, response = self._response_cache[prompt]
                if datetime.now() - timestamp < timedelta(seconds=CACHE_TTL):
                    return response
                del self._response_cache[prompt]
            return None

    async def _cache_response(self, prompt: str, response: str):
        """Cache a response with timestamp."""
        async with self._cache_lock:
            self._response_cache[prompt] = (datetime.now(), response)

    async def _process_code_blocks_parallel(self, blocks: List[Tuple[str, str]]) -> List[Tuple[int, str]]:
        """Process code blocks in parallel with optimized concurrency."""
        if not blocks:
            return []
            
        async def process_block(block: Tuple[str, str]) -> Tuple[int, str]:
            lang, code = block
            async with self.code_semaphore:
                return await self.process_code_block(lang, code)
        
        return await asyncio.gather(*(process_block(block) for block in blocks))

    async def process_message(self, message: str, no_cache: bool = False) -> str:
        """Optimized message processing with caching and parallel execution."""
        try:
            # Check for high similarity matches first (if caching is enabled)
            if not no_cache:
                # Find similar exchanges
                similar_exchanges = self.db.find_successful_exchange(message)
                if similar_exchanges:
                    best_match = similar_exchanges[0]  # (prompt, response, similarity)
                    # Check for exact match first
                    if best_match[0].lower().strip() == message.lower().strip():
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug("\n[Cache] Found exact match")
                        return best_match[1]
                    # Otherwise check for high similarity
                    elif best_match[2] >= 0.93:  # If similarity is 93% or higher
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(f"\n[Cache] Found highly similar match (similarity: {best_match[2]:.2%})")
                            logger.debug(f"Original prompt: {best_match[0]}")
                        cached_response = best_match[1]
                        
                        # Extract and execute code blocks immediately
                        blocks = self.extract_code_from_response(cached_response)
                        if blocks:
                            results = await self._process_code_blocks_parallel(blocks)
                            for idx, (ret, output) in enumerate(results, 1):
                                print(f"\n--- Code Block #{idx} Execution Result ---\n{output}\n")
                        
                        return cached_response

            # If no high similarity match or caching disabled, proceed with normal processing
            context = await self._build_context(message, no_cache=no_cache)
            async with self.llm_semaphore:
                response = await get_llm_response_async(
                    context, 
                    self.model, 
                    self.aiohttp_session
                )

            # Save successful exchanges if not in comparison mode and caching is enabled
            if not self.in_comparison_mode and not no_cache:
                # Check if the response contains valid code blocks
                blocks = self.extract_code_from_response(response)
                if blocks:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug("\n[DB] Adding successful exchange to database")
                    if self.db.add_successful_exchange(message, response):
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug("[DB] Successfully added to database")
                    else:
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug("[DB] Exchange already exists in database")

            # Cache the response if appropriate
            if not no_cache and response:
                await self._cache_response(message, response)

            return response

        except asyncio.TimeoutError:
            logger.error("LLM request timed out")
            return "Error: Request timed out. Please try again."
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return f"Error processing message: {str(e)}"

    def help_command(self) -> str:
        """Return help text describing available commands."""
        return """Available commands:
/help - Show this help message
/list [search] - List successful exchanges, optionally filtered by search term
/remove <id> - Remove a successful exchange by ID
/update <id> <response> - Update the response for a successful exchange
/compare <prompt> - Compare responses from different models
/nocache <prompt> - Process prompt without using or saving to cache

Examples:
/list python  - List exchanges containing 'python'
/remove 123   - Remove exchange with ID 123
/nocache ls -la  - Run 'ls -la' without cache
"""

    def _is_successful_exchange(self, response: str) -> bool:
        # Implement the logic to determine if a response is a successful exchange
        # This is a placeholder and should be replaced with the actual implementation
        return True

    def list_command(self, search: str) -> str:
        # Implement the logic to list successful exchanges
        # This is a placeholder and should be replaced with the actual implementation
        return "List command not implemented"

    def remove_command(self, id: str) -> str:
        # Implement the logic to remove a successful exchange
        # This is a placeholder and should be replaced with the actual implementation
        return "Remove command not implemented"

    def update_command(self, id: str, response: str) -> str:
        # Implement the logic to update a successful exchange
        # This is a placeholder and should be replaced with the actual implementation
        return "Update command not implemented"

    async def process_success_command(self, command: str):
        """
        Process a /success command by launching the GUI for managing successful exchanges.
        The GUI is launched as a separate process so that all GUI code runs on the main thread.
        """
        print("Launching Success DB GUI...")
        try:
            proc = subprocess.Popen([sys.executable, "-m", "agent.gui_success"])
            if proc.pid:
                print(f"Success DB GUI launched (PID: {proc.pid}).")
            else:
                print("Failed to launch Success DB GUI: No PID returned.")
        except Exception as e:
            print(f"Failed to launch Success DB GUI: {e}")

    async def compare_models_command(self, command: str):
        """
        Triggers model comparison mode for the last query or a new query.
        Responses from comparison mode are not saved as successful exchanges.
        
        Usage:
            /compare [prompt]  - Compare models using a new prompt
            /compare          - Compare models using the last query
        """
        from .model_comparator import compare_models, analyze_results
        
        # Parse command to get optional prompt
        parts = command.split(maxsplit=1)
        prompt = parts[1] if len(parts) > 1 else self.last_user_query
        
        if not prompt:
            print("Please enter a prompt first or provide one with the command.")
            return
            
        try:
            print(f"\nRunning comparison for prompt: {prompt}")
            print("This may take a while depending on the number of available models...")
            
            # Set a flag to prevent responses from being saved during comparison
            self.in_comparison_mode = True
            try:
                results = await compare_models(prompt, self.aiohttp_session, self.db)
            finally:
                self.in_comparison_mode = False
            
            if not results:
                print("\nNo results returned from model comparison.")
                return
                
            analysis = analyze_results(results)
            print(analysis)
            
        except Exception as e:
            logger.error(f"Error during model comparison: {e}")
            print(f"\nError during model comparison: {e}")
            print("Please try again or check the logs for more details.")

    async def run(self):
        """Enhanced main loop with optimized processing."""
        print("Minimal AI Agent for macOS (Local Mode).")
        print('Type "/h" or "/help" for available commands.')
        try:
            while True:
                try:
                    user_input = await asyncio.to_thread(input, "User: ")
                    
                    # Quick exit check
                    if user_input.lower() in ("exit", "quit", "/bye"):
                        break

                    # Handle empty input
                    if not user_input.strip():
                        user_input = self.last_user_query
                        if not user_input:
                            continue

                    # Process command flags
                    no_cache = False
                    if user_input.startswith('/nocache '):
                        no_cache = True
                        user_input = user_input[9:].strip()
                    
                    # Handle special commands
                    if user_input.startswith('/'):
                        command = user_input.split()[0].lower()
                        if command in self.command_aliases:
                            result = await self.command_aliases[command](user_input)
                            if isinstance(result, str):
                                user_input = result
                            else:
                                continue

                    # Add to history and update last query
                    self.command_history.add_command(user_input)
                    self.last_user_query = user_input

                    # Process message and handle response
                    response = await self.process_message(user_input, no_cache=no_cache)
                    print("\n[Response]\n", response)

                    # Extract and process code blocks in parallel
                    blocks = self.extract_code_from_response(response)
                    if blocks:
                        results = await self._process_code_blocks_parallel(blocks)
                        for idx, (ret, output) in enumerate(results, 1):
                            print(f"\n--- Code Block #{idx} Execution Result ---\n{output}\n")
                    
                    # Update conversation history
                    self.db.add_message("user", user_input)
                    self.db.add_message("assistant", response)

                except KeyboardInterrupt:
                    print("\nOperation cancelled. Type 'exit' to quit or continue with a new command.")
                except Exception as e:
                    logger.error(f"Error processing command: {e}")
                    print(f"\nError: {e}")
                    print("Type '/h' for help or continue with a new command.")
        finally:
            # Cleanup
            await self._cleanup()

    async def _cleanup(self):
        """Clean up resources."""
        self.command_history.save_history()
        await self.aiohttp_session.close()
        self.db.close()
        print("Agent exited. Command history saved.")

if __name__ == "__main__":
    asyncio.run(MinimalAIAgent(model=LLM_MODEL).run())
