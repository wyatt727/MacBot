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
    SIMILARITY_THRESHOLD, DEBUG_MODE
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
    def __init__(self, model: str = LLM_MODEL, command_timeout: int = 60):
        self.model = model
        self.db = ConversationDB()
        self.last_user_query = self.db.get_setting("last_user_query") or ""
        self.aiohttp_session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=RESPONSE_TIMEOUT)
        )
        # Optimized semaphores
        self.llm_semaphore = asyncio.Semaphore(MAX_CONCURRENT_LLM_CALLS)
        self.code_semaphore = asyncio.Semaphore(MAX_CONCURRENT_CODE_EXECUTIONS)
        self.command_history = CommandHistory()
        self.session_start = datetime.now()
        self.in_comparison_mode = self.db.get_setting("in_comparison_mode") == "true"
        
        # Ollama-specific optimizations
        self.ollama_config = {
            "num_thread": os.cpu_count() or 4,  # Default to CPU count or 4
            "num_gpu": 0,  # Default to CPU-only mode
            "timeout": RESPONSE_TIMEOUT
        }
        
        # Load Ollama config from environment or settings
        if os.getenv("OLLAMA_NUM_THREAD"):
            self.ollama_config["num_thread"] = int(os.getenv("OLLAMA_NUM_THREAD"))
        if os.getenv("OLLAMA_NUM_GPU"):
            self.ollama_config["num_gpu"] = int(os.getenv("OLLAMA_NUM_GPU"))
        
        # Store default timeout for restoration after /notimeout
        self.default_timeout = RESPONSE_TIMEOUT
        
        # Performance tracking
        self.performance_metrics = {
            "llm_calls": 0,
            "cache_hits": 0,
            "total_llm_time": 0,
            "avg_llm_time": 0,
            "timeouts": 0
        }
        
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
            '/nocache': lambda x: x.replace('/nocache ', ''),
            '/notimeout': lambda x: x.replace('/notimeout ', ''),
            '/perf': self.show_performance_metrics
        }

    async def show_help(self, _: str):
        """Show help information about available commands."""
        help_text = """
╭─ Available Commands ─────────────────────────────────────────────
│
│  Navigation
│  • Up/Down Arrow : Browse history
│  • Ctrl+R       : Search history
│  • Tab          : Auto-complete
│
│  Basic Commands
│  • /h, /help    : Show this help
│  • /clear       : Clear screen
│  • /history     : Show history
│  • /stats       : Show statistics
│  • /repeat      : Repeat last command
│  • /perf        : Show performance
│
│  Advanced
│  • /success     : Manage exchanges
│  • /compare     : Compare models
│  • /nocache     : Skip cache
│  • /notimeout   : Disable timeout
│  • exit, /bye   : Exit MacBot
│
│  Tips
│  • Commands are saved between sessions
│  • Use Tab for quick completion
│  • Ctrl+C to cancel operations
│
╰──────────────────────────────────────────────────────────────────"""
        print(help_text)

    async def show_command_history(self, _: str):
        """Show the command history with timestamps."""
        history = []
        for i in range(1, readline.get_current_history_length() + 1):
            cmd = readline.get_history_item(i)
            history.append(f"│  {i:3d} │ {cmd}")
        
        if not history:
            print("\n╭─ Command History ─── Empty ─────────────────────────────")
            print("╰────────────────────────────────────────────────────────")
            return
            
        width = max(len(line) for line in history) + 2
        print("\n╭─ Command History ─" + "─" * (width - 19))
        for line in history:
            print(line + " " * (width - len(line)))
        print("╰" + "─" * width)

    async def clear_screen(self, _: str):
        """Clear the terminal screen."""
        os.system('cls' if os.name == 'nt' else 'clear')

    async def show_session_stats(self, _: str):
        """Show statistics for the current session."""
        duration = datetime.now() - self.session_start
        hours = duration.seconds // 3600
        minutes = (duration.seconds % 3600) // 60
        seconds = duration.seconds % 60
        
        duration_str = []
        if hours > 0:
            duration_str.append(f"{hours}h")
        if minutes > 0 or hours > 0:
            duration_str.append(f"{minutes}m")
        duration_str.append(f"{seconds}s")
        
        total_commands = readline.get_current_history_length()
        successful = len(self.db.list_successful_exchanges())
        
        stats = f"""
╭─ Session Statistics ────────────────────────────────────────
│
│  ⏱  Duration    : {" ".join(duration_str)}
│  ⌨️  Commands    : {total_commands}
│  ✓  Successful  : {successful}
│  🔄 Cache Hits  : {self.performance_metrics['cache_hits']}
│  ⚡ LLM Calls   : {self.performance_metrics['llm_calls']}
│
╰──────────────────────────────────────────────────────────"""
        print(stats)

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

    async def _process_code_blocks_parallel(self, blocks: List[Tuple[str, str]]) -> List[Tuple[int, str]]:
        """Process code blocks in parallel with optimized concurrency."""
        start_time = datetime.now()
        results = []
        
        if blocks:
            print("\n┌─ Code Execution " + "─" * 50)
            
        for idx, (lang, code) in enumerate(blocks, 1):
            if len(blocks) > 1:
                print(f"\n├─ Block #{idx}")
            print(f"│  {lang}")
            print("│")
            for line in code.strip().split('\n'):
                print(f"│  {line}")
            print("│")
            
            async with self.code_semaphore:
                exec_start = datetime.now()
                ret, output = await self.process_code_block(lang, code)
                exec_time = (datetime.now() - exec_start).total_seconds()
                results.append((ret, output))
                
                if output.strip():
                    print("│")
                    print("│  Result:")
                    for line in output.strip().split('\n'):
                        print(f"│  {line}")
                if DEBUG_MODE:
                    print(f"│  Time: {exec_time:.2f}s")
            print("│")
        
        if blocks:
            if DEBUG_MODE:
                total_time = (datetime.now() - start_time).total_seconds()
                print(f"└─ Total time: {total_time:.2f}s")
            else:
                print("└" + "─" * 60)
        return results

    async def show_performance_metrics(self, _: str):
        """Show detailed performance metrics."""
        metrics = f"""
╭─ Performance Metrics ──────────────────────────────────────
│
│  LLM Statistics
│  • Total Calls : {self.performance_metrics['llm_calls']}
│  • Cache Hits  : {self.performance_metrics['cache_hits']}"""

        if self.performance_metrics['llm_calls'] > 0:
            metrics += f"""
│  • Avg Time    : {self.performance_metrics['avg_llm_time']:.1f}s
│  • Total Time  : {self.performance_metrics['total_llm_time']:.1f}s"""

        metrics += f"""
│  • Timeouts    : {self.performance_metrics['timeouts']}
│
│  Ollama Configuration
│  • CPU Threads : {self.ollama_config['num_thread']}
│  • GPU Layers  : {self.ollama_config['num_gpu']}
│
╰──────────────────────────────────────────────────────────"""
        print(metrics)

    async def process_message(self, message: str, no_cache: bool = False) -> Tuple[str, bool]:
        """
        Optimized message processing using only SQLite for caching.
        Returns: Tuple[response: str, was_cached: bool]
        """
        start_time = datetime.now()
        phase = "initialization"
        
        # Check for /notimeout flag
        use_extended_timeout = False
        if message.startswith('/notimeout '):
            use_extended_timeout = True
            message = message[10:].strip()  # Remove /notimeout prefix
            original_timeout = self.ollama_config["timeout"]
            self.ollama_config["timeout"] = 0  # Disable timeout
            print("ℹ️  Timeout disabled - will wait indefinitely for response")
            
            # Also update aiohttp session timeout
            self.aiohttp_session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=0)  # 0 means no timeout
            )
            
        try:
            if not no_cache:
                phase = "database_lookup"
                similar_exchanges = self.db.find_successful_exchange(message)
                
                if similar_exchanges:
                    best_match = similar_exchanges[0]
                    similarity = best_match[2]
                    
                    if similarity >= SIMILARITY_THRESHOLD:
                        print("\n╭─ Cache Hit ─────────────────────────────────────────")
                        if similarity == 1.0:
                            print("│  ✓ Exact match found")
                        else:
                            print(f"│  ✓ Similar response found ({similarity:.1%} match)")
                            print("│")
                            print("│  Similar query:")
                            print(f"│  • {best_match[0]}")
                        
                        if DEBUG_MODE:
                            lookup_time = (datetime.now() - start_time).total_seconds()
                            print(f"│  ⏱  Lookup: {lookup_time:.2f}s")
                        
                        print("╰──────────────────────────────────────────────────────")
                        
                        self.performance_metrics["cache_hits"] += 1
                        cached_response = best_match[1]
                        
                        blocks = self.extract_code_from_response(cached_response)
                        if blocks:
                            results = await self._process_code_blocks_parallel(blocks)
                        
                        return cached_response, True
            
            phase = "llm_processing"
            print("\n╭─ Generating Response ───────────────────────────────────")
            print("│  ⟳ Processing request...")
            
            # Show similar examples even if below threshold
            if not no_cache and similar_exchanges:
                best_match = similar_exchanges[0]
                similarity = best_match[2]
                if similarity >= 0.5:  # Only show if somewhat relevant
                    print("│")
                    print("│  Similar examples found:")
                    for query, response, sim in similar_exchanges[:3]:  # Show top 3
                        match_type = "Excellent" if sim >= 0.9 else "Good" if sim >= 0.7 else "Partial"
                        preview = response.split('\n')[0][:50] + "..." if len(response) > 50 else response
                        print(f"│  • {match_type} ({sim:.1%}): '{query}' → '{preview}'")
                    print("│")
                    print("│  ℹ️  Using examples for context but generating new response")
                    print("│     (similarity below cache threshold)")
            
            if DEBUG_MODE:
                context_start = datetime.now()
            
            context = await self._build_context(message, no_cache=no_cache)
            
            if DEBUG_MODE:
                context_time = (datetime.now() - context_start).total_seconds()
                print(f"│  ⏱  Context: {context_time:.2f}s")
            
            llm_start = datetime.now()
            try:
                async with self.llm_semaphore:
                    response = await get_llm_response_async(
                        context, 
                        self.model, 
                        self.aiohttp_session,
                        num_thread=self.ollama_config["num_thread"],
                        num_gpu=self.ollama_config["num_gpu"]
                    )
                llm_time = (datetime.now() - llm_start).total_seconds()
                
                # Update performance metrics
                self.performance_metrics["llm_calls"] += 1
                self.performance_metrics["total_llm_time"] += llm_time
                self.performance_metrics["avg_llm_time"] = (
                    self.performance_metrics["total_llm_time"] / 
                    self.performance_metrics["llm_calls"]
                )
                
                if llm_time > 10:
                    print(f"│  ⚠  Slow response ({llm_time:.1f}s)")
                elif DEBUG_MODE:
                    print(f"│  ⏱  LLM: {llm_time:.1f}s")
                    
            except asyncio.TimeoutError:
                self.performance_metrics["timeouts"] += 1
                raise TimeoutError(f"Response timed out after {RESPONSE_TIMEOUT}s")
            except Exception as e:
                logger.error(f"LLM response error: {str(e)}")
                print("│")
                print("│  ❌ LLM Response Failed:")
                print(f"│  • Error: {str(e)}")
                if hasattr(e, 'response') and e.response:
                    try:
                        error_json = await e.response.json()
                        if 'error' in error_json:
                            print(f"│  • Details: {error_json['error']}")
                    except:
                        if hasattr(e, 'response'):
                            error_text = await e.response.text()
                            print(f"│  • Details: {error_text[:200]}")
                print("│")
                raise
            
            blocks = self.extract_code_from_response(response)
            
            if not self.in_comparison_mode and not no_cache and blocks:
                self.db.add_successful_exchange(message, response)
                if DEBUG_MODE:
                    print("│  ✓ Response cached")
            
            if DEBUG_MODE:
                total_time = (datetime.now() - start_time).total_seconds()
                print(f"│  ⏱  Total: {total_time:.2f}s")
            print("╰──────────────────────────────────────────────────────")
            
            # Execute code blocks immediately for non-cached responses
            if blocks:
                await self._process_code_blocks_parallel(blocks)
            
            return response, False

        except Exception as e:
            total_time = (datetime.now() - start_time).total_seconds()
            print("\n╭─ Error ────────────────────────────────────────────────")
            print(f"│  ❌ {phase}: {str(e)}")
            if DEBUG_MODE:
                print(f"│  ⏱  Time: {total_time:.1f}s")
            print("╰──────────────────────────────────────────────────────")
            logger.error(f"Error in {phase}: {e}")
            return f"❌ Error in {phase}: {str(e)}", False

        finally:
            # Restore original timeout if it was changed
            if use_extended_timeout:
                self.ollama_config["timeout"] = original_timeout
                # Restore aiohttp session with default timeout
                await self.aiohttp_session.close()
                self.aiohttp_session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=self.default_timeout)
                )

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
        print("\n⟳ Launching Success DB GUI...")
        try:
            proc = subprocess.Popen([sys.executable, "-m", "agent.gui_success"])
            if proc.pid:
                print(f"✓ GUI launched (PID: {proc.pid})")
            else:
                print("❌ Failed to launch GUI: No PID returned")
        except Exception as e:
            print(f"❌ Failed to launch GUI: {e}")

    async def compare_models_command(self, command: str):
        """
        Triggers model comparison mode for the last query or a new query.
        Responses from comparison mode are not saved as successful exchanges.
        """
        from .model_comparator import compare_models, analyze_results
        
        parts = command.split(maxsplit=1)
        prompt = parts[1] if len(parts) > 1 else self.last_user_query
        
        if not prompt:
            print("\n❌ Please enter a prompt first or provide one with the command")
            return
            
        try:
            print(f"\n╭─ Model Comparison ────────────────────────────────────")
            print(f"│  Prompt: {prompt}")
            print(f"│  Status: Running comparison across available models...")
            print(f"├────────────────────────────────────────────────────────")
            
            self.in_comparison_mode = True
            self.db.set_setting("in_comparison_mode", "true")
            try:
                results = await compare_models(prompt, self.aiohttp_session, self.db)
            finally:
                self.in_comparison_mode = False
                self.db.set_setting("in_comparison_mode", "false")
            
            if not results:
                print("│  ❌ No results returned")
                print("╰────────────────────────────────────────────────────────")
                return
                
            analysis = analyze_results(results)
            print(analysis)
            print("╰────────────────────────────────────────────────────────")
            
        except Exception as e:
            logger.error(f"Error during model comparison: {e}")
            print(f"│  ❌ Error: {e}")
            print("╰────────────────────────────────────────────────────────")

    async def run(self):
        """Enhanced main loop with optimized processing."""
        print("""
╭─ MacBot AI Agent ─────────────────────────────────────────────
│
│  Welcome! Type /h for help, or just start chatting.
│  Use arrow keys ↑↓ to browse history, Tab for completion.
│
╰──────────────────────────────────────────────────────────────""")
        
        try:
            while True:
                try:
                    user_input = await asyncio.to_thread(input, "\n❯ ")
                    
                    if user_input.lower() in ("exit", "quit", "/bye"):
                        print("\n✓ Goodbye!")
                        break

                    if not user_input.strip():
                        if self.last_user_query:
                            print("ℹ️  Repeating last query...")
                            user_input = self.last_user_query
                        else:
                            continue

                    no_cache = False
                    use_extended_timeout = False
                    
                    # Handle command flags
                    if user_input.startswith('/nocache '):
                        no_cache = True
                        user_input = user_input[9:].strip()
                        print("ℹ️  Cache disabled for this query")
                    
                    if user_input.startswith('/notimeout '):
                        use_extended_timeout = True
                        user_input = user_input[10:].strip()
                        original_timeout = self.ollama_config["timeout"]
                        self.ollama_config["timeout"] = 0  # Disable timeout
                        print("ℹ️  Timeout disabled - will wait indefinitely for response")
                    
                    try:
                        if user_input.startswith('/'):
                            command = user_input.split()[0].lower()
                            if command in self.command_aliases:
                                if command not in ['/nocache', '/notimeout']:
                                    result = await self.command_aliases[command](user_input)
                                    if isinstance(result, str):
                                        user_input = result
                                    else:
                                        continue

                        self.command_history.add_command(user_input)
                        self.db.set_setting("last_user_query", user_input)
                        self.last_user_query = user_input

                        response, was_cached = await self.process_message(user_input, no_cache=no_cache)
                        
                        # Store the interaction in the database
                        self.db.add_message("user", user_input)
                        self.db.add_message("assistant", response)
                    
                    finally:
                        # Restore timeout if it was changed
                        if use_extended_timeout:
                            self.ollama_config["timeout"] = original_timeout

                except KeyboardInterrupt:
                    print("\n\n⨯ Operation cancelled")
                    continue
                except Exception as e:
                    logger.error(f"Error: {e}")
                    print(f"\n❌ Error: {str(e)}")
        finally:
            await self._cleanup()

    async def _cleanup(self):
        """Clean up resources."""
        print("""
╭─ Session Summary ──────────────────────────────────────────────
│
│  ✓ History saved
│  ✓ Database closed
│  ✓ Resources cleaned up
│
╰──────────────────────────────────────────────────────────────""")
        self.command_history.save_history()
        await self.aiohttp_session.close()
        self.db.close()

if __name__ == "__main__":
    asyncio.run(MinimalAIAgent(model=LLM_MODEL).run())
