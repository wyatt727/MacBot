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
    SIMILARITY_THRESHOLD, DEBUG_MODE, SAVE_CODE_BLOCKS
)
import json
import urllib.parse
import html
import re
from concurrent.futures import ThreadPoolExecutor
import hashlib
import functools

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a search cache
_SEARCH_CACHE = {}  # Query hash -> (results, timestamp)
_SEARCH_CACHE_TTL = 3600  # 1 hour cache TTL
_BACKGROUND_TASKS = set()  # Keep track of background tasks

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
    
    Features:
    - Web search integration for up-to-date information
    - Command history with navigation
    - Model switching at runtime
    - Performance metrics and diagnostics
    - Conversation history management
    """
    def __init__(self, 
                 model: str = LLM_MODEL, 
                 timeout: int = RESPONSE_TIMEOUT,
                 max_llm_calls: int = MAX_CONCURRENT_LLM_CALLS,
                 max_code_execs: int = MAX_CONCURRENT_CODE_EXECUTIONS,
                 debug_mode: bool = DEBUG_MODE,
                 save_code: bool = SAVE_CODE_BLOCKS,
                 command_timeout: int = 60):
        self.model = model
        self.db = ConversationDB()
        self.last_user_query = self.db.get_setting("last_user_query") or ""
        
        # Store the timeout value
        self.default_timeout = timeout
        self.aiohttp_session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=timeout)
        )
        
        # Apply configuration settings
        global DEBUG_MODE, SAVE_CODE_BLOCKS
        DEBUG_MODE = debug_mode
        SAVE_CODE_BLOCKS = save_code
        
        # Optimized semaphores
        self.llm_semaphore = asyncio.Semaphore(max_llm_calls)
        self.code_semaphore = asyncio.Semaphore(max_code_execs)
        self.command_history = CommandHistory()
        self.session_start = datetime.now()
        self.in_comparison_mode = self.db.get_setting("in_comparison_mode") == "true"
        
        # Ollama-specific optimizations
        self.ollama_config = {
            "num_thread": os.cpu_count() or 4,  # Default to CPU count or 4
            "num_gpu": 0,  # Initialize to 0, will be set below
            "timeout": timeout
        }
        
        # Auto-detect GPU capabilities
        auto_gpu_layers = self._detect_gpu_capabilities()
        if auto_gpu_layers > 0:
            logger.info(f"Auto-detected GPU capabilities: recommended {auto_gpu_layers} layers")
            
        # Load Ollama config from environment, settings, or auto-detection
        if os.getenv("OLLAMA_NUM_THREAD"):
            self.ollama_config["num_thread"] = int(os.getenv("OLLAMA_NUM_THREAD"))
        elif self.db.get_setting("ollama_num_thread"):
            try:
                self.ollama_config["num_thread"] = int(self.db.get_setting("ollama_num_thread"))
            except (ValueError, TypeError):
                pass  # Use default if invalid
                
        if os.getenv("OLLAMA_NUM_GPU"):
            self.ollama_config["num_gpu"] = int(os.getenv("OLLAMA_NUM_GPU"))
        elif self.db.get_setting("ollama_num_gpu"):
            try:
                self.ollama_config["num_gpu"] = int(self.db.get_setting("ollama_num_gpu"))
            except (ValueError, TypeError):
                # Use auto-detected GPU if available
                self.ollama_config["num_gpu"] = auto_gpu_layers
        else:
            # No environment or DB setting, use auto-detected GPU
            self.ollama_config["num_gpu"] = auto_gpu_layers
        
        # Performance metrics
        self.perf_metrics = {
            "avg_response_time": 0,
            "total_response_time": 0,
            "total_tokens": 0,
            "requests_count": 0,
            "tokens_per_second": 0,
            "cache_hits": 0,
            "timeouts": 0
        }
        
        # Store default timeout for restoration after /notimeout
        self.default_timeout = timeout
        
        # Initialize command aliases and shortcuts
        self.command_aliases = {
            'h': self.show_help,
            'help': self.show_help,
            'history': self.show_command_history,
            'clear': self.clear_screen,
            'stats': self.show_session_stats,
            'repeat': self.repeat_last_command,
            'success': self.process_success_command,
            'compare': self.compare_models_command,
            'search': lambda x: self.web_search(x) if x else "Please provide a search query",
            'web': lambda x: self.web_search(x) if x else "Please provide a search query",
            'perf': self.show_performance_metrics,
            'model': self.set_model,
            'exit': lambda _: "exit",
            'quit': lambda _: "exit",
            'bye': lambda _: "exit",
            'threads': self.set_threads,
            'gpu': self.set_gpu_layers
        }

    async def show_help(self, args: str = ""):
        """
        Display help information for using the agent.
        
        Args:
            args: Optional specific command to get help on
        """
        if args:
            # Show help for a specific command
            cmd = args.lower().strip().split()[0] if args.split() else ""
            if cmd in ["search", "web"]:
                return """
╭─ Web Search Help ─────────────────────────────────────────
│
│  SEARCH COMMANDS:
│  • /search [query]  - Search Wikipedia for information
│  • /web [query]     - Alias for /search
│  • ?[query]         - Quick search shortcut (same as /search)
│  • ?![query]        - TURBO search (ultra-fast, fewer results)
│
│  NOTE: Web search is limited to Wikipedia in this local environment.
│
│  Examples:
│    /search python programming
│    ?artificial intelligence
│
╰──────────────────────────────────────────────────────────"""
            
            elif cmd == "history":
                return """
╭─ History Command Help ────────────────────────────────────
│
│  Usage: /history [subcommand] [arguments]
│
│  Subcommands:
│  • (none)       - Show most recent conversation history
│  • search [term] - Search conversation history for a term
│  • clear        - Clear all conversation history (requires confirmation)
│  • save [file]  - Save conversation history to a file
│
│  Examples:
│    /history
│    /history search python
│    /history clear
│    /history save my_conversation.txt
│
╰──────────────────────────────────────────────────────────"""
            
            elif cmd == "model":
                return """
╭─ Model Command Help ───────────────────────────────────────
│
│  Usage: /model [model_name]
│
│  Without arguments, shows the current model being used.
│  With a model name, switches to using that model for responses.
│
│  Example: /model llama3:8b
│
╰───────────────────────────────────────────────────────────"""
            
            elif cmd == "perf":
                return """
╭─ Performance Metrics Help ─────────────────────────────────
│
│  Usage: /perf
│
│  Displays detailed performance metrics including:
│  • LLM Statistics (calls, cache hits, response times)
│  • Ollama Configuration (CPU threads, GPU layers)
│
│  Related commands:
│  • /threads - Change CPU thread count
│  • /gpu     - Change GPU layer count
│
╰───────────────────────────────────────────────────────────"""
            
            elif cmd == "notimeout":
                return """
╭─ No Timeout Command Help ─────────────────────────────────
│
│  Usage: /notimeout [your query]
│
│  Disables the response timeout for this specific query.
│  Useful for complex tasks that may take longer to process.
│
│  Example: /notimeout lets play a game of chess
│
╰───────────────────────────────────────────────────────────"""
            
            elif cmd == "nocache":
                return """
╭─ No Cache Command Help ──────────────────────────────────
│
│  Usage: /nocache [your query]
│
│  Bypasses the response cache and forces a new LLM response.
│  Useful when you want a fresh answer ignoring cached results.
│
│  Example: /nocache what's the current time?
│
╰──────────────────────────────────────────────────────────"""
                
            elif cmd == "clear":
                return """
╭─ Clear Command Help ──────────────────────────────────────
│
│  Usage: /clear
│
│  Clears the terminal screen.
│  This command has no arguments.
│
╰──────────────────────────────────────────────────────────"""
                
            elif cmd == "stats":
                return """
╭─ Stats Command Help ───────────────────────────────────────
│
│  Usage: /stats
│
│  Displays session statistics including:
│  • Commands executed
│  • Session duration
│  • Memory usage
│
╰───────────────────────────────────────────────────────────"""
                
            elif cmd == "repeat":
                return """
╭─ Repeat Command Help ───────────────────────────────────────
│
│  Usage: /repeat
│
│  Repeats the last command or query.
│  This command has no arguments.
│
╰───────────────────────────────────────────────────────────"""
                
            elif cmd == "success":
                return """
╭─ Success Command Help ──────────────────────────────────────
│
│  Usage: /success
│
│  Launches the GUI for managing successful exchanges.
│  This command has no arguments.
│
╰───────────────────────────────────────────────────────────"""
                
            elif cmd == "compare":
                return """
╭─ Compare Command Help ──────────────────────────────────────
│
│  Usage: /compare [prompt]
│
│  Compares responses from different models for the same prompt.
│  If no prompt is provided, uses the last query.
│
│  Example: /compare explain quantum computing
│
╰───────────────────────────────────────────────────────────"""
            
            elif cmd == "threads":
                return """
╭─ CPU Threads Help ───────────────────────────────────────
│
│  Usage: /threads [number]
│
│  Change the number of CPU threads used by Ollama for inference.
│  Higher values may improve performance but use more resources.
│
│  Examples:
│    /threads          - Show current thread count
│    /threads 8        - Set to use 8 threads
│    /threads 16       - Set to use 16 threads
│
│  NOTE: Changes apply to future LLM requests only.
│
╰──────────────────────────────────────────────────────────"""
            
            elif cmd == "gpu":
                return """
╭─ GPU Layers Help ───────────────────────────────────────
│
│  Usage: /gpu [number]
│
│  Change the number of GPU layers used by Ollama for inference.
│  Higher values may improve performance but use more resources.
│
│  Examples:
│    /gpu          - Show current GPU layer count
│    /gpu 8        - Set to use 8 layers
│    /gpu 16       - Set to use 16 layers
│
│  NOTE: Changes apply to future LLM requests only.
│
╰──────────────────────────────────────────────────────────"""
            
            else:
                return f"No help available for '{cmd}'. Use /help to see all commands."
        
        # General help
        return """
╭─ MacBot AI Agent Help ───────────────────────────────────────
│
│  Command Reference:
│
│  Web Integration:
│    /search [query] - Search Wikipedia for information
│    /web [query]    - Alias for /search
│    ?[query]        - Quick search shortcut 
│    ?![query]       - TURBO search (ultra-fast)
│
│  Model & Performance:
│    /model [name]   - View or change LLM model
│    /threads [num]  - View or change CPU thread count
│    /gpu [num]      - View or change GPU layer count
│    /perf           - Show performance metrics
│
│  Session Management:
│    /clear          - Clear the screen
│    /stats          - Show session statistics
│    /repeat         - Repeat last command
│
│  Options:
│    /nocache        - Skip cache for next query
│    /notimeout      - Disable timeout for next query
│
│  General:
│    /help [command] - Show help for all or specific command
│    /exit or /quit  - Exit the application
│
│  Tip: Use Tab for command completion!
│
╰───────────────────────────────────────────────────────────"""

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
│  🔄 Cache Hits  : {self.perf_metrics['requests_count'] - self.perf_metrics['requests_count']}
│  ⚡ LLM Calls   : {self.perf_metrics['requests_count']}
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
                else:
                    print("│")
                    print("│  No output")
                    
                if DEBUG_MODE:
                    print(f"│  Time: {exec_time:.2f}s")
            print("│")
        
        if blocks:
            if DEBUG_MODE:
                total_time = (datetime.now() - start_time).total_seconds()
                print(f"└─ Total execution time: {total_time:.2f}s")
            else:
                print("└" + "─" * 64)
        
        return results

    async def show_performance_metrics(self, _: str):
        """Show detailed performance metrics."""
        metrics = f"""
╭─ Performance Metrics ──────────────────────────────────────
│
│  LLM Statistics
│  • Total Calls : {self.perf_metrics['requests_count']}
│  • Cache Hits  : {self.perf_metrics['cache_hits']}"""

        if self.perf_metrics['requests_count'] > 0:
            metrics += f"""
│  • Avg Time    : {self.perf_metrics['avg_response_time']:.1f}s
│  • Total Time  : {self.perf_metrics['total_response_time']:.1f}s"""

        metrics += f"""
│  • Timeouts    : {self.perf_metrics['timeouts']}
│
│  Ollama Configuration
│  • CPU Threads : {self.ollama_config['num_thread']}
│  • GPU Layers  : {self.ollama_config['num_gpu']}
│
╰──────────────────────────────────────────────────────────"""
        
        # Return the metrics string instead of printing it
        return metrics

    async def process_message(self, message: str, no_cache: bool = False) -> Tuple[str, bool]:
        """
        Process user message and return a response.
        Returns: Tuple[response: str, was_cached: bool]
        """
        start_time = datetime.now()
        phase = "initialization"
        
        # Store for command completion context
        self.current_input = message
        
        # Turbo search mode with ?! prefix (ultra-fast search)
        if message.startswith('?!'):
            search_query = message[2:].strip()
            return await self._turbo_search(search_query), False
        
        # Standard quick search with ? prefix
        elif message.startswith('?'):
            search_query = message[1:].strip()
            search_results = await self.web_search(search_query)
            return search_results, False
        
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
        
        # Extract command if message starts with /
        command, args = self._extract_command_and_args(message)
        if command:
            # Handle direct command aliases (functions that return values)
            if command in ['search', 'web']:
                search_results = await self.web_search(args)
                return search_results, False
            
            # Handle built-in commands
            if command == "history":
                history_response = await self._handle_history_command(command, args)
                if history_response:
                    return history_response, False
            
            # Handle model switching
            if command == "model":
                if not args:
                    return f"Current model: {self.model}\nUse /model [model_name] to switch models", False
                self.model = args
                return f"Model switched to {self.model}", False
            
            # Handle performance command
            if command == "perf":
                await self.show_performance_metrics(args)
                return "Performance metrics displayed above", False
        
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
                        
                        self.perf_metrics["requests_count"] += 1
                        self.perf_metrics["cache_hits"] += 1
                        cached_response = best_match[1]
                        
                        blocks = self.extract_code_from_response(cached_response)
                        if blocks:
                            results = await self._process_code_blocks_parallel(blocks)
                            # Return an empty string to avoid duplicating the code output
                            return "", True
                        
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
                    print(f"│  • {similarity*100:.1f}%: '{best_match[0]}' → '{best_match[1][:50]}...'")
                    print("│")
                    print("│  ℹ️  Using examples for context but generating new response")
                    print("│     (similarity below cache threshold)")
            
            if DEBUG_MODE:
                context_start = datetime.now()
            
            context = await self._build_context(message, no_cache)
            
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
                        num_gpu=self.ollama_config["num_gpu"],
                        timeout=self.ollama_config["timeout"]
                    )
                llm_time = (datetime.now() - llm_start).total_seconds()
                
                # Update performance metrics
                self.perf_metrics["requests_count"] += 1
                self.perf_metrics["total_response_time"] += llm_time
                self.perf_metrics["total_tokens"] += llm_time
                self.perf_metrics["avg_response_time"] = (
                    self.perf_metrics["total_response_time"] / 
                    self.perf_metrics["requests_count"]
                )
                
                if llm_time > 10:
                    print(f"│  ⚠  Slow response ({llm_time:.1f}s)")
                elif DEBUG_MODE:
                    print(f"│  ⏱  LLM: {llm_time:.1f}s")
                    
            except asyncio.TimeoutError:
                self.perf_metrics["requests_count"] += 1
                self.perf_metrics["timeouts"] += 1
                raise TimeoutError(f"Response timed out after {self.ollama_config['timeout']}s")
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
                # Return an empty string to avoid duplicating code blocks
                return "", False
            
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

    async def web_search(self, query: str, num_results: int = 5, fast_mode: bool = True) -> str:
        """
        Web search with fallback for limited local environments.
        
        Features:
        - Result caching with TTL
        - Wikipedia search as primary source
        - Smart error handling
        - Graceful degradation
        
        Args:
            query: Search query
            num_results: Max results to return
            fast_mode: Use speed optimizations
            
        Returns:
            Formatted search results or advisory message
        """
        # Start timing
        start_time = datetime.now()
        
        if not query:
            return "Please provide a search query."
        
        # Clean and normalize query
        query = query.strip()
        search_hash = hashlib.md5(query.lower().encode()).hexdigest()
        
        # Check cache for recent results
        if search_hash in _SEARCH_CACHE:
            cached_results, timestamp = _SEARCH_CACHE[search_hash]
            if datetime.now() - timestamp < timedelta(seconds=_SEARCH_CACHE_TTL):
                elapsed = (datetime.now() - start_time).total_seconds()
                return f"🚀 Results for '{query}' (cached in {elapsed:.2f}s):\n\n{cached_results}"
        
        print(f"🔍 Searching for: {query}")
        
        try:
            # Only use Wikipedia as it's more reliable
            wiki_results = await self._search_wikipedia(query, num_results)
            
            if not wiki_results:
                return f"""
╭─ Web Search Limited ───────────────────────────────────────
│
│  No results found for: "{query}"
│
│  Note: Web search capabilities are limited in this local environment.
│  For complex searches, consider:
│  • Using more specific keywords
│  • Asking the assistant directly about general knowledge topics
│  • Using a full web browser for detailed research
│
╰───────────────────────────────────────────────────────────"""
            
            # Format results
            formatted_results = ""
            for i, result in enumerate(wiki_results, 1):
                title = result.get('title', 'No title')
                link = result.get('link', 'No link')
                snippet = result.get('snippet', 'No description')
                
                formatted_results += f"{i}. {title}\n"
                formatted_results += f"   🔗 {link}\n"
                formatted_results += f"   {snippet}\n\n"
            
            # Cache the results
            _SEARCH_CACHE[search_hash] = (formatted_results, datetime.now())
            
            # Return results with timing info
            elapsed = (datetime.now() - start_time).total_seconds()
            return f"""
╭─ Wikipedia Search Results ({elapsed:.2f}s) ──────────────────────
│
│  Results for: "{query}"
│
{formatted_results}│
│  Note: Web search capabilities are limited to Wikipedia in this environment.
│
╰───────────────────────────────────────────────────────────"""
            
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return f"""
╭─ Web Search Limited ───────────────────────────────────────
│
│  Unable to search for: "{query}"
│
│  The web search feature has limited functionality in this local environment.
│  Error: {str(e)}
│
│  For general knowledge questions, try asking the assistant directly.
│  For up-to-date information, using a full web browser is recommended.
│
╰───────────────────────────────────────────────────────────"""

    async def _search_wikipedia(self, query: str, num_results: int = 5) -> List[Dict]:
        """Search Wikipedia using its API."""
        try:
            encoded_query = urllib.parse.quote(query)
            url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={encoded_query}&format=json&utf8=1&srlimit={num_results}"
            
            async with self.aiohttp_session.get(url, timeout=5) as response:
                if response.status != 200:
                    logger.error(f"Wikipedia API error: {response.status}")
                    return []
                
                data = await response.json()
                results = []
                
                for item in data.get('query', {}).get('search', []):
                    title = html.unescape(item.get('title', ''))
                    snippet = html.unescape(re.sub(r'<.*?>', '', item.get('snippet', '')))
                    url = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
                    results.append({
                        'title': title,
                        'link': url,
                        'snippet': snippet,
                        'source': 'wikipedia'
                    })
                return results
                
        except Exception as e:
            logger.error(f"Wikipedia search error: {e}")
            return []

    async def _turbo_search(self, query: str, num_results: int = 3) -> str:
        """
        Simplified turbo search that works with limited resources.
        Just uses Wikipedia with fewer results for speed.
        """
        if not query:
            return "Please provide a search query."
            
        try:
            print(f"⚡ TURBO searching for: {query}")
            return await self.web_search(query, num_results=num_results, fast_mode=True)
        except Exception as e:
            logger.error(f"Turbo search error: {e}")
            return f"""
╭─ Turbo Search Limited ───────────────────────────────────
│
│  Unable to perform turbo search for: "{query}"
│  Error: {str(e)}
│
│  Try asking your question directly to the assistant instead.
│
╰─────────────────────────────────────────────────────────"""

    async def set_model(self, model_name: str) -> str:
        """Set the LLM model to use."""
        if not model_name.strip():
            return f"Current model: {self.model}"
        
        # Trim whitespace and validate
        new_model = model_name.strip()
        
        # Store the original model in case we need to revert
        original_model = self.model
        
        try:
            # Set the new model
            self.model = new_model
            
            # Build a minimal test context
            test_context = [{"role": "user", "content": "test"}]
            
            # Try a minimal request to validate the model
            try:
                await get_llm_response_async(
                    test_context,
                    self.model,
                    self.aiohttp_session,
                    num_thread=self.ollama_config["num_thread"],
                    num_gpu=self.ollama_config["num_gpu"],
                    timeout=5  # Short timeout for testing
                )
                
                # If we get here, the model is valid
                return f"Model successfully switched to {self.model}"
            except Exception as request_error:
                error_message = str(request_error).lower()
                
                # If the error contains "model not found", try with :latest suffix
                if "model not found" in error_message and ":" not in new_model:
                    try:
                        model_with_latest = f"{new_model}:latest"
                        self.model = model_with_latest
                        
                        await get_llm_response_async(
                            test_context,
                            self.model,
                            self.aiohttp_session,
                            num_thread=self.ollama_config["num_thread"],
                            num_gpu=self.ollama_config["num_gpu"],
                            timeout=5  # Short timeout for testing
                        )
                        
                        # If we get here, the model with :latest suffix is valid
                        return f"Model successfully switched to {self.model}"
                    except Exception as latest_error:
                        # Both original name and with :latest suffix failed
                        self.model = original_model
                        raise RuntimeError(f"Model '{new_model}' and '{model_with_latest}' not found.") from latest_error
                else:
                    # Other error, not related to model not found
                    raise request_error
            
        except Exception as e:
            # Revert to the original model
            self.model = original_model
            error_message = str(e)
            
            if "model not found" in error_message.lower():
                # Get available models
                try:
                    available_models = await self._get_available_models()
                    models_text = ", ".join(available_models[:10])
                    if len(available_models) > 10:
                        models_text += f" and {len(available_models) - 10} more"
                    
                    return f"Error: Model '{new_model}' not found. Available models: {models_text}. Current model is still {self.model}."
                except:
                    return f"Error: Model '{new_model}' not found. Current model is still {self.model}."
            else:
                return f"Error switching model: {error_message}. Current model is still {self.model}."
                
    async def _get_available_models(self) -> List[str]:
        """Get a list of available Ollama models."""
        try:
            # Execute ollama list command
            process = await asyncio.create_subprocess_exec(
                "ollama", "list",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await process.communicate()
            
            # Parse the output
            output = stdout.decode().strip().split("\n")
            
            # Skip the header line and extract model names
            if len(output) > 1:
                models = []
                for line in output[1:]:  # Skip header
                    parts = line.split()
                    if parts:
                        models.append(parts[0])
                return models
            return []
        except Exception as e:
            logger.error(f"Error getting available models: {e}")
            return []

    async def set_threads(self, thread_count: str) -> str:
        """Set the number of CPU threads for Ollama.
        
        Args:
            thread_count: String containing the number of threads to use
                          If empty, returns the current thread count
        
        Returns:
            A confirmation message
        """
        # If no argument, return current setting
        if not thread_count.strip():
            return f"Current CPU thread count: {self.ollama_config['num_thread']}"
        
        # Try to parse the thread count
        try:
            new_thread_count = int(thread_count.strip())
            if new_thread_count <= 0:
                return f"Error: Thread count must be positive. Current count: {self.ollama_config['num_thread']}"
                
            # Set the new thread count
            self.ollama_config['num_thread'] = new_thread_count
            
            # Save to settings DB for persistence
            self.db.set_setting("ollama_num_thread", str(new_thread_count))
            
            return f"CPU thread count set to {new_thread_count}"
        except ValueError:
            return f"Error: '{thread_count}' is not a valid number. Current count: {self.ollama_config['num_thread']}"

    def _setup_autocomplete(self):
        """Set up command autocompletion for the agent."""
        readline.set_completer(self._command_completer)
        readline.parse_and_bind("tab: complete")
        
        # Set up known commands for autocompletion
        self.commands = [
            "/search", "/web", "/history", "/model", "/perf", "/notimeout", 
            "/help", "/exit", "/quit", "/bye"
        ]
        
        self.history_commands = [
            "search", "clear", "save"
        ]
        
        # Cache common shell commands
        self._update_shell_commands()

    def _update_shell_commands(self):
        """
        Update the list of available shell commands for tab completion.
        Gets common shell commands from PATH.
        """
        try:
            # Start with built-in commands
            self.shell_commands = []
            
            # Add commands from standard directories in PATH
            path_dirs = os.environ.get('PATH', '').split(os.pathsep)
            for dir_path in path_dirs:
                if os.path.exists(dir_path):
                    self.shell_commands.extend([
                        cmd for cmd in os.listdir(dir_path)
                        if os.path.isfile(os.path.join(dir_path, cmd)) and 
                        os.access(os.path.join(dir_path, cmd), os.X_OK)
                    ])
            
            # Remove duplicates and sort
            self.shell_commands = sorted(set(self.shell_commands))
            
        except Exception as e:
            logger.error(f"Error updating shell commands: {e}")
            self.shell_commands = []

    def _command_completer(self, text, state):
        """
        Custom completer function for readline that completes:
        1. Agent commands (starting with /)
        2. Subcommands for known agent commands
        3. Shell commands if not an agent command
        """
        # Check if we're completing an agent command
        if text.startswith("/"):
            options = [cmd for cmd in self.commands if cmd.startswith(text)]
            return options[state] if state < len(options) else None
        
        # Check if we're completing a history subcommand
        if self.current_input and self.current_input.startswith("/history "):
            remaining = self.current_input[9:].lstrip()
            if not " " in remaining:  # No subcommand argument yet
                options = [subcmd for subcmd in self.history_commands if subcmd.startswith(text)]
                return options[state] if state < len(options) else None
            
        # Default to shell command completion
        options = [cmd for cmd in self.shell_commands if cmd.startswith(text)]
        return options[state] if state < len(options) else None

    async def run(self):
        """Enhanced main loop with optimized processing and autocompletion."""
        
        # Setup command history and autocompletion
        self.command_history.load_history()
        self._setup_autocomplete()
        
        print("""
╭─ MacBot AI Agent ──────────────────────────────────────────
│
│  Local Mode (CPU-optimized) - Type '/help' for commands, '/exit' to quit
│
╰──────────────────────────────────────────────────────────""")

        try:
            while True:
                try:
                    user_input = await asyncio.to_thread(input, "\n❯ ")
                    
                    if user_input.lower() in ("/exit", "/quit", "/bye", "exit", "quit"):
                        print("\n✓ Goodbye!")
                        break

                    if not user_input.strip():
                        if self.last_user_query:
                            print("ℹ️  Repeating last query...")
                            user_input = self.last_user_query
                        else:
                            continue

                    # Process flags first
                    no_cache = False
                    use_extended_timeout = False
                    original_timeout = None
                    
                    # Handle /nocache flag
                    if user_input.startswith('/nocache '):
                        no_cache = True
                        user_input = user_input[9:].strip()
                        print("ℹ️  Cache disabled for this query")
                    
                    # Handle /notimeout flag
                    if user_input.startswith('/notimeout '):
                        use_extended_timeout = True
                        user_input = user_input[10:].strip()
                        original_timeout = self.ollama_config["timeout"]
                        self.ollama_config["timeout"] = 0  # Disable timeout
                        # Update aiohttp session timeout
                        await self.aiohttp_session.close()
                        self.aiohttp_session = aiohttp.ClientSession(
                            timeout=aiohttp.ClientTimeout(total=0)  # 0 means no timeout
                        )
                        print("ℹ️  Timeout disabled - will wait indefinitely for response")
                    
                    try:
                        # Then process other commands
                        if user_input.startswith('/'):
                            command, args = self._extract_command_and_args(user_input)
                            if command in self.command_aliases:
                                handler = self.command_aliases[command]
                                if callable(handler):
                                    if asyncio.iscoroutinefunction(handler) or isinstance(handler, functools.partial) and asyncio.iscoroutinefunction(handler.func):
                                        # Async handler
                                        result = await handler(args)
                                    else:
                                        # Sync handler (usually a lambda)
                                        result = handler(args)
                                    
                                    if result is not None:
                                        print(result)
                                        continue

                        # Process quick search shortcuts
                        if user_input.startswith('?!') or user_input.startswith('?'):
                            result, _ = await self.process_message(user_input)
                            print(result)
                            continue

                        self.command_history.add_command(user_input)
                        self.db.set_setting("last_user_query", user_input)
                        self.last_user_query = user_input

                        response, was_cached = await self.process_message(user_input, no_cache=no_cache)
                        
                        # Store the interaction in the database if not a command
                        if not user_input.startswith('/'):
                            self.db.add_message("user", user_input)
                            # Only add non-empty responses to the database
                            if response:
                                self.db.add_message("assistant", response)
                        
                        # Only print response if it's not empty (empty means code block was already processed)
                        if response:
                            print(response)
                    
                    finally:
                        # Restore timeout if it was changed
                        if use_extended_timeout and original_timeout is not None:
                            self.ollama_config["timeout"] = original_timeout
                            # Restore aiohttp session with default timeout
                            await self.aiohttp_session.close()
                            self.aiohttp_session = aiohttp.ClientSession(
                                timeout=aiohttp.ClientTimeout(total=self.default_timeout)
                            )

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

    async def _handle_history_command(self, command: str, args: str) -> Optional[str]:
        """
        Handle history-related commands.
        
        Commands:
        - /history - Show recent conversation history
        - /history search [query] - Search conversation history
        - /history clear - Clear all conversation history
        - /history save [filename] - Save history to a file
        
        Args:
            command: The command (should be 'history')
            args: Arguments for the history command
            
        Returns:
            Response message or None if command not handled
        """
        if command != "history":
            return None
        
        if not args:
            # Show recent history
            messages = self.db.get_conversation_history(limit=10)
            if not messages:
                return "No conversation history found."
            
            history = "Recent Conversation History:\n\n"
            for i, msg in enumerate(messages, 1):
                role = msg["role"].capitalize()
                content = msg["content"]
                # Truncate long messages
                if len(content) > 100:
                    content = content[:100] + "..."
                history += f"{i}. {role}: {content}\n\n"
            
            return history
        
        # Parse subcommands
        parts = args.split(maxsplit=1)
        subcommand = parts[0].lower() if parts else ""
        subargs = parts[1] if len(parts) > 1 else ""
        
        if subcommand == "search":
            if not subargs:
                return "Please provide a search query."
            
            messages = self.db.search_conversation(subargs)
            if not messages:
                return f"No messages found matching '{subargs}'."
            
            results = f"Search Results for '{subargs}':\n\n"
            for i, msg in enumerate(messages, 1):
                role = msg["role"].capitalize()
                content = msg["content"]
                # Highlight the search term
                content = content.replace(subargs, f"**{subargs}**")
                results += f"{i}. {role}: {content[:100]}...\n\n"
            
            return results
            
        elif subcommand == "clear":
            # Confirm before clearing
            if subargs == "confirm":
                self.db.clear_conversation_history()
                return "Conversation history has been cleared."
            else:
                return "To confirm clearing all conversation history, use '/history clear confirm'"
            
        elif subcommand == "save":
            filename = subargs or f"macbot_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            
            try:
                messages = self.db.get_conversation_history(limit=100)
                with open(filename, "w") as f:
                    f.write(f"MacBot Conversation History - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    for msg in messages:
                        role = msg["role"].capitalize()
                        content = msg["content"]
                        f.write(f"{role}: {content}\n\n")
                    
                return f"Conversation history saved to {filename}"
            except Exception as e:
                return f"Error saving conversation history: {str(e)}"
            
        return f"Unknown history subcommand: {subcommand}. Available: search, clear, save"

    def _detect_gpu_capabilities(self) -> int:
        """
        Auto-detect GPU capabilities and return recommended number of GPU layers.
        
        For Apple Silicon:
        - M1: ~16-24 layers recommended
        - M2: ~24-32 layers recommended
        - M3: ~32-48 layers recommended
        - External GPUs: Varies by model
        
        Returns:
            int: Recommended number of GPU layers (0 if no GPU detected)
        """
        try:
            # Check if we're on macOS and have a GPU
            if sys.platform != 'darwin':
                return 0  # Non-macOS platforms default to CPU mode for now
                
            # Run system_profiler to get GPU information
            process = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"], 
                capture_output=True, 
                text=True, 
                timeout=5
            )
            
            output = process.stdout.lower()
            
            # Check for Apple Silicon
            if "apple m1" in output:
                return 16  # Conservative default for M1
            elif "apple m2" in output:
                return 24  # Conservative default for M2
            elif "apple m3" in output:
                return 32  # Conservative default for M3
            elif "apple" in output and "metal" in output:
                return 16  # Other Apple Silicon, conservative default
                
            # Check for external GPU or Intel Mac with discrete GPU
            if "amd" in output or "nvidia" in output or "radeon" in output:
                return 8  # Conservative default for other GPUs
                
            # No recognized GPU
            return 0
            
        except Exception as e:
            logger.error(f"Error detecting GPU capabilities: {e}")
            return 0  # Safe fallback to CPU-only mode

    async def set_gpu_layers(self, gpu_count: str) -> str:
        """Set the number of GPU layers for Ollama.
        
        Args:
            gpu_count: String containing the number of GPU layers to use
                       If empty, returns the current GPU layer count
        
        Returns:
            A confirmation message
        """
        # If no argument, return current setting
        if not gpu_count.strip():
            if self.ollama_config['num_gpu'] == 0:
                return "GPU acceleration is currently disabled. Use /gpu [number] to enable it."
            else:
                return f"Current GPU layer count: {self.ollama_config['num_gpu']}"
        
        # Try to parse the GPU count
        try:
            new_gpu_count = int(gpu_count.strip())
            if new_gpu_count < 0:
                return f"Error: GPU layer count must be non-negative. Current count: {self.ollama_config['num_gpu']}"
                
            # Set the new GPU count
            old_count = self.ollama_config['num_gpu']
            self.ollama_config['num_gpu'] = new_gpu_count
            
            # Save to settings DB for persistence
            self.db.set_setting("ollama_num_gpu", str(new_gpu_count))
            
            if new_gpu_count == 0:
                return "GPU acceleration disabled. Running in CPU-only mode."
            elif old_count == 0:
                return f"GPU acceleration enabled with {new_gpu_count} layers."
            else:
                return f"GPU layer count set to {new_gpu_count}"
        except ValueError:
            return f"Error: '{gpu_count}' is not a valid number. Current count: {self.ollama_config['num_gpu']}"

if __name__ == "__main__":
    asyncio.run(MinimalAIAgent(model=LLM_MODEL).run())
