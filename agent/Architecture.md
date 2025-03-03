---
description: This document provides an in-depth explanation of the project architecture, detailing the purpose and key features of each module, and how they interact.
globs:
---

# MacBot AI Agent Architecture

This document provides an in-depth explanation of every file in the project, detailing its purpose, key features, and how it fits into the overall architecture of the MacBot AI agent.

## Project Structure Overview

```
MacBot/
├── agent/
│   ├── __init__.py
│   ├── config.py         # Global configuration and constants
│   ├── system_prompt.py  # Reading and caching the system prompt
│   ├── db.py             # SQLite database interface for conversation history and successful exchanges
│   ├── code_executor.py  # Asynchronous code execution routines
│   ├── llm_client.py     # Asynchronous LLM API client
│   ├── agent.py          # Core agent logic (intent matching, auto-fix loop, cached responses)
│   ├── main.py           # Entry point to run the agent with command line arguments
│   └── model_comparator.py # Model comparison utilities
├── tests/
│   ├── __init__.py       
│   ├── test_db.py        # Unit tests for the database layer
│   ├── test_system_prompt.py  # Tests for system prompt reading and caching
│   ├── test_code_executor.py  # Tests for asynchronous code execution
│   ├── test_llm_client.py     # Tests for the LLM API client
│   └── test_agent.py     # Tests for overall agent behavior and intent matching
├── requirements.txt      # Python dependencies for the project
├── README.md             # Overview, installation, usage, and testing instructions
└── system-prompt.txt     # File containing the system instructions
```

## Detailed File Descriptions

### 1. `agent/config.py`
- **Purpose:**  
  Holds global configuration settings for the project.
- **Key Features:**  
  - **System Prompt Path:**  
    Specifies the location of the `system-prompt.txt` file.
  - **LLM API Settings:**  
    - `OLLAMA_API_BASE`: The URL for the LLM backend (by default "http://127.0.0.1:11434").
    - `LLM_MODEL`: The default model name (currently "deepseek-coder-v2") which determines the LLM model to use.
  - **Execution Parameters:**  
    Contains settings like `MAX_FIX_ATTEMPTS` for code auto‑fix attempts and `CONTEXT_MSG_COUNT` for determining the number of context messages.
  - **Performance Configuration:**
    - `RESPONSE_TIMEOUT`: Default timeout for LLM requests.
    - `MAX_CONCURRENT_LLM_CALLS`: Limits simultaneous API calls.
    - `MAX_CONCURRENT_CODE_EXECUTIONS`: Limits parallel code executions.
  - **Storage Configuration:**  
    Paths to the SQLite database file (`conversation.db`) and the directory where generated code files are stored.
  - **Additional Options:**  
    `SAVE_CODE_BLOCKS` allows code blocks to be permanently saved if set to True, otherwise uses temporary files.
  - **Optimization Settings:**
    - Support for configuring CPU thread count for Ollama
    - Auto-detection of GPU capabilities on Apple Silicon and other GPU-enabled Macs
    - Smart defaults for GPU layers based on detected hardware (M1/M2/M3)
    - Persistent thread and GPU configuration via database
    - Command-line interface for adjusting performance parameters

### 2. `agent/system_prompt.py`
- **Purpose:**  
  Reads and caches the system prompt (from `system-prompt.txt`) asynchronously to reduce disk I/O.
- **Key Features:**  
  - Asynchronous file I/O (using libraries such as `aiofiles`).
  - Caching with file metadata monitoring, so the prompt is reloaded if the file has changed.
  - Error logging during file access and reading.

### 3. `agent/db.py`
- **Purpose:**  
  Provides persistent storage for conversations, successful exchanges, and model comparisons.
- **Key Components:**  
  - **Conversation Table:** Stores all dialogue messages (user, assistant, results) along with timestamps.
  - **Successful Exchanges Table:** Caches successful prompt–response pairs based on similarity matching.
  - **Model Comparisons Table:** Stores results from model comparison runs including:
    - Response times
    - Token counts
    - Code execution results
    - Success/failure status
  - **Conversation Management Functions:**
    - Retrieving conversation history
    - Searching for specific terms in conversations
    - Clearing conversation history
    - Saving conversations to files
  - Methods to query, insert, and match conversation data efficiently.

### 4. `agent/code_executor.py`
- **Purpose:**  
  Handles code block extraction and asynchronous execution.
- **Key Features:**  
  - **Code Extraction:**  
    Contains functions to extract code blocks enclosed in triple-backtick marks.
  - **Asynchronous Code Execution:**  
    - When `SAVE_CODE_BLOCKS` is True: Creates persistent code files in `GENERATED_CODE_DIR`.
    - When `SAVE_CODE_BLOCKS` is False: Uses temporary files that are deleted after execution.
    - Uses asynchronous subprocess calls for execution with timeout handling.
  - **Error Reporting:**  
    Returns detailed error messages if execution fails, allowing for automated queries to fix the code.

### 5. `agent/llm_client.py`
- **Purpose:**  
  Implements an asynchronous client to interact with the LLM API.
- **Key Features:**  
  - **Asynchronous API Requests:**  
    Utilizes `aiohttp` for non‑blocking HTTP POST requests.
  - **Payload Construction:**  
    Constructs a JSON payload with model, messages, and parameters.
  - **Improved Timeout Management:**
    - Configurable timeout handling with support for disabling timeouts entirely.
    - Proper error handling for timeouts and network issues.
  - **Performance Metrics:**
    - Tracks response times and token usage.
    - Reports generation efficiency statistics.

### 6. `agent/agent.py`
- **Purpose:**  
  Houses the core logic of the AI agent, integrating conversation management, LLM invocation, and code execution.
- **Key Features:**  
  - **`MinimalAIAgent` Class:**  
    Maintains the conversation state and initializes the SQLite database interface.
  - **Web Search Integration:**  
    Allows searching the web for real-time information using `/search` or `/web` commands.
    - **Multi-Source Search**: Combines results from DuckDuckGo and Wikipedia for comprehensive coverage
    - **Smart Result Ranking**: Intelligently ranks results based on relevance, source credibility, and query matching
    - **Result Caching**: Implements a TTL-based cache for instant response to repeated queries
    - **Search Shortcuts**: Provides convenient shortcuts with `?query` and ultra-fast `?!query` turbo mode
    - **Background Prefetching**: Preloads top result content for faster subsequent interactions
  - **Conversation History Management:**  
    - `/history` command to view, search, clear, and save conversation history.
    - Robust handling of conversation data with proper formatting.
  - **Command Autocompletion:**  
    - Tab completion for agent commands and shell commands.
    - Context-aware completion for subcommands.
  - **Model Switching:**  
    Ability to change LLM models at runtime with the `/model` command.
  - **Performance Monitoring:**  
    - Tracks and displays detailed performance metrics.
    - Shows Ollama configuration and optimization settings.
  - **Improved Help System:**  
    - Context-sensitive help with `/help [command]`.
    - Detailed documentation for all features and commands.
  - **Performance Optimization:**
    - `/threads` command to view and adjust CPU thread count for Ollama.
    - `/gpu` command to view and adjust GPU acceleration layers.
    - Auto-detection of optimal CPU threads and GPU layers based on hardware.
    - Persistent performance settings stored in the database.
    - Dynamic configuration without requiring restart.
  - **Building Context and LLM Invocation:**  
    - Combines system prompt, conversation history, and user query.
    - Uses `get_llm_response_async` for fetching responses.
    - Handles code extraction and execution.
  - **Auto‑Fix and Retry Mechanism:**  
    Automatically requests revised code on execution failures.

### 7. `agent/main.py`
- **Purpose:**  
  Acts as the entry point for launching the AI agent.
- **Key Features:**  
  - **Command Line Arguments:**  
    - Allows configuration via command line options for model, performance settings, and more.
    - Uses argparse for robust argument handling.
  - **Initialization:**  
    Creates and configures the `MinimalAIAgent` with provided settings.
  - **Event Loop Management:**  
    Starts the asynchronous event loop for the agent.

### 8. `agent/model_comparator.py`
- **Purpose:**  
  Provides utilities for comparing different LLM models' performance.
- **Key Features:**  
  - **Model Benchmarking:**  
    - Runs the same prompt on multiple models.
    - Measures response times, token counts, and code execution results.
  - **Performance Analysis:**  
    Calculates efficiency metrics like tokens per second.
  - **Result Storage:**  
    Saves comparison results to the database for later analysis.

## Summary of New Features

The updated MacBot AI agent includes the following enhancements:

1. **Command Line Configuration:**  
   - Command line arguments for model selection and performance tuning.
   - Runtime model switching with the `/model` command.

2. **Web Integration:**  
   - Advanced web search capabilities with multiple sources (DuckDuckGo, Wikipedia)
   - Intelligent result ranking and caching for lightning-fast responses
   - Convenient search shortcuts: `?query` for quick searches and `?!query` for ultra-fast turbo mode
   - Background prefetching of top result content for improved performance

3. **Conversation Management:**  
   - `/history` command with view, search, clear, and save options.
   - Robust conversation database with search capabilities.

4. **Performance Optimization:**  
   - Enhanced timeout handling including the `/notimeout` command.
   - Temporary file usage when `SAVE_CODE_BLOCKS` is False.
   - Performance metrics and diagnostics via the `/perf` command.

5. **Improved User Experience:**  
   - Command autocompletion for agent and shell commands.
   - Context-sensitive help system with detailed documentation.
   - Tab completion for faster command entry.
   - Clean code execution output with clear separation between code and results.
   - Consistent formatting of all UI elements.

6. **Better Error Handling:**  
   - More robust recovery from timeouts and errors.
   - Clear error messages and user feedback.

These enhancements make the MacBot AI agent more versatile, responsive, and user-friendly while maintaining its focus on running efficiently on CPU-only Macs.

This document is kept up to date with the latest code base changes and provides a clear guide to the architecture and design decisions behind the MacBot AI agent.