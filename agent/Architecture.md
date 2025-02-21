---
description: This document provides an in-depth explanation of the project architecture, detailing the purpose and key features of each module, and how they interact.
globs:
---

# Architecture Overview

This document provides a detailed explanation of the MacBot AI agent project. It describes the responsibilities and interactions of each module and file in the codebase, ensuring maintainability, scalability, and responsiveness on CPU-only machines.

## Project Structure Overview

MacBot/
├── agent/
│   ├── __init__.py
│   ├── config.py         # Global configuration and constants for the project
│   ├── system_prompt.py  # Asynchronous reading and caching of the system prompt
│   ├── db.py             # SQLite database interface for conversation history and caching successful exchanges
│   ├── code_executor.py  # Asynchronous code extraction and execution routines with auto‑fix capabilities
│   ├── llm_client.py     # Asynchronous LLM API client for fetching responses
│   ├── model_comparator.py # Model comparison and analysis functionality
│   ├── agent.py          # Core agent logic: conversation management, LLM invocation, caching, and code execution
│   └── main.py           # Entry point for launching the AI agent
├── tests/                # Unit and integration tests
├── requirements.txt      # Python dependencies
├── README.md             # Overview, installation, usage, and testing instructions
└── system-prompt.txt     # System instructions for the AI agent

## Detailed File Descriptions

### 1. `agent/config.py`
- **Purpose:**  
  Provides global configuration settings and constants used throughout the system.
- **Key Settings:**  
  - **System Prompt Path:**  
    Specifies the location of the `system-prompt.txt` file.
  - **LLM API Settings:**  
    - `LLM_API_URL`: The URL for the LLM backend (by default "http://127.0.0.1:11434/api/chat").
    - `LLM_MODEL`: The default model name (currently "deepseek-coder-v2") which determines the LLM endpoint to be used.
  - **Execution Parameters:**  
    Contains settings like `MAX_FIX_ATTEMPTS` for code auto‑fix attempts and `CONTEXT_MSG_COUNT` for determining the number of context messages.
  - **Storage Configuration:**  
    Paths to the SQLite database file (`conversation.db`) and the directory where generated code files are stored.
  - **Additional Options:**  
    `SAVE_CODE_BLOCKS` allows code blocks to be permanently saved if set to True.

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
  - Methods to query, insert, and match conversation data efficiently.

### 4. `agent/code_executor.py`
- **Purpose:**  
  Handles code block extraction and asynchronous execution.
- **Key Features:**  
  - Uses regular expressions to extract code blocks from LLM responses.
  - Writes code blocks to files and executes them asynchronously using subprocesses.
  - Implements an auto‑fix mechanism that, if code execution fails, queries the LLM for corrected code and retries execution.
  - Provides detailed logging of errors and execution results.

### 5. `agent/model_comparator.py`
- **Purpose:**  
  Implements functionality for comparing responses from different Ollama models.
- **Key Features:**  
  - **Model Discovery:**  
    Asynchronously fetches available models from the Ollama API with retry logic.
  - **Concurrent Execution:**  
    Runs prompts against multiple models simultaneously using `asyncio.gather`.
  - **Response Analysis:**  
    - Tracks response times, token counts, and code execution results.
    - Provides detailed analysis and comparison of model outputs.
    - Generates formatted reports for easy comparison.
  - **Error Handling:**  
    Robust error handling for API failures, timeouts, and code execution issues.
  - **Database Integration:**  
    Stores comparison results for future reference and analysis.

### 6. `agent/llm_client.py`
- **Purpose:**  
  Implements an asynchronous client to interact with the LLM API.
- **Key Features:**  
  - **Asynchronous API Requests:**  
    Uses `aiohttp` to send non‑blocking HTTP POST requests to the LLM_API_URL.
  - **Payload Construction and Caching:**  
    Builds a JSON payload containing the model name, conversation messages, and a streaming flag.  
    Returns cached responses (from the in‑memory `_llm_cache`) when available.
  - **Retry Mechanism:**  
    Employs exponential backoff, retrying failed requests up to a configurable maximum number of attempts.
  - **Response Handling:**  
    Processes both streaming and standard responses by decoding JSON and returning either the `"response"` or message content.
  - **Note:**  
    This module uses a unified approach to send LLM requests instead of branching based on a Gemini/Gemeni condition.

### 7. `agent/agent.py`
- **Purpose:**  
  Houses the core logic of the AI agent, orchestrating conversation management, LLM interactions, and code execution.
- **Key Features:**  
  - **MinimalAIAgent Class:**  
    - Maintains conversation state, caches recent queries, and interfaces with the SQLite database.
    - Builds a comprehensive context for LLM requests by combining the system prompt, past successful interactions (if similarity exceeds a threshold), and the current user query.
    - Uses the `get_llm_response_async` function to retrieve responses from the LLM API.
    - Extracts code blocks from responses and processes them concurrently with a semaphore-controlled auto‑fix loop.
  - **User Interaction Flow:**  
    - Listens for user input with options to exit or launch various commands.
    - Reuses high-similarity cached responses to bypass unnecessary LLM calls.
    - Stores all interactions and successful exchanges in the database.
  - **Model Comparison:**
    - Implements the `/compare` command for comparing model responses.
    - Supports comparing the last query or a new prompt.
    - Displays detailed analysis of model performance and outputs.

### 8. `agent/main.py`
- **Purpose:**  
  Serves as the entry point for running the AI agent.
- **Key Functionality:**  
  - Instantiates the `MinimalAIAgent` using the default settings from `agent/config.py`.
  - Launches the asynchronous event loop with `asyncio.run()`, beginning the interactive session.

## Summary

- **Modularity and Asynchronous Design:**  
  The project is divided into clear modules—each handling configuration, prompt management, persistent storage, LLM interaction, and code execution. Asynchronous patterns ensure responsiveness, particularly on CPU-only machines.
  
- **Robust Error Handling and Optimization:**  
  From file I/O to network calls and code execution, every module has built-in error handling and logging. The caching of responses and use of exponential backoff in LLM requests optimize performance.
  
- **Flexible and Unified LLM Interaction:**  
  The LLM client now sends requests to a single configurable endpoint (LLM_API_URL) with support for streaming and non-streaming responses, and no longer branches for Gemini-specific logic. The default model "deepseek-coder-v2" can be changed in `config.py`, offering flexibility in integrating with different LLM backends.

This document is kept up to date with the latest code base changes and provides a clear guide to the architecture and design decisions behind the MacBot AI agent.