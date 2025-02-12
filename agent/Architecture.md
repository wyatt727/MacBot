---
description: This rule is helpful for understanding how the program works.
globs: 
---
# Architecture Overview

This document provides an in-depth explanation of every file in the project, detailing its purpose, key features, and how it fits into the overall architecture of the AI agent.

---

## Project Structure Overview


my_agent_project/
├── agent/
│   ├── __init__.py
│   ├── config.py         # Global configuration and constants
│   ├── system_prompt.py  # Reading and caching the system prompt
│   ├── db.py             # SQLite database interface for conversation history and successful exchanges
│   ├── code_executor.py  # Asynchronous code execution routines
│   ├── llm_client.py     # Asynchronous LLM API client
│   ├── agent.py          # Core agent logic (intent matching, auto-fix loop, cached responses)
│   └── main.py           # Entry point to run the agent
├── tests/
│   ├── __init__.py       # Marks the tests directory as a package
│   ├── test_db.py        # Unit tests for the database layer
│   ├── test_system_prompt.py  # Tests for system prompt reading and caching
│   ├── test_code_executor.py  # Tests for asynchronous code execution
│   ├── test_llm_client.py     # Tests for the LLM API client
│   └── test_agent.py     # Tests for overall agent behavior and intent matching
├── requirements.txt      # Python dependencies for the project
├── README.md             # Overview, installation, usage, and testing instructions
└── system-prompt.txt     # File containing the system instructions (without examples)

---

## Detailed File Descriptions

### 1. `agent/config.py`
- **Purpose:**  
  Holds global configuration settings for the project.
- **Key Features:**  
  - **System Prompt Path:**  
    Specifies the location of the `system-prompt.txt` file which contains the system instructions.
  - **LLM API Settings:**  
    Defines the API URL (pointing to a local backend by default) and the default model name (e.g., `"gemeni-1.5-flash"`).  
    This model setting determines if the Gemini/Gemeni API or the standard local endpoint is used.
  - **Execution Parameters:**  
    Contains settings such as the maximum number of auto‑fix attempts (`MAX_FIX_ATTEMPTS`) and the number of context messages (`CONTEXT_MSG_COUNT`) to be included when constructing the LLM prompt.
  - **Database and Code Storage:**  
    Specifies the path for the SQLite database file (`conversation.db`) and the directory to store generated code files.
  - **Additional Options:**  
    Includes extra configuration like whether to save executed code blocks (`SAVE_CODE_BLOCKS`).

---

### 2. `agent/system_prompt.py`
- **Purpose:**  
  Manages the reading and caching of the system prompt to reduce unnecessary disk I/O.
- **Key Features:**  
  - **Asynchronous File I/O:**  
    Uses `aiofiles` to read the `system-prompt.txt` file in a non-blocking manner.
  - **Caching Mechanism:**  
    Caches the prompt content in memory and monitors the file's modification time.  
    Reloads the prompt if the file changes.
  - **Error Handling:**  
    Logs errors that occur when accessing file metadata or reading file contents.

---

### 3. `agent/db.py`
- **Purpose:**  
  Provides an interface to a SQLite database for persistent storage of conversations and successful exchanges.
- **Key Components:**
  - **Conversation Table:**  
    Stores all conversation messages (user, assistant, result) with timestamps.
  - **Successful Exchanges Table:**  
    Stores successful prompt–response pairs. This table is used for quick intent matching.
  - **Default Initialization:**  
    When the database is empty, it is pre-populated with a set of default conversation examples.
  - **Query Functions:**  
    Methods to add messages, retrieve the most recent messages, add successful exchanges, and perform similarity matching using difflib to find cached responses for new queries.

---

### 4. `agent/code_executor.py`
- **Purpose:**  
  Provides utilities for extracting and executing code blocks obtained from LLM responses.
- **Key Features:**  
  - **Code Extraction:**  
    Contains functions (e.g., using regular expressions) to extract code blocks enclosed in triple-backtick marks.
  - **Asynchronous Code Execution:**  
    Writes code blocks to files within a designated directory (`generated_code`).
    Uses asynchronous subprocess calls (via `asyncio.create_subprocess_exec` or similar methods) to execute the code.
    Handles timeouts and captures both stdout and stderr.
  - **Error Reporting:**  
    Returns detailed error messages if execution fails, allowing for automated queries to fix the code.

---

### 5. `agent/llm_client.py`
- **Purpose:**  
  Implements an asynchronous client to interact with the LLM API.
- **Key Features:**  
  - **Asynchronous API Requests:**  
    Utilizes `aiohttp` to send non-blocking HTTP POST requests.
  - **Payload Construction:**  
    Constructs a JSON payload including the model, conversation messages (or context), and an optional streaming flag.
  - **Caching Mechanism:**  
    Uses an in-memory cache to prevent redundant API calls for identical model and prompt combinations.
  - **Gemini/Gemeni Branch:**  
    Checks if the model indicates a Gemini/Gemeni variant.  
    If so, configures the Google Generative AI client (`google.generativeai`), and uses its `chat()` function to provide a proper chat-format interaction, passing a list of messages.
    Wraps the synchronous Gemini API call in `asyncio.run_in_executor` to keep the event loop responsive.
  - **Fallback for Non‑Gemini Models:**  
    Defaults to making asynchronous HTTP POST calls to the specified LLM API endpoint.
    Implements a retry mechanism with exponential backoff to handle network errors.

---

### 6. `agent/agent.py`
- **Purpose:**  
  Houses the core logic of the AI agent, integrating conversation management, LLM invocation, and code execution.
- **Key Features:**  
  - **`MinimalAIAgent` Class:**  
    Maintains the conversation state (e.g., last user query) and initializes the SQLite database interface (`ConversationDB`).
    Creates an asynchronous HTTP session for LLM API calls.
  - **Building Context:**  
    Combines the system prompt (loaded asynchronously), any cached successful exchange (if the new query is similar to an older one), and the current user query to form a complete context.
  - **Cached Responses and Intent Matching:**  
    Checks the conversation database to determine if a similar prompt has been seen before.  
    If a high similarity is detected, reuses the cached assistant response.
  - **LLM Invocation and Code Handling:**  
    Uses `get_llm_response_async` from `llm_client.py` to fetch responses from the LLM.
    Extracts and optionally executes code blocks contained in LLM responses.
  - **Auto‑Fix and Retry Mechanism:**  
    In the event of code execution failures, automatically requests a revised version of the code from the LLM, retrying up to a defined limit.
  - **Concurrency Control:**  
    Implements semaphores to restrict the number of simultaneous LLM calls and code executions.

---

### 7. `agent/main.py`
- **Purpose:**  
  Acts as the entry point for launching the AI agent.
- **Functionality:**  
  - **Initialization:**  
    Instantiates the `MinimalAIAgent` with the default model provided in `agent/config.py`.
  - **Event Loop Management:**  
    Starts the asynchronous event loop by running the agent through `asyncio.run()`, initiating the interaction process.

---

### 8. `tests/`
- **Purpose:**  
  Contains a suite of tests covering multiple modules of the project, ensuring robust functionality.
- **Key Features in `tests/test_llm_client.py`:**  
  - **Dummy LLM Server:**  
    Uses `aiohttp`’s web server to simulate responses from the LLM API, enabling controlled testing scenarios.
  - **Asynchronous Testing:**  
    Utilizes `unittest.IsolatedAsyncioTestCase` for testing asynchronous functions and HTTP sessions.
  - **Coverage:**  
    Verifies that the `get_llm_response_async` function returns expected responses (such as a “dummy response”) based on given conditions.

Additional tests (e.g., for the database, system prompt caching, code execution, and agent logic) are included in separate test files within this directory.

---

### 9. `requirements.txt`
- **Purpose:**  
  Lists the Python package dependencies required to run the project.
- **Key Dependencies:**  
  - `aiohttp` for asynchronous HTTP requests.
  - Any other packages as needed by your project modules.

---

### 10. `README.md`
- **Purpose:**  
  Documents the project, providing detailed instructions for installation, configuration, usage, and testing.
- **Key Features:**  
  - **Usage Instructions:**  
    Explains how to run the agent (e.g., via `python -m agent.main`), what the prompts look like, and how user interactions occur.
  - **Testing Guidelines:**  
    Provides commands and instructions for running the full test suite.
  - **Configuration Details:**  
    Breaks down configurable parameters in `agent/config.py`.
  - **Project Structure:**  
    Outlines the file and directory organization to help contributors quickly identify components.
  - **Licensing and Contribution:**  
    Contains licensing information (MIT License) and guidelines for contributing to the project.

---

### 11. `system-prompt.txt`
- **Purpose:**  
  Contains the system instructions for the AI agent.  
- **Usage:**  
  This file is read by `system_prompt.py` and its content is cached for building the LLM prompt. It should not include example interactions.

---

## Summary

- **Modularity:**  
  The project is organized into distinct modules for configuration, system prompt handling, database interactions, code execution, LLM API calls, and agent logic. This separation of concerns ensures maintainability and scalability.
  
- **Asynchronous Design:**  
  All blocking operations (file I/O, network calls, subprocess executions) are handled asynchronously to keep the system responsive on a CPU-only machine.

- **Persistent and Selective Memory:**  
  The SQLite database is used both to store a complete conversation history and to cache successful prompt–response pairs. This allows the agent to quickly respond to repeated queries by reusing cached responses.

- **Robust Error Handling and Auto‑Fix:**  
  The agent implements an auto‑fix loop for code execution errors, querying the LLM for corrections and retrying execution up to a specified number of attempts.

This document should provide a clear, detailed explanation of every file and module in your project, helping both developers and maintainers understand how the system works and how each component interacts with the others.