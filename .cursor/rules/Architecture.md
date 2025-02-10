---
description: This rule is helpful for understanding how the program works.
globs: 
---
# Review Project Architecture and Interworkings

This document explains the purpose and interconnections of every file in the Self-Healing AI Agent Project. This project is organized into multiple modules to promote modularity, maintainability, and efficiency. Below is a breakdown of each file and its responsibilities.

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
  Contains global configuration variables and constants used throughout the project.
- **Key Settings:**  
  - **Paths:** Location of the `system-prompt.txt` file, the SQLite database file (`conversation.db`), and the directory for generated code files.
  - **LLM Settings:** API URL and model name (e.g., `"deepseek-coder-v2"`).
  - **Execution Parameters:** Maximum auto-fix attempts and the number of messages (context count) to include in each LLM prompt.

---

### 2. `agent/system_prompt.py`
- **Purpose:**  
  Manages reading and caching of the system prompt.
- **Functionality:**  
  - Reads the content of `system-prompt.txt` asynchronously.
  - Caches the prompt based on its file modification time to avoid unnecessary disk I/O.
  - Ensures that any changes to the system prompt are detected and re-read.

---

### 3. `agent/db.py`
- **Purpose:**  
  Provides an interface for persistent storage using SQLite.
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
  Handles asynchronous execution of code blocks.
- **Key Features:**  
  - Saves code blocks to a file within a designated directory (`generated_code`).
  - Uses `asyncio.to_thread` to offload blocking file I/O and subprocess calls.
  - Executes code (Python or shell) using `subprocess.run`, handling timeouts and errors.
  - Returns the exit code and output for further processing by the agent.

---

### 5. `agent/llm_client.py`
- **Purpose:**  
  Implements an asynchronous client for interacting with the LLM API.
- **Functionality:**  
  - Uses `aiohttp` to send a POST request to the LLM API endpoint.
  - Constructs a payload from the provided context and model settings.
  - Processes and returns the LLM response, handling errors gracefully.

---

### 6. `agent/agent.py`
- **Purpose:**  
  Contains the core logic of the AI agent.
- **Key Responsibilities:**
  - **Minimal Context Building:**  
    Builds the LLM prompt from the freshly read system prompt and the most recent user message from the database.
  - **Intent Matching:**  
    When a new user prompt arrives, queries the `successful_exchanges` table to find a similar successful prompt using a similarity threshold (via difflib).
      - If the similarity is extremely high (e.g., ≥0.95), the cached response is used immediately.
      - If the similarity is moderate (e.g., ≥0.80), the cached successful exchange is injected into the prompt context.
  - **LLM Interaction:**  
    If no sufficiently similar cached exchange is found, the agent calls the LLM API to generate a new response.
  - **Auto-Fix Loop:**  
    Processes code blocks from the LLM response, and if a code block fails, it sends an auto‑fix prompt to the LLM. It then extracts the corrected code block and retries execution (up to `MAX_FIX_ATTEMPTS` times).
  - **Database Updates:**  
    Stores every conversation message and, upon successful code execution, saves the exchange in the `successful_exchanges` table for future intent matching.

---

### 7. `agent/main.py`
- **Purpose:**  
  Serves as the entry point to launch the AI agent.
- **Functionality:**  
  - Instantiates the `MinimalAIAgent` class with the configured LLM model.
  - Runs the agent’s main event loop using `asyncio.run()`.

---

### 8. `tests/`
- **Purpose:**  
  Contains unit tests for the project’s components.
- **Examples:**
  - **`tests/test_db.py`:**  
    Tests the ConversationDB functionality including message storage and successful exchange matching.
  - **`tests/test_system_prompt.py`:**  
    Verifies the system prompt caching mechanism.
  - **`tests/test_code_executor.py`:**  
    Ensures that code execution works correctly for both Python and shell code.
  - **`tests/test_llm_client.py`:**  
    Simulates LLM API interactions using a dummy server to verify correct responses.
  - **`tests/test_agent.py`:**  
    Tests the overall agent logic, including context building and cached response retrieval.

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
  Provides an overview of the project, including installation, usage, and testing instructions.
- **Contents:**  
  - Project description and features.
  - Detailed setup and usage instructions.
  - Guidelines on how to run tests and contribute to the project.

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