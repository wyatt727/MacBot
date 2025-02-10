# My AI Agent Project

My AI Agent Project is a fully modular, multi-file Python project for an AI agent designed to run locally on CPU-only Macbooks. The agent leverages a local SQLite database to persist conversation history—including both general conversation and successful prompt–response exchanges—and uses asynchronous operations (via `aiohttp` and `asyncio`) to efficiently interact with an LLM API. The agent is optimized for precise output formatting (e.g., shell and Python code blocks) and includes an auto-fix mechanism for code execution errors.

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Testing](#testing)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Features

- **Persistent Conversation History:**  
  All interactions are stored in a local SQLite database. On first run, the database is pre-populated with default example exchanges.

- **Minimal Context Building:**  
  The agent constructs a prompt for the LLM consisting of a freshly re-read system prompt and the latest user message. This minimizes token usage while preserving precise formatting.

- **Cached Successful Exchanges:**  
  When a new user query exactly (or nearly) matches a previously successful prompt, the agent can:
  - **Bypass the LLM call entirely** if similarity is very high (e.g., ≥0.95).
  - **Inject the cached exchange into the context** if similarity is moderately high (e.g., ≥0.80), ensuring that the LLM learns the proper response format.

- **Auto-Fix Code Execution:**  
  If a code block fails execution, the agent automatically sends a fix prompt to the LLM, retrieves a corrected code block, and retries execution (up to a maximum number of attempts).

- **Asynchronous Operations:**  
  The project uses `asyncio` and `aiohttp` for non-blocking operations (such as LLM API calls and subprocess code execution) to maintain responsiveness on a CPU-only machine.

- **Local and Lightweight:**  
  Optimized for CPU-only systems, the project avoids heavy models by relying on local storage and asynchronous I/O.

## Project Structure
```
my_agent_project/
├── agent/
│   ├── __init__.py
│   ├── config.py         # Global settings and paths.
│   ├── system_prompt.py  # Reads and caches the system prompt.
│   ├── db.py             # SQLite database interface for conversation history and successful exchanges.
│   ├── code_executor.py  # Asynchronous code execution functions.
│   ├── llm_client.py     # Async LLM API client using aiohttp.
│   ├── agent.py          # Main agent logic (intent matching, auto-fix, etc.).
│   └── main.py           # Entry point for launching the agent.
├── tests/
│   ├── __init__.py
│   ├── test_db.py        # Unit tests for the database layer.
│   ├── test_system_prompt.py  # Tests for system prompt caching.
│   ├── test_code_executor.py  # Tests for asynchronous code execution.
│   ├── test_llm_client.py     # Tests for the LLM API client.
│   └── test_agent.py     # Tests for agent logic and intent matching.
├── requirements.txt      # Python dependencies.
├── README.md             # Project documentation.
└── system-prompt.txt     # File containing system instructions (no examples).
```

## Installation

1. **Clone the Repository:**

   ```bash
   git clone <repository_url>
   cd my_agent_project
   ```

2. **Install Dependencies:**

   Ensure you have Python 3.9 or later installed, then run:

   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the System Prompt:**

   Create a file named `system-prompt.txt` in the project root with your system instructions (without examples). For example:

   ''''text
   You are a MacOS command line AI agent. Provide code blocks exactly in the following format:
   - For shell commands: ```sh
   - For Python scripts: ```python
   Do not include any extraneous text outside of these code blocks.
   ''''

## Usage

To run the agent locally:

```bash
python -m agent.main
```

When you run the agent, you'll see a prompt like:

```text
Minimal AI Agent for macOS (Local Mode). Type 'exit', 'quit', or '/bye' to quit.
User:
```

- **Entering Prompts:**  
  Type your prompt (e.g., "whoami" or "open google and change the background to red") and press Enter.
  
- **Cached Responses:**  
  If your prompt matches a stored successful prompt (with high similarity), the agent will either bypass the LLM call entirely or inject the cached exchange into the prompt for guidance.

- **Auto-Fix:**  
  If a code block fails execution, the agent will automatically query the LLM for a corrected version and retry execution (up to a set number of attempts).

## Testing

A comprehensive test suite is provided to verify the functionality of each module.

Run the tests with:

```bash
python -m unittest discover tests
```

The tests cover:
- Database operations (adding, retrieving, and matching exchanges).
- System prompt reading and caching.
- Asynchronous code execution.
- LLM API client behavior (using a dummy server).
- Agent context building and intent matching.

## Configuration

You can adjust various parameters in `agent/config.py`, including:
- **LLM_API_URL** and **LLM_MODEL** for your LLM endpoint.
- **MAX_FIX_ATTEMPTS** for auto-fix retries.
- **CONTEXT_MSG_COUNT** to set the number of messages in the LLM prompt.
- **DB_FILE** and **GENERATED_CODE_DIR** for storage paths.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests. Please include tests for any new features.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Thanks to the developers behind `aiohttp` and `asyncio` for robust asynchronous tools.
- Inspired by advanced AI agent architectures and memory management strategies.
- Special thanks to all contributors and the open-source community for their support.

Enjoy your MacBot!
