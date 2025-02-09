# agent/config.py
import os

# Path to the system prompt file (should be placed in the project root)
SYSTEM_PROMPT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "system-prompt.txt")

# LLM API configuration
LLM_API_URL = "http://127.0.0.1:11434/api/chat"  # Change if needed
LLM_MODEL = "deepseek-coder-v2"                  # Change if needed

# Execution parameters
MAX_FIX_ATTEMPTS = 3
CONTEXT_MSG_COUNT = 2  # Minimal context: system prompt + current user message

# Database configuration
DB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "conversation.db")

# Directory for generated code files
GENERATED_CODE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "generated_code")
