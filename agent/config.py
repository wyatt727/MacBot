# agent/config.py
import os

# Path to the system prompt file (should be in the project root)
SYSTEM_PROMPT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "system-prompt.txt")

# LLM API configuration
LLM_API_URL = "http://127.0.0.1:11434/api/chat"  # Change if needed

# Update the model name to use Gemini 1.5 flash.
LLM_MODEL = "gemeni-1.5-flash"

# Execution parameters
MAX_FIX_ATTEMPTS = 3
CONTEXT_MSG_COUNT = 2  # Minimal context: system prompt + current user message

# Database configuration
DB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "conversation.db")

# Directory for generated code files
GENERATED_CODE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "generated_code")

# New configuration
SAVE_CODE_BLOCKS = True  # Set to True to save code blocks permanently
