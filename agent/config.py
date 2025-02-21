# agent/config.py
import os

# Path to the system prompt file (should be in the project root)
SYSTEM_PROMPT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "system-prompt.txt")

# LLM API configuration
LLM_API_URL = "http://127.0.0.1:11434/api/chat"  # Change if needed

# Update the model name to use Gemini 1.5 flash.
LLM_MODEL = "deepseek-coder-v2"

# Execution parameters
MAX_FIX_ATTEMPTS = 2

# Number of similar examples to include in context
MAX_SIMILAR_EXAMPLES = 2

# Number of context messages to include
CONTEXT_MSG_COUNT = 5

# Performance Configuration
SIMILARITY_THRESHOLD = 0.80  # Higher threshold for better quality matches
CACHE_TTL = 3600  # Cache TTL in seconds
MAX_CONCURRENT_LLM_CALLS = 3  # Limit concurrent LLM API calls
MAX_CONCURRENT_CODE_EXECUTIONS = 3  # Limit concurrent code executions
RESPONSE_TIMEOUT = 120  # Timeout for LLM API calls in seconds

# Database configuration
DB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "conversation.db")

# Directory for generated code files
GENERATED_CODE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "generated_code")

# New configuration
SAVE_CODE_BLOCKS = True  # Set to True to save code blocks permanently
