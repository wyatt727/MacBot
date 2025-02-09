# agent/agent.py
import asyncio
import aiohttp
import logging
from .db import ConversationDB
from .system_prompt import get_system_prompt
from .llm_client import get_llm_response_async
from .code_executor import execute_code_async
from .config import LLM_MODEL, CONTEXT_MSG_COUNT

logger = logging.getLogger(__name__)

class MinimalAIAgent:
    """
    Minimal AI Agent that uses a local SQLite database for conversation history.
    If a new user prompt exactly matches a previously successful prompt,
    the cached assistant response is used (bypassing the LLM call).
    Otherwise, the agent builds a minimal context (system prompt + current user message),
    calls the LLM asynchronously, processes code blocks (with an auto-fix loop), and stores
    the exchange as successful if the code executes correctly.
    """
    def __init__(self, model: str = LLM_MODEL):
        self.model = model
        self.db = ConversationDB()
        self.last_user_query = ""
        self.aiohttp_session = aiohttp.ClientSession()

    async def build_context(self) -> list:
        sys_prompt = await get_system_prompt()
        recent = self.db.get_recent_messages(CONTEXT_MSG_COUNT - 1)
        context = [{"role": "system", "content": sys_prompt}] + recent
        return context[-CONTEXT_MSG_COUNT:]

    def extract_code_from_response(self, response: str):
        from .code_executor import extract_code_blocks
        return extract_code_blocks(response)

    async def process_code_block(self, language: str, code: str) -> (int, str):
        """
        Process a code block with auto-fix attempts.
        If execution fails, send a prompt to the LLM for a fixed version and retry.
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

    async def run(self):
        print("Minimal AI Agent for macOS (Local Mode). Type 'exit', 'quit', or '/bye' to quit.\n")
        try:
            while True:
                user_input = await asyncio.to_thread(input, "User: ")
                if user_input.lower() in ("exit", "quit", "/bye"):
                    print("Exiting gracefully. Goodbye!")
                    break
                self.last_user_query = user_input
                # First, check if there's a cached successful exchange.
                cached_response = self.db.find_successful_exchange(user_input)
                if cached_response:
                    print("\n[Cached Successful Response]\n", cached_response)
                    response_to_use = cached_response
                else:
                    self.db.add_message("user", user_input)
                    context = await self.build_context()
                    response_to_use = await get_llm_response_async(context, self.model, self.aiohttp_session)
                    print("\n[LLM Response]\n", response_to_use)
                    self.db.add_message("assistant", response_to_use)
                # Process code blocks from the response.
                blocks = self.extract_code_from_response(response_to_use)
                if blocks:
                    for idx, (lang, code) in enumerate(blocks, start=1):
                        print(f"\n--- Processing Code Block #{idx} ({lang}) ---")
                        ret, output = await self.process_code_block(lang, code)
                        print(f"\n--- Code Execution Result ---\n{output}\n")
                        self.db.add_message("result", f"Final Output:\n{output}")
                        # If code executed successfully and no cached response was used, cache this exchange.
                        if ret == 0 and not cached_response:
                            self.db.add_successful_exchange(user_input, response_to_use)
                else:
                    print("[No executable code blocks found in the response.]")
        finally:
            await self.aiohttp_session.close()
            self.db.close()
            print("Agent exited.")
