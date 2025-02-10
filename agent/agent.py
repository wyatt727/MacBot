# agent/agent.py
import sys
import asyncio
import aiohttp
import subprocess
import logging
from .db import ConversationDB
from .system_prompt import get_system_prompt
from .llm_client import get_llm_response_async
from .code_executor import execute_code_async
from .config import LLM_MODEL

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

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
    """
    def __init__(self, model: str = LLM_MODEL):
        self.model = model
        self.db = ConversationDB()
        self.last_user_query = ""
        self.aiohttp_session = aiohttp.ClientSession()
        # Semaphores to limit concurrent LLM calls and code executions.
        self.llm_semaphore = asyncio.Semaphore(5)
        self.code_semaphore = asyncio.Semaphore(5)

    async def build_context(self, best_prompt=None, best_response=None, best_ratio=0) -> list:
        """
        Build the context for the LLM by combining:
          - The system prompt (loaded asynchronously).
          - An example interaction from the success DB (if the best match is at least 80% similar).
          - The current user query.
        """
        sys_prompt = await get_system_prompt()
        context = [{"role": "system", "content": sys_prompt}]
        if best_prompt is not None and best_ratio >= 0.80:
            example_text = (
                "EXAMPLE INTERACTION:\n"
                "[User]\n"
                f"{best_prompt}\n\n"
                "[MacBot]\n"
                f"{best_response}"
            )
            context.append({"role": "system", "content": example_text})
        context.append({"role": "user", "content": self.last_user_query})
        return context

    def extract_code_from_response(self, response: str):
        from .code_executor import extract_code_blocks
        return extract_code_blocks(response)

    async def process_code_block(self, language: str, code: str) -> (int, str):
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

    async def process_success_command(self, command: str):
        """
        Process a /success command by launching the GUI for managing successful exchanges.
        The GUI is launched as a separate process so that all GUI code runs on the main thread.
        """
        print("Launching Success DB GUI...")
        try:
            proc = subprocess.Popen([sys.executable, "-m", "agent.gui_success"])
            if proc.pid:
                print(f"Success DB GUI launched (PID: {proc.pid}).")
            else:
                print("Failed to launch Success DB GUI: No PID returned.")
        except Exception as e:
            print(f"Failed to launch Success DB GUI: {e}")

    async def run(self):
        print("Minimal AI Agent for macOS (Local Mode).")
        print("Type 'exit' to quit, or use '/success' commands to manage cached exchanges via a GUI.")
        try:
            while True:
                appended_success = False  # Flag to ensure we append a new exchange only once per query.
                user_input = await asyncio.to_thread(input, "User: ")
                if user_input.lower() in ("exit", "quit", "/bye"):
                    print("Exiting gracefully. Goodbye!")
                    break
                if user_input.strip().startswith("/success"):
                    await self.process_success_command(user_input.strip())
                    continue
                
                # Use the last query if the input is empty.
                if user_input.strip() == "":
                    user_input = self.last_user_query
                self.last_user_query = user_input

                # Retrieve the best matching successful exchange from the DB.
                best_prompt, best_response, best_ratio = self.db.find_successful_exchange(
                    self.last_user_query, threshold=0.80
                )
                
                # If the similarity is extremely high (95-100%), use the cached response.
                if best_prompt is not None and best_ratio >= 0.95:
                    print(f"\n[Using Cached Successful Response (similarity {best_ratio:.2f}, bypassing LLM text generation)]")
                    print(f"Cached Response:\n{best_response}\n")
                    response_to_use = best_response
                    # Only append if it's not a 100% match.
                    if best_ratio < 1.0:
                        print("[Success DB] Appending new successful exchange!")
                        #print(f"User Query: {user_input}")
                        #print(f"Assistant Response:\n{response_to_use}\n")
                        self.db.add_successful_exchange(user_input, response_to_use)
                        appended_success = True
                else:
                    # Otherwise, build context and query the LLM.
                    context = await self.build_context(best_prompt, best_response, best_ratio)
                    async with self.llm_semaphore:
                        response_to_use = await get_llm_response_async(context, self.model, self.aiohttp_session)
                    print("\n[LLM Response]\n", response_to_use)
                    self.db.add_message("assistant", response_to_use)
                
                # Process code blocks concurrently.
                blocks = self.extract_code_from_response(response_to_use)
                if blocks:
                    tasks = []
                    for idx, (lang, code) in enumerate(blocks, start=1):
                        tasks.append(self.process_code_block_with_semaphore(lang, code, idx))
                    results = await asyncio.gather(*tasks)
                    for idx, (ret, output) in enumerate(results, start=1):
                        print(f"\n--- Code Block #{idx} Execution Result ---\n{output}\n")
                        self.db.add_message("result", f"Final Output:\n{output}")
                        # Only add to the success DB if at least one code block executed successfully
                        # and if there is not already a perfect (100%) match.
                        if ret == 0 and not appended_success:
                            if best_prompt is None or best_ratio < 1.0:
                                print("[Success DB] Appending new successful exchange:")
                                print(f"User Query: {user_input}")
                                print(f"Assistant Response:\n{response_to_use}\n")
                                self.db.add_successful_exchange(user_input, response_to_use)
                                appended_success = True
                else:
                    print("[No executable code blocks found in the response.]")
                self.db.add_message("user", user_input)
        finally:
            await self.aiohttp_session.close()
            self.db.close()
            print("Agent exited.")

if __name__ == "__main__":
    asyncio.run(MinimalAIAgent(model=LLM_MODEL).run())
