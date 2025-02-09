# agent/code_executor.py
import os
import subprocess
import asyncio
from datetime import datetime
import logging
from .config import GENERATED_CODE_DIR

logger = logging.getLogger(__name__)

async def execute_code_async(language: str, code: str, timeout: int = 300) -> (int, str):
    """
    Asynchronously execute a code block.
    Blocking file I/O and subprocess calls are offloaded to threads.
    """
    def save_code():
        os.makedirs(GENERATED_CODE_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ext = ".py" if language == "python" else ".sh"
        filename = os.path.join(GENERATED_CODE_DIR, f"generated_{timestamp}{ext}")
        with open(filename, "w", encoding="utf-8") as f:
            f.write(code.strip() + "\n")
        return filename
    filepath = await asyncio.to_thread(save_code)

    def run_code():
        cmd = [os.sys.executable, filepath] if language == "python" else ["sh", filepath]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            output = result.stdout + result.stderr
            return result.returncode, output.strip()
        except subprocess.TimeoutExpired:
            return -1, f"[{language.capitalize()} Error] Execution timed out after {timeout} seconds."
        except Exception as e:
            return -1, f"[{language.capitalize()} Error] Exception: {e}"
    return await asyncio.to_thread(run_code)
