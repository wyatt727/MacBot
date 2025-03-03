# agent/code_executor.py
import os
import asyncio
import logging
from datetime import datetime
import aiofiles
import re
from .config import GENERATED_CODE_DIR, SAVE_CODE_BLOCKS

logger = logging.getLogger(__name__)

def extract_code_blocks(response: str) -> list:
    """
    Extracts code blocks from the LLM response.
    Assumes code blocks are enclosed in triple backticks.
    """
    code_blocks = re.findall(r'```(.*?)\n(.*?)```', response, re.DOTALL)
    return [(lang.strip(), code.strip()) for lang, code in code_blocks]

async def execute_code_async(language: str, code: str, timeout: int = 300) -> (int, str):
    """
    Asynchronously execute a code block.
    Uses aiofiles for file I/O and asyncio.create_subprocess_exec for subprocess management.
    """
    if SAVE_CODE_BLOCKS:
        os.makedirs(GENERATED_CODE_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ext = ".py" if language == "python" else ".sh"
        filename = os.path.join(GENERATED_CODE_DIR, f"generated_{timestamp}{ext}")
        async with aiofiles.open(filename, "w", encoding="utf-8") as f:
            await f.write(code.strip() + "\n")
    else:
        # Create a temporary file that will be automatically cleaned up
        import tempfile
        ext = ".py" if language == "python" else ".sh"
        with tempfile.NamedTemporaryFile(mode='w', suffix=ext, delete=False) as temp_file:
            temp_file.write(code.strip() + "\n")
            filename = temp_file.name
    
    if language == "python":
        cmd = [os.sys.executable, filename]
    else:
        cmd = ["sh", filename]
    
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            return -1, f"[{language.capitalize()} Error] Execution timed out after {timeout} seconds."
        output = (stdout.decode() + stderr.decode()).strip()
        
        # Clean up temporary file if not saving code blocks
        if not SAVE_CODE_BLOCKS:
            try:
                os.unlink(filename)
            except OSError:
                pass
                
        return proc.returncode, output
    except Exception as e:
        # Clean up temporary file if not saving code blocks
        if not SAVE_CODE_BLOCKS:
            try:
                os.unlink(filename)
            except OSError:
                pass
        return -1, f"[{language.capitalize()} Error] Exception: {e}"
