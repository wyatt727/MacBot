# tests/test_system_prompt.py
import unittest
import os
import asyncio
import tempfile
from agent.system_prompt import get_system_prompt

class TestSystemPrompt(unittest.IsolatedAsyncioTestCase):
    async def test_get_system_prompt(self):
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as tmp:
            tmp.write("Test system prompt")
            tmp.flush()
            tmp_path = tmp.name
        prompt = await get_system_prompt(tmp_path)
        self.assertEqual(prompt, "Test system prompt")
        os.remove(tmp_path)

if __name__ == "__main__":
    unittest.main()
