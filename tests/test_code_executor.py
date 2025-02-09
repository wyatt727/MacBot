# tests/test_code_executor.py
import unittest
import asyncio
from agent.code_executor import execute_code_async

class TestCodeExecutor(unittest.IsolatedAsyncioTestCase):
    async def test_python_code_execution(self):
        # Simple code that prints a known string.
        code = "print('Hello, World!')"
        ret, output = await execute_code_async("python", code)
        self.assertEqual(ret, 0)
        self.assertIn("Hello, World!", output)
    
    async def test_sh_code_execution(self):
        # This test assumes a Unix-like environment.
        code = "echo 'Hello, Shell!'"
        ret, output = await execute_code_async("sh", code)
        self.assertEqual(ret, 0)
        self.assertIn("Hello, Shell!", output)

if __name__ == "__main__":
    unittest.main()
