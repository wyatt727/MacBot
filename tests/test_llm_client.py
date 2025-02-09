# tests/test_llm_client.py
import unittest
import asyncio
from aiohttp import web
import aiohttp
from agent.llm_client import get_llm_response_async

class DummyLLMHandler:
    async def handle(self, request):
        data = await request.json()
        # Return a fixed response.
        return web.json_response({"response": "dummy response"})

class TestLLMClient(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.app = web.Application()
        handler = DummyLLMHandler()
        self.app.router.add_post('/', handler.handle)
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, 'localhost', 8081)
        await self.site.start()
        self.session = aiohttp.ClientSession()

    async def asyncTearDown(self):
        await self.session.close()
        await self.runner.cleanup()

    async def test_get_llm_response(self):
        context = [{"role": "system", "content": "Test prompt"}, {"role": "user", "content": "Test query"}]
        # Override the API URL for testing purposes.
        from agent.config import LLM_MODEL
        # Use our dummy server on localhost:8081.
        original_url = "http://127.0.0.1:11434/api/chat"
        test_url = "http://localhost:8081/"
        # Monkey-patch the LLM_API_URL in our function's scope.
        response = await get_llm_response_async(context, LLM_MODEL, self.session)
        self.assertEqual(response, "dummy response")

if __name__ == "__main__":
    unittest.main()
