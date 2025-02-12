# tests/test_agent.py
import unittest
import asyncio
from agent.agent import MinimalAIAgent

class TestMinimalAIAgent(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.agent = MinimalAIAgent(model="dummy-model")
        # Prepopulate the DB with a known exchange.
        self.agent.db.add_message("user", "whoami")
        self.agent.db.add_message("assistant", "```sh\nwhoami\n```")
        self.agent.last_user_query = "whoami"

    async def asyncTearDown(self):
        self.agent.db.close()
        await self.agent.aiohttp_session.close()

    async def test_build_context(self):
        context = await self.agent.build_context()
        # Context should include the system prompt and the latest user query.
        self.assertGreaterEqual(len(context), 1)
        self.assertEqual(context[-1]["role"], "user")
        self.assertEqual(context[-1]["content"], "whoami")

    async def test_cached_successful_exchange(self):
        # Add a successful exchange manually.
        self.agent.db.add_successful_exchange("whoami", "```sh\nwhoami\n```")
        # Check if querying for a similar prompt returns a successful exchange.
        best_prompt, best_response, best_ratio = self.agent.db.find_successful_exchange("whoami", threshold=0.8, bypass_threshold=0.95)
        self.assertEqual(best_response, "```sh\nwhoami\n```")
        self.assertGreaterEqual(best_ratio, 0.8)

if __name__ == "__main__":
    unittest.main()
