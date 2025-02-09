# tests/test_db.py
import unittest
from agent.db import ConversationDB

class TestConversationDB(unittest.TestCase):
    def setUp(self):
        # Use an in-memory SQLite database for testing.
        self.db = ConversationDB(":memory:")

    def tearDown(self):
        self.db.close()

    def test_add_and_get_message(self):
        self.db.add_message("user", "test message")
        messages = self.db.get_recent_messages(1)
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["content"], "test message")

    def test_successful_exchange_matching(self):
        self.db.add_successful_exchange("whoami", "```sh\nwhoami\n```")
        # Exact match should yield high similarity.
        response, similarity = self.db.find_successful_exchange("whoami", threshold=0.8, bypass_threshold=0.95)
        self.assertEqual(response, "```sh\nwhoami\n```")
        # Test with a slightly different query.
        response, similarity = self.db.find_successful_exchange("what user am I?", threshold=0.8, bypass_threshold=0.95)
        self.assertIsNotNone(response)
        self.assertGreaterEqual(similarity, 0.8)

if __name__ == "__main__":
    unittest.main()
