from agent.db import ConversationDB; db = ConversationDB(); cleaned = db.cleanup_malformed_responses(); print(f"Cleaned {cleaned} responses"); db.close()
