# agent/db.py
import sqlite3
import logging
from .config import DB_FILE, CONTEXT_MSG_COUNT

logger = logging.getLogger(__name__)

# Default conversation examples (from your original conversation.json)
DEFAULT_CONVERSATION = [
    {"role": "user", "content": "open google but make the background red"},
    {"role": "assistant", "content": (
        "```python\nwith open(__file__, \"r\") as f:\n    source = f.read()\n"
        "with open(\"red_google.py\", \"w\") as f:\n    f.write(source)\n"
        "import asyncio\nfrom playwright.async_api import async_playwright\n\n"
        "async def run():\n    async with async_playwright() as p:\n        browser = await p.chromium.launch(headless=False)\n"
        "        page = await browser.new_page()\n        await page.goto(\"https://google.com\")\n"
        "        await page.evaluate(\"document.body.style.backgroundColor = 'red'\")\n"
        "        await asyncio.sleep(300)\nasyncio.run(run())\n```")},
    {"role": "user", "content": "open my calculator then bring me to facebook"},
    {"role": "assistant", "content": "```sh\nopen -a Calculator && open https://facebook.com\n```"},
    {"role": "user", "content": "whoami"},
    {"role": "assistant", "content": "```sh\nwhoami\n```"}
]

class ConversationDB:
    def __init__(self, db_file: str = DB_FILE):
        self.db_file = db_file
        self.conn = sqlite3.connect(self.db_file, check_same_thread=False)
        self._create_tables()
        self._initialize_defaults()

    def _create_tables(self):
        cur = self.conn.cursor()
        # Table for all conversation messages.
        cur.execute("""
            CREATE TABLE IF NOT EXISTS conversation (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # New table for successful exchanges.
        cur.execute("""
            CREATE TABLE IF NOT EXISTS successful_exchanges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_prompt TEXT NOT NULL,
                assistant_response TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

    def _initialize_defaults(self):
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM conversation")
        count = cur.fetchone()[0]
        if count == 0:
            logger.info("Conversation DB empty. Inserting default examples.")
            for msg in DEFAULT_CONVERSATION:
                self.add_message(msg["role"], msg["content"])

    def add_message(self, role: str, content: str):
        cur = self.conn.cursor()
        cur.execute("INSERT INTO conversation (role, content) VALUES (?, ?)", (role, content))
        self.conn.commit()

    def get_recent_messages(self, limit: int = CONTEXT_MSG_COUNT):
        cur = self.conn.cursor()
        cur.execute("""
            SELECT role, content FROM conversation 
            WHERE role IN ('user','assistant')
            ORDER BY timestamp DESC LIMIT ?
        """, (limit,))
        rows = cur.fetchall()[::-1]  # Reverse to chronological order.
        return [{"role": role, "content": content} for role, content in rows]

    def add_successful_exchange(self, user_prompt: str, assistant_response: str):
        cur = self.conn.cursor()
        cur.execute("""
            INSERT INTO successful_exchanges (user_prompt, assistant_response)
            VALUES (?, ?)
        """, (user_prompt, assistant_response))
        self.conn.commit()

    def find_successful_exchange(self, user_prompt: str):
        cur = self.conn.cursor()
        cur.execute("""
            SELECT assistant_response FROM successful_exchanges
            WHERE user_prompt = ?
            ORDER BY timestamp DESC LIMIT 1
        """, (user_prompt,))
        row = cur.fetchone()
        if row:
            return row[0]
        return None

    def remove_successful_exchange(self, exchange_id: int):
        cur = self.conn.cursor()
        cur.execute("DELETE FROM successful_exchanges WHERE id = ?", (exchange_id,))
        self.conn.commit()

    def close(self):
        self.conn.close()
