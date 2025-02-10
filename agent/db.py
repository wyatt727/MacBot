# agent/db.py
import sqlite3
import logging
import difflib
from .config import DB_FILE, CONTEXT_MSG_COUNT

logger = logging.getLogger(__name__)

# Default conversation examples to pre-populate the DB if it is empty.
DEFAULT_CONVERSATION = [
    {"role": "user", "content": "open google but make the background red"},
    {"role": "assistant", "content": (
        "```python\nwith open(__file__, \"r\") as f:\n    source = f.read()\n"
        "with open(\"red_google.py\", \"w\") as f:\n    f.write(source)\n"
        "import asyncio\nfrom playwright.async_api import async_playwright\n\n"
        "async def run():\n    async with async_playwright() as p:\n        browser = await p.chromium.launch(headless=False)\n"
        "        page = await browser.new_page()\n        await page.goto(\"https://google.com\")\n"
        "        await page.evaluate(\"document.body.style.backgroundColor = 'red'\")\n        await asyncio.sleep(300)\nasyncio.run(run())\n```")},
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
        # Main conversation table.
        cur.execute("""
            CREATE TABLE IF NOT EXISTS conversation (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Table for storing successful exchanges.
        cur.execute("""
            CREATE TABLE IF NOT EXISTS successful_exchanges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_prompt TEXT NOT NULL,
                assistant_response TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Add composite index on user_prompt and timestamp.
        cur.execute("CREATE INDEX IF NOT EXISTS idx_user_prompt_timestamp ON successful_exchanges(user_prompt, timestamp)")
        # Create FTS virtual table for full‑text search optimization.
        cur.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS successful_exchanges_fts USING fts5(
                user_prompt, 
                assistant_response,
                content='successful_exchanges',
                content_rowid='id'
            )
        """)
        # Triggers to keep the FTS table in sync.
        cur.execute("""
            CREATE TRIGGER IF NOT EXISTS successful_exchanges_ai AFTER INSERT ON successful_exchanges BEGIN
                INSERT INTO successful_exchanges_fts(rowid, user_prompt, assistant_response)
                VALUES (new.id, new.user_prompt, new.assistant_response);
            END;
        """)
        cur.execute("""
            CREATE TRIGGER IF NOT EXISTS successful_exchanges_ad AFTER DELETE ON successful_exchanges BEGIN
                DELETE FROM successful_exchanges_fts WHERE rowid = old.id;
            END;
        """)
        cur.execute("""
            CREATE TRIGGER IF NOT EXISTS successful_exchanges_au AFTER UPDATE ON successful_exchanges BEGIN
                UPDATE successful_exchanges_fts SET user_prompt = new.user_prompt, assistant_response = new.assistant_response
                WHERE rowid = old.id;
            END;
        """)
        self.conn.commit()

    def _initialize_defaults(self):
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM conversation")
        count = cur.fetchone()[0]
        if count == 0:
            logger.info("Conversation DB empty. Inserting default examples.")
            messages = [(msg["role"], msg["content"]) for msg in DEFAULT_CONVERSATION]
            cur.executemany("INSERT INTO conversation (role, content) VALUES (?, ?)", messages)
            self.conn.commit()

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

    def find_successful_exchange(self, user_prompt: str, threshold: float = 0.80, bypass_threshold: float = 0.95):
        """
        Look for a successful exchange whose user prompt is similar to the provided user_prompt.
        Uses difflib.SequenceMatcher for string similarity.
        Returns a tuple (stored_prompt, assistant_response, similarity) if a match above threshold is found;
        Otherwise, returns (None, None, best_similarity).
        """
        cur = self.conn.cursor()
        cur.execute("SELECT user_prompt, assistant_response FROM successful_exchanges")
        rows = cur.fetchall()
        best_ratio = 0.0
        best_response = None
        best_prompt = None
        for stored_prompt, stored_response in rows:
            ratio = difflib.SequenceMatcher(None, user_prompt.lower(), stored_prompt.lower()).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_response = stored_response
                best_prompt = stored_prompt
        if best_prompt and best_ratio >= threshold:
            return best_prompt, best_response, best_ratio
        return None, None, best_ratio

    def list_successful_exchanges(self, search_term: str = ""):
        cur = self.conn.cursor()
        if search_term:
            # Use FTS MATCH for faster text-based searches.
            cur.execute("""
                SELECT se.id, se.user_prompt, se.assistant_response, se.timestamp
                FROM successful_exchanges se
                JOIN successful_exchanges_fts fts ON se.id = fts.rowid
                WHERE fts MATCH ?
                ORDER BY se.timestamp DESC
            """, (search_term,))
        else:
            cur.execute("""
                SELECT id, user_prompt, assistant_response, timestamp
                FROM successful_exchanges
                ORDER BY timestamp DESC
            """)
        rows = cur.fetchall()
        return [{"id": r[0], "user_prompt": r[1], "assistant_response": r[2], "timestamp": r[3]} for r in rows]

    def update_successful_exchange(self, exchange_id: int, new_response: str):
        cur = self.conn.cursor()
        cur.execute("UPDATE successful_exchanges SET assistant_response = ? WHERE id = ?", (new_response, exchange_id))
        self.conn.commit()

    def remove_successful_exchange(self, exchange_id: int):
        cur = self.conn.cursor()
        cur.execute("DELETE FROM successful_exchanges WHERE id = ?", (exchange_id,))
        self.conn.commit()

    def close(self):
        self.conn.close()
