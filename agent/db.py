# agent/db.py
import sqlite3
import logging
import difflib
from typing import Tuple, Optional, List, Dict
import re
from collections import Counter
from .config import DB_FILE, CONTEXT_MSG_COUNT, MAX_SIMILAR_EXAMPLES
import json

logger = logging.getLogger(__name__)

def preprocess_text(text: str) -> str:
    """Preprocess text for better matching by:
    1. Converting to lowercase
    2. Extracting and preserving special patterns (commands, paths, URLs)
    3. Normalizing whitespace
    4. Handling common variations
    """
    # Convert to lowercase
    text = text.lower()
    
    # Store special patterns to preserve
    patterns = {
        'commands': re.findall(r'\b(git|ls|cd|cat|python|pip|npm|yarn|docker|kubectl|whoami|curl|wget)\s+[^\s]+', text),
        'paths': re.findall(r'(?:/[^\s/]+)+/?', text),
        'urls': re.findall(r'https?://[^\s]+', text),
        'flags': re.findall(r'-{1,2}[a-zA-Z][^\s]*', text),  # Command flags like --help
        'vars': re.findall(r'\$[a-zA-Z_][a-zA-Z0-9_]*', text)  # Environment variables
    }
    
    # Remove special characters but preserve word boundaries
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    # Add back preserved patterns with higher weight (repeated for emphasis)
    for pattern_type, matches in patterns.items():
        if matches:
            # Repeat important patterns to give them more weight
            weight = 3 if pattern_type in ['commands', 'paths'] else 2
            text = text + ' ' + ' '.join(match * weight for match in matches)
    
    # Handle common variations
    variations = {
        'username': ['user name', 'user-name'],
        'filename': ['file name', 'file-name'],
        'pathname': ['path name', 'path-name'],
        'localhost': ['local host', 'local-host'],
        'directory': ['dir', 'folder'],
        'delete': ['remove', 'del', 'rm'],
        'copy': ['cp', 'duplicate'],
        'move': ['mv', 'rename'],
        'execute': ['run', 'start'],
        'display': ['show', 'print', 'output'],
        'create': ['make', 'new'],
        'modify': ['change', 'update', 'edit']
    }
    
    # Add common variations to improve matching
    for main_term, variants in variations.items():
        if main_term in text:
            text = text + ' ' + ' '.join(variants)
        for variant in variants:
            if variant in text:
                text = text + ' ' + main_term + ' ' + ' '.join(v for v in variants if v != variant)
    
    return text

def calculate_similarity_score(query: str, stored: str) -> float:
    """Calculate a weighted similarity score using multiple metrics:
    1. Sequence similarity (25%)
    2. Token overlap (25%)
    3. Command pattern matching (20%)
    4. Semantic similarity (20%)
    5. Length ratio penalty (10%)
    """
    # Preprocess both texts
    query_proc = preprocess_text(query)
    stored_proc = preprocess_text(stored)
    
    # 1. Sequence similarity using difflib (25%)
    sequence_sim = difflib.SequenceMatcher(None, query_proc, stored_proc).ratio() * 0.25
    
    # 2. Token overlap with position awareness (25%)
    query_tokens = query_proc.split()
    stored_tokens = stored_proc.split()
    
    if not query_tokens or not stored_tokens:
        token_sim = 0.0
    else:
        # Calculate token overlap
        common_tokens = set(query_tokens) & set(stored_tokens)
        
        # Consider token positions for matched tokens
        position_scores = []
        for token in common_tokens:
            query_pos = [i for i, t in enumerate(query_tokens) if t == token]
            stored_pos = [i for i, t in enumerate(stored_tokens) if t == token]
            
            # Calculate position similarity for this token
            pos_sim = max(1 - abs(qp/len(query_tokens) - sp/len(stored_tokens)) 
                         for qp in query_pos for sp in stored_pos)
            position_scores.append(pos_sim)
        
        # Combine token overlap with position awareness
        if position_scores:
            token_sim = (len(common_tokens) / max(len(query_tokens), len(stored_tokens)) * 
                        sum(position_scores) / len(position_scores) * 0.25)
        else:
            token_sim = 0.0
    
    # 3. Command pattern matching (20%)
    def extract_commands(text):
        # Extract full commands with arguments
        cmd_pattern = r'\b(git|ls|cd|cat|python|pip|npm|yarn|docker|kubectl|whoami|curl|wget)\s+[^\s]+'
        commands = re.findall(cmd_pattern, text.lower())
        # Also extract just the command names
        cmd_names = re.findall(r'\b(git|ls|cd|cat|python|pip|npm|yarn|docker|kubectl|whoami|curl|wget)\b', text.lower())
        return set(commands), set(cmd_names)
    
    query_cmds, query_cmd_names = extract_commands(query)
    stored_cmds, stored_cmd_names = extract_commands(stored)
    
    if not (query_cmds or query_cmd_names) and not (stored_cmds or stored_cmd_names):
        command_sim = 0.2  # Full score if no commands in either
    elif not (query_cmds or query_cmd_names) or not (stored_cmds or stored_cmd_names):
        command_sim = 0.0  # No match if commands in one but not the other
    else:
        # Weight exact command matches higher than just command name matches
        cmd_match = len(query_cmds & stored_cmds) / max(len(query_cmds | stored_cmds), 1) * 0.15
        name_match = len(query_cmd_names & stored_cmd_names) / max(len(query_cmd_names | stored_cmd_names), 1) * 0.05
        command_sim = cmd_match + name_match
    
    # 4. Semantic similarity using key concept matching (20%)
    concepts = {
        'file_ops': r'\b(file|read|write|open|close|save|delete|remove|copy|move|rename)\b',
        'system': r'\b(system|os|process|service|daemon|run|execute|kill|stop|start)\b',
        'network': r'\b(network|connect|url|http|web|download|upload|server|client)\b',
        'user': r'\b(user|name|login|account|password|auth|sudo|permission)\b',
        'path': r'\b(path|directory|folder|dir|location|root|home|pwd)\b'
    }
    
    def get_concept_matches(text):
        matches = {}
        for concept, pattern in concepts.items():
            matches[concept] = bool(re.search(pattern, text.lower()))
        return matches
    
    query_concepts = get_concept_matches(query)
    stored_concepts = get_concept_matches(stored)
    
    concept_matches = sum(1 for c in concepts if query_concepts[c] == stored_concepts[c])
    semantic_sim = (concept_matches / len(concepts)) * 0.20
    
    # 5. Length ratio penalty (10%)
    len_ratio = min(len(query_proc), len(stored_proc)) / max(len(query_proc), len(stored_proc))
    length_score = len_ratio * 0.10
    
    # Calculate final score
    total_score = sequence_sim + token_sim + command_sim + semantic_sim + length_score
    
    # Log detailed scoring for debugging
    logger.debug(f"Similarity Scores for '{query}' vs '{stored}':")
    logger.debug(f"  Sequence Similarity: {sequence_sim:.3f}")
    logger.debug(f"  Token Overlap: {token_sim:.3f}")
    logger.debug(f"  Command Matching: {command_sim:.3f}")
    logger.debug(f"  Semantic Similarity: {semantic_sim:.3f}")
    logger.debug(f"  Length Score: {length_score:.3f}")
    logger.debug(f"  Total Score: {total_score:.3f}")
    
    return total_score

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
        # Create main conversation table.
        cur.execute("""
            CREATE TABLE IF NOT EXISTS conversation (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Create successful_exchanges table.
        cur.execute("""
            CREATE TABLE IF NOT EXISTS successful_exchanges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_prompt TEXT NOT NULL,
                assistant_response TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Create model_comparisons table with enhanced timing metrics
        cur.execute("""
            CREATE TABLE IF NOT EXISTS model_comparisons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt TEXT NOT NULL,
                model TEXT NOT NULL,
                response TEXT,
                total_time REAL,
                generation_time REAL,
                execution_time REAL,
                tokens_per_second REAL,
                code_results TEXT,
                error TEXT,
                token_count INTEGER,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Create composite index on successful_exchanges.
        cur.execute("CREATE INDEX IF NOT EXISTS idx_user_prompt_timestamp ON successful_exchanges(user_prompt, timestamp)")
        # Create index on model_comparisons
        cur.execute("CREATE INDEX IF NOT EXISTS idx_model_comparisons ON model_comparisons(prompt, model, timestamp)")
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
        """
        Add a successful exchange to the DB only if an identical pair does not already exist.
        Returns True if a new exchange was added, False if it was a duplicate.
        """
        cur = self.conn.cursor()
        # Check if the exact request-response pair already exists.
        cur.execute("""
            SELECT COUNT(*) FROM successful_exchanges
            WHERE user_prompt = ? AND assistant_response = ?
        """, (user_prompt, assistant_response))
        count = cur.fetchone()[0]
        if count == 0:
            cur.execute("""
                INSERT INTO successful_exchanges (user_prompt, assistant_response)
                VALUES (?, ?)
            """, (user_prompt, assistant_response))
            self.conn.commit()
            logger.info("New successful exchange added to the DB.")
            return True  # New exchange was added
        else:
            logger.info("Duplicate exchange found; not adding to the DB.")
            return False  # Duplicate exchange, not added

    def find_successful_exchange(self, user_prompt: str, threshold: float = 0.80, max_examples: Optional[int] = None) -> List[Tuple[str, str, float]]:
        """
        Look for successful exchanges whose user prompts are semantically similar to the provided user_prompt.
        Returns the top N most similar examples based on max_examples parameter (defaults to MAX_SIMILAR_EXAMPLES).
        
        Args:
            user_prompt: The user's input prompt to find similar examples for
            threshold: Minimum similarity score threshold (0.0 to 1.0)
            max_examples: Maximum number of examples to return. If None, uses MAX_SIMILAR_EXAMPLES from config
        
        Returns:
            List of tuples (stored_prompt, assistant_response, similarity_score)
            List will contain at most max_examples items, sorted by similarity score descending
        """
        if max_examples is None:
            max_examples = MAX_SIMILAR_EXAMPLES
        
        cur = self.conn.cursor()
        cur.execute("SELECT user_prompt, assistant_response FROM successful_exchanges")
        rows = cur.fetchall()
        
        # Track all matches with their similarity scores
        matches = []
        
        for stored_prompt, stored_response in rows:
            ratio = calculate_similarity_score(user_prompt, stored_prompt)
            matches.append((stored_prompt, stored_response, ratio))
        
        # Sort by similarity score descending and take top N
        matches.sort(key=lambda x: x[2], reverse=True)
        top_matches = matches[:max_examples]
        
        # Log the matches found
        logger.info(f"Top {max_examples} matches for prompt: %s", user_prompt)
        for i, (prompt, _, ratio) in enumerate(top_matches, 1):
            logger.info(f"  {i}. Score: {ratio:.3f} - Prompt: {prompt}")
            
        return top_matches

    def list_successful_exchanges(self, search_term: str = ""):
        cur = self.conn.cursor()
        if search_term:
            # Use a LIKE query for simple text-based search.
            search_term = f"%{search_term}%"
            cur.execute("""
                SELECT id, user_prompt, assistant_response, timestamp
                FROM successful_exchanges
                WHERE user_prompt LIKE ? OR assistant_response LIKE ?
                ORDER BY timestamp DESC
            """, (search_term, search_term))
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

    def add_comparison_result(self, prompt: str, result: dict):
        """
        Adds a model comparison result to the database with enhanced timing metrics.
        
        Args:
            prompt: The prompt that was tested
            result: Dictionary containing the model's response and metrics
        """
        try:
            cur = self.conn.cursor()
            
            # Convert code_results to JSON string
            code_results_json = json.dumps(result.get('code_results', []))
            timing = result.get('timing', {})
            
            cur.execute("""
                INSERT INTO model_comparisons 
                (prompt, model, response, total_time, generation_time, execution_time, 
                tokens_per_second, code_results, error, token_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                prompt,
                result['model'],
                result.get('response', ''),
                timing.get('total_time'),
                timing.get('generation_time'),
                timing.get('execution_time'),
                timing.get('generation_tokens_per_second'),
                code_results_json,
                result.get('error'),
                result.get('token_count')
            ))
            
            self.conn.commit()
            logger.info(f"Stored comparison result for model {result['model']}")
            
        except Exception as e:
            logger.error(f"Failed to store comparison result: {e}")
            raise

    def get_comparison_results(self, prompt: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """
        Retrieves model comparison results from the database.
        
        Args:
            prompt: Optional prompt to filter by
            limit: Maximum number of results to return
            
        Returns:
            List of comparison result dictionaries
        """
        try:
            cur = self.conn.cursor()
            
            if prompt:
                cur.execute("""
                    SELECT prompt, model, response, total_time, generation_time, execution_time,
                           tokens_per_second, code_results, error, token_count, timestamp
                    FROM model_comparisons
                    WHERE prompt = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (prompt, limit))
            else:
                cur.execute("""
                    SELECT prompt, model, response, total_time, generation_time, execution_time,
                           tokens_per_second, code_results, error, token_count, timestamp
                    FROM model_comparisons
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (limit,))
                
            rows = cur.fetchall()
            results = []
            
            for row in rows:
                try:
                    code_results = json.loads(row[7]) if row[7] else []
                except json.JSONDecodeError:
                    code_results = []
                    logger.warning(f"Failed to decode code_results JSON for comparison ID {row[0]}")
                
                results.append({
                    "prompt": row[0],
                    "model": row[1],
                    "response": row[2],
                    "timing": {
                        "total_time": row[3],
                        "generation_time": row[4],
                        "execution_time": row[5],
                        "generation_tokens_per_second": row[6]
                    },
                    "code_results": code_results,
                    "error": row[8],
                    "token_count": row[9],
                    "timestamp": row[10]
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to retrieve comparison results: {e}")
            return []

    def close(self):
        self.conn.close()
