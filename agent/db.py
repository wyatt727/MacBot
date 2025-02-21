# agent/db.py
import sqlite3
import logging
import difflib
from typing import Tuple, Optional, List, Dict
import re
from collections import Counter
from .config import (
    DB_FILE, CONTEXT_MSG_COUNT, MAX_SIMILAR_EXAMPLES,
    SIMILARITY_THRESHOLD, CACHE_TTL
)
import json
from datetime import datetime, timedelta
from functools import lru_cache
import time

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Set back to INFO level

def preprocess_text(text: str) -> str:
    """Preprocess text for better matching by:
    1. Converting to lowercase
    2. Extracting and preserving special patterns
    3. Normalizing whitespace and handling variations
    """
    # Convert to lowercase and normalize whitespace
    text = ' '.join(text.lower().split())
    
    # Quick check for exact matches before heavy processing
    normalized = re.sub(r'[^a-z0-9\s]', ' ', text)
    normalized = ' '.join(normalized.split())
    
    # Store special patterns to preserve
    patterns = {
        'commands': re.findall(r'\b(git|ls|cd|cat|python|pip|npm|yarn|docker|kubectl|whoami|curl|wget)\s+[^\s]+', text),
        'paths': re.findall(r'(?:/[^\s/]+)+/?', text),
        'urls': re.findall(r'https?://[^\s]+', text)
    }
    
    # Handle common variations using a single pass
    variations = {
        'color': ['colour', 'colors', 'colours'],
        'circle': ['circles'],
        'change': ['changes', 'changing', 'modify', 'update'],
        'draw': ['drawing', 'create', 'make'],
        'username': ['user name', 'user-name'],
        'filename': ['file name', 'file-name'],
        'directory': ['dir', 'folder'],
        'delete': ['remove', 'del', 'rm']
    }
    
    # Build variations set efficiently
    words = set(normalized.split())
    variations_to_add = set()
    
    for word in words:
        # Check if word is a key in variations
        if word in variations:
            variations_to_add.update(variations[word])
        # Check if word is a value in any variation list
        for main_word, variants in variations.items():
            if word in variants:
                variations_to_add.add(main_word)
                variations_to_add.update(v for v in variants if v != word)
    
    # Add variations and preserved patterns
    result_parts = [normalized]
    if variations_to_add:
        result_parts.append(' '.join(variations_to_add))
    for pattern_type, matches in patterns.items():
        if matches:
            # Add important patterns with higher weight
            weight = 3 if pattern_type in ['commands', 'paths'] else 2
            result_parts.extend([' '.join(matches)] * weight)
    
    return ' '.join(result_parts)

def calculate_similarity_score(query: str, stored: str) -> float:
    """Calculate a weighted similarity score using multiple metrics with performance tracking."""
    start_time = time.perf_counter()
    timings = {}

    def track_timing(name: str, start: float) -> float:
        duration = time.perf_counter() - start
        timings[name] = duration
        return duration

    # Preprocess both texts
    preprocess_start = time.perf_counter()
    query_proc = preprocess_text(query)
    stored_proc = preprocess_text(stored)
    track_timing('preprocessing', preprocess_start)
    
    # 1. Sequence similarity using difflib (25%)
    seq_start = time.perf_counter()
    sequence_sim = difflib.SequenceMatcher(None, query_proc, stored_proc).ratio() * 0.25
    track_timing('sequence_similarity', seq_start)
    
    # 2. Token overlap with position awareness (25%)
    token_start = time.perf_counter()
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
    track_timing('token_overlap', token_start)
    
    # 3. Command pattern matching (20%)
    cmd_start = time.perf_counter()
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
    track_timing('command_matching', cmd_start)
    
    # 4. Semantic similarity using key concept matching (20%)
    sem_start = time.perf_counter()
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
    track_timing('semantic_similarity', sem_start)
    
    # 5. Length ratio penalty (10%)
    len_start = time.perf_counter()
    len_ratio = min(len(query_proc), len(stored_proc)) / max(len(query_proc), len(stored_proc))
    length_score = len_ratio * 0.10
    track_timing('length_ratio', len_start)
    
    # Calculate final score
    total_score = sequence_sim + token_sim + command_sim + semantic_sim + length_score
    total_time = track_timing('total', start_time)
    
    # Only log detailed performance metrics at debug level
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"\nPerformance Analysis for similarity calculation:")
        logger.debug(f"{'Component':<20} | {'Time (ms)':<10} | {'% of Total':<10} | {'Score':<10}")
        logger.debug("-" * 55)
        
        for component, duration in timings.items():
            if component != 'total':
                percentage = (duration / total_time) * 100
                score = {
                    'preprocessing': 0,
                    'sequence_similarity': sequence_sim,
                    'token_overlap': token_sim,
                    'command_matching': command_sim,
                    'semantic_similarity': semantic_sim,
                    'length_ratio': length_score
                }.get(component, 0)
                logger.debug(f"{component:<20} | {duration*1000:>9.2f} | {percentage:>9.1f}% | {score:>9.3f}")
        
        logger.debug("-" * 55)
        logger.debug(f"{'Total':<20} | {total_time*1000:>9.2f} | {'100.0':>9}% | {total_score:>9.3f}")
    
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
        self._setup_connection()
        self._create_tables()
        self._initialize_defaults()
        self._prepare_statements()
        
    def _setup_connection(self):
        """Setup database connection with optimized settings."""
        self.conn = sqlite3.connect(
            self.db_file,
            check_same_thread=False,
            timeout=30.0,  # Increased timeout
            isolation_level=None  # Autocommit mode
        )
        self.conn.execute("PRAGMA journal_mode=WAL")  # Write-Ahead Logging
        self.conn.execute("PRAGMA synchronous=NORMAL")  # Faster writes
        self.conn.execute("PRAGMA cache_size=-2000")  # 2MB cache
        self.conn.execute("PRAGMA temp_store=MEMORY")  # Store temp tables in memory
        
    def _prepare_statements(self):
        """Pre-compile commonly used SQL statements using cursor objects."""
        # Create cursors for prepared statements
        self.prepared_statements = {
            'add_message': {
                'cursor': self.conn.cursor(),
                'sql': "INSERT INTO conversation (role, content) VALUES (?, ?)"
            },
            'get_recent': {
                'cursor': self.conn.cursor(),
                'sql': "SELECT role, content FROM conversation WHERE role IN ('user','assistant') ORDER BY timestamp DESC LIMIT ?"
            },
            'add_exchange': {
                'cursor': self.conn.cursor(),
                'sql': "INSERT INTO successful_exchanges (user_prompt, assistant_response) VALUES (?, ?)"
            }
        }

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
        """Optimized message addition using prepared statement."""
        try:
            stmt = self.prepared_statements['add_message']
            stmt['cursor'].execute(stmt['sql'], (role, content))
        except sqlite3.Error as e:
            logger.error(f"Database error adding message: {e}")
            # Fallback to new cursor if the prepared one fails
            cursor = self.conn.cursor()
            cursor.execute("INSERT INTO conversation (role, content) VALUES (?, ?)", (role, content))

    def get_recent_messages(self, limit: int = CONTEXT_MSG_COUNT) -> List[Dict[str, str]]:
        """Optimized recent message retrieval using prepared statement."""
        try:
            stmt = self.prepared_statements['get_recent']
            stmt['cursor'].execute(stmt['sql'], (limit,))
            rows = stmt['cursor'].fetchall()
            return [{"role": role, "content": content} for role, content in reversed(rows)]
        except sqlite3.Error as e:
            logger.error(f"Database error getting recent messages: {e}")
            return []

    def add_successful_exchange(self, user_prompt: str, assistant_response: str) -> bool:
        """Optimized successful exchange addition with duplicate check."""
        try:
            # Check for duplicates using indexed columns
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT EXISTS(SELECT 1 FROM successful_exchanges WHERE user_prompt = ? AND assistant_response = ?)",
                (user_prompt, assistant_response)
            )
            count = cursor.fetchone()[0]
            
            if not count:
                stmt = self.prepared_statements['add_exchange']
                stmt['cursor'].execute(stmt['sql'], (user_prompt, assistant_response))
                # Clear the cache when new data is added
                self.find_successful_exchange.cache_clear()
                return True
            return False
        except sqlite3.Error as e:
            logger.error(f"Database error adding successful exchange: {e}")
            return False

    @lru_cache(maxsize=1000)
    def find_successful_exchange(self, user_prompt: str, threshold: float = SIMILARITY_THRESHOLD) -> List[Tuple[str, str, float]]:
        """Cached version of similarity search that always returns at least the best match."""
        start_time = time.perf_counter()
        timings = {}

        def track_timing(name: str, start: float) -> float:
            duration = time.perf_counter() - start
            timings[name] = duration
            return duration

        # Database query - search ALL exchanges, not just recent ones
        query_start = time.perf_counter()
        cur = self.conn.execute(
            "SELECT user_prompt, assistant_response FROM successful_exchanges"
        )
        rows = cur.fetchall()
        track_timing('database_query', query_start)
        
        if not rows:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"\n[Performance] No matches found in {track_timing('total', start_time)*1000:.2f}ms")
            return []
        
        # Calculate similarity scores
        scoring_start = time.perf_counter()
        matches = [(stored_prompt, stored_response, calculate_similarity_score(user_prompt, stored_prompt))
                  for stored_prompt, stored_response in rows]
        track_timing('similarity_scoring', scoring_start)
        
        # Sort and filter matches
        sort_start = time.perf_counter()
        sorted_matches = sorted(matches, key=lambda x: x[2], reverse=True)
        best_match = sorted_matches[0]
        
        # Get all matches above threshold, up to MAX_SIMILAR_EXAMPLES
        additional_matches = [match for match in sorted_matches[1:] 
                            if match[2] >= threshold][:MAX_SIMILAR_EXAMPLES-1]
        track_timing('sort_and_filter', sort_start)
        
        # Only log performance metrics at debug level
        if logger.isEnabledFor(logging.DEBUG):
            total_time = track_timing('total', start_time)
            logger.debug(f"\n[Performance] Found {len(matches)} potential matches in {total_time*1000:.2f}ms")
            logger.debug(f"Best match similarity: {best_match[2]:.2%}")
            logger.debug(f"Additional matches: {len(additional_matches)}")
            logger.debug(f"Time breakdown:")
            for component, duration in timings.items():
                if component != 'total':
                    percentage = (duration / total_time) * 100
                    logger.debug(f"- {component}: {duration*1000:.2f}ms ({percentage:.1f}%)")
        
        return [best_match] + additional_matches

    def list_successful_exchanges(self, search_term: str = "", offset: int = 0, limit: int = 100) -> List[Dict]:
        """
        List successful exchanges with pagination and optimized search.
        
        Args:
            search_term: Optional search term to filter results
            offset: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            List of exchange dictionaries
        """
        cur = self.conn.cursor()
        try:
            if search_term:
                # Use LIKE query with index optimization
                search_pattern = f"%{search_term}%"
                cur.execute("""
                    SELECT id, user_prompt, assistant_response, timestamp
                    FROM successful_exchanges
                    WHERE user_prompt LIKE ? OR assistant_response LIKE ?
                    ORDER BY timestamp DESC
                    LIMIT ? OFFSET ?
                """, (search_pattern, search_pattern, limit, offset))
            else:
                cur.execute("""
                    SELECT id, user_prompt, assistant_response, timestamp
                    FROM successful_exchanges
                    ORDER BY timestamp DESC
                    LIMIT ? OFFSET ?
                """, (limit, offset))
            
            rows = cur.fetchall()
            return [{"id": r[0], "user_prompt": r[1], "assistant_response": r[2], "timestamp": r[3]} for r in rows]
            
        except sqlite3.Error as e:
            logger.error(f"Database error in list_successful_exchanges: {e}")
            return []

    def get_total_exchanges_count(self, search_term: str = "") -> int:
        """
        Get total count of exchanges, optionally filtered by search term.
        
        Args:
            search_term: Optional search term to filter results
            
        Returns:
            Total number of matching exchanges
        """
        cur = self.conn.cursor()
        try:
            if search_term:
                search_pattern = f"%{search_term}%"
                cur.execute("""
                    SELECT COUNT(*)
                    FROM successful_exchanges
                    WHERE user_prompt LIKE ? OR assistant_response LIKE ?
                """, (search_pattern, search_pattern))
            else:
                cur.execute("SELECT COUNT(*) FROM successful_exchanges")
            
            return cur.fetchone()[0]
            
        except sqlite3.Error as e:
            logger.error(f"Database error in get_total_exchanges_count: {e}")
            return 0

    def batch_update_exchanges(self, exchange_ids: List[int], new_response: str) -> bool:
        """
        Update multiple exchanges with the same response.
        
        Args:
            exchange_ids: List of exchange IDs to update
            new_response: New response text to set
            
        Returns:
            bool: True if all updates were successful
        """
        cur = self.conn.cursor()
        try:
            # Start a transaction
            cur.execute("BEGIN TRANSACTION")
            
            # Update all exchanges
            cur.executemany(
                """
                UPDATE successful_exchanges 
                SET assistant_response = ?, timestamp = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                [(new_response, exchange_id) for exchange_id in exchange_ids]
            )
            
            # Commit the transaction
            cur.execute("COMMIT")
            
            # Clear the cache
            self.find_successful_exchange.cache_clear()
            
            logger.info(f"Batch updated {len(exchange_ids)} exchanges")
            return True
            
        except sqlite3.Error as e:
            # Rollback on error
            cur.execute("ROLLBACK")
            logger.error(f"Database error in batch_update_exchanges: {e}")
            return False

    def remove_successful_exchange(self, exchange_id: int) -> bool:
        """
        Remove a single exchange by ID.
        
        Args:
            exchange_id: ID of the exchange to remove
            
        Returns:
            bool: True if deletion was successful
        """
        cur = self.conn.cursor()
        try:
            # Start a transaction
            cur.execute("BEGIN TRANSACTION")
            
            # Verify the exchange exists
            cur.execute("SELECT EXISTS(SELECT 1 FROM successful_exchanges WHERE id = ?)", (exchange_id,))
            if not cur.fetchone()[0]:
                logger.error(f"Exchange ID {exchange_id} not found")
                return False
            
            # Delete the exchange
            cur.execute("DELETE FROM successful_exchanges WHERE id = ?", (exchange_id,))
            
            # Commit the transaction
            cur.execute("COMMIT")
            
            # Clear the cache
            self.find_successful_exchange.cache_clear()
            
            logger.info(f"Successfully deleted exchange {exchange_id}")
            return True
            
        except sqlite3.Error as e:
            # Rollback on error
            cur.execute("ROLLBACK")
            logger.error(f"Database error removing exchange {exchange_id}: {e}")
            return False

    def batch_delete_exchanges(self, exchange_ids: List[int]) -> Tuple[bool, List[int]]:
        """
        Delete multiple exchanges at once.
        
        Args:
            exchange_ids: List of exchange IDs to delete
            
        Returns:
            Tuple[bool, List[int]]: (overall success, list of successfully deleted IDs)
        """
        cur = self.conn.cursor()
        successful_deletes = []
        
        try:
            # Start a transaction
            cur.execute("BEGIN TRANSACTION")
            
            # Delete exchanges one by one to track success
            for exchange_id in exchange_ids:
                try:
                    cur.execute("DELETE FROM successful_exchanges WHERE id = ?", (exchange_id,))
                    if cur.rowcount > 0:
                        successful_deletes.append(exchange_id)
                except sqlite3.Error as e:
                    logger.error(f"Error deleting exchange {exchange_id}: {e}")
            
            # Commit the transaction if any deletions were successful
            if successful_deletes:
                cur.execute("COMMIT")
                # Clear the cache
                self.find_successful_exchange.cache_clear()
                logger.info(f"Successfully deleted {len(successful_deletes)} exchanges")
            else:
                cur.execute("ROLLBACK")
                logger.warning("No exchanges were deleted")
            
            return bool(successful_deletes), successful_deletes
            
        except sqlite3.Error as e:
            # Rollback on error
            cur.execute("ROLLBACK")
            logger.error(f"Database error in batch_delete_exchanges: {e}")
            return False, []

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

    def cleanup_malformed_responses(self):
        """Clean up malformed responses in the database by removing any text outside of code blocks."""
        cur = self.conn.cursor()
        try:
            # Start a transaction
            cur.execute("BEGIN TRANSACTION")
            
            # Get all responses
            cur.execute("SELECT id, assistant_response FROM successful_exchanges")
            rows = cur.fetchall()
            
            # Pattern to match complete code blocks
            code_block_pattern = r"```[a-z]*\n[\s\S]*?```"
            
            updates = []
            for row_id, response in rows:
                # Find all code blocks
                code_blocks = re.findall(code_block_pattern, response)
                if code_blocks:
                    # Join multiple code blocks with newlines if there are any
                    cleaned_response = "\n".join(code_blocks)
                    if cleaned_response != response:
                        updates.append((cleaned_response, row_id))
            
            if updates:
                # Perform the updates
                cur.executemany(
                    "UPDATE successful_exchanges SET assistant_response = ? WHERE id = ?",
                    updates
                )
                logger.info(f"Cleaned up {len(updates)} malformed responses")
                
                # Clear the cache since we modified entries
                self.find_successful_exchange.cache_clear()
                
            # Commit the transaction
            cur.execute("COMMIT")
            return len(updates)
            
        except sqlite3.Error as e:
            # Rollback on error
            cur.execute("ROLLBACK")
            logger.error(f"Database error cleaning up responses: {e}")
            return 0

    def close(self):
        """Properly close all cursors and the connection."""
        try:
            # Close all prepared statement cursors
            for stmt in self.prepared_statements.values():
                try:
                    stmt['cursor'].close()
                except Exception as e:
                    logger.warning(f"Error closing prepared statement cursor: {e}")
            
            # Close the main connection
            self.conn.close()
        except Exception as e:
            logger.error(f"Error during database cleanup: {e}")

    def update_successful_exchange(self, exchange_id: int, new_response: str, new_prompt: str = None) -> bool:
        """
        Update a successful exchange with new response and optionally new prompt.
        
        Args:
            exchange_id: ID of the exchange to update
            new_response: New response text
            new_prompt: Optional new prompt text
            
        Returns:
            bool: True if update was successful
        """
        cur = self.conn.cursor()
        try:
            # Start a transaction
            cur.execute("BEGIN TRANSACTION")
            
            # Verify the exchange exists
            cur.execute("SELECT EXISTS(SELECT 1 FROM successful_exchanges WHERE id = ?)", (exchange_id,))
            if not cur.fetchone()[0]:
                logger.error(f"Exchange ID {exchange_id} not found")
                return False
            
            # Update the exchange
            if new_prompt is not None:
                cur.execute("""
                    UPDATE successful_exchanges 
                    SET assistant_response = ?, user_prompt = ?, timestamp = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (new_response, new_prompt, exchange_id))
            else:
                cur.execute("""
                    UPDATE successful_exchanges 
                    SET assistant_response = ?, timestamp = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (new_response, exchange_id))
            
            # Commit the transaction
            cur.execute("COMMIT")
            
            # Clear the cache since we modified an entry
            self.find_successful_exchange.cache_clear()
            
            logger.info(f"Successfully updated exchange {exchange_id}")
            return True
            
        except sqlite3.Error as e:
            # Rollback on error
            cur.execute("ROLLBACK")
            logger.error(f"Database error updating exchange {exchange_id}: {e}")
            return False
