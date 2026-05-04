"""
Hybrid Memory Manager

Manages both a persistent SQLite database and an in-memory Apache Ignite
cache for NLP2SQL conversation history. Provides exact-match cache lookup,
conversation persistence, history retrieval, session clearing, and feedback
marking.
"""

import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Optional

from pyignite import Client
from pyignite.exceptions import SocketError, ReconnectError

from app.observability.logging import logger
from utils.state import ConversationEntry
from utils.config_loader import config


class HybridMemoryManager:
    """
    Manages both a persistent SQLite database and an in-memory Apache Ignite
    cache for conversation history.
    """
    IGNITE_CACHE_NAME = "conversation_cache"

    def __init__(self, db_path: str = "conversation_memory.db"):
        self.db_path = db_path
        self._ignite_client = None
        self._init_sqlite()
        self._connect_ignite()

    def _connect_ignite(self):
        """Establishes a connection to the Apache Ignite cluster."""
        try:
            self._ignite_client = Client()
            self._ignite_client.connect(
                config.get("IGNITE_HOST", "10.170.80.113"),
                config.get("IGNITE_PORT", 10800),
            )
            self._ignite_client.get_or_create_cache(self.IGNITE_CACHE_NAME)
            logger.info("[Memory Manager] Successfully connected to Apache Ignite.")
        except (OSError, SocketError) as e:
            logger.error(f"[Memory Manager] Could not connect to Ignite (socket/os error): {e}")
            self._ignite_client = None
        except ReconnectError as e:
            logger.error(f"[Memory Manager] Could not reconnect to Ignite: {e}")
            self._ignite_client = None

    def _init_sqlite(self):
        """Initializes the SQLite database and conversation_history table."""
        try:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS conversation_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        user_id TEXT NOT NULL,
                        user_query TEXT NOT NULL,
                        refined_query TEXT NOT NULL,
                        generated_sql TEXT NOT NULL,
                        tables_used TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        is_negative_feedback BOOLEAN DEFAULT 0,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                cursor.execute(
                    'CREATE INDEX IF NOT EXISTS idx_session_query ON conversation_history(session_id, user_query)'
                )
                conn.commit()
                logger.info(f"[Memory Manager] SQLite DB initialized at {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"[Memory Manager] SQLite DB initialization error: {e}")
            raise

    def get_exact_match(self, session_id: str, user_query: str) -> Optional[Dict]:
        """
        Finds an exact match for a query in the current session.
        Checks Ignite cache first, then falls back to SQLite.

        Args:
            session_id (str): Session identifier.
            user_query (str): Exact user query string.

        Returns:
            dict | None: Cached result with 'query' and 'refined_query' keys, or None.
        """
        if self._ignite_client:
            cache_key = f"{session_id}::{user_query}"
            try:
                cached_value_json = self._ignite_client.get_cache(self.IGNITE_CACHE_NAME).get(cache_key)
                if cached_value_json:
                    logger.info(f"[Memory Manager] Cache hit for session {session_id}.")
                    return json.loads(cached_value_json)
            except Exception as e:
                logger.error(f"[Memory Manager] Error reading from Ignite cache: {e}")

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT generated_sql, refined_query FROM conversation_history "
                    "WHERE session_id = ? AND user_query = ? AND is_negative_feedback = 0 "
                    "ORDER BY created_at DESC LIMIT 1",
                    (session_id, user_query),
                )
                row = cursor.fetchone()
                if row:
                    logger.info(f"[Memory Manager] DB hit for session {session_id}. Populating cache.")
                    result = {"query": row[0], "refined_query": row[1]}
                    if self._ignite_client:
                        cache_key = f"{session_id}::{user_query}"
                        self._ignite_client.get_cache(self.IGNITE_CACHE_NAME).put(
                            cache_key, json.dumps(result)
                        )
                    return result
        except sqlite3.Error as e:
            logger.error(f"[Memory Manager] Error getting exact match from SQLite: {e}")

        return None

    def save_conversation(
        self,
        session_id: str,
        user_id: str,
        user_query: str,
        refined_query: str,
        generated_sql: str,
        tables_used: List[str],
        timestamp: str,
        is_negative_feedback: bool = False,
    ):
        """
        Saves a conversation turn to both SQLite and the Ignite cache.

        Args:
            session_id (str): Session identifier.
            user_id (str): User identifier.
            user_query (str): Original user query.
            refined_query (str): Refined query after context resolution.
            generated_sql (str): The generated SQL query.
            tables_used (list[str]): Tables used in the query.
            timestamp (str): Timestamp string.
            is_negative_feedback (bool): Whether this entry has negative feedback.
        """
        try:
            sql_to_save = (
                generated_sql.get("query", generated_sql)
                if isinstance(generated_sql, dict)
                else generated_sql
            )
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''
                    INSERT INTO conversation_history
                    (session_id, user_id, user_query, refined_query, generated_sql,
                     tables_used, timestamp, is_negative_feedback)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''',
                    (
                        session_id, user_id, user_query, refined_query,
                        sql_to_save, json.dumps(tables_used), timestamp, is_negative_feedback,
                    ),
                )
                conn.commit()

            if self._ignite_client and not is_negative_feedback:
                cache_key = f"{session_id}::{user_query}"
                cache_value = json.dumps({"query": sql_to_save, "refined_query": refined_query})
                self._ignite_client.get_cache(self.IGNITE_CACHE_NAME).put(cache_key, cache_value)

            logger.info(f"[Memory Manager] Saved conversation for session {session_id}.")
        except Exception as e:
            logger.error(f"[Memory Manager] Error saving conversation for session {session_id}: {e}")

    def get_conversation_history(self, session_id: str, limit: int = 15) -> List[Dict]:
        """
        Retrieves conversation history, filtering out entries with negative feedback.

        Args:
            session_id (str): Session identifier.
            limit (int): Maximum number of entries to return.

        Returns:
            list[dict]: Chronologically ordered conversation history.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''
                    SELECT user_query, refined_query, generated_sql, tables_used, timestamp
                    FROM conversation_history
                    WHERE session_id = ? AND is_negative_feedback = 0
                    ORDER BY created_at DESC
                    LIMIT ?
                    ''',
                    (session_id, limit),
                )
                rows = cursor.fetchall()
                history = [
                    {
                        "user_query": row[0],
                        "refined_query": row[1],
                        "generated_sql": row[2],
                        "tables_used": json.loads(row[3]),
                        "timestamp": row[4],
                    }
                    for row in reversed(rows)
                ]
                logger.info(
                    f"[Memory Manager] Retrieved {len(history)} valid history entries "
                    f"for session {session_id}"
                )
                return history
        except sqlite3.Error as e:
            logger.error(f"[Memory Manager] Error retrieving history for session {session_id}: {e}")
            return []

    def clear_session_from_ignite(self, session_id: str):
        """
        Clears all in-memory Ignite cache entries for a given session.

        Args:
            session_id (str): Session identifier to clear.
        """
        if not self._ignite_client:
            logger.warning("[Memory Manager] Cannot clear session cache: Ignite client not connected.")
            return
        try:
            cache = self._ignite_client.get_cache(self.IGNITE_CACHE_NAME)
            keys_to_delete = [
                key for key, _ in cache.scan()
                if isinstance(key, str) and key.startswith(f"{session_id}::")
            ]
            if keys_to_delete:
                cache.remove_keys(keys_to_delete)
                logger.info(
                    f"[Memory Manager] Cleared {len(keys_to_delete)} cache entries "
                    f"for session {session_id}."
                )
            else:
                logger.info(f"[Memory Manager] No cache entries found for session {session_id}.")
        except Exception as e:
            logger.error(f"[Memory Manager] Error clearing Ignite cache for session {session_id}: {e}")

    def mark_feedback(self, session_id: str, user_query: str, is_negative: bool):
        """
        Updates a conversation entry with feedback and removes it from cache if negative.

        Args:
            session_id (str): Session identifier.
            user_query (str): The query to mark.
            is_negative (bool): True for negative feedback.

        Raises:
            Exception: If SQLite update fails.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE conversation_history SET is_negative_feedback = ? "
                    "WHERE session_id = ? AND user_query = ?",
                    (is_negative, session_id, user_query),
                )
                conn.commit()
                logger.info(
                    f"[Memory Manager] Marked feedback for session {session_id}. "
                    f"Updated {cursor.rowcount} rows."
                )

            if is_negative and self._ignite_client:
                cache_key = f"{session_id}::{user_query}"
                self._ignite_client.get_cache(self.IGNITE_CACHE_NAME).remove_key(cache_key)
                logger.info(
                    f"[Memory Manager] Removed negative entry from cache for session {session_id}."
                )
        except Exception as e:
            logger.error(f"[Memory Manager] Error marking feedback for session {session_id}: {e}")
            raise
