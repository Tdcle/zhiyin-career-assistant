from __future__ import annotations

from contextlib import contextmanager

import psycopg2.pool
from psycopg2.extras import RealDictCursor

try:
    from config.config import config
    from utils.logger import get_logger
except ImportError:  # imported as backend.db from repository root
    from backend.config.config import config
    from backend.utils.logger import get_logger

logger = get_logger("database")


class DatabaseBase:
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        cls = type(self)
        if cls._initialized:
            return

        try:
            self._pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=2, maxconn=10, **config.DB_CONFIG,
            )
            logger.info("database connection pool created: %s", config.DB_CONFIG["dbname"])
        except Exception as e:
            logger.critical("database connection pool creation failed: %s", e)
            raise

        logger.info("loading embedding model: %s", config.OLLAMA_MODELS.embedding)
        self.embed_model = config.create_embeddings()
        logger.info("embedding model loaded: %s", config.OLLAMA_MODELS.embedding)
        try:
            self.memory_extract_llm = config.create_tongyi(
                config.CHAT_MODELS.assistant_planner,
                temperature=0.0,
            )
            logger.info("memory extractor model loaded: %s", config.CHAT_MODELS.assistant_planner)
        except Exception as exc:
            logger.warning("memory extractor model unavailable, fallback to rules: %s", exc)
            self.memory_extract_llm = None

        self._init_tables()
        self._seed_default_users()
        cls._initialized = True

    @contextmanager
    def get_cursor(self, dict_cursor=False):
        conn = None
        try:
            conn = self._pool.getconn()
            cursor_factory = RealDictCursor if dict_cursor else None
            with conn.cursor(cursor_factory=cursor_factory) as cur:
                yield cur
                conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error("database operation failed, transaction rolled back: %s", e)
            raise
        finally:
            if conn:
                self._pool.putconn(conn)

    def close(self):
        if hasattr(self, "_pool") and self._pool:
            self._pool.closeall()
            logger.info("database connection pool closed")
            cls = type(self)
            cls._initialized = False
            cls._instance = None

    def health_check(self) -> bool:
        try:
            with self.get_cursor() as cur:
                cur.execute("SELECT 1")
                return cur.fetchone()[0] == 1
        except Exception:
            return False

    def get_pool_status(self) -> dict:
        if not hasattr(self, "_pool") or not self._pool:
            return {"status": "not_initialized"}
        return {
            "status": "running",
            "min_connections": self._pool.minconn,
            "max_connections": self._pool.maxconn,
        }
