"""Redis client bootstrap with graceful fallback behavior."""

from __future__ import annotations

import os
from threading import Lock

import redis

from config.config import config
from utils.logger import get_logger


logger = get_logger("redis_client")

_redis_client = None
_redis_lock = Lock()
_REDIS_SOCKET_TIMEOUT = float(os.getenv("REDIS_SOCKET_TIMEOUT", "8"))
_REDIS_CONNECT_TIMEOUT = float(os.getenv("REDIS_CONNECT_TIMEOUT", "3"))


def get_redis_client():
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    with _redis_lock:
        if _redis_client is not None:
            return _redis_client
        try:
            _redis_client = redis.Redis.from_url(
                config.REDIS_URL,
                decode_responses=True,
                socket_timeout=_REDIS_SOCKET_TIMEOUT,
                socket_connect_timeout=_REDIS_CONNECT_TIMEOUT,
                retry_on_timeout=True,
            )
            _redis_client.ping()
            logger.info("redis connected: %s", config.REDIS_URL)
        except Exception as exc:
            logger.warning("redis unavailable, fallback to in-memory state: %s", exc)
            _redis_client = None
    return _redis_client


def redis_is_ready() -> bool:
    client = get_redis_client()
    if client is None:
        return False
    try:
        return bool(client.ping())
    except Exception:
        return False
