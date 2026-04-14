"""Simple Redis-backed rate limiter with local fallback."""

from __future__ import annotations

from threading import Lock
from time import time

from utils.redis_client import get_redis_client


_local_lock = Lock()
_local_hits: dict[str, list[float]] = {}


def _memory_is_allowed(key: str, limit: int, window_seconds: int) -> tuple[bool, int]:
    now = time()
    cutoff = now - max(window_seconds, 1)
    with _local_lock:
        hits = _local_hits.get(key, [])
        hits = [ts for ts in hits if ts >= cutoff]
        if len(hits) >= max(limit, 1):
            _local_hits[key] = hits
            return False, 0
        hits.append(now)
        _local_hits[key] = hits
        return True, max(limit, 1) - len(hits)


def is_allowed(key: str, limit: int, window_seconds: int) -> tuple[bool, int]:
    limit = max(int(limit), 1)
    window_seconds = max(int(window_seconds), 1)
    client = get_redis_client()
    if client is not None:
        redis_key = f"rate_limit:{key}"
        try:
            pipe = client.pipeline()
            pipe.incr(redis_key, amount=1)
            pipe.ttl(redis_key)
            current, ttl = pipe.execute()
            if ttl < 0:
                client.expire(redis_key, window_seconds)
            remaining = max(limit - int(current), 0)
            return int(current) <= limit, remaining
        except Exception:
            pass
    return _memory_is_allowed(key, limit, window_seconds)
