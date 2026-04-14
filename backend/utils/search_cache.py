"""Redis-backed search result cache with lock and version helpers."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from threading import Lock
from typing import Any

from config.config import config
from utils.logger import get_logger
from utils.redis_client import get_redis_client


logger = get_logger("search_cache")

SEARCH_CACHE_SCHEMA = "v1"
SEARCH_DATA_VERSION_KEY = "search:data_version"

_local_lock = Lock()
_local_cache: dict[str, tuple[float, str]] = {}
_local_locks: dict[str, tuple[str, float]] = {}
_local_search_data_version = 1

_RELEASE_LOCK_LUA = """
if redis.call('get', KEYS[1]) == ARGV[1] then
  return redis.call('del', KEYS[1])
else
  return 0
end
"""


def _now() -> float:
    return time.time()


def add_ttl_jitter(base_ttl_seconds: int, jitter_seconds: int = 30) -> int:
    base = max(int(base_ttl_seconds), 1)
    jitter = max(int(jitter_seconds), 0)
    if jitter <= 0:
        return base
    # Deterministic per process time slot to avoid full-random cache churn.
    delta = int(_now()) % (jitter * 2 + 1) - jitter
    return max(1, base + delta)


def get_search_data_version() -> int:
    client = get_redis_client()
    if client is not None:
        try:
            value = client.get(SEARCH_DATA_VERSION_KEY)
            if value is None:
                client.set(SEARCH_DATA_VERSION_KEY, "1", nx=True)
                return 1
            return max(int(value), 1)
        except Exception:
            logger.warning("read search data version from redis failed", exc_info=True)

    global _local_search_data_version
    with _local_lock:
        return max(int(_local_search_data_version), 1)


def bump_search_data_version(reason: str = "") -> int:
    client = get_redis_client()
    if client is not None:
        try:
            value = int(client.incr(SEARCH_DATA_VERSION_KEY))
            logger.info(
                "search data version bumped: version=%s reason=%s",
                value,
                reason or "<none>",
            )
            return value
        except Exception:
            logger.warning("bump search data version in redis failed", exc_info=True)

    global _local_search_data_version
    with _local_lock:
        _local_search_data_version += 1
        value = _local_search_data_version
    logger.info(
        "search data version bumped (local): version=%s reason=%s",
        value,
        reason or "<none>",
    )
    return value


def build_search_cache_key(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"search:result:{SEARCH_CACHE_SCHEMA}:{digest}"


def get_cached_search_payload(cache_key: str) -> dict | None:
    if not cache_key:
        return None

    client = get_redis_client()
    if client is not None:
        try:
            raw = client.get(cache_key)
            if not raw:
                return None
            data = json.loads(raw)
            if isinstance(data, dict):
                return data
            return None
        except Exception:
            logger.warning("read search cache failed: key=%s", cache_key, exc_info=True)

    with _local_lock:
        row = _local_cache.get(cache_key)
        if not row:
            return None
        expire_at, raw = row
        if expire_at <= _now():
            _local_cache.pop(cache_key, None)
            return None
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def set_cached_search_payload(cache_key: str, payload: dict, ttl_seconds: int) -> bool:
    if not cache_key:
        return False
    ttl = max(int(ttl_seconds), 1)
    encoded = json.dumps(payload, ensure_ascii=False)

    client = get_redis_client()
    if client is not None:
        try:
            client.setex(cache_key, ttl, encoded)
            return True
        except Exception:
            logger.warning("write search cache failed: key=%s", cache_key, exc_info=True)

    with _local_lock:
        _local_cache[cache_key] = (_now() + ttl, encoded)
    return True


def acquire_search_lock(lock_key: str, ttl_seconds: int = 8) -> str | None:
    if not lock_key:
        return None
    token = uuid.uuid4().hex
    ttl = max(int(ttl_seconds), 1)

    client = get_redis_client()
    if client is not None:
        try:
            ok = client.set(lock_key, token, nx=True, ex=ttl)
            return token if ok else None
        except Exception:
            logger.warning("acquire search lock failed: key=%s", lock_key, exc_info=True)

    now = _now()
    with _local_lock:
        current = _local_locks.get(lock_key)
        if current and current[1] > now:
            return None
        _local_locks[lock_key] = (token, now + ttl)
        return token


def release_search_lock(lock_key: str, token: str) -> None:
    if not lock_key or not token:
        return
    client = get_redis_client()
    if client is not None:
        try:
            client.eval(_RELEASE_LOCK_LUA, 1, lock_key, token)
            return
        except Exception:
            logger.warning("release search lock failed: key=%s", lock_key, exc_info=True)

    with _local_lock:
        current = _local_locks.get(lock_key)
        if current and current[0] == token:
            _local_locks.pop(lock_key, None)


def wait_for_cached_search(
    cache_key: str,
    timeout_seconds: float,
    poll_interval_seconds: float = 0.08,
) -> dict | None:
    deadline = _now() + max(float(timeout_seconds), 0.0)
    interval = max(float(poll_interval_seconds), 0.02)

    while _now() < deadline:
        cached = get_cached_search_payload(cache_key)
        if cached is not None:
            return cached
        time.sleep(interval)
    return None
