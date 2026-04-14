"""Task state storage backed by Redis with local-memory fallback."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from threading import Lock

from config.config import config
from utils.redis_client import get_redis_client


_TASK_PREFIX = "resume_task:"
_local_tasks: dict[str, str] = {}
_local_lock = Lock()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def upsert_task(task_id: str, payload: dict, ttl_seconds: int | None = None) -> None:
    data = dict(payload)
    data["updated_at"] = _utc_now()
    encoded = json.dumps(data, ensure_ascii=False)

    client = get_redis_client()
    if client is not None:
        try:
            client.setex(
                f"{_TASK_PREFIX}{task_id}",
                ttl_seconds or config.RESUME_TASK_TTL_SECONDS,
                encoded,
            )
            return
        except Exception:
            pass

    with _local_lock:
        _local_tasks[task_id] = encoded


def get_task(task_id: str) -> dict | None:
    client = get_redis_client()
    if client is not None:
        try:
            raw = client.get(f"{_TASK_PREFIX}{task_id}")
            if raw:
                return json.loads(raw)
        except Exception:
            pass

    with _local_lock:
        raw = _local_tasks.get(task_id)
    if not raw:
        return None
    return json.loads(raw)
