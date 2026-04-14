"""Reliable resume parsing queue using Redis Streams with local fallback."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Lock, Thread
from uuid import uuid4

from redis.exceptions import TimeoutError as RedisTimeoutError

from config.config import config
from db import DatabaseManager
from utils.file_parser import FileParser
from utils.logger import get_logger
from utils.redis_client import get_redis_client
from utils.task_store import get_task, upsert_task


logger = get_logger("resume_task_queue")
db = DatabaseManager()

_STREAM_KEY = "stream:resume_parse"
_DLQ_STREAM_KEY = "stream:resume_parse:dlq"
_GROUP_NAME = "resume_parse_group"
_CONSUMER_PREFIX = "resume_worker"
_LEGACY_LIST_KEY = "queue:resume_parse"

_local_queue: Queue[dict] = Queue()
_worker_lock = Lock()
_worker_stop = Event()
_worker_thread: Thread | None = None
_stream_group_ready = False
_stream_lock = Lock()
_consumer_name = f"{_CONSUMER_PREFIX}:{os.getpid()}:{uuid4().hex[:8]}"
_last_legacy_migrate_ts = 0.0


def _safe_int(value, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _retry_delay_seconds(next_attempt: int) -> int:
    delays = list(config.RESUME_QUEUE_RETRY_DELAYS_SECONDS or [])
    if not delays:
        return 0
    idx = max(0, min(next_attempt - 1, len(delays) - 1))
    return max(0, int(delays[idx]))


def _encode_stream_fields(payload: dict) -> dict[str, str]:
    encoded: dict[str, str] = {}
    for key, value in payload.items():
        if isinstance(value, (dict, list)):
            encoded[str(key)] = json.dumps(value, ensure_ascii=False)
        else:
            encoded[str(key)] = str(value if value is not None else "")
    return encoded


def _decode_stream_fields(fields: dict) -> dict:
    payload: dict[str, str] = {}
    for key, value in (fields or {}).items():
        k = key.decode("utf-8", errors="ignore") if isinstance(key, bytes) else str(key)
        v = value.decode("utf-8", errors="ignore") if isinstance(value, bytes) else str(value)
        payload[k] = v
    return payload


def _ensure_stream_group(client) -> None:
    global _stream_group_ready
    if _stream_group_ready:
        return
    with _stream_lock:
        if _stream_group_ready:
            return
        try:
            client.xgroup_create(name=_STREAM_KEY, groupname=_GROUP_NAME, id="$", mkstream=True)
        except Exception as exc:
            if "BUSYGROUP" not in str(exc):
                raise
        _stream_group_ready = True


def _migrate_legacy_list_to_stream(client, batch: int = 200) -> None:
    global _last_legacy_migrate_ts
    now = time.time()
    if now - _last_legacy_migrate_ts < 30:
        return
    _last_legacy_migrate_ts = now

    moved = 0
    try:
        for _ in range(max(1, int(batch))):
            raw = client.rpop(_LEGACY_LIST_KEY)
            if not raw:
                break
            try:
                payload = json.loads(raw)
            except Exception:
                logger.warning("skip malformed legacy queue payload: %s", raw)
                continue
            payload = dict(payload or {})
            payload.setdefault("attempt", "0")
            payload.setdefault("max_attempt", str(max(1, int(config.RESUME_QUEUE_MAX_ATTEMPTS))))
            payload.setdefault("enqueued_at", str(int(time.time())))
            client.xadd(
                _STREAM_KEY,
                _encode_stream_fields(payload),
                maxlen=max(1000, int(config.RESUME_QUEUE_STREAM_MAXLEN)),
                approximate=True,
            )
            moved += 1
    except Exception:
        logger.warning("legacy resume queue migration failed", exc_info=True)
        return

    if moved > 0:
        logger.info("migrated legacy resume queue items into stream: moved=%s", moved)


def enqueue_resume_task(
    *,
    task_id: str,
    user_id: str,
    original_filename: str,
    save_path: str,
) -> str:
    payload = {
        "task_id": task_id,
        "user_id": user_id,
        "filename": original_filename,
        "save_path": save_path,
        "attempt": "0",
        "max_attempt": str(max(1, int(config.RESUME_QUEUE_MAX_ATTEMPTS))),
        "enqueued_at": str(int(time.time())),
    }
    client = get_redis_client()
    if client is not None:
        try:
            _ensure_stream_group(client)
            client.xadd(
                _STREAM_KEY,
                _encode_stream_fields(payload),
                maxlen=max(1000, int(config.RESUME_QUEUE_STREAM_MAXLEN)),
                approximate=True,
            )
            return "redis_stream"
        except Exception:
            logger.warning("resume task enqueue to redis stream failed, fallback to local queue", exc_info=True)

    _local_queue.put(payload)
    return "local"


def start_resume_task_worker() -> None:
    global _worker_thread, _consumer_name
    with _worker_lock:
        if _worker_thread and _worker_thread.is_alive():
            return
        _worker_stop.clear()
        _consumer_name = f"{_CONSUMER_PREFIX}:{os.getpid()}:{uuid4().hex[:8]}"
        _worker_thread = Thread(
            target=_resume_worker_loop,
            name="resume-task-worker",
            daemon=True,
        )
        _worker_thread.start()
        logger.info("resume task worker started: consumer=%s", _consumer_name)


def stop_resume_task_worker(timeout_seconds: float = 3.0) -> None:
    global _worker_thread
    with _worker_lock:
        thread = _worker_thread
        if thread is None:
            return
        _worker_stop.set()
        thread.join(timeout=timeout_seconds)
        _worker_thread = None
        logger.info("resume task worker stopped")


def _resume_worker_loop() -> None:
    while not _worker_stop.is_set():
        payload, meta = _pop_task()
        if not payload:
            continue

        ok = False
        err_text = ""
        try:
            ok, err_text = _process_resume_task(payload)
        except Exception as exc:
            ok = False
            err_text = str(exc)
            logger.error("resume worker process task failed: %s", exc, exc_info=True)

        _finalize_task(payload=payload, meta=meta, success=ok, error_message=err_text)


def _claim_stale_pending(client):
    min_idle = max(1000, int(config.RESUME_QUEUE_PENDING_IDLE_MS))
    batch = max(1, int(config.RESUME_QUEUE_CLAIM_BATCH))

    try:
        claimed = client.xautoclaim(
            name=_STREAM_KEY,
            groupname=_GROUP_NAME,
            consumername=_consumer_name,
            min_idle_time=min_idle,
            start_id="0-0",
            count=batch,
        )
        messages = []
        if isinstance(claimed, (list, tuple)) and len(claimed) >= 2:
            messages = claimed[1] or []
        if messages:
            message_id, fields = messages[0]
            return _decode_stream_fields(fields), {"backend": "redis_stream", "client": client, "message_id": message_id}
    except Exception:
        pass

    # Fallback for environments where xautoclaim is unsupported/unstable.
    try:
        pending_rows = client.xpending_range(
            _STREAM_KEY,
            _GROUP_NAME,
            min="-",
            max="+",
            count=batch,
            idle=min_idle,
        )
        if not pending_rows:
            return None, None
        message_ids = []
        for row in pending_rows:
            if isinstance(row, dict):
                mid = row.get("message_id")
            else:
                mid = row[0] if row else None
            if mid:
                message_ids.append(mid)
        if not message_ids:
            return None, None
        claimed_rows = client.xclaim(_STREAM_KEY, _GROUP_NAME, _consumer_name, min_idle, message_ids)
        if claimed_rows:
            message_id, fields = claimed_rows[0]
            return _decode_stream_fields(fields), {"backend": "redis_stream", "client": client, "message_id": message_id}
    except Exception:
        pass

    return None, None


def _pop_task() -> tuple[dict | None, dict | None]:
    client = get_redis_client()
    if client is not None:
        try:
            _ensure_stream_group(client)
            _migrate_legacy_list_to_stream(client)
            payload, meta = _claim_stale_pending(client)
            if payload:
                return payload, meta

            rows = client.xreadgroup(
                groupname=_GROUP_NAME,
                consumername=_consumer_name,
                streams={_STREAM_KEY: ">"},
                count=1,
                block=1000,
            )
            if rows:
                _, messages = rows[0]
                if messages:
                    message_id, fields = messages[0]
                    payload = _decode_stream_fields(fields)
                    return payload, {"backend": "redis_stream", "client": client, "message_id": message_id}
        except RedisTimeoutError:
            # Blocking stream read timeout means "no task now", not a hard Redis failure.
            return None, None
        except Exception:
            logger.warning("resume worker redis stream consume failed, fallback to local queue", exc_info=True)

    try:
        payload = _local_queue.get(timeout=0.4)
        return payload, {"backend": "local"}
    except Empty:
        return None, None


def _process_resume_task(payload: dict) -> tuple[bool, str]:
    task_id = str(payload.get("task_id", "")).strip()
    user_id = str(payload.get("user_id", "")).strip()
    filename = str(payload.get("filename", "")).strip()
    save_path = str(payload.get("save_path", "")).strip()
    attempt = _safe_int(payload.get("attempt", 0), 0)
    max_attempt = max(1, _safe_int(payload.get("max_attempt", config.RESUME_QUEUE_MAX_ATTEMPTS), config.RESUME_QUEUE_MAX_ATTEMPTS))

    if not task_id or not user_id or not filename or not save_path:
        logger.warning("resume worker skipped invalid task payload: %s", payload)
        return True, ""

    existing_task = get_task(task_id) or {}
    if str(existing_task.get("status", "")) == "completed":
        logger.info("resume task already completed, skip duplicate consume: task_id=%s", task_id)
        return True, ""

    try:
        upsert_task(
            task_id,
            {
                "task_id": task_id,
                "status": "processing",
                "message": f"简历解析中...（第{attempt + 1}/{max_attempt}次）",
                "user_id": user_id,
                "filename": filename,
                "attempt": attempt,
                "max_attempt": max_attempt,
            },
        )

        parsed_resume = FileParser.parse_resume(save_path, original_filename=filename)
        raw_content = str(parsed_resume.get("raw_text", "") or "").strip()
        normalized_content = str(parsed_resume.get("normalized_text", "") or "").strip()
        structured_data = parsed_resume.get("structured") or {}
        parser_version = str(parsed_resume.get("parser_version", "") or "")

        if not normalized_content or len(normalized_content) < 10:
            return False, "简历解析失败或内容过短"

        success, message = db.save_resume(
            user_id=user_id,
            filename=filename,
            content=raw_content,
            normalized_content=normalized_content,
            structured_data=structured_data,
            parser_version=parser_version,
        )
        if not success:
            return False, str(message)

        upsert_task(
            task_id,
            {
                "task_id": task_id,
                "status": "completed",
                "message": "简历更新成功",
                "user_id": user_id,
                "filename": filename,
                "attempt": attempt,
                "max_attempt": max_attempt,
            },
        )
        logger.info("resume task completed: task_id=%s user=%s", task_id, user_id)
        return True, ""
    except Exception as exc:
        logger.error("resume task process failed: task_id=%s err=%s", task_id, exc, exc_info=True)
        return False, str(exc)


def _ack_and_delete(client, message_id) -> None:
    client.xack(_STREAM_KEY, _GROUP_NAME, message_id)
    client.xdel(_STREAM_KEY, message_id)


def _handle_redis_failure(meta: dict, payload: dict, error_message: str) -> None:
    client = meta.get("client")
    message_id = meta.get("message_id")
    if client is None or not message_id:
        return

    task_id = str(payload.get("task_id", "")).strip()
    user_id = str(payload.get("user_id", "")).strip()
    filename = str(payload.get("filename", "")).strip()
    save_path = str(payload.get("save_path", "")).strip()
    attempt = _safe_int(payload.get("attempt", 0), 0)
    max_attempt = max(1, _safe_int(payload.get("max_attempt", config.RESUME_QUEUE_MAX_ATTEMPTS), config.RESUME_QUEUE_MAX_ATTEMPTS))
    next_attempt = attempt + 1

    if next_attempt <= max_attempt:
        delay_seconds = _retry_delay_seconds(next_attempt)
        upsert_task(
            task_id,
            {
                "task_id": task_id,
                "status": "retrying",
                "message": f"解析失败，{delay_seconds}s后重试（第{next_attempt}/{max_attempt}次）: {error_message}",
                "user_id": user_id,
                "filename": filename,
                "attempt": next_attempt,
                "max_attempt": max_attempt,
            },
        )

        retry_payload = dict(payload)
        retry_payload["attempt"] = str(next_attempt)
        retry_payload["last_error"] = error_message[:500]
        retry_payload["retry_at"] = str(int(time.time()) + max(0, delay_seconds))
        try:
            if delay_seconds > 0:
                time.sleep(delay_seconds)
            client.xadd(
                _STREAM_KEY,
                _encode_stream_fields(retry_payload),
                maxlen=max(1000, int(config.RESUME_QUEUE_STREAM_MAXLEN)),
                approximate=True,
            )
            _ack_and_delete(client, message_id)
            logger.warning(
                "resume task retry scheduled: task_id=%s next_attempt=%s/%s delay=%ss",
                task_id,
                next_attempt,
                max_attempt,
                delay_seconds,
            )
        except Exception as exc:
            # Keep message pending for later reclaim.
            logger.error("resume task retry scheduling failed, keep pending: task_id=%s err=%s", task_id, exc, exc_info=True)
        return

    # Exhausted retries -> DLQ + failed state.
    upsert_task(
        task_id,
        {
            "task_id": task_id,
            "status": "failed",
            "message": f"简历处理失败，已重试{max_attempt}次: {error_message}",
            "user_id": user_id,
            "filename": filename,
            "attempt": next_attempt,
            "max_attempt": max_attempt,
        },
    )
    dlq_payload = {
        "task_id": task_id,
        "user_id": user_id,
        "filename": filename,
        "save_path": save_path,
        "attempt": str(next_attempt),
        "max_attempt": str(max_attempt),
        "error": error_message[:1000],
        "failed_at": str(int(time.time())),
    }
    try:
        client.xadd(
            _DLQ_STREAM_KEY,
            _encode_stream_fields(dlq_payload),
            maxlen=max(1000, int(config.RESUME_QUEUE_STREAM_MAXLEN)),
            approximate=True,
        )
        _ack_and_delete(client, message_id)
        _safe_cleanup_file(save_path)
        logger.error("resume task moved to DLQ: task_id=%s user=%s", task_id, user_id)
    except Exception as exc:
        logger.error("resume task move to DLQ failed, keep pending: task_id=%s err=%s", task_id, exc, exc_info=True)


def _handle_local_failure(payload: dict, error_message: str) -> None:
    task_id = str(payload.get("task_id", "")).strip()
    user_id = str(payload.get("user_id", "")).strip()
    filename = str(payload.get("filename", "")).strip()
    save_path = str(payload.get("save_path", "")).strip()
    attempt = _safe_int(payload.get("attempt", 0), 0)
    max_attempt = max(1, _safe_int(payload.get("max_attempt", config.RESUME_QUEUE_MAX_ATTEMPTS), config.RESUME_QUEUE_MAX_ATTEMPTS))
    next_attempt = attempt + 1

    if next_attempt <= max_attempt:
        delay_seconds = _retry_delay_seconds(next_attempt)
        upsert_task(
            task_id,
            {
                "task_id": task_id,
                "status": "retrying",
                "message": f"解析失败，{delay_seconds}s后重试（第{next_attempt}/{max_attempt}次）: {error_message}",
                "user_id": user_id,
                "filename": filename,
                "attempt": next_attempt,
                "max_attempt": max_attempt,
            },
        )
        retry_payload = dict(payload)
        retry_payload["attempt"] = str(next_attempt)
        retry_payload["last_error"] = error_message[:500]
        if delay_seconds > 0:
            time.sleep(delay_seconds)
        _local_queue.put(retry_payload)
        logger.warning(
            "resume task retry scheduled in local queue: task_id=%s next_attempt=%s/%s delay=%ss",
            task_id,
            next_attempt,
            max_attempt,
            delay_seconds,
        )
        return

    upsert_task(
        task_id,
        {
            "task_id": task_id,
            "status": "failed",
            "message": f"简历处理失败，已重试{max_attempt}次: {error_message}",
            "user_id": user_id,
            "filename": filename,
            "attempt": next_attempt,
            "max_attempt": max_attempt,
        },
    )
    _safe_cleanup_file(save_path)
    logger.error("resume task failed in local queue after retries: task_id=%s user=%s", task_id, user_id)


def _finalize_task(payload: dict, meta: dict | None, success: bool, error_message: str) -> None:
    save_path = str(payload.get("save_path", "")).strip()
    backend = str((meta or {}).get("backend", "local"))
    if backend == "redis_stream":
        if success:
            try:
                _ack_and_delete(meta.get("client"), meta.get("message_id"))
            except Exception:
                logger.error("resume task ack failed: payload=%s", payload, exc_info=True)
                return
            _safe_cleanup_file(save_path)
            return
        _handle_redis_failure(meta, payload, error_message)
        return

    if success:
        _safe_cleanup_file(save_path)
    else:
        _handle_local_failure(payload, error_message)


def _safe_cleanup_file(save_path: str) -> None:
    if not save_path:
        return
    try:
        path = Path(save_path)
        if path.exists():
            path.unlink(missing_ok=True)
    except Exception:
        logger.warning("resume source file cleanup failed: %s", save_path, exc_info=True)


def get_resume_queue_metrics() -> dict:
    worker_alive = bool(_worker_thread and _worker_thread.is_alive())
    client = get_redis_client()
    if client is None:
        return {
            "backend": "local",
            "queue_length": int(_local_queue.qsize()),
            "pending_count": 0,
            "dlq_length": 0,
            "consumers": 0,
            "worker_alive": worker_alive,
        }

    stream_len = 0
    pending_count = 0
    consumer_count = 0
    lag_count = 0
    dlq_len = 0
    legacy_list_len = 0
    try:
        _ensure_stream_group(client)
        stream_len = int(client.xlen(_STREAM_KEY) or 0)
        dlq_len = int(client.xlen(_DLQ_STREAM_KEY) or 0)
        legacy_list_len = int(client.llen(_LEGACY_LIST_KEY) or 0)
        groups = client.xinfo_groups(_STREAM_KEY) or []
        for row in groups:
            group_name = row.get("name") if isinstance(row, dict) else None
            if str(group_name) != _GROUP_NAME:
                continue
            pending_count = int(row.get("pending", 0) or 0)
            consumer_count = int(row.get("consumers", 0) or 0)
            lag_value = row.get("lag", 0) if isinstance(row, dict) else 0
            lag_count = int(lag_value or 0)
            break
    except Exception:
        logger.warning("resume queue metrics redis fetch failed, fallback to local queue size", exc_info=True)
        return {
            "backend": "local",
            "queue_length": int(_local_queue.qsize()),
            "pending_count": 0,
            "dlq_length": 0,
            "consumers": 0,
            "worker_alive": worker_alive,
        }

    backlog = max(stream_len, pending_count + lag_count)
    return {
        "backend": "redis_stream",
        "queue_length": int(backlog + legacy_list_len),
        "pending_count": pending_count,
        "dlq_length": dlq_len,
        "legacy_list_length": legacy_list_len,
        "consumers": consumer_count,
        "worker_alive": worker_alive,
    }
