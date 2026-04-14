"""User router: profile and resume task endpoints."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status
from fastapi.responses import StreamingResponse

from config.config import config
from schemas.user import (
    CreateUserRequest,
    MemoryFactListResponse,
    MemoryFactResponse,
    MessageResponse,
    ResumeTaskStatusResponse,
    ResumeTaskSubmitResponse,
    SelectUserRequest,
    UpdateMemoryFactRequest,
    UserInfo,
    UserListResponse,
)
from utils.auth import require_user_id, resolve_user_from_request
from db import DatabaseManager
from utils.logger import get_logger
from utils.rate_limiter import is_allowed
from utils.resume_task_queue import enqueue_resume_task
from utils.task_store import get_task, upsert_task


router = APIRouter()
db = DatabaseManager()
logger = get_logger("user_router")

UPLOAD_DIR = Path(__file__).resolve().parents[1] / "static" / "resumes"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def _parse_user_display(user_display: str) -> tuple[str, str]:
    if not user_display:
        return "", ""
    if " (" in user_display and user_display.endswith(")"):
        try:
            user_id_part = user_display.split(" (")[0].strip()
            username_part = user_display.split(" (")[1].rstrip(")").strip()
            return user_id_part, username_part
        except (AttributeError, IndexError):
            pass
    return user_display, user_display


def _sanitize_filename(filename: str) -> str:
    base = os.path.basename(filename or "")
    return base.replace("/", "_").replace("\\", "_")


@router.get("/list", response_model=UserListResponse)
async def list_users():
    users = db.get_all_users()
    usernames = [user["username"] for user in users]
    return UserListResponse(users=usernames)


@router.get("/info", response_model=list[UserInfo])
async def get_users_info():
    users = db.get_all_users()
    return [UserInfo(user_id=user["user_id"], username=user["username"]) for user in users]


@router.post("/create", response_model=MessageResponse)
async def create_user(body: CreateUserRequest):
    success, msg = db.create_user(body.username)
    if not success:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=msg)
    return MessageResponse(success=True, message=msg)


@router.post("/select", response_model=UserInfo)
async def select_user(body: SelectUserRequest, request: Request):
    users = db.get_all_users()
    input_user_id, input_username = _parse_user_display(body.username)

    found_user = None
    for user in users:
        if (
            user["user_id"] == input_user_id
            or user["username"] == input_username
            or user["user_id"] == body.username
            or user["username"] == body.username
        ):
            found_user = user
            break

    if not found_user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="user not found")

    user_id = found_user["user_id"]
    username = found_user["username"]
    request.session["user_id"] = user_id
    request.session["username"] = username
    logger.info("user selected: input=%s, user_id=%s, username=%s", body.username, user_id, username)
    return UserInfo(user_id=user_id, username=username)


@router.get("/me", response_model=UserInfo)
async def get_current_user(request: Request):
    user_id = require_user_id(request)
    _, username = resolve_user_from_request(request)
    return UserInfo(user_id=user_id, username=username or user_id)


@router.get("/memories", response_model=MemoryFactListResponse)
async def list_memories(request: Request):
    user_id = require_user_id(request)
    return MemoryFactListResponse(memories=db.list_user_memory_items(user_id=user_id, limit=200))


@router.put("/memories/{fact_id}", response_model=MemoryFactResponse)
async def update_memory(fact_id: int, body: UpdateMemoryFactRequest, request: Request):
    user_id = require_user_id(request)
    updated = db.update_memory_fact(
        user_id=user_id,
        fact_id=fact_id,
        fact_key=body.fact_key,
        fact_value=body.fact_value,
        confidence=body.confidence,
        importance=body.importance,
        meta=body.meta,
    )
    if not updated:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="memory not found or invalid")
    return MemoryFactResponse(**updated)


@router.delete("/memories/{fact_id}", response_model=MessageResponse)
async def delete_memory(fact_id: int, request: Request):
    user_id = require_user_id(request)
    deleted = db.delete_memory_fact(user_id=user_id, fact_id=fact_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="memory not found")
    return MessageResponse(success=True, message="memory deleted")


@router.post("/resume", response_model=ResumeTaskSubmitResponse)
async def upload_resume(
    request: Request,
    file: UploadFile = File(...),
):
    user_id = require_user_id(request)
    allowed, remaining = is_allowed(
        key=f"resume_upload:{user_id}",
        limit=config.UPLOAD_RATE_LIMIT,
        window_seconds=config.UPLOAD_RATE_WINDOW_SECONDS,
    )
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="upload rate limit exceeded",
            headers={"X-RateLimit-Remaining": str(remaining)},
        )

    allowed_ext = {".pdf", ".png", ".jpg", ".jpeg"}
    filename = _sanitize_filename(file.filename or "")
    ext = os.path.splitext(filename)[1].lower()
    if ext not in allowed_ext:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"unsupported file type: {ext}",
        )

    content = await file.read()
    max_size_bytes = config.RESUME_MAX_FILE_MB * 1024 * 1024
    if len(content) > max_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"file too large, max {config.RESUME_MAX_FILE_MB} MB",
        )

    task_id = uuid4().hex
    stored_filename = f"{user_id}_{task_id}_{filename}"
    save_path = str(UPLOAD_DIR / stored_filename)

    with open(save_path, "wb") as fp:
        fp.write(content)

    upsert_task(
        task_id,
        {
            "task_id": task_id,
            "status": "pending",
            "message": "简历正在上传...",
            "user_id": user_id,
            "filename": filename,
        },
        ttl_seconds=config.RESUME_TASK_TTL_SECONDS,
    )

    queue_backend = enqueue_resume_task(
        task_id=task_id,
        user_id=user_id,
        original_filename=filename,
        save_path=save_path,
    )
    logger.info(
        "resume task enqueued: task_id=%s user=%s file=%s backend=%s",
        task_id,
        user_id,
        filename,
        queue_backend,
    )
    return ResumeTaskSubmitResponse(
        success=True,
        task_id=task_id,
        status="pending",
        message="简历正在上传...",
    )


@router.get("/resume/task/{task_id}", response_model=ResumeTaskStatusResponse)
async def get_resume_task_status(task_id: str, request: Request):
    user_id = require_user_id(request)
    task = get_task(task_id)
    if not task:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="task not found")
    if task.get("user_id") and task["user_id"] != user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="forbidden")
    return ResumeTaskStatusResponse(
        task_id=task_id,
        status=str(task.get("status", "unknown")),
        message=str(task.get("message", "")),
        filename=task.get("filename"),
        user_id=task.get("user_id"),
        updated_at=task.get("updated_at"),
    )


@router.get("/resume/task/{task_id}/stream")
async def stream_resume_task_status(task_id: str, request: Request):
    user_id = require_user_id(request)
    initial_task = get_task(task_id)
    if not initial_task:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="task not found")
    if initial_task.get("user_id") and initial_task["user_id"] != user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="forbidden")

    async def _event_stream():
        last_sig: tuple[str, str, str] | None = None
        keepalive_ticks = 0
        while True:
            if await request.is_disconnected():
                break

            task = get_task(task_id)
            if not task:
                payload = {
                    "task_id": task_id,
                    "status": "failed",
                    "message": "task not found",
                    "updated_at": None,
                }
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                break
            if task.get("user_id") and task["user_id"] != user_id:
                payload = {
                    "task_id": task_id,
                    "status": "failed",
                    "message": "forbidden",
                    "updated_at": task.get("updated_at"),
                }
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                break

            sig = (
                str(task.get("status", "")),
                str(task.get("message", "")),
                str(task.get("updated_at", "")),
            )
            if sig != last_sig:
                yield f"data: {json.dumps(task, ensure_ascii=False)}\n\n"
                last_sig = sig
                keepalive_ticks = 0
                if task.get("status") in {"completed", "failed"}:
                    break
            else:
                keepalive_ticks += 1
                if keepalive_ticks >= 25:
                    yield ": keepalive\n\n"
                    keepalive_ticks = 0

            await asyncio.sleep(0.8)

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/resume", response_model=dict)
async def get_resume_info(request: Request):
    user_id = require_user_id(request)
    resume = db.get_latest_resume(user_id)
    if resume:
        return {
            "has_resume": True,
            "filename": resume["filename"],
            "created_at": resume["created_at"].isoformat() if resume["created_at"] else None,
        }
    return {"has_resume": False}


@router.get("/resume/structured", response_model=dict)
async def get_resume_structured(request: Request):
    user_id = require_user_id(request)
    resume = db.get_latest_resume_structured(user_id)
    if not resume:
        return {"has_resume": False}

    structured = resume.get("structured_data") or {}
    return {
        "has_resume": True,
        "resume_id": resume.get("resume_id"),
        "filename": resume.get("filename", ""),
        "created_at": resume["created_at"].isoformat() if resume.get("created_at") else None,
        "updated_at": resume["updated_at"].isoformat() if resume.get("updated_at") else None,
        "parser_version": resume.get("parser_version", ""),
        "structured_data": structured,
    }
