"""Chat router with SSE streaming and session management."""

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from logic.chat_api_flow import (
    create_user_chat_session,
    delete_user_chat_session,
    list_user_chat_sessions,
    load_user_chat_session,
    stream_chat,
)
from schemas.chat import (
    ChatHistoryResponse,
    ChatMessage,
    ChatSessionItem,
    ChatSessionListResponse,
    CreateChatSessionRequest,
    CreateChatSessionResponse,
)
from utils.auth import require_user_id
from utils.logger import get_logger


router = APIRouter()
logger = get_logger("chat_router")


@router.get("/sessions", response_model=ChatSessionListResponse)
async def list_sessions(request: Request):
    user_id = require_user_id(request)
    sessions = list_user_chat_sessions(user_id)

    current_session_id: str | None = sessions[0]["session_id"] if sessions else None
    return ChatSessionListResponse(
        sessions=[ChatSessionItem(**item) for item in sessions],
        current_session_id=current_session_id,
    )


@router.post("/sessions", response_model=CreateChatSessionResponse)
async def create_session(body: CreateChatSessionRequest, request: Request):
    user_id = require_user_id(request)
    session = create_user_chat_session(user_id=user_id, title=body.title)
    logger.info("chat session created: user=%s session=%s", user_id, session["session_id"])
    return CreateChatSessionResponse(session=ChatSessionItem(**session))


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str, request: Request):
    user_id = require_user_id(request)
    deleted = delete_user_chat_session(user_id=user_id, session_id=session_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="session not found")
    logger.info("chat session deleted: user=%s session=%s", user_id, session_id)
    return {"success": True}


@router.get("/history", response_model=ChatHistoryResponse)
async def get_chat_history(request: Request, session_id: str | None = None):
    user_id = require_user_id(request)
    try:
        active_session_id, messages, jobs = load_user_chat_session(user_id=user_id, session_id=session_id)
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="session not found")

    return ChatHistoryResponse(
        session_id=active_session_id,
        messages=[ChatMessage(role=m["role"], content=m["content"]) for m in messages],
        jobs=jobs,
    )


@router.get("/stream")
async def chat_stream(message: str, request: Request, session_id: str | None = None):
    user_id = require_user_id(request)
    if not message.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="message cannot be empty")

    logger.info("chat stream: user=%s session=%s", user_id, session_id or "<auto>")

    async def event_generator():
        try:
            async for event in stream_chat(message=message, user_id=user_id, session_id=session_id):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except KeyError:
            payload = {"type": "error", "message": "session not found"}
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        except Exception as exc:
            logger.error("chat stream error: %s", exc, exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)}, ensure_ascii=False)}\n\n"
        finally:
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/jobs")
async def get_jobs(request: Request, session_id: str | None = None):
    user_id = require_user_id(request)
    try:
        _, _, jobs = load_user_chat_session(user_id=user_id, session_id=session_id)
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="session not found")
    return {"jobs": jobs}
