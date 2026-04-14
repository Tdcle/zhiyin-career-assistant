"""Interview router with SSE endpoints."""

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from logic.interview_api_flow import end_interview, get_interview_state, start_interview, stream_interview_chat
from schemas.interview import StartInterviewRequest
from utils.auth import require_user_id
from utils.logger import get_logger


router = APIRouter()
logger = get_logger("interview_router")


@router.post("/start")
async def start_interview_endpoint(body: StartInterviewRequest, request: Request):
    user_id = require_user_id(request)
    logger.info("interview start: user=%s job_id=%s", user_id, body.job_id)

    result = await start_interview(user_id, body.job_id)
    if "error" in result:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=result["error"])

    request.session["interview_thread_id"] = result["thread_id"]
    return result


@router.get("/stream")
async def interview_stream(message: str, request: Request):
    user_id = require_user_id(request)
    thread_id = request.session.get("interview_thread_id")
    if not thread_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="interview not started")
    if not message.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="message cannot be empty")

    logger.info("interview stream: user=%s thread=%s", user_id, thread_id)

    async def event_generator():
        try:
            async for event in stream_interview_chat(message, user_id, thread_id):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as exc:
            logger.error("interview stream error: %s", exc, exc_info=True)
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


@router.get("/end")
@router.post("/end")
async def end_interview_endpoint(request: Request):
    user_id = require_user_id(request)
    thread_id = request.session.get("interview_thread_id")
    if not thread_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="interview not started")

    logger.info("interview end: user=%s thread=%s", user_id, thread_id)

    async def event_generator():
        try:
            async for event in end_interview(user_id, thread_id):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as exc:
            logger.error("interview end error: %s", exc, exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)}, ensure_ascii=False)}\n\n"
        finally:
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/state")
async def get_interview_state_endpoint(request: Request):
    user_id = require_user_id(request)
    thread_id = request.session.get("interview_thread_id")
    if not thread_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="interview not started")
    return get_interview_state(user_id, thread_id)


@router.post("/reset")
async def reset_interview(request: Request):
    require_user_id(request)
    request.session.pop("interview_thread_id", None)
    return {"success": True}
