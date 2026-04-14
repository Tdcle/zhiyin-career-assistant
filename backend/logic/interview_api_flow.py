"""FastAPI wrappers for legacy interview flow."""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator

from logic.interview_flow import end_interview_with_summary as _legacy_end_interview_with_summary
from logic.interview_flow import handle_interview_chat as _legacy_handle_interview_chat
from logic.interview_flow import start_interview_logic as _legacy_start_interview_logic
from db import DatabaseManager


db = DatabaseManager()
DEFAULT_SCORECARD = {
    "tech_depth": 0,
    "project_depth": 0,
    "experience_match": 0,
    "communication": 0,
    "jd_fit": 0,
}


def _normalize_scorecard_for_api(raw: dict | None) -> dict:
    parsed: dict[str, float] = {}
    for key, default_value in DEFAULT_SCORECARD.items():
        raw_value = (raw or {}).get(key, default_value)
        try:
            parsed[key] = float(raw_value)
        except (TypeError, ValueError):
            parsed[key] = float(default_value)

    non_zero_values = [value for value in parsed.values() if value > 0]
    scale_to_100 = bool(non_zero_values) and max(non_zero_values) <= 10

    normalized: dict[str, int] = {}
    for key, value in parsed.items():
        adjusted = value * 10 if scale_to_100 else value
        normalized[key] = max(0, min(100, int(round(adjusted))))
    return normalized


def _extract_update_value(value):
    if isinstance(value, dict):
        return value.get("value")
    if hasattr(value, "get"):
        try:
            return value.get("value")
        except Exception:
            return None
    return None


def _load_context_and_history(thread_id: str):
    session = db.get_conversation_state(thread_id) or {}
    extra = session.get("extra_state") or {}
    history = list(session.get("recent_messages") or [])
    context = {
        "thread_id": thread_id,
        "job_context": extra.get("job_context", {}),
        "match_analysis": extra.get("match_analysis", ""),
    }
    return context, history, extra


async def start_interview(user_id: str, job_id: str) -> dict:
    job = db.get_job_details(job_id)
    if not job:
        return {"error": "job not found"}

    last_output = None
    for output in _legacy_start_interview_logic(job, user_id):
        last_output = output

    if not isinstance(last_output, tuple) or len(last_output) < 7:
        return {"error": "interview init failed"}

    radar_value = _extract_update_value(last_output[2])
    if radar_value:
        radar_value = os.path.abspath(str(radar_value))

    messages = last_output[4] if isinstance(last_output[4], list) else []
    context = last_output[6] if isinstance(last_output[6], dict) else {}
    thread_id = context.get("thread_id")
    if not thread_id:
        return {"error": "thread init failed"}

    state = db.get_conversation_state(thread_id) or {}
    extra = state.get("extra_state") or {}

    return {
        "thread_id": thread_id,
        "job_context": context.get("job_context") or job,
        "messages": messages,
        "radar_image_url": radar_value,
        "match_analysis": context.get("match_analysis", ""),
        "live_assessment_md": extra.get("live_assessment_md", ""),
        "scorecard": _normalize_scorecard_for_api(extra.get("scorecard")),
    }


async def stream_interview_chat(message: str, user_id: str, thread_id: str) -> AsyncGenerator[dict, None]:
    context, history, _ = _load_context_and_history(thread_id)
    previous_assistant = ""
    previous_scorecard = None
    previous_live_assessment = None

    for output in _legacy_handle_interview_chat(message, history, context, user_id):
        if not isinstance(output, tuple) or len(output) < 3:
            continue

        current_history = output[1] if isinstance(output[1], list) else []
        assistant_text = ""
        if current_history and isinstance(current_history[-1], dict):
            last_message = current_history[-1]
            if str(last_message.get("role", "")) == "assistant":
                assistant_text = str(last_message.get("content", ""))

        if assistant_text:
            if assistant_text.startswith(previous_assistant):
                delta = assistant_text[len(previous_assistant):]
            else:
                delta = assistant_text
            if delta:
                yield {"type": "token", "content": delta}
            previous_assistant = assistant_text

        state = db.get_conversation_state(thread_id) or {}
        extra = state.get("extra_state") or {}
        scorecard = _normalize_scorecard_for_api(extra.get("scorecard"))
        live_assessment_md = extra.get("live_assessment_md", "")
        if scorecard != previous_scorecard or live_assessment_md != previous_live_assessment:
            previous_scorecard = dict(scorecard)
            previous_live_assessment = str(live_assessment_md)
            yield {
                "type": "score",
                "scorecard": scorecard,
                "live_assessment_md": live_assessment_md,
            }


async def end_interview(user_id: str, thread_id: str) -> AsyncGenerator[dict, None]:
    context, history, _ = _load_context_and_history(thread_id)

    final_report = ""
    for output in _legacy_end_interview_with_summary(history, context, user_id):
        if not isinstance(output, tuple) or len(output) < 1:
            continue
        current_history = output[0] if isinstance(output[0], list) else []
        if current_history and isinstance(current_history[-1], dict):
            final_report = str(current_history[-1].get("content", "")) or final_report

    if final_report:
        yield {"type": "report", "content": final_report}


def get_interview_state(user_id: str, thread_id: str) -> dict:
    _ = user_id
    session = db.get_conversation_state(thread_id) or {}
    extra = session.get("extra_state") or {}
    return {
        "thread_id": thread_id,
        "job_context": extra.get("job_context", {}),
        "messages": session.get("recent_messages") or [],
        "live_assessment_md": extra.get("live_assessment_md", ""),
        "scorecard": _normalize_scorecard_for_api(extra.get("scorecard")),
        "final_report": extra.get("final_report", ""),
    }
