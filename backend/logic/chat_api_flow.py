"""Chat API flow with multi-session support."""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncGenerator
from datetime import datetime
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage

from models.chat_graph import app_graph
from db import DatabaseManager
from utils.logger import get_logger


db = DatabaseManager()
logger = get_logger("chat_api_flow")

MAX_RECENT_MESSAGES = 8
CHAT_SCENE = "chat"
CHAT_THREAD_PREFIX = "chat:"
DEFAULT_SESSION_TITLE = "新会话"


def _now_iso(value) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value or "")


def _make_thread_id(user_id: str) -> str:
    return f"{CHAT_THREAD_PREFIX}{user_id}:{uuid4().hex[:12]}"


def _serialize_recent_messages(messages) -> list[dict]:
    serialized: list[dict] = []
    for message in messages:
        if isinstance(message, HumanMessage):
            serialized.append({"role": "user", "content": str(message.content)})
        elif isinstance(message, AIMessage):
            serialized.append({"role": "assistant", "content": str(message.content)})
    return serialized[-MAX_RECENT_MESSAGES:]


def _deserialize_recent_messages(items) -> list:
    messages = []
    for item in items or []:
        role = str((item or {}).get("role", "")).strip()
        content = str((item or {}).get("content", "")).strip()
        if not content:
            continue
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    return messages


def _build_chatbot_history(items) -> list[dict]:
    rows: list[dict] = []
    for item in items or []:
        rows.append(
            {
                "role": str((item or {}).get("role", "assistant")),
                "content": str((item or {}).get("content", "")),
            }
        )
    return rows


def _safe_get_graph_state(graph_config):
    try:
        state = app_graph.get_state(graph_config)
        return state.values if state and getattr(state, "values", None) else {}
    except Exception:
        return {}


def _normalize_title(raw: str) -> str:
    text = re.sub(r"\s+", " ", (raw or "").strip())
    if not text:
        return DEFAULT_SESSION_TITLE
    return text[:30]


def _derive_title(existing_title: str, recent_messages: list[dict], latest_user_message: str) -> str:
    title = _normalize_title(existing_title)
    if title != DEFAULT_SESSION_TITLE:
        return title

    for item in recent_messages:
        if item.get("role") == "user" and item.get("content"):
            return _normalize_title(str(item["content"]))

    return _normalize_title(latest_user_message)


def _session_item_from_row(row: dict) -> dict:
    extra_state = (row or {}).get("extra_state") or {}
    recent_messages = (row or {}).get("recent_messages") or []

    title = extra_state.get("session_title") or ""
    if not title:
        title = _derive_title("", recent_messages, "")

    return {
        "session_id": str((row or {}).get("thread_id", "")),
        "title": _normalize_title(str(title)),
        "updated_at": _now_iso((row or {}).get("updated_at")),
    }


def list_user_chat_sessions(user_id: str) -> list[dict]:
    rows = db.list_conversation_states(user_id=user_id, scene=CHAT_SCENE, limit=100)
    return [_session_item_from_row(row) for row in rows]


def create_user_chat_session(user_id: str, title: str | None = None) -> dict:
    session_title = _normalize_title(title or DEFAULT_SESSION_TITLE)
    thread_id = _make_thread_id(user_id)

    db.upsert_conversation_state(
        thread_id=thread_id,
        user_id=user_id,
        scene=CHAT_SCENE,
        summary="",
        recent_messages=[],
        extra_state={
            "search_results": [],
            "search_plan": {},
            "session_title": session_title,
        },
    )

    row = db.get_conversation_state_for_user(user_id=user_id, thread_id=thread_id, scene=CHAT_SCENE)
    return _session_item_from_row(
        row
        or {
            "thread_id": thread_id,
            "extra_state": {"session_title": session_title},
            "updated_at": datetime.utcnow().isoformat(),
            "recent_messages": [],
        }
    )


def _ensure_session_row(user_id: str, session_id: str | None = None) -> tuple[str, dict]:
    if session_id:
        row = db.get_conversation_state_for_user(user_id=user_id, thread_id=session_id, scene=CHAT_SCENE)
        if not row:
            raise KeyError("session not found")
        return session_id, row

    sessions = list_user_chat_sessions(user_id)
    if sessions:
        latest_session_id = sessions[0]["session_id"]
        row = db.get_conversation_state_for_user(
            user_id=user_id,
            thread_id=latest_session_id,
            scene=CHAT_SCENE,
        )
        if row:
            return latest_session_id, row

    created = create_user_chat_session(user_id=user_id)
    created_row = db.get_conversation_state_for_user(
        user_id=user_id,
        thread_id=created["session_id"],
        scene=CHAT_SCENE,
    )
    if not created_row:
        raise RuntimeError("failed to initialize chat session")
    return created["session_id"], created_row


def load_user_chat_session(user_id: str, session_id: str | None = None) -> tuple[str, list[dict], list[dict]]:
    active_session_id, row = _ensure_session_row(user_id=user_id, session_id=session_id)
    recent_messages = (row or {}).get("recent_messages", []) or []
    extra_state = (row or {}).get("extra_state", {}) or {}
    search_results = extra_state.get("search_results", []) or []
    return active_session_id, _build_chatbot_history(recent_messages), list(search_results)


def delete_user_chat_session(user_id: str, session_id: str) -> bool:
    deleted = db.delete_conversation_state_for_user(
        user_id=user_id,
        thread_id=session_id,
        scene=CHAT_SCENE,
    )
    if not deleted:
        return False

    checkpointer = getattr(app_graph, "checkpointer", None)
    if checkpointer and hasattr(checkpointer, "delete_thread"):
        try:
            checkpointer.delete_thread(session_id)
        except Exception:
            logger.warning("clear in-memory chat state failed for thread=%s", session_id, exc_info=True)
    return True


def _persist_chat_state(
    user_id: str,
    session_id: str,
    graph_values: dict,
    existing_title: str,
    latest_user_message: str,
) -> None:
    recent_messages = _serialize_recent_messages(graph_values.get("messages", []))
    session_title = _derive_title(existing_title, recent_messages, latest_user_message)
    db.upsert_conversation_state(
        thread_id=session_id,
        user_id=user_id,
        scene=CHAT_SCENE,
        summary=str(graph_values.get("summary", "") or ""),
        recent_messages=recent_messages,
        extra_state={
            "search_results": graph_values.get("search_results") or [],
            "search_plan": graph_values.get("search_plan") or {},
            "session_title": session_title,
        },
    )


def _tool_call_hint(tool_name: str) -> str:
    if tool_name == "search_jobs_tool":
        return "正在检索职位数据库..."
    if tool_name == "analyze_job_match_by_query_tool":
        return "正在分析你与目标岗位的匹配度..."
    return "正在调用工具处理..."


def _chunk_text_for_sse(text: str, chunk_size: int = 12) -> list[str]:
    content = (text or "").strip()
    if not content:
        return []

    if len(content) <= chunk_size:
        return [content]

    chunks: list[str] = []
    start = 0
    boundary_chars = set("，。！？；：,.!?;:\n ")

    while start < len(content):
        end = min(len(content), start + chunk_size)
        split = end

        # Prefer natural boundaries to keep chunks readable.
        for idx in range(end, start, -1):
            if content[idx - 1] in boundary_chars and idx - start >= 4:
                split = idx
                break

        piece = content[start:split]
        if not piece:
            piece = content[start:end]
            split = end

        chunks.append(piece)
        start = split

    return chunks


async def stream_chat(message: str, user_id: str, session_id: str | None = None) -> AsyncGenerator[dict, None]:
    active_session_id, session_row = _ensure_session_row(user_id=user_id, session_id=session_id)
    extra_state = (session_row or {}).get("extra_state", {}) or {}
    current_title = str(extra_state.get("session_title", DEFAULT_SESSION_TITLE))

    graph_config = {"configurable": {"thread_id": active_session_id}}
    graph_values = _safe_get_graph_state(graph_config)
    has_live_state = bool(graph_values.get("messages"))

    if has_live_state:
        inputs = {
            "messages": [HumanMessage(content=message)],
            "user_id": user_id,
        }
    else:
        persisted = session_row or {}
        persisted_extra = (persisted.get("extra_state") or {}) if isinstance(persisted, dict) else {}
        inputs = {
            "messages": _deserialize_recent_messages(persisted.get("recent_messages")) + [HumanMessage(content=message)],
            "user_id": user_id,
            "summary": persisted.get("summary", ""),
            "search_results": persisted_extra.get("search_results", []),
            "search_plan": persisted_extra.get("search_plan", {}),
        }

    previous_jobs = list(extra_state.get("search_results", []) or [])

    hidden_nodes = {
        "summarize_node",
        "extract_search_results",
        "intent_parse_node",
        "result_judge_node",
        "rewrite_search_plan_node",
    }

    for msg, metadata in app_graph.stream(inputs, config=graph_config, stream_mode="messages"):
        node_name = (metadata or {}).get("langgraph_node", "")
        if node_name in hidden_nodes:
            continue

        if not isinstance(msg, AIMessage):
            continue

        for tool_call in getattr(msg, "tool_calls", None) or []:
            tool_name = str(tool_call.get("name", "")).strip()
            if tool_name:
                yield {"type": "tool_call", "hint": _tool_call_hint(tool_name)}

        content = msg.content
        if isinstance(content, str) and content:
            # If model returns a large block, split it for better perceived streaming.
            if len(content) > 24:
                for piece in _chunk_text_for_sse(content):
                    if not piece:
                        continue
                    yield {"type": "token", "content": piece}
                    await asyncio.sleep(0.015)
            else:
                yield {"type": "token", "content": content}

    final_values = _safe_get_graph_state(graph_config)
    if final_values:
        _persist_chat_state(
            user_id=user_id,
            session_id=active_session_id,
            graph_values=final_values,
            existing_title=current_title,
            latest_user_message=message,
        )
        latest_jobs = final_values.get("search_results") or []
        if latest_jobs != previous_jobs:
            yield {"type": "jobs", "jobs": latest_jobs}
