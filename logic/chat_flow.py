import os
import gradio as gr
from langchain_core.messages import AIMessage, HumanMessage

from models.chat_graph import app_graph
from utils.database import DatabaseManager
from utils.file_parser import FileParser
from utils.logger import get_logger

db = DatabaseManager()
logger = get_logger("chat_flow")
CHAT_THREAD_PREFIX = "chat_"
MAX_RECENT_MESSAGES = 8


def _parse_user_id(user_info) -> str:
    try:
        return str(user_info.split(" (")[0].strip())
    except (AttributeError, IndexError):
        return "guest"


def _chat_thread_id(user_id: str) -> str:
    return f"{CHAT_THREAD_PREFIX}{user_id}"


def _serialize_recent_messages(messages) -> list[dict]:
    serialized = []
    for message in messages:
        if isinstance(message, HumanMessage):
            serialized.append({"role": "user", "content": message.content})
        elif isinstance(message, AIMessage):
            serialized.append({"role": "assistant", "content": message.content})
    return serialized[-MAX_RECENT_MESSAGES:]


def _deserialize_recent_messages(items) -> list:
    messages = []
    for item in items or []:
        role = item.get("role")
        content = item.get("content", "")
        if not content:
            continue
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    return messages


def _build_chatbot_history(items) -> list[dict]:
    return [{"role": item.get("role", "assistant"), "content": item.get("content", "")} for item in items or []]


def _build_job_button_updates(search_results):
    hidden = [gr.update(visible=False, value="") for _ in range(6)]
    if not search_results:
        return hidden

    updates = []
    for job in search_results[:6]:
        tags_clean = (job.get("tags", "") or "").replace("|", "·")
        btn_text = (
            f"🏢 {job.get('company', '未知公司')}\n"
            f"📌 {job.get('title', '未知职位')}\n"
            f"💰 {job.get('salary', '未知')}   📍 {tags_clean}"
        )
        updates.append(gr.update(visible=True, value=btn_text))

    while len(updates) < 6:
        updates.append(gr.update(visible=False, value=""))

    return updates


def _safe_get_graph_state(graph_config):
    try:
        state = app_graph.get_state(graph_config)
        return state.values if state and getattr(state, "values", None) else {}
    except Exception:
        return {}


def _persist_chat_state(user_id: str, graph_values: dict):
    recent_messages = _serialize_recent_messages(graph_values.get("messages", []))
    db.upsert_conversation_state(
        thread_id=_chat_thread_id(user_id),
        user_id=user_id,
        scene="chat",
        summary=graph_values.get("summary", ""),
        recent_messages=recent_messages,
        extra_state={
            "search_results": graph_values.get("search_results") or [],
            "search_plan": graph_values.get("search_plan") or {},
        },
    )


def process_uploaded_resume(file_path, user_info):
    if not file_path:
        return "无文件"

    user_id = _parse_user_id(user_info)
    if user_id == "guest":
        return "⚠️ 请先选择有效的用户身份"

    logger.info("resume uploaded: user=%s path=%s", user_id, file_path)
    raw_content = FileParser.read_file(file_path)
    if not raw_content or len(raw_content) < 10:
        return "❌ 解析失败或内容过少"

    filename = os.path.basename(file_path)
    success, msg = db.save_resume(user_id, filename, raw_content)
    return f"✅ {msg}\n({filename})" if success else f"❌ 处理失败: {msg}"


def load_user_chat_session(user_info):
    user_id = _parse_user_id(user_info)
    session = db.get_conversation_state(_chat_thread_id(user_id))
    recent_messages = (session or {}).get("recent_messages", [])
    extra_state = (session or {}).get("extra_state", {}) or {}
    search_results = extra_state.get("search_results", [])
    return _build_chatbot_history(recent_messages), search_results, *_build_job_button_updates(search_results)


def clear_user_chat_session(user_info):
    user_id = _parse_user_id(user_info)
    thread_id = _chat_thread_id(user_id)
    db.delete_conversation_state(thread_id)

    checkpointer = getattr(app_graph, "checkpointer", None)
    if checkpointer and hasattr(checkpointer, "delete_thread"):
        try:
            checkpointer.delete_thread(thread_id)
        except Exception:
            logger.warning("clear in-memory chat state failed for thread=%s", thread_id, exc_info=True)

    logger.info("chat session cleared: user=%s thread=%s", user_id, thread_id)
    return [], "", [], *_build_job_button_updates([])


def respond(message, chat_history, user_info):
    current_btn_updates = [gr.update(visible=False, value="") for _ in range(6)]
    current_jobs_data = []

    if not message.strip():
        yield chat_history, "", current_jobs_data, *current_btn_updates
        return

    user_id = _parse_user_id(user_info)
    thread_id = _chat_thread_id(user_id)
    graph_config = {"configurable": {"thread_id": thread_id}}

    logger.info("chat message received: user=%s", user_id)

    chat_history.append({"role": "user", "content": message})
    yield chat_history, "", current_jobs_data, *current_btn_updates

    graph_values = _safe_get_graph_state(graph_config)
    has_live_state = bool(graph_values.get("messages"))

    if has_live_state:
        inputs = {
            "messages": [HumanMessage(content=message)],
            "user_id": user_id,
        }
    else:
        persisted = db.get_conversation_state(thread_id) or {}
        inputs = {
            "messages": _deserialize_recent_messages(persisted.get("recent_messages")) + [HumanMessage(content=message)],
            "user_id": user_id,
            "summary": persisted.get("summary", ""),
            "search_results": (persisted.get("extra_state") or {}).get("search_results", []),
            "search_plan": (persisted.get("extra_state") or {}).get("search_plan", {}),
        }

    chat_history.append({"role": "assistant", "content": "🤖 正在思考..."})
    full_response = ""

    try:
        event_stream = app_graph.stream(inputs, config=graph_config, stream_mode="messages")
        hidden_nodes = {
            "summarize_node",
            "extract_search_results",
            "intent_parse_node",
            "result_judge_node",
            "rewrite_search_plan_node",
        }

        for msg, metadata in event_stream:
            node_name = metadata.get("langgraph_node", "")
            if node_name in hidden_nodes:
                continue

            if isinstance(msg, AIMessage):
                if msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        if tool_call["name"] == "search_jobs_tool":
                            full_response += "\n正在检索职位数据库...\n\n"
                        elif tool_call["name"] == "analyze_job_match_by_query_tool":
                            full_response += "\n正在分析你与目标岗位的匹配度...\n\n"
                    chat_history[-1]["content"] = full_response
                    yield chat_history, "", current_jobs_data, *current_btn_updates

                if msg.content:
                    full_response += msg.content
                    chat_history[-1]["content"] = full_response
                    yield chat_history, "", current_jobs_data, *current_btn_updates

        final_values = _safe_get_graph_state(graph_config)
        if final_values:
            _persist_chat_state(user_id, final_values)
            search_results = final_values.get("search_results") or []
        else:
            search_results = []

        if search_results:
            current_jobs_data = search_results
            current_btn_updates = _build_job_button_updates(search_results)

        yield chat_history, "", current_jobs_data, *current_btn_updates

    except Exception as exc:
        logger.error("chat flow failed: %s", exc, exc_info=True)
        chat_history[-1]["content"] = f"{chat_history[-1]['content']}\n\n❌ 系统错误: {exc}"
        yield chat_history, "", current_jobs_data, *current_btn_updates
