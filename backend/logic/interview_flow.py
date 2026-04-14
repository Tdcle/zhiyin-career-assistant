import json
import time

import gradio as gr
from langchain_core.messages import AIMessage, HumanMessage

from models.interview_graph import interview_graph
from db import DatabaseManager
from utils.logger import get_logger
from utils.tools import analyze_job_match_tool

db = DatabaseManager()
logger = get_logger("interview_flow")
MAX_RECENT_MESSAGES = 8


def _parse_user_id(user_info) -> str:
    try:
        return str(user_info.split(" (")[0].strip())
    except (AttributeError, IndexError):
        return "guest"


def _new_interview_thread_id(user_id: str, job_id: str) -> str:
    return f"interview_{user_id}_{job_id}_{int(time.time())}"


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


def _safe_get_graph_state(graph_config):
    try:
        state = interview_graph.get_state(graph_config)
        return state.values if state and getattr(state, "values", None) else {}
    except Exception:
        return {}


def _compose_analysis_panel(match_analysis: str, live_assessment_md: str) -> str:
    parts = []
    if match_analysis:
        parts.append("### 岗位匹配分析")
        parts.append(match_analysis)
    if live_assessment_md:
        parts.append(live_assessment_md)
    return "\n\n".join(parts) if parts else "### 岗位匹配分析\n暂无数据"


def _persist_interview_state(user_id: str, thread_id: str, graph_values: dict, job_context: dict, match_analysis: str):
    db.upsert_conversation_state(
        thread_id=thread_id,
        user_id=user_id,
        scene="interview",
        summary=graph_values.get("summary", ""),
        recent_messages=_serialize_recent_messages(graph_values.get("messages", [])),
        extra_state={
            "job_context": job_context,
            "final_report": graph_values.get("final_report", ""),
            "scorecard": graph_values.get("scorecard") or {},
            "evidence_log": graph_values.get("evidence_log") or [],
            "covered_topics": graph_values.get("covered_topics") or [],
            "uncovered_topics": graph_values.get("uncovered_topics") or [],
            "recommended_next_focus": graph_values.get("recommended_next_focus", ""),
            "gap_hypotheses": graph_values.get("gap_hypotheses") or [],
            "risk_flags": graph_values.get("risk_flags") or [],
            "live_assessment_md": graph_values.get("live_assessment_md", ""),
            "interview_round": graph_values.get("interview_round", 0),
            "match_analysis": match_analysis,
        },
    )


def start_interview_logic(job_data_dict, user_info):
    if not job_data_dict or not isinstance(job_data_dict, dict):
        yield (gr.update(),) * 9
        return

    selected_job_id = str(job_data_dict.get("job_id"))
    user_id = _parse_user_id(user_info)
    thread_id = _new_interview_thread_id(user_id, selected_job_id)
    graph_config = {"configurable": {"thread_id": thread_id}}

    yield (
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(value=None),
        "### 岗位匹配分析\n正在准备面试上下文...",
        [{"role": "assistant", "content": "正在连接面试官..."}],
        "## 正在创建面试会话...",
        {"thread_id": thread_id, "job_context": job_data_dict},
        gr.update(interactive=True, value="结束面试并生成总结"),
        gr.update(visible=True),
    )

    job_detail = db.get_job_details(selected_job_id)
    if not job_detail:
        yield (
            gr.update(),
            gr.update(),
            gr.update(),
            "### 岗位匹配分析\n未找到该岗位。",
            [],
            "## 面试初始化失败",
            {"thread_id": thread_id, "job_context": {}},
            gr.update(visible=False),
            gr.update(visible=True),
        )
        return

    radar_path = None
    match_analysis = "暂时无法生成岗位匹配分析。"
    try:
        match_raw = analyze_job_match_tool.invoke(
            {"user_id": user_id, "job_description": job_detail.get("detail", "")}
        )
        match_text = match_raw.content if hasattr(match_raw, "content") else str(match_raw)
        match_data = json.loads(match_text)
        radar_path = match_data.get("image_path")
        match_analysis = match_data.get("analysis") or match_analysis
    except Exception as exc:
        logger.error("match analysis failed: %s", exc, exc_info=True)

    inputs = {
        "messages": [],
        "user_id": user_id,
        "job_context": job_detail,
        "phase": "opening",
        "summary": "",
        "scorecard": {},
        "evidence_log": [],
        "covered_topics": [],
        "uncovered_topics": [],
        "recommended_next_focus": "自我介绍",
        "gap_hypotheses": [],
        "risk_flags": [],
        "interview_round": 0,
        "should_end": False,
    }

    try:
        interview_graph.invoke(inputs, config=graph_config)
        graph_values = _safe_get_graph_state(graph_config)
    except Exception as exc:
        logger.error("interview opening failed: %s", exc, exc_info=True)
        graph_values = {}

    live_panel = _compose_analysis_panel(match_analysis, graph_values.get("live_assessment_md", ""))
    chatbot_history = [
        {"role": "assistant", "content": message.content}
        for message in graph_values.get("messages", [])
        if isinstance(message, AIMessage) and message.content
    ]
    if not chatbot_history:
        chatbot_history = [{"role": "assistant", "content": "请先做一个简短的自我介绍。"}]

    _persist_interview_state(user_id, thread_id, graph_values, job_detail, match_analysis)

    yield (
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(value=radar_path),
        live_panel,
        chatbot_history,
        f"## 模拟面试: {job_detail.get('title', '目标岗位')}",
        {"thread_id": thread_id, "job_context": job_detail, "match_analysis": match_analysis},
        gr.update(interactive=True, value="结束面试并生成总结"),
        gr.update(visible=True),
    )


def handle_interview_chat(message, history, context_state, user_info):
    if not message.strip():
        yield "", history, _compose_analysis_panel((context_state or {}).get("match_analysis", ""), "")
        return

    context_state = context_state or {}
    job_context = context_state.get("job_context", {})
    match_analysis = context_state.get("match_analysis", "")
    user_id = _parse_user_id(user_info)
    thread_id = context_state.get("thread_id")
    if not thread_id:
        yield "", history, _compose_analysis_panel(match_analysis, "### 实时评估面板\n缺少会话上下文。")
        return

    graph_config = {"configurable": {"thread_id": thread_id}}

    history.append({"role": "user", "content": message})
    yield "", history, _compose_analysis_panel(match_analysis, "### 实时评估面板\n正在更新评估...")

    graph_values = _safe_get_graph_state(graph_config)
    has_live_state = bool(graph_values.get("messages"))

    if has_live_state:
        inputs = {
            "messages": [HumanMessage(content=message)],
            "user_id": user_id,
            "job_context": job_context,
            "phase": "interviewing",
            "should_end": False,
        }
    else:
        persisted = db.get_conversation_state(thread_id) or {}
        extra_state = persisted.get("extra_state") or {}
        inputs = {
            "messages": _deserialize_recent_messages(persisted.get("recent_messages")) + [HumanMessage(content=message)],
            "user_id": user_id,
            "job_context": extra_state.get("job_context", job_context),
            "phase": "interviewing",
            "summary": persisted.get("summary", ""),
            "scorecard": extra_state.get("scorecard", {}),
            "evidence_log": extra_state.get("evidence_log", []),
            "covered_topics": extra_state.get("covered_topics", []),
            "uncovered_topics": extra_state.get("uncovered_topics", []),
            "recommended_next_focus": extra_state.get("recommended_next_focus", ""),
            "gap_hypotheses": extra_state.get("gap_hypotheses", []),
            "risk_flags": extra_state.get("risk_flags", []),
            "interview_round": extra_state.get("interview_round", 0),
            "should_end": False,
        }
        match_analysis = extra_state.get("match_analysis", match_analysis)

    history.append({"role": "assistant", "content": ""})

    try:
        event_stream = interview_graph.stream(inputs, config=graph_config, stream_mode="messages")

        full_content = ""
        for msg, metadata in event_stream:
            node_name = metadata.get("langgraph_node")
            if node_name in ("summarize_node", "assessment_node"):
                continue
            if isinstance(msg, AIMessage) and msg.content:
                full_content += msg.content
                history[-1]["content"] = full_content
                current_state = _safe_get_graph_state(graph_config)
                panel = _compose_analysis_panel(match_analysis, current_state.get("live_assessment_md", ""))
                yield "", history, panel

        final_values = _safe_get_graph_state(graph_config)
        if final_values:
            _persist_interview_state(user_id, thread_id, final_values, job_context, match_analysis)

        panel = _compose_analysis_panel(match_analysis, final_values.get("live_assessment_md", ""))
        yield "", history, panel

    except Exception as exc:
        logger.error("interview graph failed: %s", exc, exc_info=True)
        history[-1]["content"] = f"系统错误: {exc}"
        yield "", history, _compose_analysis_panel(match_analysis, "### 实时评估面板\n评估更新失败。")


def end_interview_with_summary(history, context_state, user_info):
    context_state = context_state or {}
    job_context = context_state.get("job_context", {})
    match_analysis = context_state.get("match_analysis", "")
    user_id = _parse_user_id(user_info)
    thread_id = context_state.get("thread_id")
    if not thread_id:
        yield (
            history + [{"role": "assistant", "content": "缺少面试会话，无法生成总结。"}],
            gr.update(visible=False),
            gr.update(visible=True, value="返回求职大厅"),
            _compose_analysis_panel(match_analysis, "### 实时评估面板\n缺少会话上下文。"),
        )
        return

    graph_config = {"configurable": {"thread_id": thread_id}}
    new_history = list(history)
    new_history.append({"role": "assistant", "content": "正在生成面试评估报告，请稍候..."})

    yield (
        new_history,
        gr.update(interactive=False, value="正在生成报告..."),
        gr.update(visible=True),
        _compose_analysis_panel(match_analysis, "### 实时评估面板\n正在汇总结构化状态..."),
    )

    inputs = {
        "messages": [],
        "user_id": user_id,
        "job_context": job_context,
        "phase": "finalizing",
        "should_end": True,
    }

    try:
        existing_values = _safe_get_graph_state(graph_config)
        if existing_values.get("messages"):
            inputs.update({
                "summary": existing_values.get("summary", ""),
                "scorecard": existing_values.get("scorecard", {}),
                "evidence_log": existing_values.get("evidence_log", []),
                "covered_topics": existing_values.get("covered_topics", []),
                "uncovered_topics": existing_values.get("uncovered_topics", []),
                "recommended_next_focus": existing_values.get("recommended_next_focus", ""),
                "gap_hypotheses": existing_values.get("gap_hypotheses", []),
                "risk_flags": existing_values.get("risk_flags", []),
                "interview_round": existing_values.get("interview_round", 0),
            })
        else:
            persisted = db.get_conversation_state(thread_id) or {}
            extra_state = persisted.get("extra_state") or {}
            inputs.update({
                "messages": _deserialize_recent_messages(persisted.get("recent_messages")),
                "summary": persisted.get("summary", ""),
                "job_context": extra_state.get("job_context", job_context),
                "scorecard": extra_state.get("scorecard", {}),
                "evidence_log": extra_state.get("evidence_log", []),
                "covered_topics": extra_state.get("covered_topics", []),
                "uncovered_topics": extra_state.get("uncovered_topics", []),
                "recommended_next_focus": extra_state.get("recommended_next_focus", ""),
                "gap_hypotheses": extra_state.get("gap_hypotheses", []),
                "risk_flags": extra_state.get("risk_flags", []),
                "interview_round": extra_state.get("interview_round", 0),
            })
            match_analysis = extra_state.get("match_analysis", match_analysis)

        interview_graph.invoke(inputs, config=graph_config)
        final_values = _safe_get_graph_state(graph_config)
        report_text = final_values.get("final_report") or "报告生成为空，请稍后重试。"
        _persist_interview_state(user_id, thread_id, final_values, inputs["job_context"], match_analysis)

        new_history[-1] = {"role": "assistant", "content": report_text}
        panel = _compose_analysis_panel(match_analysis, final_values.get("live_assessment_md", ""))
        yield (
            new_history,
            gr.update(visible=False),
            gr.update(visible=True, value="返回求职大厅"),
            panel,
        )

    except Exception as exc:
        logger.error("interview report generation failed: %s", exc, exc_info=True)
        new_history[-1] = {"role": "assistant", "content": f"总结生成失败: {exc}"}
        yield (
            new_history,
            gr.update(visible=False),
            gr.update(visible=True, value="返回求职大厅"),
            _compose_analysis_panel(match_analysis, "### 实时评估面板\n报告生成失败。"),
        )
