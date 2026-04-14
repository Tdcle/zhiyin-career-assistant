import json
import re
from typing import Annotated, Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

from config.config import config
from models.search_plan_resolver import apply_memory_resolution_to_plan, extract_turn_slot_signals
from db import DatabaseManager
from utils.logger import get_logger
from utils.tools import (
    _extract_role_keywords,
    _normalize_keyword_text,
    analyze_job_match_by_query_tool,
    get_user_resume_tool,
    save_preference_tool,
    search_jobs_tool,
)

from models.chat_search_helpers import (
    _apply_explicit_constraint_guard,
    _build_result_summary,
    _close_page_instance,
    _enforce_rewrite_constraints,
    _extract_explicit_constraints,
    _filter_results_by_relevance,
    _heuristic_search_plan,
    _hydrate_results_for_render,
    _is_check_fresh,
    _is_non_search_request,
    _is_transient_unknown_reason,
    _latest_user_text,
    _looks_like_search_followup,
    _normalize_search_plan,
    _parse_checked_at,
    _parse_search_payload,
    _recent_dialogue,
    _render_search_results_text,
    _safe_json_loads,
    _sanitize_messages_for_llm,
)

db_manager = DatabaseManager()
logger = get_logger("chat_graph")


class AgentState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    user_id: str
    summary: str
    search_results: Optional[List[Dict[str, Any]]]
    filtered_search_results: Optional[List[Dict[str, Any]]]
    search_plan: Dict[str, Any]
    role_keywords: List[str]
    direct_search_mode: bool
    search_retry_count: int
    search_feedback: str
    search_response_text: str
    should_retry_search: bool
    search_action: str
    relevance_feedback: str
    matched_role_count: int
    mismatch_count: int


tools = [
    search_jobs_tool,
    save_preference_tool,
    get_user_resume_tool,
    analyze_job_match_by_query_tool,
]

agent_llm = config.create_tongyi(config.CHAT_MODELS.assistant_main, streaming=True)
summary_llm = config.create_tongyi(config.CHAT_MODELS.assistant_summary)
planner_llm = config.create_tongyi(config.CHAT_MODELS.assistant_planner)
rewrite_llm = config.create_tongyi(config.CHAT_MODELS.assistant_rewrite)

agent_with_tools = agent_llm.bind_tools(tools)

JOB_ALIVE_CHECK_MAX_RESULTS = 6
JOB_ALIVE_CHECK_FRESH_HOURS = 12
JOB_ALIVE_CHECK_TIMEOUT_SECONDS = 12
JOB_ALIVE_CHECK_POLL_INTERVAL_SECONDS = 0.6
ASSUME_ACTIVE_ON_TRANSIENT_UNKNOWN = True
ASSUME_ACTIVE_ON_UNKNOWN = True


def summarize_node(state: AgentState):
    summary = state.get("summary", "")
    messages = state.get("messages", [])

    if summary:
        summary_instruction = (
            f"已有摘要：{summary}\n"
            "请把已有摘要和最新对话合并成一份更紧凑、可复用的求职摘要。"
        )
    else:
        summary_instruction = "请基于下面对话生成一份紧凑求职摘要。"

    messages_content = "\n".join(
        f"{message.type}: {message.content}"
        for message in messages
        if isinstance(message, (HumanMessage, AIMessage)) and message.content
    )
    if not messages_content:
        return {}

    prompt = (
        f"{summary_instruction}\n\n"
        "要求：\n"
        "1. 保留稳定信息：技能、经验、学历、城市、薪资、岗位偏好。\n"
        "2. 保留当前筛选条件和最近讨论过的岗位方向。\n"
        "3. 忽略寒暄和重复表达。\n\n"
        f"对话：\n{messages_content}"
    )
    response = summary_llm.invoke(prompt)
    delete_messages = [RemoveMessage(id=message.id) for message in messages[:-4]]
    return {"summary": response.content.strip(), "messages": delete_messages}


def route_from_start(state: AgentState):
    if len(state.get("messages", [])) > 10:
        return "summarize_node"
    return "intent_parse_node"


def intent_parse_node(state: AgentState):
    latest_text = _latest_user_text(state.get("messages", []))
    if not latest_text:
        return {"direct_search_mode": False}

    user_id = state.get("user_id", "")
    previous_plan = state.get("search_plan") or {}
    explicit_constraints_for_memory = _extract_explicit_constraints(latest_text)

    if _is_non_search_request(latest_text):
        return {
            "direct_search_mode": False,
            "search_response_text": "",
            "search_retry_count": 0,
            "search_feedback": "",
            "should_retry_search": False,
            "search_action": "",
            "relevance_feedback": "",
        }

    explicit_agent_signals = [
        "简历",
        "适合我",
        "匹配",
        "分析",
        "面试",
        "岗位匹配",
    ]
    if any(token in latest_text for token in explicit_agent_signals):
        return {"direct_search_mode": False}

    if user_id:
        try:
            memory_result = db_manager.ingest_user_memory_from_text(
                user_id=user_id,
                text=latest_text,
                source="chat",
            )
            if memory_result.get("updated"):
                logger.info(
                    "memory ingest updated: user=%s profile_updated=%s facts_added=%s",
                    user_id,
                    memory_result.get("profile_updated", False),
                    memory_result.get("facts_added", 0),
                )
        except Exception as exc:
            logger.warning("memory ingest failed: user=%s err=%s", user_id, exc)
        try:
            if explicit_constraints_for_memory.get("experience_mentioned"):
                explicit_exp = str(explicit_constraints_for_memory.get("experience", "") or "").strip()
                if explicit_exp in {"实习", "应届", "非实习", ""}:
                    profile = db_manager.get_memory_profile(user_id)
                    profile["desired_experience"] = explicit_exp
                    db_manager.upsert_memory_profile(user_id, profile)
                    db_manager.add_memory_fact(
                        user_id=user_id,
                        fact_key="desired_experience",
                        fact_value=explicit_exp or "不限",
                        source="chat",
                        confidence=0.92,
                        importance=5,
                        meta={
                            "scene": "chat",
                            "extractor": "intent_guard",
                            "polarity": "include" if explicit_exp else "neutral",
                        },
                    )
                    logger.info(
                        "memory experience guard applied: user=%s desired_experience=%s",
                        user_id,
                        explicit_exp or "<empty>",
                    )
        except Exception as exc:
            logger.warning("memory experience guard failed: user=%s err=%s", user_id, exc)

    summary = state.get("summary", "")
    recent_dialogue = _recent_dialogue(state.get("messages", []))
    prompt = f"""
你是求职助手的意图路由器。请判断“最新用户输入”是否要触发岗位搜索。

历史摘要：
{summary or "暂无"}

最近对话：
{recent_dialogue or "暂无"}

最新用户输入：
{latest_text}

请严格输出 JSON（不要 markdown）：
{{
  "intent": "search|summary|qa|match|resume|other",
  "confidence": 0.0,
  "search_plan": {{
    "resolved_query": "",
    "keyword_query": "",
    "city": "",
    "company": "",
    "experience": "",
    "salary_min": 0,
    "salary_unit": ""
  }},
  "reason": ""
}}

判定规则：
1. 只有明确“找岗位/筛岗位/继续搜/改搜索条件”时 intent=search。
2. “总结/概括/一句话/回顾/你记住了什么”都不是 search。
3. confidence 范围 0~1。
4. 如果不是 search，search_plan 允许为空。
5. 若是 search，search_plan 需尽量结构化，字段要求：
   - city 仅单个标准城市或空字符串
   - experience 仅简短值（如 实习/应届/1-3年）
   - salary_unit 仅 k_month / yuan_day / ""
"""
    try:
        parsed = _safe_json_loads(planner_llm.invoke(prompt).content)
    except Exception as exc:
        logger.warning("intent planner failed, disable direct search: %s", exc)
        return {
            "direct_search_mode": False,
            "search_response_text": "",
            "search_retry_count": 0,
            "search_feedback": "",
            "should_retry_search": False,
            "search_action": "",
            "relevance_feedback": "",
        }

    intent = str(parsed.get("intent", "other") or "other").strip().lower()
    try:
        confidence = float(parsed.get("confidence", 0) or 0)
    except Exception:
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    search_conf_threshold = 0.60

    llm_plan = parsed.get("search_plan") or {}
    normalized_plan = _normalize_search_plan(llm_plan, fallback_text=latest_text)
    normalized_plan, explicit_constraints = _apply_explicit_constraint_guard(normalized_plan, latest_text)
    slot_signals = extract_turn_slot_signals(latest_text)
    role_keywords = _extract_role_keywords(
        normalized_plan.get("resolved_query", ""),
        normalized_plan.get("keyword_query", ""),
    )
    if slot_signals.get("role_clear", False):
        role_keywords = []

    if slot_signals.get("city_mentioned", False):
        merged_city = normalized_plan.get("city", "")
    else:
        merged_city = normalized_plan.get("city") or previous_plan.get("city", "")

    merged_plan = {
        "resolved_query": normalized_plan.get("resolved_query") or previous_plan.get("resolved_query", latest_text),
        "keyword_query": normalized_plan.get("keyword_query") or previous_plan.get("keyword_query", latest_text),
        "city": merged_city,
        "company": normalized_plan.get("company") or previous_plan.get("company", ""),
        "experience": normalized_plan.get("experience", ""),
        "salary_min": int(normalized_plan.get("salary_min", 0) or 0),
        "salary_unit": normalized_plan.get("salary_unit", ""),
    }
    should_search = intent == "search" and confidence >= search_conf_threshold
    memory_resolution: dict = {}
    memory_apply_trace: dict = {}

    if should_search and user_id:
        try:
            memory_resolution = db_manager.resolve_memory_preferences(
                user_id=user_id,
                query=latest_text,
                scene="chat",
                limit=80,
            )
            merged_plan, memory_apply_trace = apply_memory_resolution_to_plan(
                plan=merged_plan,
                latest_text=latest_text,
                explicit_constraints=explicit_constraints,
                memory_resolution=memory_resolution,
                slot_signals=slot_signals,
            )
            role_keywords = _extract_role_keywords(
                merged_plan.get("resolved_query", ""),
                merged_plan.get("keyword_query", ""),
            )
            logger.info(
                "memory resolver: facts=%s roles_include=%s roles_exclude=%s cities_include=%s cities_exclude=%s experience=%s salary=%s/%s applied=%s",
                memory_resolution.get("fact_count", 0),
                ",".join(memory_resolution.get("roles_include", [])[:5]) or "<none>",
                ",".join(memory_resolution.get("roles_exclude", [])[:5]) or "<none>",
                ",".join(memory_resolution.get("cities_include", [])[:5]) or "<none>",
                ",".join(memory_resolution.get("cities_exclude", [])[:5]) or "<none>",
                memory_resolution.get("experience_include", "") or "<none>",
                memory_resolution.get("salary_min", 0),
                memory_resolution.get("salary_unit", "") or "<none>",
                json.dumps(memory_apply_trace, ensure_ascii=False),
            )
        except Exception as exc:
            logger.warning("memory resolver failed: user=%s err=%s", user_id, exc)

    if (
        should_search
        and not role_keywords
        and previous_plan
        and not slot_signals.get("role_clear", False)
        and _looks_like_search_followup(latest_text, previous_plan)
    ):
        previous_role_keywords = _extract_role_keywords(
            previous_plan.get("resolved_query", ""),
            previous_plan.get("keyword_query", ""),
        )
        if previous_role_keywords:
            merged_plan["keyword_query"] = previous_plan.get("keyword_query", "") or merged_plan.get("keyword_query", "")
            if (
                slot_signals.get("city_mentioned", False)
                or explicit_constraints.get("experience_mentioned", False)
                or explicit_constraints.get("salary_mentioned", False)
            ):
                merged_plan["resolved_query"] = _normalize_keyword_text(
                    " ".join(
                        part
                        for part in [latest_text, " ".join(previous_role_keywords)]
                        if str(part or "").strip()
                    )
                ) or merged_plan.get("resolved_query", latest_text)
            else:
                merged_plan["resolved_query"] = _normalize_keyword_text(
                    " ".join(
                        part for part in [previous_plan.get("resolved_query", ""), latest_text] if str(part or "").strip()
                    )
                ) or merged_plan.get("resolved_query", latest_text)
            role_keywords = previous_role_keywords
            logger.info(
                "planner inherited role constraints from previous plan: role_keywords=%s keyword_query=%s",
                ",".join(previous_role_keywords),
                merged_plan.get("keyword_query", ""),
            )
    logger.info(
        "intent router: intent=%s confidence=%.2f should_search=%s threshold=%.2f reason=%s",
        intent,
        confidence,
        should_search,
        search_conf_threshold,
        str(parsed.get("reason", "") or "").strip(),
    )
    logger.info(
        "planner explicit constraint guard: experience_mentioned=%s experience=%s salary_mentioned=%s salary_min=%s salary_unit=%s",
        explicit_constraints.get("experience_mentioned", False),
        merged_plan.get("experience", ""),
        explicit_constraints.get("salary_mentioned", False),
        merged_plan.get("salary_min", 0),
        merged_plan.get("salary_unit", ""),
    )
    logger.info(
        "planner turn slot signals: role_mentioned=%s role_clear=%s city_mentioned=%s city_clear=%s",
        slot_signals.get("role_mentioned", False),
        slot_signals.get("role_clear", False),
        slot_signals.get("city_mentioned", False),
        slot_signals.get("city_clear", False),
    )
    if should_search:
        logger.info("planner normalized search plan: %s", json.dumps(merged_plan, ensure_ascii=False))

    return {
        "direct_search_mode": should_search,
        "search_plan": merged_plan if should_search else previous_plan,
        "search_results": [],
        "filtered_search_results": [],
        "role_keywords": role_keywords if should_search else [],
        "search_response_text": "",
        "search_retry_count": 0,
        "search_feedback": "",
        "should_retry_search": False,
        "search_action": "",
        "relevance_feedback": "",
    }


def route_after_intent(state: AgentState):
    if state.get("direct_search_mode"):
        return "search_node"
    return "bot_node"


def search_node(state: AgentState):
    plan = state.get("search_plan") or {}
    logger.info(
        "search node running: query=%s city=%s experience=%s salary_min=%s salary_unit=%s",
        plan.get("keyword_query", ""),
        plan.get("city", ""),
        plan.get("experience", ""),
        plan.get("salary_min", 0),
        plan.get("salary_unit", ""),
    )
    payload = search_jobs_tool.invoke(
        {
            "resolved_query": plan.get("resolved_query", ""),
            "keyword_query": plan.get("keyword_query", ""),
            "city": plan.get("city", ""),
            "company": plan.get("company", ""),
            "experience": plan.get("experience", ""),
            "salary_min": int(plan.get("salary_min", 0) or 0),
            "salary_unit": plan.get("salary_unit", ""),
        }
    )
    content = payload.content if hasattr(payload, "content") else str(payload)
    search_results, search_response_text = _parse_search_payload(content)
    return {
        "search_results": search_results,
        "search_response_text": search_response_text,
    }


def bot_node(state: AgentState):
    user_id = state["user_id"]
    summary = state.get("summary", "")
    latest_query = _latest_user_text(state.get("messages", []))
    long_term_profile = db_manager.build_memory_context(
        user_id=user_id,
        scene="chat",
        max_facts=8,
        query=latest_query,
    ) or "暂无"

    system_prompt = f"""
你是“职小助”，一名专业求职助手。

当前用户：
- user_id: {user_id}
- 长期记忆: {long_term_profile}
- 短期摘要: {summary or "暂无"}

工具使用规则：
1. 用户提到简历推荐、适合什么岗位、基于简历做推荐时，优先调用 `get_user_resume_tool`。
2. 用户透露新的稳定信息时，调用 `save_preference_tool` 保存。
3. 用户问“我和这个岗位匹配吗”“帮我分析我和某岗位的匹配度”时，调用 `analyze_job_match_by_query_tool`。
4. 只有在确实需要复杂链式推理时，才调用 `search_jobs_tool`。

回复原则：
1. 只基于工具返回的真实内容作答，不要编造。
2. 不要暴露 JSON 或工具调用细节。
3. 如果用户是在做普通咨询、比较、追问，就直接回答。
4. 使用纯文本或轻量列表回复，不要使用 Markdown 标题，不要大段加粗。
"""
    response = agent_with_tools.invoke(
        [SystemMessage(content=system_prompt)] + _sanitize_messages_for_llm(state.get("messages", []))
    )
    return {"messages": [response]}


def extract_search_results(state: AgentState):
    if state.get("search_results") is not None and state.get("search_response_text"):
        return {}

    payload = {}
    search_results = None

    for message in reversed(state.get("messages", [])):
        if getattr(message, "name", "") != "search_jobs_tool":
            continue
        try:
            payload = _safe_json_loads(message.content)
        except Exception:
            payload = {}
        if isinstance(payload, dict):
            search_results = payload.get("results") or payload.get("ui_cards") or []
        break

    return {
        "search_results": search_results or [],
        "search_response_text": payload.get("llm_text", "") if isinstance(payload, dict) else "",
    }


def result_judge_node(state: AgentState):
    search_results = state.get("search_results") or []
    retry_count = int(state.get("search_retry_count", 0))
    latest_text = _latest_user_text(state.get("messages", []))
    plan = state.get("search_plan") or _heuristic_search_plan(latest_text)
    required_experience = (plan.get("experience", "") or "").strip()
    role_keywords = state.get("role_keywords") or _extract_role_keywords(
        plan.get("resolved_query", ""),
        plan.get("keyword_query", ""),
        latest_text,
    )
    filtered_results, dropped_results, filter_rules = _filter_results_by_relevance(
        search_results,
        role_keywords,
        required_experience,
    )
    summary = _build_result_summary(search_results, role_keywords, required_experience)
    logger.info(
        "search result summary: total=%s matched_role_count=%s mismatch_count=%s experience=%s experience_match_count=%s role_keywords=%s",
        summary["total_count"],
        summary["matched_role_count"],
        summary["mismatch_count"],
        required_experience,
        summary["experience_match_count"],
        ",".join(role_keywords) if role_keywords else "<none>",
    )
    logger.info(
        "relevance filter rules: role_keywords=%s required_experience=%s role_rule=%s experience_rule=%s",
        ",".join(filter_rules.get("role_keywords", [])) or "<none>",
        filter_rules.get("required_experience", "") or "<none>",
        filter_rules.get("role_rule", ""),
        filter_rules.get("experience_rule", ""),
    )
    if dropped_results:
        max_drop_log = 30
        dropped_lines = []
        for idx, item in enumerate(dropped_results[:max_drop_log], start=1):
            dropped_lines.append(
                "  [%s] %s @ %s | experience=%s | tags=%s | reasons=%s | experience_matched=%s | title_roles=%s | target_roles=%s | matched_roles=%s | mismatch_roles=%s"
                % (
                    idx,
                    item.get("title", ""),
                    item.get("company", ""),
                    item.get("experience", ""),
                    item.get("tags", ""),
                    ",".join(item.get("reasons", [])),
                    item.get("experience_matched", False),
                    ",".join(item.get("title_roles", [])),
                    ",".join(item.get("target_roles", [])),
                    ",".join(item.get("matched_roles", [])),
                    ",".join(item.get("mismatch_roles", [])),
                )
            )
        if len(dropped_results) > max_drop_log:
            dropped_lines.append(
                f"  ... truncated {len(dropped_results) - max_drop_log} more dropped candidates"
            )
        logger.info(
            "relevance filter dropped %s/%s candidates:\n%s",
            len(dropped_results),
            len(search_results),
            "\n".join(dropped_lines),
        )
    else:
        logger.info("relevance filter dropped 0/%s candidates", len(search_results))

    kept_lines = []
    for idx, item in enumerate(filtered_results[:10], start=1):
        kept_lines.append(
            "[%s] %s @ %s" % (idx, item.get("title", ""), item.get("company", ""))
        )
    if kept_lines:
        logger.info(
            "relevance filter kept %s/%s candidates: %s",
            len(filtered_results),
            len(search_results),
            " ; ".join(kept_lines),
        )

    if not search_results:
        if retry_count >= 1:
            return {
                "search_action": "fail",
                "should_retry_search": False,
                "relevance_feedback": "两轮检索后仍无结果。",
                "matched_role_count": 0,
                "mismatch_count": 0,
                "search_response_text": "没有找到与当前岗位方向明确匹配的职位。建议换城市、放宽技能要求，或改搜相近岗位。",
            }
        return {
            "search_action": "rewrite",
            "should_retry_search": True,
            "relevance_feedback": "当前没有检索到结果，尝试自动改写后重搜。",
            "matched_role_count": 0,
            "mismatch_count": 0,
        }

    if required_experience and summary["experience_match_count"] <= 0:
        message = f"当前结果里没有满足“{required_experience}”要求的职位，尝试自动改写后重搜。"
        if retry_count >= 1:
            return {
                "search_action": "fail",
                "should_retry_search": False,
                "relevance_feedback": f"两轮检索后仍未找到满足“{required_experience}”要求的职位。",
                "matched_role_count": summary["matched_role_count"],
                "mismatch_count": summary["mismatch_count"],
                "search_response_text": f"没有找到同时满足岗位方向和“{required_experience}”要求的职位。建议换城市、放宽技能要求，或去掉该经验限制后再试。",
                "search_results": [],
                "filtered_search_results": [],
            }
        return {
            "search_action": "rewrite",
            "should_retry_search": True,
            "relevance_feedback": message,
            "matched_role_count": summary["matched_role_count"],
            "mismatch_count": summary["mismatch_count"],
        }

    if not role_keywords:
        if len(search_results) >= 3 or retry_count >= 1:
            return {
                "search_action": "keep",
                "should_retry_search": False,
                "relevance_feedback": "未识别到明确岗位族，保留当前结果。",
                "matched_role_count": summary["matched_role_count"],
                "mismatch_count": summary["mismatch_count"],
            }
        return {
            "search_action": "rewrite",
            "should_retry_search": True,
            "relevance_feedback": "结果较少且岗位方向不够明确，尝试扩写关键词。",
            "matched_role_count": summary["matched_role_count"],
            "mismatch_count": summary["mismatch_count"],
        }

    if (
        summary["matched_role_count"] >= 2
        and summary["mismatch_count"] <= 1
        and summary["constraint_mismatch_count"] == 0
    ):
        return {
            "search_action": "keep",
            "should_retry_search": False,
            "relevance_feedback": "前几条结果与目标岗位方向基本一致，直接返回。",
            "matched_role_count": summary["matched_role_count"],
            "mismatch_count": summary["mismatch_count"],
        }

    if filtered_results and len(filtered_results) >= 1:
        return {
            "search_action": "filter",
            "should_retry_search": False,
            "filtered_search_results": filtered_results[:6],
            "relevance_feedback": "保留当前结果中与目标岗位方向和筛选条件更一致的职位。",
            "matched_role_count": summary["matched_role_count"],
            "mismatch_count": summary["mismatch_count"],
        }

    prompt = f"""
你负责判断当前职位检索结果是否与用户意图一致，只能输出严格 JSON。

用户原始输入：
{latest_text}

当前检索计划：
{json.dumps(plan, ensure_ascii=False)}

目标岗位关键词：
{", ".join(role_keywords) if role_keywords else "无"}

经验要求：
{required_experience or "无"}

当前前5条结果摘要：
{json.dumps(summary["top_results"], ensure_ascii=False)}

你只能从以下动作中选择一个：
- keep: 当前结果整体可用
- filter: 当前结果里有可用结果，但混入了不合适岗位，应过滤后返回
- rewrite: 当前结果方向不对或太少，应改写后重搜一次
- fail: 已经没有必要继续搜，直接明确告知用户

请严格输出：
{{
  "action": "keep",
  "reason": "一句中文解释"
}}

规则：
1. 如果大部分结果标题不属于目标岗位方向，优先选择 rewrite。
2. 如果用户给了经验要求而结果不满足，例如要求“实习”但结果都不是实习，优先选择 rewrite；如果已重试过一次则选择 fail。
3. 如果已有1条以上明显匹配岗位，且不匹配岗位只是混入，选择 filter。
4. 如果已经重试过一次且结果仍明显偏题，选择 fail。
5. 不要输出除 JSON 外的任何内容。
"""
    try:
        parsed = _safe_json_loads(rewrite_llm.invoke(prompt).content)
        action = parsed.get("action", "keep")
        if action not in {"keep", "filter", "rewrite", "fail"}:
            action = "keep"
        reason = parsed.get("reason", "已完成结果相关性判断。")
    except Exception as exc:
        logger.warning("relevance judge parse failed, fallback to rules: %s", exc)
        action = "filter" if filtered_results else ("rewrite" if retry_count < 1 else "fail")
        reason = "模型判断失败，已按规则选择后续动作。"

    logger.info("relevance judge action=%s reason=%s", action, reason)

    if action == "filter" and filtered_results:
        return {
            "search_action": "filter",
            "should_retry_search": False,
            "filtered_search_results": filtered_results[:6],
            "relevance_feedback": reason,
            "matched_role_count": summary["matched_role_count"],
            "mismatch_count": summary["mismatch_count"],
        }

    if action == "rewrite" and retry_count < 1:
        return {
            "search_action": "rewrite",
            "should_retry_search": True,
            "relevance_feedback": reason,
            "matched_role_count": summary["matched_role_count"],
            "mismatch_count": summary["mismatch_count"],
        }

    if action == "fail":
        return {
            "search_action": "fail",
            "should_retry_search": False,
            "relevance_feedback": reason,
            "matched_role_count": summary["matched_role_count"],
            "mismatch_count": summary["mismatch_count"],
            "search_response_text": "当前召回结果与目标岗位方向不够一致，暂时不建议继续展示。你可以补充城市、技术栈或薪资要求后再试。",
            "search_results": [],
            "filtered_search_results": [],
        }

    if action == "rewrite" and retry_count >= 1:
        return {
            "search_action": "fail",
            "should_retry_search": False,
            "relevance_feedback": reason,
            "matched_role_count": summary["matched_role_count"],
            "mismatch_count": summary["mismatch_count"],
            "search_response_text": "已经尝试自动优化一次检索条件，但结果仍与目标岗位方向不够一致。建议你补充更明确的岗位关键词后再搜。",
            "search_results": [],
            "filtered_search_results": [],
        }

    return {
        "search_action": "keep",
        "should_retry_search": False,
        "relevance_feedback": reason,
        "matched_role_count": summary["matched_role_count"],
        "mismatch_count": summary["mismatch_count"],
    }


def filter_results_node(state: AgentState):
    filtered_results = (state.get("filtered_search_results") or [])[:6]
    render_results = _hydrate_results_for_render(filtered_results)
    feedback = state.get("relevance_feedback", "").strip()
    logger.info("filter results kept %s/%s", len(filtered_results), len(state.get("search_results") or []))
    return {
        "search_results": filtered_results,
        "search_response_text": _render_search_results_text(
            render_results,
            intro=f"已根据岗位意图过滤不匹配结果：{feedback}" if feedback else "已过滤掉不匹配的职位。",
        ),
    }


def job_alive_check_node(state: AgentState):
    search_results = list((state.get("search_results") or [])[:JOB_ALIVE_CHECK_MAX_RESULTS])
    if not search_results:
        return {}

    job_ids = [
        str(item.get("job_id", "")).strip()
        for item in search_results
        if str(item.get("job_id", "")).strip()
    ]
    if not job_ids:
        return {}

    status_map = db_manager.get_job_status_checks(job_ids)
    stale_job_ids: list[str] = []
    stale_seen: set[str] = set()
    for job_id in job_ids:
        row = status_map.get(job_id)
        checked_at = _parse_checked_at((row or {}).get("checked_at"))
        if _is_check_fresh(checked_at, JOB_ALIVE_CHECK_FRESH_HOURS):
            continue
        if job_id in stale_seen:
            continue
        stale_seen.add(job_id)
        stale_job_ids.append(job_id)

    if stale_job_ids:
        logger.info(
            "job alive check refresh: total=%s stale=%s fresh=%s threshold_hours=%s",
            len(job_ids),
            len(stale_job_ids),
            len(job_ids) - len(stale_job_ids),
            JOB_ALIVE_CHECK_FRESH_HOURS,
        )
        detail_url_map = db_manager.get_job_detail_urls(stale_job_ids)
        stale_with_url: list[str] = []
        missing_url_count = 0
        for job_id in stale_job_ids:
            detail_url = str(detail_url_map.get(job_id, "")).strip()
            if detail_url:
                stale_with_url.append(job_id)
                continue
            missing_url_count += 1
            db_manager.upsert_job_status_check(
                job_id=job_id,
                status="unknown",
                reason="missing_detail_url",
                matched_keyword="",
                detail_url="",
                final_url="",
                text_preview="",
            )

        if missing_url_count:
            logger.info(
                "job alive check skipped %s stale jobs due to missing detail_url",
                missing_url_count,
            )

        if stale_with_url:
            try:
                from DrissionPage import ChromiumPage
                from utils.job_alive_probe import classify_job_page
            except Exception as exc:
                logger.warning("job alive checker unavailable, skip refresh: %s", exc)
                ChromiumPage = None
                classify_job_page = None

            page = None
            if ChromiumPage and classify_job_page:
                try:
                    page = ChromiumPage()
                    for job_id in stale_with_url:
                        detail_url = detail_url_map.get(job_id, "")
                        previous_row = status_map.get(job_id, {}) or {}
                        previous_status = str(previous_row.get("status", "")).strip().lower()
                        check_result: dict[str, Any]
                        tab = None
                        try:
                            check_result, tab = classify_job_page(
                                url=detail_url,
                                page=page,
                                timeout_seconds=JOB_ALIVE_CHECK_TIMEOUT_SECONDS,
                                poll_interval_seconds=JOB_ALIVE_CHECK_POLL_INTERVAL_SECONDS,
                            )
                        except Exception as exc:
                            logger.warning("job alive check failed: job_id=%s err=%s", job_id, exc)
                            check_result = {
                                "status": "unknown",
                                "reason": "check_exception",
                                "matched_keyword": "",
                                "final_url": detail_url,
                                "text_preview": "",
                            }
                        finally:
                            if tab is not None:
                                try:
                                    tab.close()
                                except Exception:
                                    pass

                        result_status = str(check_result.get("status", "unknown")).strip().lower() or "unknown"
                        result_reason = str(check_result.get("reason", "")).strip()
                        result_matched_keyword = str(check_result.get("matched_keyword", "")).strip()
                        result_final_url = str(check_result.get("final_url", "")).strip()
                        result_text_preview = str(check_result.get("text_preview", "")).strip()

                        if result_status == "unknown":
                            if previous_status in {"active", "closed"}:
                                logger.info(
                                    "job alive keep previous status on unknown probe result: job_id=%s previous=%s reason=%s",
                                    job_id,
                                    previous_status,
                                    result_reason or "<empty>",
                                )
                                continue

                            if ASSUME_ACTIVE_ON_UNKNOWN:
                                unknown_type = "transient" if _is_transient_unknown_reason(result_reason) else "non_transient"
                                logger.info(
                                    "job alive fallback to active on %s unknown: job_id=%s reason=%s",
                                    unknown_type,
                                    job_id,
                                    result_reason or "<empty>",
                                )
                                result_status = "active"
                                result_reason = f"assumed_active:{result_reason or 'unknown'}"[:100]
                            elif ASSUME_ACTIVE_ON_TRANSIENT_UNKNOWN and _is_transient_unknown_reason(result_reason):
                                result_status = "active"
                                result_reason = f"assumed_active:{result_reason or 'unknown'}"[:100]

                        db_manager.upsert_job_status_check(
                            job_id=job_id,
                            status=result_status,
                            reason=result_reason,
                            matched_keyword=result_matched_keyword,
                            detail_url=detail_url,
                            final_url=result_final_url,
                            text_preview=result_text_preview,
                        )
                except Exception as exc:
                    logger.warning("job alive refresh aborted: %s", exc, exc_info=True)
                finally:
                    if page is not None:
                        _close_page_instance(page)
        status_map = db_manager.get_job_status_checks(job_ids)

    kept_results: list[dict] = []
    dropped_closed: list[str] = []
    status_counter = {"active": 0, "closed": 0, "unknown": 0}

    for item in search_results:
        job_id = str(item.get("job_id", "")).strip()
        status_row = status_map.get(job_id, {})
        status = str(status_row.get("status", "unknown")).strip().lower() or "unknown"
        if status not in status_counter:
            status = "unknown"
        status_counter[status] += 1

        if status == "closed":
            dropped_closed.append(f"{item.get('title', '')} @ {item.get('company', '')}")
            continue

        enriched = dict(item)
        enriched["job_status"] = status
        checked_at = _parse_checked_at(status_row.get("checked_at"))
        if checked_at:
            enriched["job_checked_at"] = checked_at.isoformat(sep=" ", timespec="seconds")
        kept_results.append(enriched)

    logger.info(
        "job alive check summary: total=%s active=%s unknown=%s closed=%s kept=%s dropped=%s",
        len(search_results),
        status_counter["active"],
        status_counter["unknown"],
        status_counter["closed"],
        len(kept_results),
        len(dropped_closed),
    )
    if dropped_closed:
        logger.info("job alive check dropped closed jobs: %s", " ; ".join(dropped_closed[:10]))

    if not dropped_closed:
        return {}

    if not kept_results:
        return {
            "search_results": [],
            "search_response_text": "当前候选岗位链接校验后均显示已关闭，建议你换一批关键词或稍后再试。",
        }

    render_results = _hydrate_results_for_render(kept_results[:JOB_ALIVE_CHECK_MAX_RESULTS])
    return {
        "search_results": kept_results[:JOB_ALIVE_CHECK_MAX_RESULTS],
        "search_response_text": _render_search_results_text(
            render_results,
            intro=f"已过滤 {len(dropped_closed)} 个已关闭岗位，以下是仍可查看的职位：",
        ),
    }


def rewrite_search_plan_node(state: AgentState):
    latest_text = _latest_user_text(state.get("messages", []))
    plan = state.get("search_plan") or _heuristic_search_plan(latest_text)
    retry_count = int(state.get("search_retry_count", 0))
    search_results = state.get("search_results") or []
    prompt = f"""
你负责放宽一次职位检索条件。当前结果偏少，请输出更宽松但仍相关的检索计划。

用户原始输入：
{latest_text}

当前检索计划：
{json.dumps(plan, ensure_ascii=False)}

当前命中数：{len(search_results)}

请严格输出 JSON：
{{
  "resolved_query": "更宽松的自然语言检索意图",
  "keyword_query": "更宽松的短关键词，空格分隔",
  "city": "",
  "company": "",
  "experience": "",
  "salary_min": 0,
  "salary_unit": "",
  "reason": "放宽了哪些条件"
}}

规则：
1. 优先保留岗位方向、城市、实习/应届等硬条件。
2. 如果用户明确提了薪资下限，不要移除 salary_min。
3. 如果公司过滤导致结果太少，可以清空 company。
4. `city` 只能是单个标准城市名或空字符串，不要输出“北京及周边”。
5. `experience` 只能是 `实习`、`应届`、`1-3年` 这类短值，不要输出“应届/在校生”。
6. `salary_unit` 只能是 `k_month`、`yuan_day` 或空字符串。
7. 不要把 query 改得完全偏题。
"""
    try:
        parsed = _safe_json_loads(rewrite_llm.invoke(prompt).content)
        normalized_plan = _normalize_search_plan(parsed, fallback_text=latest_text)
        normalized_plan = _enforce_rewrite_constraints(normalized_plan, plan)
        normalized_plan, explicit_constraints = _apply_explicit_constraint_guard(normalized_plan, latest_text)
        role_keywords = _extract_role_keywords(
            normalized_plan.get("resolved_query", ""),
            normalized_plan.get("keyword_query", ""),
        )
        merged_plan = {
            "resolved_query": normalized_plan.get("resolved_query", plan.get("resolved_query", latest_text)),
            "keyword_query": normalized_plan.get("keyword_query", plan.get("keyword_query", latest_text)),
            "city": normalized_plan.get("city", plan.get("city", "")),
            "company": normalized_plan.get("company", plan.get("company", "")),
            "experience": normalized_plan.get("experience", ""),
            "salary_min": int(normalized_plan.get("salary_min", 0) or 0),
            "salary_unit": normalized_plan.get("salary_unit", ""),
        }
        logger.info(
            "rewrite explicit constraint guard: experience_mentioned=%s experience=%s salary_mentioned=%s salary_min=%s salary_unit=%s",
            explicit_constraints.get("experience_mentioned", False),
            merged_plan.get("experience", ""),
            explicit_constraints.get("salary_mentioned", False),
            merged_plan.get("salary_min", 0),
            merged_plan.get("salary_unit", ""),
        )
        logger.info("rewrite normalized search plan: %s", json.dumps(merged_plan, ensure_ascii=False))
        return {
            "search_plan": merged_plan,
            "role_keywords": role_keywords,
            "search_retry_count": retry_count + 1,
            "search_feedback": parsed.get("reason", "已自动放宽一次检索条件。"),
            "should_retry_search": True,
            "search_results": [],
            "filtered_search_results": [],
            "search_action": "",
        }
    except Exception as exc:
        logger.warning("rewrite parse failed, fallback to simple relaxation: %s", exc)
        relaxed_plan = dict(plan)
        relaxed_plan["company"] = ""
        relaxed_plan["keyword_query"] = re.sub(r"\s+", " ", relaxed_plan.get("keyword_query", latest_text)).strip()
        relaxed_plan, explicit_constraints = _apply_explicit_constraint_guard(relaxed_plan, latest_text)
        logger.info(
            "rewrite explicit constraint guard(fallback): experience_mentioned=%s experience=%s salary_mentioned=%s salary_min=%s salary_unit=%s",
            explicit_constraints.get("experience_mentioned", False),
            relaxed_plan.get("experience", ""),
            explicit_constraints.get("salary_mentioned", False),
            relaxed_plan.get("salary_min", 0),
            relaxed_plan.get("salary_unit", ""),
        )
        return {
            "search_plan": relaxed_plan,
            "role_keywords": _extract_role_keywords(
                relaxed_plan.get("resolved_query", ""),
                relaxed_plan.get("keyword_query", ""),
            ),
            "search_retry_count": retry_count + 1,
            "search_feedback": "已自动移除过窄限制并重试一次。",
            "should_retry_search": True,
            "search_results": [],
            "filtered_search_results": [],
            "search_action": "",
        }


def route_after_result_judge(state: AgentState):
    action = state.get("search_action", "")
    if action == "filter":
        return "filter_results_node"
    if action == "rewrite" or state.get("should_retry_search"):
        return "rewrite_search_plan_node"
    return "job_alive_check_node"


def search_response_node(state: AgentState):
    content = state.get("search_response_text", "").strip() or "没有找到合适的岗位结果。"
    feedback = state.get("search_feedback", "").strip()
    if feedback and state.get("search_retry_count", 0) > 0:
        content = f"已自动优化检索条件：{feedback}\n\n{content}"
    return {"messages": [AIMessage(content=content)]}


def after_tools_route(state: AgentState):
    if not state.get("messages"):
        return "bot_node"
    last_message = state["messages"][-1]
    if getattr(last_message, "name", "") == "search_jobs_tool":
        return "extract_search_results"
    return "bot_node"


workflow = StateGraph(AgentState)
workflow.add_node("summarize_node", summarize_node)
workflow.add_node("intent_parse_node", intent_parse_node)
workflow.add_node("search_node", search_node)
workflow.add_node("bot_node", bot_node)
workflow.add_node("tools", ToolNode(tools))
workflow.add_node("extract_search_results", extract_search_results)
workflow.add_node("result_judge_node", result_judge_node)
workflow.add_node("filter_results_node", filter_results_node)
workflow.add_node("job_alive_check_node", job_alive_check_node)
workflow.add_node("rewrite_search_plan_node", rewrite_search_plan_node)
workflow.add_node("search_response_node", search_response_node)

workflow.add_conditional_edges(
    START,
    route_from_start,
    {
        "summarize_node": "summarize_node",
        "intent_parse_node": "intent_parse_node",
    },
)
workflow.add_edge("summarize_node", "intent_parse_node")
workflow.add_conditional_edges(
    "intent_parse_node",
    route_after_intent,
    {
        "search_node": "search_node",
        "bot_node": "bot_node",
    },
)
workflow.add_edge("search_node", "result_judge_node")
workflow.add_edge("extract_search_results", "result_judge_node")
workflow.add_conditional_edges(
    "result_judge_node",
    route_after_result_judge,
    {
        "filter_results_node": "filter_results_node",
        "rewrite_search_plan_node": "rewrite_search_plan_node",
        "job_alive_check_node": "job_alive_check_node",
    },
)
workflow.add_edge("filter_results_node", "job_alive_check_node")
workflow.add_edge("job_alive_check_node", "search_response_node")
workflow.add_edge("rewrite_search_plan_node", "search_node")
workflow.add_edge("search_response_node", END)
workflow.add_conditional_edges("bot_node", tools_condition)
workflow.add_conditional_edges(
    "tools",
    after_tools_route,
    {
        "extract_search_results": "extract_search_results",
        "bot_node": "bot_node",
    },
)

app_graph = workflow.compile(checkpointer=MemorySaver())
