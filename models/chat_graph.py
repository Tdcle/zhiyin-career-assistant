import json
import re
from typing import Annotated, Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

from config.config import config
from utils.database import DatabaseManager
from utils.logger import get_logger
from utils.tools import (
    _extract_role_keywords,
    _normalize_city,
    _normalize_experience,
    _normalize_keyword_text,
    analyze_job_match_by_query_tool,
    get_user_resume_tool,
    save_preference_tool,
    search_jobs_tool,
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


def _safe_json_loads(raw: str) -> dict:
    text = (raw or "").strip()
    if "```" in text:
        chunks = text.split("```")
        if len(chunks) >= 3:
            text = chunks[1].replace("json", "", 1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        text = text[start:end + 1]
    return json.loads(text)


def _latest_user_text(messages: list) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage) and message.content:
            return message.content.strip()
    return ""


def _recent_dialogue(messages: list, limit: int = 6) -> str:
    rows = []
    for message in messages[-limit:]:
        if not isinstance(message, (HumanMessage, AIMessage)) or not message.content:
            continue
        speaker = "user" if isinstance(message, HumanMessage) else "assistant"
        rows.append(f"{speaker}: {message.content}")
    return "\n".join(rows)


def _looks_like_search_followup(latest_text: str, previous_plan: dict) -> bool:
    text = (latest_text or "").strip()
    if not text:
        return False

    if _extract_role_keywords(text):
        return True

    if re.search(r"(北京|上海|深圳|广州|杭州|成都|武汉|西安|南京|苏州|天津|重庆|远程|全国)", text):
        return True

    if re.search(r"(\d{1,2}\s*[kK]|\d{2,4}\s*元\s*/?\s*天)", text):
        return True

    if any(token in text for token in ["实习", "应届", "校招", "全职", "兼职", "初级", "中级", "高级"]):
        return True

    if previous_plan and len(text) <= 20:
        return True

    return False


def _parse_search_payload(content: str) -> tuple[list[dict], str]:
    try:
        payload = _safe_json_loads(content)
    except Exception:
        payload = {}

    if not isinstance(payload, dict):
        return [], ""

    return payload.get("ui_cards") or [], payload.get("llm_text", "") or ""


def _sanitize_messages_for_llm(messages: list) -> list:
    sanitized = []
    valid_tool_call_ids = set()

    for message in messages:
        if isinstance(message, AIMessage):
            sanitized.append(message)
            tool_calls = getattr(message, "tool_calls", None) or []
            for tool_call in tool_calls:
                tool_call_id = tool_call.get("id")
                if tool_call_id:
                    valid_tool_call_ids.add(tool_call_id)
            continue

        if isinstance(message, ToolMessage):
            if getattr(message, "tool_call_id", "") in valid_tool_call_ids:
                sanitized.append(message)
            else:
                logger.warning(
                    "dropping orphan tool message: name=%s tool_call_id=%s",
                    getattr(message, "name", ""),
                    getattr(message, "tool_call_id", ""),
                )
            continue

        sanitized.append(message)

    return sanitized


def _analyze_title_roles(title: str, role_keywords: list[str]) -> dict:
    title_roles = _extract_role_keywords(title)
    matched_roles = [role for role in title_roles if role in role_keywords]
    mismatch_roles = [role for role in title_roles if role not in role_keywords]
    return {
        "title_roles": title_roles,
        "matched_roles": matched_roles,
        "mismatch_roles": mismatch_roles,
        "match_count": len(matched_roles),
        "mismatch_count": len(mismatch_roles),
    }


def _matches_experience_requirement(item: dict, required_experience: str) -> bool:
    required = (required_experience or "").strip()
    if not required:
        return True

    text = " ".join(
        str(item.get(field, "") or "")
        for field in ("title", "tags", "experience")
    ).lower()
    if not text:
        return False

    if required == "实习":
        return any(token in text for token in ["实习", "实习生", "暑期实习", "日常实习"])
    if required == "应届":
        return any(token in text for token in ["应届", "校招", "毕业生", "管培生"])

    return required.lower() in text


def _enforce_rewrite_constraints(rewritten_plan: dict, original_plan: dict) -> dict:
    constrained = dict(rewritten_plan)
    original_experience = (original_plan.get("experience", "") or "").strip()
    if not original_experience:
        return constrained

    constrained["experience"] = original_experience
    keyword_query = constrained.get("keyword_query", "") or ""
    tokens = [token for token in keyword_query.split() if token]

    if original_experience == "实习":
        disallowed = {"初级", "中级", "高级", "全职", "社招", "校招", "应届"}
        tokens = [token for token in tokens if token not in disallowed]
        if "实习" not in tokens:
            tokens.append("实习")
    elif original_experience == "应届":
        disallowed = {"实习", "全职", "社招"}
        tokens = [token for token in tokens if token not in disallowed]
        if "应届" not in tokens and "校招" not in tokens:
            tokens.append("应届")

    constrained["keyword_query"] = " ".join(dict.fromkeys(tokens)).strip() or keyword_query
    return constrained


def _filter_results_by_relevance(
    search_results: list[dict],
    role_keywords: list[str],
    required_experience: str = "",
) -> list[dict]:
    if not role_keywords and not required_experience:
        return search_results

    filtered = []
    for item in search_results:
        if role_keywords:
            analysis = _analyze_title_roles(item.get("title", ""), role_keywords)
            if analysis["match_count"] <= 0:
                continue
            if analysis["mismatch_count"] > 0 and analysis["match_count"] <= analysis["mismatch_count"]:
                continue
        if required_experience and not _matches_experience_requirement(item, required_experience):
            continue
        filtered.append(item)
    return filtered


def _build_result_summary(
    search_results: list[dict],
    role_keywords: list[str],
    required_experience: str = "",
    limit: int = 5,
) -> dict:
    top_results = search_results[:limit]
    rows = []
    matched_role_count = 0
    mismatch_count = 0
    experience_match_count = 0

    for idx, item in enumerate(top_results, start=1):
        analysis = _analyze_title_roles(item.get("title", ""), role_keywords)
        experience_matched = _matches_experience_requirement(item, required_experience)
        if analysis["match_count"] > 0:
            matched_role_count += 1
        if analysis["mismatch_count"] > 0:
            mismatch_count += 1
        if experience_matched:
            experience_match_count += 1
        rows.append(
            {
                "rank": idx,
                "title": item.get("title", ""),
                "company": item.get("company", ""),
                "matched_roles": analysis["matched_roles"],
                "mismatch_roles": analysis["mismatch_roles"],
                "experience_matched": experience_matched,
                "tags": item.get("tags", ""),
            }
        )

    return {
        "top_results": rows,
        "matched_role_count": matched_role_count,
        "mismatch_count": mismatch_count,
        "required_experience": required_experience,
        "experience_match_count": experience_match_count,
        "constraint_mismatch_count": max(0, len(top_results) - experience_match_count) if required_experience else 0,
        "total_count": len(search_results),
    }


def _render_search_results_text(search_results: list[dict], intro: str = "") -> str:
    if not search_results:
        return intro.strip() or "没有找到合适的岗位结果。"

    lines = [intro.strip()] if intro.strip() else []
    lines.append(f"为你整理出 {len(search_results)} 个当前更贴近意图的职位：")
    for item in search_results:
        lines.extend(
            [
                f"🏢 {item.get('company', '未知公司')}",
                f"📌 职位：{item.get('title', '未知职位')}",
                f"💰 薪资：{item.get('salary', '未知')}",
                f"📍 标签：{item.get('tags', '')}",
                "",
            ]
        )
    return "\n".join(lines).strip()


def _heuristic_search_plan(latest_text: str) -> dict:
    plan = {
        "resolved_query": latest_text,
        "keyword_query": latest_text,
        "city": "",
        "company": "",
        "experience": "",
        "salary_min": 0,
        "salary_unit": "",
    }

    city_match = re.search(r"(北京|上海|深圳|广州|杭州|成都|武汉|西安|南京|苏州|天津|重庆)", latest_text)
    if city_match:
        plan["city"] = city_match.group(1)

    if any(token in latest_text for token in ["实习", "实习生"]):
        plan["experience"] = "实习"
    elif any(token in latest_text for token in ["应届", "校招", "在校"]):
        plan["experience"] = "应届"

    monthly_match = re.search(r"(\d{1,2})(?:\s*[kK]|千)", latest_text)
    if monthly_match:
        plan["salary_min"] = int(monthly_match.group(1))
        plan["salary_unit"] = "k_month"

    daily_match = re.search(r"(\d{2,4})\s*元\s*/?\s*天", latest_text)
    if daily_match:
        plan["salary_min"] = int(daily_match.group(1))
        plan["salary_unit"] = "yuan_day"

    keywords = []
    for token in ["前端", "后端", "Java", "Python", "Golang", "测试", "产品", "运营", "React", "Vue", "实习", "应届"]:
        if token.lower() in latest_text.lower():
            keywords.append(token)
    if keywords:
        plan["keyword_query"] = " ".join(keywords)

    return plan


def _normalize_salary_unit(value: str) -> str:
    raw = (value or "").strip().lower()
    if raw in {"k_month", "k", "月薪k", "月薪"}:
        return "k_month"
    if raw in {"yuan_day", "元/天", "日薪", "日薪元"}:
        return "yuan_day"
    return ""


def _normalize_search_plan(plan: dict, fallback_text: str = "") -> dict:
    resolved_query = _normalize_keyword_text(plan.get("resolved_query", "") or fallback_text)
    keyword_query = _normalize_keyword_text(plan.get("keyword_query", "") or resolved_query or fallback_text)

    return {
        "resolved_query": resolved_query or fallback_text,
        "keyword_query": keyword_query or fallback_text,
        "city": _normalize_city(
            plan.get("city", ""),
            resolved_query,
            keyword_query,
            fallback_text,
        ),
        "company": (plan.get("company", "") or "").strip(),
        "experience": _normalize_experience(
            plan.get("experience", ""),
            resolved_query,
            keyword_query,
            fallback_text,
        ),
        "salary_min": int(plan.get("salary_min", 0) or 0),
        "salary_unit": _normalize_salary_unit(plan.get("salary_unit", "")),
    }


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
    previous_plan = state.get("search_plan") or {}

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

    heuristic_search = any(token in latest_text for token in ["找", "岗位", "职位", "工作", "实习"])
    if not heuristic_search:
        heuristic_search = _looks_like_search_followup(latest_text, previous_plan)
    if not heuristic_search:
        return {"direct_search_mode": False}

    summary = state.get("summary", "")
    recent_dialogue = _recent_dialogue(state.get("messages", []))

    prompt = f"""
你负责把用户求职搜索请求解析成结构化检索计划。

历史摘要：
{summary or "暂无"}

最近对话：
{recent_dialogue or "暂无"}

最新用户输入：
{latest_text}

请严格输出 JSON：
{{
  "direct_search_mode": true,
  "search_plan": {{
    "resolved_query": "完整自然语言检索意图",
    "keyword_query": "核心检索关键词，多个短词用空格分隔，不要写完整句子",
    "city": "",
    "company": "",
    "experience": "",
    "salary_min": 0,
    "salary_unit": ""
  }}
}}

规则：
1. 只有在用户明确是在找岗位、筛岗位、搜实习时，direct_search_mode 才为 true。
2. 必须结合历史语境，处理“那换上海”“薪资再高一点”这种省略表达。
3. `city` 只能填单个标准城市名，如 `北京`、`上海`；不要写“北京及周边”“上海附近”“全国范围内”。如果用户说全国、远程优先或不限城市，就填空字符串。
4. `experience` 只能填简短标准值，如 `实习`、`应届`、`1-3年`、`3-5年`；不要写“应届/在校生”“有经验者优先”这种自然语言短语。
5. `salary_unit` 只能是 `k_month`、`yuan_day` 或空字符串，不要输出 `k`、`元/天` 这类别名。
6. `keyword_query` 只保留岗位名、技能词、经验词等短关键词，例如 `Java 实习 Spring Boot`；不要写完整描述句。
7. 不要臆造 company；没有就留空。
"""
    try:
        parsed = _safe_json_loads(planner_llm.invoke(prompt).content)
        llm_plan = parsed.get("search_plan") or {}
        normalized_plan = _normalize_search_plan(llm_plan, fallback_text=latest_text)
        role_keywords = _extract_role_keywords(
            normalized_plan.get("resolved_query", ""),
            normalized_plan.get("keyword_query", ""),
        )
        merged_plan = {
            "resolved_query": normalized_plan.get("resolved_query") or previous_plan.get("resolved_query", latest_text),
            "keyword_query": normalized_plan.get("keyword_query") or previous_plan.get("keyword_query", latest_text),
            "city": normalized_plan.get("city") or previous_plan.get("city", ""),
            "company": normalized_plan.get("company") or previous_plan.get("company", ""),
            "experience": normalized_plan.get("experience") or previous_plan.get("experience", ""),
            "salary_min": int(normalized_plan.get("salary_min", previous_plan.get("salary_min", 0)) or 0),
            "salary_unit": normalized_plan.get("salary_unit") or previous_plan.get("salary_unit", ""),
        }
        logger.info("planner normalized search plan: %s", json.dumps(merged_plan, ensure_ascii=False))
        return {
            "direct_search_mode": bool(parsed.get("direct_search_mode", False)),
            "search_plan": merged_plan,
            "search_results": [],
            "filtered_search_results": [],
            "role_keywords": role_keywords,
            "search_response_text": "",
            "search_retry_count": 0,
            "search_feedback": "",
            "should_retry_search": False,
            "search_action": "",
            "relevance_feedback": "",
        }
    except Exception as exc:
        logger.warning("planner parse failed, fallback to heuristic search plan: %s", exc)
        heuristic_plan = _heuristic_search_plan(latest_text)
        normalized_heuristic = _normalize_search_plan(heuristic_plan, fallback_text=latest_text)
        role_keywords = _extract_role_keywords(
            normalized_heuristic.get("resolved_query", ""),
            normalized_heuristic.get("keyword_query", ""),
        )
        merged_plan = {
            "resolved_query": normalized_heuristic.get("resolved_query", latest_text),
            "keyword_query": normalized_heuristic.get("keyword_query") or previous_plan.get("keyword_query", latest_text),
            "city": normalized_heuristic.get("city") or previous_plan.get("city", ""),
            "company": normalized_heuristic.get("company") or previous_plan.get("company", ""),
            "experience": normalized_heuristic.get("experience") or previous_plan.get("experience", ""),
            "salary_min": int(normalized_heuristic.get("salary_min", previous_plan.get("salary_min", 0)) or 0),
            "salary_unit": normalized_heuristic.get("salary_unit") or previous_plan.get("salary_unit", ""),
        }
        logger.info("heuristic normalized search plan: %s", json.dumps(merged_plan, ensure_ascii=False))
        return {
            "direct_search_mode": True,
            "search_plan": merged_plan,
            "search_results": [],
            "filtered_search_results": [],
            "role_keywords": role_keywords,
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
    long_term_profile = db_manager.get_user_profile(user_id) or "暂无"

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
            search_results = payload.get("ui_cards") or []
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
    filtered_results = _filter_results_by_relevance(search_results, role_keywords, required_experience)
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
    feedback = state.get("relevance_feedback", "").strip()
    logger.info("filter results kept %s/%s", len(filtered_results), len(state.get("search_results") or []))
    return {
        "search_results": filtered_results,
        "search_response_text": _render_search_results_text(
            filtered_results,
            intro=f"已根据岗位意图过滤不匹配结果：{feedback}" if feedback else "已过滤掉不匹配的职位。",
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
        role_keywords = _extract_role_keywords(
            normalized_plan.get("resolved_query", ""),
            normalized_plan.get("keyword_query", ""),
        )
        merged_plan = {
            "resolved_query": normalized_plan.get("resolved_query", plan.get("resolved_query", latest_text)),
            "keyword_query": normalized_plan.get("keyword_query", plan.get("keyword_query", latest_text)),
            "city": normalized_plan.get("city", plan.get("city", "")),
            "company": normalized_plan.get("company", plan.get("company", "")),
            "experience": normalized_plan.get("experience", plan.get("experience", "")),
            "salary_min": int(normalized_plan.get("salary_min", plan.get("salary_min", 0)) or 0),
            "salary_unit": normalized_plan.get("salary_unit", plan.get("salary_unit", "")),
        }
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
    return "search_response_node"


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
        "search_response_node": "search_response_node",
    },
)
workflow.add_edge("filter_results_node", "search_response_node")
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
