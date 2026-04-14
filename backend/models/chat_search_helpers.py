from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from typing import Any, Optional

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from db import DatabaseManager
from utils.logger import get_logger
from utils.tools import (
    _extract_role_keywords,
    _normalize_city,
    _normalize_experience,
    _normalize_keyword_text,
)

logger = get_logger("chat_graph")
db_manager = DatabaseManager()

TRANSIENT_UNKNOWN_REASONS = {
    "risk_or_login_page",
    "security_check_redirect",
    "loading_page",
    "no_signal",
    "redirected_out_of_job_detail",
    "check_exception",
}

ROLE_FAMILY_EQUIVALENTS = {
    "后端": {"后端", "Java", "Python", "Golang", "PHP", "C#", "C++"},
    "前端": {"前端", "全栈"},
    "全栈": {"全栈", "前端", "后端", "Java", "Python", "Golang", "PHP"},
    "AI": {"AI", "算法", "Python", "数据分析", "数据开发"},
    "算法": {"算法", "AI", "Python"},
}

LANGUAGE_TO_FAMILY = {
    "Java": "后端",
    "Python": "后端",
    "Golang": "后端",
    "PHP": "后端",
    "C#": "后端",
    "C++": "后端",
}


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
        followup_tokens = [
            "换", "改", "再", "提高", "降低", "放宽", "收紧",
            "不限", "优先", "只看", "继续搜", "继续找", "再搜", "再找",
            "来点", "还有吗", "还有没有", "多给", "再来",
            "那就", "然后", "另外", "并且",
        ]
        if any(token in text for token in followup_tokens):
            return True

    return False

def _is_non_search_request(latest_text: str) -> bool:
    text = (latest_text or "").strip()
    if not text:
        return False

    non_search_tokens = [
        "总结",
        "概括",
        "一句话",
        "复述",
        "回顾",
        "整理一下",
        "你记住了什么",
        "你记住的偏好",
        "筛选条件是什么",
        "我的偏好是什么",
        "个人画像",
    ]
    if any(token in text for token in non_search_tokens):
        return True

    if re.search(r"(按我刚才|根据我刚才|按之前|根据之前).*(偏好|条件).*(总结|概括|一句话|复述)", text):
        return True

    return False

def _parse_search_payload(content: str) -> tuple[list[dict], str]:
    try:
        payload = _safe_json_loads(content)
    except Exception:
        payload = {}

    if not isinstance(payload, dict):
        return [], ""

    results = payload.get("results")
    if isinstance(results, list) and results:
        return results, payload.get("llm_text", "") or ""

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

def _expand_target_roles(role_keywords: list[str]) -> set[str]:
    expanded = set(role_keywords or [])
    for role in list(expanded):
        expanded.update(ROLE_FAMILY_EQUIVALENTS.get(role, {role}))

    for role in list(expanded):
        family = LANGUAGE_TO_FAMILY.get(role)
        if not family:
            continue
        expanded.add(family)
        expanded.update(ROLE_FAMILY_EQUIVALENTS.get(family, {family}))
    return expanded

def _analyze_title_roles(title: str, role_keywords: list[str]) -> dict:
    title_roles = _extract_role_keywords(title)
    expanded_role_keywords = _expand_target_roles(role_keywords)
    # Treat algorithm/LLM related titles as AI family to avoid false negatives.
    if "AI" in expanded_role_keywords:
        expanded_role_keywords.update({"算法", "机器学习", "深度学习", "NLP", "大模型", "多模态"})
    if any(role in expanded_role_keywords for role in {"算法", "机器学习", "深度学习", "NLP", "大模型", "多模态"}):
        expanded_role_keywords.add("AI")

    matched_roles = [role for role in title_roles if role in expanded_role_keywords]
    mismatch_roles = [role for role in title_roles if role not in expanded_role_keywords]
    return {
        "title_roles": title_roles,
        "target_roles": sorted(expanded_role_keywords),
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
        if any(token in text for token in ["实习", "实习生", "暑期实习", "日常实习"]):
            return True
        # Many internship postings use attendance style text like "4天/周" instead of "实习".
        if re.search(r"([1-7]|[一二三四五六七])\s*天\s*/\s*周", text):
            return True
        if re.search(r"(每周|一周)\s*([1-7]|[一二三四五六七])\s*天", text):
            return True
        return False
    if required == "应届":
        return any(token in text for token in ["应届", "校招", "毕业生", "管培生"])
    if required in {"非实习", "全职", "正式工作"}:
        if any(token in text for token in ["实习", "实习生", "暑期实习", "日常实习", "intern"]):
            return False
        if re.search(r"([1-7]|[一二三四五六七])\s*天\s*/\s*周", text):
            return False
        if re.search(r"(每周|一周)\s*([1-7]|[一二三四五六七])\s*天", text):
            return False
        return True

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
    elif original_experience in {"非实习", "全职", "正式工作"}:
        disallowed = {"实习", "在校", "intern"}
        tokens = [token for token in tokens if token.lower() not in disallowed]
        if all(token not in tokens for token in ["全职", "社招", "正式"]):
            tokens.append("全职")

    constrained["keyword_query"] = " ".join(dict.fromkeys(tokens)).strip() or keyword_query
    return constrained

def _filter_results_by_relevance(
    search_results: list[dict],
    role_keywords: list[str],
    required_experience: str = "",
) -> tuple[list[dict], list[dict], dict]:
    rules = {
        "role_keywords": list(role_keywords or []),
        "required_experience": (required_experience or "").strip(),
        "role_rule": (
            "title must match at least one target role keyword; "
            "if mismatch roles > matched roles then drop"
            if role_keywords
            else "skipped (no role keywords)"
        ),
        "experience_rule": (
            f"title/tags/experience text must match required experience '{required_experience}'"
            if required_experience
            else "skipped (no experience constraint)"
        ),
    }

    if not role_keywords and not required_experience:
        return list(search_results), [], rules

    kept: list[dict] = []
    dropped: list[dict] = []

    for item in search_results:
        analysis = _analyze_title_roles(item.get("title", ""), role_keywords) if role_keywords else {
            "title_roles": [],
            "matched_roles": [],
            "mismatch_roles": [],
            "match_count": 0,
            "mismatch_count": 0,
        }
        experience_matched = _matches_experience_requirement(item, required_experience)
        role_pass = True
        reasons: list[str] = []

        if role_keywords:
            if analysis["match_count"] <= 0:
                role_pass = False
                reasons.append("role_mismatch:no_target_role_keyword")
            elif analysis["mismatch_count"] > 0 and analysis["match_count"] < analysis["mismatch_count"]:
                role_pass = False
                reasons.append(
                    f"role_mismatch:mismatch_dominates(matched={analysis['match_count']},mismatch={analysis['mismatch_count']})"
                )

        exp_pass = True
        if required_experience and not experience_matched:
            exp_pass = False
            reasons.append(f"experience_mismatch:required={required_experience}")

        if role_pass and exp_pass:
            kept.append(item)
        else:
            dropped.append(
                {
                    "job_id": str(item.get("job_id", "")),
                    "title": str(item.get("title", "")),
                    "company": str(item.get("company", "")),
                    "experience": str(item.get("experience", "")),
                    "tags": str(item.get("tags", "")),
                    "title_roles": analysis.get("title_roles", []),
                    "target_roles": analysis.get("target_roles", []),
                    "matched_roles": analysis.get("matched_roles", []),
                    "mismatch_roles": analysis.get("mismatch_roles", []),
                    "experience_matched": experience_matched,
                    "reasons": reasons or ["unknown_rule"],
                }
            )

    return kept, dropped, rules

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
        company = item.get("company", "未知公司")
        title = item.get("title", "未知职位")
        salary = item.get("salary", "未知")
        degree = str(item.get("degree", "") or "").strip()
        experience = str(item.get("experience", "") or "").strip()
        requirement = " / ".join([part for part in [degree, experience] if part]) or str(item.get("tags", "") or "").strip()
        summary = str(item.get("summary", "") or "").strip()
        if not summary:
            detail = str(item.get("detail", "") or "").strip()
            summary = detail[:160] if detail else ""
        detail_url = str(item.get("detail_url", "") or "").strip()

        lines.extend(
            [
                f"🏢 {company}",
                f"📌 职位：{title}",
                f"💰 薪资：{salary}",
                f"🧭 要求：{requirement or '未标注'}",
                f"📝 概要：{summary or '暂无'}",
                f"🔗 链接：{detail_url or '暂无'}",
                "",
            ]
        )
    return "\n".join(lines).strip()

def _hydrate_results_for_render(search_results: list[dict]) -> list[dict]:
    if not search_results:
        return []

    # If items already include rich fields from search tool payload, use directly.
    rich_fields = ("detail_url", "summary", "degree", "experience")
    if all(any(str(item.get(field, "")).strip() for field in rich_fields) for item in search_results):
        hydrated = []
        for item in search_results:
            merged = dict(item)
            if not str(merged.get("tags", "")).strip():
                city = str(merged.get("city", "")).strip()
                experience = str(merged.get("experience", "")).strip()
                merged["tags"] = " | ".join([part for part in [city, experience] if part])
            hydrated.append(merged)
        return hydrated

    job_ids: list[str] = []
    seen: set[str] = set()
    for item in search_results:
        job_id = str(item.get("job_id", "")).strip()
        if not job_id or job_id in seen:
            continue
        seen.add(job_id)
        job_ids.append(job_id)

    details_map = db_manager.get_jobs_by_ids(job_ids) if job_ids else {}
    hydrated: list[dict] = []
    for item in search_results:
        job_id = str(item.get("job_id", "")).strip()
        merged = dict(details_map.get(job_id, {}))
        merged.update(item)
        if not str(merged.get("tags", "")).strip():
            city = str(merged.get("city", "")).strip()
            experience = str(merged.get("experience", "")).strip()
            merged["tags"] = " | ".join([part for part in [city, experience] if part])
        hydrated.append(merged)
    return hydrated

def _parse_checked_at(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            normalized = text.replace("Z", "+00:00")
            return datetime.fromisoformat(normalized)
        except ValueError:
            return None
    return None

def _is_check_fresh(checked_at: Optional[datetime], fresh_hours: int) -> bool:
    if checked_at is None:
        return False
    if checked_at.tzinfo is not None:
        now = datetime.now(checked_at.tzinfo)
    else:
        now = datetime.now()
    return now - checked_at <= timedelta(hours=fresh_hours)

def _close_page_instance(page: Any) -> None:
    for method_name in ("quit", "close"):
        fn = getattr(page, method_name, None)
        if callable(fn):
            try:
                fn()
            except Exception:
                pass
            return

def _is_transient_unknown_reason(reason: str) -> bool:
    normalized = str(reason or "").strip().lower()
    if not normalized:
        return True
    if normalized.startswith("timeout_"):
        return True
    if normalized in TRANSIENT_UNKNOWN_REASONS:
        return True
    if "security_check" in normalized:
        return True
    if "risk" in normalized:
        return True
    return False

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

def _extract_explicit_constraints(latest_text: str) -> dict:
    text = (latest_text or "").strip()
    lowered = text.lower()

    experience_mentioned = False
    experience = ""
    salary_mentioned = False
    salary_min = 0
    salary_unit = ""

    if re.search(
        r"(不看|不找|不要|不考虑|别|非|不是).{0,3}(实习|在校|intern)|"
        r"(正式工作|全职|社招|正式岗)",
        lowered,
    ):
        experience_mentioned = True
        experience = "非实习"
    elif re.search(r"(经验不限|不限经验|无经验|不限制经验|不限是否实习|不找实习|不要实习|不看实习|不找应届|不要应届)", text):
        experience_mentioned = True
        experience = ""
    elif re.search(r"(实习|在校|intern)", lowered) and not re.search(r"(不|非|别|不用|不要|不限).{0,3}(实习|在校|intern)", text):
        experience_mentioned = True
        experience = "实习"
    elif re.search(r"(应届|校招|毕业生|new\s*grad)", lowered) and not re.search(r"(不|非|别|不用|不要|不限).{0,3}(应届|校招|毕业生)", text):
        experience_mentioned = True
        experience = "应届"
    elif "经验" in text:
        year_range = re.search(r"(\d+\s*-\s*\d+\s*年)", text)
        year_plus = re.search(r"(\d+\s*\+\s*年|\d+\s*年以上|\d+\s*年)", text)
        experience_mentioned = True
        if year_range:
            experience = re.sub(r"\s+", "", year_range.group(1))
        elif year_plus:
            experience = re.sub(r"\s+", "", year_plus.group(1))
        else:
            experience = ""

    if re.search(r"(薪资|工资|待遇).{0,4}(不限|面议)|薪资面议|工资面议", text):
        salary_mentioned = True
        salary_min = 0
        salary_unit = ""
    else:
        monthly_range = re.search(r"(\d{1,2})\s*[-~到至]\s*\d{1,2}\s*[kK千]", text)
        monthly_single = re.search(r"(\d{1,2})\s*[kK千]", text)
        daily_range = re.search(r"(\d{2,4})\s*[-~到至]\s*\d{2,4}\s*(?:元|块)\s*/?\s*(?:天|日)", text)
        daily_single = re.search(r"(\d{2,4})\s*(?:元|块)\s*/?\s*(?:天|日)", text)

        if daily_range:
            salary_mentioned = True
            salary_min = int(daily_range.group(1))
            salary_unit = "yuan_day"
        elif daily_single:
            salary_mentioned = True
            salary_min = int(daily_single.group(1))
            salary_unit = "yuan_day"
        elif monthly_range:
            salary_mentioned = True
            salary_min = int(monthly_range.group(1))
            salary_unit = "k_month"
        elif monthly_single:
            salary_mentioned = True
            salary_min = int(monthly_single.group(1))
            salary_unit = "k_month"

    return {
        "experience_mentioned": experience_mentioned,
        "experience": experience,
        "salary_mentioned": salary_mentioned,
        "salary_min": salary_min,
        "salary_unit": salary_unit,
    }

def _apply_explicit_constraint_guard(plan: dict, latest_text: str) -> tuple[dict, dict]:
    guarded = dict(plan or {})
    explicit = _extract_explicit_constraints(latest_text)

    if explicit.get("experience_mentioned"):
        guarded["experience"] = explicit.get("experience", "")
    else:
        guarded["experience"] = ""

    if explicit.get("salary_mentioned"):
        guarded["salary_min"] = int(explicit.get("salary_min", 0) or 0)
        guarded["salary_unit"] = explicit.get("salary_unit", "") or ""
    else:
        guarded["salary_min"] = 0
        guarded["salary_unit"] = ""

    if not guarded.get("salary_min", 0):
        guarded["salary_unit"] = ""

    return guarded, explicit

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
        "experience": _normalize_experience(plan.get("experience", "")),
        "salary_min": int(plan.get("salary_min", 0) or 0),
        "salary_unit": _normalize_salary_unit(plan.get("salary_unit", "")),
    }
