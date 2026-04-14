import re
from typing import TypedDict

from utils.tools import _extract_role_keywords, _normalize_city, _normalize_keyword_text


class TurnSlotSignals(TypedDict):
    role_keywords: list[str]
    role_mentioned: bool
    role_clear: bool
    city_mentioned: bool
    city_clear: bool


def _latest_text_role_keywords(latest_text: str) -> list[str]:
    return _extract_role_keywords(_normalize_keyword_text(latest_text))


def _latest_text_city(latest_text: str) -> str:
    return _normalize_city("", latest_text)


def extract_turn_slot_signals(latest_text: str) -> TurnSlotSignals:
    """Detect whether the current turn explicitly updates/clears role or city slots."""
    text = str(latest_text or "").strip()
    lowered = text.lower()
    role_keywords = _latest_text_role_keywords(text)
    city_mentions = bool(
        re.search(r"(北京|上海|深圳|广州|杭州|成都|武汉|西安|南京|苏州|天津|重庆)", text)
    )
    city_clear = bool(re.search(r"(全国|城市不限|不限城市|不限地区|地区不限|远程|remote)", lowered))
    role_clear = bool(
        re.search(r"(岗位|职位|方向|工种).{0,4}(不限|都行|都可以|随便|无所谓)", text)
        or re.search(r"(不看|不要|不想做|排除|避开).{0,4}(前端|后端|算法|运营|产品|测试|设计|ai|java|python)", lowered)
    )
    return {
        "role_keywords": role_keywords,
        "role_mentioned": bool(role_keywords) or role_clear,
        "role_clear": role_clear,
        "city_mentioned": city_mentions or city_clear,
        "city_clear": city_clear,
    }


def apply_memory_resolution_to_plan(
    plan: dict,
    latest_text: str,
    explicit_constraints: dict,
    memory_resolution: dict,
    slot_signals: TurnSlotSignals | None = None,
) -> tuple[dict, dict]:
    """
    Merge resolved memory preferences into search_plan.
    Current-turn explicit constraints always have higher priority.
    """
    resolved = dict(plan or {})
    memo = memory_resolution if isinstance(memory_resolution, dict) else {}
    signals = slot_signals if isinstance(slot_signals, dict) else {}

    role_include = [str(item or "").strip() for item in (memo.get("roles_include") or []) if str(item or "").strip()]
    role_exclude = {
        str(item or "").strip()
        for item in (memo.get("roles_exclude") or [])
        if str(item or "").strip()
    }
    city_include = [str(item or "").strip() for item in (memo.get("cities_include") or []) if str(item or "").strip()]
    city_exclude = {
        str(item or "").strip()
        for item in (memo.get("cities_exclude") or [])
        if str(item or "").strip()
    }
    exp_include = str(memo.get("experience_include", "") or "").strip()
    exp_exclude = {
        str(item or "").strip()
        for item in (memo.get("experience_exclude") or [])
        if str(item or "").strip()
    }
    salary_min = int(memo.get("salary_min", 0) or 0)
    salary_unit = str(memo.get("salary_unit", "") or "").strip()

    latest_role_keywords = _latest_text_role_keywords(latest_text)
    latest_city = _latest_text_city(latest_text)
    has_explicit_role = bool(latest_role_keywords) or bool(signals.get("role_clear", False))
    has_explicit_city = bool(signals.get("city_mentioned", False)) or bool(latest_city)
    has_explicit_experience = bool(explicit_constraints.get("experience_mentioned", False))
    has_explicit_salary = bool(explicit_constraints.get("salary_mentioned", False))

    applied = {
        "role_applied": False,
        "city_applied": False,
        "experience_applied": False,
        "salary_applied": False,
        "role_include": role_include[:6],
        "role_exclude": sorted(role_exclude)[:6],
        "city_include": city_include[:6],
        "city_exclude": sorted(city_exclude)[:6],
        "experience_include": exp_include,
        "experience_exclude": sorted(exp_exclude)[:4],
    }

    available_roles = [role for role in role_include if role not in role_exclude]
    if available_roles and not has_explicit_role:
        current_roles = _extract_role_keywords(
            resolved.get("resolved_query", ""),
            resolved.get("keyword_query", ""),
        )
        if not current_roles:
            role_prefix = " ".join(available_roles[:3])
            resolved["keyword_query"] = _normalize_keyword_text(
                " ".join(part for part in [role_prefix, resolved.get("keyword_query", "")] if str(part or "").strip())
            )
            if not resolved.get("resolved_query", "").strip():
                resolved["resolved_query"] = _normalize_keyword_text(role_prefix)
            applied["role_applied"] = True

    if city_include and not has_explicit_city:
        preferred_city = next((city for city in city_include if city not in city_exclude), "")
        current_city = str(resolved.get("city", "") or "").strip()
        if preferred_city:
            if not current_city:
                resolved["city"] = preferred_city
                applied["city_applied"] = True
            elif current_city in city_exclude:
                resolved["city"] = preferred_city
                applied["city_applied"] = True
    elif bool(signals.get("city_clear", False)):
        if str(resolved.get("city", "") or "").strip():
            resolved["city"] = ""
            applied["city_applied"] = True

    if not has_explicit_experience:
        current_experience = str(resolved.get("experience", "") or "").strip()
        if not current_experience and exp_include:
            resolved["experience"] = exp_include
            applied["experience_applied"] = True
        elif not current_experience and "实习" in exp_exclude:
            resolved["experience"] = "非实习"
            applied["experience_applied"] = True

    if not has_explicit_salary and not int(resolved.get("salary_min", 0) or 0):
        if salary_min > 0 and salary_unit:
            resolved["salary_min"] = salary_min
            resolved["salary_unit"] = salary_unit
            applied["salary_applied"] = True

    if not int(resolved.get("salary_min", 0) or 0):
        resolved["salary_unit"] = ""

    return resolved, applied
