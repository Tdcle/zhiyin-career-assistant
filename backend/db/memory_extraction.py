from __future__ import annotations

from datetime import datetime, timedelta
import json
import re

try:
    from utils.logger import get_logger
except ImportError:
    from backend.utils.logger import get_logger

from .memory_constants import (
    MEMORY_CANONICAL_FACT_KEYS,
    MEMORY_CITY_CANDIDATES,
    MEMORY_ROLE_TOKENS,
    MEMORY_SKILL_TOKENS,
)

logger = get_logger("database")


class MemoryExtractionMixin:
    def _extract_memory_from_text_rule(self, text: str) -> dict:
        raw = str(text or "").strip()
        if not raw:
            return {}
        lowered = raw.lower()
        desired_roles = []
        facts = []
        for canonical, aliases in MEMORY_ROLE_TOKENS.items():
            matched_alias = next((alias for alias in aliases if str(alias or "").lower() in lowered), "")
            if not matched_alias:
                continue
            if self._entity_is_negated(lowered, matched_alias.lower()):
                facts.append(
                    {
                        "fact_key": "desired_role",
                        "fact_value": canonical,
                        "confidence": 0.90,
                        "importance": 4,
                        "meta": {"extractor": "rule", "polarity": "exclude"},
                    }
                )
            else:
                desired_roles.append(canonical)
        desired_cities = []
        city_unlimited = any(token in raw for token in ["全国", "不限城市", "城市不限", "地点不限", "地域不限"])
        if not city_unlimited:
            for city in MEMORY_CITY_CANDIDATES:
                if city in raw:
                    desired_cities.append(city)
        desired_experience = None
        experience_unlimited = any(token in raw for token in ["经验不限", "不限经验", "无经验", "不限制经验"])
        if experience_unlimited:
            desired_experience = ""
        elif (
            re.search(r"(不看|不找|不要|不考虑|别|非|不是|排除|避开).{0,4}(实习|在校|intern)", lowered)
            or re.search(r"(正式工作|全职|社招|正式岗)", lowered)
        ):
            desired_experience = "非实习"
            facts.append(
                {
                    "fact_key": "desired_experience",
                    "fact_value": "实习",
                    "confidence": 0.90,
                    "importance": 4,
                    "meta": {"extractor": "rule", "polarity": "exclude"},
                }
            )
        elif re.search(r"(实习|在校|intern)", lowered) and not re.search(r"(不|非|别|不用|不要|不限).{0,3}(实习|在校|intern)", raw):
            desired_experience = "实习"
        elif re.search(r"(应届|校招|毕业生|new\s*grad)", lowered) and not re.search(r"(不|非|别|不用|不要|不限).{0,3}(应届|校招|毕业生)", raw):
            desired_experience = "应届"
        else:
            year_range = re.search(r"(\d+\s*-\s*\d+\s*年)", raw)
            year_plus = re.search(r"(\d+\s*\+\s*年|\d+\s*年以上|\d+\s*年)", raw)
            if year_range:
                desired_experience = re.sub(r"\s+", "", year_range.group(1))
            elif year_plus:
                desired_experience = re.sub(r"\s+", "", year_plus.group(1))
        desired_salary_min = None
        desired_salary_unit = None
        if re.search(r"(薪资|工资|待遇).{0,4}(不限|面议)|薪资面议|工资面议", raw):
            desired_salary_min = 0
            desired_salary_unit = ""
        else:
            daily = re.search(r"(\d{2,4})\s*(?:元|块)\s*(?:/|每|一)?\s*(?:天|日)", raw)
            monthly = re.search(r"(\d{1,2})\s*[kK千]", raw)
            if daily:
                desired_salary_min = int(daily.group(1))
                desired_salary_unit = "yuan_day"
            elif monthly:
                desired_salary_min = int(monthly.group(1))
                desired_salary_unit = "k_month"
        core_skills = []
        for token in MEMORY_SKILL_TOKENS:
            if token in lowered:
                core_skills.append(token)
        preference_tokens = ["双休", "远程办公", "远程", "五险一金", "不加班", "弹性工作", "弹性"]
        if not city_unlimited and any(token in raw for token in preference_tokens):
            facts.append(
                {
                    "fact_key": "preference_note",
                    "fact_value": raw[:200],
                    "confidence": 0.78,
                    "importance": 3,
                    "meta": {"extractor": "rule", "polarity": "include"},
                }
            )
        return {
            "desired_roles": self._normalize_memory_list(desired_roles, max_items=20),
            "desired_cities": self._normalize_memory_list(desired_cities, max_items=20),
            "desired_experience": desired_experience,
            "desired_salary_min": desired_salary_min,
            "desired_salary_unit": desired_salary_unit,
            "core_skills": self._normalize_memory_list(core_skills, max_items=50),
            "city_unlimited": city_unlimited,
            "facts": facts,
        }

    def _normalize_extracted_city_values(self, values) -> list[str]:
        normalized: list[str] = []
        for item in values or []:
            text = str(item or "").strip()
            if not text:
                continue
            for city in MEMORY_CITY_CANDIDATES:
                if city in text:
                    normalized.append(city)
        return self._normalize_memory_list(normalized, max_items=20)

    def _normalize_extracted_memory_payload(self, payload: dict) -> dict:
        data = payload if isinstance(payload, dict) else {}
        desired_salary_min_raw = data.get("desired_salary_min")
        desired_salary_min = None
        if desired_salary_min_raw not in (None, ""):
            try:
                desired_salary_min = int(desired_salary_min_raw)
            except Exception:
                desired_salary_min = None
        desired_salary_unit = self._normalize_salary_unit(data.get("desired_salary_unit", ""))
        if desired_salary_min is not None and desired_salary_min <= 0:
            desired_salary_min = 0
            desired_salary_unit = ""

        normalized = {
            "desired_roles": self._normalize_memory_list(data.get("desired_roles", []), max_items=20),
            "desired_cities": self._normalize_extracted_city_values(data.get("desired_cities", [])),
            "desired_experience": (
                None if data.get("desired_experience") is None else str(data.get("desired_experience") or "").strip()[:50]
            ),
            "desired_salary_min": desired_salary_min,
            "desired_salary_unit": desired_salary_unit,
            "core_skills": self._normalize_memory_list(data.get("core_skills", []), max_items=50),
            "city_unlimited": bool(data.get("city_unlimited", False)),
            "facts": [],
        }

        facts = data.get("facts") or []
        if isinstance(facts, list):
            for row in facts[:20]:
                if not isinstance(row, dict):
                    continue
                raw_fact_key = str(row.get("fact_key", "") or "").strip()[:80]
                fact_key, original_fact_key = self._normalize_fact_key(raw_fact_key)
                fact_value = str(row.get("fact_value", "") or "").strip()[:1000]
                if not fact_key or not fact_value:
                    continue
                if fact_key == "desired_city":
                    cities = self._normalize_extracted_city_values([fact_value])
                    if not cities:
                        continue
                    fact_value = cities[0]
                confidence = self._normalize_confidence(row.get("confidence", 0.75))
                importance = self._normalize_importance(row.get("importance", 3))
                expires_days = row.get("expires_days")
                expires_at = None
                if expires_days not in (None, ""):
                    try:
                        days_int = int(expires_days)
                        if days_int > 0:
                            expires_at = datetime.now() + timedelta(days=days_int)
                    except Exception:
                        expires_at = None
                meta = dict(row.get("meta") or {}) if isinstance(row.get("meta"), dict) else {}
                tags = row.get("tags")
                if isinstance(tags, list):
                    meta["tags"] = self._normalize_memory_list(tags, max_items=10)
                scene = str(row.get("scene", "") or "").strip()[:30]
                if scene:
                    meta["scene"] = scene
                evidence = str(row.get("evidence", "") or "").strip()[:200]
                if evidence:
                    meta["evidence"] = evidence
                polarity = self._normalize_fact_polarity(row.get("polarity", ""))
                if polarity != "neutral":
                    meta["polarity"] = polarity
                if original_fact_key and original_fact_key != fact_key:
                    meta.setdefault("original_fact_key", original_fact_key)
                normalized["facts"].append(
                    {
                        "fact_key": fact_key,
                        "fact_value": fact_value,
                        "confidence": confidence,
                        "importance": importance,
                        "expires_at": expires_at,
                        "meta": meta,
                    }
                )
        return normalized

    def _extract_memory_from_text_llm(self, text: str) -> dict:
        raw = str(text or "").strip()
        if not raw or not self.memory_extract_llm:
            return {}
        if len(raw) < 10:
            return {}

        content = raw[:1800]
        city_hints = ", ".join(MEMORY_CITY_CANDIDATES)
        fact_key_hints = ", ".join(sorted(MEMORY_CANONICAL_FACT_KEYS))
        prompt = f"""
You are a structured information extraction engine for a job-search assistant.
Extract only stable user preferences or profile facts from the user text.
Return strict JSON only (no markdown code block).

Output schema:
{{
  "desired_roles": ["..."],
  "desired_cities": ["..."],
  "desired_experience": "" or null,
  "desired_salary_min": 0 or null,
  "desired_salary_unit": "k_month" or "yuan_day" or "",
  "core_skills": ["..."],
  "city_unlimited": false,
  "facts": [
    {{
      "fact_key": "short_key",
      "fact_value": "value",
      "confidence": 0.0,
      "importance": 1,
      "polarity": "include|exclude|neutral",
      "expires_days": 0,
      "scene": "chat",
      "tags": ["..."],
      "evidence": "short quote"
    }}
  ]
}}

Rules:
1. Keep original language in extracted values.
2. Ignore greetings and non-memory chatter.
3. confidence range 0~1, importance range 1~5.
4. expires_days should be 0 for long-term facts.
5. Cities should prefer: {city_hints}
6. fact_key MUST be chosen from: {fact_key_hints}
7. If the user is explicitly rejecting a preference, set polarity=exclude.
8. If polarity is unclear, use neutral.

User text:
{content}
"""
        try:
            response = self.memory_extract_llm.invoke(prompt)
            response_text = response.content if hasattr(response, "content") else str(response)
            parsed = self._safe_json_loads(response_text)
            return self._normalize_extracted_memory_payload(parsed)
        except Exception as exc:
            logger.warning("memory llm extraction failed: %s", exc)
            return {}

    def _merge_memory_extraction(self, primary: dict, fallback: dict) -> dict:
        p = primary if isinstance(primary, dict) else {}
        f = fallback if isinstance(fallback, dict) else {}
        merged = {
            "desired_roles": self._normalize_memory_list(
                (p.get("desired_roles") or []) + (f.get("desired_roles") or []),
                max_items=20,
            ),
            "desired_cities": self._normalize_memory_list(
                (p.get("desired_cities") or []) + (f.get("desired_cities") or []),
                max_items=20,
            ),
            "desired_experience": p.get("desired_experience"),
            "desired_salary_min": p.get("desired_salary_min"),
            "desired_salary_unit": self._normalize_salary_unit(p.get("desired_salary_unit", "")),
            "core_skills": self._normalize_memory_list(
                (p.get("core_skills") or []) + (f.get("core_skills") or []),
                max_items=50,
            ),
            "city_unlimited": bool(p.get("city_unlimited", False) or f.get("city_unlimited", False)),
            "facts": [],
        }
        if merged["desired_experience"] is None:
            merged["desired_experience"] = f.get("desired_experience")
        if merged["desired_salary_min"] is None:
            merged["desired_salary_min"] = f.get("desired_salary_min")
            merged["desired_salary_unit"] = self._normalize_salary_unit(f.get("desired_salary_unit", ""))
        if merged["desired_salary_min"] is not None and int(merged["desired_salary_min"] or 0) <= 0:
            merged["desired_salary_min"] = 0
            merged["desired_salary_unit"] = ""

        seen_facts = set()
        for bucket in [p.get("facts") or [], f.get("facts") or []]:
            for item in bucket:
                if not isinstance(item, dict):
                    continue
                key = str(item.get("fact_key", "") or "").strip()[:80]
                value = str(item.get("fact_value", "") or "").strip()[:1000]
                if not key or not value:
                    continue
                meta = dict(item.get("meta") or {}) if isinstance(item.get("meta"), dict) else {}
                polarity = self._normalize_fact_polarity(meta.get("polarity", ""))
                if merged["city_unlimited"] and key in {"desired_city", "location_note"}:
                    continue
                if merged["city_unlimited"] and key == "preference_note" and any(
                    token in value for token in ["全国", "远程", "城市不限", "不限城市"]
                ):
                    continue
                signature = (key.lower(), value.lower(), polarity)
                if signature in seen_facts:
                    continue
                seen_facts.add(signature)
                merged["facts"].append(
                    {
                        "fact_key": key,
                        "fact_value": value,
                        "confidence": self._normalize_confidence(item.get("confidence", 0.75)),
                        "importance": self._normalize_importance(item.get("importance", 3)),
                        "expires_at": self._parse_optional_datetime(item.get("expires_at")),
                        "meta": meta,
                    }
                )
        if merged["city_unlimited"]:
            merged["desired_cities"] = []

        excluded_roles = {
            str(item.get("fact_value", "") or "").strip().lower()
            for item in merged["facts"]
            if str(item.get("fact_key", "") or "") == "desired_role"
            and self._normalize_fact_polarity((item.get("meta") or {}).get("polarity", "")) == "exclude"
        }
        if excluded_roles:
            merged["desired_roles"] = [
                role for role in (merged.get("desired_roles") or []) if str(role or "").strip().lower() not in excluded_roles
            ]
        return merged

    def _extract_memory_from_text(self, text: str) -> dict:
        rule_based = self._extract_memory_from_text_rule(text)
        llm_based = self._extract_memory_from_text_llm(text)
        merged = self._merge_memory_extraction(llm_based, rule_based)
        if not any(
            [
                merged.get("desired_roles"),
                merged.get("desired_cities"),
                merged.get("desired_experience") is not None,
                merged.get("desired_salary_min") is not None,
                merged.get("core_skills"),
                merged.get("city_unlimited"),
                merged.get("facts"),
            ]
        ):
            return {}
        merged["_llm_used"] = bool(llm_based)
        return merged

    def ingest_user_memory_from_text(self, user_id: str, text: str, source: str = "chat") -> dict:
        extracted = self._extract_memory_from_text(text)
        if not extracted:
            return {"updated": False, "facts_added": 0, "profile_updated": False}

        profile = self.get_memory_profile(user_id)
        profile_updated = False
        facts_added = 0

        desired_roles = extracted.get("desired_roles") or []
        if desired_roles:
            merged_roles = self._normalize_memory_list((profile.get("desired_roles") or []) + desired_roles, max_items=20)
            if merged_roles != (profile.get("desired_roles") or []):
                profile["desired_roles"] = merged_roles
                profile_updated = True

        if extracted.get("city_unlimited"):
            if profile.get("desired_cities"):
                profile["desired_cities"] = []
                profile_updated = True
        else:
            desired_cities = extracted.get("desired_cities") or []
            if desired_cities:
                merged_cities = self._normalize_memory_list((profile.get("desired_cities") or []) + desired_cities, max_items=20)
                if merged_cities != (profile.get("desired_cities") or []):
                    profile["desired_cities"] = merged_cities
                    profile_updated = True

        if extracted.get("desired_experience") is not None:
            exp = str(extracted.get("desired_experience") or "").strip()
            if exp != str(profile.get("desired_experience", "") or "").strip():
                profile["desired_experience"] = exp
                profile_updated = True

        if extracted.get("desired_salary_min") is not None:
            salary_min = int(extracted.get("desired_salary_min") or 0)
            salary_unit = self._normalize_salary_unit(extracted.get("desired_salary_unit", ""))
            if salary_min <= 0:
                salary_unit = ""
            if salary_min != int(profile.get("desired_salary_min", 0) or 0) or salary_unit != str(profile.get("desired_salary_unit", "") or "").strip():
                profile["desired_salary_min"] = salary_min
                profile["desired_salary_unit"] = salary_unit
                profile_updated = True

        core_skills = extracted.get("core_skills") or []
        if core_skills:
            merged_skills = self._normalize_memory_list((profile.get("core_skills") or []) + core_skills, max_items=50)
            if merged_skills != (profile.get("core_skills") or []):
                profile["core_skills"] = merged_skills
                profile_updated = True

        if profile_updated:
            self.upsert_memory_profile(user_id, profile)

        fact_items: list[dict] = []
        memory_scene = str(source or "").strip()[:30]
        for role in desired_roles:
            fact_items.append(
                {
                    "fact_key": "desired_role",
                    "fact_value": role,
                    "confidence": 0.90,
                    "importance": 4,
                    "expires_at": None,
                    "meta": {"scene": memory_scene, "extractor": "structured", "polarity": "include"},
                }
            )
        for city in extracted.get("desired_cities") or []:
            fact_items.append(
                {
                    "fact_key": "desired_city",
                    "fact_value": city,
                    "confidence": 0.90,
                    "importance": 3,
                    "expires_at": None,
                    "meta": {"scene": memory_scene, "extractor": "structured", "polarity": "include"},
                }
            )
        if extracted.get("desired_experience") is not None:
            exp = str(extracted.get("desired_experience") or "").strip() or "unknown"
            fact_items.append(
                {
                    "fact_key": "desired_experience",
                    "fact_value": exp,
                    "confidence": 0.88,
                    "importance": 4,
                    "expires_at": None,
                    "meta": {"scene": memory_scene, "extractor": "structured", "polarity": "include"},
                }
            )
        if extracted.get("desired_salary_min") is not None:
            salary_min = int(extracted.get("desired_salary_min") or 0)
            salary_unit = self._normalize_salary_unit(extracted.get("desired_salary_unit", ""))
            salary_value = f"{salary_min} {salary_unit}".strip() if salary_min > 0 else "unknown"
            fact_items.append(
                {
                    "fact_key": "desired_salary",
                    "fact_value": salary_value,
                    "confidence": 0.88,
                    "importance": 4,
                    "expires_at": None,
                    "meta": {"scene": memory_scene, "extractor": "structured", "polarity": "include"},
                }
            )
        for skill in core_skills[:8]:
            fact_items.append(
                {
                    "fact_key": "core_skill",
                    "fact_value": skill,
                    "confidence": 0.85,
                    "importance": 3,
                    "expires_at": None,
                    "meta": {"scene": memory_scene, "extractor": "structured", "polarity": "include"},
                }
            )
        for item in extracted.get("facts") or []:
            if not isinstance(item, dict):
                continue
            fact_items.append(
                {
                    "fact_key": str(item.get("fact_key", "") or "").strip()[:80],
                    "fact_value": str(item.get("fact_value", "") or "").strip()[:1000],
                    "confidence": self._normalize_confidence(item.get("confidence", 0.75)),
                    "importance": self._normalize_importance(item.get("importance", 3)),
                    "expires_at": self._parse_optional_datetime(item.get("expires_at")),
                    "meta": dict(item.get("meta") or {}) if isinstance(item.get("meta"), dict) else {},
                }
            )

        seen_fact_pairs = set()
        for item in fact_items:
            fact_key = str(item.get("fact_key", "") or "").strip()
            fact_value = str(item.get("fact_value", "") or "").strip()
            if not fact_key or not fact_value:
                continue
            normalized_polarity = self._normalize_fact_polarity((item.get("meta") or {}).get("polarity", ""))
            signature = (fact_key.lower(), fact_value.lower(), normalized_polarity)
            if signature in seen_fact_pairs:
                continue
            seen_fact_pairs.add(signature)
            item_meta = dict(item.get("meta") or {}) if isinstance(item.get("meta"), dict) else {}
            if fact_key in {"desired_role", "desired_city", "desired_experience", "desired_salary"}:
                normalized_polarity = self._normalize_fact_polarity(item_meta.get("polarity", ""))
                if normalized_polarity == "neutral":
                    item_meta["polarity"] = "include"
            if self.add_memory_fact(
                user_id=user_id,
                fact_key=fact_key,
                fact_value=fact_value,
                source=source,
                confidence=item.get("confidence", 0.75),
                importance=item.get("importance", 3),
                expires_at=item.get("expires_at"),
                meta=item_meta,
            ):
                facts_added += 1

        return {
            "updated": bool(profile_updated or facts_added > 0),
            "profile_updated": profile_updated,
            "facts_added": facts_added,
            "llm_used": bool(extracted.get("_llm_used", False)),
            "extracted": extracted,
        }

    def build_memory_context(self, user_id: str, scene: str = "chat", max_facts: int = 8, query: str = "") -> str:
        profile = self.get_memory_profile(user_id)
        facts = self.list_memory_facts(
            user_id=user_id,
            limit=max_facts,
            query=query,
            scene=scene,
            touch=bool(str(query or "").strip()),
        )
        legacy_profile = self.get_user_profile(user_id)
        resume_structured = self.get_latest_resume_structured(user_id) or {}
        resume_data = resume_structured.get("structured_data", {}) if isinstance(resume_structured, dict) else {}
        lines = []
        roles = profile.get("desired_roles") or []
        cities = profile.get("desired_cities") or []
        exp = str(profile.get("desired_experience", "") or "").strip()
        salary_min = int(profile.get("desired_salary_min", 0) or 0)
        salary_unit = str(profile.get("desired_salary_unit", "") or "").strip()
        skills = profile.get("core_skills") or []
        if not roles:
            roles = self._normalize_memory_list(
                ((resume_data.get("job_intention", {}) or {}).get("target_roles", []) if isinstance(resume_data, dict) else []),
                max_items=8,
            )
        if not cities:
            cities = self._normalize_memory_list(
                ((resume_data.get("job_intention", {}) or {}).get("target_cities", []) if isinstance(resume_data, dict) else []),
                max_items=8,
            )
        if not skills:
            skills = self._normalize_memory_list(
                (resume_data.get("skills", []) if isinstance(resume_data, dict) else []),
                max_items=12,
            )
        if roles:
            lines.append(f"- 目标岗位: {', '.join(roles[:8])}")
        if cities:
            lines.append(f"- 目标城市: {', '.join(cities[:8])}")
        if exp:
            lines.append(f"- 经验偏好: {exp}")
        if salary_min > 0 and salary_unit:
            lines.append(f"- 薪资偏好: {salary_min} ({salary_unit})")
        if skills:
            lines.append(f"- 核心技能: {', '.join(skills[:12])}")
        if facts:
            rendered_facts = []
            for row in facts[:max_facts]:
                key = str(row.get("fact_key", "") or "").strip()
                value = str(row.get("fact_value", "") or "").strip()
                if key and value:
                    rendered_facts.append(f"{key}={value}")
            if rendered_facts:
                lines.append(f"- 相关事实: {'; '.join(rendered_facts)}")
        if not lines and legacy_profile:
            lines.append(f"- 历史画像: {legacy_profile}")
        if not lines:
            return "暂无"
        return "\n".join(lines)
