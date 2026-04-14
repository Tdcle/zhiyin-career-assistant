from __future__ import annotations

from datetime import datetime, timedelta
import json
import math
import re

from psycopg2.extras import Json

try:
    from utils.logger import get_logger
except ImportError:
    from backend.utils.logger import get_logger

from .common import _make_json_safe, segment_text
from .memory_constants import (
    MEMORY_AUTO_TTL_DAYS,
    MEMORY_CANONICAL_FACT_KEYS,
    MEMORY_CITY_CANDIDATES,
    MEMORY_FACT_KEY_ALIASES,
    MEMORY_RESOLVABLE_FACT_KEYS,
    MEMORY_ROLE_TOKENS,
    MEMORY_SINGLE_VALUE_FACT_KEYS,
)

logger = get_logger("database")


class MemoryProfileMixin:
    @staticmethod
    def _normalize_memory_list(values, max_items: int = 20) -> list[str]:
        if not isinstance(values, list):
            return []
        seen = set()
        result = []
        for raw in values:
            text = str(raw or "").strip()
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            result.append(text)
            if len(result) >= max_items:
                break
        return result

    @staticmethod
    def _normalize_fact_key_token(raw_key: str) -> str:
        text = str(raw_key or "").strip().lower()
        if not text:
            return ""
        text = re.sub(r"[\s\-]+", "_", text)
        text = re.sub(r"_+", "_", text).strip("_")
        return text[:80]

    def _normalize_fact_key(self, fact_key: str) -> tuple[str, str]:
        raw = str(fact_key or "").strip()[:80]
        if not raw:
            return "", ""
        normalized = self._normalize_fact_key_token(raw)
        if not normalized:
            return "", ""
        canonical = MEMORY_FACT_KEY_ALIASES.get(normalized, normalized)
        if canonical in MEMORY_CANONICAL_FACT_KEYS:
            if canonical == raw:
                return canonical, ""
            return canonical, raw
        return "preference_note", raw

    @staticmethod
    def _normalize_fact_value_text(value: str) -> str:
        return re.sub(r"\s+", " ", str(value or "").strip())

    def _canonicalize_fact_value(self, fact_key: str, fact_value: str) -> str:
        key = str(fact_key or "").strip()
        value = self._normalize_fact_value_text(fact_value)
        if not value:
            return ""
        if key == "core_skill":
            return value.lower()
        if key == "desired_salary":
            lower = value.lower()
            nums = re.findall(r"\d+(?:\.\d+)?", lower)
            if nums:
                amount = int(float(nums[0]))
                if (
                    "yuan_day" in lower
                    or " day" in lower
                    or "/day" in lower
                    or any(token in value for token in ["元/天", "元一天", "元每天", "元每日", "/天", "每天", "每日", "日薪"])
                ):
                    return f"{amount} yuan_day"
                if "k_month" in lower or "k" in lower:
                    return f"{amount} k_month"
        return value

    @staticmethod
    def _default_fact_ttl_days(fact_key: str, fact_value: str, meta: dict | None = None) -> int:
        key = str(fact_key or "").strip()
        metadata = meta if isinstance(meta, dict) else {}
        if str(metadata.get("pin", "")).strip().lower() in {"1", "true", "yes"}:
            return 0
        ttl = int(MEMORY_AUTO_TTL_DAYS.get(key, 0) or 0)
        if ttl <= 0:
            return 0
        value = str(fact_value or "").strip().lower()
        if key in {"desired_experience", "desired_salary"} and value in {"", "unknown", "不限"}:
            return min(ttl, 45)
        return ttl

    def _derive_fact_expiry(self, fact_key: str, fact_value: str, expires_at, meta: dict | None = None):
        parsed = self._parse_optional_datetime(expires_at)
        metadata = meta if isinstance(meta, dict) else {}
        if parsed is not None:
            return parsed, metadata
        ttl_days = self._default_fact_ttl_days(fact_key, fact_value, metadata)
        if ttl_days <= 0:
            return None, metadata
        expiry = datetime.now() + timedelta(days=ttl_days)
        metadata = dict(metadata)
        metadata.setdefault("lifecycle", f"auto_ttl_{ttl_days}d")
        metadata.setdefault("ttl_days", ttl_days)
        return expiry, metadata

    def get_memory_profile(self, user_id: str) -> dict:
        default_profile = {
            "desired_roles": [],
            "desired_cities": [],
            "desired_experience": "",
            "desired_salary_min": 0,
            "desired_salary_unit": "",
            "core_skills": [],
        }
        try:
            with self.get_cursor(dict_cursor=True) as cur:
                cur.execute(
                    """
                    SELECT fact_key, fact_value, meta, updated_at, id
                    FROM memory_facts
                    WHERE user_id = %s
                      AND is_active = TRUE
                      AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
                      AND fact_key IN ('desired_role', 'desired_city', 'desired_experience', 'desired_salary', 'core_skill')
                    ORDER BY updated_at DESC, id DESC
                    """,
                    (user_id,),
                )
                rows = cur.fetchall() or []

            profile = dict(default_profile)

            excluded: dict[str, set[str]] = {
                "desired_role": set(),
                "desired_city": set(),
                "desired_experience": set(),
                "desired_salary": set(),
                "core_skill": set(),
            }
            included: dict[str, list[str]] = {
                "desired_role": [],
                "desired_city": [],
                "desired_experience": [],
                "desired_salary": [],
                "core_skill": [],
            }

            for row in rows:
                key = str(row.get("fact_key", "") or "").strip()
                value = str(row.get("fact_value", "") or "").strip()
                if not key or not value:
                    continue
                meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
                polarity = self._normalize_fact_polarity(meta.get("polarity", "include"))
                normalized_value = value
                if key == "desired_experience":
                    normalized_value = self._normalize_experience_value(value)
                elif key == "desired_salary":
                    salary_min, salary_unit = self._parse_salary_fact_value(value)
                    normalized_value = f"{salary_min} {salary_unit}".strip() if salary_min > 0 and salary_unit else ""
                if not normalized_value:
                    continue
                if polarity == "exclude":
                    excluded.setdefault(key, set()).add(normalized_value.lower())
                    continue
                included.setdefault(key, []).append(normalized_value)

            profile["desired_roles"] = self._normalize_memory_list(
                [value for value in included["desired_role"] if value.lower() not in excluded["desired_role"]],
                max_items=20,
            )
            profile["desired_cities"] = self._normalize_memory_list(
                [value for value in included["desired_city"] if value.lower() not in excluded["desired_city"]],
                max_items=20,
            )
            profile["core_skills"] = self._normalize_memory_list(
                [value for value in included["core_skill"] if value.lower() not in excluded["core_skill"]],
                max_items=50,
            )

            for value in included["desired_experience"]:
                if value.lower() not in excluded["desired_experience"]:
                    profile["desired_experience"] = value
                    break

            for value in included["desired_salary"]:
                if value.lower() in excluded["desired_salary"]:
                    continue
                salary_min, salary_unit = self._parse_salary_fact_value(value)
                if salary_min > 0 and salary_unit:
                    profile["desired_salary_min"] = salary_min
                    profile["desired_salary_unit"] = salary_unit
                    break

            if profile["desired_salary_min"] <= 0:
                profile["desired_salary_unit"] = ""
            return profile
        except Exception as e:
            logger.error("get memory profile failed: %s", e, exc_info=True)
            return dict(default_profile)

    def upsert_memory_profile(self, user_id: str, profile: dict) -> bool:
        payload = profile if isinstance(profile, dict) else {}
        desired_roles = self._normalize_memory_list(payload.get("desired_roles", []), max_items=20)
        desired_cities = self._normalize_memory_list(payload.get("desired_cities", []), max_items=20)
        desired_experience = str(payload.get("desired_experience", "") or "").strip()[:50]
        desired_salary_min = int(payload.get("desired_salary_min", 0) or 0)
        desired_salary_unit = str(payload.get("desired_salary_unit", "") or "").strip()[:20]
        core_skills = self._normalize_memory_list(payload.get("core_skills", []), max_items=50)
        if desired_salary_min <= 0:
            desired_salary_min = 0
            desired_salary_unit = ""
        target_values = {
            "desired_role": {value.lower() for value in desired_roles},
            "desired_city": {value.lower() for value in desired_cities},
            "desired_experience": {desired_experience.lower()} if desired_experience else set(),
            "desired_salary": {f"{desired_salary_min} {desired_salary_unit}".strip().lower()} if desired_salary_min > 0 and desired_salary_unit else set(),
            "core_skill": {value.lower() for value in core_skills},
        }
        try:
            with self.get_cursor() as cur:
                for fact_key, values in target_values.items():
                    cur.execute(
                        """
                        UPDATE memory_facts
                        SET
                            is_active = FALSE,
                            updated_at = CURRENT_TIMESTAMP,
                            meta = COALESCE(meta, '{}'::jsonb)
                                || jsonb_build_object(
                                    'superseded_by', 'profile_sync',
                                    'superseded_at', CURRENT_TIMESTAMP
                                )
                        WHERE user_id = %s
                          AND fact_key = %s
                          AND is_active = TRUE
                          AND NOT (lower(trim(fact_value)) = ANY(%s))
                        """,
                        (user_id, fact_key, list(values)),
                    )

            for role in desired_roles:
                self.add_memory_fact(user_id, "desired_role", role, source="memory_profile", confidence=0.92, importance=4, meta={"extractor": "profile_sync", "polarity": "include"})
            for city in desired_cities:
                self.add_memory_fact(user_id, "desired_city", city, source="memory_profile", confidence=0.90, importance=3, meta={"extractor": "profile_sync", "polarity": "include"})
            if desired_experience:
                self.add_memory_fact(user_id, "desired_experience", desired_experience, source="memory_profile", confidence=0.90, importance=4, meta={"extractor": "profile_sync", "polarity": "include"})
            if desired_salary_min > 0 and desired_salary_unit:
                self.add_memory_fact(user_id, "desired_salary", f"{desired_salary_min} {desired_salary_unit}", source="memory_profile", confidence=0.90, importance=4, meta={"extractor": "profile_sync", "polarity": "include"})
            for skill in core_skills:
                self.add_memory_fact(user_id, "core_skill", skill, source="memory_profile", confidence=0.86, importance=3, meta={"extractor": "profile_sync", "polarity": "include"})
            return True
        except Exception as e:
            logger.error("upsert memory profile failed: %s", e, exc_info=True)
            return False

    @staticmethod
    def _normalize_confidence(value, default: float = 0.75) -> float:
        try:
            parsed = float(value)
        except Exception:
            parsed = float(default)
        return max(0.0, min(1.0, parsed))

    @staticmethod
    def _normalize_importance(value, default: int = 3) -> int:
        try:
            parsed = int(value)
        except Exception:
            parsed = int(default)
        return max(1, min(5, parsed))

    @staticmethod
    def _normalize_salary_unit(unit: str) -> str:
        text = str(unit or "").strip().lower()
        if text in {"k_month", "k", "month", "monthly"}:
            return "k_month"
        if text in {"yuan_day", "day", "daily"}:
            return "yuan_day"
        return ""

    @staticmethod
    def _safe_json_loads(raw: str):
        text = str(raw or "").strip()
        if not text:
            return {}
        if "```" in text:
            chunks = text.split("```")
            if len(chunks) >= 3:
                text = chunks[1].replace("json", "", 1).strip()
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end >= start:
            text = text[start:end + 1]
        try:
            return json.loads(text)
        except Exception:
            return {}

    @staticmethod
    def _memory_terms(text: str) -> list[str]:
        raw = str(text or "").strip()
        if not raw:
            return []
        normalized = segment_text(raw)
        terms = [item for item in re.split(r"\s+", normalized) if item]
        terms.extend(re.findall(r"[a-z0-9\+\#\.\-]{2,}", raw.lower()))
        seen = set()
        result = []
        for term in terms:
            key = term.lower()
            if key in seen:
                continue
            seen.add(key)
            result.append(term)
        return result

    @staticmethod
    def _parse_optional_datetime(value):
        if value in (None, ""):
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            try:
                return datetime.fromisoformat(text.replace("Z", "+00:00"))
            except Exception:
                return None
        return None

    def _compute_memory_rank_score(self, row: dict, query_terms: list[str], scene: str = "") -> float:
        query_terms = [item for item in (query_terms or []) if item]
        combined_text = " ".join(
            [
                str(row.get("fact_key", "") or ""),
                str(row.get("fact_value", "") or ""),
                str(row.get("source", "") or ""),
            ]
        )
        meta = row.get("meta") or {}
        if isinstance(meta, dict):
            tags = meta.get("tags") or []
            if isinstance(tags, list) and tags:
                combined_text += " " + " ".join(str(tag) for tag in tags if str(tag or "").strip())
            combined_text += " " + str(meta.get("scene", "") or "")
        combined_lower = combined_text.lower()
        fact_terms = set(self._memory_terms(combined_text))
        query_set = set(query_terms)
        overlap = (len(query_set & fact_terms) / max(len(query_set), 1)) if query_set else 0.0
        token_hit = (sum(1 for term in query_set if term in combined_lower) / max(len(query_set), 1)) if query_set else 0.0
        relevance = 0.7 * overlap + 0.3 * token_hit

        confidence = self._normalize_confidence(row.get("confidence", 0.75))
        importance = self._normalize_importance(row.get("importance", 3))
        use_count = int(row.get("use_count", 0) or 0)

        updated_at = row.get("updated_at")
        age_days = 365.0
        if isinstance(updated_at, datetime):
            now = datetime.now(updated_at.tzinfo) if updated_at.tzinfo else datetime.now()
            age_days = max(0.0, (now - updated_at).total_seconds() / 86400.0)
        recency_score = math.exp(-age_days / 30.0)

        scene_bonus = 0.0
        scene_key = str(scene or "").strip().lower()
        if scene_key:
            source_key = str(row.get("source", "") or "").strip().lower()
            meta_scene = str((meta or {}).get("scene", "") or "").strip().lower() if isinstance(meta, dict) else ""
            if meta_scene == scene_key:
                scene_bonus += 0.25
            if source_key == scene_key:
                scene_bonus += 0.10

        return (
            2.2 * relevance
            + 0.55 * confidence
            + 0.20 * float(importance)
            + 0.25 * recency_score
            + 0.05 * float(min(use_count, 20))
            + scene_bonus
        )

    def list_memory_facts(
        self,
        user_id: str,
        limit: int = 20,
        query: str = "",
        scene: str = "",
        touch: bool = False,
    ) -> list[dict]:
        limit_n = max(1, int(limit))
        query_text = str(query or "").strip()
        candidate_limit = limit_n if not query_text else max(limit_n * 8, 80)
        try:
            with self.get_cursor(dict_cursor=True) as cur:
                cur.execute(
                    """
                    SELECT
                        id,
                        fact_key,
                        fact_value,
                        source,
                        is_active,
                        confidence,
                        importance,
                        expires_at,
                        last_used_at,
                        use_count,
                        meta,
                        created_at,
                        updated_at
                    FROM memory_facts
                    WHERE user_id = %s
                      AND is_active = TRUE
                      AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
                    ORDER BY updated_at DESC, id DESC
                    LIMIT %s
                    """,
                    (user_id, candidate_limit),
                )
                rows = cur.fetchall() or []
        except Exception as e:
            logger.error("list memory facts failed: %s", e, exc_info=True)
            return []

        if not query_text:
            selected_rows = rows[:limit_n]
        else:
            query_terms = self._memory_terms(query_text)
            scored_rows = []
            for row in rows:
                item = dict(row)
                item["_rank_score"] = round(self._compute_memory_rank_score(item, query_terms, scene=scene), 6)
                scored_rows.append(item)
            scored_rows.sort(
                key=lambda item: (
                    float(item.get("_rank_score", 0.0) or 0.0),
                    (
                        item.get("updated_at").timestamp()
                        if isinstance(item.get("updated_at"), datetime)
                        else 0.0
                    ),
                ),
                reverse=True,
            )
            selected_rows = scored_rows[:limit_n]

        if touch and selected_rows:
            fact_ids = [int(item["id"]) for item in selected_rows if item.get("id") is not None]
            if fact_ids:
                try:
                    with self.get_cursor() as cur:
                        cur.execute(
                            """
                            UPDATE memory_facts
                            SET
                                last_used_at = CURRENT_TIMESTAMP,
                                use_count = COALESCE(use_count, 0) + 1
                            WHERE id = ANY(%s)
                            """,
                            (fact_ids,),
                        )
                except Exception as e:
                    logger.warning("touch memory facts failed: %s", e, exc_info=True)

        return selected_rows

    @staticmethod
    def _normalize_fact_polarity(value: str) -> str:
        text = str(value or "").strip().lower()
        if text in {"exclude", "negative", "block", "blocked", "dislike"}:
            return "exclude"
        if text in {"include", "positive", "prefer", "preferred", "like"}:
            return "include"
        return "neutral"

    @staticmethod
    def _entity_is_negated(text: str, token: str) -> bool:
        body = str(text or "").lower()
        target = str(token or "").lower().strip()
        if not body or not target:
            return False
        escaped = re.escape(target)
        leading_negation = re.search(
            rf"(不看|不要|不想|不找|不考虑|排除|避开|别|非|不是).{{0,4}}{escaped}",
            body,
        )
        # Trailing negation should stay in the same short clause. This avoids
        # treating "想做前端，不考虑后端" as a rejection of "前端".
        same_clause_gap = r"[\s\w一-龥]{0,4}"
        trailing_negation = re.search(
            rf"{escaped}{same_clause_gap}(不看|不要|不想|不找|不考虑|排除|避开)",
            body,
        )
        return bool(leading_negation or trailing_negation)

    @staticmethod
    def _normalize_experience_value(value: str) -> str:
        text = str(value or "").strip()
        lowered = text.lower()
        if not text:
            return ""
        if text in {"不限", "unknown", "未说明"}:
            return ""
        if any(token in lowered for token in ["非实习", "全职", "正式工作", "正式岗", "社招"]):
            return "非实习"
        if any(token in lowered for token in ["实习", "intern", "在校"]):
            return "实习"
        if any(token in lowered for token in ["应届", "校招", "毕业生", "new grad"]):
            return "应届"
        return text[:50]

    @staticmethod
    def _parse_salary_fact_value(value: str) -> tuple[int, str]:
        text = str(value or "").strip()
        lowered = text.lower()
        if not text:
            return 0, ""
        if any(token in text for token in ["不限", "面议"]) or "unknown" in lowered:
            return 0, ""
        nums = re.findall(r"\d+(?:\.\d+)?", lowered)
        if not nums:
            return 0, ""
        amount = int(float(nums[0]))
        if any(token in lowered for token in ["yuan_day", " day", "/day"]):
            return amount, "yuan_day"
        if any(token in text for token in ["元/天", "元一天", "元每天", "元每日", "/天", "每天", "每日", "日薪"]):
            return amount, "yuan_day"
        if any(token in lowered for token in ["k_month", "月薪", "k"]):
            return amount, "k_month"
        return 0, ""

    def _parse_note_fact_slots(
        self,
        fact_key: str,
        fact_value: str,
        default_polarity: str = "neutral",
    ) -> list[tuple[str, str, str]]:
        key = str(fact_key or "").strip()
        text = str(fact_value or "").strip()
        lowered = text.lower()
        if not text:
            return []

        slots: list[tuple[str, str, str]] = []
        base_polarity = self._normalize_fact_polarity(default_polarity)

        if key in {"location_note", "preference_note"}:
            for city in MEMORY_CITY_CANDIDATES:
                if city not in text:
                    continue
                polarity = "exclude" if self._entity_is_negated(text, city) else (
                    "include" if base_polarity != "exclude" else "exclude"
                )
                slots.append(("desired_city", city, polarity))

        if key in {"preference_note", "skill_note", "experience_note"}:
            for canonical, aliases in MEMORY_ROLE_TOKENS.items():
                matched_alias = ""
                for alias in aliases:
                    alias_text = str(alias or "").strip()
                    if alias_text and alias_text.lower() in lowered:
                        matched_alias = alias_text
                        break
                if not matched_alias:
                    continue
                polarity = "exclude" if self._entity_is_negated(lowered, matched_alias.lower()) else (
                    "include" if base_polarity != "exclude" else "exclude"
                )
                slots.append(("desired_role", canonical, polarity))

        if key in {"experience_note", "preference_note"}:
            if re.search(
                r"(不看|不找|不要|不考虑|别|非|不是).{0,3}(实习|在校|intern)|"
                r"(正式工作|全职|社招|正式岗)",
                lowered,
            ):
                slots.append(("desired_experience", "非实习", "include"))
                slots.append(("desired_experience", "实习", "exclude"))
            elif re.search(r"(实习|在校|intern)", lowered) and not re.search(
                r"(不|非|别|不用|不要|不限).{0,3}(实习|在校|intern)",
                lowered,
            ):
                slots.append(("desired_experience", "实习", "include"))
            elif re.search(r"(应届|校招|毕业生|new\s*grad)", lowered) and not re.search(
                r"(不|非|别|不用|不要|不限).{0,3}(应届|校招|毕业生|new\s*grad)",
                lowered,
            ):
                slots.append(("desired_experience", "应届", "include"))

        if key in {"salary_note", "preference_note"}:
            if re.search(r"(薪资|工资|待遇|月薪|日薪|元/天|\d+\s*[kK])", text):
                salary_min, salary_unit = self._parse_salary_fact_value(text)
                if salary_min > 0 and salary_unit:
                    slots.append(("desired_salary", f"{salary_min} {salary_unit}", "include"))
                if re.search(r"(薪资|工资|待遇).{0,4}(不限|面议)|薪资面议|工资面议", text):
                    slots.append(("desired_salary", "unknown", "include"))

        deduped: list[tuple[str, str, str]] = []
        seen = set()
        for slot, value, polarity in slots:
            signature = (slot, value.strip().lower(), polarity)
            if signature in seen:
                continue
            seen.add(signature)
            deduped.append((slot, value, polarity))
        return deduped

    def resolve_memory_preferences(
        self,
        user_id: str,
        query: str = "",
        scene: str = "chat",
        limit: int = 80,
    ) -> dict:
        query_text = str(query or "").strip()
        profile = self.get_memory_profile(user_id)
        facts = self.list_memory_facts(
            user_id=user_id,
            limit=max(20, int(limit)),
            query=query_text,
            scene=scene,
            touch=bool(query_text),
        )
        query_terms = self._memory_terms(query_text)

        slot_scores: dict[str, dict[str, dict[str, float]]] = {
            "desired_role": {"include": {}, "exclude": {}},
            "desired_city": {"include": {}, "exclude": {}},
            "desired_experience": {"include": {}, "exclude": {}},
            "desired_salary": {"include": {}, "exclude": {}},
        }
        fact_trace: list[dict] = []

        def _push(slot: str, value: str, polarity: str, row: dict, score: float) -> None:
            if slot not in slot_scores:
                return
            normalized_polarity = self._normalize_fact_polarity(polarity)
            if normalized_polarity == "neutral":
                normalized_polarity = "include"

            normalized_value = str(value or "").strip()
            if slot == "desired_experience":
                normalized_value = self._normalize_experience_value(normalized_value)
            elif slot == "desired_salary":
                salary_min, salary_unit = self._parse_salary_fact_value(normalized_value)
                normalized_value = f"{salary_min} {salary_unit}".strip() if salary_min > 0 and salary_unit else ""
            elif slot == "desired_role":
                for canonical, aliases in MEMORY_ROLE_TOKENS.items():
                    if (
                        normalized_value.lower() in [str(alias).lower() for alias in aliases]
                        or normalized_value.lower() == canonical.lower()
                        or any(
                            str(alias).strip()
                            and str(alias).lower() in normalized_value.lower()
                            for alias in aliases
                        )
                    ):
                        normalized_value = canonical
                        break
            elif slot == "desired_city":
                for city in MEMORY_CITY_CANDIDATES:
                    if city in normalized_value:
                        normalized_value = city
                        break

            normalized_value = normalized_value.strip()
            if not normalized_value:
                return

            bucket = slot_scores[slot][normalized_polarity]
            bucket[normalized_value] = float(bucket.get(normalized_value, 0.0) or 0.0) + max(score, 0.05)

            fact_id = row.get("id")

            fact_trace.append(
                {
                    "id": int(fact_id) if fact_id is not None else None,
                    "slot": slot,
                    "polarity": normalized_polarity,
                    "value": normalized_value,
                    "score": round(max(score, 0.05), 6),
                    "fact_key": str(row.get("fact_key", "") or "").strip(),
                    "fact_value": str(row.get("fact_value", "") or "").strip(),
                }
            )

        for row in facts:
            key = str(row.get("fact_key", "") or "").strip()
            value = str(row.get("fact_value", "") or "").strip()
            if key not in MEMORY_RESOLVABLE_FACT_KEYS:
                continue
            if not value:
                continue
            meta = row.get("meta") or {}
            polarity = self._normalize_fact_polarity((meta or {}).get("polarity", "") if isinstance(meta, dict) else "")
            base_score = self._compute_memory_rank_score(row, query_terms, scene=scene)
            if key in {"desired_role", "desired_city", "desired_experience", "desired_salary"}:
                _push(key, value, polarity, row, base_score + 0.22)
            else:
                for slot, slot_value, slot_polarity in self._parse_note_fact_slots(key, value, default_polarity=polarity):
                    _push(slot, slot_value, slot_polarity, row, base_score + 0.08)

        for role in profile.get("desired_roles") or []:
            _push("desired_role", role, "include", {"id": None, "fact_key": "profile", "fact_value": role}, 0.32)
        for city in profile.get("desired_cities") or []:
            _push("desired_city", city, "include", {"id": None, "fact_key": "profile", "fact_value": city}, 0.32)
        profile_exp = self._normalize_experience_value(profile.get("desired_experience", ""))
        if profile_exp:
            _push("desired_experience", profile_exp, "include", {"id": None, "fact_key": "profile", "fact_value": profile_exp}, 0.30)
        profile_salary_min = int(profile.get("desired_salary_min", 0) or 0)
        profile_salary_unit = self._normalize_salary_unit(profile.get("desired_salary_unit", ""))
        if profile_salary_min > 0 and profile_salary_unit:
            _push(
                "desired_salary",
                f"{profile_salary_min} {profile_salary_unit}",
                "include",
                {"id": None, "fact_key": "profile", "fact_value": f"{profile_salary_min} {profile_salary_unit}"},
                0.30,
            )

        def _finalize_slot(slot: str, max_items: int = 6) -> tuple[list[str], list[str]]:
            include_bucket = slot_scores.get(slot, {}).get("include", {})
            exclude_bucket = slot_scores.get(slot, {}).get("exclude", {})
            include_ranked = sorted(include_bucket.items(), key=lambda item: item[1], reverse=True)
            exclude_ranked = sorted(exclude_bucket.items(), key=lambda item: item[1], reverse=True)
            include_values: list[str] = []
            for value, score in include_ranked:
                if score + 0.08 < float(exclude_bucket.get(value, 0.0) or 0.0):
                    continue
                include_values.append(value)
                if len(include_values) >= max_items:
                    break
            exclude_values = [value for value, _ in exclude_ranked[:max_items]]
            return include_values, exclude_values

        role_include, role_exclude = _finalize_slot("desired_role", max_items=8)
        city_include, city_exclude = _finalize_slot("desired_city", max_items=8)
        exp_include_list, exp_exclude_list = _finalize_slot("desired_experience", max_items=4)
        salary_include_list, salary_exclude_list = _finalize_slot("desired_salary", max_items=3)

        experience_include = exp_include_list[0] if exp_include_list else ""
        if not experience_include and "实习" in exp_exclude_list:
            experience_include = "非实习"

        salary_min = 0
        salary_unit = ""
        for item in salary_include_list:
            if item in set(salary_exclude_list):
                continue
            parsed_min, parsed_unit = self._parse_salary_fact_value(item)
            if parsed_min > 0 and parsed_unit:
                salary_min = parsed_min
                salary_unit = parsed_unit
                break

        fact_trace.sort(key=lambda row: float(row.get("score", 0.0) or 0.0), reverse=True)
        top_facts = fact_trace[:12]
        used_fact_ids: list[int] = []
        seen_fact_id = set()
        for row in top_facts:
            if row.get("id") is None:
                continue
            fid = int(row["id"])
            if fid in seen_fact_id:
                continue
            seen_fact_id.add(fid)
            used_fact_ids.append(fid)

        return {
            "roles_include": role_include,
            "roles_exclude": role_exclude,
            "cities_include": city_include,
            "cities_exclude": city_exclude,
            "experience_include": experience_include,
            "experience_exclude": exp_exclude_list,
            "salary_min": int(salary_min or 0),
            "salary_unit": salary_unit,
            "used_fact_ids": used_fact_ids,
            "top_facts": top_facts,
            "fact_count": len(facts),
            "query": query_text,
        }

    @staticmethod
    def _serialize_memory_fact_row(row: dict) -> dict:
        if not row:
            return {}

        def _iso(value):
            return value.isoformat() if isinstance(value, datetime) else value

        return {
            "id": int(row.get("id")),
            "fact_key": str(row.get("fact_key", "") or ""),
            "fact_value": str(row.get("fact_value", "") or ""),
            "source": str(row.get("source", "") or ""),
            "is_active": bool(row.get("is_active", False)),
            "confidence": float(row.get("confidence", 0.75) or 0.75),
            "importance": int(row.get("importance", 3) or 3),
            "expires_at": _iso(row.get("expires_at")),
            "last_used_at": _iso(row.get("last_used_at")),
            "use_count": int(row.get("use_count", 0) or 0),
            "meta": row.get("meta") if isinstance(row.get("meta"), dict) else {},
            "created_at": _iso(row.get("created_at")),
            "updated_at": _iso(row.get("updated_at")),
        }

    def list_user_memory_items(self, user_id: str, include_inactive: bool = False, limit: int = 200) -> list[dict]:
        limit_n = max(1, min(int(limit or 200), 500))
        try:
            with self.get_cursor(dict_cursor=True) as cur:
                cur.execute(
                    """
                    SELECT
                        id,
                        fact_key,
                        fact_value,
                        source,
                        is_active,
                        confidence,
                        importance,
                        expires_at,
                        last_used_at,
                        use_count,
                        meta,
                        created_at,
                        updated_at
                    FROM memory_facts
                    WHERE user_id = %s
                      AND (%s OR is_active = TRUE)
                    ORDER BY is_active DESC, updated_at DESC, id DESC
                    LIMIT %s
                    """,
                    (user_id, include_inactive, limit_n),
                )
                rows = cur.fetchall() or []
            return [self._serialize_memory_fact_row(dict(row)) for row in rows]
        except Exception as e:
            logger.error("list user memory items failed: %s", e, exc_info=True)
            return []

    def update_memory_fact(
        self,
        user_id: str,
        fact_id: int,
        fact_key: str,
        fact_value: str,
        confidence: float = 0.75,
        importance: int = 3,
        meta: dict | None = None,
    ) -> dict | None:
        key, original_key = self._normalize_fact_key(fact_key)
        value = self._canonicalize_fact_value(key, fact_value)
        if not key or not value:
            return None

        normalized_confidence = self._normalize_confidence(confidence)
        normalized_importance = self._normalize_importance(importance)
        user_meta = dict(meta) if isinstance(meta, dict) else {}
        user_meta.update({"edited_by_user": True, "polarity": self._normalize_fact_polarity(user_meta.get("polarity", "include"))})
        if original_key and original_key != key:
            user_meta.setdefault("original_fact_key", original_key)

        try:
            with self.get_cursor(dict_cursor=True) as cur:
                cur.execute(
                    """
                    SELECT *
                    FROM memory_facts
                    WHERE user_id = %s
                      AND id = %s
                      AND is_active = TRUE
                    LIMIT 1
                    """,
                    (user_id, int(fact_id)),
                )
                current = cur.fetchone()
                if not current:
                    return None

                current_meta = current.get("meta") if isinstance(current.get("meta"), dict) else {}
                merged_meta = dict(current_meta)
                merged_meta.update(user_meta)
                expires_dt, merged_meta = self._derive_fact_expiry(
                    fact_key=key,
                    fact_value=value,
                    expires_at=current.get("expires_at"),
                    meta=merged_meta,
                )

                target_polarity = self._normalize_fact_polarity(merged_meta.get("polarity", "include"))
                if key in MEMORY_SINGLE_VALUE_FACT_KEYS and target_polarity != "exclude":
                    cur.execute(
                        """
                        UPDATE memory_facts
                        SET
                            is_active = FALSE,
                            updated_at = CURRENT_TIMESTAMP,
                            meta = COALESCE(meta, '{}'::jsonb)
                                || jsonb_build_object(
                                    'superseded_by_value', %s,
                                    'superseded_at', CURRENT_TIMESTAMP
                                )
                        WHERE user_id = %s
                          AND fact_key = %s
                          AND id <> %s
                          AND is_active = TRUE
                          AND COALESCE(meta->>'polarity', 'include') <> 'exclude'
                        """,
                        (value[:1000], user_id, key, int(fact_id)),
                    )

                cur.execute(
                    """
                    SELECT id
                    FROM memory_facts
                    WHERE user_id = %s
                      AND fact_key = %s
                      AND lower(trim(fact_value)) = lower(trim(%s))
                      AND id <> %s
                    ORDER BY is_active DESC, updated_at DESC, id DESC
                    LIMIT 1
                    """,
                    (user_id, key, value[:1000], int(fact_id)),
                )
                duplicate = cur.fetchone()
                if duplicate:
                    duplicate_id = int(duplicate["id"])
                    cur.execute(
                        """
                        UPDATE memory_facts
                        SET
                            is_active = FALSE,
                            updated_at = CURRENT_TIMESTAMP,
                            meta = COALESCE(meta, '{}'::jsonb)
                                || jsonb_build_object(
                                    'merged_into_fact_id', %s,
                                    'merged_at', CURRENT_TIMESTAMP
                                )
                        WHERE id = %s
                          AND user_id = %s
                        """,
                        (duplicate_id, int(fact_id), user_id),
                    )
                    cur.execute(
                        """
                        UPDATE memory_facts
                        SET
                            source = 'user_edit',
                            is_active = TRUE,
                            confidence = %s,
                            importance = %s,
                            expires_at = %s,
                            meta = COALESCE(meta, '{}'::jsonb) || %s,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE id = %s
                          AND user_id = %s
                        RETURNING *
                        """,
                        (
                            normalized_confidence,
                            normalized_importance,
                            expires_dt,
                            Json(_make_json_safe(merged_meta)),
                            duplicate_id,
                            user_id,
                        ),
                    )
                    return self._serialize_memory_fact_row(dict(cur.fetchone() or {}))

                cur.execute(
                    """
                    UPDATE memory_facts
                    SET
                        fact_key = %s,
                        fact_value = %s,
                        source = 'user_edit',
                        is_active = TRUE,
                        confidence = %s,
                        importance = %s,
                        expires_at = %s,
                        meta = %s,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = %s
                      AND user_id = %s
                    RETURNING *
                    """,
                    (
                        key,
                        value[:1000],
                        normalized_confidence,
                        normalized_importance,
                        expires_dt,
                        Json(_make_json_safe(merged_meta)),
                        int(fact_id),
                        user_id,
                    ),
                )
                return self._serialize_memory_fact_row(dict(cur.fetchone() or {}))
        except Exception as e:
            logger.error("update memory fact failed: %s", e, exc_info=True)
            return None

    def delete_memory_fact(self, user_id: str, fact_id: int) -> bool:
        try:
            with self.get_cursor() as cur:
                cur.execute(
                    """
                    UPDATE memory_facts
                    SET
                        is_active = FALSE,
                        updated_at = CURRENT_TIMESTAMP,
                        meta = COALESCE(meta, '{}'::jsonb)
                            || jsonb_build_object(
                                'deleted_by_user', TRUE,
                                'deleted_at', CURRENT_TIMESTAMP
                            )
                    WHERE user_id = %s
                      AND id = %s
                      AND is_active = TRUE
                    """,
                    (user_id, int(fact_id)),
                )
                return cur.rowcount > 0
        except Exception as e:
            logger.error("delete memory fact failed: %s", e, exc_info=True)
            return False

    def add_memory_fact(
        self,
        user_id: str,
        fact_key: str,
        fact_value: str,
        source: str = "chat",
        confidence: float = 0.75,
        importance: int = 3,
        expires_at=None,
        meta: dict | None = None,
    ) -> bool:
        raw_key = str(fact_key or "").strip()[:80]
        key, original_key = self._normalize_fact_key(raw_key)
        value = self._canonicalize_fact_value(key, fact_value)
        if not key or not value:
            return False

        normalized_confidence = self._normalize_confidence(confidence)
        normalized_importance = self._normalize_importance(importance)
        metadata = dict(meta) if isinstance(meta, dict) else {}
        if original_key and original_key != key:
            metadata.setdefault("original_fact_key", original_key)
        expires_dt, metadata = self._derive_fact_expiry(
            fact_key=key,
            fact_value=value,
            expires_at=expires_at,
            meta=metadata,
        )

        try:
            with self.get_cursor() as cur:
                # Case-insensitive dedupe: reuse existing stored value spelling when possible.
                cur.execute(
                    """
                    SELECT fact_value
                    FROM memory_facts
                    WHERE user_id = %s
                      AND fact_key = %s
                      AND lower(trim(fact_value)) = lower(trim(%s))
                    ORDER BY is_active DESC, updated_at DESC, id DESC
                    LIMIT 1
                    """,
                    (user_id, key, value[:1000]),
                )
                existing_row = cur.fetchone()
                if existing_row and existing_row[0]:
                    value = str(existing_row[0])[:1000]
                else:
                    value = value[:1000]

                # Keep only one active included value for single-value keys.
                fact_polarity = self._normalize_fact_polarity(metadata.get("polarity", "include"))
                if key in MEMORY_SINGLE_VALUE_FACT_KEYS and fact_polarity != "exclude":
                    cur.execute(
                        """
                        UPDATE memory_facts
                        SET
                            is_active = FALSE,
                            updated_at = CURRENT_TIMESTAMP,
                            meta = COALESCE(meta, '{}'::jsonb)
                                || jsonb_build_object(
                                    'superseded_by_value', %s,
                                    'superseded_at', CURRENT_TIMESTAMP
                                )
                        WHERE user_id = %s
                          AND fact_key = %s
                          AND is_active = TRUE
                          AND lower(trim(fact_value)) <> lower(trim(%s))
                          AND COALESCE(meta->>'polarity', 'include') <> 'exclude'
                        """,
                        (value, user_id, key, value),
                    )

                cur.execute(
                    """
                    INSERT INTO memory_facts (
                        user_id,
                        fact_key,
                        fact_value,
                        source,
                        is_active,
                        confidence,
                        importance,
                        expires_at,
                        last_used_at,
                        use_count,
                        meta
                    )
                    VALUES (%s, %s, %s, %s, TRUE, %s, %s, %s, CURRENT_TIMESTAMP, 0, %s)
                    ON CONFLICT (user_id, fact_key, fact_value)
                    DO UPDATE SET
                        source = EXCLUDED.source,
                        is_active = TRUE,
                        confidence = GREATEST(COALESCE(memory_facts.confidence, 0.0), EXCLUDED.confidence),
                        importance = GREATEST(COALESCE(memory_facts.importance, 1), EXCLUDED.importance),
                        expires_at = CASE
                            WHEN EXCLUDED.expires_at IS NULL THEN memory_facts.expires_at
                            ELSE EXCLUDED.expires_at
                        END,
                        meta = COALESCE(memory_facts.meta, '{}'::jsonb) || EXCLUDED.meta,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (
                        user_id,
                        key,
                        value,
                        str(source or "chat")[:50],
                        normalized_confidence,
                        normalized_importance,
                        expires_dt,
                        Json(_make_json_safe(metadata)),
                    ),
                )
            return True
        except Exception as e:
            logger.error("add memory fact failed: %s", e, exc_info=True)
            return False
