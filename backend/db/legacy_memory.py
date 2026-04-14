from __future__ import annotations

try:
    from utils.logger import get_logger
except ImportError:
    from backend.utils.logger import get_logger

logger = get_logger("database")


class LegacyMemoryCompatMixin:
    def get_user_profile(self, user_id: str) -> str:
        try:
            profile = self.get_memory_profile(user_id)
            facts = self.list_memory_facts(user_id, limit=20)
            lines = []
            roles = profile.get("desired_roles") or []
            cities = profile.get("desired_cities") or []
            exp = str(profile.get("desired_experience", "") or "").strip()
            salary_min = int(profile.get("desired_salary_min", 0) or 0)
            salary_unit = str(profile.get("desired_salary_unit", "") or "").strip()
            skills = profile.get("core_skills") or []
            if roles:
                lines.append(f"目标岗位:{','.join(roles[:8])}")
            if cities:
                lines.append(f"目标城市:{','.join(cities[:8])}")
            if exp:
                lines.append(f"经验偏好:{exp}")
            if salary_min > 0 and salary_unit:
                lines.append(f"薪资偏好:{salary_min}({salary_unit})")
            if skills:
                lines.append(f"核心技能:{','.join(skills[:12])}")
            for row in facts:
                key = str(row.get("fact_key", "") or "").strip()
                value = str(row.get("fact_value", "") or "").strip()
                if key and value:
                    lines.append(f"{key}:{value}")
            return "；".join(lines)
        except Exception as e:
            logger.error("get user profile failed on merged memory tables: %s", e, exc_info=True)
            return ""

    def update_user_profile(self, user_id: str, preferences: str) -> bool:
        text = str(preferences or "").strip()
        if not text:
            return True
        try:
            result = self.ingest_user_memory_from_text(user_id=user_id, text=text, source="legacy_profile")
            if not result.get("updated"):
                self.add_memory_fact(user_id, "legacy_profile", text, source="legacy_profile")
            return True
        except Exception as e:
            logger.error("update user profile failed on merged memory tables: %s", e, exc_info=True)
            return False

    def get_long_term_memories(self, user_id: str, limit: int = 20) -> list[dict]:
        try:
            rows = self.list_memory_facts(user_id, limit=limit)
            return [
                {
                    "memory_type": str(row.get("fact_key", "") or "").strip(),
                    "content": str(row.get("fact_value", "") or "").strip(),
                    "source": str(row.get("source", "") or "").strip(),
                    "confidence": float(row.get("confidence", 0.75) or 0.75),
                    "importance": int(row.get("importance", 3) or 3),
                    "expires_at": row.get("expires_at"),
                    "last_used_at": row.get("last_used_at"),
                    "use_count": int(row.get("use_count", 0) or 0),
                    "meta": row.get("meta") if isinstance(row.get("meta"), dict) else {},
                    "updated_at": row.get("updated_at"),
                }
                for row in rows
            ]
        except Exception as e:
            logger.error("get long term memories failed on merged memory tables: %s", e, exc_info=True)
            return []

    def add_long_term_memory(
        self,
        user_id: str,
        content: str,
        memory_type: str = "preference_note",
        source: str = "chat",
    ) -> bool:
        normalized_content = (content or "").strip()
        if not normalized_content:
            return False
        try:
            fact_key = str(memory_type or "").strip()[:80] or "preference_note"
            added = self.add_memory_fact(user_id, fact_key, normalized_content, source=source)
            parsed = self.ingest_user_memory_from_text(user_id=user_id, text=normalized_content, source=source)
            return bool(added or parsed.get("updated"))
        except Exception as e:
            logger.error("add long term memory failed on merged memory tables: %s", e, exc_info=True)
            return False

    def sync_user_profile_from_memories(self, user_id: str) -> bool:
        return True
