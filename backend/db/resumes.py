from __future__ import annotations

import re

import psycopg2
from psycopg2.extras import Json

try:
    from config.config import config
    from utils.logger import get_logger
except ImportError:
    from backend.config.config import config
    from backend.utils.logger import get_logger

from .common import _make_json_safe

logger = get_logger("database")


class ResumeMixin:
    @staticmethod
    def _clean_resume_text(value, max_len: int = 2000) -> str:
        text = str(value or "").strip()
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text[:max_len] if len(text) > max_len else text

    @classmethod
    def _clean_resume_list(cls, values, max_items: int = 100, item_len: int = 255) -> list[str]:
        if not isinstance(values, list):
            return []
        seen: set[str] = set()
        result: list[str] = []
        for raw in values:
            text = cls._clean_resume_text(raw, max_len=item_len)
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

    @classmethod
    def _clean_resume_dict_list(
        cls,
        values,
        fields: list[str],
        max_items: int,
        field_limit: int = 1200,
    ) -> list[dict]:
        if not isinstance(values, list):
            return []
        result: list[dict] = []
        for raw in values:
            if not isinstance(raw, dict):
                continue
            row = {field: cls._clean_resume_text(raw.get(field, ""), max_len=field_limit) for field in fields}
            if not any(row.values()):
                continue
            result.append(row)
            if len(result) >= max_items:
                break
        return result

    @classmethod
    def _normalize_resume_structured_data(cls, structured_data: dict | None) -> dict:
        payload = structured_data if isinstance(structured_data, dict) else {}
        basic = payload.get("basic_info", {}) if isinstance(payload.get("basic_info"), dict) else {}
        intention = payload.get("job_intention", {}) if isinstance(payload.get("job_intention"), dict) else {}

        projects = cls._clean_resume_dict_list(
            payload.get("projects"),
            ["name", "role", "start_date", "end_date", "description"],
            max_items=30,
            field_limit=2000,
        )
        raw_projects = payload.get("projects") if isinstance(payload.get("projects"), list) else []
        for idx, row in enumerate(projects):
            raw = raw_projects[idx] if idx < len(raw_projects) and isinstance(raw_projects[idx], dict) else {}
            stack = raw.get("tech_stack", [])
            if isinstance(stack, str):
                stack = re.split(r"[,\n;，；、]+", stack)
            row["tech_stack"] = cls._clean_resume_list(stack if isinstance(stack, list) else [], max_items=20, item_len=80)

        normalized = {
            "basic_info": {
                "name": cls._clean_resume_text(basic.get("name", ""), max_len=100),
                "phone": cls._clean_resume_text(basic.get("phone", ""), max_len=50),
                "email": cls._clean_resume_text(basic.get("email", ""), max_len=255),
                "wechat": cls._clean_resume_text(basic.get("wechat", ""), max_len=100),
                "city": cls._clean_resume_text(basic.get("city", ""), max_len=100),
                "highest_degree": cls._clean_resume_text(basic.get("highest_degree", ""), max_len=50),
            },
            "education": cls._clean_resume_dict_list(
                payload.get("education"),
                ["school", "degree", "major", "start_date", "end_date", "description"],
                max_items=30,
                field_limit=1600,
            ),
            "work_experience": cls._clean_resume_dict_list(
                payload.get("work_experience"),
                ["company", "title", "start_date", "end_date", "description"],
                max_items=40,
                field_limit=2200,
            ),
            "projects": projects,
            "skills": cls._clean_resume_list(payload.get("skills", []), max_items=200, item_len=120),
            "certificates": cls._clean_resume_list(payload.get("certificates", []), max_items=120, item_len=255),
            "languages": cls._clean_resume_list(payload.get("languages", []), max_items=80, item_len=120),
            "job_intention": {
                "target_roles": cls._clean_resume_list(intention.get("target_roles", []), max_items=30, item_len=120),
                "target_cities": cls._clean_resume_list(intention.get("target_cities", []), max_items=30, item_len=120),
                "salary_expectation": cls._clean_resume_text(intention.get("salary_expectation", ""), max_len=80),
            },
            "profile_summary": cls._clean_resume_text(payload.get("profile_summary", ""), max_len=3000),
            "parsed_at": cls._clean_resume_text(payload.get("parsed_at", ""), max_len=50),
        }
        return normalized

    def _insert_resume_structured_rows(self, cur, resume_id: int, user_id: str, structured_data: dict | None):
        # Legacy no-op. Structured resume data now only lives in resumes.structured_data.
        return

    def save_resume(
        self,
        user_id: str,
        filename: str,
        content: str,
        normalized_content: str = "",
        structured_data: dict | None = None,
        parser_version: str = "",
    ):
        try:
            structured_data = self._normalize_resume_structured_data(structured_data)
            normalized = (normalized_content or "").strip()
            raw_content = (content or "").strip()
            embedding_source = normalized or raw_content
            if not embedding_source:
                return False, "简历内容为空"

            intention = structured_data.get("job_intention", {}) if isinstance(structured_data, dict) else {}
            skill_text = " ".join((structured_data.get("skills") or [])[:40]) if isinstance(structured_data, dict) else ""
            role_text = " ".join((intention.get("target_roles") or [])[:8]) if isinstance(intention, dict) else ""
            city_text = " ".join((intention.get("target_cities") or [])[:8]) if isinstance(intention, dict) else ""
            profile_summary = (
                str((structured_data.get("profile_summary") or "")).strip()
                if isinstance(structured_data, dict)
                else ""
            )

            embedding_parts = [embedding_source[:10000]]
            if skill_text:
                embedding_parts.append(f"技能: {skill_text}")
            if role_text or city_text:
                embedding_parts.append(f"求职意向: {role_text} {city_text}".strip())
            if profile_summary:
                embedding_parts.append(f"摘要: {profile_summary[:500]}")
            embedding_text = "\n".join(part for part in embedding_parts if part).strip()

            embedding_text = embedding_source[:10000]
            vector = self.embed_model.embed_query(embedding_text)
            if not vector or len(vector) != config.VECTOR_DIM:
                return False, "向量生成失败"
            with self.get_cursor() as cur:
                cur.execute("DELETE FROM resumes WHERE user_id = %s", (user_id,))
                cur.execute("""
                    INSERT INTO resumes (
                        user_id,
                        filename,
                        content,
                        normalized_content,
                        structured_data,
                        parser_version,
                        embedding,
                        updated_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                    RETURNING id;
                """, (
                    user_id,
                    filename,
                    raw_content,
                    normalized or raw_content,
                    Json(_make_json_safe(structured_data)),
                    str(parser_version or ""),
                    vector,
                ))
                resume_id = cur.fetchone()[0]
                return True, f"简历上传成功 (ID: {resume_id})"
        except psycopg2.ForeignKeyViolation:
            return False, "用户不存在"
        except Exception as e:
            logger.error(f"❌ 简历保存失败: {e}", exc_info=True)
            return False, str(e)

    def get_latest_resume(self, user_id: str):
        try:
            with self.get_cursor(dict_cursor=True) as cur:
                cur.execute("""
                    SELECT id, content, normalized_content, structured_data, parser_version, filename, created_at, updated_at
                    FROM resumes
                    WHERE user_id = %s ORDER BY created_at DESC LIMIT 1
                """, (user_id,))
                return cur.fetchone()
        except Exception as e:
            logger.error(f"❌ 获取简历失败: {e}")
            return None

    def get_latest_resume_structured(self, user_id: str):
        try:
            with self.get_cursor(dict_cursor=True) as cur:
                cur.execute(
                    """
                    SELECT id, filename, parser_version, created_at, updated_at, structured_data
                    FROM resumes
                    WHERE user_id = %s
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    (user_id,),
                )
                resume = cur.fetchone()
                if not resume:
                    return None

                resume_id = int(resume["id"])
                snapshot = resume.get("structured_data") if isinstance(resume.get("structured_data"), dict) else {}

            structured = self._normalize_resume_structured_data(snapshot)

            return {
                "resume_id": resume_id,
                "filename": resume.get("filename", ""),
                "parser_version": resume.get("parser_version", ""),
                "created_at": resume.get("created_at"),
                "updated_at": resume.get("updated_at"),
                "structured_data": structured,
            }
        except Exception as e:
            logger.error("get latest resume structured failed: %s", e, exc_info=True)
            return None
