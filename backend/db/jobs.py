from __future__ import annotations

try:
    from config.config import config
    from utils.logger import get_logger
except ImportError:
    from backend.config.config import config
    from backend.utils.logger import get_logger

logger = get_logger("database")


class JobMixin:
    def insert_job(self, data: dict) -> bool:
        try:
            with self.get_cursor() as cur:
                cur.execute("""
                    INSERT INTO jobs
                    (job_id, title, salary, company, industry, city, district,
                     experience, degree, welfare, detail, summary, detail_url)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (job_id) DO NOTHING
                """, (
                    data["job_id"], data["title"], data["salary"], data["company"],
                    data["industry"], data["city"], data["district"],
                    data["experience"], data["degree"], data["welfare"],
                    data["detail"], data["summary"], data["detail_url"],
                ))
                if cur.rowcount == 0:
                    return False
                self._update_tsv_for_job(
                    cur, data["job_id"],
                    data.get("title"), data.get("company"),
                    data.get("welfare"), data.get("summary"), data.get("detail"),
                )
                logger.debug(f"✅ 职位入库: {data['job_id']} - {data['title']}")
                return True
        except Exception as e:
            logger.error(f"❌ 插入职位失败: {e}")
            return False

    def build_job_embedding_text(self, data: dict, summary: str = "") -> str:
        summary_text = summary or data.get("summary", "") or ""
        return (
            f"职位: {data.get('title', '')} | "
            f"地点: {data.get('city', '')} {data.get('district', '')} | "
            f"公司: {data.get('company', '')} | "
            f"薪资: {data.get('salary', '')} | "
            f"经验要求: {data.get('experience', '')} | "
            f"学历要求: {data.get('degree', '')} | "
            f"福利: {data.get('welfare', '')} | "
            f"介绍: {summary_text}"
        )

    def save_job_with_analysis(self, data: dict, summary: str, vector) -> str:
        try:
            with self.get_cursor() as cur:
                cur.execute("SELECT 1 FROM jobs WHERE job_id = %s", (data["job_id"],))
                existed = cur.fetchone() is not None

                cur.execute("""
                    INSERT INTO jobs
                    (job_id, title, salary, company, industry, city, district,
                     experience, degree, welfare, detail, summary, detail_url, embedding)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (job_id) DO UPDATE SET
                        title = EXCLUDED.title,
                        salary = EXCLUDED.salary,
                        company = EXCLUDED.company,
                        industry = EXCLUDED.industry,
                        city = EXCLUDED.city,
                        district = EXCLUDED.district,
                        experience = EXCLUDED.experience,
                        degree = EXCLUDED.degree,
                        welfare = EXCLUDED.welfare,
                        detail = EXCLUDED.detail,
                        summary = EXCLUDED.summary,
                        detail_url = EXCLUDED.detail_url,
                        embedding = EXCLUDED.embedding
                """, (
                    data["job_id"], data["title"], data["salary"], data["company"],
                    data["industry"], data["city"], data["district"],
                    data["experience"], data["degree"], data["welfare"],
                    data["detail"], summary, data["detail_url"], vector,
                ))

                self._update_tsv_for_job(
                    cur,
                    data["job_id"],
                    data.get("title"),
                    data.get("company"),
                    data.get("welfare"),
                    summary,
                    data.get("detail"),
                )
                return "updated" if existed else "inserted"
        except Exception as e:
            logger.error(f"❌ 保存职位分析结果失败: {e}", exc_info=True)
            return "failed"

    def fetch_jobs_without_embedding(self, limit=100):
        try:
            with self.get_cursor() as cur:
                cur.execute("""
                    SELECT job_id, title, company, salary, welfare, detail,
                           city, district, experience, degree
                    FROM jobs WHERE embedding IS NULL LIMIT %s
                """, (limit,))
                return cur.fetchall()
        except Exception as e:
            logger.error(f"❌ 获取未向量化职位失败: {e}")
            return []

    def fetch_jobs_pending_analysis(self, limit=100):
        try:
            with self.get_cursor(dict_cursor=True) as cur:
                cur.execute("""
                    SELECT job_id, title, salary, company, industry, city, district,
                           experience, degree, welfare, detail, summary, detail_url
                    FROM jobs
                    WHERE embedding IS NULL
                       OR summary IS NULL
                       OR summary = ''
                    ORDER BY create_time DESC
                    LIMIT %s
                """, (limit,))
                return cur.fetchall()
        except Exception as e:
            logger.error(f"❌ 获取待补全分析的职位失败: {e}", exc_info=True)
            return []

    def update_job_analysis(self, job_id: str, summary: str, vector) -> bool:
        try:
            with self.get_cursor(dict_cursor=True) as cur:
                cur.execute(
                    "SELECT title, company, welfare, detail FROM jobs WHERE job_id = %s",
                    (job_id,),
                )
                row = cur.fetchone()
                if not row:
                    return False
                cur.execute(
                    "UPDATE jobs SET summary = %s, embedding = %s WHERE job_id = %s",
                    (summary, vector, job_id),
                )
                self._update_tsv_for_job(
                    cur, job_id,
                    row["title"], row["company"],
                    row["welfare"], summary, row["detail"],
                )
                return True
        except Exception as e:
            logger.error(f"❌ 更新分析数据失败: {e}")
            return False

    def update_embedding(self, job_id: str, text: str) -> bool:
        try:
            vector = self.embed_model.embed_query(text)
            if not vector or len(vector) != config.VECTOR_DIM:
                return False
            with self.get_cursor() as cur:
                cur.execute("UPDATE jobs SET embedding = %s WHERE job_id = %s", (vector, job_id))
                return cur.rowcount > 0
        except Exception as e:
            logger.error(f"❌ 向量更新失败: {e}")
            return False

    def get_job_details(self, job_id):
        try:
            with self.get_cursor(dict_cursor=True) as cur:
                cur.execute("SELECT * FROM jobs WHERE job_id = %s", (str(job_id),))
                result = cur.fetchone()
                if not result and str(job_id).isdigit():
                    cur.execute("SELECT * FROM jobs WHERE id = %s", (int(job_id),))
                    result = cur.fetchone()
                return result
        except Exception as e:
            logger.error(f"❌ 获取职位详情失败: {e}")
            return None

    def get_jobs_by_ids(self, job_ids: list[str]) -> dict[str, dict]:
        normalized_ids: list[str] = []
        seen: set[str] = set()
        for raw_job_id in job_ids or []:
            job_id = str(raw_job_id).strip()
            if not job_id or job_id in seen:
                continue
            seen.add(job_id)
            normalized_ids.append(job_id)

        if not normalized_ids:
            return {}

        try:
            with self.get_cursor(dict_cursor=True) as cur:
                cur.execute(
                    """
                    SELECT job_id, title, company, salary, city, district, experience, degree, welfare, summary, detail, detail_url
                    FROM jobs
                    WHERE job_id = ANY(%s)
                    """,
                    (normalized_ids,),
                )
                rows = cur.fetchall() or []
            return {
                str(row.get("job_id", "")).strip(): row
                for row in rows
                if str(row.get("job_id", "")).strip()
            }
        except Exception as e:
            logger.error("get jobs by ids failed: %s", e, exc_info=True)
            return {}

    def get_job_detail_urls(self, job_ids: list[str]) -> dict[str, str]:
        normalized_ids = [str(job_id).strip() for job_id in (job_ids or []) if str(job_id).strip()]
        if not normalized_ids:
            return {}

        try:
            with self.get_cursor(dict_cursor=True) as cur:
                cur.execute(
                    """
                    SELECT job_id, detail_url
                    FROM jobs
                    WHERE job_id = ANY(%s)
                    """,
                    (normalized_ids,),
                )
                rows = cur.fetchall() or []
            return {
                str(row.get("job_id", "")).strip(): str(row.get("detail_url", "") or "").strip()
                for row in rows
                if str(row.get("job_id", "")).strip()
            }
        except Exception as e:
            logger.error("get job detail urls failed: %s", e, exc_info=True)
            return {}

    def get_job_status_checks(self, job_ids: list[str]) -> dict[str, dict]:
        normalized_ids = [str(job_id).strip() for job_id in (job_ids or []) if str(job_id).strip()]
        if not normalized_ids:
            return {}

        try:
            with self.get_cursor(dict_cursor=True) as cur:
                cur.execute(
                    """
                    SELECT job_id, status, reason, matched_keyword, detail_url, final_url, text_preview, checked_at, updated_at
                    FROM job_status_checks
                    WHERE job_id = ANY(%s)
                    """,
                    (normalized_ids,),
                )
                rows = cur.fetchall() or []
            result = {}
            for row in rows:
                key = str(row.get("job_id", "")).strip()
                if key:
                    result[key] = row
            return result
        except Exception as e:
            logger.error("get job status checks failed: %s", e, exc_info=True)
            return {}

    def upsert_job_status_check(
        self,
        job_id: str,
        status: str,
        reason: str = "",
        matched_keyword: str = "",
        detail_url: str = "",
        final_url: str = "",
        text_preview: str = "",
    ) -> bool:
        normalized_job_id = str(job_id or "").strip()
        if not normalized_job_id:
            return False

        normalized_status = str(status or "").strip().lower()
        if normalized_status not in {"active", "closed", "unknown"}:
            normalized_status = "unknown"

        previous_status = "unknown"
        visibility_changed = False
        try:
            with self.get_cursor() as cur:
                cur.execute(
                    "SELECT status FROM job_status_checks WHERE job_id = %s",
                    (normalized_job_id,),
                )
                row = cur.fetchone()
                if row and len(row) > 0 and row[0]:
                    previous_status = str(row[0]).strip().lower() or "unknown"

                was_visible = previous_status != "closed"
                is_visible = normalized_status != "closed"
                visibility_changed = was_visible != is_visible

                cur.execute(
                    """
                    INSERT INTO job_status_checks (
                        job_id,
                        status,
                        reason,
                        matched_keyword,
                        detail_url,
                        final_url,
                        text_preview,
                        checked_at,
                        updated_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    ON CONFLICT (job_id) DO UPDATE SET
                        status = EXCLUDED.status,
                        reason = EXCLUDED.reason,
                        matched_keyword = EXCLUDED.matched_keyword,
                        detail_url = EXCLUDED.detail_url,
                        final_url = EXCLUDED.final_url,
                        text_preview = EXCLUDED.text_preview,
                        checked_at = CURRENT_TIMESTAMP,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (
                        normalized_job_id,
                        normalized_status,
                        str(reason or "")[:100],
                        str(matched_keyword or "")[:100],
                        str(detail_url or "")[:500],
                        str(final_url or "")[:1000],
                        str(text_preview or "")[:1000],
                    ),
                )
            if visibility_changed:
                try:
                    from utils.search_cache import bump_search_data_version

                    bump_search_data_version(
                        reason=f"job_status_visibility:{previous_status}->{normalized_status}"
                    )
                except Exception:
                    logger.warning("bump search data version on job status change failed", exc_info=True)
            return True
        except Exception as e:
            logger.error("upsert job status check failed: %s", e, exc_info=True)
            return False
