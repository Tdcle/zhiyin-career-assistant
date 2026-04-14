from __future__ import annotations

import jieba

try:
    from utils.logger import get_logger
except ImportError:
    from backend.utils.logger import get_logger

from .common import STOPWORDS, _salary_matches, _should_apply_experience_filter

logger = get_logger("database")


class SearchMixin:
    RRF_K = 60
    RRF_LOG_TOP_N = 15

    @staticmethod
    def _build_tsquery(keyword_query: str) -> str:
        """
        用 jieba 分词构建 tsquery。

        规则：
        - 有显式 OR: "Java OR Spring Boot" → java | (spring & boot)
        - 无 OR (空格分隔): "Java 后端 MySQL" → java | 后端 | mysql (默认 OR)
        """
        if not keyword_query or not keyword_query.strip():
            return ""

        has_explicit_or = " OR " in keyword_query or " or " in keyword_query or "|" in keyword_query

        if has_explicit_or:
            raw = keyword_query.replace(" OR ", "|").replace(" or ", "|")
            parts = [p.strip() for p in raw.split("|") if p.strip()]
        else:
            parts = keyword_query.strip().split()

        groups = []
        seen = set()

        for part in parts:
            if not part:
                continue
            words = jieba.lcut(part)
            tokens = []
            for w in words:
                w = w.strip().lower()
                if not w or w in STOPWORDS:
                    continue
                if len(w) == 1 and not w.isalnum():
                    continue
                tokens.append(w)
            if not tokens:
                continue

            group_key = " ".join(sorted(tokens))
            if group_key in seen:
                continue
            seen.add(group_key)

            if len(tokens) == 1:
                groups.append(tokens[0])
            else:
                groups.append("(" + " & ".join(tokens) + ")")

        if not groups:
            return ""
        return " | ".join(groups)

    def _vector_recall(self, semantic_query, where_clause, where_params, recall_n) -> list[dict]:
        if not semantic_query.strip():
            return []

        query_vector = self.embed_model.embed_query(semantic_query)
        if not query_vector:
            return []

        filter_sql = f"AND {where_clause}" if where_clause else ""

        sql = f"""
            SELECT jobs.id, jobs.job_id, jobs.title, jobs.company, jobs.industry, jobs.salary,
                   jobs.city, jobs.district, jobs.experience, jobs.degree, jobs.welfare,
                   jobs.summary, jobs.detail, jobs.detail_url,
                   1 - (jobs.embedding <=> %s::vector) AS vec_score
            FROM jobs
            LEFT JOIN job_status_checks ON job_status_checks.job_id = jobs.job_id
            WHERE jobs.embedding IS NOT NULL
              AND COALESCE(job_status_checks.status, 'unknown') IN ('active', 'unknown')
              {filter_sql}
            ORDER BY jobs.embedding <=> %s::vector
            LIMIT %s;
        """
        params = [query_vector] + where_params + [query_vector, recall_n]

        with self.get_cursor(dict_cursor=True) as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        for rank, row in enumerate(rows, start=1):
            row["vec_rank"] = rank
            row["vec_score"] = float(row["vec_score"])
        return rows

    def _bm25_recall(self, keyword_query, where_clause, where_params, recall_n) -> list[dict]:
        tsquery_str = self._build_tsquery(keyword_query)
        if not tsquery_str:
            return []

        filter_sql = f"AND {where_clause}" if where_clause else ""

        sql = f"""
            SELECT jobs.id, jobs.job_id, jobs.title, jobs.company, jobs.industry, jobs.salary,
                   jobs.city, jobs.district, jobs.experience, jobs.degree, jobs.welfare,
                   jobs.summary, jobs.detail, jobs.detail_url,
                   ts_rank_cd(jobs.tsv, to_tsquery('simple', %s), 32) AS bm25_score
            FROM jobs
            LEFT JOIN job_status_checks ON job_status_checks.job_id = jobs.job_id
            WHERE jobs.tsv @@ to_tsquery('simple', %s)
              AND COALESCE(job_status_checks.status, 'unknown') IN ('active', 'unknown')
              {filter_sql}
            ORDER BY bm25_score DESC
            LIMIT %s;
        """
        params = [tsquery_str, tsquery_str] + where_params + [recall_n]

        with self.get_cursor(dict_cursor=True) as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        for rank, row in enumerate(rows, start=1):
            row["bm25_rank"] = rank
            row["bm25_score"] = float(row["bm25_score"])
        return rows

    def _rrf_fuse(self, vector_results, bm25_results, top_k) -> list[dict]:
        k = self.RRF_K
        doc_map: dict[str, dict] = {}

        for row in vector_results:
            jid = row["job_id"]
            if jid not in doc_map:
                doc_map[jid] = dict(row)
                doc_map[jid]["rrf_score"] = 0.0
                doc_map[jid]["from_paths"] = []
            doc_map[jid]["rrf_score"] += 1.0 / (k + row["vec_rank"])
            doc_map[jid]["vec_rank"] = row["vec_rank"]
            doc_map[jid]["vec_score"] = row["vec_score"]
            doc_map[jid]["from_paths"].append("vec")

        for row in bm25_results:
            jid = row["job_id"]
            if jid not in doc_map:
                doc_map[jid] = dict(row)
                doc_map[jid]["rrf_score"] = 0.0
                doc_map[jid]["from_paths"] = []
            doc_map[jid]["rrf_score"] += 1.0 / (k + row["bm25_rank"])
            doc_map[jid]["bm25_rank"] = row["bm25_rank"]
            doc_map[jid]["bm25_score"] = row["bm25_score"]
            doc_map[jid]["from_paths"].append("bm25")

        fused = sorted(doc_map.values(), key=lambda x: x["rrf_score"], reverse=True)
        result = fused[:top_k]

        # 日志
        log_lines = []
        max_log_rows = max(1, int(getattr(self, "RRF_LOG_TOP_N", 15)))
        for idx, r in enumerate(result):
            if idx >= max_log_rows:
                log_lines.append(f"  ... truncated {len(result) - max_log_rows} more rows")
                break
            paths = "+".join(r.get("from_paths", []))
            log_lines.append(
                f"  [{r['title']}] {paths} "
                f"v={r.get('vec_rank', '-')} b={r.get('bm25_rank', '-')} "
                f"rrf={r['rrf_score']:.5f}"
            )
        logger.info(
            f"🔀 [RRF] vec={len(vector_results)} bm25={len(bm25_results)} "
            f"merged={len(doc_map)} return={len(result)}\n" + "\n".join(log_lines)
        )
        return result

    def hybrid_search(
        self,
        keyword_query: str,
        city: str = "",
        company: str = "",
        experience: str = "",
        salary_min: int = 0,
        salary_unit: str = "",
        top_k: int = 10,
        vector_recall_n: int = 200,
        bm25_recall_n: int = 200,
    ) -> list[dict]:
        try:
            logger.info(
                "hybrid search: keywords=%s city=%s company=%s experience=%s salary_min=%s salary_unit=%s",
                keyword_query,
                city,
                company,
                experience,
                salary_min,
                salary_unit,
            )

            where_parts = []
            where_params = []

            if city and city.strip():
                where_parts.append("city = %s")
                where_params.append(city.strip())

            if company and company.strip():
                where_parts.append("company ILIKE %s")
                where_params.append(f"%{company.strip()}%")

            if _should_apply_experience_filter(experience):
                exp = experience.strip()
                where_parts.append("(experience ILIKE %s OR title ILIKE %s)")
                where_params.extend([f"%{exp}%", f"%{exp}%"])

            where_clause = " AND ".join(where_parts) if where_parts else ""
            recall_boost = 80 if (experience or salary_min) else 0

            vector_results = self._vector_recall(
                keyword_query,
                where_clause,
                where_params,
                vector_recall_n + recall_boost,
            )
            bm25_results = self._bm25_recall(
                keyword_query,
                where_clause,
                where_params,
                bm25_recall_n + recall_boost,
            )

            logger.info("recall count: vec=%s bm25=%s", len(vector_results), len(bm25_results))

            if vector_results and bm25_results:
                merged_results = self._rrf_fuse(
                    vector_results,
                    bm25_results,
                    max(top_k * 3, top_k + 10),
                )
            elif vector_results:
                logger.info("hybrid search fallback: vector only")
                merged_results = vector_results
                for row in merged_results:
                    row["rrf_score"] = 1.0 / (self.RRF_K + row["vec_rank"])
                    row["from_paths"] = ["vec"]
            elif bm25_results:
                logger.info("hybrid search fallback: bm25 only")
                merged_results = bm25_results
                for row in merged_results:
                    row["rrf_score"] = 1.0 / (self.RRF_K + row["bm25_rank"])
                    row["from_paths"] = ["bm25"]
            else:
                logger.warning("hybrid search returned no recall results")
                return []

            filtered_results = [
                row
                for row in merged_results
                if _salary_matches(
                    row.get("salary", ""),
                    salary_min=salary_min,
                    salary_unit=salary_unit,
                )
            ]

            if salary_min and salary_unit:
                logger.info(
                    "salary filter kept %s/%s results",
                    len(filtered_results),
                    len(merged_results),
                )

            return filtered_results[:top_k]

        except Exception as e:
            logger.error("hybrid search failed: %s", e, exc_info=True)
            return []
