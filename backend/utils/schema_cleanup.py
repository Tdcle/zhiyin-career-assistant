import argparse
import sys
from pathlib import Path
from typing import Any

from psycopg2.extras import Json

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from db import DatabaseManager, _make_json_safe
from utils.logger import get_logger

logger = get_logger("schema_cleanup")


LEGACY_TABLES = [
    "resume_structured_profiles",
    "long_term_memories",
    "user_profiles",
]


def _table_exists(db: DatabaseManager, table_name: str) -> bool:
    with db.get_cursor(dict_cursor=True) as cur:
        cur.execute(
            """
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.tables
                WHERE table_schema = 'public' AND table_name = %s
            ) AS ok
            """,
            (table_name,),
        )
        row = cur.fetchone() or {}
        return bool(row.get("ok", False))


def _is_non_empty(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict)):
        return bool(value)
    return value is not None


def _merge_structured(preferred: dict, fallback: dict) -> dict:
    a = preferred if isinstance(preferred, dict) else {}
    b = fallback if isinstance(fallback, dict) else {}
    merged = dict(b)
    for key in ["basic_info", "job_intention"]:
        pa = a.get(key, {}) if isinstance(a.get(key), dict) else {}
        pb = b.get(key, {}) if isinstance(b.get(key), dict) else {}
        node = dict(pb)
        for sub_key, sub_value in pa.items():
            if _is_non_empty(sub_value):
                node[sub_key] = sub_value
        merged[key] = node
    for key in ["education", "work_experience", "projects", "skills", "certificates", "languages"]:
        pa = a.get(key, [])
        pb = b.get(key, [])
        merged[key] = pa if _is_non_empty(pa) else pb
    for key in ["profile_summary", "parsed_at"]:
        pa = a.get(key, "")
        pb = b.get(key, "")
        merged[key] = pa if _is_non_empty(pa) else pb
    return merged


def migrate_resume_structured_profiles(db: DatabaseManager, dry_run: bool = False) -> int:
    if not _table_exists(db, "resume_structured_profiles"):
        logger.info("legacy table resume_structured_profiles not found, skip migration")
        return 0

    migrated = 0
    with db.get_cursor(dict_cursor=True) as cur:
        cur.execute(
            """
            SELECT
                resume_id,
                basic_info,
                education,
                work_experience,
                projects,
                skills,
                certificates,
                languages,
                job_intention,
                profile_summary
            FROM resume_structured_profiles
            ORDER BY resume_id ASC
            """
        )
        rows = cur.fetchall() or []

    for row in rows:
        resume_id = int(row.get("resume_id"))
        legacy_payload = {
            "basic_info": row.get("basic_info", {}),
            "education": row.get("education", []),
            "work_experience": row.get("work_experience", []),
            "projects": row.get("projects", []),
            "skills": row.get("skills", []),
            "certificates": row.get("certificates", []),
            "languages": row.get("languages", []),
            "job_intention": row.get("job_intention", {}),
            "profile_summary": str(row.get("profile_summary", "") or ""),
        }
        legacy_norm = db._normalize_resume_structured_data(legacy_payload)

        with db.get_cursor(dict_cursor=True) as cur:
            cur.execute("SELECT structured_data FROM resumes WHERE id = %s", (resume_id,))
            resume_row = cur.fetchone() or {}

        if not resume_row:
            continue

        snapshot = resume_row.get("structured_data", {}) if isinstance(resume_row.get("structured_data"), dict) else {}
        snapshot_norm = db._normalize_resume_structured_data(snapshot)
        merged = db._normalize_resume_structured_data(_merge_structured(legacy_norm, snapshot_norm))
        if dry_run:
            migrated += 1
            continue

        with db.get_cursor() as cur:
            cur.execute(
                """
                UPDATE resumes
                SET structured_data = %s, updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
                """,
                (Json(_make_json_safe(merged)), resume_id),
            )
        migrated += 1

    logger.info("resume structured snapshot migration finished: %s rows", migrated)
    return migrated


def migrate_legacy_memory(db: DatabaseManager, dry_run: bool = False) -> dict:
    result = {"profile_rows": 0, "memory_rows": 0}

    if _table_exists(db, "user_profiles"):
        with db.get_cursor(dict_cursor=True) as cur:
            cur.execute("SELECT user_id, preferences FROM user_profiles")
            rows = cur.fetchall() or []
        for row in rows:
            user_id = str(row.get("user_id") or "").strip()
            preferences = str(row.get("preferences") or "").strip()
            if not user_id or not preferences:
                continue
            result["profile_rows"] += 1
            if dry_run:
                continue
            db.update_user_profile(user_id=user_id, preferences=preferences)

    if _table_exists(db, "long_term_memories"):
        with db.get_cursor(dict_cursor=True) as cur:
            cur.execute(
                """
                SELECT user_id, memory_type, content, source
                FROM long_term_memories
                ORDER BY id ASC
                """
            )
            rows = cur.fetchall() or []
        for row in rows:
            user_id = str(row.get("user_id") or "").strip()
            memory_type = str(row.get("memory_type") or "preference").strip()[:80] or "preference"
            content = str(row.get("content") or "").strip()
            source = str(row.get("source") or "legacy").strip()[:50] or "legacy"
            if not user_id or not content:
                continue
            result["memory_rows"] += 1
            if dry_run:
                continue
            db.add_memory_fact(user_id=user_id, fact_key=memory_type, fact_value=content, source=source)
            db.ingest_user_memory_from_text(user_id=user_id, text=content, source=source)

    logger.info("legacy memory migration finished: %s", result)
    return result


def drop_legacy_tables(db: DatabaseManager, dry_run: bool = False) -> list[str]:
    dropped = []
    for table_name in LEGACY_TABLES:
        if not _table_exists(db, table_name):
            continue
        dropped.append(table_name)
        if dry_run:
            continue
        with db.get_cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE")
    logger.info("legacy tables dropped: %s", dropped)
    return dropped


def main():
    parser = argparse.ArgumentParser(description="Merge redundant tables and drop legacy schema objects.")
    parser.add_argument("--dry-run", action="store_true", help="Only print planned operations.")
    args = parser.parse_args()

    db = DatabaseManager()
    logger.info("schema cleanup start (dry_run=%s)", args.dry_run)
    resume_rows = migrate_resume_structured_profiles(db, dry_run=args.dry_run)
    memory_rows = migrate_legacy_memory(db, dry_run=args.dry_run)
    dropped = drop_legacy_tables(db, dry_run=args.dry_run)
    logger.info(
        "schema cleanup done: resume_rows=%s memory_profile_rows=%s memory_rows=%s dropped=%s",
        resume_rows,
        memory_rows.get("profile_rows", 0),
        memory_rows.get("memory_rows", 0),
        dropped,
    )


if __name__ == "__main__":
    main()
