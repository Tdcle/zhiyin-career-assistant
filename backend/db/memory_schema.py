from __future__ import annotations

try:
    from utils.logger import get_logger
except ImportError:
    from backend.utils.logger import get_logger

from .memory_constants import (
    MEMORY_AUTO_TTL_DAYS,
    MEMORY_CANONICAL_FACT_KEYS,
    MEMORY_FACT_KEY_ALIASES,
    MEMORY_SINGLE_VALUE_FACT_KEYS,
    MEMORY_STALE_DEACTIVATE_DAYS,
)

logger = get_logger("database")


class MemorySchemaMixin:
    def _migrate_memory_profiles_to_facts(self, cur):
        cur.execute("SELECT to_regclass('public.memory_profiles')")
        row = cur.fetchone()
        table_name = row[0] if row else None
        if not table_name:
            return

        migrations = [
            (
                "desired_role",
                0.92,
                4,
                """
                SELECT
                    mp.user_id,
                    trim(value)::text AS fact_value,
                    mp.updated_at
                FROM memory_profiles mp
                CROSS JOIN LATERAL jsonb_array_elements_text(COALESCE(mp.desired_roles, '[]'::jsonb)) AS value
                WHERE trim(value) <> ''
                """,
            ),
            (
                "desired_city",
                0.90,
                3,
                """
                SELECT
                    mp.user_id,
                    trim(value)::text AS fact_value,
                    mp.updated_at
                FROM memory_profiles mp
                CROSS JOIN LATERAL jsonb_array_elements_text(COALESCE(mp.desired_cities, '[]'::jsonb)) AS value
                WHERE trim(value) <> ''
                """,
            ),
            (
                "desired_experience",
                0.90,
                4,
                """
                SELECT
                    user_id,
                    trim(desired_experience)::text AS fact_value,
                    updated_at
                FROM memory_profiles
                WHERE trim(COALESCE(desired_experience, '')) <> ''
                """,
            ),
            (
                "desired_salary",
                0.90,
                4,
                """
                SELECT
                    user_id,
                    (desired_salary_min::text || ' ' || desired_salary_unit)::text AS fact_value,
                    updated_at
                FROM memory_profiles
                WHERE COALESCE(desired_salary_min, 0) > 0
                  AND trim(COALESCE(desired_salary_unit, '')) <> ''
                """,
            ),
            (
                "core_skill",
                0.86,
                3,
                """
                SELECT
                    mp.user_id,
                    trim(value)::text AS fact_value,
                    mp.updated_at
                FROM memory_profiles mp
                CROSS JOIN LATERAL jsonb_array_elements_text(COALESCE(mp.core_skills, '[]'::jsonb)) AS value
                WHERE trim(value) <> ''
                """,
            ),
        ]

        migrated = 0
        for fact_key, confidence, importance, source_sql in migrations:
            cur.execute(
                f"""
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
                    meta,
                    created_at,
                    updated_at
                )
                SELECT
                    src.user_id,
                    %s,
                    src.fact_value,
                    'memory_profile',
                    TRUE,
                    %s,
                    %s,
                    NULL,
                    CURRENT_TIMESTAMP,
                    0,
                    jsonb_build_object('migrated_from', 'memory_profiles', 'polarity', 'include'),
                    COALESCE(src.updated_at, CURRENT_TIMESTAMP),
                    CURRENT_TIMESTAMP
                FROM ({source_sql}) AS src
                ON CONFLICT (user_id, fact_key, fact_value)
                DO UPDATE SET
                    is_active = TRUE,
                    confidence = GREATEST(COALESCE(memory_facts.confidence, 0.0), EXCLUDED.confidence),
                    importance = GREATEST(COALESCE(memory_facts.importance, 1), EXCLUDED.importance),
                    meta = COALESCE(memory_facts.meta, '{{}}'::jsonb) || EXCLUDED.meta,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (fact_key, confidence, importance),
            )
            migrated += max(cur.rowcount, 0)

        cur.execute("DROP TABLE IF EXISTS memory_profiles")
        logger.info("memory_profiles migrated into memory_facts: rows=%s", migrated)

    def _migrate_memory_fact_keys(self, cur):
        normalized_expr = "regexp_replace(lower(trim(fact_key)), '[\\s\\-]+', '_', 'g')"
        key_map = dict(MEMORY_FACT_KEY_ALIASES)
        for canonical_key in MEMORY_CANONICAL_FACT_KEYS:
            key_map.setdefault(canonical_key, canonical_key)
        known_normalized_keys = sorted(key_map.keys())

        migrated_count = 0
        deactivated_count = 0

        for normalized_key, target_key in key_map.items():
            cur.execute(
                f"""
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
                    meta,
                    created_at,
                    updated_at
                )
                SELECT
                    user_id,
                    %s,
                    fact_value,
                    source,
                    TRUE,
                    confidence,
                    importance,
                    expires_at,
                    last_used_at,
                    use_count,
                    COALESCE(meta, '{{}}'::jsonb)
                        || CASE
                            WHEN fact_key <> %s THEN jsonb_build_object('original_fact_key', fact_key)
                            ELSE '{{}}'::jsonb
                        END,
                    created_at,
                    CURRENT_TIMESTAMP
                FROM memory_facts
                WHERE is_active = TRUE
                  AND {normalized_expr} = %s
                  AND fact_key <> %s
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
                    last_used_at = GREATEST(
                        COALESCE(memory_facts.last_used_at, EXCLUDED.last_used_at),
                        EXCLUDED.last_used_at
                    ),
                    use_count = GREATEST(COALESCE(memory_facts.use_count, 0), EXCLUDED.use_count),
                    meta = COALESCE(memory_facts.meta, '{{}}'::jsonb) || EXCLUDED.meta,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (target_key, target_key, normalized_key, target_key),
            )
            migrated_count += max(cur.rowcount, 0)
            cur.execute(
                f"""
                UPDATE memory_facts
                SET is_active = FALSE, updated_at = CURRENT_TIMESTAMP
                WHERE is_active = TRUE
                  AND {normalized_expr} = %s
                  AND fact_key <> %s
                """,
                (normalized_key, target_key),
            )
            deactivated_count += max(cur.rowcount, 0)

        placeholders = ", ".join(["%s"] * len(known_normalized_keys))
        cur.execute(
            f"""
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
                meta,
                created_at,
                updated_at
            )
            SELECT
                user_id,
                'preference_note',
                fact_value,
                source,
                TRUE,
                confidence,
                importance,
                expires_at,
                last_used_at,
                use_count,
                COALESCE(meta, '{{}}'::jsonb) || jsonb_build_object('original_fact_key', fact_key),
                created_at,
                CURRENT_TIMESTAMP
            FROM memory_facts
            WHERE is_active = TRUE
              AND {normalized_expr} NOT IN ({placeholders})
              AND fact_key <> 'preference_note'
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
                last_used_at = GREATEST(
                    COALESCE(memory_facts.last_used_at, EXCLUDED.last_used_at),
                    EXCLUDED.last_used_at
                ),
                use_count = GREATEST(COALESCE(memory_facts.use_count, 0), EXCLUDED.use_count),
                meta = COALESCE(memory_facts.meta, '{{}}'::jsonb) || EXCLUDED.meta,
                updated_at = CURRENT_TIMESTAMP
            """,
            tuple(known_normalized_keys),
        )
        migrated_count += max(cur.rowcount, 0)
        cur.execute(
            f"""
            UPDATE memory_facts
            SET is_active = FALSE, updated_at = CURRENT_TIMESTAMP
            WHERE is_active = TRUE
              AND {normalized_expr} NOT IN ({placeholders})
              AND fact_key <> 'preference_note'
            """,
            tuple(known_normalized_keys),
        )
        deactivated_count += max(cur.rowcount, 0)

        if migrated_count or deactivated_count:
            logger.info(
                "memory fact keys normalized: migrated=%s deactivated=%s",
                migrated_count,
                deactivated_count,
            )

    def _enforce_memory_fact_policies(self, cur):
        deactivated = 0
        for fact_key in MEMORY_SINGLE_VALUE_FACT_KEYS:
            cur.execute(
                """
                WITH ranked AS (
                    SELECT
                        id,
                        ROW_NUMBER() OVER (
                            PARTITION BY user_id, fact_key
                            ORDER BY updated_at DESC, id DESC
                        ) AS rn
                    FROM memory_facts
                    WHERE is_active = TRUE
                      AND fact_key = %s
                )
                UPDATE memory_facts AS mf
                SET
                    is_active = FALSE,
                    updated_at = CURRENT_TIMESTAMP,
                    meta = COALESCE(mf.meta, '{}'::jsonb)
                        || jsonb_build_object('dedup_deactivated', TRUE)
                FROM ranked
                WHERE mf.id = ranked.id
                  AND ranked.rn > 1
                """,
                (fact_key,),
            )
            deactivated += max(cur.rowcount, 0)

        cur.execute(
            """
            UPDATE memory_facts
            SET
                is_active = FALSE,
                updated_at = CURRENT_TIMESTAMP,
                meta = COALESCE(meta, '{}'::jsonb)
                    || jsonb_build_object(
                        'invalidated_reason', 'expired',
                        'invalidated_at', CURRENT_TIMESTAMP
                    )
            WHERE is_active = TRUE
              AND expires_at IS NOT NULL
              AND expires_at <= CURRENT_TIMESTAMP
            """
        )
        deactivated += max(cur.rowcount, 0)

        mutable_keys = [key for key in MEMORY_AUTO_TTL_DAYS.keys() if key in MEMORY_CANONICAL_FACT_KEYS]
        if mutable_keys:
            placeholders = ", ".join(["%s"] * len(mutable_keys))
            cur.execute(
                f"""
                UPDATE memory_facts
                SET
                    is_active = FALSE,
                    updated_at = CURRENT_TIMESTAMP,
                    meta = COALESCE(meta, '{{}}'::jsonb)
                        || jsonb_build_object(
                            'invalidated_reason', 'stale_low_use',
                            'invalidated_at', CURRENT_TIMESTAMP,
                            'stale_days_threshold', %s
                        )
                WHERE is_active = TRUE
                  AND fact_key IN ({placeholders})
                  AND updated_at < (CURRENT_TIMESTAMP - make_interval(days => %s))
                  AND COALESCE(use_count, 0) <= 1
                  AND COALESCE(meta->>'pin', 'false') <> 'true'
                """,
                tuple([MEMORY_STALE_DEACTIVATE_DAYS, *mutable_keys, MEMORY_STALE_DEACTIVATE_DAYS]),
            )
            deactivated += max(cur.rowcount, 0)

        if deactivated:
            logger.info("memory fact policies enforced: deactivated=%s", deactivated)
