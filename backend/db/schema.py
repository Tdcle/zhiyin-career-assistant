from __future__ import annotations

try:
    from config.config import config
    from utils.logger import get_logger
except ImportError:
    from backend.config.config import config
    from backend.utils.logger import get_logger

from .common import _build_tsv_sql_and_params

logger = get_logger("database")


class SchemaMixin:
    def _init_tables(self):
        with self.get_cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")

            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS jobs (
                    id            SERIAL PRIMARY KEY,
                    job_id        VARCHAR(100) NOT NULL UNIQUE,
                    title         VARCHAR(255),
                    salary        VARCHAR(100),
                    company       VARCHAR(255),
                    industry      VARCHAR(100),
                    city          VARCHAR(50),
                    district      VARCHAR(50),
                    experience    VARCHAR(50),
                    degree        VARCHAR(50),
                    welfare       TEXT,
                    detail        TEXT,
                    summary       TEXT,
                    detail_url    VARCHAR(500),
                    embedding     vector({config.VECTOR_DIM}),
                    tsv           tsvector,
                    create_time   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            self._add_tsv_column_if_not_exists(cur)

            cur.execute("""
                CREATE OR REPLACE FUNCTION jobs_tsv_trigger_fn()
                RETURNS trigger AS $$
                BEGIN
                    IF NEW.tsv IS NOT NULL THEN
                        RETURN NEW;
                    END IF;
                    NEW.tsv :=
                        setweight(to_tsvector('simple', COALESCE(NEW.title, '')),    'A') ||
                        setweight(to_tsvector('simple', COALESCE(NEW.company, '')),  'B') ||
                        setweight(to_tsvector('simple', COALESCE(NEW.welfare, '')),  'C') ||
                        setweight(to_tsvector('simple', COALESCE(NEW.summary, '')),  'C') ||
                        setweight(to_tsvector('simple', COALESCE(NEW.detail, '')),   'D');
                    RETURN NEW;
                END;
                $$ LANGUAGE plpgsql;
            """)

            cur.execute("""
                DROP TRIGGER IF EXISTS trg_jobs_tsv ON jobs;
                CREATE TRIGGER trg_jobs_tsv
                    BEFORE INSERT OR UPDATE ON jobs
                    FOR EACH ROW EXECUTE FUNCTION jobs_tsv_trigger_fn();
            """)

            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_jobs_embedding_ivfflat
                ON jobs USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_jobs_tsv_gin ON jobs USING gin (tsv);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_jobs_city ON jobs (city);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_jobs_job_id ON jobs (job_id);")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS job_status_checks (
                    job_id VARCHAR(100) PRIMARY KEY REFERENCES jobs(job_id) ON DELETE CASCADE,
                    status VARCHAR(20) NOT NULL DEFAULT 'unknown'
                        CHECK (status IN ('active', 'closed', 'unknown')),
                    reason VARCHAR(100) NOT NULL DEFAULT '',
                    matched_keyword VARCHAR(100) NOT NULL DEFAULT '',
                    detail_url TEXT NOT NULL DEFAULT '',
                    final_url TEXT NOT NULL DEFAULT '',
                    text_preview TEXT NOT NULL DEFAULT '',
                    checked_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_job_status_checks_status ON job_status_checks (status);"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_job_status_checks_checked_at ON job_status_checks (checked_at DESC);"
            )

            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id VARCHAR(50) PRIMARY KEY,
                    username VARCHAR(100) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS auth_users (
                    user_id VARCHAR(50) PRIMARY KEY REFERENCES users(user_id) ON DELETE CASCADE,
                    password_hash VARCHAR(255) NOT NULL,
                    is_active BOOLEAN NOT NULL DEFAULT TRUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_facts (
                    id SERIAL PRIMARY KEY,
                    user_id VARCHAR(50) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
                    fact_key VARCHAR(80) NOT NULL,
                    fact_value TEXT NOT NULL,
                    source VARCHAR(50) NOT NULL DEFAULT 'chat',
                    is_active BOOLEAN NOT NULL DEFAULT TRUE,
                    confidence REAL NOT NULL DEFAULT 0.75,
                    importance INTEGER NOT NULL DEFAULT 3,
                    expires_at TIMESTAMP NULL,
                    last_used_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    use_count INTEGER NOT NULL DEFAULT 0,
                    meta JSONB NOT NULL DEFAULT '{}'::jsonb,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE (user_id, fact_key, fact_value)
                );
                """
            )
            cur.execute(
                "ALTER TABLE memory_facts ADD COLUMN IF NOT EXISTS confidence REAL NOT NULL DEFAULT 0.75;"
            )
            cur.execute(
                "ALTER TABLE memory_facts ADD COLUMN IF NOT EXISTS importance INTEGER NOT NULL DEFAULT 3;"
            )
            cur.execute(
                "ALTER TABLE memory_facts ADD COLUMN IF NOT EXISTS expires_at TIMESTAMP NULL;"
            )
            cur.execute(
                "ALTER TABLE memory_facts ADD COLUMN IF NOT EXISTS last_used_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP;"
            )
            cur.execute(
                "ALTER TABLE memory_facts ADD COLUMN IF NOT EXISTS use_count INTEGER NOT NULL DEFAULT 0;"
            )
            cur.execute(
                "ALTER TABLE memory_facts ADD COLUMN IF NOT EXISTS meta JSONB NOT NULL DEFAULT '{}'::jsonb;"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_memory_facts_user_updated ON memory_facts (user_id, updated_at DESC);"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_memory_facts_user_active ON memory_facts (user_id, is_active, updated_at DESC);"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_memory_facts_user_importance ON memory_facts (user_id, importance DESC, updated_at DESC);"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_memory_facts_expires ON memory_facts (expires_at);"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_memory_facts_user_key_active ON memory_facts (user_id, fact_key, is_active, updated_at DESC);"
            )
            self._migrate_memory_profiles_to_facts(cur)
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS resumes (
                    id SERIAL PRIMARY KEY,
                    user_id VARCHAR(50) NOT NULL REFERENCES users(user_id),
                    filename VARCHAR(255),
                    content TEXT,
                    normalized_content TEXT,
                    structured_data JSONB DEFAULT '{{}}'::jsonb,
                    parser_version VARCHAR(50) DEFAULT '',
                    embedding vector({config.VECTOR_DIM}),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            cur.execute("ALTER TABLE resumes ADD COLUMN IF NOT EXISTS normalized_content TEXT;")
            cur.execute("ALTER TABLE resumes ADD COLUMN IF NOT EXISTS structured_data JSONB DEFAULT '{}'::jsonb;")
            cur.execute("ALTER TABLE resumes ADD COLUMN IF NOT EXISTS parser_version VARCHAR(50) DEFAULT '';")
            cur.execute("ALTER TABLE resumes ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_resumes_user_id ON resumes (user_id);")
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_resumes_user_created_at ON resumes (user_id, created_at DESC);"
            )
            cur.execute("""
                CREATE TABLE IF NOT EXISTS conversation_states (
                    thread_id VARCHAR(100) PRIMARY KEY,
                    user_id VARCHAR(50) NOT NULL REFERENCES users(user_id),
                    scene VARCHAR(30) NOT NULL,
                    summary TEXT DEFAULT '',
                    recent_messages JSONB DEFAULT '[]'::jsonb,
                    extra_state JSONB DEFAULT '{}'::jsonb,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_conversation_states_user_id ON conversation_states (user_id, scene, updated_at DESC);")

            self._migrate_memory_fact_keys(cur)
            self._enforce_memory_fact_policies(cur)

        logger.info("database schema and indexes ready")

    def _add_tsv_column_if_not_exists(self, cur):
        cur.execute("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'jobs' AND column_name = 'tsv'
        """)
        if not cur.fetchone():
            logger.info("🔄 旧表升级：添加 tsv 列...")
            cur.execute("ALTER TABLE jobs ADD COLUMN tsv tsvector")
            logger.info("✅ tsv 列添加成功")

    def _update_tsv_for_job(self, cur, job_id, title, company, welfare, summary, detail):
        segs = _build_tsv_sql_and_params(title, company, welfare, summary, detail)
        cur.execute("""
            UPDATE jobs SET tsv =
                setweight(to_tsvector('simple', %s), 'A') ||
                setweight(to_tsvector('simple', %s), 'B') ||
                setweight(to_tsvector('simple', %s), 'C') ||
                setweight(to_tsvector('simple', %s), 'C') ||
                setweight(to_tsvector('simple', %s), 'D')
            WHERE job_id = %s
        """, (*segs, job_id))

    def backfill_tsv(self, batch_size=200):
        total = 0
        last_id = 0
        while True:
            with self.get_cursor(dict_cursor=True) as cur:
                cur.execute("""
                    SELECT id, job_id, title, company, welfare, summary, detail
                    FROM jobs WHERE id > %s ORDER BY id LIMIT %s
                """, (last_id, batch_size))
                rows = cur.fetchall()
                if not rows:
                    break
                for row in rows:
                    self._update_tsv_for_job(
                        cur, row["job_id"],
                        row["title"], row["company"],
                        row["welfare"], row["summary"], row["detail"],
                    )
                last_id = rows[-1]["id"]
                total += len(rows)
            logger.info(f"🔄 tsv 回填: {total} 条")
        logger.info(f"✅ tsv 回填完成: {total} 条")
        return total
