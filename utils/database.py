# utils/database.py

import logging
import re
from contextlib import contextmanager
from datetime import date, datetime

import psycopg2
import psycopg2.pool
from psycopg2.extras import Json, RealDictCursor

import jieba
from config.config import config
from utils.logger import get_logger

logger = get_logger("database")


# ====================================================================
#  jieba 初始化 + 分词工具
# ====================================================================

_CUSTOM_WORDS = [
    "Python", "Java", "JavaScript", "TypeScript", "Golang", "C++", "C#",
    "SpringBoot", "Spring Boot", "Django", "Flask", "FastAPI",
    "Vue.js", "React", "Angular", "Node.js", "MyBatis",
    "MySQL", "PostgreSQL", "Redis", "MongoDB", "Elasticsearch",
    "Docker", "Kubernetes", "K8S", "Jenkins", "Git",
    "微服务", "中间件", "消息队列", "分布式", "大数据",
    "机器学习", "深度学习", "自然语言处理", "计算机视觉",
    "小程序", "公众号",
    "后端", "前端", "全栈", "后端开发", "前端开发", "全栈开发",
    "算法工程师", "数据分析师", "产品经理", "项目经理",
    "运维工程师", "测试工程师", "架构师",
    "UI设计师", "交互设计师",
    "五险一金", "六险一金", "带薪年假", "年终奖",
    "股票期权", "定期体检", "节日福利", "餐饮补贴",
    "交通补贴", "住房补贴", "弹性工作", "远程办公",
    "本科", "硕士", "博士", "大专", "研究生",
    "双休", "单休", "大小周",
    "岗位职责", "岗位要求", "任职要求", "工作职责",
    "五险", "一金", "公积金", "社保",
]

for _w in _CUSTOM_WORDS:
    jieba.add_word(_w)
jieba.lcut("预热")

STOPWORDS = frozenset({
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人",
    "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去",
    "你", "会", "着", "没有", "看", "好", "自己", "这", "他", "她",
    "它", "们", "那", "些", "么", "什么", "及", "与", "或", "等",
    "能", "可以", "需要", "进行", "通过", "使用", "以及", "对于",
    "其他", "相关", "具有", "以上", "包括", "负责", "参与",
    "了解", "熟悉", "掌握", "熟练", "优先", "优秀", "良好", "较强",
    "，", "。", "！", "？", "、", "；", "：", "（", "）",
    "(", ")", ",", ".", "!", "?", ";", ":", "/", "\\",
    "-", "—", "·", "~", "…", "\"", "'", " ",
    "具备", "能够", "能力", "工作", "项目", "经验",
    "年", "月", "日", "个", "位", "名",
})


def segment_text(text: str) -> str:
    if not text:
        return ""
    words = jieba.lcut(text)
    tokens = []
    for w in words:
        w = w.strip()
        if not w or w in STOPWORDS:
            continue
        if len(w) == 1 and not w.isalnum():
            continue
        tokens.append(w.lower())
    return " ".join(tokens)


def segment_welfare(welfare: str) -> str:
    if not welfare:
        return ""
    parts = re.split(r'[,，、；;\s]+', welfare)
    all_tokens = []
    for part in parts:
        part = part.strip()
        if part:
            seg = segment_text(part)
            if seg:
                all_tokens.append(seg)
    return " ".join(all_tokens)


def _build_tsv_sql_and_params(title, company, welfare, summary, detail):
    return (
        segment_text(title or ""),
        segment_text(company or ""),
        segment_welfare(welfare or ""),
        segment_text(summary or ""),
        segment_text(detail or ""),
    )


def _parse_salary_info(salary_text: str) -> dict:
    text = (salary_text or "").strip().lower()
    if not text:
        return {"unit": "", "min": 0.0, "max": 0.0}

    nums = [float(item) for item in re.findall(r"\d+(?:\.\d+)?", text)]
    if not nums:
        return {"unit": "", "min": 0.0, "max": 0.0}

    low = nums[0]
    high = nums[1] if len(nums) > 1 else nums[0]

    if "元/天" in text or "元/日" in text:
        return {"unit": "yuan_day", "min": low, "max": high}
    if "k" in text:
        return {"unit": "k_month", "min": low, "max": high}
    return {"unit": "", "min": low, "max": high}


def _salary_matches(salary_text: str, salary_min: int = 0, salary_unit: str = "") -> bool:
    if not salary_min or not salary_unit:
        return True
    parsed = _parse_salary_info(salary_text)
    if parsed["unit"] != salary_unit:
        return False
    return parsed["max"] >= float(salary_min)


def _should_apply_experience_filter(experience: str) -> bool:
    exp = (experience or "").strip()
    if not exp:
        return False
    if exp in {"实习", "应届"}:
        return False
    return True


def _make_json_safe(value):
    if isinstance(value, dict):
        return {str(k): _make_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_make_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_make_json_safe(item) for item in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def _parse_salary_info(salary_text: str) -> dict:
    text = (salary_text or "").strip().lower()
    if not text:
        return {"unit": "", "min": 0.0, "max": 0.0}

    nums = [float(item) for item in re.findall(r"\d+(?:\.\d+)?", text)]
    if not nums:
        return {"unit": "", "min": 0.0, "max": 0.0}

    low = nums[0]
    high = nums[1] if len(nums) > 1 else nums[0]

    if any(token in text for token in ["元/天", "元每天", "/day", "per day", "day"]):
        return {"unit": "yuan_day", "min": low, "max": high}
    if any(token in text for token in ["k", "千/月", "月", "薪"]):
        return {"unit": "k_month", "min": low, "max": high}
    return {"unit": "", "min": low, "max": high}


def _salary_matches(salary_text: str, salary_min: int = 0, salary_unit: str = "") -> bool:
    if not salary_min or not salary_unit:
        return True
    parsed = _parse_salary_info(salary_text)
    if parsed["unit"] != salary_unit:
        return False
    return parsed["max"] >= float(salary_min)


# ====================================================================
#  数据库管理器
# ====================================================================

class DatabaseManager:

    _instance = None
    _initialized = False
    RRF_K = 60

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if DatabaseManager._initialized:
            return
        DatabaseManager._initialized = True

        try:
            self._pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=2, maxconn=10, **config.DB_CONFIG,
            )
            logger.info(f"✅ 数据库连接池创建成功: {config.DB_CONFIG['dbname']}")
        except Exception as e:
            logger.critical(f"❌ 数据库连接池创建失败: {e}")
            raise

        logger.info("loading embedding model: %s", config.OLLAMA_MODELS.embedding)
        self.embed_model = config.create_embeddings()
        logger.info("embedding model loaded: %s", config.OLLAMA_MODELS.embedding)

        self._init_tables()
        self._seed_default_users()

    # ================================================================
    #                    上下文管理器
    # ================================================================

    @contextmanager
    def get_cursor(self, dict_cursor=False):
        conn = None
        try:
            conn = self._pool.getconn()
            cursor_factory = RealDictCursor if dict_cursor else None
            with conn.cursor(cursor_factory=cursor_factory) as cur:
                yield cur
                conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"❌ 数据库操作失败，已回滚: {e}")
            raise
        finally:
            if conn:
                self._pool.putconn(conn)

    # ================================================================
    #                    表结构初始化
    # ================================================================

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

            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id VARCHAR(50) PRIMARY KEY,
                    username VARCHAR(100) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id VARCHAR(50) PRIMARY KEY REFERENCES users(user_id),
                    preferences TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS long_term_memories (
                    id SERIAL PRIMARY KEY,
                    user_id VARCHAR(50) NOT NULL REFERENCES users(user_id),
                    memory_type VARCHAR(50) NOT NULL DEFAULT 'preference',
                    content TEXT NOT NULL,
                    source VARCHAR(50) NOT NULL DEFAULT 'chat',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE (user_id, memory_type, content)
                );
            """)
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS resumes (
                    id SERIAL PRIMARY KEY,
                    user_id VARCHAR(50) NOT NULL REFERENCES users(user_id),
                    filename VARCHAR(255),
                    content TEXT,
                    embedding vector({config.VECTOR_DIM}),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_resumes_user_id ON resumes (user_id);")
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
            cur.execute("CREATE INDEX IF NOT EXISTS idx_long_term_memories_user_id ON long_term_memories (user_id, updated_at DESC);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_conversation_states_user_id ON conversation_states (user_id, scene, updated_at DESC);")

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

    # ================================================================
    #  tsv 操作工具
    # ================================================================

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

    # ================================================================
    #                    种子数据
    # ================================================================

    def _seed_default_users(self):
        with self.get_cursor() as cur:
            cur.execute("SELECT count(*) FROM users")
            if cur.fetchone()[0] == 0:
                cur.execute(
                    "INSERT INTO users (user_id, username) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                    ("admin", "管理员"),
                )
                logger.info("default user ensured")

    # ================================================================
    #                    用户管理
    # ================================================================

    def _generate_next_user_id(self, cur):
        cur.execute(r"SELECT user_id FROM users WHERE user_id ~ '^\d+$'")
        rows = cur.fetchall()
        if not rows:
            return "00001"
        max_id = 0
        for row in rows:
            try:
                uid = int(row[0]) if not isinstance(row, dict) else int(row["user_id"])
                max_id = max(max_id, uid)
            except (ValueError, TypeError):
                continue
        return f"{max_id + 1:05d}"

    def create_user(self, username: str):
        if not username or not username.strip():
            return False, "用户名不能为空"
        try:
            with self.get_cursor() as cur:
                new_user_id = self._generate_next_user_id(cur)
                cur.execute(
                    "INSERT INTO users (user_id, username) VALUES (%s, %s)",
                    (new_user_id, username.strip()),
                )
                logger.info(f"✅ 用户创建成功: ID={new_user_id}, 昵称={username}")
                return True, f"用户创建成功! ID: {new_user_id} | 昵称: {username}"
        except psycopg2.IntegrityError:
            return False, "创建失败: 用户已存在"
        except Exception as e:
            logger.error(f"❌ 用户创建失败: {e}", exc_info=True)
            return False, f"创建失败: {str(e)}"

    def get_all_users_list(self):
        try:
            with self.get_cursor() as cur:
                cur.execute("SELECT user_id, username FROM users ORDER BY user_id DESC")
                return [f"{u[0]} ({u[1]})" for u in cur.fetchall()]
        except Exception as e:
            logger.error(f"❌ 获取用户列表失败: {e}")
            return []

    # ================================================================
    #                    画像管理
    # ================================================================

    def get_user_profile(self, user_id: str) -> str:
        try:
            memories = self.get_long_term_memories(user_id)
            if memories:
                return "；".join(memory["content"] for memory in reversed(memories))
            with self.get_cursor(dict_cursor=True) as cur:
                cur.execute("SELECT preferences FROM user_profiles WHERE user_id = %s", (user_id,))
                res = cur.fetchone()
                return res["preferences"] if res else ""
        except Exception as e:
            logger.error(f"❌ 获取用户画像失败: {e}")
            return ""

    def update_user_profile(self, user_id: str, preferences: str) -> bool:
        try:
            with self.get_cursor() as cur:
                cur.execute("""
                    INSERT INTO user_profiles (user_id, preferences) VALUES (%s, %s)
                    ON CONFLICT (user_id)
                    DO UPDATE SET preferences = EXCLUDED.preferences, updated_at = CURRENT_TIMESTAMP;
                """, (user_id, preferences))
                return True
        except Exception as e:
            logger.error(f"❌ 更新画像失败: {e}", exc_info=True)
            return False

    def get_long_term_memories(self, user_id: str, limit: int = 20) -> list[dict]:
        try:
            with self.get_cursor(dict_cursor=True) as cur:
                cur.execute("""
                    SELECT memory_type, content, source, updated_at
                    FROM long_term_memories
                    WHERE user_id = %s
                    ORDER BY updated_at DESC, id DESC
                    LIMIT %s
                """, (user_id, limit))
                return cur.fetchall() or []
        except Exception as e:
            logger.error(f"❌ 获取长期记忆失败: {e}", exc_info=True)
            return []

    def add_long_term_memory(
        self,
        user_id: str,
        content: str,
        memory_type: str = "preference",
        source: str = "chat",
    ) -> bool:
        normalized_content = (content or "").strip()
        if not normalized_content:
            return False

        try:
            with self.get_cursor() as cur:
                cur.execute("""
                    INSERT INTO long_term_memories (user_id, memory_type, content, source)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (user_id, memory_type, content)
                    DO UPDATE SET source = EXCLUDED.source, updated_at = CURRENT_TIMESTAMP;
                """, (user_id, memory_type, normalized_content, source))
            self.sync_user_profile_from_memories(user_id)
            return True
        except Exception as e:
            logger.error(f"❌ 保存长期记忆失败: {e}", exc_info=True)
            return False

    def sync_user_profile_from_memories(self, user_id: str) -> bool:
        memories = self.get_long_term_memories(user_id, limit=50)
        preferences = "；".join(memory["content"] for memory in reversed(memories))
        return self.update_user_profile(user_id, preferences)

    def get_conversation_state(self, thread_id: str):
        try:
            with self.get_cursor(dict_cursor=True) as cur:
                cur.execute("""
                    SELECT thread_id, user_id, scene, summary, recent_messages, extra_state, updated_at
                    FROM conversation_states
                    WHERE thread_id = %s
                """, (thread_id,))
                return cur.fetchone()
        except Exception as e:
            logger.error(f"❌ 获取会话状态失败: {e}", exc_info=True)
            return None

    def upsert_conversation_state(
        self,
        thread_id: str,
        user_id: str,
        scene: str,
        summary: str,
        recent_messages: list[dict],
        extra_state: dict | None = None,
    ) -> bool:
        try:
            with self.get_cursor() as cur:
                cur.execute("""
                    INSERT INTO conversation_states
                        (thread_id, user_id, scene, summary, recent_messages, extra_state, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (thread_id)
                    DO UPDATE SET
                        user_id = EXCLUDED.user_id,
                        scene = EXCLUDED.scene,
                        summary = EXCLUDED.summary,
                        recent_messages = EXCLUDED.recent_messages,
                        extra_state = EXCLUDED.extra_state,
                        updated_at = CURRENT_TIMESTAMP;
                """, (
                    thread_id,
                    user_id,
                    scene,
                    summary or "",
                    Json(_make_json_safe(recent_messages or [])),
                    Json(_make_json_safe(extra_state or {})),
                ))
                return True
        except Exception as e:
            logger.error(f"❌ 保存会话状态失败: {e}", exc_info=True)
            return False

    def delete_conversation_state(self, thread_id: str) -> bool:
        try:
            with self.get_cursor() as cur:
                cur.execute("DELETE FROM conversation_states WHERE thread_id = %s", (thread_id,))
                return True
        except Exception as e:
            logger.error(f"❌ 删除会话状态失败: {e}", exc_info=True)
            return False

    # ================================================================
    #                    简历管理
    # ================================================================

    def save_resume(self, user_id: str, filename: str, content: str):
        try:
            vector = self.embed_model.embed_query(content)
            if not vector or len(vector) != config.VECTOR_DIM:
                return False, "向量生成失败"
            with self.get_cursor() as cur:
                cur.execute("""
                    INSERT INTO resumes (user_id, filename, content, embedding)
                    VALUES (%s, %s, %s, %s) RETURNING id;
                """, (user_id, filename, content, vector))
                resume_id = cur.fetchone()[0]
                return True, f"简历已入库 (ID: {resume_id})"
        except psycopg2.ForeignKeyViolation:
            return False, "用户不存在"
        except Exception as e:
            logger.error(f"❌ 简历保存失败: {e}", exc_info=True)
            return False, str(e)

    def get_latest_resume(self, user_id: str):
        try:
            with self.get_cursor(dict_cursor=True) as cur:
                cur.execute("""
                    SELECT content, filename, created_at FROM resumes
                    WHERE user_id = %s ORDER BY created_at DESC LIMIT 1
                """, (user_id,))
                return cur.fetchone()
        except Exception as e:
            logger.error(f"❌ 获取简历失败: {e}")
            return None

    # ================================================================
    #                    职位数据接口
    # ================================================================

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

    # ================================================================
    #              tsquery 构建
    # ================================================================

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

    # ================================================================
    #              向量召回
    # ================================================================

    def _vector_recall(self, semantic_query, where_clause, where_params, recall_n) -> list[dict]:
        if not semantic_query.strip():
            return []

        query_vector = self.embed_model.embed_query(semantic_query)
        if not query_vector:
            return []

        filter_sql = f"AND {where_clause}" if where_clause else ""

        sql = f"""
            SELECT id, job_id, title, company, industry, salary,
                   city, district, experience, degree, welfare,
                   summary, detail, detail_url,
                   1 - (embedding <=> %s::vector) AS vec_score
            FROM jobs
            WHERE embedding IS NOT NULL {filter_sql}
            ORDER BY embedding <=> %s::vector
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

    # ================================================================
    #              BM25 召回
    # ================================================================

    def _bm25_recall(self, keyword_query, where_clause, where_params, recall_n) -> list[dict]:
        tsquery_str = self._build_tsquery(keyword_query)
        if not tsquery_str:
            return []

        filter_sql = f"AND {where_clause}" if where_clause else ""

        sql = f"""
            SELECT id, job_id, title, company, industry, salary,
                   city, district, experience, degree, welfare,
                   summary, detail, detail_url,
                   ts_rank_cd(tsv, to_tsquery('simple', %s), 32) AS bm25_score
            FROM jobs
            WHERE tsv @@ to_tsquery('simple', %s)
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

    # ================================================================
    #              RRF 融合
    # ================================================================

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
        for r in result:
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

    # ================================================================
    #              主入口：混合检索
    # ================================================================

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
        """
        混合检索主入口，大模型工具直接调用。

        参数:
            keyword_query:    搜索关键词，空格分隔默认 OR
                              同时用于向量检索（作为语义文本）和 BM25 检索
                              例: "Java 后端 Spring Boot MySQL 双休"
            city:             城市过滤（精确匹配），为空则不限
            company:          公司过滤（模糊匹配），为空则不限
                              有值时自动走公司精确通道
            top_k:            返回条数
            vector_recall_n:  向量路召回数
            bm25_recall_n:    BM25 路召回数

        返回:
            list[dict]: 每条含 rrf_score, vec_rank, bm25_rank, from_paths 等
        """
        try:
            logger.info(
                f"🔍 [混合检索] keywords='{keyword_query}' city='{city}' company='{company}'"
            )

            # ── 构建 WHERE 子句 ──
            where_parts = []
            where_params = []

            if city and city.strip():
                where_parts.append("city = %s")
                where_params.append(city.strip())

            if company and company.strip():
                where_parts.append("company ILIKE %s")
                where_params.append(f"%{company.strip()}%")

            where_clause = " AND ".join(where_parts) if where_parts else ""

            # ── 双路召回 ──
            # 向量路：keyword_query 直接作为语义查询文本
            vector_results = self._vector_recall(
                keyword_query, where_clause, where_params, vector_recall_n
            )

            # BM25 路
            bm25_results = self._bm25_recall(
                keyword_query, where_clause, where_params, bm25_recall_n
            )

            logger.info(f"📊 [召回] vec={len(vector_results)} bm25={len(bm25_results)}")

            # ── 融合 ──
            if vector_results and bm25_results:
                return self._rrf_fuse(vector_results, bm25_results, top_k)
            elif vector_results:
                logger.info("ℹ️ 仅向量结果")
                for row in vector_results:
                    row["rrf_score"] = 1.0 / (self.RRF_K + row["vec_rank"])
                    row["from_paths"] = ["vec"]
                return vector_results[:top_k]
            elif bm25_results:
                logger.info("ℹ️ 仅 BM25 结果")
                for row in bm25_results:
                    row["rrf_score"] = 1.0 / (self.RRF_K + row["bm25_rank"])
                    row["from_paths"] = ["bm25"]
                return bm25_results[:top_k]
            else:
                logger.warning("⚠️ 双路无结果")
                return []

        except Exception as e:
            logger.error(f"❌ 混合检索失败: {e}", exc_info=True)
            return []

    # ================================================================
    #                    生命周期
    # ================================================================

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

    def close(self):
        if hasattr(self, "_pool") and self._pool:
            self._pool.closeall()
            logger.info("🔌 连接池已关闭")
            DatabaseManager._initialized = False
            DatabaseManager._instance = None

    def health_check(self) -> bool:
        try:
            with self.get_cursor() as cur:
                cur.execute("SELECT 1")
                return cur.fetchone()[0] == 1
        except:
            return False

    def get_pool_status(self) -> dict:
        if not hasattr(self, "_pool") or not self._pool:
            return {"status": "未初始化"}
        return {
            "status": "运行中",
            "min_connections": self._pool.minconn,
            "max_connections": self._pool.maxconn,
        }


# ====================================================================
#  测试
# ====================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    db = DatabaseManager()

    print("\n" + "=" * 60)
    print("测试1: Java 后端 北京")
    print("=" * 60)
    results = db.hybrid_search(
        keyword_query="Java 后端 Spring Boot MySQL 分布式",
        city="北京",
        top_k=10,
    )
    print(f"  共 {len(results)} 条:")
    for r in results:
        paths = "+".join(r.get("from_paths", []))
        print(f"  [{paths}] {r['title']} @ {r['company']} | rrf={r['rrf_score']:.5f}")

    print("\n" + "=" * 60)
    print("测试2: Python 不限城市")
    print("=" * 60)
    results = db.hybrid_search(
        keyword_query="Python 后端开发 Django Flask",
        top_k=10,
    )
    print(f"  共 {len(results)} 条:")
    for r in results:
        paths = "+".join(r.get("from_paths", []))
        print(f"  [{paths}] {r['title']} @ {r['company']} | rrf={r['rrf_score']:.5f}")

    print("\n" + "=" * 60)
    print("测试3: 指定公司")
    print("=" * 60)
    results = db.hybrid_search(
        keyword_query="Java 开发",
        company="京东",
        top_k=5,
    )
    print(f"  共 {len(results)} 条:")
    for r in results:
        paths = "+".join(r.get("from_paths", []))
        print(f"  [{paths}] {r['title']} @ {r['company']} | rrf={r['rrf_score']:.5f}")

    db.close()
