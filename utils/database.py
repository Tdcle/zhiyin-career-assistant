# utils/database.py

import logging
import psycopg2
import psycopg2.pool
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager

from langchain_ollama import OllamaEmbeddings
from config.config import config

logger = logging.getLogger("JobAgent")


class DatabaseManager:
    """
    数据库管理器 (单例 + 连接池)

    设计要点：
    1. 单例模式：无论多少模块 import 并实例化，全局只有一个实例
    2. 连接池：使用 psycopg2.pool.ThreadedConnectionPool，支持多线程并发
    3. 上下文管理器：统一管理 cursor 的获取、提交、回滚、归还
    """

    _instance = None
    _initialized = False  # 防止 __init__ 重复执行

    def __new__(cls, *args, **kwargs):
        """单例：保证全局只有一个 DatabaseManager 实例"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """初始化：连接池 + Embedding 模型 + 表结构"""
        # 单例保护：如果已经初始化过，直接跳过
        if DatabaseManager._initialized:
            return
        DatabaseManager._initialized = True

        # --- 1. 初始化连接池 ---
        try:
            self._pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=2,   # 最小保持 2 个连接
                maxconn=10,  # 最大允许 10 个连接
                **config.DB_CONFIG
            )
            logger.info(f"✅ 数据库连接池创建成功 (min=2, max=10): {config.DB_CONFIG['dbname']}")
        except Exception as e:
            logger.critical(f"❌ 数据库连接池创建失败: {e}")
            raise e

        # --- 2. 初始化 Embedding 模型 ---
        logger.info(f"🔄 正在加载向量模型: {config.EMBEDDING_MODEL_NAME} ...")
        self.embed_model = OllamaEmbeddings(
            base_url=config.OLLAMA_URL,
            model=config.EMBEDDING_MODEL_NAME
        )
        logger.info(f"✅ 向量模型加载完成: {config.EMBEDDING_MODEL_NAME}")

        # --- 3. 初始化表结构 & 种子数据 ---
        self._init_tables()
        self._seed_default_users()

    # ================================================================
    #                    核心：上下文管理器
    # ================================================================

    @contextmanager
    def get_cursor(self, dict_cursor=False):
        """
        统一的游标获取器 (上下文管理器)

        使用方式：
            with self.get_cursor() as cur:
                cur.execute("SELECT 1")

            with self.get_cursor(dict_cursor=True) as cur:
                cur.execute("SELECT * FROM users")
                rows = cur.fetchall()  # -> [{'user_id': '00001', 'username': '张三'}, ...]

        自动行为：
            - 正常退出 -> commit
            - 抛异常   -> rollback
            - 无论如何 -> 归还连接到池
        """
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
        """初始化表结构（幂等操作，可重复执行）"""
        with self.get_cursor() as cur:
            # 1. 启用 pgvector 扩展
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            # 启用 pg_trgm 扩展 (用于 ILIKE / GIN 索引加速)
            cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")

            # 2. Jobs 表 (职位信息)
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS jobs (
                    id SERIAL PRIMARY KEY,
                    job_id VARCHAR(100) NOT NULL UNIQUE,
                    title VARCHAR(255),
                    salary VARCHAR(100),
                    company VARCHAR(255),
                    industry VARCHAR(100),
                    city VARCHAR(50),
                    district VARCHAR(50),
                    experience VARCHAR(50),
                    degree VARCHAR(50),
                    welfare TEXT,
                    detail TEXT,
                    summary TEXT,
                    detail_url VARCHAR(500),
                    embedding vector({config.VECTOR_DIM}),
                    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # 3. Users 表
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id VARCHAR(50) PRIMARY KEY,
                    username VARCHAR(100) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # 4. User Profiles 表 (用户画像)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id VARCHAR(50) PRIMARY KEY REFERENCES users(user_id),
                    preferences TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # 5. Resumes 表 (简历)
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

            # 6. 创建索引 (IF NOT EXISTS 保证幂等)
            # 6.1 向量索引 — 加速 ANN 检索 (需要表中已有一定数据量才生效，但提前创建无害)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_jobs_embedding_ivfflat
                ON jobs USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
            """)

            # 6.2 GIN 三元组索引 — 加速 ILIKE 模糊搜索
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_jobs_title_trgm
                ON jobs USING gin (title gin_trgm_ops);
            """)

            # 6.3 普通索引
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_resumes_user_id
                ON resumes (user_id);
            """)

            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_jobs_job_id
                ON jobs (job_id);
            """)

        logger.info("✅ 数据库表结构 & 索引初始化完成")

    def _seed_default_users(self):
        """初始化默认用户数据（幂等）"""
        with self.get_cursor() as cur:
            cur.execute("SELECT count(*) FROM users")
            count = cur.fetchone()[0]

            if count == 0:
                logger.info("🔄 初始化默认用户数据...")
                users = [("admin", "管理员")]
                sql = "INSERT INTO users (user_id, username) VALUES (%s, %s) ON CONFLICT DO NOTHING"
                cur.executemany(sql, users)
                logger.info("✅ 默认用户创建完成")

    # ================================================================
    #                    用户管理接口
    # ================================================================

    def _generate_next_user_id(self, cur):
        """
        在已有游标上下文中，计算下一个自增用户 ID (格式: 00001)
        注意：此方法必须在外层 get_cursor() 的 with 块中调用，直接使用传入的 cur
        """
        cur.execute(r"SELECT user_id FROM users WHERE user_id ~ '^\d+$'")
        rows = cur.fetchall()

        if not rows:
            return "00001"

        max_id = 0
        for row in rows:
            try:
                uid = int(row[0]) if not isinstance(row, dict) else int(row['user_id'])
                if uid > max_id:
                    max_id = uid
            except (ValueError, TypeError):
                continue

        next_id = max_id + 1
        return f"{next_id:05d}"

    def create_user(self, username: str):
        """
        创建新用户 (自动生成 ID)

        Returns:
            tuple: (success: bool, message: str)
        """
        if not username or not username.strip():
            return False, "用户名不能为空"

        try:
            with self.get_cursor() as cur:
                new_user_id = self._generate_next_user_id(cur)

                cur.execute(
                    "INSERT INTO users (user_id, username) VALUES (%s, %s)",
                    (new_user_id, username.strip())
                )

                logger.info(f"✅ 用户创建成功: ID={new_user_id}, 昵称={username}")
                return True, f"用户创建成功! ID: {new_user_id} | 昵称: {username}"

        except psycopg2.IntegrityError as e:
            logger.warning(f"⚠️ 用户创建冲突: {e}")
            return False, f"创建失败: 用户已存在"
        except Exception as e:
            logger.error(f"❌ 用户创建失败: {e}", exc_info=True)
            return False, f"创建失败: {str(e)}"

    def get_all_users_list(self):
        """
        获取所有用户列表 (供下拉框使用)

        Returns:
            list[str]: ["00001 (张三)", "admin (管理员)", ...]
        """
        try:
            with self.get_cursor() as cur:
                cur.execute("SELECT user_id, username FROM users ORDER BY user_id DESC")
                users = cur.fetchall()
                return [f"{u[0]} ({u[1]})" for u in users]
        except Exception as e:
            logger.error(f"❌ 获取用户列表失败: {e}")
            return []

    # ================================================================
    #                    画像管理接口
    # ================================================================

    def get_user_profile(self, user_id: str) -> str:
        """获取用户画像偏好文本"""
        try:
            with self.get_cursor(dict_cursor=True) as cur:
                cur.execute(
                    "SELECT preferences FROM user_profiles WHERE user_id = %s",
                    (user_id,)
                )
                res = cur.fetchone()
                return res['preferences'] if res else ""
        except Exception as e:
            logger.error(f"❌ 获取用户画像失败 (user_id={user_id}): {e}")
            return ""

    def update_user_profile(self, user_id: str, preferences: str) -> bool:
        """
        更新用户偏好 (Upsert)
        前提：users 表中必须存在该 user_id
        """
        try:
            with self.get_cursor() as cur:
                sql = """
                    INSERT INTO user_profiles (user_id, preferences)
                    VALUES (%s, %s)
                    ON CONFLICT (user_id)
                    DO UPDATE SET preferences = EXCLUDED.preferences,
                                  updated_at = CURRENT_TIMESTAMP;
                """
                cur.execute(sql, (user_id, preferences))
                logger.info(f"✅ 用户画像已更新: user_id={user_id}")
                return True
        except Exception as e:
            logger.error(f"❌ 更新用户画像失败 (user_id={user_id}): {e}", exc_info=True)
            return False

    # ================================================================
    #                    简历管理接口
    # ================================================================

    def save_resume(self, user_id: str, filename: str, content: str):
        """
        保存简历文本并生成向量

        Returns:
            tuple: (success: bool, message: str)
        """
        try:
            # 1. 生成向量 (在数据库事务外完成，避免长时间占用连接)
            logger.info(f"🔄 正在为用户 {user_id} 的简历生成向量...")
            vector = self.embed_model.embed_query(content)

            if not vector:
                return False, "向量生成失败：Embedding 模型返回空结果"

            if len(vector) != config.VECTOR_DIM:
                return False, f"向量维度不匹配: 期望 {config.VECTOR_DIM}, 实际 {len(vector)}"

            # 2. 存入数据库
            with self.get_cursor() as cur:
                sql = """
                    INSERT INTO resumes (user_id, filename, content, embedding)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id;
                """
                cur.execute(sql, (user_id, filename, content, vector))
                resume_id = cur.fetchone()[0]

                logger.info(f"✅ 简历已入库: user_id={user_id}, resume_id={resume_id}, file={filename}")
                return True, f"简历已入库 (ID: {resume_id})"

        except psycopg2.ForeignKeyViolation:
            logger.warning(f"⚠️ 用户不存在，无法保存简历: user_id={user_id}")
            return False, "用户不存在，请先创建用户"
        except Exception as e:
            logger.error(f"❌ 简历保存失败 (user_id={user_id}): {e}", exc_info=True)
            return False, str(e)

    def get_latest_resume(self, user_id: str):
        """
        获取用户最新上传的一份简历

        Returns:
            dict | None: {'content': ..., 'filename': ..., 'created_at': ...}
        """
        try:
            with self.get_cursor(dict_cursor=True) as cur:
                sql = """
                    SELECT content, filename, created_at
                    FROM resumes
                    WHERE user_id = %s
                    ORDER BY created_at DESC
                    LIMIT 1
                """
                cur.execute(sql, (user_id,))
                return cur.fetchone()
        except Exception as e:
            logger.error(f"❌ 获取简历失败 (user_id={user_id}): {e}")
            return None

    # ================================================================
    #                    职位数据接口 (爬虫/向量化)
    # ================================================================

    def insert_job(self, data: dict) -> bool:
        """插入原始职位数据 (ON CONFLICT 幂等)"""
        try:
            with self.get_cursor() as cur:
                sql = """
                    INSERT INTO jobs
                    (job_id, title, salary, company, industry, city, district,
                     experience, degree, welfare, detail, summary, detail_url)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (job_id) DO NOTHING
                """
                cur.execute(sql, (
                    data['job_id'], data['title'], data['salary'], data['company'],
                    data['industry'], data['city'], data['district'],
                    data['experience'], data['degree'], data['welfare'],
                    data['detail'], data['summary'], data['detail_url']
                ))
                inserted = cur.rowcount > 0
                if inserted:
                    logger.debug(f"✅ 职位入库: {data['job_id']} - {data['title']}")
                return inserted
        except Exception as e:
            logger.error(f"❌ 插入职位失败 (job_id={data.get('job_id')}): {e}")
            return False

    def fetch_jobs_without_embedding(self, limit: int = 100):
        """获取未向量化的职位数据"""
        try:
            with self.get_cursor() as cur:
                sql = """
                    SELECT job_id, title, company, salary, welfare, detail,
                           city, district, experience, degree
                    FROM jobs
                    WHERE embedding IS NULL
                    LIMIT %s
                """
                cur.execute(sql, (limit,))
                return cur.fetchall()
        except Exception as e:
            logger.error(f"❌ 获取未向量化职位失败: {e}")
            return []

    def update_job_analysis(self, job_id: str, summary: str, vector) -> bool:
        """同时更新职位摘要和向量"""
        try:
            with self.get_cursor() as cur:
                sql = """
                    UPDATE jobs
                    SET summary = %s, embedding = %s
                    WHERE job_id = %s
                """
                cur.execute(sql, (summary, vector, job_id))
                updated = cur.rowcount > 0
                if updated:
                    logger.debug(f"✅ 职位分析数据已更新: {job_id}")
                return updated
        except Exception as e:
            logger.error(f"❌ 更新分析数据失败 (job_id={job_id}): {e}")
            return False

    def update_embedding(self, job_id: str, text: str) -> bool:
        """仅更新职位向量"""
        try:
            vector = self.embed_model.embed_query(text)
            if not vector or len(vector) != config.VECTOR_DIM:
                logger.warning(f"⚠️ 向量生成异常 (job_id={job_id}): dim={len(vector) if vector else 0}")
                return False

            with self.get_cursor() as cur:
                sql = "UPDATE jobs SET embedding = %s WHERE job_id = %s"
                cur.execute(sql, (vector, job_id))
                return cur.rowcount > 0
        except Exception as e:
            logger.error(f"❌ 向量更新失败 (job_id={job_id}): {e}")
            return False

    def get_job_details(self, job_id):
        """
        根据 job_id 获取职位完整信息

        兼容逻辑：优先匹配 job_id 字段，若传入纯数字则额外尝试匹配自增 id
        """
        try:
            with self.get_cursor(dict_cursor=True) as cur:
                # 优先匹配 job_id (字符串唯一标识)
                cur.execute("SELECT * FROM jobs WHERE job_id = %s", (str(job_id),))
                result = cur.fetchone()

                # 若未找到且传入的是纯数字，尝试匹配自增 id
                if not result and str(job_id).isdigit():
                    cur.execute("SELECT * FROM jobs WHERE id = %s", (int(job_id),))
                    result = cur.fetchone()

                if not result:
                    logger.warning(f"⚠️ 职位未找到: job_id={job_id}")

                return result
        except Exception as e:
            logger.error(f"❌ 获取职位详情失败 (job_id={job_id}): {e}")
            return None

    # ================================================================
    #                    搜索与分析接口
    # ================================================================

    def vector_search(self, title: str = "", semantic_query: str = "", city: str = "", company: str = "", experience: str = "", welfare: str = "", top_k: int = 5):
        try:
            from utils.logger import sys_logger

            # -------------------------------------------------------------------------
            # 阶段一：精确公司过滤层 (Exact Company Match)
            # -------------------------------------------------------------------------
            if company:
                clean_company = company.strip()
                with self.get_cursor(dict_cursor=True) as cur:
                    sql_exact = """
                                SELECT id, job_id, title, company, industry, salary,
                                       city, district, experience, degree, welfare,
                                       summary, detail, detail_url
                                FROM jobs 
                                WHERE company ILIKE %s
                            """
                    params_exact = [f"%{clean_company}%"]
                    if city:
                        sql_exact += " AND city ILIKE %s"
                        params_exact.append(f"%{city}%")

                    sql_exact += " LIMIT 50"

                    cur.execute(sql_exact, params_exact)
                    exact_candidates = cur.fetchall()

                if exact_candidates:
                    # 使用 title 进行基础算分
                    search_keywords = title
                    keywords = [kw.strip().lower() for kw in search_keywords.split() if kw.strip()]

                    for row in exact_candidates:
                        score = 0.0
                        title_lower = (row.get('title') or '').lower()
                        for kw in keywords:
                            if kw in title_lower:
                                score += 1.0

                        row['exact_match_score'] = score
                        row['vec_dist'] = 0.0
                        row['bonus'] = score
                        row['final_score'] = -score

                    exact_candidates.sort(key=lambda x: x['exact_match_score'], reverse=True)
                    final_results = exact_candidates[:top_k]

                    log_details = [f"[{r['title']}|score:{r['exact_match_score']:.1f}]" for r in final_results]
                    sys_logger.info(
                        f"✅ [阶段一: 精准命中] 公司 '{clean_company}', 返回={len(final_results)} \n得分明细: {', '.join(log_details)}")
                    return final_results
                else:
                    sys_logger.info(f"⚠️ [阶段一: 未命中] 未查到公司 '{clean_company}'，进入向量召回...")

            # -------------------------------------------------------------------------
            # 阶段二：向量模糊召回与重排兜底 (Vector Recall & Rerank)
            # -------------------------------------------------------------------------
            # 1. 消除向量非对称性，把 experience 拼进来引导向量去找“实习生聚集区”
            query_parts = []
            if title: query_parts.append(f"职位: {title}")
            if city: query_parts.append(f"地点: {city}")
            if company: query_parts.append(f"公司: {company}")
            if experience: query_parts.append(f"经验要求: {experience}")
            if welfare: query_parts.append(f"福利: {welfare}")
            if semantic_query: query_parts.append(f"介绍: {semantic_query}")

            symmetric_query = "\n".join(query_parts)

            query_vector = self.embed_model.embed_query(symmetric_query)
            if not query_vector:
                sys_logger.warning("⚠️ 查询向量生成失败")
                return []

            with self.get_cursor(dict_cursor=True) as cur:
                where_sql = ""
                dynamic_params = []
                if city:
                    where_sql += " AND city ILIKE %s"
                    dynamic_params.append(f"%{city}%")

                recall_count = max(100, top_k * 10)

                sql_vector = f"""
                            SELECT id, job_id, title, company, industry, salary,
                                   city, district, experience, degree, welfare,
                                   summary, detail, detail_url,
                                   (embedding <=> %s::vector) AS vec_dist
                            FROM jobs
                            WHERE embedding IS NOT NULL {where_sql}
                            ORDER BY embedding <=> %s::vector
                            LIMIT %s;
                        """
                params_vector = [query_vector] + dynamic_params + [query_vector, recall_count]
                cur.execute(sql_vector, params_vector)
                candidates = cur.fetchall()

            # 4. 目标重排 (Targeted Rerank)：强行把实习等特殊岗捞到 Top 20
            clean_comp = company.lower().strip() if company else ""
            clean_title = title.lower().strip() if title else ""
            clean_exp = experience.lower().replace("经验", "").strip() if experience else ""

            search_title_kws = [k for k in clean_title.split() if k]

            for row in candidates:
                bonus = 0.0
                title_lower = (row.get('title') or '').lower()
                company_lower = (row.get('company') or '').lower()
                exp_lower = (row.get('experience') or '').lower()

                # A. 职位名词直接命中 (权重 0.15)
                for kw in search_title_kws:
                    if kw in title_lower:
                        bonus += 0.15
                        break

                # B. 【关键防沉底】如果是找实习/应届，且匹配到了，给予巨额加分上浮 (0.20)
                if clean_exp and (clean_exp in exp_lower or clean_exp in title_lower):
                    bonus += 0.20

                # C. 公司精确匹配 (兜底重排绝杀：权重 0.30)
                if clean_comp and clean_comp in company_lower:
                    bonus += 0.30

                row['vec_dist'] = float(row['vec_dist'])
                row['bonus'] = bonus
                row['final_score'] = row['vec_dist'] - bonus

            candidates.sort(key=lambda x: x['final_score'])
            final_results = candidates[:top_k]

            log_details = [
                f"[{r['title']}|dist:{r['vec_dist']:.3f}|bonus:{r['bonus']:.2f}|score:{r['final_score']:.3f}]" for r in
                final_results]

            clean_log_query = symmetric_query.replace('\n', ' | ')
            sys_logger.info(
                f"🔍 [阶段二: 向量召回] V8: bridge='{clean_log_query}', 候选={len(candidates)}, 返回={len(final_results)} \n得分明细: {', '.join(log_details)}"
            )
            return final_results

        except Exception as e:
            from utils.logger import sys_logger
            sys_logger.error(f"❌ 混合检索失败: {e}", exc_info=True)
            return []



    # ================================================================
    #                    生命周期管理
    # ================================================================

    def close(self):
        """关闭连接池，释放所有连接"""
        if hasattr(self, '_pool') and self._pool:
            self._pool.closeall()
            logger.info("🔌 数据库连接池已关闭")
            DatabaseManager._initialized = False
            DatabaseManager._instance = None

    def health_check(self) -> bool:
        """健康检查：验证连接池和数据库是否可用"""
        try:
            with self.get_cursor() as cur:
                cur.execute("SELECT 1")
                result = cur.fetchone()
                return result[0] == 1
        except Exception as e:
            logger.error(f"❌ 健康检查失败: {e}")
            return False

    def get_pool_status(self) -> dict:
        """获取连接池状态 (用于监控/调试)"""
        if not hasattr(self, '_pool') or not self._pool:
            return {"status": "未初始化"}

        return {
            "status": "运行中",
            "min_connections": self._pool.minconn,
            "max_connections": self._pool.maxconn,
        }