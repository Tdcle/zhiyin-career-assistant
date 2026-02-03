# database.py
import psycopg2
from psycopg2.extras import RealDictCursor
from langchain_ollama import OllamaEmbeddings
from config.config import config
from datetime import datetime


class DatabaseManager:
    def __init__(self):
        """初始化：连接数据库、加载模型、初始化表结构"""
        try:
            self.conn = psycopg2.connect(**config.DB_CONFIG)
            self.conn.autocommit = True
            # print(f"✅ 数据库连接成功: {config.DB_CONFIG['dbname']}")
        except Exception as e:
            print(f"❌ 数据库连接失败: {e}")
            raise e

        # 初始化 Embedding 模型
        # print(f"🔄 正在加载向量模型: {config.MODEL_NAME} ...")
        self.embed_model = OllamaEmbeddings(
            base_url=config.OLLAMA_URL,
            model=config.MODEL_NAME
        )

        self._init_tables()
        self._seed_default_users()  # 初始化默认用户

    def _init_tables(self):
        """初始化表结构"""
        with self.conn.cursor() as cur:
            # 1. 启用向量扩展
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

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

            # 3. Users 表 (用户基础信息 - 新增)
            cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id VARCHAR(50) PRIMARY KEY,
                username VARCHAR(100) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)

            # 4. User Profiles 表 (用户画像 - 升级)
            # 增加了 FOREIGN KEY 约束
            cur.execute("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id VARCHAR(50) PRIMARY KEY REFERENCES users(user_id),
                preferences TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)

            # 5. resumes 简历表
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

    def _seed_default_users(self):
        """初始化默认用户数据"""
        with self.conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM users")
            count = cur.fetchone()[0]

            if count == 0:
                print("🔄 初始化默认用户数据...")
                users = [("admin", "管理员")]
                sql = "INSERT INTO users (user_id, username) VALUES (%s, %s) ON CONFLICT DO NOTHING"
                cur.executemany(sql, users)

    def _generate_next_user_id(self, cur):
        """
        核心逻辑：计算下一个 ID (格式: 00001)
        """
        # 1. 查找所有由纯数字组成的 user_id
        # Postgres 正则: '^\d+$' 匹配纯数字
        cur.execute(r"SELECT user_id FROM users WHERE user_id ~ '^\d+$'")
        rows = cur.fetchall()

        if not rows:
            return "00001"  # 如果没有数字ID，从1开始

        # 2. 找出最大的数字
        max_id = 0
        for row in rows:
            try:
                # 将 "00005" 转为 int 5
                uid = int(row[0])
                if uid > max_id:
                    max_id = uid
            except:
                continue

        # 3. 加 1 并补零
        next_id = max_id + 1
        return f"{next_id:05d}"  # 格式化为 5 位，不足补 0

    # ================= 用户管理接口 =================

    def create_user(self, username):
        """
        创建一个新用户 (自动生成 ID)
        """
        if not username:
            return False, "用户名不能为空"

        try:
            with self.conn.cursor() as cur:
                # 1. 自动生成 ID
                new_user_id = self._generate_next_user_id(cur)

                # 2. 插入数据库
                cur.execute(
                    "INSERT INTO users (user_id, username) VALUES (%s, %s)",
                    (new_user_id, username)
                )
                return True, f"用户创建成功! ID: {new_user_id} | 昵称: {username}"
        except Exception as e:
            print(f"创建用户失败: {e}")
            return False, f"创建失败: {str(e)}"

    def get_all_users_list(self):
        try:
            with self.conn.cursor() as cur:
                # 按创建时间倒序排列
                cur.execute("SELECT user_id, username FROM users ORDER BY user_id DESC")
                users = cur.fetchall()
                return [f"{u[0]} ({u[1]})" for u in users]
        except Exception as e:
            print(f"获取用户列表失败: {e}")
            return []

    # ================= 画像管理接口 =================

    def get_user_profile(self, user_id):
        """获取用户画像"""
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT preferences FROM user_profiles WHERE user_id = %s", (user_id,))
            res = cur.fetchone()
            return res['preferences'] if res else ""

    def update_user_profile(self, user_id, preferences):
        """
        更新用户偏好 (Upsert)
        前提：users 表中必须存在该 user_id
        """
        try:
            with self.conn.cursor() as cur:
                # PostgreSQL 的 Upsert 语法
                sql = """
                INSERT INTO user_profiles (user_id, preferences) 
                VALUES (%s, %s)
                ON CONFLICT (user_id) 
                DO UPDATE SET preferences = EXCLUDED.preferences, updated_at = CURRENT_TIMESTAMP;
                """
                cur.execute(sql, (user_id, preferences))
                return True
        except Exception as e:
            print(f"❌ 更新用户画像失败: {e}")
            return False

    # ================= 简历专用方法 (Write Path) =================
    def save_resume(self, user_id, filename, content):
        """
        后台任务：保存简历文本并生成向量
        """
        try:
            # 1. 生成向量 (调用 Ollama)
            # print(f"🔄 正在为用户 {user_id} 的简历生成向量...")
            vector = self.embed_model.embed_query(content)

            if not vector:
                return False, "向量生成失败"

            # 2. 存入数据库
            with self.conn.cursor() as cur:
                # 策略：这里演示简单的“追加模式”，如果需要“覆盖模式”可以先 DELETE
                # cur.execute("DELETE FROM resumes WHERE user_id = %s", (user_id,))

                sql = """
                INSERT INTO resumes (user_id, filename, content, embedding)
                VALUES (%s, %s, %s, %s)
                RETURNING id;
                """
                cur.execute(sql, (user_id, filename, content, vector))
                resume_id = cur.fetchone()[0]
                return True, f"简历已入库 (ID: {resume_id})"
        except Exception as e:
            print(f"❌ 简历保存失败: {e}")
            return False, str(e)

    # ================= 简历专用方法 (Read Path - Tool用) =================

    def get_latest_resume(self, user_id):
        """
        获取用户最新上传的一份简历内容
        """
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                sql = """
                SELECT content, filename, created_at 
                FROM resumes 
                WHERE user_id = %s 
                ORDER BY created_at DESC 
                LIMIT 1
                """
                cur.execute(sql, (user_id,))
                result = cur.fetchone()
                return result
        except Exception as e:
            print(f"❌ 获取简历失败: {e}")
            return None

    # ================= 爬虫/向量化专用方法 (保持不变) =================

    def insert_job(self, data):
        """插入原始职位数据"""
        with self.conn.cursor() as cur:
            sql = """
            INSERT INTO jobs 
            (job_id, title, salary, company, industry, city, district, experience, degree, welfare, detail, summary, detail_url)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (job_id) DO NOTHING
            """
            try:
                cur.execute(sql, (
                    data['job_id'], data['title'], data['salary'], data['company'],
                    data['industry'], data['city'], data['district'],
                    data['experience'], data['degree'], data['welfare'],
                    data['detail'], data['summary'], data['detail_url']
                ))
                return cur.rowcount > 0
            except Exception as e:
                print(f"插入失败: {e}")
                return False

    def fetch_jobs_without_embedding(self, limit=100):
        """获取未向量化的数据"""
        with self.conn.cursor() as cur:
            sql = """
            SELECT job_id, title, company, salary, welfare, detail, city, district, experience, degree 
            FROM jobs 
            WHERE embedding IS NULL 
            LIMIT %s
            """
            cur.execute(sql, (limit,))
            return cur.fetchall()

    def update_job_analysis(self, job_id, summary, vector):
        """同时更新摘要和向量"""
        try:
            with self.conn.cursor() as cur:
                sql = """
                UPDATE jobs 
                SET summary = %s, embedding = %s 
                WHERE job_id = %s
                """
                cur.execute(sql, (summary, vector, job_id))
                return True
        except Exception as e:
            print(f"更新分析数据失败 (ID: {job_id}): {e}")
            return False

    def update_embedding(self, job_id, text):
        """仅更新向量"""
        try:
            vector = self.embed_model.embed_query(text)
            if not vector or len(vector) != config.VECTOR_DIM:
                return False

            with self.conn.cursor() as cur:
                sql = "UPDATE jobs SET embedding = %s WHERE job_id = %s"
                cur.execute(sql, (vector, job_id))
                return True
        except Exception as e:
            print(f"向量更新失败 (ID: {job_id}): {e}")
            return False

    # ================= 搜索与分析接口 (保持不变) =================

    def vector_search(self, query_text: str, top_k=5):
        """混合检索"""
        try:
            query_vector = self.embed_model.embed_query(query_text)

            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                sql = f"""
                SELECT title, company, industry, salary, city, district, welfare, summary, detail_url,
                       (embedding <=> %s::vector) as vector_dist,
                       (CASE WHEN title ILIKE %s THEN 0.2 ELSE 0 END) as title_score,
                       (CASE WHEN summary ILIKE %s THEN 0.1 ELSE 0 END) as detail_score
                FROM jobs
                WHERE embedding IS NOT NULL
                ORDER BY (embedding <=> %s::vector) - (CASE WHEN title ILIKE %s THEN 0.2 ELSE 0 END) ASC
                LIMIT {top_k};
                """
                like_kw = f"%{query_text}%"
                cur.execute(sql, (query_vector, like_kw, like_kw, query_vector, like_kw))
                results = cur.fetchall()
                return results

        except Exception as e:
            print(f"❌ 检索失败: {e}")
            return []

    def get_market_analytics(self, keyword):
        """统计职位数量"""
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            sql = "SELECT COUNT(*) as job_count FROM jobs WHERE title ILIKE %s OR detail ILIKE %s"
            kw = f"%{keyword}%"
            cur.execute(sql, (kw, kw))
            return cur.fetchone()

    def close(self):
        self.conn.close()