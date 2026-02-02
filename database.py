# database.py (完整升级版)
import psycopg2
from psycopg2.extras import RealDictCursor
from langchain_ollama import OllamaEmbeddings
from config import config


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

    def _init_tables(self):
        """初始化表结构"""
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            # Jobs 表
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

            # 用户画像表
            cur.execute("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id VARCHAR(50) PRIMARY KEY,
                preferences TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)

    # ================= 爬虫/向量化专用方法 =================

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
                # rowcount: 1=插入成功, 0=重复忽略
                return cur.rowcount > 0
            except Exception as e:
                print(f"插入失败: {e}")
                return False

    def fetch_jobs_without_embedding(self, limit=100):
        """获取未向量化的数据"""
        with self.conn.cursor() as cur:
            sql = "SELECT job_id, title, company, salary, welfare, detail FROM jobs WHERE embedding IS NULL LIMIT %s"
            cur.execute(sql, (limit,))
            return cur.fetchall()

    def update_job_analysis(self, job_id, summary, vector):
        """
        同时更新摘要(summary)和向量(embedding)
        用于第二步的处理流程
        """
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
        """生成并更新向量"""
        try:
            vector = self.embed_model.embed_query(text)

            # 维度检查
            if not vector or len(vector) != config.VECTOR_DIM:
                print(f"❌ 维度错误: 模型返回 {len(vector)}，数据库需要 {config.VECTOR_DIM}")
                return False

            with self.conn.cursor() as cur:
                sql = "UPDATE jobs SET embedding = %s WHERE job_id = %s"
                cur.execute(sql, (vector, job_id))
                return True
        except Exception as e:
            print(f"向量更新失败 (ID: {job_id}): {e}")
            return False

    # ================= RAG/Agent 专用方法 =================

    def get_user_profile(self, user_id):
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT preferences FROM user_profiles WHERE user_id = %s", (user_id,))
            res = cur.fetchone()
            return res['preferences'] if res else ""

    def update_user_profile(self, user_id, preferences):
        """
        更新用户偏好 (Upsert: 不存在则插入，存在则更新)
        """
        try:
            with self.conn.cursor() as cur:
                sql = """
                INSERT INTO user_profiles (user_id, preferences) 
                VALUES (%s, %s)
                ON CONFLICT (user_id) 
                DO UPDATE SET preferences = EXCLUDED.preferences, updated_at = CURRENT_TIMESTAMP;
                """
                cur.execute(sql, (user_id, preferences))
                # print(f"✅ 用户 {user_id} 画像已更新")
                return True
        except Exception as e:
            print(f"❌ 更新用户画像失败: {e}")
            return False

    def vector_search(self, query_text: str, top_k=5):
        """
        【修复版】混合检索 (Hybrid Search)
        修复了 KeyError: 'city' 问题，明确查询 city 和 district 字段
        """
        try:
            # 1. 生成向量
            query_vector = self.embed_model.embed_query(query_text)

            # 2. 混合查询 SQL
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

                # 参数准备
                like_kw = f"%{query_text[:4]}%"

                cur.execute(sql, (query_vector, like_kw, like_kw, query_vector, like_kw))
                results = cur.fetchall()
                return results

        except Exception as e:
            print(f"❌ 检索失败: {e}")
            import traceback
            traceback.print_exc()
            return []

    def get_market_analytics(self, keyword):
        """
        简单的市场分析工具：统计某个关键词（如 Python）的职位数量
        """
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            sql = "SELECT COUNT(*) as job_count FROM jobs WHERE title ILIKE %s OR detail ILIKE %s"
            kw = f"%{keyword}%"
            cur.execute(sql, (kw, kw))
            return cur.fetchone()

    def close(self):
        self.conn.close()

