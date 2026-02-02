import os
from dotenv import load_dotenv

# ==========================================================
# 【核心修复】全局代理屏蔽设置
# 必须放在任何网络请求库 (如 requests, httpx, langchain) 导入之前
# 这样即便你开着梯子，Python 也会直连本地服务
os.environ["NO_PROXY"] = "localhost,127.0.0.1,0.0.0.0,postgres"
# ==========================================================

# 加载 .env 文件中的环境变量
load_dotenv()

class Config:
    # API Keys
    DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
    if not DASHSCOPE_API_KEY:
        raise ValueError("请在 .env 文件中配置 DASHSCOPE_API_KEY")

    # 数据库配置字典
    DB_CONFIG = {
        'host': os.getenv("DB_HOST", "localhost"),
        'port': os.getenv("DB_PORT", "5433"),
        'user': os.getenv("DB_USER", "postgres"),
        'password': os.getenv("DB_PASSWORD"),
        'dbname': os.getenv("DB_NAME", "boss_zhipin")
    }

    # 向量模型配置
    VECTOR_DIM = int(os.getenv("VECTOR_DIM", 768))
    MODEL_NAME = os.getenv("MODEL_NAME", "nomic-embed-text-v2-moe")
    OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

# 实例化配置对象，方便其他文件 import config
config = Config()