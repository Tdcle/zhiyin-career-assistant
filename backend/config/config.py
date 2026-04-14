import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values, load_dotenv
from langchain_community.chat_models import ChatTongyi
from langchain_ollama import ChatOllama, OllamaEmbeddings

os.environ["NO_PROXY"] = "localhost,127.0.0.1,0.0.0.0,postgres,redis"

_BASE_DIR = Path(__file__).resolve().parents[1]
_ENV_CANDIDATES = [
    _BASE_DIR.parent / "config" / ".env",
    _BASE_DIR / "config" / ".env",
]


def _load_env() -> None:
    """Load .env files without allowing empty values to clobber valid config."""
    loaded = False
    initial_keys = set(os.environ.keys())
    for env_path in _ENV_CANDIDATES:
        if not env_path.exists():
            continue
        loaded = True
        for key, value in dotenv_values(env_path).items():
            if not key or value in (None, ""):
                continue
            if key in initial_keys:
                continue
            os.environ[key] = value
    if not loaded:
        load_dotenv()


_load_env()


@dataclass(frozen=True)
class ChatModels:
    assistant_main: str
    assistant_summary: str
    resume_structurer: str
    assistant_planner: str
    assistant_rewrite: str
    interviewer: str
    interview_assessment: str
    interview_report: str
    interview_summary: str
    eval_dataset_generator: str
    eval_judge: str
    data_summary: str


@dataclass(frozen=True)
class OllamaModels:
    embedding: str
    match_analysis: str


def _parse_cors_origins(raw: str) -> list[str]:
    if not raw:
        return ["http://localhost:5173", "http://127.0.0.1:5173"]
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_int_csv(raw: str, default: list[int]) -> list[int]:
    if not raw:
        return list(default)
    values: list[int] = []
    for part in str(raw).split(","):
        text = part.strip()
        if not text:
            continue
        try:
            number = int(text)
            if number >= 0:
                values.append(number)
        except Exception:
            continue
    return values or list(default)


class Config:
    APP_NAME = os.getenv("APP_NAME", "JobAgent")
    ENV = os.getenv("APP_ENV", "dev")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
    LOG_DIR = os.getenv("LOG_DIR", "logs")

    DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
    if not DASHSCOPE_API_KEY:
        raise ValueError("请在 .env 文件中配置 DASHSCOPE_API_KEY")

    DB_CONFIG = {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": os.getenv("DB_PORT", "5432"),
        "user": os.getenv("DB_USER", "postgres"),
        "password": os.getenv("DB_PASSWORD"),
        "dbname": os.getenv("DB_NAME", "boss_zhipin"),
    }

    VECTOR_DIM = int(os.getenv("VECTOR_DIM", "768"))
    OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
    RERANKER_MODEL_PATH = os.getenv("RERANKER_MODEL_PATH", "./models/bge-reranker-v2-m3")

    CHAT_MODELS = ChatModels(
        assistant_main=os.getenv("MODEL_ASSISTANT_MAIN", "qwen-plus-2025-07-28"),
        assistant_summary=os.getenv("MODEL_ASSISTANT_SUMMARY", "qwen-plus-2025-07-28"),
        resume_structurer=os.getenv("MODEL_RESUME_STRUCTURER", "qwen-plus-1220"),
        assistant_planner=os.getenv("MODEL_ASSISTANT_PLANNER", "qwen-max-2025-01-25"),
        assistant_rewrite=os.getenv("MODEL_ASSISTANT_REWRITE", "qwen-max-2025-01-25"),
        interviewer=os.getenv("MODEL_INTERVIEWER", "glm-5"),
        interview_assessment=os.getenv("MODEL_INTERVIEW_ASSESSMENT", "qwen-plus-2025-07-28"),
        interview_report=os.getenv("MODEL_INTERVIEW_REPORT", "qwen-plus-2025-07-28"),
        interview_summary=os.getenv("MODEL_INTERVIEW_SUMMARY", "qwen-plus-2025-07-28"),
        eval_dataset_generator=os.getenv("MODEL_EVAL_DATASET_GENERATOR", "qwen3.5-35b-a3b"),
        eval_judge=os.getenv("MODEL_EVAL_JUDGE", "qwen-plus-2025-07-28"),
        data_summary=os.getenv("MODEL_DATA_SUMMARY", "qwen-plus-2025-07-28"),
    )

    OLLAMA_MODELS = OllamaModels(
        embedding=os.getenv("MODEL_OLLAMA_EMBEDDING", "nomic-embed-text-v2-moe"),
        match_analysis=os.getenv("MODEL_OLLAMA_MATCH_ANALYSIS", "qwen2.5:3b"),
    )
    EMBEDDING_MODEL_NAME = OLLAMA_MODELS.embedding
    MARK_MODEL_NAME = OLLAMA_MODELS.match_analysis

    CORS_ORIGINS = _parse_cors_origins(os.getenv("CORS_ORIGINS", ""))
    SESSION_SECRET_KEY = os.getenv("SESSION_SECRET_KEY", "dev-session-secret-change-me")
    SESSION_HTTPS_ONLY = os.getenv("SESSION_HTTPS_ONLY", "false").lower() == "true"

    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "dev-jwt-secret-change-me")
    JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "10080"))

    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    RESUME_TASK_TTL_SECONDS = int(os.getenv("RESUME_TASK_TTL_SECONDS", "86400"))
    RESUME_QUEUE_MAX_ATTEMPTS = int(os.getenv("RESUME_QUEUE_MAX_ATTEMPTS", "3"))
    RESUME_QUEUE_RETRY_DELAYS_SECONDS = _parse_int_csv(
        os.getenv("RESUME_QUEUE_RETRY_DELAYS_SECONDS", "15,60,180"),
        [15, 60, 180],
    )
    RESUME_QUEUE_PENDING_IDLE_MS = int(os.getenv("RESUME_QUEUE_PENDING_IDLE_MS", "120000"))
    RESUME_QUEUE_CLAIM_BATCH = int(os.getenv("RESUME_QUEUE_CLAIM_BATCH", "20"))
    RESUME_QUEUE_STREAM_MAXLEN = int(os.getenv("RESUME_QUEUE_STREAM_MAXLEN", "20000"))
    RESUME_MAX_FILE_MB = int(os.getenv("RESUME_MAX_FILE_MB", "10"))
    UPLOAD_RATE_LIMIT = int(os.getenv("UPLOAD_RATE_LIMIT", "10"))
    UPLOAD_RATE_WINDOW_SECONDS = int(os.getenv("UPLOAD_RATE_WINDOW_SECONDS", "3600"))
    SEARCH_CACHE_ENABLED = os.getenv("SEARCH_CACHE_ENABLED", "true").lower() == "true"
    SEARCH_CACHE_TTL_SECONDS = int(os.getenv("SEARCH_CACHE_TTL_SECONDS", "1200"))
    SEARCH_CACHE_EMPTY_TTL_SECONDS = int(os.getenv("SEARCH_CACHE_EMPTY_TTL_SECONDS", "45"))
    SEARCH_CACHE_LOCK_TTL_SECONDS = int(os.getenv("SEARCH_CACHE_LOCK_TTL_SECONDS", "8"))
    SEARCH_CACHE_LOCK_WAIT_SECONDS = float(os.getenv("SEARCH_CACHE_LOCK_WAIT_SECONDS", "1.2"))

    MONITOR_WINDOW_SECONDS = int(os.getenv("MONITOR_WINDOW_SECONDS", "300"))
    MONITOR_MAX_EVENTS = int(os.getenv("MONITOR_MAX_EVENTS", "5000"))
    MONITOR_ALERT_MIN_SAMPLES = int(os.getenv("MONITOR_ALERT_MIN_SAMPLES", "30"))
    MONITOR_ALERT_CHECK_INTERVAL_SECONDS = int(os.getenv("MONITOR_ALERT_CHECK_INTERVAL_SECONDS", "60"))
    MONITOR_ALERT_P95_MS = int(os.getenv("MONITOR_ALERT_P95_MS", "2500"))
    MONITOR_ALERT_5XX_RATE = float(os.getenv("MONITOR_ALERT_5XX_RATE", "0.10"))
    MONITOR_ALERT_RESUME_QUEUE_LEN = int(os.getenv("MONITOR_ALERT_RESUME_QUEUE_LEN", "20"))

    @classmethod
    def create_tongyi(cls, model_name: str, streaming: bool = False, temperature: float | None = None):
        kwargs = {
            "api_key": cls.DASHSCOPE_API_KEY,
            "model": model_name,
            "streaming": streaming,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        return ChatTongyi(**kwargs)

    @classmethod
    def create_ollama_chat(cls, model_name: str, temperature: float = 0.1):
        return ChatOllama(
            base_url=cls.OLLAMA_URL,
            model=model_name,
            temperature=temperature,
        )

    @classmethod
    def create_embeddings(cls):
        return OllamaEmbeddings(
            base_url=cls.OLLAMA_URL,
            model=cls.OLLAMA_MODELS.embedding,
        )


config = Config()
