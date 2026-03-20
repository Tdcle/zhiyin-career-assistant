import os
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain_community.chat_models import ChatTongyi
from langchain_ollama import ChatOllama, OllamaEmbeddings

os.environ["NO_PROXY"] = "localhost,127.0.0.1,0.0.0.0,postgres"
load_dotenv()


@dataclass(frozen=True)
class ChatModels:
    assistant_main: str
    assistant_summary: str
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
        "port": os.getenv("DB_PORT", "5433"),
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
