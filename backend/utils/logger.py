import logging
import os
from datetime import datetime

from config.config import config

LOG_DIR = config.LOG_DIR
os.makedirs(LOG_DIR, exist_ok=True)

_FORMATTER = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
current_log_path = os.path.join(
    LOG_DIR,
    f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
)


def _build_file_handler(path: str) -> logging.Handler:
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(_FORMATTER)
    return handler


def _build_stream_handler() -> logging.Handler:
    handler = logging.StreamHandler()
    handler.setFormatter(_FORMATTER)
    return handler


def setup_logger() -> logging.Logger:
    logger = logging.getLogger(config.APP_NAME)
    logger.setLevel(getattr(logging, config.LOG_LEVEL, logging.INFO))
    logger.propagate = False

    if not logger.handlers:
        logger.addHandler(_build_file_handler(current_log_path))
        logger.addHandler(_build_stream_handler())

    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    root_logger = setup_logger()
    if not name:
        return root_logger
    normalized = name.replace("/", ".").replace("\\", ".")
    if normalized == config.APP_NAME or normalized.startswith(f"{config.APP_NAME}."):
        return logging.getLogger(normalized)
    return logging.getLogger(f"{config.APP_NAME}.{normalized}")


def add_file_handler(path: str) -> logging.Handler:
    logger = setup_logger()
    handler = _build_file_handler(path)
    logger.addHandler(handler)
    return handler


sys_logger = setup_logger()
