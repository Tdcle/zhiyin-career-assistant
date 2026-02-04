# utils/logger.py
import logging
import os
from datetime import datetime

# 配置日志目录
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

def setup_logger(name="JobAgent"):
    """
    初始化并返回一个 Logger 对象
    """
    # 生成带时间戳的文件名
    log_filename = f"{LOG_DIR}/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # 防止重复添加 Handler (避免日志重复打印)
    if not logger.handlers:
        # 1. 文件输出 (详细记录)
        file_handler = logging.FileHandler(log_filename, encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # 2. 控制台输出 (开发调试用)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger, log_filename

# ==========================================
# 【新增】这里直接初始化一个全局实例
# 这样其他文件只需要: from utils.logger import sys_logger 即可使用
# ==========================================
sys_logger, current_log_path = setup_logger()