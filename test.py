import os
# ========================================================
# 【关键修复】强制不使用代理
# 这行代码必须放在 import langchain 之前，或者是代码的最开始
os.environ["NO_PROXY"] = "localhost,127.0.0.1"
# ========================================================

from langchain_ollama import OllamaEmbeddings

# 1. 配置
MODEL_NAME = "nomic-embed-text-v2-moe"
# 建议：把 localhost 显式改成 127.0.0.1，避免 IPv6 解析问题
OLLAMA_URL = "http://127.0.0.1:11434"

print(f"正在连接 Ollama: {OLLAMA_URL} ...")
print(f"使用的模型: {MODEL_NAME}")

try:
    # 2. 初始化
    embeddings = OllamaEmbeddings(
        base_url=OLLAMA_URL,
        model=MODEL_NAME
    )

    # 3. 测试
    print("开始生成向量...")
    vector = embeddings.embed_query("你好，世界")

    print("✅ 测试成功！")
    print(f"向量维度: {len(vector)}")
    print(f"前5位数据: {vector[:5]}")

except Exception as e:
    print("\n❌ 测试失败！")
    print(f"错误信息: {e}")