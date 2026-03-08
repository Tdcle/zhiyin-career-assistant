import os
import sys
import json
import logging
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from datasets import Dataset
import warnings

# 忽略内部废弃警告
warnings.filterwarnings("ignore", category=DeprecationWarning)

# 将项目根目录加入 sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# ================= 1. 自动归档与全局日志配置 =================
EVAL_DIR = os.path.dirname(__file__)
TEST_FILE = os.path.join(EVAL_DIR, "synthetic_eval_dataset.json")

# 生成本次运行的独立文件夹
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = os.path.join(EVAL_DIR, "results", f"run_{timestamp}")
os.makedirs(RUN_DIR, exist_ok=True)


# 拦截系统标准输出 (print)，使其同时输出到控制台和日志文件
class OutputLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


log_file_path = os.path.join(RUN_DIR, "eval_process.log")
sys.stdout = OutputLogger(log_file_path)
sys.stderr = sys.stdout  # 捕获错误信息

# 尝试捕获 database.py 里的 sys_logger，使其搜索日志也落盘到同个 log 文件
try:
    from utils.logger import sys_logger

    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s'))
    sys_logger.addHandler(file_handler)
except ImportError:
    pass

from utils.database import DatabaseManager
from config.config import config
from langchain_community.chat_models import ChatTongyi
from langchain_ollama import OllamaEmbeddings
from langchain_core.messages import HumanMessage
from ragas import evaluate

# ================= 2. 兼容处理 RAGAS 的大版本更新 =================
try:
    from ragas.metrics import ContextPrecision, ContextRecall, Faithfulness, AnswerRelevancy

    eval_metrics = [ContextPrecision(), ContextRecall(), Faithfulness(), AnswerRelevancy()]
except ImportError:
    from ragas.metrics import context_precision, context_recall, faithfulness, answer_relevancy

    eval_metrics = [
        context_precision() if isinstance(context_precision, type) else context_precision,
        context_recall() if isinstance(context_recall, type) else context_recall,
        faithfulness() if isinstance(faithfulness, type) else faithfulness,
        answer_relevancy() if isinstance(answer_relevancy, type) else answer_relevancy,
    ]

try:
    from ragas.run_config import RunConfig

    run_config = RunConfig(max_workers=4)
except ImportError:
    run_config = None

# ================= 3. 初始化系统与模型 =================
db = DatabaseManager()

eval_llm = ChatTongyi(api_key=config.DASHSCOPE_API_KEY, model="qwen3-235b-a22b")
eval_embeddings = OllamaEmbeddings(base_url=config.OLLAMA_URL, model=config.EMBEDDING_MODEL_NAME)

try:
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    ragas_llm = LangchainLLMWrapper(eval_llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(eval_embeddings)
except ImportError:
    ragas_llm = eval_llm
    ragas_embeddings = eval_embeddings


def format_job_context(job: dict) -> str:
    """统一的职位信息格式化函数"""
    if not job:
        return ""
    welfare = job.get('welfare') or ''
    intro = job.get('summary') or job.get('detail') or ''
    return (
        f"🏢 **{job.get('company', '')}**（{job.get('industry', '')} | {job.get('city', '')} {job.get('district', '')}）\n"
        f"📌 职位：{job.get('title', '')}\n"
        f"💰 薪资：{job.get('salary', '')} · {welfare}\n"
        f"📋 要求：{job.get('degree', '')} / {job.get('experience', '')}\n"
        f"📝 概要：{intro}\n"
        f"🔗 链接：{job.get('detail_url', '')}"
    )


def generate_rag_answer(query: str, contexts: list):
    """简化的提示词，对齐生成数据集时的参考答案约束"""
    context_str = "\n\n---\n\n".join([f"【职位 {i + 1}】\n{ctx}" for i, ctx in enumerate(contexts)])
    prompt = f"""
    你是专业的求职顾问。请根据以下【检索到的真实职位信息】，回答【求职者问题】。

    【要求】：
    请客观分析求职者需求与该岗位的匹配度（指出匹配点和可能的不足）
    你的分析必须严谨，并且绝对不能编造职位信息中未提及的数据。
    回答简洁，只需要回答一小段话不分点，控制在200字以内。
    
    示例：
    

    【检索到的职位信息】：
    {context_str}

    【求职者问题】：
    {query}
    """
    try:
        response = eval_llm.invoke([HumanMessage(content=prompt)])
        return response.content
    except Exception as e:
        print(f"生成回答失败: {e}")
        return "生成失败"


def main():
    if not os.path.exists(TEST_FILE):
        print(f"找不到测试集: {TEST_FILE}")
        return

    with open(TEST_FILE, "r", encoding="utf-8") as f:
        synthetic_dataset = json.load(f)

    print(f"🚀 本次运行日志与结果将保存在: {RUN_DIR}")
    print("开始执行评测流水线 (分离 Broad Search 与 Precise Match)...")

    # 用于保存的综合报告字典
    report_data = {
        "timestamp": timestamp,
        "broad_search_metrics": {},
        "precise_match_metrics": {}
    }

    # 第一部分：纯检索评估指标与明细记录
    total_broad_queries = 0
    hits_at_5 = 0
    hits_at_10 = 0
    broad_search_records = []

    # 第二部分：RAGAS 生成评估数据集
    data_samples = {
        "question": [], "user_input": [],
        "answer": [], "response": [],
        "contexts": [], "retrieved_contexts": [],
        "ground_truth": [], "reference": []
    }

    for item in tqdm(synthetic_dataset, desc="运行系统流水线"):
        ground_truth_id = item["ground_truth_job_id"]
        gt_job = db.get_job_details(ground_truth_id)
        if not gt_job:
            continue

        for test_case in item["test_cases"]:
            query_type = test_case.get("query_type", "")
            query = test_case["user_question"]
            semantic_query = test_case.get("semantic_query", "")
            city = test_case.get("city", "")
            experience = test_case.get("experience", "")
            reference_answer = test_case.get("reference_answer", "")

            # ================= 第一部分: 泛化搜索 (评估纯数据库召回率) =================
            if query_type == "broad_search":
                total_broad_queries += 1

                # 检索 Top 10 用于计算 Recall
                retrieved_jobs = db.vector_search(
                    query_text=semantic_query,
                    city=city,
                    experience=experience,
                    top_k=10
                )

                retrieved_ids = [job.get("job_id") for job in retrieved_jobs]

                # 判断是否命中
                is_hit_5 = ground_truth_id in retrieved_ids[:5]
                is_hit_10 = ground_truth_id in retrieved_ids[:10]

                if is_hit_5: hits_at_5 += 1
                if is_hit_10: hits_at_10 += 1

                # 压入泛化搜索明细记录，供后续排查使用
                broad_search_records.append({
                    "user_question": query,
                    "semantic_query": semantic_query,
                    "city": city,
                    "experience": experience,
                    "ground_truth_id": ground_truth_id,
                    "hit_at_5": is_hit_5,
                    "hit_at_10": is_hit_10,
                    "retrieved_ids": ", ".join(retrieved_ids)
                })

            # ================= 第二部分: 精确匹配 (交由 RAGAS 评估大模型生成能力) =================
            elif query_type == "precise_match":
                retrieved_jobs = db.vector_search(
                    query_text=semantic_query,
                    city=city,
                    experience=experience,
                    top_k=5
                )

                contexts = [format_job_context(job) for job in retrieved_jobs]
                if not contexts:
                    contexts = ["未检索到任何职位"]

                # 生成回答
                answer = generate_rag_answer(query, contexts)

                # 压入 RAGAS 数据集
                data_samples["question"].append(query)
                data_samples["user_input"].append(query)
                data_samples["contexts"].append(contexts)
                data_samples["retrieved_contexts"].append(contexts)
                data_samples["answer"].append(answer)
                data_samples["response"].append(answer)
                data_samples["ground_truth"].append(reference_answer)
                data_samples["reference"].append(reference_answer)

    # ================= 打印并保存第一部分检索结果 =================
    print("\n" + "=" * 40)
    print("📊 第一部分：泛化搜索 (Broad Search) 纯检索评测")
    print("=" * 40)
    if total_broad_queries > 0:
        recall_5 = (hits_at_5 / total_broad_queries) * 100
        recall_10 = (hits_at_10 / total_broad_queries) * 100
        print(f"总测试问题数: {total_broad_queries}")
        print(f"🎯 Recall@5  : {recall_5:.2f}% ({hits_at_5}/{total_broad_queries})")
        print(f"🎯 Recall@10 : {recall_10:.2f}% ({hits_at_10}/{total_broad_queries})")

        # 存入报告字典
        report_data["broad_search_metrics"] = {
            "total_queries": total_broad_queries,
            "recall_5": round(recall_5, 2),
            "recall_10": round(recall_10, 2),
            "hits_at_5": hits_at_5,
            "hits_at_10": hits_at_10
        }
    else:
        print("⚠️ 未在数据集中找到 query_type='broad_search' 的用例")

    # ================= 执行并保存第二部分 RAGAS 评测 =================
    ragas_score_df = None
    if len(data_samples["question"]) > 0:
        hf_dataset = Dataset.from_dict(data_samples)

        print("\n🧠 第二部分：开始使用 RAGAS 进行生成质量打分 (Precise Match)...")
        try:
            evaluate_kwargs = {
                "dataset": hf_dataset,
                "metrics": eval_metrics,
                "llm": ragas_llm,
                "embeddings": ragas_embeddings
            }
            if run_config:
                evaluate_kwargs["run_config"] = run_config

            score = evaluate(**evaluate_kwargs)

            print("\n" + "=" * 40)
            print("🏆 RAGAS 精确人岗匹配 (Precise Match) 生成报告")
            print("=" * 40)

            score_dict = {}
            if hasattr(score, 'items'):
                score_dict = score
                if hasattr(score, 'to_pandas'):
                    ragas_score_df = score.to_pandas()
            elif hasattr(score, 'to_pandas'):
                ragas_score_df = score.to_pandas()
                numeric_df = ragas_score_df.select_dtypes(include=['number'])
                score_dict = numeric_df.mean().to_dict()
            else:
                try:
                    score_dict = dict(score)
                except:
                    pass

            # 打印提取出的分数并存入报告
            if score_dict:
                for metric_name, value in score_dict.items():
                    print(f"{metric_name.ljust(20)}: {value:.4f} ({value * 100:.2f}%)")
                    report_data["precise_match_metrics"][metric_name] = round(value, 4)

            print("=" * 40)

        except Exception as e:
            print(f"\nRAGAS 评测过程中出现错误: {e}")
            report_data["precise_match_metrics"]["error"] = str(e)
    else:
        print("\n⚠️ 未在数据集中找到 query_type='precise_match' 的用例，跳过 RAGAS 评测。")

    # ================= 结果与明细统一落盘保存 =================
    print(f"\n📁 正在打包写入所有明细至目录: {RUN_DIR}")

    # 1. 泛化搜索记录 CSV 落盘 (新增)
    if broad_search_records:
        broad_df = pd.DataFrame(broad_search_records)
        broad_csv_path = os.path.join(RUN_DIR, "broad_search_details.csv")
        broad_df.to_csv(broad_csv_path, index=False, encoding="utf-8-sig")
        print(f"✅ 泛化搜索明细已保存至: {broad_csv_path}")

    # 2. RAGAS 生成质量明细 CSV 落盘
    if ragas_score_df is not None:
        csv_path = os.path.join(RUN_DIR, "ragas_details.csv")
        ragas_score_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"✅ RAGAS评估明细已保存至: {csv_path}")

    # 3. 最终汇总 JSON 落盘
    report_path = os.path.join(RUN_DIR, "eval_report.json")
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, ensure_ascii=False, indent=4)
        print(f"✅ 核心评测报告已保存至: {report_path}")
    except Exception as e:
        print(f"❌ 报告保存失败: {e}")

    print(f"\n🎉 评测完成！请查阅文件夹: {RUN_DIR}")


if __name__ == "__main__":
    main()