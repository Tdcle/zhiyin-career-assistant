import os
import sys
import json
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

from utils.database import DatabaseManager
from config.config import config
from langchain_community.chat_models import ChatTongyi
from langchain_ollama import OllamaEmbeddings
from langchain_core.messages import HumanMessage

from ragas import evaluate

# ================= 兼容处理 RAGAS 的大版本更新 =================
# 1. 兼容指标类导入 (新版要求大写类名并实例化)
try:
    from ragas.metrics import ContextPrecision, ContextRecall, Faithfulness, AnswerRelevancy

    eval_metrics = [ContextPrecision(), ContextRecall(), Faithfulness(), AnswerRelevancy()]
except ImportError:
    from ragas.metrics import context_precision, context_recall, faithfulness, answer_relevancy

    # 动态判断：如果是类就加上 () 实例化
    eval_metrics = [
        context_precision() if isinstance(context_precision, type) else context_precision,
        context_recall() if isinstance(context_recall, type) else context_recall,
        faithfulness() if isinstance(faithfulness, type) else faithfulness,
        answer_relevancy() if isinstance(answer_relevancy, type) else answer_relevancy,
    ]

# 尝试导入 RunConfig 以限制并发防止超时
try:
    from ragas.run_config import RunConfig

    run_config = RunConfig(max_workers=4)
except ImportError:
    run_config = None

# ================= 1. 初始化系统 =================
db = DatabaseManager()
EVAL_DIR = os.path.dirname(__file__)
TEST_FILE = os.path.join(EVAL_DIR, "synthetic_eval_dataset.json")

# 【新增】：结果保存目录
RESULTS_DIR = os.path.join(EVAL_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

eval_llm = ChatTongyi(api_key=config.DASHSCOPE_API_KEY, model="qwen-max")
eval_embeddings = OllamaEmbeddings(base_url=config.OLLAMA_URL, model=config.EMBEDDING_MODEL_NAME)

# 2. 兼容模型封装 (新版 RAGAS 要求使用 Wrapper)
try:
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    ragas_llm = LangchainLLMWrapper(eval_llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(eval_embeddings)
except ImportError:
    ragas_llm = eval_llm
    ragas_embeddings = eval_embeddings


def format_job_context(job: dict) -> str:
    """统一的职位信息格式化函数，确保评测环境与真实链路一致"""
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
    """模拟真实的 RAG 生成环节"""
    context_str = "\n\n---\n\n".join([f"【职位 {i + 1}】\n{ctx}" for i, ctx in enumerate(contexts)])

    prompt = f"""
    你是专业的求职顾问。请根据以下检索到的职位信息，回答求职者的问题。
    如果检索到的信息无法回答，请直接说不知道，禁止编造。

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

    print("开始执行评测流水线 (分离 Broad Search 与 Precise Match)...")

    # 用于保存的综合报告字典
    report_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "broad_search_metrics": {},
        "precise_match_metrics": {}
    }

    # 第一部分：纯检索评估指标记录
    total_broad_queries = 0
    hits_at_5 = 0
    hits_at_10 = 0

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
                if ground_truth_id in retrieved_ids[:5]:
                    hits_at_5 += 1
                if ground_truth_id in retrieved_ids[:10]:
                    hits_at_10 += 1

            # ================= 第二部分: 精确匹配 (交由 RAGAS 评估大模型生成能力) =================
            elif query_type == "precise_match":
                # 检索 (Top 5 即可，模拟真实推荐数量)
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
            # 加入 run_config 控制并发，防止 TimeoutError
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

            # 安全地从 EvaluationResult 对象中提取分数
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
                    print(f"原始打分对象: {score}")

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

    # ================= 结果落盘保存 =================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. 保存汇总 JSON 报告
    report_path = os.path.join(RESULTS_DIR, f"eval_report_{timestamp}.json")
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, ensure_ascii=False, indent=4)
        print(f"\n✅ 评测汇总报告已保存至: {report_path}")
    except Exception as e:
        print(f"❌ 报告保存失败: {e}")

    # 2. 保存 RAGAS 每条测试用例的得分明细 CSV
    if ragas_score_df is not None:
        csv_path = os.path.join(RESULTS_DIR, f"ragas_details_{timestamp}.csv")
        try:
            # utf-8-sig 可以防止用 Excel 打开中文乱码
            ragas_score_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
            print(f"✅ RAGAS明细数据已保存至: {csv_path}")
        except Exception as e:
            print(f"❌ 明细数据保存失败: {e}")


if __name__ == "__main__":
    main()