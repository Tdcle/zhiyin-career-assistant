import os
import sys
import json
from tqdm import tqdm

# 将项目根目录加入 sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.database import DatabaseManager

# 初始化数据库
db = DatabaseManager()
TEST_FILE = os.path.join(os.path.dirname(__file__), "synthetic_eval_dataset.json")


def evaluate():
    if not os.path.exists(TEST_FILE):
        print(f"❌ 找不到测试集文件: {TEST_FILE}")
        return

    with open(TEST_FILE, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    total_queries = 0
    # 统计 Recall@5 和 Recall@10
    hits_at_5 = 0
    hits_at_10 = 0

    print(f"🚀 开始自动化评测 RAG 检索系统 (共 {len(dataset)} 个职位)...")

    # 我们不仅看整体，还可以按 query_type 分类看效果
    type_stats = {
        "keyword": {"total": 0, "hits": 0},
        "natural_language": {"total": 0, "hits": 0},
        "fuzzy": {"total": 0, "hits": 0}
    }

    for item in tqdm(dataset, desc="评测中"):
        ground_truth_id = item["ground_truth_job_id"]

        for test_case in item["test_cases"]:
            query_type = test_case["query_type"]
            query_text = test_case["query_text"]
            total_queries += 1
            type_stats[query_type]["total"] += 1

            # 调用你的核心混合检索方法 (获取前 10 名)
            # 注意：这里我们故意不传 city 和 experience，直接用用户的原始自然语言去考量向量+正则的硬实力
            results = db.vector_search(query_text=query_text, top_k=10)

            # 提取返回结果中的 job_id 列表
            retrieved_ids = [res["job_id"] for res in results]

            # 检查命中情况
            is_hit = False
            if ground_truth_id in retrieved_ids[:5]:
                hits_at_5 += 1
                is_hit = True

            if ground_truth_id in retrieved_ids[:10]:
                hits_at_10 += 1
                if not is_hit:  # 只有在前10命中但没在前5命中时，才算分类的命中(为了简化逻辑，分类我们只看 Top 10)
                    pass
                type_stats[query_type]["hits"] += 1
            elif is_hit:
                type_stats[query_type]["hits"] += 1

    # 打印超级震撼的评测报告
    print("\n" + "=" * 40)
    print("📊 RAG 检索系统评测报告 (RRF 混合检索)")
    print("=" * 40)
    print(f"总测试问题数: {total_queries}")

    recall_5 = (hits_at_5 / total_queries) * 100
    recall_10 = (hits_at_10 / total_queries) * 100

    print(f"🎯 Recall@5  (前 5 名命中率): {recall_5:.2f}%")
    print(f"🎯 Recall@10 (前10名命中率): {recall_10:.2f}%")
    print("-" * 40)

    print("📈 按查询类型细分 (Recall@10):")
    for q_type, stats in type_stats.items():
        if stats["total"] > 0:
            rate = (stats["hits"] / stats["total"]) * 100
            print(f"   - {q_type.ljust(16)}: {rate:.2f}% ({stats['hits']}/{stats['total']})")

    print("=" * 40)


if __name__ == "__main__":
    evaluate()