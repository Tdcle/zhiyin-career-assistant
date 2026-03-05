import os
import sys
import json
import re
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.database import DatabaseManager
from config.config import config
from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import HumanMessage, SystemMessage

# ================= 1. 初始化 =================
db = DatabaseManager()
llm = ChatTongyi(
    api_key=config.DASHSCOPE_API_KEY,
    model="qwen-max",
    temperature=0.7
)

OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "synthetic_eval_dataset.json")
SAMPLE_SIZE = 5  # 测试通过后可调大


def fetch_random_jobs(limit=50):
    print(f"🔄 正在从数据库随机抽取 {limit} 条职位数据...")
    try:
        with db.get_cursor(dict_cursor=True) as cur:
            sql = """
                SELECT job_id, title, company, city, experience, salary, summary 
                FROM jobs 
                WHERE summary IS NOT NULL AND summary != ''
                ORDER BY RANDOM() 
                LIMIT %s
            """
            cur.execute(sql, (limit,))
            jobs = cur.fetchall()
            print(f"✅ 成功抽取 {len(jobs)} 条职位数据")
            return jobs
    except Exception as e:
        print(f"❌ 抽样失败: {e}")
        return []


def generate_queries_for_job(job):
    """生成两种不同意图的测试数据"""
    system_prompt = """
    你是一个资深的招聘专家和数据标注工程师。
    我将提供一个真实的职位信息。请你模拟真实的求职者，为该职位生成 2 个完全不同场景的求职提问，并提取对应的检索参数。

    请严格输出 JSON 数组格式，不要包含 Markdown 标记代码块。
    数组必须包含且仅包含 2 个对象，分别对应以下两种 query_type：

    【Type 1: 泛化搜索 (broad_search)】
    - 场景：用户只想找某一类工作，没有特定公司偏好。
    - user_question 示例："我想找北京的前端开发的实习，期望薪资150一天左右，会Vue，有什么推荐吗？"
    - 约束：user_question 中 **绝对不能** 出现公司名称。

    【Type 2: 精确人岗匹配 (precise_match)】
    - 场景：用户看中了特定公司，询问自身条件是否匹配。
    - user_question 示例："我想去[公司名称]，我会[技能]，有[X年]经验，这个适合我吗？"
    - 约束：user_question 中 **必须明确包含** 该职位的公司名称。

    每个对象必须包含以下字段：
    1. "query_type": "broad_search" 或 "precise_match"
    2. "user_question": 模拟真实求职者的自然语言提问（根据上述要求）。
    3. "semantic_query": 从提问中提炼的核心语义检索词，用空格隔开。若是 precise_match，必须把公司名也放进来（如：腾讯 前端 Vue）。
    4. "city": 提取的城市（如：北京。若无则为空 ""）
    5. "experience": 提取的经验要求（如：1-3年、实习。若无则为空 ""）
    6. "reference_answer": 作为 AI 助手的标准完美回答（Ground Truth）。客观分析用户需求与该岗位的匹配度并进行推荐。
    """

    human_prompt = f"""
    【职位信息】
    - 岗位名称：{job.get('title')}
    - 公司：{job.get('company')}
    - 城市：{job.get('city')}
    - 经验要求：{job.get('experience')}
    - 薪资：{job.get('salary')}
    - 职位摘要：{job.get('summary')}
    """

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ]

    content = ""
    try:
        response = llm.invoke(messages)
        content = response.content.strip()

        md_block_marker = "`" * 3
        if md_block_marker in content:
            pattern = md_block_marker + r"(?:json)?\s*(.*?)" + md_block_marker
            match = re.search(pattern, content, re.DOTALL)
            if match:
                content = match.group(1).strip()

        queries = json.loads(content)
        return queries
    except Exception as e:
        print(f"\n⚠️ 生成失败: {e}")
        return []


def main():
    jobs = fetch_random_jobs(limit=SAMPLE_SIZE)
    if not jobs:
        return

    dataset = []
    print("🚀 开始大模型逆向生成 RAGAS 标准测试集 (区分意图)...")
    for job in tqdm(jobs, desc="生成进度"):
        generated_queries = generate_queries_for_job(job)
        if generated_queries:
            dataset_item = {
                "ground_truth_job_id": job['job_id'],
                "job_meta": {
                    "title": job['title'],
                    "company": job['company'],
                    "city": job['city']
                },
                "test_cases": generated_queries
            }
            dataset.append(dataset_item)

    print(f"\n💾 正在保存测试集到 {OUTPUT_FILE} ...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)
    print("🎉 包含 [泛化搜索] 与 [精确匹配] 的双场景数据集生成完毕！")


if __name__ == "__main__":
    main()