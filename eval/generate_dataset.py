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
    model="deepseek-v3.2",
    temperature=0.7
)

OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "synthetic_eval_dataset.json")
SAMPLE_SIZE = 20  # 测试通过后可调大


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
    """【修改点】全维度参数提取，对齐 V5.0 极简对称检索架构"""
    system_prompt = """
    你是一个资深的招聘专家和数据标注工程师。
    我将提供一个真实的职位信息。请你模拟真实的求职者，为该职位生成 2 个完全不同场景的求职提问，并提取出对应的全维度检索参数。

    请严格输出 JSON 数组格式，不要包含 Markdown 标记代码块。
    数组必须包含且仅包含 2 个对象，分别对应以下两种 query_type：

    【Type 1: 泛化搜索 (broad_search)】
    - 场景：用户只想找某一类工作，没有特定公司偏好。
    - user_question 示例："我想找北京的前端开发的实习，要求本科，期望薪资150一天左右，会Vue，有双休吗？"
    - 约束：user_question 中绝对不能出现公司名称。"reference_answer" 字段请直接留空字符串 ""。

    【Type 2: 精确人岗匹配 (precise_match)】
    - 场景：用户看中了特定公司，询问自身条件是否匹配。
    - user_question 示例："我想去[公司名称]，我是大专学历，会[技能]，有[X年]经验，期望薪资[XX]，这个适合我吗？"
    - 约束：user_question 必须明确包含公司名称。
    - "reference_answer" 约束：你必须客观分析用户需求与该岗位的匹配度（指出匹配点和可能的不足），作为标准答案，写成一段话150字以内。

    每个对象必须包含以下 11 个字段。如果 user_question 中没有提及某项信息，对应的字段必须留空字符串 ""：
    1. "query_type": "broad_search" 或 "precise_match"
    2. "user_question": 模拟真实求职者的自然语言提问。
    3. "title": 提取的岗位（如：前端开发）。
    3. "semantic_query": 技术栈词汇（如：Vue、CSS，绝不能出现公司名称）。
    4. "city": 提取的城市（如：北京）。
    5. "experience": 提取用户的经验或工作性质（如：3-5年、实习、应届）。
    6. "company": 提取的公司名称（仅 precise_match 场景可能有）。
    7. "salary": 提取的薪资诉求（如：15K、150元/天）。
    8. "degree": 提取的学历要求（如：本科、大专）。
    9. "welfare": 提取的福利诉求（如：双休、五险一金）。
    10. "reference_answer": broad_search 留空；precise_match 写匹配度客观分析。
    """

    # 【优化点】在喂给大模型的岗位信息中，补全 degree 和 welfare
    human_prompt = f"""
    【职位信息】
    - 岗位名称：{job.get('title')}
    - 公司：{job.get('company')}
    - 城市：{job.get('city')}
    - 经验要求：{job.get('experience')}
    - 学历要求：{job.get('degree', '未说明')}
    - 薪资：{job.get('salary')}
    - 福利：{job.get('welfare', '未说明')}
    - 职位摘要：{job.get('summary')}
    """

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ]

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