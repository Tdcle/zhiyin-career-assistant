import os
import sys
import json
import re
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from backend.db import DatabaseManager
from backend.config.config import config
from langchain_core.messages import HumanMessage, SystemMessage
from backend.utils.logger import get_logger

# ================= 1. 初始化 =================
db = DatabaseManager()
logger = get_logger("eval.generate_dataset")
llm = config.create_tongyi(config.CHAT_MODELS.eval_dataset_generator, temperature=0.7)

OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "synthetic_eval_dataset.json")
SAMPLE_SIZE = 25  # 测试通过后可调大

SUPPORTED_CITIES = [
    "北京", "上海", "深圳", "广州", "杭州", "成都",
    "武汉", "西安", "南京", "苏州", "天津", "重庆",
]
CANONICAL_EXPERIENCE_VALUES = {
    "",
    "实习",
    "应届",
    "经验不限",
    "1年以内",
    "1-3年",
    "3-5年",
    "5-10年",
    "10年以上",
}


def fetch_random_jobs(limit=50):
    logger.info("sampling %s jobs for synthetic dataset", limit)
    try:
        with db.get_cursor(dict_cursor=True) as cur:
            sql = """
                SELECT job_id, title, company, city, experience, salary, degree, welfare, summary 
                FROM jobs 
                WHERE summary IS NOT NULL AND summary != ''
                ORDER BY RANDOM() 
                LIMIT %s
            """
            cur.execute(sql, (limit,))
            jobs = cur.fetchall()
            logger.info("sampled %s jobs", len(jobs))
            return jobs
    except Exception as e:
        logger.error("dataset sampling failed: %s", e, exc_info=True)
        return []


def normalize_city(city: str, *texts: str) -> str:
    raw = (city or "").strip()
    combined = " ".join(part for part in (raw, *texts) if part)
    if any(token in combined for token in ["全国", "不限城市", "城市不限", "远程"]):
        return ""
    for candidate in SUPPORTED_CITIES:
        if candidate in combined:
            return candidate
    return raw if raw in SUPPORTED_CITIES else ""


def normalize_experience(experience: str, *texts: str) -> str:
    combined = " ".join(part for part in (experience, *texts) if part).strip().lower()
    raw = (experience or "").strip()
    if not combined:
        return ""
    if any(token in combined for token in ["实习", "在校", "intern"]):
        return "实习"
    if any(token in combined for token in ["应届", "校招", "毕业生", "new grad"]):
        return "应届"
    if raw in CANONICAL_EXPERIENCE_VALUES:
        return raw
    if re.search(r"\d+\s*年", raw):
        return ""
    return ""


def normalize_salary(salary: str, *texts: str) -> str:
    combined = " ".join(part for part in (salary, *texts) if part).strip()
    if not combined:
        return ""
    monthly_match = re.search(r"(\d{1,2})(?:\s*[kK]|千)", combined)
    if monthly_match:
        return f"{int(monthly_match.group(1))}K"
    daily_match = re.search(r"(\d{2,4})\s*元\s*/?\s*天", combined)
    if daily_match:
        return f"{int(daily_match.group(1))}元/天"
    return ""


def normalize_query_text(text: str) -> str:
    if not text:
        return ""
    normalized = re.sub(r"[\/|,，、;；]+", " ", text)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def build_semantic_query(title: str, semantic_query: str, city: str, company: str) -> str:
    merged = normalize_query_text(f"{title} {semantic_query}")
    if company:
        merged = merged.replace(company, " ")
    if city:
        merged = merged.replace(city, " ")
    merged = normalize_query_text(merged)
    return merged


def sanitize_test_case(raw_case: dict, job: dict) -> dict:
    query_type = raw_case.get("query_type", "")
    title = normalize_query_text(raw_case.get("title", "") or job.get("title", ""))
    city = normalize_city(raw_case.get("city", ""), raw_case.get("user_question", ""), job.get("city", ""))
    company = (raw_case.get("company", "") or "").strip()
    if query_type == "broad_search":
        company = ""
    elif not company:
        company = (job.get("company", "") or "").strip()

    semantic_query = build_semantic_query(
        title=title,
        semantic_query=raw_case.get("semantic_query", ""),
        city=city,
        company=company,
    )
    experience = normalize_experience(
        raw_case.get("experience", ""),
        raw_case.get("user_question", ""),
        job.get("experience", ""),
    )
    salary = normalize_salary(raw_case.get("salary", ""), raw_case.get("user_question", ""))
    degree = (raw_case.get("degree", "") or "").strip()
    welfare = normalize_query_text(raw_case.get("welfare", ""))
    reference_answer = (raw_case.get("reference_answer", "") or "").strip()

    if query_type == "broad_search":
        reference_answer = ""

    return {
        "query_type": query_type,
        "user_question": (raw_case.get("user_question", "") or "").strip(),
        "title": title,
        "semantic_query": semantic_query,
        "city": city,
        "experience": experience,
        "company": company,
        "salary": salary,
        "degree": degree,
        "welfare": welfare,
        "reference_answer": reference_answer,
    }


def build_fallback_cases(job: dict) -> list[dict]:
    city = normalize_city(job.get("city", ""))
    title = normalize_query_text(job.get("title", ""))
    company = (job.get("company", "") or "").strip()
    experience = normalize_experience(job.get("experience", ""))
    salary = normalize_salary(job.get("salary", ""))
    degree = (job.get("degree", "") or "").strip()
    welfare = normalize_query_text(job.get("welfare", ""))
    semantic_query = build_semantic_query(title=title, semantic_query="", city=city, company=company)

    broad_question = f"我想找{city}的{title}岗位，有什么推荐吗？" if city else f"我想找{title}岗位，有什么推荐吗？"
    precise_question = f"我想去{company}做{title}，这个岗位适合我吗？" if company else f"我想做{title}，这个岗位适合我吗？"
    precise_answer = "匹配度需要结合你的学历、经验和技能进一步判断，但从目标岗位名称看，方向是对口的，建议重点核对岗位要求里的经验、技能和城市条件。"

    return [
        {
            "query_type": "broad_search",
            "user_question": broad_question,
            "title": title,
            "semantic_query": semantic_query,
            "city": city,
            "experience": experience if experience in {"实习", "应届"} else "",
            "company": "",
            "salary": salary,
            "degree": degree,
            "welfare": welfare,
            "reference_answer": "",
        },
        {
            "query_type": "precise_match",
            "user_question": precise_question,
            "title": title,
            "semantic_query": semantic_query,
            "city": city,
            "experience": experience,
            "company": company,
            "salary": salary,
            "degree": degree,
            "welfare": welfare,
            "reference_answer": precise_answer,
        },
    ]


def postprocess_generated_cases(raw_cases: list, job: dict) -> list[dict]:
    sanitized = []
    seen_types = set()

    for raw_case in raw_cases or []:
        query_type = raw_case.get("query_type", "")
        if query_type not in {"broad_search", "precise_match"} or query_type in seen_types:
            continue
        sanitized_case = sanitize_test_case(raw_case, job)
        if not sanitized_case["user_question"] or not sanitized_case["semantic_query"]:
            continue
        sanitized.append(sanitized_case)
        seen_types.add(query_type)

    if len(sanitized) == 2:
        return sanitized

    fallback_cases = build_fallback_cases(job)
    for case in fallback_cases:
        if case["query_type"] not in seen_types:
            sanitized.append(case)
            seen_types.add(case["query_type"])

    return sanitized[:2]


def generate_queries_for_job(job):
    """生成更贴近当前检索接口约束的评测样本。"""
    system_prompt = """
    你是一个资深的招聘专家和数据标注工程师。
    我将提供一个真实的职位信息。请你模拟真实的求职者，为该职位生成 2 个完全不同场景的求职提问，并提取出对应的检索参数。

    请严格输出 JSON 数组格式，不要包含 Markdown 标记代码块。
    数组必须包含且仅包含 2 个对象，分别对应以下两种 query_type：

    【Type 1: 泛化搜索 (broad_search)】
    - 场景：用户只想找某一类工作，没有特定公司偏好。
    - user_question 示例："我想找北京的前端开发工作，会Vue和CSS，有什么推荐吗？"
    - 约束：user_question 中绝对不能出现公司名称。"reference_answer" 字段请直接留空字符串 ""。

    【Type 2: 精确人岗匹配 (precise_match)】
    - 场景：用户看中了特定公司，询问自身条件是否匹配。
    - user_question 示例："我想去[公司名称]做[岗位]，我有相关经验和技能，这个适合我吗？"
    - 约束：user_question 必须明确包含公司名称。
    - "reference_answer" 约束：你必须客观分析用户需求与该岗位的匹配度（指出匹配点和可能的不足），作为标准答案，写成一段话150字以内。

    每个对象必须包含以下 11 个字段。如果 user_question 中没有提及某项信息，对应的字段必须留空字符串 ""：
    1. "query_type": "broad_search" 或 "precise_match"
    2. "user_question": 模拟真实求职者的自然语言提问。
    3. "title": 岗位名称，尽量与职位名称一致或接近。
    4. "semantic_query": 用于数据库检索的短关键词，必须是空格分隔的短词，通常由岗位名 + 技能词组成，例如 "Java 实习 Spring Boot"；不能写完整句子，不能包含公司名称。
    5. "city": 只能填单个标准城市名，如 "北京"、"上海"；不要写“北京及周边”“全国范围内”。
    6. "experience": 只能填以下之一：""、"实习"、"应届"、"经验不限"、"1年以内"、"1-3年"、"3-5年"、"5-10年"、"10年以上"。
       如果用户问题里只是“2年经验”“4年经验”这类自由描述，不要填进 experience，直接留空字符串。
    7. "company": precise_match 必须包含公司名；broad_search 必须为空字符串。
    8. "salary": 只能填标准格式，如 "15K"、"25K"、"150元/天"；不要写“7千到8千左右”这类长表达。
    9. "degree": 学历短词，如 "本科"、"大专"、"硕士"。
    10. "welfare": 福利短词，如 "双休"、"五险一金"。
    11. "reference_answer": broad_search 留空；precise_match 写匹配度客观分析。

    额外要求：
    1. broad_search 的样本要优先测试数据库检索能力，因此 user_question 和 semantic_query 要围绕岗位名、技能词、城市展开，不要堆砌过多限制。
    2. precise_match 的样本要优先测试人岗匹配，因此 semantic_query 仍然应该是可检索短词，而不是完整自然语言。
    3. 如果岗位本身不是实习或校招，不要强行生成“实习”“应届”型样本。
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
        return postprocess_generated_cases(queries, job)
    except Exception as e:
        logger.error("query generation failed: %s", e, exc_info=True)
        return build_fallback_cases(job)


def main():
    jobs = fetch_random_jobs(limit=SAMPLE_SIZE)
    if not jobs:
        return

    dataset = []
    logger.info("starting synthetic dataset generation")
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

    logger.info("saving dataset to %s", OUTPUT_FILE)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)
    logger.info("synthetic dataset generation completed")


if __name__ == "__main__":
    main()
