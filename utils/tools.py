import json
import os
import re

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder

from config.config import config
from utils.database import DatabaseManager
from utils.logger import get_logger
from utils.plotter import create_radar_chart

logger = get_logger("tools")
db = DatabaseManager()
llm = config.create_ollama_chat(config.OLLAMA_MODELS.match_analysis, temperature=0.1)

logger.info("loading reranker model from %s", config.RERANKER_MODEL_PATH)
try:
    reranker = CrossEncoder(config.RERANKER_MODEL_PATH, device="cpu")
    logger.info("reranker model loaded")
except Exception as exc:
    logger.warning("reranker unavailable, fallback to plain retrieval: %s", exc)
    reranker = None


class SearchInput(BaseModel):
    resolved_query: str = Field(..., description="用户的完整中文求职意图，尽量保留原话。")
    keyword_query: str = Field(default="", description="用于数据库检索的核心中文关键词，多个词用空格分隔。")
    city: str = Field(default="", description="城市硬过滤条件，如 北京、上海、深圳。")
    company: str = Field(default="", description="公司过滤条件；没有明确提及时留空。")
    experience: str = Field(default="", description="经验或岗位类型，如 实习、应届、3-5年。")
    salary_min: int = Field(default=0, description="最低期望薪资数值；没有明确要求时填 0。")
    salary_unit: str = Field(default="", description="薪资单位，只能是 k、元/天 或空字符串。")


class PreferenceInput(BaseModel):
    user_id: str = Field(description="用户 ID")
    preference: str = Field(description="要写入长期画像的偏好或个人事实，使用中文自然语言。")


class GetResumeInput(BaseModel):
    user_id: str = Field(description="用户 ID")


class MatchInput(BaseModel):
    user_id: str = Field(description="用户 ID")
    job_description: str = Field(description="目标岗位 JD 原文，优先使用中文。")


class MatchByQueryInput(BaseModel):
    user_id: str = Field(description="用户 ID")
    target_query: str = Field(description="用户提到的目标岗位、公司名或 JD 片段，尽量保留中文表述。")
    city: str = Field(default="", description="可选城市过滤条件。")
    company: str = Field(default="", description="可选公司过滤条件。")


def _infer_memory_type(preference: str) -> str:
    text = (preference or "").lower()
    if any(token in text for token in ["简历", "项目", "经历", "负责", "做过"]):
        return "experience"
    if any(token in text for token in ["城市", "北京", "上海", "深圳", "杭州", "远程", "remote"]):
        return "location_preference"
    if any(token in text for token in ["薪资", "月薪", "年薪", "k"]):
        return "salary_preference"
    if any(token in text for token in ["本科", "硕士", "学历", "大专", "博士"]):
        return "education"
    if any(token in text for token in ["python", "java", "react", "vue", "golang", "测试", "运维", "前端", "后端"]):
        return "skill"
    return "preference"


def _safe_json_text(raw: str) -> dict:
    text = (raw or "").strip()
    if "```" in text:
        match = re.search(r"```(?:json)?(.*?)```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        text = text[start:end + 1]
    return json.loads(text)


def _build_effective_search_query(
    resolved_query: str,
    keyword_query: str,
    experience: str,
    salary_min: int,
    salary_unit: str,
) -> str:
    parts = [part.strip() for part in [keyword_query, resolved_query] if part and part.strip()]
    query = parts[0] if parts else ""

    additions = []
    if experience == "实习":
        additions.extend(["实习", "在校"])
    elif experience == "应届":
        additions.extend(["应届", "校招"])
    elif experience:
        additions.append(experience)

    if salary_min and salary_unit == "k_month":
        additions.append(f"{salary_min}k")
    elif salary_min and salary_unit == "yuan_day":
        additions.append(f"{salary_min}元/天")

    for token in additions:
        if token and token.lower() not in query.lower():
            query = f"{query} {token}".strip()

    return query or resolved_query.strip()


SUPPORTED_CITIES = [
    "北京",
    "上海",
    "深圳",
    "广州",
    "杭州",
    "成都",
    "武汉",
    "西安",
    "南京",
    "苏州",
    "天津",
    "重庆",
]


def _normalize_city(city: str, *texts: str) -> str:
    raw = (city or "").strip()
    combined = " ".join(part for part in (raw, *texts) if part)

    if any(token in combined for token in ["全国", "不限城市", "城市不限"]):
        return ""

    for candidate in SUPPORTED_CITIES:
        if candidate in combined:
            return candidate
    return raw


def _normalize_experience(experience: str, *texts: str) -> str:
    combined = " ".join(part for part in (experience, *texts) if part)
    if not combined.strip():
        return ""

    normalized = combined.lower()
    if any(token in normalized for token in ["实习", "在校", "intern"]):
        return "实习"
    if any(token in normalized for token in ["应届", "校招", "毕业生", "new grad"]):
        return "应届"

    parts = [part.strip() for part in re.split(r"[\/|,，、;；\s]+", experience or "") if part.strip()]
    return parts[0] if parts else (experience or "").strip()


def _normalize_keyword_text(text: str) -> str:
    if not text:
        return ""
    normalized = re.sub(r"[\/|,，、;；]+", " ", text)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


ROLE_KEYWORD_GROUPS = {
    "前端": ["前端", "前端开发", "web前端", "frontend", "front-end", "front end"],
    "后端": ["后端", "后端开发", "服务端", "服务端开发", "backend", "back-end"],
    "全栈": ["全栈", "全栈开发", "fullstack", "full-stack", "full stack"],
    "测试": ["测试", "测试开发", "软件测试", "qa"],
    "运维": ["运维", "devops", "sre"],
    "产品": ["产品经理", "产品", "product manager", "pm"],
    "运营": ["运营", "内容运营", "用户运营", "产品运营"],
    "设计": ["设计", "ui", "ux", "交互设计", "视觉设计"],
    "算法": ["算法", "算法工程师"],
    "数据分析": ["数据分析", "商业分析", "bi"],
    "数据开发": ["数据开发", "大数据", "数仓", "etl"],
    "Java": ["java"],
    "Python": ["python"],
    "Golang": ["golang", "go开发", "go语言"],
    "C++": ["c++"],
    "C#": ["c#", ".net", "dotnet"],
    "PHP": ["php"],
    "Android": ["android", "安卓"],
    "iOS": ["ios"],
    "客户端": ["客户端"],
    "AI": ["ai", "人工智能", "大模型", "机器学习", "深度学习", "nlp", "自然语言处理"],
}

GENERIC_SEARCH_TERMS = {
    "实习",
    "应届",
    "校招",
    "社招",
    "全职",
    "兼职",
    "远程",
    "双休",
    "五险一金",
    "k",
}


def _extract_role_keywords(*texts: str) -> list[str]:
    combined = " ".join(text for text in texts if text).lower()
    if not combined:
        return []

    matched = []
    for canonical, aliases in ROLE_KEYWORD_GROUPS.items():
        if any(alias.lower() in combined for alias in aliases):
            matched.append(canonical)
    return matched


def _title_match_score(title: str, role_keywords: list[str]) -> int:
    normalized_title = (title or "").lower()
    if not normalized_title or not role_keywords:
        return 0

    score = 0
    for canonical in role_keywords:
        aliases = ROLE_KEYWORD_GROUPS.get(canonical, [])
        if any(alias.lower() in normalized_title for alias in aliases):
            score += 1
    return score


def _filter_candidates_by_title(candidates: list[dict], role_keywords: list[str]) -> list[dict]:
    if not role_keywords:
        return candidates

    filtered = []
    for row in candidates:
        title_score = _title_match_score(row.get("title", ""), role_keywords)
        if title_score <= 0:
            continue
        enriched = dict(row)
        enriched["title_match_score"] = title_score
        filtered.append(enriched)

    filtered.sort(
        key=lambda item: (
            item.get("title_match_score", 0),
            item.get("rrf_score", 0),
            item.get("vec_score", 0),
            item.get("bm25_score", 0),
        ),
        reverse=True,
    )
    return filtered


@tool("search_jobs_tool", args_schema=SearchInput)
def search_jobs_tool(
    resolved_query: str,
    keyword_query: str = "",
    city: str = "",
    company: str = "",
    experience: str = "",
    salary_min: int = 0,
    salary_unit: str = "",
):
    """按中文岗位意图检索职位列表，并返回面向前端展示的候选结果。"""
    normalized_city = _normalize_city(city, resolved_query, keyword_query)
    normalized_experience = _normalize_experience(experience, resolved_query, keyword_query)
    normalized_keyword_query = _normalize_keyword_text(keyword_query)
    normalized_resolved_query = _normalize_keyword_text(resolved_query)

    effective_query = _build_effective_search_query(
        resolved_query=normalized_resolved_query,
        keyword_query=normalized_keyword_query,
        experience=normalized_experience,
        salary_min=salary_min,
        salary_unit=salary_unit,
    )
    logger.info(
        "search tool invoked: query=%s effective_query=%s city=%s->%s company=%s experience=%s->%s salary_min=%s salary_unit=%s",
        resolved_query,
        effective_query,
        city,
        normalized_city,
        company,
        experience,
        normalized_experience,
        salary_min,
        salary_unit,
    )
    role_keywords = _extract_role_keywords(normalized_resolved_query, normalized_keyword_query)
    logger.info("search role keywords=%s", ",".join(role_keywords) if role_keywords else "<none>")

    candidates = db.hybrid_search(
        keyword_query=effective_query,
        city=normalized_city,
        company=company,
        experience=normalized_experience,
        salary_min=salary_min,
        salary_unit=salary_unit,
        top_k=20,
    )
    logger.info("search tool recalled %s candidates", len(candidates))

    candidates = _filter_candidates_by_title(candidates, role_keywords)
    logger.info("search tool kept %s candidates after title filter", len(candidates))

    if not candidates:
        target_text = " / ".join(role_keywords) if role_keywords else "当前筛选条件"
        return json.dumps(
            {
                "llm_text": f"没有找到职位名称与 {target_text} 明确匹配的岗位。建议换个城市、放宽薪资，或改搜相近岗位方向。",
                "ui_cards": [],
            },
            ensure_ascii=False,
        )

    if not reranker or len(candidates) < 2:
        final_results = candidates[:6]
    else:
        rerank_pairs = []
        for res in candidates:
            doc_text = (
                f"职位: {res.get('title')} | "
                f"地点: {res.get('city')} {res.get('district')} | "
                f"公司: {res.get('company')} | "
                f"薪资: {res.get('salary')} | "
                f"经验要求: {res.get('experience')} | "
                f"学历要求: {res.get('degree')} | "
                f"福利: {res.get('welfare')} | "
                f"详情: {res.get('summary') or res.get('detail', '')[:200]}"
            ).replace("\n", " ")
            rerank_pairs.append([normalized_resolved_query or resolved_query, doc_text])

        scores = reranker.predict(rerank_pairs)
        scored_results = list(zip(candidates, scores))
        scored_results.sort(key=lambda item: item[1], reverse=True)
        logger.info(
            "search rerank complete: max=%.4f min=%.4f",
            float(scored_results[0][1]),
            float(scored_results[-1][1]),
        )
        final_results = [item[0] for item in scored_results[:6]]

    ui_cards = []
    for res in final_results:
        ui_cards.append(
            {
                "job_id": res["job_id"],
                "title": res["title"],
                "company": res["company"],
                "salary": res["salary"],
                "tags": f"{res.get('city', '')} | {res.get('experience', '')}",
            }
        )

    llm_lines = [f"为你找到 {len(final_results)} 个较匹配的职位："]
    for res in final_results:
        intro = res.get("summary") or (res.get("detail", "")[:150])
        llm_lines.extend(
            [
                f"🏢 {res['company']}（{res.get('industry', '')} | {res.get('city', '')} {res.get('district', '')}）",
                f"📌 职位：{res['title']}",
                f"💰 薪资：{res['salary']} · {res.get('welfare') or '未标注'}",
                f"📋 要求：{res.get('degree', '')} / {res.get('experience', '')}",
                f"📝 概要：{intro}",
                f"🔗 链接：{res.get('detail_url', '')}",
                "",
            ]
        )

    return json.dumps({"llm_text": "\n".join(llm_lines), "ui_cards": ui_cards}, ensure_ascii=False)


@tool("get_user_resume_tool", args_schema=GetResumeInput)
def get_user_resume_tool(user_id: str):
    """读取用户最近一次上传的简历内容，供后续分析或推荐使用。"""
    logger.info("fetching latest resume for user=%s", user_id)
    resume_data = db.get_latest_resume(user_id)

    if not resume_data:
        return "数据库反馈：该用户尚未上传简历，请先在左侧面板上传。"

    content = resume_data["content"]
    if len(content) > 4000:
        content = content[:4000] + "\n...(后文截断)..."

    return f"【用户简历内容（文件名: {resume_data['filename']}）】\n{content}"


@tool("save_preference_tool", args_schema=PreferenceInput)
def save_preference_tool(user_id: str, preference: str):
    """保存用户偏好或背景事实到长期记忆。"""
    logger.info("saving memory for user=%s preference=%s", user_id, preference)
    old_pref = db.get_user_profile(user_id)
    if old_pref and preference in old_pref:
        return f"系统反馈：信息 '{preference}' 已存在。"

    db.add_long_term_memory(
        user_id=user_id,
        content=preference,
        memory_type=_infer_memory_type(preference),
        source="tool",
    )
    return f"系统反馈：已写入长期画像 - {preference}"


@tool("analyze_job_match_tool", args_schema=MatchInput)
def analyze_job_match_tool(user_id: str, job_description: str):
    """基于用户简历和目标 JD 生成岗位匹配分析与雷达图数据。"""
    resume_data = db.get_latest_resume(user_id)
    if not resume_data:
        return json.dumps(
            {
                "error": "请先上传简历，否则无法进行匹配分析。",
                "image_path": None,
                "analysis": "未找到简历，请先上传。",
                "scores": [0, 0, 0, 0, 0],
            },
            ensure_ascii=False,
        )

    resume_text = resume_data["content"]
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """你是一位资深技术招聘专家。请深度对比【求职者简历】和【目标岗位JD】。

任务要求：
1. 从 5 个维度打分（0-100）：技术栈、经验、学历、项目、软技能。
2. 输出一份简洁但专业的 Markdown 分析。
3. 严格输出 JSON，不要带 markdown code fence。

JSON 结构：
{
  "scores": [85, 75, 90, 60, 80],
  "analysis": "### 核心优势\\n...\\n\\n### 能力缺口\\n..."
}
""",
            ),
            ("human", "【简历内容】\n{resume}\n\n【岗位JD】\n{jd}"),
        ]
    )

    try:
        response = (prompt | llm).invoke({"resume": resume_text, "jd": job_description})
        data = _safe_json_text(response.content)
        scores = data.get("scores", [60, 60, 60, 60, 60])
        comment = data.get("analysis", "分析完成，但未生成详细报告。")
    except Exception as exc:
        logger.exception("match analysis llm failed")
        scores = [50, 50, 50, 50, 50]
        comment = f"### 分析服务异常\n\n模型调用或解析失败，请稍后重试。\n错误信息: {exc}"

    abs_path = None
    try:
        img_path = create_radar_chart(scores, ["技术栈", "工作经验", "学历背景", "项目契合", "软技能"])
        if img_path:
            abs_path = os.path.abspath(img_path)
    except Exception as exc:
        logger.warning("radar chart generation failed: %s", exc)

    return json.dumps({"image_path": abs_path, "analysis": comment, "scores": scores}, ensure_ascii=False)


@tool("analyze_job_match_by_query_tool", args_schema=MatchByQueryInput)
def analyze_job_match_by_query_tool(
    user_id: str,
    target_query: str,
    city: str = "",
    company: str = "",
):
    """先按查询词找到目标岗位，再基于该岗位执行人岗匹配分析。"""
    normalized_city = _normalize_city(city, target_query)
    normalized_query = _normalize_keyword_text(target_query)
    logger.info(
        "match-by-query invoked: user=%s target_query=%s city=%s->%s company=%s",
        user_id,
        target_query,
        city,
        normalized_city,
        company,
    )

    candidates = db.hybrid_search(
        keyword_query=normalized_query,
        city=normalized_city,
        company=company,
        top_k=3,
    )
    role_keywords = _extract_role_keywords(normalized_query)
    candidates = _filter_candidates_by_title(candidates, role_keywords)

    if not candidates:
        return json.dumps(
            {
                "llm_text": "没有找到可用于匹配分析的目标岗位。请补充公司名、职位名或城市后再试。",
                "selected_job": None,
                "image_path": None,
                "analysis": None,
            },
            ensure_ascii=False,
        )

    selected_job = candidates[0]
    match_raw = analyze_job_match_tool.invoke(
        {
            "user_id": user_id,
            "job_description": selected_job.get("detail", "") or selected_job.get("summary", ""),
        }
    )
    match_text = match_raw.content if hasattr(match_raw, "content") else str(match_raw)

    try:
        match_data = json.loads(match_text)
    except Exception:
        match_data = {
            "image_path": None,
            "analysis": "匹配分析结果解析失败，请稍后重试。",
            "scores": [0, 0, 0, 0, 0],
        }

    llm_text = (
        f"已基于岗位 {selected_job.get('title', '未知职位')} @ {selected_job.get('company', '未知公司')} "
        f"完成人岗匹配分析。\n\n"
        f"薪资：{selected_job.get('salary', '未知')}\n"
        f"地点：{selected_job.get('city', '')} {selected_job.get('district', '')}\n"
        f"要求：{selected_job.get('degree', '不限')} / {selected_job.get('experience', '不限')}\n\n"
        f"{match_data.get('analysis', '暂无分析结果')}"
    )

    return json.dumps(
        {
            "llm_text": llm_text,
            "selected_job": {
                "job_id": selected_job.get("job_id"),
                "title": selected_job.get("title"),
                "company": selected_job.get("company"),
                "salary": selected_job.get("salary"),
                "tags": f"{selected_job.get('city', '')} | {selected_job.get('experience', '')}",
            },
            "image_path": match_data.get("image_path"),
            "analysis": match_data.get("analysis"),
        },
        ensure_ascii=False,
    )
