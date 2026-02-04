# utils/tools.py
import json
import os
import re
import logging
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from sentence_transformers import CrossEncoder

# 项目内部导入
from utils.database import DatabaseManager
from utils.global_state import GlobalState
from utils.plotter import create_radar_chart
from config.config import config

# 获取日志记录器（注意：这里假设 app.py 已经初始化了配置名为 "JobAgent" 的 logger）
# 这样可以确保工具的日志和主程序的日志写入同一个文件
logger = logging.getLogger("JobAgent")

db = DatabaseManager()

llm = ChatOllama(base_url=config.OLLAMA_URL, model=config.MARK_MODEL_NAME, temperature=0.1)

# ================= 1. 初始化重排序模型 =================
logger.info("🔄 [System] 正在加载 BGE-Reranker 模型...")
try:
    # 请确保路径正确，建议改为相对路径或环境变量
    MODEL_PATH = "F:/Python_Project/LLM_study/models/bge-reranker-v2-m3"
    reranker = CrossEncoder(MODEL_PATH, device='cpu')
    logger.info("✅ [System] Reranker 模型加载完成")
except Exception as e:
    logger.warning(f"⚠️ [System] Reranker 加载失败，将降级为普通检索: {e}")
    reranker = None



# ================= 2. 定义输入结构 =================
class SearchInput(BaseModel):
    user_id: str = Field(description="用户ID")
    query: str = Field(
        description=(
            "用于检索职位的组合关键词字符串。"
            "关键词应包含以下维度中的一个或多个（用空格分隔）："
            "1. 地点（如：北京、海淀区）"
            "2. 职位方向/技术栈（如：Python、Java、大模型、前端）"
            "3. 薪资范围（如：20k、30-50k）"
            "4. 公司名（如：百度、字节）"
            "5. 福利要求（如：双休、不加班）"
            "例如输入：'北京 海淀 Python 20k 双休'"
        )
    )

class TrendInput(BaseModel):
    keyword: str = Field(description="关键词。")

class PreferenceInput(BaseModel):
    user_id: str = Field(description="用户ID")
    preference: str = Field(description="需要保存的信息。")

class GetResumeInput(BaseModel):
    user_id: str = Field(description="用户ID")

class MatchInput(BaseModel):
    user_id: str = Field(description="用户ID")
    job_description: str = Field(description="目标职位的详细描述或职位JD内容")


@tool("search_jobs_tool", args_schema=SearchInput)
def search_jobs_tool(user_id: str, query: str):
    """
    【核心工具】搜索职位 (Vector Recall -> Rerank)。
    执行流程：
    1. 数据库混合检索召回 Top 20。
    2. BGE-Reranker 模型精细打分。
    3. 返回得分最高的 Top 5。
    副作用：将 Top 5 的结构化数据存入 GlobalState 供前端列表展示。
    """

    logger.info(f"🛠️ [Tool:Search] 开始搜索: '{query}'")

    # 1. 召回
    candidates = db.vector_search(query, top_k=20)
    logger.info(f"📊 [Tool:Search] 数据库召回: {len(candidates)} 条")

    if not candidates:
        return "数据库反馈：没有找到匹配的职位。"

    # 2. 精排
    if not reranker or len(candidates) < 2:
        final_results = candidates[:5]
    else:
        # --- 阶段二：精排 / 重排序 (Rerank) ---
        rerank_pairs = []

        for res in candidates:
            # 数据构造/数据清洗：去掉换行符，防止干扰模型
            doc_text = (
                f"职位: {res.get('title')} | "
                f"地点: {res.get('city')} {res.get('district')} | "
                f"公司: {res.get('company')} | "
                f"薪资: {res.get('salary')} | "
                f"经验要求: {res.get('experience')} | "
                f"学历要求: {res.get('degree')} | "
                f"福利: {res.get('welfare')} | "
                f"详情: {res.get('summary')}"
            ).replace('\n', ' ')

            rerank_pairs.append([query, doc_text])

        # 模型打分 (返回一个分数列表)
        scores = reranker.predict(rerank_pairs)
        # 将 [职位, 分数] 绑定在一起
        scored_results = list(zip(candidates, scores))
        # 按分数从高到低排序 (Reverse=True)
        scored_results.sort(key=lambda x: x[1], reverse=True)

        # 记录分数情况到日志文件
        top_score = scored_results[0][1]
        low_score = scored_results[-1][1]
        logger.info(f"🧠 [Tool:Search] 重排序完成. Max: {top_score:.4f} | Min: {low_score:.4f}")

        # 取前 5 名
        final_results = [item[0] for item in scored_results[:5]]

    # 前端只需要 ID, Title, Company 用于展示卡片
    simple_results_for_ui = []
    for res in final_results:
        simple_results_for_ui.append({
            "job_id": res['job_id'],  # 关键
            "title": res['title'],
            "company": res['company'],
            "salary": res['salary'],
            "tags": f"{res.get('city')} | {res.get('experience')}"
        })

    logger.info(f"🛑 [DEBUG-TOOL] 准备写入缓存 -> Key(user_id): '{user_id}' | 数据量: {len(simple_results_for_ui)}")
    # 存入缓存，Key 为 user_id (确保多用户隔离)
    GlobalState.set_search_results(user_id, simple_results_for_ui)

    check = GlobalState.get_search_results(user_id)
    logger.info(f"🛑 [DEBUG-TOOL] 写入后立即读取验证 -> Key: '{user_id}' | 结果数量: {len(check)}")

    # 3. 结果格式化
    context_str = f"经 AI 深度筛选，为您找到最匹配的 {len(final_results)} 个职位：\n"
    for res in final_results:
        intro = res.get('summary') if res.get('summary') else res['detail'][:200]
        context_str += f"""
        - **公司**: {res['company']} ({res['industry']} | {res['city']} {res['district']})
        - **职位**: {res['title']}
        - **薪资**: {res['salary']} ({res['welfare']})
        - **概要**: {intro}
        - **链接**: {res['detail_url']}
        ------------------------------------------------
        """
    logger.info(f"✅ [Tool:Search] 工具返回完成")
    return context_str


@tool("get_user_resume_tool", args_schema=GetResumeInput)
def get_user_resume_tool(user_id: str):
    """
    【简历获取工具】当用户询问与自身情况匹配的职位，或者需要基于简历进行推荐时调用。
    该工具会从数据库中提取用户最新上传的简历文本内容。
    """
    print(f"🛠️ [Tool] 正在从数据库获取用户 [{user_id}] 的简历...")

    resume_data = db.get_latest_resume(user_id)

    if not resume_data:
        return "数据库反馈：该用户暂未上传简历，请提示用户先在左侧面板上传简历。"

    # 截断一下防止 Token 爆炸，但要保留足够信息
    content = resume_data['content']
    if len(content) > 4000:
        content = content[:4000] + "\n...(后文截断)..."

    return f"【用户简历内容 (文件名: {resume_data['filename']})】:\n{content}"

@tool("analyze_trend_tool", args_schema=TrendInput)
def analyze_trend_tool(keyword: str):
    """分析市场趋势"""
    logger.info(f"🛠️ [Tool:Trend] 分析关键词: {keyword}")
    result = db.get_market_analytics(keyword)
    count = result['job_count'] if result else 0
    return f"数据库统计：关于 '{keyword}' 的职位现有 {count} 个。"


@tool("save_preference_tool", args_schema=PreferenceInput)
def save_preference_tool(user_id: str, preference: str):
    """【记忆工具】保存信息"""
    logger.info(f"🧠 [Tool:Memory] 正在保存用户[{user_id}] 偏好: {preference}")

    old_pref = db.get_user_profile(user_id)
    if old_pref and preference in old_pref:
        return f"系统反馈：信息 '{preference}' 已存在。"

    new_pref = f"{old_pref}; {preference}" if old_pref else preference
    db.update_user_profile(user_id, new_pref)

    return f"系统反馈：已写入长期画像 - {preference}"


@tool("analyze_job_match_tool", args_schema=MatchInput)
def analyze_job_match_tool(user_id: str, job_description: str):
    """
    深度人岗匹配分析工具。
    对比用户简历和职位JD，生成雷达图和详细分析建议。
    返回: JSON 字符串，包含图片路径和分析文本。
    """
    # 1. 获取简历
    resume_data = db.get_latest_resume(user_id)
    if not resume_data:
        # 返回符合前端预期的空结构，防止报错
        return json.dumps({
            "error": "请先上传简历，否则无法进行匹配分析。",
            "image_path": None,
            "analysis": "❌ 未找到简历，请先上传。",
            "scores": [0, 0, 0, 0, 0]
        }, ensure_ascii=False)

    resume_text = resume_data['content']

    # 2. LLM 打分与深度分析 (Prompt 升级)
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一位拥有10年经验的资深技术招聘专家。请深度对比【求职者简历】和【目标职位JD】。

        任务要求：
        1. **打分**：从5个维度进行打分（0-100分）：技术栈(Tech)、经验(Exp)、学历(Edu)、项目(Proj)、软技能(Soft)。
        2. **分析**：撰写一份专业的面试评估报告。
           - 语气：使用“您”称呼求职者，专业、诚恳、一针见血。
           - 格式：Markdown。
           - 内容结构：
             ### ✅ 核心优势 (列出2-3点完全匹配的亮点)
             ### ⚠️ 能力缺口 (明确指出缺失的技术栈或经验不足)
             ### 💡 面试准备建议 (针对缺口的具体复习方向或话术)
             ### 🔮 综合评价 (一句话总结成功率)

        请严格输出纯 JSON 格式，不要包含 ```json ... ``` 包裹：
        {{
            "scores": [85, 75, 90, 60, 80],
            "analysis": "### ✅ 核心优势\\n您在...\\n\\n### ⚠️ 能力缺口\\n..."
        }}
        """),
        ("human", "【简历内容】:\n{resume}\n\n【职位JD】:\n{jd}")
    ])

    try:
        # 调用本地模型链
        chain = prompt | llm
        # 截断输入防止超长
        response = chain.invoke({"resume": resume_text, "jd": job_description})
        content = response.content.strip()

        # --- 增强的 JSON 清洗逻辑 ---
        # 1. 尝试去除 markdown 代码块标记
        if "```" in content:
            # 提取 ```json 和 ``` 之间的内容，或者 ``` 和 ``` 之间的内容
            pattern = r"```(?:json)?(.*?)```"
            match = re.search(pattern, content, re.DOTALL)
            if match:
                content = match.group(1).strip()
            else:
                # 兜底：简单的 split
                content = content.split("```")[1].replace("json", "").strip()

        # 2. 尝试找到第一个 { 和最后一个 }，应对模型废话
        start_idx = content.find("{")
        end_idx = content.rfind("}")
        if start_idx != -1 and end_idx != -1:
            content = content[start_idx: end_idx + 1]

        # 解析 JSON
        data = json.loads(content)
        scores = data.get("scores", [60, 60, 60, 60, 60])
        comment = data.get("analysis", "分析完成，但未生成详细报告。")

    except Exception as e:
        print(f"❌ LLM分析失败: {e}")
        # 降级处理：返回默认分和错误提示
        scores = [50, 50, 50, 50, 50]
        comment = f"### ❌ 分析服务异常\n\n模型调用或解析失败，请稍后重试。\n错误信息: {str(e)}"

    # 3. 生成雷达图
    categories = ["技术栈", "工作经验", "学历背景", "项目契合", "软技能"]
    abs_path = None
    try:
        # 假设 create_radar_chart 返回的是相对路径或绝对路径
        img_path = create_radar_chart(scores, categories)
        if img_path:
            # 确保转换为绝对路径或前端可访问的路径
            abs_path = os.path.abspath(img_path)
    except Exception as e:
        print(f"❌ 绘图失败: {e}")
        # 绘图失败不影响文字分析展示
        pass

    # 4. 返回结构化数据
    return json.dumps({
        "image_path": abs_path,
        "analysis": comment,
        "scores": scores
    }, ensure_ascii=False)