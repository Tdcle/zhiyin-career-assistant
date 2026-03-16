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
    MODEL_PATH = "./models/bge-reranker-v2-m3"
    reranker = CrossEncoder(MODEL_PATH, device='cpu')
    logger.info("✅ [System] Reranker 模型加载完成")
except Exception as e:
    logger.warning(f"⚠️ [System] Reranker 加载失败，将降级为普通检索: {e}")
    reranker = None



# ================= 2. 定义输入结构 =================
# ==========================================
# 1. 定义更智能的 SearchInput 模型
# ==========================================
class SearchInput(BaseModel):
    resolved_query: str = Field(...,description="【极其重要】结合多轮对话上下文，将用户的搜索意图改写为一句完整的自然语言。例如用户之前说'北京前端实习'，现在问'那测试呢'，需改写为'在北京寻找一份测试的实习工作'。用于最终的语义精排。")
    keyword_query: str = Field(default="",description="提取的核心关键词集合（用空格分隔）。必须将职位名词(如'前端')、经验要求(如'实习'、'应届')、福利诉求(如'双休')、核心技能等组合在一起。例如：'前端 实习 双休'。如果没有则为空")
    city: str = Field(default="", description="提取的城市硬性条件，如'北京'。如果没有则为空")
    company: str = Field(default="", description="提取的公司名称，如'腾讯'。如果没有则为空")


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


# 注意：确保文件顶部已经导入了你的 db 和 reranker 实例

@tool("search_jobs_tool", args_schema=SearchInput)
def search_jobs_tool(
        resolved_query: str,
        keyword_query: str = "",
        city: str = "",
        company: str = "",
):
    """
    【核心工具】搜索职位 (混合检索 + BGE-Reranker)。
    """
    logger.info(
        f"🛠️ [Tool:Search] 意图改写: '{resolved_query}' | 综合关键词: '{keyword_query}' | city='{city}', company='{company}'")

    # 1. 数据库广度召回 + Python 初筛 (召回 Top 20 候选池)
    # 把 experience 重新传给底层，防止百万级库里实习生被全职岗淹没
    candidates = db.hybrid_search(
        keyword_query=keyword_query,
        city=city,
        company=company,
        top_k=20
    )

    logger.info(f"📊 [Tool:Search] 数据库底层召回: {len(candidates)} 条候选岗位")

    if not candidates:
        return json.dumps({
            "llm_text": "数据库反馈：没有找到匹配的职位。",
            "ui_cards": []
        }, ensure_ascii=False)

    # 2. BGE-Reranker 深度交叉精排 (终极审判层)
    if not reranker or len(candidates) < 2:
        final_results = candidates[:6]
    else:
        rerank_pairs = []

        # 使用具备完整多轮上下文的 resolved_query 去和岗位信息做 Cross-Encoder 计算
        for res in candidates:
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

            rerank_pairs.append([resolved_query, doc_text])

        scores = reranker.predict(rerank_pairs)
        scored_results = list(zip(candidates, scores))
        scored_results.sort(key=lambda x: x[1], reverse=True)

        top_score = scored_results[0][1]
        low_score = scored_results[-1][1]
        logger.info(
            f"🧠 [Tool:Search] Rerank 完成. Max分: {top_score:.4f} | Min分: {low_score:.4f} (依据 Query: '{resolved_query}')")

        final_results = [item[0] for item in scored_results[:6]]

    # 3. 构造两份数据
    ui_cards = []
    for res in final_results:
        ui_cards.append({
            "job_id": res['job_id'],
            "title": res['title'],
            "company": res['company'],
            "salary": res['salary'],
            "tags": f"{res.get('city', '')} | {res.get('experience', '')}"
        })

    llm_text = f"为用户找到 {len(final_results)} 个匹配职位，请按格式展示：\n"
    for i, res in enumerate(final_results, 1):
        intro = res.get('summary') if res.get('summary') else res.get('detail', '')[:150]
        welfare_text = res.get('welfare') or '未标注'
        llm_text += (
            f"🏢 **{res['company']}**（{res.get('industry', '')} | {res.get('city', '')} {res.get('district', '')}）\n"
            f"📌 职位：{res['title']}\n"
            f"💰 薪资：{res['salary']} · {welfare_text}\n"
            f"📋 要求：{res.get('degree', '')} / {res.get('experience', '')}\n"
            f"📝 概要：{intro}\n"
            f"🔗 链接：{res.get('detail_url', '')}\n"
            f"\n---\n"
        )

    return json.dumps({
        "llm_text": llm_text,
        "ui_cards": ui_cards
    }, ensure_ascii=False)


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