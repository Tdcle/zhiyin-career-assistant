# tools.py
import logging
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from utils.database import DatabaseManager
from sentence_transformers import CrossEncoder
from utils.file_parser import FileParser # 导入刚才写的文件

# 获取日志记录器（注意：这里假设 app.py 已经初始化了配置名为 "JobAgent" 的 logger）
# 这样可以确保工具的日志和主程序的日志写入同一个文件
logger = logging.getLogger("JobAgent")

db = DatabaseManager()

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


@tool("search_jobs_tool", args_schema=SearchInput)
def search_jobs_tool(query: str):
    """
    【核心工具】搜索职位 (Vector Recall -> Rerank)。
    执行流程：
    1. 数据库混合检索召回 Top 20。
    2. BGE-Reranker 模型精细打分。
    3. 返回得分最高的 Top 5。
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

    # 3. 结果格式化
    context_str = f"经 AI 深度筛选，为您找到最匹配的 {len(final_results)} 个职位：\n"
    for res in final_results:
        intro = res.get('summary') if res.get('summary') else res['detail'][:200]
        context_str += f"""
        - **职位**: {res['title']}
        - **公司**: {res['company']} ({res['industry']} | {res['city']} {res['district']})
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