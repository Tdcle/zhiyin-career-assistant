# tools.py
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from database import DatabaseManager

db = DatabaseManager()


class SearchInput(BaseModel):
    query: str = Field(description="用于数据库检索的关键词。")


class TrendInput(BaseModel):
    keyword: str = Field(description="关键词。")


class PreferenceInput(BaseModel):
    user_id: str = Field(description="用户ID")
    preference: str = Field(description="需要保存的信息（如求职偏好、个人称呼、技术栈等）。")


@tool("search_jobs_tool", args_schema=SearchInput)
def search_jobs_tool(query: str):
    """
    【核心工具】搜索职位。
    返回结果包含了职位的完整详情，请以精美的 Markdown 格式展示给用户。
    """
    # 稍微放宽检索数量，方便优中选优
    results = db.vector_search(query, top_k=6)

    if not results:
        return "数据库反馈：没有找到匹配的职位。请建议用户尝试更通用的关键词。"

    # 我们直接返回结构化数据，让大模型去决定怎么排版，但提供足够素材
    context_str = f"共检索到 {len(results)} 个职位，以下是详细数据：\n"
    for i, res in enumerate(results):

        context_str += f"""
        ---
        【职位名称】: {res['title']}
        【公司名称】: {res['company']} ({res['industry']} | {res['city']}{res['district']})
        【薪资待遇】: {res['salary']}
        【福利标签】: {res['welfare']}
        【职位详情】: {res['summary']}
        【详情链接】: {res['detail_url']}
        """
    return context_str


@tool("analyze_trend_tool", args_schema=TrendInput)
def analyze_trend_tool(keyword: str):
    """分析市场趋势"""
    result = db.get_market_analytics(keyword)
    count = result['job_count'] if result else 0
    return f"数据库统计：关于 '{keyword}' 的职位现有 {count} 个。"


@tool("save_preference_tool", args_schema=PreferenceInput)
def save_preference_tool(user_id: str, preference: str):
    """
    【记忆工具】用于保存用户的任何长期信息。
    包括但不限于：求职偏好（地点、薪资）、个人称呼（我叫xxx）、技术特长等。
    """
    old_pref = db.get_user_profile(user_id)

    # 智能去重逻辑
    if old_pref and preference in old_pref:
        return f"系统反馈：信息 '{preference}' 已存在，无需重复保存。"

    new_pref = f"{old_pref}; {preference}" if old_pref else preference
    db.update_user_profile(user_id, new_pref)

    return f"系统反馈：已成功将信息【{preference}】写入长期画像。请根据此新信息决定下一步行动（如重新搜索或继续对话）。"