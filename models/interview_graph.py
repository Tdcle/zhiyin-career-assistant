# models/interview_graph.py
import os
import json
import re
from utils.logger import sys_logger
from typing import Annotated, Dict, Any, Optional
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, RemoveMessage

from langchain_community.chat_models import ChatTongyi
from langgraph.checkpoint.memory import MemorySaver

from config.config import config
from utils.database import DatabaseManager
from utils.tools import save_preference_tool

db_manager = DatabaseManager()


# ================= 1. 定义面试专用状态 =================
class InterviewState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str
    job_context: Dict[str, Any]
    summary: str
    # 面试总结相关（由 report_node 写入）
    final_report: Optional[str]
    # 控制信号
    should_end: bool


# ================= 2. 初始化模型 =================
tools = [save_preference_tool]

llm = ChatTongyi(
    api_key=config.DASHSCOPE_API_KEY,
    model="qwen-max",
    streaming=True
)

report_llm = ChatTongyi(
    api_key=config.DASHSCOPE_API_KEY,
    model="qwen-plus",
    streaming=False  # 【改为非流式】报告一次性生成，由 interview_flow 控制展示
)

summ_llm = ChatTongyi(
    api_key=config.DASHSCOPE_API_KEY,
    model="qwen-turbo",
    streaming=False
)

llm_with_tools = llm.bind_tools(tools)


# ================= 3. 节点逻辑 =================

def summarize_node(state: InterviewState):
    """【记忆压缩节点】"""
    summary = state.get("summary", "")
    messages = state["messages"]

    if summary:
        summary_message = (
            f"这是之前的面试摘要: {summary}\n\n"
            "请将上面的摘要与下方的新对话合并，生成更全面的摘要。"
        )
    else:
        summary_message = "请将下面的面试对话生成一份精炼摘要。"

    messages_content = "\n".join(
        [f"{m.type}: {m.content}" for m in messages if isinstance(m, (HumanMessage, AIMessage))]
    )

    prompt = f"""
    {summary_message}
    【要求】
    1. 保留核心技术问答：面试官问了什么，候选人答得如何（对/错/深度）。
    2. 保留候选人自述的关键信息（技能、经验）。
    3. 忽略客套话。

    【内容】
    {messages_content}
    """

    response = summ_llm.invoke(prompt)
    new_summary = response.content
    delete_messages = [RemoveMessage(id=m.id) for m in messages[:-4]]
    return {"summary": new_summary, "messages": delete_messages}


def should_summarize(state: InterviewState):
    """【条件边】入口路由"""
    if state.get("should_end", False):
        return "report_node"

    messages = state["messages"]
    if len(messages) > 8:
        return "summarize_node"
    return "interviewer_node"


def interviewer_node(state: InterviewState):
    """【面试官节点】"""
    user_id = state["user_id"]
    summary = state.get("summary", "")
    job_ctx = state.get("job_context", {})

    company = job_ctx.get("company", "该公司")
    title = job_ctx.get("title", "该岗位")
    jd = job_ctx.get("detail", "")[:1000]

    db_profile = db_manager.get_user_profile(user_id)
    if not db_profile:
        db_profile = "暂无"

    system_prompt_str = f"""
    你现在是【{company}】的资深技术面试官。
    正在面试岗位：【{title}】。

    【岗位JD摘要】：
    {jd}

    【候选人长期画像(参考)】：
    {db_profile}

    【之前的面试摘要】：
    {summary}

    【你的行为准则】：
    1. **角色定位**：你是严格的考核者，不是老师。禁止长篇大论的教学。
    2. **记忆与画像**：
       - 如果候选人在对话中透露了新的技能、年限、项目经历，**必须立即调用 `save_preference_tool` 保存**。
       - 结合【长期画像】和JD进行提问。
    3. **冷启动**：如果这是对话开始，根据候选人介绍和JD直接抛出技术问题。
    4. **点评规则**：
       - 候选人回答后，仅进行"判卷式"简评（对/错/不完整）。
       - 严禁解释技术原理（除非被问）。
       - 字数控制在 100 字以内。
    5. **追问机制**：点评后立即追问细节，或开启新话题。禁止客套。
    """

    sys_msg = SystemMessage(content=system_prompt_str)
    messages = [sys_msg] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def report_node(state: InterviewState):
    """
    【面试总结报告节点】

    简化设计：
    - 只生成一份完整的 Markdown 评估报告
    - 不额外生成分数 JSON（报告正文中已包含评分）
    - 将详细反馈写入用户画像
    - 不往 messages 中写入任何内容（避免重复显示）
    """
    import re

    user_id = state["user_id"]
    summary = state.get("summary", "")
    messages = state["messages"]
    job_ctx = state.get("job_context", {})

    company = job_ctx.get("company", "该公司")
    title = job_ctx.get("title", "该岗位")
    jd = job_ctx.get("detail", "")[:800]

    # 收集完整对话素材
    recent_dialogue = "\n".join(
        [f"{'面试官' if isinstance(m, AIMessage) else '候选人'}: {m.content}"
         for m in messages if isinstance(m, (HumanMessage, AIMessage)) and m.content]
    )

    full_context = ""
    if summary:
        full_context += f"【前期面试摘要】\n{summary}\n\n"
    full_context += f"【近期对话记录】\n{recent_dialogue}"

    report_prompt = f"""
    你是一位资深的面试评估专家。候选人刚完成了【{company}】的【{title}】岗位面试。

    【岗位 JD】：
    {jd}

    【完整面试记录】：
    {full_context}

    请生成一份专业、深度、一针见血的面试评估报告。

    格式要求（严格遵循 Markdown）：

    ## 📋 面试评估报告

    **面试岗位**：{title} @ {company}

    ### 📊 综合评分：X / 10 分
    （一句话概括评分理由）

    ### ✅ 表现亮点
    （列出 2-3 个候选人回答出色的地方，必须具体引用其原话或回答要点）

    ### ⚠️ 明显不足
    （列出 2-3 个回答不好或暴露知识盲区的地方，必须指出哪里错了、正确答案是什么、为什么这很重要）

    ### 📚 针对性学习建议
    （针对上述每个不足，给出具体的学习方向和推荐资源，用编号列表）

    ### 🎯 面试通过概率预估
    （给出百分比 + 一句话理由 + 如果要达到通过线还需要多久的学习）

    ---

    注意：
    - 评分和内容必须一致，如果表现很差就给低分，不要客气
    - 直接输出报告内容即可，不要输出任何 JSON、代码块或额外格式

    最后，在报告末尾另起一行，以如下格式输出需要写入候选人档案的信息（用于长期跟踪改进）：
    PROFILE_UPDATE:简洁描述候选人的核心优势和所有薄弱环节，包括具体的技术知识点缺失
    """

    response = report_llm.invoke(report_prompt)
    report_content = response.content

    # --- 解析并提取画像更新内容 ---
    profile_update = ""
    clean_report = report_content

    try:
        profile_match = re.search(r'PROFILE_UPDATE:(.*?)$', report_content, re.DOTALL)
        if profile_match:
            profile_update = profile_match.group(1).strip()
            # 从展示报告中去掉这行
            clean_report = report_content[:profile_match.start()].rstrip()
    except AttributeError:
        pass

    # --- 写入用户画像 ---
    if profile_update:
        try:
            old_profile = db_manager.get_user_profile(user_id)
            timestamp_note = f"[面试反馈-{title}] {profile_update}"
            new_profile = f"{old_profile}; {timestamp_note}" if old_profile else timestamp_note
            db_manager.update_user_profile(user_id, new_profile)
            sys_logger.info(f"🧠 [Report] 画像已更新: {profile_update[:100]}...")
        except Exception as e:
            sys_logger.error(f"画像更新失败: {e}")

    # 【关键】只写独立字段，不动 messages
    return {
        "final_report": clean_report,
    }


# ================= 4. 构建图 =================
workflow = StateGraph(InterviewState)

workflow.add_node("interviewer_node", interviewer_node)
workflow.add_node("summarize_node", summarize_node)
workflow.add_node("tools", ToolNode(tools))
workflow.add_node("report_node", report_node)

checkpointer = MemorySaver()

workflow.add_conditional_edges(
    START,
    should_summarize,
    {
        "summarize_node": "summarize_node",
        "interviewer_node": "interviewer_node",
        "report_node": "report_node"
    }
)

workflow.add_edge("summarize_node", "interviewer_node")
workflow.add_conditional_edges("interviewer_node", tools_condition)
workflow.add_edge("tools", "interviewer_node")
workflow.add_edge("report_node", END)

interview_graph = workflow.compile(checkpointer=checkpointer)