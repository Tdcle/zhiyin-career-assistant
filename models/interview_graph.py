import os
from typing import Annotated, Dict, Any
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, RemoveMessage

from langchain_community.chat_models import ChatTongyi
from langgraph.checkpoint.memory import MemorySaver

from config.config import config
from utils.database import DatabaseManager
from utils.tools import save_preference_tool  # 只要这个工具即可

db_manager = DatabaseManager()


# ================= 1. 定义面试专用状态 =================
class InterviewState(TypedDict):
    # 消息历史
    messages: Annotated[list, add_messages]
    # 用户ID
    user_id: str
    # 岗位上下文 (标题、公司、JD) - 用于生成 SystemPrompt
    job_context: Dict[str, Any]
    # 对话摘要
    summary: str


# ================= 2. 初始化模型 =================
# 面试官模型：绑定“保存画像”工具，让它自主决定何时记录
tools = [save_preference_tool]

llm = ChatTongyi(
    api_key=config.DASHSCOPE_API_KEY,
    model="qwen-max",
    streaming=True
)

summ_llm = ChatTongyi(
    api_key=config.DASHSCOPE_API_KEY,
    model="qwen-turbo",  # 总结用便宜的模型
    streaming=False
)

llm_with_tools = llm.bind_tools(tools)


# ================= 3. 节点逻辑 =================

def summarize_node(state: InterviewState):
    """【记忆压缩节点】处理过长的面试记录"""
    summary = state.get("summary", "")
    messages = state["messages"]

    # 构造 Prompt
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

    # 删除旧消息，保留最近 4 条 (2轮) 保持连贯
    delete_messages = [RemoveMessage(id=m.id) for m in messages[:-4]]

    return {"summary": new_summary, "messages": delete_messages}


def should_summarize(state: InterviewState):
    """【条件边】判断是否需要总结"""
    messages = state["messages"]
    if len(messages) > 8:  # 每 4 轮对话总结一次
        return "summarize_node"
    return "interviewer_node"


def interviewer_node(state: InterviewState):
    """【面试官节点】核心逻辑"""
    user_id = state["user_id"]
    summary = state.get("summary", "")
    job_ctx = state.get("job_context", {})

    # 1. 获取 JD 信息
    company = job_ctx.get("company", "该公司")
    title = job_ctx.get("title", "该岗位")
    jd = job_ctx.get("detail", "")[:1000]  # 截断防止溢出

    # 2. 获取长期记忆 (用户画像)
    # 这一步让面试官拥有"长时记忆"，知道你以前是干嘛的
    db_profile = db_manager.get_user_profile(user_id)
    if not db_profile: db_profile = "暂无"

    # 3. 构造 System Prompt (严谨版)
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
       - 候选人回答后，仅进行“判卷式”简评（对/错/不完整）。
       - 严禁解释技术原理（除非被问）。
       - 字数控制在 100 字以内。
    5. **追问机制**：点评后立即追问细节，或开启新话题。禁止客套。
    """

    sys_msg = SystemMessage(content=system_prompt_str)
    # 将 SystemMessage 放在最前面
    messages = [sys_msg] + state["messages"]

    # 调用模型
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


# ================= 4. 构建图 =================
workflow = StateGraph(InterviewState)

workflow.add_node("interviewer_node", interviewer_node)
workflow.add_node("summarize_node", summarize_node)
workflow.add_node("tools", ToolNode(tools))  # 工具节点负责执行 save_preference_tool

# 开启记忆
checkpointer = MemorySaver()

# 定义边
workflow.add_conditional_edges(
    START,
    should_summarize,
    {
        "summarize_node": "summarize_node",
        "interviewer_node": "interviewer_node"
    }
)

workflow.add_edge("summarize_node", "interviewer_node")

# 工具调用逻辑：如果面试官决定保存画像，就去 tools 节点，回来后继续面试
workflow.add_conditional_edges(
    "interviewer_node",
    tools_condition,
)
workflow.add_edge("tools", "interviewer_node")

# 编译图
interview_graph = workflow.compile(checkpointer=checkpointer)