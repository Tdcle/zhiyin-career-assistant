import os
from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, RemoveMessage

from langchain_community.chat_models import ChatTongyi
from langgraph.checkpoint.memory import MemorySaver

from config.config import config
from utils.database import DatabaseManager
from utils.tools import search_jobs_tool, analyze_trend_tool, save_preference_tool

os.environ["NO_PROXY"] = "localhost,127.0.0.1"

db_manager = DatabaseManager()



# ================= 1. 定义状态 (State) =================
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str
    summary: str


# ================= 2. 初始化模型 =================
tools = [search_jobs_tool, analyze_trend_tool, save_preference_tool]

llm = ChatTongyi(
    api_key=config.DASHSCOPE_API_KEY,
    model="qwen-plus",
    streaming=True
)

summ_llm = ChatTongyi(
    api_key=config.DASHSCOPE_API_KEY,
    model="qwen-turbo", # 总结用便宜的模型
    streaming=False
)

llm_with_tools = llm.bind_tools(tools)


# ================= 3. 节点逻辑实现 =================

def summarize_node(state: AgentState):
    """
    【记忆压缩节点】
    """
    summary = state.get("summary", "")
    messages = state["messages"]

    # 构造 Prompt
    if summary:
        summary_message = (
            f"这是一份之前的对话摘要: {summary}\n\n"
            "请将上面的摘要内容与下面的新对话内容合并，生成一份新的、更全面的对话摘要。"
        )
    else:
        summary_message = "请将下面的对话内容生成一份精炼的摘要。"

    # 【修正点】: 必须包含 AIMessage，否则 AI 记不住自己说过什么
    # 同时排除 SystemMessage，因为系统指令不需要被总结
    messages_content = "\n".join(
        [f"{m.type}: {m.content}" for m in messages if isinstance(m, (HumanMessage, AIMessage))]
    )

    prompt = f"""
    {summary_message}

    【要求】
    1. 重点保留：用户的个人信息（姓名、技术栈）、明确的求职偏好（地点、薪资）、AI 推荐过的关键职位名称。
    2. 过滤：无意义的寒暄（你好、谢谢）。
    3. 输出：直接输出摘要段落，不要加前缀。

    【需要合并的新对话】
    {messages_content}
    """

    response = summ_llm.invoke(prompt)
    new_summary = response.content

    # 删除倒数第2条之前的所有消息（保留最近一轮问答，保证对话连贯性）
    delete_messages = [RemoveMessage(id=m.id) for m in messages[:-2]]

    return {"summary": new_summary, "messages": delete_messages}


def should_summarize(state: AgentState):
    """
    【条件边】判断是否需要进行总结
    """
    messages = state["messages"]
    # 阈值建议：可以设为 6-10 之间。太长会浪费 Token，太短总结太频繁。
    if len(messages) > 8:
        return "summarize_node"
    return "bot_node"


def bot_node(state: AgentState):
    """
    【主大脑节点】
    """
    user_id = state["user_id"]
    summary = state.get("summary", "")

    # 1. 动态获取数据库里的长期画像
    db_profile = db_manager.get_user_profile(user_id)
    if not db_profile: db_profile = "暂无偏好"

    # 2. 构造 System Prompt
    # 【重点】这里必须把 summary 塞进去，因为旧的 messages 已经被 summarize_node 删了！
    system_prompt_content = f"""
    你是由一位高级职业顾问，帮助用户寻找职位，不会回答与求职无关的问题，如果用户提出与求职无关的问题，请直接拒绝。

    【记忆模块】
    1. **长期画像 (Database)**: 
       {db_profile}
       (这是用户最核心的设定。如果用户提到新的设定，请自觉保存)

    2. **对话摘要 (Summary)**: 
       {summary}
       (这是之前的聊天背景，包含了被压缩的历史信息)

    【决策机制】
    - **更新信息** -> 用户自报家门或修改需求 -> 调用 `save_preference_tool` (User ID: {user_id})。
    - **寻找职位** -> 用户找工作 -> 调用 `search_jobs_tool`。
    - **闲聊** -> 用户问其他问题 -> 基于【记忆模块】直接回答，**不要调工具**。

    【展示原则】
    - 推荐职位时，必须以 Markdown 卡片形式展示。
    - 必须展示摘要(Summary)和链接。
    """

    # 构造消息列表：SystemPrompt + (被删减后的) History
    sys_msg = SystemMessage(content=system_prompt_content)
    messages = [sys_msg] + state["messages"]

    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


# ================= 4. 构建图 =================
workflow = StateGraph(AgentState)

workflow.add_node("bot_node", bot_node)
workflow.add_node("summarize_node", summarize_node)
workflow.add_node("tools", ToolNode(tools))

# 开启记忆
checkpointer = MemorySaver()

# 逻辑流
workflow.add_conditional_edges(
    START,
    should_summarize,
    {
        "summarize_node": "summarize_node",
        "bot_node": "bot_node"
    }
)

workflow.add_edge("summarize_node", "bot_node")
workflow.add_conditional_edges("bot_node", tools_condition)
workflow.add_edge("tools", "bot_node")

# 编译 (必须传入 checkpointer)
app_graph = workflow.compile(checkpointer=checkpointer)