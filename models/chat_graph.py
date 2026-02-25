# models/chat_graph.py
import os
from typing import Annotated, List, Dict, Any, Optional
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, RemoveMessage

from langchain_community.chat_models import ChatTongyi
from langgraph.checkpoint.memory import MemorySaver

from config.config import config
from utils.database import DatabaseManager
from utils.tools import search_jobs_tool, analyze_trend_tool, save_preference_tool, get_user_resume_tool

db_manager = DatabaseManager()


# ================= 1. 定义状态 =================
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str
    summary: str
    # 【新增】职位搜索结果，供前端 UI 读取
    # 这个字段不参与 LLM 对话，仅作为 Tool -> UI 的数据通道
    search_results: Optional[List[Dict[str, Any]]]


# ================= 2. 初始化模型 =================
tools = [search_jobs_tool, analyze_trend_tool, save_preference_tool, get_user_resume_tool]

llm = ChatTongyi(
    api_key=config.DASHSCOPE_API_KEY,
    model="qwen-plus",
    streaming=True
)

summ_llm = ChatTongyi(
    api_key=config.DASHSCOPE_API_KEY,
    model="qwen-turbo",
    streaming=False
)

llm_with_tools = llm.bind_tools(tools)


# ================= 3. 节点逻辑 =================

def summarize_node(state: AgentState):
    """【记忆压缩节点】"""
    summary = state.get("summary", "")
    messages = state["messages"]

    if summary:
        summary_message = (
            f"这是一份之前的对话摘要: {summary}\n\n"
            "请将上面的摘要内容与下面的新对话内容合并，生成一份新的、更全面的对话摘要。"
        )
    else:
        summary_message = "请将下面的对话内容生成一份精炼的摘要。"

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

    delete_messages = [RemoveMessage(id=m.id) for m in messages[:-2]]
    return {"summary": new_summary, "messages": delete_messages}


def should_summarize(state: AgentState):
    """【条件边】判断是否需要进行总结"""
    messages = state["messages"]
    if len(messages) > 8:
        return "summarize_node"
    return "bot_node"


def bot_node(state: AgentState):
    """【主大脑节点】"""
    user_id = state["user_id"]
    summary = state.get("summary", "")

    db_profile = db_manager.get_user_profile(user_id)
    if not db_profile:
        db_profile = "暂无偏好"

    system_prompt_content = f"""
    你是由一位高级职业顾问，帮助用户寻找职位，不会回答与求职无关的问题，如果用户提出与求职无关的问题，请直接拒绝。

    【记忆模块】
    1. **长期画像 (Database)**: 
       {db_profile}

    2. **对话摘要 (Summary)**: 
       {summary}

    【决策机制】
    - **更新信息** -> 用户自报家门或修改需求 -> 调用 `save_preference_tool` (User ID: {user_id})。
    - **寻找职位** -> 用户找工作 -> 调用 `search_jobs_tool`。
    - **闲聊** -> 用户问其他问题 -> 基于【记忆模块】直接回答，**不要调工具**。

    【展示原则】
    - 推荐职位时，必须以 Markdown 卡片形式展示。
    - 必须展示摘要(Summary)和链接。
    """

    sys_msg = SystemMessage(content=system_prompt_content)
    messages = [sys_msg] + state["messages"]

    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def extract_search_results(state: AgentState):
    """
    【新增节点】后处理节点：从 ToolMessage 中提取搜索结果

    逻辑：遍历最近的消息，找到 search_jobs_tool 返回的 ToolMessage，
    从中解析出结构化的职位数据，写入 state["search_results"]。
    """
    import json

    messages = state["messages"]
    search_results = None

    # 倒序查找最近的 search_jobs_tool 结果
    for msg in reversed(messages):
        if hasattr(msg, 'name') and msg.name == "search_jobs_tool":
            try:
                # Tool 返回的是 JSON 字符串
                data = json.loads(msg.content)
                if isinstance(data, dict) and "ui_cards" in data:
                    search_results = data["ui_cards"]
            except (json.JSONDecodeError, TypeError):
                pass
            break

    return {"search_results": search_results}


# ================= 4. 自定义路由：工具调用后是否需要提取结果 =================

def after_tools_route(state: AgentState):
    """
    工具执行完毕后的路由：
    - 如果刚执行了 search_jobs_tool -> 先去 extract 节点提取 UI 数据
    - 否则 -> 直接回 bot_node 继续对话
    """
    messages = state["messages"]

    # 检查最后一条消息是否是 search_jobs_tool 的返回
    if messages:
        last_msg = messages[-1]
        if hasattr(last_msg, 'name') and last_msg.name == "search_jobs_tool":
            return "extract_search_results"

    return "bot_node"


# ================= 5. 构建图 =================
workflow = StateGraph(AgentState)

workflow.add_node("bot_node", bot_node)
workflow.add_node("summarize_node", summarize_node)
workflow.add_node("tools", ToolNode(tools))
workflow.add_node("extract_search_results", extract_search_results)  # 【新增】

checkpointer = MemorySaver()

# 入口路由
workflow.add_conditional_edges(
    START,
    should_summarize,
    {
        "summarize_node": "summarize_node",
        "bot_node": "bot_node"
    }
)

workflow.add_edge("summarize_node", "bot_node")

# bot -> 工具判断
workflow.add_conditional_edges("bot_node", tools_condition)

# 【修改】工具执行完后，走自定义路由
workflow.add_conditional_edges(
    "tools",
    after_tools_route,
    {
        "extract_search_results": "extract_search_results",
        "bot_node": "bot_node"
    }
)

# extract 完成后回到 bot 继续生成回复
workflow.add_edge("extract_search_results", "bot_node")

app_graph = workflow.compile(checkpointer=checkpointer)