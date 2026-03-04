# logic/chat_flow.py
import os
import json
import gradio as gr
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from models.chat_graph import app_graph
from utils.logger import sys_logger
from utils.database import DatabaseManager
from utils.file_parser import FileParser

# 【删除】不再需要 GlobalState
# from utils.global_state import GlobalState

db = DatabaseManager()


def process_uploaded_resume(file_path, user_info):
    if not file_path:
        return "无文件"
    try:
        user_id = user_info.split(" (")[0].strip()
    except (AttributeError, IndexError):
        return "⚠️ 请先选择有效的用户身份"

    sys_logger.info(f"📄 [Resume] 用户[{user_id}] 上传文件: {file_path}")
    raw_content = FileParser.read_file(file_path)
    if not raw_content or len(raw_content) < 10:
        return "❌ 解析失败或内容过少"
    filename = os.path.basename(file_path)
    success, msg = db.save_resume(user_id, filename, raw_content)
    if success:
        return f"✅ 简历已入库！\n({filename})"
    else:
        return f"❌ 处理失败: {msg}"


def respond(message, chat_history, user_info):
    """
    主对话逻辑：
    - 流式输出 LLM 回复
    - 从 Graph State 中读取 search_results 更新按钮 (取代 GlobalState)
    """

    # --- 1. 初始化 ---
    current_btn_updates = [gr.update(visible=False, value="")] * 6
    current_jobs_data = []

    if not message.strip():
        yield chat_history, "", current_jobs_data, *current_btn_updates
        return

    try:
        user_id = str(user_info.split(" (")[0].strip())
    except (AttributeError, IndexError):
        user_id = "guest"

    sys_logger.info(f"\n📨 [Chat] 用户[{user_id}]: {message}")

    # --- 2. 用户消息上屏 ---
    chat_history.append({"role": "user", "content": message})
    yield chat_history, "", current_jobs_data, *current_btn_updates

    # --- 3. 构造输入 ---
    input_message = HumanMessage(content=message)
    system_prompt = SystemMessage(
        content=f"当前服务的用户ID是: {user_id}。如果用户要求根据简历推荐工作，请调用 get_user_resume_tool。在调用搜索工具时，请务必传入正确的 user_id。"
    )

    inputs = {"messages": [system_prompt, input_message], "user_id": user_id}
    graph_config = {"configurable": {"thread_id": user_id}}

    chat_history.append({"role": "assistant", "content": "🤔 正在思考..."})
    full_response = ""

    try:
        # --- 4. 流式调用 Graph ---
        event_stream = app_graph.stream(
            inputs, config=graph_config, stream_mode="messages"
        )

        for msg, metadata in event_stream:
            node_name = metadata.get('langgraph_node', '')

            # 跳过总结节点和提取节点的中间输出
            if node_name in ("summarize_node", "extract_search_results"):
                continue

            if isinstance(msg, AIMessage):
                # 处理工具调用提示
                if msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        if tool_call['name'] == 'search_jobs_tool':
                            full_response += "\n> 🔍 *正在检索职位数据库...*\n\n"
                    chat_history[-1]["content"] = full_response
                    yield chat_history, "", current_jobs_data, *current_btn_updates

                # 处理文本内容
                if msg.content:
                    full_response += msg.content
                    chat_history[-1]["content"] = full_response
                    yield chat_history, "", current_jobs_data, *current_btn_updates

        # --- 5.【关键改动】流结束后，从 Graph State 中读取搜索结果 ---
        final_state = app_graph.get_state(graph_config)
        search_results = final_state.values.get("search_results")

        if search_results and isinstance(search_results, list) and len(search_results) > 0:
            sys_logger.info(f"✅ [UI] 从 Graph State 读取到 {len(search_results)} 条职位数据")

            current_jobs_data = search_results
            new_updates = []

            for i in range(len(search_results)):
                if i < len(search_results):
                    job = search_results[i]
                    tags_clean = job.get('tags', '').replace('|', '·')
                    btn_text = (
                        f"🏢 {job['company']}\n"
                        f"💼 {job['title']}\n"
                        f"💰 {job['salary']}   📍 {tags_clean}"
                    )
                    new_updates.append(gr.update(visible=True, value=btn_text))
                else:
                    new_updates.append(gr.update(visible=False, value=""))

            current_btn_updates = new_updates

        # 最终 yield
        yield chat_history, "", current_jobs_data, *current_btn_updates

    except Exception as e:
        sys_logger.error(str(e), exc_info=True)
        chat_history[-1]["content"] += f"\n\n❌ 系统错误: {str(e)}"
        yield chat_history, "", current_jobs_data, *current_btn_updates