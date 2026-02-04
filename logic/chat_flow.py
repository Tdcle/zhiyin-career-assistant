# logic/chat_flow.py
import os
import gradio as gr
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from models.chat_graph import app_graph
from utils.logger import sys_logger
from utils.database import DatabaseManager
from utils.file_parser import FileParser
from utils.global_state import GlobalState

db = DatabaseManager()


def process_uploaded_resume(file_path, user_info):
    if not file_path: return "无文件"
    try:
        user_id = user_info.split(" (")[0].strip()
    except:
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
    主对话逻辑：原生按钮版 (修复闪烁问题)
    """

    # --- 1. 初始化状态变量 (在循环外定义，防止被重置) ---
    # 默认状态：5个按钮全部隐藏
    current_btn_updates = [gr.update(visible=False, value="")] * 5
    current_jobs_data = []  # 用于存储到 gr.State 的数据

    if not message.strip():
        yield chat_history, "", current_jobs_data, *current_btn_updates
        return

    try:
        user_id = str(user_info.split(" (")[0].strip())
    except:
        user_id = "guest"

    sys_logger.info(f"\n📨 [Chat] 用户[{user_id}]: {message}")

    chat_history.append({"role": "user", "content": message})
    yield chat_history, "", current_jobs_data, *current_btn_updates

    input_message = HumanMessage(content=message)
    system_prompt = SystemMessage(
        content=f"当前服务的用户ID是: {user_id}。如果用户要求根据简历推荐工作，请调用 get_user_resume_tool 获取该用户的简历信息。在调用搜索工具时，请务必传入正确的 user_id。"
    )

    inputs = {"messages": [system_prompt, input_message], "user_id": user_id}
    config = {"configurable": {"thread_id": user_id}}

    chat_history.append({"role": "assistant", "content": "🤔 正在思考..."})
    full_response = ""

    try:
        event_stream = app_graph.stream(inputs, config=config, stream_mode="messages")

        for msg, metadata in event_stream:
            if metadata.get('langgraph_node') == "summarize_node": continue

            if isinstance(msg, AIMessage):
                # 1. 处理工具调用
                if msg.tool_calls:
                    for tool in msg.tool_calls:
                        if tool['name'] == 'search_jobs_tool':
                            full_response += "\n> 🔍 *正在检索职位数据库...*\n\n"
                    chat_history[-1]["content"] = full_response
                    # Yield 当前保持的状态
                    yield chat_history, "", current_jobs_data, *current_btn_updates

                # 2. 处理文本内容
                if msg.content:
                    full_response += msg.content
                    chat_history[-1]["content"] = full_response

                    # --- 核心修复逻辑 ---
                    # 检查缓存中是否有新数据
                    new_jobs = GlobalState.get_search_results(user_id)

                    if new_jobs:
                        sys_logger.info(f"✅ [UI-HIT] 发现 {len(new_jobs)} 条数据，更新按钮状态...")

                        current_jobs_data = new_jobs

                        new_updates = []
                        for i in range(5):
                            if i < len(new_jobs):
                                job = new_jobs[i]
                                tags_clean = job['tags'].replace('|', '·')

                                # 【核心修改】回归纯文本，用 \n 换行
                                # 第一行：公司名 (将在 CSS 中被 ::first-line 选中放大)
                                # 第二行：职位名
                                # 第三行：薪资与其他
                                btn_text = (
                                    f"🏢 {job['company']}\n"
                                    f"💼 {job['title']}\n"
                                    f"💰 {job['salary']}   📍 {tags_clean}"
                                )

                                new_updates.append(gr.update(visible=True, value=btn_text))
                            else:
                                new_updates.append(gr.update(visible=False, value=""))

                        current_btn_updates = new_updates
                        GlobalState.clear(user_id)

                    # 无论这次循环有没有读到新数据，都 Yield 最新的 current_btn_updates
                    # 这样如果之前已经变显示了，这里就会一直发“显示”，不会变回“隐藏”
                    yield chat_history, "", current_jobs_data, *current_btn_updates

    except Exception as e:
        sys_logger.error(str(e), exc_info=True)
        chat_history[-1]["content"] += f"\n\n❌ 系统错误: {str(e)}"
        yield chat_history, "", current_jobs_data, *current_btn_updates