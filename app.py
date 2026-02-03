import gradio as gr
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from models.graph import app_graph
from utils.logger import setup_logger
from utils.database import DatabaseManager
from utils.file_parser import FileParser

# --- 1. 初始化系统 ---
sys_logger, current_log_path = setup_logger(name="JobAgent")
sys_logger.info("🚀 系统启动中...")

db = DatabaseManager()


# --- 2. 业务逻辑 ---

def process_uploaded_resume(file_path, user_info):
    """
    【后台核心逻辑】
    1. 解析文件 (PDF/图片) -> 文本
    2. 存入数据库 (自动向量化)
    """
    if not file_path:
        return "无文件"

    # 解析 User ID
    try:
        user_id = user_info.split(" (")[0].strip()
    except:
        return "⚠️ 请先选择有效的用户身份"

    sys_logger.info(f"📄 [Resume] 用户[{user_id}] 上传文件: {file_path}")

    # A. 调用解析器提取文本 (不经过模型，纯规则解析)
    raw_content = FileParser.read_file(file_path)

    if not raw_content or len(raw_content) < 10:
        return "❌ 解析失败或内容过少，请检查文件"

    # 获取文件名
    import os
    filename = os.path.basename(file_path)

    # B. 存入数据库
    success, msg = db.save_resume(user_id, filename, raw_content)

    if success:
        sys_logger.info(f"✅ [Resume] 简历已入库: {filename}")
        return f"✅ 简历解析并入库成功！\n(文件名: {filename})\n现在的我已记住了你的经历，快问我推荐什么工作吧！"
    else:
        sys_logger.error(f"❌ [Resume] 入库失败: {msg}")
        return f"❌ 处理失败: {msg}"


def respond(message, chat_history, user_info):
    """交互逻辑"""
    if not message.strip():
        yield chat_history, ""
        return

    try:
        user_id = user_info.split(" (")[0].strip()
    except:
        user_id = "guest"

    sys_logger.info(f"\n📨 [Chat] 用户[{user_id}]: {message}")

    chat_history.append({"role": "user", "content": message})
    yield chat_history, ""

    # 【重要】在 Prompt 中注入 UserID，这样模型在调用 get_user_resume_tool 时才知道传什么 ID
    # 或者我们可以直接在 SystemPrompt 里注入
    input_message = HumanMessage(content=message)

    # 构造 SystemMessage 告知模型当前服务的用户ID
    system_prompt = SystemMessage(
        content=f"当前服务的用户ID是: {user_id}。如果用户要求根据简历推荐工作，请调用 get_user_resume_tool 获取该用户的简历信息。")

    inputs = {"messages": [system_prompt, input_message], "user_id": user_id}
    config = {"configurable": {"thread_id": user_id}}

    chat_history.append({"role": "assistant", "content": "🤔 正在思考..."})
    full_response = ""

    try:
        event_stream = app_graph.stream(inputs, config=config, stream_mode="messages")

        for msg, metadata in event_stream:
            if metadata.get('langgraph_node') == "summarize_node": continue

            if isinstance(msg, AIMessage):
                if msg.tool_calls:
                    # 如果调用了获取简历的工具
                    if any(tool['name'] == 'get_user_resume_tool' for tool in msg.tool_calls):
                        full_response += "\n> 📂 *正在从数据库调取您的简历档案...*\n\n"
                    elif any(tool['name'] == 'search_jobs_tool' for tool in msg.tool_calls):
                        full_response += "\n> 🔍 *正在根据条件检索职位...*\n\n"
                    else:
                        full_response += "\n> 🛠️ *正在调用工具...*\n\n"

                    chat_history[-1]["content"] = full_response
                    yield chat_history, ""

                if msg.content:
                    full_response += msg.content
                    chat_history[-1]["content"] = full_response
                    yield chat_history, ""

        sys_logger.info(f"✅ [Chat] 回复完成")

    except Exception as e:
        sys_logger.error(str(e), exc_info=True)
        chat_history[-1]["content"] += f"\n\n❌ 系统错误: {str(e)}"
        yield chat_history, ""


# ... (create_new_user, refresh_user_dropdown 等保持不变) ...

# --- 3. 构建 UI ---

try:
    with open("assets/style.css", "r", encoding="utf-8") as f:
        custom_css = f.read()
except FileNotFoundError:
    custom_css = ""

with gr.Blocks(title="AI 智能招聘助手") as demo:
    with gr.Row(variant="panel"):
        gr.Markdown("## 🤖 AI 智能招聘助手")

    with gr.Row():
        # --- 左侧 ---
        with gr.Column(scale=1):
            gr.Markdown("### 👤 用户切换")
            user_dropdown = gr.Dropdown(
                choices=db.get_all_users_list(),
                value=lambda: db.get_all_users_list()[0] if db.get_all_users_list() else None,
                label="当前身份",
                interactive=True
            )
            refresh_btn = gr.Button("🔄 刷新列表", size="sm")

            gr.Markdown("---")
            gr.Markdown("### ➕ 新建用户")
            with gr.Group():
                new_name_input = gr.Textbox(label="用户昵称")
                create_btn = gr.Button("创建并切换", variant="secondary")
                create_status = gr.Markdown(value="")

            gr.Markdown("---")
            gr.Markdown("### 📄 简历入库")

            # 【核心修改】上传即处理，不需要“开始分析”按钮了
            resume_file = gr.File(
                label="拖拽上传简历 (PDF/图片)",
                file_types=[".pdf", ".png", ".jpg", ".jpeg"],
                type="filepath",
                height=120
            )
            # 状态反馈区
            upload_status = gr.Textbox(label="系统处理状态", value="等待上传...", interactive=False)

            gr.Markdown("---")
            gr.Markdown(f"📡 **系统状态**\nLog: `{current_log_path}`")

        # --- 右侧 ---
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="对话历史",
                height=700,
                elem_classes="chatbot-container",
                avatar_images=("assets/user.png", "assets/bot.png")
            )

            with gr.Row():
                msg_input = gr.Textbox(
                    show_label=False,
                    placeholder="例如：根据我的简历推荐3个匹配的职位...",
                    scale=5,
                    container=False,
                    autofocus=True
                )
                send_btn = gr.Button("发送 🚀", variant="primary", scale=1)

    # --- 事件绑定 ---

    # 1. 基础事件
    refresh_btn.click(lambda: gr.Dropdown(choices=db.get_all_users_list()), None, user_dropdown)
    create_btn.click(lambda name: db.create_user(name) and (f"创建成功", gr.Dropdown(choices=db.get_all_users_list())),
                     inputs=[new_name_input], outputs=[create_status, user_dropdown])

    # 2. 【核心】简历上传事件 -> 触发后台处理
    resume_file.upload(
        fn=process_uploaded_resume,
        inputs=[resume_file, user_dropdown],  # 传入文件和当前用户ID
        outputs=[upload_status]  # 结果显示在状态框
    )

    # 3. 对话事件
    msg_input.submit(respond, [msg_input, chatbot, user_dropdown], [chatbot, msg_input])
    send_btn.click(respond, [msg_input, chatbot, user_dropdown], [chatbot, msg_input])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, theme=gr.themes.Soft(primary_hue="blue"), css=custom_css)