import gradio as gr
from langchain_core.messages import HumanMessage, AIMessage
from models.graph import app_graph
from utils.logger import setup_logger
from utils.database import DatabaseManager  # 导入数据库管理器

# --- 1. 初始化系统 ---
sys_logger, current_log_path = setup_logger(name="JobAgent")
sys_logger.info("🚀 系统启动中...")

# 初始化数据库
db = DatabaseManager()


# --- 2. 核心业务逻辑 ---

def refresh_user_dropdown():
    """从数据库读取最新用户列表，刷新下拉框"""
    new_choices = db.get_all_users_list()
    # 返回 gr.Dropdown 的更新对象
    return gr.Dropdown(choices=new_choices, value=new_choices[0] if new_choices else None)


def create_new_user(new_name):
    """
    处理创建用户请求
    只需要传入 name，ID 由后端自动生成
    """
    if not new_name.strip():
        return "⚠️ 用户昵称不能为空", gr.Dropdown()

    # 调用数据库创建方法 (只传 name)
    success, msg = db.create_user(new_name)

    if success:
        # 成功后，重新获取列表，并自动选中刚刚创建的那个用户 (通常是列表第一个或根据ID判断)
        # 这里为了简单，我们刷新列表并默认选中最新的（因为我们SQL是按ID倒序排的）
        all_users = db.get_all_users_list()

        # 解析出新生成的 ID (从 msg 里或者重新查询，这里简单处理默认选第一个)
        # 我们的 get_all_users_list SQL 是 ORDER BY user_id DESC，所以最新的在第一个
        new_value = all_users[0] if all_users else None

        return f"✅ {msg}", gr.Dropdown(choices=all_users, value=new_value)
    else:
        return f"❌ {msg}", gr.Dropdown()


def respond(message, chat_history, user_info):
    """
    交互逻辑
    """
    if not message.strip():
        yield chat_history, ""
        return

    # 解析 user_id (取括号前的内容)
    try:
        user_id = user_info.split(" (")[0].strip()
    except:
        user_id = "guest"

    sys_logger.info(f"\n📨 [Chat] 用户[{user_id}]: {message}")

    # 1. 更新前端用户消息
    chat_history.append({"role": "user", "content": message})
    yield chat_history, ""

    # 2. 构造输入
    input_message = HumanMessage(content=message)
    inputs = {"messages": [input_message], "user_id": user_id}
    config = {"configurable": {"thread_id": user_id}}

    # 3. 预填充 AI 回复
    chat_history.append({"role": "assistant", "content": "🤔 正在思考..."})
    full_response = ""

    try:
        event_stream = app_graph.stream(inputs, config=config, stream_mode="messages")
        for msg, metadata in event_stream:
            if metadata.get('langgraph_node') == "summarize_node": continue

            if isinstance(msg, AIMessage):
                if msg.tool_calls:
                    full_response += "\n> 🛠️ *正在调用工具检索数据...*\n\n"
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


# --- 3. 构建 UI 界面 ---

try:
    with open("assets/style.css", "r", encoding="utf-8") as f:
        custom_css = f.read()
except FileNotFoundError:
    custom_css = ""

with gr.Blocks(title="AI 智能招聘助手", css=custom_css) as demo:
    with gr.Row(variant="panel"):
        gr.Markdown("## 🤖 AI 智能招聘助手")

    with gr.Row():
        # --- 左侧：用户管理面板 ---
        with gr.Column(scale=1):
            gr.Markdown("### 👤 用户切换")

            # 1. 用户选择下拉框
            user_dropdown = gr.Dropdown(
                choices=db.get_all_users_list(),  # 初始加载
                value=lambda: db.get_all_users_list()[0] if db.get_all_users_list() else None,
                label="当前身份",
                interactive=True
            )

            # 刷新按钮 (为了保险起见，有时需要手动刷新)
            refresh_btn = gr.Button("🔄 刷新列表", size="sm")

            gr.Markdown("---")
            gr.Markdown("### ➕ 新建用户")

            # 2. 新建用户表单
            with gr.Group():
                new_name_input = gr.Textbox(label="用户昵称", placeholder="例如: 王五 (Java)")
                create_btn = gr.Button("创建并切换", variant="secondary")
                create_status = gr.Markdown(value="")

            gr.Markdown("---")
            gr.Markdown(f"📡 **系统状态**\n\n日志: `{current_log_path}`")

        # --- 右侧：聊天主界面 ---
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="对话历史",
                # type="messages",  # 新版格式
                height=650,
                elem_classes="chatbot-container",
                avatar_images=("assets/user.png", "assets/bot.png")
            )

            with gr.Row():
                msg_input = gr.Textbox(
                    show_label=False,
                    placeholder="请输入您的需求...",
                    scale=5,
                    container=False,
                    autofocus=True
                )
                send_btn = gr.Button("发送 🚀", variant="primary", scale=1)

            gr.Examples(
                examples=["帮我找北京丰台的Java工作", "我叫小明，想找高薪职位"],
                inputs=[msg_input],
                label="试一试"
            )

    # --- 事件绑定逻辑 ---

    # 1. 刷新按钮：点击更新下拉框
    refresh_btn.click(
        fn=refresh_user_dropdown,
        inputs=None,
        outputs=[user_dropdown]
    )

    # 2. 创建用户按钮
    create_btn.click(
        fn=create_new_user,
        inputs=[new_name_input],
        outputs=[create_status, user_dropdown]  # 更新状态文字 + 刷新下拉框
    )

    # 3. 聊天发送
    msg_input.submit(
        fn=respond,
        inputs=[msg_input, chatbot, user_dropdown],
        outputs=[chatbot, msg_input]
    )
    send_btn.click(
        fn=respond,
        inputs=[msg_input, chatbot, user_dropdown],
        outputs=[chatbot, msg_input]
    )

if __name__ == "__main__":
    sys_logger.info("Web 服务正在启动...")
    demo.launch(server_name="127.0.0.1", server_port=7860, theme=gr.themes.Soft(primary_hue="blue"))