# app.py
import gradio as gr
import os
from utils.logger import sys_logger, current_log_path
from utils.database import DatabaseManager

# 导入业务逻辑
from logic.chat_flow import process_uploaded_resume, respond
from logic.interview_flow import start_interview_logic, handle_interview_chat

# 初始化
sys_logger.info("🚀 系统启动中...")
os.makedirs("static", exist_ok=True)
db = DatabaseManager()

# 读取 CSS
css_path = "assets/style.css"
try:
    with open(css_path, "r", encoding="utf-8") as f:
        full_css = f.read()
except FileNotFoundError:
    full_css = ""


def create_new_user(new_name):
    if not new_name.strip(): return "⚠️ 不能为空", gr.Dropdown()
    success, msg = db.create_user(new_name)
    all_users = db.get_all_users_list()
    return (f"✅ {msg}", gr.Dropdown(choices=all_users, value=all_users[0] if success else None))


def refresh_user_dropdown():
    new_choices = db.get_all_users_list()
    return gr.Dropdown(choices=new_choices, value=new_choices[0] if new_choices else None)


def create_click_handler(index):
    def handler(jobs_data, user_info):
        if not jobs_data or index >= len(jobs_data):
            yield (gr.update(),) * 7
            return
        yield from start_interview_logic(jobs_data[index], user_info)

    return handler


# --- UI 构建 ---
with gr.Blocks(title="AI 智能招聘助手") as demo:
    jobs_state = gr.State([])
    interview_context_state = gr.State({})

    with gr.Row(variant="panel"):
        gr.Markdown("## 🤖 AI 智能招聘助手 (Pro版)")

    with gr.Row():
        # === 左侧边栏 ===
        with gr.Column(scale=1):
            gr.Markdown("### 👤 用户切换")
            user_dropdown = gr.Dropdown(
                choices=db.get_all_users_list(),
                value=lambda: db.get_all_users_list()[0] if db.get_all_users_list() else None,
                label="当前身份", interactive=True
            )
            refresh_btn = gr.Button("🔄 刷新", size="sm")

            with gr.Accordion("➕ 新建用户", open=False):
                new_name_input = gr.Textbox(label="昵称")
                create_btn = gr.Button("创建")
                create_status = gr.Markdown()

            gr.Markdown("---")
            gr.Markdown("### 📄 简历管理")
            resume_file = gr.File(label="上传简历", file_types=[".pdf", ".png", ".jpg"], height=100)
            upload_status = gr.Textbox(label="状态", interactive=False)

            gr.Markdown(f"\nLog: `{os.path.basename(current_log_path)}`")

        # === 右侧主区域 ===
        with gr.Column(scale=3):
            # --- 模式 1: 求职大厅 ---
            with gr.Group(visible=True) as main_group:
                chatbot = gr.Chatbot(label="求职助手", height=500, elem_classes="chatbot-container",
                                     avatar_images=("assets/user.png", "assets/bot.png"))

                with gr.Row():
                    msg_input = gr.Textbox(show_label=False, placeholder="例如：帮我找北京的Java工作...", scale=5,
                                           autofocus=True)
                    send_btn = gr.Button("发送", variant="primary", scale=1)

                gr.Markdown("### 🎯 推荐职位列表")

                with gr.Row():
                    with gr.Column(scale=1, min_width=250):
                        btn_job_0 = gr.Button(visible=False, elem_classes="job-card-btn")
                    with gr.Column(scale=1, min_width=250):
                        btn_job_1 = gr.Button(visible=False, elem_classes="job-card-btn")

                with gr.Row():
                    with gr.Column(scale=1, min_width=250):
                        btn_job_2 = gr.Button(visible=False, elem_classes="job-card-btn")
                    with gr.Column(scale=1, min_width=250):
                        btn_job_3 = gr.Button(visible=False, elem_classes="job-card-btn")

                with gr.Row():
                    with gr.Column(scale=1, min_width=250):
                        btn_job_4 = gr.Button(visible=False, elem_classes="job-card-btn")
                    with gr.Column(scale=1, min_width=250):
                        gr.Markdown("", visible=True)

                        # --- 模式 2: 模拟面试室 ---
            with gr.Group(visible=False) as interview_group:
                interview_header = gr.Markdown("## 🎙️ 面试连接中...")

                with gr.Row():
                    with gr.Column(scale=1):
                        radar_image = gr.Image(label="岗位匹配雷达", interactive=False)
                        analysis_text = gr.Markdown(
                            "等待分析...",
                            elem_classes="match-analysis-card"
                        )

                    with gr.Column(scale=2):
                        interview_chatbot = gr.Chatbot(label="面试官", height=600,
                                                       avatar_images=("assets/user.png", "assets/bot.png"))
                        interview_input = gr.Textbox(placeholder="回答面试官的问题...", show_label=False)

                end_interview_btn = gr.Button("结束面试 / 返回大厅", variant="stop")

    # === 事件绑定 ===
    refresh_btn.click(refresh_user_dropdown, None, user_dropdown)
    create_btn.click(create_new_user, [new_name_input], [create_status, user_dropdown])
    resume_file.upload(process_uploaded_resume, [resume_file, user_dropdown], [upload_status])

    chat_outputs = [chatbot, msg_input, jobs_state, btn_job_0, btn_job_1, btn_job_2, btn_job_3, btn_job_4]
    msg_input.submit(respond, [msg_input, chatbot, user_dropdown], chat_outputs)
    send_btn.click(respond, [msg_input, chatbot, user_dropdown], chat_outputs)

    interview_outputs = [
        main_group, interview_group,
        radar_image, analysis_text,
        interview_chatbot, interview_header,
        interview_context_state
    ]

    btn_job_0.click(create_click_handler(0), inputs=[jobs_state, user_dropdown], outputs=interview_outputs)
    btn_job_1.click(create_click_handler(1), inputs=[jobs_state, user_dropdown], outputs=interview_outputs)
    btn_job_2.click(create_click_handler(2), inputs=[jobs_state, user_dropdown], outputs=interview_outputs)
    btn_job_3.click(create_click_handler(3), inputs=[jobs_state, user_dropdown], outputs=interview_outputs)
    btn_job_4.click(create_click_handler(4), inputs=[jobs_state, user_dropdown], outputs=interview_outputs)

    # 【重要】删除了之前的 interview_chat dummy 函数
    # 只保留 handle_interview_chat，并确保 inputs 正确
    interview_input.submit(
        handle_interview_chat,
        inputs=[interview_input, interview_chatbot, interview_context_state, user_dropdown],
        outputs=[interview_input, interview_chatbot]
    )

    end_interview_btn.click(
        lambda: (gr.update(visible=True), gr.update(visible=False)),
        None,
        [main_group, interview_group]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, theme=gr.themes.Soft(primary_hue="blue"), css=full_css)