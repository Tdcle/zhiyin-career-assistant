import gradio as gr
from langchain_core.messages import HumanMessage
from graph import app_graph


def predict(message, history, user_id):
    """
    核心交互函数
    """
    # 1. 如果用户没填ID，给个默认值
    if not user_id:
        user_id = "default_user"

    print(f"\n📨 收到用户[{user_id}]的消息: {message}")

    # 2. 配置线程 ID (Thread ID)
    config = {"configurable": {"thread_id": user_id}}

    # 3. 构造输入
    input_message = HumanMessage(content=message)

    inputs = {
        "messages": [input_message],
        "user_id": user_id
    }

    try:
        # 4. 调用大脑
        print("🧠 Agent 正在思考 (读取记忆中)...")
        result = app_graph.invoke(inputs, config=config)

        # 5. 提取最终回复
        last_message = result["messages"][-1]
        response_content = last_message.content

        # 打印一下当前的 Summary 用于调试
        current_summary = result.get("summary", "")
        if current_summary:
            print(f"📘 [当前记忆摘要]: {current_summary[:50]}...")

        print("✅ 回复生成完毕")
        return response_content

    except Exception as e:
        print(f"❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        return f"系统出错了: {e}"


# ================= 搭建 UI 界面 =================
# 自定义 CSS
custom_css = """
footer {visibility: hidden}
.bubble-wrap {font-size: 16px !important}
"""

# 【修复点1】: 这里的 css 参数去掉，只保留 title
with gr.Blocks(title="AI 智能招聘助手") as demo:
    gr.Markdown("## 🤖 AI 智能招聘助手")

    with gr.Row():
        user_id_input = gr.Textbox(
            label="用户 ID (记忆密钥)",
            value="test_user_001",
            info="系统根据此 ID 保存对话上下文。切换 ID 会开启新的对话。"
        )

    # ChatInterface
    gr.ChatInterface(
        fn=predict,
        additional_inputs=[user_id_input],
        examples=[
            ["我叫乌龙茶，是一个Python全栈开发"],
            ["帮我找个北京的工作"],
            ["我们刚才聊了什么？"]
        ]
    )

if __name__ == "__main__":
    print("正在启动 Web 服务...")
    # 【修复点2】: 按照警告提示，将 css 移到 launch 方法中
    # 同时也建议加上 allowed_paths (如果有本地文件需要访问的话，这里暂时不需要)
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        css=custom_css
    )