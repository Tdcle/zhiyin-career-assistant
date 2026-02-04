# logic/interview_flow.py
import json
import gradio as gr
from utils.database import DatabaseManager
from config.config import config
from utils.tools import analyze_job_match_tool
from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from utils.logger import sys_logger

# 【关键】导入新创建的面试图
from models.interview_graph import interview_graph

db = DatabaseManager()

# 仅保留一个简单的 Assistant 用于开场白 (或者也可以把开场白放进 Graph，但这里分开比较灵活)
llm_assistant = ChatTongyi(
    api_key=config.DASHSCOPE_API_KEY,
    model="qwen-turbo",
    streaming=False
)


def start_interview_logic(job_data_dict, user_info):
    """
    【核心逻辑】初始化模拟面试
    """
    if not job_data_dict or not isinstance(job_data_dict, dict):
        yield (
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(visible=True, value="❌ 无法获取职位数据"),
            None
        )
        return

    selected_job_id = job_data_dict.get('job_id')
    job_title = job_data_dict.get('title', '未知职位')

    # UI Loading
    yield (
        gr.update(visible=False), gr.update(visible=True), gr.update(value=None),
        "### ⏳ 正在分析简历匹配度...",
        [{"role": "assistant", "content": f"⏳ 正在连接【{job_title}】面试官..."}],
        f"🎙️ 正在连接面试房间...", None
    )

    # 查库
    job_detail = db.get_job_details(selected_job_id)
    if not job_detail:
        yield (gr.update(), gr.update(), gr.update(), "❌ 职位不存在", [], "❌ 错误", None)
        return

    try:
        user_id = user_info.split(" (")[0].strip()
        resume_data = db.get_latest_resume(user_id)
        # 获取长期记忆仅用于日志展示，实际逻辑在 Graph 中处理
        user_profile = db.get_user_profile(user_id)
        if user_profile:
            sys_logger.info(f"🧠 [Init] 用户画像: {user_profile}")
    except:
        user_id = "guest"

    # 画图
    radar_path = None
    analysis_txt = "暂时无法生成匹配分析。"
    try:
        match_res = analyze_job_match_tool.invoke({
            "user_id": user_id, "job_description": job_detail['detail']
        })
        if hasattr(match_res, 'content'):
            match_str = match_res.content
        else:
            match_str = str(match_res)
        match_data = json.loads(match_str)
        radar_path = match_data.get('image_path')
        analysis_txt = match_data.get('analysis')
    except Exception as e:
        sys_logger.error(f"雷达图失败: {e}")

    # 开场白 (依然由 Assistant 快速生成，不进入 Graph 循环，为了快)
    ai_opening = "请先做一下自我介绍。"
    try:
        prompt = f"""
        你是【{job_detail['company']}】的面试官。
        请生成一段简短开场白：欢迎候选人，简述流程，并请他做自我介绍。
        不要问技术题。
        """
        ai_msg = llm_assistant.invoke(prompt)
        ai_opening = ai_msg.content
    except:
        pass

    # 【关键】构造 Graph 所需的初始 Context
    # 注意：这里不需要 summary 和 turn_count，因为 Graph 会自己管理
    # 我们只需要把 job_detail 传给前端，前端在下一次调用 handle_interview_chat 时会传回来
    context_state = {
        "job_context": job_detail
    }

    yield (
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(value=radar_path),
        f"### 岗位匹配分析\n{analysis_txt}",
        [{"role": "assistant", "content": ai_opening}],
        f"🎙️ 正在面试：{job_detail['title']}",
        context_state
    )


def handle_interview_chat(message, history, context_state, user_info):
    """
    【核心逻辑】面试对话 (调用 Graph 版)
    """
    if not message.strip():
        yield "", history
        return

    # 1. 准备参数
    if not context_state: context_state = {}
    job_context = context_state.get("job_context", {})

    try:
        user_id = str(user_info.split(" (")[0].strip())
    except:
        user_id = "guest"

    # 2. UI 立即显示用户输入
    history.append({"role": "user", "content": message})
    yield "", history

    # 3. 准备 Graph 输入
    # 注意：LangGraph 的 MemorySaver 会根据 thread_id 自动加载之前的 messages
    # 我们只需要传入 新的消息 + 上下文更新
    inputs = {
        "messages": [HumanMessage(content=message)],
        "user_id": user_id,
        "job_context": job_context  # 确保 Context 始终是最新的
    }

    config = {"configurable": {"thread_id": f"interview_{user_id}"}}  # 使用独立于 Chat 的 Thread ID

    # 4. 流式调用 Graph
    history.append({"role": "assistant", "content": ""})  # 占位

    try:
        # stream_mode="messages" 会返回一个个 Message chunk
        event_stream = interview_graph.stream(inputs, config=config, stream_mode="messages")

        full_content = ""
        for msg, metadata in event_stream:
            # 过滤掉总结节点的中间消息，只看面试官的回复
            if metadata.get('langgraph_node') == "summarize_node":
                continue

            # 过滤掉 Tool 调用请求，只显示 Tool 的结果或 AI 的文本
            if isinstance(msg, AIMessage):
                # 如果 AI 决定调用保存画像工具，这里会收到 ToolCall，我们不显示 ToolCall 代码
                if msg.content:
                    full_content += msg.content
                    history[-1]['content'] = full_content
                    yield "", history

            # (可选) 如果你想显示 "记忆已保存" 的提示，可以处理 ToolMessage
            # 但通常面试中保持沉浸感，不需要显示 "已保存"

    except Exception as e:
        sys_logger.error(f"Graph Error: {e}", exc_info=True)
        history[-1]['content'] = f"❌ 面试官掉线了: {str(e)}"
        yield "", history