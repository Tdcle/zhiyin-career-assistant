# logic/interview_flow.py
import json
import os
import gradio as gr
from utils.database import DatabaseManager
from utils.plotter import create_radar_chart
from config.config import config
from utils.tools import analyze_job_match_tool
from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from utils.logger import sys_logger

from models.interview_graph import interview_graph

db = DatabaseManager()

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
        yield (gr.update(),) * 9
        return

    selected_job_id = job_data_dict.get('job_id')
    job_title = job_data_dict.get('title', '未知职位')

    yield (
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(value=None),
        "### ⏳ 正在分析简历匹配度...",
        [{"role": "assistant", "content": f"⏳ 正在连接【{job_title}】面试官..."}],
        f"🎙️ 正在连接面试房间...",
        None,
        gr.update(visible=True, interactive=True, value="📝 结束面试并生成总结"),
        gr.update(visible=False)
    )

    job_detail = db.get_job_details(selected_job_id)
    if not job_detail:
        yield (
            gr.update(), gr.update(),
            gr.update(), "❌ 职位不存在",
            [], "❌ 错误", None,
            gr.update(visible=False), gr.update(visible=True)
        )
        return

    try:
        user_id = user_info.split(" (")[0].strip()
    except (AttributeError, IndexError):
        user_id = "guest"

    radar_path = None
    analysis_txt = "暂时无法生成匹配分析。"
    try:
        match_res = analyze_job_match_tool.invoke({
            "user_id": user_id, "job_description": job_detail['detail']
        })
        match_str = match_res.content if hasattr(match_res, 'content') else str(match_res)
        match_data = json.loads(match_str)
        radar_path = match_data.get('image_path')
        analysis_txt = match_data.get('analysis')
    except Exception as e:
        sys_logger.error(f"雷达图失败: {e}")

    ai_opening = "请先做一下自我介绍。"
    try:
        prompt = f"""
        你是【{job_detail['company']}】的面试官。
        请生成一段简短开场白：欢迎候选人，简述流程，并请他做自我介绍。
        不要问技术题。
        """
        ai_msg = llm_assistant.invoke(prompt)
        ai_opening = ai_msg.content
    except Exception:
        pass

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
        context_state,
        gr.update(visible=True, interactive=True, value="📝 结束面试并生成总结"),
        gr.update(visible=False)
    )


def handle_interview_chat(message, history, context_state, user_info):
    """
    【核心逻辑】面试对话
    """
    if not message.strip():
        yield "", history
        return

    if not context_state:
        context_state = {}
    job_context = context_state.get("job_context", {})

    try:
        user_id = str(user_info.split(" (")[0].strip())
    except (AttributeError, IndexError):
        user_id = "guest"

    history.append({"role": "user", "content": message})
    yield "", history

    inputs = {
        "messages": [HumanMessage(content=message)],
        "user_id": user_id,
        "job_context": job_context,
        "should_end": False
    }

    graph_config = {"configurable": {"thread_id": f"interview_{user_id}"}}
    history.append({"role": "assistant", "content": ""})

    try:
        event_stream = interview_graph.stream(
            inputs, config=graph_config, stream_mode="messages"
        )

        full_content = ""
        for msg, metadata in event_stream:
            if metadata.get('langgraph_node') == "summarize_node":
                continue
            if isinstance(msg, AIMessage) and msg.content:
                full_content += msg.content
                history[-1]['content'] = full_content
                yield "", history

    except Exception as e:
        sys_logger.error(f"Graph Error: {e}", exc_info=True)
        history[-1]['content'] = f"❌ 面试官掉线了: {str(e)}"
        yield "", history


def end_interview_with_summary(history, context_state, user_info):
    """
    【核心逻辑】结束面试并生成总结报告

    修复要点：
    1. 报告显示在全新的、独立的聊天气泡中
    2. 不拼接任何额外的评分表格（LLM 报告正文已包含评分）
    3. 不修改左侧雷达图和分析区
    """
    if not context_state:
        context_state = {}
    job_context = context_state.get("job_context", {})

    try:
        user_id = str(user_info.split(" (")[0].strip())
    except (AttributeError, IndexError):
        user_id = "guest"

    sys_logger.info(f"📝 [Interview] 用户 {user_id} 请求生成面试总结")

    # --- 1. 构造全新的 history 列表，确保新气泡独立 ---
    # 深拷贝当前历史，避免引用问题
    new_history = list(history)

    # 添加加载提示（独立气泡）
    new_history.append({
        "role": "assistant",
        "content": "⏳ **面试已结束，正在生成评估报告，请稍候...**"
    })

    # 第一次 yield：显示加载状态
    yield (
        new_history,
        gr.update(interactive=False, value="⏳ 正在生成报告..."),
        gr.update(visible=False)
    )

    # --- 2. 调用 Graph ---
    inputs = {
        "messages": [HumanMessage(content="[系统指令] 面试结束，请生成评估报告。")],
        "user_id": user_id,
        "job_context": job_context,
        "should_end": True
    }

    graph_config = {"configurable": {"thread_id": f"interview_{user_id}"}}

    try:
        # 同步执行，等待 report_node 完成
        interview_graph.invoke(inputs, config=graph_config)

        # --- 3. 从 State 中读取报告 ---
        final_state = interview_graph.get_state(graph_config)
        report_text = final_state.values.get("final_report", "")

        if not report_text:
            report_text = "⚠️ 报告生成为空，请稍后重试。"

        sys_logger.info(f"📊 [Interview] 报告生成完成，长度: {len(report_text)}")

        # --- 4. 用报告内容替换加载提示气泡 ---
        new_history[-1] = {
            "role": "assistant",
            "content": report_text
        }

        yield (
            new_history,
            gr.update(visible=False),
            gr.update(visible=True, value="🏠 返回求职大厅")
        )

    except Exception as e:
        sys_logger.error(f"面试总结生成失败: {e}", exc_info=True)
        new_history[-1] = {
            "role": "assistant",
            "content": f"❌ 总结生成失败: {str(e)}\n\n请点击返回按钮退出。"
        }
        yield (
            new_history,
            gr.update(visible=False),
            gr.update(visible=True, value="🏠 返回求职大厅")
        )