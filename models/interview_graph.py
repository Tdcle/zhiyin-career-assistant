import json
from typing import Annotated, Any, Dict, List, Optional

from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

from config.config import config
from utils.database import DatabaseManager
from utils.logger import get_logger
from utils.tools import save_preference_tool

db_manager = DatabaseManager()
logger = get_logger("interview_graph")

DEFAULT_SCORECARD = {
    "tech_depth": 0,
    "project_depth": 0,
    "experience_match": 0,
    "communication": 0,
    "jd_fit": 0,
}

DEFAULT_TOPICS = [
    "self_intro",
    "core_skill",
    "project_depth",
    "tradeoff",
    "problem_solving",
    "jd_fit",
]


class InterviewState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    user_id: str
    job_context: Dict[str, Any]
    summary: str
    phase: str
    interview_round: int
    scorecard: Dict[str, int]
    evidence_log: List[Dict[str, Any]]
    covered_topics: List[str]
    uncovered_topics: List[str]
    recommended_next_focus: str
    gap_hypotheses: List[str]
    risk_flags: List[str]
    stop_reason: str
    live_assessment_md: str
    final_report: Optional[str]
    should_end: bool


tools = [save_preference_tool]

interviewer_llm = config.create_tongyi(config.CHAT_MODELS.interviewer, streaming=True)
assessment_llm = config.create_tongyi(config.CHAT_MODELS.interview_assessment)
report_llm = config.create_tongyi(config.CHAT_MODELS.interview_report)
summary_llm = config.create_tongyi(config.CHAT_MODELS.interview_summary)

interviewer_with_tools = interviewer_llm.bind_tools(tools)


def _safe_json_loads(raw: str) -> dict:
    text = (raw or "").strip()
    if "```" in text:
        chunks = text.split("```")
        if len(chunks) >= 3:
            text = chunks[1].replace("json", "", 1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        text = text[start:end + 1]
    return json.loads(text)


def _normalize_scores(raw_scores: dict | None) -> dict:
    merged = dict(DEFAULT_SCORECARD)
    for key in DEFAULT_SCORECARD:
        try:
            value = int((raw_scores or {}).get(key, merged[key]))
        except (TypeError, ValueError):
            value = merged[key]
        merged[key] = max(0, min(100, value))
    return merged


def _merge_unique(existing: list | None, delta: list | None) -> list:
    merged = []
    for item in (existing or []) + (delta or []):
        if not item:
            continue
        if item not in merged:
            merged.append(item)
    return merged


def _format_dialogue(messages: list, limit: int = 10) -> str:
    lines = []
    kept = messages[-limit:]
    for message in kept:
        if not isinstance(message, (HumanMessage, AIMessage)) or not message.content:
            continue
        speaker = "candidate" if isinstance(message, HumanMessage) else "interviewer"
        lines.append(f"{speaker}: {message.content}")
    return "\n".join(lines)


def _render_live_assessment(
    interview_round: int,
    scorecard: dict,
    recommended_next_focus: str,
    uncovered_topics: list,
    risk_flags: list,
) -> str:
    lines = [
        "### 实时评估面板",
        f"- 面试轮次: {interview_round}",
        f"- 技术深度: {scorecard.get('tech_depth', 0)}",
        f"- 项目深度: {scorecard.get('project_depth', 0)}",
        f"- 经验匹配: {scorecard.get('experience_match', 0)}",
        f"- 表达沟通: {scorecard.get('communication', 0)}",
        f"- JD 契合度: {scorecard.get('jd_fit', 0)}",
        f"- 下一问焦点: {recommended_next_focus or '继续核验核心能力'}",
        f"- 未覆盖主题: {', '.join(uncovered_topics) if uncovered_topics else '已覆盖核心主题'}",
        f"- 风险信号: {', '.join(risk_flags) if risk_flags else '暂无明显风险'}",
    ]
    return "\n".join(lines)


def summarize_node(state: InterviewState):
    summary = state.get("summary", "")
    dialogue = _format_dialogue(state.get("messages", []), limit=12)
    if not dialogue:
        return {}

    if summary:
        instruction = (
            f"已有摘要：{summary}\n"
            "请把已有摘要和最新对话合并成更紧凑的面试摘要，保留能力证据、暴露短板和已覆盖主题。"
        )
    else:
        instruction = "请基于下面对话生成一段紧凑面试摘要，保留能力证据、暴露短板和已覆盖主题。"

    prompt = (
        f"{instruction}\n\n"
        "要求：\n"
        "1. 只写摘要正文，不要加标题。\n"
        "2. 保留可复用信息：候选人经历、技术栈、项目细节、明显风险。\n"
        "3. 避免复述寒暄和重复问答。\n\n"
        f"对话：\n{dialogue}"
    )
    response = summary_llm.invoke(prompt)
    messages = state.get("messages", [])
    delete_messages = [RemoveMessage(id=message.id) for message in messages[:-6]]
    return {"summary": response.content.strip(), "messages": delete_messages}


def route_from_start(state: InterviewState):
    phase = state.get("phase", "interviewing")
    if phase != "finalizing" and len(state.get("messages", [])) > 12:
        return "summarize_node"
    if phase == "finalizing" or state.get("should_end", False):
        return "report_node"
    if phase == "opening":
        return "interviewer_node"
    return "assessment_node"


def route_after_summary(state: InterviewState):
    phase = state.get("phase", "interviewing")
    if phase == "finalizing" or state.get("should_end", False):
        return "report_node"
    if phase == "opening":
        return "interviewer_node"
    return "assessment_node"


def assessment_node(state: InterviewState):
    messages = state.get("messages", [])
    latest_answer = ""
    for message in reversed(messages):
        if isinstance(message, HumanMessage) and message.content:
            latest_answer = message.content
            break

    if not latest_answer:
        return {}

    job_context = state.get("job_context", {})
    prompt = f"""
你是后台评审 agent，只负责评估，不直接和候选人对话。

岗位信息：
- 公司: {job_context.get("company", "未知公司")}
- 岗位: {job_context.get("title", "未知岗位")}
- JD 摘要: {(job_context.get("detail", "") or "")[:1000]}

已有摘要：
{state.get("summary", "暂无")}

最近对话：
{_format_dialogue(messages, limit=8)}

上一轮结构化状态：
- scorecard: {json.dumps(state.get("scorecard") or DEFAULT_SCORECARD, ensure_ascii=False)}
- covered_topics: {json.dumps(state.get("covered_topics") or [], ensure_ascii=False)}
- uncovered_topics: {json.dumps(state.get("uncovered_topics") or DEFAULT_TOPICS, ensure_ascii=False)}
- risk_flags: {json.dumps(state.get("risk_flags") or [], ensure_ascii=False)}

请严格输出 JSON，字段必须完整：
{{
  "scorecard": {{
    "tech_depth": 0,
    "project_depth": 0,
    "experience_match": 0,
    "communication": 0,
    "jd_fit": 0
  }},
  "evidence_delta": [
    {{
      "dimension": "project_depth",
      "signal": "positive",
      "quote": "候选人的关键表述",
      "reason": "为什么支持这个判断"
    }}
  ],
  "covered_topics": ["self_intro"],
  "uncovered_topics": ["tradeoff"],
  "recommended_next_focus": "project_tradeoff",
  "gap_hypotheses": ["项目复杂度证据不足"],
  "risk_flags": ["回答偏抽象"],
  "should_end": false,
  "stop_reason": ""
}}

要求：
1. 分数要和证据一致，不能客气。
2. 如果候选人已经明显无法补充新信息，或核心主题已覆盖，should_end 可设为 true。
3. uncovered_topics 尽量控制在 1-3 个。
"""
    try:
        parsed = _safe_json_loads(assessment_llm.invoke(prompt).content)
    except Exception as exc:
        logger.error("assessment parse failed: %s", exc, exc_info=True)
        parsed = {}

    interview_round = int(state.get("interview_round", 0)) + 1
    scorecard = _normalize_scores(parsed.get("scorecard"))
    covered_topics = _merge_unique(state.get("covered_topics"), parsed.get("covered_topics"))
    uncovered_topics = parsed.get("uncovered_topics") or [
        item for item in DEFAULT_TOPICS if item not in covered_topics
    ]
    evidence_log = _merge_unique(state.get("evidence_log"), parsed.get("evidence_delta"))
    evidence_log = evidence_log[-12:]
    risk_flags = _merge_unique(state.get("risk_flags"), parsed.get("risk_flags"))
    gap_hypotheses = _merge_unique(state.get("gap_hypotheses"), parsed.get("gap_hypotheses"))
    recommended_next_focus = parsed.get("recommended_next_focus") or (
        uncovered_topics[0] if uncovered_topics else "综合复盘"
    )

    should_end = bool(parsed.get("should_end", False))
    stop_reason = parsed.get("stop_reason", "")
    if interview_round >= 6 and not should_end:
        should_end = True
        stop_reason = "已达到轮次上限"
    if not uncovered_topics and not should_end:
        should_end = True
        stop_reason = "核心主题已覆盖"

    live_assessment_md = _render_live_assessment(
        interview_round=interview_round,
        scorecard=scorecard,
        recommended_next_focus=recommended_next_focus,
        uncovered_topics=uncovered_topics,
        risk_flags=risk_flags,
    )

    return {
        "phase": "interviewing",
        "interview_round": interview_round,
        "scorecard": scorecard,
        "covered_topics": covered_topics,
        "uncovered_topics": uncovered_topics,
        "recommended_next_focus": recommended_next_focus,
        "gap_hypotheses": gap_hypotheses,
        "risk_flags": risk_flags,
        "evidence_log": evidence_log,
        "should_end": should_end,
        "stop_reason": stop_reason,
        "live_assessment_md": live_assessment_md,
    }


def interviewer_node(state: InterviewState):
    user_id = state["user_id"]
    job_context = state.get("job_context", {})
    company = job_context.get("company", "目标公司")
    title = job_context.get("title", "目标岗位")
    jd = (job_context.get("detail", "") or "")[:1200]
    profile = db_manager.get_user_profile(user_id) or "暂无"
    phase = state.get("phase", "interviewing")

    if phase == "opening":
        prompt = f"""
你现在是 {company} 的技术面试官，正在面试岗位：{title}。

候选人画像：
- user_id: {user_id}
- 长期画像: {profile}

要求：
1. 只输出一段简短开场白，然后请候选人做自我介绍。
2. 不要直接进入技术追问。
3. 保持专业、克制、有压迫感，但不要失礼。
4. 输出控制在 120 字以内。
"""
    else:
        prompt = f"""
你现在是 {company} 的技术面试官，正在面试岗位：{title}。

岗位 JD 摘要：
{jd}

候选人画像：
- user_id: {user_id}
- 长期画像: {profile}
- 历史摘要: {state.get("summary", "暂无")}

后台评审状态：
- 当前轮次: {state.get("interview_round", 0)}
- scorecard: {json.dumps(state.get("scorecard") or DEFAULT_SCORECARD, ensure_ascii=False)}
- 下一问焦点: {state.get("recommended_next_focus", "继续核验核心能力")}
- 未覆盖主题: {json.dumps(state.get("uncovered_topics") or [], ensure_ascii=False)}
- 风险信号: {json.dumps(state.get("risk_flags") or [], ensure_ascii=False)}
- stop_reason: {state.get("stop_reason", "")}

要求：
1. 先用一句话点评候选人刚才的回答，再继续追问。
2. 追问必须围绕“下一问焦点”或“未覆盖主题”，不要泛泛而谈。
3. 若候选人暴露了新的长期信息，可以调用 `save_preference_tool` 保存。
4. 单轮回复控制在 180 字以内。
5. 不要给教学式提示，不要抢答，不要总结整场面试。
"""

    response = interviewer_with_tools.invoke([SystemMessage(content=prompt)] + state.get("messages", []))
    return {"messages": [response], "phase": "interviewing"}


def report_node(state: InterviewState):
    user_id = state["user_id"]
    job_context = state.get("job_context", {})
    company = job_context.get("company", "目标公司")
    title = job_context.get("title", "目标岗位")
    jd = (job_context.get("detail", "") or "")[:1000]
    scorecard = _normalize_scores(state.get("scorecard"))
    evidence_log = state.get("evidence_log") or []
    uncovered_topics = state.get("uncovered_topics") or []
    risk_flags = state.get("risk_flags") or []
    gap_hypotheses = state.get("gap_hypotheses") or []
    avg_score = round(sum(scorecard.values()) / max(len(scorecard), 1) / 10, 1)

    report_prompt = f"""
你是资深技术面试评审专家。请基于结构化状态输出一份专业 Markdown 报告。

岗位信息：
- 公司: {company}
- 岗位: {title}
- JD 摘要: {jd}

面试摘要：
{state.get("summary", "暂无")}

最近对话：
{_format_dialogue(state.get("messages", []), limit=10)}

结构化状态：
- 轮次: {state.get("interview_round", 0)}
- scorecard: {json.dumps(scorecard, ensure_ascii=False)}
- evidence_log: {json.dumps(evidence_log, ensure_ascii=False)}
- uncovered_topics: {json.dumps(uncovered_topics, ensure_ascii=False)}
- gap_hypotheses: {json.dumps(gap_hypotheses, ensure_ascii=False)}
- risk_flags: {json.dumps(risk_flags, ensure_ascii=False)}
- stop_reason: {state.get("stop_reason", "用户主动结束")}

请严格使用如下结构：
## 面试评估报告
**面试岗位**: {title} @ {company}
**综合评分**: {avg_score} / 10
### 亮点
### 主要不足
### 针对性建议
### 通过概率预估

要求：
1. 亮点和不足必须引用 evidence_log 中的具体证据。
2. 建议要针对岗位和候选人差距，不要空话。
3. 通过概率给出百分比，并说明依据。
4. 输出报告正文，不要输出 JSON。
"""
    logger.info("generating interview report for user=%s", user_id)
    response = report_llm.invoke(report_prompt)
    return {
        "final_report": response.content.strip(),
        "phase": "finalizing",
        "live_assessment_md": _render_live_assessment(
            interview_round=state.get("interview_round", 0),
            scorecard=scorecard,
            recommended_next_focus=state.get("recommended_next_focus", "综合复盘"),
            uncovered_topics=uncovered_topics,
            risk_flags=risk_flags,
        ),
    }


workflow = StateGraph(InterviewState)
workflow.add_node("summarize_node", summarize_node)
workflow.add_node("assessment_node", assessment_node)
workflow.add_node("interviewer_node", interviewer_node)
workflow.add_node("tools", ToolNode(tools))
workflow.add_node("report_node", report_node)

workflow.add_conditional_edges(
    START,
    route_from_start,
    {
        "summarize_node": "summarize_node",
        "assessment_node": "assessment_node",
        "interviewer_node": "interviewer_node",
        "report_node": "report_node",
    },
)
workflow.add_conditional_edges(
    "summarize_node",
    route_after_summary,
    {
        "assessment_node": "assessment_node",
        "interviewer_node": "interviewer_node",
        "report_node": "report_node",
    },
)
workflow.add_edge("assessment_node", "interviewer_node")
workflow.add_conditional_edges("interviewer_node", tools_condition)
workflow.add_edge("tools", "interviewer_node")
workflow.add_edge("report_node", END)

interview_graph = workflow.compile(checkpointer=MemorySaver())
