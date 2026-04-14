# Shared text/search helpers for the database package.



from __future__ import annotations



from datetime import date, datetime

import re



import jieba



_CUSTOM_WORDS = [
    "Python", "Java", "JavaScript", "TypeScript", "Golang", "C++", "C#",
    "SpringBoot", "Spring Boot", "Django", "Flask", "FastAPI",
    "Vue.js", "React", "Angular", "Node.js", "MyBatis",
    "MySQL", "PostgreSQL", "Redis", "MongoDB", "Elasticsearch",
    "Docker", "Kubernetes", "K8S", "Jenkins", "Git",
    "微服务", "中间件", "消息队列", "分布式", "大数据",
    "机器学习", "深度学习", "自然语言处理", "计算机视觉",
    "小程序", "公众号",
    "后端", "前端", "全栈", "后端开发", "前端开发", "全栈开发",
    "算法工程师", "数据分析师", "产品经理", "项目经理",
    "运维工程师", "测试工程师", "架构师",
    "UI设计师", "交互设计师",
    "五险一金", "六险一金", "带薪年假", "年终奖",
    "股票期权", "定期体检", "节日福利", "餐饮补贴",
    "交通补贴", "住房补贴", "弹性工作", "远程办公",
    "本科", "硕士", "博士", "大专", "研究生",
    "双休", "单休", "大小周",
    "岗位职责", "岗位要求", "任职要求", "工作职责",
    "五险", "一金", "公积金", "社保",
]

for _w in _CUSTOM_WORDS:
    jieba.add_word(_w)

jieba.lcut("预热")

STOPWORDS = frozenset({
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人",
    "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去",
    "你", "会", "着", "没有", "看", "好", "自己", "这", "他", "她",
    "它", "们", "那", "些", "么", "什么", "及", "与", "或", "等",
    "能", "可以", "需要", "进行", "通过", "使用", "以及", "对于",
    "其他", "相关", "具有", "以上", "包括", "负责", "参与",
    "了解", "熟悉", "掌握", "熟练", "优先", "优秀", "良好", "较强",
    "，", "。", "！", "？", "、", "；", "：", "（", "）",
    "(", ")", ",", ".", "!", "?", ";", ":", "/", "\\",
    "-", "—", "·", "~", "…", "\"", "'", " ",
    "具备", "能够", "能力", "工作", "项目", "经验",
    "年", "月", "日", "个", "位", "名",
})

def segment_text(text: str) -> str:
    if not text:
        return ""
    words = jieba.lcut(text)
    tokens = []
    for w in words:
        w = w.strip()
        if not w or w in STOPWORDS:
            continue
        if len(w) == 1 and not w.isalnum():
            continue
        tokens.append(w.lower())
    return " ".join(tokens)

def segment_welfare(welfare: str) -> str:
    if not welfare:
        return ""
    parts = re.split(r'[,，、；;\s]+', welfare)
    all_tokens = []
    for part in parts:
        part = part.strip()
        if part:
            seg = segment_text(part)
            if seg:
                all_tokens.append(seg)
    return " ".join(all_tokens)

def _build_tsv_sql_and_params(title, company, welfare, summary, detail):
    return (
        segment_text(title or ""),
        segment_text(company or ""),
        segment_welfare(welfare or ""),
        segment_text(summary or ""),
        segment_text(detail or ""),
    )

def _should_apply_experience_filter(experience: str) -> bool:
    exp = (experience or "").strip()
    if not exp:
        return False
    normalized = exp.lower()
    # Broad categories should be handled by semantic ranking, not rigid SQL filters.
    broad_tokens = ["intern", "new grad", "实习", "应届", "非实习", "全职", "正式", "社招", "校招"]
    if any(token in normalized or token in exp for token in broad_tokens):
        return False
    return True

def _make_json_safe(value):
    if isinstance(value, dict):
        return {str(k): _make_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_make_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_make_json_safe(item) for item in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value

def _parse_salary_info(salary_text: str) -> dict:
    text = (salary_text or "").strip().lower()
    if not text:
        return {"unit": "", "min": 0.0, "max": 0.0}

    nums = [float(item) for item in re.findall(r"\d+(?:\.\d+)?", text)]
    if not nums:
        return {"unit": "", "min": 0.0, "max": 0.0}

    low = nums[0]
    high = nums[1] if len(nums) > 1 else nums[0]

    if any(token in text for token in ["元/天", "元每天", "/day", "per day", "day"]):
        return {"unit": "yuan_day", "min": low, "max": high}
    if any(token in text for token in ["k", "千/月", "月", "薪"]):
        return {"unit": "k_month", "min": low, "max": high}
    return {"unit": "", "min": low, "max": high}

def _salary_matches(salary_text: str, salary_min: int = 0, salary_unit: str = "") -> bool:
    if not salary_min or not salary_unit:
        return True
    parsed = _parse_salary_info(salary_text)
    if parsed["unit"] != salary_unit:
        return False
    return parsed["max"] >= float(salary_min)



__all__ = [

    "STOPWORDS",

    "segment_text",

    "segment_welfare",

    "_build_tsv_sql_and_params",

    "_should_apply_experience_filter",

    "_make_json_safe",

    "_parse_salary_info",

    "_salary_matches",

]
