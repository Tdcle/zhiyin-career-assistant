import os
import re
import json
import ast
from datetime import datetime
from typing import Any

import cv2
import fitz  # PyMuPDF
import numpy as np
from langchain_core.prompts import ChatPromptTemplate

from config.config import config
from utils.logger import get_logger

try:
    from paddleocr import PaddleOCR

    HAS_OCR = True
except ImportError:
    HAS_OCR = False

logger = get_logger("file_parser")

RESUME_PARSER_VERSION = "resume_parser_v3_llm"

DATE_RANGE_RE = re.compile(
    r"((?:19|20)\d{2}(?:[./-]\d{1,2})?)\s*(?:至|到|-|—|~)\s*((?:19|20)\d{2}(?:[./-]\d{1,2})?|至今|现在|present)",
    re.IGNORECASE,
)
PHONE_RE = re.compile(r"(?<!\d)(1[3-9]\d{9})(?!\d)")
EMAIL_RE = re.compile(r"([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})")
WECHAT_RE = re.compile(r"(?:微信|wechat)[:：]?\s*([A-Za-z][A-Za-z0-9_-]{5,30})", re.IGNORECASE)
SALARY_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(k|K|元/天|万/年|万|元/月)")

SECTION_ALIASES = {
    "basic_info": ["基本信息", "个人信息", "联系方式", "个人资料"],
    "education": ["教育经历", "教育背景", "教育"],
    "work_experience": ["工作经历", "工作经验", "实习经历", "实践经历"],
    "projects": ["项目经历", "项目经验", "项目背景"],
    "skills": ["专业技能", "技能", "技能清单", "技术栈", "核心技能"],
    "certificates": ["证书", "获奖经历", "荣誉奖项", "荣誉"],
    "job_intention": ["求职意向", "意向岗位", "期望职位", "期望薪资", "期望城市"],
    "self_evaluation": ["自我评价", "个人总结", "个人评价"],
}

CITY_KEYWORDS = [
    "北京",
    "上海",
    "深圳",
    "广州",
    "杭州",
    "成都",
    "武汉",
    "西安",
    "南京",
    "苏州",
    "天津",
    "重庆",
]

DEGREE_KEYWORDS = ["博士", "硕士", "本科", "大专", "高中", "中专"]

SKILL_LEXICON = [
    "python",
    "java",
    "golang",
    "go",
    "c++",
    "c#",
    "php",
    "javascript",
    "typescript",
    "react",
    "vue",
    "node",
    "spring",
    "spring boot",
    "django",
    "flask",
    "fastapi",
    "mysql",
    "postgresql",
    "redis",
    "mongodb",
    "elasticsearch",
    "docker",
    "kubernetes",
    "git",
    "linux",
    "pytorch",
    "tensorflow",
    "llm",
    "rag",
    "agent",
    "nlp",
    "机器学习",
    "深度学习",
    "大模型",
]

LANGUAGE_KEYWORDS = ["英语", "日语", "韩语", "法语", "德语", "雅思", "托福", "CET-4", "CET-6"]

LLM_SECTION_KEYS = [
    "basic_info",
    "education",
    "work_experience",
    "projects",
    "skills",
    "certificates",
    "languages",
    "job_intention",
    "self_evaluation",
    "other",
]

LLM_ENABLED = os.getenv("RESUME_ENABLE_LLM_STRUCTURER", "true").strip().lower() != "false"
LLM_MAX_SOURCE_CHARS = int(os.getenv("RESUME_LLM_MAX_SOURCE_CHARS", "12000"))
LLM_MAX_PAGE_CHARS = int(os.getenv("RESUME_LLM_MAX_PAGE_CHARS", "2200"))


class FileParser:
    _ocr_instance = None
    _resume_structurer_llm = None
    _resume_structurer_disabled_logged = False

    @classmethod
    def get_ocr_engine(cls):
        if not HAS_OCR:
            logger.warning("PaddleOCR not found, image/scanned PDF parsing unavailable.")
            return None

        if cls._ocr_instance is None:
            logger.info("initializing OCR engine...")
            cls._ocr_instance = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
        return cls._ocr_instance

    @staticmethod
    def _pixmap_to_tensor(pix):
        img_array = np.frombuffer(pix.samples, dtype=np.uint8)
        img_array = img_array.reshape(pix.h, pix.w, pix.n)

        if pix.n == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        return img_array

    @staticmethod
    def _ocr_image_array(img_array):
        ocr = FileParser.get_ocr_engine()
        if not ocr:
            return ""

        try:
            result = ocr.ocr(img_array, cls=True)
            text_result = []
            if result and result[0]:
                for line in result[0]:
                    text_content = line[1][0]
                    if text_content:
                        text_result.append(text_content)
            return "\n".join(text_result)
        except Exception as exc:
            logger.error("ocr failed: %s", exc, exc_info=True)
            return ""

    @staticmethod
    def parse_pdf(file_path):
        text_content = ""
        try:
            doc = fitz.open(file_path)
            for page_num, page in enumerate(doc):
                page_text = page.get_text() or ""

                if len(page_text.strip()) < 120 and HAS_OCR:
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    img_array = FileParser._pixmap_to_tensor(pix)
                    ocr_text = FileParser._ocr_image_array(img_array)
                    page_text = f"[OCR]\n{ocr_text}".strip()

                text_content += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
            doc.close()
            return text_content.strip()
        except Exception as exc:
            logger.error("pdf parse failed: %s", exc, exc_info=True)
            return ""

    @staticmethod
    def parse_image(file_path):
        if not HAS_OCR:
            return ""

        try:
            ocr = FileParser.get_ocr_engine()
            result = ocr.ocr(file_path, cls=True)
            text_result = []
            if result and result[0]:
                for line in result[0]:
                    text = line[1][0]
                    if text:
                        text_result.append(text)
            return "\n".join(text_result).strip()
        except Exception as exc:
            logger.error("image parse failed: %s", exc, exc_info=True)
            return ""

    @staticmethod
    def read_file(file_path):
        if not os.path.exists(file_path):
            return ""

        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            return FileParser.parse_pdf(file_path)
        if ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            return FileParser.parse_image(file_path)
        if ext in [".txt", ".md"]:
            with open(file_path, "r", encoding="utf-8") as fp:
                return fp.read()
        return ""

    @staticmethod
    def normalize_resume_text(raw_text: str) -> str:
        text = raw_text or ""
        text = text.replace("[OCR]", "")
        text = re.sub(r"\n---\s*Page\s+\d+\s*---\n", "\n", text)
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def _safe_json_loads(raw: str) -> dict:
        text = (raw or "").strip()
        if not text:
            raise ValueError("empty json text")

        def _extract_candidate(value: str) -> str:
            candidate = (value or "").strip()
            if "```" in candidate:
                match = re.search(r"```(?:json)?(.*?)```", candidate, re.DOTALL | re.IGNORECASE)
                if match:
                    candidate = match.group(1).strip()
            candidate = candidate.replace("\ufeff", "").strip()
            start_obj = candidate.find("{")
            end_obj = candidate.rfind("}")
            start_arr = candidate.find("[")
            end_arr = candidate.rfind("]")
            if start_obj != -1 and end_obj != -1 and end_obj > start_obj:
                return candidate[start_obj:end_obj + 1].strip()
            if start_arr != -1 and end_arr != -1 and end_arr > start_arr:
                return candidate[start_arr:end_arr + 1].strip()
            return candidate

        def _repair_json_text(value: str) -> str:
            repaired = (value or "").strip()
            repaired = repaired.replace("“", "\"").replace("”", "\"").replace("’", "'").replace("‘", "'")
            repaired = re.sub(r"//.*?$", "", repaired, flags=re.MULTILINE)
            repaired = re.sub(r"/\*.*?\*/", "", repaired, flags=re.DOTALL)
            repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
            repaired = re.sub(r"\bNone\b", "null", repaired)
            repaired = re.sub(r"\bTrue\b", "true", repaired)
            repaired = re.sub(r"\bFalse\b", "false", repaired)
            repaired = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", repaired)
            return repaired.strip()

        candidate = _extract_candidate(text)
        errors: list[Exception] = []

        for attempt in (candidate, _repair_json_text(candidate)):
            if not attempt:
                continue
            try:
                data = json.loads(attempt)
                if isinstance(data, dict):
                    return data
                if isinstance(data, list):
                    return {"items": data}
                raise ValueError("json is neither object nor array")
            except Exception as exc:
                errors.append(exc)

        try:
            py_obj = ast.literal_eval(candidate)
            if isinstance(py_obj, dict):
                return py_obj
            if isinstance(py_obj, list):
                return {"items": py_obj}
        except Exception as exc:
            errors.append(exc)

        repaired_candidate = _repair_json_text(candidate)
        try:
            py_obj = ast.literal_eval(repaired_candidate)
            if isinstance(py_obj, dict):
                return py_obj
            if isinstance(py_obj, list):
                return {"items": py_obj}
        except Exception as exc:
            errors.append(exc)

        message = "; ".join(str(err) for err in errors[-2:]) if errors else "json parse failed"
        raise ValueError(message)

    @staticmethod
    def _extract_pages_from_raw(raw_text: str) -> list[dict]:
        text = raw_text or ""
        matches = list(re.finditer(r"\n---\s*Page\s+(\d+)\s*---\n", text))
        if not matches:
            cleaned = text.strip()
            return [{"page": 1, "text": cleaned}] if cleaned else []

        pages: list[dict] = []
        for idx, match in enumerate(matches):
            page_number = int(match.group(1))
            content_start = match.end()
            content_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            page_text = text[content_start:content_end].strip()
            if page_text:
                pages.append({"page": page_number, "text": page_text})
        return pages

    @classmethod
    def _build_paged_digest(cls, pages: list[dict]) -> str:
        if not pages:
            return ""
        lines: list[str] = []
        for page in pages:
            number = int(page.get("page", 0) or 0)
            content = str(page.get("text", "") or "").strip()
            if not content:
                continue
            if len(content) > LLM_MAX_PAGE_CHARS:
                content = f"{content[:LLM_MAX_PAGE_CHARS]}\n...(truncated)"
            lines.append(f"[Page {number}]\n{content}")
        return "\n\n".join(lines).strip()

    @classmethod
    def _get_resume_structurer_llm(cls):
        if not LLM_ENABLED:
            return None
        if cls._resume_structurer_llm is not None:
            return cls._resume_structurer_llm
        try:
            cls._resume_structurer_llm = config.create_tongyi(
                config.CHAT_MODELS.resume_structurer,
                streaming=False,
                temperature=0.1,
            )
            return cls._resume_structurer_llm
        except Exception as exc:
            if not cls._resume_structurer_disabled_logged:
                logger.warning("resume structurer llm unavailable, fallback to heuristic parser: %s", exc)
                cls._resume_structurer_disabled_logged = True
            return None

    @staticmethod
    def _normalize_text_item(value: Any, max_len: int = 4000) -> str:
        text = str(value or "").strip()
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        if len(text) > max_len:
            return text[:max_len]
        return text

    @classmethod
    def _dedupe_str_list(cls, values: list[Any], max_items: int = 80) -> list[str]:
        seen = set()
        result: list[str] = []
        for raw in values or []:
            item = cls._normalize_text_item(raw, max_len=120)
            if not item:
                continue
            key = item.lower()
            if key in seen:
                continue
            seen.add(key)
            result.append(item)
            if len(result) >= max_items:
                break
        return result

    @classmethod
    def _normalize_structured_payload(cls, payload: dict | None) -> dict:
        payload = payload if isinstance(payload, dict) else {}
        basic = payload.get("basic_info", {}) if isinstance(payload.get("basic_info"), dict) else {}
        intention = payload.get("job_intention", {}) if isinstance(payload.get("job_intention"), dict) else {}

        def normalize_items(items: Any, keys: list[str], max_items: int, text_limit: int = 1200) -> list[dict]:
            normalized: list[dict] = []
            for raw in items or []:
                if not isinstance(raw, dict):
                    continue
                row = {key: cls._normalize_text_item(raw.get(key, ""), max_len=text_limit) for key in keys}
                if any(row.values()):
                    normalized.append(row)
                if len(normalized) >= max_items:
                    break
            return normalized

        projects = normalize_items(
            payload.get("projects"),
            ["name", "role", "start_date", "end_date", "description"],
            max_items=25,
            text_limit=1600,
        )
        for idx, row in enumerate(projects):
            raw = payload.get("projects", [])[idx] if isinstance(payload.get("projects"), list) and idx < len(payload.get("projects")) else {}
            tech_stack = []
            if isinstance(raw, dict):
                stack_value = raw.get("tech_stack", [])
                if isinstance(stack_value, str):
                    stack_value = re.split(r"[,\n;，；、]+", stack_value)
                if isinstance(stack_value, list):
                    tech_stack = cls._dedupe_str_list(stack_value, max_items=20)
            row["tech_stack"] = tech_stack

        structured = {
            "basic_info": {
                "name": cls._normalize_text_item(basic.get("name", ""), max_len=80),
                "phone": cls._normalize_text_item(basic.get("phone", ""), max_len=40),
                "email": cls._normalize_text_item(basic.get("email", ""), max_len=120),
                "wechat": cls._normalize_text_item(basic.get("wechat", ""), max_len=80),
                "city": cls._normalize_text_item(basic.get("city", ""), max_len=60),
                "highest_degree": cls._normalize_text_item(basic.get("highest_degree", ""), max_len=40),
            },
            "education": normalize_items(
                payload.get("education"),
                ["school", "degree", "major", "start_date", "end_date", "description"],
                max_items=20,
                text_limit=1200,
            ),
            "work_experience": normalize_items(
                payload.get("work_experience"),
                ["company", "title", "start_date", "end_date", "description"],
                max_items=25,
                text_limit=1800,
            ),
            "projects": projects,
            "skills": cls._dedupe_str_list(payload.get("skills", []), max_items=120),
            "certificates": cls._dedupe_str_list(payload.get("certificates", []), max_items=80),
            "languages": cls._dedupe_str_list(payload.get("languages", []), max_items=40),
            "job_intention": {
                "target_roles": cls._dedupe_str_list(intention.get("target_roles", []), max_items=20),
                "target_cities": cls._dedupe_str_list(intention.get("target_cities", []), max_items=20),
                "salary_expectation": cls._normalize_text_item(intention.get("salary_expectation", ""), max_len=80),
            },
            "profile_summary": cls._normalize_text_item(payload.get("profile_summary", ""), max_len=1500),
        }
        return structured

    @classmethod
    def _merge_structured_payload(cls, preferred: dict, fallback: dict) -> dict:
        if not isinstance(preferred, dict):
            preferred = {}
        if not isinstance(fallback, dict):
            fallback = {}

        merged = cls._normalize_structured_payload(preferred)
        backup = cls._normalize_structured_payload(fallback)

        basic = merged.get("basic_info", {})
        backup_basic = backup.get("basic_info", {})
        for key in ["name", "phone", "email", "wechat", "city", "highest_degree"]:
            if not basic.get(key):
                basic[key] = backup_basic.get(key, "")
        merged["basic_info"] = basic

        for list_key in ["education", "work_experience", "projects", "skills", "certificates", "languages"]:
            if not merged.get(list_key):
                merged[list_key] = backup.get(list_key, [])

        merged_intention = merged.get("job_intention", {})
        backup_intention = backup.get("job_intention", {})
        if not merged_intention.get("target_roles"):
            merged_intention["target_roles"] = backup_intention.get("target_roles", [])
        if not merged_intention.get("target_cities"):
            merged_intention["target_cities"] = backup_intention.get("target_cities", [])
        if not merged_intention.get("salary_expectation"):
            merged_intention["salary_expectation"] = backup_intention.get("salary_expectation", "")
        merged["job_intention"] = merged_intention

        if not merged.get("profile_summary"):
            merged["profile_summary"] = backup.get("profile_summary", "")

        merged["parsed_at"] = datetime.now().isoformat(timespec="seconds")
        return merged

    @classmethod
    def _stage1_classify_blocks_with_llm(cls, normalized_text: str, paged_digest: str) -> list[dict]:
        llm = cls._get_resume_structurer_llm()
        if not llm:
            return []

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are a resume segmentation engine.\n"
                        "Classify resume content into section blocks.\n"
                        "Return strict JSON only in this schema:\n"
                        "{{\"blocks\": [{{\"section\": \"basic_info|education|work_experience|projects|skills|certificates|languages|job_intention|self_evaluation|other\", \"page_from\": 1, \"page_to\": 1, \"text\": \"...\"}}]}}\n"
                        "Rules:\n"
                        "1) Keep `text` close to source wording, do not paraphrase.\n"
                        "2) Merge continuous lines that belong to one section into one block.\n"
                        "3) If uncertain, use section `other`."
                    ),
                ),
                (
                    "human",
                    (
                        "Resume pages:\n{paged_digest}\n\n"
                        "Fallback full text:\n{full_text}"
                    ),
                ),
            ]
        )

        full_text = normalized_text[:LLM_MAX_SOURCE_CHARS]
        paged_input = paged_digest[:LLM_MAX_SOURCE_CHARS] if paged_digest else full_text

        try:
            response = (prompt | llm).invoke({"paged_digest": paged_input, "full_text": full_text})
            data = cls._safe_json_loads(getattr(response, "content", ""))
        except Exception as exc:
            logger.warning("resume stage1 llm parse failed, fallback to heuristic parser: %s", exc)
            return []

        raw_blocks = data.get("blocks") if isinstance(data, dict) else []
        if not isinstance(raw_blocks, list):
            return []

        blocks: list[dict] = []
        for raw in raw_blocks:
            if not isinstance(raw, dict):
                continue
            section = str(raw.get("section", "other") or "other").strip()
            if section not in LLM_SECTION_KEYS:
                section = "other"
            block_text = cls._normalize_text_item(raw.get("text", ""), max_len=3000)
            if not block_text:
                continue
            page_from = int(raw.get("page_from", 0) or 0)
            page_to = int(raw.get("page_to", 0) or 0)
            blocks.append(
                {
                    "section": section,
                    "page_from": page_from if page_from > 0 else 0,
                    "page_to": page_to if page_to > 0 else 0,
                    "text": block_text,
                }
            )
        return blocks

    @classmethod
    def _group_blocks_for_stage2(cls, blocks: list[dict], normalized_text: str) -> dict:
        grouped = {key: [] for key in LLM_SECTION_KEYS}
        for block in blocks or []:
            section = str(block.get("section", "other") or "other")
            if section not in grouped:
                section = "other"
            text = cls._normalize_text_item(block.get("text", ""), max_len=3000)
            if text:
                grouped[section].append(text)

        if not any(grouped.values()):
            grouped["other"] = [normalized_text[:LLM_MAX_SOURCE_CHARS]]
        return grouped

    @classmethod
    def _stage2_extract_fields_with_llm(cls, grouped_blocks: dict, normalized_text: str) -> dict:
        llm = cls._get_resume_structurer_llm()
        if not llm:
            return {}

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are a resume information extraction engine.\n"
                        "Extract structured data from grouped blocks.\n"
                        "Return strict JSON only with this exact schema keys:\n"
                        "{{"
                        "\"basic_info\": {{\"name\": \"\", \"phone\": \"\", \"email\": \"\", \"wechat\": \"\", \"city\": \"\", \"highest_degree\": \"\"}},"
                        "\"education\": [{{\"school\": \"\", \"degree\": \"\", \"major\": \"\", \"start_date\": \"\", \"end_date\": \"\", \"description\": \"\"}}],"
                        "\"work_experience\": [{{\"company\": \"\", \"title\": \"\", \"start_date\": \"\", \"end_date\": \"\", \"description\": \"\"}}],"
                        "\"projects\": [{{\"name\": \"\", \"role\": \"\", \"start_date\": \"\", \"end_date\": \"\", \"tech_stack\": [\"\"], \"description\": \"\"}}],"
                        "\"skills\": [\"\"],"
                        "\"certificates\": [\"\"],"
                        "\"languages\": [\"\"],"
                        "\"job_intention\": {{\"target_roles\": [\"\"], \"target_cities\": [\"\"], \"salary_expectation\": \"\"}},"
                        "\"profile_summary\": \"\""
                        "}}\n"
                        "Rules:\n"
                        "1) Preserve multiple items for education/work/projects.\n"
                        "2) Keep unknown fields as empty string/list.\n"
                        "3) Do not output confidence, evidence, comments, or markdown."
                    ),
                ),
                (
                    "human",
                    (
                        "Grouped blocks JSON:\n{grouped_blocks}\n\n"
                        "Full text fallback:\n{full_text}"
                    ),
                ),
            ]
        )

        grouped_text = json.dumps(grouped_blocks, ensure_ascii=False)
        full_text = normalized_text[:LLM_MAX_SOURCE_CHARS]
        try:
            response = (prompt | llm).invoke({"grouped_blocks": grouped_text, "full_text": full_text})
            data = cls._safe_json_loads(getattr(response, "content", ""))
            if not isinstance(data, dict):
                return {}
            return data
        except Exception as exc:
            logger.warning("resume stage2 llm parse failed, fallback to heuristic parser: %s", exc)
            return {}

    @staticmethod
    def _normalize_heading(line: str) -> str:
        cleaned = re.sub(r"[：:\s]+$", "", line.strip())
        cleaned = re.sub(r"^[#*\-\d.\s]+", "", cleaned)
        return cleaned

    @classmethod
    def _detect_section_key(cls, line: str) -> str | None:
        heading = cls._normalize_heading(line)
        if not heading:
            return None
        if DATE_RANGE_RE.search(heading):
            return None
        if re.search(r"\d", heading):
            return None
        if re.search(r"[，,。；;、]", heading):
            return None
        if len(heading) > 24:
            return None
        for key, aliases in SECTION_ALIASES.items():
            for alias in aliases:
                if heading == alias or heading.startswith(alias):
                    return key
        return None

    @classmethod
    def _split_sections(cls, text: str) -> dict[str, str]:
        sections: dict[str, list[str]] = {"raw": []}
        current = "raw"
        for raw_line in (text or "").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            key = cls._detect_section_key(line)
            if key:
                current = key
                sections.setdefault(current, [])
                continue
            sections.setdefault(current, []).append(line)
        return {k: "\n".join(v).strip() for k, v in sections.items()}

    @staticmethod
    def _dedupe_keep_order(items: list[str]) -> list[str]:
        seen = set()
        result = []
        for item in items:
            token = (item or "").strip()
            if not token:
                continue
            key = token.lower()
            if key in seen:
                continue
            seen.add(key)
            result.append(token)
        return result

    @classmethod
    def _extract_basic_info(cls, full_text: str, sections: dict[str, str]) -> dict:
        text = "\n".join(
            [
                sections.get("basic_info", ""),
                sections.get("job_intention", ""),
                full_text,
            ]
        )

        phone = PHONE_RE.search(text)
        email = EMAIL_RE.search(text)
        wechat = WECHAT_RE.search(text)

        name = ""
        m_name = re.search(r"(?:姓名|Name)[:：]?\s*([^\s，,|/]{2,12})", text, re.IGNORECASE)
        if m_name:
            name = m_name.group(1).strip()
        else:
            for line in full_text.splitlines()[:8]:
                line = line.strip()
                m_inline_name = re.match(r"^([\u4e00-\u9fa5]{2,4})(?:\s|$)", line)
                if m_inline_name:
                    candidate = m_inline_name.group(1).strip()
                    if (
                        candidate
                        and candidate not in {"个人信息", "基本信息", "联系方式"}
                        and not re.search(r"大学|学院|学校|硕士|本科|博士|教育|经历", candidate)
                    ):
                        name = candidate
                        break
                if (
                    re.fullmatch(r"[\u4e00-\u9fa5]{2,4}", line)
                    and not re.search(r"\d|@|邮箱|电话|手机|微信", line)
                    and not re.search(r"大学|学院|学校|硕士|本科|博士|简历|教育|经历", line)
                ):
                    name = line
                    break

        city = ""
        m_city = re.search(r"(?:现居|居住地|所在城市|城市)[:：]?\s*([^\s，,;；|/]{2,12})", text)
        if m_city:
            city = m_city.group(1).strip()
        else:
            for city_item in CITY_KEYWORDS:
                if city_item in text:
                    city = city_item
                    break

        highest_degree = ""
        for degree in DEGREE_KEYWORDS:
            if degree in text:
                highest_degree = degree
                break

        return {
            "name": name,
            "phone": phone.group(1) if phone else "",
            "email": email.group(1) if email else "",
            "wechat": wechat.group(1) if wechat else "",
            "city": city,
            "highest_degree": highest_degree,
        }

    @staticmethod
    def _timeline_blocks(section_text: str) -> list[str]:
        lines = [line.strip() for line in (section_text or "").splitlines() if line.strip()]
        if not lines:
            return []

        blocks: list[list[str]] = []
        current: list[str] = []
        for line in lines:
            if DATE_RANGE_RE.search(line) and current:
                blocks.append(current)
                current = [line]
            else:
                current.append(line)
        if current:
            blocks.append(current)

        return ["\n".join(block).strip() for block in blocks if block]

    @staticmethod
    def _find_date_range(text: str) -> tuple[str, str]:
        m = DATE_RANGE_RE.search(text or "")
        if not m:
            return "", ""
        return m.group(1).strip(), m.group(2).strip()

    @staticmethod
    def _extract_degree(text: str) -> str:
        for degree in DEGREE_KEYWORDS:
            if degree in text:
                return degree
        return ""

    @classmethod
    def _parse_education_items(cls, section_text: str) -> list[dict]:
        items = []
        for block in cls._timeline_blocks(section_text):
            start_date, end_date = cls._find_date_range(block)
            school_match = re.search(
                r"([A-Za-z\u4e00-\u9fa5·（）()\-]{2,80}(?:大学|学院|学校|University|College))",
                block,
            )
            major_match = re.search(r"(?:专业|Major)[:：]?\s*([^\n，,;；|/]{2,40})", block, re.IGNORECASE)
            degree = cls._extract_degree(block)
            item = {
                "school": school_match.group(1).strip() if school_match else "",
                "degree": degree,
                "major": major_match.group(1).strip() if major_match else "",
                "start_date": start_date,
                "end_date": end_date,
                "description": block[:400],
            }
            if item["school"] or item["start_date"] or item["description"]:
                items.append(item)
        return items

    @classmethod
    def _parse_work_items(cls, section_text: str) -> list[dict]:
        items = []
        for block in cls._timeline_blocks(section_text):
            start_date, end_date = cls._find_date_range(block)
            company_match = re.search(
                r"([A-Za-z\u4e00-\u9fa5·（）()\-]{2,80}(?:公司|集团|科技|信息|网络|银行|研究院|有限|Inc|Ltd))",
                block,
                re.IGNORECASE,
            )
            title_match = re.search(r"(?:岗位|职位|担任)[:：]?\s*([^\n，,;；|/]{2,40})", block)

            first_line = block.splitlines()[0] if block.splitlines() else ""
            if not title_match:
                line_without_dates = DATE_RANGE_RE.sub("", first_line).strip(" -|/")
                if 1 < len(line_without_dates) <= 40:
                    title_guess = line_without_dates
                else:
                    title_guess = ""
            else:
                title_guess = title_match.group(1).strip()

            item = {
                "company": company_match.group(1).strip() if company_match else "",
                "title": title_guess,
                "start_date": start_date,
                "end_date": end_date,
                "description": block[:800],
            }
            if item["company"] or item["title"] or item["description"]:
                items.append(item)
        return items

    @classmethod
    def _parse_project_items(cls, section_text: str) -> list[dict]:
        items = []
        for block in cls._timeline_blocks(section_text):
            start_date, end_date = cls._find_date_range(block)
            name_match = re.search(r"(?:项目名称|项目)[:：]?\s*([^\n]{2,80})", block)
            role_match = re.search(r"(?:角色|职责|担任)[:：]?\s*([^\n]{2,60})", block)

            first_line = block.splitlines()[0] if block.splitlines() else ""
            if name_match:
                project_name = name_match.group(1).strip()
            else:
                project_name = DATE_RANGE_RE.sub("", first_line).strip(" -|/")[:80]

            tech_stack = []
            lowered = block.lower()
            for token in SKILL_LEXICON:
                if token.lower() in lowered:
                    tech_stack.append(token)

            item = {
                "name": project_name,
                "role": role_match.group(1).strip() if role_match else "",
                "start_date": start_date,
                "end_date": end_date,
                "tech_stack": cls._dedupe_keep_order(tech_stack),
                "description": block[:1000],
            }
            if item["name"] or item["description"]:
                items.append(item)
        return items

    @classmethod
    def _extract_skills(cls, full_text: str, section_text: str) -> list[str]:
        skills: list[str] = []

        if section_text:
            tokens = re.split(r"[\n,，、/|；; ]+", section_text)
            for token in tokens:
                token = token.strip()
                if not token:
                    continue
                if re.fullmatch(r"[\u4e00-\u9fa5A-Za-z0-9.+#-]{2,30}", token):
                    skills.append(token)

        lowered = full_text.lower()
        for token in SKILL_LEXICON:
            if token.lower() in lowered:
                skills.append(token)

        return cls._dedupe_keep_order(skills)[:80]

    @classmethod
    def _extract_certificates(cls, full_text: str, section_text: str) -> list[str]:
        certs: list[str] = []
        for line in (section_text or "").splitlines():
            text = line.strip()
            if not text:
                continue
            if len(text) <= 60:
                certs.append(text)

        common_patterns = [
            r"(PMP|软考|CFA|CPA|教师资格证|证券从业|计算机二级|英语六级|英语四级|CET-6|CET-4)",
        ]
        for pattern in common_patterns:
            for m in re.finditer(pattern, full_text, flags=re.IGNORECASE):
                certs.append(m.group(1))
        return cls._dedupe_keep_order(certs)

    @classmethod
    def _extract_languages(cls, full_text: str) -> list[str]:
        found = []
        lowered = full_text.lower()
        for key in LANGUAGE_KEYWORDS:
            if key.lower() in lowered:
                found.append(key)
        return cls._dedupe_keep_order(found)

    @classmethod
    def _extract_job_intention(cls, full_text: str, section_text: str) -> dict:
        merged = "\n".join([section_text or "", full_text or ""])
        target_roles = _extract_role_like_keywords(merged)
        target_cities = [city for city in CITY_KEYWORDS if city in merged]

        salary = ""
        m_salary = SALARY_RE.search(merged)
        if m_salary:
            salary = f"{m_salary.group(1)}{m_salary.group(2)}"

        return {
            "target_roles": cls._dedupe_keep_order(target_roles),
            "target_cities": cls._dedupe_keep_order(target_cities),
            "salary_expectation": salary,
        }

    @classmethod
    def _build_profile_summary(cls, sections: dict[str, str]) -> str:
        parts = []
        for key in ["self_evaluation", "work_experience", "projects"]:
            text = (sections.get(key) or "").strip()
            if not text:
                continue
            parts.append(text[:220])
            if len(parts) >= 2:
                break
        return "\n".join(parts).strip()

    @classmethod
    def extract_structured_resume(cls, normalized_text: str) -> dict:
        sections = cls._split_sections(normalized_text)
        basic_info = cls._extract_basic_info(normalized_text, sections)
        education_items = cls._parse_education_items(sections.get("education", ""))
        work_items = cls._parse_work_items(sections.get("work_experience", ""))
        project_items = cls._parse_project_items(sections.get("projects", ""))
        skills = cls._extract_skills(normalized_text, sections.get("skills", ""))
        certificates = cls._extract_certificates(normalized_text, sections.get("certificates", ""))
        languages = cls._extract_languages(normalized_text)
        job_intention = cls._extract_job_intention(normalized_text, sections.get("job_intention", ""))
        profile_summary = cls._build_profile_summary(sections)

        return {
            "basic_info": basic_info,
            "education": education_items,
            "work_experience": work_items,
            "projects": project_items,
            "skills": skills,
            "certificates": certificates,
            "languages": languages,
            "job_intention": job_intention,
            "profile_summary": profile_summary,
            "parsed_at": datetime.now().isoformat(timespec="seconds"),
        }

    @classmethod
    def parse_resume(cls, file_path: str, original_filename: str = "") -> dict:
        raw_text = cls.read_file(file_path)
        normalized_text = cls.normalize_resume_text(raw_text)
        heuristic_structured = cls.extract_structured_resume(normalized_text) if normalized_text else {}
        paged_digest = cls._build_paged_digest(cls._extract_pages_from_raw(raw_text))

        llm_structured: dict = {}
        if normalized_text and LLM_ENABLED:
            stage1_blocks = cls._stage1_classify_blocks_with_llm(normalized_text, paged_digest)
            grouped_blocks = cls._group_blocks_for_stage2(stage1_blocks, normalized_text)
            llm_structured = cls._stage2_extract_fields_with_llm(grouped_blocks, normalized_text)

        structured = cls._merge_structured_payload(llm_structured, heuristic_structured)
        basic = structured.get("basic_info", {}) if isinstance(structured, dict) else {}
        if isinstance(basic, dict) and not str(basic.get("name", "")).strip():
            inferred_name = _extract_name_from_filename(original_filename or file_path)
            if inferred_name:
                basic["name"] = inferred_name
                structured["basic_info"] = basic

        parser_version = RESUME_PARSER_VERSION
        if llm_structured:
            parser_version = f"{RESUME_PARSER_VERSION}+llm"
        else:
            parser_version = f"{RESUME_PARSER_VERSION}+heuristic_fallback"
        return {
            "raw_text": raw_text,
            "normalized_text": normalized_text,
            "structured": structured,
            "parser_version": parser_version,
        }


def _extract_role_like_keywords(text: str) -> list[str]:
    lowered = (text or "").lower()
    mapping = {
        "后端": ["后端", "backend", "java", "python", "golang", "go"],
        "前端": ["前端", "frontend", "react", "vue", "javascript"],
        "全栈": ["全栈", "fullstack", "full-stack"],
        "算法": ["算法", "machine learning", "机器学习", "深度学习"],
        "AI": ["ai", "大模型", "nlp", "llm", "agent"],
        "测试": ["测试", "qa"],
        "数据": ["数据分析", "数据开发", "大数据", "etl"],
        "产品": ["产品经理", "product manager", "pm"],
    }
    result = []
    for role, aliases in mapping.items():
        if any(alias.lower() in lowered for alias in aliases):
            result.append(role)
    return result


def _extract_name_from_filename(filename: str) -> str:
    base = os.path.splitext(os.path.basename(filename or ""))[0]
    if not base:
        return ""
    parts = re.split(r"[_\-\s]+", base)
    for token in parts:
        token = token.strip()
        if re.fullmatch(r"[\u4e00-\u9fa5]{2,4}", token):
            return token
    if re.fullmatch(r"[\u4e00-\u9fa5]{2,4}", base):
        return base
    return ""
