"""Standalone job-link availability probe.

Usage examples:
    # validate explicit urls
    python utils/job_alive_probe.py --url "https://www.zhipin.com/job_detail/xxx.html"

    # validate urls from file (one url per line)
    python utils/job_alive_probe.py --file urls.txt

    # sample latest jobs from DB and validate their detail_url
    python utils/job_alive_probe.py --from-db-limit 20
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from DrissionPage import ChromiumPage

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from utils.logger import get_logger


logger = get_logger("job_alive_probe")

ACTIVE_KEYWORDS = [
    "招聘中",
    "正在招聘",
    "立即沟通",
    "感兴趣",
]

# Context-only active signals. These need job-detail context to avoid homepage false positives.
ACTIVE_CONTEXT_KEYWORDS = [
    "最新",
    "刚刚发布",
    "最近更新",
]

CLOSED_KEYWORDS = [
    "职位已关闭",
    "查看更多优选岗位",
    "该职位已关闭",
    "职位关闭",
    "停止招聘",
    "已下线",
    "职位不存在",
    "页面不存在",
    "你访问的页面不存在",
    "链接已失效",
]

RISK_KEYWORDS = [
    "登录后查看",
    "请先登录",
    "验证",
    "验证码",
    "访问异常",
    "风险",
    "行为异常",
    "请求过于频繁",
]

LOADING_KEYWORDS = [
    "请稍候",
    "正在加载",
    "加载中",
    "BOSS 正在加载中",
]

# Fallback strategy: if link is not explicitly judged as closed, treat timeout/unknown as active.
ASSUME_ACTIVE_WHEN_NOT_CLOSED = True


def _contains_any(text: str, words: list[str]) -> tuple[bool, str]:
    for word in words:
        if word in text:
            return True, word
    return False, ""


def _normalize_text(text: str) -> str:
    normalized = (text or "").replace("\u00a0", " ")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def _extract_text(tab: Any, timeout_seconds: int) -> str:
    try:
        body_ele = tab.ele("tag:body", timeout=timeout_seconds)
        if body_ele:
            return _normalize_text(body_ele.text or "")
    except Exception:
        pass
    return ""


def _safe_attr(obj: Any, attr: str, default: str = "") -> str:
    try:
        value = getattr(obj, attr, default)
        if isinstance(value, str):
            return value
        return str(value or default)
    except Exception:
        return default


def _url_path(url: str) -> str:
    try:
        return (urlparse(url).path or "").strip().lower()
    except Exception:
        return ""


def _is_zhipin_url(url: str) -> bool:
    try:
        host = (urlparse(url).netloc or "").lower()
    except Exception:
        return False
    return "zhipin.com" in host


def _looks_like_job_detail_context(final_url: str, title: str, text: str) -> bool:
    path = _url_path(final_url)
    if "/job_detail/" in path or path.startswith("/job_detail"):
        return True
    if "岗位职责" in title:
        return True
    if "岗位职责" in text or "职位详情" in text:
        return True
    return False


def _is_home_or_list_redirect(final_url: str) -> bool:
    if not _is_zhipin_url(final_url):
        return False
    path = _url_path(final_url)
    if path in {"", "/", "/index.html", "/web/geek", "/web/geek/job", "/web/geek/jobs"}:
        return True
    if path.startswith("/web/geek/job") or path.startswith("/web/geek/jobs"):
        return True
    if path.startswith("/web/common/error"):
        return True
    return False


def classify_current_tab(tab: Any, input_url: str, timeout_seconds: int = 10) -> dict[str, Any]:
    text = _extract_text(tab, timeout_seconds=timeout_seconds)
    title = _safe_attr(tab, "title")
    final_url = _safe_attr(tab, "url", input_url)

    closed_hit_text, closed_keyword_text = _contains_any(text, CLOSED_KEYWORDS)
    closed_hit_title, closed_keyword_title = _contains_any(title, CLOSED_KEYWORDS)
    active_hit_text, active_keyword_text = _contains_any(text, ACTIVE_KEYWORDS)
    active_hit_title, active_keyword_title = _contains_any(title, ACTIVE_KEYWORDS)
    context_active_hit_text, context_active_keyword_text = _contains_any(text, ACTIVE_CONTEXT_KEYWORDS)
    context_active_hit_title, context_active_keyword_title = _contains_any(title, ACTIVE_CONTEXT_KEYWORDS)
    risk_hit_text, risk_keyword_text = _contains_any(text, RISK_KEYWORDS)
    risk_hit_title, risk_keyword_title = _contains_any(title, RISK_KEYWORDS)
    loading_hit_text, loading_keyword_text = _contains_any(text, LOADING_KEYWORDS)
    loading_hit_title, loading_keyword_title = _contains_any(title, LOADING_KEYWORDS)

    closed_hit = closed_hit_text or closed_hit_title
    closed_keyword = closed_keyword_text or closed_keyword_title
    active_hit = active_hit_text or active_hit_title
    active_keyword = active_keyword_text or active_keyword_title
    context_active_hit = context_active_hit_text or context_active_hit_title
    context_active_keyword = context_active_keyword_text or context_active_keyword_title
    risk_hit = risk_hit_text or risk_hit_title
    risk_keyword = risk_keyword_text or risk_keyword_title

    status = "unknown"
    reason = "no_signal"
    matched_keyword = ""

    if closed_hit:
        status = "closed"
        reason = "closed_keyword_hit"
        matched_keyword = closed_keyword
    elif "security-check" in final_url:
        status = "unknown"
        reason = "security_check_redirect"
    elif _is_zhipin_url(final_url) and (not _looks_like_job_detail_context(final_url, title, text)):
        if _is_home_or_list_redirect(final_url):
            status = "closed"
            reason = "redirect_home_or_list"
            matched_keyword = _url_path(final_url) or "/"
        else:
            status = "unknown"
            reason = "redirected_out_of_job_detail"
    elif active_hit:
        status = "active"
        reason = "active_keyword_hit"
        matched_keyword = active_keyword
    elif context_active_hit and _looks_like_job_detail_context(final_url, title, text):
        status = "active"
        reason = "active_context_keyword_hit"
        matched_keyword = context_active_keyword
    elif risk_hit:
        status = "unknown"
        reason = "risk_or_login_page"
        matched_keyword = risk_keyword
    elif loading_hit_text or loading_hit_title:
        status = "unknown"
        reason = "loading_page"
        matched_keyword = loading_keyword_text or loading_keyword_title

    return {
        "input_url": input_url,
        "final_url": final_url,
        "status": status,
        "reason": reason,
        "matched_keyword": matched_keyword,
        "title": title,
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "text_length": len(text),
        "text_preview": text[:220],
    }


def classify_job_page(
    url: str,
    page: ChromiumPage,
    timeout_seconds: int = 10,
    poll_interval_seconds: float = 0.6,
) -> tuple[dict[str, Any], Any]:
    tab = page.new_tab(url)
    started_at = time.time()
    checks = 0
    timeout_seconds = max(3, int(timeout_seconds))
    per_check_timeout = 2
    transient_reasons = {
        "loading_page",
        "security_check_redirect",
        "risk_or_login_page",
        "no_signal",
        "redirected_out_of_job_detail",
    }

    last_result: dict[str, Any] | None = None
    while True:
        checks += 1
        result = classify_current_tab(
            tab=tab,
            input_url=url,
            timeout_seconds=per_check_timeout,
        )
        elapsed = time.time() - started_at
        if result["status"] in {"active", "closed"}:
            result["elapsed_seconds"] = round(elapsed, 2)
            result["poll_checks"] = checks
            return result, tab

        if elapsed >= timeout_seconds:
            if ASSUME_ACTIVE_WHEN_NOT_CLOSED and result.get("status") == "unknown":
                base_reason = str(result.get("reason", "no_signal") or "no_signal").strip()
                result["status"] = "active"
                result["reason"] = f"assumed_active:{base_reason}"
            else:
                result["reason"] = f"timeout_{result['reason']}"
            result["elapsed_seconds"] = round(elapsed, 2)
            result["poll_checks"] = checks
            return result, tab

        last_result = result
        if result["reason"] in transient_reasons:
            time.sleep(max(0.1, float(poll_interval_seconds)))
            continue

        if ASSUME_ACTIVE_WHEN_NOT_CLOSED and result.get("status") == "unknown":
            base_reason = str(result.get("reason", "no_signal") or "no_signal").strip()
            result["status"] = "active"
            result["reason"] = f"assumed_active:{base_reason}"
        result["elapsed_seconds"] = round(elapsed, 2)
        result["poll_checks"] = checks
        return result, tab

    # Should never reach here, but keep a safe fallback.
    if last_result is None:
        last_result = classify_current_tab(tab=tab, input_url=url, timeout_seconds=per_check_timeout)
    last_result["elapsed_seconds"] = round(time.time() - started_at, 2)
    last_result["poll_checks"] = checks
    return last_result, tab


def _safe_close_tab(tab: Any) -> None:
    try:
        tab.close()
    except Exception:
        pass


def _load_urls_from_file(path: str) -> list[str]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"url file not found: {path}")
    urls: list[str] = []
    for raw in file_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        urls.append(line)
    return urls


def _load_urls_from_db(limit: int) -> list[str]:
    # Lazy import to avoid DB initialization for help-only usage.
    from db import DatabaseManager

    db = DatabaseManager()
    with db.get_cursor() as cur:
        cur.execute(
            """
            SELECT detail_url
            FROM jobs
            WHERE detail_url IS NOT NULL
              AND detail_url <> ''
            ORDER BY create_time DESC
            LIMIT %s
            """,
            (limit,),
        )
        rows = cur.fetchall() or []
    return [str(row[0]).strip() for row in rows if row and str(row[0]).strip()]


def _build_url_list(args: argparse.Namespace) -> list[str]:
    urls: list[str] = []
    if args.url:
        urls.extend([item.strip() for item in args.url if item and item.strip()])
    if args.file:
        urls.extend(_load_urls_from_file(args.file))
    if args.from_db_limit > 0:
        urls.extend(_load_urls_from_db(args.from_db_limit))

    deduped: list[str] = []
    seen: set[str] = set()
    for url in urls:
        if url in seen:
            continue
        seen.add(url)
        deduped.append(url)
    return deduped


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate BOSS job detail urls (active/closed/unknown)")
    parser.add_argument("--url", action="append", default=[], help="Job detail url, repeatable")
    parser.add_argument("--file", default="", help="Path to txt file, one url per line")
    parser.add_argument("--from-db-limit", type=int, default=0, help="Load latest detail_url from DB")
    parser.add_argument("--timeout", type=int, default=12, help="Max wait seconds for final status (with polling)")
    parser.add_argument("--poll-interval", type=float, default=0.6, help="Polling interval in seconds")
    parser.add_argument("--sleep", type=float, default=1.2, help="Sleep seconds between url checks")
    parser.add_argument("--output", default="", help="Optional output JSON file path")
    parser.add_argument(
        "--pause-on-unknown",
        action="store_true",
        help="Pause on unknown status so you can inspect/login manually, then re-check same tab",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        urls = _build_url_list(args)
    except Exception as exc:
        logger.error("load urls failed: %s", exc, exc_info=True)
        return 2

    if not urls:
        logger.warning("no urls provided, use --url/--file/--from-db-limit")
        return 1

    logger.info(
        "job alive probe start: total_urls=%s timeout=%s poll_interval=%s",
        len(urls),
        args.timeout,
        args.poll_interval,
    )
    page = ChromiumPage()
    results: list[dict[str, Any]] = []
    try:
        for idx, url in enumerate(urls, start=1):
            logger.info("checking %s/%s: %s", idx, len(urls), url)
            result, tab = classify_job_page(
                url=url,
                page=page,
                timeout_seconds=max(3, int(args.timeout)),
                poll_interval_seconds=max(0.1, float(args.poll_interval)),
            )
            try:
                should_pause = bool(args.pause_on_unknown) or (len(urls) == 1 and sys.stdin.isatty())
                if should_pause and result["status"] == "unknown" and sys.stdin.isatty():
                    logger.info(
                        "unknown detected, tab kept open for manual inspection: title=%s final_url=%s reason=%s",
                        result["title"],
                        result["final_url"],
                        result["reason"],
                    )
                    input("当前为 unknown。请在浏览器中检查/登录后，按回车继续二次判定...")
                    result = classify_current_tab(
                        tab=tab,
                        input_url=url,
                        timeout_seconds=max(3, int(args.timeout)),
                    )
                    logger.info(
                        "recheck after pause: status=%s reason=%s matched=%s",
                        result["status"],
                        result["reason"],
                        result["matched_keyword"],
                    )

                results.append(result)
                logger.info(
                    "result %s/%s: status=%s reason=%s matched=%s",
                    idx,
                    len(urls),
                    result["status"],
                    result["reason"],
                    result["matched_keyword"],
                )
            finally:
                _safe_close_tab(tab)
            time.sleep(max(0.0, float(args.sleep)))
    finally:
        for method_name in ("quit", "close"):
            fn = getattr(page, method_name, None)
            if callable(fn):
                try:
                    fn()
                except Exception:
                    pass
                break

    summary = {
        "total": len(results),
        "active": sum(1 for item in results if item["status"] == "active"),
        "closed": sum(1 for item in results if item["status"] == "closed"),
        "unknown": sum(1 for item in results if item["status"] == "unknown"),
    }
    payload = {
        "summary": summary,
        "results": results,
    }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("result written to %s", str(output_path))

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    logger.info("job alive probe done: %s", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
