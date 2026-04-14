"""Nightly scheduled crawler for BOSS jobs.

Run as an independent worker process:
    python backend/utils/nightly_crawler.py

By default it runs only in the night window (00:00-06:00), and refreshes in cycles.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import quote

from DrissionPage import ChromiumPage

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from utils.logger import get_logger


logger = get_logger("nightly_crawler")
_FETCH_HELPERS: tuple | None = None


# Nightly scheduler window: [00:00, 06:00)
NIGHT_START_HOUR = 0
NIGHT_END_HOUR = 6

# Scheduler cadence
DEFAULT_RUN_INTERVAL_MINUTES = 60

# Per keyword+city crawl budget
TARGET_COUNT_PER_COMBO = 50
MAX_IDLE_ROUNDS = 12
MAX_SECONDS_PER_COMBO = 8 * 60

# Anti-blocking pacing
DETAIL_FETCH_SLEEP_RANGE = (1.2, 2.6)
SCROLL_SLEEP_RANGE = (1.5, 3.0)
COMBO_COOLDOWN_RANGE = (4.0, 8.0)


@dataclass(frozen=True)
class CityTarget:
    name: str
    code: str = ""


@dataclass(frozen=True)
class CrawlTarget:
    keyword: str
    city_name: str
    city_code: str = ""


# Fill/adjust as needed.
TARGET_ROLE_KEYWORDS = [
    "大模型",
    "AI Agent",
    "机器学习",
    "深度学习",
    "算法工程师",
    "NLP",
    "多模态",
    "AIGC",
    "数据科学",
    "数据分析",
    "数据开发",
    "Python",
    "Java",
    "Golang",
    "C++",
    "前端开发",
    "后端开发",
    "测试开发",
    "运维开发",
    "产品经理",
]

# If city code is unknown, leave code as "" and fill later.
# Entries with empty code (except "全国") are skipped to avoid duplicate national crawling.
TARGET_CITIES = [
    CityTarget("全国", ""),
    CityTarget("北京", "101010100"),
    CityTarget("上海", "101020100"),
    CityTarget("广州", "101280100"),
    CityTarget("深圳", "101280600"),
    CityTarget("杭州", "101210100"),
    CityTarget("南京", "101190100"),
    CityTarget("武汉", "101200100"),
    CityTarget("成都", "101270100"),
    CityTarget("西安", "101110100"),
    CityTarget("苏州", "101190400"),
    CityTarget("天津", "101030100"),
]


def _build_targets() -> list[CrawlTarget]:
    targets: list[CrawlTarget] = []
    for keyword in TARGET_ROLE_KEYWORDS:
        for city in TARGET_CITIES:
            if not city.code and city.name != "全国":
                logger.warning("skip city without code: city=%s keyword=%s", city.name, keyword)
                continue
            targets.append(CrawlTarget(keyword=keyword, city_name=city.name, city_code=city.code))
    return targets


def _get_fetch_helpers():
    global _FETCH_HELPERS
    if _FETCH_HELPERS is None:
        from utils.fetch_data import build_job_data_from_list_item, crawl_job_detail, persist_job_record

        _FETCH_HELPERS = (
            build_job_data_from_list_item,
            crawl_job_detail,
            persist_job_record,
        )
    return _FETCH_HELPERS


def _build_jobs_url(keyword: str, city_code: str = "") -> str:
    query = quote(keyword.strip())
    params = [f"query={query}"]
    if city_code.strip():
        params.insert(0, f"city={city_code.strip()}")
    return f"https://www.zhipin.com/web/geek/jobs?{'&'.join(params)}"


def _in_night_window(now: datetime | None = None) -> bool:
    current = now or datetime.now()
    return NIGHT_START_HOUR <= current.hour < NIGHT_END_HOUR


def _seconds_until_next_window(now: datetime | None = None) -> int:
    current = now or datetime.now()
    next_start = current.replace(hour=NIGHT_START_HOUR, minute=0, second=0, microsecond=0)
    if current.hour >= NIGHT_END_HOUR:
        next_start = next_start + timedelta(days=1)
    elif current.hour < NIGHT_START_HOUR:
        pass
    else:
        return 0
    return max(1, int((next_start - current).total_seconds()))


def _safe_close_page(page: ChromiumPage) -> None:
    for method_name in ("quit", "close"):
        fn = getattr(page, method_name, None)
        if callable(fn):
            try:
                fn()
            except Exception:
                pass
            return


def _sleep_in_range(seconds_range: tuple[float, float]) -> None:
    low, high = seconds_range
    time.sleep(random.uniform(low, high))


def crawl_target_once(page: ChromiumPage, target: CrawlTarget, target_count: int = TARGET_COUNT_PER_COMBO) -> dict:
    build_job_data_from_list_item, crawl_job_detail, persist_job_record = _get_fetch_helpers()
    url = _build_jobs_url(target.keyword, target.city_code)
    seen_job_ids: set[str] = set()
    success_count = 0
    inserted_count = 0
    updated_count = 0
    fail_count = 0
    no_progress_rounds = 0
    start_ts = time.time()

    logger.info(
        "crawl target start: keyword=%s city=%s code=%s target_count=%s",
        target.keyword,
        target.city_name,
        target.city_code or "<nationwide>",
        target_count,
    )

    page.listen.start("wapi/zpgeek/search/joblist.json")
    page.get(url)

    while success_count < target_count:
        if time.time() - start_ts > MAX_SECONDS_PER_COMBO:
            logger.warning(
                "crawl target timeout: keyword=%s city=%s success=%s/%s",
                target.keyword,
                target.city_name,
                success_count,
                target_count,
            )
            break
        if no_progress_rounds >= MAX_IDLE_ROUNDS:
            logger.warning(
                "crawl target stalled: keyword=%s city=%s success=%s/%s no_progress_rounds=%s",
                target.keyword,
                target.city_name,
                success_count,
                target_count,
                no_progress_rounds,
            )
            break

        progress_before = success_count
        packet = page.listen.wait(timeout=7)
        if not packet:
            no_progress_rounds += 1
            page.scroll.to_bottom()
            _sleep_in_range(SCROLL_SLEEP_RANGE)
            continue

        body = packet.response.body
        if not isinstance(body, dict):
            no_progress_rounds += 1
            continue

        job_list = ((body.get("zpData") or {}).get("jobList") or [])
        if not job_list:
            no_progress_rounds += 1
            page.scroll.to_bottom()
            _sleep_in_range(SCROLL_SLEEP_RANGE)
            continue

        for raw_job in job_list:
            if success_count >= target_count:
                break
            job_id = str(raw_job.get("encryptJobId", "") or "").strip()
            if not job_id or job_id in seen_job_ids:
                continue
            seen_job_ids.add(job_id)

            try:
                job_data = build_job_data_from_list_item(raw_job)
                detail = crawl_job_detail(page, job_data["detail_url"])
                if not detail:
                    fail_count += 1
                    _sleep_in_range(DETAIL_FETCH_SLEEP_RANGE)
                    continue

                job_data["detail"] = detail
                status = persist_job_record(job_data)
                if status == "inserted":
                    inserted_count += 1
                    success_count += 1
                elif status == "updated":
                    updated_count += 1
                    success_count += 1
                else:
                    fail_count += 1
            except Exception as exc:
                fail_count += 1
                logger.error(
                    "crawl target item failed: keyword=%s city=%s job_id=%s err=%s",
                    target.keyword,
                    target.city_name,
                    job_id,
                    exc,
                    exc_info=True,
                )
            finally:
                _sleep_in_range(DETAIL_FETCH_SLEEP_RANGE)

        if success_count == progress_before:
            no_progress_rounds += 1
        else:
            no_progress_rounds = 0

        if success_count < target_count:
            page.scroll.to_bottom()
            _sleep_in_range(SCROLL_SLEEP_RANGE)

    elapsed = round(time.time() - start_ts, 2)
    stats = {
        "keyword": target.keyword,
        "city": target.city_name,
        "city_code": target.city_code,
        "target_count": target_count,
        "success_count": success_count,
        "inserted_count": inserted_count,
        "updated_count": updated_count,
        "fail_count": fail_count,
        "seen_unique_count": len(seen_job_ids),
        "elapsed_seconds": elapsed,
    }
    logger.info("crawl target done: %s", stats)
    return stats


def run_refresh_cycle(target_count: int = TARGET_COUNT_PER_COMBO) -> dict:
    targets = _build_targets()
    logger.info("nightly refresh cycle start: target_groups=%s target_count_per_group=%s", len(targets), target_count)

    summary = {
        "target_groups": len(targets),
        "success_total": 0,
        "inserted_total": 0,
        "updated_total": 0,
        "fail_total": 0,
        "results": [],
    }

    if not targets:
        logger.warning("nightly refresh skipped: no enabled crawl targets")
        return summary

    page = ChromiumPage()
    try:
        for index, target in enumerate(targets, start=1):
            logger.info(
                "nightly refresh progress: %s/%s keyword=%s city=%s",
                index,
                len(targets),
                target.keyword,
                target.city_name,
            )
            stats = crawl_target_once(page=page, target=target, target_count=target_count)
            summary["results"].append(stats)
            summary["success_total"] += int(stats["success_count"])
            summary["inserted_total"] += int(stats["inserted_count"])
            summary["updated_total"] += int(stats["updated_count"])
            summary["fail_total"] += int(stats["fail_count"])
            _sleep_in_range(COMBO_COOLDOWN_RANGE)
    finally:
        _safe_close_page(page)

    logger.info(
        "nightly refresh cycle done: success=%s inserted=%s updated=%s fail=%s",
        summary["success_total"],
        summary["inserted_total"],
        summary["updated_total"],
        summary["fail_total"],
    )
    changed_count = int(summary["inserted_total"]) + int(summary["updated_total"])
    if changed_count > 0:
        try:
            from utils.search_cache import bump_search_data_version

            bump_search_data_version(reason=f"nightly_refresh_changed:{changed_count}")
        except Exception:
            logger.warning("nightly refresh bump search data version failed", exc_info=True)
    return summary


def run_scheduler_forever(interval_minutes: int = DEFAULT_RUN_INTERVAL_MINUTES) -> None:
    interval_seconds = max(60, int(interval_minutes * 60))
    logger.info(
        "nightly scheduler started: window=%02d:00-%02d:00 interval_minutes=%s",
        NIGHT_START_HOUR,
        NIGHT_END_HOUR,
        interval_minutes,
    )
    while True:
        now = datetime.now()
        if _in_night_window(now):
            logger.info("within night window, starting refresh cycle at %s", now.isoformat(timespec="seconds"))
            try:
                run_refresh_cycle(target_count=TARGET_COUNT_PER_COMBO)
            except Exception as exc:
                logger.error("nightly refresh cycle failed: %s", exc, exc_info=True)
            sleep_seconds = interval_seconds + random.randint(10, 120)
            logger.info("nightly scheduler sleep: %s seconds", sleep_seconds)
            time.sleep(sleep_seconds)
            continue

        wait_seconds = _seconds_until_next_window(now)
        logger.info(
            "outside night window, next run in %.2f hours (at %s)",
            wait_seconds / 3600,
            (now + timedelta(seconds=wait_seconds)).strftime("%Y-%m-%d %H:%M:%S"),
        )
        time.sleep(wait_seconds)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nightly crawler scheduler")
    parser.add_argument("--once", action="store_true", help="Run a single refresh cycle now and exit")
    parser.add_argument(
        "--interval-minutes",
        type=int,
        default=DEFAULT_RUN_INTERVAL_MINUTES,
        help="Scheduler cycle interval (effective during 00:00-06:00)",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=TARGET_COUNT_PER_COMBO,
        help="Per keyword+city target count",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.once:
        run_refresh_cycle(target_count=max(1, int(args.target_count)))
    else:
        run_scheduler_forever(interval_minutes=max(1, int(args.interval_minutes)))
