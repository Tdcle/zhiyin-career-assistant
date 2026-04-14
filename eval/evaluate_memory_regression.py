from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import traceback
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

from psycopg2.extras import Json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from backend.db import DatabaseManager, _make_json_safe  # noqa: E402


DEFAULT_CASES_FILE = Path(__file__).resolve().with_name("memory_regression_cases.json")
DEFAULT_RESULTS_DIR = Path(__file__).resolve().with_name("results")
DEFAULT_USER_ID = "__memory_regression__"
DEFAULT_USERNAME = "memory_regression"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _normalize_text(value: Any) -> str:
    return str(value if value is not None else "").strip()


def _norm_casefold(value: Any) -> str:
    return _normalize_text(value).casefold()


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _contains_value(values: Any, expected: Any) -> bool:
    expected_norm = _norm_casefold(expected)
    return any(_norm_casefold(item) == expected_norm for item in _as_list(values))


def _contains_substring(values: Any, expected: Any) -> bool:
    expected_norm = _norm_casefold(expected)
    return any(expected_norm in _norm_casefold(item) for item in _as_list(values))


def _combined_plan_text(plan: dict[str, Any]) -> str:
    parts = [
        plan.get("keyword_query", ""),
        plan.get("resolved_query", ""),
        plan.get("city", ""),
        plan.get("experience", ""),
    ]
    return " ".join(_normalize_text(part) for part in parts if _normalize_text(part))


def ensure_eval_user(db: DatabaseManager, user_id: str, username: str) -> None:
    with db.get_cursor() as cur:
        cur.execute(
            """
            INSERT INTO users (user_id, username)
            VALUES (%s, %s)
            ON CONFLICT (user_id)
            DO UPDATE SET username = EXCLUDED.username
            """,
            (user_id, username),
        )


def cleanup_eval_user_data(db: DatabaseManager, user_id: str) -> None:
    with db.get_cursor() as cur:
        cur.execute("DELETE FROM memory_facts WHERE user_id = %s", (user_id,))
        cur.execute("DELETE FROM resumes WHERE user_id = %s", (user_id,))


def insert_resume_structured(db: DatabaseManager, user_id: str, structured_data: dict[str, Any]) -> None:
    with db.get_cursor() as cur:
        cur.execute("DELETE FROM resumes WHERE user_id = %s", (user_id,))
        cur.execute(
            """
            INSERT INTO resumes (
                user_id,
                filename,
                content,
                normalized_content,
                structured_data,
                parser_version,
                updated_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
            """,
            (
                user_id,
                "memory_regression.json",
                "",
                "",
                Json(_make_json_safe(structured_data or {})),
                "memory_regression",
            ),
        )


def find_memory_fact(
    db: DatabaseManager,
    user_id: str,
    match: dict[str, Any],
    include_inactive: bool = True,
) -> dict[str, Any] | None:
    expected_key = _norm_casefold(match.get("fact_key"))
    expected_value = _norm_casefold(match.get("fact_value"))
    for row in db.list_user_memory_items(user_id, include_inactive=include_inactive, limit=500):
        if expected_key and _norm_casefold(row.get("fact_key")) != expected_key:
            continue
        if expected_value and _norm_casefold(row.get("fact_value")) != expected_value:
            continue
        return row
    return None


def setup_memory_facts(db: DatabaseManager, user_id: str, facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    inserted: list[dict[str, Any]] = []
    for item in facts or []:
        fact_key = _normalize_text(item.get("fact_key"))
        fact_value = _normalize_text(item.get("fact_value"))
        if not fact_key or not fact_value:
            inserted.append({"ok": False, "input": item, "error": "empty fact_key or fact_value"})
            continue

        ok = db.add_memory_fact(
            user_id=user_id,
            fact_key=fact_key,
            fact_value=fact_value,
            source=_normalize_text(item.get("source")) or "test_setup",
            confidence=float(item.get("confidence", 0.75) or 0.75),
            importance=int(item.get("importance", 3) or 3),
            expires_at=item.get("expires_at"),
            meta=item.get("meta") if isinstance(item.get("meta"), dict) else None,
        )
        row = find_memory_fact(db, user_id, {"fact_key": fact_key, "fact_value": fact_value}, include_inactive=True)

        if item.get("is_active") is False and row:
            db.delete_memory_fact(user_id, int(row["id"]))
            row = find_memory_fact(db, user_id, {"fact_key": fact_key, "fact_value": fact_value}, include_inactive=True)

        inserted.append({"ok": bool(ok), "input": item, "row": row})
    return inserted


def apply_memory_operation(db: DatabaseManager, user_id: str, operation: dict[str, Any] | None) -> dict[str, Any]:
    if not operation:
        return {}

    op_type = _normalize_text(operation.get("type")).lower()
    target = find_memory_fact(db, user_id, operation.get("match") or {}, include_inactive=False)
    if not target:
        return {"ok": False, "type": op_type, "error": "target fact not found", "operation": operation}

    fact_id = int(target["id"])
    if op_type == "update":
        payload = operation.get("payload") or {}
        updated = db.update_memory_fact(
            user_id=user_id,
            fact_id=fact_id,
            fact_key=_normalize_text(payload.get("fact_key")),
            fact_value=_normalize_text(payload.get("fact_value")),
            confidence=float(payload.get("confidence", 0.75) or 0.75),
            importance=int(payload.get("importance", 3) or 3),
            meta=payload.get("meta") if isinstance(payload.get("meta"), dict) else None,
        )
        return {"ok": bool(updated), "type": op_type, "target": target, "result": updated}

    if op_type == "delete":
        deleted = db.delete_memory_fact(user_id, fact_id)
        return {"ok": bool(deleted), "type": op_type, "target": target}

    return {"ok": False, "type": op_type, "error": "unsupported operation", "operation": operation}


def extract_explicit_constraints_for_eval(latest_text: str, base_plan: dict[str, Any]) -> dict[str, Any]:
    text = _normalize_text(latest_text)
    lowered = text.casefold()

    salary_min = int(base_plan.get("salary_min", 0) or 0) if isinstance(base_plan, dict) else 0
    salary_unit = _normalize_text(base_plan.get("salary_unit")) if isinstance(base_plan, dict) else ""
    salary_mentioned = salary_min > 0 or bool(
        re.search(r"\d{1,4}\s*(?:k|/day|per\s*day|day)", lowered)
    )

    experience = _normalize_text(base_plan.get("experience")) if isinstance(base_plan, dict) else ""
    experience_mentioned = bool(experience) or bool(re.search(r"\b(?:intern|internship|new\s*grad)\b", lowered))

    return {
        "experience_mentioned": experience_mentioned,
        "experience": experience,
        "salary_mentioned": salary_mentioned,
        "salary_min": salary_min,
        "salary_unit": salary_unit,
    }


def resolve_plan_with_memory(
    db: DatabaseManager,
    user_id: str,
    user_text: str,
    base_plan: dict[str, Any] | None,
) -> dict[str, Any]:
    from backend.models.search_plan_resolver import (  # noqa: WPS433
        apply_memory_resolution_to_plan,
        extract_turn_slot_signals,
    )

    initial_plan = deepcopy(base_plan or {})
    memory_resolution = db.resolve_memory_preferences(
        user_id=user_id,
        query=user_text or "",
        scene="chat",
        limit=80,
    )
    explicit_constraints = extract_explicit_constraints_for_eval(user_text, initial_plan)
    slot_signals = extract_turn_slot_signals(user_text or "")
    final_plan, trace = apply_memory_resolution_to_plan(
        initial_plan,
        user_text or "",
        explicit_constraints,
        memory_resolution,
        slot_signals=slot_signals,
    )
    return {
        "memory_resolution": memory_resolution,
        "explicit_constraints": explicit_constraints,
        "slot_signals": slot_signals,
        "trace": trace,
        "plan": final_plan,
    }


def assert_expected_profile(profile: dict[str, Any], expected: dict[str, Any] | None) -> list[str]:
    failures: list[str] = []
    if not expected:
        return failures

    for key, expected_value in expected.items():
        if key.endswith("_not_contains"):
            field = key[: -len("_not_contains")]
            actual = profile.get(field, [])
            for item in _as_list(expected_value):
                if _contains_value(actual, item):
                    failures.append(f"profile.{field} should not contain {item!r}; actual={actual!r}")
            continue

        if key.endswith("_contains"):
            field = key[: -len("_contains")]
            actual = profile.get(field, [])
            for item in _as_list(expected_value):
                if not _contains_value(actual, item):
                    failures.append(f"profile.{field} missing expected value {item!r}; actual={actual!r}")
            continue

        actual_value = profile.get(key)
        if isinstance(expected_value, list):
            actual_list = _as_list(actual_value)
            if sorted(_norm_casefold(item) for item in actual_list) != sorted(_norm_casefold(item) for item in expected_value):
                failures.append(f"profile.{key} expected {expected_value!r}; actual={actual_value!r}")
        elif actual_value != expected_value:
            failures.append(f"profile.{key} expected {expected_value!r}; actual={actual_value!r}")

    return failures


def assert_expected_facts(facts: list[dict[str, Any]], expected: list[dict[str, Any]] | None) -> list[str]:
    failures: list[str] = []
    if expected is None:
        return failures

    active_facts = [row for row in facts if row.get("is_active", True)]
    for item in expected:
        expected_key = _norm_casefold(item.get("fact_key"))
        exact_value = item.get("fact_value")
        contains_value = item.get("fact_value_contains")

        candidates = [row for row in active_facts if _norm_casefold(row.get("fact_key")) == expected_key]
        if exact_value is not None:
            matched = any(_norm_casefold(row.get("fact_value")) == _norm_casefold(exact_value) for row in candidates)
            if not matched:
                failures.append(f"active fact {item!r} not found; candidates={candidates!r}")
        elif contains_value is not None:
            matched = any(_norm_casefold(contains_value) in _norm_casefold(row.get("fact_value")) for row in candidates)
            if not matched:
                failures.append(f"active fact containing {item!r} not found; candidates={candidates!r}")
        elif not candidates:
            failures.append(f"active fact key {item.get('fact_key')!r} not found")

    if expected == [] and active_facts:
        failures.append(f"expected no active facts; actual={active_facts!r}")

    return failures


def assert_expected_resolution(resolution: dict[str, Any], expected: dict[str, Any] | None) -> list[str]:
    failures: list[str] = []
    if not expected:
        return failures

    mapping = {
        "roles_include_contains": "roles_include",
        "roles_exclude_contains": "roles_exclude",
        "cities_include_contains": "cities_include",
        "cities_exclude_contains": "cities_exclude",
        "experience_exclude_contains": "experience_exclude",
    }
    for key, expected_value in expected.items():
        if key in mapping:
            field = mapping[key]
            actual = resolution.get(field, [])
            for item in _as_list(expected_value):
                if not _contains_value(actual, item):
                    failures.append(f"resolution.{field} missing {item!r}; actual={actual!r}")
            continue

        actual_value = resolution.get(key)
        if actual_value != expected_value:
            failures.append(f"resolution.{key} expected {expected_value!r}; actual={actual_value!r}")

    return failures


def assert_expected_plan(plan: dict[str, Any], expected: dict[str, Any] | None) -> list[str]:
    failures: list[str] = []
    if not expected:
        return failures

    combined_text = _combined_plan_text(plan)
    for key, expected_value in expected.items():
        if key == "keyword_query_contains":
            keyword_text = _normalize_text(plan.get("keyword_query")) or combined_text
            for item in _as_list(expected_value):
                if _norm_casefold(item) not in _norm_casefold(keyword_text):
                    failures.append(f"plan.keyword_query missing {item!r}; actual={plan.get('keyword_query')!r}")
            continue

        if key == "keyword_query_not_contains":
            keyword_text = _normalize_text(plan.get("keyword_query"))
            for item in _as_list(expected_value):
                if _norm_casefold(item) in _norm_casefold(keyword_text):
                    failures.append(f"plan.keyword_query should not contain {item!r}; actual={keyword_text!r}")
            continue

        actual_value = plan.get(key)
        if actual_value != expected_value:
            failures.append(f"plan.{key} expected {expected_value!r}; actual={actual_value!r}")

    return failures


def assert_expected_context(context: str, expected: list[str] | None) -> list[str]:
    failures: list[str] = []
    if not expected:
        return failures

    context_norm = _norm_casefold(context)
    for item in expected:
        if _norm_casefold(item) not in context_norm:
            failures.append(f"context missing {item!r}; actual={context!r}")
    return failures


def run_case(db: DatabaseManager, case: dict[str, Any], user_id: str) -> dict[str, Any]:
    cleanup_eval_user_data(db, user_id)
    detail: dict[str, Any] = {
        "case_id": case.get("case_id"),
        "category": case.get("category"),
        "description": case.get("description"),
        "passed": False,
        "failures": [],
    }

    try:
        detail["setup_results"] = setup_memory_facts(db, user_id, case.get("setup_facts") or [])

        if isinstance(case.get("setup_resume_structured"), dict):
            insert_resume_structured(db, user_id, case["setup_resume_structured"])

        if case.get("category") == "ingest":
            detail["ingest_result"] = db.ingest_user_memory_from_text(
                user_id=user_id,
                text=case.get("user_text") or "",
                source="memory_regression",
            )

        if case.get("operation"):
            detail["operation_result"] = apply_memory_operation(db, user_id, case.get("operation"))

        profile = db.get_memory_profile(user_id)
        facts = db.list_user_memory_items(user_id, include_inactive=True, limit=500)
        detail["actual_profile"] = profile
        detail["actual_facts"] = facts

        if case.get("expected_resolution") or case.get("expected_plan"):
            try:
                resolved = resolve_plan_with_memory(
                    db=db,
                    user_id=user_id,
                    user_text=case.get("user_text") or "",
                    base_plan=case.get("base_plan") or {},
                )
                detail.update(
                    {
                        "actual_resolution": resolved.get("memory_resolution", {}),
                        "actual_plan": resolved.get("plan", {}),
                        "plan_trace": resolved.get("trace", {}),
                        "slot_signals": resolved.get("slot_signals", {}),
                        "explicit_constraints": resolved.get("explicit_constraints", {}),
                    }
                )
            except Exception as exc:
                detail["actual_resolution"] = {}
                detail["actual_plan"] = deepcopy(case.get("base_plan") or {})
                detail["plan_resolution_error"] = f"{type(exc).__name__}: {exc}"
                detail["plan_resolution_traceback"] = traceback.format_exc()

        if case.get("expected_context_contains") is not None:
            detail["actual_context"] = db.build_memory_context(
                user_id=user_id,
                scene="chat",
                max_facts=8,
                query=case.get("user_text") or "",
            )

        failures: list[str] = []
        failures.extend(assert_expected_profile(profile, case.get("expected_profile")))
        failures.extend(assert_expected_facts(facts, case.get("expected_facts")))
        failures.extend(assert_expected_resolution(detail.get("actual_resolution", {}), case.get("expected_resolution")))
        failures.extend(assert_expected_plan(detail.get("actual_plan", {}), case.get("expected_plan")))
        failures.extend(assert_expected_context(detail.get("actual_context", ""), case.get("expected_context_contains")))

        if detail.get("plan_resolution_error") and (case.get("expected_resolution") or case.get("expected_plan")):
            failures.append(detail["plan_resolution_error"])

        detail["failures"] = failures
        detail["passed"] = not failures
        return detail
    except Exception as exc:
        detail["failures"] = [f"case crashed: {type(exc).__name__}: {exc}"]
        detail["traceback"] = traceback.format_exc()
        return detail


def summarize_results(details: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(details)
    passed = sum(1 for item in details if item.get("passed"))
    failed = total - passed

    by_category: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "passed": 0, "failed": 0})
    for item in details:
        category = _normalize_text(item.get("category")) or "unknown"
        by_category[category]["total"] += 1
        if item.get("passed"):
            by_category[category]["passed"] += 1
        else:
            by_category[category]["failed"] += 1

    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": round((passed / total) * 100, 2) if total else 0.0,
        "by_category": dict(sorted(by_category.items())),
        "failure_case_ids": [item.get("case_id") for item in details if not item.get("passed")],
    }


def write_csv_summary(path: Path, details: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["case_id", "category", "passed", "failure_count", "failures"],
        )
        writer.writeheader()
        for item in details:
            writer.writerow(
                {
                    "case_id": item.get("case_id"),
                    "category": item.get("category"),
                    "passed": bool(item.get("passed")),
                    "failure_count": len(item.get("failures") or []),
                    "failures": " | ".join(item.get("failures") or []),
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate memory regression cases.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_FILE, help="Path to memory regression cases JSON.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR, help="Directory for run outputs.")
    parser.add_argument("--user-id", default=DEFAULT_USER_ID, help="Isolated eval user id.")
    parser.add_argument("--username", default=DEFAULT_USERNAME, help="Isolated eval username.")
    parser.add_argument("--case-id", action="append", default=[], help="Run only selected case id. Can be repeated.")
    parser.add_argument("--category", action="append", default=[], help="Run only selected category. Can be repeated.")
    parser.add_argument("--keep-data", action="store_true", help="Keep eval user's memory and resume rows after run.")
    parser.add_argument("--disable-llm", action="store_true", help="Disable memory extraction LLM and use rule fallback only.")
    parser.add_argument("--no-fail-exit", action="store_true", help="Always exit with code 0 even when cases fail.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cases = _read_json(args.cases)
    if not isinstance(cases, list):
        raise ValueError(f"cases file must contain a list: {args.cases}")

    selected_case_ids = set(args.case_id or [])
    selected_categories = set(args.category or [])
    if selected_case_ids:
        cases = [case for case in cases if case.get("case_id") in selected_case_ids]
    if selected_categories:
        cases = [case for case in cases if case.get("category") in selected_categories]
    if not cases:
        raise ValueError("no memory regression cases selected")

    db = DatabaseManager()
    if args.disable_llm and hasattr(db, "memory_extract_llm"):
        db.memory_extract_llm = None

    ensure_eval_user(db, args.user_id, args.username)

    run_id = datetime.now().strftime("memory_%Y%m%d_%H%M%S")
    run_dir = args.results_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    details: list[dict[str, Any]] = []
    category_counter = Counter(_normalize_text(case.get("category")) or "unknown" for case in cases)
    print(f"Loaded {len(cases)} memory regression cases from {args.cases}")
    print(f"Categories: {dict(sorted(category_counter.items()))}")

    for index, case in enumerate(cases, start=1):
        case_id = case.get("case_id")
        print(f"[{index}/{len(cases)}] {case_id}")
        detail = run_case(db, case, args.user_id)
        details.append(detail)
        status = "PASS" if detail.get("passed") else "FAIL"
        print(f"  {status}")
        for failure in detail.get("failures") or []:
            print(f"  - {failure}")

    if not args.keep_data:
        cleanup_eval_user_data(db, args.user_id)

    summary = summarize_results(details)
    report = {
        "run_id": run_id,
        "cases_file": str(args.cases),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "user_id": args.user_id,
        "llm_disabled": bool(args.disable_llm),
        "summary": summary,
    }

    report_path = run_dir / "memory_eval_report.json"
    details_path = run_dir / "memory_eval_details.json"
    csv_path = run_dir / "memory_eval_details.csv"
    _write_json(report_path, report)
    _write_json(details_path, details)
    write_csv_summary(csv_path, details)

    print("")
    print("Memory regression summary")
    print(f"  total     : {summary['total']}")
    print(f"  passed    : {summary['passed']}")
    print(f"  failed    : {summary['failed']}")
    print(f"  pass rate : {summary['pass_rate']}%")
    print(f"  report    : {report_path}")
    print(f"  details   : {details_path}")
    print(f"  csv       : {csv_path}")

    if summary["failed"] and not args.no_fail_exit:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
