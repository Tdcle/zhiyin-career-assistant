---
name: search-debug
description: Diagnose job search failures and retrieval quality issues for the find_a_good_job repository. Use when a query returns too few or too many jobs, the wrong role family, stale recommendation cards, incorrect planner fields, or mismatches between search logs and UI output.
---

# Search Debug

Use this skill when debugging the search stack in `find_a_good_job`.

Start from one concrete failing query. Record:
- exact user text
- expected jobs
- actual jobs
- whether the problem is recall, filtering, ranking, or stale UI state

Read `references/search-flow.md` for the file map and `references/log-signals.md` for the key log lines.

## Workflow

1. Reproduce the issue with one query and capture the latest logs.
2. Classify the failure:
- Planner issue: wrong `city`, `experience`, `salary_unit`, or `keyword_query`
- Retrieval issue: `hybrid_search` returns zero or irrelevant results
- Post-filter issue: title filter or salary filter removes good candidates
- UI/state issue: recommendation cards do not refresh or clear
3. Inspect the matching layer before changing code:
- Planner and normalization: `models/chat_graph.py`
- Search tool assembly and title filtering: `utils/tools.py`
- Database recall and hard filters: `utils/database.py`
- Card updates and chat state: `logic/chat_flow.py`
4. Verify the fix with the same query and compare the before/after log lines.

## Checks

Use this order when triaging:

1. Confirm `planner normalized search plan` is reasonable.
2. Confirm `search tool invoked` shows normalized values you expect.
3. Confirm `hybrid search` is not over-constrained by `city`, `company`, `experience`, or `salary`.
4. Confirm `recall count` and `[RRF]` show plausible candidate volume.
5. Confirm `search tool kept X candidates after title filter` is not removing everything unexpectedly.
6. Confirm `chat_flow` clears and rebuilds job buttons for each new search.

## Common Patterns

- Query like `frontend` does not refresh cards after `testing`
  Check follow-up intent detection in `models/chat_graph.py`.
- Query like `Beijing nearby Java intern` returns zero recall
  Check normalization of `city` and `experience` before `hybrid_search`.
- Query like `frontend intern` returns `design intern`
  Check role extraction and title filtering in `utils/tools.py`.
- Cards remain from the previous search
  Check `search_results` reset in `models/chat_graph.py` and card updates in `logic/chat_flow.py`.

## Validation

After a fix:

1. rerun the exact same query
2. compare planner logs, recall logs, and final kept count
3. confirm the UI cards match the new search rather than stale state


