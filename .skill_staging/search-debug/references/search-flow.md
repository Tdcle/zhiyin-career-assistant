# Search Flow

Repository root: `D:\PythonProject\find_a_good_job`

## File Map

- `models/chat_graph.py`
  Parses user intent, normalizes `search_plan`, handles retry logic, and stores `search_results`.
- `utils/tools.py`
  Builds `effective_query`, normalizes search parameters, calls `db.hybrid_search`, and applies title filtering.
- `utils/database.py`
  Runs `hybrid_search`, vector recall, BM25 recall, and hard filters such as `city`, `company`, and some `experience` values.
- `logic/chat_flow.py`
  Streams chat output, clears or rebuilds card state, and persists conversation state.
- `app.py`
  Wires the chat UI and recommendation buttons.

## Debug Path

1. Planner
  Check `intent_parse_node`, `_normalize_search_plan`, and retry rewrite behavior.
2. Tool
  Check `_normalize_city`, `_normalize_experience`, `_build_effective_search_query`, `_extract_role_keywords`, and `_filter_candidates_by_title`.
3. Database
  Check `hybrid_search`, `_should_apply_experience_filter`, `_vector_recall`, `_bm25_recall`, and `_rrf_fuse`.
4. UI
  Check `clear_user_chat_session`, `respond`, and `_build_job_button_updates`.

## High-Value Questions

- Did planner produce the wrong constraints?
- Did normalization fix or worsen them?
- Did SQL hard filters remove valid jobs before recall?
- Did title filtering remove jobs that should remain?
- Did the UI show stale cards despite correct search state?

