# Log Signals

Use these log lines in order.

## Planner

- `planner normalized search plan`
  The canonical search plan after prompt output and code-side normalization.

## Search Tool

- `search tool invoked`
  Shows raw and normalized `city` / `experience` plus the final `effective_query`.
- `search role keywords`
  Shows which role family the title filter will require.
- `search tool recalled X candidates`
  Candidate count before title filtering.
- `search tool kept X candidates after title filter`
  Candidate count after title filtering.

## Database

- `hybrid search`
  Shows the exact parameters passed to `db.hybrid_search`.
- `recall count: vec=... bm25=...`
  First signal of whether recall is too narrow or too broad.
- `[RRF]`
  Shows merged ranking and the top returned titles.

## UI / State

- `chat session cleared`
  Confirms the thread state and stored conversation state were removed.
- `chat message received`
  Confirms a fresh request entered the chat path.

## Quick Interpretation

- `vec=0 bm25=0`
  Usually a bad hard filter or bad query terms.
- `recalled > 0` and `kept 0`
  Title filter is probably too strict or role extraction is wrong.
- Good recall, good kept count, wrong cards in UI
  State reset or button update bug.

