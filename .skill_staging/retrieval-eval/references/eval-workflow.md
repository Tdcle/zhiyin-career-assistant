# Eval Workflow

Repository root: `D:\PythonProject\find_a_good_job`

## Commands

- Generate or refresh the synthetic dataset:
  `python eval/generate_dataset.py`
- Run the evaluation pipeline:
  `python eval/evaluate_RAGAS.py`

## Output Directory

Each run writes to:

- `eval/results/run_YYYYMMDD_HHMMSS/`

Important files:

- `eval_report.json`
  Summary metrics for broad search and precise match.
- `broad_search_details.csv`
  Best first file for retrieval regressions.
- `ragas_details.csv`
  Per-row metric breakdown when RAGAS finishes successfully.
- `eval_process.log`
  Raw execution logs and stack traces.

## How To Read Results

1. Start with `eval_report.json`.
2. If Recall or MRR regressed, open `broad_search_details.csv`.
3. Inspect these columns first:
- `user_question`
- `semantic_query`
- `search_keyword_query`
- `search_city`
- `search_experience`
- `retrieved_ids`
- `hit_at_5`
- `hit_at_10`
4. Only inspect `ragas_details.csv` after retrieval looks sane.

## Typical Failure Modes

- Dataset produced bad search parameters.
- Eval normalization made `city` or `experience` too strict.
- Database recall is broad but misses the ground-truth job in the top K.
- The dataset expects match-style behavior from a retrieval-only run.

