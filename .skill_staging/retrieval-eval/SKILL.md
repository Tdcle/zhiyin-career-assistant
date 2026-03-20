---
name: retrieval-eval
description: Run and interpret retrieval evaluation for the find_a_good_job repository. Use when validating search changes, regenerating the synthetic evaluation dataset, checking Recall or MRR regressions, or understanding how dataset fields map into the current `db.hybrid_search` parameters.
---

# Retrieval Eval

Use this skill when measuring retrieval quality instead of debugging one live query.

Read `references/eval-workflow.md` first. Read `references/dataset-fields.md` when dataset values or normalized search parameters look suspicious.

## Workflow

1. Decide whether you need a fresh dataset.
- If the retrieval parameter schema changed, regenerate the dataset.
- If only ranking or filtering changed, you can often reuse the current dataset.
2. Run the evaluation script from repo root:
- `python eval/generate_dataset.py` when regeneration is needed
- `python eval/evaluate_RAGAS.py` for the current evaluation pipeline
3. Inspect outputs in the newest `eval/results/run_*/` directory.
4. Read files in this order:
- `eval_report.json`
- `broad_search_details.csv`
- `ragas_details.csv` if generation quality also matters
- `eval_process.log` for raw execution logs

## What To Optimize For

- Retrieval changes
  Prioritize `broad_search` metrics such as Recall@5, Recall@10, average rank, and MRR.
- Match-analysis or answer-generation changes
  Also inspect RAGAS metrics and low-scoring rows.

## Review Checklist

1. Compare raw dataset fields to normalized search params in `broad_search_details.csv`.
2. Check whether `search_experience` or `search_city` became over-constrained.
3. Check whether `search_keyword_query` is too verbose or too thin.
4. For regressions, inspect several misses rather than only the summary metrics.
5. Tie each miss back to one layer:
- dataset generation
- eval parameter normalization
- database recall
- title filtering

## Notes

- The evaluation path now calls `db.hybrid_search()` directly.
- The eval script does not use reranker behavior from `utils/tools.py`.
- A poor RAGAS score can be caused by bad retrieval, not only bad answer generation.

