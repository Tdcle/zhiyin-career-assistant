# Dataset Fields

The synthetic dataset lives at `eval/synthetic_eval_dataset.json`.

## Important Fields

- `query_type`
  `broad_search` for retrieval metrics, `precise_match` for answer quality.
- `user_question`
  The natural-language prompt.
- `semantic_query`
  Search-oriented short keywords, not a full sentence.
- `city`
  Single standard city or empty string.
- `experience`
  Canonical values such as `实习`, `应届`, `1-3年`, or empty string.
- `company`
  Empty for `broad_search`; present for `precise_match` when needed.
- `salary`
  Canonical text such as `15K` or `150元/天`.

## Eval Mapping

`eval/evaluate_RAGAS.py` maps dataset fields to `db.hybrid_search()` through a normalization layer:

- `semantic_query` and `title` become `keyword_query`
- `city` becomes normalized `search_city`
- `experience` becomes normalized `search_experience`
- `salary` becomes `search_salary_min` plus `search_salary_unit`

## Practical Rule

When evaluating retrieval regressions, trust the normalized `search_*` columns in `broad_search_details.csv` more than the raw dataset fields. The regression may come from normalization rather than the original sample.

