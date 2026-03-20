# Repository Guidelines

## Project Structure & Module Organization
`app.py` is the Gradio entry point and wires the chat and interview UI. Core workflows live in `logic/` (`chat_flow.py`, `interview_flow.py`), while graph orchestration is in `models/`. Shared infrastructure such as database access, logging, parsing, and maintenance scripts lives in `utils/`. Configuration is centralized in `config/config.py` and loaded from `config/.env`. Evaluation scripts and datasets are kept in `eval/`. UI assets belong in `assets/`; generated runtime files such as `logs/` and `static/` should not be committed.

## Build, Test, and Development Commands
Use a local virtual environment and run from the repository root.

```bash
python -m venv .venv
.venv\Scripts\activate
python app.py
```

`python app.py` starts the Gradio app. `python eval/evaluate_retrieval.py` runs retrieval metrics against `eval/synthetic_eval_dataset.json`. `python eval/evaluate_RAGAS.py` runs the heavier end-to-end evaluation pipeline and writes outputs under `eval/results/`. Utility scripts in `utils/` are intended for one-off maintenance tasks such as TSV rebuilding or database backup.

## Coding Style & Naming Conventions
Follow existing Python style: 4-space indentation, snake_case for functions and variables, PascalCase for classes, and short module names grouped by responsibility. Keep business logic out of `app.py`; prefer adding helpers in `logic/`, `models/`, or `utils/`. Preserve the current style of small focused functions and explicit logging via `utils.logger`. No formatter or linter config is checked in, so match surrounding code closely and keep imports tidy.

## Testing Guidelines
There is no dedicated `tests/` package yet; evaluation scripts in `eval/` are the current regression checks. Before opening a PR, run `python eval/evaluate_retrieval.py` for retrieval changes and `python eval/evaluate_RAGAS.py` for generation or ranking changes when dependencies are available. Name any new test data or scripts descriptively, for example `eval/test_search_cases.json` or `eval/evaluate_<feature>.py`.

## Commit & Pull Request Guidelines
Recent history uses short, feature-focused commit messages in Chinese, for example `升级混合检索、优化提示词模板`. Keep commits scoped to one change and use the same concise style. PRs should describe user-visible impact, note any config or schema changes, list commands run for validation, and include screenshots when UI behavior in `app.py` or `assets/style.css` changes.

## Security & Configuration Tips
Do not commit secrets. `config/.env` contains runtime settings such as `DASHSCOPE_API_KEY`, PostgreSQL credentials, and Ollama endpoints. Document any new environment variable in both the PR description and `config/config.py`.
