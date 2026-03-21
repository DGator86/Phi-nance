# Contributing to Phi-nance

Thanks for contributing to Phi-nance. This guide defines expected engineering and documentation standards.

## Development Setup

```bash
git clone https://github.com/DGator86/Phi-nance.git
cd Phi-nance
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
pre-commit install
cp .env.example .env
```

## Coding Standards

- **Type hints are required** for new/changed public interfaces.
- **Use custom exceptions** from `phi.exceptions` / `phinance.exceptions` where applicable.
- **Use centralized logging** (`phi.logging.get_logger` or package logger helper), not ad hoc `print` statements in runtime modules.
- **Validate external inputs** with helpers in `phi.utils.validation` where practical.
- Follow configured linting and formatting conventions (`ruff`, `black` profile settings in `pyproject.toml`).
- Maintain consistent docstring style for public classes/functions (Google- or NumPy-style acceptable; keep one style per module).

## Testing Expectations

Run before opening a PR:

```bash
pytest
ruff check .
mypy phinance phi
```

Recommended coverage run:

```bash
pytest --cov=phi --cov=phinance --cov=app_streamlit --cov-report=term-missing
```

For refactors, add or update tests that cover:

- nominal logic,
- error/validation paths,
- regression scenarios introduced by the change.

## Branching

- Integrate on **`MAIN`**; use short-lived feature branches. See [`DEV_WORKFLOW.md`](DEV_WORKFLOW.md) for conventions and cleanup tips.
- Avoid history-rewriting “clean slate” branches unless the whole team agrees on a migration plan.

## Pull Request Process

1. Create a focused branch and keep commits logically grouped.
2. Ensure lint/type/test checks pass locally.
3. Update documentation for any user-facing behavior or config changes.
4. Include concise PR notes: what changed, why, risk/rollback considerations.
5. Confirm CI passes (tests, lint, coverage gates as configured).

## Documentation Responsibilities

When changing behavior, also update relevant docs:

- `README.md` for top-level UX changes,
- `Architecture.md` for structural changes,
- `docs/*.md` topic guides,
- `.env.example` for environment variable additions/removals.

## Commit Hygiene

- Use clear commit messages in imperative mood.
- Avoid unrelated formatting-only churn.
- Keep sensitive values out of source control and logs.
