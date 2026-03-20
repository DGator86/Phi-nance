# Testing

## Run tests

```bash
pytest
```

For coverage-oriented runs:

```bash
pytest --cov=phi --cov=phinance --cov=app_streamlit --cov-report=term-missing
```

## What to test for new code

- Correctness of business logic
- Validation and error paths
- Logging side effects where operationally important
- Integration seams (data adapters, backtest runners, UI handlers)

## Style and quality checks

```bash
ruff check .
mypy phinance phi
```

Docstring style checks (if enabled in your environment):

```bash
pydocstyle phi phinance
```
