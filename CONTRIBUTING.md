# Contributing to Phi-nance

Thank you for your interest in contributing!

## Development Setup

```bash
git clone https://github.com/DGator86/Phi-nance
cd Phi-nance
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install flake8
```

## Running Tests

```bash
pytest --cov=phi --cov=app_streamlit --cov-report=xml --cov-report=term-missing --cov-fail-under=80
```

## Linting

```bash
flake8 phi/ tests/ --max-line-length=120 --ignore=E501,W503
```

## Submitting a Pull Request

1. Fork the repository and create a feature branch from `MAIN`.
2. Make your changes with clear, descriptive commits.
3. Ensure all tests pass (`pytest --cov=phi --cov=app_streamlit --cov-report=xml --cov-report=term-missing --cov-fail-under=80`) and linting is clean.
4. Open a pull request against `MAIN` with a description of what you changed and why.

## Code Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) with a maximum line length of 120 characters.
- Use [black](https://github.com/psf/black)-compatible formatting.
- Add NumPy-style docstrings to all public functions and classes.
- Mock all external API calls in tests (no real network calls).

- Use centralized logging (`from phi.logging import get_logger`) for production modules; avoid `print(...)` outside intentional CLI UX output.


## Writing Unit Tests

- Keep tests in `tests/` and prefer isolated unit tests with mocked external APIs/filesystem.
- Use `tmp_path` for file IO and `monkeypatch`/`pytest-mock` for dependency isolation.
- Avoid network calls; mock vendor/fetch functions at module boundaries.
