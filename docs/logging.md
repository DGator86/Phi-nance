# Logging

Phi-nance uses centralized logger setup via `phi.logging` and package-level logging helpers.

## Defaults

- Level from `LOG_LEVEL` (default `INFO`).
- File output under `${LOGS_DIR}/phi.log`.
- Console logging enabled for operational visibility.

## Usage in code

```python
from phi.logging import get_logger

logger = get_logger(__name__)
logger.info("Backtest started", extra={"symbol": "SPY"})
```

## Guidance

- Prefer structured logger calls over `print` in production paths.
- Use `logger.exception(...)` inside exception handlers for tracebacks.
- Keep secrets out of log messages.
