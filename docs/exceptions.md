# Exceptions

Use typed exceptions for predictable control flow and clearer error boundaries.

## Key modules

- `phi/exceptions.py`
- `phinance/exceptions.py`

## Guidance

- Raise specific exception types (`DataFetchError`, `CacheError`, etc.) instead of generic `Exception`.
- Catch narrow exception classes at integration boundaries (UI, CLI entry points).
- Preserve context with `raise ... from exc` when wrapping lower-level errors.
