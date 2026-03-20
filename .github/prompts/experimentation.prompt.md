You are implementing experimentation support for Phi-nance.

Checklist:
1. Define/update tracker abstractions in `phinance/experiment`.
2. Ensure runner validates YAML config and executes target callables.
3. Preserve backward compatibility (`tracker=None`).
4. Log reproducibility metadata (git hash + environment snapshot).
5. Add/update tests and documentation.
