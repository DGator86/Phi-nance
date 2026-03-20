---
applyTo: "notebooks/**/*.ipynb"
---
# Notebook authoring guidelines

- Prefer concise, executable cells with markdown context before each code block.
- Use helpers from `phinance.experiment.visualization` instead of duplicating plotting logic.
- Keep notebooks robust in fresh environments (avoid hidden state and rely on explicit imports).
- Assume MLflow tracking data already exists; provide graceful messages when runs are missing.
- Keep sample paths and IDs as placeholders (`YOUR_SWEEP_RUN_ID`) unless deterministic fixture data is used.
