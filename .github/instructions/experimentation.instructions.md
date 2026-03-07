# Experimentation Instructions

- Prefer `phinance.experiment.runner.run_experiment` for reproducible runs.
- Keep integrations optional: tracker may be `None`.
- Log run parameters, metrics, and at least one reproducibility artifact.
- Ensure new targets accept `tracker=None` and return final metrics as a dictionary.
