# Development workflow

## Default branch

- **`MAIN`** (capitalized on this remote) is the integration branch. Keep it **deployable**: passing CI when possible, no intentional broken commits.
- Prefer **short-lived feature branches** merged via PR (or local merge with a clear message) rather than long-lived parallel “experiment” branches.

## Day-to-day

```bash
git fetch origin
git checkout MAIN
git pull origin MAIN
# create a branch for your change
git checkout -b fix/short-description
# ... edit, test ...
pytest tests/path/to/test_foo.py   # or full pytest before push
git push -u origin fix/short-description
# open PR → merge → delete remote branch
```

## After merge: cleanup

Periodically remove **merged** remote branches to reduce noise:

```bash
git fetch --prune
# list merged into MAIN (example)
git branch -r --merged origin/MAIN
```

Deleting stale branches on GitHub is safe **after** they are merged and you no longer need the PR reference.

## What we avoid

- **Orphan “clean history” branches** that rewrite all commits—high cost for collaborators and no benefit over disciplined merges on `MAIN`.
- **Mega-renames** (`phi` → `src/phinance`) without a dedicated migration plan, version bump, and doc update—do those on a long branch if ever needed.

## Local runs

| Environment | Command |
|-------------|---------|
| Linux / macOS VPS | `./start.sh` (see repo root) |
| Windows | `.\run_local.ps1` from repo root (venv `python`, picks a free port if 8501 is busy) |
| Expert workbench only | `streamlit run app_streamlit/expert_workbench.py` |

## CI

GitHub Actions (e.g. `.github/workflows/test.yml`) should stay green before merging risky changes. Run `pytest`, `ruff`, and targeted `mypy` locally as in [`CONTRIBUTING.md`](CONTRIBUTING.md).
