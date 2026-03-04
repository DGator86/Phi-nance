# Run Strategy Lab v2 (app_v2.py)

## Important: use `streamlit run`, not `python`

Do **not** run `python app.py` or `python app_v2.py`. Streamlit must be started with:

```bash
python3 -m streamlit run app_v2.py
```

## From repo root (e.g. `/workspaces/Phi-nance` or `...\Phi-nance`)

```bash
python3 -m streamlit run app_v2.py
```

**If you see "File does not exist: app_v2.py"** (e.g. on Codespaces):

1. Pull the latest code: `git pull origin Main`
2. Or use the launcher: `bash run_streamlit.sh` (tries app_v2.py, then app.py, then dashboard.py)
3. Or run an app that exists: `python3 -m streamlit run app.py` or `python3 -m streamlit run dashboard.py`

## If you see "Could not import phinence"

Install the project so the `phinence` package is available:

```bash
pip install -e ".[gui]"
```

Or from repo root:

```bash
pip install -e .
pip install streamlit backtesting openai
```

Then run again:

```bash
python3 -m streamlit run app_v2.py
```

## Optional: run from scripts/ (when that folder exists)

```bash
python3 -m streamlit run scripts/app_v2.py
```
