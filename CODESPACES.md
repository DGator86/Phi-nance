# Running the app on Codespaces

## Do not run with `python app.py`

Streamlit apps **must** be started with the `streamlit run` command. If you run `python app.py` or `python dashboard.py`, you will only get warnings and no browser app.

**Correct command:**
```bash
python3 -m streamlit run app.py
```
or
```bash
python3 -m streamlit run app_v2.py
```
or
```bash
python3 -m streamlit run dashboard.py
```

## If `app_v2.py` is missing

Your Codespaces clone may not have the latest files. Try:

```bash
git pull origin Main
# or
git pull origin MAIN
```

Then run:
```bash
python3 -m streamlit run app_v2.py
```

If `app_v2.py` still doesn’t exist, use one of the apps that are present:

```bash
python3 -m streamlit run app.py
# or
python3 -m streamlit run dashboard.py
```

## One-command launcher

From the repo root:

```bash
bash run_streamlit.sh
```

This runs the first available app: `app_v2.py` → `app.py` → `dashboard.py` → `scripts/app_v2.py`.

## Open in browser

After Streamlit starts, open the URL it prints (e.g. http://localhost:8501). On Codespaces, use “Ports” and “Forward Port” for 8501 if needed.
