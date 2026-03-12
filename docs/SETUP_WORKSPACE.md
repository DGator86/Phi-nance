# Setup for /workspaces/Phi-nance

Your workspace has a different structure. Here's how to run the Strategy Lab:

## Step 1: Install dependencies

```bash
pip install streamlit backtesting openai
pip install pydantic pyarrow pandas numpy httpx python-dotenv
```

## Step 2: Run the app

```bash
python3 -m streamlit run app.py
```

**Important:** Use `python3 -m streamlit` instead of just `streamlit` (the command might not be in PATH).

The `app.py` file is now in the root of your workspace and will work with your structure.

## If you see import errors

The app tries to import from `phinence.gui.runner`. If that fails, it will use a fallback. To fix imports:

1. Make sure `src/phinence/` exists (or your code is in a structure Python can find).
2. The app adds the current directory and `src/` to `sys.path`, so if `src/phinence/gui/runner.py` exists, it should work.

## Alternative: Use your existing dashboard.py

If `dashboard.py` is already a Streamlit app, you can run:

```bash
streamlit run dashboard.py
```

And we can adapt that to add the Strategy Lab features.
