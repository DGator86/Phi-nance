# Quick Start — If Setup Fails

If `pip install -e '.[gui]'` says "pyproject.toml not found" or `scripts/app.py` doesn't exist:

## 1. Check you're in the right folder

You need to be in the folder that contains:
- `pyproject.toml`
- `scripts/` folder  
- `src/` folder

**Check:** Run `ls` (Mac/Linux) or `dir` (Windows). You should see `pyproject.toml` listed.

If you don't see it, navigate to the correct folder:
```bash
cd /path/to/Phi-nance  # Use your actual path
```

## 2. Install dependencies directly (works without pyproject.toml)

If editable install fails, install packages directly:

```bash
pip install streamlit backtesting openai
pip install pydantic pyarrow pandas numpy httpx python-dotenv
```

Then run:
```bash
python3 -m streamlit run scripts/app.py
```

## 3. Check file locations

```bash
# Find pyproject.toml
find . -name "pyproject.toml" 2>/dev/null

# Find app.py  
find . -name "app.py" 2>/dev/null

# List current directory
ls -la
```

If files are in a different location, adjust paths accordingly.

## 4. Run app directly (no install needed)

If you're in the project root (where `scripts/` and `src/` are), Python can find modules without installing:

```bash
# Just install the GUI dependencies
pip install streamlit backtesting openai

# Run the app (Python will import from src/ automatically)
python3 -m streamlit run scripts/app.py
```

The app imports from `src/phinence/` which Python can find when you run from the project root.
