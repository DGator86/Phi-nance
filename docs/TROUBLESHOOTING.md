# Troubleshooting Guide

## If the GUI isn't opening:

### 1. Check if Streamlit is running
Look for output like:
```
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
```

### 2. Try the standalone app (no dependencies on phinence modules)
```bash
python3 -m streamlit run app.py
```

### 3. Check for port conflicts
If port 8501 is busy, use a different port:
```bash
python3 -m streamlit run app.py --server.port 8502
```

### 4. Check Python path
Make sure you're in the project root:
```bash
cd "c:\Users\Darrin Vogeli\OneDrive - Penetron\Desktop\Phi-nance"
python3 -m streamlit run app.py
```

### 5. Install missing dependencies
```bash
pip install streamlit backtesting openai pandas numpy
```

### 6. Check for errors
Run with verbose output:
```bash
python3 -m streamlit run app.py --logger.level=debug
```

### 7. Try dashboard.py instead
If app.py doesn't work, try the Lumibot dashboard:
```bash
python3 -m streamlit run dashboard.py
```

## Common Errors:

**"ModuleNotFoundError: No module named 'streamlit'"**
→ `pip install streamlit`

**"ModuleNotFoundError: No module named 'backtesting'"**
→ `pip install backtesting`

**"Address already in use"**
→ Kill the process using port 8501 or use `--server.port 8502`

**"FileNotFoundError: scripts/app.py"**
→ Use `app.py` in root instead: `python3 -m streamlit run app.py`

**Import errors from phinence modules**
→ Use the standalone `app.py` which doesn't require phinence modules
