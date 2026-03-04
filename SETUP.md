# Set up Phi-nance (from scratch)

Do this **once** in the folder that contains `scripts`, `src`, and `pyproject.toml` (your Phi-nance project folder).

---

## What you need

- **Python 3.10 or newer**  
  - Check: open a terminal and run `python --version` or `python3 --version`.  
  - If you don’t have it: install from [python.org](https://www.python.org/downloads/). During setup, check **“Add Python to PATH”**.

---

## Step 1: Open a terminal in the project folder

- **Windows:** In File Explorer, go to the Phi-nance folder. Click the address bar, type `cmd` and press Enter (or right‑click → “Open in Terminal”).
- **Mac:** Open Terminal, then run `cd` and drag the Phi-nance folder onto the window, then press Enter.
- **Linux:** Open a terminal and run `cd /path/to/Phi-nance` (use your actual path).

You should see a prompt and the folder should be the “current directory.”

---

## Step 2: Install the project and GUI

Run **one** of these — **paste only one line** and press Enter:

```bash
pip install -e '.[gui]'
```

If that fails, try:

```bash
python3 -m pip install -e '.[gui]'
```

**Important:** Don’t paste the command twice. If you see “not a valid editable requirement”, run the command again as a single line.

Wait until it finishes without errors. That installs Phi-nance and the Strategy Lab (Streamlit + backtesting + optional OpenAI for the agent).

---

## Step 3: Start the Strategy Lab

Run (one line only):

```bash
streamlit run scripts/app.py
```

If you get “streamlit: command not found”, the install in Step 2 didn’t succeed. Run Step 2 again, then try:

```bash
python3 -m streamlit run scripts/app.py
```

- Your browser should open to **Phi-nance Strategy Lab**.
- If it doesn’t, look in the terminal for something like `Local URL: http://localhost:8501` and open that URL in your browser.

---

## Step 4: Use the app

1. In the **left sidebar**, check one or more strategies (e.g. **Buy & Hold**, **SMA Crossover**).
2. Leave **Symbol** as `SPY` (or change it).
3. Click **Run backtest**.
4. Look at the **Results** table and the **Best overall** line.
5. (Optional) Scroll down to **Ask the Agent** and type a question. For the agent to answer, you need an OpenAI API key in a `.env` file — see below.

---

## Optional: “Ask the Agent” (AI chat)

To use the chat that explains results and suggests strategies:

1. Get an API key from [OpenAI](https://platform.openai.com/api-keys) (account required).
2. In the Phi-nance folder, create a file named **`.env`** (same folder as `scripts` and `src`).
3. Put this in it (use your real key):

   ```
   OPENAI_API_KEY=sk-your-key-here
   ```

4. Restart the Strategy Lab (in the terminal press Ctrl+C, then run `streamlit run scripts/app.py` again).

If you skip this, the rest of the app still works; only the chat will say you need a key.

---

## If something goes wrong

| Problem | What to do |
|--------|------------|
| **“python” or “pip” not found** | Install Python from [python.org](https://www.python.org/downloads/) and tick “Add Python to PATH.” Then try `python -m pip install -e ".[gui]"` and `python -m streamlit run scripts/app.py`. |
| **“No module named 'streamlit'”** | Step 2 didn’t finish correctly. Run `pip install -e ".[gui]"` again from the Phi-nance folder. |
| **“No module named 'backtesting'”** | Same: run `pip install -e ".[gui]"` again. |
| **Backtest error in the app** | Make sure you’re in the project folder when you run `streamlit run scripts/app.py`. If it still fails, copy the full error message and use it when asking for help. |
| **Browser didn’t open** | In the terminal, find the line like `Local URL: http://localhost:8501` and open that link in your browser. |

---

## Windows: double‑click setup and run

- **First time:** Double‑click **`setup.bat`** in the Phi-nance folder. It installs everything. Wait until it says “Setup complete.”
- **Every time you want the app:** Double‑click **`run_strategy_lab.bat`**. It starts the Strategy Lab (and will try to update the install if needed).

---

## Summary

1. Open terminal in the Phi-nance folder.  
2. Run: `pip install -e '.[gui]'` (single line; single quotes).  
3. Run: `streamlit run scripts/app.py` (or `python3 -m streamlit run scripts/app.py`).  
4. Use the app in your browser.

That’s the full setup.
