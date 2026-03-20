# Run the Strategy Lab (GUI)

Follow these steps **in order**. Use a terminal (Command Prompt, PowerShell, or Terminal app).

## Step 1: Open a terminal in the project folder

- **Windows:** Open File Explorer, go to the folder that contains `scripts` and `src` (your Phi-nance project). In the address bar type `cmd` and press Enter, or right‑click in the folder and choose "Open in Terminal" / "Open PowerShell window here".
- **Mac/Linux:** Open Terminal, then type `cd` followed by a space, drag the project folder onto the terminal window, and press Enter.

## Step 2: Install the GUI (one time)

Run this **exactly**:

```bash
pip install phi-nance[gui]
```

If that fails, try:

```bash
python -m pip install "phi-nance[gui]"
```

Wait until it finishes without errors.

## Step 3: Start the Strategy Lab

Run:

```bash
streamlit run scripts/app.py
```

- Your browser should open to **Phi-nance Strategy Lab**.
- If it doesn’t, look in the terminal for a line like `Local URL: http://localhost:8501` and open that address in your browser.

## If something doesn’t work

- **"pip is not recognized"**  
  Use: `python -m pip install "phi-nance[gui]"` and then `python -m streamlit run scripts/app.py`.

- **"No module named 'streamlit'"**  
  You didn’t install the GUI. Run Step 2 again: `pip install phi-nance[gui]`.

- **"No module named 'backtesting'"**  
  Same as above: install with `pip install phi-nance[gui]`.

- **Backtest fails with an error**  
  The Buy & Hold bug is fixed. If you still see an error, copy the full message from the terminal or the app and use it when asking for help.

- **Agent doesn’t answer**  
  The "Ask the Agent" feature needs an OpenAI API key. Create a file named `.env` in the project folder (same place as `scripts`) with one line:  
  `OPENAI_API_KEY=your-key-here`  
  Then restart the app (Ctrl+C in the terminal, then run `streamlit run scripts/app.py` again).
