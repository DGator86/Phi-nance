# How to See the Updated GUI

## The app has been restarted with the new combined strategies feature!

### To see the updates:

1. **Hard refresh your browser** (this clears cached files):
   - **Windows/Linux**: Press `Ctrl + F5` or `Ctrl + Shift + R`
   - **Mac**: Press `Cmd + Shift + R`

2. **Or manually refresh**:
   - Go to `http://localhost:8501`
   - Click the refresh button in your browser

3. **If you still don't see it**:
   - Close the browser tab completely
   - Open a new tab and go to `http://localhost:8501`

### What you should see:

In the **left sidebar**, you should now see:

1. **"1. Pick your strategies"** (checkboxes)
2. **"2. Run mode"** ← NEW!
   - Radio buttons: "Compare individually" or "Combine strategies"
   - If you select "Combine strategies" and have 2+ strategies selected, you'll see:
   - **"Voting mode:"** dropdown ← NEW!
     - Majority Vote
     - Unanimous
     - Weighted Average
3. **"3. Symbol & dates"** (inputs)

### If it's still not showing:

1. Check the terminal/console where Streamlit is running for errors
2. Try stopping Streamlit completely:
   ```bash
   # Kill all Python processes
   taskkill /F /IM python.exe
   ```
3. Then restart:
   ```bash
   python3 -m streamlit run scripts/app.py
   ```

### Alternative: Use the standalone app

If `scripts/app.py` isn't working, try the standalone version:
```bash
python3 -m streamlit run app.py
```

This version has the same combined strategies feature but doesn't require the phinence modules.
