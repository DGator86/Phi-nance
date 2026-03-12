# Fix: "untracked working tree files would be overwritten by merge: app.py"

Your Codespaces workspace has a local **app.py** that would be overwritten by `git pull`. Do the following.

## Step 1: Remove or move app.py so the pull can run

If you already have a backup, remove the current `app.py`:

```bash
rm app.py
```

If you haven’t backed it up yet, move it instead:

```bash
mv app.py app.py.codespaces-backup
```

(If you see "No such file or directory", then `app.py` is already gone; go to Step 2.)

## Step 2: Pull from Main

```bash
git pull origin Main
```

This should complete and bring in **app_v2.py** and the rest of the repo.

## Step 3: Run Strategy Lab v2

```bash
python3 -m streamlit run app_v2.py
```

---

## If pull still says "would be overwritten"

Make sure no `app.py` is left:

```bash
ls -la app.py
```

If it exists, run:

```bash
rm -f app.py
git pull origin Main
```

---

## After a successful pull

- **Strategy Lab v2:** `python3 -m streamlit run app_v2.py`
- **Strategy Lab (simple):** `python3 -m streamlit run app.py`
- **Dashboard:** `python3 -m streamlit run dashboard.py`

Always use `python3 -m streamlit run <file>`, not `python app.py`.
