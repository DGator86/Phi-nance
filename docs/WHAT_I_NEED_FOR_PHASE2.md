# What I need from you to finish Phase 2

The **APIs are wired correctly**; they just need **valid keys in `.env`** (we never store keys in code or in chat). If you get 401/403, the key is invalid or expired.

---

## 1. Create `.env` from the example

```bash
# In repo root
copy .env.example .env
# Then edit .env and paste your keys (after rotating any you pasted in chat).
```

---

## 2. Required keys (exactly these names)

| Variable | Used for | Where to get it |
|----------|----------|------------------|
| **POLYGON_API_KEY** or **MASSIVE_API_KEY** | Historical 1m bars (2+ years) | [Polygon.io](https://polygon.io) or [Massive.com](https://massive.com) (same key works for both). |
| **TRADIER_ACCESS_TOKEN** | Live 1m bars + options chain | [Tradier Developer](https://developer.tradier.com); use sandbox token for testing. |

Use **one** of Polygon or Massive for history; Tradier for live.

---

## 3. Run Phase 2

From repo root:

```bash
# Full Phase 2: FINRA short volume + Polygon backfill + Tradier live bars
python -m scripts.run_phase2

# Only FINRA short volume (no keys needed)
python -m scripts.run_phase2 --short-volume-only

# Backfill specific tickers and years
python -m scripts.run_phase2 --tickers SPY QQQ AAPL --years 2

# If you use the official Massive client
python -m scripts.run_phase2 --use-massive
```

---

## 4. If something fails

- **"Backfill skipped: no Polygon/Massive API key"**  
  Add `POLYGON_API_KEY=...` (or `MASSIVE_API_KEY=...`) to `.env`.

- **"API key rejected (401/403)"**  
  Key is wrong or expired. Create a new key in the provider dashboard and set it in `.env`.

- **"Tradier key rejected"**  
  Set `TRADIER_ACCESS_TOKEN=...` in `.env` (sandbox or live).

- **FINRA "No data"**  
  Normal for weekend/holiday (no file for that date). Try a recent trading day or wait until 6 PM ET on a trading day.

---

## 5. After a successful run

- **data/bars/{TICKER}/*.parquet** — historical 1m bars (and live_YYYYMMDD.parquet if Tradier ran).
- **data/short_volume/YYYYMMDD.parquet** — FINRA short volume for that date.

Then run the no-gap check (e.g. from code or a small script using `phinence.store.parquet_store.check_no_gap_more_than_n_bars`) to satisfy Phase 2A “done when.”
