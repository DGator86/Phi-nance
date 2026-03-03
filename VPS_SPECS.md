# Phi-nance — Recommended VPS Specs

Given the app’s features (Streamlit dashboard, ML training, backtests, regime engine, optional L2 feed), these specs are a good fit.

---
6:
7: ## Your current VPS (**165.245.142.100**)
8: - **Host**: DigitalOcean (ATL1)
9: - **Plan**: 8 GB Memory / 2 Intel vCPUs / 160 GB Disk
10: - **OS**: Ubuntu 24.04 (LTS) x64
11:
12: *This setup has excellent RAM (Recommended) and sufficient CPU for a smooth beta experience.*
13:
14: ---

## Workload summary

| Component | Load |
|-----------|------|
| **Streamlit dashboard** | 6 tabs; moderate CPU when running pipeline/backtests. |
| **Regime engine** | Pandas/NumPy over OHLCV (single symbol or small universe). |
| **ML (LightGBM, sklearn)** | CPU-bound, uses all cores (`n_jobs=-1`). Training on one symbol, ~10 years of daily data. |
| **Lumibot backtests** | One strategy at a time, Yahoo data; moderate CPU/memory. |
| **Data/cache** | Parquet OHLCV cache, JSON API cache, saved models. |
| **L2 feed (optional)** | One Polygon WebSocket; low CPU. |

No GPU required. Everything runs on CPU.

---

## Recommended tiers

### Minimum (beta / light use)

- **2 vCPU**
- **4 GB RAM**
- **40–80 GB SSD**
- Good for: dashboard, occasional backtest, single-symbol ML training. May swap under heavy training.

### Recommended (comfortable beta / daily use)

- **4 vCPU**
- **8 GB RAM**
- **80 GB SSD**
- Good for: ML training (LightGBM using 4 cores), multiple backtests, universe scans, no swapping under normal use.

### If you add more later

- **RL training** (stable-baselines3): 8 GB RAM is safer; 16 GB if you run large envs.
- **Many concurrent users**: scale RAM and consider more vCPUs.

---

## What to pick on each provider

| Provider | Minimum tier | Recommended tier |
|----------|----------------|------------------|
| **DigitalOcean** | Basic 2 vCPU / 4 GB (~$24/mo) or 1 vCPU / 2 GB (~$12) if very light | Basic 4 vCPU / 8 GB (~$48/mo) |
| **Vultr** | Cloud Compute 2 vCPU / 4 GB | Cloud Compute 4 vCPU / 8 GB |
| **Linode** | Shared 2 GB or 4 GB | Shared 8 GB (4 vCPU) |
| **Hetzner** | CX22 (2 vCPU, 4 GB) or CPX21 (4 vCPU, 8 GB) | CPX21 or CPX31 (4–8 vCPU, 8–16 GB) |

**OS:** Ubuntu 24.04 LTS x64 (as in `deploy_vps.sh`).

---

## Disk

- **20 GB** absolute minimum (OS + venv + app + small cache).
- **40–80 GB** recommended so `data/cache`, `logs`, `models`, and `.av_cache` can grow without worry.

SSD is preferred for snappier dashboard and faster ML/backtest runs.
