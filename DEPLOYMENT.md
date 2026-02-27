# Deployment Guide

Consolidated deployment reference for Phi-nance.

---

## Prerequisites

- Python 3.11+
- `pip install -r requirements.txt`
- API keys: Alpha Vantage (`AV_API_KEY`), optionally MarketDataApp (`MARKETDATAAPP_API_TOKEN`)

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

---

## Local Setup

```bash
git clone https://github.com/DGator86/Phi-nance
cd Phi-nance
pip install -r requirements.txt
python -m streamlit run app_streamlit/live_workbench.py
```

Open [http://localhost:8501](http://localhost:8501).

---

## VPS Deployment

Based on `VPS_SETUP.md`. Use the provided `deploy_vps.sh` script:

```bash
chmod +x deploy_vps.sh
./deploy_vps.sh
```

### Manual steps

```bash
# Install deps
sudo apt update && sudo apt install -y python3.11 python3-pip nginx

# Clone + install
git clone https://github.com/DGator86/Phi-nance /opt/phinance
cd /opt/phinance && pip install -r requirements.txt

# Configure env
cp .env.example .env
nano .env  # fill in API keys

# Run with systemd (see VPS_SETUP.md for service file)
sudo systemctl start phinance
sudo systemctl enable phinance
```

Nginx reverse proxy config: see `streamlit_nginx.conf`.

---

## Oracle Cloud Free Tier

Based on `ORACLE_DEPLOY.md` and `ORACLE_FREE_START_HERE.md`.

1. Create an **Always Free** Compute instance (ARM or x86, Ubuntu 22.04).
2. Open ports 8501 (Streamlit) and 80/443 (nginx) in the Security List.
3. SSH in and run:

```bash
sudo apt update && sudo apt install -y python3.11 python3-pip nginx git
git clone https://github.com/DGator86/Phi-nance /opt/phinance
cd /opt/phinance
pip install -r requirements.txt
cp .env.example .env && nano .env
./deploy_oracle.sh
```

See `ORACLE_FREE_START_HERE.md` for the full step-by-step Oracle walkthrough.

---

## Troubleshooting

### Missing API keys

```
Error: AV_API_KEY not set
```
→ Copy `.env.example` to `.env` and set your Alpha Vantage key.

### Port conflicts

```
OSError: [Errno 98] Address already in use
```
→ Change the Streamlit port: `streamlit run app_streamlit/live_workbench.py --server.port 8502`

### Memory errors (OOM)

- Reduce date range in the Dataset Builder.
- Use `1D` timeframe instead of intraday for large date ranges.
- On Oracle Free Tier (1 GB RAM), limit to 1–2 symbols at a time.

### yfinance timeout / rate limit

The data cache uses exponential backoff (max 3 retries). If you still hit rate limits, wait a few minutes and retry, or use Alpha Vantage with a paid API key.
