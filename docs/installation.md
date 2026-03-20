# Installation

## Requirements

- Python 3.10+
- `pip`
- (Optional) `python3.12-venv` system package for creating virtual environments

## Local setup

```bash
git clone https://github.com/DGator86/Phi-nance.git
cd Phi-nance
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
cp .env.example .env
```

### Windows / Python 3.13

**Ray** is not part of the default `requirements.txt` because many platforms (notably **Windows with Python 3.13**) have no matching `ray` wheel on PyPI. The stack runs without it; distributed backtests and `RayEnvRunner` stay off until you install Ray on a supported environment. If your OS/Python has wheels, use:

```bash
pip install -r requirements-ray.txt
```

Editable install with Ray: `pip install -e ".[distributed]"`.

**Polygon + yfinance + Alpaca:** the default file pins `polygon-api-client>=1.16` so `websockets` is new enough for both Polygon and yfinance. The deprecated **`alpaca-trade-api`** package is **not** installed by default (it requires `websockets<11` and cannot coexist). Live Alpaca flows use **`alpaca-py`** (already in `requirements.txt`). The legacy `phi.live.broker.AlpacaBroker` path still expects `alpaca-trade-api`; install only if you accept conflicts: `pip install -r requirements-legacy-alpaca.txt` or `pip install -e ".[legacy-alpaca]"`.

## Verify installation

```bash
python scripts/engine_health.py
pytest
```

## Run app

```bash
streamlit run app_streamlit/main.py --server.headless true
```
