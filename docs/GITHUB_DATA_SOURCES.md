# GitHub data sources (market data, short volume, etc.)

Use env vars for API keys; see `.env.example`. Below: repos that provide or help with market/short-volume data.

---

## FINRA short volume (Phase 2C)

| Repo | Description |
|------|-------------|
| [amor71/FINRAShortData](https://github.com/amor71/FINRAShortData) | `pip install finrashortdata`. Async fetch by offset or date range. GPL-3.0. |
| [shinjimisato/finra-daily-shortExempt-vol-notebook](https://github.com/shinjimisato/finra-daily-shortExempt-vol-notebook) | Colab: scrape FINRA CDN, analyze, visualize; anomaly detection. |
| [arthurwu1227/FINRA-shortsale-data](https://github.com/arthurwu1227/FINRA-shortsale-data) | Scripts for historical FINRA short sale data (e.g. Aug 2009–May 2023). |
| [babyblake3/Finra-API-Calls](https://github.com/babyblake3/Finra-API-Calls) | FINRA API usage examples. |

**This repo:** We use the **public CDN** (no key): `https://cdn.finra.org/equity/regsho/daily/CNMSshvol{YYYYMMDD}.txt`. See `data/providers/finra_short_volume.py`.

---

## Polygon / Massive (historical bars)

| Repo | Description |
|------|-------------|
| [polygon-io/client-python](https://github.com/polygon-io/client-python) (now [massive-com/client-python](https://github.com/massive-com/client-python)) | Official client. `list_aggs(ticker, multiplier=1, timespan="minute", ...)`. |
| [TonyMiri/Polygon_database_builder](https://github.com/TonyMiri/Polygon_database_builder) | Build DB of historical prices including 1m bars. |
| [shinathan/polygon.io-stock-database](https://github.com/shinathan/polygon.io-stock-database) | Jupyter notebooks for 1m/5m/daily stock DB. |

---

## Tradier (live bars + options)

| Repo | Description |
|------|-------------|
| [cablehead/python-tradier](https://github.com/cablehead/python-tradier) | Python client: quotes, expirations, chains(symbol, expiration). |
| [phileaton/tradier-python](https://github.com/phileaton/tradier-python) | `pip install tradier-python`; market data, profile. |
| [MySybil/tradier-options-plotter](https://github.com/MySybil/tradier-options-plotter) | Historic options + IV from Tradier Sandbox. |

---

## GEX / dealer flow

| Repo | Description |
|------|-------------|
| [proshotv2/gamma-vanna-options-exposure](https://github.com/proshotv2/gamma-vanna-options-exposure) | GEX/VEX from Tradier chain; formulas ported into this repo. |
| [hayden4r4/GEX-python](https://github.com/hayden4r4/GEX-python) | GEX from TDA option chains. |
| [Matteo-Ferrara/gex-tracker](https://github.com/Matteo-Ferrara/gex-tracker) | Dealer gamma exposure tracker. |

---

## Other market data (reference)

| Repo | Description |
|------|-------------|
| [yfinance](https://github.com/ranaroussi/yfinance) | Yahoo Finance API; free, delayed. |
| [alpaca-trade-api](https://github.com/alpacahq/alpaca-trade-api) | Alpaca brokerage/market data. |
| [alpha-vantage](https://github.com/RomelTorres/alpha_vantage) | Alpha Vantage API wrapper. |
