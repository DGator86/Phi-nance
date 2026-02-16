# Helpful GitHub Repos for Phi-nance

Use **environment variables** for all API keys (e.g. `TRADIER_ACCESS_TOKEN`, `POLYGON_API_KEY`). Never commit keys. See `.env.example` for required vars.

## Ported into this repo

- **proshotv2/gamma-vanna-options-exposure:** GEX/Vanna math in `src/phinence/engines/gex_math.py` (GEX = Γ×OI×100×Spot², VEX = Vanna×OI×100×Spot; vanna formula from README).
- **cablehead/python-tradier:** Quotes, expirations, chains(symbol, expiration) in `data/providers/tradier.py`; `get_quote()`, `options_expirations()`, `fetch_chain_snapshot(..., expiration=)`.
- **polygon-io/client-python (massive-com):** Optional `fetch_1m_bars_massive()` and `backfill_ticker_year(..., use_massive_client=True)` when `pip install massive` (or `pip install phinence[massive]`).

---

## Tradier (live bars + options chain)

| Repo | Description |
|------|-------------|
| [cablehead/python-tradier](https://github.com/cablehead/python-tradier) | Python client for Tradier API (market data, options). |
| [phileaton/tradier-python](https://github.com/phileaton/tradier-python) | `pip install tradier-python`; profile, accounts, market data. |
| [MySybil/tradier-options-plotter](https://github.com/MySybil/tradier-options-plotter) | CLI for historic options data + IV from Tradier Sandbox. |

**Official:** [Tradier Developer Hub](https://documentation.tradier.com/) — options chains at `/v1/markets/options/chains`.

---

## Polygon / Massive (historical 1m bars)

| Repo | Description |
|------|-------------|
| [polygon-io/client-python](https://github.com/polygon-io/client-python) | Official Polygon Python client (Polygon → Massive rebrand). |
| [MarketMakerLite/polygon](https://github.com/MarketMakerLite/polygon) | Historical data scripts, e.g. `historical_data/advanced.py`. |
| [TonyMiri/Polygon_database_builder](https://github.com/TonyMiri/Polygon_database_builder) | Build DB of historical prices including 1m bars. |
| [shinathan/polygon.io-stock-database](https://github.com/shinathan/polygon.io-stock-database) | Jupyter notebooks for 1m/5m/daily stock DB. |

**Note:** Polygon.io is now [Massive](https://www.massive.com/). Your “Massive.com” key is likely for this.

---

## GEX / dealer flow (Hedge Engine)

| Repo | Description |
|------|-------------|
| [proshotv2/gamma-vanna-options-exposure](https://github.com/proshotv2/gamma-vanna-options-exposure) | GEX/VEX from option chains; **Tradier** data source. |
| [hayden4r4/GEX-python](https://github.com/hayden4r4/GEX-python) | GEX from TDA option chains: Γ × OI × 100 (calls) / -100 (puts). |
| [Matteo-Ferrara/gex-tracker](https://github.com/Matteo-Ferrara/gex-tracker) | Dealer gamma exposure tracker. |
| [GMestreM/gex_data](https://github.com/GMestreM/gex_data) | SPX GEX from CBOE data. |

---

## Walk-forward / backtest

| Repo | Description |
|------|-------------|
| [kernc/backtesting.py](https://github.com/kernc/backtesting.py) | Popular backtesting framework; metrics, vectorized. |
| [MitchMedeiros/walk-forward-optimization-app](https://github.com/MitchMedeiros/walk-forward-optimization-app) | Walk-forward optimization (Dash + TA-Lib + vectorbt). |
| [JensGBG/backtest-module](https://github.com/JensGBG/backtest-module) | Backtest + walk-forward + Monte Carlo permutation. |
| [CornellQuantFund/CQF_Backtesting](https://github.com/CornellQuantFund/CQF_Backtesting) | Backtest engine with costs and benchmarks. |

---

## Suggested use in this repo

- **Tradier:** Keep current `data/providers/tradier.py`; optionally refactor to use `python-tradier` or `tradier-python` for auth and endpoints.
- **Polygon/Massive:** Swap or supplement `data/providers/polygon_backfill.py` with `polygon-io/client-python` (or Massive’s client) for aggregates.
- **GEX:** Use **proshotv2/gamma-vanna-options-exposure** (Tradier-sourced) as reference for EOD GEX in `src/phinence/engines/hedge.py`.
- **Walk-forward:** Use **MitchMedeiros/walk-forward-optimization-app** or **JensGBG/backtest-module** as patterns for `src/phinence/validation/walk_forward.py`.
