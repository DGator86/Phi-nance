# Options Module Phase 1

Phase 1 introduces core pricing and data primitives for options work in `phi`.

## Included models

- **Black-Scholes** (`phi.options.models.black_scholes`)
  - `price_european(option_type, S, K, T, r, sigma)`
  - `greeks(...)` returns `delta, gamma, theta, vega, rho`
- **Binomial Tree (CRR)** (`phi.options.models.binomial_tree`)
  - `price_american(...)`
  - `price_european(...)` (for convergence checks)

## Greeks wrapper

Use `phi.options.greeks.get_greeks(...)` for unified model access.

```python
from phi.options.greeks import get_greeks

g = get_greeks("call", S=100, K=100, T=1.0, r=0.05, sigma=0.2, model="bs")
```

## Options chain cache

`phi.data.cache` now supports options chain pulls with yfinance:

```python
from phi.data.cache import fetch_options_chain, get_cached_options

chain = fetch_options_chain("AAPL")                 # nearest expiry
chain = fetch_options_chain("AAPL", "2026-06-19") # specific expiry
cached = get_cached_options("AAPL", "2026-06-19")
```

Cache location:

`data_cache/options/{SYMBOL}/{EXPIRATION}/chain.parquet`
