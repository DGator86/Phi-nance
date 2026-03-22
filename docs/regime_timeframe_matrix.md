# Regime matrix: feature windows vs calendar timeframes

Phi-nance uses several **different** “regime” ideas. They complement each other; they are not duplicates.

## 1. Taxonomy matrix (`regime_engine/`)

The **28 species** in `regime_engine/species.py` are a fixed KPCOFGS-style lattice: each leaf has `base_regime` (TREND, RANGE, BREAKOUT, EXHAUST_REV) and optional `phylum_regime` (LOWVOL / HIGHVOL). The scanner combines features, taxonomy, and probabilities into **8 collapsed bins** plus species scores. This is the full **Market Field Theory** stack—not the same as a single k-means column.

## 2. Easy mode “three-window” stack (`train_multi_window_regimes`)

**SHORT / MEDIUM / LONG** are three **k-means** fits on the **same** OHLCV series with different **feature windows** (12 / 20 / 40 bars). That is an NPR-style **smoothing ladder**, not three calendar timeframes.

## 3. Timeframe regime matrix (`phi.regime.mtf_matrix`)

`build_regime_matrix()` **resamples** OHLCV to pandas offsets (`1min` … `1ME`), runs the same trainable detector on each resampled series, maps clusters to **TREND_DN / RANGE / TREND_UP**, then **forward-fills** each column to the base index.

**Important:** you only get columns for rules **not finer than your bar spacing**. Daily easy-mode data ⇒ typically **1D, 1W, 1ME** (and similar). To populate **1m–4H** columns you must pass **intraday** OHLCV (same API).

`confluence_score()` averages direction scores (+1 / 0 / -1) across columns with equal weights (custom weights optional in code).

## Programmatic usage

```python
from phi.regime.mtf_matrix import build_regime_matrix, confluence_score

matrix, meta = build_regime_matrix(ohlcv)
score = confluence_score(matrix)
# meta["skipped"] explains rules dropped (e.g. finer than data bars)
```

## UI

**Automatic backtest** (easy mode) runs both the three-window stack and the timeframe matrix after each analysis and shows the last bar + a small table + skipped-rule JSON.
