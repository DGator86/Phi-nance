"""
Live Scanner — Combines AlphaVantageFetcher + RegimeEngine for real-ticker scanning.

Workflow
--------
1.  Fetch 1-minute OHLCV from Alpha Vantage for each ticker.
2.  Run RegimeEngine.run_latest() on each.
3.  Return ranked DataFrame with regime probabilities + composite score.

Designed for:
  - Morning pre-market scan (fetch full day, rank by score)
  - Intraday re-scan (compact mode, rolling update every N minutes)
  - Backtestable via inject_data() for replay

Usage
-----
>>> from regime_engine.live_scanner import LiveScanner
>>> scanner = LiveScanner()
>>> results = scanner.scan(['AAPL', 'MSFT', 'NVDA', 'TSLA', 'SPY'])
>>> print(results.head())

# Re-scan using only the latest N bars (fast intraday mode)
>>> results = scanner.scan(['AAPL', 'MSFT'], outputsize='compact')

# Single ticker with full output
>>> full = scanner.run_ticker('AAPL')
>>> print(full['regime_probs'].tail())
>>> print(full['mix'].tail())
"""

from __future__ import annotations

import logging
import pathlib
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np

from .data_fetcher import AlphaVantageFetcher, YFinanceIntradayFetcher
from .scanner import RegimeEngine, UniverseScanner, load_config, TickerResult
from .species import REGIME_BINS

logger = logging.getLogger(__name__)


class LiveScanner:
    """
    End-to-end live scanner: Alpha Vantage → Regime Engine → Ranked output.

    Parameters
    ----------
    api_key       : Alpha Vantage API key (defaults to built-in key)
    config_path   : path to config.yaml (default: regime_engine/config.yaml)
    cache_dir     : local cache directory for AV responses
    rate_limit    : minimum seconds between AV API calls
    min_bars      : minimum bars required to run engine on a ticker

    Examples
    --------
    >>> ls = LiveScanner()
    >>> df = ls.scan(['AAPL', 'MSFT', 'NVDA'])
    >>> trending = ls.filter_regime(['AAPL','MSFT','NVDA'], regime='TREND_UP', min_prob=0.25)
    """

    def __init__(
        self,
        api_key:      Optional[str] = None,
        config_path:  Optional[str | pathlib.Path] = None,
        cache_dir:    str   = ".av_cache",
        rate_limit:   float = 12.0,
        min_bars:     int   = 300,
        fetcher_type: str   = "alphavantage",
    ) -> None:
        """
        Parameters
        ----------
        fetcher_type : 'alphavantage' (default) | 'yfinance'
            Use 'yfinance' for no-API-key, no-rate-limit intraday scanning
            (10-min to 10-hour signal horizon).  Use 'alphavantage' for full
            historical depth (30 days of 1-min bars) when an API key is set.
        """
        if fetcher_type == "yfinance":
            self.fetcher: Any = YFinanceIntradayFetcher()
        else:
            self.fetcher = AlphaVantageFetcher(
                api_key=api_key,
                cache_dir=cache_dir,
                rate_limit=rate_limit,
            )
        self.cfg      = load_config(config_path)
        self.engine   = RegimeEngine(self.cfg)
        self.scanner  = UniverseScanner(config=self.cfg)
        self.min_bars = min_bars

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scan(
        self,
        symbols:    List[str],
        interval:   str  = "1min",
        outputsize: str  = "full",
        adjusted:   bool = True,
        extended:   bool = False,
        sort_by:    str  = "score",
        ascending:  bool = False,
        top_n:      int  = 50,
    ) -> pd.DataFrame:
        """
        Fetch data for all symbols and run the regime engine.

        Parameters
        ----------
        symbols    : list of ticker symbols
        interval   : '1min', '5min', '15min', '30min', '60min'
        outputsize : 'compact' (100 bars) | 'full' (up to 30 days)
        adjusted   : use adjusted prices
        extended   : include pre/post-market hours
        sort_by    : column to rank by ('score', 'p_TREND_UP', …)
        ascending  : sort direction
        top_n      : return only top N rows

        Returns
        -------
        pd.DataFrame — ranked scanner output
        """
        logger.info("Fetching %d symbols (%s, %s)…", len(symbols), interval, outputsize)

        universe = self.fetcher.fetch_universe(
            symbols,
            interval=interval,
            outputsize=outputsize,
            adjusted=adjusted,
            extended=extended,
            cache_ttl=5 if outputsize == "compact" else 30,
        )

        logger.info("Running regime engine on %d symbols…", len(universe))
        results = self.scanner.scan(universe, sort_by=sort_by, ascending=ascending)

        return results.head(top_n) if top_n and len(results) > top_n else results

    def run_ticker(
        self,
        symbol:     str,
        interval:   str  = "1min",
        outputsize: str  = "full",
        adjusted:   bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch a single ticker and return full engine output (all time series).

        Returns
        -------
        Same dict as RegimeEngine.run():
          features, logits, node_log_probs, species_lp,
          regime_probs, signals, weights, projections, mix
        Plus key 'ohlcv' with the raw fetched data.
        """
        df = self.fetcher.intraday(
            symbol, interval=interval, outputsize=outputsize, adjusted=adjusted
        )
        if len(df) < self.min_bars:
            raise ValueError(
                f"{symbol}: only {len(df)} bars fetched, need {self.min_bars}. "
                f"Try outputsize='full' or a less recent date."
            )
        result = self.engine.run(df)
        result["ohlcv"] = df
        return result

    def run_ticker_month(
        self,
        symbol: str,
        month:  str,
        interval: str = "1min",
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch a full month of intraday data and run the engine.
        Useful for backtesting.

        Parameters
        ----------
        month : 'YYYY-MM'  e.g. '2025-01'
        """
        df = self.fetcher.intraday_month(symbol, month=month, interval=interval)
        result = self.engine.run(df)
        result["ohlcv"] = df
        return result

    def filter_regime(
        self,
        symbols:   List[str],
        regime:    str,
        min_prob:  float = 0.25,
        interval:  str   = "1min",
        outputsize: str  = "compact",
    ) -> pd.DataFrame:
        """
        Return only tickers where a specific regime is probable.

        Parameters
        ----------
        regime   : e.g. 'TREND_UP', 'RANGE', 'BREAKOUT_UP'
        min_prob : minimum probability for the regime
        """
        universe = self.fetcher.fetch_universe(
            symbols, interval=interval, outputsize=outputsize, cache_ttl=2
        )
        return self.scanner.scan_top_regime(universe, regime=regime, min_prob=min_prob)

    def latest_score(self, symbol: str) -> TickerResult:
        """
        Quick latest-bar score for a single symbol (compact fetch).
        """
        df = self.fetcher.intraday(symbol, interval="1min", outputsize="compact")
        result = self.engine.run_latest(df)
        result.ticker = symbol
        return result

    # ------------------------------------------------------------------
    # Backtesting helpers
    # ------------------------------------------------------------------

    def inject_data(self, universe: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Run engine on a pre-fetched universe (no API calls).
        Useful for backtesting with stored OHLCV data.
        """
        return self.scanner.scan(universe)

    def replay(
        self,
        ohlcv: pd.DataFrame,
        step: int = 1,
    ) -> pd.DataFrame:
        """
        Replay a single ticker's OHLCV bar-by-bar, running the engine on
        each expanding window.  Returns one row per bar from min_bars onward.

        Parameters
        ----------
        ohlcv : full OHLCV DataFrame
        step  : emit a row every `step` bars (1 = every bar, 5 = every 5th)

        Returns
        -------
        pd.DataFrame — one row per emitted bar with full engine output
        """
        rows = []
        for end in range(self.min_bars, len(ohlcv) + 1, step):
            window = ohlcv.iloc[:end]
            try:
                res = self.engine.run_latest(window)
                row: Dict[str, Any] = {
                    "timestamp":        window.index[-1],
                    "score":            res.score,
                    "composite_signal": res.composite_signal,
                    "c_field":          res.c_field,
                    "c_consensus":      res.c_consensus,
                    "c_liquidity":      res.c_liquidity,
                    "top_species":      res.top_species,
                }
                for rbin, prob in res.regime_probs.items():
                    row[f"p_{rbin}"] = prob
                rows.append(row)
            except Exception as e:
                logger.debug("Replay error at bar %d: %s", end, e)

        return pd.DataFrame(rows).set_index("timestamp") if rows else pd.DataFrame()
