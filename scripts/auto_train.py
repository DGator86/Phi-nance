#!/usr/bin/env python3
"""Auto-training script for PhiAI models."""

from __future__ import annotations

import argparse
import logging
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from phi.backtest.direct import run_direct_backtest
from phi.config import settings
from phi.data.cache import fetch_and_cache
from phi.logging import get_logger
from phi.phiai.auto_tune import run_phiai_optimization, save_best_params
from phi.regime import create_detector_from_params
from phi.utils.validation import sanitize_ticker

logger = get_logger(__name__)

DEFAULT_INDICATORS: dict[str, dict[str, object]] = {
    "RSI": {"enabled": True, "auto_tune": True, "params": {}},
    "MACD": {"enabled": True, "auto_tune": True, "params": {}},
    "BBands": {"enabled": True, "auto_tune": True, "params": {}},
}
_MAXIMIZE_METRICS = {"sharpe", "roi", "cagr", "win_rate", "accuracy"}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for auto-training."""
    parser = argparse.ArgumentParser(description="Auto-train PhiAI models.")
    parser.add_argument("--tickers", nargs="+", required=True, help="Tickers to train on (space-separated)")
    parser.add_argument("--timeframe", default="1D", help="Data timeframe (e.g., 1D, 1H)")
    parser.add_argument("--years", type=int, default=3, help="Years of historical data to use")
    parser.add_argument("--end-date", help="End date in YYYY-MM-DD format (default: today)")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of Optuna trials")
    parser.add_argument("--windows", type=int, default=3, help="Number of walk-forward windows")
    parser.add_argument("--parallel", type=int, default=1, help="Number of parallel jobs")
    parser.add_argument("--metric", default="sharpe", help="Optimization metric (sharpe, roi, etc.)")
    parser.add_argument("--output-dir", default="./runs/best_params", help="Directory to save best parameters")
    parser.add_argument("--vendor", default="yfinance", help="Data vendor")
    parser.add_argument("--force-refresh", action="store_true", help="Force refresh data even if cached")
    parser.add_argument("--regime-optimize", action="store_true", help="Enable regime detector + boosts optimization")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--verbose", action="store_true", help="Increase log level to DEBUG")
    parser.add_argument("--deploy-live", action="store_true", help="Copy best config to LIVE_CONFIG_PATH")
    return parser.parse_args()


def _enabled_indicator_names(indicators: dict[str, dict[str, object]]) -> list[str]:
    return [name for name, cfg in indicators.items() if isinstance(cfg, dict) and cfg.get("enabled") is True]


def _to_regime_key(raw: object) -> str:
    text = str(raw)
    match = re.search(r"(\d+)$", text)
    return match.group(1) if match else text


def _walk_forward_slices(n_rows: int, windows: int) -> list[tuple[slice, slice]]:
    if windows <= 1 or n_rows < (windows + 1) * 10:
        return [(slice(0, n_rows), slice(0, n_rows))]
    fold_size = n_rows // (windows + 1)
    if fold_size <= 0:
        return [(slice(0, n_rows), slice(0, n_rows))]

    folds: list[tuple[slice, slice]] = []
    for window_idx in range(1, windows + 1):
        val_start = window_idx * fold_size
        val_stop = val_start + fold_size if window_idx < windows else n_rows
        if val_stop - val_start < 10:
            continue
        folds.append((slice(0, val_start), slice(val_start, val_stop)))
    return folds or [(slice(0, n_rows), slice(0, n_rows))]


def _suggest_regime_trial_config(
    trial: optuna.Trial,
    indicators: list[str],
    *,
    seed: int,
) -> dict[str, Any]:
    detector_type = trial.suggest_categorical("regime__detector_type", ["hmm", "kmeans", "gmm"])
    n_regimes = trial.suggest_int("regime__n_regimes", 2, 5)
    feature_window = trial.suggest_int("regime__feature_window", 10, 50)
    params: dict[str, Any] = {
        "n_regimes": n_regimes,
        "feature_window": feature_window,
        "random_state": seed,
    }
    if detector_type == "hmm":
        params["covariance_type"] = trial.suggest_categorical("regime__covariance_type", ["full", "diag"])

    boosts: dict[str, dict[str, float]] = {}
    for regime_idx in range(n_regimes):
        regime_key = str(regime_idx)
        boosts[regime_key] = {}
        for indicator in indicators:
            boosts[regime_key][indicator] = trial.suggest_float(
                f"boost__r{regime_idx}__{indicator}",
                0.5,
                2.0,
            )

    return {
        "regime_detector": {"type": detector_type, "params": params},
        "regime_boosts": boosts,
    }


def run_regime_optimization(
    ohlcv: pd.DataFrame,
    indicators_config: dict[str, dict[str, Any]],
    *,
    n_trials: int,
    windows: int,
    parallel_jobs: int,
    metric: str,
    seed: int,
) -> dict[str, Any]:
    """Optimize indicators plus regime detector and per-regime boosts."""
    base = run_phiai_optimization(
        ohlcv=ohlcv,
        indicators_config=indicators_config,
        n_trials=n_trials,
        walk_forward_windows=windows,
        parallel_jobs=parallel_jobs,
        metric=metric,
        seed=seed,
    )
    indicator_params = base.get("best_params", {})
    tuned_indicators = {name: dict(cfg) for name, cfg in indicators_config.items()}
    for name, params in indicator_params.items():
        if name in tuned_indicators:
            tuned_indicators[name]["params"] = dict(params)
            tuned_indicators[name]["enabled"] = True

    enabled_names = _enabled_indicator_names(tuned_indicators)
    direction = "maximize" if metric in _MAXIMIZE_METRICS else "minimize"
    folds = _walk_forward_slices(len(ohlcv), windows)

    def objective(trial: optuna.Trial) -> float:
        regime_cfg = _suggest_regime_trial_config(trial, enabled_names, seed=seed)
        detector_type = str(regime_cfg["regime_detector"]["type"])
        params = dict(regime_cfg["regime_detector"]["params"])
        feature_window = int(params["feature_window"])
        boosts = dict(regime_cfg["regime_boosts"])

        fold_scores: list[float] = []
        for train_slice, val_slice in folds:
            train = ohlcv.iloc[train_slice]
            validation = ohlcv.iloc[val_slice]
            if train.empty or validation.empty:
                continue
            detector = create_detector_from_params(detector_type, params)
            detector.fit(train, window=feature_window)
            regimes = detector.predict(validation)
            label_map = {label: _to_regime_key(label) for label in regimes.astype(str).unique()}

            try:
                results, _ = run_direct_backtest(
                    ohlcv=validation,
                    symbol="OPT",
                    indicators=tuned_indicators,
                    blend_weights={name: 1.0 / len(enabled_names) for name in enabled_names} if enabled_names else {},
                    blend_method="regime_weighted",
                    regime_series=regimes,
                    regime_label_map=label_map,
                    regime_boosts=boosts,
                )
            except Exception:
                logger.debug("Skipping fold due to backtest error", exc_info=True)
                continue
            fold_scores.append(float(results.get(metric, -1e9)))

        score = float(np.mean(fold_scores)) if fold_scores else -1e9
        trial.set_user_attr("regime_detector", regime_cfg["regime_detector"])
        trial.set_user_attr("regime_boosts", boosts)
        return score

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction=direction, sampler=TPESampler(seed=seed))
    study.optimize(objective, n_trials=max(1, n_trials), n_jobs=max(1, parallel_jobs))

    return {
        "best_params": indicator_params,
        "best_value": float(study.best_value),
        "optimized_indicators": tuned_indicators,
        "regime_detector": study.best_trial.user_attrs.get("regime_detector", {}),
        "regime_boosts": study.best_trial.user_attrs.get("regime_boosts", {}),
        "study": study,
        "explanation": "Regime-aware optimization completed.",
    }




def _deploy_live_config(payload: dict[str, Any]) -> None:
    settings.LIVE_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    settings.LIVE_CONFIG_PATH.write_text(__import__("json").dumps(payload, indent=2), encoding="utf-8")
    logger.info("Deployed live config to %s", settings.LIVE_CONFIG_PATH)


def main() -> None:
    """Run auto-training for each requested ticker."""
    args = parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug("Verbose mode enabled")

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    end_date = datetime.now().date() if not args.end_date else datetime.strptime(args.end_date, "%Y-%m-%d").date()
    start_date = end_date - timedelta(days=365 * args.years)

    logger.info("Starting auto-training for %s from %s to %s", args.tickers, start_date, end_date)

    processed = 0
    saved = 0
    skipped = 0

    for raw_ticker in args.tickers:
        processed += 1
        try:
            ticker = sanitize_ticker(raw_ticker)
            logger.info("Processing %s", ticker)

            data = fetch_and_cache(
                vendor=args.vendor,
                symbol=ticker,
                timeframe=args.timeframe,
                start=start_date.isoformat(),
                end=end_date.isoformat(),
                force_refresh=args.force_refresh,
            )
            if data.empty:
                skipped += 1
                logger.warning("No data for %s, skipping", ticker)
                continue

            if args.regime_optimize:
                result = run_regime_optimization(
                    ohlcv=data,
                    indicators_config=DEFAULT_INDICATORS,
                    n_trials=args.n_trials,
                    windows=args.windows,
                    parallel_jobs=args.parallel,
                    metric=args.metric,
                    seed=args.seed,
                )
            else:
                result = run_phiai_optimization(
                    ohlcv=data,
                    indicators_config=DEFAULT_INDICATORS,
                    n_trials=args.n_trials,
                    walk_forward_windows=args.windows,
                    parallel_jobs=args.parallel,
                    metric=args.metric,
                    seed=args.seed,
                )

            dataset_id = f"{ticker}_{args.timeframe}_{start_date}_{end_date}"
            if args.regime_optimize:
                payload = {
                    "dataset_id": dataset_id,
                    "metric": args.metric,
                    "best_value": float(result.get("best_value", 0.0)),
                    "indicators": result.get("best_params", {}),
                    "regime_detector": result.get("regime_detector", {}),
                    "regime_boosts": result.get("regime_boosts", {}),
                }
                (output_path / f"{dataset_id}.json").write_text(__import__("json").dumps(payload, indent=2), encoding="utf-8")
            else:
                save_best_params(
                    result["best_params"],
                    dataset_id=dataset_id,
                    metric=args.metric,
                    best_value=float(result.get("best_value", 0.0)),
                    output_dir=output_path,
                )

            saved += 1
            if args.deploy_live:
                deploy_payload = {
                    "dataset_id": dataset_id,
                    "initial_capital": 100000.0,
                    "allocation_strategy": "equal_weight",
                    "allocation_params": {},
                    "indicators": result.get("best_params", {}),
                    "metric": args.metric,
                }
                _deploy_live_config(deploy_payload)
            logger.info("Best params for %s: %s", ticker, result.get("explanation", "(no explanation)"))
        except Exception:
            skipped += 1
            logger.exception("Failed to process %s", raw_ticker)

    logger.info("Auto-training completed. processed=%d saved=%d skipped=%d", processed, saved, skipped)


if __name__ == "__main__":
    main()
