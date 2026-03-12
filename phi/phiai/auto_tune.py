"""PhiAI auto-tuning with Bayesian optimization and walk-forward validation."""

from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from optuna.samplers import NSGAIISampler, RandomSampler

from phi.config import settings
from phi.exceptions import OptimizationError
from phi.logging import get_logger
from phi.run_config import RunConfig

logger = get_logger(__name__)

_MAXIMIZE_METRICS = {"sharpe", "roi", "cagr", "win_rate", "accuracy"}


def tune_parameter_space(
    objective_func,
    param_space: dict[str, tuple[Any, Any] | list[Any]],
    method: str = "bayesian",
    n_trials: int = 50,
    direction: str = "minimize",
    seed: int = 42,
) -> tuple[dict[str, Any], float]:
    """Generic PhiAI optimization helper for arbitrary parameter spaces.

    ``param_space`` supports either:
      - (min, max) tuples for numeric parameters
      - list values for categorical parameters
    """

    if not param_space:
        raise OptimizationError("param_space must not be empty")

    method_key = str(method).strip().lower()
    if method_key in {"bayesian", "tpe"}:
        sampler = TPESampler(seed=seed)
    elif method_key in {"genetic", "nsga2"}:
        sampler = NSGAIISampler(seed=seed)
    elif method_key in {"random", "rand"}:
        sampler = RandomSampler(seed=seed)
    else:
        raise OptimizationError(f"Unsupported optimization method: {method}")

    def _suggest(trial: optuna.Trial) -> dict[str, Any]:
        params: dict[str, Any] = {}
        for name, spec in param_space.items():
            if isinstance(spec, tuple) and len(spec) == 2:
                low, high = spec
                if isinstance(low, int) and isinstance(high, int):
                    params[name] = trial.suggest_int(name, int(low), int(high))
                else:
                    params[name] = trial.suggest_float(name, float(low), float(high))
            elif isinstance(spec, list) and spec:
                params[name] = trial.suggest_categorical(name, spec)
            else:
                raise OptimizationError(f"Invalid search space for '{name}': {spec!r}")
        return params

    def wrapped_objective(trial: optuna.Trial) -> float:
        return float(objective_func(_suggest(trial)))

    study = optuna.create_study(direction=direction, sampler=sampler)
    study.optimize(wrapped_objective, n_trials=max(1, int(n_trials)))
    return dict(study.best_params), float(study.best_value)

_PARAM_GRIDS_DAILY: dict[str, dict[str, list[Any]]] = {
    "RSI": {"rsi_period": [7, 14, 21], "oversold": [25, 30, 35], "overbought": [65, 70, 75]},
    "MACD": {"fast_period": [8, 12, 16], "slow_period": [21, 26, 31], "signal_period": [7, 9, 11]},
    "Bollinger": {"bb_period": [15, 20, 25], "num_std": [1.5, 2.0, 2.5]},
    "Dual SMA": {"fast_period": [5, 10, 20], "slow_period": [30, 50, 100]},
    "Mean Reversion": {"sma_period": [10, 20, 40]},
    "Breakout": {"channel_period": [10, 20, 40]},
    "VWAP": {"band_pct": [0.2, 0.5, 1.0]},
}

_PARAM_GRIDS_INTRADAY: dict[str, dict[str, list[Any]]] = {
    "RSI": {"rsi_period": [3, 5, 7, 9, 14], "oversold": [25, 30, 35], "overbought": [65, 70, 75]},
    "MACD": {"fast_period": [3, 5, 8], "slow_period": [12, 17, 21], "signal_period": [3, 5, 7]},
    "Bollinger": {"bb_period": [10, 14, 20], "num_std": [1.5, 2.0, 2.5]},
    "Dual SMA": {"fast_period": [3, 5, 9], "slow_period": [12, 20, 30]},
    "Mean Reversion": {"sma_period": [5, 10, 15]},
    "Breakout": {"channel_period": [5, 10, 15]},
    "VWAP": {"band_pct": [0.2, 0.3, 0.5, 0.8, 1.0]},
}


class PhiAI:
    """PhiAI orchestrator for full auto mode."""

    def __init__(self, max_indicators: int = 5, allow_shorts: bool = False, risk_cap: float | None = None) -> None:
        self.max_indicators = max_indicators
        self.allow_shorts = allow_shorts
        self.risk_cap = risk_cap
        self.changes: list[dict[str, str]] = []

    def explain(self) -> str:
        lines = [
            f"PhiAI configuration: max_indicators={self.max_indicators}, "
            f"allow_shorts={self.allow_shorts}, risk_cap={self.risk_cap}",
        ]
        if not self.changes:
            lines.append("No adjustments were made.")
        else:
            lines.append(f"{len(self.changes)} adjustment(s):")
            for change in self.changes:
                lines.append(f"  • {change.get('what', 'unknown')}: {change.get('reason', '')}")
        return "\n".join(lines)


def _is_intraday(timeframe: str) -> bool:
    return timeframe in {"1m", "5m", "15m", "30m", "1H"}


def _grid_for(indicator_name: str, timeframe: str, config: dict[str, Any]) -> dict[str, list[Any]] | None:
    configured_grid = config.get("param_grid")
    if isinstance(configured_grid, dict) and configured_grid:
        return {k: list(v) for k, v in configured_grid.items() if isinstance(v, list) and v}
    grid_source = _PARAM_GRIDS_INTRADAY if _is_intraday(timeframe) else _PARAM_GRIDS_DAILY
    return grid_source.get(indicator_name)




def _infer_periods_per_year(index: pd.Index, default: float = 252.0) -> float:
    """Infer annualization periods from a DatetimeIndex frequency when possible."""
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 3:
        return default

    freq = index.freqstr or pd.infer_freq(index)
    if not freq:
        return default

    try:
        offset = pd.tseries.frequencies.to_offset(freq)
    except ValueError:
        return default

    nanos = getattr(offset, "nanos", 0)
    if nanos <= 0:
        return default

    minutes = nanos / (60 * 1e9)
    if minutes <= 0:
        return default

    trading_minutes_per_year = 252.0 * 6.5 * 60.0
    return max(1.0, trading_minutes_per_year / minutes)


def _metric_from_returns(strategy_returns: pd.Series, metric: str, periods_per_year: float | None = None) -> float:
    """Compute optimization metric from strategy returns."""
    clean = strategy_returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if clean.empty:
        return -1e9

    if metric == "roi":
        return float((1.0 + clean).prod() - 1.0)
    if metric == "sharpe":
        std = float(clean.std())
        if std <= 1e-12:
            return 0.0
        if periods_per_year is None:
            periods_per_year = 252.0
            warnings.warn(
                "Could not infer annualization periods for Sharpe; defaulting to 252.",
                RuntimeWarning,
                stacklevel=2,
            )
        return float((clean.mean() / std) * np.sqrt(periods_per_year))
    if metric == "max_drawdown":
        curve = (1.0 + clean).cumprod()
        drawdown = curve / curve.cummax() - 1.0
        return float(drawdown.min())
    return float((1.0 + clean).prod() - 1.0)


def _evaluate_indicator_set(
    ohlcv: pd.DataFrame,
    indicators_config: dict[str, dict[str, Any]],
    metric: str,
    periods_per_year: float | None = None,
) -> float:
    from phi.indicators.simple import compute_indicator

    close = ohlcv["close"].astype(float)
    returns = close.pct_change().fillna(0.0)
    signals: list[pd.Series] = []

    for name, cfg in indicators_config.items():
        if not cfg.get("enabled", False):
            continue
        params = cfg.get("params", {}) if isinstance(cfg.get("params", {}), dict) else {}
        signal = compute_indicator(name, ohlcv, params).reindex(ohlcv.index).fillna(0.0)
        signals.append(signal)

    if not signals:
        return -1e9

    composite_signal = pd.concat(signals, axis=1).mean(axis=1).clip(-1.0, 1.0)
    strategy_returns = composite_signal.shift(1).fillna(0.0) * returns
    return _metric_from_returns(strategy_returns, metric=metric, periods_per_year=periods_per_year)


def _walk_forward_score(
    ohlcv: pd.DataFrame,
    indicators_config: dict[str, dict[str, Any]],
    walk_forward_windows: int,
    metric: str,
) -> float:
    n_rows = len(ohlcv)
    periods_per_year = _infer_periods_per_year(ohlcv.index)
    if walk_forward_windows <= 1 or n_rows < (walk_forward_windows + 1) * 10:
        return _evaluate_indicator_set(
            ohlcv=ohlcv,
            indicators_config=indicators_config,
            metric=metric,
            periods_per_year=periods_per_year,
        )

    fold_size = n_rows // (walk_forward_windows + 1)
    if fold_size <= 0:
        return _evaluate_indicator_set(
            ohlcv=ohlcv,
            indicators_config=indicators_config,
            metric=metric,
            periods_per_year=periods_per_year,
        )

    fold_scores: list[float] = []
    for window_idx in range(1, walk_forward_windows + 1):
        start = window_idx * fold_size
        stop = start + fold_size if window_idx < walk_forward_windows else n_rows
        validation = ohlcv.iloc[start:stop]
        if len(validation) < 10:
            continue
        fold_scores.append(
            _evaluate_indicator_set(
                validation,
                indicators_config,
                metric=metric,
                periods_per_year=periods_per_year,
            )
        )

    if not fold_scores:
        return _evaluate_indicator_set(
            ohlcv=ohlcv,
            indicators_config=indicators_config,
            metric=metric,
            periods_per_year=periods_per_year,
        )

    return float(np.mean(fold_scores))


def _build_dataset_id(ohlcv: pd.DataFrame, run_config: RunConfig | None) -> str:
    if run_config and run_config.dataset_id:
        return run_config.dataset_id

    if ohlcv.empty:
        return "empty"
    base = f"{ohlcv.index[0]}|{ohlcv.index[-1]}|{len(ohlcv)}"
    digest = hashlib.sha1(base.encode("utf-8")).hexdigest()[:12]
    return f"dataset_{digest}"


def _best_params_dir() -> Path:
    root = settings.DATA_CACHE_DIR / "phiai_best_params"
    root.mkdir(parents=True, exist_ok=True)
    return root


def save_best_params(
    best_params: dict[str, dict[str, Any]],
    dataset_id: str,
    best_value: float,
    metric: str,
    output_dir: Path | str | None = None,
) -> Path:
    """Persist optimized indicator params and score for later reuse."""
    payload = {
        "dataset_id": dataset_id,
        "best_params": best_params,
        "best_value": float(best_value),
        "metric": metric,
    }
    out_root = Path(output_dir) if output_dir is not None else _best_params_dir()
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / f"{dataset_id}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Saved PhiAI best params to %s", out_path)
    return out_path


def load_best_params(dataset_id: str) -> dict[str, Any] | None:
    """Load previously saved optimized params for a dataset id."""
    path = _best_params_dir() / f"{dataset_id}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OptimizationError(f"Failed loading saved params for {dataset_id}: {exc}") from exc


def run_phiai_optimization(
    ohlcv: pd.DataFrame,
    indicators_config: dict[str, dict[str, Any]],
    run_config: RunConfig | None = None,
    n_trials: int | None = None,
    walk_forward_windows: int | None = None,
    parallel_jobs: int | None = None,
    metric: str = "sharpe",
    seed: int = 42,
) -> dict[str, Any]:
    """Run Bayesian optimization to tune indicator parameters."""
    if ohlcv is None or ohlcv.empty:
        return {
            "best_params": {},
            "best_value": -1e9,
            "optimized_indicators": indicators_config,
            "study": None,
            "explanation": "PhiAI skipped: empty OHLCV data.",
        }

    timeframe = run_config.timeframe if run_config else "1D"
    resolved_trials = int(
        n_trials
        if n_trials is not None
        else (run_config.phiai_n_trials if run_config and run_config.phiai_n_trials is not None else settings.PHIAI_DEFAULT_N_TRIALS)
    )
    resolved_windows = int(
        walk_forward_windows
        if walk_forward_windows is not None
        else (
            run_config.phiai_walk_forward_windows
            if run_config and run_config.phiai_walk_forward_windows is not None
            else settings.PHIAI_WALK_FORWARD_WINDOWS
        )
    )
    resolved_jobs = int(
        parallel_jobs
        if parallel_jobs is not None
        else (run_config.phiai_parallel_jobs if run_config and run_config.phiai_parallel_jobs is not None else settings.PHIAI_PARALLEL_JOBS)
    )

    tuned_names = [
        name
        for name, cfg in indicators_config.items()
        if cfg.get("enabled", False) and cfg.get("auto_tune", False) and _grid_for(name, timeframe, cfg)
    ]

    if not tuned_names:
        return {
            "best_params": {},
            "best_value": _walk_forward_score(ohlcv, indicators_config, max(1, resolved_windows), metric),
            "optimized_indicators": indicators_config,
            "study": None,
            "explanation": "PhiAI made no changes.",
        }

    logger.info(
        "Starting PhiAI optimization: trials=%s windows=%s n_jobs=%s metric=%s indicators=%s",
        resolved_trials,
        resolved_windows,
        resolved_jobs,
        metric,
        tuned_names,
    )

    direction = "maximize" if metric in _MAXIMIZE_METRICS else "minimize"

    def objective(trial: optuna.Trial) -> float:
        trial_indicators = {name: dict(cfg) for name, cfg in indicators_config.items()}
        best_for_trial: dict[str, dict[str, Any]] = {}

        for name in tuned_names:
            cfg = trial_indicators[name]
            params = dict(cfg.get("params", {}) or {})
            grid = _grid_for(name, timeframe, cfg) or {}
            for param_name, values in grid.items():
                params[param_name] = trial.suggest_categorical(f"{name}__{param_name}", values)
            cfg["params"] = params
            best_for_trial[name] = params
            trial_indicators[name] = cfg

        score = _walk_forward_score(
            ohlcv=ohlcv,
            indicators_config=trial_indicators,
            walk_forward_windows=max(1, resolved_windows),
            metric=metric,
        )
        trial.set_user_attr("best_params", best_for_trial)
        return score

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = TPESampler(seed=seed)
    study = optuna.create_study(direction=direction, sampler=sampler)
    try:
        study.optimize(objective, n_trials=max(1, resolved_trials), n_jobs=max(1, resolved_jobs))
    except (RuntimeError, ValueError) as exc:
        raise OptimizationError(f"PhiAI optimization failed: {exc}") from exc

    best_params: dict[str, dict[str, Any]] = study.best_trial.user_attrs.get("best_params", {})
    optimized_indicators = {name: dict(cfg) for name, cfg in indicators_config.items()}
    changes: list[str] = []

    for name, params in best_params.items():
        current = dict(optimized_indicators.get(name, {}))
        current["enabled"] = True
        current["auto_tune"] = False
        current["params"] = params
        optimized_indicators[name] = current
        changes.append(f"- {name} params: optimized via Optuna")

    dataset_id = _build_dataset_id(ohlcv=ohlcv, run_config=run_config)
    save_best_params(best_params=best_params, dataset_id=dataset_id, best_value=study.best_value, metric=metric)

    explanation = "PhiAI adjustments:\n" + "\n".join(changes) if changes else "PhiAI made no changes."
    logger.info("PhiAI optimization complete: best_value=%s dataset_id=%s", study.best_value, dataset_id)

    return {
        "best_params": best_params,
        "best_value": float(study.best_value),
        "optimized_indicators": optimized_indicators,
        "study": study,
        "dataset_id": dataset_id,
        "explanation": explanation,
    }
