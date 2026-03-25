"""CLI: train regime detector from YAML + optional MLflow logging."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from phi.data.unified_data import get_ohlcv
from phi.logging import get_logger
from phi.regime.artifacts import write_regime_training_manifest
from phi.regime.evaluation import summarize_regime_metrics
from phi.regime.explain import feature_regime_correlation_proxy, try_shap_random_forest_summary
from phi.regime.train import train_regime_detector
from phi.regime.yaml_config import load_regime_training_yaml

logger = get_logger(__name__)


def _maybe_mlflow_log(
    cfg: dict[str, Any],
    params: dict[str, Any],
    metrics: dict[str, float],
    model_path: Path,
) -> None:
    block = cfg.get("mlflow") or {}
    if not block.get("enabled"):
        return
    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError("mlflow is not installed; pip install mlflow or disable mlflow.enabled") from exc

    uri = str(block.get("tracking_uri", "./mlruns"))
    exp = str(block.get("experiment_name", "phi_regime"))
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(exp)
    run_name = block.get("run_name")
    with mlflow.start_run(run_name=run_name):
        for k, v in params.items():
            if v is None:
                continue
            mlflow.log_param(k, v)
        for k, v in metrics.items():
            mlflow.log_metric(k, float(v))
        mlflow.log_artifact(str(model_path))
        man = model_path.with_suffix(".manifest.json")
        if man.exists():
            mlflow.log_artifact(str(man))


def run_training_from_config_dict(cfg: dict[str, Any]) -> dict[str, Any]:
    """Execute training from a loaded config mapping (for tests / API reuse)."""
    data = cfg.get("data") or {}
    model = cfg.get("model") or {}
    out = cfg.get("output") or {}

    symbol = str(data.get("symbol", "SPY")).upper()
    start = str(data["start"])
    end = str(data["end"])
    timeframe = str(data.get("timeframe", "1D"))
    vendor = str(data.get("vendor", "yfinance"))

    method = str(model.get("method", "hmm"))
    n_regimes = int(model.get("n_regimes", 3))
    window = int(model.get("window", 20))
    fit_params = dict(model.get("fit_params") or {})

    df = get_ohlcv(symbol, start, end, timeframe, vendor)

    save_path = out.get("save_path")
    if save_path:
        save_path = Path(save_path)
    detector, path = train_regime_detector(
        df,
        method=method,
        n_regimes=n_regimes,
        window=window,
        save=True,
        save_path=save_path,
        **fit_params,
    )
    if path is None:
        raise RuntimeError("Expected saved model path")

    preds = detector.predict(df)
    metrics = summarize_regime_metrics(df, preds)
    explain_corr = feature_regime_correlation_proxy(df, preds, window=window)
    metrics["explain_top_feature_corr"] = float(max(explain_corr.values())) if explain_corr else 0.0

    training_config = {
        "data": {"symbol": symbol, "start": start, "end": end, "timeframe": timeframe, "vendor": vendor},
        "model": {"method": method, "n_regimes": n_regimes, "window": window, "fit_params": fit_params},
        "output": {"save_path": str(path)},
    }

    extra: dict[str, Any] = {"feature_regime_correlation": dict(list(explain_corr.items())[:15])}
    if (cfg.get("explain") or {}).get("shap_surrogate"):
        shap_summary = try_shap_random_forest_summary(df, preds, window=window)
        if shap_summary:
            extra["shap"] = shap_summary

    write_regime_training_manifest(
        path,
        training_config=training_config,
        metrics=metrics,
        extra=extra,
    )

    params_flat: dict[str, Any] = {
        "symbol": symbol,
        "method": method,
        "n_regimes": n_regimes,
        "window": window,
        "start": start,
        "end": end,
        "vendor": vendor,
    }
    _maybe_mlflow_log(cfg, params_flat, metrics, path)

    logger.info("Saved regime model to %s metrics=%s", path, metrics)
    return {"model_path": str(path), "metrics": metrics, "training_config": training_config}


def main() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(Path.cwd() / ".env")
    except ImportError:
        pass

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        default=os.environ.get("PHINANCE_REGIME_TRAIN_CONFIG", ""),
        help="YAML config path (default: env PHINANCE_REGIME_TRAIN_CONFIG)",
    )
    p.add_argument("--dry-run", action="store_true", help="Parse config and print resolved settings only")
    args = p.parse_args()
    if not args.config:
        p.error("Pass --config or set PHINANCE_REGIME_TRAIN_CONFIG")
    cfg_path = Path(args.config)
    cfg = load_regime_training_yaml(cfg_path)
    if args.dry_run:
        print(json.dumps(cfg, indent=2, default=str))
        return
    result = run_training_from_config_dict(cfg)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
