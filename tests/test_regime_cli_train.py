"""Config-driven regime training (mocked OHLCV)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from phi.regime.cli_train import run_training_from_config_dict


def _tiny_ohlcv() -> pd.DataFrame:
    idx = pd.date_range("2022-01-01", periods=80, freq="D")
    c = pd.Series(range(100, 180), index=idx, dtype=float)
    return pd.DataFrame(
        {"open": c, "high": c + 1, "low": c - 1, "close": c, "volume": 1000.0},
        index=idx,
    )


def test_run_training_from_config_dict_writes_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "phi.regime.cli_train.get_ohlcv",
        lambda *_a, **_k: _tiny_ohlcv(),
    )
    cfg = {
        "data": {
            "symbol": "SPY",
            "start": "2022-01-01",
            "end": "2022-06-01",
            "timeframe": "1D",
            "vendor": "yfinance",
        },
        "model": {"method": "kmeans", "n_regimes": 2, "window": 10, "fit_params": {}},
        "output": {"save_path": str(tmp_path / "m.pkl")},
        "mlflow": {"enabled": False},
    }
    out = run_training_from_config_dict(cfg)
    p = Path(out["model_path"])
    assert p.is_file()
    assert p.with_suffix(".manifest.json").is_file()
    assert "regime_entropy_norm" in out["metrics"]
