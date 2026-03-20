import numpy as np
import pandas as pd

from phinance.meta.search import run_meta_search
from phinance.meta.vault_integration import load_vault


def _synthetic_ohlcv(n: int = 100) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    close = 100 + np.cumsum(np.random.default_rng(21).normal(0, 1.0, size=n))
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.linspace(900, 1800, n),
        },
        index=idx,
    )


def test_meta_discovery_writes_vault(tmp_path):
    df = _synthetic_ohlcv()
    cfg = tmp_path / "meta.yaml"
    cfg.write_text(
        """
gp:
  population_size: 6
  generations: 1
  top_k: 2
  random_seed: 5
""".strip(),
        encoding="utf-8",
    )
    vault = tmp_path / "vault.json"

    result = run_meta_search(df, config_path=cfg, vault_path=vault)

    assert len(result["best_strategies"]) > 0
    payload = load_vault(vault)
    assert len(payload.get("strategies", [])) > 0
