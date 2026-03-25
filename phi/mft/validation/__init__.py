"""Walk-forward and paper-trading validation. Ablation: engine earns >0.02 AUC."""

from phi.mft.validation.ablations import ablation_threshold_met
from phi.mft.validation.backtest_runner import make_synthetic_bars, run_backtest_fold
from phi.mft.validation.walk_forward import WalkForwardHarness

__all__ = [
    "WalkForwardHarness",
    "ablation_threshold_met",
    "run_backtest_fold",
    "make_synthetic_bars",
]
