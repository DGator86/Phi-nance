"""Walk-forward and paper-trading validation. Ablation: engine earns >0.02 AUC."""

from phinence.validation.ablations import ablation_threshold_met
from phinence.validation.backtest_runner import make_synthetic_bars, run_backtest_fold
from phinence.validation.walk_forward import WalkForwardHarness

__all__ = [
    "WalkForwardHarness",
    "ablation_threshold_met",
    "run_backtest_fold",
    "make_synthetic_bars",
]
