"""
Ablations: engine earns complexity only if mean OOS AUC improves by >0.02 (2 pp)
or materially improves cone calibration without hurting AUC.
"""

from __future__ import annotations

ABLATION_AUC_THRESHOLD = 0.02  # 2 percentage points


def ablation_threshold_met(
    baseline_auc: float,
    with_engine_auc: float,
    baseline_cone_75: float | None = None,
    with_engine_cone_75: float | None = None,
) -> bool:
    """
    Engine stays if:
    - (with_engine_auc - baseline_auc) >= 0.02, or
    - cone calibration materially better and with_engine_auc >= baseline_auc.
    """
    if with_engine_auc - baseline_auc >= ABLATION_AUC_THRESHOLD:
        return True
    if baseline_cone_75 is not None and with_engine_cone_75 is not None:
        # Materially better: e.g. 75% cone coverage closer to 75%
        if with_engine_auc >= baseline_auc and abs(with_engine_cone_75 - 0.75) < abs(baseline_cone_75 - 0.75):
            return True
    return False
