"""
phinance.blending — Signal blending engine.

Public API
----------
    from phinance.blending import blend_signals, BLEND_METHODS

Sub-modules
-----------
  methods          — Pure blend implementations (weighted_sum, voting, ...)
  weights          — Weight calculation and normalisation helpers
  regime_detector  — Lightweight market-regime classifier
  blender          — Orchestrator: blend_signals() public function
"""

from phinance.blending.blender import blend_signals
from phinance.blending.methods import BLEND_METHODS

__all__ = ["blend_signals", "BLEND_METHODS"]
