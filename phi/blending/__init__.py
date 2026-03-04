"""
Phi-nance Blending Module
=========================

Blend multiple indicator signals:
  - Weighted Sum
  - Regime-Weighted
  - Voting
  - PhiAI Chooses
"""

from .blender import blend_signals, BLEND_METHODS

__all__ = ["blend_signals", "BLEND_METHODS"]
