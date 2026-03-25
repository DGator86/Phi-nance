"""
phinance.blending — Signal blending engine.

Canonical implementation: ``phi.blending`` (phi/blending/).
This module re-exports ``blend_signals`` and ``BLEND_METHODS`` from there so
all callers share a single implementation regardless of which import path they
use.  The local ``methods``, ``weights``, and ``regime_detector`` sub-modules
are preserved for backward compatibility and internal use.

Public API
----------
    from phinance.blending import blend_signals, BLEND_METHODS
"""

from phi.blending import ALLOWED_METHODS, blend_signals

# Backward-compatible alias used by phinance internals and older callers.
BLEND_METHODS = ALLOWED_METHODS

__all__ = ["blend_signals", "BLEND_METHODS", "ALLOWED_METHODS"]
