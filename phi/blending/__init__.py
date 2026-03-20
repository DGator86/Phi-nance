"""Public API for signal blending."""

from phi.logging import get_logger

logger = get_logger(__name__)

from .blender import ALLOWED_METHODS, DEFAULT_REGIME_BOOSTS, blend_signals

# Backward-compatible alias.
BLEND_METHODS = ALLOWED_METHODS

__all__ = ["blend_signals", "ALLOWED_METHODS", "BLEND_METHODS", "DEFAULT_REGIME_BOOSTS"]
