"""Blending method registry."""

BLEND_METHODS = {
    "weighted_sum": "phi.blending.manual.WeightedSum",
    "voting": "phi.blending.manual.Voting",
    "regime_weighted": "phi.blending.manual.RegimeWeighted",
    "ai_driven": "phi.blending.ai.AIDrivenBlend",
    "phiai_chooses": "phi.blending.ai.AIDrivenBlend",
}
