"""Motion state = force / mass (expected-move-style tendencies)."""

from __future__ import annotations

from phi.force_field.schemas import ForceVector, MassState, MotionState


def derive_motion(force: ForceVector, mass: MassState) -> MotionState:
    inv = 1.0 / max(1e-9, float(mass.raw_mass))
    return MotionState(
        expected_direction=float(force.directional) * inv,
        expected_expansion=float(force.expansion) * inv,
        pinning_strength=float(force.magnet) * inv,
        execution_difficulty=max(0.0, float(force.friction)) * inv,
        systemic_coupling=float(force.systemic) * inv,
    )
