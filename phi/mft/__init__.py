"""
phi.mft — Market Field Theory projection pipeline.

Moved from ``src/phinence`` (formerly ``phinence.*``).

Sub-packages
------------
  engines/     — regime, liquidity, hedge, sentiment field engines
  composer/    — MFM → ProjectionPacket (direction, drift, vol-cones)
  contracts/   — AssignedPacket, ProjectionPacket (data contracts only)
  mfm/         — MarketFieldMap merger
  store/       — ParquetBarStore / InMemoryBarStore / bar_store_protocol
  assignment/  — AssignmentEngine
  validation/  — walk-forward + paper trading + backtest runner

Hard boundary (per .cursor/rules/projection-only.mdc):
  No strategy selection, order routing, or sizing here.

Backward compat: ``from phinence.*`` still works via the shims in src/phinence/.
"""
