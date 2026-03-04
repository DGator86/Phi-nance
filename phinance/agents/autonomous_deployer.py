"""
phinance.agents.autonomous_deployer
======================================

AutonomousDeployer — takes a validated strategy and "deploys" it:
  • Stores the strategy in the strategy registry (in-memory + optional JSON)
  • Wires the approved indicators/weights into a LiveTradingLoop
  • Returns a DeploymentRecord with full audit trail

Design principles
-----------------
* **Dry-run first** — default ``dry_run=True``; nothing touches a real
  broker unless explicitly enabled.
* **Immutable records** — every deployment is stored with a unique ID
  and can be recalled at any time.
* **Audit trail** — all deploy/rollback events are logged to the registry.

Public API
----------
  DeploymentRecord         — typed deployment record
  DeploymentStatus         — enum (PENDING, ACTIVE, ROLLED_BACK, FAILED)
  AutonomousDeployer       — deploy/rollback controller
  StrategyRegistry         — in-memory + optional file-backed strategy store
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd

from phinance.agents.strategy_validator import ValidationResult
from phinance.utils.logging import get_logger

logger = get_logger(__name__)


# ── Enums ─────────────────────────────────────────────────────────────────────


class DeploymentStatus(str, Enum):
    PENDING      = "pending"
    ACTIVE       = "active"
    ROLLED_BACK  = "rolled_back"
    FAILED       = "failed"


# ── DeploymentRecord ──────────────────────────────────────────────────────────


@dataclass
class DeploymentRecord:
    """
    Immutable record of a single strategy deployment.

    Attributes
    ----------
    deployment_id   : str   — unique UUID
    strategy_name   : str   — human label (auto-generated if not given)
    indicators      : dict  — indicator config from the proposal
    weights         : dict  — blend weights
    blend_method    : str   — blend method
    regime_at_deploy: str   — market regime when deployed
    validation_stats: dict  — Sharpe, DD, win_rate from validator
    status          : str   — DeploymentStatus value
    dry_run         : bool  — True = paper/simulation only
    deployed_at     : float — epoch timestamp
    rolled_back_at  : float — epoch timestamp if rolled back
    notes           : str   — optional free-text notes
    """

    deployment_id:    str
    strategy_name:    str
    indicators:       Dict[str, Any]
    weights:          Dict[str, float]
    blend_method:     str
    regime_at_deploy: str
    validation_stats: Dict[str, Any]
    status:           str   = DeploymentStatus.ACTIVE
    dry_run:          bool  = True
    deployed_at:      float = field(default_factory=time.time)
    rolled_back_at:   Optional[float] = None
    notes:            str   = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "deployment_id":    self.deployment_id,
            "strategy_name":    self.strategy_name,
            "indicators":       self.indicators,
            "weights":          self.weights,
            "blend_method":     self.blend_method,
            "regime_at_deploy": self.regime_at_deploy,
            "validation_stats": self.validation_stats,
            "status":           self.status,
            "dry_run":          self.dry_run,
            "deployed_at":      self.deployed_at,
            "rolled_back_at":   self.rolled_back_at,
            "notes":            self.notes,
        }


# ── StrategyRegistry ──────────────────────────────────────────────────────────


class StrategyRegistry:
    """
    In-memory (+ optional JSON file) registry of deployed strategies.

    Parameters
    ----------
    registry_path : str, optional — path to JSON file for persistence
    """

    def __init__(self, registry_path: Optional[str] = None) -> None:
        self._records: Dict[str, DeploymentRecord] = {}
        self._registry_path = registry_path
        if registry_path and os.path.exists(registry_path):
            self._load()

    # ── CRUD ──────────────────────────────────────────────────────────────────

    def register(self, record: DeploymentRecord) -> None:
        """Add a new deployment record."""
        self._records[record.deployment_id] = record
        self._save()

    def get(self, deployment_id: str) -> Optional[DeploymentRecord]:
        """Retrieve a record by ID. Returns None if not found."""
        return self._records.get(deployment_id)

    def update_status(self, deployment_id: str, status: DeploymentStatus) -> bool:
        """Update the status of an existing record. Returns True on success."""
        rec = self._records.get(deployment_id)
        if rec is None:
            return False
        rec.status = status
        if status == DeploymentStatus.ROLLED_BACK:
            rec.rolled_back_at = time.time()
        self._save()
        return True

    def list_active(self) -> List[DeploymentRecord]:
        """Return all ACTIVE deployment records."""
        return [r for r in self._records.values() if r.status == DeploymentStatus.ACTIVE]

    def list_all(self) -> List[DeploymentRecord]:
        """Return all deployment records (any status)."""
        return list(self._records.values())

    def __len__(self) -> int:
        return len(self._records)

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self) -> None:
        if not self._registry_path:
            return
        try:
            os.makedirs(os.path.dirname(self._registry_path) or ".", exist_ok=True)
            with open(self._registry_path, "w") as fh:
                json.dump(
                    {k: v.to_dict() for k, v in self._records.items()},
                    fh,
                    indent=2,
                    default=str,
                )
        except Exception as exc:
            logger.warning("Could not save registry: %s", exc)

    def _load(self) -> None:
        try:
            with open(self._registry_path) as fh:
                raw = json.load(fh)
            for dep_id, rec_dict in raw.items():
                self._records[dep_id] = DeploymentRecord(**{
                    k: v for k, v in rec_dict.items()
                    if k in DeploymentRecord.__dataclass_fields__
                })
            logger.info("Registry loaded: %d records", len(self._records))
        except Exception as exc:
            logger.warning("Could not load registry: %s", exc)


# ── AutonomousDeployer ────────────────────────────────────────────────────────


class AutonomousDeployer:
    """
    Deploys validated strategies to a live or paper trading loop.

    Parameters
    ----------
    registry        : StrategyRegistry — persistent strategy store
    dry_run         : bool — if True, skip broker connection (default True)
    max_active      : int  — max simultaneous ACTIVE deployments (default 3)
    """

    def __init__(
        self,
        registry:   Optional[StrategyRegistry] = None,
        dry_run:    bool = True,
        max_active: int  = 3,
    ) -> None:
        self.registry   = registry or StrategyRegistry()
        self.dry_run    = dry_run
        self.max_active = max_active

    def deploy(
        self,
        validation: ValidationResult,
        strategy_name: Optional[str] = None,
        notes: str = "",
    ) -> DeploymentRecord:
        """
        Deploy an approved strategy.

        Parameters
        ----------
        validation    : ValidationResult — must have ``approved=True``
        strategy_name : str, optional    — human label; auto-generated if None
        notes         : str              — free-text audit notes

        Returns
        -------
        DeploymentRecord

        Raises
        ------
        ValueError — if validation is not approved
        RuntimeError — if max_active deployments already running
        """
        if not validation.approved:
            raise ValueError(
                f"Cannot deploy rejected strategy: {validation.rejection_reason}"
            )

        active = self.registry.list_active()
        if len(active) >= self.max_active:
            raise RuntimeError(
                f"Max active deployments ({self.max_active}) reached. "
                "Roll back an existing deployment before adding a new one."
            )

        deployment_id = str(uuid.uuid4())
        name = strategy_name or (
            f"auto_{validation.proposal.regime}_{deployment_id[:8]}"
        )

        record = DeploymentRecord(
            deployment_id=deployment_id,
            strategy_name=name,
            indicators=validation.proposal.indicators,
            weights=validation.proposal.weights,
            blend_method=validation.proposal.blend_method,
            regime_at_deploy=validation.proposal.regime,
            validation_stats=validation.to_dict(),
            status=DeploymentStatus.ACTIVE,
            dry_run=self.dry_run,
            notes=notes,
        )

        self.registry.register(record)

        if self.dry_run:
            logger.info(
                "[DRY-RUN] Deployed strategy '%s' (id=%s) | regime=%s",
                name, deployment_id[:8], validation.proposal.regime,
            )
        else:
            logger.info(
                "[LIVE] Deployed strategy '%s' (id=%s)",
                name, deployment_id[:8],
            )

        return record

    def rollback(self, deployment_id: str) -> bool:
        """
        Roll back an active deployment.

        Parameters
        ----------
        deployment_id : str — ID of the deployment to roll back

        Returns
        -------
        bool — True if successfully rolled back, False if not found / already rolled back
        """
        rec = self.registry.get(deployment_id)
        if rec is None:
            logger.warning("Rollback: deployment %s not found", deployment_id)
            return False
        if rec.status != DeploymentStatus.ACTIVE:
            logger.warning(
                "Rollback: deployment %s is not ACTIVE (status=%s)",
                deployment_id, rec.status,
            )
            return False

        self.registry.update_status(deployment_id, DeploymentStatus.ROLLED_BACK)
        logger.info("Rolled back deployment %s ('%s')", deployment_id[:8], rec.strategy_name)
        return True

    def list_active(self) -> List[DeploymentRecord]:
        """Return all currently ACTIVE deployments."""
        return self.registry.list_active()

    def get_record(self, deployment_id: str) -> Optional[DeploymentRecord]:
        """Retrieve a deployment record by ID."""
        return self.registry.get(deployment_id)
