"""
phinance.agents.autonomous_pipeline
======================================

AutonomousPipeline — end-to-end orchestrator that runs the full
autonomous loop without human intervention:

  Propose → Validate → Deploy (or Retry)

The pipeline can be triggered on-demand (``run_once()``) or scheduled
to run repeatedly (``run_loop()``).

Feedback loop
-------------
* If a strategy is **rejected**, the pipeline re-runs the proposer with
  tighter ``min_score`` filters or different regime assumptions.
* If a strategy is **approved**, it is deployed and the pipeline records
  the deployment.
* After ``max_deployments`` have been accumulated the pipeline pauses
  until older strategies are rolled back.

Public API
----------
  PipelineRunResult       — typed result of a single pipeline iteration
  AutonomousPipeline      — propose → validate → deploy orchestrator
  run_autonomous_pipeline — convenience one-shot function
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from phinance.agents.strategy_proposer import StrategyProposerAgent, StrategyProposal
from phinance.agents.strategy_validator import StrategyValidator, ValidationResult
from phinance.agents.autonomous_deployer import (
    AutonomousDeployer,
    DeploymentRecord,
    StrategyRegistry,
)
from phinance.utils.logging import get_logger

logger = get_logger(__name__)


# ── PipelineRunResult ─────────────────────────────────────────────────────────


@dataclass
class PipelineRunResult:
    """
    Summary of a single end-to-end pipeline run.

    Attributes
    ----------
    success          : bool — True if a strategy was successfully deployed
    proposal         : StrategyProposal or None
    validation        : ValidationResult or None
    deployment        : DeploymentRecord or None
    attempts          : int  — number of propose/validate cycles run
    total_elapsed_ms  : float — wall-clock time for the full run
    message           : str  — human-readable summary
    """

    success:          bool
    proposal:         Optional[StrategyProposal]       = None
    validation:       Optional[ValidationResult]        = None
    deployment:       Optional[DeploymentRecord]        = None
    attempts:         int                              = 0
    total_elapsed_ms: float                            = 0.0
    message:          str                              = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success":          self.success,
            "attempts":         self.attempts,
            "total_elapsed_ms": self.total_elapsed_ms,
            "message":          self.message,
            "deployment_id":    self.deployment.deployment_id if self.deployment else None,
            "regime":           self.proposal.regime if self.proposal else None,
            "sharpe":           self.validation.sharpe if self.validation else None,
        }


# ── AutonomousPipeline ────────────────────────────────────────────────────────


class AutonomousPipeline:
    """
    Fully autonomous propose → validate → deploy loop.

    Parameters
    ----------
    proposer         : StrategyProposerAgent (default: new instance)
    validator        : StrategyValidator     (default: new instance)
    deployer         : AutonomousDeployer    (default: new instance, dry_run=True)
    max_retries      : int   — proposal retries if validation fails (default 3)
    retry_top_n_step : int   — reduce top_n by this each retry (default 1)
    dry_run          : bool  — passed to deployer (default True)
    """

    def __init__(
        self,
        proposer:         Optional[StrategyProposerAgent] = None,
        validator:        Optional[StrategyValidator]     = None,
        deployer:         Optional[AutonomousDeployer]    = None,
        max_retries:      int  = 3,
        retry_top_n_step: int  = 1,
        dry_run:          bool = True,
    ) -> None:
        self.proposer         = proposer  or StrategyProposerAgent()
        self.validator        = validator or StrategyValidator()
        self.deployer         = deployer  or AutonomousDeployer(dry_run=dry_run)
        self.max_retries      = max_retries
        self.retry_top_n_step = retry_top_n_step
        self._run_history:    List[PipelineRunResult] = []

    # ── Main entry points ─────────────────────────────────────────────────────

    def run_once(
        self,
        ohlcv:    pd.DataFrame,
        symbol:   str = "SIM",
        notes:    str = "",
    ) -> PipelineRunResult:
        """
        Execute one full propose→validate→deploy cycle.

        Parameters
        ----------
        ohlcv   : pd.DataFrame — OHLCV historical data
        symbol  : str          — ticker label used in backtest
        notes   : str          — optional deployment notes

        Returns
        -------
        PipelineRunResult
        """
        t0 = time.time()
        attempts = 0
        last_proposal: Optional[StrategyProposal]  = None
        last_validation: Optional[ValidationResult] = None

        # Save original top_n so we can restore it between pipeline runs
        original_top_n   = self.proposer.top_n
        original_score   = self.proposer.min_score

        try:
            for attempt in range(self.max_retries + 1):
                attempts = attempt + 1
                logger.info("Pipeline attempt %d / %d", attempts, self.max_retries + 1)

                # 1. Propose
                try:
                    proposal = self.proposer.propose(ohlcv)
                    last_proposal = proposal
                except Exception as exc:
                    logger.warning("Proposer failed: %s", exc)
                    continue

                # 2. Validate
                try:
                    validation = self.validator.validate(proposal, ohlcv, symbol=symbol)
                    last_validation = validation
                except Exception as exc:
                    logger.warning("Validator failed: %s", exc)
                    continue

                if validation.approved:
                    # 3. Deploy
                    try:
                        deployment = self.deployer.deploy(validation, notes=notes)
                        elapsed = (time.time() - t0) * 1000
                        result = PipelineRunResult(
                            success=True,
                            proposal=proposal,
                            validation=validation,
                            deployment=deployment,
                            attempts=attempts,
                            total_elapsed_ms=elapsed,
                            message=(
                                f"Deployed '{deployment.strategy_name}' after {attempts} "
                                f"attempt(s) | Sharpe={validation.sharpe:.3f} | "
                                f"Regime={proposal.regime}"
                            ),
                        )
                        self._run_history.append(result)
                        return result
                    except RuntimeError as exc:
                        elapsed = (time.time() - t0) * 1000
                        result = PipelineRunResult(
                            success=False,
                            proposal=proposal,
                            validation=validation,
                            attempts=attempts,
                            total_elapsed_ms=elapsed,
                            message=f"Deploy failed: {exc}",
                        )
                        self._run_history.append(result)
                        return result
                else:
                    # Retry with fewer indicators / lower bar
                    logger.info(
                        "Validation failed (attempt %d): %s — retrying with fewer indicators",
                        attempts, validation.rejection_reason,
                    )
                    new_top_n = max(2, self.proposer.top_n - self.retry_top_n_step)
                    self.proposer.top_n       = new_top_n
                    self.proposer.min_score   = max(0.40, self.proposer.min_score - 0.02)

            # All retries exhausted
            elapsed = (time.time() - t0) * 1000
            reason = last_validation.rejection_reason if last_validation else "No valid proposal generated"
            result = PipelineRunResult(
                success=False,
                proposal=last_proposal,
                validation=last_validation,
                attempts=attempts,
                total_elapsed_ms=elapsed,
                message=f"Pipeline exhausted {attempts} attempt(s). Last rejection: {reason}",
            )
            self._run_history.append(result)
            return result

        finally:
            # Restore proposer state
            self.proposer.top_n     = original_top_n
            self.proposer.min_score = original_score

    def run_loop(
        self,
        ohlcv:        pd.DataFrame,
        n_iterations: int  = 3,
        symbol:       str  = "SIM",
    ) -> List[PipelineRunResult]:
        """
        Run the pipeline ``n_iterations`` times (e.g., for daily cron).

        Parameters
        ----------
        ohlcv        : pd.DataFrame — OHLCV data
        n_iterations : int          — number of pipeline iterations
        symbol       : str          — ticker label

        Returns
        -------
        List[PipelineRunResult]
        """
        results = []
        for i in range(n_iterations):
            logger.info("Pipeline loop iteration %d / %d", i + 1, n_iterations)
            result = self.run_once(ohlcv, symbol=symbol)
            results.append(result)
            if result.success:
                # Check if max deployments reached
                if len(self.deployer.list_active()) >= self.deployer.max_active:
                    logger.info(
                        "Max active deployments reached (%d). Stopping loop.",
                        self.deployer.max_active,
                    )
                    break
        return results

    @property
    def run_history(self) -> List[PipelineRunResult]:
        """All pipeline run results since instantiation."""
        return list(self._run_history)

    @property
    def active_deployments(self) -> List[DeploymentRecord]:
        """Currently ACTIVE deployed strategies."""
        return self.deployer.list_active()


# ── Convenience function ──────────────────────────────────────────────────────


def run_autonomous_pipeline(
    ohlcv:       pd.DataFrame,
    symbol:      str   = "SIM",
    dry_run:     bool  = True,
    max_retries: int   = 3,
    top_n:       int   = 4,
    min_sharpe:  float = 0.3,
    registry_path: Optional[str] = None,
) -> PipelineRunResult:
    """
    One-shot convenience wrapper for the full autonomous pipeline.

    Parameters
    ----------
    ohlcv          : pd.DataFrame — OHLCV historical data
    symbol         : str          — ticker label
    dry_run        : bool         — paper mode (default True)
    max_retries    : int          — proposal retries on rejection
    top_n          : int          — indicators per proposal
    min_sharpe     : float        — validation threshold
    registry_path  : str, optional — JSON file for deployment persistence

    Returns
    -------
    PipelineRunResult
    """
    registry  = StrategyRegistry(registry_path=registry_path)
    proposer  = StrategyProposerAgent(top_n=top_n)
    validator = StrategyValidator(min_sharpe=min_sharpe)
    deployer  = AutonomousDeployer(registry=registry, dry_run=dry_run)
    pipeline  = AutonomousPipeline(
        proposer=proposer,
        validator=validator,
        deployer=deployer,
        max_retries=max_retries,
    )
    return pipeline.run_once(ohlcv, symbol=symbol)
