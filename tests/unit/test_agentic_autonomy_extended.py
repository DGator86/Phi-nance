"""
tests.unit.test_agentic_autonomy_extended
==========================================

Extended unit tests for the Full Agentic Autonomy components:
  • StrategyProposerAgent  (strategy_proposer.py)
  • StrategyValidator      (strategy_validator.py)
  • AutonomousDeployer     (autonomous_deployer.py)
  • StrategyRegistry       (autonomous_deployer.py)
  • DeploymentRecord       (autonomous_deployer.py)

All tests are pure-unit: no external network calls.
"""

from __future__ import annotations

import os
import sys
import json
import time
import tempfile
import dataclasses

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
import pytest

from tests.fixtures.ohlcv import make_ohlcv

from phinance.agents.strategy_proposer import (
    StrategyProposal,
    StrategyProposerAgent,
)
from phinance.agents.strategy_validator import (
    ValidationResult,
    StrategyValidator,
)
from phinance.agents.autonomous_deployer import (
    AutonomousDeployer,
    DeploymentRecord,
    DeploymentStatus,
    StrategyRegistry,
)

# ── Shared fixtures ───────────────────────────────────────────────────────────

DF_300 = make_ohlcv(n=300)
DF_200 = make_ohlcv(n=200)

# Permissive validator — accepts everything (min_trades=0 bypasses trade requirement)
_PERMISSIVE_VALIDATOR = StrategyValidator(
    min_sharpe=-99.0,
    max_drawdown=1.0,
    min_win_rate=0.0,
    min_trades=0,
)


def _get_approved_validation_result(df=None):
    """Return a ValidationResult with approved=True for deployer tests."""
    df = df or DF_300
    proposer  = StrategyProposerAgent(top_n=2)
    proposal  = proposer.propose(df)
    vr        = _PERMISSIVE_VALIDATOR.validate(proposal, df)
    # Force approval so deployer tests always work regardless of backtest outcome
    return dataclasses.replace(vr, approved=True)


# ═══════════════════════════════════════════════════════════════════════════════
# StrategyProposerAgent
# ═══════════════════════════════════════════════════════════════════════════════

class TestStrategyProposerAgent:

    def setup_method(self):
        self.agent = StrategyProposerAgent(top_n=3)

    # ── Basic shape ──────────────────────────────────────────────────────────

    def test_propose_returns_strategy_proposal(self):
        proposal = self.agent.propose(DF_300)
        assert isinstance(proposal, StrategyProposal)

    def test_proposal_indicators_is_dict(self):
        proposal = self.agent.propose(DF_300)
        # indicators is a dict of {name: config}
        assert isinstance(proposal.indicators, dict)
        assert len(proposal.indicators) > 0

    def test_proposal_has_weights(self):
        proposal = self.agent.propose(DF_300)
        assert isinstance(proposal.weights, dict)
        assert len(proposal.weights) > 0

    def test_weights_sum_to_one(self):
        proposal = self.agent.propose(DF_300)
        total = sum(proposal.weights.values())
        assert abs(total - 1.0) < 1e-6

    def test_proposal_has_regime(self):
        proposal = self.agent.propose(DF_300)
        assert isinstance(proposal.regime, str)
        assert len(proposal.regime) > 0

    def test_proposal_has_scores(self):
        proposal = self.agent.propose(DF_300)
        assert isinstance(proposal.scores, dict)
        assert len(proposal.scores) > 0

    def test_proposal_has_blend_method(self):
        proposal = self.agent.propose(DF_300)
        assert isinstance(proposal.blend_method, str)

    def test_proposal_has_rationale(self):
        proposal = self.agent.propose(DF_300)
        assert isinstance(proposal.rationale, str)

    def test_proposal_has_created_at(self):
        proposal = self.agent.propose(DF_300)
        assert proposal.created_at > 0

    # ── top_n respected ──────────────────────────────────────────────────────

    def test_top_n_one(self):
        agent = StrategyProposerAgent(top_n=1)
        proposal = agent.propose(DF_300)
        assert len(proposal.indicators) == 1

    def test_top_n_two(self):
        agent = StrategyProposerAgent(top_n=2)
        proposal = agent.propose(DF_300)
        assert len(proposal.indicators) == 2

    def test_top_n_five(self):
        agent = StrategyProposerAgent(top_n=5)
        proposal = agent.propose(DF_300)
        assert len(proposal.indicators) <= 5

    def test_top_n_too_large_clipped(self):
        agent = StrategyProposerAgent(top_n=999)
        proposal = agent.propose(DF_300)
        assert len(proposal.indicators) > 0

    # ── Serialisation ────────────────────────────────────────────────────────

    def test_proposal_to_dict(self):
        proposal = self.agent.propose(DF_300)
        d = proposal.to_dict()
        assert isinstance(d, dict)
        assert "indicators" in d
        assert "weights" in d
        assert "regime" in d
        assert "scores" in d

    def test_proposal_to_dict_json_serialisable(self):
        proposal = self.agent.propose(DF_300)
        d = proposal.to_dict()
        json_str = json.dumps(d)
        assert isinstance(json_str, str)
        loaded = json.loads(json_str)
        assert loaded["regime"] == proposal.regime

    # ── AgentBase interface ──────────────────────────────────────────────────

    def test_agent_name(self):
        assert isinstance(self.agent.name, str)
        assert len(self.agent.name) > 0

    def test_agent_analyze(self):
        from phinance.agents.base import AgentResult
        result = self.agent.analyze({"ohlcv": DF_300})
        assert isinstance(result, AgentResult)

    def test_agent_capabilities(self):
        caps = self.agent.capabilities
        assert len(caps) > 0

    # ── Different data sizes ─────────────────────────────────────────────────

    def test_propose_with_200_bars(self):
        proposal = self.agent.propose(DF_200)
        assert isinstance(proposal, StrategyProposal)

    def test_propose_with_minimal_data(self):
        df_small = make_ohlcv(n=50, start="2023-01-01")
        proposal = self.agent.propose(df_small)
        assert isinstance(proposal, StrategyProposal)

    # ── Indicators keys match weights keys ───────────────────────────────────

    def test_indicators_keys_match_weights(self):
        proposal = self.agent.propose(DF_300)
        assert set(proposal.indicators.keys()) == set(proposal.weights.keys())

    def test_scores_keys_subset_of_indicators(self):
        proposal = self.agent.propose(DF_300)
        ind_names = set(proposal.indicators.keys())
        score_names = set(proposal.scores.keys())
        assert score_names.issubset(ind_names)


# ═══════════════════════════════════════════════════════════════════════════════
# StrategyValidator
# ═══════════════════════════════════════════════════════════════════════════════

class TestStrategyValidator:

    def setup_method(self):
        self.proposer  = StrategyProposerAgent(top_n=2)
        self.validator = _PERMISSIVE_VALIDATOR

    def _get_proposal(self, df=None):
        df = df or DF_300
        return self.proposer.propose(df)

    # ── Basic shape ──────────────────────────────────────────────────────────

    def test_validate_returns_validation_result(self):
        proposal = self._get_proposal()
        vr = self.validator.validate(proposal, DF_300)
        assert isinstance(vr, ValidationResult)

    def test_validation_result_has_approved(self):
        proposal = self._get_proposal()
        vr = self.validator.validate(proposal, DF_300)
        assert isinstance(vr.approved, bool)

    def test_validation_result_has_sharpe(self):
        proposal = self._get_proposal()
        vr = self.validator.validate(proposal, DF_300)
        assert isinstance(vr.sharpe, float)

    def test_validation_result_has_max_drawdown(self):
        proposal = self._get_proposal()
        vr = self.validator.validate(proposal, DF_300)
        assert isinstance(vr.max_drawdown, float)
        assert vr.max_drawdown >= 0.0

    def test_validation_result_has_win_rate(self):
        proposal = self._get_proposal()
        vr = self.validator.validate(proposal, DF_300)
        assert 0.0 <= vr.win_rate <= 1.0

    def test_validation_result_has_total_return(self):
        proposal = self._get_proposal()
        vr = self.validator.validate(proposal, DF_300)
        assert isinstance(vr.total_return, float)

    def test_validation_result_has_num_trades(self):
        proposal = self._get_proposal()
        vr = self.validator.validate(proposal, DF_300)
        assert isinstance(vr.num_trades, int)
        assert vr.num_trades >= 0

    def test_validation_elapsed_ms_positive(self):
        proposal = self._get_proposal()
        vr = self.validator.validate(proposal, DF_300)
        assert vr.elapsed_ms >= 0

    def test_validation_has_backtest_stats_dict(self):
        proposal = self._get_proposal()
        vr = self.validator.validate(proposal, DF_300)
        assert isinstance(vr.backtest_stats, dict)

    # ── Strict thresholds → rejection ────────────────────────────────────────

    def test_rejection_on_high_sharpe_threshold(self):
        strict = StrategyValidator(min_sharpe=999.0)
        proposal = self._get_proposal()
        vr = strict.validate(proposal, DF_300)
        assert vr.approved is False
        assert isinstance(vr.rejection_reason, str)

    def test_rejection_reason_non_empty_on_rejection(self):
        strict = StrategyValidator(min_sharpe=999.0)
        proposal = self._get_proposal()
        vr = strict.validate(proposal, DF_300)
        assert len(vr.rejection_reason) > 0

    # ── Serialisation ────────────────────────────────────────────────────────

    def test_validation_result_to_dict(self):
        proposal = self._get_proposal()
        vr = self.validator.validate(proposal, DF_300)
        d = vr.to_dict()
        assert isinstance(d, dict)
        assert "approved" in d
        assert "sharpe" in d

    def test_validation_result_to_dict_json_serialisable(self):
        proposal = self._get_proposal()
        vr = self.validator.validate(proposal, DF_300)
        d = vr.to_dict()
        json.dumps(d)   # must not raise


# ═══════════════════════════════════════════════════════════════════════════════
# DeploymentRecord & DeploymentStatus
# ═══════════════════════════════════════════════════════════════════════════════

class TestDeploymentRecord:

    def _make_record(self, **kwargs) -> DeploymentRecord:
        import uuid
        defaults = dict(
            deployment_id    = str(uuid.uuid4()),
            strategy_name    = "TestStrategy",
            indicators       = {"RSI": {"period": 14}},
            weights          = {"RSI": 1.0},
            blend_method     = "weighted",
            regime_at_deploy = "TREND_UP",
            validation_stats = {"sharpe": 0.5, "max_drawdown": 0.1},
            status           = DeploymentStatus.ACTIVE,
            dry_run          = True,
            deployed_at      = time.time(),
            rolled_back_at   = 0.0,
            notes            = "",
        )
        defaults.update(kwargs)
        return DeploymentRecord(**defaults)

    def test_create_record(self):
        rec = self._make_record()
        assert rec.deployment_id

    def test_status_active(self):
        rec = self._make_record(status=DeploymentStatus.ACTIVE)
        assert rec.status == DeploymentStatus.ACTIVE

    def test_status_rolled_back(self):
        rec = self._make_record(status=DeploymentStatus.ROLLED_BACK)
        assert rec.status == DeploymentStatus.ROLLED_BACK

    def test_status_pending(self):
        rec = self._make_record(status=DeploymentStatus.PENDING)
        assert rec.status == DeploymentStatus.PENDING

    def test_status_failed(self):
        rec = self._make_record(status=DeploymentStatus.FAILED)
        assert rec.status == DeploymentStatus.FAILED

    def test_to_dict(self):
        rec = self._make_record()
        d = rec.to_dict()
        assert isinstance(d, dict)
        assert "deployment_id" in d
        assert "strategy_name" in d

    def test_to_dict_json_serialisable(self):
        rec = self._make_record()
        json.dumps(rec.to_dict())   # must not raise

    def test_dry_run_default(self):
        rec = self._make_record()
        assert rec.dry_run is True


# ═══════════════════════════════════════════════════════════════════════════════
# StrategyRegistry
# ═══════════════════════════════════════════════════════════════════════════════

class TestStrategyRegistry:

    def setup_method(self):
        self.reg = StrategyRegistry()

    def _make_record(self, name="Strat") -> DeploymentRecord:
        import uuid
        return DeploymentRecord(
            deployment_id    = str(uuid.uuid4()),
            strategy_name    = name,
            indicators       = {"RSI": {}},
            weights          = {"RSI": 1.0},
            blend_method     = "weighted",
            regime_at_deploy = "RANGE",
            validation_stats = {"sharpe": 0.3},
            status           = DeploymentStatus.ACTIVE,
            dry_run          = True,
            deployed_at      = time.time(),
            rolled_back_at   = 0.0,
            notes            = "",
        )

    def test_register_and_get(self):
        rec = self._make_record()
        self.reg.register(rec)
        got = self.reg.get(rec.deployment_id)
        assert got is not None
        assert got.deployment_id == rec.deployment_id

    def test_get_nonexistent_returns_none(self):
        assert self.reg.get("does-not-exist") is None

    def test_len_empty(self):
        assert len(self.reg) == 0

    def test_len_after_register(self):
        self.reg.register(self._make_record("A"))
        self.reg.register(self._make_record("B"))
        assert len(self.reg) == 2

    def test_list_active(self):
        rec = self._make_record("ActiveStrat")
        self.reg.register(rec)
        active = self.reg.list_active()
        assert any(r.deployment_id == rec.deployment_id for r in active)

    def test_update_status(self):
        rec = self._make_record()
        self.reg.register(rec)
        ok = self.reg.update_status(rec.deployment_id, DeploymentStatus.ROLLED_BACK)
        assert ok is True
        updated = self.reg.get(rec.deployment_id)
        assert updated.status == DeploymentStatus.ROLLED_BACK

    def test_update_status_nonexistent(self):
        ok = self.reg.update_status("ghost-id", DeploymentStatus.FAILED)
        assert ok is False

    def test_list_all(self):
        self.reg.register(self._make_record("X"))
        self.reg.register(self._make_record("Y"))
        all_records = self.reg.list_all()
        assert len(all_records) == 2

    def test_registry_with_file_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "registry.json")
            reg = StrategyRegistry(registry_path=path)
            rec = self._make_record("FileStrat")
            reg.register(rec)
            assert os.path.exists(path)


# ═══════════════════════════════════════════════════════════════════════════════
# AutonomousDeployer
# ═══════════════════════════════════════════════════════════════════════════════

class TestAutonomousDeployer:

    def setup_method(self):
        self.deployer = AutonomousDeployer(dry_run=True)

    def test_deploy_returns_deployment_record(self):
        vr = _get_approved_validation_result()
        rec = self.deployer.deploy(vr)
        assert isinstance(rec, DeploymentRecord)

    def test_deployment_id_is_string(self):
        vr = _get_approved_validation_result()
        rec = self.deployer.deploy(vr)
        assert isinstance(rec.deployment_id, str)
        assert len(rec.deployment_id) > 0

    def test_deployment_dry_run_flag(self):
        vr = _get_approved_validation_result()
        rec = self.deployer.deploy(vr)
        assert rec.dry_run is True

    def test_deployment_status_valid(self):
        vr = _get_approved_validation_result()
        rec = self.deployer.deploy(vr)
        assert rec.status in (
            DeploymentStatus.ACTIVE,
            DeploymentStatus.FAILED,
            DeploymentStatus.PENDING,
        )

    def test_rollback_deployed_strategy(self):
        vr = _get_approved_validation_result()
        rec = self.deployer.deploy(vr)
        ok = self.deployer.rollback(rec.deployment_id)
        assert isinstance(ok, bool)

    def test_list_active_after_deploy(self):
        vr = _get_approved_validation_result()
        self.deployer.deploy(vr)
        active = self.deployer.list_active()
        assert isinstance(active, list)

    def test_get_record(self):
        vr = _get_approved_validation_result()
        rec = self.deployer.deploy(vr)
        got = self.deployer.get_record(rec.deployment_id)
        assert got is not None

    def test_get_nonexistent_record(self):
        got = self.deployer.get_record("ghost")
        assert got is None

    def test_deploy_with_custom_name(self):
        vr = _get_approved_validation_result()
        rec = self.deployer.deploy(vr, strategy_name="CustomName")
        assert rec.strategy_name == "CustomName"

    def test_deploy_sets_validation_stats(self):
        vr = _get_approved_validation_result()
        rec = self.deployer.deploy(vr)
        assert isinstance(rec.validation_stats, dict)

    def test_multiple_deployments_unique_ids(self):
        vr1 = _get_approved_validation_result()
        vr2 = _get_approved_validation_result()
        rec1 = self.deployer.deploy(vr1)
        rec2 = self.deployer.deploy(vr2)
        assert rec1.deployment_id != rec2.deployment_id

    def test_deploy_unapproved_raises(self):
        vr = _get_approved_validation_result()
        vr_rejected = dataclasses.replace(vr, approved=False, rejection_reason="bad sharpe")
        with pytest.raises(Exception):
            self.deployer.deploy(vr_rejected)
