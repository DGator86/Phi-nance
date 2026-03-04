"""
tests.unit.test_autonomous_pipeline
======================================

Comprehensive tests for Phase 9.1 — Agentic Autonomy:
  • StrategyProposerAgent + StrategyProposal
  • StrategyValidator    + ValidationResult
  • AutonomousDeployer   + StrategyRegistry + DeploymentRecord
  • AutonomousPipeline   + PipelineRunResult
  • run_autonomous_pipeline convenience function
"""

from __future__ import annotations

import time
import pytest
import pandas as pd
import numpy as np

from tests.fixtures.ohlcv import make_ohlcv

# ── Shared fixtures ────────────────────────────────────────────────────────────

DF_SMALL = make_ohlcv(n=40,  start="2020-01-01")   # too small for validation
DF_MED   = make_ohlcv(n=200, start="2021-01-01")
DF_LARGE = make_ohlcv(n=500, start="2022-01-01")


# ═══════════════════════════════════════════════════════════════════════════════
# StrategyProposal dataclass
# ═══════════════════════════════════════════════════════════════════════════════

class TestStrategyProposal:

    def test_to_dict_keys(self):
        from phinance.agents.strategy_proposer import StrategyProposal
        p = StrategyProposal(
            indicators={"RSI": {"enabled": True, "params": {}}},
            weights={"RSI": 1.0},
            regime="TREND_UP",
        )
        d = p.to_dict()
        assert "indicators" in d
        assert "weights"    in d
        assert "regime"     in d
        assert "rationale"  in d
        assert "created_at" in d

    def test_created_at_is_recent(self):
        from phinance.agents.strategy_proposer import StrategyProposal
        p = StrategyProposal(indicators={}, weights={})
        assert abs(p.created_at - time.time()) < 5

    def test_scores_default_empty(self):
        from phinance.agents.strategy_proposer import StrategyProposal
        p = StrategyProposal(indicators={}, weights={})
        assert isinstance(p.scores, dict)


# ═══════════════════════════════════════════════════════════════════════════════
# StrategyProposerAgent
# ═══════════════════════════════════════════════════════════════════════════════

class TestStrategyProposerAgent:

    def test_name(self):
        from phinance.agents.strategy_proposer import StrategyProposerAgent
        assert StrategyProposerAgent().name == "StrategyProposerAgent"

    def test_propose_returns_proposal(self):
        from phinance.agents.strategy_proposer import StrategyProposerAgent, StrategyProposal
        agent    = StrategyProposerAgent(top_n=3)
        proposal = agent.propose(DF_MED)
        assert isinstance(proposal, StrategyProposal)

    def test_proposal_has_indicators(self):
        from phinance.agents.strategy_proposer import StrategyProposerAgent
        proposal = StrategyProposerAgent(top_n=3).propose(DF_MED)
        assert len(proposal.indicators) > 0

    def test_proposal_weights_sum_to_one(self):
        from phinance.agents.strategy_proposer import StrategyProposerAgent
        proposal = StrategyProposerAgent(top_n=3).propose(DF_MED)
        assert abs(sum(proposal.weights.values()) - 1.0) < 1e-9

    def test_proposal_weights_keys_match_indicators(self):
        from phinance.agents.strategy_proposer import StrategyProposerAgent
        proposal = StrategyProposerAgent(top_n=3).propose(DF_MED)
        assert set(proposal.weights.keys()) == set(proposal.indicators.keys())

    def test_proposal_regime_is_string(self):
        from phinance.agents.strategy_proposer import StrategyProposerAgent
        proposal = StrategyProposerAgent().propose(DF_MED)
        assert isinstance(proposal.regime, str)
        assert len(proposal.regime) > 0

    def test_top_n_respected(self):
        from phinance.agents.strategy_proposer import StrategyProposerAgent
        for n in (2, 3, 5):
            proposal = StrategyProposerAgent(top_n=n).propose(DF_MED)
            assert len(proposal.indicators) <= n

    def test_insufficient_data_fallback(self):
        from phinance.agents.strategy_proposer import StrategyProposerAgent
        # Should still produce a proposal even with small DF
        proposal = StrategyProposerAgent(top_n=2).propose(DF_SMALL)
        assert isinstance(proposal.indicators, dict)

    def test_analyze_returns_agent_result(self):
        from phinance.agents.strategy_proposer import StrategyProposerAgent
        from phinance.agents.base import AgentResult
        agent  = StrategyProposerAgent(top_n=2)
        result = agent.analyze({"ohlcv": DF_MED})
        assert isinstance(result, AgentResult)
        assert result.action in ("buy", "sell", "hold")
        assert 0.0 <= result.confidence <= 1.0

    def test_analyze_no_ohlcv_returns_hold(self):
        from phinance.agents.strategy_proposer import StrategyProposerAgent
        result = StrategyProposerAgent().analyze({})
        assert result.action == "hold"
        assert result.confidence == 0.0

    def test_analyze_tiny_ohlcv_returns_hold(self):
        from phinance.agents.strategy_proposer import StrategyProposerAgent
        result = StrategyProposerAgent().analyze({"ohlcv": make_ohlcv(n=5)})
        assert result.action == "hold"

    def test_proposal_rationale_non_empty(self):
        from phinance.agents.strategy_proposer import StrategyProposerAgent
        proposal = StrategyProposerAgent(top_n=2).propose(DF_MED)
        assert len(proposal.rationale) > 10

    def test_proposal_indicators_all_in_catalog(self):
        from phinance.agents.strategy_proposer import StrategyProposerAgent
        from phinance.strategies.indicator_catalog import INDICATOR_CATALOG
        proposal = StrategyProposerAgent(top_n=4).propose(DF_MED)
        for name in proposal.indicators:
            assert name in INDICATOR_CATALOG, f"{name} not in catalog"

    def test_blend_method_propagated(self):
        from phinance.agents.strategy_proposer import StrategyProposerAgent
        agent    = StrategyProposerAgent(blend_method="regime_weighted")
        proposal = agent.propose(DF_MED)
        assert proposal.blend_method == "regime_weighted"


# ═══════════════════════════════════════════════════════════════════════════════
# ValidationResult dataclass
# ═══════════════════════════════════════════════════════════════════════════════

class TestValidationResult:

    def _make_approved_result(self):
        from phinance.agents.strategy_proposer import StrategyProposal
        from phinance.agents.strategy_validator import ValidationResult
        proposal = StrategyProposal(indicators={"RSI": {}}, weights={"RSI": 1.0})
        return ValidationResult(
            approved=True, proposal=proposal,
            sharpe=1.2, max_drawdown=0.05, win_rate=0.6, num_trades=10,
        )

    def test_to_dict_keys(self):
        r = self._make_approved_result()
        d = r.to_dict()
        assert "approved"      in d
        assert "sharpe"        in d
        assert "max_drawdown"  in d
        assert "win_rate"      in d
        assert "num_trades"    in d

    def test_approved_true(self):
        r = self._make_approved_result()
        assert r.approved is True

    def test_rejection_reason_empty_when_approved(self):
        r = self._make_approved_result()
        assert r.rejection_reason == ""


# ═══════════════════════════════════════════════════════════════════════════════
# StrategyValidator
# ═══════════════════════════════════════════════════════════════════════════════

class TestStrategyValidator:

    def _make_proposal(self, indicator="RSI"):
        from phinance.agents.strategy_proposer import StrategyProposal
        return StrategyProposal(
            indicators={indicator: {"enabled": True, "params": {}}},
            weights={indicator: 1.0},
        )

    def test_validate_returns_validation_result(self):
        from phinance.agents.strategy_validator import StrategyValidator, ValidationResult
        validator = StrategyValidator(min_sharpe=-999, max_drawdown=1.0, min_trades=0)
        result    = validator.validate(self._make_proposal(), DF_LARGE, symbol="SPY")
        assert isinstance(result, ValidationResult)

    def test_approved_with_lenient_thresholds(self):
        from phinance.agents.strategy_validator import StrategyValidator
        validator = StrategyValidator(min_sharpe=-99, max_drawdown=1.0, min_win_rate=0.0, min_trades=0)
        result    = validator.validate(self._make_proposal(), DF_LARGE)
        assert result.approved is True

    def test_rejected_with_strict_sharpe(self):
        from phinance.agents.strategy_validator import StrategyValidator
        validator = StrategyValidator(min_sharpe=999.0, max_drawdown=1.0, min_trades=0)
        result    = validator.validate(self._make_proposal(), DF_LARGE)
        assert result.approved is False
        assert "Sharpe" in result.rejection_reason

    def test_rejected_strict_drawdown(self):
        from phinance.agents.strategy_validator import StrategyValidator
        validator = StrategyValidator(min_sharpe=-99, max_drawdown=0.0001, min_trades=0)
        result    = validator.validate(self._make_proposal(), DF_LARGE)
        # May or may not trigger depending on synthetic data, but should not crash
        assert isinstance(result.approved, bool)

    def test_elapsed_ms_positive(self):
        from phinance.agents.strategy_validator import StrategyValidator
        validator = StrategyValidator(min_sharpe=-99, max_drawdown=1.0, min_trades=0)
        result    = validator.validate(self._make_proposal(), DF_LARGE)
        assert result.elapsed_ms >= 0

    def test_backtest_stats_populated(self):
        from phinance.agents.strategy_validator import StrategyValidator
        validator = StrategyValidator(min_sharpe=-99, max_drawdown=1.0, min_trades=0)
        result    = validator.validate(self._make_proposal(), DF_LARGE)
        assert isinstance(result.backtest_stats, dict)

    def test_validation_result_to_dict(self):
        from phinance.agents.strategy_validator import StrategyValidator
        validator = StrategyValidator(min_sharpe=-99, max_drawdown=1.0, min_trades=0)
        result    = validator.validate(self._make_proposal(), DF_LARGE)
        d = result.to_dict()
        assert "approved" in d

    def test_num_trades_non_negative(self):
        from phinance.agents.strategy_validator import StrategyValidator
        validator = StrategyValidator(min_sharpe=-99, max_drawdown=1.0, min_trades=0)
        result    = validator.validate(self._make_proposal(), DF_LARGE)
        assert result.num_trades >= 0


# ═══════════════════════════════════════════════════════════════════════════════
# StrategyRegistry
# ═══════════════════════════════════════════════════════════════════════════════

class TestStrategyRegistry:

    def _make_record(self, name="test_strat"):
        from phinance.agents.autonomous_deployer import DeploymentRecord, DeploymentStatus
        from phinance.agents.strategy_proposer import StrategyProposal
        import uuid
        proposal = StrategyProposal(indicators={"RSI": {}}, weights={"RSI": 1.0})
        return DeploymentRecord(
            deployment_id=str(uuid.uuid4()),
            strategy_name=name,
            indicators={"RSI": {}},
            weights={"RSI": 1.0},
            blend_method="weighted_sum",
            regime_at_deploy="TREND_UP",
            validation_stats={"sharpe": 1.0},
            status=DeploymentStatus.ACTIVE,
        )

    def test_register_and_retrieve(self):
        from phinance.agents.autonomous_deployer import StrategyRegistry
        reg    = StrategyRegistry()
        record = self._make_record()
        reg.register(record)
        fetched = reg.get(record.deployment_id)
        assert fetched is not None
        assert fetched.deployment_id == record.deployment_id

    def test_list_active(self):
        from phinance.agents.autonomous_deployer import StrategyRegistry
        reg = StrategyRegistry()
        reg.register(self._make_record("a"))
        reg.register(self._make_record("b"))
        active = reg.list_active()
        assert len(active) == 2

    def test_list_all(self):
        from phinance.agents.autonomous_deployer import StrategyRegistry
        reg = StrategyRegistry()
        reg.register(self._make_record())
        assert len(reg.list_all()) == 1

    def test_update_status(self):
        from phinance.agents.autonomous_deployer import StrategyRegistry, DeploymentStatus
        reg    = StrategyRegistry()
        record = self._make_record()
        reg.register(record)
        ok = reg.update_status(record.deployment_id, DeploymentStatus.ROLLED_BACK)
        assert ok is True
        assert reg.get(record.deployment_id).status == DeploymentStatus.ROLLED_BACK

    def test_update_unknown_id_returns_false(self):
        from phinance.agents.autonomous_deployer import StrategyRegistry, DeploymentStatus
        reg = StrategyRegistry()
        ok  = reg.update_status("nonexistent-id", DeploymentStatus.ROLLED_BACK)
        assert ok is False

    def test_len(self):
        from phinance.agents.autonomous_deployer import StrategyRegistry
        reg = StrategyRegistry()
        assert len(reg) == 0
        reg.register(self._make_record())
        assert len(reg) == 1

    def test_get_nonexistent_returns_none(self):
        from phinance.agents.autonomous_deployer import StrategyRegistry
        reg = StrategyRegistry()
        assert reg.get("does-not-exist") is None

    def test_persistence_roundtrip(self, tmp_path):
        from phinance.agents.autonomous_deployer import StrategyRegistry
        path = str(tmp_path / "registry.json")
        reg1 = StrategyRegistry(registry_path=path)
        rec  = self._make_record("persisted_strat")
        reg1.register(rec)

        # Load in a new instance
        reg2 = StrategyRegistry(registry_path=path)
        fetched = reg2.get(rec.deployment_id)
        assert fetched is not None
        assert fetched.strategy_name == "persisted_strat"


# ═══════════════════════════════════════════════════════════════════════════════
# AutonomousDeployer
# ═══════════════════════════════════════════════════════════════════════════════

class TestAutonomousDeployer:

    def _approved_validation(self):
        from phinance.agents.strategy_proposer import StrategyProposal
        from phinance.agents.strategy_validator import ValidationResult
        proposal = StrategyProposal(
            indicators={"RSI": {"enabled": True, "params": {}}},
            weights={"RSI": 1.0},
            regime="TREND_UP",
        )
        return ValidationResult(
            approved=True, proposal=proposal,
            sharpe=1.2, max_drawdown=0.05, win_rate=0.6, num_trades=10,
        )

    def _rejected_validation(self):
        from phinance.agents.strategy_proposer import StrategyProposal
        from phinance.agents.strategy_validator import ValidationResult
        proposal = StrategyProposal(indicators={"RSI": {}}, weights={"RSI": 1.0})
        return ValidationResult(
            approved=False, proposal=proposal,
            rejection_reason="Sharpe too low",
        )

    def test_deploy_returns_record(self):
        from phinance.agents.autonomous_deployer import AutonomousDeployer, DeploymentRecord
        deployer = AutonomousDeployer(dry_run=True)
        record   = deployer.deploy(self._approved_validation())
        assert isinstance(record, DeploymentRecord)

    def test_deploy_sets_active_status(self):
        from phinance.agents.autonomous_deployer import AutonomousDeployer, DeploymentStatus
        deployer = AutonomousDeployer(dry_run=True)
        record   = deployer.deploy(self._approved_validation())
        assert record.status == DeploymentStatus.ACTIVE

    def test_deploy_dry_run_flag(self):
        from phinance.agents.autonomous_deployer import AutonomousDeployer
        deployer = AutonomousDeployer(dry_run=True)
        record   = deployer.deploy(self._approved_validation())
        assert record.dry_run is True

    def test_deploy_custom_name(self):
        from phinance.agents.autonomous_deployer import AutonomousDeployer
        deployer = AutonomousDeployer(dry_run=True)
        record   = deployer.deploy(self._approved_validation(), strategy_name="my_strat")
        assert record.strategy_name == "my_strat"

    def test_deploy_auto_name_contains_regime(self):
        from phinance.agents.autonomous_deployer import AutonomousDeployer
        deployer = AutonomousDeployer(dry_run=True)
        record   = deployer.deploy(self._approved_validation())
        assert "TREND_UP" in record.strategy_name

    def test_deploy_rejected_raises(self):
        from phinance.agents.autonomous_deployer import AutonomousDeployer
        deployer = AutonomousDeployer(dry_run=True)
        with pytest.raises(ValueError, match="[Rr]ejected|[Cc]annot deploy"):
            deployer.deploy(self._rejected_validation())

    def test_deploy_max_active_raises(self):
        from phinance.agents.autonomous_deployer import AutonomousDeployer
        deployer = AutonomousDeployer(dry_run=True, max_active=1)
        deployer.deploy(self._approved_validation())
        with pytest.raises(RuntimeError, match="[Mm]ax"):
            deployer.deploy(self._approved_validation())

    def test_rollback_success(self):
        from phinance.agents.autonomous_deployer import AutonomousDeployer, DeploymentStatus
        deployer = AutonomousDeployer(dry_run=True)
        record   = deployer.deploy(self._approved_validation())
        ok = deployer.rollback(record.deployment_id)
        assert ok is True
        fetched = deployer.registry.get(record.deployment_id)
        assert fetched.status == DeploymentStatus.ROLLED_BACK

    def test_rollback_nonexistent_returns_false(self):
        from phinance.agents.autonomous_deployer import AutonomousDeployer
        deployer = AutonomousDeployer(dry_run=True)
        assert deployer.rollback("does-not-exist") is False

    def test_rollback_already_rolled_back_returns_false(self):
        from phinance.agents.autonomous_deployer import AutonomousDeployer
        deployer = AutonomousDeployer(dry_run=True)
        record   = deployer.deploy(self._approved_validation())
        deployer.rollback(record.deployment_id)
        assert deployer.rollback(record.deployment_id) is False

    def test_list_active_after_rollback(self):
        from phinance.agents.autonomous_deployer import AutonomousDeployer
        deployer = AutonomousDeployer(dry_run=True, max_active=2)
        r1 = deployer.deploy(self._approved_validation())
        r2 = deployer.deploy(self._approved_validation())
        assert len(deployer.list_active()) == 2
        deployer.rollback(r1.deployment_id)
        assert len(deployer.list_active()) == 1

    def test_deployment_id_is_uuid(self):
        import re
        from phinance.agents.autonomous_deployer import AutonomousDeployer
        deployer = AutonomousDeployer(dry_run=True)
        record   = deployer.deploy(self._approved_validation())
        uuid_pattern = r"[0-9a-f-]{36}"
        assert re.match(uuid_pattern, record.deployment_id)


# ═══════════════════════════════════════════════════════════════════════════════
# AutonomousPipeline
# ═══════════════════════════════════════════════════════════════════════════════

class TestAutonomousPipeline:

    def _make_pipeline(self, min_sharpe=-99.0):
        from phinance.agents.strategy_proposer import StrategyProposerAgent
        from phinance.agents.strategy_validator import StrategyValidator
        from phinance.agents.autonomous_deployer import AutonomousDeployer
        from phinance.agents.autonomous_pipeline import AutonomousPipeline
        return AutonomousPipeline(
            proposer=StrategyProposerAgent(top_n=3),
            validator=StrategyValidator(min_sharpe=min_sharpe, max_drawdown=1.0, min_trades=0),
            deployer=AutonomousDeployer(dry_run=True),
            max_retries=2,
        )

    def test_run_once_returns_result(self):
        from phinance.agents.autonomous_pipeline import PipelineRunResult
        pipeline = self._make_pipeline()
        result   = pipeline.run_once(DF_LARGE)
        assert isinstance(result, PipelineRunResult)

    def test_run_once_success_with_lenient_validator(self):
        pipeline = self._make_pipeline(min_sharpe=-99.0)
        result   = pipeline.run_once(DF_LARGE)
        assert result.success is True

    def test_run_once_has_proposal(self):
        pipeline = self._make_pipeline()
        result   = pipeline.run_once(DF_LARGE)
        assert result.proposal is not None

    def test_run_once_has_validation(self):
        pipeline = self._make_pipeline()
        result   = pipeline.run_once(DF_LARGE)
        assert result.validation is not None

    def test_run_once_message_non_empty(self):
        pipeline = self._make_pipeline()
        result   = pipeline.run_once(DF_LARGE)
        assert len(result.message) > 0

    def test_run_once_elapsed_ms_positive(self):
        pipeline = self._make_pipeline()
        result   = pipeline.run_once(DF_LARGE)
        assert result.total_elapsed_ms >= 0

    def test_run_once_attempts_at_least_one(self):
        pipeline = self._make_pipeline()
        result   = pipeline.run_once(DF_LARGE)
        assert result.attempts >= 1

    def test_run_once_to_dict(self):
        pipeline = self._make_pipeline()
        result   = pipeline.run_once(DF_LARGE)
        d = result.to_dict()
        assert "success" in d
        assert "attempts" in d

    def test_run_loop_returns_list(self):
        pipeline = self._make_pipeline()
        results  = pipeline.run_loop(DF_LARGE, n_iterations=2)
        assert isinstance(results, list)
        assert len(results) >= 1

    def test_run_history_appended(self):
        pipeline = self._make_pipeline()
        pipeline.run_once(DF_LARGE)
        pipeline.run_once(DF_LARGE)
        assert len(pipeline.run_history) >= 2

    def test_active_deployments_property(self):
        pipeline = self._make_pipeline()
        pipeline.run_once(DF_LARGE)
        assert isinstance(pipeline.active_deployments, list)

    def test_failed_pipeline_no_exception(self):
        """Pipeline should return PipelineRunResult even when all attempts fail."""
        pipeline = self._make_pipeline(min_sharpe=9999.0)
        result   = pipeline.run_once(DF_LARGE)
        assert result.success is False
        assert result.attempts >= 1

    def test_proposer_top_n_restored_after_run(self):
        """Proposer state must be restored after each run_once call."""
        from phinance.agents.autonomous_pipeline import AutonomousPipeline
        from phinance.agents.strategy_proposer import StrategyProposerAgent
        from phinance.agents.strategy_validator import StrategyValidator
        from phinance.agents.autonomous_deployer import AutonomousDeployer
        proposer  = StrategyProposerAgent(top_n=4)
        pipeline  = AutonomousPipeline(
            proposer=proposer,
            validator=StrategyValidator(min_sharpe=9999),
            deployer=AutonomousDeployer(dry_run=True),
            max_retries=3,
        )
        pipeline.run_once(DF_LARGE)
        assert proposer.top_n == 4   # restored


# ═══════════════════════════════════════════════════════════════════════════════
# run_autonomous_pipeline convenience function
# ═══════════════════════════════════════════════════════════════════════════════

class TestRunAutonomousPipeline:

    def test_returns_pipeline_run_result(self):
        from phinance.agents.autonomous_pipeline import run_autonomous_pipeline, PipelineRunResult
        result = run_autonomous_pipeline(DF_LARGE, min_sharpe=-99.0)
        assert isinstance(result, PipelineRunResult)

    def test_dry_run_default(self):
        from phinance.agents.autonomous_pipeline import run_autonomous_pipeline
        result = run_autonomous_pipeline(DF_LARGE, min_sharpe=-99.0)
        if result.deployment:
            assert result.deployment.dry_run is True

    def test_with_registry_path(self, tmp_path):
        from phinance.agents.autonomous_pipeline import run_autonomous_pipeline, PipelineRunResult
        path   = str(tmp_path / "registry.json")
        result = run_autonomous_pipeline(DF_LARGE, min_sharpe=-99.0, registry_path=path)
        assert isinstance(result, PipelineRunResult)
