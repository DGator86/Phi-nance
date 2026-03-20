from phinance.agents.risk_monitor import RiskMonitorAgent
from phinance.agents.strategy_rd import StrategyRDAgent


def test_strategy_rd_contains_option_template():
    agent = StrategyRDAgent(use_rl=False)
    proposal = agent.propose_strategy({"regime": "bull", "volatility": 0.2, "recent_performance": 0.1})
    assert "name" in proposal
    assert any(t.get("name") == "option_strategy" for t in agent.template_library)


def test_risk_monitor_outputs_hedge_action():
    agent = RiskMonitorAgent(use_rl=False)
    out = agent.get_risk_limits(
        {"drawdown": 0.05, "var_95": 0.02, "beta": 1.0, "delta": 0.2, "gamma": 0.1, "vega": 0.1, "correlation": 0.2, "leverage_ratio": 0.5, "rebalance_age": 0.1},
        {"regime": "bear", "volatility": 0.4, "exploration_count": 0.2},
    )
    assert "hedge_action" in out
    assert "hedge_template" in out
