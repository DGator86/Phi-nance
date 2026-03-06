from __future__ import annotations

from phinance.rl.hierarchical.options import (
    Option,
    execution_initiation,
    execution_termination,
    risk_monitor_initiation,
    risk_monitor_termination,
    strategy_rd_initiation,
)


class _DummyPolicy:
    def act(self, state, deterministic: bool = True):  # noqa: ANN001,ARG002
        return 1


def test_execution_option_initiation_and_termination() -> None:
    context = {"order": {"status": "open", "remaining_shares": 50}}
    assert execution_initiation(context)
    assert not execution_termination(context, {"remaining_shares": 10})
    assert execution_termination(context, {"remaining_shares": 0})


def test_strategy_initiation_interval() -> None:
    assert strategy_rd_initiation({"global_step": 20, "strategy_interval": 10})
    assert not strategy_rd_initiation({"global_step": 21, "strategy_interval": 10})


def test_risk_monitor_is_always_available() -> None:
    assert risk_monitor_initiation({})
    assert not risk_monitor_termination({}, {})


def test_option_max_steps_termination() -> None:
    option = Option(
        name="dummy",
        policy=_DummyPolicy(),
        initiation_condition=lambda ctx: True,
        termination_condition=lambda ctx, info: False,
        max_steps=2,
    )
    assert not option.should_terminate({"option_elapsed": 1}, {})
    assert option.should_terminate({"option_elapsed": 2}, {})
