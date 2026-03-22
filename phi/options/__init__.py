"""Public API for options pricing and backtesting."""

from phi.logging import get_logger

logger = get_logger(__name__)

from .backtest import compute_greeks, run_options_backtest
from .contract import OptionContract, OptionType
from .data_adapter import adapt_for_backtesting, fetch_options_data
from .market import fetch_options_market_data
from .position import OptionPosition
from .pricing import black_scholes_price, delta, gamma, theta, vega
from .regime_playbook import (
    OptionsRegimePlaybook,
    RegimePlaybookEntry,
    build_default_options_regime_playbook,
    get_default_options_regime_playbook,
    load_options_regime_playbook,
    playbook_entry_for_label,
    playbook_to_summary_dict,
    quick_detailed_regime_from_ohlcv,
    resolve_playbook_regime_key,
)
from .regime_strategy_map import (
    APPROVED_STRATEGIES,
    REGIME_STRATEGY_MAP,
    is_approved_strategy,
    map_regime_probabilities_to_strategies,
    strategies_for_regime,
)
from .signal_card import OptionsSignalCard
from .signal_generator import build_options_signal_card

__all__ = [
    "OptionType",
    "OptionContract",
    "OptionPosition",
    "black_scholes_price",
    "delta",
    "gamma",
    "vega",
    "theta",
    "compute_greeks",
    "run_options_backtest",
    "fetch_options_market_data",
    "adapt_for_backtesting",
    "fetch_options_data",
    "APPROVED_STRATEGIES",
    "REGIME_STRATEGY_MAP",
    "strategies_for_regime",
    "map_regime_probabilities_to_strategies",
    "is_approved_strategy",
    "OptionsRegimePlaybook",
    "RegimePlaybookEntry",
    "build_default_options_regime_playbook",
    "get_default_options_regime_playbook",
    "load_options_regime_playbook",
    "playbook_entry_for_label",
    "playbook_to_summary_dict",
    "quick_detailed_regime_from_ohlcv",
    "resolve_playbook_regime_key",
    "OptionsSignalCard",
    "build_options_signal_card",
]
