"""
Strategies Package for Options Backtesting

This package provides the Strategy base class and concrete strategy implementations
for options trading.

Core Components:
    Strategy: Abstract base class for all trading strategies

Example Strategies:
    ShortStraddleHighIVStrategy: Sell ATM straddles when IV rank is high
    IronCondorStrategy: Sell iron condors with delta-based strike selection
    VolatilityRegimeStrategy: Adaptive strategy based on VIX regime

Usage:
    from backtester.strategies import ShortStraddleHighIVStrategy

    strategy = ShortStraddleHighIVStrategy(
        name='Short Straddle IV70',
        initial_capital=100000,
        iv_rank_threshold=70,
        profit_target_pct=0.50
    )

    # Check entry conditions
    if strategy.should_enter(market_data):
        # Create position
        pass
"""

from backtester.strategies.strategy import (
    # Main class
    Strategy,

    # Exceptions
    StrategyError,
    StrategyValidationError,
    PositionError,
    RiskLimitError,
    InsufficientCapitalError,

    # Constants
    DEFAULT_MAX_POSITIONS,
    DEFAULT_MAX_TOTAL_DELTA,
    DEFAULT_MAX_TOTAL_VEGA,
    DEFAULT_MAX_CAPITAL_UTILIZATION,
    DEFAULT_MARGIN_PER_CONTRACT,
)

# Example strategy implementations
from backtester.strategies.short_straddle_strategy import ShortStraddleHighIVStrategy
from backtester.strategies.iron_condor_strategy import IronCondorStrategy
from backtester.strategies.volatility_regime_strategy import (
    VolatilityRegimeStrategy,
    VolatilityRegime,
)

__all__ = [
    # Main class
    'Strategy',

    # Exceptions
    'StrategyError',
    'StrategyValidationError',
    'PositionError',
    'RiskLimitError',
    'InsufficientCapitalError',

    # Constants
    'DEFAULT_MAX_POSITIONS',
    'DEFAULT_MAX_TOTAL_DELTA',
    'DEFAULT_MAX_TOTAL_VEGA',
    'DEFAULT_MAX_CAPITAL_UTILIZATION',
    'DEFAULT_MARGIN_PER_CONTRACT',

    # Example Strategies
    'ShortStraddleHighIVStrategy',
    'IronCondorStrategy',
    'VolatilityRegimeStrategy',
    'VolatilityRegime',
]
