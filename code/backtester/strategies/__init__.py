"""
Strategies Package for Options Backtesting

This package provides the Strategy base class, concrete strategy implementations,
and the StrategyBuilder fluent API for declarative strategy creation.

Core Components:
    Strategy: Abstract base class for all trading strategies
    StrategyBuilder: Fluent API for declarative strategy creation

Example Strategies:
    ShortStraddleHighIVStrategy: Sell ATM straddles when IV rank is high
    IronCondorStrategy: Sell iron condors with delta-based strike selection
    VolatilityRegimeStrategy: Adaptive strategy based on VIX regime

StrategyBuilder Usage:
    from backtester.strategies import (
        StrategyBuilder,
        iv_rank_above, profit_target, stop_loss, dte_below,
        short_straddle, risk_percent
    )

    strategy = (StrategyBuilder()
        .name("High IV Short Straddle")
        .underlying("SPY")
        .entry_condition(iv_rank_above(70))
        .structure(short_straddle(dte=30))
        .exit_condition(profit_target(0.50) | stop_loss(2.0) | dte_below(7))
        .position_size(risk_percent(0.02))
        .build())
"""

from backtester.strategies.strategy import (
    Strategy,
    StrategyError,
    StrategyValidationError,
    PositionError,
    RiskLimitError,
    InsufficientCapitalError,
    DEFAULT_MAX_POSITIONS,
    DEFAULT_MAX_TOTAL_DELTA,
    DEFAULT_MAX_TOTAL_VEGA,
    DEFAULT_MAX_CAPITAL_UTILIZATION,
    DEFAULT_MARGIN_PER_CONTRACT,
)

from backtester.strategies.short_straddle_strategy import ShortStraddleHighIVStrategy
from backtester.strategies.iron_condor_strategy import IronCondorStrategy
from backtester.strategies.volatility_regime_strategy import (
    VolatilityRegimeStrategy,
    VolatilityRegime,
)

from backtester.strategies.strategy_builder import (
    StrategyBuilder,
    BuiltStrategy,
    Condition,
    AndCondition,
    OrCondition,
    NotCondition,
    AlwaysTrue,
    AlwaysFalse,
    IVRankAbove,
    IVRankBelow,
    IVRankBetween,
    VIXAbove,
    VIXBelow,
    VIXBetween,
    DTEAbove,
    DTEBelow,
    DTEBetween,
    DayOfWeek,
    NoOpenPositions,
    MaxOpenPositions,
    ProfitTarget,
    StopLoss,
    FixedStopLoss,
    TrailingStop,
    HoldingPeriod,
    ExpirationApproaching,
    iv_rank_above,
    iv_rank_below,
    iv_rank_between,
    vix_above,
    vix_below,
    vix_between,
    dte_above,
    dte_below,
    dte_between,
    day_of_week,
    no_open_positions,
    max_open_positions,
    profit_target,
    stop_loss,
    fixed_stop_loss,
    trailing_stop,
    holding_period,
    expiration_approaching,
    StructureSpec,
    short_straddle,
    long_straddle,
    short_strangle,
    long_strangle,
    iron_condor,
    bull_call_spread,
    bear_put_spread,
    PositionSizer,
    fixed_contracts,
    risk_percent,
    capital_percent,
    delta_target,
    premium_target,
    StrategyBuilderError,
    BuilderValidationError,
    ConditionError,
)

__all__ = [
    # Base Strategy
    "Strategy",
    "StrategyError",
    "StrategyValidationError",
    "PositionError",
    "RiskLimitError",
    "InsufficientCapitalError",
    "DEFAULT_MAX_POSITIONS",
    "DEFAULT_MAX_TOTAL_DELTA",
    "DEFAULT_MAX_TOTAL_VEGA",
    "DEFAULT_MAX_CAPITAL_UTILIZATION",
    "DEFAULT_MARGIN_PER_CONTRACT",
    # Example Strategies
    "ShortStraddleHighIVStrategy",
    "IronCondorStrategy",
    "VolatilityRegimeStrategy",
    "VolatilityRegime",
    # Strategy Builder
    "StrategyBuilder",
    "BuiltStrategy",
    # Condition Classes
    "Condition",
    "AndCondition",
    "OrCondition",
    "NotCondition",
    "AlwaysTrue",
    "AlwaysFalse",
    # Entry Conditions
    "IVRankAbove",
    "IVRankBelow",
    "IVRankBetween",
    "VIXAbove",
    "VIXBelow",
    "VIXBetween",
    "DTEAbove",
    "DTEBelow",
    "DTEBetween",
    "DayOfWeek",
    "NoOpenPositions",
    "MaxOpenPositions",
    # Exit Conditions
    "ProfitTarget",
    "StopLoss",
    "FixedStopLoss",
    "TrailingStop",
    "HoldingPeriod",
    "ExpirationApproaching",
    # Condition Factories
    "iv_rank_above",
    "iv_rank_below",
    "iv_rank_between",
    "vix_above",
    "vix_below",
    "vix_between",
    "dte_above",
    "dte_below",
    "dte_between",
    "day_of_week",
    "no_open_positions",
    "max_open_positions",
    "profit_target",
    "stop_loss",
    "fixed_stop_loss",
    "trailing_stop",
    "holding_period",
    "expiration_approaching",
    # Structure Factories
    "StructureSpec",
    "short_straddle",
    "long_straddle",
    "short_strangle",
    "long_strangle",
    "iron_condor",
    "bull_call_spread",
    "bear_put_spread",
    # Position Sizing
    "PositionSizer",
    "fixed_contracts",
    "risk_percent",
    "capital_percent",
    "delta_target",
    "premium_target",
    # Builder Exceptions
    "StrategyBuilderError",
    "BuilderValidationError",
    "ConditionError",
]
