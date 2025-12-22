"""
Utilities Package for Options Backtesting

This package provides utility functions and helpers for options
trading strategies and backtesting.

Core Components:
    conditions: Condition helper functions for strategy entry/exit logic

Usage:
    from backtester.utils.conditions import (
        calculate_iv_percentile,
        is_major_event_date,
        check_position_limit,
        days_to_expiry,
        calculate_profit_pct
    )

    # Calculate IV percentile
    iv_pct = calculate_iv_percentile(current_iv, historical_ivs)

    # Check position limits
    can_add = check_position_limit(current_count=3, max_count=5)
"""

from backtester.utils.conditions import (
    # IV calculations
    calculate_iv_percentile,
    calculate_iv_rank,

    # Event checking
    is_major_event_date,
    get_upcoming_events,

    # Position limits
    check_position_limit,
    check_capital_limit,
    check_delta_limit,

    # Time calculations
    days_to_expiry,
    is_expiration_day,
    is_within_dte_range,

    # Profit calculations
    calculate_profit_pct,
    has_reached_profit_target,
    has_reached_stop_loss,

    # VIX conditions
    is_vix_above_threshold,
    is_vix_below_threshold,
    is_vix_in_range,

    # Condition builders
    create_threshold_condition,
    create_range_condition,
    combine_conditions,
    negate_condition,

    # Exceptions
    ConditionError,
    InvalidInputError,

    # Constants
    DEFAULT_IV_WINDOW,
    DEFAULT_EVENT_BUFFER_DAYS,
    EVENT_TYPE_EARNINGS,
    EVENT_TYPE_FOMC,
    EVENT_TYPE_CPI,
    EVENT_TYPE_JOBS,
    EVENT_TYPE_DIVIDEND,
    EVENT_TYPE_EXPIRATION,
)

__all__ = [
    # IV calculations
    'calculate_iv_percentile',
    'calculate_iv_rank',

    # Event checking
    'is_major_event_date',
    'get_upcoming_events',

    # Position limits
    'check_position_limit',
    'check_capital_limit',
    'check_delta_limit',

    # Time calculations
    'days_to_expiry',
    'is_expiration_day',
    'is_within_dte_range',

    # Profit calculations
    'calculate_profit_pct',
    'has_reached_profit_target',
    'has_reached_stop_loss',

    # VIX conditions
    'is_vix_above_threshold',
    'is_vix_below_threshold',
    'is_vix_in_range',

    # Condition builders
    'create_threshold_condition',
    'create_range_condition',
    'combine_conditions',
    'negate_condition',

    # Exceptions
    'ConditionError',
    'InvalidInputError',

    # Constants
    'DEFAULT_IV_WINDOW',
    'DEFAULT_EVENT_BUFFER_DAYS',
    'EVENT_TYPE_EARNINGS',
    'EVENT_TYPE_FOMC',
    'EVENT_TYPE_CPI',
    'EVENT_TYPE_JOBS',
    'EVENT_TYPE_DIVIDEND',
    'EVENT_TYPE_EXPIRATION',
]
