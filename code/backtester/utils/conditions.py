"""
Condition Helper Utilities for Options Strategies

This module provides reusable condition helper functions for building
trading strategy entry and exit logic. These utilities handle common
calculations needed for options trading decisions.

Key Features:
    - IV percentile/rank calculations
    - Event date checking (earnings, FOMC, etc.)
    - Position limit validation
    - Days-to-expiry calculation
    - Profit percentage calculation
    - Generic condition builders

Usage:
    from backtester.utils.conditions import (
        calculate_iv_percentile,
        is_major_event_date,
        check_position_limit,
        days_to_expiry,
        calculate_profit_pct
    )

    # Check if IV is elevated
    iv_pct = calculate_iv_percentile(current_iv=0.25, historical_ivs=[0.15, 0.20, 0.30])

    # Check if today is an event day
    is_event = is_major_event_date(date, event_calendar)

    # Check position limits
    can_add = check_position_limit(current_count=3, max_count=5)

References:
    - Tastytrade Research: IV Rank methodology
    - CBOE VIX calculations
"""

import logging
from datetime import date, datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np

# Configure module logger
logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Default IV percentile window
DEFAULT_IV_WINDOW = 252  # One year of trading days

# Common event types
EVENT_TYPE_EARNINGS = 'earnings'
EVENT_TYPE_FOMC = 'fomc'
EVENT_TYPE_CPI = 'cpi'
EVENT_TYPE_JOBS = 'jobs'
EVENT_TYPE_DIVIDEND = 'dividend'
EVENT_TYPE_EXPIRATION = 'expiration'

# Days to avoid around events
DEFAULT_EVENT_BUFFER_DAYS = 1


# =============================================================================
# Exceptions
# =============================================================================

class ConditionError(Exception):
    """Base exception for condition utility errors."""
    pass


class InvalidInputError(ConditionError):
    """Exception raised for invalid input values."""
    pass


# =============================================================================
# IV Percentile and Rank Calculations
# =============================================================================

def calculate_iv_percentile(
    current_iv: float,
    historical_ivs: Sequence[float],
    window: Optional[int] = None
) -> float:
    """
    Calculate IV percentile (IV rank) based on historical IV values.

    IV Percentile represents where the current IV stands relative to
    its historical range. A percentile of 50 means current IV is at the
    median of historical values.

    Formula:
        IV Percentile = (Current IV - Min IV) / (Max IV - Min IV) * 100

    Alternatively, this function calculates the percentile rank:
        Percentile = (Count of values below current) / Total count * 100

    Args:
        current_iv: Current implied volatility (decimal, e.g., 0.20 for 20%)
        historical_ivs: Sequence of historical IV values
        window: Number of historical values to use. If None, uses all.
                Typically 252 for one year of trading days.

    Returns:
        IV percentile from 0 to 100

    Raises:
        InvalidInputError: If inputs are invalid

    Example:
        >>> historical = [0.15, 0.18, 0.22, 0.25, 0.30]
        >>> current = 0.20
        >>> percentile = calculate_iv_percentile(current, historical)
        >>> print(f"IV Percentile: {percentile:.1f}")
        IV Percentile: 33.3

    Note:
        - Returns 0 if current IV is at or below historical minimum
        - Returns 100 if current IV is at or above historical maximum
        - Returns 50 if only one historical value equals current
    """
    # Validate inputs
    if current_iv is None:
        raise InvalidInputError("current_iv cannot be None")
    if not np.isfinite(current_iv):
        raise InvalidInputError(f"current_iv must be finite, got {current_iv}")
    if current_iv < 0:
        raise InvalidInputError(f"current_iv must be non-negative, got {current_iv}")

    if historical_ivs is None or len(historical_ivs) == 0:
        raise InvalidInputError("historical_ivs cannot be empty")

    # Convert to numpy array and apply window
    iv_array = np.asarray(historical_ivs, dtype=np.float64)

    # Filter out non-finite values
    iv_array = iv_array[np.isfinite(iv_array)]

    if len(iv_array) == 0:
        raise InvalidInputError("historical_ivs contains no valid values")

    # Apply window if specified
    if window is not None and window > 0:
        iv_array = iv_array[-window:]

    # Calculate range
    iv_min = np.min(iv_array)
    iv_max = np.max(iv_array)

    # Handle edge cases
    if iv_max == iv_min:
        # No range in historical data
        if current_iv == iv_min:
            return 50.0
        elif current_iv > iv_max:
            return 100.0
        else:
            return 0.0

    # Calculate percentile using range method
    percentile = (current_iv - iv_min) / (iv_max - iv_min) * 100.0

    # Clamp to 0-100
    return max(0.0, min(100.0, percentile))


def calculate_iv_rank(
    current_iv: float,
    historical_ivs: Sequence[float],
    window: Optional[int] = None
) -> float:
    """
    Calculate IV rank (percentile rank) based on historical IV values.

    IV Rank represents the percentage of historical values that are below
    the current IV. This is slightly different from IV Percentile which
    uses the range method.

    Formula:
        IV Rank = (Count of values <= current) / Total count * 100

    Args:
        current_iv: Current implied volatility (decimal)
        historical_ivs: Sequence of historical IV values
        window: Number of historical values to use. If None, uses all.

    Returns:
        IV rank from 0 to 100

    Raises:
        InvalidInputError: If inputs are invalid

    Example:
        >>> historical = [0.15, 0.18, 0.22, 0.25, 0.30]
        >>> current = 0.20
        >>> rank = calculate_iv_rank(current, historical)
        >>> print(f"IV Rank: {rank:.1f}")
        IV Rank: 40.0
    """
    # Validate inputs
    if current_iv is None:
        raise InvalidInputError("current_iv cannot be None")
    if not np.isfinite(current_iv):
        raise InvalidInputError(f"current_iv must be finite, got {current_iv}")

    if historical_ivs is None or len(historical_ivs) == 0:
        raise InvalidInputError("historical_ivs cannot be empty")

    # Convert to numpy array
    iv_array = np.asarray(historical_ivs, dtype=np.float64)
    iv_array = iv_array[np.isfinite(iv_array)]

    if len(iv_array) == 0:
        raise InvalidInputError("historical_ivs contains no valid values")

    # Apply window if specified
    if window is not None and window > 0:
        iv_array = iv_array[-window:]

    # Calculate rank (percentage of values <= current)
    count_below_or_equal = np.sum(iv_array <= current_iv)
    rank = count_below_or_equal / len(iv_array) * 100.0

    return rank


# =============================================================================
# Event Date Checking
# =============================================================================

def is_major_event_date(
    check_date: Union[date, datetime],
    event_calendar: Optional[Dict[str, List[Union[date, datetime]]]] = None,
    buffer_days: int = DEFAULT_EVENT_BUFFER_DAYS,
    event_types: Optional[List[str]] = None
) -> bool:
    """
    Check if a date is on or near a major market event.

    This function checks whether the given date falls within a buffer
    period of any scheduled market events (earnings, FOMC, CPI, etc.).

    Args:
        check_date: Date to check
        event_calendar: Dictionary mapping event types to lists of event dates.
            Example: {'earnings': [date(2024, 1, 25), ...], 'fomc': [...]}
            If None, always returns False.
        buffer_days: Number of days before/after event to avoid.
            Default is 1 (avoid day before, day of, and day after).
        event_types: List of event types to check. If None, checks all.
            Valid types: 'earnings', 'fomc', 'cpi', 'jobs', 'dividend', 'expiration'

    Returns:
        True if date is on or near a major event, False otherwise

    Example:
        >>> calendar = {
        ...     'earnings': [date(2024, 1, 25)],
        ...     'fomc': [date(2024, 1, 31)]
        ... }
        >>> is_major_event_date(date(2024, 1, 25), calendar)
        True
        >>> is_major_event_date(date(2024, 1, 20), calendar)
        False
    """
    if event_calendar is None:
        return False

    # Normalize check_date to date object
    if isinstance(check_date, datetime):
        check_date = check_date.date()

    # Determine which event types to check
    types_to_check = event_types if event_types else list(event_calendar.keys())

    for event_type in types_to_check:
        if event_type not in event_calendar:
            continue

        event_dates = event_calendar[event_type]

        for event_date in event_dates:
            # Normalize event_date
            if isinstance(event_date, datetime):
                event_date = event_date.date()

            # Calculate date range to avoid
            start_avoid = event_date - timedelta(days=buffer_days)
            end_avoid = event_date + timedelta(days=buffer_days)

            if start_avoid <= check_date <= end_avoid:
                logger.debug(
                    f"Date {check_date} is near {event_type} event on {event_date}"
                )
                return True

    return False


def get_upcoming_events(
    start_date: Union[date, datetime],
    end_date: Union[date, datetime],
    event_calendar: Optional[Dict[str, List[Union[date, datetime]]]] = None,
    event_types: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Get list of events within a date range.

    Args:
        start_date: Start of date range (inclusive)
        end_date: End of date range (inclusive)
        event_calendar: Dictionary of event types to event dates
        event_types: List of event types to include. If None, includes all.

    Returns:
        List of dictionaries with event information:
            - 'date': Event date
            - 'type': Event type string

    Example:
        >>> events = get_upcoming_events(
        ...     date(2024, 1, 1), date(2024, 1, 31), calendar
        ... )
        >>> for e in events:
        ...     print(f"{e['type']}: {e['date']}")
    """
    if event_calendar is None:
        return []

    # Normalize dates
    if isinstance(start_date, datetime):
        start_date = start_date.date()
    if isinstance(end_date, datetime):
        end_date = end_date.date()

    events = []
    types_to_check = event_types if event_types else list(event_calendar.keys())

    for event_type in types_to_check:
        if event_type not in event_calendar:
            continue

        for event_date in event_calendar[event_type]:
            if isinstance(event_date, datetime):
                event_date = event_date.date()

            if start_date <= event_date <= end_date:
                events.append({
                    'date': event_date,
                    'type': event_type
                })

    # Sort by date
    events.sort(key=lambda x: x['date'])
    return events


# =============================================================================
# Position Limit Checking
# =============================================================================

def check_position_limit(
    current_count: int,
    max_count: int
) -> bool:
    """
    Check if a new position can be added within limits.

    Args:
        current_count: Current number of positions
        max_count: Maximum allowed positions

    Returns:
        True if a new position can be added, False otherwise

    Example:
        >>> check_position_limit(current_count=3, max_count=5)
        True
        >>> check_position_limit(current_count=5, max_count=5)
        False
    """
    if current_count is None or max_count is None:
        raise InvalidInputError("position counts cannot be None")
    if current_count < 0 or max_count < 0:
        raise InvalidInputError("position counts must be non-negative")

    return current_count < max_count


def check_capital_limit(
    current_allocated: float,
    position_size: float,
    max_capital: float
) -> bool:
    """
    Check if a new position can be added within capital limits.

    Args:
        current_allocated: Currently allocated capital
        position_size: Size of new position to add
        max_capital: Maximum allowed capital allocation

    Returns:
        True if new position can be added within limits

    Example:
        >>> check_capital_limit(
        ...     current_allocated=50000,
        ...     position_size=10000,
        ...     max_capital=100000
        ... )
        True
    """
    if current_allocated is None or position_size is None or max_capital is None:
        raise InvalidInputError("capital values cannot be None")

    return (current_allocated + position_size) <= max_capital


def check_delta_limit(
    current_delta: float,
    position_delta: float,
    max_delta: float
) -> bool:
    """
    Check if a new position can be added within delta limits.

    Args:
        current_delta: Current portfolio delta
        position_delta: Delta of new position
        max_delta: Maximum allowed absolute delta

    Returns:
        True if new position can be added within delta limits

    Example:
        >>> check_delta_limit(
        ...     current_delta=25.0,
        ...     position_delta=-10.0,
        ...     max_delta=50.0
        ... )
        True
    """
    if current_delta is None or position_delta is None or max_delta is None:
        raise InvalidInputError("delta values cannot be None")

    new_delta = current_delta + position_delta
    return abs(new_delta) <= max_delta


# =============================================================================
# Time and Expiry Calculations
# =============================================================================

def days_to_expiry(
    expiration: Union[date, datetime],
    current_date: Optional[Union[date, datetime]] = None
) -> int:
    """
    Calculate calendar days until expiration.

    Args:
        expiration: Option expiration date
        current_date: Reference date. If None, uses today.

    Returns:
        Number of calendar days to expiration (0 if expired)

    Example:
        >>> exp = date(2024, 3, 15)
        >>> current = date(2024, 3, 1)
        >>> dte = days_to_expiry(exp, current)
        >>> print(f"DTE: {dte}")
        DTE: 14
    """
    if expiration is None:
        raise InvalidInputError("expiration cannot be None")

    # Normalize to date objects
    if isinstance(expiration, datetime):
        expiration = expiration.date()
    if current_date is None:
        current_date = date.today()
    elif isinstance(current_date, datetime):
        current_date = current_date.date()

    delta = expiration - current_date
    return max(delta.days, 0)


def is_expiration_day(
    check_date: Union[date, datetime],
    expiration: Union[date, datetime]
) -> bool:
    """
    Check if a date is the expiration date.

    Args:
        check_date: Date to check
        expiration: Option expiration date

    Returns:
        True if dates match, False otherwise
    """
    if isinstance(check_date, datetime):
        check_date = check_date.date()
    if isinstance(expiration, datetime):
        expiration = expiration.date()

    return check_date == expiration


def is_within_dte_range(
    expiration: Union[date, datetime],
    current_date: Optional[Union[date, datetime]] = None,
    min_dte: int = 0,
    max_dte: int = 365
) -> bool:
    """
    Check if option DTE is within specified range.

    Args:
        expiration: Option expiration date
        current_date: Reference date. If None, uses today.
        min_dte: Minimum days to expiry (inclusive)
        max_dte: Maximum days to expiry (inclusive)

    Returns:
        True if DTE is within range

    Example:
        >>> exp = date(2024, 3, 15)
        >>> current = date(2024, 3, 1)
        >>> is_within_dte_range(exp, current, min_dte=7, max_dte=30)
        True
    """
    dte = days_to_expiry(expiration, current_date)
    return min_dte <= dte <= max_dte


# =============================================================================
# Profit and P&L Calculations
# =============================================================================

def calculate_profit_pct(
    pnl: float,
    initial_premium: float
) -> float:
    """
    Calculate profit as percentage of initial premium.

    For credit strategies, profit percentage represents how much
    of the initial credit has been captured (or lost if negative).

    Args:
        pnl: Current profit/loss in dollars
        initial_premium: Initial premium received/paid (absolute value)

    Returns:
        Profit as decimal (e.g., 0.25 for 25% profit)

    Raises:
        InvalidInputError: If initial_premium is zero

    Example:
        >>> # Collected $500 premium, now up $125
        >>> calculate_profit_pct(pnl=125, initial_premium=500)
        0.25
        >>> # Collected $500 premium, now down $250
        >>> calculate_profit_pct(pnl=-250, initial_premium=500)
        -0.5
    """
    if initial_premium is None:
        raise InvalidInputError("initial_premium cannot be None")
    if abs(initial_premium) < 1e-10:
        raise InvalidInputError("initial_premium cannot be zero")
    if pnl is None:
        raise InvalidInputError("pnl cannot be None")

    return pnl / abs(initial_premium)


def has_reached_profit_target(
    pnl: float,
    initial_premium: float,
    target_pct: float = 0.25
) -> bool:
    """
    Check if profit target has been reached.

    Args:
        pnl: Current P&L in dollars
        initial_premium: Initial premium received/paid
        target_pct: Target profit as decimal (default 0.25 = 25%)

    Returns:
        True if profit target reached

    Example:
        >>> has_reached_profit_target(pnl=130, initial_premium=500, target_pct=0.25)
        True
    """
    profit_pct = calculate_profit_pct(pnl, initial_premium)
    return profit_pct >= target_pct


def has_reached_stop_loss(
    pnl: float,
    initial_premium: float,
    stop_pct: float = -1.0
) -> bool:
    """
    Check if stop loss has been triggered.

    Args:
        pnl: Current P&L in dollars
        initial_premium: Initial premium received/paid
        stop_pct: Stop loss as decimal (default -1.0 = 100% loss)

    Returns:
        True if stop loss triggered

    Example:
        >>> has_reached_stop_loss(pnl=-600, initial_premium=500, stop_pct=-1.0)
        True
    """
    profit_pct = calculate_profit_pct(pnl, initial_premium)
    return profit_pct <= stop_pct


# =============================================================================
# VIX and Volatility Conditions
# =============================================================================

def is_vix_above_threshold(
    vix: float,
    threshold: float = 20.0
) -> bool:
    """
    Check if VIX is above a threshold.

    Higher VIX typically indicates elevated implied volatility,
    which may be favorable for premium selling strategies.

    Args:
        vix: Current VIX level
        threshold: VIX threshold (default 20)

    Returns:
        True if VIX is above threshold

    Example:
        >>> is_vix_above_threshold(vix=25.5, threshold=20)
        True
    """
    if vix is None or threshold is None:
        raise InvalidInputError("vix and threshold cannot be None")

    return vix > threshold


def is_vix_below_threshold(
    vix: float,
    threshold: float = 15.0
) -> bool:
    """
    Check if VIX is below a threshold.

    Lower VIX typically indicates low implied volatility,
    which may be favorable for premium buying strategies.

    Args:
        vix: Current VIX level
        threshold: VIX threshold (default 15)

    Returns:
        True if VIX is below threshold
    """
    if vix is None or threshold is None:
        raise InvalidInputError("vix and threshold cannot be None")

    return vix < threshold


def is_vix_in_range(
    vix: float,
    min_vix: float = 15.0,
    max_vix: float = 30.0
) -> bool:
    """
    Check if VIX is within a specified range.

    Args:
        vix: Current VIX level
        min_vix: Minimum VIX (inclusive)
        max_vix: Maximum VIX (inclusive)

    Returns:
        True if VIX is within range
    """
    if vix is None:
        raise InvalidInputError("vix cannot be None")

    return min_vix <= vix <= max_vix


# =============================================================================
# Generic Condition Builders
# =============================================================================

def create_threshold_condition(
    key: str,
    threshold: float,
    comparison: str = 'above'
) -> Callable[[Dict[str, Any]], bool]:
    """
    Create a threshold-based condition function.

    Args:
        key: Key to look up in market_data dictionary
        threshold: Threshold value for comparison
        comparison: 'above', 'below', 'equal', 'above_or_equal', 'below_or_equal'

    Returns:
        Callable that takes market_data dict and returns bool

    Example:
        >>> cond = create_threshold_condition('iv_percentile', 50, 'above')
        >>> cond({'iv_percentile': 60})
        True
    """
    comparisons = {
        'above': lambda x, t: x > t,
        'below': lambda x, t: x < t,
        'equal': lambda x, t: x == t,
        'above_or_equal': lambda x, t: x >= t,
        'below_or_equal': lambda x, t: x <= t,
    }

    if comparison not in comparisons:
        raise InvalidInputError(
            f"Invalid comparison: {comparison}. "
            f"Must be one of: {list(comparisons.keys())}"
        )

    compare_func = comparisons[comparison]

    def condition(market_data: Dict[str, Any]) -> bool:
        value = market_data.get(key)
        if value is None:
            return False
        return compare_func(value, threshold)

    return condition


def create_range_condition(
    key: str,
    min_value: float,
    max_value: float
) -> Callable[[Dict[str, Any]], bool]:
    """
    Create a range-based condition function.

    Args:
        key: Key to look up in market_data dictionary
        min_value: Minimum value (inclusive)
        max_value: Maximum value (inclusive)

    Returns:
        Callable that takes market_data dict and returns bool

    Example:
        >>> cond = create_range_condition('vix', 15, 30)
        >>> cond({'vix': 25})
        True
    """
    def condition(market_data: Dict[str, Any]) -> bool:
        value = market_data.get(key)
        if value is None:
            return False
        return min_value <= value <= max_value

    return condition


def combine_conditions(
    conditions: List[Callable[[Dict[str, Any]], bool]],
    logic: str = 'and'
) -> Callable[[Dict[str, Any]], bool]:
    """
    Combine multiple conditions with AND/OR logic.

    Args:
        conditions: List of condition functions
        logic: 'and' (all must be true) or 'or' (any must be true)

    Returns:
        Combined condition function

    Example:
        >>> cond1 = create_threshold_condition('iv_percentile', 50, 'above')
        >>> cond2 = create_threshold_condition('vix', 20, 'above')
        >>> combined = combine_conditions([cond1, cond2], 'and')
        >>> combined({'iv_percentile': 60, 'vix': 25})
        True
    """
    if logic not in ('and', 'or'):
        raise InvalidInputError(f"logic must be 'and' or 'or', got '{logic}'")

    if logic == 'and':
        def combined(market_data: Dict[str, Any]) -> bool:
            return all(cond(market_data) for cond in conditions)
    else:
        def combined(market_data: Dict[str, Any]) -> bool:
            return any(cond(market_data) for cond in conditions)

    return combined


def negate_condition(
    condition: Callable[[Dict[str, Any]], bool]
) -> Callable[[Dict[str, Any]], bool]:
    """
    Negate a condition (logical NOT).

    Args:
        condition: Condition function to negate

    Returns:
        Negated condition function

    Example:
        >>> cond = create_threshold_condition('vix', 30, 'above')
        >>> not_cond = negate_condition(cond)
        >>> not_cond({'vix': 25})  # VIX not above 30
        True
    """
    def negated(market_data: Dict[str, Any]) -> bool:
        return not condition(market_data)

    return negated


# =============================================================================
# Module Exports
# =============================================================================

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
