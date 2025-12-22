"""
Comprehensive Tests for Condition Helper Utilities

This module provides thorough testing of the condition helper utilities
used for strategy entry/exit logic including:
- IV percentile/rank calculation
- Event date checking
- Position limit checking
- DTE calculation
- Profit percentage calculation
- VIX conditions
- Generic condition builders

Test Coverage:
- Normal operation scenarios
- Edge cases and boundary conditions
- Error handling
- Input validation
"""

import pytest
import numpy as np
from datetime import date, datetime, timedelta

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
)


# =============================================================================
# Test IV Percentile Calculation
# =============================================================================

class TestIVPercentile:
    """Tests for IV percentile calculation."""

    def test_basic_percentile(self):
        """Test basic IV percentile calculation."""
        historical = [0.10, 0.15, 0.20, 0.25, 0.30]
        current = 0.20

        percentile = calculate_iv_percentile(current, historical)

        # (0.20 - 0.10) / (0.30 - 0.10) = 0.10 / 0.20 = 0.50 = 50%
        assert abs(percentile - 50.0) < 0.01

    def test_percentile_at_minimum(self):
        """Test percentile when current IV is at historical minimum."""
        historical = [0.15, 0.20, 0.25, 0.30]
        current = 0.15

        percentile = calculate_iv_percentile(current, historical)
        assert percentile == 0.0

    def test_percentile_at_maximum(self):
        """Test percentile when current IV is at historical maximum."""
        historical = [0.15, 0.20, 0.25, 0.30]
        current = 0.30

        percentile = calculate_iv_percentile(current, historical)
        assert percentile == 100.0

    def test_percentile_below_minimum(self):
        """Test percentile when current IV is below historical minimum."""
        historical = [0.15, 0.20, 0.25, 0.30]
        current = 0.10

        percentile = calculate_iv_percentile(current, historical)
        assert percentile == 0.0

    def test_percentile_above_maximum(self):
        """Test percentile when current IV is above historical maximum."""
        historical = [0.15, 0.20, 0.25, 0.30]
        current = 0.40

        percentile = calculate_iv_percentile(current, historical)
        assert percentile == 100.0

    def test_percentile_with_window(self):
        """Test percentile with window parameter."""
        historical = [0.10, 0.15, 0.20, 0.25, 0.30]  # Full range
        # With window=3, only use last 3: [0.20, 0.25, 0.30]
        current = 0.25

        percentile = calculate_iv_percentile(current, historical, window=3)
        # (0.25 - 0.20) / (0.30 - 0.20) = 0.05 / 0.10 = 50%
        assert abs(percentile - 50.0) < 0.01

    def test_percentile_single_value(self):
        """Test percentile with single historical value."""
        historical = [0.20]
        current = 0.20

        percentile = calculate_iv_percentile(current, historical)
        assert percentile == 50.0  # Equal to only value

    def test_percentile_all_same_values(self):
        """Test percentile when all historical values are the same."""
        historical = [0.20, 0.20, 0.20]
        current = 0.20

        percentile = calculate_iv_percentile(current, historical)
        assert percentile == 50.0

        # Current above range
        current_high = 0.25
        percentile_high = calculate_iv_percentile(current_high, historical)
        assert percentile_high == 100.0

        # Current below range
        current_low = 0.15
        percentile_low = calculate_iv_percentile(current_low, historical)
        assert percentile_low == 0.0

    def test_percentile_invalid_current(self):
        """Test percentile with invalid current IV."""
        historical = [0.15, 0.20, 0.25]

        with pytest.raises(InvalidInputError):
            calculate_iv_percentile(None, historical)

        with pytest.raises(InvalidInputError):
            calculate_iv_percentile(float('inf'), historical)

        with pytest.raises(InvalidInputError):
            calculate_iv_percentile(-0.10, historical)

    def test_percentile_empty_historical(self):
        """Test percentile with empty historical data."""
        with pytest.raises(InvalidInputError):
            calculate_iv_percentile(0.20, [])

        with pytest.raises(InvalidInputError):
            calculate_iv_percentile(0.20, None)

    def test_percentile_with_nan_values(self):
        """Test percentile with NaN values in historical data."""
        historical = [0.10, 0.15, float('nan'), 0.25, 0.30]
        current = 0.20

        # Should filter out NaN values
        percentile = calculate_iv_percentile(current, historical)
        # Uses [0.10, 0.15, 0.25, 0.30]
        # (0.20 - 0.10) / (0.30 - 0.10) = 50%
        assert abs(percentile - 50.0) < 0.01

    def test_percentile_numpy_array(self):
        """Test percentile with numpy array input."""
        historical = np.array([0.10, 0.15, 0.20, 0.25, 0.30])
        current = 0.225

        percentile = calculate_iv_percentile(current, historical)
        expected = (0.225 - 0.10) / (0.30 - 0.10) * 100
        assert abs(percentile - expected) < 0.01


# =============================================================================
# Test IV Rank Calculation
# =============================================================================

class TestIVRank:
    """Tests for IV rank calculation."""

    def test_basic_rank(self):
        """Test basic IV rank calculation."""
        historical = [0.15, 0.18, 0.22, 0.25, 0.30]
        current = 0.20

        rank = calculate_iv_rank(current, historical)
        # Values <= 0.20: 0.15, 0.18, 0.20 doesn't exist, so 2 values
        # 2/5 = 40%
        assert abs(rank - 40.0) < 0.01

    def test_rank_at_minimum(self):
        """Test rank at minimum value."""
        historical = [0.15, 0.20, 0.25]
        current = 0.15

        rank = calculate_iv_rank(current, historical)
        # 1 value <= 0.15 out of 3 = 33.3%
        assert abs(rank - 33.33) < 1.0

    def test_rank_at_maximum(self):
        """Test rank at maximum value."""
        historical = [0.15, 0.20, 0.25]
        current = 0.25

        rank = calculate_iv_rank(current, historical)
        # All 3 values <= 0.25 = 100%
        assert rank == 100.0

    def test_rank_with_window(self):
        """Test rank with window parameter."""
        historical = [0.10, 0.15, 0.20, 0.25, 0.30]
        current = 0.22

        rank_full = calculate_iv_rank(current, historical)
        rank_window = calculate_iv_rank(current, historical, window=3)

        # Different historical data should give different ranks
        # Full: [0.10, 0.15, 0.20, 0.25, 0.30] - 3 values <= 0.22 = 60%
        # Window: [0.20, 0.25, 0.30] - 1 value <= 0.22 = 33.3%
        assert abs(rank_full - 60.0) < 0.01
        assert abs(rank_window - 33.33) < 1.0


# =============================================================================
# Test Event Date Checking
# =============================================================================

class TestEventDateChecking:
    """Tests for event date checking."""

    @pytest.fixture
    def sample_calendar(self):
        """Create a sample event calendar."""
        return {
            'earnings': [date(2024, 1, 25), date(2024, 4, 25)],
            'fomc': [date(2024, 1, 31), date(2024, 3, 20)],
            'cpi': [date(2024, 1, 11), date(2024, 2, 13)]
        }

    def test_event_day(self, sample_calendar):
        """Test checking actual event day."""
        is_event = is_major_event_date(date(2024, 1, 25), sample_calendar)
        assert is_event

    def test_day_before_event(self, sample_calendar):
        """Test checking day before event."""
        is_event = is_major_event_date(date(2024, 1, 24), sample_calendar)
        assert is_event

    def test_day_after_event(self, sample_calendar):
        """Test checking day after event."""
        is_event = is_major_event_date(date(2024, 1, 26), sample_calendar)
        assert is_event

    def test_non_event_day(self, sample_calendar):
        """Test checking non-event day."""
        is_event = is_major_event_date(date(2024, 1, 15), sample_calendar)
        assert not is_event

    def test_custom_buffer(self, sample_calendar):
        """Test with custom buffer days."""
        # Event on 1/25, with buffer=3 should include 1/22-1/28
        is_event = is_major_event_date(date(2024, 1, 22), sample_calendar, buffer_days=3)
        assert is_event

        is_event = is_major_event_date(date(2024, 1, 21), sample_calendar, buffer_days=3)
        assert not is_event

    def test_specific_event_types(self, sample_calendar):
        """Test checking specific event types."""
        # Only check earnings
        is_event = is_major_event_date(
            date(2024, 1, 25),
            sample_calendar,
            event_types=['earnings']
        )
        assert is_event

        # Only check FOMC - earnings day is not FOMC
        is_event = is_major_event_date(
            date(2024, 1, 25),
            sample_calendar,
            event_types=['fomc']
        )
        assert not is_event

    def test_empty_calendar(self):
        """Test with empty or None calendar."""
        is_event = is_major_event_date(date(2024, 1, 25), None)
        assert not is_event

        is_event = is_major_event_date(date(2024, 1, 25), {})
        assert not is_event

    def test_datetime_input(self, sample_calendar):
        """Test with datetime input."""
        is_event = is_major_event_date(datetime(2024, 1, 25, 10, 30), sample_calendar)
        assert is_event

    def test_get_upcoming_events(self, sample_calendar):
        """Test getting upcoming events."""
        events = get_upcoming_events(
            date(2024, 1, 1),
            date(2024, 1, 31),
            sample_calendar
        )

        assert len(events) == 3  # 1 earnings, 1 FOMC, 1 CPI in January

    def test_get_upcoming_events_filtered(self, sample_calendar):
        """Test getting specific event types."""
        events = get_upcoming_events(
            date(2024, 1, 1),
            date(2024, 1, 31),
            sample_calendar,
            event_types=['earnings']
        )

        assert len(events) == 1
        assert events[0]['type'] == 'earnings'


# =============================================================================
# Test Position Limit Checking
# =============================================================================

class TestPositionLimitChecking:
    """Tests for position limit checking."""

    def test_can_add_position(self):
        """Test when position can be added."""
        assert check_position_limit(current_count=3, max_count=5)

    def test_cannot_add_position(self):
        """Test when limit is reached."""
        assert not check_position_limit(current_count=5, max_count=5)
        assert not check_position_limit(current_count=6, max_count=5)

    def test_empty_portfolio(self):
        """Test empty portfolio."""
        assert check_position_limit(current_count=0, max_count=5)

    def test_invalid_inputs(self):
        """Test invalid inputs."""
        with pytest.raises(InvalidInputError):
            check_position_limit(current_count=None, max_count=5)

        with pytest.raises(InvalidInputError):
            check_position_limit(current_count=-1, max_count=5)


# =============================================================================
# Test Capital Limit Checking
# =============================================================================

class TestCapitalLimitChecking:
    """Tests for capital limit checking."""

    def test_within_limit(self):
        """Test when within capital limit."""
        assert check_capital_limit(
            current_allocated=50000,
            position_size=10000,
            max_capital=100000
        )

    def test_at_limit(self):
        """Test when at exact limit."""
        assert check_capital_limit(
            current_allocated=50000,
            position_size=50000,
            max_capital=100000
        )

    def test_exceed_limit(self):
        """Test when exceeding limit."""
        assert not check_capital_limit(
            current_allocated=50000,
            position_size=60000,
            max_capital=100000
        )

    def test_invalid_inputs(self):
        """Test invalid inputs."""
        with pytest.raises(InvalidInputError):
            check_capital_limit(None, 10000, 100000)


# =============================================================================
# Test Delta Limit Checking
# =============================================================================

class TestDeltaLimitChecking:
    """Tests for delta limit checking."""

    def test_within_limit(self):
        """Test when within delta limit."""
        assert check_delta_limit(
            current_delta=25.0,
            position_delta=-10.0,
            max_delta=50.0
        )

    def test_at_limit(self):
        """Test when at exact limit."""
        assert check_delta_limit(
            current_delta=30.0,
            position_delta=20.0,
            max_delta=50.0
        )

    def test_exceed_positive_limit(self):
        """Test exceeding positive delta limit."""
        assert not check_delta_limit(
            current_delta=40.0,
            position_delta=20.0,
            max_delta=50.0
        )

    def test_exceed_negative_limit(self):
        """Test exceeding negative delta limit."""
        assert not check_delta_limit(
            current_delta=-40.0,
            position_delta=-20.0,
            max_delta=50.0
        )

    def test_offsetting_delta(self):
        """Test offsetting delta positions."""
        # Start negative, add positive
        assert check_delta_limit(
            current_delta=-30.0,
            position_delta=25.0,  # Net = -5
            max_delta=10.0
        )


# =============================================================================
# Test Days to Expiry Calculation
# =============================================================================

class TestDaysToExpiry:
    """Tests for days to expiry calculation."""

    def test_basic_dte(self):
        """Test basic DTE calculation."""
        exp = date(2024, 3, 15)
        current = date(2024, 3, 1)

        dte = days_to_expiry(exp, current)
        assert dte == 14

    def test_expiration_day(self):
        """Test DTE on expiration day."""
        exp = date(2024, 3, 15)
        current = date(2024, 3, 15)

        dte = days_to_expiry(exp, current)
        assert dte == 0

    def test_already_expired(self):
        """Test DTE when already expired."""
        exp = date(2024, 3, 15)
        current = date(2024, 3, 20)

        dte = days_to_expiry(exp, current)
        assert dte == 0

    def test_datetime_input(self):
        """Test with datetime input."""
        exp = datetime(2024, 3, 15, 16, 0)
        current = datetime(2024, 3, 1, 9, 30)

        dte = days_to_expiry(exp, current)
        assert dte == 14

    def test_no_current_date(self):
        """Test with no current date (uses today)."""
        # Use a future expiration
        exp = date.today() + timedelta(days=30)
        dte = days_to_expiry(exp)
        assert dte == 30

    def test_invalid_expiration(self):
        """Test invalid expiration input."""
        with pytest.raises(InvalidInputError):
            days_to_expiry(None)


# =============================================================================
# Test Expiration Day Check
# =============================================================================

class TestExpirationDay:
    """Tests for expiration day check."""

    def test_is_expiration_day(self):
        """Test expiration day detection."""
        exp = date(2024, 3, 15)
        assert is_expiration_day(date(2024, 3, 15), exp)
        assert not is_expiration_day(date(2024, 3, 14), exp)

    def test_datetime_inputs(self):
        """Test with datetime inputs."""
        exp = datetime(2024, 3, 15, 16, 0)
        check = datetime(2024, 3, 15, 10, 0)
        assert is_expiration_day(check, exp)


# =============================================================================
# Test DTE Range Check
# =============================================================================

class TestDTERange:
    """Tests for DTE range checking."""

    def test_within_range(self):
        """Test DTE within range."""
        exp = date(2024, 3, 15)
        current = date(2024, 3, 1)

        assert is_within_dte_range(exp, current, min_dte=7, max_dte=30)

    def test_below_range(self):
        """Test DTE below range."""
        exp = date(2024, 3, 15)
        current = date(2024, 3, 10)  # 5 DTE

        assert not is_within_dte_range(exp, current, min_dte=7, max_dte=30)

    def test_above_range(self):
        """Test DTE above range."""
        exp = date(2024, 5, 15)
        current = date(2024, 3, 1)  # 75 DTE

        assert not is_within_dte_range(exp, current, min_dte=7, max_dte=30)


# =============================================================================
# Test Profit Percentage Calculation
# =============================================================================

class TestProfitPercentage:
    """Tests for profit percentage calculation."""

    def test_basic_profit_pct(self):
        """Test basic profit percentage."""
        pnl = 125.0
        premium = 500.0

        pct = calculate_profit_pct(pnl, premium)
        assert abs(pct - 0.25) < 0.01

    def test_loss_pct(self):
        """Test loss percentage."""
        pnl = -250.0
        premium = 500.0

        pct = calculate_profit_pct(pnl, premium)
        assert abs(pct - (-0.50)) < 0.01

    def test_zero_pnl(self):
        """Test zero P&L."""
        pnl = 0.0
        premium = 500.0

        pct = calculate_profit_pct(pnl, premium)
        assert pct == 0.0

    def test_zero_premium_error(self):
        """Test error on zero premium."""
        with pytest.raises(InvalidInputError):
            calculate_profit_pct(100.0, 0.0)

    def test_negative_premium(self):
        """Test with negative premium (debit)."""
        pnl = 100.0
        premium = -500.0  # Debit position

        pct = calculate_profit_pct(pnl, premium)
        # Should use absolute value of premium
        assert abs(pct - 0.20) < 0.01


# =============================================================================
# Test Profit Target Check
# =============================================================================

class TestProfitTarget:
    """Tests for profit target checking."""

    def test_target_reached(self):
        """Test when profit target is reached."""
        assert has_reached_profit_target(pnl=130, initial_premium=500, target_pct=0.25)

    def test_target_not_reached(self):
        """Test when profit target is not reached."""
        assert not has_reached_profit_target(pnl=100, initial_premium=500, target_pct=0.25)

    def test_exactly_at_target(self):
        """Test exactly at target."""
        assert has_reached_profit_target(pnl=125, initial_premium=500, target_pct=0.25)


# =============================================================================
# Test Stop Loss Check
# =============================================================================

class TestStopLoss:
    """Tests for stop loss checking."""

    def test_stop_triggered(self):
        """Test when stop loss is triggered."""
        assert has_reached_stop_loss(pnl=-600, initial_premium=500, stop_pct=-1.0)

    def test_stop_not_triggered(self):
        """Test when stop loss is not triggered."""
        assert not has_reached_stop_loss(pnl=-400, initial_premium=500, stop_pct=-1.0)

    def test_exactly_at_stop(self):
        """Test exactly at stop loss."""
        assert has_reached_stop_loss(pnl=-500, initial_premium=500, stop_pct=-1.0)


# =============================================================================
# Test VIX Conditions
# =============================================================================

class TestVIXConditions:
    """Tests for VIX condition checking."""

    def test_vix_above_threshold(self):
        """Test VIX above threshold."""
        assert is_vix_above_threshold(vix=25.5, threshold=20)
        assert not is_vix_above_threshold(vix=18.0, threshold=20)

    def test_vix_below_threshold(self):
        """Test VIX below threshold."""
        assert is_vix_below_threshold(vix=12.0, threshold=15)
        assert not is_vix_below_threshold(vix=18.0, threshold=15)

    def test_vix_in_range(self):
        """Test VIX in range."""
        assert is_vix_in_range(vix=20.0, min_vix=15, max_vix=25)
        assert not is_vix_in_range(vix=30.0, min_vix=15, max_vix=25)
        assert not is_vix_in_range(vix=10.0, min_vix=15, max_vix=25)

    def test_vix_at_boundaries(self):
        """Test VIX at boundary values."""
        assert is_vix_in_range(vix=15.0, min_vix=15, max_vix=25)  # At min
        assert is_vix_in_range(vix=25.0, min_vix=15, max_vix=25)  # At max

    def test_invalid_vix(self):
        """Test invalid VIX values."""
        with pytest.raises(InvalidInputError):
            is_vix_above_threshold(vix=None, threshold=20)


# =============================================================================
# Test Condition Builders
# =============================================================================

class TestConditionBuilders:
    """Tests for condition builder functions."""

    def test_threshold_above(self):
        """Test threshold condition - above."""
        cond = create_threshold_condition('iv_percentile', 50, 'above')

        assert cond({'iv_percentile': 60})
        assert not cond({'iv_percentile': 40})
        assert not cond({'iv_percentile': 50})  # Not above

    def test_threshold_below(self):
        """Test threshold condition - below."""
        cond = create_threshold_condition('vix', 15, 'below')

        assert cond({'vix': 12})
        assert not cond({'vix': 18})

    def test_threshold_equal(self):
        """Test threshold condition - equal."""
        cond = create_threshold_condition('dte', 30, 'equal')

        assert cond({'dte': 30})
        assert not cond({'dte': 29})

    def test_threshold_above_or_equal(self):
        """Test threshold condition - above_or_equal."""
        cond = create_threshold_condition('iv_percentile', 50, 'above_or_equal')

        assert cond({'iv_percentile': 60})
        assert cond({'iv_percentile': 50})
        assert not cond({'iv_percentile': 49})

    def test_threshold_missing_key(self):
        """Test threshold condition with missing key."""
        cond = create_threshold_condition('iv_percentile', 50, 'above')
        assert not cond({'vix': 20})  # iv_percentile missing

    def test_invalid_comparison(self):
        """Test invalid comparison operator."""
        with pytest.raises(InvalidInputError):
            create_threshold_condition('vix', 20, 'invalid_op')

    def test_range_condition(self):
        """Test range condition."""
        cond = create_range_condition('vix', 15, 30)

        assert cond({'vix': 20})
        assert cond({'vix': 15})  # At min
        assert cond({'vix': 30})  # At max
        assert not cond({'vix': 35})
        assert not cond({'vix': 10})

    def test_combine_conditions_and(self):
        """Test combining conditions with AND logic."""
        cond1 = create_threshold_condition('iv_percentile', 50, 'above')
        cond2 = create_threshold_condition('vix', 20, 'above')

        combined = combine_conditions([cond1, cond2], 'and')

        # Both true
        assert combined({'iv_percentile': 60, 'vix': 25})
        # Only one true
        assert not combined({'iv_percentile': 60, 'vix': 15})
        # Neither true
        assert not combined({'iv_percentile': 40, 'vix': 15})

    def test_combine_conditions_or(self):
        """Test combining conditions with OR logic."""
        cond1 = create_threshold_condition('iv_percentile', 50, 'above')
        cond2 = create_threshold_condition('vix', 20, 'above')

        combined = combine_conditions([cond1, cond2], 'or')

        # Both true
        assert combined({'iv_percentile': 60, 'vix': 25})
        # Only one true
        assert combined({'iv_percentile': 60, 'vix': 15})
        assert combined({'iv_percentile': 40, 'vix': 25})
        # Neither true
        assert not combined({'iv_percentile': 40, 'vix': 15})

    def test_invalid_logic(self):
        """Test invalid logic operator."""
        cond = create_threshold_condition('vix', 20, 'above')

        with pytest.raises(InvalidInputError):
            combine_conditions([cond], 'xor')

    def test_negate_condition(self):
        """Test negating a condition."""
        cond = create_threshold_condition('vix', 30, 'above')
        not_cond = negate_condition(cond)

        assert not_cond({'vix': 25})  # VIX not above 30
        assert not not_cond({'vix': 35})  # VIX above 30


# =============================================================================
# Test Complex Condition Scenarios
# =============================================================================

class TestComplexScenarios:
    """Tests for complex condition scenarios."""

    def test_entry_conditions_pattern(self):
        """Test typical entry conditions pattern."""
        # IV > 50th percentile AND VIX > 20 AND position limit not reached
        iv_cond = create_threshold_condition('iv_percentile', 50, 'above')
        vix_cond = create_threshold_condition('vix', 20, 'above')

        def position_cond(data):
            return check_position_limit(data.get('current_positions', 0), 5)

        entry_conditions = combine_conditions([iv_cond, vix_cond, position_cond], 'and')

        # All conditions met
        market_data = {
            'iv_percentile': 60,
            'vix': 25,
            'current_positions': 3
        }
        assert entry_conditions(market_data)

        # Position limit reached
        market_data['current_positions'] = 5
        assert not entry_conditions(market_data)

    def test_exit_conditions_pattern(self):
        """Test typical exit conditions pattern."""
        # Exit if profit target OR stop loss OR DTE <= 1
        def profit_cond(data):
            return data.get('pnl_pct', 0) >= 0.25

        def stop_cond(data):
            return data.get('pnl_pct', 0) <= -1.0

        def time_cond(data):
            return data.get('dte', 999) <= 1

        exit_conditions = combine_conditions([profit_cond, stop_cond, time_cond], 'or')

        # Profit target hit
        assert exit_conditions({'pnl_pct': 0.30, 'dte': 10})

        # Stop loss hit
        assert exit_conditions({'pnl_pct': -1.5, 'dte': 10})

        # Time exit
        assert exit_conditions({'pnl_pct': 0.10, 'dte': 1})

        # No exit condition met
        assert not exit_conditions({'pnl_pct': 0.10, 'dte': 5})


# =============================================================================
# Test Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_iv_percentile_very_low_iv(self):
        """Test IV percentile with very low IV values."""
        historical = [0.01, 0.02, 0.03, 0.04, 0.05]
        current = 0.025

        percentile = calculate_iv_percentile(current, historical)
        assert 0 <= percentile <= 100

    def test_iv_percentile_very_high_iv(self):
        """Test IV percentile with very high IV values."""
        historical = [0.50, 0.60, 0.70, 0.80, 0.90]
        current = 0.75

        percentile = calculate_iv_percentile(current, historical)
        assert 0 <= percentile <= 100

    def test_dte_same_day(self):
        """Test DTE when expiration is same day."""
        today = date.today()
        dte = days_to_expiry(today, today)
        assert dte == 0

    def test_profit_pct_very_small_premium(self):
        """Test profit percentage with very small premium."""
        pnl = 1.0
        premium = 0.01

        pct = calculate_profit_pct(pnl, premium)
        assert pct == 100.0  # 100x the premium

    def test_condition_with_none_values(self):
        """Test conditions handling None values."""
        cond = create_threshold_condition('vix', 20, 'above')

        # Should return False for None value, not raise error
        assert not cond({'vix': None})

    def test_empty_conditions_list(self):
        """Test combining empty conditions list."""
        combined_and = combine_conditions([], 'and')
        combined_or = combine_conditions([], 'or')

        # AND of empty should be True (vacuous truth)
        assert combined_and({})

        # OR of empty should be False
        assert not combined_or({})


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
