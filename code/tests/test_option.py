"""
Unit Tests for Option Class

This module contains comprehensive tests for the Option class, ensuring
correct position tracking, P&L calculations, and Greeks integration.

Test Categories:
    1. Construction Tests
        - Valid initialization
        - Input validation
        - Type checking

    2. P&L Calculation Tests
        - Long/short call P&L
        - Long/short put P&L
        - Edge cases

    3. Moneyness Tests
        - ITM/ATM/OTM classification
        - Intrinsic value calculation

    4. Greeks Integration Tests
        - Calculate greeks method
        - Greeks caching

    5. Time Calculation Tests
        - Time to expiry
        - Expired options handling

References:
    - Options position P&L: https://www.cboe.com/education/
"""

import pytest
import numpy as np
from datetime import datetime, timedelta

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backtester.core.option import (
    Option,
    OptionError,
    OptionExpiredError,
    OptionValidationError,
    create_long_call,
    create_short_call,
    create_long_put,
    create_short_put,
    CONTRACT_MULTIPLIER,
    DEFAULT_ATM_THRESHOLD,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_call():
    """Create a sample long call option."""
    return Option(
        option_type='call',
        position_type='long',
        underlying='SPY',
        strike=450.0,
        expiration=datetime(2024, 6, 21),
        quantity=10,
        entry_price=5.50,
        entry_date=datetime(2024, 3, 1),
        underlying_price_at_entry=445.0,
        implied_vol_at_entry=0.18
    )


@pytest.fixture
def sample_put():
    """Create a sample long put option."""
    return Option(
        option_type='put',
        position_type='long',
        underlying='SPY',
        strike=440.0,
        expiration=datetime(2024, 6, 21),
        quantity=5,
        entry_price=4.00,
        entry_date=datetime(2024, 3, 1),
        underlying_price_at_entry=445.0,
        implied_vol_at_entry=0.18
    )


@pytest.fixture
def short_call():
    """Create a sample short call option."""
    return Option(
        option_type='call',
        position_type='short',
        underlying='SPY',
        strike=455.0,
        expiration=datetime(2024, 6, 21),
        quantity=10,
        entry_price=3.50,
        entry_date=datetime(2024, 3, 1),
        underlying_price_at_entry=445.0,
        implied_vol_at_entry=0.18
    )


@pytest.fixture
def short_put():
    """Create a sample short put option."""
    return Option(
        option_type='put',
        position_type='short',
        underlying='SPY',
        strike=435.0,
        expiration=datetime(2024, 6, 21),
        quantity=5,
        entry_price=2.50,
        entry_date=datetime(2024, 3, 1),
        underlying_price_at_entry=445.0,
        implied_vol_at_entry=0.18
    )


# =============================================================================
# Construction Tests
# =============================================================================

class TestOptionConstruction:
    """Tests for Option initialization."""

    def test_basic_construction(self, sample_call):
        """Test basic option construction."""
        assert sample_call.option_type == 'call'
        assert sample_call.position_type == 'long'
        assert sample_call.underlying == 'SPY'
        assert sample_call.strike == 450.0
        assert sample_call.quantity == 10
        assert sample_call.entry_price == 5.50

    def test_option_type_normalization(self):
        """Test that option_type is normalized."""
        # 'c' should become 'call'
        opt = Option(
            option_type='c',
            position_type='long',
            underlying='SPY',
            strike=450.0,
            expiration=datetime(2024, 6, 21),
            quantity=1,
            entry_price=5.00,
            entry_date=datetime(2024, 3, 1),
            underlying_price_at_entry=445.0
        )
        assert opt.option_type == 'call'

        # 'p' should become 'put'
        opt = Option(
            option_type='p',
            position_type='long',
            underlying='SPY',
            strike=450.0,
            expiration=datetime(2024, 6, 21),
            quantity=1,
            entry_price=5.00,
            entry_date=datetime(2024, 3, 1),
            underlying_price_at_entry=445.0
        )
        assert opt.option_type == 'put'

    def test_underlying_uppercase(self):
        """Test that underlying is uppercased."""
        opt = Option(
            option_type='call',
            position_type='long',
            underlying='spy',
            strike=450.0,
            expiration=datetime(2024, 6, 21),
            quantity=1,
            entry_price=5.00,
            entry_date=datetime(2024, 3, 1),
            underlying_price_at_entry=445.0
        )
        assert opt.underlying == 'SPY'

    def test_properties_set_correctly(self, sample_call):
        """Test all properties are accessible."""
        assert sample_call.is_call == True
        assert sample_call.is_put == False
        assert sample_call.is_long == True
        assert sample_call.is_short == False
        assert sample_call.position_sign == 1
        assert sample_call.expiration == datetime(2024, 6, 21)


class TestOptionValidation:
    """Tests for input validation during construction."""

    def test_invalid_option_type(self):
        """Test that invalid option_type raises error."""
        with pytest.raises(OptionValidationError):
            Option(
                option_type='forward',
                position_type='long',
                underlying='SPY',
                strike=450.0,
                expiration=datetime(2024, 6, 21),
                quantity=10,
                entry_price=5.00,
                entry_date=datetime(2024, 3, 1),
                underlying_price_at_entry=445.0
            )

    def test_invalid_position_type(self):
        """Test that invalid position_type raises error."""
        with pytest.raises(OptionValidationError):
            Option(
                option_type='call',
                position_type='neutral',
                underlying='SPY',
                strike=450.0,
                expiration=datetime(2024, 6, 21),
                quantity=10,
                entry_price=5.00,
                entry_date=datetime(2024, 3, 1),
                underlying_price_at_entry=445.0
            )

    def test_empty_underlying(self):
        """Test that empty underlying raises error."""
        with pytest.raises(OptionValidationError):
            Option(
                option_type='call',
                position_type='long',
                underlying='',
                strike=450.0,
                expiration=datetime(2024, 6, 21),
                quantity=10,
                entry_price=5.00,
                entry_date=datetime(2024, 3, 1),
                underlying_price_at_entry=445.0
            )

    def test_negative_strike(self):
        """Test that negative strike raises error."""
        with pytest.raises(OptionValidationError):
            Option(
                option_type='call',
                position_type='long',
                underlying='SPY',
                strike=-450.0,
                expiration=datetime(2024, 6, 21),
                quantity=10,
                entry_price=5.00,
                entry_date=datetime(2024, 3, 1),
                underlying_price_at_entry=445.0
            )

    def test_zero_quantity(self):
        """Test that zero quantity raises error."""
        with pytest.raises(OptionValidationError):
            Option(
                option_type='call',
                position_type='long',
                underlying='SPY',
                strike=450.0,
                expiration=datetime(2024, 6, 21),
                quantity=0,
                entry_price=5.00,
                entry_date=datetime(2024, 3, 1),
                underlying_price_at_entry=445.0
            )

    def test_negative_entry_price(self):
        """Test that negative entry_price raises error."""
        with pytest.raises(OptionValidationError):
            Option(
                option_type='call',
                position_type='long',
                underlying='SPY',
                strike=450.0,
                expiration=datetime(2024, 6, 21),
                quantity=10,
                entry_price=-5.00,
                entry_date=datetime(2024, 3, 1),
                underlying_price_at_entry=445.0
            )

    def test_none_expiration(self):
        """Test that None expiration raises error."""
        with pytest.raises(OptionValidationError):
            Option(
                option_type='call',
                position_type='long',
                underlying='SPY',
                strike=450.0,
                expiration=None,
                quantity=10,
                entry_price=5.00,
                entry_date=datetime(2024, 3, 1),
                underlying_price_at_entry=445.0
            )


# =============================================================================
# P&L Calculation Tests
# =============================================================================

class TestPnLCalculation:
    """Tests for P&L calculation."""

    def test_long_call_profit(self, sample_call):
        """Test P&L for profitable long call."""
        # Entry: $5.50, Current: $7.50
        # P&L = (7.50 - 5.50) * 10 * 100 = $2,000
        sample_call.update_price(7.50, datetime(2024, 4, 1))
        pnl = sample_call.calculate_pnl()

        expected = (7.50 - 5.50) * 10 * CONTRACT_MULTIPLIER
        assert abs(pnl - expected) < 0.01
        assert pnl == 2000.0

    def test_long_call_loss(self, sample_call):
        """Test P&L for losing long call."""
        # Entry: $5.50, Current: $3.00
        # P&L = (3.00 - 5.50) * 10 * 100 = -$2,500
        sample_call.update_price(3.00, datetime(2024, 4, 1))
        pnl = sample_call.calculate_pnl()

        expected = (3.00 - 5.50) * 10 * CONTRACT_MULTIPLIER
        assert abs(pnl - expected) < 0.01
        assert pnl == -2500.0

    def test_short_call_profit(self, short_call):
        """Test P&L for profitable short call (price decreases)."""
        # Entry: $3.50, Current: $1.50
        # Short P&L = (3.50 - 1.50) * 10 * 100 = $2,000
        short_call.update_price(1.50, datetime(2024, 4, 1))
        pnl = short_call.calculate_pnl()

        # For short: P&L = (entry - current) * qty * 100
        expected = (3.50 - 1.50) * 10 * CONTRACT_MULTIPLIER
        assert abs(pnl - expected) < 0.01
        assert pnl == 2000.0

    def test_short_call_loss(self, short_call):
        """Test P&L for losing short call (price increases)."""
        # Entry: $3.50, Current: $6.00
        # Short P&L = (3.50 - 6.00) * 10 * 100 = -$2,500
        short_call.update_price(6.00, datetime(2024, 4, 1))
        pnl = short_call.calculate_pnl()

        expected = (3.50 - 6.00) * 10 * CONTRACT_MULTIPLIER
        assert abs(pnl - expected) < 0.01
        assert pnl == -2500.0

    def test_long_put_profit(self, sample_put):
        """Test P&L for profitable long put."""
        # Entry: $4.00, Current: $8.00
        # P&L = (8.00 - 4.00) * 5 * 100 = $2,000
        sample_put.update_price(8.00, datetime(2024, 4, 1))
        pnl = sample_put.calculate_pnl()

        expected = (8.00 - 4.00) * 5 * CONTRACT_MULTIPLIER
        assert abs(pnl - expected) < 0.01
        assert pnl == 2000.0

    def test_short_put_profit(self, short_put):
        """Test P&L for profitable short put (price decreases)."""
        # Entry: $2.50, Current: $1.00
        # Short P&L = (2.50 - 1.00) * 5 * 100 = $750
        short_put.update_price(1.00, datetime(2024, 4, 1))
        pnl = short_put.calculate_pnl()

        expected = (2.50 - 1.00) * 5 * CONTRACT_MULTIPLIER
        assert abs(pnl - expected) < 0.01
        assert pnl == 750.0

    def test_pnl_at_entry_is_zero(self, sample_call):
        """Test that P&L at entry is zero."""
        # No price update, current = entry
        pnl = sample_call.calculate_pnl()
        assert pnl == 0.0

    def test_pnl_at_hypothetical_price(self, sample_call):
        """Test P&L at a hypothetical price."""
        hypothetical_pnl = sample_call.calculate_pnl_at_price(10.00)
        expected = (10.00 - 5.50) * 10 * CONTRACT_MULTIPLIER
        assert abs(hypothetical_pnl - expected) < 0.01


class TestPayoffAtExpiry:
    """Tests for expiry payoff calculations."""

    def test_call_payoff_itm(self, sample_call):
        """Test call payoff when ITM at expiry."""
        # Strike 450, spot 460 -> intrinsic = $10 per share
        # Payoff = 10 * 10 * 100 = $10,000
        payoff = sample_call.get_payoff_at_expiry(spot_price=460.0)
        expected = (460.0 - 450.0) * 10 * CONTRACT_MULTIPLIER
        assert abs(payoff - expected) < 0.01
        assert payoff == 10000.0

    def test_call_payoff_otm(self, sample_call):
        """Test call payoff when OTM at expiry."""
        # Strike 450, spot 440 -> intrinsic = 0
        payoff = sample_call.get_payoff_at_expiry(spot_price=440.0)
        assert payoff == 0.0

    def test_put_payoff_itm(self, sample_put):
        """Test put payoff when ITM at expiry."""
        # Strike 440, spot 430 -> intrinsic = $10 per share
        # Payoff = 10 * 5 * 100 = $5,000
        payoff = sample_put.get_payoff_at_expiry(spot_price=430.0)
        expected = (440.0 - 430.0) * 5 * CONTRACT_MULTIPLIER
        assert abs(payoff - expected) < 0.01
        assert payoff == 5000.0

    def test_put_payoff_otm(self, sample_put):
        """Test put payoff when OTM at expiry."""
        # Strike 440, spot 450 -> intrinsic = 0
        payoff = sample_put.get_payoff_at_expiry(spot_price=450.0)
        assert payoff == 0.0

    def test_short_call_payoff(self, short_call):
        """Test short call payoff (negative when ITM)."""
        # Short call: negative payoff when ITM
        # Strike 455, spot 465 -> short pays out $10 per share
        payoff = short_call.get_payoff_at_expiry(spot_price=465.0)
        expected = -(465.0 - 455.0) * 10 * CONTRACT_MULTIPLIER
        assert abs(payoff - expected) < 0.01
        assert payoff == -10000.0

    def test_pnl_at_expiry(self, sample_call):
        """Test total P&L at expiry."""
        # Long call, strike 450, entry $5.50, spot at expiry = 460
        # Intrinsic = $10, P&L per share = 10 - 5.50 = $4.50
        # Total P&L = 4.50 * 10 * 100 = $4,500
        pnl = sample_call.calculate_pnl_at_expiry(spot_price=460.0)
        expected = (10.0 - 5.50) * 10 * CONTRACT_MULTIPLIER
        assert abs(pnl - expected) < 0.01
        assert pnl == 4500.0


# =============================================================================
# Moneyness Tests
# =============================================================================

class TestMoneyness:
    """Tests for moneyness classification."""

    def test_call_itm(self, sample_call):
        """Test ITM detection for call."""
        # Strike 450, spot 460 -> ITM
        assert sample_call.is_itm(spot_price=460.0) == True

    def test_call_otm(self, sample_call):
        """Test OTM detection for call."""
        # Strike 450, spot 440 -> OTM
        assert sample_call.is_otm(spot_price=440.0) == True

    def test_call_atm(self, sample_call):
        """Test ATM detection for call."""
        # Strike 450, spot 450 -> ATM
        assert sample_call.is_atm(spot_price=450.0) == True

        # Spot within threshold (2%)
        assert sample_call.is_atm(spot_price=445.0, threshold=0.02) == True
        assert sample_call.is_atm(spot_price=455.0, threshold=0.02) == True

    def test_put_itm(self, sample_put):
        """Test ITM detection for put."""
        # Strike 440, spot 430 -> ITM
        assert sample_put.is_itm(spot_price=430.0) == True

    def test_put_otm(self, sample_put):
        """Test OTM detection for put."""
        # Strike 440, spot 450 -> OTM
        assert sample_put.is_otm(spot_price=450.0) == True

    def test_moneyness_ratio(self, sample_call):
        """Test moneyness ratio calculation."""
        # Strike 450, spot 460 -> moneyness = 460/450 = 1.022
        moneyness = sample_call.get_moneyness(spot_price=460.0)
        assert abs(moneyness - 460.0/450.0) < 0.001

    def test_moneyness_string(self, sample_call):
        """Test moneyness string classification."""
        assert sample_call.get_moneyness_str(spot_price=460.0) == 'ITM'
        assert sample_call.get_moneyness_str(spot_price=440.0) == 'OTM'
        assert sample_call.get_moneyness_str(spot_price=450.0) == 'ATM'


class TestIntrinsicTimeValue:
    """Tests for intrinsic and time value calculations."""

    def test_call_intrinsic_value_itm(self, sample_call):
        """Test intrinsic value for ITM call."""
        # Strike 450, spot 460 -> intrinsic = $10
        intrinsic = sample_call.get_intrinsic_value(spot_price=460.0)
        assert abs(intrinsic - 10.0) < 0.01

    def test_call_intrinsic_value_otm(self, sample_call):
        """Test intrinsic value for OTM call (should be 0)."""
        intrinsic = sample_call.get_intrinsic_value(spot_price=440.0)
        assert intrinsic == 0.0

    def test_put_intrinsic_value_itm(self, sample_put):
        """Test intrinsic value for ITM put."""
        # Strike 440, spot 430 -> intrinsic = $10
        intrinsic = sample_put.get_intrinsic_value(spot_price=430.0)
        assert abs(intrinsic - 10.0) < 0.01

    def test_time_value(self, sample_call):
        """Test time value calculation."""
        # Current price $5.50, spot 445 (OTM) -> intrinsic = 0
        # Time value = 5.50 - 0 = $5.50
        time_value = sample_call.get_time_value(spot_price=445.0)
        assert abs(time_value - 5.50) < 0.01

    def test_time_value_itm(self, sample_call):
        """Test time value for ITM option."""
        # Update price to $12, spot 460
        # Intrinsic = $10, Time value = 12 - 10 = $2
        sample_call.update_price(12.0, datetime(2024, 4, 1))
        time_value = sample_call.get_time_value(spot_price=460.0)
        assert abs(time_value - 2.0) < 0.01


# =============================================================================
# Time Calculation Tests
# =============================================================================

class TestTimeCalculations:
    """Tests for time-to-expiry calculations."""

    def test_time_to_expiry_years(self, sample_call):
        """Test time to expiry in years."""
        # Entry: March 1, Expiry: June 21
        # Days = 112 (approximately)
        tte = sample_call.get_time_to_expiry(current_date=datetime(2024, 3, 1))

        # Should be roughly 112/365 = 0.307 years
        expected = 112 / 365
        assert abs(tte - expected) < 0.01

    def test_days_to_expiry(self, sample_call):
        """Test days to expiry."""
        dte = sample_call.get_days_to_expiry(current_date=datetime(2024, 3, 1))
        assert dte == 112  # March 1 to June 21

    def test_time_to_expiry_at_expiry(self, sample_call):
        """Test time to expiry at expiration date."""
        tte = sample_call.get_time_to_expiry(current_date=datetime(2024, 6, 21))
        assert tte == 0.0

    def test_is_expired(self, sample_call):
        """Test expired detection."""
        # Update timestamp to after expiry
        assert sample_call.is_expired == False

        # Manually check with a date past expiry
        sample_call._current_timestamp = datetime(2024, 6, 22)
        assert sample_call.is_expired == True


class TestExpiredOptionHandling:
    """Tests for handling expired options."""

    def test_update_price_after_expiry_raises_error(self, sample_call):
        """Test that updating price after expiry raises error."""
        with pytest.raises(OptionExpiredError):
            sample_call.update_price(
                new_price=10.0,
                timestamp=datetime(2024, 6, 22)  # After June 21 expiry
            )

    def test_time_value_expired_raises_error(self, sample_call):
        """Test that time value for expired option raises error."""
        sample_call._current_timestamp = datetime(2024, 6, 22)
        with pytest.raises(OptionExpiredError):
            sample_call.get_time_value(spot_price=450.0)


# =============================================================================
# Greeks Integration Tests
# =============================================================================

class TestGreeksCalculation:
    """Tests for Greeks calculation integration."""

    def test_calculate_greeks_returns_dict(self, sample_call):
        """Test that calculate_greeks returns proper dict."""
        greeks = sample_call.calculate_greeks(
            spot=450.0,
            vol=0.20,
            rate=0.05,
            current_date=datetime(2024, 3, 15)
        )

        assert 'delta' in greeks
        assert 'gamma' in greeks
        assert 'theta' in greeks
        assert 'vega' in greeks
        assert 'rho' in greeks

    def test_call_delta_positive(self, sample_call):
        """Test that call delta is positive."""
        greeks = sample_call.calculate_greeks(
            spot=450.0,
            vol=0.20,
            rate=0.05,
            current_date=datetime(2024, 3, 15)
        )

        assert 0 < greeks['delta'] < 1

    def test_put_delta_negative(self, sample_put):
        """Test that put delta is negative."""
        greeks = sample_put.calculate_greeks(
            spot=445.0,
            vol=0.20,
            rate=0.05,
            current_date=datetime(2024, 3, 15)
        )

        assert -1 < greeks['delta'] < 0

    def test_greeks_cached(self, sample_call):
        """Test that Greeks are cached."""
        sample_call.calculate_greeks(
            spot=450.0,
            vol=0.20,
            rate=0.05,
            current_date=datetime(2024, 3, 15)
        )

        # Access cached greeks via property
        cached = sample_call.greeks
        assert len(cached) == 5

    def test_calculate_greeks_expired_raises_error(self, sample_call):
        """Test that calculating Greeks for expired option raises error."""
        with pytest.raises(OptionExpiredError):
            sample_call.calculate_greeks(
                spot=450.0,
                vol=0.20,
                rate=0.05,
                current_date=datetime(2024, 6, 22)  # After expiry
            )


# =============================================================================
# Factory Function Tests
# =============================================================================

class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_long_call(self):
        """Test create_long_call factory."""
        option = create_long_call(
            underlying='SPY',
            strike=450.0,
            expiration=datetime(2024, 6, 21),
            quantity=10,
            entry_price=5.50,
            entry_date=datetime(2024, 3, 1),
            underlying_price=445.0,
            implied_vol=0.18
        )

        assert option.is_call == True
        assert option.is_long == True

    def test_create_short_call(self):
        """Test create_short_call factory."""
        option = create_short_call(
            underlying='SPY',
            strike=450.0,
            expiration=datetime(2024, 6, 21),
            quantity=10,
            entry_price=5.50,
            entry_date=datetime(2024, 3, 1),
            underlying_price=445.0
        )

        assert option.is_call == True
        assert option.is_short == True

    def test_create_long_put(self):
        """Test create_long_put factory."""
        option = create_long_put(
            underlying='SPY',
            strike=440.0,
            expiration=datetime(2024, 6, 21),
            quantity=5,
            entry_price=4.00,
            entry_date=datetime(2024, 3, 1),
            underlying_price=445.0
        )

        assert option.is_put == True
        assert option.is_long == True

    def test_create_short_put(self):
        """Test create_short_put factory."""
        option = create_short_put(
            underlying='SPY',
            strike=435.0,
            expiration=datetime(2024, 6, 21),
            quantity=5,
            entry_price=2.50,
            entry_date=datetime(2024, 3, 1),
            underlying_price=445.0
        )

        assert option.is_put == True
        assert option.is_short == True


# =============================================================================
# Serialization Tests
# =============================================================================

class TestSerialization:
    """Tests for serialization methods."""

    def test_to_dict(self, sample_call):
        """Test to_dict method."""
        d = sample_call.to_dict()

        assert d['option_type'] == 'call'
        assert d['position_type'] == 'long'
        assert d['underlying'] == 'SPY'
        assert d['strike'] == 450.0
        assert d['quantity'] == 10
        assert d['entry_price'] == 5.50

    def test_from_dict_roundtrip(self, sample_call):
        """Test from_dict roundtrip."""
        d = sample_call.to_dict()
        restored = Option.from_dict(d)

        assert restored.option_type == sample_call.option_type
        assert restored.position_type == sample_call.position_type
        assert restored.underlying == sample_call.underlying
        assert restored.strike == sample_call.strike
        assert restored.quantity == sample_call.quantity
        assert restored.entry_price == sample_call.entry_price


# =============================================================================
# String Representation Tests
# =============================================================================

class TestStringRepresentation:
    """Tests for string representations."""

    def test_str_representation(self, sample_call):
        """Test __str__ method."""
        s = str(sample_call)
        assert 'LONG' in s
        assert '10' in s
        assert 'SPY' in s
        assert '450.0' in s
        assert 'CALL' in s

    def test_repr_representation(self, sample_call):
        """Test __repr__ method."""
        r = repr(sample_call)
        assert 'Option(' in r
        assert 'call' in r
        assert 'long' in r


# =============================================================================
# Equality and Hashing Tests
# =============================================================================

class TestEqualityHashing:
    """Tests for equality and hashing."""

    def test_equality(self):
        """Test option equality."""
        opt1 = create_long_call(
            underlying='SPY',
            strike=450.0,
            expiration=datetime(2024, 6, 21),
            quantity=10,
            entry_price=5.50,
            entry_date=datetime(2024, 3, 1),
            underlying_price=445.0
        )
        opt2 = create_long_call(
            underlying='SPY',
            strike=450.0,
            expiration=datetime(2024, 6, 21),
            quantity=10,
            entry_price=5.50,
            entry_date=datetime(2024, 3, 1),
            underlying_price=445.0
        )

        assert opt1 == opt2

    def test_inequality_different_strike(self):
        """Test options with different strikes are not equal."""
        opt1 = create_long_call(
            underlying='SPY',
            strike=450.0,
            expiration=datetime(2024, 6, 21),
            quantity=10,
            entry_price=5.50,
            entry_date=datetime(2024, 3, 1),
            underlying_price=445.0
        )
        opt2 = create_long_call(
            underlying='SPY',
            strike=455.0,  # Different
            expiration=datetime(2024, 6, 21),
            quantity=10,
            entry_price=5.50,
            entry_date=datetime(2024, 3, 1),
            underlying_price=445.0
        )

        assert opt1 != opt2

    def test_hashable(self):
        """Test options can be used in sets."""
        opt1 = create_long_call(
            underlying='SPY',
            strike=450.0,
            expiration=datetime(2024, 6, 21),
            quantity=10,
            entry_price=5.50,
            entry_date=datetime(2024, 3, 1),
            underlying_price=445.0
        )
        opt2 = create_long_call(
            underlying='SPY',
            strike=455.0,
            expiration=datetime(2024, 6, 21),
            quantity=10,
            entry_price=5.50,
            entry_date=datetime(2024, 3, 1),
            underlying_price=445.0
        )

        s = {opt1, opt2}
        assert len(s) == 2


# =============================================================================
# Market Value Tests
# =============================================================================

class TestMarketValue:
    """Tests for market value calculations."""

    def test_long_market_value(self, sample_call):
        """Test market value for long position."""
        # Current price $5.50, 10 contracts
        # Market value = 5.50 * 10 * 100 = $5,500
        mv = sample_call.market_value
        assert mv == 5500.0

    def test_short_market_value(self, short_call):
        """Test market value for short position (negative)."""
        # Current price $3.50, 10 contracts, short
        # Market value = -3.50 * 10 * 100 = -$3,500
        mv = short_call.market_value
        assert mv == -3500.0

    def test_notional_value(self, sample_call):
        """Test notional value calculation."""
        # Strike 450, 10 contracts
        # Notional = 450 * 10 * 100 = $450,000
        notional = sample_call.notional_value
        assert notional == 450000.0


# =============================================================================
# Price History Tests
# =============================================================================

class TestPriceHistory:
    """Tests for price history tracking."""

    def test_initial_price_history(self, sample_call):
        """Test initial price history."""
        history = sample_call.get_price_history()
        assert len(history) == 1
        assert history[0] == (datetime(2024, 3, 1), 5.50)

    def test_price_history_updates(self, sample_call):
        """Test price history after updates."""
        sample_call.update_price(6.00, datetime(2024, 3, 15))
        sample_call.update_price(7.00, datetime(2024, 4, 1))

        history = sample_call.get_price_history()
        assert len(history) == 3
        assert history[1] == (datetime(2024, 3, 15), 6.00)
        assert history[2] == (datetime(2024, 4, 1), 7.00)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
