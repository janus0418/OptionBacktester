"""
Unit Tests for OptionStructure Class

This module contains comprehensive tests for the OptionStructure class,
ensuring correct multi-leg position management, Greeks aggregation,
P&L calculations, and payoff analysis.

Test Categories:
    1. Construction Tests
        - Basic initialization
        - Input validation
        - Auto-detection of underlying

    2. Option Management Tests
        - Adding options
        - Removing options
        - Underlying consistency validation

    3. Net Greeks Calculation Tests
        - Single leg Greeks
        - Multi-leg aggregation (2, 3, 4 legs)
        - Long/short position handling
        - Greeks caching

    4. P&L Calculation Tests
        - Single leg P&L
        - Multi-leg total P&L
        - Net premium tracking (credits and debits)

    5. Max Profit/Loss Tests
        - Defined risk structures
        - Undefined risk handling

    6. Breakeven Calculation Tests
        - Single breakeven
        - Multiple breakevens
        - Edge cases

    7. Payoff Diagram Tests
        - Payoff at expiry
        - P&L at expiry
        - Diagram generation

    8. Serialization Tests
        - to_dict / from_dict roundtrip

    9. Edge Cases
        - Empty structure
        - Single leg structure
        - Multi-expiration structures

Financial Correctness:
    Tests validate that:
    - Net Greeks = sum of position-adjusted individual Greeks
    - P&L correctly handles long/short positions
    - Max profit/loss computed correctly for various structures
    - Breakevens solved numerically to correct precision

References:
    - Options structures: https://www.cboe.com/education/
    - Payoff diagrams: Hull (2018) Options, Futures, and Other Derivatives
"""

import pytest
import numpy as np
from datetime import datetime, timedelta

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backtester.core.option import (
    Option,
    create_long_call,
    create_short_call,
    create_long_put,
    create_short_put,
    CONTRACT_MULTIPLIER,
)
from backtester.core.option_structure import (
    OptionStructure,
    OptionStructureError,
    OptionStructureValidationError,
    EmptyStructureError,
    BREAKEVEN_TOLERANCE,
    GREEK_NAMES,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_long_call():
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
def sample_long_put():
    """Create a sample long put option."""
    return Option(
        option_type='put',
        position_type='long',
        underlying='SPY',
        strike=450.0,
        expiration=datetime(2024, 6, 21),
        quantity=10,
        entry_price=6.00,
        entry_date=datetime(2024, 3, 1),
        underlying_price_at_entry=445.0,
        implied_vol_at_entry=0.18
    )


@pytest.fixture
def sample_short_call():
    """Create a sample short call option."""
    return Option(
        option_type='call',
        position_type='short',
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
def sample_short_put():
    """Create a sample short put option."""
    return Option(
        option_type='put',
        position_type='short',
        underlying='SPY',
        strike=450.0,
        expiration=datetime(2024, 6, 21),
        quantity=10,
        entry_price=6.00,
        entry_date=datetime(2024, 3, 1),
        underlying_price_at_entry=445.0,
        implied_vol_at_entry=0.18
    )


@pytest.fixture
def empty_structure():
    """Create an empty structure."""
    return OptionStructure(structure_type='custom', underlying='SPY')


@pytest.fixture
def long_straddle(sample_long_call, sample_long_put):
    """Create a long straddle (long call + long put at same strike)."""
    structure = OptionStructure(structure_type='straddle', underlying='SPY')
    structure.add_option(sample_long_call)
    structure.add_option(sample_long_put)
    return structure


@pytest.fixture
def short_straddle(sample_short_call, sample_short_put):
    """Create a short straddle (short call + short put at same strike)."""
    structure = OptionStructure(structure_type='straddle', underlying='SPY')
    structure.add_option(sample_short_call)
    structure.add_option(sample_short_put)
    return structure


@pytest.fixture
def iron_condor():
    """Create an iron condor (4-leg structure)."""
    structure = OptionStructure(structure_type='iron_condor', underlying='SPY')

    # Sell put at 430
    structure.add_option(Option(
        option_type='put',
        position_type='short',
        underlying='SPY',
        strike=430.0,
        expiration=datetime(2024, 6, 21),
        quantity=10,
        entry_price=3.00,
        entry_date=datetime(2024, 3, 1),
        underlying_price_at_entry=445.0,
    ))

    # Buy put at 420
    structure.add_option(Option(
        option_type='put',
        position_type='long',
        underlying='SPY',
        strike=420.0,
        expiration=datetime(2024, 6, 21),
        quantity=10,
        entry_price=1.50,
        entry_date=datetime(2024, 3, 1),
        underlying_price_at_entry=445.0,
    ))

    # Sell call at 460
    structure.add_option(Option(
        option_type='call',
        position_type='short',
        underlying='SPY',
        strike=460.0,
        expiration=datetime(2024, 6, 21),
        quantity=10,
        entry_price=3.50,
        entry_date=datetime(2024, 3, 1),
        underlying_price_at_entry=445.0,
    ))

    # Buy call at 470
    structure.add_option(Option(
        option_type='call',
        position_type='long',
        underlying='SPY',
        strike=470.0,
        expiration=datetime(2024, 6, 21),
        quantity=10,
        entry_price=1.75,
        entry_date=datetime(2024, 3, 1),
        underlying_price_at_entry=445.0,
    ))

    return structure


@pytest.fixture
def bull_call_spread():
    """Create a bull call spread (long lower strike call, short higher strike call)."""
    structure = OptionStructure(structure_type='vertical_spread', underlying='SPY')

    # Long call at 445
    structure.add_option(Option(
        option_type='call',
        position_type='long',
        underlying='SPY',
        strike=445.0,
        expiration=datetime(2024, 6, 21),
        quantity=10,
        entry_price=8.00,
        entry_date=datetime(2024, 3, 1),
        underlying_price_at_entry=445.0,
    ))

    # Short call at 455
    structure.add_option(Option(
        option_type='call',
        position_type='short',
        underlying='SPY',
        strike=455.0,
        expiration=datetime(2024, 6, 21),
        quantity=10,
        entry_price=4.00,
        entry_date=datetime(2024, 3, 1),
        underlying_price_at_entry=445.0,
    ))

    return structure


# =============================================================================
# Construction Tests
# =============================================================================

class TestOptionStructureConstruction:
    """Tests for OptionStructure initialization."""

    def test_basic_construction(self):
        """Test basic structure construction."""
        structure = OptionStructure(
            structure_type='straddle',
            underlying='SPY'
        )
        assert structure.structure_type == 'straddle'
        assert structure.underlying == 'SPY'
        assert structure.is_empty == True
        assert structure.num_legs == 0

    def test_auto_generated_id(self):
        """Test that structure ID is auto-generated."""
        structure = OptionStructure(structure_type='custom')
        assert structure.structure_id is not None
        assert len(structure.structure_id) == 8

    def test_custom_id(self):
        """Test custom structure ID."""
        structure = OptionStructure(
            structure_type='custom',
            structure_id='my-struct'
        )
        assert structure.structure_id == 'my-struct'

    def test_structure_type_normalization(self):
        """Test that structure_type is normalized."""
        structure = OptionStructure(structure_type='  STRADDLE  ')
        assert structure.structure_type == 'straddle'

    def test_underlying_uppercase(self):
        """Test that underlying is uppercased."""
        structure = OptionStructure(
            structure_type='custom',
            underlying='spy'
        )
        assert structure.underlying == 'SPY'

    def test_empty_structure_type_raises_error(self):
        """Test that empty structure_type raises error."""
        with pytest.raises(OptionStructureValidationError):
            OptionStructure(structure_type='')

    def test_invalid_structure_type_raises_error(self):
        """Test that invalid structure_type raises error."""
        with pytest.raises(OptionStructureValidationError):
            OptionStructure(structure_type=None)


# =============================================================================
# Option Management Tests
# =============================================================================

class TestOptionManagement:
    """Tests for adding and removing options."""

    def test_add_single_option(self, empty_structure, sample_long_call):
        """Test adding a single option."""
        empty_structure.add_option(sample_long_call)

        assert empty_structure.num_legs == 1
        assert empty_structure.is_empty == False
        assert empty_structure.options[0] == sample_long_call

    def test_add_multiple_options(self, empty_structure, sample_long_call, sample_long_put):
        """Test adding multiple options."""
        empty_structure.add_option(sample_long_call)
        empty_structure.add_option(sample_long_put)

        assert empty_structure.num_legs == 2

    def test_underlying_from_first_option(self, sample_long_call):
        """Test that underlying is set from first option."""
        structure = OptionStructure(structure_type='custom')
        assert structure.underlying is None

        structure.add_option(sample_long_call)
        assert structure.underlying == 'SPY'

    def test_different_underlying_raises_error(self, empty_structure, sample_long_call):
        """Test that adding option with different underlying raises error."""
        empty_structure.add_option(sample_long_call)

        different_underlying = Option(
            option_type='call',
            position_type='long',
            underlying='QQQ',  # Different!
            strike=450.0,
            expiration=datetime(2024, 6, 21),
            quantity=10,
            entry_price=5.50,
            entry_date=datetime(2024, 3, 1),
            underlying_price_at_entry=445.0,
        )

        with pytest.raises(OptionStructureValidationError):
            empty_structure.add_option(different_underlying)

    def test_add_non_option_raises_error(self, empty_structure):
        """Test that adding non-Option raises error."""
        with pytest.raises(OptionStructureValidationError):
            empty_structure.add_option("not an option")

    def test_remove_option_by_index(self, long_straddle):
        """Test removing option by index."""
        assert long_straddle.num_legs == 2

        removed = long_straddle.remove_option(index=0)

        assert long_straddle.num_legs == 1
        assert removed.option_type == 'call'

    def test_remove_option_by_index_invalid(self, long_straddle):
        """Test removing option with invalid index."""
        with pytest.raises(IndexError):
            long_straddle.remove_option(index=10)

    def test_remove_from_empty_raises_error(self, empty_structure):
        """Test removing from empty structure raises error."""
        with pytest.raises(EmptyStructureError):
            empty_structure.remove_option()

    def test_remove_last_option(self, long_straddle):
        """Test removing last option (no args)."""
        removed = long_straddle.remove_option()

        assert long_straddle.num_legs == 1
        assert removed.option_type == 'put'

    def test_get_option_by_index(self, long_straddle):
        """Test getting option by index."""
        call = long_straddle.get_option(0)
        put = long_straddle.get_option(1)

        assert call.option_type == 'call'
        assert put.option_type == 'put'

    def test_get_option_invalid_index(self, long_straddle):
        """Test getting option with invalid index."""
        with pytest.raises(IndexError):
            long_straddle.get_option(10)


# =============================================================================
# Net Premium Tests
# =============================================================================

class TestNetPremium:
    """Tests for net premium calculation."""

    def test_long_straddle_net_premium_debit(self, long_straddle):
        """Test long straddle has debit (negative net premium)."""
        # Long call: 5.50 * 10 * 100 = 5500 debit
        # Long put: 6.00 * 10 * 100 = 6000 debit
        # Total: -11500 (debit)
        expected = -(5.50 + 6.00) * 10 * CONTRACT_MULTIPLIER
        assert abs(long_straddle.net_premium - expected) < 0.01

    def test_short_straddle_net_premium_credit(self, short_straddle):
        """Test short straddle has credit (positive net premium)."""
        # Short call: 5.50 * 10 * 100 = 5500 credit
        # Short put: 6.00 * 10 * 100 = 6000 credit
        # Total: +11500 (credit)
        expected = (5.50 + 6.00) * 10 * CONTRACT_MULTIPLIER
        assert abs(short_straddle.net_premium - expected) < 0.01

    def test_iron_condor_net_premium(self, iron_condor):
        """Test iron condor net premium."""
        # Short put 430: +3.00 * 10 * 100 = +3000
        # Long put 420: -1.50 * 10 * 100 = -1500
        # Short call 460: +3.50 * 10 * 100 = +3500
        # Long call 470: -1.75 * 10 * 100 = -1750
        # Total: 3000 - 1500 + 3500 - 1750 = +3250
        expected = (3.00 - 1.50 + 3.50 - 1.75) * 10 * CONTRACT_MULTIPLIER
        assert abs(iron_condor.net_premium - expected) < 0.01

    def test_bull_call_spread_net_premium(self, bull_call_spread):
        """Test bull call spread net premium (debit)."""
        # Long call 445: -8.00 * 10 * 100 = -8000 (debit)
        # Short call 455: +4.00 * 10 * 100 = +4000 (credit)
        # Total: -4000 (net debit)
        expected = (4.00 - 8.00) * 10 * CONTRACT_MULTIPLIER
        assert abs(bull_call_spread.net_premium - expected) < 0.01


# =============================================================================
# Net Greeks Calculation Tests
# =============================================================================

class TestNetGreeksCalculation:
    """Tests for net Greeks calculation."""

    def test_empty_structure_raises_error(self, empty_structure):
        """Test that calculating Greeks for empty structure raises error."""
        with pytest.raises(EmptyStructureError):
            empty_structure.calculate_net_greeks(spot=450, vol=0.20)

    def test_single_leg_greeks(self, empty_structure, sample_long_call):
        """Test net Greeks for single leg equals that option's Greeks."""
        empty_structure.add_option(sample_long_call)

        net_greeks = empty_structure.calculate_net_greeks(
            spot=450.0,
            vol=0.20,
            rate=0.05,
            current_date=datetime(2024, 3, 15)
        )

        # For single long call, net Greeks = option Greeks * quantity * position_sign
        option_greeks = sample_long_call.greeks

        # Single long call with quantity 10
        assert abs(net_greeks['delta'] - option_greeks['delta'] * 10) < 0.001
        assert abs(net_greeks['gamma'] - option_greeks['gamma'] * 10) < 0.001

    def test_long_straddle_net_delta_near_zero_atm(self, long_straddle):
        """Test long straddle has near-zero delta at ATM."""
        net_greeks = long_straddle.calculate_net_greeks(
            spot=450.0,  # ATM
            vol=0.20,
            rate=0.05,
            current_date=datetime(2024, 3, 15)
        )

        # ATM straddle should have delta near zero
        # Call delta ~0.5, Put delta ~ -0.5, so net ~ 0
        # With 10 contracts each, net delta should be small relative to total
        # (Forward delta adjustment causes slight positive skew)
        assert abs(net_greeks['delta']) < 2.0  # Within 2 deltas

    def test_long_straddle_net_gamma_positive(self, long_straddle):
        """Test long straddle has positive gamma."""
        net_greeks = long_straddle.calculate_net_greeks(
            spot=450.0,
            vol=0.20,
            rate=0.05,
            current_date=datetime(2024, 3, 15)
        )

        # Both long call and long put have positive gamma
        assert net_greeks['gamma'] > 0

    def test_short_straddle_net_theta_positive(self, short_straddle):
        """Test short straddle has positive theta (collects time decay)."""
        net_greeks = short_straddle.calculate_net_greeks(
            spot=450.0,
            vol=0.20,
            rate=0.05,
            current_date=datetime(2024, 3, 15)
        )

        # Short positions: theta is positive (benefit from time decay)
        assert net_greeks['theta'] > 0

    def test_short_straddle_net_vega_negative(self, short_straddle):
        """Test short straddle has negative vega (hurt by vol increase)."""
        net_greeks = short_straddle.calculate_net_greeks(
            spot=450.0,
            vol=0.20,
            rate=0.05,
            current_date=datetime(2024, 3, 15)
        )

        # Short positions: vega is negative (hurt by vol increases)
        assert net_greeks['vega'] < 0

    def test_iron_condor_net_delta_near_zero(self, iron_condor):
        """Test iron condor has near-zero delta in center."""
        net_greeks = iron_condor.calculate_net_greeks(
            spot=445.0,  # Center of iron condor
            vol=0.20,
            rate=0.05,
            current_date=datetime(2024, 3, 15)
        )

        # Iron condor should be approximately delta neutral in center
        assert abs(net_greeks['delta']) < 5.0

    def test_is_delta_neutral(self, long_straddle):
        """Test is_delta_neutral method."""
        long_straddle.calculate_net_greeks(
            spot=450.0,
            vol=0.20,
            rate=0.05,
            current_date=datetime(2024, 3, 15)
        )

        # ATM straddle should be approximately delta neutral
        # (Forward delta adjustment causes slight skew, so use larger threshold)
        assert long_straddle.is_delta_neutral(threshold=2.0)

    def test_greeks_properties(self, long_straddle):
        """Test Greeks properties after calculation."""
        long_straddle.calculate_net_greeks(
            spot=450.0,
            vol=0.20,
            rate=0.05,
            current_date=datetime(2024, 3, 15)
        )

        # Properties should return cached values
        assert long_straddle.net_gamma > 0
        assert long_straddle.net_theta < 0  # Long pays theta
        assert long_straddle.net_vega > 0   # Long benefits from vol


# =============================================================================
# P&L Calculation Tests
# =============================================================================

class TestPnLCalculation:
    """Tests for P&L calculation."""

    def test_empty_structure_pnl_raises_error(self, empty_structure):
        """Test P&L for empty structure raises error."""
        with pytest.raises(EmptyStructureError):
            empty_structure.calculate_pnl()

    def test_pnl_at_entry_is_zero(self, long_straddle):
        """Test P&L at entry is zero."""
        pnl = long_straddle.calculate_pnl()
        assert abs(pnl) < 0.01

    def test_long_straddle_profit(self, long_straddle):
        """Test P&L for profitable long straddle."""
        # Update prices: call goes up, put goes down (big move up)
        long_straddle.options[0].update_price(15.0, datetime(2024, 4, 1))  # Call: 5.50 -> 15.0
        long_straddle.options[1].update_price(1.0, datetime(2024, 4, 1))   # Put: 6.00 -> 1.0

        pnl = long_straddle.calculate_pnl()

        # Call P&L: (15.0 - 5.50) * 10 * 100 = $9,500
        # Put P&L: (1.0 - 6.00) * 10 * 100 = -$5,000
        # Total: $4,500
        expected = ((15.0 - 5.50) + (1.0 - 6.00)) * 10 * CONTRACT_MULTIPLIER
        assert abs(pnl - expected) < 0.01

    def test_short_straddle_profit(self, short_straddle):
        """Test P&L for profitable short straddle (no move)."""
        # Prices decay: both options lose value
        short_straddle.options[0].update_price(3.0, datetime(2024, 4, 1))  # Call: 5.50 -> 3.0
        short_straddle.options[1].update_price(4.0, datetime(2024, 4, 1))  # Put: 6.00 -> 4.0

        pnl = short_straddle.calculate_pnl()

        # Short call P&L: (5.50 - 3.0) * 10 * 100 = $2,500
        # Short put P&L: (6.00 - 4.0) * 10 * 100 = $2,000
        # Total: $4,500
        expected = ((5.50 - 3.0) + (6.00 - 4.0)) * 10 * CONTRACT_MULTIPLIER
        assert abs(pnl - expected) < 0.01

    def test_pnl_percent(self, short_straddle):
        """Test P&L percentage calculation."""
        # Prices decay by 50%
        short_straddle.options[0].update_price(2.75, datetime(2024, 4, 1))  # 50% decay
        short_straddle.options[1].update_price(3.0, datetime(2024, 4, 1))   # 50% decay

        pnl_pct = short_straddle.calculate_pnl_percent()

        # Should be approximately 50% of premium captured
        assert 0.4 < pnl_pct < 0.6

    def test_current_value(self, long_straddle):
        """Test current market value calculation."""
        value = long_straddle.get_current_value()

        # Long call: 5.50 * 10 * 100 = 5500
        # Long put: 6.00 * 10 * 100 = 6000
        # Total: 11500
        expected = (5.50 + 6.00) * 10 * CONTRACT_MULTIPLIER
        assert abs(value - expected) < 0.01


# =============================================================================
# Payoff at Expiry Tests
# =============================================================================

class TestPayoffAtExpiry:
    """Tests for payoff at expiration calculations."""

    def test_long_straddle_payoff_atm(self, long_straddle):
        """Test long straddle payoff at ATM (minimum payoff = 0)."""
        spots = np.array([450.0])  # ATM
        payoffs = long_straddle.get_payoff_at_expiry(spots)

        # At strike, both call and put have zero intrinsic
        assert abs(payoffs[0]) < 0.01

    def test_long_straddle_payoff_above_strike(self, long_straddle):
        """Test long straddle payoff above strike (call has intrinsic)."""
        spots = np.array([470.0])  # 20 above strike
        payoffs = long_straddle.get_payoff_at_expiry(spots)

        # Call payoff: (470 - 450) * 10 * 100 = $20,000
        # Put payoff: 0
        expected = (470.0 - 450.0) * 10 * CONTRACT_MULTIPLIER
        assert abs(payoffs[0] - expected) < 0.01

    def test_long_straddle_payoff_below_strike(self, long_straddle):
        """Test long straddle payoff below strike (put has intrinsic)."""
        spots = np.array([430.0])  # 20 below strike
        payoffs = long_straddle.get_payoff_at_expiry(spots)

        # Call payoff: 0
        # Put payoff: (450 - 430) * 10 * 100 = $20,000
        expected = (450.0 - 430.0) * 10 * CONTRACT_MULTIPLIER
        assert abs(payoffs[0] - expected) < 0.01

    def test_short_straddle_payoff_atm(self, short_straddle):
        """Test short straddle payoff at ATM (maximum payoff = 0)."""
        spots = np.array([450.0])
        payoffs = short_straddle.get_payoff_at_expiry(spots)

        # At strike, short positions pay nothing
        assert abs(payoffs[0]) < 0.01

    def test_short_straddle_payoff_away_from_strike(self, short_straddle):
        """Test short straddle payoff away from strike (loss)."""
        spots = np.array([470.0])
        payoffs = short_straddle.get_payoff_at_expiry(spots)

        # Short call loses: -(470 - 450) * 10 * 100 = -$20,000
        expected = -(470.0 - 450.0) * 10 * CONTRACT_MULTIPLIER
        assert abs(payoffs[0] - expected) < 0.01

    def test_pnl_at_expiry(self, long_straddle):
        """Test P&L at expiry includes initial premium."""
        spots = np.array([470.0])
        pnl = long_straddle.get_pnl_at_expiry(spots)

        # Payoff: $20,000
        # Net premium: -$11,500 (debit)
        # P&L: $20,000 - $11,500 = $8,500
        payoff = (470.0 - 450.0) * 10 * CONTRACT_MULTIPLIER
        expected = payoff + long_straddle.net_premium
        assert abs(pnl[0] - expected) < 0.01

    def test_iron_condor_payoff_in_profit_zone(self, iron_condor):
        """Test iron condor payoff in profit zone (between short strikes)."""
        spots = np.array([445.0])  # Between 430 and 460
        payoffs = iron_condor.get_payoff_at_expiry(spots)

        # All options expire OTM, payoff = 0
        assert abs(payoffs[0]) < 0.01

    def test_iron_condor_payoff_at_wing(self, iron_condor):
        """Test iron condor payoff at wing (max loss)."""
        spots = np.array([410.0])  # Below lower wing
        payoffs = iron_condor.get_payoff_at_expiry(spots)

        # Short put 430: (430-410) * -10 * 100 = -$20,000
        # Long put 420: (420-410) * 10 * 100 = $10,000
        # Net: -$10,000 (max loss on put side)
        expected = ((420.0 - 410.0) - (430.0 - 410.0)) * 10 * CONTRACT_MULTIPLIER
        assert abs(payoffs[0] - expected) < 0.01


# =============================================================================
# Payoff Diagram Tests
# =============================================================================

class TestPayoffDiagram:
    """Tests for payoff diagram generation."""

    def test_payoff_diagram_shape(self, long_straddle):
        """Test payoff diagram returns correct shape."""
        spots, payoffs = long_straddle.get_payoff_diagram(
            spot_range=(400, 500),
            num_points=101
        )

        assert len(spots) == 101
        assert len(payoffs) == 101

    def test_payoff_diagram_auto_range(self, long_straddle):
        """Test payoff diagram with auto spot range."""
        spots, payoffs = long_straddle.get_payoff_diagram()

        # Should have reasonable range around strikes
        assert len(spots) > 50
        assert min(spots) < 450  # Below strike
        assert max(spots) > 450  # Above strike

    def test_payoff_diagram_minimum_at_strike(self, long_straddle):
        """Test long straddle has minimum payoff at strike."""
        spots, payoffs = long_straddle.get_payoff_diagram(
            spot_range=(400, 500),
            num_points=101
        )

        min_payoff_idx = np.argmin(payoffs)
        # Minimum should be near the strike (450)
        assert abs(spots[min_payoff_idx] - 450.0) < 5.0


# =============================================================================
# Max Profit/Loss Tests
# =============================================================================

class TestMaxProfitLoss:
    """Tests for max profit and max loss calculations."""

    def test_empty_structure_max_profit_raises_error(self, empty_structure):
        """Test max profit for empty structure raises error."""
        with pytest.raises(EmptyStructureError):
            empty_structure.calculate_max_profit()

    def test_long_straddle_max_profit_unlimited(self, long_straddle):
        """Test long straddle has high max profit (unlimited potential)."""
        max_profit = long_straddle.calculate_max_profit()

        # Should be large (evaluated at 3x strike)
        assert max_profit > 0
        # At spot = 1350 (3x 450), call intrinsic = 900 * 10 * 100 = $900,000
        # Minus premium = $900,000 - $11,500 = ~$888,500

    def test_long_straddle_max_loss(self, long_straddle):
        """Test long straddle max loss is limited to premium paid."""
        max_loss = long_straddle.calculate_max_loss()

        # Max loss occurs at strike where both options expire worthless
        # Max loss = -premium = -$11,500
        expected_max_loss = long_straddle.net_premium  # Already negative
        # Allow for numerical tolerance (spot_range discretization)
        assert abs(max_loss - expected_max_loss) < 500  # Allow numerical tolerance

    def test_short_straddle_max_profit(self, short_straddle):
        """Test short straddle max profit is limited to premium received."""
        max_profit = short_straddle.calculate_max_profit()

        # Max profit occurs at strike = premium received = $11,500
        expected = short_straddle.net_premium
        # Allow for numerical tolerance (spot_range discretization)
        assert abs(max_profit - expected) < 500

    def test_short_straddle_max_loss_large(self, short_straddle):
        """Test short straddle has large max loss potential."""
        max_loss = short_straddle.calculate_max_loss()

        # Should be very negative (large loss potential)
        assert max_loss < -50000

    def test_iron_condor_max_profit(self, iron_condor):
        """Test iron condor max profit = net credit."""
        max_profit = iron_condor.calculate_max_profit()

        # Max profit = net premium = $3,250
        expected = iron_condor.net_premium
        assert abs(max_profit - expected) < 100

    def test_iron_condor_max_loss(self, iron_condor):
        """Test iron condor max loss = spread width - credit."""
        max_loss = iron_condor.calculate_max_loss()

        # Spread width = 10 * 10 * 100 = $10,000
        # Max loss = -10,000 + 3,250 = -$6,750
        spread_width = 10.0 * 10 * CONTRACT_MULTIPLIER
        expected = -spread_width + iron_condor.net_premium
        assert abs(max_loss - expected) < 200

    def test_bull_call_spread_max_profit(self, bull_call_spread):
        """Test bull call spread max profit."""
        max_profit = bull_call_spread.calculate_max_profit()

        # Spread width = (455 - 445) * 10 * 100 = $10,000
        # Net debit = -$4,000
        # Max profit = 10,000 - 4,000 = $6,000
        spread_width = (455 - 445) * 10 * CONTRACT_MULTIPLIER
        expected = spread_width + bull_call_spread.net_premium
        assert abs(max_profit - expected) < 100

    def test_bull_call_spread_max_loss(self, bull_call_spread):
        """Test bull call spread max loss = debit paid."""
        max_loss = bull_call_spread.calculate_max_loss()

        # Max loss = debit = -$4,000
        expected = bull_call_spread.net_premium  # Already negative
        assert abs(max_loss - expected) < 100


# =============================================================================
# Breakeven Calculation Tests
# =============================================================================

class TestBreakevenCalculation:
    """Tests for breakeven point calculations."""

    def test_empty_structure_breakeven_raises_error(self, empty_structure):
        """Test breakeven for empty structure raises error."""
        with pytest.raises(EmptyStructureError):
            empty_structure.calculate_breakeven_points()

    def test_long_straddle_two_breakevens(self, long_straddle):
        """Test long straddle has two breakeven points."""
        breakevens = long_straddle.calculate_breakeven_points()

        assert len(breakevens) == 2

        # Premium per share = (5.50 + 6.00) = 11.50
        # Lower BE = 450 - 11.50 = 438.50
        # Upper BE = 450 + 11.50 = 461.50
        lower_expected = 450.0 - (5.50 + 6.00)
        upper_expected = 450.0 + (5.50 + 6.00)

        assert abs(breakevens[0] - lower_expected) < 0.5
        assert abs(breakevens[1] - upper_expected) < 0.5

    def test_short_straddle_two_breakevens(self, short_straddle):
        """Test short straddle has two breakeven points."""
        breakevens = short_straddle.calculate_breakeven_points()

        assert len(breakevens) == 2

        # Same breakevens as long straddle
        lower_expected = 450.0 - (5.50 + 6.00)
        upper_expected = 450.0 + (5.50 + 6.00)

        assert abs(breakevens[0] - lower_expected) < 0.5
        assert abs(breakevens[1] - upper_expected) < 0.5

    def test_iron_condor_four_breakevens_or_two(self, iron_condor):
        """Test iron condor breakeven points."""
        breakevens = iron_condor.calculate_breakeven_points()

        # Iron condor should have 2 breakevens (one on each side)
        # Lower BE: around 430 - some credit offset
        # Upper BE: around 460 + some credit offset
        assert len(breakevens) >= 2

    def test_bull_call_spread_one_breakeven(self, bull_call_spread):
        """Test bull call spread has one breakeven."""
        breakevens = bull_call_spread.calculate_breakeven_points()

        # BE = lower strike + debit/share = 445 + 4 = 449
        assert len(breakevens) >= 1

        debit_per_share = (8.00 - 4.00)
        expected = 445.0 + debit_per_share

        assert abs(breakevens[0] - expected) < 0.5


# =============================================================================
# Expiration Helper Tests
# =============================================================================

class TestExpirationHelpers:
    """Tests for expiration-related helper methods."""

    def test_earliest_expiration(self, iron_condor):
        """Test getting earliest expiration."""
        exp = iron_condor.get_earliest_expiration()
        assert exp == datetime(2024, 6, 21)

    def test_latest_expiration(self, iron_condor):
        """Test getting latest expiration."""
        exp = iron_condor.get_latest_expiration()
        assert exp == datetime(2024, 6, 21)

    def test_expiration_empty_structure(self, empty_structure):
        """Test expiration for empty structure."""
        assert empty_structure.get_earliest_expiration() is None
        assert empty_structure.get_latest_expiration() is None

    def test_is_same_expiration(self, iron_condor):
        """Test all legs same expiration."""
        assert iron_condor.is_same_expiration() == True

    def test_days_to_expiry(self, long_straddle):
        """Test days to expiry calculation."""
        dte = long_straddle.get_days_to_expiry(
            current_date=datetime(2024, 3, 1)
        )

        # March 1 to June 21 = 112 days
        assert dte == 112


# =============================================================================
# Multi-Expiration Structure Tests
# =============================================================================

class TestMultiExpirationStructure:
    """Tests for structures with multiple expirations."""

    def test_calendar_spread_different_expirations(self):
        """Test structure with different expirations (calendar spread)."""
        structure = OptionStructure(
            structure_type='calendar_spread',
            underlying='SPY'
        )

        # Near-term short call
        structure.add_option(Option(
            option_type='call',
            position_type='short',
            underlying='SPY',
            strike=450.0,
            expiration=datetime(2024, 4, 19),  # Near
            quantity=10,
            entry_price=4.00,
            entry_date=datetime(2024, 3, 1),
            underlying_price_at_entry=445.0,
        ))

        # Far-term long call
        structure.add_option(Option(
            option_type='call',
            position_type='long',
            underlying='SPY',
            strike=450.0,
            expiration=datetime(2024, 6, 21),  # Far
            quantity=10,
            entry_price=7.00,
            entry_date=datetime(2024, 3, 1),
            underlying_price_at_entry=445.0,
        ))

        assert structure.is_same_expiration() == False
        assert structure.get_earliest_expiration() == datetime(2024, 4, 19)
        assert structure.get_latest_expiration() == datetime(2024, 6, 21)


# =============================================================================
# Serialization Tests
# =============================================================================

class TestSerialization:
    """Tests for serialization methods."""

    def test_to_dict(self, long_straddle):
        """Test to_dict method."""
        d = long_straddle.to_dict()

        assert d['structure_type'] == 'straddle'
        assert d['underlying'] == 'SPY'
        assert d['num_legs'] == 2
        assert len(d['options']) == 2
        assert d['net_premium'] == long_straddle.net_premium

    def test_from_dict_roundtrip(self, long_straddle):
        """Test from_dict roundtrip."""
        d = long_straddle.to_dict()
        restored = OptionStructure.from_dict(d)

        assert restored.structure_type == long_straddle.structure_type
        assert restored.underlying == long_straddle.underlying
        assert restored.num_legs == long_straddle.num_legs
        assert abs(restored.net_premium - long_straddle.net_premium) < 0.01

    def test_from_dict_with_options(self, iron_condor):
        """Test from_dict preserves all options."""
        d = iron_condor.to_dict()
        restored = OptionStructure.from_dict(d)

        assert restored.num_legs == 4

        # Check strikes are preserved
        strikes = [opt.strike for opt in restored.options]
        assert 420.0 in strikes
        assert 430.0 in strikes
        assert 460.0 in strikes
        assert 470.0 in strikes


# =============================================================================
# String Representation Tests
# =============================================================================

class TestStringRepresentation:
    """Tests for string representations."""

    def test_str_representation(self, long_straddle):
        """Test __str__ method."""
        s = str(long_straddle)

        assert 'STRADDLE' in s
        assert 'SPY' in s

    def test_repr_representation(self, long_straddle):
        """Test __repr__ method."""
        r = repr(long_straddle)

        assert 'OptionStructure(' in r
        assert 'straddle' in r
        assert 'SPY' in r

    def test_empty_structure_str(self, empty_structure):
        """Test string for empty structure."""
        s = str(empty_structure)
        assert 'empty' in s.lower()


# =============================================================================
# Iterator and Container Tests
# =============================================================================

class TestIteratorContainer:
    """Tests for iterator and container methods."""

    def test_len(self, iron_condor):
        """Test __len__."""
        assert len(iron_condor) == 4

    def test_iteration(self, long_straddle):
        """Test iteration over options."""
        options = list(long_straddle)

        assert len(options) == 2
        assert options[0].option_type == 'call'
        assert options[1].option_type == 'put'

    def test_getitem(self, long_straddle):
        """Test __getitem__."""
        call = long_straddle[0]
        put = long_straddle[1]

        assert call.option_type == 'call'
        assert put.option_type == 'put'


# =============================================================================
# Equality and Hashing Tests
# =============================================================================

class TestEqualityHashing:
    """Tests for equality and hashing."""

    def test_equality_by_id(self):
        """Test structures with same ID are equal."""
        s1 = OptionStructure(structure_type='straddle', structure_id='test123')
        s2 = OptionStructure(structure_type='straddle', structure_id='test123')

        assert s1 == s2

    def test_inequality_different_id(self):
        """Test structures with different ID are not equal."""
        s1 = OptionStructure(structure_type='straddle', structure_id='test123')
        s2 = OptionStructure(structure_type='straddle', structure_id='test456')

        assert s1 != s2

    def test_hashable(self):
        """Test structures can be used in sets."""
        s1 = OptionStructure(structure_type='straddle', structure_id='test123')
        s2 = OptionStructure(structure_type='straddle', structure_id='test456')

        s = {s1, s2}
        assert len(s) == 2


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_leg_structure(self, sample_long_call):
        """Test structure with single leg."""
        structure = OptionStructure(structure_type='naked_call')
        structure.add_option(sample_long_call)

        # All operations should work
        assert structure.num_legs == 1
        pnl = structure.calculate_pnl()
        assert abs(pnl) < 0.01

        payoff = structure.get_payoff_at_expiry(np.array([450.0]))
        assert payoff[0] == 0.0  # ATM call payoff = 0

    def test_zero_quantity_option(self):
        """Test that zero quantity option is rejected."""
        structure = OptionStructure(structure_type='custom')

        with pytest.raises(Exception):  # OptionValidationError
            structure.add_option(Option(
                option_type='call',
                position_type='long',
                underlying='SPY',
                strike=450.0,
                expiration=datetime(2024, 6, 21),
                quantity=0,  # Invalid!
                entry_price=5.00,
                entry_date=datetime(2024, 3, 1),
                underlying_price_at_entry=445.0,
            ))

    def test_very_wide_spot_range(self, long_straddle):
        """Test calculations with very wide spot range."""
        spots = np.linspace(1, 10000, 1000)
        payoffs = long_straddle.get_payoff_at_expiry(spots)

        # Should handle without errors
        assert len(payoffs) == 1000
        assert np.all(np.isfinite(payoffs))

    def test_remove_and_readd_option(self, long_straddle, sample_long_call):
        """Test removing and re-adding option."""
        initial_premium = long_straddle.net_premium

        removed = long_straddle.remove_option(index=0)
        premium_after_remove = long_straddle.net_premium

        # Premium should have changed
        assert premium_after_remove != initial_premium

        long_straddle.add_option(removed)

        # Premium should be restored
        assert abs(long_straddle.net_premium - initial_premium) < 0.01


# =============================================================================
# Financial Correctness Tests
# =============================================================================

class TestFinancialCorrectness:
    """Tests for financial correctness of calculations."""

    def test_long_call_plus_short_call_cancels(self):
        """Test long call + short call at same strike = zero position."""
        structure = OptionStructure(structure_type='box')

        # Long call
        structure.add_option(Option(
            option_type='call',
            position_type='long',
            underlying='SPY',
            strike=450.0,
            expiration=datetime(2024, 6, 21),
            quantity=10,
            entry_price=5.00,
            entry_date=datetime(2024, 3, 1),
            underlying_price_at_entry=445.0,
        ))

        # Short call at same strike
        structure.add_option(Option(
            option_type='call',
            position_type='short',
            underlying='SPY',
            strike=450.0,
            expiration=datetime(2024, 6, 21),
            quantity=10,
            entry_price=5.00,
            entry_date=datetime(2024, 3, 1),
            underlying_price_at_entry=445.0,
        ))

        # Net premium should be zero
        assert abs(structure.net_premium) < 0.01

        # Net delta should be zero
        greeks = structure.calculate_net_greeks(
            spot=445.0,
            vol=0.20,
            rate=0.05,
            current_date=datetime(2024, 3, 15)
        )
        assert abs(greeks['delta']) < 0.001

        # Payoff should be zero everywhere
        payoffs = structure.get_payoff_at_expiry(np.array([400, 450, 500]))
        assert np.allclose(payoffs, 0, atol=0.01)

    def test_synthetic_long_stock(self):
        """Test long call + short put = synthetic long stock."""
        structure = OptionStructure(structure_type='synthetic_long')

        # Long call at 450
        structure.add_option(Option(
            option_type='call',
            position_type='long',
            underlying='SPY',
            strike=450.0,
            expiration=datetime(2024, 6, 21),
            quantity=10,
            entry_price=5.00,
            entry_date=datetime(2024, 3, 1),
            underlying_price_at_entry=445.0,
        ))

        # Short put at 450
        structure.add_option(Option(
            option_type='put',
            position_type='short',
            underlying='SPY',
            strike=450.0,
            expiration=datetime(2024, 6, 21),
            quantity=10,
            entry_price=5.00,
            entry_date=datetime(2024, 3, 1),
            underlying_price_at_entry=445.0,
        ))

        # Net delta should be approximately 1 per share (10 contracts = delta 10)
        greeks = structure.calculate_net_greeks(
            spot=450.0,
            vol=0.20,
            rate=0.05,
            current_date=datetime(2024, 3, 15)
        )

        # Delta should be close to quantity (synthetic stock behavior)
        # Call delta ~0.5 + Short put delta (which is -(-0.5) = 0.5) = 1.0 per contract
        assert abs(greeks['delta'] - 10.0) < 1.0  # ~10 deltas

    def test_put_call_parity_in_structure(self):
        """Test put-call parity relationship in structures."""
        # Long call + Short put + K*e^(-rT) = S
        # At expiry: Long call + Short put payoff = S - K

        structure = OptionStructure(structure_type='conversion')

        # Long call at 450
        structure.add_option(Option(
            option_type='call',
            position_type='long',
            underlying='SPY',
            strike=450.0,
            expiration=datetime(2024, 6, 21),
            quantity=1,
            entry_price=5.00,
            entry_date=datetime(2024, 3, 1),
            underlying_price_at_entry=445.0,
        ))

        # Short put at 450
        structure.add_option(Option(
            option_type='put',
            position_type='short',
            underlying='SPY',
            strike=450.0,
            expiration=datetime(2024, 6, 21),
            quantity=1,
            entry_price=5.00,
            entry_date=datetime(2024, 3, 1),
            underlying_price_at_entry=445.0,
        ))

        # Payoff at various spots should be linear: (S - K) * 100
        spots = np.array([400, 425, 450, 475, 500])
        payoffs = structure.get_payoff_at_expiry(spots)

        expected_payoffs = (spots - 450.0) * CONTRACT_MULTIPLIER
        np.testing.assert_allclose(payoffs, expected_payoffs, atol=0.01)


# =============================================================================
# Price Update Tests
# =============================================================================

class TestPriceUpdates:
    """Tests for price update methods."""

    def test_update_all_prices(self, long_straddle):
        """Test updating all prices."""
        long_straddle.update_all_prices(
            prices={0: 8.00, 1: 9.00},
            timestamp=datetime(2024, 4, 1)
        )

        assert long_straddle.options[0].current_price == 8.00
        assert long_straddle.options[1].current_price == 9.00

    def test_update_partial_prices(self, long_straddle):
        """Test updating only some prices."""
        original_put_price = long_straddle.options[1].current_price

        long_straddle.update_all_prices(
            prices={0: 8.00},  # Only update call
            timestamp=datetime(2024, 4, 1)
        )

        assert long_straddle.options[0].current_price == 8.00
        assert long_straddle.options[1].current_price == original_put_price


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
