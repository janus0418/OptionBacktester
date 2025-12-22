"""
Comprehensive tests for concrete option structures.

Tests all concrete structures:
    - LongStraddle, ShortStraddle
    - LongStrangle, ShortStrangle
    - BullCallSpread, BearPutSpread, BullPutSpread, BearCallSpread
    - IronCondor, IronButterfly

Each structure is tested for:
    - Factory method creation
    - Max profit/loss calculations
    - Breakeven calculations
    - Validation (strike ordering, expiration, etc.)
    - Edge cases
"""

import pytest
from datetime import datetime, timedelta
import numpy as np

from backtester.structures.straddle import LongStraddle, ShortStraddle
from backtester.structures.strangle import LongStrangle, ShortStrangle
from backtester.structures.spread import (
    BullCallSpread,
    BearPutSpread,
    BullPutSpread,
    BearCallSpread,
)
from backtester.structures.condor import IronCondor, IronButterfly
from backtester.core.option_structure import OptionStructureValidationError


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def base_params():
    """Common parameters for structure creation."""
    return {
        'underlying': 'SPY',
        'expiration': datetime(2024, 3, 15),
        'quantity': 10,
        'entry_date': datetime(2024, 3, 1),
        'underlying_price': 450.0,
    }


# =============================================================================
# Long Straddle Tests
# =============================================================================

class TestLongStraddle:
    """Tests for LongStraddle structure."""

    def test_create_long_straddle(self, base_params):
        """Test basic creation of long straddle."""
        straddle = LongStraddle.create(
            strike=450.0,
            call_price=6.50,
            put_price=6.25,
            **base_params
        )

        assert straddle.strike == 450.0
        assert straddle.num_legs == 2
        assert straddle.structure_type == 'long_straddle'
        assert straddle.call_option.is_long
        assert straddle.put_option.is_long

    def test_long_straddle_max_loss(self, base_params):
        """Test max loss calculation for long straddle."""
        straddle = LongStraddle.create(
            strike=450.0,
            call_price=6.50,
            put_price=6.25,
            **base_params
        )

        # Max loss = total premium paid
        expected_max_loss = (6.50 + 6.25) * 10 * 100
        assert abs(straddle.max_loss - expected_max_loss) < 1e-6

    def test_long_straddle_max_profit(self, base_params):
        """Test max profit is unlimited."""
        straddle = LongStraddle.create(
            strike=450.0,
            call_price=6.50,
            put_price=6.25,
            **base_params
        )

        assert straddle.max_profit == float('inf')

    def test_long_straddle_breakevens(self, base_params):
        """Test breakeven calculation."""
        straddle = LongStraddle.create(
            strike=450.0,
            call_price=6.50,
            put_price=6.25,
            **base_params
        )

        # Breakevens: strike ± total premium
        total_premium = 6.50 + 6.25
        expected_upper = 450.0 + total_premium
        expected_lower = 450.0 - total_premium

        assert abs(straddle.upper_breakeven - expected_upper) < 1e-6
        assert abs(straddle.lower_breakeven - expected_lower) < 1e-6

    def test_long_straddle_breakeven_range(self, base_params):
        """Test breakeven range calculation."""
        straddle = LongStraddle.create(
            strike=450.0,
            call_price=6.50,
            put_price=6.25,
            **base_params
        )

        expected_range = 2 * (6.50 + 6.25)
        assert abs(straddle.breakeven_range - expected_range) < 1e-6


# =============================================================================
# Short Straddle Tests
# =============================================================================

class TestShortStraddle:
    """Tests for ShortStraddle structure."""

    def test_create_short_straddle(self, base_params):
        """Test basic creation of short straddle."""
        straddle = ShortStraddle.create(
            strike=450.0,
            call_price=6.50,
            put_price=6.25,
            **base_params
        )

        assert straddle.strike == 450.0
        assert straddle.num_legs == 2
        assert straddle.structure_type == 'short_straddle'
        assert straddle.call_option.is_short
        assert straddle.put_option.is_short

    def test_short_straddle_max_profit(self, base_params):
        """Test max profit is total premium."""
        straddle = ShortStraddle.create(
            strike=450.0,
            call_price=6.50,
            put_price=6.25,
            **base_params
        )

        expected_max_profit = (6.50 + 6.25) * 10 * 100
        assert abs(straddle.max_profit - expected_max_profit) < 1e-6

    def test_short_straddle_max_loss(self, base_params):
        """Test max loss is unlimited."""
        straddle = ShortStraddle.create(
            strike=450.0,
            call_price=6.50,
            put_price=6.25,
            **base_params
        )

        assert straddle.max_loss == float('-inf')

    def test_short_straddle_breakevens(self, base_params):
        """Test breakeven calculation."""
        straddle = ShortStraddle.create(
            strike=450.0,
            call_price=6.50,
            put_price=6.25,
            **base_params
        )

        total_premium = 6.50 + 6.25
        expected_upper = 450.0 + total_premium
        expected_lower = 450.0 - total_premium

        assert abs(straddle.upper_breakeven - expected_upper) < 1e-6
        assert abs(straddle.lower_breakeven - expected_lower) < 1e-6


# =============================================================================
# Long Strangle Tests
# =============================================================================

class TestLongStrangle:
    """Tests for LongStrangle structure."""

    def test_create_long_strangle(self, base_params):
        """Test creation of long strangle."""
        strangle = LongStrangle.create(
            call_strike=460.0,
            put_strike=440.0,
            call_price=3.50,
            put_price=3.25,
            **base_params
        )

        assert strangle.call_strike == 460.0
        assert strangle.put_strike == 440.0
        assert strangle.num_legs == 2
        assert strangle.structure_type == 'long_strangle'

    def test_long_strangle_strike_validation(self, base_params):
        """Test validation of strike ordering."""
        with pytest.raises(OptionStructureValidationError):
            LongStrangle.create(
                call_strike=440.0,  # Should be higher than put
                put_strike=460.0,
                call_price=3.50,
                put_price=3.25,
                **base_params
            )

    def test_long_strangle_max_loss(self, base_params):
        """Test max loss calculation."""
        strangle = LongStrangle.create(
            call_strike=460.0,
            put_strike=440.0,
            call_price=3.50,
            put_price=3.25,
            **base_params
        )

        expected_max_loss = (3.50 + 3.25) * 10 * 100
        assert abs(strangle.max_loss - expected_max_loss) < 1e-6

    def test_long_strangle_strike_width(self, base_params):
        """Test strike width calculation."""
        strangle = LongStrangle.create(
            call_strike=460.0,
            put_strike=440.0,
            call_price=3.50,
            put_price=3.25,
            **base_params
        )

        assert strangle.strike_width == 20.0


# =============================================================================
# Short Strangle Tests
# =============================================================================

class TestShortStrangle:
    """Tests for ShortStrangle structure."""

    def test_create_short_strangle(self, base_params):
        """Test creation of short strangle."""
        strangle = ShortStrangle.create(
            call_strike=460.0,
            put_strike=440.0,
            call_price=3.50,
            put_price=3.25,
            **base_params
        )

        assert strangle.call_strike == 460.0
        assert strangle.put_strike == 440.0
        assert strangle.structure_type == 'short_strangle'
        assert strangle.call_option.is_short
        assert strangle.put_option.is_short

    def test_short_strangle_max_profit(self, base_params):
        """Test max profit calculation."""
        strangle = ShortStrangle.create(
            call_strike=460.0,
            put_strike=440.0,
            call_price=3.50,
            put_price=3.25,
            **base_params
        )

        expected_max_profit = (3.50 + 3.25) * 10 * 100
        assert abs(strangle.max_profit - expected_max_profit) < 1e-6


# =============================================================================
# Bull Call Spread Tests
# =============================================================================

class TestBullCallSpread:
    """Tests for BullCallSpread structure."""

    def test_create_bull_call_spread(self, base_params):
        """Test creation of bull call spread."""
        spread = BullCallSpread.create(
            long_strike=450.0,
            short_strike=460.0,
            long_price=6.50,
            short_price=3.00,
            **base_params
        )

        assert spread.long_strike == 450.0
        assert spread.short_strike == 460.0
        assert spread.structure_type == 'bull_call_spread'

    def test_bull_call_spread_max_profit(self, base_params):
        """Test max profit calculation."""
        spread = BullCallSpread.create(
            long_strike=450.0,
            short_strike=460.0,
            long_price=6.50,
            short_price=3.00,
            **base_params
        )

        # Max profit = spread width - net debit
        spread_width = (460.0 - 450.0) * 10 * 100
        net_debit = (6.50 - 3.00) * 10 * 100
        expected_max_profit = spread_width - net_debit

        assert abs(spread.max_profit - expected_max_profit) < 1e-6

    def test_bull_call_spread_max_loss(self, base_params):
        """Test max loss is net debit."""
        spread = BullCallSpread.create(
            long_strike=450.0,
            short_strike=460.0,
            long_price=6.50,
            short_price=3.00,
            **base_params
        )

        expected_max_loss = (6.50 - 3.00) * 10 * 100
        assert abs(spread.max_loss - expected_max_loss) < 1e-6

    def test_bull_call_spread_breakeven(self, base_params):
        """Test breakeven calculation."""
        spread = BullCallSpread.create(
            long_strike=450.0,
            short_strike=460.0,
            long_price=6.50,
            short_price=3.00,
            **base_params
        )

        # Breakeven = long strike + net debit
        expected_breakeven = 450.0 + (6.50 - 3.00)
        assert abs(spread.breakeven - expected_breakeven) < 1e-6

    def test_bull_call_spread_strike_validation(self, base_params):
        """Test validation of strike ordering."""
        with pytest.raises(OptionStructureValidationError):
            BullCallSpread.create(
                long_strike=460.0,  # Should be lower
                short_strike=450.0,
                long_price=3.00,
                short_price=6.50,
                **base_params
            )


# =============================================================================
# Bear Put Spread Tests
# =============================================================================

class TestBearPutSpread:
    """Tests for BearPutSpread structure."""

    def test_create_bear_put_spread(self, base_params):
        """Test creation of bear put spread."""
        spread = BearPutSpread.create(
            long_strike=450.0,
            short_strike=440.0,
            long_price=6.50,
            short_price=3.00,
            **base_params
        )

        assert spread.long_strike == 450.0
        assert spread.short_strike == 440.0
        assert spread.structure_type == 'bear_put_spread'

    def test_bear_put_spread_max_profit(self, base_params):
        """Test max profit calculation."""
        spread = BearPutSpread.create(
            long_strike=450.0,
            short_strike=440.0,
            long_price=6.50,
            short_price=3.00,
            **base_params
        )

        spread_width = (450.0 - 440.0) * 10 * 100
        net_debit = (6.50 - 3.00) * 10 * 100
        expected_max_profit = spread_width - net_debit

        assert abs(spread.max_profit - expected_max_profit) < 1e-6

    def test_bear_put_spread_strike_validation(self, base_params):
        """Test validation of strike ordering."""
        with pytest.raises(OptionStructureValidationError):
            BearPutSpread.create(
                long_strike=440.0,  # Should be higher
                short_strike=450.0,
                long_price=3.00,
                short_price=6.50,
                **base_params
            )


# =============================================================================
# Bull Put Spread (Credit) Tests
# =============================================================================

class TestBullPutSpread:
    """Tests for BullPutSpread structure."""

    def test_create_bull_put_spread(self, base_params):
        """Test creation of bull put spread."""
        spread = BullPutSpread.create(
            long_strike=440.0,
            short_strike=450.0,
            long_price=3.00,
            short_price=6.50,
            **base_params
        )

        assert spread.long_strike == 440.0
        assert spread.short_strike == 450.0
        assert spread.structure_type == 'bull_put_spread'

    def test_bull_put_spread_max_profit(self, base_params):
        """Test max profit is net credit."""
        spread = BullPutSpread.create(
            long_strike=440.0,
            short_strike=450.0,
            long_price=3.00,
            short_price=6.50,
            **base_params
        )

        expected_max_profit = (6.50 - 3.00) * 10 * 100
        assert abs(spread.max_profit - expected_max_profit) < 1e-6

    def test_bull_put_spread_max_loss(self, base_params):
        """Test max loss calculation."""
        spread = BullPutSpread.create(
            long_strike=440.0,
            short_strike=450.0,
            long_price=3.00,
            short_price=6.50,
            **base_params
        )

        spread_width = (450.0 - 440.0) * 10 * 100
        net_credit = (6.50 - 3.00) * 10 * 100
        expected_max_loss = spread_width - net_credit

        assert abs(spread.max_loss - expected_max_loss) < 1e-6


# =============================================================================
# Bear Call Spread (Credit) Tests
# =============================================================================

class TestBearCallSpread:
    """Tests for BearCallSpread structure."""

    def test_create_bear_call_spread(self, base_params):
        """Test creation of bear call spread."""
        spread = BearCallSpread.create(
            long_strike=460.0,
            short_strike=450.0,
            long_price=3.00,
            short_price=6.50,
            **base_params
        )

        assert spread.long_strike == 460.0
        assert spread.short_strike == 450.0
        assert spread.structure_type == 'bear_call_spread'

    def test_bear_call_spread_max_profit(self, base_params):
        """Test max profit is net credit."""
        spread = BearCallSpread.create(
            long_strike=460.0,
            short_strike=450.0,
            long_price=3.00,
            short_price=6.50,
            **base_params
        )

        expected_max_profit = (6.50 - 3.00) * 10 * 100
        assert abs(spread.max_profit - expected_max_profit) < 1e-6

    def test_bear_call_spread_spread_width(self, base_params):
        """Test spread width calculation."""
        spread = BearCallSpread.create(
            long_strike=460.0,
            short_strike=450.0,
            long_price=3.00,
            short_price=6.50,
            **base_params
        )

        assert spread.spread_width == 10.0


# =============================================================================
# Iron Condor Tests
# =============================================================================

class TestIronCondor:
    """Tests for IronCondor structure."""

    def test_create_iron_condor(self, base_params):
        """Test creation of iron condor."""
        ic = IronCondor.create(
            put_buy_strike=430.0,
            put_sell_strike=440.0,
            call_sell_strike=460.0,
            call_buy_strike=470.0,
            put_buy_price=1.50,
            put_sell_price=3.00,
            call_sell_price=3.25,
            call_buy_price=1.75,
            **base_params
        )

        assert ic.put_buy_strike == 430.0
        assert ic.put_sell_strike == 440.0
        assert ic.call_sell_strike == 460.0
        assert ic.call_buy_strike == 470.0
        assert ic.structure_type == 'iron_condor'
        assert ic.num_legs == 4

    def test_iron_condor_max_profit(self, base_params):
        """Test max profit is net credit."""
        ic = IronCondor.create(
            put_buy_strike=430.0,
            put_sell_strike=440.0,
            call_sell_strike=460.0,
            call_buy_strike=470.0,
            put_buy_price=1.50,
            put_sell_price=3.00,
            call_sell_price=3.25,
            call_buy_price=1.75,
            **base_params
        )

        # Net credit = (put sell + call sell) - (put buy + call buy)
        expected_credit = (3.00 + 3.25 - 1.50 - 1.75) * 10 * 100
        assert abs(ic.max_profit - expected_credit) < 1e-6

    def test_iron_condor_wing_widths(self, base_params):
        """Test wing width calculations."""
        ic = IronCondor.create(
            put_buy_strike=430.0,
            put_sell_strike=440.0,
            call_sell_strike=460.0,
            call_buy_strike=470.0,
            put_buy_price=1.50,
            put_sell_price=3.00,
            call_sell_price=3.25,
            call_buy_price=1.75,
            **base_params
        )

        assert ic.put_wing_width == 10.0
        assert ic.call_wing_width == 10.0

    def test_iron_condor_body_width(self, base_params):
        """Test body width calculation."""
        ic = IronCondor.create(
            put_buy_strike=430.0,
            put_sell_strike=440.0,
            call_sell_strike=460.0,
            call_buy_strike=470.0,
            put_buy_price=1.50,
            put_sell_price=3.00,
            call_sell_price=3.25,
            call_buy_price=1.75,
            **base_params
        )

        assert ic.body_width == 20.0

    def test_iron_condor_strike_validation(self, base_params):
        """Test validation of strike ordering."""
        with pytest.raises(OptionStructureValidationError):
            IronCondor.create(
                put_buy_strike=440.0,  # Wrong order
                put_sell_strike=430.0,
                call_sell_strike=460.0,
                call_buy_strike=470.0,
                put_buy_price=1.50,
                put_sell_price=3.00,
                call_sell_price=3.25,
                call_buy_price=1.75,
                **base_params
            )


# =============================================================================
# Iron Butterfly Tests
# =============================================================================

class TestIronButterfly:
    """Tests for IronButterfly structure."""

    def test_create_iron_butterfly(self, base_params):
        """Test creation of iron butterfly."""
        ib = IronButterfly.create(
            lower_strike=440.0,
            middle_strike=450.0,
            upper_strike=460.0,
            lower_price=2.00,
            middle_put_price=6.50,
            middle_call_price=6.75,
            upper_price=2.25,
            **base_params
        )

        assert ib.lower_strike == 440.0
        assert ib.middle_strike == 450.0
        assert ib.upper_strike == 460.0
        assert ib.structure_type == 'iron_butterfly'
        assert ib.num_legs == 4

    def test_iron_butterfly_max_profit(self, base_params):
        """Test max profit is net credit."""
        ib = IronButterfly.create(
            lower_strike=440.0,
            middle_strike=450.0,
            upper_strike=460.0,
            lower_price=2.00,
            middle_put_price=6.50,
            middle_call_price=6.75,
            upper_price=2.25,
            **base_params
        )

        # Net credit = (middle put + middle call) - (lower + upper)
        expected_credit = (6.50 + 6.75 - 2.00 - 2.25) * 10 * 100
        assert abs(ib.max_profit - expected_credit) < 1e-6

    def test_iron_butterfly_wing_widths(self, base_params):
        """Test wing width calculations."""
        ib = IronButterfly.create(
            lower_strike=440.0,
            middle_strike=450.0,
            upper_strike=460.0,
            lower_price=2.00,
            middle_put_price=6.50,
            middle_call_price=6.75,
            upper_price=2.25,
            **base_params
        )

        assert ib.put_wing_width == 10.0
        assert ib.call_wing_width == 10.0

    def test_iron_butterfly_strike_validation(self, base_params):
        """Test validation of strike ordering."""
        with pytest.raises(OptionStructureValidationError):
            IronButterfly.create(
                lower_strike=460.0,  # Wrong order
                middle_strike=450.0,
                upper_strike=440.0,
                lower_price=2.00,
                middle_put_price=6.50,
                middle_call_price=6.75,
                upper_price=2.25,
                **base_params
            )


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_zero_premium_straddle(self, base_params):
        """Test straddle with zero premiums."""
        straddle = ShortStraddle.create(
            strike=450.0,
            call_price=0.0,
            put_price=0.0,
            **base_params
        )

        assert straddle.max_profit == 0.0
        assert straddle.upper_breakeven == 450.0
        assert straddle.lower_breakeven == 450.0

    def test_very_wide_strangle(self, base_params):
        """Test strangle with very wide strikes."""
        strangle = LongStrangle.create(
            call_strike=500.0,
            put_strike=400.0,
            call_price=1.00,
            put_price=1.00,
            **base_params
        )

        assert strangle.strike_width == 100.0
        assert strangle.max_loss == 2.00 * 10 * 100

    def test_minimum_spread_width(self, base_params):
        """Test spread with minimum width."""
        spread = BullCallSpread.create(
            long_strike=450.0,
            short_strike=451.0,
            long_price=5.00,
            short_price=4.50,
            **base_params
        )

        assert spread.spread_width == 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
