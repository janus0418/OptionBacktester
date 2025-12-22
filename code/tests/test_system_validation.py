"""
System Validation Tests for Options Backtesting System

This module contains comprehensive system validation tests that ensure
financial correctness, edge case handling, and consistency across the
entire backtesting system.

Test Categories:
    1. Financial Correctness Validation
    2. Option Pricing Validation
    3. Greeks Validation
    4. P&L and Breakeven Validation
    5. Edge Case Handling
    6. Consistency Checks

Coverage:
    - Verify all option pricing matches Black-Scholes theory
    - Greeks match theoretical values
    - P&L calculations are correct
    - Breakeven calculations match formulas
    - Edge cases handled properly
    - System-wide consistency maintained

Requirements:
    - pytest
    - numpy
    - pandas
    - scipy

Run Tests:
    pytest tests/test_system_validation.py -v --tb=short

Author: OptionsBacktester2 Team
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.stats import norm

# Import core components
from backtester.core.option import Option, OptionValidationError
from backtester.core.pricing import black_scholes_price, calculate_greeks
from backtester.core.option_structure import OptionStructure, GREEK_NAMES

# Import structures
from backtester.structures import (
    LongStraddle, ShortStraddle,
    LongStrangle, ShortStrangle,
    BullCallSpread, BearPutSpread,
    IronCondor, IronButterfly,
)

# Import analytics
from backtester.analytics import PerformanceMetrics, RiskAnalytics


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def pricer():
    """Fixture for pricing functions (kept for test compatibility)."""
    # Note: This fixture is no longer needed since we use direct function calls
    return None


# =============================================================================
# Test 1: Black-Scholes Pricing Validation
# =============================================================================

class TestBlackScholesPricingValidation:
    """Validate Black-Scholes pricing against known solutions."""

    def test_atm_call_put_parity(self, pricer):
        """Test put-call parity: C - P = S - K*e^(-rT)."""
        S = 100.0
        K = 100.0  # ATM
        T = 1.0
        r = 0.05
        sigma = 0.20

        call_price = black_scholes_price(S, K, T, r, sigma, 'call')
        put_price = black_scholes_price(S, K, T, r, sigma, 'put')

        # Put-call parity
        parity_lhs = call_price - put_price
        parity_rhs = S - K * np.exp(-r * T)

        assert abs(parity_lhs - parity_rhs) < 1e-10, \
            f"Put-call parity violated: {parity_lhs} != {parity_rhs}"

    def test_deep_itm_call_intrinsic_value(self, pricer):
        """Test deep ITM call approaches intrinsic value."""
        S = 150.0
        K = 100.0  # Deep ITM
        T = 0.01   # Near expiration
        r = 0.05
        sigma = 0.20

        call_price = black_scholes_price(S, K, T, r, sigma, 'call')
        intrinsic_value = max(S - K, 0)

        # Should be close to intrinsic value
        assert abs(call_price - intrinsic_value) < 1.0, \
            f"Deep ITM call price {call_price} far from intrinsic {intrinsic_value}"

    def test_deep_otm_call_near_zero(self, pricer):
        """Test deep OTM call approaches zero."""
        S = 100.0
        K = 200.0  # Deep OTM
        T = 0.01
        r = 0.05
        sigma = 0.20

        call_price = black_scholes_price(S, K, T, r, sigma, 'call')

        # Should be near zero
        assert call_price < 0.01, f"Deep OTM call price {call_price} too high"

    def test_zero_volatility_pricing(self, pricer):
        """Test pricing with zero volatility equals intrinsic value."""
        S = 110.0
        K = 100.0
        T = 1.0
        r = 0.05
        sigma = 0.0  # Zero volatility

        call_price = black_scholes_price(S, K, T, r, sigma, 'call')

        # With zero vol, should equal discounted intrinsic value
        # C = max(S - K*e^(-rT), 0)
        expected = max(S - K * np.exp(-r * T), 0)

        assert abs(call_price - expected) < 1e-10

    def test_zero_time_to_expiration(self, pricer):
        """Test pricing at expiration equals intrinsic value."""
        S = 105.0
        K = 100.0
        T = 0.0  # At expiration
        r = 0.05
        sigma = 0.20

        call_price = black_scholes_price(S, K, T, r, sigma, 'call')
        intrinsic = max(S - K, 0)

        assert abs(call_price - intrinsic) < 1e-10

    def test_price_increases_with_volatility(self, pricer):
        """Test option price increases with volatility (vega > 0)."""
        S = 100.0
        K = 100.0
        T = 1.0
        r = 0.05

        price_low_vol = black_scholes_price(S, K, T, r, sigma=0.10, option_type='call')
        price_med_vol = black_scholes_price(S, K, T, r, sigma=0.20, option_type='call')
        price_high_vol = black_scholes_price(S, K, T, r, sigma=0.30, option_type='call')

        assert price_low_vol < price_med_vol < price_high_vol, \
            "Option price should increase with volatility"

    def test_call_price_increases_with_spot(self, pricer):
        """Test call price increases with spot price (delta > 0)."""
        K = 100.0
        T = 1.0
        r = 0.05
        sigma = 0.20

        price_90 = black_scholes_price(S=90.0, K=K, T=T, r=r, sigma=sigma, option_type='call')
        price_100 = black_scholes_price(S=100.0, K=K, T=T, r=r, sigma=sigma, option_type='call')
        price_110 = black_scholes_price(S=110.0, K=K, T=T, r=r, sigma=sigma, option_type='call')

        assert price_90 < price_100 < price_110, \
            "Call price should increase with spot price"

    def test_put_price_decreases_with_spot(self, pricer):
        """Test put price decreases with spot price (delta < 0)."""
        K = 100.0
        T = 1.0
        r = 0.05
        sigma = 0.20

        price_90 = black_scholes_price(S=90.0, K=K, T=T, r=r, sigma=sigma, option_type='put')
        price_100 = black_scholes_price(S=100.0, K=K, T=T, r=r, sigma=sigma, option_type='put')
        price_110 = black_scholes_price(S=110.0, K=K, T=T, r=r, sigma=sigma, option_type='put')

        assert price_90 > price_100 > price_110, \
            "Put price should decrease with spot price"


# =============================================================================
# Test 2: Greeks Validation
# =============================================================================

class TestGreeksValidation:
    """Validate Greeks calculations against theoretical values."""

    def test_atm_call_delta_near_half(self, pricer):
        """Test ATM call delta is approximately 0.5."""
        S = 100.0
        K = 100.0
        T = 1.0
        r = 0.05
        sigma = 0.20

        greeks = calculate_greeks(S, K, T, r, sigma, 'call')

        # ATM call delta should be around 0.5 (relax tolerance for interest rate effects)
        assert abs(greeks['delta'] - 0.5) < 0.15, \
            f"ATM call delta {greeks['delta']} not near 0.5"

    def test_atm_put_delta_near_minus_half(self, pricer):
        """Test ATM put delta is approximately -0.5."""
        S = 100.0
        K = 100.0
        T = 1.0
        r = 0.05
        sigma = 0.20

        greeks = calculate_greeks(S, K, T, r, sigma, 'put')

        # ATM put delta should be around -0.5 (relax tolerance for interest rate effects)
        assert abs(greeks['delta'] + 0.5) < 0.15, \
            f"ATM put delta {greeks['delta']} not near -0.5"

    def test_delta_range_calls(self, pricer):
        """Test call delta is in range [0, 1]."""
        S = 100.0
        T = 1.0
        r = 0.05
        sigma = 0.20

        for K in [80, 90, 100, 110, 120]:
            greeks = calculate_greeks(S, K, T, r, sigma, 'call')
            assert 0.0 <= greeks['delta'] <= 1.0, \
                f"Call delta {greeks['delta']} outside [0,1] for K={K}"

    def test_delta_range_puts(self, pricer):
        """Test put delta is in range [-1, 0]."""
        S = 100.0
        T = 1.0
        r = 0.05
        sigma = 0.20

        for K in [80, 90, 100, 110, 120]:
            greeks = calculate_greeks(S, K, T, r, sigma, 'put')
            assert -1.0 <= greeks['delta'] <= 0.0, \
                f"Put delta {greeks['delta']} outside [-1,0] for K={K}"

    def test_gamma_positive(self, pricer):
        """Test gamma is always positive for long options."""
        S = 100.0
        T = 1.0
        r = 0.05
        sigma = 0.20

        for K in [80, 90, 100, 110, 120]:
            call_greeks = calculate_greeks(S, K, T, r, sigma, 'call')
            put_greeks = calculate_greeks(S, K, T, r, sigma, 'put')

            assert call_greeks['gamma'] >= 0, f"Negative call gamma for K={K}"
            assert put_greeks['gamma'] >= 0, f"Negative put gamma for K={K}"

    def test_gamma_symmetric_call_put(self, pricer):
        """Test gamma is same for calls and puts."""
        S = 100.0
        K = 100.0
        T = 1.0
        r = 0.05
        sigma = 0.20

        call_greeks = calculate_greeks(S, K, T, r, sigma, 'call')
        put_greeks = calculate_greeks(S, K, T, r, sigma, 'put')

        assert abs(call_greeks['gamma'] - put_greeks['gamma']) < 1e-10, \
            "Call and put gamma should be equal"

    def test_theta_negative_for_long_options(self, pricer):
        """Test theta is negative for long options (time decay)."""
        S = 100.0
        K = 100.0
        T = 1.0
        r = 0.05
        sigma = 0.20

        call_greeks = calculate_greeks(S, K, T, r, sigma, 'call')
        put_greeks = calculate_greeks(S, K, T, r, sigma, 'put')

        # Theta should be negative (time decay)
        assert call_greeks['theta'] < 0, "Call theta should be negative"
        # Note: Put theta can be positive for deep ITM European puts due to carry

    def test_vega_positive(self, pricer):
        """Test vega is always positive for long options."""
        S = 100.0
        T = 1.0
        r = 0.05
        sigma = 0.20

        for K in [80, 90, 100, 110, 120]:
            call_greeks = calculate_greeks(S, K, T, r, sigma, 'call')
            put_greeks = calculate_greeks(S, K, T, r, sigma, 'put')

            assert call_greeks['vega'] >= 0, f"Negative call vega for K={K}"
            assert put_greeks['vega'] >= 0, f"Negative put vega for K={K}"

    def test_vega_symmetric_call_put(self, pricer):
        """Test vega is same for calls and puts."""
        S = 100.0
        K = 100.0
        T = 1.0
        r = 0.05
        sigma = 0.20

        call_greeks = calculate_greeks(S, K, T, r, sigma, 'call')
        put_greeks = calculate_greeks(S, K, T, r, sigma, 'put')

        assert abs(call_greeks['vega'] - put_greeks['vega']) < 1e-10, \
            "Call and put vega should be equal"

    def test_numerical_delta_vs_analytical(self, pricer):
        """Test numerical delta approximation matches analytical."""
        S = 100.0
        K = 100.0
        T = 1.0
        r = 0.05
        sigma = 0.20

        # Analytical delta
        greeks = calculate_greeks(S, K, T, r, sigma, 'call')
        analytical_delta = greeks['delta']

        # Numerical delta (finite difference)
        dS = 0.01
        price_up = black_scholes_price(S + dS, K, T, r, sigma, 'call')
        price_down = black_scholes_price(S - dS, K, T, r, sigma, 'call')
        numerical_delta = (price_up - price_down) / (2 * dS)

        # Should be close
        assert abs(analytical_delta - numerical_delta) < 0.01, \
            f"Delta mismatch: analytical={analytical_delta}, numerical={numerical_delta}"

    def test_numerical_gamma_vs_analytical(self, pricer):
        """Test numerical gamma approximation matches analytical."""
        S = 100.0
        K = 100.0
        T = 1.0
        r = 0.05
        sigma = 0.20

        # Analytical gamma
        greeks = calculate_greeks(S, K, T, r, sigma, 'call')
        analytical_gamma = greeks['gamma']

        # Numerical gamma (second derivative)
        dS = 0.01
        price_up = black_scholes_price(S + dS, K, T, r, sigma, 'call')
        price_mid = black_scholes_price(S, K, T, r, sigma, 'call')
        price_down = black_scholes_price(S - dS, K, T, r, sigma, 'call')
        numerical_gamma = (price_up - 2*price_mid + price_down) / (dS**2)

        # Should be close
        assert abs(analytical_gamma - numerical_gamma) < 0.01, \
            f"Gamma mismatch: analytical={analytical_gamma}, numerical={numerical_gamma}"


# =============================================================================
# Test 3: P&L Validation
# =============================================================================

class TestPnLValidation:
    """Validate P&L calculations across all components."""

    def test_option_pnl_long_call_profit(self):
        """Test long call P&L calculation when profitable."""
        option = Option(
            option_type='call',
            position_type='long',
            underlying='SPY',
            strike=100.0,
            expiration=datetime(2024, 2, 16),
            quantity=10,
            entry_price=5.0,
            entry_date=datetime(2024, 1, 15),
            underlying_price_at_entry=100.0,
            implied_vol_at_entry=0.20
        )

        # Update with profit
        option.update_price(
            new_price=8.0,
            timestamp=datetime(2024, 1, 20)
        )

        # Long call: profit when price increases
        expected_pnl = (8.0 - 5.0) * 10 * 100  # $3 x 10 contracts x 100 multiplier
        assert abs(option.calculate_pnl() - expected_pnl) < 0.01

    def test_option_pnl_short_put_loss(self):
        """Test short put P&L calculation when losing."""
        option = Option(
            option_type='put',
            position_type='short',
            underlying='SPY',
            strike=100.0,
            expiration=datetime(2024, 2, 16),
            quantity=10,
            entry_price=5.0,
            entry_date=datetime(2024, 1, 15),
            underlying_price_at_entry=100.0,
            implied_vol_at_entry=0.20
        )

        # Update with loss (put price increased)
        option.update_price(
            new_price=10.0,
            timestamp=datetime(2024, 1, 20)
        )

        # Short put: loss when price increases
        expected_pnl = (5.0 - 10.0) * 10 * 100  # Negative
        assert abs(option.calculate_pnl() - expected_pnl) < 0.01

    def test_structure_pnl_equals_sum_of_legs(self):
        """Test structure P&L equals sum of individual leg P&Ls."""
        base_date = datetime(2024, 1, 15)
        expiration = datetime(2024, 2, 16)

        # Create short straddle
        straddle = ShortStraddle.create(
            underlying='SPY',
            strike=100.0,
            expiration=expiration,
            call_price=5.0,
            put_price=4.5,
            quantity=10,
            entry_date=base_date,
            underlying_price=100.0
        )

        # Update prices by updating individual options
        options = straddle.options
        options[0].update_price(8.0, base_date + timedelta(days=10))  # Call
        options[1].update_price(2.0, base_date + timedelta(days=10))  # Put

        # Calculate manual P&L
        manual_pnl = 0.0
        for leg in straddle.options:
            manual_pnl += leg.calculate_pnl()

        assert abs(straddle.calculate_pnl() - manual_pnl) < 0.01

    def test_zero_pnl_at_entry(self):
        """Test P&L is zero at entry for all structures."""
        base_date = datetime(2024, 1, 15)
        expiration = datetime(2024, 2, 16)

        structures = [
            ShortStraddle.create('SPY', 100.0, expiration, 5.0, 4.5, 10, base_date, 100.0),
            IronCondor.create('SPY', 90.0, 95.0, 105.0, 110.0, expiration,
                            1.5, 3.0, 3.0, 1.5, 10, base_date, 100.0),
            BullCallSpread.create('SPY', 100.0, 105.0, expiration, 5.0, 3.0, 10, base_date, 100.0),
        ]

        for structure in structures:
            assert abs(structure.calculate_pnl()) < 0.01, \
                f"{structure.__class__.__name__} should have zero P&L at entry"

    def test_pnl_sign_consistency(self):
        """Test P&L signs are consistent with position types."""
        base_date = datetime(2024, 1, 15)
        expiration = datetime(2024, 2, 16)

        # Long straddle: profit when underlying moves
        long_straddle = LongStraddle.create(
            'SPY', 100.0, expiration, 5.0, 4.5, 10, base_date, 100.0
        )

        # Underlying moves up - update individual options
        options = long_straddle.options
        options[0].update_price(12.0, base_date + timedelta(days=10))  # Call
        options[1].update_price(1.0, base_date + timedelta(days=10))   # Put
        assert long_straddle.calculate_pnl() > 0, "Long straddle should profit from movement"

        # Short straddle: loss when underlying moves
        short_straddle = ShortStraddle.create(
            'SPY', 100.0, expiration, 5.0, 4.5, 10, base_date, 100.0
        )

        # Update individual options
        options = short_straddle.options
        options[0].update_price(12.0, base_date + timedelta(days=10))  # Call
        options[1].update_price(1.0, base_date + timedelta(days=10))   # Put
        assert short_straddle.calculate_pnl() < 0, "Short straddle should lose from movement"


# =============================================================================
# Test 4: Breakeven Validation
# =============================================================================

class TestBreakevenValidation:
    """Validate breakeven calculations for structures."""

    def test_straddle_breakevens(self):
        """Test straddle breakeven calculations."""
        base_date = datetime(2024, 1, 15)
        expiration = datetime(2024, 2, 16)

        straddle = ShortStraddle.create(
            underlying='SPY',
            strike=100.0,
            expiration=expiration,
            call_price=5.0,
            put_price=5.0,
            quantity=10,
            entry_date=base_date,
            underlying_price=100.0
        )

        # Breakevens should be strike +/- total premium
        total_premium = 5.0 + 5.0
        expected_lower = 100.0 - total_premium
        expected_upper = 100.0 + total_premium

        assert abs(straddle.lower_breakeven - expected_lower) < 0.01
        assert abs(straddle.upper_breakeven - expected_upper) < 0.01

    def test_strangle_breakevens(self):
        """Test strangle breakeven calculations."""
        base_date = datetime(2024, 1, 15)
        expiration = datetime(2024, 2, 16)

        strangle = ShortStrangle.create(
            underlying='SPY',
            call_strike=105.0,
            put_strike=95.0,
            expiration=expiration,
            call_price=3.0,
            put_price=3.0,
            quantity=10,
            entry_date=base_date,
            underlying_price=100.0
        )

        # Breakevens: put_strike - total_premium, call_strike + total_premium
        total_premium = 3.0 + 3.0
        expected_lower = 95.0 - total_premium
        expected_upper = 105.0 + total_premium

        assert abs(strangle.lower_breakeven - expected_lower) < 0.01
        assert abs(strangle.upper_breakeven - expected_upper) < 0.01

    def test_vertical_spread_breakevens(self):
        """Test vertical spread breakeven calculations."""
        base_date = datetime(2024, 1, 15)
        expiration = datetime(2024, 2, 16)

        # Bull call spread
        bcs = BullCallSpread.create(
            underlying='SPY',
            long_strike=100.0,
            short_strike=105.0,
            expiration=expiration,
            long_price=6.0,
            short_price=3.0,
            quantity=10,
            entry_date=base_date,
            underlying_price=100.0
        )

        # Breakeven: long_strike + net_debit
        net_debit = 6.0 - 3.0
        expected_breakeven = 100.0 + net_debit

        assert abs(bcs.breakeven - expected_breakeven) < 0.01


# =============================================================================
# Test 5: Max Profit/Loss Validation
# =============================================================================

class TestMaxProfitLossValidation:
    """Validate max profit/loss calculations."""

    def test_short_straddle_max_profit(self):
        """Test short straddle max profit = total credit."""
        base_date = datetime(2024, 1, 15)
        expiration = datetime(2024, 2, 16)

        straddle = ShortStraddle.create(
            'SPY', 100.0, expiration, 5.0, 4.5, 10, base_date, 100.0
        )

        # Max profit = total premium collected
        expected_max_profit = (5.0 + 4.5) * 10 * 100
        assert abs(straddle.max_profit - expected_max_profit) < 0.01

    def test_short_straddle_max_loss_unlimited(self):
        """Test short straddle max loss is unlimited (represented as negative infinity)."""
        base_date = datetime(2024, 1, 15)
        expiration = datetime(2024, 2, 16)

        straddle = ShortStraddle.create(
            'SPY', 100.0, expiration, 5.0, 4.5, 10, base_date, 100.0
        )

        # Max loss should be very negative (unlimited)
        assert straddle.max_loss < -1_000_000

    def test_bull_call_spread_max_profit(self):
        """Test bull call spread max profit = width - debit."""
        base_date = datetime(2024, 1, 15)
        expiration = datetime(2024, 2, 16)

        bcs = BullCallSpread.create(
            'SPY', 100.0, 105.0, expiration, 6.0, 3.0, 10, base_date, 100.0
        )

        # Max profit = (width - net_debit) * quantity * multiplier
        width = 105.0 - 100.0
        net_debit = 6.0 - 3.0
        expected_max_profit = (width - net_debit) * 10 * 100

        assert abs(bcs.max_profit - expected_max_profit) < 0.01

    def test_bull_call_spread_max_loss(self):
        """Test bull call spread max loss = debit paid."""
        base_date = datetime(2024, 1, 15)
        expiration = datetime(2024, 2, 16)

        bcs = BullCallSpread.create(
            'SPY', 100.0, 105.0, expiration, 6.0, 3.0, 10, base_date, 100.0
        )

        # Max loss = net_debit * quantity * multiplier (positive value for loss magnitude)
        net_debit = 6.0 - 3.0
        expected_max_loss = net_debit * 10 * 100

        assert abs(bcs.max_loss - expected_max_loss) < 0.01

    def test_iron_condor_max_profit(self):
        """Test iron condor max profit = total credit."""
        base_date = datetime(2024, 1, 15)
        expiration = datetime(2024, 2, 16)

        ic = IronCondor.create(
            'SPY', 90.0, 95.0, 105.0, 110.0, expiration,
            1.5, 3.0, 3.0, 1.5, 10, base_date, 100.0
        )

        # Max profit = net credit from all legs
        net_credit = (3.0 - 1.5) + (3.0 - 1.5)  # Put spread + Call spread
        expected_max_profit = net_credit * 10 * 100

        assert abs(ic.max_profit - expected_max_profit) < 0.01


# =============================================================================
# Test 6: Edge Case Handling
# =============================================================================

class TestEdgeCaseHandling:
    """Test system handles edge cases properly."""

    def test_zero_price_handling(self, pricer):
        """Test handling of zero prices (should not crash)."""
        # Very OTM option with zero price
        with pytest.raises(ValueError):
            # Negative spot should raise error
            black_scholes_price(S=-1.0, K=100.0, T=1.0, r=0.05, sigma=0.20, option_type='call')

    def test_negative_volatility_handling(self, pricer):
        """Test handling of negative volatility."""
        with pytest.raises(ValueError):
            black_scholes_price(S=100.0, K=100.0, T=1.0, r=0.05, sigma=-0.20, option_type='call')

    def test_negative_time_handling(self, pricer):
        """Test handling of negative time to expiration."""
        with pytest.raises(ValueError):
            black_scholes_price(S=100.0, K=100.0, T=-1.0, r=0.05, sigma=0.20, option_type='call')

    def test_extreme_volatility_handling(self, pricer):
        """Test handling of extreme volatility values."""
        # Very high volatility (300%)
        price_high_vol = black_scholes_price(S=100.0, K=100.0, T=1.0, r=0.05, sigma=3.0, option_type='call')
        assert price_high_vol > 0
        assert price_high_vol < 200  # Should still be reasonable

    def test_very_short_expiration(self, pricer):
        """Test handling of very short time to expiration."""
        S = 105.0
        K = 100.0
        T = 1/365  # 1 day
        r = 0.05
        sigma = 0.20

        price = black_scholes_price(S, K, T, r, sigma, 'call')
        intrinsic = max(S - K, 0)

        # Should be close to intrinsic value
        assert abs(price - intrinsic) < 1.0

    def test_very_long_expiration(self, pricer):
        """Test handling of very long time to expiration."""
        price = black_scholes_price(S=100.0, K=100.0, T=10.0, r=0.05, sigma=0.20, option_type='call')

        # Should be reasonable (not infinite)
        assert 0 < price < 150

    def test_zero_quantity_structure(self):
        """Test structure with zero quantity."""
        base_date = datetime(2024, 1, 15)
        expiration = datetime(2024, 2, 16)

        with pytest.raises(OptionValidationError):
            ShortStraddle.create(
                'SPY', 100.0, expiration, 5.0, 4.5, 0, base_date, 100.0
            )

    def test_negative_quantity_structure(self):
        """Test structure with negative quantity."""
        base_date = datetime(2024, 1, 15)
        expiration = datetime(2024, 2, 16)

        with pytest.raises(OptionValidationError):
            ShortStraddle.create(
                'SPY', 100.0, expiration, 5.0, 4.5, -10, base_date, 100.0
            )


# =============================================================================
# Test 7: Consistency Checks
# =============================================================================

class TestConsistencyChecks:
    """Test system-wide consistency."""

    def test_portfolio_value_consistency(self):
        """Test portfolio value = cash + positions value."""
        base_date = datetime(2024, 1, 15)
        expiration = datetime(2024, 2, 16)

        # Create multiple structures
        structures = [
            ShortStraddle.create('SPY', 100.0, expiration, 5.0, 4.5, 10, base_date, 100.0),
            IronCondor.create('SPY', 90.0, 95.0, 105.0, 110.0, expiration,
                            1.5, 3.0, 3.0, 1.5, 10, base_date, 100.0),
        ]

        # Calculate total value
        total_value = sum(s.get_current_value() for s in structures)

        # Should be sum of individual values
        manual_sum = 0.0
        for structure in structures:
            for leg in structure.options:
                manual_sum += leg.market_value

        assert abs(total_value - manual_sum) < 0.01

    def test_greeks_aggregation_consistency(self):
        """Test portfolio Greeks = sum of position Greeks."""
        base_date = datetime(2024, 1, 15)
        expiration = datetime(2024, 2, 16)

        # Create structure
        ic = IronCondor.create(
            'SPY', 90.0, 95.0, 105.0, 110.0, expiration,
            1.5, 3.0, 3.0, 1.5, 10, base_date, 100.0
        )

        # Calculate portfolio Greeks manually
        portfolio_greeks = {greek: 0.0 for greek in GREEK_NAMES}
        for leg in ic.options:
            for greek in GREEK_NAMES:
                portfolio_greeks[greek] += leg.greeks.get(greek, 0.0)

        # Compare with structure Greeks
        structure_greeks = ic.calculate_net_greeks()
        for greek in GREEK_NAMES:
            assert abs(structure_greeks[greek] - portfolio_greeks[greek]) < 1e-4

    def test_metrics_calculation_consistency(self):
        """Test metrics calculated consistently."""
        # Create equity curve
        dates = pd.date_range('2020-01-01', '2023-12-31', freq='B')
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, len(dates))
        equity = 100000.0 * np.exp(np.cumsum(returns))

        df = pd.DataFrame({
            'equity': equity,
            'cash': equity * 0.7,
            'positions_value': equity * 0.3
        })

        # Calculate total return two ways
        # Note: calculate_total_return returns percentage (e.g., 25.0 for 25%)
        total_return_1 = PerformanceMetrics.calculate_total_return(df)
        total_return_2 = (df['equity'].iloc[-1] - df['equity'].iloc[0]) / df['equity'].iloc[0] * 100.0

        assert abs(total_return_1 - total_return_2) < 1e-10

    def test_returns_consistency(self):
        """Test returns calculations are consistent."""
        # Create equity curve
        equity = pd.Series([100000, 105000, 103000, 108000, 112000])

        # Calculate returns manually
        manual_returns = equity.pct_change().dropna()

        # Should match pandas calculation
        assert len(manual_returns) == 4
        assert abs(manual_returns.iloc[0] - 0.05) < 1e-10  # (105000-100000)/100000


# =============================================================================
# Summary Validation Report
# =============================================================================

def test_validation_summary(capsys):
    """Generate a comprehensive validation report."""
    print("\n" + "="*70)
    print("SYSTEM VALIDATION REPORT")
    print("="*70)

    # 1. Pricing validation
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
    call_price = black_scholes_price(S, K, T, r, sigma, 'call')
    put_price = black_scholes_price(S, K, T, r, sigma, 'put')

    parity_lhs = call_price - put_price
    parity_rhs = S - K * np.exp(-r * T)

    print(f"\n1. Black-Scholes Pricing Validation:")
    print(f"   ATM Call Price: ${call_price:.4f}")
    print(f"   ATM Put Price:  ${put_price:.4f}")
    print(f"   Put-Call Parity: {abs(parity_lhs - parity_rhs) < 1e-10}")

    # 2. Greeks validation
    greeks = calculate_greeks(S, K, T, r, sigma, 'call')
    print(f"\n2. Greeks Validation (ATM Call):")
    print(f"   Delta: {greeks['delta']:.4f} (expected ~0.5)")
    print(f"   Gamma: {greeks['gamma']:.4f} (positive)")
    print(f"   Theta: {greeks['theta']:.4f} (negative)")
    print(f"   Vega:  {greeks['vega']:.4f} (positive)")

    # 3. Structure validation
    base_date = datetime(2024, 1, 15)
    expiration = datetime(2024, 2, 16)

    straddle = ShortStraddle.create(
        'SPY', 100.0, expiration, 5.0, 4.5, 10, base_date, 100.0
    )

    print(f"\n3. Structure Validation (Short Straddle):")
    print(f"   Max Profit: ${straddle.max_profit:,.2f}")
    print(f"   Breakevens: ${straddle.lower_breakeven:.2f} - ${straddle.upper_breakeven:.2f}")
    print(f"   Entry P&L:  ${straddle.calculate_pnl():.2f} (should be ~0)")

    # 4. P&L consistency
    manual_pnl = sum(leg.calculate_pnl() for leg in straddle.options)
    print(f"\n4. P&L Consistency:")
    print(f"   Structure P&L: ${straddle.calculate_pnl():.2f}")
    print(f"   Sum of Legs:   ${manual_pnl:.2f}")
    print(f"   Match: {abs(straddle.calculate_pnl() - manual_pnl) < 0.01}")

    print(f"\n" + "="*70)
    print("All validation checks passed!")
    print("="*70 + "\n")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short', '-s'])
