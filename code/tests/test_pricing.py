"""
Unit Tests for Options Pricing Module

This module contains comprehensive tests for the Black-Scholes pricing
functions and Greeks calculations, ensuring financial accuracy and
numerical stability.

Test Categories:
    1. Black-Scholes Pricing Tests
        - Known analytical solutions
        - Put-call parity verification
        - Edge cases (T=0, sigma=0, etc.)

    2. Greeks Tests
        - Individual Greeks against known values
        - Greeks sum properties (e.g., gamma same for calls/puts)
        - Edge case handling

    3. Implied Volatility Tests
        - Round-trip verification (price -> IV -> price)
        - Different methods (Newton, Brent)
        - Edge cases and failure modes

    4. Numerical Stability Tests
        - Extreme parameter values
        - Deep ITM/OTM options
        - Near-expiration options

References:
    - Hull, J.C. (2018). Options, Futures, and Other Derivatives
    - Wilmott, P. (2006). Paul Wilmott on Quantitative Finance
"""

import pytest
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backtester.core.pricing import (
    black_scholes_call,
    black_scholes_put,
    black_scholes_price,
    calculate_delta,
    calculate_gamma,
    calculate_theta,
    calculate_vega,
    calculate_rho,
    calculate_greeks,
    calculate_implied_volatility,
    black_scholes_call_vectorized,
    black_scholes_put_vectorized,
    PricingError,
    ImpliedVolatilityError,
    DAYS_PER_YEAR,
)


# =============================================================================
# Test Fixtures and Constants
# =============================================================================

# Standard test parameters
SPOT = 100.0
STRIKE = 100.0
TIME = 0.25  # 3 months
RATE = 0.05
SIGMA = 0.20

# Tolerance for price comparisons
PRICE_TOL = 1e-4  # $0.0001
GREEK_TOL = 1e-4
IV_TOL = 1e-6


@pytest.fixture
def standard_params():
    """Standard option parameters for testing."""
    return {
        'S': SPOT,
        'K': STRIKE,
        'T': TIME,
        'r': RATE,
        'sigma': SIGMA
    }


# =============================================================================
# Black-Scholes Pricing Tests
# =============================================================================

class TestBlackScholesCall:
    """Tests for black_scholes_call function."""

    def test_atm_call_price(self, standard_params):
        """Test ATM call price against known value."""
        # For ATM option: S=K=100, T=0.25, r=5%, sigma=20%
        # Expected price approximately $5.88
        price = black_scholes_call(**standard_params)

        # Manual calculation for verification
        d1 = (np.log(SPOT/STRIKE) + (RATE + 0.5*SIGMA**2)*TIME) / (SIGMA*np.sqrt(TIME))
        d2 = d1 - SIGMA*np.sqrt(TIME)
        expected = SPOT*norm.cdf(d1) - STRIKE*np.exp(-RATE*TIME)*norm.cdf(d2)

        assert abs(price - expected) < PRICE_TOL

    def test_call_itm(self):
        """Test ITM call price."""
        # Deep ITM call should be close to intrinsic + time value
        price = black_scholes_call(S=120, K=100, T=0.25, r=0.05, sigma=0.20)
        intrinsic = 120 - 100  # $20

        # ITM call should be >= intrinsic value
        assert price >= intrinsic - PRICE_TOL

    def test_call_otm(self):
        """Test OTM call price."""
        # Deep OTM call should have small but positive price
        price = black_scholes_call(S=80, K=100, T=0.25, r=0.05, sigma=0.20)

        assert price > 0
        assert price < 5  # Should be relatively small

    def test_call_at_expiry_itm(self):
        """Test call at expiration when ITM."""
        price = black_scholes_call(S=110, K=100, T=0, r=0.05, sigma=0.20)
        assert abs(price - 10.0) < PRICE_TOL  # Intrinsic value

    def test_call_at_expiry_otm(self):
        """Test call at expiration when OTM."""
        price = black_scholes_call(S=90, K=100, T=0, r=0.05, sigma=0.20)
        assert abs(price) < PRICE_TOL  # Worthless

    def test_call_zero_volatility_itm(self):
        """Test call with zero volatility when ITM."""
        # With sigma=0, call = max(S - K*e^(-rT), 0)
        price = black_scholes_call(S=110, K=100, T=0.25, r=0.05, sigma=0)
        expected = 110 - 100 * np.exp(-0.05 * 0.25)
        assert abs(price - expected) < PRICE_TOL

    def test_call_zero_volatility_otm(self):
        """Test call with zero volatility when OTM."""
        price = black_scholes_call(S=90, K=100, T=0.25, r=0.05, sigma=0)
        assert abs(price) < PRICE_TOL

    def test_call_zero_spot(self):
        """Test call when spot is zero."""
        price = black_scholes_call(S=0, K=100, T=0.25, r=0.05, sigma=0.20)
        assert abs(price) < PRICE_TOL

    def test_call_zero_strike(self):
        """Test call when strike is zero (option equals spot)."""
        price = black_scholes_call(S=100, K=0, T=0.25, r=0.05, sigma=0.20)
        assert abs(price - 100) < PRICE_TOL

    def test_call_non_negative(self, standard_params):
        """Test that call price is always non-negative."""
        price = black_scholes_call(**standard_params)
        assert price >= 0


class TestBlackScholesPut:
    """Tests for black_scholes_put function."""

    def test_atm_put_price(self, standard_params):
        """Test ATM put price against known value."""
        price = black_scholes_put(**standard_params)

        # Manual calculation
        d1 = (np.log(SPOT/STRIKE) + (RATE + 0.5*SIGMA**2)*TIME) / (SIGMA*np.sqrt(TIME))
        d2 = d1 - SIGMA*np.sqrt(TIME)
        expected = STRIKE*np.exp(-RATE*TIME)*norm.cdf(-d2) - SPOT*norm.cdf(-d1)

        assert abs(price - expected) < PRICE_TOL

    def test_put_itm(self):
        """Test ITM put price."""
        # Deep ITM put
        price = black_scholes_put(S=80, K=100, T=0.25, r=0.05, sigma=0.20)

        # For European puts with positive interest rates, price can be
        # less than intrinsic because the strike is received at expiration.
        # The put should be >= discounted intrinsic: max(K*e^(-rT) - S, 0)
        discounted_intrinsic = max(100 * np.exp(-0.05 * 0.25) - 80, 0)
        assert price >= discounted_intrinsic - PRICE_TOL

        # Should still be a substantial value for deep ITM
        assert price > 15.0

    def test_put_otm(self):
        """Test OTM put price."""
        price = black_scholes_put(S=120, K=100, T=0.25, r=0.05, sigma=0.20)

        assert price > 0
        assert price < 5

    def test_put_at_expiry_itm(self):
        """Test put at expiration when ITM."""
        price = black_scholes_put(S=90, K=100, T=0, r=0.05, sigma=0.20)
        assert abs(price - 10.0) < PRICE_TOL

    def test_put_at_expiry_otm(self):
        """Test put at expiration when OTM."""
        price = black_scholes_put(S=110, K=100, T=0, r=0.05, sigma=0.20)
        assert abs(price) < PRICE_TOL

    def test_put_zero_spot(self):
        """Test put when spot is zero."""
        price = black_scholes_put(S=0, K=100, T=0.25, r=0.05, sigma=0.20)
        expected = 100 * np.exp(-0.05 * 0.25)
        assert abs(price - expected) < PRICE_TOL


class TestPutCallParity:
    """Tests for put-call parity relationship."""

    def test_put_call_parity_atm(self, standard_params):
        """Test put-call parity: C - P = S - K*e^(-rT)."""
        call = black_scholes_call(**standard_params)
        put = black_scholes_put(**standard_params)

        lhs = call - put
        rhs = SPOT - STRIKE * np.exp(-RATE * TIME)

        assert abs(lhs - rhs) < PRICE_TOL

    def test_put_call_parity_itm_call(self):
        """Test put-call parity with ITM call (OTM put)."""
        S, K, T, r, sigma = 120, 100, 0.5, 0.03, 0.25

        call = black_scholes_call(S, K, T, r, sigma)
        put = black_scholes_put(S, K, T, r, sigma)

        lhs = call - put
        rhs = S - K * np.exp(-r * T)

        assert abs(lhs - rhs) < PRICE_TOL

    def test_put_call_parity_otm_call(self):
        """Test put-call parity with OTM call (ITM put)."""
        S, K, T, r, sigma = 80, 100, 0.5, 0.03, 0.25

        call = black_scholes_call(S, K, T, r, sigma)
        put = black_scholes_put(S, K, T, r, sigma)

        lhs = call - put
        rhs = S - K * np.exp(-r * T)

        assert abs(lhs - rhs) < PRICE_TOL

    @pytest.mark.parametrize("spot,strike,time,rate,sigma", [
        (100, 100, 0.25, 0.05, 0.20),
        (50, 60, 0.5, 0.02, 0.30),
        (200, 180, 1.0, 0.04, 0.15),
        (100, 100, 0.01, 0.05, 0.40),  # Near expiry
        (100, 100, 2.0, 0.01, 0.10),   # Long dated
    ])
    def test_put_call_parity_various_params(self, spot, strike, time, rate, sigma):
        """Test put-call parity across various parameter combinations."""
        call = black_scholes_call(spot, strike, time, rate, sigma)
        put = black_scholes_put(spot, strike, time, rate, sigma)

        lhs = call - put
        rhs = spot - strike * np.exp(-rate * time)

        assert abs(lhs - rhs) < PRICE_TOL


# =============================================================================
# Greeks Tests
# =============================================================================

class TestDelta:
    """Tests for delta calculation."""

    def test_call_delta_bounds(self, standard_params):
        """Test that call delta is between 0 and 1."""
        delta = calculate_delta(**standard_params, option_type='call')
        assert 0 <= delta <= 1

    def test_put_delta_bounds(self, standard_params):
        """Test that put delta is between -1 and 0."""
        delta = calculate_delta(**standard_params, option_type='put')
        assert -1 <= delta <= 0

    def test_call_put_delta_relationship(self, standard_params):
        """Test that call_delta - put_delta = 1."""
        call_delta = calculate_delta(**standard_params, option_type='call')
        put_delta = calculate_delta(**standard_params, option_type='put')

        assert abs(call_delta - put_delta - 1.0) < GREEK_TOL

    def test_atm_call_delta_approximately_half(self):
        """Test that ATM call delta is approximately 0.5."""
        # ATM with r=0 should be close to 0.5
        delta = calculate_delta(S=100, K=100, T=0.25, r=0.0, sigma=0.20, option_type='call')
        assert abs(delta - 0.5) < 0.05  # Within 5%

    def test_deep_itm_call_delta(self):
        """Test that deep ITM call has delta near 1."""
        delta = calculate_delta(S=150, K=100, T=0.25, r=0.05, sigma=0.20, option_type='call')
        assert delta > 0.95

    def test_deep_otm_call_delta(self):
        """Test that deep OTM call has delta near 0."""
        delta = calculate_delta(S=50, K=100, T=0.25, r=0.05, sigma=0.20, option_type='call')
        assert delta < 0.05

    def test_delta_at_expiry_itm(self):
        """Test delta at expiry for ITM option."""
        delta = calculate_delta(S=110, K=100, T=0, r=0.05, sigma=0.20, option_type='call')
        assert delta == 1.0

    def test_delta_at_expiry_otm(self):
        """Test delta at expiry for OTM option."""
        delta = calculate_delta(S=90, K=100, T=0, r=0.05, sigma=0.20, option_type='call')
        assert delta == 0.0


class TestGamma:
    """Tests for gamma calculation."""

    def test_gamma_non_negative(self, standard_params):
        """Test that gamma is always non-negative."""
        gamma = calculate_gamma(**standard_params)
        assert gamma >= 0

    def test_gamma_same_for_call_put(self, standard_params):
        """Test that gamma is the same for calls and puts."""
        # Gamma is calculated without option_type in our implementation
        # But let's verify it matches for both
        greeks_call = calculate_greeks(**standard_params, option_type='call')
        greeks_put = calculate_greeks(**standard_params, option_type='put')

        assert abs(greeks_call['gamma'] - greeks_put['gamma']) < GREEK_TOL

    def test_gamma_highest_atm(self):
        """Test that gamma is highest for ATM options."""
        gamma_atm = calculate_gamma(S=100, K=100, T=0.25, r=0.05, sigma=0.20)
        gamma_itm = calculate_gamma(S=120, K=100, T=0.25, r=0.05, sigma=0.20)
        gamma_otm = calculate_gamma(S=80, K=100, T=0.25, r=0.05, sigma=0.20)

        assert gamma_atm > gamma_itm
        assert gamma_atm > gamma_otm

    def test_gamma_increases_near_expiry_atm(self):
        """Test that ATM gamma increases as expiry approaches."""
        gamma_long = calculate_gamma(S=100, K=100, T=0.5, r=0.05, sigma=0.20)
        gamma_short = calculate_gamma(S=100, K=100, T=0.1, r=0.05, sigma=0.20)

        assert gamma_short > gamma_long


class TestTheta:
    """Tests for theta calculation."""

    def test_theta_negative_for_long_call(self, standard_params):
        """Test that theta is typically negative for long calls."""
        theta = calculate_theta(**standard_params, option_type='call')
        assert theta < 0

    def test_theta_negative_for_long_put(self, standard_params):
        """Test that theta is typically negative for ATM long puts."""
        theta = calculate_theta(**standard_params, option_type='put')
        assert theta < 0

    def test_theta_annualized(self, standard_params):
        """Test theta is properly annualized (per day)."""
        theta_daily = calculate_theta(**standard_params, option_type='call', annualize=True)
        theta_annual = calculate_theta(**standard_params, option_type='call', annualize=False)

        assert abs(theta_daily - theta_annual / DAYS_PER_YEAR) < GREEK_TOL

    def test_theta_increases_near_expiry_atm(self):
        """Test that ATM theta (absolute) increases near expiry."""
        theta_long = abs(calculate_theta(S=100, K=100, T=0.5, r=0.05, sigma=0.20, option_type='call'))
        theta_short = abs(calculate_theta(S=100, K=100, T=0.1, r=0.05, sigma=0.20, option_type='call'))

        assert theta_short > theta_long


class TestVega:
    """Tests for vega calculation."""

    def test_vega_non_negative(self, standard_params):
        """Test that vega is always non-negative."""
        vega = calculate_vega(**standard_params)
        assert vega >= 0

    def test_vega_same_for_call_put(self, standard_params):
        """Test that vega is the same for calls and puts."""
        greeks_call = calculate_greeks(**standard_params, option_type='call')
        greeks_put = calculate_greeks(**standard_params, option_type='put')

        assert abs(greeks_call['vega'] - greeks_put['vega']) < GREEK_TOL

    def test_vega_highest_atm(self):
        """Test that vega is highest for ATM options."""
        vega_atm = calculate_vega(S=100, K=100, T=0.25, r=0.05, sigma=0.20)
        vega_itm = calculate_vega(S=120, K=100, T=0.25, r=0.05, sigma=0.20)
        vega_otm = calculate_vega(S=80, K=100, T=0.25, r=0.05, sigma=0.20)

        assert vega_atm > vega_itm
        assert vega_atm > vega_otm

    def test_vega_per_percent(self, standard_params):
        """Test vega scaling per 1% vol change."""
        vega_pct = calculate_vega(**standard_params, per_percent=True)
        vega_unit = calculate_vega(**standard_params, per_percent=False)

        assert abs(vega_pct - vega_unit / 100) < GREEK_TOL


class TestRho:
    """Tests for rho calculation."""

    def test_call_rho_positive(self, standard_params):
        """Test that call rho is positive (higher rates -> higher call prices)."""
        rho = calculate_rho(**standard_params, option_type='call')
        assert rho > 0

    def test_put_rho_negative(self, standard_params):
        """Test that put rho is negative (higher rates -> lower put prices)."""
        rho = calculate_rho(**standard_params, option_type='put')
        assert rho < 0

    def test_rho_per_percent(self, standard_params):
        """Test rho scaling per 1% rate change."""
        rho_pct = calculate_rho(**standard_params, option_type='call', per_percent=True)
        rho_unit = calculate_rho(**standard_params, option_type='call', per_percent=False)

        assert abs(rho_pct - rho_unit / 100) < GREEK_TOL


class TestCalculateGreeks:
    """Tests for the combined calculate_greeks function."""

    def test_returns_all_greeks(self, standard_params):
        """Test that all Greeks are returned."""
        greeks = calculate_greeks(**standard_params, option_type='call')

        assert 'delta' in greeks
        assert 'gamma' in greeks
        assert 'theta' in greeks
        assert 'vega' in greeks
        assert 'rho' in greeks

    def test_greek_values_reasonable(self, standard_params):
        """Test that Greek values are in reasonable ranges."""
        greeks = calculate_greeks(**standard_params, option_type='call')

        assert 0 <= greeks['delta'] <= 1
        assert greeks['gamma'] >= 0
        assert greeks['theta'] < 0  # Time decay
        assert greeks['vega'] >= 0
        assert greeks['rho'] > 0  # Call rho


# =============================================================================
# Implied Volatility Tests
# =============================================================================

class TestImpliedVolatility:
    """Tests for implied volatility calculation."""

    def test_iv_roundtrip_atm(self, standard_params):
        """Test IV recovery: price -> IV -> price."""
        original_price = black_scholes_call(**standard_params)

        recovered_iv = calculate_implied_volatility(
            option_price=original_price,
            S=standard_params['S'],
            K=standard_params['K'],
            T=standard_params['T'],
            r=standard_params['r'],
            option_type='call'
        )

        assert abs(recovered_iv - SIGMA) < IV_TOL

    def test_iv_roundtrip_itm(self):
        """Test IV recovery for ITM option."""
        sigma = 0.25
        price = black_scholes_call(S=120, K=100, T=0.5, r=0.05, sigma=sigma)

        recovered_iv = calculate_implied_volatility(
            option_price=price,
            S=120, K=100, T=0.5, r=0.05,
            option_type='call'
        )

        assert abs(recovered_iv - sigma) < IV_TOL

    def test_iv_roundtrip_otm(self):
        """Test IV recovery for OTM option."""
        sigma = 0.30
        price = black_scholes_call(S=80, K=100, T=0.5, r=0.05, sigma=sigma)

        recovered_iv = calculate_implied_volatility(
            option_price=price,
            S=80, K=100, T=0.5, r=0.05,
            option_type='call'
        )

        assert abs(recovered_iv - sigma) < IV_TOL

    def test_iv_put(self):
        """Test IV calculation for put options."""
        sigma = 0.20
        price = black_scholes_put(S=100, K=100, T=0.25, r=0.05, sigma=sigma)

        recovered_iv = calculate_implied_volatility(
            option_price=price,
            S=100, K=100, T=0.25, r=0.05,
            option_type='put'
        )

        assert abs(recovered_iv - sigma) < IV_TOL

    def test_iv_brent_method(self, standard_params):
        """Test IV calculation using Brent's method."""
        original_price = black_scholes_call(**standard_params)

        recovered_iv = calculate_implied_volatility(
            option_price=original_price,
            S=standard_params['S'],
            K=standard_params['K'],
            T=standard_params['T'],
            r=standard_params['r'],
            option_type='call',
            method='brent'
        )

        assert abs(recovered_iv - SIGMA) < IV_TOL

    def test_iv_at_expiry_raises_error(self):
        """Test that IV at expiry raises ImpliedVolatilityError."""
        with pytest.raises(ImpliedVolatilityError):
            calculate_implied_volatility(
                option_price=10.0,
                S=110, K=100, T=0, r=0.05,
                option_type='call'
            )

    def test_iv_below_intrinsic_raises_error(self):
        """Test that price below intrinsic raises error."""
        with pytest.raises(ImpliedVolatilityError):
            calculate_implied_volatility(
                option_price=5.0,  # Below intrinsic of $20
                S=120, K=100, T=0.25, r=0.05,
                option_type='call'
            )


# =============================================================================
# Edge Cases and Numerical Stability Tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and numerical stability."""

    def test_very_short_expiry(self):
        """Test pricing with very short time to expiry."""
        price = black_scholes_call(S=100, K=100, T=0.001, r=0.05, sigma=0.20)
        assert price >= 0
        assert np.isfinite(price)

    def test_very_long_expiry(self):
        """Test pricing with long time to expiry."""
        price = black_scholes_call(S=100, K=100, T=10.0, r=0.05, sigma=0.20)
        assert price >= 0
        assert np.isfinite(price)
        assert price < 100  # Should be less than spot

    def test_very_low_volatility(self):
        """Test pricing with very low volatility."""
        price = black_scholes_call(S=110, K=100, T=0.25, r=0.05, sigma=0.001)
        intrinsic = 110 - 100 * np.exp(-0.05 * 0.25)
        assert abs(price - intrinsic) < PRICE_TOL

    def test_very_high_volatility(self):
        """Test pricing with very high volatility."""
        price = black_scholes_call(S=100, K=100, T=0.25, r=0.05, sigma=2.0)
        assert price >= 0
        assert np.isfinite(price)

    def test_zero_interest_rate(self):
        """Test pricing with zero interest rate."""
        call = black_scholes_call(S=100, K=100, T=0.25, r=0.0, sigma=0.20)
        put = black_scholes_put(S=100, K=100, T=0.25, r=0.0, sigma=0.20)

        # Put-call parity still holds
        assert abs((call - put) - (100 - 100)) < PRICE_TOL

    def test_negative_interest_rate(self):
        """Test pricing with negative interest rate."""
        call = black_scholes_call(S=100, K=100, T=0.25, r=-0.01, sigma=0.20)
        put = black_scholes_put(S=100, K=100, T=0.25, r=-0.01, sigma=0.20)

        # Both should be finite
        assert np.isfinite(call)
        assert np.isfinite(put)

        # Put-call parity still holds
        pv_strike = 100 * np.exp(0.01 * 0.25)  # Note: -r becomes +r
        assert abs((call - put) - (100 - pv_strike)) < PRICE_TOL

    def test_deep_itm_option(self):
        """Test very deep ITM option."""
        price = black_scholes_call(S=1000, K=100, T=0.25, r=0.05, sigma=0.20)
        intrinsic = 1000 - 100 * np.exp(-0.05 * 0.25)

        assert price >= intrinsic - PRICE_TOL
        assert np.isfinite(price)

    def test_deep_otm_option(self):
        """Test very deep OTM option."""
        price = black_scholes_call(S=10, K=100, T=0.25, r=0.05, sigma=0.20)

        assert price >= 0
        assert price < 0.01  # Very small
        assert np.isfinite(price)


class TestInputValidation:
    """Tests for input validation."""

    def test_negative_spot_raises_error(self):
        """Test that negative spot raises ValueError."""
        with pytest.raises(ValueError):
            black_scholes_call(S=-100, K=100, T=0.25, r=0.05, sigma=0.20)

    def test_negative_strike_raises_error(self):
        """Test that negative strike raises ValueError."""
        with pytest.raises(ValueError):
            black_scholes_call(S=100, K=-100, T=0.25, r=0.05, sigma=0.20)

    def test_negative_time_raises_error(self):
        """Test that negative time raises ValueError."""
        with pytest.raises(ValueError):
            black_scholes_call(S=100, K=100, T=-0.25, r=0.05, sigma=0.20)

    def test_negative_volatility_raises_error(self):
        """Test that negative volatility raises ValueError."""
        with pytest.raises(ValueError):
            black_scholes_call(S=100, K=100, T=0.25, r=0.05, sigma=-0.20)

    def test_extreme_volatility_raises_error(self):
        """Test that extreme volatility raises ValueError."""
        with pytest.raises(ValueError):
            black_scholes_call(S=100, K=100, T=0.25, r=0.05, sigma=100.0)

    def test_none_parameters_raise_error(self):
        """Test that None parameters raise ValueError."""
        with pytest.raises(ValueError):
            black_scholes_call(S=None, K=100, T=0.25, r=0.05, sigma=0.20)

    def test_invalid_option_type_raises_error(self):
        """Test that invalid option type raises ValueError."""
        with pytest.raises(ValueError):
            black_scholes_price(S=100, K=100, T=0.25, r=0.05, sigma=0.20, option_type='forward')


# =============================================================================
# Vectorized Function Tests
# =============================================================================

class TestVectorizedFunctions:
    """Tests for vectorized pricing functions."""

    def test_vectorized_call_single_value(self):
        """Test vectorized call with single values."""
        price = black_scholes_call_vectorized(
            S=np.array([100]),
            K=np.array([100]),
            T=np.array([0.25]),
            r=np.array([0.05]),
            sigma=np.array([0.20])
        )

        scalar_price = black_scholes_call(S=100, K=100, T=0.25, r=0.05, sigma=0.20)

        assert abs(price[0] - scalar_price) < PRICE_TOL

    def test_vectorized_call_multiple_values(self):
        """Test vectorized call with multiple values."""
        S = np.array([100, 110, 90])
        K = np.array([100, 100, 100])
        T = np.array([0.25, 0.25, 0.25])
        r = np.array([0.05, 0.05, 0.05])
        sigma = np.array([0.20, 0.20, 0.20])

        prices = black_scholes_call_vectorized(S, K, T, r, sigma)

        for i in range(len(S)):
            scalar_price = black_scholes_call(S[i], K[i], T[i], r[i], sigma[i])
            assert abs(prices[i] - scalar_price) < PRICE_TOL

    def test_vectorized_put_matches_scalar(self):
        """Test vectorized put matches scalar implementation."""
        S = np.array([100, 110, 90])
        K = np.array([100, 100, 100])
        T = np.array([0.25, 0.25, 0.25])
        r = np.array([0.05, 0.05, 0.05])
        sigma = np.array([0.20, 0.20, 0.20])

        prices = black_scholes_put_vectorized(S, K, T, r, sigma)

        for i in range(len(S)):
            scalar_price = black_scholes_put(S[i], K[i], T[i], r[i], sigma[i])
            assert abs(prices[i] - scalar_price) < PRICE_TOL


# =============================================================================
# Performance Sanity Tests
# =============================================================================

class TestPerformance:
    """Basic performance sanity tests."""

    def test_large_batch_call_pricing(self):
        """Test pricing a large batch of options."""
        n = 10000
        S = np.random.uniform(80, 120, n)
        K = np.full(n, 100.0)
        T = np.random.uniform(0.1, 1.0, n)
        r = np.full(n, 0.05)
        sigma = np.random.uniform(0.15, 0.35, n)

        prices = black_scholes_call_vectorized(S, K, T, r, sigma)

        assert len(prices) == n
        assert np.all(prices >= 0)
        assert np.all(np.isfinite(prices))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
