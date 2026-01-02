"""
Tests for American Option Pricing module.

Tests the binomial tree implementation for American-style options,
including early exercise, dividend handling, and Greeks calculation.
"""

import pytest
import numpy as np
from backtester.core.american_pricing import (
    BinomialPricer,
    AmericanOptionPricer,
    AmericanPricingError,
    TreeConstructionError,
    price_american_option,
    calculate_american_greeks,
    DEFAULT_TREE_STEPS,
)
from backtester.core.pricing import black_scholes_call, black_scholes_put


class TestBinomialPricerConstruction:
    def test_default_steps(self):
        pricer = BinomialPricer()
        assert pricer.steps == DEFAULT_TREE_STEPS

    def test_custom_steps(self):
        pricer = BinomialPricer(steps=100)
        assert pricer.steps == 100

    def test_invalid_steps_zero(self):
        with pytest.raises(ValueError):
            BinomialPricer(steps=0)

    def test_invalid_steps_negative(self):
        with pytest.raises(ValueError):
            BinomialPricer(steps=-10)


class TestBinomialPricerValidation:
    @pytest.fixture
    def pricer(self):
        return BinomialPricer(steps=100)

    def test_negative_spot(self, pricer):
        with pytest.raises(ValueError, match="Spot price"):
            pricer.price(S=-100, K=100, T=0.25, r=0.05, sigma=0.20, option_type="call")

    def test_negative_strike(self, pricer):
        with pytest.raises(ValueError, match="Strike price"):
            pricer.price(S=100, K=-100, T=0.25, r=0.05, sigma=0.20, option_type="call")

    def test_negative_time(self, pricer):
        with pytest.raises(ValueError, match="Time"):
            pricer.price(S=100, K=100, T=-0.25, r=0.05, sigma=0.20, option_type="call")

    def test_negative_volatility(self, pricer):
        with pytest.raises(ValueError, match="Volatility"):
            pricer.price(S=100, K=100, T=0.25, r=0.05, sigma=-0.20, option_type="call")

    def test_invalid_option_type(self, pricer):
        with pytest.raises(ValueError, match="option_type"):
            pricer.price(
                S=100, K=100, T=0.25, r=0.05, sigma=0.20, option_type="invalid"
            )

    def test_nan_input(self, pricer):
        with pytest.raises(ValueError):
            pricer.price(
                S=np.nan, K=100, T=0.25, r=0.05, sigma=0.20, option_type="call"
            )


class TestBinomialPricerEdgeCases:
    @pytest.fixture
    def pricer(self):
        return BinomialPricer(steps=100)

    def test_zero_time_call_itm(self, pricer):
        price = pricer.price(S=110, K=100, T=0, r=0.05, sigma=0.20, option_type="call")
        assert price == pytest.approx(10.0, rel=1e-6)

    def test_zero_time_call_otm(self, pricer):
        price = pricer.price(S=90, K=100, T=0, r=0.05, sigma=0.20, option_type="call")
        assert price == pytest.approx(0.0, rel=1e-6)

    def test_zero_time_put_itm(self, pricer):
        price = pricer.price(S=90, K=100, T=0, r=0.05, sigma=0.20, option_type="put")
        assert price == pytest.approx(10.0, rel=1e-6)

    def test_zero_time_put_otm(self, pricer):
        price = pricer.price(S=110, K=100, T=0, r=0.05, sigma=0.20, option_type="put")
        assert price == pytest.approx(0.0, rel=1e-6)

    def test_zero_volatility(self, pricer):
        price = pricer.price(S=100, K=100, T=0.25, r=0.05, sigma=0, option_type="call")
        assert price >= 0


class TestBinomialVsBlackScholes:
    """Test that binomial converges to Black-Scholes for European options"""

    @pytest.fixture
    def pricer(self):
        return BinomialPricer(steps=500)

    def test_call_convergence_atm(self, pricer):
        S, K, T, r, sigma = 100, 100, 0.25, 0.05, 0.20

        bs_price = black_scholes_call(S, K, T, r, sigma)
        binomial_price = pricer.price(S, K, T, r, sigma, "call", is_american=False)

        assert binomial_price == pytest.approx(bs_price, rel=0.01)

    def test_call_convergence_itm(self, pricer):
        S, K, T, r, sigma = 110, 100, 0.25, 0.05, 0.20

        bs_price = black_scholes_call(S, K, T, r, sigma)
        binomial_price = pricer.price(S, K, T, r, sigma, "call", is_american=False)

        assert binomial_price == pytest.approx(bs_price, rel=0.01)

    def test_call_convergence_otm(self, pricer):
        S, K, T, r, sigma = 90, 100, 0.25, 0.05, 0.20

        bs_price = black_scholes_call(S, K, T, r, sigma)
        binomial_price = pricer.price(S, K, T, r, sigma, "call", is_american=False)

        assert binomial_price == pytest.approx(bs_price, rel=0.01)

    def test_put_convergence_atm(self, pricer):
        S, K, T, r, sigma = 100, 100, 0.25, 0.05, 0.20

        bs_price = black_scholes_put(S, K, T, r, sigma)
        binomial_price = pricer.price(S, K, T, r, sigma, "put", is_american=False)

        assert binomial_price == pytest.approx(bs_price, rel=0.01)

    def test_put_convergence_itm(self, pricer):
        S, K, T, r, sigma = 90, 100, 0.25, 0.05, 0.20

        bs_price = black_scholes_put(S, K, T, r, sigma)
        binomial_price = pricer.price(S, K, T, r, sigma, "put", is_american=False)

        assert binomial_price == pytest.approx(bs_price, rel=0.01)


class TestAmericanEarlyExercise:
    """Test early exercise premium for American options"""

    @pytest.fixture
    def pricer(self):
        return BinomialPricer(steps=200)

    def test_american_put_worth_more_than_european(self, pricer):
        S, K, T, r, sigma = 90, 100, 0.5, 0.05, 0.25

        american_price = pricer.price(S, K, T, r, sigma, "put", is_american=True)
        european_price = pricer.price(S, K, T, r, sigma, "put", is_american=False)

        assert american_price >= european_price

    def test_deep_itm_put_has_early_exercise_premium(self, pricer):
        S, K, T, r, sigma = 70, 100, 0.5, 0.10, 0.25

        american_price = pricer.price(S, K, T, r, sigma, "put", is_american=True)
        european_price = pricer.price(S, K, T, r, sigma, "put", is_american=False)

        premium = american_price - european_price
        assert premium > 0.1

    def test_american_call_no_dividends_equals_european(self, pricer):
        S, K, T, r, sigma = 100, 100, 0.25, 0.05, 0.20

        american_price = pricer.price(S, K, T, r, sigma, "call", is_american=True)
        european_price = pricer.price(S, K, T, r, sigma, "call", is_american=False)

        assert american_price == pytest.approx(european_price, rel=0.001)

    def test_american_option_at_least_intrinsic(self, pricer):
        S, K, T, r, sigma = 80, 100, 0.25, 0.05, 0.20

        put_price = pricer.price(S, K, T, r, sigma, "put", is_american=True)
        intrinsic = max(K - S, 0)

        assert put_price >= intrinsic


class TestDividendHandling:
    """Test discrete dividend handling"""

    @pytest.fixture
    def pricer(self):
        return BinomialPricer(steps=200)

    def test_dividend_reduces_call_value(self, pricer):
        S, K, T, r, sigma = 100, 100, 0.5, 0.05, 0.20

        call_no_div = pricer.price(S, K, T, r, sigma, "call")
        call_with_div = pricer.price(S, K, T, r, sigma, "call", dividends=[(0.25, 2.0)])

        assert call_with_div < call_no_div

    def test_dividend_increases_put_value(self, pricer):
        S, K, T, r, sigma = 100, 100, 0.5, 0.05, 0.20

        put_no_div = pricer.price(S, K, T, r, sigma, "put")
        put_with_div = pricer.price(S, K, T, r, sigma, "put", dividends=[(0.25, 2.0)])

        assert put_with_div > put_no_div

    def test_multiple_dividends(self, pricer):
        S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20

        divs = [(0.25, 1.0), (0.5, 1.0), (0.75, 1.0)]
        call_price = pricer.price(S, K, T, r, sigma, "call", dividends=divs)

        assert call_price > 0
        assert call_price < S

    def test_dividend_after_expiration_ignored(self, pricer):
        S, K, T, r, sigma = 100, 100, 0.25, 0.05, 0.20

        call_no_div = pricer.price(S, K, T, r, sigma, "call")
        call_late_div = pricer.price(S, K, T, r, sigma, "call", dividends=[(0.5, 5.0)])

        assert call_no_div == pytest.approx(call_late_div, rel=1e-6)


class TestGreeksCalculation:
    """Test Greeks calculation via finite differences"""

    @pytest.fixture
    def pricer(self):
        return BinomialPricer(steps=200)

    def test_greeks_returns_dict(self, pricer):
        greeks = pricer.calculate_greeks(
            S=100, K=100, T=0.25, r=0.05, sigma=0.20, option_type="call"
        )

        assert isinstance(greeks, dict)
        assert "delta" in greeks
        assert "gamma" in greeks
        assert "theta" in greeks
        assert "vega" in greeks
        assert "rho" in greeks

    def test_call_delta_positive(self, pricer):
        greeks = pricer.calculate_greeks(
            S=100, K=100, T=0.25, r=0.05, sigma=0.20, option_type="call"
        )
        assert greeks["delta"] > 0

    def test_call_delta_between_zero_and_one(self, pricer):
        greeks = pricer.calculate_greeks(
            S=100, K=100, T=0.25, r=0.05, sigma=0.20, option_type="call"
        )
        assert 0 < greeks["delta"] < 1

    def test_put_delta_negative(self, pricer):
        greeks = pricer.calculate_greeks(
            S=100, K=100, T=0.25, r=0.05, sigma=0.20, option_type="put"
        )
        assert greeks["delta"] < 0

    def test_put_delta_between_minus_one_and_zero(self, pricer):
        greeks = pricer.calculate_greeks(
            S=100, K=100, T=0.25, r=0.05, sigma=0.20, option_type="put"
        )
        assert -1 < greeks["delta"] < 0

    def test_gamma_positive(self, pricer):
        greeks = pricer.calculate_greeks(
            S=100, K=100, T=0.25, r=0.05, sigma=0.20, option_type="call"
        )
        assert greeks["gamma"] > 0

    def test_theta_typically_negative_for_long_call(self, pricer):
        greeks = pricer.calculate_greeks(
            S=100, K=100, T=0.25, r=0.05, sigma=0.20, option_type="call"
        )
        assert greeks["theta"] < 0

    def test_vega_positive(self, pricer):
        greeks = pricer.calculate_greeks(
            S=100, K=100, T=0.25, r=0.05, sigma=0.20, option_type="call"
        )
        assert greeks["vega"] > 0

    def test_call_rho_positive(self, pricer):
        greeks = pricer.calculate_greeks(
            S=100, K=100, T=0.25, r=0.05, sigma=0.20, option_type="call"
        )
        assert greeks["rho"] > 0

    def test_put_rho_negative(self, pricer):
        greeks = pricer.calculate_greeks(
            S=100, K=100, T=0.25, r=0.05, sigma=0.20, option_type="put"
        )
        assert greeks["rho"] < 0

    def test_itm_call_high_delta(self, pricer):
        greeks = pricer.calculate_greeks(
            S=120, K=100, T=0.25, r=0.05, sigma=0.20, option_type="call"
        )
        assert greeks["delta"] > 0.7

    def test_otm_call_low_delta(self, pricer):
        greeks = pricer.calculate_greeks(
            S=80, K=100, T=0.25, r=0.05, sigma=0.20, option_type="call"
        )
        assert greeks["delta"] < 0.3

    def test_atm_has_highest_gamma(self, pricer):
        gamma_atm = pricer.calculate_greeks(
            S=100, K=100, T=0.25, r=0.05, sigma=0.20, option_type="call"
        )["gamma"]
        gamma_itm = pricer.calculate_greeks(
            S=120, K=100, T=0.25, r=0.05, sigma=0.20, option_type="call"
        )["gamma"]
        gamma_otm = pricer.calculate_greeks(
            S=80, K=100, T=0.25, r=0.05, sigma=0.20, option_type="call"
        )["gamma"]

        assert gamma_atm > gamma_itm
        assert gamma_atm > gamma_otm


class TestAmericanOptionPricer:
    """Test unified American option pricer"""

    @pytest.fixture
    def pricer(self):
        return AmericanOptionPricer()

    def test_call_uses_black_scholes_no_dividends(self, pricer):
        S, K, T, r, sigma = 100, 100, 0.25, 0.05, 0.20

        american_price = pricer.price(S, K, T, r, sigma, "call")
        bs_price = black_scholes_call(S, K, T, r, sigma)

        assert american_price == pytest.approx(bs_price, rel=1e-6)

    def test_put_uses_binomial(self, pricer):
        S, K, T, r, sigma = 90, 100, 0.5, 0.10, 0.25

        american_price = pricer.price(S, K, T, r, sigma, "put")
        european_price = black_scholes_put(S, K, T, r, sigma)

        assert american_price >= european_price

    def test_early_exercise_premium_positive_for_put(self, pricer):
        premium = pricer.early_exercise_premium(
            S=70, K=100, T=0.5, r=0.10, sigma=0.25, option_type="put"
        )
        assert premium > 0

    def test_early_exercise_premium_zero_for_call_no_dividends(self, pricer):
        premium = pricer.early_exercise_premium(
            S=100, K=100, T=0.25, r=0.05, sigma=0.20, option_type="call"
        )
        assert premium == pytest.approx(0.0, abs=0.01)

    def test_force_binomial_for_call(self):
        pricer = AmericanOptionPricer(use_binomial_always=True)
        price = pricer.price(
            S=100, K=100, T=0.25, r=0.05, sigma=0.20, option_type="call"
        )
        assert price > 0


class TestConvenienceFunctions:
    """Test module-level convenience functions"""

    def test_price_american_option(self):
        price = price_american_option(
            S=100, K=100, T=0.25, r=0.05, sigma=0.20, option_type="put"
        )
        assert price > 0

    def test_calculate_american_greeks(self):
        greeks = calculate_american_greeks(
            S=100, K=100, T=0.25, r=0.05, sigma=0.20, option_type="put"
        )

        assert "delta" in greeks
        assert "gamma" in greeks
        assert "theta" in greeks
        assert "vega" in greeks
        assert "rho" in greeks

    def test_custom_steps(self):
        price_low = price_american_option(
            S=100, K=100, T=0.25, r=0.05, sigma=0.20, option_type="put", steps=50
        )
        price_high = price_american_option(
            S=100, K=100, T=0.25, r=0.05, sigma=0.20, option_type="put", steps=500
        )

        assert abs(price_low - price_high) < 0.1


class TestAccuracyBenchmarks:
    """Compare binomial prices to known analytical values"""

    @pytest.fixture
    def pricer(self):
        return BinomialPricer(steps=500)

    def test_atm_call_matches_bs(self, pricer):
        S, K, T, r, sigma = 100, 100, 0.25, 0.05, 0.20

        bs_price = black_scholes_call(S, K, T, r, sigma)
        binomial_price = pricer.price(S, K, T, r, sigma, "call", is_american=False)

        assert binomial_price == pytest.approx(bs_price, abs=0.05)

    def test_atm_put_matches_bs(self, pricer):
        S, K, T, r, sigma = 100, 100, 0.25, 0.05, 0.20

        bs_price = black_scholes_put(S, K, T, r, sigma)
        binomial_price = pricer.price(S, K, T, r, sigma, "put", is_american=False)

        assert binomial_price == pytest.approx(bs_price, abs=0.05)

    def test_high_vol_options(self, pricer):
        S, K, T, r, sigma = 100, 100, 0.25, 0.05, 0.50

        bs_call = black_scholes_call(S, K, T, r, sigma)
        binomial_call = pricer.price(S, K, T, r, sigma, "call", is_american=False)

        assert binomial_call == pytest.approx(bs_call, rel=0.02)

    def test_long_dated_options(self, pricer):
        S, K, T, r, sigma = 100, 100, 2.0, 0.05, 0.20

        bs_call = black_scholes_call(S, K, T, r, sigma)
        binomial_call = pricer.price(S, K, T, r, sigma, "call", is_american=False)

        assert binomial_call == pytest.approx(bs_call, rel=0.02)


class TestPerformance:
    """Test performance characteristics"""

    def test_pricing_speed_reasonable(self):
        import time

        pricer = BinomialPricer(steps=100)

        start = time.time()
        for _ in range(100):
            pricer.price(S=100, K=100, T=0.25, r=0.05, sigma=0.20, option_type="put")
        elapsed = time.time() - start

        assert elapsed < 5.0

    def test_greeks_speed_reasonable(self):
        import time

        pricer = BinomialPricer(steps=100)

        start = time.time()
        for _ in range(20):
            pricer.calculate_greeks(
                S=100, K=100, T=0.25, r=0.05, sigma=0.20, option_type="put"
            )
        elapsed = time.time() - start

        assert elapsed < 5.0


class TestModuleExports:
    """Test that all expected exports are available"""

    def test_classes_exported(self):
        from backtester.core.american_pricing import (
            BinomialPricer,
            AmericanOptionPricer,
        )

        assert BinomialPricer is not None
        assert AmericanOptionPricer is not None

    def test_exceptions_exported(self):
        from backtester.core.american_pricing import (
            AmericanPricingError,
            TreeConstructionError,
        )

        assert issubclass(AmericanPricingError, Exception)
        assert issubclass(TreeConstructionError, AmericanPricingError)

    def test_functions_exported(self):
        from backtester.core.american_pricing import (
            price_american_option,
            calculate_american_greeks,
        )

        assert callable(price_american_option)
        assert callable(calculate_american_greeks)

    def test_constants_exported(self):
        from backtester.core.american_pricing import DEFAULT_TREE_STEPS

        assert isinstance(DEFAULT_TREE_STEPS, int)
        assert DEFAULT_TREE_STEPS > 0
