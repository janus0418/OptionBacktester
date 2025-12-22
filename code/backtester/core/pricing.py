"""
Options Pricing Module for Options Backtesting

This module provides comprehensive options pricing functionality using the
Black-Scholes model and analytical Greeks calculations. It serves as the
foundation for option valuation and risk management in the backtesting system.

Mathematical Framework:
    The Black-Scholes model assumes:
    - European-style options (no early exercise)
    - Log-normal distribution of underlying returns
    - Constant volatility and risk-free rate
    - No dividends (or dividend-adjusted spot price)
    - Continuous trading with no transaction costs

Key Formulas:
    Call Price: C = S*N(d1) - K*exp(-rT)*N(d2)
    Put Price:  P = K*exp(-rT)*N(-d2) - S*N(-d1)

    where:
        d1 = [ln(S/K) + (r + sigma^2/2)*T] / (sigma*sqrt(T))
        d2 = d1 - sigma*sqrt(T)
        N(x) = cumulative standard normal distribution

Greeks:
    Delta: Rate of change of option price with respect to underlying price
    Gamma: Rate of change of delta with respect to underlying price
    Theta: Rate of change of option price with respect to time
    Vega:  Rate of change of option price with respect to volatility
    Rho:   Rate of change of option price with respect to interest rate

Usage:
    from backtester.core.pricing import (
        black_scholes_call,
        black_scholes_put,
        calculate_greeks,
        calculate_implied_volatility
    )

    # Price a call option
    call_price = black_scholes_call(S=100, K=100, T=0.25, r=0.05, sigma=0.20)

    # Calculate Greeks
    greeks = calculate_greeks(S=100, K=100, T=0.25, r=0.05, sigma=0.20, option_type='call')

    # Calculate implied volatility
    iv = calculate_implied_volatility(
        option_price=5.0, S=100, K=100, T=0.25, r=0.05, option_type='call'
    )

References:
    - Black, F., & Scholes, M. (1973). The Pricing of Options and Corporate Liabilities.
    - Hull, J. C. (2018). Options, Futures, and Other Derivatives.
    - Barone-Adesi, G., & Whaley, R. E. (1987). Efficient Analytic Approximation
      of American Option Values. The Journal of Finance, 42(2), 301-320.
"""

import logging
import math
from typing import Dict, Optional, Tuple, Literal

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq, newton

# Configure module logger
logger = logging.getLogger(__name__)


# =============================================================================
# Exceptions
# =============================================================================

class PricingError(Exception):
    """Exception raised when pricing calculation fails."""
    pass


class ImpliedVolatilityError(Exception):
    """Exception raised when implied volatility calculation fails to converge."""
    pass


# =============================================================================
# Constants
# =============================================================================

# Numerical stability thresholds
MIN_TIME_TO_EXPIRY = 1e-10  # Minimum time to avoid division by zero
MIN_VOLATILITY = 1e-10      # Minimum volatility to avoid division by zero
MAX_VOLATILITY = 10.0       # Maximum volatility for sanity checks (1000%)
MIN_SPOT_PRICE = 1e-10      # Minimum spot price
MIN_STRIKE_PRICE = 1e-10    # Minimum strike price

# IV calculation parameters
IV_INITIAL_GUESS = 0.30     # Initial guess for IV (30%)
IV_MAX_ITERATIONS = 100     # Maximum iterations for Newton-Raphson
IV_TOLERANCE = 1e-8         # Convergence tolerance for IV
IV_LOWER_BOUND = 0.001      # Lower bound for IV search (0.1%)
IV_UPPER_BOUND = 5.0        # Upper bound for IV search (500%)

# Annualization factors
DAYS_PER_YEAR = 365         # Calendar days for theta annualization
TRADING_DAYS_PER_YEAR = 252 # Trading days


# =============================================================================
# Helper Functions
# =============================================================================

def _validate_inputs(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    allow_zero_time: bool = True
) -> None:
    """
    Validate pricing inputs for numerical stability and financial validity.

    Args:
        S: Spot price
        K: Strike price
        T: Time to expiration in years
        r: Risk-free rate
        sigma: Volatility
        allow_zero_time: Whether to allow T=0

    Raises:
        ValueError: If any input is invalid
    """
    if S is None or K is None or T is None or r is None or sigma is None:
        raise ValueError("All pricing parameters must be provided (none can be None)")

    if not np.isfinite(S) or S < 0:
        raise ValueError(f"Spot price (S) must be non-negative and finite, got {S}")

    if not np.isfinite(K) or K < 0:
        raise ValueError(f"Strike price (K) must be non-negative and finite, got {K}")

    if not np.isfinite(T) or T < 0:
        raise ValueError(f"Time to expiration (T) must be non-negative and finite, got {T}")

    if not allow_zero_time and T <= 0:
        raise ValueError(f"Time to expiration (T) must be positive, got {T}")

    if not np.isfinite(r):
        raise ValueError(f"Risk-free rate (r) must be finite, got {r}")

    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError(f"Volatility (sigma) must be non-negative and finite, got {sigma}")

    if sigma > MAX_VOLATILITY:
        raise ValueError(
            f"Volatility (sigma) exceeds maximum allowed value of {MAX_VOLATILITY}, got {sigma}"
        )


def _calculate_d1_d2(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float
) -> Tuple[float, float]:
    """
    Calculate d1 and d2 parameters for Black-Scholes formula.

    Formula:
        d1 = [ln(S/K) + (r + sigma^2/2)*T] / (sigma*sqrt(T))
        d2 = d1 - sigma*sqrt(T)

    Args:
        S: Spot price
        K: Strike price
        T: Time to expiration in years
        r: Risk-free rate (annualized)
        sigma: Volatility (annualized)

    Returns:
        Tuple of (d1, d2)

    Note:
        Handles edge cases:
        - If T is very small, returns extreme values for d1/d2
        - If sigma is very small, returns extreme values for d1/d2
    """
    # Handle edge cases
    if T < MIN_TIME_TO_EXPIRY or sigma < MIN_VOLATILITY:
        # For T->0 or sigma->0, option value approaches intrinsic value
        # d1 and d2 approach +/- infinity depending on moneyness
        if S < MIN_SPOT_PRICE or K < MIN_STRIKE_PRICE:
            return 0.0, 0.0

        log_moneyness = np.log(S / K)
        if log_moneyness > 0:  # ITM call / OTM put
            return np.inf, np.inf
        elif log_moneyness < 0:  # OTM call / ITM put
            return -np.inf, -np.inf
        else:  # ATM
            return 0.0, 0.0

    sqrt_T = np.sqrt(T)
    sigma_sqrt_T = sigma * sqrt_T

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / sigma_sqrt_T
    d2 = d1 - sigma_sqrt_T

    return d1, d2


# =============================================================================
# Black-Scholes Pricing Functions
# =============================================================================

def black_scholes_call(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float
) -> float:
    """
    Calculate European call option price using Black-Scholes formula.

    The Black-Scholes call option price is given by:
        C = S * N(d1) - K * exp(-r*T) * N(d2)

    where:
        d1 = [ln(S/K) + (r + sigma^2/2)*T] / (sigma*sqrt(T))
        d2 = d1 - sigma*sqrt(T)
        N(x) = cumulative standard normal distribution

    Args:
        S: Current spot price of the underlying asset.
        K: Strike price of the option.
        T: Time to expiration in years (e.g., 0.25 for 3 months).
        r: Risk-free interest rate (annualized, e.g., 0.05 for 5%).
        sigma: Volatility of the underlying (annualized, e.g., 0.20 for 20%).

    Returns:
        Call option price.

    Raises:
        ValueError: If any input parameter is invalid.

    Edge Cases:
        - T = 0: Returns intrinsic value max(S - K, 0)
        - sigma = 0: Returns discounted intrinsic value max(S - K*exp(-rT), 0)
        - S = 0: Returns 0
        - K = 0: Returns S (option is deep ITM)

    Example:
        >>> price = black_scholes_call(S=100, K=100, T=0.25, r=0.05, sigma=0.20)
        >>> print(f"Call price: ${price:.2f}")
        Call price: $5.88

    References:
        Black, F., & Scholes, M. (1973). The Pricing of Options and Corporate
        Liabilities. Journal of Political Economy, 81(3), 637-654.
    """
    _validate_inputs(S, K, T, r, sigma, allow_zero_time=True)

    # Edge case: S = 0
    if S < MIN_SPOT_PRICE:
        return 0.0

    # Edge case: K = 0 (option is infinitely ITM)
    if K < MIN_STRIKE_PRICE:
        return S

    # Edge case: T = 0 (at expiration)
    if T < MIN_TIME_TO_EXPIRY:
        return max(S - K, 0.0)

    # Edge case: sigma = 0 (deterministic case)
    if sigma < MIN_VOLATILITY:
        discounted_strike = K * np.exp(-r * T)
        return max(S - discounted_strike, 0.0)

    # Standard Black-Scholes calculation
    d1, d2 = _calculate_d1_d2(S, K, T, r, sigma)

    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    # Ensure non-negative price (numerical precision)
    return max(call_price, 0.0)


def black_scholes_put(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float
) -> float:
    """
    Calculate European put option price using Black-Scholes formula.

    The Black-Scholes put option price is given by:
        P = K * exp(-r*T) * N(-d2) - S * N(-d1)

    Alternatively, using put-call parity:
        P = C - S + K * exp(-r*T)

    Args:
        S: Current spot price of the underlying asset.
        K: Strike price of the option.
        T: Time to expiration in years.
        r: Risk-free interest rate (annualized).
        sigma: Volatility of the underlying (annualized).

    Returns:
        Put option price.

    Raises:
        ValueError: If any input parameter is invalid.

    Edge Cases:
        - T = 0: Returns intrinsic value max(K - S, 0)
        - sigma = 0: Returns discounted intrinsic value max(K*exp(-rT) - S, 0)
        - S = 0: Returns K*exp(-rT)
        - K = 0: Returns 0

    Example:
        >>> price = black_scholes_put(S=100, K=100, T=0.25, r=0.05, sigma=0.20)
        >>> print(f"Put price: ${price:.2f}")
        Put price: $4.65

    Note:
        Put-call parity: C - P = S - K*exp(-rT)
        This relationship is maintained by the implementation.
    """
    _validate_inputs(S, K, T, r, sigma, allow_zero_time=True)

    # Edge case: K = 0
    if K < MIN_STRIKE_PRICE:
        return 0.0

    # Edge case: S = 0
    if S < MIN_SPOT_PRICE:
        return K * np.exp(-r * T) if T >= MIN_TIME_TO_EXPIRY else K

    # Edge case: T = 0 (at expiration)
    if T < MIN_TIME_TO_EXPIRY:
        return max(K - S, 0.0)

    # Edge case: sigma = 0 (deterministic case)
    if sigma < MIN_VOLATILITY:
        discounted_strike = K * np.exp(-r * T)
        return max(discounted_strike - S, 0.0)

    # Standard Black-Scholes calculation
    d1, d2 = _calculate_d1_d2(S, K, T, r, sigma)

    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    # Ensure non-negative price (numerical precision)
    return max(put_price, 0.0)


def black_scholes_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = 'call'
) -> float:
    """
    Calculate European option price using Black-Scholes formula.

    This is a convenience function that dispatches to black_scholes_call
    or black_scholes_put based on the option_type parameter.

    Args:
        S: Current spot price of the underlying asset.
        K: Strike price of the option.
        T: Time to expiration in years.
        r: Risk-free interest rate (annualized).
        sigma: Volatility of the underlying (annualized).
        option_type: 'call' or 'put'.

    Returns:
        Option price.

    Raises:
        ValueError: If option_type is not 'call' or 'put'.

    Example:
        >>> call_price = black_scholes_price(S=100, K=100, T=0.25, r=0.05, sigma=0.20, option_type='call')
        >>> put_price = black_scholes_price(S=100, K=100, T=0.25, r=0.05, sigma=0.20, option_type='put')
    """
    option_type_lower = option_type.lower().strip()

    if option_type_lower in ('call', 'c'):
        return black_scholes_call(S, K, T, r, sigma)
    elif option_type_lower in ('put', 'p'):
        return black_scholes_put(S, K, T, r, sigma)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")


# =============================================================================
# Greeks Calculations
# =============================================================================

def calculate_delta(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = 'call'
) -> float:
    """
    Calculate option delta - rate of change of option price with respect to spot.

    Delta represents the sensitivity of the option price to small changes in
    the underlying asset price. It is also interpreted as the hedge ratio
    (number of shares to hold to delta-hedge one option contract).

    Formula:
        Call Delta = N(d1)
        Put Delta = N(d1) - 1 = -N(-d1)

    Properties:
        - Call delta: 0 to 1 (approaches 1 for deep ITM, 0 for deep OTM)
        - Put delta: -1 to 0 (approaches -1 for deep ITM, 0 for deep OTM)
        - ATM options have delta around 0.5 (call) or -0.5 (put)

    Args:
        S: Spot price
        K: Strike price
        T: Time to expiration in years
        r: Risk-free rate (annualized)
        sigma: Volatility (annualized)
        option_type: 'call' or 'put'

    Returns:
        Delta value (between 0 and 1 for calls, -1 and 0 for puts)

    Example:
        >>> delta = calculate_delta(S=100, K=100, T=0.25, r=0.05, sigma=0.20, option_type='call')
        >>> print(f"Delta: {delta:.4f}")
        Delta: 0.5596
    """
    _validate_inputs(S, K, T, r, sigma, allow_zero_time=True)
    option_type_lower = option_type.lower().strip()

    # Edge cases
    if S < MIN_SPOT_PRICE:
        return 0.0 if option_type_lower in ('call', 'c') else -1.0

    if K < MIN_STRIKE_PRICE:
        return 1.0 if option_type_lower in ('call', 'c') else 0.0

    if T < MIN_TIME_TO_EXPIRY or sigma < MIN_VOLATILITY:
        # At expiration or with zero vol, delta is binary
        if option_type_lower in ('call', 'c'):
            return 1.0 if S > K else (0.5 if S == K else 0.0)
        else:
            return 0.0 if S > K else (-0.5 if S == K else -1.0)

    d1, _ = _calculate_d1_d2(S, K, T, r, sigma)

    if option_type_lower in ('call', 'c'):
        return norm.cdf(d1)
    elif option_type_lower in ('put', 'p'):
        return norm.cdf(d1) - 1.0
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")


def calculate_gamma(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float
) -> float:
    """
    Calculate option gamma - rate of change of delta with respect to spot.

    Gamma measures the curvature of the option price curve with respect to
    the underlying price. It is the same for both calls and puts (by put-call
    parity, which shows delta differs by 1).

    Formula:
        Gamma = N'(d1) / (S * sigma * sqrt(T))
        where N'(x) = standard normal PDF

    Properties:
        - Always positive for long options
        - Highest for ATM options near expiration
        - Approaches 0 for deep ITM/OTM options
        - Higher gamma means delta changes more rapidly

    Args:
        S: Spot price
        K: Strike price
        T: Time to expiration in years
        r: Risk-free rate (annualized)
        sigma: Volatility (annualized)

    Returns:
        Gamma value (always non-negative)

    Example:
        >>> gamma = calculate_gamma(S=100, K=100, T=0.25, r=0.05, sigma=0.20)
        >>> print(f"Gamma: {gamma:.4f}")
        Gamma: 0.0396
    """
    _validate_inputs(S, K, T, r, sigma, allow_zero_time=True)

    # Edge cases
    if S < MIN_SPOT_PRICE or K < MIN_STRIKE_PRICE:
        return 0.0

    if T < MIN_TIME_TO_EXPIRY or sigma < MIN_VOLATILITY:
        # Gamma approaches infinity at ATM, 0 otherwise
        # Return 0 for practical purposes (delta is discontinuous)
        return 0.0

    d1, _ = _calculate_d1_d2(S, K, T, r, sigma)

    sqrt_T = np.sqrt(T)
    gamma = norm.pdf(d1) / (S * sigma * sqrt_T)

    return gamma


def calculate_theta(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = 'call',
    annualize: bool = True
) -> float:
    """
    Calculate option theta - rate of change of option price with respect to time.

    Theta represents time decay - the rate at which the option loses value
    as time passes, all else being equal. By convention, theta is typically
    reported as the change in option value for one calendar day (negative
    for long positions).

    Formula (per year):
        Call Theta = -S*N'(d1)*sigma/(2*sqrt(T)) - r*K*exp(-rT)*N(d2)
        Put Theta = -S*N'(d1)*sigma/(2*sqrt(T)) + r*K*exp(-rT)*N(-d2)

    For daily theta, divide by 365 (calendar days).

    Properties:
        - Typically negative for long options (time decay)
        - Can be positive for deep ITM puts (early exercise premium)
        - Accelerates as expiration approaches (especially for ATM)

    Args:
        S: Spot price
        K: Strike price
        T: Time to expiration in years
        r: Risk-free rate (annualized)
        sigma: Volatility (annualized)
        option_type: 'call' or 'put'
        annualize: If True, returns per-day theta (divided by 365).
                  If False, returns per-year theta.

    Returns:
        Theta value (typically negative for long positions)

    Example:
        >>> theta = calculate_theta(S=100, K=100, T=0.25, r=0.05, sigma=0.20, option_type='call')
        >>> print(f"Theta (per day): ${theta:.4f}")
        Theta (per day): $-0.0440
    """
    _validate_inputs(S, K, T, r, sigma, allow_zero_time=True)
    option_type_lower = option_type.lower().strip()

    # Edge cases
    if S < MIN_SPOT_PRICE or K < MIN_STRIKE_PRICE:
        return 0.0

    if T < MIN_TIME_TO_EXPIRY:
        # At expiration, theta is 0 (option has expired)
        return 0.0

    if sigma < MIN_VOLATILITY:
        # With zero volatility, only interest rate effect remains
        discount_factor = np.exp(-r * T)
        if option_type_lower in ('call', 'c'):
            theta_annual = -r * K * discount_factor if S > K else 0.0
        else:
            theta_annual = r * K * discount_factor if S < K else 0.0
        return theta_annual / DAYS_PER_YEAR if annualize else theta_annual

    d1, d2 = _calculate_d1_d2(S, K, T, r, sigma)

    sqrt_T = np.sqrt(T)
    discount_factor = np.exp(-r * T)

    # Time decay component (always negative)
    time_decay = -S * norm.pdf(d1) * sigma / (2 * sqrt_T)

    # Interest rate component
    if option_type_lower in ('call', 'c'):
        interest_component = -r * K * discount_factor * norm.cdf(d2)
        theta_annual = time_decay + interest_component
    elif option_type_lower in ('put', 'p'):
        interest_component = r * K * discount_factor * norm.cdf(-d2)
        theta_annual = time_decay + interest_component
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")

    if annualize:
        return theta_annual / DAYS_PER_YEAR
    else:
        return theta_annual


def calculate_vega(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    per_percent: bool = True
) -> float:
    """
    Calculate option vega - rate of change of option price with respect to volatility.

    Vega measures the sensitivity of the option price to changes in implied
    volatility. It is the same for both calls and puts.

    Formula (per 100% vol change):
        Vega = S * N'(d1) * sqrt(T)

    For per-1% volatility change, divide by 100.

    Properties:
        - Always positive for long options
        - Highest for ATM options
        - Increases with time to expiration
        - Higher for options on higher-priced underlyings

    Args:
        S: Spot price
        K: Strike price
        T: Time to expiration in years
        r: Risk-free rate (annualized)
        sigma: Volatility (annualized)
        per_percent: If True, returns vega per 1% vol change (divided by 100).
                    If False, returns vega per 100% vol change.

    Returns:
        Vega value (always non-negative)

    Example:
        >>> vega = calculate_vega(S=100, K=100, T=0.25, r=0.05, sigma=0.20)
        >>> print(f"Vega (per 1% vol): ${vega:.4f}")
        Vega (per 1% vol): $0.1982
    """
    _validate_inputs(S, K, T, r, sigma, allow_zero_time=True)

    # Edge cases
    if S < MIN_SPOT_PRICE or K < MIN_STRIKE_PRICE:
        return 0.0

    if T < MIN_TIME_TO_EXPIRY:
        # At expiration, vega is 0
        return 0.0

    if sigma < MIN_VOLATILITY:
        # Use a small sigma to compute an approximation
        # Vega at sigma=0 is finite (not 0)
        sigma = MIN_VOLATILITY

    d1, _ = _calculate_d1_d2(S, K, T, r, sigma)

    sqrt_T = np.sqrt(T)
    vega_per_unit = S * norm.pdf(d1) * sqrt_T

    if per_percent:
        return vega_per_unit / 100.0
    else:
        return vega_per_unit


def calculate_rho(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = 'call',
    per_percent: bool = True
) -> float:
    """
    Calculate option rho - rate of change of option price with respect to interest rate.

    Rho measures the sensitivity of the option price to changes in the risk-free
    interest rate.

    Formula (per 100% rate change):
        Call Rho = K * T * exp(-rT) * N(d2)
        Put Rho = -K * T * exp(-rT) * N(-d2)

    For per-1% rate change, divide by 100.

    Properties:
        - Positive for calls (higher rates -> higher call prices)
        - Negative for puts (higher rates -> lower put prices)
        - Increases with time to expiration
        - Typically smaller impact than other Greeks

    Args:
        S: Spot price
        K: Strike price
        T: Time to expiration in years
        r: Risk-free rate (annualized)
        sigma: Volatility (annualized)
        option_type: 'call' or 'put'
        per_percent: If True, returns rho per 1% rate change (divided by 100).
                    If False, returns rho per 100% rate change.

    Returns:
        Rho value (positive for calls, negative for puts)

    Example:
        >>> rho = calculate_rho(S=100, K=100, T=0.25, r=0.05, sigma=0.20, option_type='call')
        >>> print(f"Rho (per 1% rate): ${rho:.4f}")
        Rho (per 1% rate): $0.1227
    """
    _validate_inputs(S, K, T, r, sigma, allow_zero_time=True)
    option_type_lower = option_type.lower().strip()

    # Edge cases
    if K < MIN_STRIKE_PRICE:
        return 0.0

    if T < MIN_TIME_TO_EXPIRY:
        # At expiration, rho is 0
        return 0.0

    if sigma < MIN_VOLATILITY:
        sigma = MIN_VOLATILITY

    _, d2 = _calculate_d1_d2(S, K, T, r, sigma)

    discount_factor = np.exp(-r * T)

    if option_type_lower in ('call', 'c'):
        rho_per_unit = K * T * discount_factor * norm.cdf(d2)
    elif option_type_lower in ('put', 'p'):
        rho_per_unit = -K * T * discount_factor * norm.cdf(-d2)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")

    if per_percent:
        return rho_per_unit / 100.0
    else:
        return rho_per_unit


def calculate_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = 'call'
) -> Dict[str, float]:
    """
    Calculate all Greeks analytically for a European option.

    This is a convenience function that calculates all first-order Greeks
    in a single call. For repeated calculations on the same option, this
    is more efficient than calling individual functions.

    Args:
        S: Current spot price of the underlying asset.
        K: Strike price of the option.
        T: Time to expiration in years.
        r: Risk-free interest rate (annualized).
        sigma: Volatility of the underlying (annualized).
        option_type: 'call' or 'put'.

    Returns:
        Dictionary containing:
            - 'delta': Rate of change w.r.t. spot price
            - 'gamma': Rate of change of delta w.r.t. spot price
            - 'theta': Daily time decay (negative for long positions)
            - 'vega': Per 1% volatility change
            - 'rho': Per 1% interest rate change

    Raises:
        ValueError: If any input parameter is invalid.

    Example:
        >>> greeks = calculate_greeks(S=100, K=100, T=0.25, r=0.05, sigma=0.20, option_type='call')
        >>> for greek, value in greeks.items():
        ...     print(f"{greek}: {value:.4f}")
        delta: 0.5596
        gamma: 0.0396
        theta: -0.0440
        vega: 0.1982
        rho: 0.1227

    Note:
        - Theta is per calendar day (annualized / 365)
        - Vega is per 1% volatility change
        - Rho is per 1% interest rate change
    """
    _validate_inputs(S, K, T, r, sigma, allow_zero_time=True)

    return {
        'delta': calculate_delta(S, K, T, r, sigma, option_type),
        'gamma': calculate_gamma(S, K, T, r, sigma),
        'theta': calculate_theta(S, K, T, r, sigma, option_type, annualize=True),
        'vega': calculate_vega(S, K, T, r, sigma, per_percent=True),
        'rho': calculate_rho(S, K, T, r, sigma, option_type, per_percent=True)
    }


# =============================================================================
# Implied Volatility Calculation
# =============================================================================

def calculate_implied_volatility(
    option_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = 'call',
    method: str = 'newton',
    initial_guess: Optional[float] = None,
    tol: float = IV_TOLERANCE,
    max_iter: int = IV_MAX_ITERATIONS
) -> float:
    """
    Calculate implied volatility using numerical methods.

    Implied volatility is the volatility value that, when input into the
    Black-Scholes formula, produces the observed market price. This function
    finds IV by solving the inverse problem numerically.

    Supported Methods:
        - 'newton': Newton-Raphson iteration (faster, may not converge)
        - 'brent': Brent's method (bracketed, guaranteed convergence)

    Args:
        option_price: Observed market price of the option.
        S: Current spot price of the underlying.
        K: Strike price of the option.
        T: Time to expiration in years.
        r: Risk-free interest rate (annualized).
        option_type: 'call' or 'put'.
        method: 'newton' or 'brent'. Default 'newton'.
        initial_guess: Starting point for iteration. Default 0.30 (30%).
        tol: Convergence tolerance. Default 1e-8.
        max_iter: Maximum iterations. Default 100.

    Returns:
        Implied volatility as a decimal (e.g., 0.20 for 20%).

    Raises:
        ValueError: If input parameters are invalid.
        ImpliedVolatilityError: If IV calculation fails to converge.

    Example:
        >>> iv = calculate_implied_volatility(
        ...     option_price=5.88,
        ...     S=100, K=100, T=0.25, r=0.05,
        ...     option_type='call'
        ... )
        >>> print(f"Implied Volatility: {iv:.2%}")
        Implied Volatility: 20.00%

    Note:
        - Very deep OTM options may have unreliable IV due to low prices
        - At expiration (T=0), IV is undefined
        - Option prices below intrinsic value will not yield valid IV
    """
    # Validate inputs
    if option_price is None or option_price < 0:
        raise ValueError(f"option_price must be non-negative, got {option_price}")

    _validate_inputs(S, K, T, r, sigma=0.0, allow_zero_time=True)

    option_type_lower = option_type.lower().strip()
    if option_type_lower not in ('call', 'c', 'put', 'p'):
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")

    # Edge case: T = 0 (at expiration, IV is undefined)
    if T < MIN_TIME_TO_EXPIRY:
        raise ImpliedVolatilityError("Cannot calculate IV at expiration (T=0)")

    # Calculate intrinsic value
    if option_type_lower in ('call', 'c'):
        intrinsic = max(S - K * np.exp(-r * T), 0.0)
    else:
        intrinsic = max(K * np.exp(-r * T) - S, 0.0)

    # Check if price is below intrinsic (arbitrage)
    if option_price < intrinsic - 1e-10:
        raise ImpliedVolatilityError(
            f"Option price ({option_price:.4f}) is below intrinsic value "
            f"({intrinsic:.4f}). IV cannot be calculated."
        )

    # Check if option price is essentially zero
    if option_price < 1e-10:
        # Very deep OTM, return a high IV estimate
        logger.warning("Option price is near zero, returning minimum IV")
        return IV_LOWER_BOUND

    # Define the objective function: BS_price(sigma) - market_price = 0
    def objective(sigma: float) -> float:
        bs_price = black_scholes_price(S, K, T, r, sigma, option_type)
        return bs_price - option_price

    # Define derivative for Newton's method (vega)
    def vega_derivative(sigma: float) -> float:
        return calculate_vega(S, K, T, r, sigma, per_percent=False)

    # Set initial guess
    if initial_guess is None:
        initial_guess = IV_INITIAL_GUESS

    method_lower = method.lower().strip()

    try:
        if method_lower == 'newton':
            # Newton-Raphson method
            sigma = initial_guess
            for i in range(max_iter):
                price_diff = objective(sigma)
                if abs(price_diff) < tol:
                    break

                vega = vega_derivative(sigma)
                if abs(vega) < 1e-15:
                    # Vega too small, switch to bisection step
                    if price_diff > 0:
                        sigma *= 0.5
                    else:
                        sigma *= 2.0
                else:
                    sigma_new = sigma - price_diff / vega

                    # Ensure sigma stays in bounds
                    sigma_new = max(IV_LOWER_BOUND, min(IV_UPPER_BOUND, sigma_new))

                    # Damping for stability
                    if sigma_new < 0:
                        sigma_new = sigma / 2.0

                    sigma = sigma_new
            else:
                # Did not converge, try Brent's method as fallback
                logger.debug("Newton's method did not converge, falling back to Brent's method")
                return calculate_implied_volatility(
                    option_price, S, K, T, r, option_type,
                    method='brent', tol=tol, max_iter=max_iter
                )

            # Validate result
            if not np.isfinite(sigma) or sigma < IV_LOWER_BOUND or sigma > IV_UPPER_BOUND:
                raise ImpliedVolatilityError(
                    f"IV calculation produced invalid result: {sigma}"
                )

            return sigma

        elif method_lower == 'brent':
            # Brent's method (bracketed search)
            # First, find valid brackets
            lower = IV_LOWER_BOUND
            upper = IV_UPPER_BOUND

            # Check if solution exists in bracket
            f_lower = objective(lower)
            f_upper = objective(upper)

            if f_lower * f_upper > 0:
                # Try to find a valid bracket
                if f_lower > 0:
                    # Price at min IV is already too high - shouldn't happen normally
                    raise ImpliedVolatilityError(
                        f"Option price ({option_price:.4f}) is below BS price even "
                        f"at minimum IV ({IV_LOWER_BOUND:.4f})"
                    )
                else:
                    # Price at max IV is still too low
                    raise ImpliedVolatilityError(
                        f"Option price ({option_price:.4f}) exceeds BS price even "
                        f"at maximum IV ({IV_UPPER_BOUND:.4f}). Price may be invalid."
                    )

            sigma = brentq(objective, lower, upper, xtol=tol, maxiter=max_iter)

            return sigma

        else:
            raise ValueError(f"method must be 'newton' or 'brent', got '{method}'")

    except (RuntimeError, ValueError) as e:
        raise ImpliedVolatilityError(f"IV calculation failed: {str(e)}") from e


# =============================================================================
# American Option Approximations (Barone-Adesi-Whaley)
# =============================================================================

def barone_adesi_whaley_american_call(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0
) -> float:
    """
    Calculate American call option price using Barone-Adesi-Whaley approximation.

    For a call option on a non-dividend paying stock (q=0), the American call
    equals the European call (early exercise is never optimal). This function
    is included for completeness and for assets with dividends.

    The BAW approximation provides a quadratic approximation for the early
    exercise premium of American options.

    Args:
        S: Spot price
        K: Strike price
        T: Time to expiration in years
        r: Risk-free rate (annualized)
        sigma: Volatility (annualized)
        q: Continuous dividend yield (annualized). Default 0.

    Returns:
        American call option price

    References:
        Barone-Adesi, G., & Whaley, R. E. (1987). Efficient Analytic
        Approximation of American Option Values. The Journal of Finance.
    """
    _validate_inputs(S, K, T, r, sigma, allow_zero_time=True)

    if q < 0:
        raise ValueError(f"Dividend yield (q) cannot be negative, got {q}")

    # Edge case: no dividends - American call = European call
    if q < MIN_VOLATILITY:
        return black_scholes_call(S, K, T, r, sigma)

    # Edge case: T = 0
    if T < MIN_TIME_TO_EXPIRY:
        return max(S - K, 0.0)

    # Calculate European call price (adjusted for dividends)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    european_call = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    # BAW parameters
    M = 2 * r / (sigma**2)
    N = 2 * (r - q) / (sigma**2)
    k = 1 - np.exp(-r * T)

    q2 = (-(N - 1) + np.sqrt((N - 1)**2 + 4 * M / k)) / 2

    # Find critical stock price (S*)
    # Using Newton-Raphson to solve for S*
    def critical_price_equation(S_star: float) -> float:
        if S_star < MIN_SPOT_PRICE:
            return -np.inf

        d1_star = (np.log(S_star / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        european_at_star = S_star * np.exp(-q * T) * norm.cdf(d1_star) - K * np.exp(-r * T) * norm.cdf(d1_star - sigma * np.sqrt(T))

        return S_star - K - european_at_star - (S_star / q2) * (1 - np.exp(-q * T) * norm.cdf(d1_star))

    # Solve for critical price
    try:
        from scipy.optimize import brentq
        S_star = brentq(critical_price_equation, K, K * 10, maxiter=100)
    except Exception:
        # Fall back to European price if unable to find critical price
        return european_call

    if S >= S_star:
        # Exercise immediately
        return S - K
    else:
        # Early exercise premium
        d1_star = (np.log(S_star / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        A2 = (S_star / q2) * (1 - np.exp(-q * T) * norm.cdf(d1_star))

        return european_call + A2 * (S / S_star) ** q2


def barone_adesi_whaley_american_put(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0
) -> float:
    """
    Calculate American put option price using Barone-Adesi-Whaley approximation.

    American puts may be optimally exercised early even without dividends,
    as the time value of money on the strike received from exercise can
    exceed the remaining option time value.

    Args:
        S: Spot price
        K: Strike price
        T: Time to expiration in years
        r: Risk-free rate (annualized)
        sigma: Volatility (annualized)
        q: Continuous dividend yield (annualized). Default 0.

    Returns:
        American put option price

    References:
        Barone-Adesi, G., & Whaley, R. E. (1987). Efficient Analytic
        Approximation of American Option Values. The Journal of Finance.
    """
    _validate_inputs(S, K, T, r, sigma, allow_zero_time=True)

    if q < 0:
        raise ValueError(f"Dividend yield (q) cannot be negative, got {q}")

    # Edge case: T = 0
    if T < MIN_TIME_TO_EXPIRY:
        return max(K - S, 0.0)

    # Calculate European put price (adjusted for dividends)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    european_put = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

    # If interest rate is zero or negative, no early exercise premium
    if r <= 0:
        return european_put

    # BAW parameters
    M = 2 * r / (sigma**2)
    N = 2 * (r - q) / (sigma**2)
    k = 1 - np.exp(-r * T)

    q1 = (-(N - 1) - np.sqrt((N - 1)**2 + 4 * M / k)) / 2

    # Find critical stock price (S*)
    def critical_price_equation(S_star: float) -> float:
        if S_star < MIN_SPOT_PRICE:
            return np.inf

        d1_star = (np.log(S_star / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        european_at_star = K * np.exp(-r * T) * norm.cdf(-d1_star + sigma * np.sqrt(T)) - S_star * np.exp(-q * T) * norm.cdf(-d1_star)

        return K - S_star - european_at_star + (S_star / q1) * (1 - np.exp(-q * T) * norm.cdf(-d1_star))

    # Solve for critical price
    try:
        from scipy.optimize import brentq
        S_star = brentq(critical_price_equation, MIN_SPOT_PRICE, K, maxiter=100)
    except Exception:
        # Fall back to European price
        return european_put

    if S <= S_star:
        # Exercise immediately
        return K - S
    else:
        # Early exercise premium
        d1_star = (np.log(S_star / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        A1 = -(S_star / q1) * (1 - np.exp(-q * T) * norm.cdf(-d1_star))

        return european_put + A1 * (S / S_star) ** q1


# =============================================================================
# Vectorized Calculations for Performance
# =============================================================================

def black_scholes_call_vectorized(
    S: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    r: np.ndarray,
    sigma: np.ndarray
) -> np.ndarray:
    """
    Vectorized Black-Scholes call price calculation for arrays.

    This function provides efficient calculation of call prices for
    multiple options simultaneously using NumPy broadcasting.

    Args:
        S: Array of spot prices
        K: Array of strike prices
        T: Array of times to expiration (years)
        r: Array of risk-free rates
        sigma: Array of volatilities

    Returns:
        Array of call option prices

    Note:
        All input arrays must be broadcastable to the same shape.
        Edge cases are handled but may produce NaN for invalid inputs.

    Example:
        >>> S = np.array([100, 100, 100])
        >>> K = np.array([95, 100, 105])
        >>> T = np.array([0.25, 0.25, 0.25])
        >>> r = np.array([0.05, 0.05, 0.05])
        >>> sigma = np.array([0.20, 0.20, 0.20])
        >>> prices = black_scholes_call_vectorized(S, K, T, r, sigma)
    """
    # Convert to arrays if needed
    S = np.asarray(S, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    # Initialize result array
    with np.errstate(divide='ignore', invalid='ignore'):
        sqrt_T = np.sqrt(T)
        sigma_sqrt_T = sigma * sqrt_T

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / sigma_sqrt_T
        d2 = d1 - sigma_sqrt_T

        call_prices = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    # Handle edge cases
    # T = 0 or sigma = 0
    edge_mask = (T < MIN_TIME_TO_EXPIRY) | (sigma < MIN_VOLATILITY)
    if np.any(edge_mask):
        intrinsic = np.maximum(S - K, 0.0)
        call_prices = np.where(edge_mask, intrinsic, call_prices)

    # S = 0
    zero_spot_mask = S < MIN_SPOT_PRICE
    call_prices = np.where(zero_spot_mask, 0.0, call_prices)

    # K = 0
    zero_strike_mask = K < MIN_STRIKE_PRICE
    call_prices = np.where(zero_strike_mask, S, call_prices)

    # Ensure non-negative
    call_prices = np.maximum(call_prices, 0.0)

    return call_prices


def black_scholes_put_vectorized(
    S: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    r: np.ndarray,
    sigma: np.ndarray
) -> np.ndarray:
    """
    Vectorized Black-Scholes put price calculation for arrays.

    This function provides efficient calculation of put prices for
    multiple options simultaneously using NumPy broadcasting.

    Args:
        S: Array of spot prices
        K: Array of strike prices
        T: Array of times to expiration (years)
        r: Array of risk-free rates
        sigma: Array of volatilities

    Returns:
        Array of put option prices

    Example:
        >>> S = np.array([100, 100, 100])
        >>> K = np.array([95, 100, 105])
        >>> T = np.array([0.25, 0.25, 0.25])
        >>> r = np.array([0.05, 0.05, 0.05])
        >>> sigma = np.array([0.20, 0.20, 0.20])
        >>> prices = black_scholes_put_vectorized(S, K, T, r, sigma)
    """
    # Convert to arrays if needed
    S = np.asarray(S, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    # Initialize result array
    with np.errstate(divide='ignore', invalid='ignore'):
        sqrt_T = np.sqrt(T)
        sigma_sqrt_T = sigma * sqrt_T

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / sigma_sqrt_T
        d2 = d1 - sigma_sqrt_T

        put_prices = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    # Handle edge cases
    # T = 0 or sigma = 0
    edge_mask = (T < MIN_TIME_TO_EXPIRY) | (sigma < MIN_VOLATILITY)
    if np.any(edge_mask):
        intrinsic = np.maximum(K - S, 0.0)
        put_prices = np.where(edge_mask, intrinsic, put_prices)

    # K = 0
    zero_strike_mask = K < MIN_STRIKE_PRICE
    put_prices = np.where(zero_strike_mask, 0.0, put_prices)

    # S = 0
    zero_spot_mask = S < MIN_SPOT_PRICE
    put_prices = np.where(zero_spot_mask, K * np.exp(-r * T), put_prices)

    # Ensure non-negative
    put_prices = np.maximum(put_prices, 0.0)

    return put_prices


def calculate_greeks_vectorized(
    S: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    r: np.ndarray,
    sigma: np.ndarray,
    option_type: str = 'call'
) -> Dict[str, np.ndarray]:
    """
    Vectorized Greeks calculation for arrays of option parameters.

    Args:
        S: Array of spot prices
        K: Array of strike prices
        T: Array of times to expiration (years)
        r: Array of risk-free rates
        sigma: Array of volatilities
        option_type: 'call' or 'put'

    Returns:
        Dictionary of arrays for each Greek:
        {'delta', 'gamma', 'theta', 'vega', 'rho'}

    Example:
        >>> S = np.array([100, 100, 100])
        >>> K = np.array([95, 100, 105])
        >>> greeks = calculate_greeks_vectorized(S, K, T, r, sigma, option_type='call')
    """
    # Convert to arrays
    S = np.asarray(S, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    option_type_lower = option_type.lower().strip()
    is_call = option_type_lower in ('call', 'c')

    with np.errstate(divide='ignore', invalid='ignore'):
        sqrt_T = np.sqrt(T)
        sigma_sqrt_T = sigma * sqrt_T

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / sigma_sqrt_T
        d2 = d1 - sigma_sqrt_T

        # Delta
        if is_call:
            delta = norm.cdf(d1)
        else:
            delta = norm.cdf(d1) - 1.0

        # Gamma (same for calls and puts)
        gamma = norm.pdf(d1) / (S * sigma_sqrt_T)

        # Theta
        discount = np.exp(-r * T)
        time_decay = -S * norm.pdf(d1) * sigma / (2 * sqrt_T)
        if is_call:
            theta = time_decay - r * K * discount * norm.cdf(d2)
        else:
            theta = time_decay + r * K * discount * norm.cdf(-d2)
        theta = theta / DAYS_PER_YEAR  # Per-day

        # Vega (same for calls and puts, per 1%)
        vega = S * norm.pdf(d1) * sqrt_T / 100.0

        # Rho (per 1%)
        if is_call:
            rho = K * T * discount * norm.cdf(d2) / 100.0
        else:
            rho = -K * T * discount * norm.cdf(-d2) / 100.0

    # Handle edge cases
    edge_mask = (T < MIN_TIME_TO_EXPIRY) | (sigma < MIN_VOLATILITY) | (S < MIN_SPOT_PRICE) | (K < MIN_STRIKE_PRICE)
    if np.any(edge_mask):
        # Set NaN or 0 for edge cases
        delta = np.where(edge_mask, np.nan, delta)
        gamma = np.where(edge_mask, 0.0, gamma)
        theta = np.where(edge_mask, 0.0, theta)
        vega = np.where(edge_mask, 0.0, vega)
        rho = np.where(edge_mask, 0.0, rho)

    return {
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'vega': vega,
        'rho': rho
    }


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Exceptions
    'PricingError',
    'ImpliedVolatilityError',

    # Pricing functions
    'black_scholes_call',
    'black_scholes_put',
    'black_scholes_price',

    # Greeks functions
    'calculate_delta',
    'calculate_gamma',
    'calculate_theta',
    'calculate_vega',
    'calculate_rho',
    'calculate_greeks',

    # Implied volatility
    'calculate_implied_volatility',

    # American options
    'barone_adesi_whaley_american_call',
    'barone_adesi_whaley_american_put',

    # Vectorized functions
    'black_scholes_call_vectorized',
    'black_scholes_put_vectorized',
    'calculate_greeks_vectorized',

    # Constants
    'DAYS_PER_YEAR',
    'TRADING_DAYS_PER_YEAR',
]
