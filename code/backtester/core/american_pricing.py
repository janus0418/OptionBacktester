"""
American Option Pricing using Binomial Tree (Cox-Ross-Rubinstein Model)

This module provides accurate pricing for American-style options that can be
exercised at any time before expiration. It implements the Cox-Ross-Rubinstein
(CRR) binomial tree model, which is the industry standard for American option
valuation.

Key Features:
    - Early exercise handling for puts and dividend-paying stocks
    - Discrete dividend support with ex-dividend adjustments
    - Greeks calculation via finite differences
    - Convergence to Black-Scholes as steps increase
    - Efficient numpy-based tree construction

Mathematical Framework:
    The binomial tree discretizes price movement into up/down steps:
        u = exp(sigma * sqrt(dt))   - up factor
        d = 1/u                      - down factor
        p = (exp(r*dt) - d) / (u - d) - risk-neutral probability

    At each node, the option value is:
        V = max(exercise_value, continuation_value)

    where continuation_value = exp(-r*dt) * [p*V_up + (1-p)*V_down]

Usage:
    >>> from backtester.core.american_pricing import BinomialPricer, AmericanOptionPricer
    >>>
    >>> # Direct binomial pricing
    >>> pricer = BinomialPricer(steps=200)
    >>> put_price = pricer.price(S=100, K=100, T=0.25, r=0.05, sigma=0.20, option_type='put')
    >>>
    >>> # With dividends
    >>> divs = [(0.1, 2.0), (0.2, 2.0)]  # (time, amount) tuples
    >>> call_price = pricer.price(S=100, K=100, T=0.25, r=0.05, sigma=0.20,
    ...                           option_type='call', dividends=divs)
    >>>
    >>> # Using unified interface
    >>> american = AmericanOptionPricer()
    >>> price = american.price(S=100, K=100, T=0.25, r=0.05, sigma=0.20, option_type='put')
    >>> greeks = american.calculate_greeks(S=100, K=100, T=0.25, r=0.05, sigma=0.20, option_type='put')

References:
    - Cox, J.C., Ross, S.A., & Rubinstein, M. (1979). "Option pricing: A simplified approach"
    - Hull, J.C. (2018). "Options, Futures, and Other Derivatives"
"""

import numpy as np
from typing import Optional, Dict, List, Tuple, Literal
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Default number of tree steps (tradeoff: accuracy vs speed)
DEFAULT_TREE_STEPS = 200

# Minimum values for numerical stability
MIN_TIME_TO_EXPIRY = 1e-10
MIN_SPOT_PRICE = 1e-10
MIN_STRIKE_PRICE = 1e-10
MIN_VOLATILITY = 1e-10

# Greeks finite difference bump sizes
DELTA_BUMP_PCT = 0.01  # 1% spot price bump
VEGA_BUMP = 0.01  # 1% volatility bump
RHO_BUMP = 0.01  # 1% rate bump
THETA_DAYS = 1  # 1 day for theta


# =============================================================================
# Exceptions
# =============================================================================


class AmericanPricingError(Exception):
    """Base exception for American pricing errors"""

    pass


class TreeConstructionError(AmericanPricingError):
    """Exception raised when tree construction fails"""

    pass


# =============================================================================
# Binomial Pricer Class
# =============================================================================


class BinomialPricer:
    """
    Binomial tree for American option pricing using Cox-Ross-Rubinstein model.

    This pricer handles:
        - Early exercise for American puts and dividend-paying calls
        - Discrete dividend adjustments
        - Greeks calculation via finite differences
        - Convergence to Black-Scholes in the limit

    Attributes:
        steps: Number of time steps in the tree

    Example:
        >>> pricer = BinomialPricer(steps=200)
        >>> price = pricer.price(S=100, K=100, T=0.25, r=0.05, sigma=0.20, option_type='put')
        >>> greeks = pricer.calculate_greeks(S=100, K=100, T=0.25, r=0.05, sigma=0.20, option_type='put')
    """

    def __init__(self, steps: int = DEFAULT_TREE_STEPS):
        """
        Initialize binomial pricer.

        Args:
            steps: Number of time steps in tree.
                   More steps = higher accuracy but slower computation.
                   Recommended: 100-500 for production, 50 for quick estimates.

        Raises:
            ValueError: If steps < 1
        """
        if steps < 1:
            raise ValueError(f"steps must be >= 1, got {steps}")

        self.steps = steps

    def price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str,
        dividends: Optional[List[Tuple[float, float]]] = None,
        is_american: bool = True,
    ) -> float:
        """
        Price an option using the binomial tree.

        Args:
            S: Current spot price of the underlying
            K: Strike price
            T: Time to expiration in years
            r: Risk-free interest rate (annualized, e.g., 0.05 for 5%)
            sigma: Volatility (annualized, e.g., 0.20 for 20%)
            option_type: 'call' or 'put'
            dividends: List of (time, amount) tuples for discrete dividends.
                       Time is in years from now, amount is cash dividend.
            is_american: If True, allow early exercise. If False, price European.

        Returns:
            Option price

        Raises:
            ValueError: If inputs are invalid

        Example:
            >>> pricer = BinomialPricer(steps=200)
            >>> # American put
            >>> put_price = pricer.price(S=100, K=100, T=0.25, r=0.05, sigma=0.20, option_type='put')
            >>> # Call with dividends
            >>> divs = [(0.1, 2.0)]  # $2 dividend in 0.1 years
            >>> call_price = pricer.price(S=100, K=100, T=0.25, r=0.05, sigma=0.20,
            ...                           option_type='call', dividends=divs)
        """
        # Validate inputs
        self._validate_inputs(S, K, T, r, sigma, option_type)

        # Handle edge cases
        if T <= MIN_TIME_TO_EXPIRY:
            return self._intrinsic_value(S, K, option_type)

        if sigma <= MIN_VOLATILITY:
            # Zero vol = deterministic, just discount intrinsic
            forward = S * np.exp(r * T)
            return self._intrinsic_value(forward, K, option_type) * np.exp(-r * T)

        # Normalize option type
        option_type_lower = option_type.lower().strip()
        is_call = option_type_lower in ("call", "c")

        # Adjust spot for dividends (present value method)
        S_adj = self._adjust_spot_for_dividends(S, r, T, dividends)

        # Time step
        dt = T / self.steps

        # CRR up/down factors
        u = np.exp(sigma * np.sqrt(dt))
        d = 1.0 / u

        # Risk-neutral probability
        exp_r_dt = np.exp(r * dt)
        p = (exp_r_dt - d) / (u - d)

        # Validate probability
        if p <= 0 or p >= 1:
            raise TreeConstructionError(
                f"Invalid risk-neutral probability: {p}. "
                f"This may occur with extreme inputs (r={r}, sigma={sigma}, T={T})"
            )

        # Discount factor per step
        disc = np.exp(-r * dt)

        # Build tree and compute price
        return self._build_tree(S_adj, K, u, d, p, disc, is_call, is_american)

    def _validate_inputs(
        self, S: float, K: float, T: float, r: float, sigma: float, option_type: str
    ) -> None:
        """Validate pricing inputs"""
        if S is None or not np.isfinite(S) or S < 0:
            raise ValueError(f"Spot price S must be non-negative and finite, got {S}")

        if K is None or not np.isfinite(K) or K < 0:
            raise ValueError(f"Strike price K must be non-negative and finite, got {K}")

        if T is None or not np.isfinite(T) or T < 0:
            raise ValueError(f"Time T must be non-negative and finite, got {T}")

        if r is None or not np.isfinite(r):
            raise ValueError(f"Risk-free rate r must be finite, got {r}")

        if sigma is None or not np.isfinite(sigma) or sigma < 0:
            raise ValueError(
                f"Volatility sigma must be non-negative and finite, got {sigma}"
            )

        option_type_lower = option_type.lower().strip() if option_type else ""
        if option_type_lower not in ("call", "c", "put", "p"):
            raise ValueError(
                f"option_type must be 'call' or 'put', got '{option_type}'"
            )

    def _adjust_spot_for_dividends(
        self,
        S: float,
        r: float,
        T: float,
        dividends: Optional[List[Tuple[float, float]]],
    ) -> float:
        """
        Adjust spot price for present value of discrete dividends.

        This uses the "escrow method" - subtract PV of dividends from spot.
        """
        if not dividends:
            return S

        pv_dividends = 0.0
        for div_time, div_amount in dividends:
            if 0 < div_time <= T:
                # Present value of this dividend
                pv_dividends += div_amount * np.exp(-r * div_time)

        adjusted_spot = S - pv_dividends

        # Ensure non-negative
        return max(adjusted_spot, MIN_SPOT_PRICE)

    def _build_tree(
        self,
        S: float,
        K: float,
        u: float,
        d: float,
        p: float,
        disc: float,
        is_call: bool,
        is_american: bool,
    ) -> float:
        """
        Build binomial tree and compute option price by backward induction.

        Uses efficient numpy arrays instead of explicit tree structure.
        """
        n = self.steps

        # Initialize asset prices at maturity (step n)
        # S * u^(n-i) * d^i for i = 0, 1, ..., n
        ST = S * (u ** np.arange(n, -1, -1)) * (d ** np.arange(0, n + 1))

        # Initialize option values at maturity
        if is_call:
            V = np.maximum(ST - K, 0.0)
        else:
            V = np.maximum(K - ST, 0.0)

        # Step backward through tree
        for j in range(n - 1, -1, -1):
            # Asset prices at this time step
            S_j = S * (u ** np.arange(j, -1, -1)) * (d ** np.arange(0, j + 1))

            # Continuation value (discounted expected value)
            V = disc * (p * V[:-1] + (1 - p) * V[1:])

            # For American options, check early exercise
            if is_american:
                if is_call:
                    exercise = np.maximum(S_j - K, 0.0)
                else:
                    exercise = np.maximum(K - S_j, 0.0)
                V = np.maximum(V, exercise)

        return float(V[0])

    @staticmethod
    def _intrinsic_value(S: float, K: float, option_type: str) -> float:
        """Calculate intrinsic value of option"""
        option_type_lower = option_type.lower().strip() if option_type else ""
        if option_type_lower in ("call", "c"):
            return max(0.0, S - K)
        else:
            return max(0.0, K - S)

    def calculate_greeks(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str,
        dividends: Optional[List[Tuple[float, float]]] = None,
    ) -> Dict[str, float]:
        """
        Calculate Greeks via finite differences.

        Uses central differences for better accuracy where possible.

        Args:
            S: Spot price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'
            dividends: Optional discrete dividends

        Returns:
            Dictionary with keys:
                - 'delta': dV/dS (sensitivity to spot)
                - 'gamma': d²V/dS² (convexity)
                - 'theta': dV/dt per day (time decay)
                - 'vega': dV/dsigma per 1% vol change
                - 'rho': dV/dr per 1% rate change

        Example:
            >>> pricer = BinomialPricer(steps=200)
            >>> greeks = pricer.calculate_greeks(S=100, K=100, T=0.25, r=0.05,
            ...                                  sigma=0.20, option_type='put')
            >>> print(f"Delta: {greeks['delta']:.4f}")
        """
        # Base price
        V = self.price(S, K, T, r, sigma, option_type, dividends)

        # Delta: dV/dS using central difference
        dS = S * DELTA_BUMP_PCT
        if dS < MIN_SPOT_PRICE:
            dS = MIN_SPOT_PRICE

        V_up = self.price(S + dS, K, T, r, sigma, option_type, dividends)
        V_down = self.price(
            max(S - dS, MIN_SPOT_PRICE), K, T, r, sigma, option_type, dividends
        )
        delta = (V_up - V_down) / (2 * dS)

        # Gamma: d²V/dS² using central difference
        gamma = (V_up - 2 * V + V_down) / (dS**2)

        # Theta: dV/dt (expressed as per-day decay)
        dt = THETA_DAYS / 365.0
        if T > dt:
            V_tomorrow = self.price(S, K, T - dt, r, sigma, option_type, dividends)
            theta = V_tomorrow - V  # Already per day
        else:
            # Near expiration
            theta = -V  # Will lose all value

        # Vega: dV/dsigma (per 1% volatility change)
        dsigma = VEGA_BUMP
        V_vol_up = self.price(S, K, T, r, sigma + dsigma, option_type, dividends)
        vega = (V_vol_up - V) / 100.0  # Per 1% change

        # Rho: dV/dr (per 1% rate change)
        dr = RHO_BUMP
        V_rate_up = self.price(S, K, T, r + dr, sigma, option_type, dividends)
        rho = (V_rate_up - V) / 100.0  # Per 1% change

        return {
            "delta": delta,
            "gamma": gamma,
            "theta": theta,
            "vega": vega,
            "rho": rho,
        }

    def calculate_early_exercise_boundary(
        self,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str,
        num_points: int = 50,
    ) -> List[Tuple[float, float]]:
        """
        Calculate the early exercise boundary over time.

        For a put, this is the spot price below which early exercise is optimal.
        For a call (with dividends), this is the spot price above which exercise is optimal.

        Args:
            K: Strike price
            T: Time to expiration
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'
            num_points: Number of time points to compute

        Returns:
            List of (time_to_expiration, critical_spot_price) tuples
        """
        is_call = option_type.lower().strip() in ("call", "c")

        # For non-dividend calls, there's no early exercise boundary
        if is_call:
            logger.warning(
                "Early exercise boundary for calls without dividends is infinite"
            )
            return [(t, np.inf) for t in np.linspace(0, T, num_points)]

        boundary = []
        times = np.linspace(0.01, T, num_points)

        for t in times:
            S_low = 0.01
            S_high = K * 2
            S_mid = (S_low + S_high) / 2

            for _ in range(50):
                S_mid = (S_low + S_high) / 2
                V = self.price(S_mid, K, t, r, sigma, option_type)
                intrinsic = self._intrinsic_value(S_mid, K, option_type)

                if abs(V - intrinsic) < 1e-6:
                    break
                elif V > intrinsic:
                    S_high = S_mid
                else:
                    S_low = S_mid

            boundary.append((t, S_mid))

        return boundary


# =============================================================================
# Unified American Option Pricer
# =============================================================================


class AmericanOptionPricer:
    """
    Unified interface for American option pricing.

    Automatically selects the optimal pricing method:
        - Black-Scholes for calls without dividends (no early exercise premium)
        - Binomial tree for puts and dividend-paying calls

    This provides both accuracy and computational efficiency.

    Attributes:
        use_binomial_always: If True, always use binomial tree
        binomial_pricer: Underlying BinomialPricer instance

    Example:
        >>> pricer = AmericanOptionPricer()
        >>>
        >>> # American put (uses binomial)
        >>> put_price = pricer.price(S=100, K=100, T=0.25, r=0.05, sigma=0.20, option_type='put')
        >>>
        >>> # American call no dividends (uses Black-Scholes)
        >>> call_price = pricer.price(S=100, K=100, T=0.25, r=0.05, sigma=0.20, option_type='call')
        >>>
        >>> # Call with dividends (uses binomial)
        >>> call_with_div = pricer.price(S=100, K=100, T=0.25, r=0.05, sigma=0.20,
        ...                              option_type='call', dividends=[(0.1, 2.0)])
    """

    def __init__(
        self, use_binomial_always: bool = False, tree_steps: int = DEFAULT_TREE_STEPS
    ):
        """
        Initialize American option pricer.

        Args:
            use_binomial_always: If True, always use binomial tree even when
                                 Black-Scholes would suffice. Useful for consistency.
            tree_steps: Number of steps in binomial tree (for accuracy/speed tradeoff)
        """
        self.use_binomial_always = use_binomial_always
        self.binomial_pricer = BinomialPricer(steps=tree_steps)

    def price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str,
        dividends: Optional[List[Tuple[float, float]]] = None,
    ) -> float:
        """
        Price American option using optimal method.

        For calls without dividends, uses Black-Scholes (American = European).
        For puts or calls with dividends, uses binomial tree.

        Args:
            S: Spot price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'
            dividends: Optional list of (time, amount) tuples

        Returns:
            Option price
        """
        option_type_lower = option_type.lower().strip()
        is_call = option_type_lower in ("call", "c")

        # For American calls with no dividends, use Black-Scholes
        # (it's never optimal to exercise early, so American = European)
        if is_call and not dividends and not self.use_binomial_always:
            from backtester.core.pricing import black_scholes_call

            return black_scholes_call(S, K, T, r, sigma)

        # Otherwise use binomial tree
        return self.binomial_pricer.price(S, K, T, r, sigma, option_type, dividends)

    def calculate_greeks(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str,
        dividends: Optional[List[Tuple[float, float]]] = None,
    ) -> Dict[str, float]:
        """
        Calculate Greeks for American option.

        Always uses binomial tree for consistency (even for calls),
        as finite differences work well with the tree method.

        Args:
            S: Spot price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'
            dividends: Optional discrete dividends

        Returns:
            Dictionary with delta, gamma, theta, vega, rho
        """
        return self.binomial_pricer.calculate_greeks(
            S, K, T, r, sigma, option_type, dividends
        )

    def early_exercise_premium(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str,
        dividends: Optional[List[Tuple[float, float]]] = None,
    ) -> float:
        """
        Calculate the early exercise premium (American price - European price).

        This quantifies the value of the ability to exercise early.

        Args:
            S: Spot price
            K: Strike price
            T: Time to expiration
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'
            dividends: Optional discrete dividends

        Returns:
            Early exercise premium (>= 0)
        """
        # American price
        american_price = self.binomial_pricer.price(
            S, K, T, r, sigma, option_type, dividends, is_american=True
        )

        # European price (same tree, no early exercise)
        european_price = self.binomial_pricer.price(
            S, K, T, r, sigma, option_type, dividends, is_american=False
        )

        return max(0.0, american_price - european_price)


# =============================================================================
# Convenience Functions
# =============================================================================


def price_american_option(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str,
    dividends: Optional[List[Tuple[float, float]]] = None,
    steps: int = DEFAULT_TREE_STEPS,
) -> float:
    """
    Convenience function to price an American option.

    Args:
        S: Spot price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free rate
        sigma: Volatility
        option_type: 'call' or 'put'
        dividends: Optional list of (time, amount) tuples
        steps: Number of tree steps

    Returns:
        Option price

    Example:
        >>> price = price_american_option(S=100, K=100, T=0.25, r=0.05, sigma=0.20, option_type='put')
    """
    pricer = AmericanOptionPricer(tree_steps=steps)
    return pricer.price(S, K, T, r, sigma, option_type, dividends)


def calculate_american_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str,
    dividends: Optional[List[Tuple[float, float]]] = None,
    steps: int = DEFAULT_TREE_STEPS,
) -> Dict[str, float]:
    """
    Convenience function to calculate Greeks for an American option.

    Args:
        S: Spot price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free rate
        sigma: Volatility
        option_type: 'call' or 'put'
        dividends: Optional discrete dividends
        steps: Number of tree steps

    Returns:
        Dictionary with delta, gamma, theta, vega, rho

    Example:
        >>> greeks = calculate_american_greeks(S=100, K=100, T=0.25, r=0.05, sigma=0.20, option_type='put')
        >>> print(f"Delta: {greeks['delta']:.4f}")
    """
    pricer = AmericanOptionPricer(tree_steps=steps)
    return pricer.calculate_greeks(S, K, T, r, sigma, option_type, dividends)


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Classes
    "BinomialPricer",
    "AmericanOptionPricer",
    # Exceptions
    "AmericanPricingError",
    "TreeConstructionError",
    # Convenience functions
    "price_american_option",
    "calculate_american_greeks",
    # Constants
    "DEFAULT_TREE_STEPS",
]
