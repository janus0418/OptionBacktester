"""
Core Module for Options Backtesting

This module provides the fundamental building blocks for options pricing,
valuation, and position management in the backtesting system.

Components:
    - pricing: Black-Scholes pricing and Greeks calculations
    - option: Option class for position tracking and P&L
    - option_structure: OptionStructure class for multi-leg positions

The core module implements mathematically correct options pricing using
the Black-Scholes model and provides comprehensive Greeks calculations
for risk management.

Key Classes:
    - Option: Represents a single option position (long/short call/put)
    - OptionStructure: Container for multi-leg option structures

Key Functions:
    - black_scholes_call: European call option pricing
    - black_scholes_put: European put option pricing
    - black_scholes_price: General BS pricing (call or put)
    - calculate_greeks: All first-order Greeks
    - calculate_implied_volatility: IV calculation via Newton/Brent

Usage:
    from backtester.core import (
        Option,
        OptionStructure,
        black_scholes_call,
        black_scholes_put,
        calculate_greeks,
        calculate_implied_volatility
    )

    # Price options
    call_price = black_scholes_call(S=100, K=100, T=0.25, r=0.05, sigma=0.20)
    put_price = black_scholes_put(S=100, K=100, T=0.25, r=0.05, sigma=0.20)

    # Calculate Greeks
    greeks = calculate_greeks(S=100, K=100, T=0.25, r=0.05, sigma=0.20, option_type='call')

    # Create option position
    option = Option(
        option_type='call',
        position_type='long',
        underlying='SPY',
        strike=450.0,
        expiration=datetime(2024, 3, 15),
        quantity=10,
        entry_price=5.50,
        entry_date=datetime(2024, 1, 15),
        underlying_price_at_entry=445.0
    )

    # Track P&L
    option.update_price(6.25, datetime(2024, 2, 1))
    print(f"P&L: ${option.calculate_pnl():,.2f}")

    # Create multi-leg structure
    structure = OptionStructure(structure_type='straddle', underlying='SPY')
    structure.add_option(call_option)
    structure.add_option(put_option)
    net_greeks = structure.calculate_net_greeks(spot=450, vol=0.20)

Financial Accuracy:
    All pricing functions are validated against closed-form solutions and
    adhere to standard financial conventions:
    - Greeks scaling: theta per day, vega/rho per 1%
    - Put-call parity maintained
    - Proper handling of edge cases (T=0, sigma=0, etc.)

Example Workflow:
    >>> from backtester.core import Option, OptionStructure, black_scholes_call, calculate_greeks
    >>> from datetime import datetime
    >>>
    >>> # Price a call option
    >>> price = black_scholes_call(S=100, K=100, T=0.25, r=0.05, sigma=0.20)
    >>> print(f"Call price: ${price:.2f}")
    Call price: $5.88
    >>>
    >>> # Calculate all Greeks
    >>> greeks = calculate_greeks(S=100, K=100, T=0.25, r=0.05, sigma=0.20, option_type='call')
    >>> print(f"Delta: {greeks['delta']:.4f}, Gamma: {greeks['gamma']:.4f}")
    Delta: 0.5596, Gamma: 0.0396
"""

# Import pricing functions
from backtester.core.pricing import (
    # Exceptions
    PricingError,
    ImpliedVolatilityError,
    # Core pricing functions
    black_scholes_call,
    black_scholes_put,
    black_scholes_price,
    # Individual Greeks
    calculate_delta,
    calculate_gamma,
    calculate_theta,
    calculate_vega,
    calculate_rho,
    calculate_greeks,
    # Implied volatility
    calculate_implied_volatility,
    # American options (Barone-Adesi-Whaley)
    barone_adesi_whaley_american_call,
    barone_adesi_whaley_american_put,
    # Vectorized functions for performance
    black_scholes_call_vectorized,
    black_scholes_put_vectorized,
    calculate_greeks_vectorized,
    # Constants
    DAYS_PER_YEAR,
    TRADING_DAYS_PER_YEAR,
)

# Import Option class and related
from backtester.core.option import (
    # Main class
    Option,
    # Exceptions
    OptionError,
    OptionExpiredError,
    OptionValidationError,
    # Factory functions
    create_long_call,
    create_short_call,
    create_long_put,
    create_short_put,
    # Constants
    CONTRACT_MULTIPLIER,
    DEFAULT_ATM_THRESHOLD,
)

# Import OptionStructure class and related
from backtester.core.option_structure import (
    OptionStructure,
    OptionStructureError,
    OptionStructureValidationError,
    EmptyStructureError,
    BREAKEVEN_TOLERANCE,
    GREEK_NAMES,
)

# Import American option pricing
from backtester.core.american_pricing import (
    BinomialPricer,
    AmericanOptionPricer,
    AmericanPricingError,
    TreeConstructionError,
    price_american_option,
    calculate_american_greeks,
    DEFAULT_TREE_STEPS,
)

# Define public API
__all__ = [
    # =========================================================================
    # Option Class
    # =========================================================================
    "Option",
    # Option Exceptions
    "OptionError",
    "OptionExpiredError",
    "OptionValidationError",
    # Option Factory Functions
    "create_long_call",
    "create_short_call",
    "create_long_put",
    "create_short_put",
    # =========================================================================
    # OptionStructure Class
    # =========================================================================
    "OptionStructure",
    # OptionStructure Exceptions
    "OptionStructureError",
    "OptionStructureValidationError",
    "EmptyStructureError",
    # =========================================================================
    # Pricing Functions
    # =========================================================================
    # Core BS pricing
    "black_scholes_call",
    "black_scholes_put",
    "black_scholes_price",
    # Pricing Exceptions
    "PricingError",
    "ImpliedVolatilityError",
    # =========================================================================
    # Greeks
    # =========================================================================
    "calculate_delta",
    "calculate_gamma",
    "calculate_theta",
    "calculate_vega",
    "calculate_rho",
    "calculate_greeks",
    # =========================================================================
    # Implied Volatility
    # =========================================================================
    "calculate_implied_volatility",
    # =========================================================================
    # American Options (BAW approximation)
    # =========================================================================
    "barone_adesi_whaley_american_call",
    "barone_adesi_whaley_american_put",
    # American Options (Binomial Tree)
    "BinomialPricer",
    "AmericanOptionPricer",
    "AmericanPricingError",
    "TreeConstructionError",
    "price_american_option",
    "calculate_american_greeks",
    "DEFAULT_TREE_STEPS",
    # =========================================================================
    # Vectorized Functions
    # =========================================================================
    "black_scholes_call_vectorized",
    "black_scholes_put_vectorized",
    "calculate_greeks_vectorized",
    # =========================================================================
    # Constants
    # =========================================================================
    "CONTRACT_MULTIPLIER",
    "DEFAULT_ATM_THRESHOLD",
    "DAYS_PER_YEAR",
    "TRADING_DAYS_PER_YEAR",
    "BREAKEVEN_TOLERANCE",
    "GREEK_NAMES",
]

# Module metadata
__version__ = "0.1.0"
__author__ = "Options Backtester Team"
