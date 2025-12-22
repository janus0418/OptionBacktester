"""
Option Class for Options Backtesting

This module provides the Option class representing a single option position
(long or short call/put) in the backtesting system. It encapsulates all
attributes and behaviors needed to track, value, and manage individual
option positions throughout a backtest.

Key Features:
    - Position tracking (long/short, call/put)
    - P&L calculation with proper contract multiplier
    - Greeks calculation and caching
    - Moneyness analysis (ITM/ATM/OTM)
    - Intrinsic and time value decomposition
    - Time to expiry calculations

Usage:
    from backtester.core.option import Option
    from datetime import datetime

    # Create a long call position
    option = Option(
        option_type='call',
        position_type='long',
        underlying='SPY',
        strike=450.0,
        expiration=datetime(2024, 3, 15),
        quantity=10,
        entry_price=5.50,
        entry_date=datetime(2024, 1, 15),
        underlying_price_at_entry=445.0,
        implied_vol_at_entry=0.18
    )

    # Update price and calculate P&L
    option.update_price(new_price=7.25, timestamp=datetime(2024, 2, 1))
    print(f"P&L: ${option.calculate_pnl():,.2f}")

    # Calculate Greeks
    greeks = option.calculate_greeks(spot=450.0, vol=0.20, rate=0.05)
    print(f"Delta: {greeks['delta']:.4f}")

P&L Conventions:
    - Long position: P&L = (current_price - entry_price) * quantity * 100
    - Short position: P&L = (entry_price - current_price) * quantity * 100
    - Contract multiplier is 100 shares per contract

References:
    - Hull, J. C. (2018). Options, Futures, and Other Derivatives.
    - Options positions: https://www.cboe.com/education/
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import warnings

import numpy as np
import pandas as pd

from backtester.core.pricing import (
    black_scholes_price,
    calculate_greeks,
    calculate_implied_volatility,
    ImpliedVolatilityError,
    DAYS_PER_YEAR,
    TRADING_DAYS_PER_YEAR
)

# Configure module logger
logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Contract multiplier (shares per contract)
CONTRACT_MULTIPLIER = 100

# ATM threshold for moneyness classification (2% of spot by default)
DEFAULT_ATM_THRESHOLD = 0.02

# Valid option and position types
VALID_OPTION_TYPES = {'call', 'put', 'c', 'p'}
VALID_POSITION_TYPES = {'long', 'short'}


# =============================================================================
# Exceptions
# =============================================================================

class OptionError(Exception):
    """Base exception for Option class errors."""
    pass


class OptionExpiredError(OptionError):
    """Exception raised when attempting operations on expired options."""
    pass


class OptionValidationError(OptionError):
    """Exception raised when option validation fails."""
    pass


# =============================================================================
# Option Class
# =============================================================================

class Option:
    """
    Represents a single option position (long/short call/put).

    This class encapsulates all information needed to track and manage
    an option position throughout a backtest, including entry details,
    current valuation, Greeks, and P&L calculation.

    Attributes:
        option_type (str): 'call' or 'put'
        position_type (str): 'long' or 'short'
        underlying (str): Ticker symbol (e.g., 'SPY')
        strike (float): Strike price
        expiration (datetime): Expiration date
        quantity (int): Number of contracts
        entry_price (float): Premium paid/received at entry (per share)
        entry_date (datetime): Trade entry timestamp
        underlying_price_at_entry (float): Spot price when position opened
        implied_vol_at_entry (Optional[float]): IV at entry
        current_price (float): Current option price (per share)
        current_timestamp (Optional[datetime]): Timestamp of last price update
        greeks (Dict[str, float]): Dictionary of current Greeks

    Properties:
        is_call (bool): True if call option
        is_put (bool): True if put option
        is_long (bool): True if long position
        is_short (bool): True if short position
        is_expired (bool): True if option has expired
        position_sign (int): +1 for long, -1 for short
        notional_value (float): Strike * quantity * 100
        market_value (float): Current price * quantity * 100 * position_sign

    Example:
        >>> from datetime import datetime
        >>> option = Option(
        ...     option_type='call',
        ...     position_type='long',
        ...     underlying='SPY',
        ...     strike=450.0,
        ...     expiration=datetime(2024, 3, 15),
        ...     quantity=10,
        ...     entry_price=5.50,
        ...     entry_date=datetime(2024, 1, 15),
        ...     underlying_price_at_entry=445.0
        ... )
        >>> print(option)
        LONG 10 SPY 450.0 CALL exp 2024-03-15 @ $5.50
    """

    __slots__ = (
        '_option_type',
        '_position_type',
        '_underlying',
        '_strike',
        '_expiration',
        '_quantity',
        '_entry_price',
        '_entry_date',
        '_underlying_price_at_entry',
        '_implied_vol_at_entry',
        '_current_price',
        '_current_timestamp',
        '_greeks',
        '_price_history'
    )

    def __init__(
        self,
        option_type: str,
        position_type: str,
        underlying: str,
        strike: float,
        expiration: datetime,
        quantity: int,
        entry_price: float,
        entry_date: datetime,
        underlying_price_at_entry: float,
        implied_vol_at_entry: Optional[float] = None
    ) -> None:
        """
        Initialize an Option position.

        Args:
            option_type: 'call' or 'put'
            position_type: 'long' or 'short'
            underlying: Ticker symbol of the underlying asset
            strike: Strike price of the option
            expiration: Expiration date (datetime object)
            quantity: Number of contracts (positive integer)
            entry_price: Premium paid/received per share at entry
            entry_date: Date/time when position was opened
            underlying_price_at_entry: Spot price of underlying at entry
            implied_vol_at_entry: Implied volatility at entry (optional)

        Raises:
            OptionValidationError: If any parameter is invalid

        Example:
            >>> option = Option(
            ...     option_type='call',
            ...     position_type='long',
            ...     underlying='SPY',
            ...     strike=450.0,
            ...     expiration=datetime(2024, 3, 15),
            ...     quantity=10,
            ...     entry_price=5.50,
            ...     entry_date=datetime(2024, 1, 15),
            ...     underlying_price_at_entry=445.0
            ... )
        """
        # Validate and set option_type
        self._validate_option_type(option_type)
        self._option_type = option_type.lower().strip()
        if self._option_type in ('c',):
            self._option_type = 'call'
        elif self._option_type in ('p',):
            self._option_type = 'put'

        # Validate and set position_type
        self._validate_position_type(position_type)
        self._position_type = position_type.lower().strip()

        # Validate and set underlying
        if not underlying or not isinstance(underlying, str):
            raise OptionValidationError("underlying must be a non-empty string")
        self._underlying = underlying.upper().strip()

        # Validate and set strike
        if strike is None or strike <= 0:
            raise OptionValidationError(f"strike must be positive, got {strike}")
        if not np.isfinite(strike):
            raise OptionValidationError(f"strike must be finite, got {strike}")
        self._strike = float(strike)

        # Validate and set expiration
        if expiration is None:
            raise OptionValidationError("expiration cannot be None")
        if not isinstance(expiration, datetime):
            raise OptionValidationError(
                f"expiration must be a datetime object, got {type(expiration)}"
            )
        self._expiration = expiration

        # Validate and set quantity
        if quantity is None or quantity <= 0:
            raise OptionValidationError(f"quantity must be positive, got {quantity}")
        if not isinstance(quantity, (int, np.integer)):
            raise OptionValidationError(
                f"quantity must be an integer, got {type(quantity)}"
            )
        self._quantity = int(quantity)

        # Validate and set entry_price
        if entry_price is None or entry_price < 0:
            raise OptionValidationError(f"entry_price must be non-negative, got {entry_price}")
        if not np.isfinite(entry_price):
            raise OptionValidationError(f"entry_price must be finite, got {entry_price}")
        self._entry_price = float(entry_price)

        # Validate and set entry_date
        if entry_date is None:
            raise OptionValidationError("entry_date cannot be None")
        if not isinstance(entry_date, datetime):
            raise OptionValidationError(
                f"entry_date must be a datetime object, got {type(entry_date)}"
            )
        self._entry_date = entry_date

        # Validate and set underlying_price_at_entry
        if underlying_price_at_entry is None or underlying_price_at_entry <= 0:
            raise OptionValidationError(
                f"underlying_price_at_entry must be positive, got {underlying_price_at_entry}"
            )
        if not np.isfinite(underlying_price_at_entry):
            raise OptionValidationError(
                f"underlying_price_at_entry must be finite, got {underlying_price_at_entry}"
            )
        self._underlying_price_at_entry = float(underlying_price_at_entry)

        # Validate and set implied_vol_at_entry (optional)
        if implied_vol_at_entry is not None:
            if implied_vol_at_entry < 0:
                raise OptionValidationError(
                    f"implied_vol_at_entry must be non-negative, got {implied_vol_at_entry}"
                )
            if not np.isfinite(implied_vol_at_entry):
                raise OptionValidationError(
                    f"implied_vol_at_entry must be finite, got {implied_vol_at_entry}"
                )
            self._implied_vol_at_entry = float(implied_vol_at_entry)
        else:
            self._implied_vol_at_entry = None

        # Initialize current price to entry price
        self._current_price = self._entry_price
        self._current_timestamp = self._entry_date

        # Initialize Greeks cache
        self._greeks: Dict[str, float] = {}

        # Initialize price history
        self._price_history: List[Tuple[datetime, float]] = [
            (self._entry_date, self._entry_price)
        ]

        logger.debug(
            f"Created option: {self.position_type.upper()} {self.quantity} "
            f"{self.underlying} {self.strike} {self.option_type.upper()} "
            f"exp {self.expiration.date()} @ ${self.entry_price:.2f}"
        )

    # =========================================================================
    # Validation Methods
    # =========================================================================

    @staticmethod
    def _validate_option_type(option_type: str) -> None:
        """Validate option_type parameter."""
        if not option_type or not isinstance(option_type, str):
            raise OptionValidationError("option_type must be a non-empty string")
        if option_type.lower().strip() not in VALID_OPTION_TYPES:
            raise OptionValidationError(
                f"option_type must be 'call' or 'put', got '{option_type}'"
            )

    @staticmethod
    def _validate_position_type(position_type: str) -> None:
        """Validate position_type parameter."""
        if not position_type or not isinstance(position_type, str):
            raise OptionValidationError("position_type must be a non-empty string")
        if position_type.lower().strip() not in VALID_POSITION_TYPES:
            raise OptionValidationError(
                f"position_type must be 'long' or 'short', got '{position_type}'"
            )

    # =========================================================================
    # Properties - Option Attributes
    # =========================================================================

    @property
    def option_type(self) -> str:
        """Get option type ('call' or 'put')."""
        return self._option_type

    @property
    def position_type(self) -> str:
        """Get position type ('long' or 'short')."""
        return self._position_type

    @property
    def underlying(self) -> str:
        """Get underlying ticker symbol."""
        return self._underlying

    @property
    def strike(self) -> float:
        """Get strike price."""
        return self._strike

    @property
    def expiration(self) -> datetime:
        """Get expiration date."""
        return self._expiration

    @property
    def quantity(self) -> int:
        """Get number of contracts."""
        return self._quantity

    @property
    def entry_price(self) -> float:
        """Get entry price (premium per share)."""
        return self._entry_price

    @property
    def entry_date(self) -> datetime:
        """Get entry date."""
        return self._entry_date

    @property
    def underlying_price_at_entry(self) -> float:
        """Get underlying price at entry."""
        return self._underlying_price_at_entry

    @property
    def implied_vol_at_entry(self) -> Optional[float]:
        """Get implied volatility at entry."""
        return self._implied_vol_at_entry

    @property
    def current_price(self) -> float:
        """Get current option price (premium per share)."""
        return self._current_price

    @property
    def current_timestamp(self) -> Optional[datetime]:
        """Get timestamp of last price update."""
        return self._current_timestamp

    @property
    def greeks(self) -> Dict[str, float]:
        """Get current Greeks (may be empty if not calculated)."""
        return self._greeks.copy()

    # =========================================================================
    # Properties - Computed Attributes
    # =========================================================================

    @property
    def is_call(self) -> bool:
        """Check if option is a call."""
        return self._option_type == 'call'

    @property
    def is_put(self) -> bool:
        """Check if option is a put."""
        return self._option_type == 'put'

    @property
    def is_long(self) -> bool:
        """Check if position is long."""
        return self._position_type == 'long'

    @property
    def is_short(self) -> bool:
        """Check if position is short."""
        return self._position_type == 'short'

    @property
    def position_sign(self) -> int:
        """
        Get position sign: +1 for long, -1 for short.

        This is used in P&L calculations:
        - Long: benefit from price increases
        - Short: benefit from price decreases
        """
        return 1 if self.is_long else -1

    @property
    def is_expired(self) -> bool:
        """
        Check if option has expired.

        Returns True if current timestamp is past expiration date.
        If current_timestamp is None, uses entry_date.
        """
        check_date = self._current_timestamp or self._entry_date
        return check_date >= self._expiration

    @property
    def notional_value(self) -> float:
        """
        Get notional value of the position.

        Notional = Strike * Quantity * Contract Multiplier
        """
        return self._strike * self._quantity * CONTRACT_MULTIPLIER

    @property
    def market_value(self) -> float:
        """
        Get current market value of the position.

        For long: Current Price * Quantity * 100 (positive)
        For short: -Current Price * Quantity * 100 (negative, representing liability)
        """
        return self._current_price * self._quantity * CONTRACT_MULTIPLIER * self.position_sign

    @property
    def entry_cost(self) -> float:
        """
        Get initial cost/credit of the position.

        For long: Entry Price * Quantity * 100 (debit paid)
        For short: -Entry Price * Quantity * 100 (credit received)
        """
        return self._entry_price * self._quantity * CONTRACT_MULTIPLIER * self.position_sign

    # =========================================================================
    # Core Methods
    # =========================================================================

    def calculate_greeks(
        self,
        spot: float,
        vol: float,
        rate: float = 0.04,
        current_date: Optional[datetime] = None
    ) -> Dict[str, float]:
        """
        Calculate Greeks for this option using the pricing module.

        Calculates delta, gamma, theta, vega, and rho based on current
        market conditions. The Greeks are cached for subsequent access
        via the greeks property.

        Args:
            spot: Current spot price of the underlying
            vol: Current implied volatility (annualized, e.g., 0.20 for 20%)
            rate: Risk-free interest rate (annualized). Default 0.04 (4%)
            current_date: Current date for time-to-expiry calculation.
                         If None, uses current_timestamp or entry_date.

        Returns:
            Dictionary containing:
                - 'delta': Rate of change w.r.t. spot
                - 'gamma': Rate of change of delta w.r.t. spot
                - 'theta': Daily time decay
                - 'vega': Per 1% volatility change
                - 'rho': Per 1% interest rate change

        Raises:
            OptionExpiredError: If option has expired
            ValueError: If spot or vol is invalid

        Note:
            Greeks are for a single contract. For position Greeks,
            multiply by quantity and position_sign where appropriate:
            - Position Delta = delta * quantity * position_sign
            - Position Gamma = gamma * quantity * position_sign
            - Position Theta = theta * quantity * position_sign
            - Position Vega = vega * quantity * position_sign
            - Position Rho = rho * quantity * position_sign

        Example:
            >>> greeks = option.calculate_greeks(spot=450.0, vol=0.20, rate=0.05)
            >>> print(f"Delta: {greeks['delta']:.4f}")
        """
        # Validate inputs
        if spot is None or spot <= 0:
            raise ValueError(f"spot must be positive, got {spot}")
        if vol is None or vol <= 0:
            raise ValueError(f"vol must be positive, got {vol}")
        if rate is None:
            raise ValueError("rate cannot be None")

        # Determine current date
        if current_date is None:
            current_date = self._current_timestamp or self._entry_date

        # Calculate time to expiry
        time_to_expiry = self.get_time_to_expiry(current_date)

        if time_to_expiry <= 0:
            raise OptionExpiredError(
                f"Option expired on {self._expiration.date()}. "
                f"Cannot calculate Greeks for expired options."
            )

        # Calculate Greeks using pricing module
        self._greeks = calculate_greeks(
            S=spot,
            K=self._strike,
            T=time_to_expiry,
            r=rate,
            sigma=vol,
            option_type=self._option_type
        )

        return self._greeks.copy()

    def update_price(
        self,
        new_price: float,
        timestamp: datetime
    ) -> None:
        """
        Update the current option price and timestamp.

        Records the price update in the price history and updates
        the current price and timestamp. This should be called
        as market data changes during a backtest.

        Args:
            new_price: New option price (per share)
            timestamp: Timestamp of the price update

        Raises:
            ValueError: If new_price is negative or timestamp is invalid
            OptionExpiredError: If attempting to update price after expiration

        Example:
            >>> option.update_price(new_price=6.25, timestamp=datetime(2024, 2, 1))
            >>> print(f"Current price: ${option.current_price:.2f}")
        """
        # Validate inputs
        if new_price is None or new_price < 0:
            raise ValueError(f"new_price must be non-negative, got {new_price}")
        if not np.isfinite(new_price):
            raise ValueError(f"new_price must be finite, got {new_price}")
        if timestamp is None:
            raise ValueError("timestamp cannot be None")
        if not isinstance(timestamp, datetime):
            raise ValueError(f"timestamp must be datetime, got {type(timestamp)}")

        # Check if option has expired at this timestamp
        if timestamp > self._expiration:
            raise OptionExpiredError(
                f"Cannot update price: timestamp {timestamp} is after "
                f"expiration {self._expiration}"
            )

        # Update state
        self._current_price = float(new_price)
        self._current_timestamp = timestamp

        # Record in price history
        self._price_history.append((timestamp, new_price))

        logger.debug(
            f"Updated {self.underlying} {self.strike} {self.option_type.upper()} "
            f"price to ${new_price:.2f} at {timestamp}"
        )

    def calculate_pnl(self) -> float:
        """
        Calculate current P&L for this position.

        P&L Convention:
            - Long: P&L = (current_price - entry_price) * quantity * 100
            - Short: P&L = (entry_price - current_price) * quantity * 100

        This simplifies to:
            P&L = (current_price - entry_price) * quantity * 100 * position_sign

        Returns:
            Current P&L in dollars (positive = profit, negative = loss)

        Example:
            >>> # Long call, entry $5.50, current $7.25, 10 contracts
            >>> # P&L = (7.25 - 5.50) * 10 * 100 = $1,750
            >>> pnl = option.calculate_pnl()
            >>> print(f"P&L: ${pnl:,.2f}")
        """
        price_change = self._current_price - self._entry_price
        return price_change * self._quantity * CONTRACT_MULTIPLIER * self.position_sign

    def calculate_pnl_at_price(self, price: float) -> float:
        """
        Calculate hypothetical P&L at a given option price.

        Args:
            price: Hypothetical option price (per share)

        Returns:
            P&L in dollars at the given price

        Example:
            >>> # What would P&L be if price went to $8.00?
            >>> hypothetical_pnl = option.calculate_pnl_at_price(8.00)
        """
        if price < 0:
            raise ValueError(f"price must be non-negative, got {price}")

        price_change = price - self._entry_price
        return price_change * self._quantity * CONTRACT_MULTIPLIER * self.position_sign

    def get_payoff_at_expiry(self, spot_price: float) -> float:
        """
        Calculate intrinsic payoff at expiration.

        This is the payoff from the option itself (not accounting for
        entry premium). For full P&L at expiry, use calculate_pnl_at_expiry().

        Payoff formulas:
            - Call payoff: max(S - K, 0)
            - Put payoff: max(K - S, 0)

        For positions:
            - Long: payoff * quantity * 100 (receive payoff)
            - Short: -payoff * quantity * 100 (pay payoff)

        Args:
            spot_price: Underlying price at expiration

        Returns:
            Payoff in dollars (before premium consideration)

        Example:
            >>> # Call with strike 450, spot at expiry = 460
            >>> # Payoff per share = max(460 - 450, 0) = $10
            >>> payoff = option.get_payoff_at_expiry(spot_price=460.0)
        """
        if spot_price is None or spot_price < 0:
            raise ValueError(f"spot_price must be non-negative, got {spot_price}")

        intrinsic = self.get_intrinsic_value(spot_price)
        return intrinsic * self._quantity * CONTRACT_MULTIPLIER * self.position_sign

    def calculate_pnl_at_expiry(self, spot_price: float) -> float:
        """
        Calculate total P&L at expiration given a spot price.

        This includes the payoff and the entry premium:
            P&L = Payoff - Entry Cost (for long)
            P&L = Entry Credit - Payoff (for short)

        Args:
            spot_price: Underlying price at expiration

        Returns:
            Total P&L in dollars at expiration

        Example:
            >>> # Long call, strike 450, entry $5.50, spot at expiry = 460
            >>> # Intrinsic = $10, Entry cost = $5.50
            >>> # P&L per share = 10 - 5.50 = $4.50
            >>> pnl = option.calculate_pnl_at_expiry(spot_price=460.0)
        """
        intrinsic = self.get_intrinsic_value(spot_price)
        return self.calculate_pnl_at_price(intrinsic)

    # =========================================================================
    # Moneyness Methods
    # =========================================================================

    def is_itm(self, spot_price: float) -> bool:
        """
        Check if option is in-the-money.

        ITM Definition:
            - Call: spot > strike (option has intrinsic value)
            - Put: spot < strike (option has intrinsic value)

        Args:
            spot_price: Current underlying price

        Returns:
            True if option is ITM

        Example:
            >>> # Call with strike 450
            >>> option.is_itm(spot_price=460.0)  # True
            >>> option.is_itm(spot_price=440.0)  # False
        """
        if spot_price is None or spot_price <= 0:
            raise ValueError(f"spot_price must be positive, got {spot_price}")

        if self.is_call:
            return spot_price > self._strike
        else:
            return spot_price < self._strike

    def is_atm(
        self,
        spot_price: float,
        threshold: float = DEFAULT_ATM_THRESHOLD
    ) -> bool:
        """
        Check if option is at-the-money (within threshold).

        ATM is defined as spot being within a percentage threshold
        of the strike price. This accounts for the fact that strikes
        rarely equal spot exactly.

        Args:
            spot_price: Current underlying price
            threshold: Percentage threshold for ATM classification.
                      Default 0.02 (2%). Option is ATM if:
                      |spot/strike - 1| <= threshold

        Returns:
            True if option is ATM (within threshold of strike)

        Example:
            >>> # Call with strike 450, spot = 448
            >>> # |448/450 - 1| = 0.0044 < 0.02
            >>> option.is_atm(spot_price=448.0)  # True
            >>> option.is_atm(spot_price=430.0)  # False
        """
        if spot_price is None or spot_price <= 0:
            raise ValueError(f"spot_price must be positive, got {spot_price}")
        if threshold <= 0:
            raise ValueError(f"threshold must be positive, got {threshold}")

        moneyness = spot_price / self._strike
        return abs(moneyness - 1.0) <= threshold

    def is_otm(self, spot_price: float) -> bool:
        """
        Check if option is out-of-the-money.

        OTM Definition:
            - Call: spot < strike (no intrinsic value)
            - Put: spot > strike (no intrinsic value)

        Args:
            spot_price: Current underlying price

        Returns:
            True if option is OTM

        Example:
            >>> # Call with strike 450
            >>> option.is_otm(spot_price=440.0)  # True
            >>> option.is_otm(spot_price=460.0)  # False
        """
        if spot_price is None or spot_price <= 0:
            raise ValueError(f"spot_price must be positive, got {spot_price}")

        if self.is_call:
            return spot_price < self._strike
        else:
            return spot_price > self._strike

    def get_moneyness(self, spot_price: float) -> float:
        """
        Get moneyness ratio (spot/strike).

        Moneyness > 1: ITM for calls, OTM for puts
        Moneyness = 1: ATM
        Moneyness < 1: OTM for calls, ITM for puts

        Args:
            spot_price: Current underlying price

        Returns:
            Moneyness ratio (spot/strike)

        Example:
            >>> option.get_moneyness(spot_price=460.0)
            1.0222...
        """
        if spot_price is None or spot_price <= 0:
            raise ValueError(f"spot_price must be positive, got {spot_price}")

        return spot_price / self._strike

    def get_moneyness_str(
        self,
        spot_price: float,
        atm_threshold: float = DEFAULT_ATM_THRESHOLD
    ) -> str:
        """
        Get human-readable moneyness classification.

        Args:
            spot_price: Current underlying price
            atm_threshold: Threshold for ATM classification

        Returns:
            'ITM', 'ATM', or 'OTM'

        Example:
            >>> option.get_moneyness_str(spot_price=460.0)
            'ITM'
        """
        if self.is_atm(spot_price, atm_threshold):
            return 'ATM'
        elif self.is_itm(spot_price):
            return 'ITM'
        else:
            return 'OTM'

    # =========================================================================
    # Value Decomposition Methods
    # =========================================================================

    def get_intrinsic_value(self, spot_price: float) -> float:
        """
        Calculate intrinsic value (per share).

        Intrinsic value is the immediate exercise value:
            - Call: max(S - K, 0)
            - Put: max(K - S, 0)

        Args:
            spot_price: Current underlying price

        Returns:
            Intrinsic value per share (non-negative)

        Example:
            >>> # Call with strike 450, spot = 460
            >>> option.get_intrinsic_value(spot_price=460.0)
            10.0
        """
        if spot_price is None or spot_price < 0:
            raise ValueError(f"spot_price must be non-negative, got {spot_price}")

        if self.is_call:
            return max(spot_price - self._strike, 0.0)
        else:
            return max(self._strike - spot_price, 0.0)

    def get_time_value(self, spot_price: Optional[float] = None) -> float:
        """
        Calculate time value (extrinsic value) per share.

        Time value = Current Price - Intrinsic Value

        Time value represents the additional value from the possibility
        of the option becoming more valuable before expiration.

        Args:
            spot_price: Current underlying price. If None, uses
                       underlying_price_at_entry.

        Returns:
            Time value per share (can be negative if price < intrinsic,
            which indicates potential arbitrage)

        Raises:
            OptionExpiredError: If option has expired (no time value)

        Example:
            >>> # Call with strike 450, spot = 455, current_price = $8
            >>> # Intrinsic = 5, Time value = 8 - 5 = $3
            >>> option.get_time_value(spot_price=455.0)
            3.0
        """
        if self.is_expired:
            raise OptionExpiredError(
                "Cannot calculate time value for expired option"
            )

        if spot_price is None:
            spot_price = self._underlying_price_at_entry

        intrinsic = self.get_intrinsic_value(spot_price)
        return self._current_price - intrinsic

    # =========================================================================
    # Time Methods
    # =========================================================================

    def get_time_to_expiry(
        self,
        current_date: Optional[datetime] = None
    ) -> float:
        """
        Get time to expiration in years.

        Uses calendar days / 365 for consistency with standard
        Black-Scholes implementations.

        Args:
            current_date: Reference date. If None, uses current_timestamp
                         or entry_date.

        Returns:
            Time to expiration in years. Returns 0 if expired.

        Example:
            >>> # 30 days to expiration
            >>> tte = option.get_time_to_expiry()
            >>> print(f"Time to expiry: {tte:.4f} years ({tte * 365:.0f} days)")
        """
        if current_date is None:
            current_date = self._current_timestamp or self._entry_date

        # Calculate days to expiry
        time_delta = self._expiration - current_date
        days_to_expiry = time_delta.total_seconds() / (24 * 3600)

        # Return 0 if expired
        if days_to_expiry <= 0:
            return 0.0

        return days_to_expiry / DAYS_PER_YEAR

    def get_days_to_expiry(
        self,
        current_date: Optional[datetime] = None
    ) -> int:
        """
        Get days to expiration (calendar days).

        Args:
            current_date: Reference date. If None, uses current_timestamp
                         or entry_date.

        Returns:
            Calendar days to expiration. Returns 0 if expired.

        Example:
            >>> dte = option.get_days_to_expiry()
            >>> print(f"DTE: {dte}")
        """
        if current_date is None:
            current_date = self._current_timestamp or self._entry_date

        time_delta = self._expiration - current_date
        days = time_delta.days

        return max(days, 0)

    def get_trading_days_to_expiry(
        self,
        current_date: Optional[datetime] = None,
        trading_days_per_year: int = TRADING_DAYS_PER_YEAR
    ) -> float:
        """
        Get approximate trading days to expiration.

        Estimates trading days by scaling calendar days.

        Args:
            current_date: Reference date.
            trading_days_per_year: Number of trading days per year.
                                  Default 252.

        Returns:
            Estimated trading days to expiration.
        """
        tte_years = self.get_time_to_expiry(current_date)
        return tte_years * trading_days_per_year

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_price_history(self) -> List[Tuple[datetime, float]]:
        """
        Get price update history.

        Returns:
            List of (timestamp, price) tuples in chronological order.
        """
        return self._price_history.copy()

    def get_return(self) -> float:
        """
        Get percentage return on the position.

        For long positions: (current - entry) / entry
        For short positions: (entry - current) / entry

        Returns:
            Percentage return as a decimal (e.g., 0.15 for 15% return)

        Raises:
            ValueError: If entry_price is zero (cannot calculate return)
        """
        if self._entry_price <= 0:
            raise ValueError("Cannot calculate return with zero entry price")

        return (self._current_price - self._entry_price) / self._entry_price * self.position_sign

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert option to dictionary representation.

        Returns:
            Dictionary with all option attributes.
        """
        return {
            'option_type': self._option_type,
            'position_type': self._position_type,
            'underlying': self._underlying,
            'strike': self._strike,
            'expiration': self._expiration,
            'quantity': self._quantity,
            'entry_price': self._entry_price,
            'entry_date': self._entry_date,
            'underlying_price_at_entry': self._underlying_price_at_entry,
            'implied_vol_at_entry': self._implied_vol_at_entry,
            'current_price': self._current_price,
            'current_timestamp': self._current_timestamp,
            'greeks': self._greeks.copy(),
            'pnl': self.calculate_pnl(),
            'market_value': self.market_value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Option':
        """
        Create Option from dictionary representation.

        Args:
            data: Dictionary with option attributes

        Returns:
            New Option instance

        Raises:
            KeyError: If required keys are missing
        """
        option = cls(
            option_type=data['option_type'],
            position_type=data['position_type'],
            underlying=data['underlying'],
            strike=data['strike'],
            expiration=data['expiration'],
            quantity=data['quantity'],
            entry_price=data['entry_price'],
            entry_date=data['entry_date'],
            underlying_price_at_entry=data['underlying_price_at_entry'],
            implied_vol_at_entry=data.get('implied_vol_at_entry')
        )

        # Restore current state if available
        if 'current_price' in data and 'current_timestamp' in data:
            if data['current_timestamp'] is not None:
                option.update_price(
                    new_price=data['current_price'],
                    timestamp=data['current_timestamp']
                )

        return option

    # =========================================================================
    # Special Methods
    # =========================================================================

    def __repr__(self) -> str:
        """Return detailed string representation."""
        return (
            f"Option("
            f"type={self._option_type!r}, "
            f"position={self._position_type!r}, "
            f"underlying={self._underlying!r}, "
            f"strike={self._strike}, "
            f"expiration={self._expiration.date()}, "
            f"quantity={self._quantity}, "
            f"entry_price={self._entry_price}, "
            f"current_price={self._current_price}"
            f")"
        )

    def __str__(self) -> str:
        """Return human-readable string."""
        return (
            f"{self._position_type.upper()} {self._quantity} "
            f"{self._underlying} {self._strike} {self._option_type.upper()} "
            f"exp {self._expiration.date()} @ ${self._entry_price:.2f}"
        )

    def __eq__(self, other: object) -> bool:
        """Check equality based on key attributes."""
        if not isinstance(other, Option):
            return NotImplemented
        return (
            self._option_type == other._option_type and
            self._position_type == other._position_type and
            self._underlying == other._underlying and
            self._strike == other._strike and
            self._expiration == other._expiration and
            self._quantity == other._quantity and
            self._entry_price == other._entry_price and
            self._entry_date == other._entry_date
        )

    def __hash__(self) -> int:
        """Return hash based on key attributes."""
        return hash((
            self._option_type,
            self._position_type,
            self._underlying,
            self._strike,
            self._expiration,
            self._quantity,
            self._entry_price,
            self._entry_date
        ))


# =============================================================================
# Factory Functions
# =============================================================================

def create_long_call(
    underlying: str,
    strike: float,
    expiration: datetime,
    quantity: int,
    entry_price: float,
    entry_date: datetime,
    underlying_price: float,
    implied_vol: Optional[float] = None
) -> Option:
    """
    Factory function to create a long call option.

    Args:
        underlying: Ticker symbol
        strike: Strike price
        expiration: Expiration date
        quantity: Number of contracts
        entry_price: Premium paid per share
        entry_date: Entry timestamp
        underlying_price: Spot price at entry
        implied_vol: Optional implied volatility at entry

    Returns:
        Option instance configured as long call

    Example:
        >>> call = create_long_call(
        ...     underlying='SPY',
        ...     strike=450.0,
        ...     expiration=datetime(2024, 3, 15),
        ...     quantity=10,
        ...     entry_price=5.50,
        ...     entry_date=datetime(2024, 1, 15),
        ...     underlying_price=445.0
        ... )
    """
    return Option(
        option_type='call',
        position_type='long',
        underlying=underlying,
        strike=strike,
        expiration=expiration,
        quantity=quantity,
        entry_price=entry_price,
        entry_date=entry_date,
        underlying_price_at_entry=underlying_price,
        implied_vol_at_entry=implied_vol
    )


def create_short_call(
    underlying: str,
    strike: float,
    expiration: datetime,
    quantity: int,
    entry_price: float,
    entry_date: datetime,
    underlying_price: float,
    implied_vol: Optional[float] = None
) -> Option:
    """
    Factory function to create a short call option.

    Args:
        underlying: Ticker symbol
        strike: Strike price
        expiration: Expiration date
        quantity: Number of contracts
        entry_price: Premium received per share
        entry_date: Entry timestamp
        underlying_price: Spot price at entry
        implied_vol: Optional implied volatility at entry

    Returns:
        Option instance configured as short call
    """
    return Option(
        option_type='call',
        position_type='short',
        underlying=underlying,
        strike=strike,
        expiration=expiration,
        quantity=quantity,
        entry_price=entry_price,
        entry_date=entry_date,
        underlying_price_at_entry=underlying_price,
        implied_vol_at_entry=implied_vol
    )


def create_long_put(
    underlying: str,
    strike: float,
    expiration: datetime,
    quantity: int,
    entry_price: float,
    entry_date: datetime,
    underlying_price: float,
    implied_vol: Optional[float] = None
) -> Option:
    """
    Factory function to create a long put option.

    Args:
        underlying: Ticker symbol
        strike: Strike price
        expiration: Expiration date
        quantity: Number of contracts
        entry_price: Premium paid per share
        entry_date: Entry timestamp
        underlying_price: Spot price at entry
        implied_vol: Optional implied volatility at entry

    Returns:
        Option instance configured as long put
    """
    return Option(
        option_type='put',
        position_type='long',
        underlying=underlying,
        strike=strike,
        expiration=expiration,
        quantity=quantity,
        entry_price=entry_price,
        entry_date=entry_date,
        underlying_price_at_entry=underlying_price,
        implied_vol_at_entry=implied_vol
    )


def create_short_put(
    underlying: str,
    strike: float,
    expiration: datetime,
    quantity: int,
    entry_price: float,
    entry_date: datetime,
    underlying_price: float,
    implied_vol: Optional[float] = None
) -> Option:
    """
    Factory function to create a short put option.

    Args:
        underlying: Ticker symbol
        strike: Strike price
        expiration: Expiration date
        quantity: Number of contracts
        entry_price: Premium received per share
        entry_date: Entry timestamp
        underlying_price: Spot price at entry
        implied_vol: Optional implied volatility at entry

    Returns:
        Option instance configured as short put
    """
    return Option(
        option_type='put',
        position_type='short',
        underlying=underlying,
        strike=strike,
        expiration=expiration,
        quantity=quantity,
        entry_price=entry_price,
        entry_date=entry_date,
        underlying_price_at_entry=underlying_price,
        implied_vol_at_entry=implied_vol
    )


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Main class
    'Option',

    # Exceptions
    'OptionError',
    'OptionExpiredError',
    'OptionValidationError',

    # Factory functions
    'create_long_call',
    'create_short_call',
    'create_long_put',
    'create_short_put',

    # Constants
    'CONTRACT_MULTIPLIER',
    'DEFAULT_ATM_THRESHOLD',
]
