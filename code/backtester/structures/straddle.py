"""
Straddle Option Structures

This module implements Long and Short Straddle option structures. A straddle
consists of a call and a put at the same strike and expiration, providing
exposure to volatility movements in either direction.

Financial Characteristics:
    - Long Straddle: Buy call + buy put at same strike
      - Max Profit: Unlimited (if underlying moves significantly)
      - Max Loss: Total premium paid
      - Breakevens: Strike ± total premium paid
      - Use case: Expect large move, don't know direction

    - Short Straddle: Sell call + sell put at same strike
      - Max Profit: Total premium received
      - Max Loss: Unlimited (if underlying moves significantly)
      - Breakevens: Strike ± total premium received
      - Use case: Expect no move, collect premium

Greeks:
    - Delta neutral at ATM (call delta + put delta ≈ 0)
    - Positive gamma (benefits from volatility)
    - Negative theta (time decay works against long, for short)
    - Positive vega (benefits from IV increase for long, decreases for short)

Usage:
    from backtester.structures.straddle import LongStraddle, ShortStraddle
    from datetime import datetime

    # Create short straddle
    straddle = ShortStraddle.create(
        underlying='SPY',
        strike=450.0,
        expiration=datetime(2024, 3, 15),
        call_price=5.50,
        put_price=5.25,
        quantity=10,
        entry_date=datetime(2024, 3, 1),
        underlying_price=450.0,
        call_iv=0.18,
        put_iv=0.17
    )

    print(f"Max Profit: ${straddle.max_profit:,.2f}")
    print(f"Breakevens: ${straddle.lower_breakeven:.2f}, ${straddle.upper_breakeven:.2f}")

References:
    - CBOE Straddle Strategy: https://www.cboe.com/strategies/straddle/
    - Hull, J. C. (2018). Options, Futures, and Other Derivatives, Chapter 12.
"""

import logging
from datetime import datetime
from typing import Optional, Tuple

import numpy as np

from backtester.core.option import (
    Option,
    create_long_call,
    create_short_call,
    create_long_put,
    create_short_put,
    OptionValidationError,
)
from backtester.core.option_structure import (
    OptionStructure,
    OptionStructureValidationError,
)

# Configure module logger
logger = logging.getLogger(__name__)


# =============================================================================
# Long Straddle
# =============================================================================

class LongStraddle(OptionStructure):
    """
    Long Straddle: Buy call + buy put at the same strike.

    This is a volatility play that profits from large moves in either direction.
    The position has unlimited profit potential and limited loss (total premium).

    Financial Formulas:
        Max Profit = Unlimited
        Max Loss = Call Premium + Put Premium
        Upper Breakeven = Strike + Total Premium
        Lower Breakeven = Strike - Total Premium

    Greeks:
        Delta ≈ 0 at ATM (delta neutral)
        Gamma > 0 (benefits from large moves)
        Theta < 0 (loses value from time decay)
        Vega > 0 (benefits from volatility increase)

    Example:
        >>> straddle = LongStraddle.create(
        ...     underlying='SPY',
        ...     strike=450.0,
        ...     expiration=datetime(2024, 3, 15),
        ...     call_price=5.50,
        ...     put_price=5.25,
        ...     quantity=10,
        ...     entry_date=datetime(2024, 3, 1),
        ...     underlying_price=450.0
        ... )
        >>> print(f"Max Loss: ${straddle.max_loss:,.2f}")
    """

    __slots__ = (
        '_strike',
        '_call_option',
        '_put_option',
        '_max_profit',
        '_max_loss',
        '_upper_breakeven',
        '_lower_breakeven',
    )

    def __init__(
        self,
        underlying: str,
        strike: float,
        expiration: datetime,
        call_option: Option,
        put_option: Option,
        structure_id: Optional[str] = None,
        entry_date: Optional[datetime] = None
    ) -> None:
        """
        Initialize a Long Straddle structure.

        Args:
            underlying: Ticker symbol
            strike: Strike price (same for call and put)
            expiration: Expiration date (same for call and put)
            call_option: Long call option
            put_option: Long put option
            structure_id: Optional unique identifier
            entry_date: Optional entry timestamp

        Raises:
            OptionStructureValidationError: If options don't match straddle requirements
        """
        # Validate inputs
        if not isinstance(call_option, Option) or not isinstance(put_option, Option):
            raise OptionStructureValidationError(
                "Both call_option and put_option must be Option instances"
            )

        # Validate that both are long positions
        if not call_option.is_long or not put_option.is_long:
            raise OptionStructureValidationError(
                "Long straddle requires both options to be long positions"
            )

        # Validate option types
        if not call_option.is_call:
            raise OptionStructureValidationError(
                "call_option must be a call option"
            )
        if not put_option.is_put:
            raise OptionStructureValidationError(
                "put_option must be a put option"
            )

        # Validate same strike
        if abs(call_option.strike - put_option.strike) > 1e-6:
            raise OptionStructureValidationError(
                f"Straddle requires same strike. Got call: {call_option.strike}, "
                f"put: {put_option.strike}"
            )

        # Validate same expiration
        if call_option.expiration != put_option.expiration:
            raise OptionStructureValidationError(
                f"Straddle requires same expiration. Got call: {call_option.expiration}, "
                f"put: {put_option.expiration}"
            )

        # Validate same underlying
        if call_option.underlying != put_option.underlying:
            raise OptionStructureValidationError(
                f"Straddle requires same underlying. Got call: {call_option.underlying}, "
                f"put: {put_option.underlying}"
            )

        # Initialize base structure
        super().__init__(
            structure_type='long_straddle',
            underlying=underlying,
            structure_id=structure_id,
            entry_date=entry_date
        )

        # Store strike
        self._strike = strike

        # Add options to structure
        self.add_option(call_option)
        self.add_option(put_option)

        # Store references for convenience
        self._call_option = call_option
        self._put_option = put_option

        # Calculate and cache max profit/loss and breakevens
        self._calculate_metrics()

        logger.debug(
            f"Created Long Straddle: {underlying} {strike} strike, "
            f"Max Loss: ${self._max_loss:,.2f}, "
            f"Breakevens: ${self._lower_breakeven:.2f} - ${self._upper_breakeven:.2f}"
        )

    def _calculate_metrics(self) -> None:
        """Calculate and cache max profit, max loss, and breakeven points."""
        # Max Loss = Total premium paid (net premium is negative for debit)
        self._max_loss = -self.net_premium  # Convert to positive loss

        # Max Profit = Unlimited (represented by a very large number)
        self._max_profit = float('inf')

        # Breakevens: Strike ± Total Premium
        total_premium = abs(self.net_premium) / (self._call_option.quantity * 100)
        self._upper_breakeven = self._strike + total_premium
        self._lower_breakeven = self._strike - total_premium

    @classmethod
    def create(
        cls,
        underlying: str,
        strike: float,
        expiration: datetime,
        call_price: float,
        put_price: float,
        quantity: int,
        entry_date: datetime,
        underlying_price: float,
        call_iv: Optional[float] = None,
        put_iv: Optional[float] = None,
        structure_id: Optional[str] = None
    ) -> 'LongStraddle':
        """
        Factory method to create a Long Straddle.

        Args:
            underlying: Ticker symbol (e.g., 'SPY')
            strike: Strike price for both call and put
            expiration: Expiration date for both options
            call_price: Call premium per share
            put_price: Put premium per share
            quantity: Number of contracts (same for both legs)
            entry_date: Entry timestamp
            underlying_price: Spot price at entry
            call_iv: Optional call implied volatility
            put_iv: Optional put implied volatility
            structure_id: Optional unique identifier

        Returns:
            LongStraddle instance

        Example:
            >>> straddle = LongStraddle.create(
            ...     underlying='SPY',
            ...     strike=450.0,
            ...     expiration=datetime(2024, 3, 15),
            ...     call_price=5.50,
            ...     put_price=5.25,
            ...     quantity=10,
            ...     entry_date=datetime(2024, 3, 1),
            ...     underlying_price=450.0
            ... )
        """
        # Create long call
        call = create_long_call(
            underlying=underlying,
            strike=strike,
            expiration=expiration,
            quantity=quantity,
            entry_price=call_price,
            entry_date=entry_date,
            underlying_price=underlying_price,
            implied_vol=call_iv
        )

        # Create long put
        put = create_long_put(
            underlying=underlying,
            strike=strike,
            expiration=expiration,
            quantity=quantity,
            entry_price=put_price,
            entry_date=entry_date,
            underlying_price=underlying_price,
            implied_vol=put_iv
        )

        # Create and return straddle
        return cls(
            underlying=underlying,
            strike=strike,
            expiration=expiration,
            call_option=call,
            put_option=put,
            structure_id=structure_id,
            entry_date=entry_date
        )

    # =========================================================================
    # Properties - Straddle-specific Attributes
    # =========================================================================

    @property
    def strike(self) -> float:
        """Get the strike price."""
        return self._strike

    @property
    def call_option(self) -> Option:
        """Get the call option leg."""
        return self._call_option

    @property
    def put_option(self) -> Option:
        """Get the put option leg."""
        return self._put_option

    @property
    def max_profit(self) -> float:
        """
        Get maximum profit (unlimited for long straddle).

        Returns infinity to represent unlimited profit potential.
        """
        return self._max_profit

    @property
    def max_loss(self) -> float:
        """
        Get maximum loss (total premium paid).

        Returns positive number representing maximum dollar loss.
        """
        return self._max_loss

    @property
    def upper_breakeven(self) -> float:
        """
        Get upper breakeven point.

        Upper Breakeven = Strike + Total Premium Paid
        """
        return self._upper_breakeven

    @property
    def lower_breakeven(self) -> float:
        """
        Get lower breakeven point.

        Lower Breakeven = Strike - Total Premium Paid
        """
        return self._lower_breakeven

    @property
    def breakeven_range(self) -> float:
        """
        Get the width of the breakeven range.

        Range = Upper Breakeven - Lower Breakeven = 2 * Total Premium
        """
        return self._upper_breakeven - self._lower_breakeven


# =============================================================================
# Short Straddle
# =============================================================================

class ShortStraddle(OptionStructure):
    """
    Short Straddle: Sell call + sell put at the same strike.

    This is a premium collection strategy that profits from low volatility.
    The position has limited profit (total premium) and unlimited loss potential.

    Financial Formulas:
        Max Profit = Call Premium + Put Premium
        Max Loss = Unlimited
        Upper Breakeven = Strike + Total Premium
        Lower Breakeven = Strike - Total Premium

    Greeks:
        Delta ≈ 0 at ATM (delta neutral)
        Gamma < 0 (loses from large moves)
        Theta > 0 (benefits from time decay)
        Vega < 0 (loses from volatility increase)

    Example:
        >>> straddle = ShortStraddle.create(
        ...     underlying='SPY',
        ...     strike=450.0,
        ...     expiration=datetime(2024, 3, 15),
        ...     call_price=5.50,
        ...     put_price=5.25,
        ...     quantity=10,
        ...     entry_date=datetime(2024, 3, 1),
        ...     underlying_price=450.0
        ... )
        >>> print(f"Max Profit: ${straddle.max_profit:,.2f}")
    """

    __slots__ = (
        '_strike',
        '_call_option',
        '_put_option',
        '_max_profit',
        '_max_loss',
        '_upper_breakeven',
        '_lower_breakeven',
    )

    def __init__(
        self,
        underlying: str,
        strike: float,
        expiration: datetime,
        call_option: Option,
        put_option: Option,
        structure_id: Optional[str] = None,
        entry_date: Optional[datetime] = None
    ) -> None:
        """
        Initialize a Short Straddle structure.

        Args:
            underlying: Ticker symbol
            strike: Strike price (same for call and put)
            expiration: Expiration date (same for call and put)
            call_option: Short call option
            put_option: Short put option
            structure_id: Optional unique identifier
            entry_date: Optional entry timestamp

        Raises:
            OptionStructureValidationError: If options don't match straddle requirements
        """
        # Validate inputs
        if not isinstance(call_option, Option) or not isinstance(put_option, Option):
            raise OptionStructureValidationError(
                "Both call_option and put_option must be Option instances"
            )

        # Validate that both are short positions
        if not call_option.is_short or not put_option.is_short:
            raise OptionStructureValidationError(
                "Short straddle requires both options to be short positions"
            )

        # Validate option types
        if not call_option.is_call:
            raise OptionStructureValidationError(
                "call_option must be a call option"
            )
        if not put_option.is_put:
            raise OptionStructureValidationError(
                "put_option must be a put option"
            )

        # Validate same strike
        if abs(call_option.strike - put_option.strike) > 1e-6:
            raise OptionStructureValidationError(
                f"Straddle requires same strike. Got call: {call_option.strike}, "
                f"put: {put_option.strike}"
            )

        # Validate same expiration
        if call_option.expiration != put_option.expiration:
            raise OptionStructureValidationError(
                f"Straddle requires same expiration. Got call: {call_option.expiration}, "
                f"put: {put_option.expiration}"
            )

        # Validate same underlying
        if call_option.underlying != put_option.underlying:
            raise OptionStructureValidationError(
                f"Straddle requires same underlying. Got call: {call_option.underlying}, "
                f"put: {put_option.underlying}"
            )

        # Initialize base structure
        super().__init__(
            structure_type='short_straddle',
            underlying=underlying,
            structure_id=structure_id,
            entry_date=entry_date
        )

        # Store strike
        self._strike = strike

        # Add options to structure
        self.add_option(call_option)
        self.add_option(put_option)

        # Store references for convenience
        self._call_option = call_option
        self._put_option = put_option

        # Calculate and cache max profit/loss and breakevens
        self._calculate_metrics()

        logger.debug(
            f"Created Short Straddle: {underlying} {strike} strike, "
            f"Max Profit: ${self._max_profit:,.2f}, "
            f"Breakevens: ${self._lower_breakeven:.2f} - ${self._upper_breakeven:.2f}"
        )

    def _calculate_metrics(self) -> None:
        """Calculate and cache max profit, max loss, and breakeven points."""
        # Max Profit = Total premium received (net premium is positive for credit)
        self._max_profit = self.net_premium

        # Max Loss = Unlimited (represented by negative infinity)
        self._max_loss = float('-inf')

        # Breakevens: Strike ± Total Premium
        total_premium = self.net_premium / (self._call_option.quantity * 100)
        self._upper_breakeven = self._strike + total_premium
        self._lower_breakeven = self._strike - total_premium

    @classmethod
    def create(
        cls,
        underlying: str,
        strike: float,
        expiration: datetime,
        call_price: float,
        put_price: float,
        quantity: int,
        entry_date: datetime,
        underlying_price: float,
        call_iv: Optional[float] = None,
        put_iv: Optional[float] = None,
        structure_id: Optional[str] = None
    ) -> 'ShortStraddle':
        """
        Factory method to create a Short Straddle.

        Args:
            underlying: Ticker symbol (e.g., 'SPY')
            strike: Strike price for both call and put
            expiration: Expiration date for both options
            call_price: Call premium per share (received)
            put_price: Put premium per share (received)
            quantity: Number of contracts (same for both legs)
            entry_date: Entry timestamp
            underlying_price: Spot price at entry
            call_iv: Optional call implied volatility
            put_iv: Optional put implied volatility
            structure_id: Optional unique identifier

        Returns:
            ShortStraddle instance

        Example:
            >>> straddle = ShortStraddle.create(
            ...     underlying='SPY',
            ...     strike=450.0,
            ...     expiration=datetime(2024, 3, 15),
            ...     call_price=5.50,
            ...     put_price=5.25,
            ...     quantity=10,
            ...     entry_date=datetime(2024, 3, 1),
            ...     underlying_price=450.0
            ... )
        """
        # Create short call
        call = create_short_call(
            underlying=underlying,
            strike=strike,
            expiration=expiration,
            quantity=quantity,
            entry_price=call_price,
            entry_date=entry_date,
            underlying_price=underlying_price,
            implied_vol=call_iv
        )

        # Create short put
        put = create_short_put(
            underlying=underlying,
            strike=strike,
            expiration=expiration,
            quantity=quantity,
            entry_price=put_price,
            entry_date=entry_date,
            underlying_price=underlying_price,
            implied_vol=put_iv
        )

        # Create and return straddle
        return cls(
            underlying=underlying,
            strike=strike,
            expiration=expiration,
            call_option=call,
            put_option=put,
            structure_id=structure_id,
            entry_date=entry_date
        )

    # =========================================================================
    # Properties - Straddle-specific Attributes
    # =========================================================================

    @property
    def strike(self) -> float:
        """Get the strike price."""
        return self._strike

    @property
    def call_option(self) -> Option:
        """Get the call option leg."""
        return self._call_option

    @property
    def put_option(self) -> Option:
        """Get the put option leg."""
        return self._put_option

    @property
    def max_profit(self) -> float:
        """
        Get maximum profit (total premium received).

        Returns positive number representing maximum dollar profit.
        """
        return self._max_profit

    @property
    def max_loss(self) -> float:
        """
        Get maximum loss (unlimited for short straddle).

        Returns negative infinity to represent unlimited loss potential.
        """
        return self._max_loss

    @property
    def upper_breakeven(self) -> float:
        """
        Get upper breakeven point.

        Upper Breakeven = Strike + Total Premium Received
        """
        return self._upper_breakeven

    @property
    def lower_breakeven(self) -> float:
        """
        Get lower breakeven point.

        Lower Breakeven = Strike - Total Premium Received
        """
        return self._lower_breakeven

    @property
    def breakeven_range(self) -> float:
        """
        Get the width of the breakeven range.

        Range = Upper Breakeven - Lower Breakeven = 2 * Total Premium
        """
        return self._upper_breakeven - self._lower_breakeven


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    'LongStraddle',
    'ShortStraddle',
]
