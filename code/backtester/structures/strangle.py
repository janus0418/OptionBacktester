"""
Strangle Option Structures

This module implements Long and Short Strangle option structures. A strangle
consists of a call and a put at different strikes (call strike > put strike)
with the same expiration, providing exposure to volatility at a lower cost
than straddles.

Financial Characteristics:
    - Long Strangle: Buy OTM call + buy OTM put
      - Max Profit: Unlimited (if underlying moves significantly)
      - Max Loss: Total premium paid
      - Breakevens: Call strike + total premium, Put strike - total premium
      - Use case: Expect large move, cheaper than straddle

    - Short Strangle: Sell OTM call + sell OTM put
      - Max Profit: Total premium received
      - Max Loss: Unlimited (if underlying moves significantly)
      - Breakevens: Call strike + total premium, Put strike - total premium
      - Use case: Expect range-bound movement, collect premium

Greeks:
    - Delta closer to neutral than individual options
    - Positive gamma for long (benefits from volatility)
    - Negative theta for long (time decay works against)
    - Positive vega for long (benefits from IV increase)

Usage:
    from backtester.structures.strangle import LongStrangle, ShortStrangle
    from datetime import datetime

    # Create short strangle
    strangle = ShortStrangle.create(
        underlying='SPY',
        call_strike=460.0,
        put_strike=440.0,
        expiration=datetime(2024, 3, 15),
        call_price=3.50,
        put_price=3.25,
        quantity=10,
        entry_date=datetime(2024, 3, 1),
        underlying_price=450.0
    )

    print(f"Max Profit: ${strangle.max_profit:,.2f}")
    print(f"Breakevens: ${strangle.lower_breakeven:.2f}, ${strangle.upper_breakeven:.2f}")

References:
    - CBOE Strangle Strategy: https://www.cboe.com/strategies/strangle/
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
# Long Strangle
# =============================================================================

class LongStrangle(OptionStructure):
    """
    Long Strangle: Buy OTM call + buy OTM put at different strikes.

    This is a volatility play similar to a straddle but with lower cost
    and wider breakeven range. It profits from large moves in either direction.

    Financial Formulas:
        Max Profit = Unlimited
        Max Loss = Call Premium + Put Premium
        Upper Breakeven = Call Strike + Total Premium
        Lower Breakeven = Put Strike - Total Premium

    Greeks:
        Delta depends on spot relative to strikes
        Gamma > 0 (benefits from large moves)
        Theta < 0 (loses value from time decay)
        Vega > 0 (benefits from volatility increase)

    Example:
        >>> strangle = LongStrangle.create(
        ...     underlying='SPY',
        ...     call_strike=460.0,
        ...     put_strike=440.0,
        ...     expiration=datetime(2024, 3, 15),
        ...     call_price=3.50,
        ...     put_price=3.25,
        ...     quantity=10,
        ...     entry_date=datetime(2024, 3, 1),
        ...     underlying_price=450.0
        ... )
    """

    __slots__ = (
        '_call_strike',
        '_put_strike',
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
        call_strike: float,
        put_strike: float,
        expiration: datetime,
        call_option: Option,
        put_option: Option,
        structure_id: Optional[str] = None,
        entry_date: Optional[datetime] = None
    ) -> None:
        """
        Initialize a Long Strangle structure.

        Args:
            underlying: Ticker symbol
            call_strike: Strike price for call (higher)
            put_strike: Strike price for put (lower)
            expiration: Expiration date (same for call and put)
            call_option: Long call option
            put_option: Long put option
            structure_id: Optional unique identifier
            entry_date: Optional entry timestamp

        Raises:
            OptionStructureValidationError: If options don't match strangle requirements
        """
        # Validate inputs
        if not isinstance(call_option, Option) or not isinstance(put_option, Option):
            raise OptionStructureValidationError(
                "Both call_option and put_option must be Option instances"
            )

        # Validate that both are long positions
        if not call_option.is_long or not put_option.is_long:
            raise OptionStructureValidationError(
                "Long strangle requires both options to be long positions"
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

        # Validate strike ordering (call strike > put strike)
        if call_strike <= put_strike:
            raise OptionStructureValidationError(
                f"Strangle requires call_strike > put_strike. "
                f"Got call: {call_strike}, put: {put_strike}"
            )

        # Validate same expiration
        if call_option.expiration != put_option.expiration:
            raise OptionStructureValidationError(
                f"Strangle requires same expiration. Got call: {call_option.expiration}, "
                f"put: {put_option.expiration}"
            )

        # Validate same underlying
        if call_option.underlying != put_option.underlying:
            raise OptionStructureValidationError(
                f"Strangle requires same underlying. Got call: {call_option.underlying}, "
                f"put: {put_option.underlying}"
            )

        # Initialize base structure
        super().__init__(
            structure_type='long_strangle',
            underlying=underlying,
            structure_id=structure_id,
            entry_date=entry_date
        )

        # Store strikes
        self._call_strike = call_strike
        self._put_strike = put_strike

        # Add options to structure
        self.add_option(call_option)
        self.add_option(put_option)

        # Store references for convenience
        self._call_option = call_option
        self._put_option = put_option

        # Calculate and cache max profit/loss and breakevens
        self._calculate_metrics()

        logger.debug(
            f"Created Long Strangle: {underlying} {put_strike}/{call_strike} strikes, "
            f"Max Loss: ${self._max_loss:,.2f}, "
            f"Breakevens: ${self._lower_breakeven:.2f} - ${self._upper_breakeven:.2f}"
        )

    def _calculate_metrics(self) -> None:
        """Calculate and cache max profit, max loss, and breakeven points."""
        # Max Loss = Total premium paid (net premium is negative for debit)
        self._max_loss = -self.net_premium  # Convert to positive loss

        # Max Profit = Unlimited
        self._max_profit = float('inf')

        # Breakevens:
        # Upper = Call Strike + Total Premium
        # Lower = Put Strike - Total Premium
        total_premium = abs(self.net_premium) / (self._call_option.quantity * 100)
        self._upper_breakeven = self._call_strike + total_premium
        self._lower_breakeven = self._put_strike - total_premium

    @classmethod
    def create(
        cls,
        underlying: str,
        call_strike: float,
        put_strike: float,
        expiration: datetime,
        call_price: float,
        put_price: float,
        quantity: int,
        entry_date: datetime,
        underlying_price: float,
        call_iv: Optional[float] = None,
        put_iv: Optional[float] = None,
        structure_id: Optional[str] = None
    ) -> 'LongStrangle':
        """
        Factory method to create a Long Strangle.

        Args:
            underlying: Ticker symbol (e.g., 'SPY')
            call_strike: Strike price for call (should be > put_strike)
            put_strike: Strike price for put (should be < call_strike)
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
            LongStrangle instance

        Raises:
            OptionStructureValidationError: If call_strike <= put_strike

        Example:
            >>> strangle = LongStrangle.create(
            ...     underlying='SPY',
            ...     call_strike=460.0,
            ...     put_strike=440.0,
            ...     expiration=datetime(2024, 3, 15),
            ...     call_price=3.50,
            ...     put_price=3.25,
            ...     quantity=10,
            ...     entry_date=datetime(2024, 3, 1),
            ...     underlying_price=450.0
            ... )
        """
        # Validate strike ordering
        if call_strike <= put_strike:
            raise OptionStructureValidationError(
                f"call_strike must be > put_strike. Got call: {call_strike}, put: {put_strike}"
            )

        # Create long call
        call = create_long_call(
            underlying=underlying,
            strike=call_strike,
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
            strike=put_strike,
            expiration=expiration,
            quantity=quantity,
            entry_price=put_price,
            entry_date=entry_date,
            underlying_price=underlying_price,
            implied_vol=put_iv
        )

        # Create and return strangle
        return cls(
            underlying=underlying,
            call_strike=call_strike,
            put_strike=put_strike,
            expiration=expiration,
            call_option=call,
            put_option=put,
            structure_id=structure_id,
            entry_date=entry_date
        )

    # =========================================================================
    # Properties - Strangle-specific Attributes
    # =========================================================================

    @property
    def call_strike(self) -> float:
        """Get the call strike price."""
        return self._call_strike

    @property
    def put_strike(self) -> float:
        """Get the put strike price."""
        return self._put_strike

    @property
    def strike_width(self) -> float:
        """Get the distance between strikes."""
        return self._call_strike - self._put_strike

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
        """Get maximum profit (unlimited for long strangle)."""
        return self._max_profit

    @property
    def max_loss(self) -> float:
        """Get maximum loss (total premium paid)."""
        return self._max_loss

    @property
    def upper_breakeven(self) -> float:
        """
        Get upper breakeven point.

        Upper Breakeven = Call Strike + Total Premium Paid
        """
        return self._upper_breakeven

    @property
    def lower_breakeven(self) -> float:
        """
        Get lower breakeven point.

        Lower Breakeven = Put Strike - Total Premium Paid
        """
        return self._lower_breakeven

    @property
    def breakeven_range(self) -> float:
        """Get the width of the breakeven range."""
        return self._upper_breakeven - self._lower_breakeven


# =============================================================================
# Short Strangle
# =============================================================================

class ShortStrangle(OptionStructure):
    """
    Short Strangle: Sell OTM call + sell OTM put at different strikes.

    This is a premium collection strategy that profits from low volatility.
    The position has limited profit and unlimited loss potential.

    Financial Formulas:
        Max Profit = Call Premium + Put Premium
        Max Loss = Unlimited
        Upper Breakeven = Call Strike + Total Premium
        Lower Breakeven = Put Strike - Total Premium

    Greeks:
        Delta depends on spot relative to strikes
        Gamma < 0 (loses from large moves)
        Theta > 0 (benefits from time decay)
        Vega < 0 (loses from volatility increase)

    Example:
        >>> strangle = ShortStrangle.create(
        ...     underlying='SPY',
        ...     call_strike=460.0,
        ...     put_strike=440.0,
        ...     expiration=datetime(2024, 3, 15),
        ...     call_price=3.50,
        ...     put_price=3.25,
        ...     quantity=10,
        ...     entry_date=datetime(2024, 3, 1),
        ...     underlying_price=450.0
        ... )
    """

    __slots__ = (
        '_call_strike',
        '_put_strike',
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
        call_strike: float,
        put_strike: float,
        expiration: datetime,
        call_option: Option,
        put_option: Option,
        structure_id: Optional[str] = None,
        entry_date: Optional[datetime] = None
    ) -> None:
        """
        Initialize a Short Strangle structure.

        Args:
            underlying: Ticker symbol
            call_strike: Strike price for call (higher)
            put_strike: Strike price for put (lower)
            expiration: Expiration date (same for call and put)
            call_option: Short call option
            put_option: Short put option
            structure_id: Optional unique identifier
            entry_date: Optional entry timestamp

        Raises:
            OptionStructureValidationError: If options don't match strangle requirements
        """
        # Validate inputs
        if not isinstance(call_option, Option) or not isinstance(put_option, Option):
            raise OptionStructureValidationError(
                "Both call_option and put_option must be Option instances"
            )

        # Validate that both are short positions
        if not call_option.is_short or not put_option.is_short:
            raise OptionStructureValidationError(
                "Short strangle requires both options to be short positions"
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

        # Validate strike ordering (call strike > put strike)
        if call_strike <= put_strike:
            raise OptionStructureValidationError(
                f"Strangle requires call_strike > put_strike. "
                f"Got call: {call_strike}, put: {put_strike}"
            )

        # Validate same expiration
        if call_option.expiration != put_option.expiration:
            raise OptionStructureValidationError(
                f"Strangle requires same expiration. Got call: {call_option.expiration}, "
                f"put: {put_option.expiration}"
            )

        # Validate same underlying
        if call_option.underlying != put_option.underlying:
            raise OptionStructureValidationError(
                f"Strangle requires same underlying. Got call: {call_option.underlying}, "
                f"put: {put_option.underlying}"
            )

        # Initialize base structure
        super().__init__(
            structure_type='short_strangle',
            underlying=underlying,
            structure_id=structure_id,
            entry_date=entry_date
        )

        # Store strikes
        self._call_strike = call_strike
        self._put_strike = put_strike

        # Add options to structure
        self.add_option(call_option)
        self.add_option(put_option)

        # Store references for convenience
        self._call_option = call_option
        self._put_option = put_option

        # Calculate and cache max profit/loss and breakevens
        self._calculate_metrics()

        logger.debug(
            f"Created Short Strangle: {underlying} {put_strike}/{call_strike} strikes, "
            f"Max Profit: ${self._max_profit:,.2f}, "
            f"Breakevens: ${self._lower_breakeven:.2f} - ${self._upper_breakeven:.2f}"
        )

    def _calculate_metrics(self) -> None:
        """Calculate and cache max profit, max loss, and breakeven points."""
        # Max Profit = Total premium received (net premium is positive for credit)
        self._max_profit = self.net_premium

        # Max Loss = Unlimited
        self._max_loss = float('-inf')

        # Breakevens:
        # Upper = Call Strike + Total Premium
        # Lower = Put Strike - Total Premium
        total_premium = self.net_premium / (self._call_option.quantity * 100)
        self._upper_breakeven = self._call_strike + total_premium
        self._lower_breakeven = self._put_strike - total_premium

    @classmethod
    def create(
        cls,
        underlying: str,
        call_strike: float,
        put_strike: float,
        expiration: datetime,
        call_price: float,
        put_price: float,
        quantity: int,
        entry_date: datetime,
        underlying_price: float,
        call_iv: Optional[float] = None,
        put_iv: Optional[float] = None,
        structure_id: Optional[str] = None
    ) -> 'ShortStrangle':
        """
        Factory method to create a Short Strangle.

        Args:
            underlying: Ticker symbol (e.g., 'SPY')
            call_strike: Strike price for call (should be > put_strike)
            put_strike: Strike price for put (should be < call_strike)
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
            ShortStrangle instance

        Raises:
            OptionStructureValidationError: If call_strike <= put_strike

        Example:
            >>> strangle = ShortStrangle.create(
            ...     underlying='SPY',
            ...     call_strike=460.0,
            ...     put_strike=440.0,
            ...     expiration=datetime(2024, 3, 15),
            ...     call_price=3.50,
            ...     put_price=3.25,
            ...     quantity=10,
            ...     entry_date=datetime(2024, 3, 1),
            ...     underlying_price=450.0
            ... )
        """
        # Validate strike ordering
        if call_strike <= put_strike:
            raise OptionStructureValidationError(
                f"call_strike must be > put_strike. Got call: {call_strike}, put: {put_strike}"
            )

        # Create short call
        call = create_short_call(
            underlying=underlying,
            strike=call_strike,
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
            strike=put_strike,
            expiration=expiration,
            quantity=quantity,
            entry_price=put_price,
            entry_date=entry_date,
            underlying_price=underlying_price,
            implied_vol=put_iv
        )

        # Create and return strangle
        return cls(
            underlying=underlying,
            call_strike=call_strike,
            put_strike=put_strike,
            expiration=expiration,
            call_option=call,
            put_option=put,
            structure_id=structure_id,
            entry_date=entry_date
        )

    # =========================================================================
    # Properties - Strangle-specific Attributes
    # =========================================================================

    @property
    def call_strike(self) -> float:
        """Get the call strike price."""
        return self._call_strike

    @property
    def put_strike(self) -> float:
        """Get the put strike price."""
        return self._put_strike

    @property
    def strike_width(self) -> float:
        """Get the distance between strikes."""
        return self._call_strike - self._put_strike

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
        """Get maximum profit (total premium received)."""
        return self._max_profit

    @property
    def max_loss(self) -> float:
        """Get maximum loss (unlimited for short strangle)."""
        return self._max_loss

    @property
    def upper_breakeven(self) -> float:
        """
        Get upper breakeven point.

        Upper Breakeven = Call Strike + Total Premium Received
        """
        return self._upper_breakeven

    @property
    def lower_breakeven(self) -> float:
        """
        Get lower breakeven point.

        Lower Breakeven = Put Strike - Total Premium Received
        """
        return self._lower_breakeven

    @property
    def breakeven_range(self) -> float:
        """Get the width of the breakeven range."""
        return self._upper_breakeven - self._lower_breakeven


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    'LongStrangle',
    'ShortStrangle',
]
