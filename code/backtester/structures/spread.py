"""
Spread Option Structures

This module implements vertical spread structures: Bull Call Spread, Bear Put Spread,
Bull Put Spread (credit spread), and Bear Call Spread (credit spread).

Spreads are defined-risk strategies that limit both profit and loss by combining
long and short options at different strikes.

Financial Characteristics:

    Bull Call Spread (Debit):
        - Buy lower strike call + Sell higher strike call
        - Max Profit = Spread Width - Net Debit
        - Max Loss = Net Debit
        - Breakeven = Lower Strike + Net Debit
        - Use: Moderately bullish

    Bear Put Spread (Debit):
        - Buy higher strike put + Sell lower strike put
        - Max Profit = Spread Width - Net Debit
        - Max Loss = Net Debit
        - Breakeven = Higher Strike - Net Debit
        - Use: Moderately bearish

    Bull Put Spread (Credit):
        - Sell higher strike put + Buy lower strike put
        - Max Profit = Net Credit
        - Max Loss = Spread Width - Net Credit
        - Breakeven = Higher Strike - Net Credit
        - Use: Moderately bullish, collect premium

    Bear Call Spread (Credit):
        - Sell lower strike call + Buy higher strike call
        - Max Profit = Net Credit
        - Max Loss = Spread Width - Net Credit
        - Breakeven = Lower Strike + Net Credit
        - Use: Moderately bearish, collect premium

Usage:
    from backtester.structures.spread import BullCallSpread, IronCondor
    from datetime import datetime

    # Create bull call spread
    spread = BullCallSpread.create(
        underlying='SPY',
        long_strike=450.0,
        short_strike=460.0,
        expiration=datetime(2024, 3, 15),
        long_price=6.50,
        short_price=3.00,
        quantity=10,
        entry_date=datetime(2024, 3, 1),
        underlying_price=448.0
    )

References:
    - CBOE Spreads: https://www.cboe.com/education/tools/strategies/
    - Hull, J. C. (2018). Options, Futures, and Other Derivatives.
"""

import logging
from datetime import datetime
from typing import Optional

from backtester.core.option import (
    Option,
    create_long_call,
    create_short_call,
    create_long_put,
    create_short_put,
)
from backtester.core.option_structure import (
    OptionStructure,
    OptionStructureValidationError,
)

# Configure module logger
logger = logging.getLogger(__name__)


# =============================================================================
# Bull Call Spread
# =============================================================================

class BullCallSpread(OptionStructure):
    """
    Bull Call Spread: Buy lower strike call + Sell higher strike call.

    This is a moderately bullish debit spread with defined risk and reward.

    Financial Formulas:
        Max Profit = (Short Strike - Long Strike) - Net Debit
        Max Loss = Net Debit (total premium paid)
        Breakeven = Long Strike + Net Debit

    Greeks:
        Delta > 0 (bullish, but less than long call alone)
        Gamma can be positive or negative depending on spot
        Theta moderately negative (offset by short call)
        Vega moderately positive (offset by short call)

    Example:
        >>> spread = BullCallSpread.create(
        ...     underlying='SPY',
        ...     long_strike=450.0,
        ...     short_strike=460.0,
        ...     expiration=datetime(2024, 3, 15),
        ...     long_price=6.50,
        ...     short_price=3.00,
        ...     quantity=10,
        ...     entry_date=datetime(2024, 3, 1),
        ...     underlying_price=448.0
        ... )
    """

    __slots__ = (
        '_long_strike',
        '_short_strike',
        '_long_call',
        '_short_call',
        '_max_profit',
        '_max_loss',
        '_breakeven',
    )

    def __init__(
        self,
        underlying: str,
        long_strike: float,
        short_strike: float,
        expiration: datetime,
        long_call: Option,
        short_call: Option,
        structure_id: Optional[str] = None,
        entry_date: Optional[datetime] = None
    ) -> None:
        """
        Initialize a Bull Call Spread.

        Args:
            underlying: Ticker symbol
            long_strike: Strike of long call (lower)
            short_strike: Strike of short call (higher)
            expiration: Expiration date
            long_call: Long call option
            short_call: Short call option
            structure_id: Optional unique identifier
            entry_date: Optional entry timestamp

        Raises:
            OptionStructureValidationError: If validation fails
        """
        # Validate options
        if not isinstance(long_call, Option) or not isinstance(short_call, Option):
            raise OptionStructureValidationError(
                "Both long_call and short_call must be Option instances"
            )

        if not long_call.is_long or not long_call.is_call:
            raise OptionStructureValidationError(
                "long_call must be a long call option"
            )

        if not short_call.is_short or not short_call.is_call:
            raise OptionStructureValidationError(
                "short_call must be a short call option"
            )

        # Validate strike ordering (long strike < short strike for bull call spread)
        if long_strike >= short_strike:
            raise OptionStructureValidationError(
                f"Bull call spread requires long_strike < short_strike. "
                f"Got long: {long_strike}, short: {short_strike}"
            )

        if long_call.expiration != short_call.expiration:
            raise OptionStructureValidationError(
                "Both options must have same expiration"
            )

        if long_call.underlying != short_call.underlying:
            raise OptionStructureValidationError(
                "Both options must have same underlying"
            )

        # Initialize base structure
        super().__init__(
            structure_type='bull_call_spread',
            underlying=underlying,
            structure_id=structure_id,
            entry_date=entry_date
        )

        self._long_strike = long_strike
        self._short_strike = short_strike

        # Add options
        self.add_option(long_call)
        self.add_option(short_call)

        self._long_call = long_call
        self._short_call = short_call

        # Calculate metrics
        self._calculate_metrics()

        logger.debug(
            f"Created Bull Call Spread: {underlying} {long_strike}/{short_strike}, "
            f"Max Profit: ${self._max_profit:,.2f}, Max Loss: ${self._max_loss:,.2f}"
        )

    def _calculate_metrics(self) -> None:
        """Calculate max profit, max loss, and breakeven."""
        # Net debit (negative since we pay)
        net_debit = -self.net_premium

        # Spread width
        spread_width = (self._short_strike - self._long_strike) * self._long_call.quantity * 100

        # Max Profit = Spread Width - Net Debit
        self._max_profit = spread_width - net_debit

        # Max Loss = Net Debit
        self._max_loss = net_debit

        # Breakeven = Long Strike + (Net Debit / contracts / 100)
        debit_per_share = net_debit / (self._long_call.quantity * 100)
        self._breakeven = self._long_strike + debit_per_share

    @classmethod
    def create(
        cls,
        underlying: str,
        long_strike: float,
        short_strike: float,
        expiration: datetime,
        long_price: float,
        short_price: float,
        quantity: int,
        entry_date: datetime,
        underlying_price: float,
        long_iv: Optional[float] = None,
        short_iv: Optional[float] = None,
        structure_id: Optional[str] = None
    ) -> 'BullCallSpread':
        """Factory method to create a Bull Call Spread."""
        if long_strike >= short_strike:
            raise OptionStructureValidationError(
                f"long_strike must be < short_strike. Got {long_strike} >= {short_strike}"
            )

        long_call = create_long_call(
            underlying=underlying,
            strike=long_strike,
            expiration=expiration,
            quantity=quantity,
            entry_price=long_price,
            entry_date=entry_date,
            underlying_price=underlying_price,
            implied_vol=long_iv
        )

        short_call = create_short_call(
            underlying=underlying,
            strike=short_strike,
            expiration=expiration,
            quantity=quantity,
            entry_price=short_price,
            entry_date=entry_date,
            underlying_price=underlying_price,
            implied_vol=short_iv
        )

        return cls(
            underlying=underlying,
            long_strike=long_strike,
            short_strike=short_strike,
            expiration=expiration,
            long_call=long_call,
            short_call=short_call,
            structure_id=structure_id,
            entry_date=entry_date
        )

    @property
    def long_strike(self) -> float:
        return self._long_strike

    @property
    def short_strike(self) -> float:
        return self._short_strike

    @property
    def spread_width(self) -> float:
        """Get the width of the spread (short strike - long strike)."""
        return self._short_strike - self._long_strike

    @property
    def max_profit(self) -> float:
        return self._max_profit

    @property
    def max_loss(self) -> float:
        return self._max_loss

    @property
    def breakeven(self) -> float:
        return self._breakeven


# =============================================================================
# Bear Put Spread
# =============================================================================

class BearPutSpread(OptionStructure):
    """
    Bear Put Spread: Buy higher strike put + Sell lower strike put.

    This is a moderately bearish debit spread with defined risk and reward.

    Financial Formulas:
        Max Profit = (Long Strike - Short Strike) - Net Debit
        Max Loss = Net Debit
        Breakeven = Long Strike - Net Debit

    Example:
        >>> spread = BearPutSpread.create(
        ...     underlying='SPY',
        ...     long_strike=450.0,
        ...     short_strike=440.0,
        ...     expiration=datetime(2024, 3, 15),
        ...     long_price=6.50,
        ...     short_price=3.00,
        ...     quantity=10,
        ...     entry_date=datetime(2024, 3, 1),
        ...     underlying_price=448.0
        ... )
    """

    __slots__ = (
        '_long_strike',
        '_short_strike',
        '_long_put',
        '_short_put',
        '_max_profit',
        '_max_loss',
        '_breakeven',
    )

    def __init__(
        self,
        underlying: str,
        long_strike: float,
        short_strike: float,
        expiration: datetime,
        long_put: Option,
        short_put: Option,
        structure_id: Optional[str] = None,
        entry_date: Optional[datetime] = None
    ) -> None:
        """Initialize a Bear Put Spread."""
        # Validate options
        if not isinstance(long_put, Option) or not isinstance(short_put, Option):
            raise OptionStructureValidationError(
                "Both long_put and short_put must be Option instances"
            )

        if not long_put.is_long or not long_put.is_put:
            raise OptionStructureValidationError(
                "long_put must be a long put option"
            )

        if not short_put.is_short or not short_put.is_put:
            raise OptionStructureValidationError(
                "short_put must be a short put option"
            )

        # Validate strike ordering (long strike > short strike for bear put spread)
        if long_strike <= short_strike:
            raise OptionStructureValidationError(
                f"Bear put spread requires long_strike > short_strike. "
                f"Got long: {long_strike}, short: {short_strike}"
            )

        if long_put.expiration != short_put.expiration:
            raise OptionStructureValidationError(
                "Both options must have same expiration"
            )

        if long_put.underlying != short_put.underlying:
            raise OptionStructureValidationError(
                "Both options must have same underlying"
            )

        # Initialize base structure
        super().__init__(
            structure_type='bear_put_spread',
            underlying=underlying,
            structure_id=structure_id,
            entry_date=entry_date
        )

        self._long_strike = long_strike
        self._short_strike = short_strike

        # Add options
        self.add_option(long_put)
        self.add_option(short_put)

        self._long_put = long_put
        self._short_put = short_put

        # Calculate metrics
        self._calculate_metrics()

        logger.debug(
            f"Created Bear Put Spread: {underlying} {long_strike}/{short_strike}, "
            f"Max Profit: ${self._max_profit:,.2f}, Max Loss: ${self._max_loss:,.2f}"
        )

    def _calculate_metrics(self) -> None:
        """Calculate max profit, max loss, and breakeven."""
        # Net debit (negative since we pay)
        net_debit = -self.net_premium

        # Spread width
        spread_width = (self._long_strike - self._short_strike) * self._long_put.quantity * 100

        # Max Profit = Spread Width - Net Debit
        self._max_profit = spread_width - net_debit

        # Max Loss = Net Debit
        self._max_loss = net_debit

        # Breakeven = Long Strike - (Net Debit / contracts / 100)
        debit_per_share = net_debit / (self._long_put.quantity * 100)
        self._breakeven = self._long_strike - debit_per_share

    @classmethod
    def create(
        cls,
        underlying: str,
        long_strike: float,
        short_strike: float,
        expiration: datetime,
        long_price: float,
        short_price: float,
        quantity: int,
        entry_date: datetime,
        underlying_price: float,
        long_iv: Optional[float] = None,
        short_iv: Optional[float] = None,
        structure_id: Optional[str] = None
    ) -> 'BearPutSpread':
        """Factory method to create a Bear Put Spread."""
        if long_strike <= short_strike:
            raise OptionStructureValidationError(
                f"long_strike must be > short_strike. Got {long_strike} <= {short_strike}"
            )

        long_put = create_long_put(
            underlying=underlying,
            strike=long_strike,
            expiration=expiration,
            quantity=quantity,
            entry_price=long_price,
            entry_date=entry_date,
            underlying_price=underlying_price,
            implied_vol=long_iv
        )

        short_put = create_short_put(
            underlying=underlying,
            strike=short_strike,
            expiration=expiration,
            quantity=quantity,
            entry_price=short_price,
            entry_date=entry_date,
            underlying_price=underlying_price,
            implied_vol=short_iv
        )

        return cls(
            underlying=underlying,
            long_strike=long_strike,
            short_strike=short_strike,
            expiration=expiration,
            long_put=long_put,
            short_put=short_put,
            structure_id=structure_id,
            entry_date=entry_date
        )

    @property
    def long_strike(self) -> float:
        return self._long_strike

    @property
    def short_strike(self) -> float:
        return self._short_strike

    @property
    def spread_width(self) -> float:
        """Get the width of the spread (long strike - short strike)."""
        return self._long_strike - self._short_strike

    @property
    def max_profit(self) -> float:
        return self._max_profit

    @property
    def max_loss(self) -> float:
        return self._max_loss

    @property
    def breakeven(self) -> float:
        return self._breakeven


# =============================================================================
# Bull Put Spread (Credit Spread)
# =============================================================================

class BullPutSpread(OptionStructure):
    """
    Bull Put Spread: Sell higher strike put + Buy lower strike put.

    This is a moderately bullish credit spread with defined risk and reward.

    Financial Formulas:
        Max Profit = Net Credit
        Max Loss = (Short Strike - Long Strike) - Net Credit
        Breakeven = Short Strike - Net Credit

    Example:
        >>> spread = BullPutSpread.create(
        ...     underlying='SPY',
        ...     long_strike=440.0,
        ...     short_strike=450.0,
        ...     expiration=datetime(2024, 3, 15),
        ...     long_price=3.00,
        ...     short_price=6.50,
        ...     quantity=10,
        ...     entry_date=datetime(2024, 3, 1),
        ...     underlying_price=452.0
        ... )
    """

    __slots__ = (
        '_long_strike',
        '_short_strike',
        '_long_put',
        '_short_put',
        '_max_profit',
        '_max_loss',
        '_breakeven',
    )

    def __init__(
        self,
        underlying: str,
        long_strike: float,
        short_strike: float,
        expiration: datetime,
        long_put: Option,
        short_put: Option,
        structure_id: Optional[str] = None,
        entry_date: Optional[datetime] = None
    ) -> None:
        """Initialize a Bull Put Spread."""
        # Validate options
        if not isinstance(long_put, Option) or not isinstance(short_put, Option):
            raise OptionStructureValidationError(
                "Both long_put and short_put must be Option instances"
            )

        if not long_put.is_long or not long_put.is_put:
            raise OptionStructureValidationError(
                "long_put must be a long put option"
            )

        if not short_put.is_short or not short_put.is_put:
            raise OptionStructureValidationError(
                "short_put must be a short put option"
            )

        # Validate strike ordering (short strike > long strike)
        if short_strike <= long_strike:
            raise OptionStructureValidationError(
                f"Bull put spread requires short_strike > long_strike. "
                f"Got short: {short_strike}, long: {long_strike}"
            )

        if long_put.expiration != short_put.expiration:
            raise OptionStructureValidationError(
                "Both options must have same expiration"
            )

        if long_put.underlying != short_put.underlying:
            raise OptionStructureValidationError(
                "Both options must have same underlying"
            )

        # Initialize base structure
        super().__init__(
            structure_type='bull_put_spread',
            underlying=underlying,
            structure_id=structure_id,
            entry_date=entry_date
        )

        self._long_strike = long_strike
        self._short_strike = short_strike

        # Add options
        self.add_option(long_put)
        self.add_option(short_put)

        self._long_put = long_put
        self._short_put = short_put

        # Calculate metrics
        self._calculate_metrics()

        logger.debug(
            f"Created Bull Put Spread: {underlying} {long_strike}/{short_strike}, "
            f"Max Profit: ${self._max_profit:,.2f}, Max Loss: ${self._max_loss:,.2f}"
        )

    def _calculate_metrics(self) -> None:
        """Calculate max profit, max loss, and breakeven."""
        # Net credit (positive since we receive)
        net_credit = self.net_premium

        # Max Profit = Net Credit
        self._max_profit = net_credit

        # Spread width
        spread_width = (self._short_strike - self._long_strike) * self._long_put.quantity * 100

        # Max Loss = Spread Width - Net Credit
        self._max_loss = spread_width - net_credit

        # Breakeven = Short Strike - (Net Credit / contracts / 100)
        credit_per_share = net_credit / (self._long_put.quantity * 100)
        self._breakeven = self._short_strike - credit_per_share

    @classmethod
    def create(
        cls,
        underlying: str,
        long_strike: float,
        short_strike: float,
        expiration: datetime,
        long_price: float,
        short_price: float,
        quantity: int,
        entry_date: datetime,
        underlying_price: float,
        long_iv: Optional[float] = None,
        short_iv: Optional[float] = None,
        structure_id: Optional[str] = None
    ) -> 'BullPutSpread':
        """Factory method to create a Bull Put Spread."""
        if short_strike <= long_strike:
            raise OptionStructureValidationError(
                f"short_strike must be > long_strike. Got {short_strike} <= {long_strike}"
            )

        long_put = create_long_put(
            underlying=underlying,
            strike=long_strike,
            expiration=expiration,
            quantity=quantity,
            entry_price=long_price,
            entry_date=entry_date,
            underlying_price=underlying_price,
            implied_vol=long_iv
        )

        short_put = create_short_put(
            underlying=underlying,
            strike=short_strike,
            expiration=expiration,
            quantity=quantity,
            entry_price=short_price,
            entry_date=entry_date,
            underlying_price=underlying_price,
            implied_vol=short_iv
        )

        return cls(
            underlying=underlying,
            long_strike=long_strike,
            short_strike=short_strike,
            expiration=expiration,
            long_put=long_put,
            short_put=short_put,
            structure_id=structure_id,
            entry_date=entry_date
        )

    @property
    def long_strike(self) -> float:
        return self._long_strike

    @property
    def short_strike(self) -> float:
        return self._short_strike

    @property
    def spread_width(self) -> float:
        """Get the width of the spread (short strike - long strike)."""
        return self._short_strike - self._long_strike

    @property
    def max_profit(self) -> float:
        return self._max_profit

    @property
    def max_loss(self) -> float:
        return self._max_loss

    @property
    def breakeven(self) -> float:
        return self._breakeven


# =============================================================================
# Bear Call Spread (Credit Spread)
# =============================================================================

class BearCallSpread(OptionStructure):
    """
    Bear Call Spread: Sell lower strike call + Buy higher strike call.

    This is a moderately bearish credit spread with defined risk and reward.

    Financial Formulas:
        Max Profit = Net Credit
        Max Loss = (Long Strike - Short Strike) - Net Credit
        Breakeven = Short Strike + Net Credit

    Example:
        >>> spread = BearCallSpread.create(
        ...     underlying='SPY',
        ...     long_strike=460.0,
        ...     short_strike=450.0,
        ...     expiration=datetime(2024, 3, 15),
        ...     long_price=3.00,
        ...     short_price=6.50,
        ...     quantity=10,
        ...     entry_date=datetime(2024, 3, 1),
        ...     underlying_price=448.0
        ... )
    """

    __slots__ = (
        '_long_strike',
        '_short_strike',
        '_long_call',
        '_short_call',
        '_max_profit',
        '_max_loss',
        '_breakeven',
    )

    def __init__(
        self,
        underlying: str,
        long_strike: float,
        short_strike: float,
        expiration: datetime,
        long_call: Option,
        short_call: Option,
        structure_id: Optional[str] = None,
        entry_date: Optional[datetime] = None
    ) -> None:
        """Initialize a Bear Call Spread."""
        # Validate options
        if not isinstance(long_call, Option) or not isinstance(short_call, Option):
            raise OptionStructureValidationError(
                "Both long_call and short_call must be Option instances"
            )

        if not long_call.is_long or not long_call.is_call:
            raise OptionStructureValidationError(
                "long_call must be a long call option"
            )

        if not short_call.is_short or not short_call.is_call:
            raise OptionStructureValidationError(
                "short_call must be a short call option"
            )

        # Validate strike ordering (long strike > short strike)
        if long_strike <= short_strike:
            raise OptionStructureValidationError(
                f"Bear call spread requires long_strike > short_strike. "
                f"Got long: {long_strike}, short: {short_strike}"
            )

        if long_call.expiration != short_call.expiration:
            raise OptionStructureValidationError(
                "Both options must have same expiration"
            )

        if long_call.underlying != short_call.underlying:
            raise OptionStructureValidationError(
                "Both options must have same underlying"
            )

        # Initialize base structure
        super().__init__(
            structure_type='bear_call_spread',
            underlying=underlying,
            structure_id=structure_id,
            entry_date=entry_date
        )

        self._long_strike = long_strike
        self._short_strike = short_strike

        # Add options
        self.add_option(long_call)
        self.add_option(short_call)

        self._long_call = long_call
        self._short_call = short_call

        # Calculate metrics
        self._calculate_metrics()

        logger.debug(
            f"Created Bear Call Spread: {underlying} {short_strike}/{long_strike}, "
            f"Max Profit: ${self._max_profit:,.2f}, Max Loss: ${self._max_loss:,.2f}"
        )

    def _calculate_metrics(self) -> None:
        """Calculate max profit, max loss, and breakeven."""
        # Net credit (positive since we receive)
        net_credit = self.net_premium

        # Max Profit = Net Credit
        self._max_profit = net_credit

        # Spread width
        spread_width = (self._long_strike - self._short_strike) * self._long_call.quantity * 100

        # Max Loss = Spread Width - Net Credit
        self._max_loss = spread_width - net_credit

        # Breakeven = Short Strike + (Net Credit / contracts / 100)
        credit_per_share = net_credit / (self._long_call.quantity * 100)
        self._breakeven = self._short_strike + credit_per_share

    @classmethod
    def create(
        cls,
        underlying: str,
        long_strike: float,
        short_strike: float,
        expiration: datetime,
        long_price: float,
        short_price: float,
        quantity: int,
        entry_date: datetime,
        underlying_price: float,
        long_iv: Optional[float] = None,
        short_iv: Optional[float] = None,
        structure_id: Optional[str] = None
    ) -> 'BearCallSpread':
        """Factory method to create a Bear Call Spread."""
        if long_strike <= short_strike:
            raise OptionStructureValidationError(
                f"long_strike must be > short_strike. Got {long_strike} <= {short_strike}"
            )

        long_call = create_long_call(
            underlying=underlying,
            strike=long_strike,
            expiration=expiration,
            quantity=quantity,
            entry_price=long_price,
            entry_date=entry_date,
            underlying_price=underlying_price,
            implied_vol=long_iv
        )

        short_call = create_short_call(
            underlying=underlying,
            strike=short_strike,
            expiration=expiration,
            quantity=quantity,
            entry_price=short_price,
            entry_date=entry_date,
            underlying_price=underlying_price,
            implied_vol=short_iv
        )

        return cls(
            underlying=underlying,
            long_strike=long_strike,
            short_strike=short_strike,
            expiration=expiration,
            long_call=long_call,
            short_call=short_call,
            structure_id=structure_id,
            entry_date=entry_date
        )

    @property
    def long_strike(self) -> float:
        return self._long_strike

    @property
    def short_strike(self) -> float:
        return self._short_strike

    @property
    def spread_width(self) -> float:
        """Get the width of the spread (long strike - short strike)."""
        return self._long_strike - self._short_strike

    @property
    def max_profit(self) -> float:
        return self._max_profit

    @property
    def max_loss(self) -> float:
        return self._max_loss

    @property
    def breakeven(self) -> float:
        return self._breakeven


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    'BullCallSpread',
    'BearPutSpread',
    'BullPutSpread',
    'BearCallSpread',
]
