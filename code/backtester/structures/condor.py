"""
Condor Option Structures

This module implements Iron Condor and Iron Butterfly structures. These are
advanced premium collection strategies that combine credit spreads to create
defined-risk positions.

Financial Characteristics:

    Iron Condor:
        - Sell OTM put + Buy lower OTM put + Sell OTM call + Buy higher OTM call
        - Four different strikes (all ordered: buy put < sell put < sell call < buy call)
        - Max Profit = Net Credit
        - Max Loss = Wing Width - Net Credit
        - Two Breakevens: Sell Put - Credit, Sell Call + Credit
        - Use: Range-bound market, collect premium

    Iron Butterfly:
        - Sell ATM put + Sell ATM call + Buy OTM put + Buy OTM call
        - Three strikes: Lower (buy put), Middle (sell both), Higher (buy call)
        - Max Profit = Net Credit
        - Max Loss = Wing Width - Net Credit
        - Two Breakevens: Middle - Credit, Middle + Credit
        - Use: Expect price to stay near middle strike

Greeks:
    - Near-neutral delta at initiation
    - Negative gamma (loses from large moves)
    - Positive theta (benefits from time decay)
    - Negative vega (loses from IV increase)

Usage:
    from backtester.structures.condor import IronCondor, IronButterfly
    from datetime import datetime

    # Create iron condor
    ic = IronCondor.create(
        underlying='SPY',
        put_buy_strike=430.0,
        put_sell_strike=440.0,
        call_sell_strike=460.0,
        call_buy_strike=470.0,
        expiration=datetime(2024, 3, 15),
        put_buy_price=1.50,
        put_sell_price=3.00,
        call_sell_price=3.25,
        call_buy_price=1.75,
        quantity=10,
        entry_date=datetime(2024, 3, 1),
        underlying_price=450.0
    )

References:
    - CBOE Iron Condor: https://www.cboe.com/education/tools/iron-condor/
    - OptionsPlaybook: https://www.optionsplaybook.com/option-strategies/
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
# Iron Condor
# =============================================================================

class IronCondor(OptionStructure):
    """
    Iron Condor: Combines bull put spread and bear call spread.

    Structure: Buy put + Sell put + Sell call + Buy call (4 different strikes)

    Financial Formulas:
        Max Profit = Net Credit
        Max Loss = Wing Width - Net Credit
        Lower Breakeven = Short Put Strike - Net Credit
        Upper Breakeven = Short Call Strike + Net Credit

    Greeks:
        Delta ≈ 0 (neutral positioning)
        Gamma < 0 (short gamma, loses from volatility)
        Theta > 0 (benefits from time decay)
        Vega < 0 (loses from volatility expansion)

    Example:
        >>> ic = IronCondor.create(
        ...     underlying='SPY',
        ...     put_buy_strike=430.0,
        ...     put_sell_strike=440.0,
        ...     call_sell_strike=460.0,
        ...     call_buy_strike=470.0,
        ...     expiration=datetime(2024, 3, 15),
        ...     put_buy_price=1.50,
        ...     put_sell_price=3.00,
        ...     call_sell_price=3.25,
        ...     call_buy_price=1.75,
        ...     quantity=10,
        ...     entry_date=datetime(2024, 3, 1),
        ...     underlying_price=450.0
        ... )
    """

    __slots__ = (
        '_put_buy_strike',
        '_put_sell_strike',
        '_call_sell_strike',
        '_call_buy_strike',
        '_put_buy',
        '_put_sell',
        '_call_sell',
        '_call_buy',
        '_max_profit',
        '_max_loss',
        '_upper_breakeven',
        '_lower_breakeven',
    )

    def __init__(
        self,
        underlying: str,
        put_buy_strike: float,
        put_sell_strike: float,
        call_sell_strike: float,
        call_buy_strike: float,
        expiration: datetime,
        put_buy: Option,
        put_sell: Option,
        call_sell: Option,
        call_buy: Option,
        structure_id: Optional[str] = None,
        entry_date: Optional[datetime] = None
    ) -> None:
        """
        Initialize an Iron Condor.

        Args:
            underlying: Ticker symbol
            put_buy_strike: Long put strike (lowest)
            put_sell_strike: Short put strike
            call_sell_strike: Short call strike
            call_buy_strike: Long call strike (highest)
            expiration: Expiration date (same for all)
            put_buy: Long put option
            put_sell: Short put option
            call_sell: Short call option
            call_buy: Long call option
            structure_id: Optional unique identifier
            entry_date: Optional entry timestamp

        Raises:
            OptionStructureValidationError: If validation fails
        """
        # Validate all options are Option instances
        if not all(isinstance(opt, Option) for opt in [put_buy, put_sell, call_sell, call_buy]):
            raise OptionStructureValidationError(
                "All four legs must be Option instances"
            )

        # Validate option types and positions
        if not (put_buy.is_long and put_buy.is_put):
            raise OptionStructureValidationError("put_buy must be a long put")
        if not (put_sell.is_short and put_sell.is_put):
            raise OptionStructureValidationError("put_sell must be a short put")
        if not (call_sell.is_short and call_sell.is_call):
            raise OptionStructureValidationError("call_sell must be a short call")
        if not (call_buy.is_long and call_buy.is_call):
            raise OptionStructureValidationError("call_buy must be a long call")

        # Validate strike ordering: put_buy < put_sell < call_sell < call_buy
        if not (put_buy_strike < put_sell_strike < call_sell_strike < call_buy_strike):
            raise OptionStructureValidationError(
                f"Strikes must be ordered: put_buy < put_sell < call_sell < call_buy. "
                f"Got: {put_buy_strike} < {put_sell_strike} < {call_sell_strike} < {call_buy_strike}"
            )

        # Validate same expiration
        expirations = [put_buy.expiration, put_sell.expiration, call_sell.expiration, call_buy.expiration]
        if not all(exp == expirations[0] for exp in expirations):
            raise OptionStructureValidationError(
                "All options must have same expiration"
            )

        # Validate same underlying
        underlyings = [put_buy.underlying, put_sell.underlying, call_sell.underlying, call_buy.underlying]
        if not all(u == underlyings[0] for u in underlyings):
            raise OptionStructureValidationError(
                "All options must have same underlying"
            )

        # Initialize base structure
        super().__init__(
            structure_type='iron_condor',
            underlying=underlying,
            structure_id=structure_id,
            entry_date=entry_date
        )

        # Store strikes
        self._put_buy_strike = put_buy_strike
        self._put_sell_strike = put_sell_strike
        self._call_sell_strike = call_sell_strike
        self._call_buy_strike = call_buy_strike

        # Add all legs
        self.add_option(put_buy)
        self.add_option(put_sell)
        self.add_option(call_sell)
        self.add_option(call_buy)

        # Store references
        self._put_buy = put_buy
        self._put_sell = put_sell
        self._call_sell = call_sell
        self._call_buy = call_buy

        # Calculate metrics
        self._calculate_metrics()

        logger.debug(
            f"Created Iron Condor: {underlying} "
            f"{put_buy_strike}/{put_sell_strike}/{call_sell_strike}/{call_buy_strike}, "
            f"Max Profit: ${self._max_profit:,.2f}, Max Loss: ${self._max_loss:,.2f}"
        )

    def _calculate_metrics(self) -> None:
        """Calculate max profit, max loss, and breakeven points."""
        # Net credit (should be positive)
        net_credit = self.net_premium

        # Max Profit = Net Credit
        self._max_profit = net_credit

        # Wing widths (should be equal for standard iron condor)
        put_wing = (self._put_sell_strike - self._put_buy_strike) * self._put_buy.quantity * 100
        call_wing = (self._call_buy_strike - self._call_sell_strike) * self._call_buy.quantity * 100

        # Max Loss = Larger wing width - Net Credit
        wing_width = max(put_wing, call_wing)
        self._max_loss = wing_width - net_credit

        # Breakevens
        credit_per_share = net_credit / (self._put_buy.quantity * 100)
        self._lower_breakeven = self._put_sell_strike - credit_per_share
        self._upper_breakeven = self._call_sell_strike + credit_per_share

    @classmethod
    def create(
        cls,
        underlying: str,
        put_buy_strike: float,
        put_sell_strike: float,
        call_sell_strike: float,
        call_buy_strike: float,
        expiration: datetime,
        put_buy_price: float,
        put_sell_price: float,
        call_sell_price: float,
        call_buy_price: float,
        quantity: int,
        entry_date: datetime,
        underlying_price: float,
        put_buy_iv: Optional[float] = None,
        put_sell_iv: Optional[float] = None,
        call_sell_iv: Optional[float] = None,
        call_buy_iv: Optional[float] = None,
        structure_id: Optional[str] = None
    ) -> 'IronCondor':
        """
        Factory method to create an Iron Condor.

        Args:
            underlying: Ticker symbol
            put_buy_strike: Long put strike (lowest)
            put_sell_strike: Short put strike
            call_sell_strike: Short call strike
            call_buy_strike: Long call strike (highest)
            expiration: Expiration date
            put_buy_price: Long put premium
            put_sell_price: Short put premium
            call_sell_price: Short call premium
            call_buy_price: Long call premium
            quantity: Number of contracts (same for all legs)
            entry_date: Entry timestamp
            underlying_price: Spot price at entry
            put_buy_iv: Optional long put IV
            put_sell_iv: Optional short put IV
            call_sell_iv: Optional short call IV
            call_buy_iv: Optional long call IV
            structure_id: Optional unique identifier

        Returns:
            IronCondor instance

        Raises:
            OptionStructureValidationError: If strikes not properly ordered
        """
        # Validate strike ordering
        if not (put_buy_strike < put_sell_strike < call_sell_strike < call_buy_strike):
            raise OptionStructureValidationError(
                f"Strikes must be ordered: put_buy < put_sell < call_sell < call_buy. "
                f"Got: {put_buy_strike} < {put_sell_strike} < {call_sell_strike} < {call_buy_strike}"
            )

        # Create all four legs
        put_buy = create_long_put(
            underlying=underlying,
            strike=put_buy_strike,
            expiration=expiration,
            quantity=quantity,
            entry_price=put_buy_price,
            entry_date=entry_date,
            underlying_price=underlying_price,
            implied_vol=put_buy_iv
        )

        put_sell = create_short_put(
            underlying=underlying,
            strike=put_sell_strike,
            expiration=expiration,
            quantity=quantity,
            entry_price=put_sell_price,
            entry_date=entry_date,
            underlying_price=underlying_price,
            implied_vol=put_sell_iv
        )

        call_sell = create_short_call(
            underlying=underlying,
            strike=call_sell_strike,
            expiration=expiration,
            quantity=quantity,
            entry_price=call_sell_price,
            entry_date=entry_date,
            underlying_price=underlying_price,
            implied_vol=call_sell_iv
        )

        call_buy = create_long_call(
            underlying=underlying,
            strike=call_buy_strike,
            expiration=expiration,
            quantity=quantity,
            entry_price=call_buy_price,
            entry_date=entry_date,
            underlying_price=underlying_price,
            implied_vol=call_buy_iv
        )

        return cls(
            underlying=underlying,
            put_buy_strike=put_buy_strike,
            put_sell_strike=put_sell_strike,
            call_sell_strike=call_sell_strike,
            call_buy_strike=call_buy_strike,
            expiration=expiration,
            put_buy=put_buy,
            put_sell=put_sell,
            call_sell=call_sell,
            call_buy=call_buy,
            structure_id=structure_id,
            entry_date=entry_date
        )

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def put_buy_strike(self) -> float:
        return self._put_buy_strike

    @property
    def put_sell_strike(self) -> float:
        return self._put_sell_strike

    @property
    def call_sell_strike(self) -> float:
        return self._call_sell_strike

    @property
    def call_buy_strike(self) -> float:
        return self._call_buy_strike

    @property
    def put_wing_width(self) -> float:
        """Width of put spread."""
        return self._put_sell_strike - self._put_buy_strike

    @property
    def call_wing_width(self) -> float:
        """Width of call spread."""
        return self._call_buy_strike - self._call_sell_strike

    @property
    def body_width(self) -> float:
        """Width between short strikes (profit zone)."""
        return self._call_sell_strike - self._put_sell_strike

    @property
    def max_profit(self) -> float:
        return self._max_profit

    @property
    def max_loss(self) -> float:
        return self._max_loss

    @property
    def upper_breakeven(self) -> float:
        return self._upper_breakeven

    @property
    def lower_breakeven(self) -> float:
        return self._lower_breakeven

    @property
    def breakeven_range(self) -> float:
        """Width of the breakeven range."""
        return self._upper_breakeven - self._lower_breakeven


# =============================================================================
# Iron Butterfly
# =============================================================================

class IronButterfly(OptionStructure):
    """
    Iron Butterfly: Sell ATM straddle + Buy OTM strangle for protection.

    Structure: Buy put + Sell put + Sell call + Buy call (3 strikes: lower, middle, upper)

    Financial Formulas:
        Max Profit = Net Credit
        Max Loss = Wing Width - Net Credit
        Lower Breakeven = Middle Strike - Net Credit
        Upper Breakeven = Middle Strike + Net Credit

    Greeks:
        Delta ≈ 0 (neutral at middle strike)
        Gamma < 0 (short gamma position)
        Theta > 0 (benefits from time decay)
        Vega < 0 (loses from volatility expansion)

    Example:
        >>> ib = IronButterfly.create(
        ...     underlying='SPY',
        ...     lower_strike=440.0,
        ...     middle_strike=450.0,
        ...     upper_strike=460.0,
        ...     expiration=datetime(2024, 3, 15),
        ...     lower_price=2.00,
        ...     middle_put_price=6.50,
        ...     middle_call_price=6.75,
        ...     upper_price=2.25,
        ...     quantity=10,
        ...     entry_date=datetime(2024, 3, 1),
        ...     underlying_price=450.0
        ... )
    """

    __slots__ = (
        '_lower_strike',
        '_middle_strike',
        '_upper_strike',
        '_lower_put',
        '_middle_put',
        '_middle_call',
        '_upper_call',
        '_max_profit',
        '_max_loss',
        '_upper_breakeven',
        '_lower_breakeven',
    )

    def __init__(
        self,
        underlying: str,
        lower_strike: float,
        middle_strike: float,
        upper_strike: float,
        expiration: datetime,
        lower_put: Option,
        middle_put: Option,
        middle_call: Option,
        upper_call: Option,
        structure_id: Optional[str] = None,
        entry_date: Optional[datetime] = None
    ) -> None:
        """
        Initialize an Iron Butterfly.

        Args:
            underlying: Ticker symbol
            lower_strike: Long put strike (lowest)
            middle_strike: Short put and call strike (ATM)
            upper_strike: Long call strike (highest)
            expiration: Expiration date
            lower_put: Long put option
            middle_put: Short put option
            middle_call: Short call option
            upper_call: Long call option
            structure_id: Optional unique identifier
            entry_date: Optional entry timestamp

        Raises:
            OptionStructureValidationError: If validation fails
        """
        # Validate all options
        if not all(isinstance(opt, Option) for opt in [lower_put, middle_put, middle_call, upper_call]):
            raise OptionStructureValidationError(
                "All four legs must be Option instances"
            )

        # Validate option types and positions
        if not (lower_put.is_long and lower_put.is_put):
            raise OptionStructureValidationError("lower_put must be a long put")
        if not (middle_put.is_short and middle_put.is_put):
            raise OptionStructureValidationError("middle_put must be a short put")
        if not (middle_call.is_short and middle_call.is_call):
            raise OptionStructureValidationError("middle_call must be a short call")
        if not (upper_call.is_long and upper_call.is_call):
            raise OptionStructureValidationError("upper_call must be a long call")

        # Validate strike ordering: lower < middle < upper
        if not (lower_strike < middle_strike < upper_strike):
            raise OptionStructureValidationError(
                f"Strikes must be ordered: lower < middle < upper. "
                f"Got: {lower_strike} < {middle_strike} < {upper_strike}"
            )

        # Validate middle strikes match
        if abs(middle_put.strike - middle_call.strike) > 1e-6:
            raise OptionStructureValidationError(
                f"Middle put and call must have same strike. "
                f"Got put: {middle_put.strike}, call: {middle_call.strike}"
            )

        # Validate same expiration
        expirations = [lower_put.expiration, middle_put.expiration, middle_call.expiration, upper_call.expiration]
        if not all(exp == expirations[0] for exp in expirations):
            raise OptionStructureValidationError(
                "All options must have same expiration"
            )

        # Validate same underlying
        underlyings = [lower_put.underlying, middle_put.underlying, middle_call.underlying, upper_call.underlying]
        if not all(u == underlyings[0] for u in underlyings):
            raise OptionStructureValidationError(
                "All options must have same underlying"
            )

        # Initialize base structure
        super().__init__(
            structure_type='iron_butterfly',
            underlying=underlying,
            structure_id=structure_id,
            entry_date=entry_date
        )

        # Store strikes
        self._lower_strike = lower_strike
        self._middle_strike = middle_strike
        self._upper_strike = upper_strike

        # Add all legs
        self.add_option(lower_put)
        self.add_option(middle_put)
        self.add_option(middle_call)
        self.add_option(upper_call)

        # Store references
        self._lower_put = lower_put
        self._middle_put = middle_put
        self._middle_call = middle_call
        self._upper_call = upper_call

        # Calculate metrics
        self._calculate_metrics()

        logger.debug(
            f"Created Iron Butterfly: {underlying} "
            f"{lower_strike}/{middle_strike}/{upper_strike}, "
            f"Max Profit: ${self._max_profit:,.2f}, Max Loss: ${self._max_loss:,.2f}"
        )

    def _calculate_metrics(self) -> None:
        """Calculate max profit, max loss, and breakeven points."""
        # Net credit (should be positive)
        net_credit = self.net_premium

        # Max Profit = Net Credit
        self._max_profit = net_credit

        # Wing widths (should be equal for standard iron butterfly)
        put_wing = (self._middle_strike - self._lower_strike) * self._lower_put.quantity * 100
        call_wing = (self._upper_strike - self._middle_strike) * self._upper_call.quantity * 100

        # Max Loss = Wing width - Net Credit
        wing_width = max(put_wing, call_wing)
        self._max_loss = wing_width - net_credit

        # Breakevens (symmetrical around middle strike for standard butterfly)
        credit_per_share = net_credit / (self._lower_put.quantity * 100)
        self._lower_breakeven = self._middle_strike - credit_per_share
        self._upper_breakeven = self._middle_strike + credit_per_share

    @classmethod
    def create(
        cls,
        underlying: str,
        lower_strike: float,
        middle_strike: float,
        upper_strike: float,
        expiration: datetime,
        lower_price: float,
        middle_put_price: float,
        middle_call_price: float,
        upper_price: float,
        quantity: int,
        entry_date: datetime,
        underlying_price: float,
        lower_iv: Optional[float] = None,
        middle_put_iv: Optional[float] = None,
        middle_call_iv: Optional[float] = None,
        upper_iv: Optional[float] = None,
        structure_id: Optional[str] = None
    ) -> 'IronButterfly':
        """
        Factory method to create an Iron Butterfly.

        Args:
            underlying: Ticker symbol
            lower_strike: Long put strike (lowest)
            middle_strike: Short put and call strike (ATM)
            upper_strike: Long call strike (highest)
            expiration: Expiration date
            lower_price: Long put premium
            middle_put_price: Short put premium
            middle_call_price: Short call premium
            upper_price: Long call premium
            quantity: Number of contracts
            entry_date: Entry timestamp
            underlying_price: Spot price at entry
            lower_iv: Optional long put IV
            middle_put_iv: Optional short put IV
            middle_call_iv: Optional short call IV
            upper_iv: Optional long call IV
            structure_id: Optional unique identifier

        Returns:
            IronButterfly instance

        Raises:
            OptionStructureValidationError: If strikes not properly ordered
        """
        # Validate strike ordering
        if not (lower_strike < middle_strike < upper_strike):
            raise OptionStructureValidationError(
                f"Strikes must be ordered: lower < middle < upper. "
                f"Got: {lower_strike} < {middle_strike} < {upper_strike}"
            )

        # Create all four legs
        lower_put = create_long_put(
            underlying=underlying,
            strike=lower_strike,
            expiration=expiration,
            quantity=quantity,
            entry_price=lower_price,
            entry_date=entry_date,
            underlying_price=underlying_price,
            implied_vol=lower_iv
        )

        middle_put = create_short_put(
            underlying=underlying,
            strike=middle_strike,
            expiration=expiration,
            quantity=quantity,
            entry_price=middle_put_price,
            entry_date=entry_date,
            underlying_price=underlying_price,
            implied_vol=middle_put_iv
        )

        middle_call = create_short_call(
            underlying=underlying,
            strike=middle_strike,
            expiration=expiration,
            quantity=quantity,
            entry_price=middle_call_price,
            entry_date=entry_date,
            underlying_price=underlying_price,
            implied_vol=middle_call_iv
        )

        upper_call = create_long_call(
            underlying=underlying,
            strike=upper_strike,
            expiration=expiration,
            quantity=quantity,
            entry_price=upper_price,
            entry_date=entry_date,
            underlying_price=underlying_price,
            implied_vol=upper_iv
        )

        return cls(
            underlying=underlying,
            lower_strike=lower_strike,
            middle_strike=middle_strike,
            upper_strike=upper_strike,
            expiration=expiration,
            lower_put=lower_put,
            middle_put=middle_put,
            middle_call=middle_call,
            upper_call=upper_call,
            structure_id=structure_id,
            entry_date=entry_date
        )

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def lower_strike(self) -> float:
        return self._lower_strike

    @property
    def middle_strike(self) -> float:
        return self._middle_strike

    @property
    def upper_strike(self) -> float:
        return self._upper_strike

    @property
    def put_wing_width(self) -> float:
        """Width of put spread."""
        return self._middle_strike - self._lower_strike

    @property
    def call_wing_width(self) -> float:
        """Width of call spread."""
        return self._upper_strike - self._middle_strike

    @property
    def max_profit(self) -> float:
        return self._max_profit

    @property
    def max_loss(self) -> float:
        return self._max_loss

    @property
    def upper_breakeven(self) -> float:
        return self._upper_breakeven

    @property
    def lower_breakeven(self) -> float:
        return self._lower_breakeven

    @property
    def breakeven_range(self) -> float:
        """Width of the breakeven range."""
        return self._upper_breakeven - self._lower_breakeven


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    'IronCondor',
    'IronButterfly',
]
