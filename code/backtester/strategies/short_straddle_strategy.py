"""
Short Straddle High IV Strategy

This module implements a short straddle strategy that enters when implied volatility
is elevated. The strategy sells ATM call and put options to collect premium,
expecting low realized volatility during the holding period.

Strategy Logic:
    Entry Conditions:
        - IV rank > threshold (default 70%)
        - Sufficient capital available
        - DTE >= minimum threshold (default 30 days)

    Exit Conditions:
        - Profit target reached (default 50% of max profit)
        - Loss limit breached (default 2x max profit)
        - DTE threshold (default 7 days to expiration)

Risk Characteristics:
    - Max Profit: Total premium collected
    - Max Loss: Unlimited (managed by stop loss)
    - Delta: Approximately neutral at entry
    - Theta: Positive (benefits from time decay)
    - Vega: Negative (loses from volatility expansion)

Usage:
    from backtester.strategies.short_straddle_strategy import ShortStraddleHighIVStrategy
    from datetime import datetime

    strategy = ShortStraddleHighIVStrategy(
        name='Short Straddle IV70',
        initial_capital=100000,
        iv_rank_threshold=70,
        profit_target_pct=0.50,
        loss_limit_pct=2.0,
        exit_dte=7,
        min_entry_dte=30
    )

    # Check if should enter
    market_data = {
        'spot': 450.0,
        'iv_rank': 75,
        'dte': 45,
        'atm_strike': 450.0,
        'atm_call_price': 6.50,
        'atm_put_price': 6.25
    }

    if strategy.should_enter(market_data):
        # Create position
        pass

References:
    - Natenberg, S. (1994). Option Volatility and Pricing.
    - Tastyworks IV Rank Methodology
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from backtester.strategies.strategy import Strategy
from backtester.core.option_structure import OptionStructure
from backtester.structures.straddle import ShortStraddle

# Configure module logger
logger = logging.getLogger(__name__)


# =============================================================================
# Short Straddle High IV Strategy
# =============================================================================

class ShortStraddleHighIVStrategy(Strategy):
    """
    Short straddle strategy that enters on high IV rank.

    This strategy sells ATM straddles when implied volatility is elevated,
    profiting from volatility contraction and time decay.

    Parameters:
        iv_rank_threshold: Minimum IV rank for entry (0-100). Default 70.
        profit_target_pct: Exit when P&L >= this % of max profit. Default 0.50 (50%).
        loss_limit_pct: Exit when loss >= this multiple of max profit. Default 2.0.
        exit_dte: Exit if DTE <= this value. Default 7.
        min_entry_dte: Minimum DTE required for entry. Default 30.
        max_positions_per_underlying: Max concurrent positions per ticker. Default 1.

    Example:
        >>> strategy = ShortStraddleHighIVStrategy(
        ...     name='Short Straddle IV70',
        ...     initial_capital=100000,
        ...     iv_rank_threshold=70,
        ...     profit_target_pct=0.50
        ... )
    """

    __slots__ = (
        '_iv_rank_threshold',
        '_profit_target_pct',
        '_loss_limit_pct',
        '_exit_dte',
        '_min_entry_dte',
        '_max_positions_per_underlying',
    )

    def __init__(
        self,
        name: str = 'ShortStraddleHighIV',
        description: str = 'Short straddle on high IV rank',
        initial_capital: float = 100000.0,
        position_limits: Optional[Dict[str, Any]] = None,
        iv_rank_threshold: float = 70.0,
        profit_target_pct: float = 0.50,
        loss_limit_pct: float = 2.0,
        exit_dte: int = 7,
        min_entry_dte: int = 30,
        max_positions_per_underlying: int = 1,
        strategy_id: Optional[str] = None
    ) -> None:
        """
        Initialize Short Straddle High IV Strategy.

        Args:
            name: Strategy name
            description: Strategy description
            initial_capital: Starting capital
            position_limits: Risk limits dictionary
            iv_rank_threshold: Minimum IV rank for entry (0-100)
            profit_target_pct: Profit target as fraction of max profit
            loss_limit_pct: Loss limit as multiple of max profit
            exit_dte: Exit if days to expiration <= this
            min_entry_dte: Minimum DTE required for entry
            max_positions_per_underlying: Max positions per ticker
            strategy_id: Optional unique identifier
        """
        # Initialize base strategy
        super().__init__(
            name=name,
            description=description,
            initial_capital=initial_capital,
            position_limits=position_limits,
            strategy_id=strategy_id
        )

        # Validate and store strategy-specific parameters
        if not 0 <= iv_rank_threshold <= 100:
            raise ValueError(f"iv_rank_threshold must be 0-100, got {iv_rank_threshold}")
        self._iv_rank_threshold = iv_rank_threshold

        if not 0 < profit_target_pct <= 1.0:
            raise ValueError(f"profit_target_pct must be 0-1, got {profit_target_pct}")
        self._profit_target_pct = profit_target_pct

        if loss_limit_pct <= 0:
            raise ValueError(f"loss_limit_pct must be positive, got {loss_limit_pct}")
        self._loss_limit_pct = loss_limit_pct

        if exit_dte < 0:
            raise ValueError(f"exit_dte must be non-negative, got {exit_dte}")
        self._exit_dte = exit_dte

        if min_entry_dte <= 0:
            raise ValueError(f"min_entry_dte must be positive, got {min_entry_dte}")
        self._min_entry_dte = min_entry_dte

        if max_positions_per_underlying <= 0:
            raise ValueError(f"max_positions_per_underlying must be positive, got {max_positions_per_underlying}")
        self._max_positions_per_underlying = max_positions_per_underlying

        logger.info(
            f"Initialized {name}: IV threshold={iv_rank_threshold}%, "
            f"Profit target={profit_target_pct:.0%}, Loss limit={loss_limit_pct}x"
        )

    # =========================================================================
    # Strategy-specific Properties
    # =========================================================================

    @property
    def iv_rank_threshold(self) -> float:
        """Get IV rank threshold."""
        return self._iv_rank_threshold

    @property
    def profit_target_pct(self) -> float:
        """Get profit target percentage."""
        return self._profit_target_pct

    @property
    def loss_limit_pct(self) -> float:
        """Get loss limit multiple."""
        return self._loss_limit_pct

    @property
    def exit_dte(self) -> int:
        """Get exit DTE threshold."""
        return self._exit_dte

    @property
    def min_entry_dte(self) -> int:
        """Get minimum entry DTE."""
        return self._min_entry_dte

    # =========================================================================
    # Abstract Method Implementations
    # =========================================================================

    def should_enter(self, market_data: Dict[str, Any]) -> bool:
        """
        Determine if entry conditions are met.

        Args:
            market_data: Dictionary with market information:
                - 'iv_rank': IV rank (0-100)
                - 'dte': Days to expiration
                - 'underlying': Ticker symbol
                - 'spot': Current spot price
                - Additional optional fields

        Returns:
            True if entry conditions met, False otherwise
        """
        # Extract required fields
        iv_rank = market_data.get('iv_rank')
        dte = market_data.get('dte')
        underlying = market_data.get('underlying')

        # Validate required data
        if iv_rank is None or dte is None or underlying is None:
            logger.debug("Missing required market data for entry decision")
            return False

        # Check IV rank threshold
        if iv_rank < self._iv_rank_threshold:
            logger.debug(f"IV rank {iv_rank:.1f}% below threshold {self._iv_rank_threshold}%")
            return False

        # Check minimum DTE
        if dte < self._min_entry_dte:
            logger.debug(f"DTE {dte} below minimum {self._min_entry_dte}")
            return False

        # Check max positions per underlying
        positions_on_underlying = sum(
            1 for s in self._structures
            if s.underlying == underlying
        )
        if positions_on_underlying >= self._max_positions_per_underlying:
            logger.debug(
                f"Max positions ({self._max_positions_per_underlying}) "
                f"reached for {underlying}"
            )
            return False

        logger.info(
            f"Entry signal: {underlying} IV rank={iv_rank:.1f}%, DTE={dte}"
        )
        return True

    def should_exit(
        self,
        structure: OptionStructure,
        market_data: Dict[str, Any]
    ) -> bool:
        """
        Determine if exit conditions are met for a position.

        Args:
            structure: OptionStructure to evaluate
            market_data: Dictionary with current market information:
                - 'dte': Days to expiration
                - Additional optional fields

        Returns:
            True if exit conditions met, False otherwise
        """
        # Get current DTE
        dte = market_data.get('dte')
        if dte is not None and dte <= self._exit_dte:
            logger.info(
                f"Exit signal: DTE={dte} <= threshold {self._exit_dte} "
                f"for {structure.structure_id}"
            )
            return True

        # Calculate P&L percentage
        try:
            current_pnl = structure.calculate_pnl()

            # Get max profit (total premium for short straddle)
            if hasattr(structure, 'max_profit'):
                max_profit = structure.max_profit
            else:
                max_profit = structure.net_premium

            if max_profit <= 0:
                logger.warning(f"Invalid max_profit for {structure.structure_id}")
                return False

            pnl_pct = current_pnl / max_profit

            # Check profit target
            if pnl_pct >= self._profit_target_pct:
                logger.info(
                    f"Exit signal: Profit target reached {pnl_pct:.1%} >= "
                    f"{self._profit_target_pct:.1%} for {structure.structure_id}"
                )
                return True

            # Check loss limit (negative P&L exceeds threshold)
            if pnl_pct <= -self._loss_limit_pct:
                logger.info(
                    f"Exit signal: Loss limit breached {pnl_pct:.1%} <= "
                    f"-{self._loss_limit_pct:.1%} for {structure.structure_id}"
                )
                return True

        except Exception as e:
            logger.error(f"Error calculating P&L for exit decision: {e}")
            return False

        return False

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def create_position(
        self,
        underlying: str,
        strike: float,
        expiration: datetime,
        call_price: float,
        put_price: float,
        quantity: int,
        entry_date: datetime,
        spot_price: float,
        call_iv: Optional[float] = None,
        put_iv: Optional[float] = None
    ) -> ShortStraddle:
        """
        Create a short straddle position.

        Args:
            underlying: Ticker symbol
            strike: ATM strike price
            expiration: Expiration date
            call_price: Call premium to sell
            put_price: Put premium to sell
            quantity: Number of contracts
            entry_date: Entry timestamp
            spot_price: Spot price at entry
            call_iv: Optional call IV
            put_iv: Optional put IV

        Returns:
            ShortStraddle instance
        """
        straddle = ShortStraddle.create(
            underlying=underlying,
            strike=strike,
            expiration=expiration,
            call_price=call_price,
            put_price=put_price,
            quantity=quantity,
            entry_date=entry_date,
            underlying_price=spot_price,
            call_iv=call_iv,
            put_iv=put_iv
        )

        logger.info(
            f"Created short straddle: {underlying} {strike} strike, "
            f"Premium: ${straddle.net_premium:,.2f}, "
            f"Breakevens: {straddle.lower_breakeven:.2f}-{straddle.upper_breakeven:.2f}"
        )

        return straddle

    def get_strategy_stats(self) -> Dict[str, Any]:
        """
        Get strategy-specific statistics.

        Returns:
            Dictionary with strategy performance metrics
        """
        stats = self.get_statistics()

        # Add strategy-specific metrics
        stats['strategy_type'] = 'short_straddle_high_iv'
        stats['iv_rank_threshold'] = self._iv_rank_threshold
        stats['profit_target_pct'] = self._profit_target_pct
        stats['loss_limit_pct'] = self._loss_limit_pct
        stats['exit_dte'] = self._exit_dte
        stats['min_entry_dte'] = self._min_entry_dte

        # Calculate average days held for closed positions
        closed_trades = [
            t for t in self._trade_history
            if t.get('action') == 'close' and 'hold_time_days' in t
        ]
        if closed_trades:
            avg_hold_days = sum(t['hold_time_days'] for t in closed_trades) / len(closed_trades)
            stats['avg_hold_days'] = avg_hold_days
        else:
            stats['avg_hold_days'] = 0

        return stats


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    'ShortStraddleHighIVStrategy',
]
