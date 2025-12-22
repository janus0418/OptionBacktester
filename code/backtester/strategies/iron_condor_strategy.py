"""
Iron Condor Strategy

Implements an iron condor strategy that enters when IV rank is moderate,
placing strikes at ±1 standard deviation to create a defined-risk position.

Strategy Logic:
    Entry Conditions:
        - IV rank > threshold (default 50%)
        - Strikes placed at ±1 standard deviation
        - DTE >= minimum threshold (default 45 days)

    Exit Conditions:
        - Profit target (default 50% of max profit)
        - Loss limit (default 2x max profit)
        - DTE threshold (default 7 days)

Risk Characteristics:
    - Max Profit: Net credit received
    - Max Loss: Wing width - net credit
    - Delta: Approximately neutral
    - Defined risk structure

Usage:
    from backtester.strategies.iron_condor_strategy import IronCondorStrategy

    strategy = IronCondorStrategy(
        name='IC IV50',
        initial_capital=100000,
        iv_rank_threshold=50,
        profit_target_pct=0.50,
        std_dev_width=1.0
    )

References:
    - CBOE Iron Condor Education
    - Options as a Strategic Investment, McMillan
"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np

from backtester.strategies.strategy import Strategy
from backtester.core.option_structure import OptionStructure
from backtester.structures.condor import IronCondor

logger = logging.getLogger(__name__)


class IronCondorStrategy(Strategy):
    """
    Iron condor strategy with delta-based strike selection.

    Places iron condors with strikes at ±1 standard deviation,
    targeting premium collection in range-bound markets.

    Parameters:
        iv_rank_threshold: Minimum IV rank for entry. Default 50.
        profit_target_pct: Exit at this % of max profit. Default 0.50.
        loss_limit_pct: Exit at this multiple of max profit loss. Default 2.0.
        exit_dte: Exit if DTE <= this. Default 7.
        min_entry_dte: Minimum DTE for entry. Default 45.
        std_dev_width: Standard deviations for strike selection. Default 1.0.
        wing_width_pct: Wing width as % of underlying. Default 0.02 (2%).
    """

    __slots__ = (
        '_iv_rank_threshold',
        '_profit_target_pct',
        '_loss_limit_pct',
        '_exit_dte',
        '_min_entry_dte',
        '_std_dev_width',
        '_wing_width_pct',
    )

    def __init__(
        self,
        name: str = 'IronCondor',
        description: str = 'Iron condor on moderate IV',
        initial_capital: float = 100000.0,
        position_limits: Optional[Dict[str, Any]] = None,
        iv_rank_threshold: float = 50.0,
        profit_target_pct: float = 0.50,
        loss_limit_pct: float = 2.0,
        exit_dte: int = 7,
        min_entry_dte: int = 45,
        std_dev_width: float = 1.0,
        wing_width_pct: float = 0.02,
        strategy_id: Optional[str] = None
    ) -> None:
        """Initialize Iron Condor Strategy."""
        super().__init__(
            name=name,
            description=description,
            initial_capital=initial_capital,
            position_limits=position_limits,
            strategy_id=strategy_id
        )

        # Validate parameters
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

        if std_dev_width <= 0:
            raise ValueError(f"std_dev_width must be positive, got {std_dev_width}")
        self._std_dev_width = std_dev_width

        if wing_width_pct <= 0:
            raise ValueError(f"wing_width_pct must be positive, got {wing_width_pct}")
        self._wing_width_pct = wing_width_pct

        logger.info(
            f"Initialized {name}: IV threshold={iv_rank_threshold}%, "
            f"Std dev={std_dev_width}, Wing width={wing_width_pct:.1%}"
        )

    def should_enter(self, market_data: Dict[str, Any]) -> bool:
        """
        Determine if entry conditions are met.

        Args:
            market_data: Dictionary with:
                - 'iv_rank': IV rank (0-100)
                - 'dte': Days to expiration
                - 'underlying': Ticker symbol

        Returns:
            True if entry conditions met
        """
        iv_rank = market_data.get('iv_rank')
        dte = market_data.get('dte')

        if iv_rank is None or dte is None:
            logger.debug("Missing required market data")
            return False

        if iv_rank < self._iv_rank_threshold:
            logger.debug(f"IV rank {iv_rank:.1f}% below threshold {self._iv_rank_threshold}%")
            return False

        if dte < self._min_entry_dte:
            logger.debug(f"DTE {dte} below minimum {self._min_entry_dte}")
            return False

        logger.info(f"Entry signal: IV rank={iv_rank:.1f}%, DTE={dte}")
        return True

    def should_exit(self, structure: OptionStructure, market_data: Dict[str, Any]) -> bool:
        """
        Determine if exit conditions are met.

        Args:
            structure: Position to evaluate
            market_data: Current market data with 'dte'

        Returns:
            True if exit conditions met
        """
        # Check DTE
        dte = market_data.get('dte')
        if dte is not None and dte <= self._exit_dte:
            logger.info(f"Exit signal: DTE={dte} for {structure.structure_id}")
            return True

        # Check P&L
        try:
            current_pnl = structure.calculate_pnl()
            max_profit = getattr(structure, 'max_profit', structure.net_premium)

            if max_profit <= 0:
                return False

            pnl_pct = current_pnl / max_profit

            # Profit target
            if pnl_pct >= self._profit_target_pct:
                logger.info(f"Exit signal: Profit target {pnl_pct:.1%}")
                return True

            # Loss limit
            if pnl_pct <= -self._loss_limit_pct:
                logger.info(f"Exit signal: Loss limit {pnl_pct:.1%}")
                return True

        except Exception as e:
            logger.error(f"Error in exit logic: {e}")
            return False

        return False

    def calculate_strikes(
        self,
        spot: float,
        vol: float,
        dte: int
    ) -> Dict[str, float]:
        """
        Calculate iron condor strikes based on standard deviation.

        Args:
            spot: Current spot price
            vol: Implied volatility (annualized)
            dte: Days to expiration

        Returns:
            Dictionary with strike prices:
                - 'put_buy': Lower put strike
                - 'put_sell': Short put strike
                - 'call_sell': Short call strike
                - 'call_buy': Upper call strike
        """
        # Calculate expected move (1 standard deviation)
        time_to_expiry = dte / 365.0
        std_dev_move = spot * vol * np.sqrt(time_to_expiry) * self._std_dev_width

        # Wing width
        wing_width = spot * self._wing_width_pct

        # Calculate strikes (rounded to nearest dollar or $5)
        call_sell_strike = self._round_strike(spot + std_dev_move)
        put_sell_strike = self._round_strike(spot - std_dev_move)
        call_buy_strike = self._round_strike(call_sell_strike + wing_width)
        put_buy_strike = self._round_strike(put_sell_strike - wing_width)

        return {
            'put_buy': put_buy_strike,
            'put_sell': put_sell_strike,
            'call_sell': call_sell_strike,
            'call_buy': call_buy_strike
        }

    def _round_strike(self, strike: float) -> float:
        """Round strike to nearest $5 for standard options."""
        return round(strike / 5.0) * 5.0

    def create_position(
        self,
        underlying: str,
        strikes: Dict[str, float],
        expiration: datetime,
        prices: Dict[str, float],
        quantity: int,
        entry_date: datetime,
        spot_price: float
    ) -> IronCondor:
        """
        Create an iron condor position.

        Args:
            underlying: Ticker symbol
            strikes: Dict with 'put_buy', 'put_sell', 'call_sell', 'call_buy' strikes
            expiration: Expiration date
            prices: Dict with corresponding option prices
            quantity: Number of contracts
            entry_date: Entry timestamp
            spot_price: Spot price at entry

        Returns:
            IronCondor instance
        """
        condor = IronCondor.create(
            underlying=underlying,
            put_buy_strike=strikes['put_buy'],
            put_sell_strike=strikes['put_sell'],
            call_sell_strike=strikes['call_sell'],
            call_buy_strike=strikes['call_buy'],
            expiration=expiration,
            put_buy_price=prices['put_buy'],
            put_sell_price=prices['put_sell'],
            call_sell_price=prices['call_sell'],
            call_buy_price=prices['call_buy'],
            quantity=quantity,
            entry_date=entry_date,
            underlying_price=spot_price
        )

        logger.info(
            f"Created iron condor: {underlying} "
            f"{strikes['put_buy']}/{strikes['put_sell']}/{strikes['call_sell']}/{strikes['call_buy']}, "
            f"Credit: ${condor.net_premium:,.2f}"
        )

        return condor


__all__ = ['IronCondorStrategy']
