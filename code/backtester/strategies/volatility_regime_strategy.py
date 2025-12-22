"""
Volatility Regime Strategy

Adaptive strategy that selects different option structures based on volatility regime.
Uses VIX levels to determine market volatility environment and choose appropriate tactics.

Strategy Logic:
    High VIX (>25): Sell premium (short straddle/strangle) - elevated premiums
    Medium VIX (15-25): Iron condors - moderate premium collection with defined risk
    Low VIX (<15): Buy premium (long straddle/strangle) - cheap options, expect expansion

Volatility Regimes:
    HIGH: VIX > high_vix_threshold (default 25)
        - Action: Sell premium (short straddle)
        - Rationale: High premiums, expect reversion to mean

    MEDIUM: low_vix_threshold < VIX <= high_vix_threshold
        - Action: Iron condors
        - Rationale: Moderate volatility, defined risk

    LOW: VIX <= low_vix_threshold (default 15)
        - Action: Buy premium (long straddle)
        - Rationale: Cheap options, expect volatility expansion

Usage:
    from backtester.strategies.volatility_regime_strategy import VolatilityRegimeStrategy

    strategy = VolatilityRegimeStrategy(
        name='VIX Regime',
        initial_capital=100000,
        high_vix_threshold=25.0,
        low_vix_threshold=15.0
    )

    market_data = {
        'vix': 22.0,
        'spot': 450.0,
        'dte': 45
    }

    if strategy.should_enter(market_data):
        regime = strategy.get_current_regime(market_data)
        # Select appropriate structure based on regime

References:
    - VIX as a Predictor of Volatility - CBOE
    - Dynamic Option Trading Strategies - Connolly (1997)
"""

import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

from backtester.strategies.strategy import Strategy
from backtester.core.option_structure import OptionStructure

logger = logging.getLogger(__name__)


class VolatilityRegime(Enum):
    """Volatility regime classifications."""
    LOW = 'low'
    MEDIUM = 'medium'
    HIGH = 'high'


class VolatilityRegimeStrategy(Strategy):
    """
    Adaptive strategy that trades based on volatility regime.

    Selects option structures based on VIX level:
    - High VIX: Sell premium (short straddle/strangle)
    - Medium VIX: Iron condors (defined risk premium selling)
    - Low VIX: Buy premium (long straddle/strangle)

    Parameters:
        high_vix_threshold: VIX level above which regime is HIGH. Default 25.
        low_vix_threshold: VIX level below which regime is LOW. Default 15.
        profit_target_pct: Profit target for all positions. Default 0.50.
        loss_limit_pct: Loss limit for all positions. Default 2.0.
        exit_dte: Exit if DTE <= this. Default 7.
        min_entry_dte: Minimum DTE for entry. Default 30.
        min_regime_duration: Minimum days in regime before entry. Default 3.
    """

    __slots__ = (
        '_high_vix_threshold',
        '_low_vix_threshold',
        '_profit_target_pct',
        '_loss_limit_pct',
        '_exit_dte',
        '_min_entry_dte',
        '_min_regime_duration',
        '_regime_history',
    )

    def __init__(
        self,
        name: str = 'VolatilityRegime',
        description: str = 'Adaptive strategy based on VIX regime',
        initial_capital: float = 100000.0,
        position_limits: Optional[Dict[str, Any]] = None,
        high_vix_threshold: float = 25.0,
        low_vix_threshold: float = 15.0,
        profit_target_pct: float = 0.50,
        loss_limit_pct: float = 2.0,
        exit_dte: int = 7,
        min_entry_dte: int = 30,
        min_regime_duration: int = 3,
        strategy_id: Optional[str] = None
    ) -> None:
        """Initialize Volatility Regime Strategy."""
        super().__init__(
            name=name,
            description=description,
            initial_capital=initial_capital,
            position_limits=position_limits,
            strategy_id=strategy_id
        )

        # Validate thresholds
        if low_vix_threshold >= high_vix_threshold:
            raise ValueError(
                f"low_vix_threshold ({low_vix_threshold}) must be < "
                f"high_vix_threshold ({high_vix_threshold})"
            )
        if low_vix_threshold <= 0 or high_vix_threshold <= 0:
            raise ValueError("VIX thresholds must be positive")

        self._high_vix_threshold = high_vix_threshold
        self._low_vix_threshold = low_vix_threshold

        # Validate other parameters
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

        if min_regime_duration < 0:
            raise ValueError(f"min_regime_duration must be non-negative, got {min_regime_duration}")
        self._min_regime_duration = min_regime_duration

        # Track regime history (date -> regime)
        self._regime_history: Dict[datetime, VolatilityRegime] = {}

        logger.info(
            f"Initialized {name}: VIX thresholds Low<{low_vix_threshold}, "
            f"High>{high_vix_threshold}"
        )

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def high_vix_threshold(self) -> float:
        """Get high VIX threshold."""
        return self._high_vix_threshold

    @property
    def low_vix_threshold(self) -> float:
        """Get low VIX threshold."""
        return self._low_vix_threshold

    # =========================================================================
    # Regime Classification
    # =========================================================================

    def get_current_regime(self, market_data: Dict[str, Any]) -> Optional[VolatilityRegime]:
        """
        Classify current volatility regime based on VIX.

        Args:
            market_data: Dictionary with 'vix' level

        Returns:
            VolatilityRegime (HIGH, MEDIUM, or LOW) or None if VIX unavailable
        """
        vix = market_data.get('vix')
        if vix is None:
            return None

        if vix > self._high_vix_threshold:
            return VolatilityRegime.HIGH
        elif vix < self._low_vix_threshold:
            return VolatilityRegime.LOW
        else:
            return VolatilityRegime.MEDIUM

    def update_regime_history(
        self,
        date: datetime,
        regime: VolatilityRegime
    ) -> None:
        """
        Update regime history.

        Args:
            date: Date of observation
            regime: Volatility regime
        """
        self._regime_history[date] = regime

    def get_regime_duration(
        self,
        current_date: datetime,
        regime: VolatilityRegime
    ) -> int:
        """
        Get number of consecutive days in current regime.

        Args:
            current_date: Current date
            regime: Regime to check duration for

        Returns:
            Number of consecutive days in regime
        """
        if not self._regime_history:
            return 0

        # Sort dates in reverse
        sorted_dates = sorted(self._regime_history.keys(), reverse=True)

        duration = 0
        for date in sorted_dates:
            if date > current_date:
                continue
            if self._regime_history[date] == regime:
                duration += 1
            else:
                break

        return duration

    # =========================================================================
    # Abstract Method Implementations
    # =========================================================================

    def should_enter(self, market_data: Dict[str, Any]) -> bool:
        """
        Determine if entry conditions are met.

        Args:
            market_data: Dictionary with:
                - 'vix': VIX level
                - 'dte': Days to expiration
                - 'date': Current date (optional)

        Returns:
            True if entry conditions met
        """
        vix = market_data.get('vix')
        dte = market_data.get('dte')

        if vix is None or dte is None:
            logger.debug("Missing required market data")
            return False

        # Check minimum DTE
        if dte < self._min_entry_dte:
            logger.debug(f"DTE {dte} below minimum {self._min_entry_dte}")
            return False

        # Get current regime
        regime = self.get_current_regime(market_data)
        if regime is None:
            return False

        # Check regime duration if date provided
        current_date = market_data.get('date')
        if current_date and self._min_regime_duration > 0:
            duration = self.get_regime_duration(current_date, regime)
            if duration < self._min_regime_duration:
                logger.debug(
                    f"Regime duration {duration} below minimum {self._min_regime_duration}"
                )
                return False

        logger.info(
            f"Entry signal: VIX={vix:.2f}, Regime={regime.value.upper()}, DTE={dte}"
        )
        return True

    def should_exit(
        self,
        structure: OptionStructure,
        market_data: Dict[str, Any]
    ) -> bool:
        """
        Determine if exit conditions are met.

        Args:
            structure: Position to evaluate
            market_data: Current market data with 'dte' and optionally 'vix'

        Returns:
            True if exit conditions met
        """
        # Check DTE
        dte = market_data.get('dte')
        if dte is not None and dte <= self._exit_dte:
            logger.info(f"Exit signal: DTE={dte}")
            return True

        # Check P&L
        try:
            current_pnl = structure.calculate_pnl()

            # Determine max profit based on structure type
            if hasattr(structure, 'max_profit'):
                max_profit = structure.max_profit
            else:
                # Fallback to net premium
                max_profit = abs(structure.net_premium)

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

        # Optional: Exit if regime changes significantly
        vix = market_data.get('vix')
        if vix is not None:
            current_regime = self.get_current_regime(market_data)
            structure_type = structure.structure_type

            # If we're in a long premium position and VIX spikes, take profit
            if 'long' in structure_type and current_regime == VolatilityRegime.HIGH:
                logger.info("Exit signal: Long position in high VIX regime")
                return True

            # If we're in a short premium position and VIX drops too low, exit
            if 'short' in structure_type and current_regime == VolatilityRegime.LOW:
                logger.info("Exit signal: Short position in low VIX regime")
                return True

        return False

    def get_recommended_structure(
        self,
        market_data: Dict[str, Any]
    ) -> Optional[str]:
        """
        Get recommended structure type based on current regime.

        Args:
            market_data: Market data with VIX

        Returns:
            Recommended structure type string or None
        """
        regime = self.get_current_regime(market_data)

        if regime == VolatilityRegime.HIGH:
            return 'short_straddle'
        elif regime == VolatilityRegime.MEDIUM:
            return 'iron_condor'
        elif regime == VolatilityRegime.LOW:
            return 'long_straddle'
        else:
            return None

    def get_strategy_stats(self) -> Dict[str, Any]:
        """Get strategy-specific statistics."""
        stats = self.get_statistics()

        # Add strategy-specific metrics
        stats['strategy_type'] = 'volatility_regime'
        stats['high_vix_threshold'] = self._high_vix_threshold
        stats['low_vix_threshold'] = self._low_vix_threshold
        stats['profit_target_pct'] = self._profit_target_pct
        stats['loss_limit_pct'] = self._loss_limit_pct

        # Regime statistics
        if self._regime_history:
            regime_counts = {}
            for regime in self._regime_history.values():
                regime_counts[regime.value] = regime_counts.get(regime.value, 0) + 1
            stats['regime_distribution'] = regime_counts
        else:
            stats['regime_distribution'] = {}

        return stats


__all__ = ['VolatilityRegimeStrategy', 'VolatilityRegime']
