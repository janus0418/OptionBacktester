"""
Strategy Base Class for Options Backtesting

This module provides the Strategy base class that serves as the foundation
for all trading strategy implementations. It manages position tracking,
capital allocation, portfolio Greeks aggregation, and P&L calculation.

Key Features:
    - Active and closed position management via OptionStructure
    - Portfolio-level Greeks aggregation
    - Capital and margin tracking
    - Risk limit enforcement
    - Realized and unrealized P&L calculation
    - Abstract entry/exit condition framework

Design Philosophy:
    The Strategy class is an abstract base that provides infrastructure
    for position and risk management. Concrete strategies (implemented in
    Run 8) will inherit from this class and override should_enter() and
    should_exit() methods with specific trading logic.

Financial Correctness:
    - Portfolio Greeks = sum of net Greeks across all active structures
    - Total P&L = unrealized P&L (open) + realized P&L (closed)
    - Margin = sum of margin requirements for all active positions
    - Available capital = initial_capital - allocated capital

Usage:
    from backtester.strategies.strategy import Strategy

    class MyStrategy(Strategy):
        def should_enter(self, market_data):
            # Custom entry logic
            return some_condition

        def should_exit(self, structure, market_data):
            # Custom exit logic
            return some_exit_condition

References:
    - Hull, J. C. (2018). Options, Futures, and Other Derivatives.
    - Taleb, N. N. (1997). Dynamic Hedging.
"""

import logging
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from backtester.core.option_structure import (
    OptionStructure,
    OptionStructureError,
    EmptyStructureError,
    GREEK_NAMES,
)
from backtester.core.option import CONTRACT_MULTIPLIER

# Configure module logger
logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Default risk limit values
DEFAULT_MAX_POSITIONS = 10
DEFAULT_MAX_TOTAL_DELTA = 100.0  # Maximum absolute delta exposure
DEFAULT_MAX_TOTAL_VEGA = 500.0   # Maximum absolute vega exposure
DEFAULT_MAX_CAPITAL_UTILIZATION = 0.80  # 80% of capital max

# Margin calculation constants (simplified)
# In practice, margin varies by broker and structure type
NAKED_OPTION_MARGIN_FACTOR = 0.20  # 20% of notional for naked options
SPREAD_MARGIN_FACTOR = 1.0         # Width of spread for defined risk
DEFAULT_MARGIN_PER_CONTRACT = 2000.0  # Default margin per contract


# =============================================================================
# Exceptions
# =============================================================================

class StrategyError(Exception):
    """Base exception for Strategy errors."""
    pass


class StrategyValidationError(StrategyError):
    """Exception raised when strategy validation fails."""
    pass


class PositionError(StrategyError):
    """Exception raised for position management errors."""
    pass


class RiskLimitError(StrategyError):
    """Exception raised when risk limits are breached."""
    pass


class InsufficientCapitalError(StrategyError):
    """Exception raised when there is insufficient capital."""
    pass


# =============================================================================
# Strategy Base Class
# =============================================================================

class Strategy(ABC):
    """
    Abstract base class for options trading strategies.

    This class provides the infrastructure for managing positions, tracking
    capital and P&L, calculating portfolio Greeks, and enforcing risk limits.
    Concrete strategy implementations must override the abstract methods
    should_enter() and should_exit().

    Attributes:
        name (str): Strategy name for identification
        description (str): Strategy description
        structures (List[OptionStructure]): Currently active positions
        closed_structures (List[OptionStructure]): Historical closed positions
        capital (float): Current available capital
        initial_capital (float): Starting capital
        position_limits (Dict): Risk limit configuration

    Properties:
        num_open_positions (int): Count of active positions
        num_closed_positions (int): Count of closed positions
        total_allocated_capital (float): Capital tied up in open positions
        available_capital (float): Capital available for new positions

    Example:
        >>> class ShortStraddleStrategy(Strategy):
        ...     def should_enter(self, market_data):
        ...         return market_data['iv_percentile'] > 50
        ...
        ...     def should_exit(self, structure, market_data):
        ...         pnl_pct = structure.calculate_pnl_percent()
        ...         return pnl_pct >= 0.25 or pnl_pct <= -1.0
    """

    __slots__ = (
        '_strategy_id',
        '_name',
        '_description',
        '_structures',
        '_closed_structures',
        '_initial_capital',
        '_capital',
        '_position_limits',
        '_realized_pnl',
        '_trade_history',
        '_created_at',
    )

    def __init__(
        self,
        name: str,
        description: str = '',
        initial_capital: float = 100000.0,
        position_limits: Optional[Dict[str, Any]] = None,
        strategy_id: Optional[str] = None
    ) -> None:
        """
        Initialize a Strategy instance.

        Args:
            name: Strategy name for identification
            description: Optional description of the strategy
            initial_capital: Starting capital (default $100,000)
            position_limits: Dictionary of risk limits. Keys include:
                - 'max_positions': Maximum number of concurrent positions
                - 'max_total_delta': Maximum absolute net delta
                - 'max_total_vega': Maximum absolute net vega
                - 'max_capital_utilization': Max fraction of capital to use
                - 'max_single_position_size': Max capital per position
            strategy_id: Unique identifier. Auto-generated if not provided.

        Raises:
            StrategyValidationError: If parameters are invalid

        Example:
            >>> strategy = Strategy(
            ...     name='ShortStraddle',
            ...     description='Daily short straddle on SPY',
            ...     initial_capital=100000,
            ...     position_limits={'max_positions': 5, 'max_total_delta': 50}
            ... )
        """
        # Validate name
        if not name or not isinstance(name, str):
            raise StrategyValidationError("name must be a non-empty string")
        self._name = name.strip()

        # Set description
        self._description = description or ''

        # Generate unique ID
        self._strategy_id = strategy_id or str(uuid.uuid4())[:8]

        # Validate and set initial capital
        if initial_capital is None or initial_capital <= 0:
            raise StrategyValidationError(
                f"initial_capital must be positive, got {initial_capital}"
            )
        if not np.isfinite(initial_capital):
            raise StrategyValidationError(
                f"initial_capital must be finite, got {initial_capital}"
            )
        self._initial_capital = float(initial_capital)
        self._capital = float(initial_capital)

        # Set up position limits with defaults
        self._position_limits = {
            'max_positions': DEFAULT_MAX_POSITIONS,
            'max_total_delta': DEFAULT_MAX_TOTAL_DELTA,
            'max_total_vega': DEFAULT_MAX_TOTAL_VEGA,
            'max_capital_utilization': DEFAULT_MAX_CAPITAL_UTILIZATION,
            'max_single_position_size': initial_capital * 0.20,  # 20% of capital
        }
        if position_limits:
            self._position_limits.update(position_limits)

        # Initialize position tracking
        self._structures: List[OptionStructure] = []
        self._closed_structures: List[OptionStructure] = []

        # Track realized P&L separately
        self._realized_pnl = 0.0

        # Trade history for analysis
        self._trade_history: List[Dict[str, Any]] = []

        # Creation timestamp
        self._created_at = datetime.now()

        logger.debug(
            f"Created strategy: {self._name} (id={self._strategy_id}) "
            f"with capital ${self._initial_capital:,.2f}"
        )

    # =========================================================================
    # Properties - Basic Attributes
    # =========================================================================

    @property
    def strategy_id(self) -> str:
        """Get strategy unique identifier."""
        return self._strategy_id

    @property
    def name(self) -> str:
        """Get strategy name."""
        return self._name

    @property
    def description(self) -> str:
        """Get strategy description."""
        return self._description

    @property
    def structures(self) -> List[OptionStructure]:
        """Get list of active positions (copy for safety)."""
        return self._structures.copy()

    @property
    def closed_structures(self) -> List[OptionStructure]:
        """Get list of closed positions (copy for safety)."""
        return self._closed_structures.copy()

    @property
    def initial_capital(self) -> float:
        """Get initial starting capital."""
        return self._initial_capital

    @property
    def capital(self) -> float:
        """Get current capital (initial - allocated + realized P&L)."""
        return self._capital

    @property
    def position_limits(self) -> Dict[str, Any]:
        """Get position limits configuration (copy for safety)."""
        return self._position_limits.copy()

    @property
    def realized_pnl(self) -> float:
        """Get total realized P&L from closed positions."""
        return self._realized_pnl

    @property
    def trade_history(self) -> List[Dict[str, Any]]:
        """Get trade history (copy for safety)."""
        return self._trade_history.copy()

    @property
    def created_at(self) -> datetime:
        """Get strategy creation timestamp."""
        return self._created_at

    # =========================================================================
    # Properties - Computed Attributes
    # =========================================================================

    @property
    def num_open_positions(self) -> int:
        """Get number of currently open positions."""
        return len(self._structures)

    @property
    def num_closed_positions(self) -> int:
        """Get number of closed positions."""
        return len(self._closed_structures)

    @property
    def total_allocated_capital(self) -> float:
        """
        Get total capital allocated to open positions.

        This represents the capital at risk in current positions,
        calculated as the sum of margin requirements.
        """
        return self.get_margin_requirement()

    @property
    def available_capital(self) -> float:
        """
        Get capital available for new positions.

        Available = Current Capital - Allocated to open positions
        """
        return max(0.0, self._capital - self.total_allocated_capital)

    @property
    def capital_utilization(self) -> float:
        """
        Get current capital utilization ratio.

        Utilization = Allocated Capital / Initial Capital
        """
        if self._initial_capital <= 0:
            return 0.0
        return self.total_allocated_capital / self._initial_capital

    # =========================================================================
    # Position Management Methods
    # =========================================================================

    def open_position(
        self,
        structure: OptionStructure,
        validate_limits: bool = True
    ) -> None:
        """
        Open a new position by adding a structure to active positions.

        This method validates risk limits before adding the position
        and updates capital allocation.

        Args:
            structure: OptionStructure representing the position to open
            validate_limits: Whether to validate risk limits (default True)

        Raises:
            StrategyValidationError: If structure is invalid
            RiskLimitError: If opening would breach risk limits
            InsufficientCapitalError: If insufficient capital available

        Example:
            >>> strategy.open_position(short_straddle_structure)
        """
        # Validate structure
        if not isinstance(structure, OptionStructure):
            raise StrategyValidationError(
                f"Expected OptionStructure, got {type(structure).__name__}"
            )

        if structure.is_empty:
            raise StrategyValidationError("Cannot open position with empty structure")

        # Check if structure already exists
        if structure in self._structures:
            raise PositionError(
                f"Structure {structure.structure_id} is already an open position"
            )

        # Validate risk limits if requested
        if validate_limits:
            self._validate_new_position(structure)

        # Calculate margin requirement for this position
        position_margin = self._calculate_structure_margin(structure)

        # Check capital availability
        if position_margin > self.available_capital:
            raise InsufficientCapitalError(
                f"Insufficient capital to open position. "
                f"Required: ${position_margin:,.2f}, "
                f"Available: ${self.available_capital:,.2f}"
            )

        # Add to active positions
        self._structures.append(structure)

        # Record trade
        self._trade_history.append({
            'action': 'open',
            'structure_id': structure.structure_id,
            'structure_type': structure.structure_type,
            'underlying': structure.underlying,
            'timestamp': datetime.now(),
            'net_premium': structure.net_premium,
            'margin_required': position_margin,
            'num_legs': structure.num_legs,
        })

        logger.info(
            f"Opened position: {structure.structure_type.upper()} on "
            f"{structure.underlying}, ID={structure.structure_id}, "
            f"Premium=${structure.net_premium:,.2f}"
        )

    def close_position(
        self,
        structure: OptionStructure,
        exit_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Close an open position and record the results.

        Moves the structure from active to closed, calculates realized P&L,
        and updates capital.

        Args:
            structure: OptionStructure to close (must be in active positions)
            exit_data: Optional dictionary with exit information:
                - 'exit_reason': String describing why position was closed
                - 'exit_timestamp': When position was closed
                - 'exit_prices': Dict of exit prices per leg

        Returns:
            Dictionary with close statistics:
                - 'structure_id': Structure identifier
                - 'realized_pnl': P&L from this position
                - 'hold_time_days': Days position was held
                - 'exit_reason': Reason for closing
                - 'return_pct': Percentage return

        Raises:
            PositionError: If structure is not an open position

        Example:
            >>> result = strategy.close_position(
            ...     straddle,
            ...     exit_data={'exit_reason': 'profit_target', 'exit_timestamp': now}
            ... )
            >>> print(f"Realized P&L: ${result['realized_pnl']:,.2f}")
        """
        # Validate structure is in active positions
        if structure not in self._structures:
            raise PositionError(
                f"Structure {structure.structure_id} is not an open position"
            )

        # Calculate P&L before removing
        try:
            realized_pnl = structure.calculate_pnl()
        except EmptyStructureError:
            realized_pnl = 0.0

        # Get entry info for calculations
        entry_date = structure.entry_date or datetime.now()
        exit_timestamp = (exit_data or {}).get('exit_timestamp', datetime.now())

        if isinstance(exit_timestamp, datetime):
            hold_time = exit_timestamp - entry_date
            hold_time_days = hold_time.days
        else:
            hold_time_days = 0

        # Calculate return percentage
        if abs(structure.net_premium) > 1e-10:
            return_pct = realized_pnl / abs(structure.net_premium)
        else:
            return_pct = 0.0

        # Remove from active and add to closed
        self._structures.remove(structure)
        self._closed_structures.append(structure)

        # Update realized P&L and capital
        self._realized_pnl += realized_pnl
        self._capital += realized_pnl

        # Build result dictionary
        exit_reason = (exit_data or {}).get('exit_reason', 'manual')

        result = {
            'structure_id': structure.structure_id,
            'structure_type': structure.structure_type,
            'underlying': structure.underlying,
            'realized_pnl': realized_pnl,
            'hold_time_days': hold_time_days,
            'exit_reason': exit_reason,
            'return_pct': return_pct,
            'entry_date': entry_date,
            'exit_date': exit_timestamp,
            'net_premium': structure.net_premium,
        }

        # Record trade
        self._trade_history.append({
            'action': 'close',
            'structure_id': structure.structure_id,
            'structure_type': structure.structure_type,
            'underlying': structure.underlying,
            'timestamp': exit_timestamp,
            'realized_pnl': realized_pnl,
            'return_pct': return_pct,
            'exit_reason': exit_reason,
            'hold_time_days': hold_time_days,
        })

        logger.info(
            f"Closed position: {structure.structure_type.upper()} on "
            f"{structure.underlying}, ID={structure.structure_id}, "
            f"P&L=${realized_pnl:,.2f} ({return_pct:.1%}), "
            f"Reason={exit_reason}"
        )

        return result

    def update_positions(
        self,
        market_data: Dict[str, Any],
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Update all open positions with current market data.

        This method should be called at each time step during a backtest
        to update option prices and recalculate P&L.

        Args:
            market_data: Dictionary containing market data with option prices.
                Expected format varies by implementation, but typically includes
                price data keyed by option identifier.
            timestamp: Current timestamp. If None, uses datetime.now()

        Note:
            This method updates prices on each structure via their
            update_prices_from_market_data method.

        Example:
            >>> strategy.update_positions(market_data={'SPY_450_call': 5.25}, timestamp=now)
        """
        if timestamp is None:
            timestamp = datetime.now()

        for structure in self._structures:
            try:
                structure.update_prices_from_market_data(
                    market_data=market_data,
                    timestamp=timestamp
                )
            except Exception as e:
                logger.warning(
                    f"Failed to update prices for structure "
                    f"{structure.structure_id}: {e}"
                )

    def get_position_by_id(self, structure_id: str) -> Optional[OptionStructure]:
        """
        Get an open position by its structure ID.

        Args:
            structure_id: The structure's unique identifier

        Returns:
            OptionStructure if found, None otherwise
        """
        for structure in self._structures:
            if structure.structure_id == structure_id:
                return structure
        return None

    # =========================================================================
    # Portfolio Metrics Methods
    # =========================================================================

    def calculate_portfolio_greeks(
        self,
        spot: Optional[float] = None,
        vol: Optional[float] = None,
        rate: float = 0.04,
        current_date: Optional[datetime] = None
    ) -> Dict[str, float]:
        """
        Calculate aggregate Greeks across all open positions.

        Portfolio Greeks = Sum of net Greeks from each active structure.

        Args:
            spot: Current spot price (optional, uses cached if not provided)
            vol: Current implied volatility (optional)
            rate: Risk-free rate (default 0.04 = 4%)
            current_date: Current date for time-to-expiry calculation

        Returns:
            Dictionary with portfolio Greeks:
                - 'delta': Total delta exposure
                - 'gamma': Total gamma exposure
                - 'theta': Total daily theta
                - 'vega': Total vega per 1% vol change
                - 'rho': Total rho per 1% rate change

        Note:
            Returns zero Greeks if no open positions.

        Example:
            >>> greeks = strategy.calculate_portfolio_greeks(spot=450, vol=0.20)
            >>> print(f"Portfolio Delta: {greeks['delta']:.2f}")
        """
        # Initialize portfolio Greeks
        portfolio_greeks = {name: 0.0 for name in GREEK_NAMES}

        if not self._structures:
            return portfolio_greeks

        for structure in self._structures:
            try:
                structure_greeks = structure.calculate_net_greeks(
                    spot=spot,
                    vol=vol,
                    rate=rate,
                    current_date=current_date
                )
                for name in GREEK_NAMES:
                    portfolio_greeks[name] += structure_greeks.get(name, 0.0)
            except EmptyStructureError:
                # Skip empty structures
                continue
            except Exception as e:
                logger.warning(
                    f"Failed to calculate Greeks for structure "
                    f"{structure.structure_id}: {e}"
                )

        return portfolio_greeks

    def get_total_exposure(self) -> float:
        """
        Calculate total capital at risk across all positions.

        For options, this is typically the margin requirement plus
        the net premium paid/received.

        Returns:
            Total capital exposure in dollars

        Example:
            >>> exposure = strategy.get_total_exposure()
            >>> print(f"Total Exposure: ${exposure:,.2f}")
        """
        total_exposure = 0.0

        for structure in self._structures:
            # Margin requirement
            total_exposure += self._calculate_structure_margin(structure)

            # For debit structures, add the premium paid
            if structure.net_premium < 0:
                total_exposure += abs(structure.net_premium)

        return total_exposure

    def get_margin_requirement(self) -> float:
        """
        Calculate total margin requirement for all open positions.

        This is a simplified margin calculation. In practice, margin
        varies by broker, account type, and structure type.

        Returns:
            Total margin requirement in dollars

        Note:
            Uses simplified margin rules:
            - Naked options: 20% of notional
            - Spreads: Width of spread
            - For complex structures, sums per-leg margin

        Example:
            >>> margin = strategy.get_margin_requirement()
            >>> print(f"Margin Required: ${margin:,.2f}")
        """
        total_margin = 0.0

        for structure in self._structures:
            total_margin += self._calculate_structure_margin(structure)

        return total_margin

    def _calculate_structure_margin(self, structure: OptionStructure) -> float:
        """
        Calculate margin requirement for a single structure.

        Uses a simplified margin calculation similar to broker requirements:
        - For defined-risk spreads: Width of spread * contracts * 100
        - For naked options: Greater of two formulas:
          a) 20% of underlying - OTM amount + premium
          b) 10% of strike + premium
        - Falls back to default margin per contract if calculation fails

        Args:
            structure: OptionStructure to calculate margin for

        Returns:
            Margin requirement in dollars

        Note:
            This is a simplified approximation. Real broker margin varies
            by account type, underlying, and market conditions.
        """
        if structure.is_empty:
            return 0.0

        # Get structure properties
        net_premium = structure.net_premium
        is_credit = net_premium > 0

        # Try to determine if it's a defined-risk or undefined-risk structure
        try:
            # Check if max loss is bounded (defined-risk)
            max_loss = structure.calculate_max_loss()

            # If max loss is bounded and reasonable (< 5x premium), use that
            if is_credit and abs(max_loss) < abs(net_premium) * 5:
                # Defined risk - margin is max loss
                return abs(max_loss)

        except Exception:
            pass

        # For undefined risk or when max_loss is very large,
        # use a simplified broker-style margin calculation
        try:
            # Calculate based on underlying value and number of contracts
            total_margin = 0.0

            for option in structure.options:
                # Standard option margin: 20% of underlying + premium - OTM amount
                underlying_value = option.underlying_price_at_entry * CONTRACT_MULTIPLIER
                premium_value = option.entry_price * CONTRACT_MULTIPLIER

                # Calculate OTM amount
                if option.is_call:
                    otm_amount = max(option.strike - option.underlying_price_at_entry, 0)
                else:
                    otm_amount = max(option.underlying_price_at_entry - option.strike, 0)
                otm_value = otm_amount * CONTRACT_MULTIPLIER

                # Formula: 20% of underlying + premium - OTM amount
                margin1 = underlying_value * 0.20 + premium_value - otm_value

                # Alternative: 10% of strike + premium
                strike_value = option.strike * CONTRACT_MULTIPLIER
                margin2 = strike_value * 0.10 + premium_value

                # Take greater of the two, but ensure minimum
                option_margin = max(margin1, margin2, premium_value)

                # For long positions, margin is just the premium paid
                if option.is_long:
                    option_margin = premium_value

                total_margin += option_margin * option.quantity

            # Credit received can offset margin requirement
            if is_credit:
                total_margin = max(total_margin - net_premium, net_premium)

            return max(total_margin, 0.0)

        except Exception:
            # Fall back to default margin per contract
            return structure.num_legs * DEFAULT_MARGIN_PER_CONTRACT

    def validate_risk_limits(self) -> Tuple[bool, List[str]]:
        """
        Validate that current positions are within risk limits.

        Checks all configured risk limits including:
        - Maximum positions
        - Maximum delta exposure
        - Maximum vega exposure
        - Maximum capital utilization

        Returns:
            Tuple of (is_valid, list_of_violations)
            - is_valid: True if all limits are satisfied
            - list_of_violations: List of string descriptions of any violations

        Example:
            >>> is_valid, violations = strategy.validate_risk_limits()
            >>> if not is_valid:
            ...     for v in violations:
            ...         print(f"Violation: {v}")
        """
        violations = []

        # Check max positions
        max_positions = self._position_limits.get('max_positions', DEFAULT_MAX_POSITIONS)
        if self.num_open_positions > max_positions:
            violations.append(
                f"Position count ({self.num_open_positions}) exceeds "
                f"max ({max_positions})"
            )

        # Check delta limits
        try:
            greeks = self.calculate_portfolio_greeks()

            max_delta = self._position_limits.get('max_total_delta', DEFAULT_MAX_TOTAL_DELTA)
            if abs(greeks['delta']) > max_delta:
                violations.append(
                    f"Delta exposure ({greeks['delta']:.2f}) exceeds "
                    f"max ({max_delta})"
                )

            max_vega = self._position_limits.get('max_total_vega', DEFAULT_MAX_TOTAL_VEGA)
            if abs(greeks['vega']) > max_vega:
                violations.append(
                    f"Vega exposure ({greeks['vega']:.2f}) exceeds "
                    f"max ({max_vega})"
                )
        except Exception as e:
            logger.warning(f"Could not calculate Greeks for risk validation: {e}")

        # Check capital utilization
        max_utilization = self._position_limits.get(
            'max_capital_utilization',
            DEFAULT_MAX_CAPITAL_UTILIZATION
        )
        if self.capital_utilization > max_utilization:
            violations.append(
                f"Capital utilization ({self.capital_utilization:.1%}) exceeds "
                f"max ({max_utilization:.1%})"
            )

        return len(violations) == 0, violations

    def _validate_new_position(self, structure: OptionStructure) -> None:
        """
        Validate that adding a new position would not breach risk limits.

        Args:
            structure: Structure to potentially add

        Raises:
            RiskLimitError: If adding would breach limits
        """
        # Check position count
        max_positions = self._position_limits.get('max_positions', DEFAULT_MAX_POSITIONS)
        if self.num_open_positions >= max_positions:
            raise RiskLimitError(
                f"Cannot open new position: would exceed max positions "
                f"({max_positions})"
            )

        # Check single position size
        position_margin = self._calculate_structure_margin(structure)
        max_single = self._position_limits.get(
            'max_single_position_size',
            self._initial_capital * 0.20
        )
        if position_margin > max_single:
            raise RiskLimitError(
                f"Position size (${position_margin:,.2f}) exceeds max "
                f"single position size (${max_single:,.2f})"
            )

        # Check capital utilization
        max_utilization = self._position_limits.get(
            'max_capital_utilization',
            DEFAULT_MAX_CAPITAL_UTILIZATION
        )
        new_utilization = (
            (self.total_allocated_capital + position_margin) / self._initial_capital
        )
        if new_utilization > max_utilization:
            raise RiskLimitError(
                f"Adding position would exceed capital utilization limit "
                f"({new_utilization:.1%} > {max_utilization:.1%})"
            )

    # =========================================================================
    # P&L Calculation Methods
    # =========================================================================

    def calculate_unrealized_pnl(self) -> float:
        """
        Calculate unrealized P&L from all open positions.

        Returns:
            Total unrealized P&L in dollars (positive = profit)

        Example:
            >>> unrealized = strategy.calculate_unrealized_pnl()
            >>> print(f"Unrealized P&L: ${unrealized:,.2f}")
        """
        unrealized = 0.0

        for structure in self._structures:
            try:
                unrealized += structure.calculate_pnl()
            except EmptyStructureError:
                continue

        return unrealized

    def calculate_total_pnl(self) -> float:
        """
        Calculate total P&L (realized + unrealized).

        Total P&L = Realized (from closed positions) + Unrealized (from open)

        Returns:
            Total P&L in dollars (positive = profit)

        Example:
            >>> total_pnl = strategy.calculate_total_pnl()
            >>> print(f"Total P&L: ${total_pnl:,.2f}")
        """
        return self._realized_pnl + self.calculate_unrealized_pnl()

    def calculate_return(self) -> float:
        """
        Calculate percentage return on initial capital.

        Return = Total P&L / Initial Capital

        Returns:
            Return as decimal (e.g., 0.15 for 15% return)

        Example:
            >>> ret = strategy.calculate_return()
            >>> print(f"Return: {ret:.2%}")
        """
        if self._initial_capital <= 0:
            return 0.0
        return self.calculate_total_pnl() / self._initial_capital

    def get_equity(self) -> float:
        """
        Get current equity (initial capital + total P&L).

        Returns:
            Current equity value in dollars

        Example:
            >>> equity = strategy.get_equity()
            >>> print(f"Current Equity: ${equity:,.2f}")
        """
        return self._initial_capital + self.calculate_total_pnl()

    # =========================================================================
    # Abstract Methods (to be implemented by concrete strategies)
    # =========================================================================

    @abstractmethod
    def should_enter(self, market_data: Dict[str, Any]) -> bool:
        """
        Determine whether to enter a new position based on market data.

        This is an abstract method that must be implemented by concrete
        strategy subclasses with specific entry logic.

        Args:
            market_data: Dictionary containing current market information.
                May include:
                - 'spot': Current underlying price
                - 'iv_percentile': Current IV percentile
                - 'vix': VIX level
                - 'date': Current date
                - 'option_chain': Available options
                - Strategy-specific indicators

        Returns:
            True if entry conditions are met, False otherwise

        Example implementation:
            >>> def should_enter(self, market_data):
            ...     # Enter when IV is above 50th percentile
            ...     return market_data.get('iv_percentile', 0) > 50
        """
        pass

    @abstractmethod
    def should_exit(
        self,
        structure: OptionStructure,
        market_data: Dict[str, Any]
    ) -> bool:
        """
        Determine whether to exit an existing position.

        This is an abstract method that must be implemented by concrete
        strategy subclasses with specific exit logic.

        Args:
            structure: The OptionStructure to evaluate for exit
            market_data: Dictionary containing current market information

        Returns:
            True if exit conditions are met, False otherwise

        Example implementation:
            >>> def should_exit(self, structure, market_data):
            ...     # Exit at 25% profit or 100% loss
            ...     pnl_pct = structure.calculate_pnl_percent()
            ...     return pnl_pct >= 0.25 or pnl_pct <= -1.0
        """
        pass

    # =========================================================================
    # Statistics and Reporting Methods
    # =========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive strategy statistics.

        Returns:
            Dictionary with strategy performance statistics:
                - 'name': Strategy name
                - 'num_open_positions': Current open positions
                - 'num_closed_positions': Total closed positions
                - 'total_trades': Total number of trades
                - 'realized_pnl': P&L from closed positions
                - 'unrealized_pnl': P&L from open positions
                - 'total_pnl': Total P&L
                - 'return_pct': Percentage return
                - 'equity': Current equity
                - 'capital_utilization': Capital utilization ratio
                - 'win_rate': Percentage of profitable trades
                - 'avg_win': Average profit on winning trades
                - 'avg_loss': Average loss on losing trades

        Example:
            >>> stats = strategy.get_statistics()
            >>> print(f"Win Rate: {stats['win_rate']:.1%}")
        """
        # Basic stats
        stats = {
            'name': self._name,
            'strategy_id': self._strategy_id,
            'num_open_positions': self.num_open_positions,
            'num_closed_positions': self.num_closed_positions,
            'total_trades': self.num_closed_positions,
            'realized_pnl': self._realized_pnl,
            'unrealized_pnl': self.calculate_unrealized_pnl(),
            'total_pnl': self.calculate_total_pnl(),
            'return_pct': self.calculate_return(),
            'equity': self.get_equity(),
            'initial_capital': self._initial_capital,
            'capital': self._capital,
            'capital_utilization': self.capital_utilization,
        }

        # Win/loss analysis from trade history
        closed_trades = [
            t for t in self._trade_history
            if t.get('action') == 'close' and 'realized_pnl' in t
        ]

        if closed_trades:
            wins = [t['realized_pnl'] for t in closed_trades if t['realized_pnl'] > 0]
            losses = [t['realized_pnl'] for t in closed_trades if t['realized_pnl'] <= 0]

            stats['win_rate'] = len(wins) / len(closed_trades) if closed_trades else 0.0
            stats['avg_win'] = np.mean(wins) if wins else 0.0
            stats['avg_loss'] = np.mean(losses) if losses else 0.0
            stats['profit_factor'] = (
                sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else float('inf')
            )
            stats['max_win'] = max(wins) if wins else 0.0
            stats['max_loss'] = min(losses) if losses else 0.0
        else:
            stats['win_rate'] = 0.0
            stats['avg_win'] = 0.0
            stats['avg_loss'] = 0.0
            stats['profit_factor'] = 0.0
            stats['max_win'] = 0.0
            stats['max_loss'] = 0.0

        return stats

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert strategy state to dictionary representation.

        Returns:
            Dictionary with strategy state
        """
        return {
            'strategy_id': self._strategy_id,
            'name': self._name,
            'description': self._description,
            'initial_capital': self._initial_capital,
            'capital': self._capital,
            'realized_pnl': self._realized_pnl,
            'position_limits': self._position_limits,
            'num_open_positions': self.num_open_positions,
            'num_closed_positions': self.num_closed_positions,
            'open_structures': [s.to_dict() for s in self._structures],
            'statistics': self.get_statistics(),
            'created_at': self._created_at,
        }

    # =========================================================================
    # Special Methods
    # =========================================================================

    def __repr__(self) -> str:
        """Return detailed string representation."""
        return (
            f"Strategy("
            f"name={self._name!r}, "
            f"id={self._strategy_id!r}, "
            f"open_positions={self.num_open_positions}, "
            f"capital=${self._capital:,.2f}, "
            f"total_pnl=${self.calculate_total_pnl():,.2f}"
            f")"
        )

    def __str__(self) -> str:
        """Return human-readable string."""
        return (
            f"Strategy '{self._name}': "
            f"{self.num_open_positions} open positions, "
            f"Equity ${self.get_equity():,.2f}, "
            f"P&L ${self.calculate_total_pnl():,.2f} ({self.calculate_return():.1%})"
        )


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Main class
    'Strategy',

    # Exceptions
    'StrategyError',
    'StrategyValidationError',
    'PositionError',
    'RiskLimitError',
    'InsufficientCapitalError',

    # Constants
    'DEFAULT_MAX_POSITIONS',
    'DEFAULT_MAX_TOTAL_DELTA',
    'DEFAULT_MAX_TOTAL_VEGA',
    'DEFAULT_MAX_CAPITAL_UTILIZATION',
    'DEFAULT_MARGIN_PER_CONTRACT',
]
