"""
PositionManager Class for Options Backtesting

This module provides the PositionManager class that tracks positions across
multiple strategies in a backtest. It manages active and closed positions,
calculates portfolio-level metrics, and provides position lookup functionality.

Key Features:
    - Multi-strategy position tracking
    - Position-level and portfolio-level metrics
    - Margin requirement calculation
    - Portfolio Greeks aggregation
    - Mark-to-market valuation
    - Position history tracking

Design Philosophy:
    The PositionManager acts as a central registry for all positions in a
    backtest, abstracting position management from individual strategies.
    This enables portfolio-level risk analysis and cross-strategy position
    tracking.

Financial Correctness:
    - Portfolio value = sum of mark-to-market values across all positions
    - Portfolio Greeks = sum of net Greeks across all structures
    - Margin = sum of individual structure margin requirements
    - P&L tracking separates realized vs unrealized

Usage:
    from backtester.engine.position_manager import PositionManager
    from backtester.core.option_structure import OptionStructure

    manager = PositionManager()

    # Add positions from different strategies
    manager.add_position(iron_condor, strategy_name='IronCondor')
    manager.add_position(straddle, strategy_name='Straddle')

    # Get portfolio metrics
    total_value = manager.calculate_portfolio_value(market_data)
    greeks = manager.get_portfolio_greeks(market_data)

References:
    - Hull, J. C. (2018). Options, Futures, and Other Derivatives.
    - Position management in options trading systems
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

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

# Default margin calculation factors
DEFAULT_NAKED_OPTION_MARGIN_FACTOR = 0.20
DEFAULT_SPREAD_MARGIN_FACTOR = 1.0
DEFAULT_MARGIN_PER_CONTRACT = 2000.0


# =============================================================================
# Exceptions
# =============================================================================

class PositionManagerError(Exception):
    """Base exception for PositionManager errors."""
    pass


class PositionNotFoundError(PositionManagerError):
    """Exception raised when a position is not found."""
    pass


class DuplicatePositionError(PositionManagerError):
    """Exception raised when adding a duplicate position."""
    pass


# =============================================================================
# PositionRecord Data Class
# =============================================================================

class PositionRecord:
    """
    Record containing position details and metadata.

    This class wraps an OptionStructure with additional tracking information
    such as strategy name, open timestamp, and status.

    Attributes:
        structure (OptionStructure): The option structure
        strategy_name (str): Name of the strategy that owns this position
        open_timestamp (datetime): When the position was opened
        close_timestamp (Optional[datetime]): When position was closed
        is_active (bool): Whether the position is currently active
        realized_pnl (Optional[float]): Realized P&L if closed
    """

    __slots__ = (
        '_structure',
        '_strategy_name',
        '_open_timestamp',
        '_close_timestamp',
        '_is_active',
        '_realized_pnl',
        '_record_id',
    )

    def __init__(
        self,
        structure: OptionStructure,
        strategy_name: str,
        open_timestamp: Optional[datetime] = None,
        record_id: Optional[str] = None
    ) -> None:
        """
        Initialize a PositionRecord.

        Args:
            structure: The OptionStructure for this position
            strategy_name: Name of the owning strategy
            open_timestamp: When the position was opened. Defaults to now.
            record_id: Unique identifier. Auto-generated if not provided.
        """
        self._structure = structure
        self._strategy_name = strategy_name
        self._open_timestamp = open_timestamp or datetime.now()
        self._close_timestamp: Optional[datetime] = None
        self._is_active = True
        self._realized_pnl: Optional[float] = None
        self._record_id = record_id or str(uuid.uuid4())[:12]

    @property
    def structure(self) -> OptionStructure:
        """Get the option structure."""
        return self._structure

    @property
    def structure_id(self) -> str:
        """Get the structure ID."""
        return self._structure.structure_id

    @property
    def record_id(self) -> str:
        """Get the record ID."""
        return self._record_id

    @property
    def strategy_name(self) -> str:
        """Get the strategy name."""
        return self._strategy_name

    @property
    def open_timestamp(self) -> datetime:
        """Get the open timestamp."""
        return self._open_timestamp

    @property
    def close_timestamp(self) -> Optional[datetime]:
        """Get the close timestamp."""
        return self._close_timestamp

    @property
    def is_active(self) -> bool:
        """Check if position is active."""
        return self._is_active

    @property
    def realized_pnl(self) -> Optional[float]:
        """Get realized P&L if closed."""
        return self._realized_pnl

    def close(
        self,
        close_timestamp: Optional[datetime] = None,
        realized_pnl: Optional[float] = None
    ) -> None:
        """
        Mark the position as closed.

        Args:
            close_timestamp: When the position was closed
            realized_pnl: The realized P&L from this position
        """
        self._is_active = False
        self._close_timestamp = close_timestamp or datetime.now()
        self._realized_pnl = realized_pnl

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'record_id': self._record_id,
            'structure_id': self._structure.structure_id,
            'strategy_name': self._strategy_name,
            'structure_type': self._structure.structure_type,
            'underlying': self._structure.underlying,
            'open_timestamp': self._open_timestamp,
            'close_timestamp': self._close_timestamp,
            'is_active': self._is_active,
            'realized_pnl': self._realized_pnl,
            'num_legs': self._structure.num_legs,
            'net_premium': self._structure.net_premium,
        }

    def __repr__(self) -> str:
        """Return string representation."""
        status = "ACTIVE" if self._is_active else "CLOSED"
        return (
            f"PositionRecord({self._structure.structure_type.upper()} "
            f"on {self._structure.underlying}, "
            f"strategy={self._strategy_name!r}, "
            f"status={status})"
        )


# =============================================================================
# PositionManager Class
# =============================================================================

class PositionManager:
    """
    Track positions across multiple strategies.

    This class serves as a central registry for all option positions in a
    backtest. It provides methods for adding, removing, and querying positions,
    as well as calculating portfolio-level metrics.

    Attributes:
        positions (Dict[str, PositionRecord]): Active positions by structure_id
        closed_positions (List[PositionRecord]): Historical closed positions
        strategy_positions (Dict[str, List[str]]): Position IDs by strategy

    Example:
        >>> manager = PositionManager()
        >>> manager.add_position(iron_condor, 'IronCondorStrategy')
        >>> manager.add_position(straddle, 'StraddleStrategy')
        >>>
        >>> # Get all active positions
        >>> all_positions = manager.get_all_positions()
        >>>
        >>> # Get positions for a specific strategy
        >>> condor_positions = manager.get_positions_by_strategy('IronCondorStrategy')
        >>>
        >>> # Calculate portfolio metrics
        >>> value = manager.calculate_portfolio_value(market_data)
        >>> greeks = manager.get_portfolio_greeks(market_data)
    """

    __slots__ = (
        '_positions',
        '_closed_positions',
        '_strategy_positions',
        '_position_history',
        '_total_realized_pnl',
    )

    def __init__(self) -> None:
        """
        Initialize the PositionManager.

        Example:
            >>> manager = PositionManager()
        """
        # Active positions indexed by structure_id
        self._positions: Dict[str, PositionRecord] = {}

        # Closed positions (historical)
        self._closed_positions: List[PositionRecord] = []

        # Positions grouped by strategy name
        self._strategy_positions: Dict[str, List[str]] = {}

        # Complete position history for analysis
        self._position_history: List[Dict[str, Any]] = []

        # Accumulated realized P&L
        self._total_realized_pnl = 0.0

        logger.debug("PositionManager initialized")

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def num_positions(self) -> int:
        """Get number of active positions."""
        return len(self._positions)

    @property
    def num_closed_positions(self) -> int:
        """Get number of closed positions."""
        return len(self._closed_positions)

    @property
    def total_realized_pnl(self) -> float:
        """Get total realized P&L from closed positions."""
        return self._total_realized_pnl

    @property
    def strategy_names(self) -> List[str]:
        """Get list of strategy names with positions."""
        return list(self._strategy_positions.keys())

    # =========================================================================
    # Position Management Methods
    # =========================================================================

    def add_position(
        self,
        structure: OptionStructure,
        strategy_name: str,
        open_timestamp: Optional[datetime] = None
    ) -> str:
        """
        Add a new position to tracking.

        Args:
            structure: OptionStructure representing the position
            strategy_name: Name of the strategy that owns this position
            open_timestamp: When the position was opened. Defaults to now.

        Returns:
            The structure_id of the added position

        Raises:
            DuplicatePositionError: If structure is already tracked
            ValueError: If structure is None or invalid

        Example:
            >>> structure_id = manager.add_position(
            ...     iron_condor,
            ...     strategy_name='IronCondor',
            ...     open_timestamp=datetime(2023, 6, 1)
            ... )
        """
        # Validate structure
        if structure is None:
            raise ValueError("structure cannot be None")
        if not isinstance(structure, OptionStructure):
            raise ValueError(
                f"Expected OptionStructure, got {type(structure).__name__}"
            )

        # Check for duplicates
        if structure.structure_id in self._positions:
            raise DuplicatePositionError(
                f"Structure {structure.structure_id} is already being tracked"
            )

        # Create position record
        record = PositionRecord(
            structure=structure,
            strategy_name=strategy_name,
            open_timestamp=open_timestamp
        )

        # Add to positions
        self._positions[structure.structure_id] = record

        # Add to strategy grouping
        if strategy_name not in self._strategy_positions:
            self._strategy_positions[strategy_name] = []
        self._strategy_positions[strategy_name].append(structure.structure_id)

        # Record in history
        self._position_history.append({
            'action': 'open',
            'structure_id': structure.structure_id,
            'strategy_name': strategy_name,
            'timestamp': record.open_timestamp,
            'structure_type': structure.structure_type,
            'underlying': structure.underlying,
            'net_premium': structure.net_premium,
        })

        logger.debug(
            f"Added position: {structure.structure_type.upper()} on "
            f"{structure.underlying}, strategy={strategy_name}"
        )

        return structure.structure_id

    def remove_position(
        self,
        structure_id: str,
        close_timestamp: Optional[datetime] = None,
        realized_pnl: Optional[float] = None
    ) -> OptionStructure:
        """
        Remove a closed position from tracking.

        Moves the position from active to closed and records the closing details.

        Args:
            structure_id: ID of the structure to remove
            close_timestamp: When the position was closed. Defaults to now.
            realized_pnl: The realized P&L from this position. If None,
                         calculated from the structure.

        Returns:
            The removed OptionStructure

        Raises:
            PositionNotFoundError: If structure_id is not found

        Example:
            >>> closed_structure = manager.remove_position(
            ...     structure_id='abc123',
            ...     realized_pnl=1500.0
            ... )
        """
        if structure_id not in self._positions:
            raise PositionNotFoundError(
                f"Position {structure_id} not found in active positions"
            )

        # Get the record
        record = self._positions[structure_id]

        # Calculate realized P&L if not provided
        if realized_pnl is None:
            try:
                realized_pnl = record.structure.calculate_pnl()
            except EmptyStructureError:
                realized_pnl = 0.0

        # Close the record
        record.close(close_timestamp=close_timestamp, realized_pnl=realized_pnl)

        # Remove from active positions
        del self._positions[structure_id]

        # Remove from strategy grouping
        strategy_name = record.strategy_name
        if strategy_name in self._strategy_positions:
            if structure_id in self._strategy_positions[strategy_name]:
                self._strategy_positions[strategy_name].remove(structure_id)
            # Clean up empty strategy lists
            if not self._strategy_positions[strategy_name]:
                del self._strategy_positions[strategy_name]

        # Add to closed positions
        self._closed_positions.append(record)

        # Update total realized P&L
        if realized_pnl is not None:
            self._total_realized_pnl += realized_pnl

        # Record in history
        self._position_history.append({
            'action': 'close',
            'structure_id': structure_id,
            'strategy_name': record.strategy_name,
            'timestamp': record.close_timestamp,
            'realized_pnl': realized_pnl,
        })

        logger.debug(
            f"Removed position: {record.structure.structure_type.upper()} on "
            f"{record.structure.underlying}, P&L=${realized_pnl:,.2f}"
        )

        return record.structure

    def has_position(self, structure_id: str) -> bool:
        """
        Check if a position is currently tracked.

        Args:
            structure_id: ID of the structure to check

        Returns:
            True if position is active, False otherwise
        """
        return structure_id in self._positions

    def get_position(self, structure_id: str) -> Optional[OptionStructure]:
        """
        Get a specific active position by ID.

        Args:
            structure_id: ID of the structure to retrieve

        Returns:
            OptionStructure if found, None otherwise

        Example:
            >>> structure = manager.get_position('abc123')
            >>> if structure:
            ...     print(f"Found: {structure}")
        """
        record = self._positions.get(structure_id)
        return record.structure if record else None

    def get_position_record(self, structure_id: str) -> Optional[PositionRecord]:
        """
        Get the full position record by ID.

        Args:
            structure_id: ID of the structure to retrieve

        Returns:
            PositionRecord if found, None otherwise
        """
        return self._positions.get(structure_id)

    # =========================================================================
    # Position Query Methods
    # =========================================================================

    def get_all_positions(self) -> List[OptionStructure]:
        """
        Get all active positions.

        Returns:
            List of all active OptionStructure instances

        Example:
            >>> positions = manager.get_all_positions()
            >>> for pos in positions:
            ...     print(f"{pos.structure_type}: {pos.underlying}")
        """
        return [record.structure for record in self._positions.values()]

    def get_all_records(self) -> List[PositionRecord]:
        """
        Get all active position records.

        Returns:
            List of all active PositionRecord instances
        """
        return list(self._positions.values())

    def get_positions_by_strategy(self, strategy_name: str) -> List[OptionStructure]:
        """
        Get positions for a specific strategy.

        Args:
            strategy_name: Name of the strategy to filter by

        Returns:
            List of OptionStructure instances for the strategy

        Example:
            >>> condor_positions = manager.get_positions_by_strategy('IronCondor')
        """
        structure_ids = self._strategy_positions.get(strategy_name, [])
        structures = []
        for sid in structure_ids:
            record = self._positions.get(sid)
            if record:
                structures.append(record.structure)
        return structures

    def get_positions_by_underlying(self, underlying: str) -> List[OptionStructure]:
        """
        Get positions for a specific underlying.

        Args:
            underlying: Ticker symbol to filter by

        Returns:
            List of OptionStructure instances for the underlying

        Example:
            >>> spy_positions = manager.get_positions_by_underlying('SPY')
        """
        underlying_upper = underlying.upper()
        return [
            record.structure
            for record in self._positions.values()
            if record.structure.underlying == underlying_upper
        ]

    def get_closed_positions(self) -> List[PositionRecord]:
        """
        Get all closed position records.

        Returns:
            List of closed PositionRecord instances
        """
        return self._closed_positions.copy()

    # =========================================================================
    # Portfolio Metrics Methods
    # =========================================================================

    def calculate_total_margin(self) -> float:
        """
        Calculate margin requirement across all positions.

        Uses simplified margin calculation similar to broker requirements.

        Returns:
            Total margin requirement in dollars

        Example:
            >>> margin = manager.calculate_total_margin()
            >>> print(f"Total Margin: ${margin:,.2f}")
        """
        total_margin = 0.0

        for record in self._positions.values():
            structure = record.structure
            margin = self._calculate_structure_margin(structure)
            total_margin += margin

        return total_margin

    def _calculate_structure_margin(self, structure: OptionStructure) -> float:
        """
        Calculate margin for a single structure.

        Uses simplified broker-style margin calculation.

        Args:
            structure: OptionStructure to calculate margin for

        Returns:
            Margin requirement in dollars
        """
        if structure.is_empty:
            return 0.0

        net_premium = structure.net_premium
        is_credit = net_premium > 0

        # Try to use max loss for defined-risk structures
        try:
            max_loss = structure.calculate_max_loss()
            if is_credit and abs(max_loss) < abs(net_premium) * 5:
                return abs(max_loss)
        except Exception:
            pass

        # Fall back to per-leg margin calculation
        try:
            total_margin = 0.0
            for option in structure.options:
                underlying_value = option.underlying_price_at_entry * CONTRACT_MULTIPLIER
                premium_value = option.entry_price * CONTRACT_MULTIPLIER

                # Standard margin calculation
                margin1 = underlying_value * DEFAULT_NAKED_OPTION_MARGIN_FACTOR + premium_value
                margin2 = option.strike * CONTRACT_MULTIPLIER * 0.10 + premium_value

                option_margin = max(margin1, margin2, premium_value)

                # Long positions only require premium
                if option.is_long:
                    option_margin = premium_value

                total_margin += option_margin * option.quantity

            # Credit offset
            if is_credit:
                total_margin = max(total_margin - net_premium, net_premium)

            return max(total_margin, 0.0)

        except Exception:
            return structure.num_legs * DEFAULT_MARGIN_PER_CONTRACT

    def calculate_portfolio_value(self, market_data: Dict[str, Any]) -> float:
        """
        Calculate mark-to-market value of all positions.

        Returns the current market value of all active positions:
        - Long positions: positive value (assets)
        - Short positions: negative value (liabilities)

        This is NOT the same as P&L. For proper equity calculation:
        equity = cash + portfolio_value

        Where cash includes premiums received from short sales, and
        portfolio_value includes the liability (negative) for those
        short positions.

        Args:
            market_data: Dictionary containing current market data with
                        option prices and spot price

        Returns:
            Total portfolio market value in dollars (negative for net short)

        Example:
            >>> value = manager.calculate_portfolio_value({'spot': 450, ...})
            >>> print(f"Portfolio Value: ${value:,.2f}")
        """
        total_value = 0.0

        for record in self._positions.values():
            structure = record.structure

            if structure.is_empty:
                continue

            try:
                # Get current market value of the structure
                # For long positions: positive (asset)
                # For short positions: negative (liability)
                market_value = structure.get_current_value()
                total_value += market_value
            except EmptyStructureError:
                continue
            except Exception as e:
                logger.warning(
                    f"Error calculating value for {structure.structure_id}: {e}"
                )

        return total_value

    def get_portfolio_greeks(
        self,
        market_data: Dict[str, Any],
        rate: float = 0.04
    ) -> Dict[str, float]:
        """
        Calculate aggregate Greeks across all positions.

        Sums the net Greeks from each active structure to get
        portfolio-level Greek exposures.

        Args:
            market_data: Dictionary containing 'spot' and 'iv' keys
            rate: Risk-free rate for Greek calculations

        Returns:
            Dictionary with portfolio Greeks:
                - 'delta': Total delta exposure
                - 'gamma': Total gamma exposure
                - 'theta': Total daily theta
                - 'vega': Total vega exposure
                - 'rho': Total rho exposure

        Example:
            >>> greeks = manager.get_portfolio_greeks({'spot': 450, 'iv': 0.20})
            >>> print(f"Portfolio Delta: {greeks['delta']:.2f}")
        """
        # Initialize portfolio Greeks
        portfolio_greeks = {name: 0.0 for name in GREEK_NAMES}

        if not self._positions:
            return portfolio_greeks

        # Get market parameters
        spot = market_data.get('spot', 0.0)
        vol = market_data.get('iv', 0.20)
        current_date = market_data.get('date', datetime.now())

        for record in self._positions.values():
            structure = record.structure

            if structure.is_empty:
                continue

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
                continue
            except Exception as e:
                logger.warning(
                    f"Error calculating Greeks for {structure.structure_id}: {e}"
                )

        return portfolio_greeks

    def calculate_unrealized_pnl(self) -> float:
        """
        Calculate total unrealized P&L across all active positions.

        Returns:
            Total unrealized P&L in dollars
        """
        unrealized = 0.0

        for record in self._positions.values():
            try:
                unrealized += record.structure.calculate_pnl()
            except EmptyStructureError:
                continue

        return unrealized

    def calculate_total_pnl(self) -> float:
        """
        Calculate total P&L (realized + unrealized).

        Returns:
            Total P&L in dollars
        """
        return self._total_realized_pnl + self.calculate_unrealized_pnl()

    # =========================================================================
    # Position Update Methods
    # =========================================================================

    def update_all_positions(
        self,
        market_data: Dict[str, Any],
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Update all positions with current market data.

        Args:
            market_data: Dictionary containing option prices
            timestamp: Current timestamp

        Example:
            >>> manager.update_all_positions(market_data, datetime.now())
        """
        if timestamp is None:
            timestamp = datetime.now()

        for record in self._positions.values():
            try:
                record.structure.update_prices_from_market_data(
                    market_data=market_data,
                    timestamp=timestamp
                )
            except Exception as e:
                logger.warning(
                    f"Failed to update position {record.structure_id}: {e}"
                )

    # =========================================================================
    # Statistics and Reporting Methods
    # =========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive position statistics.

        Returns:
            Dictionary with position statistics

        Example:
            >>> stats = manager.get_statistics()
            >>> print(f"Active Positions: {stats['num_active']}")
        """
        stats = {
            'num_active': self.num_positions,
            'num_closed': self.num_closed_positions,
            'total_trades': self.num_positions + self.num_closed_positions,
            'realized_pnl': self._total_realized_pnl,
            'unrealized_pnl': self.calculate_unrealized_pnl(),
            'total_pnl': self.calculate_total_pnl(),
            'total_margin': self.calculate_total_margin(),
            'num_strategies': len(self._strategy_positions),
            'strategies': list(self._strategy_positions.keys()),
        }

        # Per-strategy breakdown
        strategy_stats = {}
        for strategy_name in self._strategy_positions:
            positions = self.get_positions_by_strategy(strategy_name)
            strategy_stats[strategy_name] = {
                'num_positions': len(positions),
                'position_ids': [p.structure_id for p in positions],
            }
        stats['by_strategy'] = strategy_stats

        # Underlying breakdown
        underlyings = set()
        for record in self._positions.values():
            if record.structure.underlying:
                underlyings.add(record.structure.underlying)
        stats['underlyings'] = list(underlyings)

        return stats

    def get_position_summary(self) -> pd.DataFrame:
        """
        Get a summary DataFrame of all positions.

        Returns:
            DataFrame with position summary

        Example:
            >>> summary = manager.get_position_summary()
            >>> print(summary)
        """
        records = []

        for record in self._positions.values():
            structure = record.structure
            try:
                pnl = structure.calculate_pnl()
            except EmptyStructureError:
                pnl = 0.0

            records.append({
                'structure_id': structure.structure_id,
                'strategy': record.strategy_name,
                'type': structure.structure_type,
                'underlying': structure.underlying,
                'num_legs': structure.num_legs,
                'net_premium': structure.net_premium,
                'current_pnl': pnl,
                'open_date': record.open_timestamp,
                'is_active': record.is_active,
            })

        if not records:
            return pd.DataFrame()

        return pd.DataFrame(records)

    def get_history(self) -> List[Dict[str, Any]]:
        """
        Get position history (opens and closes).

        Returns:
            List of position history events
        """
        return self._position_history.copy()

    def clear(self) -> None:
        """
        Clear all positions and history.

        Use with caution - this removes all tracking data.
        """
        self._positions.clear()
        self._closed_positions.clear()
        self._strategy_positions.clear()
        self._position_history.clear()
        self._total_realized_pnl = 0.0

        logger.info("PositionManager cleared")

    # =========================================================================
    # Iterator Protocol
    # =========================================================================

    def __iter__(self) -> Iterator[OptionStructure]:
        """Iterate over active positions."""
        return iter(self.get_all_positions())

    def __len__(self) -> int:
        """Return number of active positions."""
        return self.num_positions

    def __contains__(self, structure_id: str) -> bool:
        """Check if structure_id is in active positions."""
        return structure_id in self._positions

    # =========================================================================
    # Special Methods
    # =========================================================================

    def __repr__(self) -> str:
        """Return detailed string representation."""
        return (
            f"PositionManager("
            f"active={self.num_positions}, "
            f"closed={self.num_closed_positions}, "
            f"strategies={len(self._strategy_positions)}, "
            f"realized_pnl=${self._total_realized_pnl:,.2f}"
            f")"
        )

    def __str__(self) -> str:
        """Return human-readable string."""
        return (
            f"PositionManager: "
            f"{self.num_positions} active positions, "
            f"{self.num_closed_positions} closed, "
            f"Total P&L ${self.calculate_total_pnl():,.2f}"
        )


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Main classes
    'PositionManager',
    'PositionRecord',

    # Exceptions
    'PositionManagerError',
    'PositionNotFoundError',
    'DuplicatePositionError',

    # Constants
    'DEFAULT_MARGIN_PER_CONTRACT',
]
