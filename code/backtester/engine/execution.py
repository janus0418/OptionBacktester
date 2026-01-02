"""
ExecutionModel Class for Options Backtesting

This module provides the ExecutionModel class that simulates realistic order
execution for options trades. It models bid/ask spread, commissions, slippage,
and partial fills to provide accurate backtest results.

Key Features:
    - Bid/ask spread modeling (buy at ask, sell at bid)
    - Configurable commission per contract
    - Optional slippage percentage
    - Fill price calculation
    - Entry and exit execution simulation
    - Transaction cost breakdown

Design Philosophy:
    The ExecutionModel separates execution concerns from trading logic. It
    provides realistic fill prices that account for market microstructure,
    enabling accurate P&L and risk calculations.

Financial Correctness:
    - Buy orders fill at ask price (or mid + half spread)
    - Sell orders fill at bid price (or mid - half spread)
    - Commission = per_contract_commission * num_contracts * num_legs
    - Slippage is applied as a percentage on top of the fill price
    - Total cost includes premium, commission, and slippage

Usage:
    from backtester.engine.execution import ExecutionModel

    execution = ExecutionModel(
        commission_per_contract=0.65,
        slippage_pct=0.001,
        use_bid_ask=True
    )

    # Execute entry
    entry_result = execution.execute_entry(structure, market_data)
    print(f"Entry cost: ${entry_result['total_cost']:,.2f}")

    # Execute exit
    exit_result = execution.execute_exit(structure, market_data)
    print(f"Exit proceeds: ${exit_result['total_proceeds']:,.2f}")

References:
    - Market microstructure: https://www.cmegroup.com/education/courses/
    - Options execution: https://www.cboe.com/
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from backtester.core.option_structure import (
    OptionStructure,
    EmptyStructureError,
)
from backtester.core.option import Option, CONTRACT_MULTIPLIER

# Configure module logger
logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Standard commission rates (per contract)
DEFAULT_COMMISSION_PER_CONTRACT = 0.65  # $0.65 per contract

# Default slippage percentage
DEFAULT_SLIPPAGE_PCT = 0.0  # No slippage by default

# Default bid/ask spread as percentage of mid price
DEFAULT_SPREAD_PCT = 0.02  # 2% of mid price

# Minimum tick size for options
MINIMUM_TICK_SIZE = 0.01

# Maximum reasonable spread percentage
MAX_SPREAD_PCT = 0.50  # 50% - beyond this is likely bad data


# =============================================================================
# Exceptions
# =============================================================================


class ExecutionError(Exception):
    """Base exception for execution errors."""

    pass


class ExecutionConfigError(ExecutionError):
    """Exception raised for configuration errors."""

    pass


class PriceNotAvailableError(ExecutionError):
    """Exception raised when price data is not available."""

    pass


class FillError(ExecutionError):
    """Exception raised when order cannot be filled."""

    pass


# =============================================================================
# Execution Result Classes
# =============================================================================


class ExecutionResult:
    """
    Container for execution results.

    Attributes:
        entry_prices (Dict[str, float]): Fill prices by option identifier
        commissions (float): Total commission paid
        slippage (float): Total slippage cost
        total_cost (float): Total cost including premium, commission, slippage
        timestamp (datetime): When execution occurred
        fills (List[Dict]): Detailed fill information per leg
    """

    __slots__ = (
        "_entry_prices",
        "_commissions",
        "_slippage",
        "_total_cost",
        "_total_proceeds",
        "_timestamp",
        "_fills",
        "_is_entry",
        "_num_contracts",
    )

    def __init__(
        self,
        entry_prices: Dict[str, float],
        commissions: float,
        slippage: float,
        total_cost: float,
        total_proceeds: float,
        timestamp: datetime,
        fills: List[Dict[str, Any]],
        is_entry: bool,
        num_contracts: int,
    ) -> None:
        """Initialize execution result."""
        self._entry_prices = entry_prices
        self._commissions = commissions
        self._slippage = slippage
        self._total_cost = total_cost
        self._total_proceeds = total_proceeds
        self._timestamp = timestamp
        self._fills = fills
        self._is_entry = is_entry
        self._num_contracts = num_contracts

    @property
    def entry_prices(self) -> Dict[str, float]:
        """Get fill prices by option identifier."""
        return self._entry_prices.copy()

    @property
    def commissions(self) -> float:
        """Get total commission."""
        return self._commissions

    @property
    def slippage(self) -> float:
        """Get total slippage cost."""
        return self._slippage

    @property
    def total_cost(self) -> float:
        """Get total cost (for entries)."""
        return self._total_cost

    @property
    def total_proceeds(self) -> float:
        """Get total proceeds (for exits)."""
        return self._total_proceeds

    @property
    def timestamp(self) -> datetime:
        """Get execution timestamp."""
        return self._timestamp

    @property
    def fills(self) -> List[Dict[str, Any]]:
        """Get detailed fill information."""
        return self._fills.copy()

    @property
    def is_entry(self) -> bool:
        """Check if this is an entry execution."""
        return self._is_entry

    @property
    def num_contracts(self) -> int:
        """Get total number of contracts executed."""
        return self._num_contracts

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "entry_prices": self._entry_prices,
            "commissions": self._commissions,
            "slippage": self._slippage,
            "total_cost": self._total_cost,
            "total_proceeds": self._total_proceeds,
            "timestamp": self._timestamp,
            "fills": self._fills,
            "is_entry": self._is_entry,
            "num_contracts": self._num_contracts,
        }

    def __repr__(self) -> str:
        """Return string representation."""
        action = "Entry" if self._is_entry else "Exit"
        return (
            f"ExecutionResult({action}, "
            f"cost=${self._total_cost:,.2f}, "
            f"proceeds=${self._total_proceeds:,.2f}, "
            f"commission=${self._commissions:.2f}, "
            f"contracts={self._num_contracts})"
        )


# =============================================================================
# ExecutionModel Class
# =============================================================================


class ExecutionModel:
    """
    Simulate realistic order execution for options trades.

    This class models the execution of option trades, accounting for:
    - Bid/ask spread: buy at ask, sell at bid
    - Commissions: per-contract fees
    - Slippage: additional cost from market impact

    Attributes:
        commission_per_contract (float): Commission per contract
        slippage_pct (float): Slippage as percentage of fill price
        use_bid_ask (bool): Whether to use bid/ask prices
        default_spread_pct (float): Default spread if bid/ask not available

    Example:
        >>> execution = ExecutionModel(
        ...     commission_per_contract=0.65,
        ...     slippage_pct=0.001,
        ...     use_bid_ask=True
        ... )
        >>> result = execution.execute_entry(structure, market_data)
        >>> print(f"Entry cost: ${result['total_cost']:,.2f}")
    """

    __slots__ = (
        "_commission_per_contract",
        "_slippage_pct",
        "_use_bid_ask",
        "_default_spread_pct",
        "_execution_log",
    )

    def __init__(
        self,
        commission_per_contract: float = DEFAULT_COMMISSION_PER_CONTRACT,
        slippage_pct: float = DEFAULT_SLIPPAGE_PCT,
        use_bid_ask: bool = True,
        default_spread_pct: float = DEFAULT_SPREAD_PCT,
    ) -> None:
        """
        Initialize the ExecutionModel.

        Args:
            commission_per_contract: Commission per contract (default $0.65)
            slippage_pct: Slippage as percentage of fill price (default 0%)
            use_bid_ask: Whether to use bid/ask prices for fills (default True)
                        If False, uses mid price for all fills.
            default_spread_pct: Default spread percentage when bid/ask not
                               available (default 2%)

        Raises:
            ExecutionConfigError: If parameters are invalid

        Example:
            >>> execution = ExecutionModel(
            ...     commission_per_contract=0.65,
            ...     slippage_pct=0.001,
            ...     use_bid_ask=True
            ... )
        """
        # Validate commission
        if commission_per_contract < 0:
            raise ExecutionConfigError(
                f"commission_per_contract must be non-negative, "
                f"got {commission_per_contract}"
            )
        self._commission_per_contract = float(commission_per_contract)

        # Validate slippage
        if slippage_pct < 0 or slippage_pct > 1:
            raise ExecutionConfigError(
                f"slippage_pct must be between 0 and 1, got {slippage_pct}"
            )
        self._slippage_pct = float(slippage_pct)

        # Set bid/ask usage
        self._use_bid_ask = bool(use_bid_ask)

        # Validate default spread
        if default_spread_pct < 0 or default_spread_pct > MAX_SPREAD_PCT:
            raise ExecutionConfigError(
                f"default_spread_pct must be between 0 and {MAX_SPREAD_PCT}, "
                f"got {default_spread_pct}"
            )
        self._default_spread_pct = float(default_spread_pct)

        # Execution log for analysis
        self._execution_log: List[Dict[str, Any]] = []

        logger.debug(
            f"ExecutionModel initialized: commission=${commission_per_contract:.2f}, "
            f"slippage={slippage_pct:.2%}, use_bid_ask={use_bid_ask}"
        )

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def commission_per_contract(self) -> float:
        """Get commission per contract."""
        return self._commission_per_contract

    @property
    def slippage_pct(self) -> float:
        """Get slippage percentage."""
        return self._slippage_pct

    @property
    def use_bid_ask(self) -> bool:
        """Check if bid/ask prices are used."""
        return self._use_bid_ask

    @property
    def default_spread_pct(self) -> float:
        """Get default spread percentage."""
        return self._default_spread_pct

    @property
    def execution_log(self) -> List[Dict[str, Any]]:
        """Get execution log."""
        return self._execution_log.copy()

    # =========================================================================
    # Main Execution Methods
    # =========================================================================

    def execute_entry(
        self,
        structure: OptionStructure,
        market_data: Dict[str, Any],
        timestamp: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Simulate opening a position.

        For each leg in the structure:
        - Long positions: buy at ask (or mid + half spread)
        - Short positions: sell at bid (or mid - half spread)

        Args:
            structure: OptionStructure to open
            market_data: Dictionary containing option chain with bid/ask prices
            timestamp: Execution timestamp. Defaults to now.

        Returns:
            Dictionary with execution details:
                - 'entry_prices': Dict[option_id, fill_price]
                - 'commissions': Total commission
                - 'slippage': Total slippage cost
                - 'total_cost': Total cost (negative for credits)
                - 'total_premium': Net premium from fills
                - 'fills': List of fill details per leg
                - 'timestamp': Execution timestamp

        Raises:
            ExecutionError: If execution fails
            EmptyStructureError: If structure has no options

        Example:
            >>> result = execution.execute_entry(iron_condor, market_data)
            >>> print(f"Entry cost: ${result['total_cost']:,.2f}")
        """
        if structure.is_empty:
            raise EmptyStructureError("Cannot execute entry for empty structure")

        if timestamp is None:
            timestamp = datetime.now()

        # Get option chain from market data
        option_chain = market_data.get("option_chain")
        spot = market_data.get("spot", 0.0)

        entry_prices = {}
        fills = []
        total_premium = 0.0
        total_slippage = 0.0
        total_contracts = 0

        for option in structure.options:
            # Determine if this leg is opening long or short
            is_buying = option.is_long  # Long = buy, Short = sell

            # Get fill price
            fill_price = self.get_fill_price(
                option=option,
                market_data=market_data,
                side="buy" if is_buying else "sell",
            )

            # Apply slippage
            slippage_amount = 0.0
            if self._slippage_pct > 0:
                slippage_amount = fill_price * self._slippage_pct
                if is_buying:
                    fill_price += slippage_amount  # Pay more when buying
                else:
                    fill_price -= slippage_amount  # Receive less when selling
                    slippage_amount = -slippage_amount  # Slippage always costs us

            # Ensure non-negative price
            fill_price = max(fill_price, 0.0)

            # Calculate premium for this leg
            # Buying: pay premium (negative to portfolio)
            # Selling: receive premium (positive to portfolio)
            leg_premium = fill_price * option.quantity * CONTRACT_MULTIPLIER
            if is_buying:
                leg_premium = -leg_premium  # Paying premium

            total_premium += leg_premium
            total_slippage += (
                abs(slippage_amount) * option.quantity * CONTRACT_MULTIPLIER
            )
            total_contracts += option.quantity

            # Create option identifier
            option_id = self._create_option_id(option)
            entry_prices[option_id] = fill_price

            # Record fill
            fills.append(
                {
                    "option_id": option_id,
                    "option_type": option.option_type,
                    "position_type": option.position_type,
                    "strike": option.strike,
                    "expiration": option.expiration,
                    "quantity": option.quantity,
                    "side": "buy" if is_buying else "sell",
                    "fill_price": fill_price,
                    "slippage": slippage_amount,
                    "premium": leg_premium,
                }
            )

        # Calculate commission
        total_commission = self._calculate_commission(num_contracts=total_contracts)

        # Total cost = premium paid + commission + slippage
        # For credit trades, premium is positive (receive), so total_cost is negative
        total_cost = -total_premium + total_commission + total_slippage

        # Log execution
        execution_record = {
            "type": "entry",
            "structure_id": structure.structure_id,
            "structure_type": structure.structure_type,
            "timestamp": timestamp,
            "num_legs": len(structure.options),
            "num_contracts": total_contracts,
            "total_premium": total_premium,
            "total_cost": total_cost,
            "commissions": total_commission,
            "slippage": total_slippage,
            "fills": fills,
        }
        self._execution_log.append(execution_record)

        logger.debug(
            f"Executed entry: {structure.structure_type.upper()}, "
            f"premium=${total_premium:,.2f}, cost=${total_cost:,.2f}"
        )

        return {
            "entry_prices": entry_prices,
            "commissions": total_commission,
            "slippage": total_slippage,
            "total_cost": total_cost,
            "total_premium": total_premium,
            "fills": fills,
            "timestamp": timestamp,
            "num_contracts": total_contracts,
        }

    def execute_exit(
        self,
        structure: OptionStructure,
        market_data: Dict[str, Any],
        timestamp: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Simulate closing a position.

        For each leg in the structure:
        - Long positions: sell at bid (close buy by selling)
        - Short positions: buy at ask (close sell by buying)

        Args:
            structure: OptionStructure to close
            market_data: Dictionary containing option chain with bid/ask prices
            timestamp: Execution timestamp. Defaults to now.

        Returns:
            Dictionary with execution details:
                - 'exit_prices': Dict[option_id, fill_price]
                - 'commissions': Total commission
                - 'slippage': Total slippage cost
                - 'total_proceeds': Net proceeds from closing
                - 'total_premium': Net premium from fills
                - 'fills': List of fill details per leg
                - 'timestamp': Execution timestamp

        Raises:
            ExecutionError: If execution fails
            EmptyStructureError: If structure has no options

        Example:
            >>> result = execution.execute_exit(iron_condor, market_data)
            >>> print(f"Exit proceeds: ${result['total_proceeds']:,.2f}")
        """
        if structure.is_empty:
            raise EmptyStructureError("Cannot execute exit for empty structure")

        if timestamp is None:
            timestamp = datetime.now()

        exit_prices = {}
        fills = []
        total_premium = 0.0
        total_slippage = 0.0
        total_contracts = 0

        for option in structure.options:
            # Determine if this leg is closing long or short
            # Long positions: sell to close
            # Short positions: buy to close
            is_buying = option.is_short  # Short = buy to close, Long = sell to close

            # Get fill price
            fill_price = self.get_fill_price(
                option=option,
                market_data=market_data,
                side="buy" if is_buying else "sell",
            )

            # Apply slippage
            slippage_amount = 0.0
            if self._slippage_pct > 0:
                slippage_amount = fill_price * self._slippage_pct
                if is_buying:
                    fill_price += slippage_amount  # Pay more when buying
                else:
                    fill_price -= slippage_amount  # Receive less when selling
                    slippage_amount = -slippage_amount

            # Ensure non-negative price
            fill_price = max(fill_price, 0.0)

            # Calculate premium for this leg
            # Buying to close: pay premium (negative)
            # Selling to close: receive premium (positive)
            leg_premium = fill_price * option.quantity * CONTRACT_MULTIPLIER
            if is_buying:
                leg_premium = -leg_premium  # Paying to close

            total_premium += leg_premium
            total_slippage += (
                abs(slippage_amount) * option.quantity * CONTRACT_MULTIPLIER
            )
            total_contracts += option.quantity

            # Create option identifier
            option_id = self._create_option_id(option)
            exit_prices[option_id] = fill_price

            # Record fill
            fills.append(
                {
                    "option_id": option_id,
                    "option_type": option.option_type,
                    "position_type": option.position_type,
                    "strike": option.strike,
                    "expiration": option.expiration,
                    "quantity": option.quantity,
                    "side": "buy" if is_buying else "sell",
                    "fill_price": fill_price,
                    "slippage": slippage_amount,
                    "premium": leg_premium,
                }
            )

        # Calculate commission
        total_commission = self._calculate_commission(num_contracts=total_contracts)

        # Total proceeds = premium received - commission - slippage
        # For debit to close, premium is negative, so proceeds are negative
        total_proceeds = total_premium - total_commission - total_slippage

        # Log execution
        execution_record = {
            "type": "exit",
            "structure_id": structure.structure_id,
            "structure_type": structure.structure_type,
            "timestamp": timestamp,
            "num_legs": len(structure.options),
            "num_contracts": total_contracts,
            "total_premium": total_premium,
            "total_proceeds": total_proceeds,
            "commissions": total_commission,
            "slippage": total_slippage,
            "fills": fills,
        }
        self._execution_log.append(execution_record)

        logger.debug(
            f"Executed exit: {structure.structure_type.upper()}, "
            f"premium=${total_premium:,.2f}, proceeds=${total_proceeds:,.2f}"
        )

        return {
            "exit_prices": exit_prices,
            "commissions": total_commission,
            "slippage": total_slippage,
            "total_proceeds": total_proceeds,
            "total_premium": total_premium,
            "fills": fills,
            "timestamp": timestamp,
            "num_contracts": total_contracts,
        }

    # =========================================================================
    # Fill Price Calculation
    # =========================================================================

    def get_fill_price(
        self, option: Option, market_data: Dict[str, Any], side: str
    ) -> float:
        """
        Calculate fill price for an option.

        For 'buy' side: use ask price (or mid + half spread)
        For 'sell' side: use bid price (or mid - half spread)

        Args:
            option: Option to get fill price for
            market_data: Dictionary containing option chain
            side: 'buy' or 'sell'

        Returns:
            Fill price per share

        Raises:
            PriceNotAvailableError: If price cannot be determined
            ValueError: If side is invalid

        Example:
            >>> fill = execution.get_fill_price(call_option, market_data, 'buy')
            >>> print(f"Fill price: ${fill:.2f}")
        """
        if side not in ("buy", "sell"):
            raise ValueError(f"side must be 'buy' or 'sell', got '{side}'")

        # Try to get price from option chain in market data
        option_chain = market_data.get("option_chain")

        if option_chain is not None and not option_chain.empty:
            # Find the matching option in the chain
            price_data = self._find_option_price(option, option_chain)

            if price_data is not None:
                bid = price_data.get("bid", 0.0)
                ask = price_data.get("ask", 0.0)

                if self._use_bid_ask and bid > 0 and ask > 0:
                    # Use actual bid/ask
                    return ask if side == "buy" else bid

                # Calculate mid if bid/ask not available or not using them
                mid = price_data.get("mid", 0.0)
                if mid <= 0 and bid > 0 and ask > 0:
                    mid = (bid + ask) / 2

                if mid > 0:
                    if not self._use_bid_ask:
                        return mid

                    # Apply default spread
                    half_spread = mid * self._default_spread_pct / 2
                    return mid + half_spread if side == "buy" else mid - half_spread

        # Fall back to option's current price
        current_price = option.current_price
        if current_price > 0:
            if not self._use_bid_ask:
                return current_price

            half_spread = current_price * self._default_spread_pct / 2
            return (
                current_price + half_spread
                if side == "buy"
                else max(current_price - half_spread, 0)
            )

        # Last resort: use entry price
        entry_price = option.entry_price
        if entry_price > 0:
            if not self._use_bid_ask:
                return entry_price

            half_spread = entry_price * self._default_spread_pct / 2
            return (
                entry_price + half_spread
                if side == "buy"
                else max(entry_price - half_spread, 0)
            )

        raise PriceNotAvailableError(
            f"No price available for {option.underlying} "
            f"{option.strike} {option.option_type.upper()}"
        )

    def _find_option_price(
        self, option: Option, option_chain: pd.DataFrame
    ) -> Optional[Dict[str, float]]:
        """
        Find option price data in the option chain.

        Args:
            option: Option to look up
            option_chain: DataFrame with option chain data

        Returns:
            Dictionary with bid, ask, mid prices or None if not found
        """
        if option_chain.empty:
            return None

        # Determine column names (handle different schemas)
        strike_col = "strike"
        option_type_col = (
            "call_put" if "call_put" in option_chain.columns else "option_type"
        )
        expiration_col = "expiration"
        bid_col = "bid"
        ask_col = "ask"

        # Normalize option type for matching
        option_type_value = option.option_type.capitalize()

        # Filter for matching strike
        matches = option_chain[option_chain[strike_col] == option.strike]

        if matches.empty:
            # Try approximate strike match (within 0.01)
            matches = option_chain[abs(option_chain[strike_col] - option.strike) < 0.01]

        if matches.empty:
            return None

        # Filter by option type
        if option_type_col in matches.columns:
            type_matches = matches[
                matches[option_type_col].str.lower() == option.option_type.lower()
            ]
            if not type_matches.empty:
                matches = type_matches

        # Filter by expiration if available
        if expiration_col in matches.columns and len(matches) > 1:
            # Try to match expiration
            exp_matches = matches[
                pd.to_datetime(matches[expiration_col]).dt.date
                == option.expiration.date()
            ]
            if not exp_matches.empty:
                matches = exp_matches

        if matches.empty:
            return None

        # Take the first (or best) match
        match = matches.iloc[0]

        # Extract prices
        bid = (
            float(match[bid_col])
            if bid_col in match and pd.notna(match[bid_col])
            else 0.0
        )
        ask = (
            float(match[ask_col])
            if ask_col in match and pd.notna(match[ask_col])
            else 0.0
        )

        # Calculate mid
        if bid > 0 and ask > 0:
            mid = (bid + ask) / 2
        elif bid > 0:
            mid = bid
        elif ask > 0:
            mid = ask
        else:
            mid = 0.0

        return {
            "bid": max(bid, 0.0),
            "ask": max(ask, 0.0),
            "mid": max(mid, 0.0),
        }

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _calculate_commission(self, num_contracts: int) -> float:
        """
        Calculate commission for a given number of contracts.

        Args:
            num_contracts: Total number of contracts

        Returns:
            Total commission in dollars
        """
        return self._commission_per_contract * num_contracts

    def _create_option_id(self, option: Option) -> str:
        """
        Create a unique identifier for an option.

        Args:
            option: Option to create ID for

        Returns:
            String identifier
        """
        exp_str = option.expiration.strftime("%Y%m%d")
        return (
            f"{option.underlying}_{option.strike:.0f}_"
            f"{option.option_type[0].upper()}_{exp_str}"
        )

    # =========================================================================
    # Configuration Methods
    # =========================================================================

    def set_commission(self, commission_per_contract: float) -> None:
        """
        Update commission per contract.

        Args:
            commission_per_contract: New commission rate

        Raises:
            ExecutionConfigError: If commission is negative
        """
        if commission_per_contract < 0:
            raise ExecutionConfigError(
                f"commission_per_contract must be non-negative, "
                f"got {commission_per_contract}"
            )
        self._commission_per_contract = float(commission_per_contract)
        logger.debug(f"Commission updated to ${commission_per_contract:.2f}")

    def set_slippage(self, slippage_pct: float) -> None:
        """
        Update slippage percentage.

        Args:
            slippage_pct: New slippage percentage (0 to 1)

        Raises:
            ExecutionConfigError: If slippage is out of range
        """
        if slippage_pct < 0 or slippage_pct > 1:
            raise ExecutionConfigError(
                f"slippage_pct must be between 0 and 1, got {slippage_pct}"
            )
        self._slippage_pct = float(slippage_pct)
        logger.debug(f"Slippage updated to {slippage_pct:.2%}")

    # =========================================================================
    # Analysis Methods
    # =========================================================================

    def get_execution_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of all executions.

        Returns:
            Dictionary with execution statistics
        """
        if not self._execution_log:
            return {
                "num_executions": 0,
                "num_entries": 0,
                "num_exits": 0,
                "total_commissions": 0.0,
                "total_slippage": 0.0,
            }

        entries = [e for e in self._execution_log if e["type"] == "entry"]
        exits = [e for e in self._execution_log if e["type"] == "exit"]

        return {
            "num_executions": len(self._execution_log),
            "num_entries": len(entries),
            "num_exits": len(exits),
            "total_commissions": sum(e["commissions"] for e in self._execution_log),
            "total_slippage": sum(e["slippage"] for e in self._execution_log),
            "avg_commission_per_execution": (
                sum(e["commissions"] for e in self._execution_log)
                / len(self._execution_log)
            ),
            "total_contracts": sum(e["num_contracts"] for e in self._execution_log),
        }

    def clear_log(self) -> None:
        """Clear the execution log."""
        self._execution_log.clear()
        logger.debug("Execution log cleared")

    # =========================================================================
    # Special Methods
    # =========================================================================

    def __repr__(self) -> str:
        """Return detailed string representation."""
        return (
            f"ExecutionModel("
            f"commission=${self._commission_per_contract:.2f}, "
            f"slippage={self._slippage_pct:.2%}, "
            f"use_bid_ask={self._use_bid_ask})"
        )

    def __str__(self) -> str:
        """Return human-readable string."""
        return (
            f"ExecutionModel: "
            f"${self._commission_per_contract:.2f}/contract, "
            f"{self._slippage_pct:.2%} slippage, "
            f"{'bid/ask' if self._use_bid_ask else 'mid'} fills"
        )


# =============================================================================
# Advanced Execution Model with Volume Impact
# =============================================================================

# Kyle's lambda calibration constants
KYLE_LAMBDA_COEFFICIENT = 0.5
MAX_VOLUME_IMPACT = 0.10
DEFAULT_DAILY_VOLUME = 1000
MAX_FILL_RATIO = 0.10


class VolumeImpactModel:
    """
    Kyle's Lambda model for market impact estimation.

    Based on Kyle (1985): Price impact is proportional to the square root
    of order size relative to market volume.

    Impact = lambda * sqrt(order_size / daily_volume)

    where lambda is calibrated to market conditions.
    """

    def __init__(
        self,
        lambda_coefficient: float = KYLE_LAMBDA_COEFFICIENT,
        max_impact: float = MAX_VOLUME_IMPACT,
    ):
        self.lambda_coefficient = lambda_coefficient
        self.max_impact = max_impact

    def calculate_impact(
        self, order_size: int, daily_volume: int, is_buy: bool = True
    ) -> float:
        """
        Calculate market impact for an order.

        Args:
            order_size: Number of contracts in the order
            daily_volume: Average daily volume for this option
            is_buy: True for buy orders, False for sell orders

        Returns:
            Market impact as a decimal (e.g., 0.02 for 2% impact)
        """
        if daily_volume <= 0:
            return self.max_impact

        volume_ratio = order_size / daily_volume

        impact = self.lambda_coefficient * np.sqrt(volume_ratio)

        return min(impact, self.max_impact)

    def calculate_impact_cost(
        self, order_size: int, daily_volume: int, mid_price: float, is_buy: bool = True
    ) -> float:
        """
        Calculate dollar cost of market impact.

        Args:
            order_size: Number of contracts
            daily_volume: Daily volume
            mid_price: Mid price of the option
            is_buy: True for buy, False for sell

        Returns:
            Dollar cost of impact per contract
        """
        impact_pct = self.calculate_impact(order_size, daily_volume, is_buy)
        return mid_price * impact_pct


class AdvancedExecutionResult:
    """
    Detailed execution result with volume impact breakdown.
    """

    __slots__ = (
        "_fill_price",
        "_total_cost",
        "_total_proceeds",
        "_commission",
        "_slippage",
        "_volume_impact",
        "_volume_impact_cost",
        "_timestamp",
        "_filled_quantity",
        "_unfilled_quantity",
        "_order_type",
        "_order_style",
        "_is_partial_fill",
        "_execution_quality",
    )

    def __init__(
        self,
        fill_price: float,
        total_cost: float,
        total_proceeds: float,
        commission: float,
        slippage: float,
        volume_impact: float,
        volume_impact_cost: float,
        timestamp: datetime,
        filled_quantity: int,
        unfilled_quantity: int,
        order_type: str,
        order_style: str,
        is_partial_fill: bool,
    ):
        self._fill_price = fill_price
        self._total_cost = total_cost
        self._total_proceeds = total_proceeds
        self._commission = commission
        self._slippage = slippage
        self._volume_impact = volume_impact
        self._volume_impact_cost = volume_impact_cost
        self._timestamp = timestamp
        self._filled_quantity = filled_quantity
        self._unfilled_quantity = unfilled_quantity
        self._order_type = order_type
        self._order_style = order_style
        self._is_partial_fill = is_partial_fill

        effective_spread = (
            (slippage + volume_impact_cost)
            / (fill_price * filled_quantity * CONTRACT_MULTIPLIER)
            if filled_quantity > 0 and fill_price > 0
            else 0
        )
        self._execution_quality = 1.0 - effective_spread

    @property
    def fill_price(self) -> float:
        return self._fill_price

    @property
    def total_cost(self) -> float:
        return self._total_cost

    @property
    def total_proceeds(self) -> float:
        return self._total_proceeds

    @property
    def commission(self) -> float:
        return self._commission

    @property
    def slippage(self) -> float:
        return self._slippage

    @property
    def volume_impact(self) -> float:
        return self._volume_impact

    @property
    def volume_impact_cost(self) -> float:
        return self._volume_impact_cost

    @property
    def timestamp(self) -> datetime:
        return self._timestamp

    @property
    def filled_quantity(self) -> int:
        return self._filled_quantity

    @property
    def unfilled_quantity(self) -> int:
        return self._unfilled_quantity

    @property
    def order_type(self) -> str:
        return self._order_type

    @property
    def order_style(self) -> str:
        return self._order_style

    @property
    def is_partial_fill(self) -> bool:
        return self._is_partial_fill

    @property
    def fill_rate(self) -> float:
        total = self._filled_quantity + self._unfilled_quantity
        return self._filled_quantity / total if total > 0 else 0.0

    @property
    def execution_quality(self) -> float:
        return self._execution_quality

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fill_price": self._fill_price,
            "total_cost": self._total_cost,
            "total_proceeds": self._total_proceeds,
            "commission": self._commission,
            "slippage": self._slippage,
            "volume_impact": self._volume_impact,
            "volume_impact_cost": self._volume_impact_cost,
            "timestamp": self._timestamp,
            "filled_quantity": self._filled_quantity,
            "unfilled_quantity": self._unfilled_quantity,
            "order_type": self._order_type,
            "order_style": self._order_style,
            "is_partial_fill": self._is_partial_fill,
            "fill_rate": self.fill_rate,
            "execution_quality": self._execution_quality,
        }

    def __repr__(self) -> str:
        fill_status = (
            f"PARTIAL {self._filled_quantity}/{self._filled_quantity + self._unfilled_quantity}"
            if self._is_partial_fill
            else "FILLED"
        )
        return (
            f"AdvancedExecutionResult({fill_status}, "
            f"price=${self._fill_price:.2f}, "
            f"impact={self._volume_impact:.2%}, "
            f"quality={self._execution_quality:.2%})"
        )


class AdvancedExecutionModel:
    """
    Realistic execution simulation with volume impact modeling.

    Features:
        - Kyle's lambda model for market impact
        - Partial fills based on available volume
        - Limit and stop order support
        - Order queue management
        - Execution quality metrics

    The model accounts for the fact that large orders move the market,
    and that options with low volume may not be fully fillable.

    Example:
        >>> model = AdvancedExecutionModel(
        ...     commission_per_contract=0.65,
        ...     base_slippage_pct=0.01,
        ...     use_volume_impact=True,
        ...     allow_partial_fills=True
        ... )
        >>> result = model.execute_order('buy', 100, market_data)
        >>> print(f"Filled {result.filled_quantity} @ ${result.fill_price:.2f}")
    """

    def __init__(
        self,
        commission_per_contract: float = DEFAULT_COMMISSION_PER_CONTRACT,
        base_slippage_pct: float = 0.01,
        use_volume_impact: bool = True,
        allow_partial_fills: bool = False,
        max_fill_ratio: float = MAX_FILL_RATIO,
        kyle_lambda: float = KYLE_LAMBDA_COEFFICIENT,
    ):
        """
        Initialize advanced execution model.

        Args:
            commission_per_contract: Commission per contract ($)
            base_slippage_pct: Base slippage as decimal (0.01 = 1%)
            use_volume_impact: Enable Kyle's lambda volume impact
            allow_partial_fills: Allow partial fills for large orders
            max_fill_ratio: Maximum order size as ratio of daily volume
            kyle_lambda: Kyle's lambda coefficient for impact calculation
        """
        if commission_per_contract < 0:
            raise ExecutionConfigError(
                f"commission must be non-negative, got {commission_per_contract}"
            )
        if base_slippage_pct < 0 or base_slippage_pct > 1:
            raise ExecutionConfigError(
                f"base_slippage_pct must be 0-1, got {base_slippage_pct}"
            )
        if max_fill_ratio <= 0 or max_fill_ratio > 1:
            raise ExecutionConfigError(
                f"max_fill_ratio must be 0-1, got {max_fill_ratio}"
            )

        self._commission_per_contract = commission_per_contract
        self._base_slippage_pct = base_slippage_pct
        self._use_volume_impact = use_volume_impact
        self._allow_partial_fills = allow_partial_fills
        self._max_fill_ratio = max_fill_ratio

        self._volume_impact_model = VolumeImpactModel(
            lambda_coefficient=kyle_lambda, max_impact=MAX_VOLUME_IMPACT
        )

        self._pending_orders: List[Dict[str, Any]] = []
        self._execution_history: List[AdvancedExecutionResult] = []

    @property
    def commission_per_contract(self) -> float:
        return self._commission_per_contract

    @property
    def base_slippage_pct(self) -> float:
        return self._base_slippage_pct

    @property
    def use_volume_impact(self) -> bool:
        return self._use_volume_impact

    @property
    def allow_partial_fills(self) -> bool:
        return self._allow_partial_fills

    @property
    def execution_history(self) -> List[AdvancedExecutionResult]:
        return self._execution_history.copy()

    def execute_order(
        self,
        order_type: str,
        quantity: int,
        market_data: Dict[str, Any],
        order_style: str = "market",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        timestamp: Optional[datetime] = None,
    ) -> AdvancedExecutionResult:
        """
        Execute an order with realistic fill simulation.

        Args:
            order_type: 'buy' or 'sell'
            quantity: Number of contracts
            market_data: Dict with 'bid', 'ask', 'mid', 'volume', 'timestamp'
            order_style: 'market', 'limit', or 'stop'
            limit_price: Limit price (required for limit orders)
            stop_price: Stop trigger price (required for stop orders)
            timestamp: Execution timestamp (defaults to market_data timestamp or now)

        Returns:
            AdvancedExecutionResult with fill details

        Raises:
            ExecutionError: If order cannot be executed
            ValueError: If invalid order parameters
        """
        if order_type not in ("buy", "sell"):
            raise ValueError(f"order_type must be 'buy' or 'sell', got '{order_type}'")
        if order_style not in ("market", "limit", "stop"):
            raise ValueError(
                f"order_style must be 'market', 'limit', or 'stop', got '{order_style}'"
            )
        if quantity <= 0:
            raise ValueError(f"quantity must be positive, got {quantity}")
        if order_style == "limit" and limit_price is None:
            raise ValueError("limit_price required for limit orders")
        if order_style == "stop" and stop_price is None:
            raise ValueError("stop_price required for stop orders")

        bid = market_data.get("bid", 0.0)
        ask = market_data.get("ask", 0.0)
        mid = market_data.get("mid", (bid + ask) / 2 if bid > 0 and ask > 0 else 0.0)
        volume = market_data.get("volume", DEFAULT_DAILY_VOLUME)

        if timestamp is None:
            timestamp = market_data.get("timestamp", datetime.now())

        if bid <= 0 or ask <= 0:
            raise PriceNotAvailableError("Invalid bid/ask prices in market data")

        is_buy = order_type == "buy"

        if not self._can_execute_order(
            order_style, order_type, bid, ask, limit_price, stop_price
        ):
            return self._create_unfilled_result(
                quantity, order_type, order_style, timestamp
            )

        filled_qty, unfilled_qty = self._determine_fill_quantity(quantity, volume)

        is_partial = unfilled_qty > 0

        if filled_qty == 0:
            return self._create_unfilled_result(
                quantity, order_type, order_style, timestamp
            )

        volume_impact = 0.0
        volume_impact_cost = 0.0

        if self._use_volume_impact:
            volume_impact = self._volume_impact_model.calculate_impact(
                filled_qty, volume, is_buy
            )
            volume_impact_cost = (
                self._volume_impact_model.calculate_impact_cost(
                    filled_qty, volume, mid, is_buy
                )
                * filled_qty
                * CONTRACT_MULTIPLIER
            )

        base_price = ask if is_buy else bid

        if order_style == "limit" and limit_price is not None:
            if is_buy:
                base_price = min(ask, limit_price)
            else:
                base_price = max(bid, limit_price)

        slippage_amount = base_price * self._base_slippage_pct

        if is_buy:
            fill_price = base_price + slippage_amount + (mid * volume_impact)
        else:
            fill_price = base_price - slippage_amount - (mid * volume_impact)

        fill_price = max(fill_price, 0.01)

        commission = self._commission_per_contract * filled_qty

        premium = fill_price * filled_qty * CONTRACT_MULTIPLIER
        slippage_cost = slippage_amount * filled_qty * CONTRACT_MULTIPLIER

        if is_buy:
            total_cost = premium + commission + slippage_cost + volume_impact_cost
            total_proceeds = 0.0
        else:
            total_cost = 0.0
            total_proceeds = premium - commission - slippage_cost - volume_impact_cost

        result = AdvancedExecutionResult(
            fill_price=fill_price,
            total_cost=total_cost,
            total_proceeds=total_proceeds,
            commission=commission,
            slippage=slippage_cost,
            volume_impact=volume_impact,
            volume_impact_cost=volume_impact_cost,
            timestamp=timestamp,
            filled_quantity=filled_qty,
            unfilled_quantity=unfilled_qty,
            order_type=order_type,
            order_style=order_style,
            is_partial_fill=is_partial,
        )

        self._execution_history.append(result)

        if is_partial:
            logger.warning(
                f"Partial fill: {filled_qty}/{quantity} contracts at ${fill_price:.2f} "
                f"(volume impact: {volume_impact:.2%})"
            )

        return result

    def _can_execute_order(
        self,
        order_style: str,
        order_type: str,
        bid: float,
        ask: float,
        limit_price: Optional[float],
        stop_price: Optional[float],
    ) -> bool:
        """Check if order can be executed given current market conditions."""
        if order_style == "market":
            return True

        is_buy = order_type == "buy"

        if order_style == "limit":
            if is_buy:
                return ask <= limit_price
            else:
                return bid >= limit_price

        if order_style == "stop":
            mid = (bid + ask) / 2
            if is_buy:
                return mid >= stop_price
            else:
                return mid <= stop_price

        return False

    def _determine_fill_quantity(
        self, requested_qty: int, daily_volume: int
    ) -> Tuple[int, int]:
        """Determine how much of the order can be filled."""
        if not self._allow_partial_fills:
            return requested_qty, 0

        max_fillable = int(daily_volume * self._max_fill_ratio)
        max_fillable = max(max_fillable, 1)

        if requested_qty <= max_fillable:
            return requested_qty, 0

        return max_fillable, requested_qty - max_fillable

    def _create_unfilled_result(
        self, quantity: int, order_type: str, order_style: str, timestamp: datetime
    ) -> AdvancedExecutionResult:
        """Create result for unfilled order."""
        return AdvancedExecutionResult(
            fill_price=0.0,
            total_cost=0.0,
            total_proceeds=0.0,
            commission=0.0,
            slippage=0.0,
            volume_impact=0.0,
            volume_impact_cost=0.0,
            timestamp=timestamp,
            filled_quantity=0,
            unfilled_quantity=quantity,
            order_type=order_type,
            order_style=order_style,
            is_partial_fill=True,
        )

    def execute_structure_entry(
        self,
        structure: OptionStructure,
        market_data: Dict[str, Any],
        timestamp: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Execute entry for an option structure with volume impact.

        Args:
            structure: OptionStructure to enter
            market_data: Market data with option chain
            timestamp: Execution timestamp

        Returns:
            Dict with aggregated execution results
        """
        if structure.is_empty:
            raise EmptyStructureError("Cannot execute entry for empty structure")

        if timestamp is None:
            timestamp = datetime.now()

        results = []
        total_cost = 0.0
        total_premium = 0.0
        total_commission = 0.0
        total_slippage = 0.0
        total_volume_impact = 0.0
        filled_contracts = 0
        unfilled_contracts = 0

        option_chain = market_data.get("option_chain")

        for option in structure.options:
            order_type = "buy" if option.is_long else "sell"

            option_market_data = self._get_option_market_data(
                option, option_chain, market_data
            )

            result = self.execute_order(
                order_type=order_type,
                quantity=option.quantity,
                market_data=option_market_data,
                order_style="market",
                timestamp=timestamp,
            )

            results.append(
                {
                    "option": self._create_option_id_simple(option),
                    "result": result.to_dict(),
                }
            )

            total_cost += result.total_cost
            if order_type == "sell":
                total_premium += result.total_proceeds
            else:
                total_premium -= (
                    result.total_cost
                    - result.commission
                    - result.slippage
                    - result.volume_impact_cost
                )

            total_commission += result.commission
            total_slippage += result.slippage
            total_volume_impact += result.volume_impact_cost
            filled_contracts += result.filled_quantity
            unfilled_contracts += result.unfilled_quantity

        return {
            "entry_results": results,
            "total_cost": total_cost,
            "total_premium": total_premium,
            "total_commission": total_commission,
            "total_slippage": total_slippage,
            "total_volume_impact": total_volume_impact,
            "filled_contracts": filled_contracts,
            "unfilled_contracts": unfilled_contracts,
            "timestamp": timestamp,
            "execution_quality": 1.0
            - (total_slippage + total_volume_impact) / max(abs(total_premium), 1.0),
        }

    def _get_option_market_data(
        self,
        option: Option,
        option_chain: Optional[pd.DataFrame],
        market_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Extract market data for a specific option."""
        default_data = {
            "bid": option.current_price * 0.98
            if option.current_price > 0
            else option.entry_price * 0.98,
            "ask": option.current_price * 1.02
            if option.current_price > 0
            else option.entry_price * 1.02,
            "mid": option.current_price
            if option.current_price > 0
            else option.entry_price,
            "volume": DEFAULT_DAILY_VOLUME,
            "timestamp": market_data.get("timestamp", datetime.now()),
        }

        if option_chain is None or (
            hasattr(option_chain, "empty") and option_chain.empty
        ):
            return default_data

        try:
            matches = option_chain[option_chain["strike"] == option.strike]

            if "option_type" in option_chain.columns:
                matches = matches[
                    matches["option_type"].str.lower() == option.option_type.lower()
                ]
            elif "call_put" in option_chain.columns:
                matches = matches[
                    matches["call_put"].str.lower() == option.option_type.lower()
                ]

            if len(matches) == 0:
                return default_data

            row = matches.iloc[0]

            bid = float(row.get("bid", default_data["bid"]))
            ask = float(row.get("ask", default_data["ask"]))
            volume = int(row.get("volume", default_data["volume"]))

            return {
                "bid": bid if bid > 0 else default_data["bid"],
                "ask": ask if ask > 0 else default_data["ask"],
                "mid": (bid + ask) / 2 if bid > 0 and ask > 0 else default_data["mid"],
                "volume": volume if volume > 0 else default_data["volume"],
                "timestamp": market_data.get("timestamp", datetime.now()),
            }
        except Exception:
            return default_data

    def _create_option_id_simple(self, option: Option) -> str:
        """Create simple option identifier."""
        exp_str = option.expiration.strftime("%Y%m%d")
        return f"{option.underlying}_{option.strike:.0f}_{option.option_type[0].upper()}_{exp_str}"

    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get statistics about all executions."""
        if not self._execution_history:
            return {
                "num_executions": 0,
                "avg_fill_rate": 0.0,
                "avg_volume_impact": 0.0,
                "avg_execution_quality": 0.0,
                "total_volume_impact_cost": 0.0,
            }

        fill_rates = [r.fill_rate for r in self._execution_history]
        impacts = [
            r.volume_impact for r in self._execution_history if r.filled_quantity > 0
        ]
        qualities = [
            r.execution_quality
            for r in self._execution_history
            if r.filled_quantity > 0
        ]
        impact_costs = [r.volume_impact_cost for r in self._execution_history]

        return {
            "num_executions": len(self._execution_history),
            "num_partial_fills": sum(
                1 for r in self._execution_history if r.is_partial_fill
            ),
            "avg_fill_rate": np.mean(fill_rates) if fill_rates else 0.0,
            "avg_volume_impact": np.mean(impacts) if impacts else 0.0,
            "max_volume_impact": max(impacts) if impacts else 0.0,
            "avg_execution_quality": np.mean(qualities) if qualities else 0.0,
            "total_volume_impact_cost": sum(impact_costs),
            "total_commission": sum(r.commission for r in self._execution_history),
        }

    def clear_history(self) -> None:
        """Clear execution history."""
        self._execution_history.clear()

    def __repr__(self) -> str:
        return (
            f"AdvancedExecutionModel("
            f"commission=${self._commission_per_contract:.2f}, "
            f"slippage={self._base_slippage_pct:.2%}, "
            f"volume_impact={self._use_volume_impact}, "
            f"partial_fills={self._allow_partial_fills})"
        )


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Main classes
    "ExecutionModel",
    "ExecutionResult",
    "AdvancedExecutionModel",
    "AdvancedExecutionResult",
    "VolumeImpactModel",
    # Exceptions
    "ExecutionError",
    "ExecutionConfigError",
    "PriceNotAvailableError",
    "FillError",
    # Constants
    "DEFAULT_COMMISSION_PER_CONTRACT",
    "DEFAULT_SLIPPAGE_PCT",
    "DEFAULT_SPREAD_PCT",
    "KYLE_LAMBDA_COEFFICIENT",
    "MAX_VOLUME_IMPACT",
]
