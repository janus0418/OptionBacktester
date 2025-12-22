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
        '_entry_prices',
        '_commissions',
        '_slippage',
        '_total_cost',
        '_total_proceeds',
        '_timestamp',
        '_fills',
        '_is_entry',
        '_num_contracts',
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
        num_contracts: int
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
            'entry_prices': self._entry_prices,
            'commissions': self._commissions,
            'slippage': self._slippage,
            'total_cost': self._total_cost,
            'total_proceeds': self._total_proceeds,
            'timestamp': self._timestamp,
            'fills': self._fills,
            'is_entry': self._is_entry,
            'num_contracts': self._num_contracts,
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
        '_commission_per_contract',
        '_slippage_pct',
        '_use_bid_ask',
        '_default_spread_pct',
        '_execution_log',
    )

    def __init__(
        self,
        commission_per_contract: float = DEFAULT_COMMISSION_PER_CONTRACT,
        slippage_pct: float = DEFAULT_SLIPPAGE_PCT,
        use_bid_ask: bool = True,
        default_spread_pct: float = DEFAULT_SPREAD_PCT
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
        timestamp: Optional[datetime] = None
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
        option_chain = market_data.get('option_chain')
        spot = market_data.get('spot', 0.0)

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
                side='buy' if is_buying else 'sell'
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
            total_slippage += abs(slippage_amount) * option.quantity * CONTRACT_MULTIPLIER
            total_contracts += option.quantity

            # Create option identifier
            option_id = self._create_option_id(option)
            entry_prices[option_id] = fill_price

            # Record fill
            fills.append({
                'option_id': option_id,
                'option_type': option.option_type,
                'position_type': option.position_type,
                'strike': option.strike,
                'expiration': option.expiration,
                'quantity': option.quantity,
                'side': 'buy' if is_buying else 'sell',
                'fill_price': fill_price,
                'slippage': slippage_amount,
                'premium': leg_premium,
            })

        # Calculate commission
        total_commission = self._calculate_commission(
            num_contracts=total_contracts
        )

        # Total cost = premium paid + commission + slippage
        # For credit trades, premium is positive (receive), so total_cost is negative
        total_cost = -total_premium + total_commission + total_slippage

        # Log execution
        execution_record = {
            'type': 'entry',
            'structure_id': structure.structure_id,
            'structure_type': structure.structure_type,
            'timestamp': timestamp,
            'num_legs': len(structure.options),
            'num_contracts': total_contracts,
            'total_premium': total_premium,
            'total_cost': total_cost,
            'commissions': total_commission,
            'slippage': total_slippage,
            'fills': fills,
        }
        self._execution_log.append(execution_record)

        logger.debug(
            f"Executed entry: {structure.structure_type.upper()}, "
            f"premium=${total_premium:,.2f}, cost=${total_cost:,.2f}"
        )

        return {
            'entry_prices': entry_prices,
            'commissions': total_commission,
            'slippage': total_slippage,
            'total_cost': total_cost,
            'total_premium': total_premium,
            'fills': fills,
            'timestamp': timestamp,
            'num_contracts': total_contracts,
        }

    def execute_exit(
        self,
        structure: OptionStructure,
        market_data: Dict[str, Any],
        timestamp: Optional[datetime] = None
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
                side='buy' if is_buying else 'sell'
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
            total_slippage += abs(slippage_amount) * option.quantity * CONTRACT_MULTIPLIER
            total_contracts += option.quantity

            # Create option identifier
            option_id = self._create_option_id(option)
            exit_prices[option_id] = fill_price

            # Record fill
            fills.append({
                'option_id': option_id,
                'option_type': option.option_type,
                'position_type': option.position_type,
                'strike': option.strike,
                'expiration': option.expiration,
                'quantity': option.quantity,
                'side': 'buy' if is_buying else 'sell',
                'fill_price': fill_price,
                'slippage': slippage_amount,
                'premium': leg_premium,
            })

        # Calculate commission
        total_commission = self._calculate_commission(
            num_contracts=total_contracts
        )

        # Total proceeds = premium received - commission - slippage
        # For debit to close, premium is negative, so proceeds are negative
        total_proceeds = total_premium - total_commission - total_slippage

        # Log execution
        execution_record = {
            'type': 'exit',
            'structure_id': structure.structure_id,
            'structure_type': structure.structure_type,
            'timestamp': timestamp,
            'num_legs': len(structure.options),
            'num_contracts': total_contracts,
            'total_premium': total_premium,
            'total_proceeds': total_proceeds,
            'commissions': total_commission,
            'slippage': total_slippage,
            'fills': fills,
        }
        self._execution_log.append(execution_record)

        logger.debug(
            f"Executed exit: {structure.structure_type.upper()}, "
            f"premium=${total_premium:,.2f}, proceeds=${total_proceeds:,.2f}"
        )

        return {
            'exit_prices': exit_prices,
            'commissions': total_commission,
            'slippage': total_slippage,
            'total_proceeds': total_proceeds,
            'total_premium': total_premium,
            'fills': fills,
            'timestamp': timestamp,
            'num_contracts': total_contracts,
        }

    # =========================================================================
    # Fill Price Calculation
    # =========================================================================

    def get_fill_price(
        self,
        option: Option,
        market_data: Dict[str, Any],
        side: str
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
        if side not in ('buy', 'sell'):
            raise ValueError(f"side must be 'buy' or 'sell', got '{side}'")

        # Try to get price from option chain in market data
        option_chain = market_data.get('option_chain')

        if option_chain is not None and not option_chain.empty:
            # Find the matching option in the chain
            price_data = self._find_option_price(option, option_chain)

            if price_data is not None:
                bid = price_data.get('bid', 0.0)
                ask = price_data.get('ask', 0.0)

                if self._use_bid_ask and bid > 0 and ask > 0:
                    # Use actual bid/ask
                    return ask if side == 'buy' else bid

                # Calculate mid if bid/ask not available or not using them
                mid = price_data.get('mid', 0.0)
                if mid <= 0 and bid > 0 and ask > 0:
                    mid = (bid + ask) / 2

                if mid > 0:
                    if not self._use_bid_ask:
                        return mid

                    # Apply default spread
                    half_spread = mid * self._default_spread_pct / 2
                    return mid + half_spread if side == 'buy' else mid - half_spread

        # Fall back to option's current price
        current_price = option.current_price
        if current_price > 0:
            if not self._use_bid_ask:
                return current_price

            half_spread = current_price * self._default_spread_pct / 2
            return current_price + half_spread if side == 'buy' else max(current_price - half_spread, 0)

        # Last resort: use entry price
        entry_price = option.entry_price
        if entry_price > 0:
            if not self._use_bid_ask:
                return entry_price

            half_spread = entry_price * self._default_spread_pct / 2
            return entry_price + half_spread if side == 'buy' else max(entry_price - half_spread, 0)

        raise PriceNotAvailableError(
            f"No price available for {option.underlying} "
            f"{option.strike} {option.option_type.upper()}"
        )

    def _find_option_price(
        self,
        option: Option,
        option_chain: pd.DataFrame
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
        strike_col = 'strike'
        option_type_col = 'call_put' if 'call_put' in option_chain.columns else 'option_type'
        expiration_col = 'expiration'
        bid_col = 'bid'
        ask_col = 'ask'

        # Normalize option type for matching
        option_type_value = option.option_type.capitalize()

        # Filter for matching strike
        matches = option_chain[option_chain[strike_col] == option.strike]

        if matches.empty:
            # Try approximate strike match (within 0.01)
            matches = option_chain[
                abs(option_chain[strike_col] - option.strike) < 0.01
            ]

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
                pd.to_datetime(matches[expiration_col]).dt.date == option.expiration.date()
            ]
            if not exp_matches.empty:
                matches = exp_matches

        if matches.empty:
            return None

        # Take the first (or best) match
        match = matches.iloc[0]

        # Extract prices
        bid = float(match[bid_col]) if bid_col in match and pd.notna(match[bid_col]) else 0.0
        ask = float(match[ask_col]) if ask_col in match and pd.notna(match[ask_col]) else 0.0

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
            'bid': max(bid, 0.0),
            'ask': max(ask, 0.0),
            'mid': max(mid, 0.0),
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
        exp_str = option.expiration.strftime('%Y%m%d')
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
                'num_executions': 0,
                'num_entries': 0,
                'num_exits': 0,
                'total_commissions': 0.0,
                'total_slippage': 0.0,
            }

        entries = [e for e in self._execution_log if e['type'] == 'entry']
        exits = [e for e in self._execution_log if e['type'] == 'exit']

        return {
            'num_executions': len(self._execution_log),
            'num_entries': len(entries),
            'num_exits': len(exits),
            'total_commissions': sum(e['commissions'] for e in self._execution_log),
            'total_slippage': sum(e['slippage'] for e in self._execution_log),
            'avg_commission_per_execution': (
                sum(e['commissions'] for e in self._execution_log) / len(self._execution_log)
            ),
            'total_contracts': sum(e['num_contracts'] for e in self._execution_log),
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
# Module Exports
# =============================================================================

__all__ = [
    # Main classes
    'ExecutionModel',
    'ExecutionResult',

    # Exceptions
    'ExecutionError',
    'ExecutionConfigError',
    'PriceNotAvailableError',
    'FillError',

    # Constants
    'DEFAULT_COMMISSION_PER_CONTRACT',
    'DEFAULT_SLIPPAGE_PCT',
    'DEFAULT_SPREAD_PCT',
]
