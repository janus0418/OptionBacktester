"""
OptionStructure Base Class for Options Backtesting

This module provides the OptionStructure base class for managing multi-leg
option positions. It serves as the foundation for all concrete option
structures (straddles, strangles, spreads, condors, etc.).

Key Features:
    - Container for multiple Option objects (legs)
    - Net Greeks aggregation across all legs
    - Total P&L calculation with proper position handling
    - Max profit/loss calculation
    - Breakeven point determination
    - Payoff diagram generation
    - Serialization support

Design Philosophy:
    The OptionStructure is a generic container that can hold any combination
    of option legs. Concrete implementations (straddle, iron condor, etc.)
    are built in Run 8 and will inherit from this base class.

Financial Correctness:
    - Net Greeks = sum of position-adjusted Greeks across all legs
    - P&L = sum of individual option P&Ls
    - Max profit/loss calculated from payoff at extreme spot prices
    - Breakevens solved numerically for arbitrary structures

Usage:
    from backtester.core.option_structure import OptionStructure
    from backtester.core.option import Option
    from datetime import datetime

    # Create a structure
    structure = OptionStructure(structure_type='custom', underlying='SPY')

    # Add legs manually
    call = Option(option_type='call', position_type='short', ...)
    put = Option(option_type='put', position_type='short', ...)
    structure.add_option(call)
    structure.add_option(put)

    # Calculate aggregate metrics
    net_greeks = structure.calculate_net_greeks()
    total_pnl = structure.calculate_pnl()
    max_profit, max_loss = structure.calculate_max_profit(), structure.calculate_max_loss()

References:
    - Hull, J. C. (2018). Options, Futures, and Other Derivatives.
    - Options Strategy Payoffs: https://www.cboe.com/education/
"""

import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from scipy.optimize import brentq

from backtester.core.option import (
    Option,
    OptionError,
    OptionExpiredError,
    OptionValidationError,
    CONTRACT_MULTIPLIER,
)

# Configure module logger
logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Numerical tolerance for breakeven calculations
BREAKEVEN_TOLERANCE = 1e-6

# Default spot price range for max profit/loss (as fraction of current spot)
DEFAULT_SPOT_RANGE_FACTOR = 2.0

# Minimum number of points for payoff diagram
MIN_PAYOFF_POINTS = 100

# Greeks names for aggregation
GREEK_NAMES = ['delta', 'gamma', 'theta', 'vega', 'rho']

# Breakeven calculation search margin factor (20% of minimum strike)
DEFAULT_BREAKEVEN_SEARCH_MARGIN_FACTOR = 0.20

# Spot range factors for max profit/loss calculations
DEFAULT_MIN_SPOT_FACTOR = 0.01  # Near zero for lower bound
DEFAULT_MAX_SPOT_FACTOR = 3.0   # 3x highest strike for upper bound


# =============================================================================
# Exceptions
# =============================================================================

class OptionStructureError(Exception):
    """Base exception for OptionStructure errors."""
    pass


class OptionStructureValidationError(OptionStructureError):
    """Exception raised when structure validation fails."""
    pass


class EmptyStructureError(OptionStructureError):
    """Exception raised when attempting operations on empty structure."""
    pass


# =============================================================================
# OptionStructure Base Class
# =============================================================================

class OptionStructure:
    """
    Base class for multi-leg option structures.

    This class provides a container for multiple Option objects and methods
    to calculate aggregate metrics (net Greeks, total P&L, max profit/loss,
    breakeven points, and payoff diagrams).

    The class is designed to be:
    1. Flexible: Can hold any combination of options
    2. Validated: Ensures all options share the same underlying
    3. Financial: Correctly aggregates Greeks and P&L

    Attributes:
        structure_id (str): Unique identifier for this structure
        structure_type (str): Type of structure ('straddle', 'custom', etc.)
        options (List[Option]): List of option legs
        underlying (str): Underlying asset ticker
        entry_date (Optional[datetime]): When structure was opened
        net_premium (float): Net credit/debit at entry (positive = credit)

    Properties:
        is_empty (bool): True if no options in structure
        num_legs (int): Number of option legs
        net_delta (float): Sum of position-adjusted deltas
        net_gamma (float): Sum of position-adjusted gammas
        net_theta (float): Sum of position-adjusted thetas
        net_vega (float): Sum of position-adjusted vegas
        net_rho (float): Sum of position-adjusted rhos

    Example:
        >>> structure = OptionStructure(structure_type='straddle', underlying='SPY')
        >>> structure.add_option(call_option)
        >>> structure.add_option(put_option)
        >>> net_greeks = structure.calculate_net_greeks()
        >>> print(f"Net Delta: {net_greeks['delta']:.4f}")
    """

    __slots__ = (
        '_structure_id',
        '_structure_type',
        '_options',
        '_underlying',
        '_entry_date',
        '_net_premium',
        '_cached_greeks',
        '_cached_greeks_params',
    )

    def __init__(
        self,
        structure_type: str = 'custom',
        underlying: Optional[str] = None,
        structure_id: Optional[str] = None,
        entry_date: Optional[datetime] = None
    ) -> None:
        """
        Initialize an OptionStructure.

        Args:
            structure_type: Type of structure (e.g., 'straddle', 'strangle',
                          'iron_condor', 'custom'). Default 'custom'.
            underlying: Underlying asset ticker. If None, will be set from
                       first added option.
            structure_id: Unique identifier. If None, auto-generated UUID.
            entry_date: When structure was opened. If None, set from first
                       added option.

        Example:
            >>> structure = OptionStructure(
            ...     structure_type='straddle',
            ...     underlying='SPY',
            ...     entry_date=datetime(2024, 3, 1)
            ... )
        """
        # Generate unique ID if not provided
        self._structure_id = structure_id or str(uuid.uuid4())[:8]

        # Validate and set structure type
        if not structure_type or not isinstance(structure_type, str):
            raise OptionStructureValidationError(
                "structure_type must be a non-empty string"
            )
        self._structure_type = structure_type.lower().strip()

        # Initialize underlying (may be set from first option)
        if underlying is not None:
            if not isinstance(underlying, str) or not underlying.strip():
                raise OptionStructureValidationError(
                    "underlying must be a non-empty string if provided"
                )
            self._underlying = underlying.upper().strip()
        else:
            self._underlying = None

        # Initialize options list
        self._options: List[Option] = []

        # Entry date (set from first option if not provided)
        self._entry_date = entry_date

        # Net premium calculated when options are added
        self._net_premium = 0.0

        # Greeks cache
        self._cached_greeks: Dict[str, float] = {}
        self._cached_greeks_params: Optional[Tuple] = None

        logger.debug(
            f"Created OptionStructure: id={self._structure_id}, "
            f"type={self._structure_type}, underlying={self._underlying}"
        )

    # =========================================================================
    # Properties - Basic Attributes
    # =========================================================================

    @property
    def structure_id(self) -> str:
        """Get structure unique identifier."""
        return self._structure_id

    @property
    def structure_type(self) -> str:
        """Get structure type."""
        return self._structure_type

    @property
    def underlying(self) -> Optional[str]:
        """Get underlying asset ticker."""
        return self._underlying

    @property
    def options(self) -> List[Option]:
        """Get list of option legs (copy for safety)."""
        return self._options.copy()

    @property
    def entry_date(self) -> Optional[datetime]:
        """Get structure entry date."""
        return self._entry_date

    @property
    def net_premium(self) -> float:
        """
        Get net premium at entry.

        For credit structures (e.g., short straddle): positive value
        For debit structures (e.g., long straddle): negative value

        Calculated as: sum of (entry_price * position_sign * quantity * 100)
        """
        return self._net_premium

    @property
    def is_empty(self) -> bool:
        """Check if structure has no options."""
        return len(self._options) == 0

    @property
    def num_legs(self) -> int:
        """Get number of option legs."""
        return len(self._options)

    # =========================================================================
    # Option Management Methods
    # =========================================================================

    def add_option(self, option: Option) -> None:
        """
        Add an option leg to the structure.

        Validates that the option:
        1. Is a valid Option instance
        2. Has the same underlying as existing options (if any)

        Args:
            option: Option instance to add

        Raises:
            OptionStructureValidationError: If option is invalid or
                has different underlying than existing options

        Example:
            >>> structure.add_option(call_option)
            >>> structure.add_option(put_option)
        """
        # Validate option type
        if not isinstance(option, Option):
            raise OptionStructureValidationError(
                f"Expected Option instance, got {type(option).__name__}"
            )

        # Validate underlying consistency
        if self._underlying is None:
            # First option sets the underlying
            self._underlying = option.underlying
        elif option.underlying != self._underlying:
            raise OptionStructureValidationError(
                f"Option underlying '{option.underlying}' does not match "
                f"structure underlying '{self._underlying}'"
            )

        # Set entry date from first option if not set
        if self._entry_date is None:
            self._entry_date = option.entry_date

        # Add option to list
        self._options.append(option)

        # Update net premium
        # Short positions receive premium (positive), long positions pay (negative)
        # entry_cost is already signed correctly: positive for long, negative for short
        # For net_premium, we want: short = credit (positive), long = debit (negative)
        # So we use -position_sign * entry_price * quantity * 100
        option_premium = option.entry_price * option.quantity * CONTRACT_MULTIPLIER
        if option.is_short:
            self._net_premium += option_premium  # Credit received
        else:
            self._net_premium -= option_premium  # Debit paid

        # Invalidate Greeks cache
        self._cached_greeks = {}
        self._cached_greeks_params = None

        logger.debug(
            f"Added option to structure {self._structure_id}: "
            f"{option.position_type.upper()} {option.quantity} "
            f"{option.strike} {option.option_type.upper()}"
        )

    def remove_option(self, option_id: Optional[str] = None, index: Optional[int] = None) -> Option:
        """
        Remove an option leg from the structure.

        Can remove by option_id (matching structure_id or string representation)
        or by index. If neither provided, removes the last option.

        Args:
            option_id: Identifier to match against option's string representation
            index: Index of option to remove (0-based)

        Returns:
            The removed Option instance

        Raises:
            EmptyStructureError: If structure is empty
            IndexError: If index is out of range
            ValueError: If option_id not found

        Example:
            >>> removed = structure.remove_option(index=0)
            >>> removed = structure.remove_option(option_id='SPY 450 CALL')
        """
        if self.is_empty:
            raise EmptyStructureError("Cannot remove option from empty structure")

        if index is not None:
            if index < 0 or index >= len(self._options):
                raise IndexError(
                    f"Index {index} out of range for structure with "
                    f"{len(self._options)} options"
                )
            removed = self._options.pop(index)

        elif option_id is not None:
            # Find option matching the id
            found_index = None
            for i, opt in enumerate(self._options):
                # Match against string representation
                if option_id in str(opt):
                    found_index = i
                    break
            if found_index is None:
                raise ValueError(f"No option found matching '{option_id}'")
            removed = self._options.pop(found_index)

        else:
            # Remove last option
            removed = self._options.pop()

        # Update net premium (reverse the addition)
        option_premium = removed.entry_price * removed.quantity * CONTRACT_MULTIPLIER
        if removed.is_short:
            self._net_premium -= option_premium
        else:
            self._net_premium += option_premium

        # Invalidate Greeks cache
        self._cached_greeks = {}
        self._cached_greeks_params = None

        # Clear underlying if no options left
        if self.is_empty and self._underlying is not None:
            # Keep underlying for structure identity, but note structure is empty
            pass

        logger.debug(
            f"Removed option from structure {self._structure_id}: "
            f"{removed.position_type.upper()} {removed.quantity} "
            f"{removed.strike} {removed.option_type.upper()}"
        )

        return removed

    def get_option(self, index: int) -> Option:
        """
        Get option at specified index.

        Args:
            index: Index of option (0-based)

        Returns:
            Option at the specified index

        Raises:
            IndexError: If index is out of range
        """
        if index < 0 or index >= len(self._options):
            raise IndexError(
                f"Index {index} out of range for structure with "
                f"{len(self._options)} options"
            )
        return self._options[index]

    # =========================================================================
    # Greeks Calculation Methods
    # =========================================================================

    def calculate_net_greeks(
        self,
        spot: Optional[float] = None,
        vol: Optional[float] = None,
        rate: float = 0.04,
        current_date: Optional[datetime] = None
    ) -> Dict[str, float]:
        """
        Calculate net (aggregate) Greeks across all option legs.

        Net Greeks are the sum of position-adjusted Greeks for each leg:
            Net Delta = sum(delta_i * quantity_i * position_sign_i)
            Net Gamma = sum(gamma_i * quantity_i * position_sign_i)
            etc.

        If spot, vol, and current_date are provided, recalculates Greeks
        for each option. Otherwise, uses cached Greeks from each option.

        Args:
            spot: Current spot price (optional, uses cached if not provided)
            vol: Current implied volatility (optional)
            rate: Risk-free rate (default 0.04 = 4%)
            current_date: Current date for time-to-expiry calculation

        Returns:
            Dictionary with net Greeks:
                - 'delta': Net delta exposure
                - 'gamma': Net gamma exposure
                - 'theta': Net daily theta
                - 'vega': Net vega per 1% vol change
                - 'rho': Net rho per 1% rate change

        Raises:
            EmptyStructureError: If structure has no options

        Note:
            Position Greeks account for position direction:
            - Long call: positive delta, positive gamma
            - Short call: negative delta, negative gamma
            - Long put: negative delta, positive gamma
            - Short put: positive delta, negative gamma

        Example:
            >>> greeks = structure.calculate_net_greeks(spot=450, vol=0.20, rate=0.05)
            >>> print(f"Net Delta: {greeks['delta']:.4f}")
        """
        if self.is_empty:
            raise EmptyStructureError("Cannot calculate Greeks for empty structure")

        # Check cache
        cache_key = (spot, vol, rate, current_date)
        if cache_key == self._cached_greeks_params and self._cached_greeks:
            return self._cached_greeks.copy()

        # Initialize net Greeks
        net_greeks = {name: 0.0 for name in GREEK_NAMES}

        for option in self._options:
            # Get or calculate Greeks for this option
            if spot is not None and vol is not None:
                try:
                    option_greeks = option.calculate_greeks(
                        spot=spot,
                        vol=vol,
                        rate=rate,
                        current_date=current_date
                    )
                except OptionExpiredError:
                    # Expired options have zero Greeks
                    option_greeks = {name: 0.0 for name in GREEK_NAMES}
            else:
                # Use cached Greeks from option
                option_greeks = option.greeks
                if not option_greeks:
                    # No cached Greeks available
                    logger.warning(
                        f"No Greeks available for option {option}, using zeros"
                    )
                    option_greeks = {name: 0.0 for name in GREEK_NAMES}

            # Add position-adjusted Greeks
            # Position sign and quantity adjustment
            position_multiplier = option.position_sign * option.quantity

            for greek_name in GREEK_NAMES:
                greek_value = option_greeks.get(greek_name, 0.0)
                net_greeks[greek_name] += greek_value * position_multiplier

        # Cache results
        self._cached_greeks = net_greeks.copy()
        self._cached_greeks_params = cache_key

        return net_greeks

    @property
    def net_delta(self) -> float:
        """Get net delta (uses cached Greeks)."""
        if not self._cached_greeks:
            return 0.0
        return self._cached_greeks.get('delta', 0.0)

    @property
    def net_gamma(self) -> float:
        """Get net gamma (uses cached Greeks)."""
        if not self._cached_greeks:
            return 0.0
        return self._cached_greeks.get('gamma', 0.0)

    @property
    def net_theta(self) -> float:
        """Get net theta (uses cached Greeks)."""
        if not self._cached_greeks:
            return 0.0
        return self._cached_greeks.get('theta', 0.0)

    @property
    def net_vega(self) -> float:
        """Get net vega (uses cached Greeks)."""
        if not self._cached_greeks:
            return 0.0
        return self._cached_greeks.get('vega', 0.0)

    @property
    def net_rho(self) -> float:
        """Get net rho (uses cached Greeks)."""
        if not self._cached_greeks:
            return 0.0
        return self._cached_greeks.get('rho', 0.0)

    def is_delta_neutral(self, threshold: float = 0.10) -> bool:
        """
        Check if structure is approximately delta-neutral.

        Args:
            threshold: Maximum absolute delta to be considered neutral

        Returns:
            True if |net_delta| <= threshold

        Example:
            >>> if structure.is_delta_neutral():
            ...     print("Structure is delta-neutral")
        """
        return abs(self.net_delta) <= threshold

    # =========================================================================
    # P&L Calculation Methods
    # =========================================================================

    def calculate_pnl(self) -> float:
        """
        Calculate total current P&L for the structure.

        Total P&L = sum of P&L for each option leg

        Returns:
            Total P&L in dollars (positive = profit, negative = loss)

        Raises:
            EmptyStructureError: If structure has no options

        Example:
            >>> pnl = structure.calculate_pnl()
            >>> print(f"Total P&L: ${pnl:,.2f}")
        """
        if self.is_empty:
            raise EmptyStructureError("Cannot calculate P&L for empty structure")

        total_pnl = 0.0
        for option in self._options:
            total_pnl += option.calculate_pnl()

        return total_pnl

    def calculate_pnl_percent(self) -> float:
        """
        Calculate P&L as percentage of initial investment/credit.

        For credit structures: pnl / net_premium (profit is positive)
        For debit structures: pnl / abs(net_premium) (profit is positive)

        Returns:
            P&L as decimal (e.g., 0.25 for 25% profit)

        Raises:
            EmptyStructureError: If structure has no options
            ValueError: If net_premium is zero

        Example:
            >>> pnl_pct = structure.calculate_pnl_percent()
            >>> print(f"P&L: {pnl_pct:.1%}")
        """
        if self.is_empty:
            raise EmptyStructureError("Cannot calculate P&L for empty structure")

        if abs(self._net_premium) < 1e-10:
            raise ValueError("Cannot calculate percentage with zero net premium")

        pnl = self.calculate_pnl()
        return pnl / abs(self._net_premium)

    def get_current_value(self) -> float:
        """
        Get current market value of the structure.

        For long positions: positive (asset value)
        For short positions: negative (liability)

        Returns:
            Current market value in dollars

        Raises:
            EmptyStructureError: If structure has no options
        """
        if self.is_empty:
            raise EmptyStructureError("Cannot get value for empty structure")

        total_value = 0.0
        for option in self._options:
            total_value += option.market_value

        return total_value

    # =========================================================================
    # Payoff Calculation Methods
    # =========================================================================

    def get_payoff_at_expiry(self, spot_prices: np.ndarray) -> np.ndarray:
        """
        Calculate payoff at expiration for an array of spot prices.

        Payoff is the intrinsic value at expiration, accounting for
        all legs and their positions (long/short).

        Args:
            spot_prices: NumPy array of spot prices to evaluate

        Returns:
            NumPy array of total payoffs at each spot price

        Raises:
            EmptyStructureError: If structure has no options

        Note:
            This calculates the payoff only, not the total P&L.
            Total P&L = Payoff - Initial Debit (or + Initial Credit)

        Example:
            >>> spots = np.linspace(400, 500, 101)
            >>> payoffs = structure.get_payoff_at_expiry(spots)
            >>> plt.plot(spots, payoffs)
        """
        if self.is_empty:
            raise EmptyStructureError("Cannot calculate payoff for empty structure")

        spot_prices = np.asarray(spot_prices, dtype=np.float64)
        payoffs = np.zeros_like(spot_prices)

        for option in self._options:
            for i, spot in enumerate(spot_prices):
                payoffs[i] += option.get_payoff_at_expiry(spot)

        return payoffs

    def get_pnl_at_expiry(self, spot_prices: np.ndarray) -> np.ndarray:
        """
        Calculate total P&L at expiration for an array of spot prices.

        P&L includes the payoff minus the initial debit (or plus initial credit).

        Args:
            spot_prices: NumPy array of spot prices to evaluate

        Returns:
            NumPy array of total P&L at each spot price

        Raises:
            EmptyStructureError: If structure has no options

        Example:
            >>> spots = np.linspace(400, 500, 101)
            >>> pnl = structure.get_pnl_at_expiry(spots)
            >>> plt.plot(spots, pnl)
        """
        payoffs = self.get_payoff_at_expiry(spot_prices)
        # P&L = Payoff + Net Premium (positive premium = credit received)
        return payoffs + self._net_premium

    def get_payoff_diagram(
        self,
        spot_range: Optional[Tuple[float, float]] = None,
        num_points: int = 101
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate payoff diagram data.

        Args:
            spot_range: Tuple of (min_spot, max_spot). If None, auto-calculated
                       based on strikes with margin.
            num_points: Number of points in the diagram (default 101)

        Returns:
            Tuple of (spot_prices, payoffs) arrays

        Raises:
            EmptyStructureError: If structure has no options

        Example:
            >>> spots, payoffs = structure.get_payoff_diagram()
            >>> plt.plot(spots, payoffs)
            >>> plt.xlabel('Spot Price')
            >>> plt.ylabel('Payoff ($)')
        """
        if self.is_empty:
            raise EmptyStructureError("Cannot generate payoff diagram for empty structure")

        # Determine spot range if not provided
        if spot_range is None:
            strikes = [opt.strike for opt in self._options]
            min_strike = min(strikes)
            max_strike = max(strikes)
            strike_range = max_strike - min_strike
            if strike_range < 1.0:
                # Single strike (e.g., straddle) - use percentage of strike
                margin = min_strike * DEFAULT_BREAKEVEN_SEARCH_MARGIN_FACTOR
            else:
                # Multiple strikes - use strike range as margin
                margin = strike_range * 0.5

            spot_range = (min_strike - margin, max_strike + margin)

        spot_prices = np.linspace(spot_range[0], spot_range[1], num_points)
        payoffs = self.get_payoff_at_expiry(spot_prices)

        return spot_prices, payoffs

    # =========================================================================
    # Max Profit/Loss Calculation
    # =========================================================================

    def calculate_max_profit(
        self,
        spot_range: Optional[Tuple[float, float]] = None,
        num_points: int = 1000
    ) -> float:
        """
        Calculate theoretical maximum profit at expiration.

        Evaluates payoff across a range of spot prices to find the maximum.

        Args:
            spot_range: Range of spot prices to evaluate. If None, uses
                       wide range based on strikes.
            num_points: Number of points to evaluate (default 1000)

        Returns:
            Maximum possible profit in dollars

        Raises:
            EmptyStructureError: If structure has no options

        Note:
            For structures with unlimited profit potential (e.g., long calls),
            this returns the profit at the upper bound of the spot range.

        Example:
            >>> max_profit = structure.calculate_max_profit()
            >>> print(f"Max Profit: ${max_profit:,.2f}")
        """
        if self.is_empty:
            raise EmptyStructureError("Cannot calculate max profit for empty structure")

        # Determine spot range
        if spot_range is None:
            strikes = [opt.strike for opt in self._options]
            min_strike = min(strikes)
            max_strike = max(strikes)

            # Use wide range to capture max profit
            spot_range = (
                min_strike * DEFAULT_MIN_SPOT_FACTOR,  # Near zero
                max_strike * DEFAULT_MAX_SPOT_FACTOR   # 3x highest strike
            )

        spots = np.linspace(spot_range[0], spot_range[1], num_points)
        pnl_values = self.get_pnl_at_expiry(spots)

        return float(np.max(pnl_values))

    def calculate_max_loss(
        self,
        spot_range: Optional[Tuple[float, float]] = None,
        num_points: int = 1000
    ) -> float:
        """
        Calculate theoretical maximum loss at expiration.

        Evaluates payoff across a range of spot prices to find the minimum.

        Args:
            spot_range: Range of spot prices to evaluate. If None, uses
                       wide range based on strikes.
            num_points: Number of points to evaluate (default 1000)

        Returns:
            Maximum possible loss in dollars (as a negative number)

        Raises:
            EmptyStructureError: If structure has no options

        Note:
            For structures with unlimited loss potential (e.g., naked calls),
            this returns the loss at the upper bound of the spot range.

        Example:
            >>> max_loss = structure.calculate_max_loss()
            >>> print(f"Max Loss: ${max_loss:,.2f}")
        """
        if self.is_empty:
            raise EmptyStructureError("Cannot calculate max loss for empty structure")

        # Determine spot range
        if spot_range is None:
            strikes = [opt.strike for opt in self._options]
            min_strike = min(strikes)
            max_strike = max(strikes)

            # Use wide range to capture max loss
            spot_range = (
                min_strike * DEFAULT_MIN_SPOT_FACTOR,  # Near zero
                max_strike * DEFAULT_MAX_SPOT_FACTOR   # 3x highest strike
            )

        spots = np.linspace(spot_range[0], spot_range[1], num_points)
        pnl_values = self.get_pnl_at_expiry(spots)

        return float(np.min(pnl_values))

    # =========================================================================
    # Breakeven Calculation
    # =========================================================================

    def calculate_breakeven_points(
        self,
        spot_range: Optional[Tuple[float, float]] = None,
        num_search_points: int = 1000,
        tolerance: float = BREAKEVEN_TOLERANCE
    ) -> List[float]:
        """
        Calculate breakeven spot prices at expiration.

        Breakeven points are where total P&L = 0. This method finds all
        such points within the specified range by:
        1. Scanning for sign changes in P&L
        2. Using root-finding on each zero-crossing

        Args:
            spot_range: Range to search for breakevens. If None, auto-calculated.
            num_search_points: Number of points for initial scan
            tolerance: Numerical tolerance for root-finding

        Returns:
            List of breakeven spot prices (sorted ascending)

        Raises:
            EmptyStructureError: If structure has no options

        Example:
            >>> breakevens = structure.calculate_breakeven_points()
            >>> for be in breakevens:
            ...     print(f"Breakeven at ${be:.2f}")
        """
        if self.is_empty:
            raise EmptyStructureError(
                "Cannot calculate breakeven points for empty structure"
            )

        # Determine spot range
        if spot_range is None:
            strikes = [opt.strike for opt in self._options]
            min_strike = min(strikes)
            max_strike = max(strikes)
            margin = max(max_strike - min_strike, min_strike * DEFAULT_BREAKEVEN_SEARCH_MARGIN_FACTOR)
            spot_range = (
                max(min_strike - margin * 2, 0.01),
                max_strike + margin * 2
            )

        # Create P&L function
        def pnl_func(spot: float) -> float:
            return float(self.get_pnl_at_expiry(np.array([spot]))[0])

        # Scan for sign changes
        spots = np.linspace(spot_range[0], spot_range[1], num_search_points)
        pnl_values = self.get_pnl_at_expiry(spots)

        breakevens = []

        for i in range(len(spots) - 1):
            # Check for sign change
            if pnl_values[i] * pnl_values[i + 1] < 0:
                # Found a zero crossing - use root finding
                try:
                    root = brentq(
                        pnl_func,
                        spots[i],
                        spots[i + 1],
                        xtol=tolerance
                    )
                    breakevens.append(root)
                except (ValueError, RuntimeError):
                    # Root finding failed, use midpoint
                    breakevens.append((spots[i] + spots[i + 1]) / 2)

            # Check for exact zero
            elif abs(pnl_values[i]) < tolerance:
                if not breakevens or abs(spots[i] - breakevens[-1]) > tolerance:
                    breakevens.append(spots[i])

        return sorted(breakevens)

    # =========================================================================
    # Price Update Methods
    # =========================================================================

    def update_all_prices(
        self,
        prices: Dict[int, float],
        timestamp: datetime
    ) -> None:
        """
        Update prices for multiple option legs.

        Args:
            prices: Dictionary mapping option index to new price
            timestamp: Timestamp of price update

        Example:
            >>> structure.update_all_prices({0: 3.50, 1: 4.25}, datetime.now())
        """
        for index, price in prices.items():
            if 0 <= index < len(self._options):
                self._options[index].update_price(price, timestamp)

        # Invalidate Greeks cache
        self._cached_greeks = {}
        self._cached_greeks_params = None

    def update_prices_from_market_data(
        self,
        market_data: Dict[str, Any],
        timestamp: datetime,
        price_field: str = 'mid'
    ) -> None:
        """
        Update option prices from market data.

        Args:
            market_data: Dictionary containing either:
                - 'option_chain': DataFrame with columns strike, expiration, call_put, bid, ask
                - Or direct price keys like "SPY_500.0_call_2024-02-15"
            timestamp: Timestamp of price update
            price_field: Which price field to use ('bid', 'ask', 'mid')

        Note:
            Supports both DataFrame option_chain format (from DataStream) and
            direct dictionary format for flexibility.

            When exact matches aren't found, uses interpolation/estimation:
            1. Tries to find options with same strike but different expiration
            2. Tries to find options with same expiration but different strike
            3. Falls back to Black-Scholes estimation using market IV
        """
        import pandas as pd
        import numpy as np

        # Check if market_data contains an option_chain DataFrame
        option_chain = market_data.get('option_chain')
        spot_price = market_data.get('spot', 0)
        market_iv = market_data.get('iv', 0.20)  # Default 20% IV

        if option_chain is not None and isinstance(option_chain, pd.DataFrame) and not option_chain.empty:
            # Use DataFrame to update prices
            for option in self._options:
                try:
                    new_price = self._find_option_price(
                        option, option_chain, spot_price, market_iv,
                        timestamp, price_field
                    )

                    if new_price is not None and new_price > 0:
                        option.update_price(new_price, timestamp)

                except OptionExpiredError:
                    logger.warning(f"Cannot update expired option: {option}")
                except Exception as e:
                    logger.debug(f"Could not update option {option.strike} {option.option_type}: {e}")
        else:
            # Fall back to direct key lookup (original behavior)
            for option in self._options:
                # Create key for lookup
                key = f"{option.underlying}_{option.strike}_{option.option_type}_{option.expiration.date()}"

                if key in market_data:
                    price_data = market_data[key]
                    if isinstance(price_data, dict):
                        new_price = price_data.get(price_field, price_data.get('mid', 0))
                    else:
                        new_price = float(price_data)

                    try:
                        option.update_price(new_price, timestamp)
                    except OptionExpiredError:
                        logger.warning(f"Cannot update expired option: {option}")

        # Invalidate Greeks cache after any price updates
        self._cached_greeks = {}
        self._cached_greeks_params = None

    def _find_option_price(
        self,
        option,
        option_chain,
        spot_price: float,
        market_iv: float,
        timestamp: datetime,
        price_field: str = 'mid'
    ) -> Optional[float]:
        """
        Find or estimate option price using multiple strategies.

        Strategy order:
        1. Exact match (same strike, expiration, type)
        2. Interpolate from nearby strikes with same expiration
        3. Interpolate from same strike with nearby expirations
        4. Black-Scholes estimation using market IV

        Returns:
            Estimated price or None if cannot determine
        """
        import pandas as pd
        import numpy as np
        from scipy.stats import norm

        # Handle option type variants
        option_type_variants = [option.option_type, option.option_type.capitalize()]

        # Convert expiration to date for comparison
        option_exp = option.expiration
        if hasattr(option_exp, 'date'):
            option_exp_date = option_exp.date()
        else:
            option_exp_date = option_exp

        def get_mid_price(row):
            """Extract mid price from a row."""
            bid = row.get('bid', 0) or 0
            ask = row.get('ask', 0) or 0
            if price_field == 'bid':
                return bid
            elif price_field == 'ask':
                return ask
            else:  # mid
                if ask > 0 and bid > 0:
                    return (bid + ask) / 2.0
                elif ask > 0:
                    return ask
                return bid

        # Strategy 1: Exact match
        mask = (
            (option_chain['strike'] == option.strike) &
            (option_chain['call_put'].isin(option_type_variants))
        )
        if 'expiration' in option_chain.columns:
            chain_exp = pd.to_datetime(option_chain['expiration']).dt.date
            mask = mask & (chain_exp == option_exp_date)

        matching = option_chain[mask]
        if not matching.empty:
            return get_mid_price(matching.iloc[0])

        # Strategy 2: Interpolate from nearby strikes with similar expiration
        # Find options of same type with similar DTE
        type_mask = option_chain['call_put'].isin(option_type_variants)
        same_type = option_chain[type_mask].copy()

        if not same_type.empty and 'expiration' in same_type.columns:
            # Calculate DTE for each option in chain
            same_type['chain_dte'] = (pd.to_datetime(same_type['expiration']) - timestamp).dt.days
            target_dte = (option.expiration - timestamp).days

            # Find options with similar DTE (within 7 days)
            similar_dte = same_type[abs(same_type['chain_dte'] - target_dte) <= 7]

            if len(similar_dte) >= 2:
                # Find strikes above and below our target
                strikes_below = similar_dte[similar_dte['strike'] < option.strike].sort_values('strike', ascending=False)
                strikes_above = similar_dte[similar_dte['strike'] > option.strike].sort_values('strike')

                if not strikes_below.empty and not strikes_above.empty:
                    lower = strikes_below.iloc[0]
                    upper = strikes_above.iloc[0]

                    lower_price = get_mid_price(lower)
                    upper_price = get_mid_price(upper)

                    if lower_price > 0 and upper_price > 0:
                        # Linear interpolation by strike
                        weight = (option.strike - lower['strike']) / (upper['strike'] - lower['strike'])
                        interpolated = lower_price + weight * (upper_price - lower_price)

                        # Adjust for DTE difference if needed
                        avg_chain_dte = (lower['chain_dte'] + upper['chain_dte']) / 2
                        if avg_chain_dte > 0 and target_dte > 0:
                            dte_ratio = target_dte / avg_chain_dte
                            # Time value scales roughly with sqrt of time
                            time_adjustment = np.sqrt(dte_ratio)
                            interpolated *= time_adjustment

                        return max(interpolated, 0.01)

        # Strategy 3: Black-Scholes estimation
        if spot_price > 0:
            return self._estimate_price_black_scholes(
                option, spot_price, market_iv, timestamp
            )

        return None

    def _estimate_price_black_scholes(
        self,
        option,
        spot_price: float,
        iv: float,
        timestamp: datetime,
        risk_free_rate: float = 0.05
    ) -> Optional[float]:
        """
        Estimate option price using Black-Scholes model.

        Args:
            option: Option object with strike, expiration, option_type
            spot_price: Current underlying price
            iv: Implied volatility (decimal, e.g., 0.20 for 20%)
            timestamp: Current timestamp
            risk_free_rate: Risk-free rate (default 5%)

        Returns:
            Estimated option price
        """
        import numpy as np
        from scipy.stats import norm

        try:
            # Calculate time to expiration in years
            dte_days = (option.expiration - timestamp).days
            if dte_days <= 0:
                # At or past expiration - return intrinsic value
                if option.option_type.lower() == 'call':
                    return max(spot_price - option.strike, 0)
                else:
                    return max(option.strike - spot_price, 0)

            T = dte_days / 365.0
            S = spot_price
            K = option.strike
            r = risk_free_rate
            sigma = iv if iv > 0 else 0.20  # Default 20% if IV not available

            # Black-Scholes formula
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)

            if option.option_type.lower() == 'call':
                price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            else:  # put
                price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

            return max(price, 0.01)  # Ensure minimum price

        except Exception as e:
            logger.debug(f"Black-Scholes estimation failed: {e}")
            return None

    # =========================================================================
    # Expiration Helpers
    # =========================================================================

    def get_earliest_expiration(self) -> Optional[datetime]:
        """
        Get the earliest expiration date among all legs.

        Returns:
            Earliest expiration datetime, or None if empty

        Example:
            >>> exp = structure.get_earliest_expiration()
            >>> days_to_exp = (exp - datetime.now()).days
        """
        if self.is_empty:
            return None
        return min(opt.expiration for opt in self._options)

    def get_latest_expiration(self) -> Optional[datetime]:
        """
        Get the latest expiration date among all legs.

        Returns:
            Latest expiration datetime, or None if empty
        """
        if self.is_empty:
            return None
        return max(opt.expiration for opt in self._options)

    def is_same_expiration(self) -> bool:
        """
        Check if all legs have the same expiration.

        Returns:
            True if all options expire on the same date

        Example:
            >>> if structure.is_same_expiration():
            ...     print("All legs expire together")
        """
        if self.is_empty or len(self._options) == 1:
            return True

        first_exp = self._options[0].expiration
        return all(opt.expiration == first_exp for opt in self._options)

    def get_days_to_expiry(
        self,
        current_date: Optional[datetime] = None
    ) -> Optional[int]:
        """
        Get days to earliest expiration.

        Args:
            current_date: Reference date. If None, uses current timestamp
                         from first option.

        Returns:
            Days to earliest expiration, or None if empty
        """
        earliest = self.get_earliest_expiration()
        if earliest is None:
            return None

        if current_date is None and self._options:
            current_date = self._options[0].current_timestamp

        if current_date is None:
            current_date = datetime.now()

        delta = earliest - current_date
        return max(delta.days, 0)

    # =========================================================================
    # Serialization Methods
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert structure to dictionary representation.

        Returns:
            Dictionary with all structure attributes and options

        Example:
            >>> d = structure.to_dict()
            >>> json.dumps(d, default=str)
        """
        return {
            'structure_id': self._structure_id,
            'structure_type': self._structure_type,
            'underlying': self._underlying,
            'entry_date': self._entry_date,
            'net_premium': self._net_premium,
            'num_legs': len(self._options),
            'options': [opt.to_dict() for opt in self._options],
            'current_pnl': self.calculate_pnl() if not self.is_empty else 0.0,
            'current_value': self.get_current_value() if not self.is_empty else 0.0,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OptionStructure':
        """
        Create OptionStructure from dictionary representation.

        Args:
            data: Dictionary with structure attributes

        Returns:
            New OptionStructure instance

        Raises:
            KeyError: If required keys are missing

        Example:
            >>> structure = OptionStructure.from_dict(saved_data)
        """
        structure = cls(
            structure_type=data.get('structure_type', 'custom'),
            underlying=data.get('underlying'),
            structure_id=data.get('structure_id'),
            entry_date=data.get('entry_date')
        )

        # Restore options
        for opt_data in data.get('options', []):
            option = Option.from_dict(opt_data)
            structure.add_option(option)

        return structure

    # =========================================================================
    # Special Methods
    # =========================================================================

    def __repr__(self) -> str:
        """Return detailed string representation."""
        return (
            f"OptionStructure("
            f"id={self._structure_id!r}, "
            f"type={self._structure_type!r}, "
            f"underlying={self._underlying!r}, "
            f"num_legs={len(self._options)}, "
            f"net_premium={self._net_premium:.2f}"
            f")"
        )

    def __str__(self) -> str:
        """Return human-readable string."""
        if self.is_empty:
            return f"{self._structure_type.upper()} on {self._underlying or 'N/A'} (empty)"

        legs_str = ", ".join(
            f"{opt.position_type[0].upper()}{opt.quantity} "
            f"{opt.strike}{opt.option_type[0].upper()}"
            for opt in self._options
        )

        premium_str = f"+${self._net_premium:,.0f}" if self._net_premium >= 0 else f"-${-self._net_premium:,.0f}"

        return (
            f"{self._structure_type.upper()} on {self._underlying}: "
            f"[{legs_str}] {premium_str}"
        )

    def __len__(self) -> int:
        """Return number of option legs."""
        return len(self._options)

    def __iter__(self):
        """Iterate over option legs."""
        return iter(self._options)

    def __getitem__(self, index: int) -> Option:
        """Get option leg by index."""
        return self._options[index]

    def __eq__(self, other: object) -> bool:
        """Check equality based on structure ID."""
        if not isinstance(other, OptionStructure):
            return NotImplemented
        return self._structure_id == other._structure_id

    def __hash__(self) -> int:
        """Return hash based on structure ID."""
        return hash(self._structure_id)


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Main class
    'OptionStructure',

    # Exceptions
    'OptionStructureError',
    'OptionStructureValidationError',
    'EmptyStructureError',

    # Constants
    'BREAKEVEN_TOLERANCE',
    'GREEK_NAMES',
]
