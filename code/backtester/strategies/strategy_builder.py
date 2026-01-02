"""
Strategy Builder - Fluent API for Declarative Strategy Creation

This module provides a fluent builder pattern for creating options trading strategies
declaratively. Instead of subclassing Strategy, users can compose strategies using
chainable methods and composable condition/structure factories.

Key Features:
    - Fluent builder API with method chaining
    - Composable entry/exit conditions with AND/OR operators
    - Pre-built condition factories (iv_rank_above, profit_target, etc.)
    - Structure factory functions (short_straddle, iron_condor, etc.)
    - Position sizing functions (risk_percent, fixed_contracts, etc.)
    - Validation at build time

Usage:
    from backtester.strategies.strategy_builder import (
        StrategyBuilder,
        iv_rank_above, profit_target, stop_loss, dte_below,
        short_straddle, iron_condor,
        risk_percent, fixed_contracts
    )

    strategy = (StrategyBuilder()
        .name("High IV Short Straddle")
        .underlying("SPY")
        .entry_condition(iv_rank_above(70))
        .structure(short_straddle(dte=30))
        .exit_condition(profit_target(0.50) | stop_loss(2.0) | dte_below(7))
        .position_size(risk_percent(2.0))
        .build())

References:
    - Builder Pattern: Gang of Four Design Patterns
    - Fluent Interface: Martin Fowler's Domain Specific Languages
"""

import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from backtester.strategies.strategy import (
    Strategy,
    StrategyError,
    StrategyValidationError,
)
from backtester.core.option_structure import OptionStructure
from backtester.structures.straddle import ShortStraddle, LongStraddle
from backtester.structures.strangle import ShortStrangle, LongStrangle
from backtester.structures.condor import IronCondor
from backtester.structures.spread import BullCallSpread, BearPutSpread

# Configure module logger
logger = logging.getLogger(__name__)


# =============================================================================
# Exceptions
# =============================================================================


class StrategyBuilderError(StrategyError):
    """Base exception for StrategyBuilder errors."""

    pass


class BuilderValidationError(StrategyBuilderError):
    """Exception raised when builder validation fails."""

    pass


class ConditionError(StrategyBuilderError):
    """Exception raised for condition-related errors."""

    pass


# =============================================================================
# Condition Classes - Composable with & and | operators
# =============================================================================


class Condition(ABC):
    """
    Abstract base class for composable conditions.

    Conditions can be combined using:
        - & (AND): Both conditions must be true
        - | (OR): Either condition must be true
        - ~ (NOT): Negate the condition

    Example:
        >>> cond = iv_rank_above(70) & dte_above(30)
        >>> cond.evaluate(market_data)  # True if both conditions met
    """

    @abstractmethod
    def evaluate(self, context: Dict[str, Any]) -> bool:
        """
        Evaluate the condition against the given context.

        Args:
            context: Dictionary containing market data, position info, etc.

        Returns:
            True if condition is satisfied, False otherwise
        """
        pass

    @abstractmethod
    def describe(self) -> str:
        """Return human-readable description of the condition."""
        pass

    def __and__(self, other: "Condition") -> "AndCondition":
        """Combine with AND logic."""
        return AndCondition(self, other)

    def __or__(self, other: "Condition") -> "OrCondition":
        """Combine with OR logic."""
        return OrCondition(self, other)

    def __invert__(self) -> "NotCondition":
        """Negate the condition."""
        return NotCondition(self)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.describe()}>"


class AndCondition(Condition):
    """Composite condition that requires ALL sub-conditions to be true."""

    def __init__(self, *conditions: Condition):
        self.conditions = list(conditions)

    def evaluate(self, context: Dict[str, Any]) -> bool:
        return all(c.evaluate(context) for c in self.conditions)

    def describe(self) -> str:
        return " AND ".join(f"({c.describe()})" for c in self.conditions)


class OrCondition(Condition):
    """Composite condition that requires ANY sub-condition to be true."""

    def __init__(self, *conditions: Condition):
        self.conditions = list(conditions)

    def evaluate(self, context: Dict[str, Any]) -> bool:
        return any(c.evaluate(context) for c in self.conditions)

    def describe(self) -> str:
        return " OR ".join(f"({c.describe()})" for c in self.conditions)


class NotCondition(Condition):
    """Negates a condition."""

    def __init__(self, condition: Condition):
        self.condition = condition

    def evaluate(self, context: Dict[str, Any]) -> bool:
        return not self.condition.evaluate(context)

    def describe(self) -> str:
        return f"NOT ({self.condition.describe()})"


class AlwaysTrue(Condition):
    """Condition that always returns True."""

    def evaluate(self, context: Dict[str, Any]) -> bool:
        return True

    def describe(self) -> str:
        return "always"


class AlwaysFalse(Condition):
    """Condition that always returns False."""

    def evaluate(self, context: Dict[str, Any]) -> bool:
        return False

    def describe(self) -> str:
        return "never"


# =============================================================================
# Entry Condition Factories
# =============================================================================


class IVRankAbove(Condition):
    """Condition: IV rank is above threshold."""

    def __init__(self, threshold: float):
        if not 0 <= threshold <= 100:
            raise ConditionError(f"IV rank threshold must be 0-100, got {threshold}")
        self.threshold = threshold

    def evaluate(self, context: Dict[str, Any]) -> bool:
        iv_rank = context.get("iv_rank") or context.get("iv_percentile")
        if iv_rank is None:
            return False
        return iv_rank > self.threshold

    def describe(self) -> str:
        return f"IV rank > {self.threshold}%"


class IVRankBelow(Condition):
    """Condition: IV rank is below threshold."""

    def __init__(self, threshold: float):
        if not 0 <= threshold <= 100:
            raise ConditionError(f"IV rank threshold must be 0-100, got {threshold}")
        self.threshold = threshold

    def evaluate(self, context: Dict[str, Any]) -> bool:
        iv_rank = context.get("iv_rank") or context.get("iv_percentile")
        if iv_rank is None:
            return False
        return iv_rank < self.threshold

    def describe(self) -> str:
        return f"IV rank < {self.threshold}%"


class IVRankBetween(Condition):
    """Condition: IV rank is within range."""

    def __init__(self, low: float, high: float):
        if not 0 <= low <= 100 or not 0 <= high <= 100:
            raise ConditionError("IV rank thresholds must be 0-100")
        if low >= high:
            raise ConditionError(f"low ({low}) must be less than high ({high})")
        self.low = low
        self.high = high

    def evaluate(self, context: Dict[str, Any]) -> bool:
        iv_rank = context.get("iv_rank") or context.get("iv_percentile")
        if iv_rank is None:
            return False
        return self.low <= iv_rank <= self.high

    def describe(self) -> str:
        return f"IV rank between {self.low}% and {self.high}%"


class VIXAbove(Condition):
    """Condition: VIX is above threshold."""

    def __init__(self, threshold: float):
        if threshold < 0:
            raise ConditionError(f"VIX threshold must be positive, got {threshold}")
        self.threshold = threshold

    def evaluate(self, context: Dict[str, Any]) -> bool:
        vix = context.get("vix")
        if vix is None:
            return False
        return vix > self.threshold

    def describe(self) -> str:
        return f"VIX > {self.threshold}"


class VIXBelow(Condition):
    """Condition: VIX is below threshold."""

    def __init__(self, threshold: float):
        if threshold < 0:
            raise ConditionError(f"VIX threshold must be positive, got {threshold}")
        self.threshold = threshold

    def evaluate(self, context: Dict[str, Any]) -> bool:
        vix = context.get("vix")
        if vix is None:
            return False
        return vix < self.threshold

    def describe(self) -> str:
        return f"VIX < {self.threshold}"


class VIXBetween(Condition):
    """Condition: VIX is within range."""

    def __init__(self, low: float, high: float):
        if low < 0 or high < 0:
            raise ConditionError("VIX thresholds must be positive")
        if low >= high:
            raise ConditionError(f"low ({low}) must be less than high ({high})")
        self.low = low
        self.high = high

    def evaluate(self, context: Dict[str, Any]) -> bool:
        vix = context.get("vix")
        if vix is None:
            return False
        return self.low <= vix <= self.high

    def describe(self) -> str:
        return f"VIX between {self.low} and {self.high}"


class DTEAbove(Condition):
    """Condition: Days to expiration is above threshold."""

    def __init__(self, days: int):
        if days < 0:
            raise ConditionError(f"DTE threshold must be non-negative, got {days}")
        self.days = days

    def evaluate(self, context: Dict[str, Any]) -> bool:
        dte = context.get("dte")
        if dte is None:
            return False
        return dte > self.days

    def describe(self) -> str:
        return f"DTE > {self.days}"


class DTEBelow(Condition):
    """Condition: Days to expiration is below threshold."""

    def __init__(self, days: int):
        if days < 0:
            raise ConditionError(f"DTE threshold must be non-negative, got {days}")
        self.days = days

    def evaluate(self, context: Dict[str, Any]) -> bool:
        dte = context.get("dte")
        if dte is None:
            return False
        return dte < self.days

    def describe(self) -> str:
        return f"DTE < {self.days}"


class DTEBetween(Condition):
    """Condition: Days to expiration is within range."""

    def __init__(self, min_days: int, max_days: int):
        if min_days < 0 or max_days < 0:
            raise ConditionError("DTE thresholds must be non-negative")
        if min_days >= max_days:
            raise ConditionError(
                f"min_days ({min_days}) must be less than max_days ({max_days})"
            )
        self.min_days = min_days
        self.max_days = max_days

    def evaluate(self, context: Dict[str, Any]) -> bool:
        dte = context.get("dte")
        if dte is None:
            return False
        return self.min_days <= dte <= self.max_days

    def describe(self) -> str:
        return f"DTE between {self.min_days} and {self.max_days}"


class DayOfWeek(Condition):
    """Condition: Current day is one of specified weekdays."""

    WEEKDAYS = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6,
    }

    def __init__(self, *days: str):
        self.allowed_days = set()
        for day in days:
            day_lower = day.lower()
            if day_lower not in self.WEEKDAYS:
                raise ConditionError(f"Invalid weekday: {day}")
            self.allowed_days.add(self.WEEKDAYS[day_lower])

    def evaluate(self, context: Dict[str, Any]) -> bool:
        current_date = context.get("date") or context.get("current_date")
        if current_date is None:
            return False
        if isinstance(current_date, datetime):
            current_date = current_date.date()
        return current_date.weekday() in self.allowed_days

    def describe(self) -> str:
        day_names = [
            name for name, num in self.WEEKDAYS.items() if num in self.allowed_days
        ]
        return f"day is {', '.join(day_names)}"


class NoOpenPositions(Condition):
    """Condition: No positions are currently open on the underlying."""

    def __init__(self, underlying: Optional[str] = None):
        self.underlying = underlying

    def evaluate(self, context: Dict[str, Any]) -> bool:
        open_positions = context.get("open_positions", [])
        if not open_positions:
            return True
        if self.underlying:
            return not any(p.underlying == self.underlying for p in open_positions)
        return len(open_positions) == 0

    def describe(self) -> str:
        if self.underlying:
            return f"no open positions on {self.underlying}"
        return "no open positions"


class MaxOpenPositions(Condition):
    """Condition: Number of open positions is below maximum."""

    def __init__(self, max_positions: int):
        if max_positions < 0:
            raise ConditionError(
                f"max_positions must be non-negative, got {max_positions}"
            )
        self.max_positions = max_positions

    def evaluate(self, context: Dict[str, Any]) -> bool:
        open_positions = context.get("open_positions", [])
        return len(open_positions) < self.max_positions

    def describe(self) -> str:
        return f"open positions < {self.max_positions}"


# =============================================================================
# Exit Condition Factories
# =============================================================================


class ProfitTarget(Condition):
    """Condition: Profit target reached (as percentage of max profit)."""

    def __init__(self, target_pct: float):
        """
        Args:
            target_pct: Target as decimal (e.g., 0.50 for 50%)
        """
        if not 0 < target_pct <= 1.0:
            raise ConditionError(f"target_pct must be 0-1, got {target_pct}")
        self.target_pct = target_pct

    def evaluate(self, context: Dict[str, Any]) -> bool:
        pnl_pct = context.get("pnl_pct") or context.get("profit_pct")
        if pnl_pct is None:
            # Try to calculate from position
            position = context.get("position")
            if position and hasattr(position, "calculate_pnl_percent"):
                try:
                    pnl_pct = position.calculate_pnl_percent()
                except Exception:
                    return False
            else:
                return False
        return pnl_pct >= self.target_pct

    def describe(self) -> str:
        return f"profit >= {self.target_pct:.0%}"


class StopLoss(Condition):
    """Condition: Stop loss triggered (as multiple of max profit)."""

    def __init__(self, loss_multiple: float):
        """
        Args:
            loss_multiple: Loss as multiple of max profit (e.g., 2.0 for 200% loss)
        """
        if loss_multiple <= 0:
            raise ConditionError(f"loss_multiple must be positive, got {loss_multiple}")
        self.loss_multiple = loss_multiple

    def evaluate(self, context: Dict[str, Any]) -> bool:
        pnl_pct = context.get("pnl_pct") or context.get("profit_pct")
        if pnl_pct is None:
            position = context.get("position")
            if position and hasattr(position, "calculate_pnl_percent"):
                try:
                    pnl_pct = position.calculate_pnl_percent()
                except Exception:
                    return False
            else:
                return False
        return pnl_pct <= -self.loss_multiple

    def describe(self) -> str:
        return f"loss >= {self.loss_multiple:.0%}"


class FixedStopLoss(Condition):
    """Condition: Stop loss triggered (fixed dollar amount)."""

    def __init__(self, max_loss: float):
        """
        Args:
            max_loss: Maximum loss in dollars (positive number)
        """
        if max_loss <= 0:
            raise ConditionError(f"max_loss must be positive, got {max_loss}")
        self.max_loss = max_loss

    def evaluate(self, context: Dict[str, Any]) -> bool:
        pnl = context.get("pnl") or context.get("unrealized_pnl")
        if pnl is None:
            position = context.get("position")
            if position and hasattr(position, "calculate_pnl"):
                try:
                    pnl = position.calculate_pnl()
                except Exception:
                    return False
            else:
                return False
        return pnl <= -self.max_loss

    def describe(self) -> str:
        return f"loss >= ${self.max_loss:,.2f}"


class TrailingStop(Condition):
    """Condition: Trailing stop triggered."""

    def __init__(self, trail_pct: float):
        """
        Args:
            trail_pct: Trailing stop as percentage from peak (e.g., 0.25 for 25%)
        """
        if not 0 < trail_pct < 1.0:
            raise ConditionError(f"trail_pct must be 0-1, got {trail_pct}")
        self.trail_pct = trail_pct

    def evaluate(self, context: Dict[str, Any]) -> bool:
        current_pnl = context.get("pnl") or context.get("unrealized_pnl")
        peak_pnl = context.get("peak_pnl") or context.get("max_pnl")

        if current_pnl is None or peak_pnl is None:
            return False

        if peak_pnl <= 0:
            return False

        drawdown = (peak_pnl - current_pnl) / peak_pnl
        return drawdown >= self.trail_pct

    def describe(self) -> str:
        return f"trailing stop {self.trail_pct:.0%} from peak"


class HoldingPeriod(Condition):
    """Condition: Position has been held for N days."""

    def __init__(self, days: int):
        if days <= 0:
            raise ConditionError(f"days must be positive, got {days}")
        self.days = days

    def evaluate(self, context: Dict[str, Any]) -> bool:
        entry_date = context.get("entry_date")
        current_date = context.get("date") or context.get("current_date")

        if entry_date is None or current_date is None:
            return False

        if isinstance(entry_date, datetime):
            entry_date = entry_date.date()
        if isinstance(current_date, datetime):
            current_date = current_date.date()

        days_held = (current_date - entry_date).days
        return days_held >= self.days

    def describe(self) -> str:
        return f"held >= {self.days} days"


class ExpirationApproaching(Condition):
    """Condition: Expiration is within N days (exit before expiry)."""

    def __init__(self, days_before_expiry: int = 7):
        if days_before_expiry < 0:
            raise ConditionError(f"days_before_expiry must be non-negative")
        self.days_before_expiry = days_before_expiry

    def evaluate(self, context: Dict[str, Any]) -> bool:
        dte = context.get("dte")
        if dte is None:
            # Try to get from position
            position = context.get("position")
            if position and hasattr(position, "min_dte"):
                dte = position.min_dte

        if dte is None:
            return False

        return dte <= self.days_before_expiry

    def describe(self) -> str:
        return f"DTE <= {self.days_before_expiry}"


# =============================================================================
# Condition Factory Functions (for cleaner API)
# =============================================================================


def iv_rank_above(threshold: float) -> IVRankAbove:
    """Create condition: IV rank is above threshold."""
    return IVRankAbove(threshold)


def iv_rank_below(threshold: float) -> IVRankBelow:
    """Create condition: IV rank is below threshold."""
    return IVRankBelow(threshold)


def iv_rank_between(low: float, high: float) -> IVRankBetween:
    """Create condition: IV rank is within range."""
    return IVRankBetween(low, high)


def vix_above(threshold: float) -> VIXAbove:
    """Create condition: VIX is above threshold."""
    return VIXAbove(threshold)


def vix_below(threshold: float) -> VIXBelow:
    """Create condition: VIX is below threshold."""
    return VIXBelow(threshold)


def vix_between(low: float, high: float) -> VIXBetween:
    """Create condition: VIX is within range."""
    return VIXBetween(low, high)


def dte_above(days: int) -> DTEAbove:
    """Create condition: DTE is above threshold."""
    return DTEAbove(days)


def dte_below(days: int) -> DTEBelow:
    """Create condition: DTE is below threshold."""
    return DTEBelow(days)


def dte_between(min_days: int, max_days: int) -> DTEBetween:
    """Create condition: DTE is within range."""
    return DTEBetween(min_days, max_days)


def day_of_week(*days: str) -> DayOfWeek:
    """Create condition: Current day is one of specified weekdays."""
    return DayOfWeek(*days)


def no_open_positions(underlying: Optional[str] = None) -> NoOpenPositions:
    """Create condition: No positions are currently open."""
    return NoOpenPositions(underlying)


def max_open_positions(max_positions: int) -> MaxOpenPositions:
    """Create condition: Open positions below maximum."""
    return MaxOpenPositions(max_positions)


def profit_target(target_pct: float) -> ProfitTarget:
    """Create condition: Profit target reached."""
    return ProfitTarget(target_pct)


def stop_loss(loss_multiple: float) -> StopLoss:
    """Create condition: Stop loss triggered (as multiple of max profit)."""
    return StopLoss(loss_multiple)


def fixed_stop_loss(max_loss: float) -> FixedStopLoss:
    """Create condition: Stop loss triggered (fixed dollar amount)."""
    return FixedStopLoss(max_loss)


def trailing_stop(trail_pct: float) -> TrailingStop:
    """Create condition: Trailing stop triggered."""
    return TrailingStop(trail_pct)


def holding_period(days: int) -> HoldingPeriod:
    """Create condition: Position held for N days."""
    return HoldingPeriod(days)


def expiration_approaching(days_before: int = 7) -> ExpirationApproaching:
    """Create condition: Expiration is within N days."""
    return ExpirationApproaching(days_before)


# =============================================================================
# Structure Factory - Deferred Structure Creation
# =============================================================================


@dataclass
class StructureSpec:
    """
    Specification for creating an option structure.

    This is a deferred factory that creates structures when called
    with market data and option chain.
    """

    structure_type: str
    target_dte: int = 30
    delta_target: Optional[float] = None
    width: Optional[float] = None
    quantity: int = 1
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def create(
        self, underlying: str, spot: float, option_chain: Any, entry_date: datetime
    ) -> OptionStructure:
        """
        Create the actual structure from market data.

        Args:
            underlying: Ticker symbol
            spot: Current spot price
            option_chain: DataFrame or object with option chain data
            entry_date: Entry timestamp

        Returns:
            Created OptionStructure
        """
        # Find closest expiration to target DTE
        expiration = entry_date + timedelta(days=self.target_dte)

        # Select strike(s) based on structure type
        atm_strike = round(spot)  # Simplified ATM selection

        if self.structure_type == "short_straddle":
            return self._create_short_straddle(
                underlying, atm_strike, expiration, spot, option_chain, entry_date
            )
        elif self.structure_type == "long_straddle":
            return self._create_long_straddle(
                underlying, atm_strike, expiration, spot, option_chain, entry_date
            )
        elif self.structure_type == "short_strangle":
            return self._create_short_strangle(
                underlying, atm_strike, expiration, spot, option_chain, entry_date
            )
        elif self.structure_type == "long_strangle":
            return self._create_long_strangle(
                underlying, atm_strike, expiration, spot, option_chain, entry_date
            )
        elif self.structure_type == "iron_condor":
            return self._create_iron_condor(
                underlying, atm_strike, expiration, spot, option_chain, entry_date
            )
        elif self.structure_type == "bull_call_spread":
            return self._create_bull_call_spread(
                underlying, atm_strike, expiration, spot, option_chain, entry_date
            )
        elif self.structure_type == "bear_put_spread":
            return self._create_bear_put_spread(
                underlying, atm_strike, expiration, spot, option_chain, entry_date
            )
        else:
            raise StrategyBuilderError(f"Unknown structure type: {self.structure_type}")

    def _get_option_prices(
        self, option_chain: Any, strike: float, expiration: datetime
    ) -> Tuple[float, float]:
        """Get call and put prices from option chain."""
        # This is a simplified implementation
        # Real implementation would query the option chain
        call_price = self.extra_params.get("call_price", 5.0)
        put_price = self.extra_params.get("put_price", 5.0)
        return call_price, put_price

    def _create_short_straddle(
        self, underlying, strike, expiration, spot, option_chain, entry_date
    ) -> ShortStraddle:
        call_price, put_price = self._get_option_prices(
            option_chain, strike, expiration
        )
        return ShortStraddle.create(
            underlying=underlying,
            strike=strike,
            expiration=expiration,
            call_price=call_price,
            put_price=put_price,
            quantity=self.quantity,
            entry_date=entry_date,
            underlying_price=spot,
        )

    def _create_long_straddle(
        self, underlying, strike, expiration, spot, option_chain, entry_date
    ) -> LongStraddle:
        call_price, put_price = self._get_option_prices(
            option_chain, strike, expiration
        )
        return LongStraddle.create(
            underlying=underlying,
            strike=strike,
            expiration=expiration,
            call_price=call_price,
            put_price=put_price,
            quantity=self.quantity,
            entry_date=entry_date,
            underlying_price=spot,
        )

    def _create_short_strangle(
        self, underlying, atm_strike, expiration, spot, option_chain, entry_date
    ) -> ShortStrangle:
        width = self.width or (atm_strike * 0.05)  # Default 5% OTM
        call_strike = atm_strike + width
        put_strike = atm_strike - width

        call_price = self.extra_params.get("call_price", 3.0)
        put_price = self.extra_params.get("put_price", 3.0)

        return ShortStrangle.create(
            underlying=underlying,
            call_strike=call_strike,
            put_strike=put_strike,
            expiration=expiration,
            call_price=call_price,
            put_price=put_price,
            quantity=self.quantity,
            entry_date=entry_date,
            underlying_price=spot,
        )

    def _create_long_strangle(
        self, underlying, atm_strike, expiration, spot, option_chain, entry_date
    ) -> LongStrangle:
        width = self.width or (atm_strike * 0.05)
        call_strike = atm_strike + width
        put_strike = atm_strike - width

        call_price = self.extra_params.get("call_price", 3.0)
        put_price = self.extra_params.get("put_price", 3.0)

        return LongStrangle.create(
            underlying=underlying,
            call_strike=call_strike,
            put_strike=put_strike,
            expiration=expiration,
            call_price=call_price,
            put_price=put_price,
            quantity=self.quantity,
            entry_date=entry_date,
            underlying_price=spot,
        )

    def _create_iron_condor(
        self, underlying, atm_strike, expiration, spot, option_chain, entry_date
    ) -> IronCondor:
        width = self.width or (atm_strike * 0.05)
        wing_width = self.extra_params.get("wing_width", width * 0.5)

        # Short strikes (inner)
        call_sell = atm_strike + width
        put_sell = atm_strike - width

        # Long strikes (outer)
        call_buy = call_sell + wing_width
        put_buy = put_sell - wing_width

        return IronCondor.create(
            underlying=underlying,
            put_buy_strike=put_buy,
            put_sell_strike=put_sell,
            call_sell_strike=call_sell,
            call_buy_strike=call_buy,
            expiration=expiration,
            put_buy_price=self.extra_params.get("put_buy_price", 1.0),
            put_sell_price=self.extra_params.get("put_sell_price", 2.0),
            call_sell_price=self.extra_params.get("call_sell_price", 2.0),
            call_buy_price=self.extra_params.get("call_buy_price", 1.0),
            quantity=self.quantity,
            entry_date=entry_date,
            underlying_price=spot,
        )

    def _create_bull_call_spread(
        self, underlying, atm_strike, expiration, spot, option_chain, entry_date
    ) -> BullCallSpread:
        width = self.width or (atm_strike * 0.02)
        long_strike = atm_strike
        short_strike = atm_strike + width

        return BullCallSpread.create(
            underlying=underlying,
            long_strike=long_strike,
            short_strike=short_strike,
            expiration=expiration,
            long_price=self.extra_params.get("long_price", 5.0),
            short_price=self.extra_params.get("short_price", 3.0),
            quantity=self.quantity,
            entry_date=entry_date,
            underlying_price=spot,
        )

    def _create_bear_put_spread(
        self, underlying, atm_strike, expiration, spot, option_chain, entry_date
    ) -> BearPutSpread:
        width = self.width or (atm_strike * 0.02)
        # For bear put spread: long_strike > short_strike
        short_strike = atm_strike - width
        long_strike = atm_strike

        return BearPutSpread.create(
            underlying=underlying,
            long_strike=long_strike,
            short_strike=short_strike,
            expiration=expiration,
            long_price=self.extra_params.get("long_price", 5.0),
            short_price=self.extra_params.get("short_price", 3.0),
            quantity=self.quantity,
            entry_date=entry_date,
            underlying_price=spot,
        )

    def describe(self) -> str:
        """Return human-readable description."""
        desc = f"{self.structure_type.replace('_', ' ').title()}"
        desc += f", {self.target_dte} DTE"
        if self.delta_target:
            desc += f", delta={self.delta_target}"
        if self.width:
            desc += f", width={self.width}"
        return desc


# =============================================================================
# Structure Factory Functions
# =============================================================================


def short_straddle(dte: int = 30, quantity: int = 1, **kwargs) -> StructureSpec:
    """Create specification for short straddle."""
    return StructureSpec(
        structure_type="short_straddle",
        target_dte=dte,
        quantity=quantity,
        extra_params=kwargs,
    )


def long_straddle(dte: int = 30, quantity: int = 1, **kwargs) -> StructureSpec:
    """Create specification for long straddle."""
    return StructureSpec(
        structure_type="long_straddle",
        target_dte=dte,
        quantity=quantity,
        extra_params=kwargs,
    )


def short_strangle(
    dte: int = 30, width: Optional[float] = None, quantity: int = 1, **kwargs
) -> StructureSpec:
    """Create specification for short strangle."""
    return StructureSpec(
        structure_type="short_strangle",
        target_dte=dte,
        width=width,
        quantity=quantity,
        extra_params=kwargs,
    )


def long_strangle(
    dte: int = 30, width: Optional[float] = None, quantity: int = 1, **kwargs
) -> StructureSpec:
    """Create specification for long strangle."""
    return StructureSpec(
        structure_type="long_strangle",
        target_dte=dte,
        width=width,
        quantity=quantity,
        extra_params=kwargs,
    )


def iron_condor(
    dte: int = 30,
    width: Optional[float] = None,
    wing_width: Optional[float] = None,
    quantity: int = 1,
    **kwargs,
) -> StructureSpec:
    """Create specification for iron condor."""
    if wing_width:
        kwargs["wing_width"] = wing_width
    return StructureSpec(
        structure_type="iron_condor",
        target_dte=dte,
        width=width,
        quantity=quantity,
        extra_params=kwargs,
    )


def bull_call_spread(
    dte: int = 30, width: Optional[float] = None, quantity: int = 1, **kwargs
) -> StructureSpec:
    """Create specification for bull call spread."""
    return StructureSpec(
        structure_type="bull_call_spread",
        target_dte=dte,
        width=width,
        quantity=quantity,
        extra_params=kwargs,
    )


def bear_put_spread(
    dte: int = 30, width: Optional[float] = None, quantity: int = 1, **kwargs
) -> StructureSpec:
    """Create specification for bear put spread."""
    return StructureSpec(
        structure_type="bear_put_spread",
        target_dte=dte,
        width=width,
        quantity=quantity,
        extra_params=kwargs,
    )


# =============================================================================
# Position Sizing Functions
# =============================================================================


@dataclass
class PositionSizer:
    """
    Determines position size based on capital and risk parameters.
    """

    method: str
    value: float
    max_contracts: int = 100

    def calculate(
        self,
        available_capital: float,
        structure_spec: StructureSpec,
        market_data: Dict[str, Any],
    ) -> int:
        """
        Calculate number of contracts to trade.

        Args:
            available_capital: Available capital for the position
            structure_spec: Structure specification
            market_data: Current market data

        Returns:
            Number of contracts to trade
        """
        if self.method == "fixed":
            contracts = int(self.value)
        elif self.method == "risk_percent":
            # Size based on percentage of capital at risk
            risk_per_contract = market_data.get("estimated_max_loss", 1000)
            risk_capital = available_capital * self.value
            contracts = int(risk_capital / risk_per_contract)
        elif self.method == "capital_percent":
            # Size based on percentage of capital to allocate
            margin_per_contract = market_data.get("margin_per_contract", 2000)
            allocation = available_capital * self.value
            contracts = int(allocation / margin_per_contract)
        elif self.method == "delta_target":
            # Size to achieve target delta
            delta_per_contract = abs(market_data.get("delta_per_contract", 1.0))
            if delta_per_contract > 0:
                contracts = int(abs(self.value) / delta_per_contract)
            else:
                contracts = 1
        elif self.method == "premium_target":
            # Size to collect target premium
            premium_per_contract = market_data.get("premium_per_contract", 100)
            if premium_per_contract > 0:
                contracts = int(self.value / premium_per_contract)
            else:
                contracts = 1
        else:
            contracts = 1

        # Apply limits
        contracts = max(1, min(contracts, self.max_contracts))
        return contracts

    def describe(self) -> str:
        if self.method == "fixed":
            return f"{int(self.value)} contracts"
        elif self.method == "risk_percent":
            return f"risk {self.value:.1%} of capital"
        elif self.method == "capital_percent":
            return f"allocate {self.value:.1%} of capital"
        elif self.method == "delta_target":
            return f"target delta of {self.value}"
        elif self.method == "premium_target":
            return f"target premium of ${self.value:,.2f}"
        return f"{self.method}: {self.value}"


def fixed_contracts(n: int) -> PositionSizer:
    """Size position as fixed number of contracts."""
    return PositionSizer(method="fixed", value=float(n))


def risk_percent(pct: float, max_contracts: int = 100) -> PositionSizer:
    """Size position based on percentage of capital at risk."""
    if not 0 < pct <= 1.0:
        raise StrategyBuilderError(f"risk_percent must be 0-1, got {pct}")
    return PositionSizer(method="risk_percent", value=pct, max_contracts=max_contracts)


def capital_percent(pct: float, max_contracts: int = 100) -> PositionSizer:
    """Size position based on percentage of capital to allocate."""
    if not 0 < pct <= 1.0:
        raise StrategyBuilderError(f"capital_percent must be 0-1, got {pct}")
    return PositionSizer(
        method="capital_percent", value=pct, max_contracts=max_contracts
    )


def delta_target(target_delta: float, max_contracts: int = 100) -> PositionSizer:
    """Size position to achieve target delta."""
    return PositionSizer(
        method="delta_target", value=target_delta, max_contracts=max_contracts
    )


def premium_target(target_premium: float, max_contracts: int = 100) -> PositionSizer:
    """Size position to collect target premium."""
    if target_premium <= 0:
        raise StrategyBuilderError(f"target_premium must be positive")
    return PositionSizer(
        method="premium_target", value=target_premium, max_contracts=max_contracts
    )


# =============================================================================
# Built Strategy Class (Generated by StrategyBuilder)
# =============================================================================


class BuiltStrategy(Strategy):
    """
    Strategy instance created by StrategyBuilder.

    This class wraps the builder configuration and implements
    the Strategy interface using the composed conditions and specs.
    """

    __slots__ = (
        "_underlying",
        "_entry_condition",
        "_exit_condition",
        "_structure_spec",
        "_position_sizer",
        "_builder_config",
    )

    def __init__(
        self,
        name: str,
        description: str,
        initial_capital: float,
        position_limits: Dict[str, Any],
        underlying: Optional[str],
        entry_condition: Condition,
        exit_condition: Condition,
        structure_spec: StructureSpec,
        position_sizer: PositionSizer,
        builder_config: Dict[str, Any],
    ):
        super().__init__(
            name=name,
            description=description,
            initial_capital=initial_capital,
            position_limits=position_limits,
        )
        self._underlying = underlying
        self._entry_condition = entry_condition
        self._exit_condition = exit_condition
        self._structure_spec = structure_spec
        self._position_sizer = position_sizer
        self._builder_config = builder_config

    @property
    def underlying(self) -> Optional[str]:
        """Get configured underlying."""
        return self._underlying

    @property
    def entry_condition(self) -> Condition:
        """Get entry condition."""
        return self._entry_condition

    @property
    def exit_condition(self) -> Condition:
        """Get exit condition."""
        return self._exit_condition

    @property
    def structure_spec(self) -> StructureSpec:
        """Get structure specification."""
        return self._structure_spec

    @property
    def position_sizer(self) -> PositionSizer:
        """Get position sizer."""
        return self._position_sizer

    def should_enter(self, market_data: Dict[str, Any]) -> bool:
        """Check if entry conditions are met."""
        # Build context from market data
        context = dict(market_data)
        context["open_positions"] = self._structures

        # Filter by underlying if specified
        if self._underlying and market_data.get("underlying") != self._underlying:
            return False

        return self._entry_condition.evaluate(context)

    def should_exit(
        self, structure: OptionStructure, market_data: Dict[str, Any]
    ) -> bool:
        """Check if exit conditions are met for a position."""
        # Build context from market data and position
        context = dict(market_data)
        context["position"] = structure
        context["entry_date"] = structure.entry_date

        # Calculate P&L percentage
        try:
            max_profit = getattr(structure, "max_profit", structure.net_premium)
            if max_profit and abs(max_profit) > 1e-10:
                context["pnl_pct"] = structure.calculate_pnl() / abs(max_profit)
            context["pnl"] = structure.calculate_pnl()
        except Exception:
            pass

        return self._exit_condition.evaluate(context)

    def create_structure(
        self,
        current_date: datetime,
        market_data: Dict[str, Any],
        option_chain: Any,
        available_capital: float,
    ) -> OptionStructure:
        """Create the strategy structure."""
        underlying = self._underlying or market_data.get("underlying", "SPY")
        spot = market_data.get("spot", market_data.get("price", 100.0))

        # Calculate position size
        contracts = self._position_sizer.calculate(
            available_capital=available_capital,
            structure_spec=self._structure_spec,
            market_data=market_data,
        )

        # Create structure with calculated size
        spec = StructureSpec(
            structure_type=self._structure_spec.structure_type,
            target_dte=self._structure_spec.target_dte,
            delta_target=self._structure_spec.delta_target,
            width=self._structure_spec.width,
            quantity=contracts,
            extra_params=self._structure_spec.extra_params,
        )

        return spec.create(
            underlying=underlying,
            spot=spot,
            option_chain=option_chain,
            entry_date=current_date,
        )

    def describe(self) -> str:
        """Return human-readable strategy description."""
        parts = [f"Strategy: {self._name}"]
        if self._underlying:
            parts.append(f"Underlying: {self._underlying}")
        parts.append(f"Structure: {self._structure_spec.describe()}")
        parts.append(f"Entry: {self._entry_condition.describe()}")
        parts.append(f"Exit: {self._exit_condition.describe()}")
        parts.append(f"Position Size: {self._position_sizer.describe()}")
        return "\n".join(parts)


# =============================================================================
# Strategy Builder
# =============================================================================


class StrategyBuilder:
    """
    Fluent builder for creating options trading strategies.

    The builder pattern allows declarative strategy creation by chaining
    method calls. Call .build() at the end to create the strategy instance.

    Example:
        >>> strategy = (StrategyBuilder()
        ...     .name("High IV Short Straddle")
        ...     .underlying("SPY")
        ...     .entry_condition(iv_rank_above(70) & dte_between(25, 45))
        ...     .structure(short_straddle(dte=30))
        ...     .exit_condition(
        ...         profit_target(0.50) | stop_loss(2.0) | dte_below(7)
        ...     )
        ...     .position_size(risk_percent(0.02))
        ...     .build())

    Builder Methods:
        .name(str) - Set strategy name
        .description(str) - Set strategy description
        .underlying(str) - Set target underlying
        .initial_capital(float) - Set starting capital
        .entry_condition(Condition) - Set entry condition(s)
        .exit_condition(Condition) - Set exit condition(s)
        .structure(StructureSpec) - Set structure to trade
        .position_size(PositionSizer) - Set position sizing method
        .max_positions(int) - Set maximum concurrent positions
        .max_delta(float) - Set maximum portfolio delta
        .build() - Create the strategy instance
    """

    def __init__(self):
        """Initialize builder with defaults."""
        self._name: Optional[str] = None
        self._description: str = ""
        self._underlying: Optional[str] = None
        self._initial_capital: float = 100000.0
        self._entry_condition: Optional[Condition] = None
        self._exit_condition: Optional[Condition] = None
        self._structure_spec: Optional[StructureSpec] = None
        self._position_sizer: PositionSizer = fixed_contracts(1)
        self._position_limits: Dict[str, Any] = {}

    def name(self, name: str) -> "StrategyBuilder":
        """Set strategy name."""
        if not name or not isinstance(name, str):
            raise BuilderValidationError("name must be a non-empty string")
        self._name = name.strip()
        return self

    def description(self, desc: str) -> "StrategyBuilder":
        """Set strategy description."""
        self._description = desc
        return self

    def underlying(self, symbol: str) -> "StrategyBuilder":
        """Set target underlying ticker."""
        if symbol:
            self._underlying = symbol.upper()
        return self

    def initial_capital(self, amount: float) -> "StrategyBuilder":
        """Set initial capital."""
        if amount <= 0:
            raise BuilderValidationError(
                f"initial_capital must be positive, got {amount}"
            )
        self._initial_capital = float(amount)
        return self

    def entry_condition(self, condition: Condition) -> "StrategyBuilder":
        """Set entry condition(s)."""
        if not isinstance(condition, Condition):
            raise BuilderValidationError(
                f"entry_condition must be a Condition, got {type(condition).__name__}"
            )
        self._entry_condition = condition
        return self

    def exit_condition(self, condition: Condition) -> "StrategyBuilder":
        """Set exit condition(s)."""
        if not isinstance(condition, Condition):
            raise BuilderValidationError(
                f"exit_condition must be a Condition, got {type(condition).__name__}"
            )
        self._exit_condition = condition
        return self

    def structure(self, spec: StructureSpec) -> "StrategyBuilder":
        """Set structure specification."""
        if not isinstance(spec, StructureSpec):
            raise BuilderValidationError(
                f"structure must be a StructureSpec, got {type(spec).__name__}"
            )
        self._structure_spec = spec
        return self

    def position_size(self, sizer: PositionSizer) -> "StrategyBuilder":
        """Set position sizing method."""
        if not isinstance(sizer, PositionSizer):
            raise BuilderValidationError(
                f"position_size must be a PositionSizer, got {type(sizer).__name__}"
            )
        self._position_sizer = sizer
        return self

    def max_positions(self, n: int) -> "StrategyBuilder":
        """Set maximum number of concurrent positions."""
        if n < 1:
            raise BuilderValidationError(f"max_positions must be >= 1, got {n}")
        self._position_limits["max_positions"] = n
        return self

    def max_delta(self, delta: float) -> "StrategyBuilder":
        """Set maximum portfolio delta exposure."""
        if delta <= 0:
            raise BuilderValidationError(f"max_delta must be positive, got {delta}")
        self._position_limits["max_total_delta"] = delta
        return self

    def max_vega(self, vega: float) -> "StrategyBuilder":
        """Set maximum portfolio vega exposure."""
        if vega <= 0:
            raise BuilderValidationError(f"max_vega must be positive, got {vega}")
        self._position_limits["max_total_vega"] = vega
        return self

    def max_capital_utilization(self, pct: float) -> "StrategyBuilder":
        """Set maximum capital utilization."""
        if not 0 < pct <= 1.0:
            raise BuilderValidationError(
                f"max_capital_utilization must be 0-1, got {pct}"
            )
        self._position_limits["max_capital_utilization"] = pct
        return self

    def validate(self) -> List[str]:
        """
        Validate builder configuration.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        if not self._name:
            errors.append("name is required")

        if not self._entry_condition:
            errors.append("entry_condition is required")

        if not self._exit_condition:
            errors.append("exit_condition is required")

        if not self._structure_spec:
            errors.append("structure is required")

        if self._initial_capital <= 0:
            errors.append("initial_capital must be positive")

        return errors

    def build(self) -> BuiltStrategy:
        """
        Build and return the strategy instance.

        Returns:
            BuiltStrategy instance configured from builder

        Raises:
            BuilderValidationError: If validation fails
        """
        errors = self.validate()
        if errors:
            raise BuilderValidationError(
                f"Strategy validation failed: {'; '.join(errors)}"
            )

        assert self._name is not None
        assert self._entry_condition is not None
        assert self._exit_condition is not None
        assert self._structure_spec is not None

        description = self._description
        if not description:
            parts = []
            if self._underlying:
                parts.append(f"{self._underlying}")
            parts.append(self._structure_spec.structure_type.replace("_", " "))
            parts.append(f"entry: {self._entry_condition.describe()}")
            description = ", ".join(parts)

        config = {
            "name": self._name,
            "underlying": self._underlying,
            "initial_capital": self._initial_capital,
            "entry_condition": self._entry_condition.describe(),
            "exit_condition": self._exit_condition.describe(),
            "structure": self._structure_spec.describe(),
            "position_sizer": self._position_sizer.describe(),
            "position_limits": self._position_limits,
        }

        logger.info(f"Building strategy: {self._name}")

        return BuiltStrategy(
            name=self._name,
            description=description,
            initial_capital=self._initial_capital,
            position_limits=self._position_limits,
            underlying=self._underlying,
            entry_condition=self._entry_condition,
            exit_condition=self._exit_condition,
            structure_spec=self._structure_spec,
            position_sizer=self._position_sizer,
            builder_config=config,
        )


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Builder
    "StrategyBuilder",
    "BuiltStrategy",
    # Conditions Base
    "Condition",
    "AndCondition",
    "OrCondition",
    "NotCondition",
    "AlwaysTrue",
    "AlwaysFalse",
    # Entry Condition Classes
    "IVRankAbove",
    "IVRankBelow",
    "IVRankBetween",
    "VIXAbove",
    "VIXBelow",
    "VIXBetween",
    "DTEAbove",
    "DTEBelow",
    "DTEBetween",
    "DayOfWeek",
    "NoOpenPositions",
    "MaxOpenPositions",
    # Exit Condition Classes
    "ProfitTarget",
    "StopLoss",
    "FixedStopLoss",
    "TrailingStop",
    "HoldingPeriod",
    "ExpirationApproaching",
    # Condition Factory Functions
    "iv_rank_above",
    "iv_rank_below",
    "iv_rank_between",
    "vix_above",
    "vix_below",
    "vix_between",
    "dte_above",
    "dte_below",
    "dte_between",
    "day_of_week",
    "no_open_positions",
    "max_open_positions",
    "profit_target",
    "stop_loss",
    "fixed_stop_loss",
    "trailing_stop",
    "holding_period",
    "expiration_approaching",
    # Structure Specification
    "StructureSpec",
    # Structure Factory Functions
    "short_straddle",
    "long_straddle",
    "short_strangle",
    "long_strangle",
    "iron_condor",
    "bull_call_spread",
    "bear_put_spread",
    # Position Sizing
    "PositionSizer",
    "fixed_contracts",
    "risk_percent",
    "capital_percent",
    "delta_target",
    "premium_target",
    # Exceptions
    "StrategyBuilderError",
    "BuilderValidationError",
    "ConditionError",
]
