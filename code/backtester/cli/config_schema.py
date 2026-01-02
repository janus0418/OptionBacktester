"""
Configuration Schema for Strategy Definitions

Defines the schema for YAML/JSON strategy configuration files,
including validation logic and type coercion.
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""

    def __init__(self, message: str, errors: Optional[List[str]] = None):
        self.errors = errors or []
        super().__init__(message)


class StructureType(str, Enum):
    """Supported option structure types."""

    SHORT_STRADDLE = "short_straddle"
    LONG_STRADDLE = "long_straddle"
    SHORT_STRANGLE = "short_strangle"
    LONG_STRANGLE = "long_strangle"
    IRON_CONDOR = "iron_condor"
    BULL_CALL_SPREAD = "bull_call_spread"
    BEAR_PUT_SPREAD = "bear_put_spread"
    BULL_PUT_SPREAD = "bull_put_spread"
    BEAR_CALL_SPREAD = "bear_call_spread"


class PositionSizeMethod(str, Enum):
    """Position sizing methods."""

    FIXED_CONTRACTS = "fixed_contracts"
    RISK_PERCENT = "risk_percent"
    CAPITAL_PERCENT = "capital_percent"
    DELTA_TARGET = "delta_target"
    PREMIUM_TARGET = "premium_target"


class DataSourceType(str, Enum):
    """Supported data source types."""

    DOLT = "dolt"
    CSV = "csv"
    MOCK = "mock"


@dataclass
class EntryConditionConfig:
    """Configuration for entry conditions."""

    type: str
    params: Dict[str, Any] = field(default_factory=dict)

    # Compound conditions
    and_conditions: Optional[List["EntryConditionConfig"]] = None
    or_conditions: Optional[List["EntryConditionConfig"]] = None
    negate: bool = False


@dataclass
class ExitConditionConfig:
    """Configuration for exit conditions."""

    profit_target: Optional[float] = None  # e.g., 0.50 = 50% of premium
    stop_loss: Optional[float] = None  # e.g., 2.0 = 200% of premium
    fixed_stop_loss: Optional[float] = None  # Absolute dollar amount
    trailing_stop: Optional[float] = None  # e.g., 0.20 = 20% trailing
    max_holding_days: Optional[int] = None
    min_dte: Optional[int] = None  # Exit when DTE falls below


@dataclass
class StructureConfig:
    """Configuration for option structure."""

    type: StructureType
    dte: int = 30  # Target days to expiration
    delta: Optional[float] = None  # For delta-based strike selection
    width: Optional[float] = None  # For spreads/condors (in dollars or delta)
    quantity: int = 1


@dataclass
class PositionSizeConfig:
    """Configuration for position sizing."""

    method: PositionSizeMethod = PositionSizeMethod.FIXED_CONTRACTS
    value: float = 1.0  # Contracts, percent, or target value


@dataclass
class RiskLimitsConfig:
    """Risk management configuration."""

    max_positions: int = 5
    max_daily_loss: Optional[float] = None  # Dollar amount
    max_daily_loss_pct: Optional[float] = None  # Percent of capital
    max_drawdown: Optional[float] = None  # Percent
    max_portfolio_delta: Optional[float] = None
    max_portfolio_vega: Optional[float] = None
    max_capital_utilization: float = 0.80


@dataclass
class BacktestConfig:
    """Backtest execution configuration."""

    start_date: Union[str, date, datetime]
    end_date: Union[str, date, datetime]
    initial_capital: float = 100000.0
    commission_per_contract: float = 0.65
    slippage_pct: float = 0.01


@dataclass
class DataSourceConfig:
    """Data source configuration."""

    type: DataSourceType = DataSourceType.DOLT
    database: Optional[str] = None
    directory: Optional[str] = None
    host: Optional[str] = None
    port: Optional[int] = None


@dataclass
class StrategyConfig:
    """Complete strategy configuration."""

    name: str
    version: str = "1.0"
    description: Optional[str] = None

    # Core strategy settings
    underlying: str = "SPY"
    structure: Optional[StructureConfig] = None
    entry_conditions: List[EntryConditionConfig] = field(default_factory=list)
    exit_conditions: Optional[ExitConditionConfig] = None
    position_size: Optional[PositionSizeConfig] = None

    # Risk management
    risk_limits: Optional[RiskLimitsConfig] = None

    # Backtest settings
    backtest: Optional[BacktestConfig] = None

    # Data source
    data_source: Optional[DataSourceConfig] = None

    # Template reference (if using a template)
    template: Optional[str] = None
    template_params: Dict[str, Any] = field(default_factory=dict)


class ConfigValidator:
    """Validates strategy configuration."""

    VALID_ENTRY_CONDITIONS = {
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
        "always_true",
        "always_false",
    }

    @classmethod
    def validate(cls, config: StrategyConfig) -> List[str]:
        """
        Validate a strategy configuration.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Required fields
        if not config.name:
            errors.append("Strategy name is required")

        if not config.underlying:
            errors.append("Underlying symbol is required")

        # Validate structure if present
        if config.structure:
            errors.extend(cls._validate_structure(config.structure))

        # Validate entry conditions
        for i, cond in enumerate(config.entry_conditions):
            cond_errors = cls._validate_entry_condition(cond, i)
            errors.extend(cond_errors)

        # Validate exit conditions
        if config.exit_conditions:
            errors.extend(cls._validate_exit_conditions(config.exit_conditions))

        # Validate position sizing
        if config.position_size:
            errors.extend(cls._validate_position_size(config.position_size))

        # Validate risk limits
        if config.risk_limits:
            errors.extend(cls._validate_risk_limits(config.risk_limits))

        # Validate backtest config
        if config.backtest:
            errors.extend(cls._validate_backtest(config.backtest))

        # Template validation
        if config.template and config.structure:
            errors.append(
                "Cannot specify both 'template' and 'structure' - use one or the other"
            )

        return errors

    @classmethod
    def _validate_structure(cls, structure: StructureConfig) -> List[str]:
        """Validate structure configuration."""
        errors = []

        if structure.dte <= 0:
            errors.append(f"Structure DTE must be positive, got {structure.dte}")

        if structure.dte > 365:
            errors.append(
                f"Structure DTE {structure.dte} seems unreasonably large (>365)"
            )

        if structure.delta is not None and not (0 < abs(structure.delta) <= 1):
            errors.append(f"Delta must be between 0 and 1, got {structure.delta}")

        if structure.quantity < 1:
            errors.append(f"Quantity must be at least 1, got {structure.quantity}")

        # Structure-specific validation
        if structure.type in (StructureType.IRON_CONDOR,):
            if structure.width is None:
                errors.append(f"{structure.type.value} requires 'width' parameter")

        return errors

    @classmethod
    def _validate_entry_condition(
        cls, condition: EntryConditionConfig, index: int
    ) -> List[str]:
        """Validate entry condition configuration."""
        errors = []
        prefix = f"Entry condition [{index}]"

        if condition.type not in cls.VALID_ENTRY_CONDITIONS:
            errors.append(f"{prefix}: Unknown condition type '{condition.type}'")
            return errors

        # Type-specific parameter validation
        params = condition.params

        if condition.type in ("iv_rank_above", "iv_rank_below"):
            if "threshold" not in params:
                errors.append(
                    f"{prefix}: {condition.type} requires 'threshold' parameter"
                )
            elif not (0 <= params["threshold"] <= 100):
                errors.append(f"{prefix}: IV rank threshold must be 0-100")

        elif condition.type == "iv_rank_between":
            if "low" not in params or "high" not in params:
                errors.append(
                    f"{prefix}: iv_rank_between requires 'low' and 'high' parameters"
                )
            elif params.get("low", 0) >= params.get("high", 100):
                errors.append(f"{prefix}: 'low' must be less than 'high'")

        elif condition.type in ("vix_above", "vix_below"):
            if "level" not in params:
                errors.append(f"{prefix}: {condition.type} requires 'level' parameter")
            elif params["level"] < 0:
                errors.append(f"{prefix}: VIX level must be positive")

        elif condition.type == "day_of_week":
            if "days" not in params:
                errors.append(f"{prefix}: day_of_week requires 'days' parameter")
            else:
                valid_days = {0, 1, 2, 3, 4, 5, 6}
                for day in params["days"]:
                    if day not in valid_days:
                        errors.append(f"{prefix}: Invalid day {day}, must be 0-6")

        return errors

    @classmethod
    def _validate_exit_conditions(cls, exit_config: ExitConditionConfig) -> List[str]:
        """Validate exit condition configuration."""
        errors = []

        if exit_config.profit_target is not None:
            if exit_config.profit_target <= 0:
                errors.append("Profit target must be positive")
            elif exit_config.profit_target > 1:
                errors.append(
                    "Profit target > 1 (100%) is unusual - verify this is intentional"
                )

        if exit_config.stop_loss is not None:
            if exit_config.stop_loss <= 0:
                errors.append("Stop loss multiplier must be positive")

        if exit_config.trailing_stop is not None:
            if not (0 < exit_config.trailing_stop < 1):
                errors.append(
                    "Trailing stop must be between 0 and 1 (e.g., 0.20 for 20%)"
                )

        if (
            exit_config.max_holding_days is not None
            and exit_config.max_holding_days < 1
        ):
            errors.append("Max holding days must be at least 1")

        if exit_config.min_dte is not None and exit_config.min_dte < 0:
            errors.append("Min DTE cannot be negative")

        return errors

    @classmethod
    def _validate_position_size(cls, size_config: PositionSizeConfig) -> List[str]:
        """Validate position sizing configuration."""
        errors = []

        if size_config.value <= 0:
            errors.append("Position size value must be positive")

        if size_config.method == PositionSizeMethod.RISK_PERCENT:
            if size_config.value > 0.10:
                errors.append(
                    f"Risk percent {size_config.value:.1%} is very high (>10%)"
                )

        if size_config.method == PositionSizeMethod.CAPITAL_PERCENT:
            if size_config.value > 0.50:
                errors.append(f"Capital percent {size_config.value:.1%} exceeds 50%")

        return errors

    @classmethod
    def _validate_risk_limits(cls, limits: RiskLimitsConfig) -> List[str]:
        """Validate risk limits configuration."""
        errors = []

        if limits.max_positions < 1:
            errors.append("Max positions must be at least 1")

        if limits.max_drawdown is not None and limits.max_drawdown > 0.50:
            errors.append(f"Max drawdown {limits.max_drawdown:.1%} is very high (>50%)")

        if limits.max_capital_utilization <= 0 or limits.max_capital_utilization > 1:
            errors.append("Max capital utilization must be between 0 and 1")

        return errors

    @classmethod
    def _validate_backtest(cls, backtest: BacktestConfig) -> List[str]:
        """Validate backtest configuration."""
        errors = []

        # Parse dates if strings
        start = cls._parse_date(backtest.start_date)
        end = cls._parse_date(backtest.end_date)

        if start is None:
            errors.append(f"Invalid start_date format: {backtest.start_date}")
        if end is None:
            errors.append(f"Invalid end_date format: {backtest.end_date}")

        if start and end and start >= end:
            errors.append("start_date must be before end_date")

        if backtest.initial_capital <= 0:
            errors.append("Initial capital must be positive")

        if backtest.commission_per_contract < 0:
            errors.append("Commission cannot be negative")

        if backtest.slippage_pct < 0:
            errors.append("Slippage percentage cannot be negative")

        return errors

    @staticmethod
    def _parse_date(date_value: Union[str, date, datetime]) -> Optional[date]:
        """Parse various date formats."""
        if isinstance(date_value, datetime):
            return date_value.date()
        if isinstance(date_value, date):
            return date_value
        if isinstance(date_value, str):
            for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%d-%m-%Y"):
                try:
                    return datetime.strptime(date_value, fmt).date()
                except ValueError:
                    continue
        return None


def validate_config(config: StrategyConfig) -> None:
    """
    Validate configuration and raise exception if invalid.

    Args:
        config: Strategy configuration to validate

    Raises:
        ConfigValidationError: If validation fails
    """
    errors = ConfigValidator.validate(config)
    if errors:
        raise ConfigValidationError(
            f"Configuration validation failed with {len(errors)} error(s)",
            errors=errors,
        )
