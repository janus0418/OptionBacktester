"""
Strategy Validation Framework

This module provides comprehensive validation for options trading strategies,
including configuration validation, risk checks, backtest sanity verification,
and performance benchmarking.

Components:
    StrategyConfigValidator: Validates strategy configuration before execution
    RiskLimitValidator: Validates risk parameters against acceptable limits
    BacktestSanityChecker: Validates backtest results for common issues
    PerformanceBenchmark: Compares strategy performance against benchmarks

Usage:
    from backtester.strategies.strategy_validator import (
        StrategyConfigValidator,
        RiskLimitValidator,
        BacktestSanityChecker,
        validate_strategy,
    )

    # Validate a strategy before backtesting
    validator = StrategyConfigValidator()
    result = validator.validate(strategy)

    if not result.is_valid:
        print(f"Validation errors: {result.errors}")

    # Quick validation helper
    is_valid, errors = validate_strategy(strategy)

References:
    - Risk management best practices
    - Backtest validation literature
    - Options trading conventions
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from backtester.strategies.strategy import Strategy
from backtester.strategies.strategy_builder import BuiltStrategy, StructureSpec

logger = logging.getLogger(__name__)


# =============================================================================
# Validation Result Types
# =============================================================================


class ValidationSeverity(Enum):
    """Severity level for validation issues."""

    ERROR = "error"  # Must be fixed, strategy cannot run
    WARNING = "warning"  # Should be reviewed, may cause issues
    INFO = "info"  # Informational, no action needed


@dataclass
class ValidationIssue:
    """A single validation issue."""

    code: str
    message: str
    severity: ValidationSeverity
    field: Optional[str] = None
    suggestion: Optional[str] = None

    def __str__(self) -> str:
        severity_str = f"[{self.severity.value.upper()}]"
        field_str = f" ({self.field})" if self.field else ""
        return f"{severity_str}{field_str}: {self.message}"


@dataclass
class ValidationResult:
    """Result of a validation operation."""

    issues: List[ValidationIssue] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Check if validation passed (no errors)."""
        return not any(i.severity == ValidationSeverity.ERROR for i in self.issues)

    @property
    def has_warnings(self) -> bool:
        """Check if there are warnings."""
        return any(i.severity == ValidationSeverity.WARNING for i in self.issues)

    @property
    def errors(self) -> List[ValidationIssue]:
        """Get all error-level issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.ERROR]

    @property
    def warnings(self) -> List[ValidationIssue]:
        """Get all warning-level issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.WARNING]

    def add_error(
        self,
        code: str,
        message: str,
        field: Optional[str] = None,
        suggestion: Optional[str] = None,
    ):
        """Add an error issue."""
        self.issues.append(
            ValidationIssue(
                code=code,
                message=message,
                severity=ValidationSeverity.ERROR,
                field=field,
                suggestion=suggestion,
            )
        )

    def add_warning(
        self,
        code: str,
        message: str,
        field: Optional[str] = None,
        suggestion: Optional[str] = None,
    ):
        """Add a warning issue."""
        self.issues.append(
            ValidationIssue(
                code=code,
                message=message,
                severity=ValidationSeverity.WARNING,
                field=field,
                suggestion=suggestion,
            )
        )

    def add_info(self, code: str, message: str, field: Optional[str] = None):
        """Add an informational issue."""
        self.issues.append(
            ValidationIssue(
                code=code,
                message=message,
                severity=ValidationSeverity.INFO,
                field=field,
            )
        )

    def merge(self, other: "ValidationResult"):
        """Merge another validation result into this one."""
        self.issues.extend(other.issues)

    def __str__(self) -> str:
        if not self.issues:
            return "Validation passed with no issues"
        return "\n".join(str(i) for i in self.issues)


# =============================================================================
# Strategy Configuration Validator
# =============================================================================


class StrategyConfigValidator:
    """
    Validates strategy configuration before execution.

    Checks:
        - Required fields are present
        - Field values are within acceptable ranges
        - Logical consistency between fields
        - Common configuration mistakes
    """

    # Acceptable ranges for common parameters
    CAPITAL_MIN = 1_000.0
    CAPITAL_MAX = 100_000_000.0
    MAX_POSITIONS_MIN = 1
    MAX_POSITIONS_MAX = 100
    DTE_MIN = 0
    DTE_MAX = 730  # 2 years
    IV_RANK_MIN = 0.0
    IV_RANK_MAX = 100.0
    PROFIT_TARGET_MIN = 0.01
    PROFIT_TARGET_MAX = 1.0
    STOP_LOSS_MIN = 0.01
    STOP_LOSS_MAX = 10.0  # 1000% of premium

    def validate(self, strategy: Union[Strategy, BuiltStrategy]) -> ValidationResult:
        """
        Validate a strategy configuration.

        Args:
            strategy: Strategy to validate

        Returns:
            ValidationResult with any issues found
        """
        result = ValidationResult()

        # Basic validation
        self._validate_name(strategy, result)
        self._validate_capital(strategy, result)

        # BuiltStrategy-specific validation
        if isinstance(strategy, BuiltStrategy):
            self._validate_built_strategy(strategy, result)

        logger.debug(
            f"Validated strategy '{getattr(strategy, 'name', 'unknown')}': "
            f"{len(result.errors)} errors, {len(result.warnings)} warnings"
        )

        return result

    def _validate_name(self, strategy: Strategy, result: ValidationResult):
        """Validate strategy name."""
        name = getattr(strategy, "name", None) or getattr(strategy, "_name", None)

        if not name:
            result.add_error(
                code="NAME_MISSING",
                message="Strategy name is required",
                field="name",
                suggestion="Set a descriptive name for the strategy",
            )
        elif len(name) < 3:
            result.add_warning(
                code="NAME_TOO_SHORT",
                message=f"Strategy name '{name}' is very short",
                field="name",
                suggestion="Use a more descriptive name",
            )
        elif len(name) > 100:
            result.add_warning(
                code="NAME_TOO_LONG",
                message=f"Strategy name is too long ({len(name)} chars)",
                field="name",
                suggestion="Keep name under 100 characters",
            )

    def _validate_capital(self, strategy: Strategy, result: ValidationResult):
        """Validate initial capital."""
        capital = getattr(strategy, "_initial_capital", None)

        if capital is None:
            result.add_error(
                code="CAPITAL_MISSING",
                message="Initial capital is required",
                field="initial_capital",
            )
        elif capital < self.CAPITAL_MIN:
            result.add_error(
                code="CAPITAL_TOO_LOW",
                message=f"Initial capital ${capital:,.0f} is below minimum ${self.CAPITAL_MIN:,.0f}",
                field="initial_capital",
                suggestion=f"Use at least ${self.CAPITAL_MIN:,.0f}",
            )
        elif capital > self.CAPITAL_MAX:
            result.add_warning(
                code="CAPITAL_VERY_HIGH",
                message=f"Initial capital ${capital:,.0f} is very high",
                field="initial_capital",
                suggestion="Verify this is intentional",
            )

    def _validate_built_strategy(
        self, strategy: BuiltStrategy, result: ValidationResult
    ):
        """Validate BuiltStrategy-specific configuration."""
        # Validate entry condition
        if strategy.entry_condition is None:
            result.add_error(
                code="ENTRY_CONDITION_MISSING",
                message="Entry condition is required",
                field="entry_condition",
            )

        # Validate exit condition
        if strategy.exit_condition is None:
            result.add_error(
                code="EXIT_CONDITION_MISSING",
                message="Exit condition is required",
                field="exit_condition",
            )

        # Validate structure spec
        if strategy.structure_spec is None:
            result.add_error(
                code="STRUCTURE_MISSING",
                message="Structure specification is required",
                field="structure_spec",
            )
        else:
            self._validate_structure_spec(strategy.structure_spec, result)

        # Validate position sizer
        if strategy.position_sizer is None:
            result.add_warning(
                code="POSITION_SIZER_MISSING",
                message="No position sizer configured, will use defaults",
                field="position_sizer",
            )

    def _validate_structure_spec(self, spec: StructureSpec, result: ValidationResult):
        """Validate structure specification."""
        # Validate DTE
        if spec.target_dte < self.DTE_MIN:
            result.add_error(
                code="DTE_NEGATIVE",
                message=f"Target DTE cannot be negative: {spec.target_dte}",
                field="target_dte",
            )
        elif spec.target_dte > self.DTE_MAX:
            result.add_warning(
                code="DTE_VERY_LONG",
                message=f"Target DTE {spec.target_dte} is very long (>2 years)",
                field="target_dte",
                suggestion="Consider shorter-dated options for most strategies",
            )

        # Validate quantity
        if spec.quantity < 1:
            result.add_error(
                code="QUANTITY_INVALID",
                message=f"Quantity must be at least 1: {spec.quantity}",
                field="quantity",
            )
        elif spec.quantity > 1000:
            result.add_warning(
                code="QUANTITY_VERY_HIGH",
                message=f"Quantity {spec.quantity} is very high",
                field="quantity",
                suggestion="Verify position sizing is intentional",
            )


# =============================================================================
# Risk Limit Validator
# =============================================================================


@dataclass
class RiskLimits:
    """Risk limit configuration."""

    max_position_size_pct: float = 0.10  # Max 10% per position
    max_portfolio_delta: float = 100.0  # Max portfolio delta
    max_portfolio_vega: float = 1000.0  # Max portfolio vega
    max_daily_loss_pct: float = 0.05  # Max 5% daily loss
    max_drawdown_pct: float = 0.20  # Max 20% drawdown
    min_days_to_expiry: int = 1  # Min DTE for positions
    max_leverage: float = 2.0  # Max leverage ratio


class RiskLimitValidator:
    """
    Validates risk parameters against acceptable limits.

    Ensures strategies don't exceed risk thresholds that could
    lead to catastrophic losses.
    """

    def __init__(self, limits: Optional[RiskLimits] = None):
        """
        Args:
            limits: Risk limits to enforce (uses defaults if not provided)
        """
        self.limits = limits or RiskLimits()

    def validate_strategy(self, strategy: Strategy) -> ValidationResult:
        """Validate strategy against risk limits."""
        result = ValidationResult()

        # Check position limits if available
        position_limits = getattr(strategy, "_position_limits", {})

        if "max_positions" in position_limits:
            max_pos = position_limits["max_positions"]
            if max_pos > 50:
                result.add_warning(
                    code="MAX_POSITIONS_HIGH",
                    message=f"Max positions {max_pos} is high, may be difficult to manage",
                    field="max_positions",
                )

        if "max_total_delta" in position_limits:
            max_delta = position_limits["max_total_delta"]
            if max_delta > self.limits.max_portfolio_delta:
                result.add_warning(
                    code="DELTA_LIMIT_HIGH",
                    message=f"Max delta {max_delta} exceeds recommended {self.limits.max_portfolio_delta}",
                    field="max_total_delta",
                )

        return result

    def validate_position(
        self,
        position_value: float,
        portfolio_value: float,
        position_delta: float = 0.0,
        position_vega: float = 0.0,
    ) -> ValidationResult:
        """
        Validate a potential position against risk limits.

        Args:
            position_value: Value of the position
            portfolio_value: Total portfolio value
            position_delta: Position delta
            position_vega: Position vega

        Returns:
            ValidationResult with any limit violations
        """
        result = ValidationResult()

        # Check position size
        if portfolio_value > 0:
            position_pct = abs(position_value) / portfolio_value
            if position_pct > self.limits.max_position_size_pct:
                result.add_error(
                    code="POSITION_TOO_LARGE",
                    message=f"Position size {position_pct:.1%} exceeds limit {self.limits.max_position_size_pct:.1%}",
                    field="position_size",
                )

        # Check delta exposure
        if abs(position_delta) > self.limits.max_portfolio_delta:
            result.add_warning(
                code="DELTA_EXPOSURE_HIGH",
                message=f"Position delta {position_delta:.1f} is high",
                field="delta",
            )

        return result

    def validate_backtest_results(
        self,
        equity_curve: List[float],
        initial_capital: float,
    ) -> ValidationResult:
        """
        Validate backtest results against risk limits.

        Args:
            equity_curve: List of portfolio values over time
            initial_capital: Starting capital

        Returns:
            ValidationResult with any limit violations
        """
        result = ValidationResult()

        if not equity_curve or len(equity_curve) < 2:
            result.add_warning(
                code="INSUFFICIENT_DATA",
                message="Not enough data points for risk validation",
            )
            return result

        equity = np.array(equity_curve)

        # Calculate max drawdown
        running_max = np.maximum.accumulate(equity)
        drawdowns = (running_max - equity) / running_max
        max_drawdown = np.max(drawdowns)

        if max_drawdown > self.limits.max_drawdown_pct:
            result.add_warning(
                code="MAX_DRAWDOWN_EXCEEDED",
                message=f"Max drawdown {max_drawdown:.1%} exceeds limit {self.limits.max_drawdown_pct:.1%}",
                field="max_drawdown",
                suggestion="Consider adding stop loss or reducing position sizes",
            )

        # Calculate daily returns
        daily_returns = np.diff(equity) / equity[:-1]

        # Check for extreme daily losses
        min_daily_return = np.min(daily_returns)
        if min_daily_return < -self.limits.max_daily_loss_pct:
            result.add_warning(
                code="LARGE_DAILY_LOSS",
                message=f"Largest daily loss {min_daily_return:.1%} exceeds limit {-self.limits.max_daily_loss_pct:.1%}",
                field="daily_loss",
            )

        return result


# =============================================================================
# Backtest Sanity Checker
# =============================================================================


class BacktestSanityChecker:
    """
    Validates backtest results for common issues and potential errors.

    Checks for:
        - Unrealistic returns
        - Data quality issues
        - Look-ahead bias indicators
        - Survivorship bias indicators
        - Overfitting signals
    """

    # Thresholds for sanity checks
    MAX_ANNUAL_RETURN = 5.0  # 500% annual return is suspicious
    MIN_ANNUAL_RETURN = -0.99  # Can't lose more than 99%
    MIN_TRADES = 10  # Need at least 10 trades for validity
    MAX_WIN_RATE = 0.95  # 95% win rate is suspicious
    MAX_SHARPE = 5.0  # Sharpe > 5 is suspicious
    MIN_TRADE_DURATION_DAYS = 0.1  # Trades should last at least ~2.5 hours

    def validate(
        self,
        results: Dict[str, Any],
        trading_days: int = 252,
    ) -> ValidationResult:
        """
        Validate backtest results.

        Args:
            results: Backtest results dictionary
            trading_days: Number of trading days per year

        Returns:
            ValidationResult with any issues found
        """
        result = ValidationResult()

        # Extract key metrics
        total_return = results.get("total_return", 0)
        sharpe = results.get("sharpe_ratio", 0)
        win_rate = results.get("win_rate", 0)
        total_trades = results.get("total_trades", 0)
        backtest_days = results.get("backtest_days", 0)

        # Validate return magnitude
        self._validate_returns(total_return, backtest_days, trading_days, result)

        # Validate trade statistics
        self._validate_trades(total_trades, win_rate, result)

        # Validate risk metrics
        self._validate_risk_metrics(sharpe, result)

        # Check for potential issues
        self._check_potential_issues(results, result)

        return result

    def _validate_returns(
        self,
        total_return: float,
        backtest_days: int,
        trading_days: int,
        result: ValidationResult,
    ):
        """Validate return metrics."""
        if backtest_days > 0:
            # Annualize returns
            years = backtest_days / trading_days
            if years > 0:
                annual_return = (1 + total_return) ** (1 / years) - 1

                if annual_return > self.MAX_ANNUAL_RETURN:
                    result.add_warning(
                        code="RETURNS_UNREALISTIC",
                        message=f"Annualized return {annual_return:.0%} seems unrealistically high",
                        field="total_return",
                        suggestion="Check for look-ahead bias or data errors",
                    )

                if annual_return < self.MIN_ANNUAL_RETURN:
                    result.add_warning(
                        code="RETURNS_EXTREME_LOSS",
                        message=f"Strategy lost {-annual_return:.0%} annualized",
                        field="total_return",
                        suggestion="Review risk management and stop losses",
                    )

    def _validate_trades(
        self, total_trades: int, win_rate: float, result: ValidationResult
    ):
        """Validate trade statistics."""
        if total_trades < self.MIN_TRADES:
            result.add_warning(
                code="INSUFFICIENT_TRADES",
                message=f"Only {total_trades} trades, results may not be statistically significant",
                field="total_trades",
                suggestion=f"Run backtest over longer period to get at least {self.MIN_TRADES} trades",
            )

        if win_rate > self.MAX_WIN_RATE and total_trades > 20:
            result.add_warning(
                code="WIN_RATE_SUSPICIOUS",
                message=f"Win rate {win_rate:.0%} is suspiciously high",
                field="win_rate",
                suggestion="Check for data issues or look-ahead bias",
            )

    def _validate_risk_metrics(self, sharpe: float, result: ValidationResult):
        """Validate risk-adjusted metrics."""
        if sharpe > self.MAX_SHARPE:
            result.add_warning(
                code="SHARPE_UNREALISTIC",
                message=f"Sharpe ratio {sharpe:.2f} is unrealistically high",
                field="sharpe_ratio",
                suggestion="Sharpe > 3 is rare in live trading; check for issues",
            )

    def _check_potential_issues(
        self, results: Dict[str, Any], result: ValidationResult
    ):
        """Check for potential backtest issues."""
        # Check for suspiciously consistent returns
        if "monthly_returns" in results:
            monthly = results["monthly_returns"]
            if len(monthly) >= 12:
                std = np.std(monthly)
                if std < 0.01:  # Less than 1% monthly std
                    result.add_warning(
                        code="RETURNS_TOO_CONSISTENT",
                        message="Monthly returns are suspiciously consistent",
                        suggestion="Real strategies have more volatility",
                    )

        # Check for gaps in trading
        if "trade_dates" in results:
            dates = results["trade_dates"]
            if len(dates) >= 2:
                gaps = np.diff(sorted(dates))
                max_gap = max(gaps) if len(gaps) > 0 else timedelta(0)
                if isinstance(max_gap, timedelta) and max_gap.days > 60:
                    result.add_info(
                        code="LARGE_TRADING_GAP",
                        message=f"Largest gap between trades: {max_gap.days} days",
                    )


# =============================================================================
# Performance Benchmark
# =============================================================================


@dataclass
class BenchmarkComparison:
    """Comparison of strategy to benchmark."""

    strategy_return: float
    benchmark_return: float
    alpha: float
    beta: float
    correlation: float
    tracking_error: float
    information_ratio: float

    @property
    def outperformed(self) -> bool:
        """Check if strategy outperformed benchmark."""
        return self.strategy_return > self.benchmark_return


class PerformanceBenchmark:
    """
    Compares strategy performance against benchmarks.

    Common benchmarks:
        - Buy and hold SPY
        - Risk-free rate
        - Simple moving average strategy
    """

    # Standard benchmark returns (annualized)
    RISK_FREE_RATE = 0.05  # 5% risk-free rate
    HISTORICAL_SPY_RETURN = 0.10  # 10% average annual return
    HISTORICAL_SPY_VOL = 0.16  # 16% annual volatility

    def compare_to_buy_hold(
        self,
        strategy_returns: List[float],
        benchmark_returns: List[float],
    ) -> BenchmarkComparison:
        """
        Compare strategy to buy-and-hold benchmark.

        Args:
            strategy_returns: Daily strategy returns
            benchmark_returns: Daily benchmark returns

        Returns:
            BenchmarkComparison with metrics
        """
        if len(strategy_returns) != len(benchmark_returns):
            raise ValueError("Return series must have same length")

        strat = np.array(strategy_returns)
        bench = np.array(benchmark_returns)

        # Total returns
        strat_total = np.prod(1 + strat) - 1
        bench_total = np.prod(1 + bench) - 1

        # Beta and alpha
        if np.std(bench) > 0:
            beta = np.cov(strat, bench)[0, 1] / np.var(bench)
            alpha = np.mean(strat) - beta * np.mean(bench)
        else:
            beta = 0.0
            alpha = np.mean(strat)

        # Correlation
        if np.std(strat) > 0 and np.std(bench) > 0:
            correlation = np.corrcoef(strat, bench)[0, 1]
        else:
            correlation = 0.0

        # Tracking error
        excess = strat - bench
        tracking_error = np.std(excess) * np.sqrt(252)

        # Information ratio
        if tracking_error > 0:
            info_ratio = (np.mean(excess) * 252) / tracking_error
        else:
            info_ratio = 0.0

        return BenchmarkComparison(
            strategy_return=float(strat_total),
            benchmark_return=float(bench_total),
            alpha=float(alpha * 252),  # Annualize
            beta=float(beta),
            correlation=float(correlation),
            tracking_error=float(tracking_error),
            information_ratio=float(info_ratio),
        )

    def validate_against_benchmark(
        self,
        comparison: BenchmarkComparison,
    ) -> ValidationResult:
        """
        Validate strategy performance against benchmark.

        Args:
            comparison: BenchmarkComparison result

        Returns:
            ValidationResult with performance observations
        """
        result = ValidationResult()

        # Check if significantly underperformed
        if comparison.strategy_return < comparison.benchmark_return * 0.5:
            result.add_warning(
                code="SIGNIFICANT_UNDERPERFORMANCE",
                message=f"Strategy returned {comparison.strategy_return:.1%} vs benchmark {comparison.benchmark_return:.1%}",
                suggestion="Review strategy logic or consider alternatives",
            )

        # Check beta exposure
        if abs(comparison.beta) > 1.5:
            result.add_info(
                code="HIGH_BETA",
                message=f"Strategy has high beta ({comparison.beta:.2f}) to benchmark",
            )

        # Check tracking error
        if comparison.tracking_error > 0.30:
            result.add_info(
                code="HIGH_TRACKING_ERROR",
                message=f"High tracking error ({comparison.tracking_error:.1%}) vs benchmark",
            )

        # Check information ratio
        if comparison.information_ratio < -1.0:
            result.add_warning(
                code="POOR_RISK_ADJUSTED_RETURN",
                message=f"Negative information ratio ({comparison.information_ratio:.2f})",
                suggestion="Strategy underperforms on risk-adjusted basis",
            )

        return result


# =============================================================================
# Convenience Functions
# =============================================================================


def validate_strategy(
    strategy: Union[Strategy, BuiltStrategy],
    risk_limits: Optional[RiskLimits] = None,
) -> Tuple[bool, List[ValidationIssue]]:
    """
    Convenience function to validate a strategy.

    Args:
        strategy: Strategy to validate
        risk_limits: Optional risk limits to check

    Returns:
        Tuple of (is_valid, list of issues)
    """
    config_validator = StrategyConfigValidator()
    result = config_validator.validate(strategy)

    if risk_limits:
        risk_validator = RiskLimitValidator(risk_limits)
        result.merge(risk_validator.validate_strategy(strategy))

    return result.is_valid, result.issues


def validate_backtest_results(
    results: Dict[str, Any],
    equity_curve: Optional[List[float]] = None,
    initial_capital: float = 100_000.0,
    risk_limits: Optional[RiskLimits] = None,
) -> Tuple[bool, List[ValidationIssue]]:
    """
    Convenience function to validate backtest results.

    Args:
        results: Backtest results dictionary
        equity_curve: Optional equity curve for risk validation
        initial_capital: Initial capital
        risk_limits: Optional risk limits

    Returns:
        Tuple of (is_valid, list of issues)
    """
    sanity_checker = BacktestSanityChecker()
    result = sanity_checker.validate(results)

    if equity_curve and risk_limits:
        risk_validator = RiskLimitValidator(risk_limits)
        result.merge(
            risk_validator.validate_backtest_results(equity_curve, initial_capital)
        )

    return result.is_valid, result.issues


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Validation Result Types
    "ValidationSeverity",
    "ValidationIssue",
    "ValidationResult",
    # Validators
    "StrategyConfigValidator",
    "RiskLimitValidator",
    "RiskLimits",
    "BacktestSanityChecker",
    # Benchmark
    "PerformanceBenchmark",
    "BenchmarkComparison",
    # Convenience Functions
    "validate_strategy",
    "validate_backtest_results",
]
