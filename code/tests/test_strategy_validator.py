"""
Tests for Strategy Validation Framework

This module tests the validation components for strategy configuration,
risk limits, backtest sanity checks, and performance benchmarking.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta

from backtester.strategies.strategy_validator import (
    # Validation Result Types
    ValidationSeverity,
    ValidationIssue,
    ValidationResult,
    # Validators
    StrategyConfigValidator,
    RiskLimitValidator,
    RiskLimits,
    BacktestSanityChecker,
    # Benchmark
    PerformanceBenchmark,
    BenchmarkComparison,
    # Convenience Functions
    validate_strategy,
    validate_backtest_results,
)
from backtester.strategies.strategy_templates import HighIVStraddleTemplate
from backtester.strategies.strategy_builder import (
    StrategyBuilder,
    iv_rank_above,
    profit_target,
    stop_loss,
    dte_below,
    short_straddle,
    risk_percent,
)


# =============================================================================
# ValidationResult Tests
# =============================================================================


class TestValidationResult:
    """Tests for ValidationResult class."""

    def test_empty_result_is_valid(self):
        """Empty result should be valid."""
        result = ValidationResult()
        assert result.is_valid is True
        assert result.has_warnings is False
        assert len(result.errors) == 0
        assert len(result.warnings) == 0

    def test_add_error_makes_invalid(self):
        """Adding an error should make result invalid."""
        result = ValidationResult()
        result.add_error("TEST_ERROR", "Test error message")

        assert result.is_valid is False
        assert len(result.errors) == 1
        assert result.errors[0].code == "TEST_ERROR"

    def test_add_warning_keeps_valid(self):
        """Adding a warning should keep result valid but flag warnings."""
        result = ValidationResult()
        result.add_warning("TEST_WARNING", "Test warning message")

        assert result.is_valid is True
        assert result.has_warnings is True
        assert len(result.warnings) == 1

    def test_add_info_no_effect(self):
        """Adding info should not affect validity."""
        result = ValidationResult()
        result.add_info("TEST_INFO", "Test info message")

        assert result.is_valid is True
        assert result.has_warnings is False
        assert len(result.issues) == 1

    def test_merge_results(self):
        """Test merging two validation results."""
        result1 = ValidationResult()
        result1.add_error("ERROR1", "First error")

        result2 = ValidationResult()
        result2.add_warning("WARNING1", "First warning")

        result1.merge(result2)

        assert len(result1.issues) == 2
        assert len(result1.errors) == 1
        assert len(result1.warnings) == 1

    def test_validation_issue_str(self):
        """Test string representation of validation issue."""
        issue = ValidationIssue(
            code="TEST",
            message="Test message",
            severity=ValidationSeverity.ERROR,
            field="test_field",
        )

        str_repr = str(issue)
        assert "[ERROR]" in str_repr
        assert "test_field" in str_repr
        assert "Test message" in str_repr


# =============================================================================
# StrategyConfigValidator Tests
# =============================================================================


class TestStrategyConfigValidator:
    """Tests for StrategyConfigValidator."""

    def test_validate_valid_strategy(self):
        """Test validation of a valid strategy."""
        strategy = HighIVStraddleTemplate.create()
        validator = StrategyConfigValidator()

        result = validator.validate(strategy)

        assert result.is_valid is True

    def test_validate_built_strategy_structure(self):
        """Test that BuiltStrategy structure is validated."""
        strategy = HighIVStraddleTemplate.create()
        validator = StrategyConfigValidator()

        result = validator.validate(strategy)

        # Should pass - has all required components
        assert result.is_valid is True

    def test_validate_capital_too_low(self):
        """Test validation catches capital below minimum."""
        strategy = HighIVStraddleTemplate.create(initial_capital=500.0)
        validator = StrategyConfigValidator()

        result = validator.validate(strategy)

        assert not result.is_valid
        error_codes = [e.code for e in result.errors]
        assert "CAPITAL_TOO_LOW" in error_codes

    def test_validate_capital_very_high_warning(self):
        """Test validation warns on very high capital."""
        strategy = HighIVStraddleTemplate.create(initial_capital=500_000_000.0)
        validator = StrategyConfigValidator()

        result = validator.validate(strategy)

        # Very high capital is a warning, not error
        assert result.has_warnings
        warning_codes = [w.code for w in result.warnings]
        assert "CAPITAL_VERY_HIGH" in warning_codes

    def test_validate_name_validation(self):
        """Test strategy name validation."""
        # Create a strategy with very short name
        strategy = (
            StrategyBuilder()
            .name("AB")  # Too short
            .underlying("SPY")
            .entry_condition(iv_rank_above(70))
            .structure(short_straddle(dte=30))
            .exit_condition(profit_target(0.50))
            .position_size(risk_percent(0.02))
            .build()
        )

        validator = StrategyConfigValidator()
        result = validator.validate(strategy)

        warning_codes = [w.code for w in result.warnings]
        assert "NAME_TOO_SHORT" in warning_codes


# =============================================================================
# RiskLimitValidator Tests
# =============================================================================


class TestRiskLimitValidator:
    """Tests for RiskLimitValidator."""

    def test_default_limits(self):
        """Test default risk limits are set."""
        limits = RiskLimits()

        assert limits.max_position_size_pct == 0.10
        assert limits.max_drawdown_pct == 0.20
        assert limits.max_daily_loss_pct == 0.05

    def test_custom_limits(self):
        """Test custom risk limits."""
        limits = RiskLimits(
            max_position_size_pct=0.05,
            max_drawdown_pct=0.15,
        )

        assert limits.max_position_size_pct == 0.05
        assert limits.max_drawdown_pct == 0.15

    def test_validate_position_too_large(self):
        """Test position size validation."""
        limits = RiskLimits(max_position_size_pct=0.10)
        validator = RiskLimitValidator(limits)

        # Position is 15% of portfolio - exceeds 10% limit
        result = validator.validate_position(
            position_value=15000,
            portfolio_value=100000,
        )

        assert not result.is_valid
        error_codes = [e.code for e in result.errors]
        assert "POSITION_TOO_LARGE" in error_codes

    def test_validate_position_acceptable(self):
        """Test acceptable position passes validation."""
        limits = RiskLimits(max_position_size_pct=0.10)
        validator = RiskLimitValidator(limits)

        # Position is 5% of portfolio - under 10% limit
        result = validator.validate_position(
            position_value=5000,
            portfolio_value=100000,
        )

        assert result.is_valid

    def test_validate_backtest_max_drawdown(self):
        """Test backtest drawdown validation."""
        limits = RiskLimits(max_drawdown_pct=0.20)
        validator = RiskLimitValidator(limits)

        # Create equity curve with 30% drawdown
        equity = [100000, 105000, 110000, 80000, 90000, 95000]

        result = validator.validate_backtest_results(equity, 100000)

        warning_codes = [w.code for w in result.warnings]
        assert "MAX_DRAWDOWN_EXCEEDED" in warning_codes

    def test_validate_backtest_large_daily_loss(self):
        """Test backtest daily loss validation."""
        limits = RiskLimits(max_daily_loss_pct=0.05)
        validator = RiskLimitValidator(limits)

        # Create equity curve with 10% daily loss
        equity = [100000, 102000, 92000, 95000]  # 10% drop day 2->3

        result = validator.validate_backtest_results(equity, 100000)

        warning_codes = [w.code for w in result.warnings]
        assert "LARGE_DAILY_LOSS" in warning_codes

    def test_validate_backtest_acceptable(self):
        """Test acceptable backtest passes validation."""
        limits = RiskLimits(max_drawdown_pct=0.20, max_daily_loss_pct=0.05)
        validator = RiskLimitValidator(limits)

        # Create stable equity curve
        equity = [100000, 101000, 102000, 101500, 103000, 104000]

        result = validator.validate_backtest_results(equity, 100000)

        assert result.is_valid


# =============================================================================
# BacktestSanityChecker Tests
# =============================================================================


class TestBacktestSanityChecker:
    """Tests for BacktestSanityChecker."""

    def test_validate_unrealistic_returns(self):
        """Test detection of unrealistic returns."""
        checker = BacktestSanityChecker()

        # 1000% return in 1 year is suspicious
        results = {
            "total_return": 10.0,  # 1000% return
            "backtest_days": 252,
            "total_trades": 50,
            "win_rate": 0.60,
            "sharpe_ratio": 2.0,
        }

        result = checker.validate(results)

        warning_codes = [w.code for w in result.warnings]
        assert "RETURNS_UNREALISTIC" in warning_codes

    def test_validate_insufficient_trades(self):
        """Test detection of insufficient trades."""
        checker = BacktestSanityChecker()

        results = {
            "total_return": 0.15,
            "backtest_days": 252,
            "total_trades": 5,  # Only 5 trades
            "win_rate": 0.80,
            "sharpe_ratio": 1.5,
        }

        result = checker.validate(results)

        warning_codes = [w.code for w in result.warnings]
        assert "INSUFFICIENT_TRADES" in warning_codes

    def test_validate_suspicious_win_rate(self):
        """Test detection of suspicious win rate."""
        checker = BacktestSanityChecker()

        results = {
            "total_return": 0.20,
            "backtest_days": 252,
            "total_trades": 100,
            "win_rate": 0.98,  # 98% win rate is suspicious
            "sharpe_ratio": 1.5,
        }

        result = checker.validate(results)

        warning_codes = [w.code for w in result.warnings]
        assert "WIN_RATE_SUSPICIOUS" in warning_codes

    def test_validate_unrealistic_sharpe(self):
        """Test detection of unrealistic Sharpe ratio."""
        checker = BacktestSanityChecker()

        results = {
            "total_return": 0.30,
            "backtest_days": 252,
            "total_trades": 50,
            "win_rate": 0.65,
            "sharpe_ratio": 7.0,  # Sharpe > 5 is suspicious
        }

        result = checker.validate(results)

        warning_codes = [w.code for w in result.warnings]
        assert "SHARPE_UNREALISTIC" in warning_codes

    def test_validate_reasonable_results(self):
        """Test that reasonable results pass validation."""
        checker = BacktestSanityChecker()

        results = {
            "total_return": 0.15,  # 15% return
            "backtest_days": 252,
            "total_trades": 50,
            "win_rate": 0.55,
            "sharpe_ratio": 1.2,
        }

        result = checker.validate(results)

        # Should have no errors
        assert result.is_valid


# =============================================================================
# PerformanceBenchmark Tests
# =============================================================================


class TestPerformanceBenchmark:
    """Tests for PerformanceBenchmark."""

    def test_compare_to_buy_hold(self):
        """Test comparison to buy-and-hold benchmark."""
        benchmark = PerformanceBenchmark()

        # Strategy returns: 0.5% per day for 10 days
        strategy_returns = [0.005] * 10
        # Benchmark returns: 0.3% per day for 10 days
        benchmark_returns = [0.003] * 10

        comparison = benchmark.compare_to_buy_hold(strategy_returns, benchmark_returns)

        assert comparison.strategy_return > comparison.benchmark_return
        assert comparison.outperformed is True
        assert isinstance(comparison.alpha, float)
        assert isinstance(comparison.beta, float)

    def test_compare_underperformance(self):
        """Test detection of underperformance."""
        benchmark = PerformanceBenchmark()

        # Strategy underperforms
        strategy_returns = [0.001] * 10
        benchmark_returns = [0.005] * 10

        comparison = benchmark.compare_to_buy_hold(strategy_returns, benchmark_returns)

        assert comparison.outperformed is False
        assert comparison.strategy_return < comparison.benchmark_return

    def test_validate_against_benchmark(self):
        """Test validation of benchmark comparison."""
        benchmark = PerformanceBenchmark()

        # Create a comparison showing significant underperformance
        comparison = BenchmarkComparison(
            strategy_return=0.05,
            benchmark_return=0.20,  # 4x better
            alpha=-0.10,
            beta=0.8,
            correlation=0.7,
            tracking_error=0.15,
            information_ratio=-1.5,
        )

        result = benchmark.validate_against_benchmark(comparison)

        warning_codes = [w.code for w in result.warnings]
        assert "SIGNIFICANT_UNDERPERFORMANCE" in warning_codes

    def test_information_ratio_warning(self):
        """Test warning on poor information ratio."""
        benchmark = PerformanceBenchmark()

        comparison = BenchmarkComparison(
            strategy_return=0.10,
            benchmark_return=0.12,
            alpha=-0.02,
            beta=1.0,
            correlation=0.8,
            tracking_error=0.10,
            information_ratio=-1.5,  # Poor risk-adjusted return
        )

        result = benchmark.validate_against_benchmark(comparison)

        warning_codes = [w.code for w in result.warnings]
        assert "POOR_RISK_ADJUSTED_RETURN" in warning_codes


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience validation functions."""

    def test_validate_strategy_function(self):
        """Test validate_strategy convenience function."""
        strategy = HighIVStraddleTemplate.create()

        is_valid, issues = validate_strategy(strategy)

        assert is_valid is True
        assert isinstance(issues, list)

    def test_validate_strategy_with_risk_limits(self):
        """Test validate_strategy with custom risk limits."""
        strategy = HighIVStraddleTemplate.create()
        limits = RiskLimits(max_position_size_pct=0.05)

        is_valid, issues = validate_strategy(strategy, risk_limits=limits)

        assert isinstance(is_valid, bool)
        assert isinstance(issues, list)

    def test_validate_backtest_results_function(self):
        """Test validate_backtest_results convenience function."""
        results = {
            "total_return": 0.15,
            "backtest_days": 252,
            "total_trades": 50,
            "win_rate": 0.55,
            "sharpe_ratio": 1.2,
        }

        is_valid, issues = validate_backtest_results(results)

        assert isinstance(is_valid, bool)
        assert isinstance(issues, list)

    def test_validate_backtest_with_equity_curve(self):
        """Test backtest validation with equity curve."""
        results = {
            "total_return": 0.15,
            "backtest_days": 252,
            "total_trades": 50,
            "win_rate": 0.55,
            "sharpe_ratio": 1.2,
        }
        equity_curve = [100000, 105000, 110000, 108000, 115000]
        limits = RiskLimits(max_drawdown_pct=0.10)

        is_valid, issues = validate_backtest_results(
            results,
            equity_curve=equity_curve,
            initial_capital=100000,
            risk_limits=limits,
        )

        assert isinstance(is_valid, bool)
        assert isinstance(issues, list)


# =============================================================================
# Integration Tests
# =============================================================================


class TestValidatorIntegration:
    """Integration tests for validators."""

    def test_full_strategy_validation_flow(self):
        """Test complete strategy validation workflow."""
        # Create strategy
        strategy = HighIVStraddleTemplate.create(
            underlying="SPY",
            iv_threshold=70,
            profit_target_pct=0.50,
            initial_capital=100000,
        )

        # Validate configuration
        config_validator = StrategyConfigValidator()
        config_result = config_validator.validate(strategy)

        # Validate risk limits
        limits = RiskLimits()
        risk_validator = RiskLimitValidator(limits)
        risk_result = risk_validator.validate_strategy(strategy)

        # Both should pass for a valid strategy
        assert config_result.is_valid
        # Risk result may have warnings but should be valid

    def test_backtest_validation_flow(self):
        """Test complete backtest validation workflow."""
        # Simulated backtest results
        results = {
            "total_return": 0.18,
            "backtest_days": 365,
            "total_trades": 45,
            "win_rate": 0.58,
            "sharpe_ratio": 1.4,
        }

        # Simulated equity curve
        np.random.seed(42)
        daily_returns = np.random.normal(0.0005, 0.01, 365)
        equity = 100000 * np.cumprod(1 + daily_returns)
        equity_curve = equity.tolist()

        # Validate with sanity checker
        checker = BacktestSanityChecker()
        sanity_result = checker.validate(results)

        # Validate risk limits
        limits = RiskLimits()
        risk_validator = RiskLimitValidator(limits)
        risk_result = risk_validator.validate_backtest_results(equity_curve, 100000)

        # Both should be valid for reasonable results
        assert sanity_result.is_valid

    def test_benchmark_comparison_flow(self):
        """Test complete benchmark comparison workflow."""
        # Generate random returns
        np.random.seed(42)
        strategy_returns = np.random.normal(0.0008, 0.015, 252).tolist()
        benchmark_returns = np.random.normal(0.0004, 0.012, 252).tolist()

        # Compare to benchmark
        benchmark = PerformanceBenchmark()
        comparison = benchmark.compare_to_buy_hold(strategy_returns, benchmark_returns)

        # Validate comparison
        result = benchmark.validate_against_benchmark(comparison)

        # Should produce a valid result
        assert isinstance(comparison.alpha, float)
        assert isinstance(comparison.beta, float)
        assert isinstance(result.is_valid, bool)


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_equity_curve(self):
        """Test handling of empty equity curve."""
        validator = RiskLimitValidator()

        result = validator.validate_backtest_results([], 100000)

        # Should warn about insufficient data
        warning_codes = [w.code for w in result.warnings]
        assert "INSUFFICIENT_DATA" in warning_codes

    def test_single_point_equity_curve(self):
        """Test handling of single-point equity curve."""
        validator = RiskLimitValidator()

        result = validator.validate_backtest_results([100000], 100000)

        warning_codes = [w.code for w in result.warnings]
        assert "INSUFFICIENT_DATA" in warning_codes

    def test_zero_portfolio_value(self):
        """Test handling of zero portfolio value."""
        validator = RiskLimitValidator()

        result = validator.validate_position(
            position_value=5000,
            portfolio_value=0,  # Zero portfolio
        )

        # Should not crash, position check is skipped
        assert isinstance(result, ValidationResult)

    def test_negative_returns(self):
        """Test validation with negative returns."""
        checker = BacktestSanityChecker()

        results = {
            "total_return": -0.50,  # 50% loss
            "backtest_days": 252,
            "total_trades": 50,
            "win_rate": 0.30,
            "sharpe_ratio": -1.0,
        }

        result = checker.validate(results)

        # Should detect extreme loss
        warning_codes = [w.code for w in result.warnings]
        # May have warnings about extreme loss

    def test_mismatched_return_lengths(self):
        """Test benchmark comparison with mismatched lengths."""
        benchmark = PerformanceBenchmark()

        strategy_returns = [0.01] * 10
        benchmark_returns = [0.01] * 20  # Different length

        with pytest.raises(ValueError):
            benchmark.compare_to_buy_hold(strategy_returns, benchmark_returns)

    def test_zero_std_returns(self):
        """Test benchmark comparison with zero std returns."""
        benchmark = PerformanceBenchmark()

        # All same returns - zero variance
        strategy_returns = [0.01] * 10
        benchmark_returns = [0.01] * 10

        # Should not crash
        comparison = benchmark.compare_to_buy_hold(strategy_returns, benchmark_returns)

        assert isinstance(comparison, BenchmarkComparison)
