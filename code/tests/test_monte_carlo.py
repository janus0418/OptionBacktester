"""
Comprehensive Tests for Monte Carlo Simulation Module

This module contains extensive tests for:
    - MonteCarloSimulator: GBM simulation, bootstrap resampling, block bootstrap
    - Confidence intervals for performance metrics
    - VaR/CVaR estimation via simulation
    - Distribution analysis
    - Drawdown distribution analysis

Test Coverage:
    - Unit tests for each simulation method with known inputs/outputs
    - Statistical property tests (mean, variance, distribution shape)
    - Reproducibility tests (random seed)
    - Boundary condition tests (edge cases)
    - Error condition testing

Requirements:
    - pytest
    - numpy
    - pandas
    - scipy

Run Tests:
    pytest tests/test_monte_carlo.py -v --tb=short

Author: OptionsBacktester2 Team
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

# Import Monte Carlo classes
from backtester.analytics.monte_carlo import (
    MonteCarloSimulator,
    SimulationResult,
    ConfidenceInterval,
    MonteCarloError,
    InsufficientDataError,
    InvalidParameterError,
    SimulationError,
    DEFAULT_NUM_SIMULATIONS,
    DEFAULT_NUM_PERIODS,
    DEFAULT_CONFIDENCE_LEVEL,
    DEFAULT_RANDOM_SEED,
    TRADING_DAYS_PER_YEAR,
    MIN_OBSERVATIONS_FOR_SIMULATION,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_returns():
    """Create sample returns for testing."""
    np.random.seed(42)
    # Generate 252 daily returns (1 year)
    returns = np.random.normal(0.0004, 0.01, 252)  # ~10% annual, 15% vol
    dates = pd.date_range(start="2023-01-01", periods=252, freq="B")
    return pd.Series(returns, index=dates)


@pytest.fixture
def small_returns():
    """Create minimal valid returns for edge case testing."""
    np.random.seed(42)
    returns = np.random.normal(0.0004, 0.01, MIN_OBSERVATIONS_FOR_SIMULATION)
    return pd.Series(returns)


@pytest.fixture
def known_returns():
    """Create returns with known statistical properties."""
    # Create returns that we know the mean and std of
    np.random.seed(123)
    n = 1000
    mean = 0.001
    std = 0.02
    returns = np.random.normal(mean, std, n)
    return pd.Series(returns)


@pytest.fixture
def trending_returns():
    """Create returns with autocorrelation for block bootstrap testing."""
    np.random.seed(42)
    n = 252
    returns = np.zeros(n)
    # Add some autocorrelation
    for i in range(1, n):
        returns[i] = 0.3 * returns[i - 1] + np.random.normal(0.0004, 0.01)
    returns[0] = np.random.normal(0.0004, 0.01)
    return pd.Series(returns)


# =============================================================================
# Tests for GBM Simulation
# =============================================================================


class TestSimulateReturnsGBM:
    """Tests for Geometric Brownian Motion simulation."""

    def test_basic_simulation(self, sample_returns):
        """Test basic GBM simulation runs successfully."""
        result = MonteCarloSimulator.simulate_returns_gbm(
            sample_returns, num_simulations=1000, num_periods=100, random_seed=42
        )

        assert isinstance(result, SimulationResult)
        assert result.simulated_paths.shape == (1000, 101)  # periods + 1
        assert len(result.final_values) == 1000
        assert "method" in result.parameters
        assert result.parameters["method"] == "gbm"

    def test_simulation_reproducibility(self, sample_returns):
        """Test that same seed produces same results."""
        result1 = MonteCarloSimulator.simulate_returns_gbm(
            sample_returns, num_simulations=100, random_seed=42
        )
        result2 = MonteCarloSimulator.simulate_returns_gbm(
            sample_returns, num_simulations=100, random_seed=42
        )

        np.testing.assert_array_almost_equal(
            result1.simulated_paths, result2.simulated_paths
        )

    def test_different_seeds_different_results(self, sample_returns):
        """Test that different seeds produce different results."""
        result1 = MonteCarloSimulator.simulate_returns_gbm(
            sample_returns, num_simulations=100, random_seed=42
        )
        result2 = MonteCarloSimulator.simulate_returns_gbm(
            sample_returns, num_simulations=100, random_seed=123
        )

        assert not np.allclose(result1.simulated_paths, result2.simulated_paths)

    def test_simulation_starts_at_initial_value(self, sample_returns):
        """Test that all paths start at the initial value."""
        initial = 100.0
        result = MonteCarloSimulator.simulate_returns_gbm(
            sample_returns, num_simulations=100, initial_value=initial, random_seed=42
        )

        assert np.allclose(result.simulated_paths[:, 0], initial)

    def test_statistics_calculated(self, sample_returns):
        """Test that statistics are calculated correctly."""
        result = MonteCarloSimulator.simulate_returns_gbm(
            sample_returns, num_simulations=1000, random_seed=42
        )

        stats = result.statistics
        assert "mean" in stats
        assert "median" in stats
        assert "std" in stats
        assert "skewness" in stats
        assert "kurtosis" in stats
        assert "percentile_5" in stats
        assert "percentile_95" in stats

    def test_parameters_stored(self, sample_returns):
        """Test that simulation parameters are stored."""
        result = MonteCarloSimulator.simulate_returns_gbm(
            sample_returns, num_simulations=500, num_periods=100, random_seed=42
        )

        params = result.parameters
        assert params["method"] == "gbm"
        assert params["num_simulations"] == 500
        assert params["num_periods"] == 100
        assert "mu" in params
        assert "sigma" in params
        assert "annualized_return" in params
        assert "annualized_volatility" in params

    def test_insufficient_data_raises_error(self):
        """Test that insufficient data raises InsufficientDataError."""
        short_returns = pd.Series([0.01, 0.02])  # Too few observations

        with pytest.raises(InsufficientDataError):
            MonteCarloSimulator.simulate_returns_gbm(short_returns)

    def test_invalid_num_simulations_raises_error(self, sample_returns):
        """Test that invalid num_simulations raises error."""
        with pytest.raises(InvalidParameterError):
            MonteCarloSimulator.simulate_returns_gbm(sample_returns, num_simulations=0)

        with pytest.raises(InvalidParameterError):
            MonteCarloSimulator.simulate_returns_gbm(sample_returns, num_simulations=-1)

    def test_invalid_num_periods_raises_error(self, sample_returns):
        """Test that invalid num_periods raises error."""
        with pytest.raises(InvalidParameterError):
            MonteCarloSimulator.simulate_returns_gbm(sample_returns, num_periods=0)

    def test_invalid_initial_value_raises_error(self, sample_returns):
        """Test that invalid initial_value raises error."""
        with pytest.raises(InvalidParameterError):
            MonteCarloSimulator.simulate_returns_gbm(sample_returns, initial_value=0)

        with pytest.raises(InvalidParameterError):
            MonteCarloSimulator.simulate_returns_gbm(sample_returns, initial_value=-100)

    def test_handles_list_input(self, sample_returns):
        """Test that list input is converted to Series."""
        returns_list = sample_returns.tolist()
        result = MonteCarloSimulator.simulate_returns_gbm(
            returns_list, num_simulations=100, random_seed=42
        )

        assert isinstance(result, SimulationResult)

    def test_handles_nan_values(self, sample_returns):
        """Test that NaN values are handled."""
        returns_with_nan = sample_returns.copy()
        returns_with_nan.iloc[0] = np.nan
        returns_with_nan.iloc[10] = np.nan

        result = MonteCarloSimulator.simulate_returns_gbm(
            returns_with_nan, num_simulations=100, random_seed=42
        )

        assert isinstance(result, SimulationResult)


# =============================================================================
# Tests for Bootstrap Simulation
# =============================================================================


class TestSimulateReturnsBootstrap:
    """Tests for bootstrap resampling simulation."""

    def test_basic_simulation(self, sample_returns):
        """Test basic bootstrap simulation runs successfully."""
        result = MonteCarloSimulator.simulate_returns_bootstrap(
            sample_returns, num_simulations=1000, num_periods=100, random_seed=42
        )

        assert isinstance(result, SimulationResult)
        assert result.simulated_paths.shape == (1000, 101)
        assert result.parameters["method"] == "bootstrap"

    def test_simulation_reproducibility(self, sample_returns):
        """Test that same seed produces same results."""
        result1 = MonteCarloSimulator.simulate_returns_bootstrap(
            sample_returns, num_simulations=100, random_seed=42
        )
        result2 = MonteCarloSimulator.simulate_returns_bootstrap(
            sample_returns, num_simulations=100, random_seed=42
        )

        np.testing.assert_array_almost_equal(
            result1.simulated_paths, result2.simulated_paths
        )

    def test_preserves_empirical_distribution(self, known_returns):
        """Test that bootstrap preserves empirical mean and std."""
        # Bootstrap should preserve the empirical distribution
        result = MonteCarloSimulator.simulate_returns_bootstrap(
            known_returns,
            num_simulations=10000,
            num_periods=len(known_returns),
            random_seed=42,
        )

        # Calculate returns from paths
        all_returns = []
        for path in result.simulated_paths:
            path_returns = np.diff(path) / path[:-1]
            all_returns.extend(path_returns)
        all_returns = np.array(all_returns)

        # Mean and std should be close to original
        original_mean = known_returns.mean()
        original_std = known_returns.std()

        # Use looser tolerance due to resampling variance
        assert abs(np.mean(all_returns) - original_mean) < 0.005
        assert abs(np.std(all_returns) - original_std) < 0.01

    def test_parameters_stored(self, sample_returns):
        """Test that simulation parameters are stored."""
        result = MonteCarloSimulator.simulate_returns_bootstrap(
            sample_returns, num_simulations=500, num_periods=100, random_seed=42
        )

        params = result.parameters
        assert params["method"] == "bootstrap"
        assert params["num_simulations"] == 500
        assert "historical_mean" in params
        assert "historical_std" in params
        assert "sample_size" in params


# =============================================================================
# Tests for Block Bootstrap Simulation
# =============================================================================


class TestSimulateReturnsBlockBootstrap:
    """Tests for block bootstrap resampling simulation."""

    def test_basic_simulation(self, trending_returns):
        """Test basic block bootstrap simulation runs successfully."""
        result = MonteCarloSimulator.simulate_returns_block_bootstrap(
            trending_returns,
            num_simulations=1000,
            num_periods=100,
            block_size=20,
            random_seed=42,
        )

        assert isinstance(result, SimulationResult)
        assert result.parameters["method"] == "block_bootstrap"
        assert result.parameters["block_size"] == 20

    def test_simulation_reproducibility(self, trending_returns):
        """Test that same seed produces same results."""
        result1 = MonteCarloSimulator.simulate_returns_block_bootstrap(
            trending_returns, num_simulations=100, block_size=10, random_seed=42
        )
        result2 = MonteCarloSimulator.simulate_returns_block_bootstrap(
            trending_returns, num_simulations=100, block_size=10, random_seed=42
        )

        np.testing.assert_array_almost_equal(
            result1.simulated_paths, result2.simulated_paths
        )

    def test_block_size_larger_than_data(self, small_returns):
        """Test handling when block size exceeds data length."""
        # Should reduce block size automatically
        result = MonteCarloSimulator.simulate_returns_block_bootstrap(
            small_returns,
            num_simulations=100,
            num_periods=50,
            block_size=100,  # Larger than data length
            random_seed=42,
        )

        assert isinstance(result, SimulationResult)

    def test_invalid_block_size_raises_error(self, sample_returns):
        """Test that invalid block_size raises error."""
        with pytest.raises(InvalidParameterError):
            MonteCarloSimulator.simulate_returns_block_bootstrap(
                sample_returns, block_size=0
            )


# =============================================================================
# Tests for Confidence Intervals
# =============================================================================


class TestConfidenceIntervals:
    """Tests for confidence interval calculation."""

    def test_metric_confidence_interval(self, sample_returns):
        """Test confidence interval calculation for a metric."""
        result = MonteCarloSimulator.simulate_returns_bootstrap(
            sample_returns, num_simulations=1000, random_seed=42
        )

        # Define a metric function (total return)
        total_return_func = lambda path: (path[-1] / path[0]) - 1

        ci = MonteCarloSimulator.calculate_metric_confidence_interval(
            result.simulated_paths,
            total_return_func,
            confidence_level=0.95,
            metric_name="total_return",
        )

        assert isinstance(ci, ConfidenceInterval)
        assert ci.metric_name == "total_return"
        assert ci.confidence_level == 0.95
        assert ci.lower_bound < ci.point_estimate < ci.upper_bound
        assert ci.standard_error > 0

    def test_confidence_interval_bounds_order(self, sample_returns):
        """Test that lower bound < upper bound."""
        result = MonteCarloSimulator.simulate_returns_bootstrap(
            sample_returns, num_simulations=1000, random_seed=42
        )

        metric_func = lambda path: path[-1] / path[0] - 1

        for conf in [0.90, 0.95, 0.99]:
            ci = MonteCarloSimulator.calculate_metric_confidence_interval(
                result.simulated_paths, metric_func, confidence_level=conf
            )
            assert ci.lower_bound < ci.upper_bound

    def test_wider_ci_at_higher_confidence(self, sample_returns):
        """Test that higher confidence gives wider intervals."""
        result = MonteCarloSimulator.simulate_returns_bootstrap(
            sample_returns, num_simulations=1000, random_seed=42
        )

        metric_func = lambda path: path[-1]

        ci_90 = MonteCarloSimulator.calculate_metric_confidence_interval(
            result.simulated_paths, metric_func, 0.90
        )
        ci_95 = MonteCarloSimulator.calculate_metric_confidence_interval(
            result.simulated_paths, metric_func, 0.95
        )
        ci_99 = MonteCarloSimulator.calculate_metric_confidence_interval(
            result.simulated_paths, metric_func, 0.99
        )

        width_90 = ci_90.upper_bound - ci_90.lower_bound
        width_95 = ci_95.upper_bound - ci_95.lower_bound
        width_99 = ci_99.upper_bound - ci_99.lower_bound

        assert width_90 < width_95 < width_99

    def test_invalid_confidence_raises_error(self, sample_returns):
        """Test that invalid confidence level raises error."""
        result = MonteCarloSimulator.simulate_returns_bootstrap(
            sample_returns, num_simulations=100, random_seed=42
        )
        metric_func = lambda path: path[-1]

        with pytest.raises(InvalidParameterError):
            MonteCarloSimulator.calculate_metric_confidence_interval(
                result.simulated_paths, metric_func, confidence_level=0
            )

        with pytest.raises(InvalidParameterError):
            MonteCarloSimulator.calculate_metric_confidence_interval(
                result.simulated_paths, metric_func, confidence_level=1.0
            )

    def test_calculate_all_confidence_intervals(self, sample_returns):
        """Test calculation of all standard confidence intervals."""
        result = MonteCarloSimulator.simulate_returns_bootstrap(
            sample_returns, num_simulations=1000, random_seed=42
        )

        all_cis = MonteCarloSimulator.calculate_all_confidence_intervals(
            result, confidence_level=0.95
        )

        expected_metrics = [
            "total_return",
            "max_drawdown",
            "volatility",
            "sharpe_ratio",
            "final_value",
        ]

        for metric in expected_metrics:
            assert metric in all_cis
            assert isinstance(all_cis[metric], ConfidenceInterval)


# =============================================================================
# Tests for VaR/CVaR Estimation
# =============================================================================


class TestVaREstimation:
    """Tests for VaR and CVaR estimation via simulation."""

    def test_estimate_var_monte_carlo(self, sample_returns):
        """Test basic VaR estimation."""
        result = MonteCarloSimulator.simulate_returns_bootstrap(
            sample_returns, num_simulations=10000, random_seed=42
        )

        var_metrics = MonteCarloSimulator.estimate_var_monte_carlo(
            result, confidence_level=0.95, time_horizon=1
        )

        assert "var" in var_metrics
        assert "cvar" in var_metrics
        assert "var_dollar" in var_metrics
        assert "cvar_dollar" in var_metrics
        assert var_metrics["var"] < 0  # VaR is typically negative (loss)
        assert var_metrics["cvar"] <= var_metrics["var"]  # CVaR <= VaR

    def test_var_confidence_levels(self, sample_returns):
        """Test VaR at different confidence levels."""
        result = MonteCarloSimulator.simulate_returns_bootstrap(
            sample_returns, num_simulations=10000, random_seed=42
        )

        var_90 = MonteCarloSimulator.estimate_var_monte_carlo(
            result, confidence_level=0.90
        )["var"]
        var_95 = MonteCarloSimulator.estimate_var_monte_carlo(
            result, confidence_level=0.95
        )["var"]
        var_99 = MonteCarloSimulator.estimate_var_monte_carlo(
            result, confidence_level=0.99
        )["var"]

        # Higher confidence = more extreme VaR (more negative)
        assert var_90 > var_95 > var_99

    def test_estimate_var_at_multiple_levels(self, sample_returns):
        """Test VaR estimation at multiple confidence levels."""
        result = MonteCarloSimulator.simulate_returns_bootstrap(
            sample_returns, num_simulations=5000, random_seed=42
        )

        var_table = MonteCarloSimulator.estimate_var_at_multiple_levels(result)

        assert isinstance(var_table, pd.DataFrame)
        assert len(var_table) == 3  # Default: 0.90, 0.95, 0.99
        assert "confidence_level" in var_table.columns
        assert "var" in var_table.columns
        assert "cvar" in var_table.columns

    def test_invalid_confidence_raises_error(self, sample_returns):
        """Test that invalid confidence level raises error."""
        result = MonteCarloSimulator.simulate_returns_bootstrap(
            sample_returns, num_simulations=100, random_seed=42
        )

        with pytest.raises(InvalidParameterError):
            MonteCarloSimulator.estimate_var_monte_carlo(result, confidence_level=1.5)

    def test_invalid_time_horizon_raises_error(self, sample_returns):
        """Test that time horizon exceeding periods raises error."""
        result = MonteCarloSimulator.simulate_returns_bootstrap(
            sample_returns, num_simulations=100, num_periods=50, random_seed=42
        )

        with pytest.raises(InvalidParameterError):
            MonteCarloSimulator.estimate_var_monte_carlo(
                result,
                time_horizon=100,  # Exceeds num_periods
            )


# =============================================================================
# Tests for Distribution Analysis
# =============================================================================


class TestDistributionAnalysis:
    """Tests for return distribution analysis."""

    def test_analyze_return_distribution(self, sample_returns):
        """Test return distribution analysis."""
        result = MonteCarloSimulator.simulate_returns_bootstrap(
            sample_returns, num_simulations=5000, random_seed=42
        )

        analysis = MonteCarloSimulator.analyze_return_distribution(result)

        assert "moments" in analysis
        assert "percentiles" in analysis
        assert "probability_of_profit" in analysis
        assert "probability_of_loss" in analysis
        assert "expected_profit_given_profit" in analysis
        assert "expected_loss_given_loss" in analysis
        assert "tail_statistics" in analysis

    def test_probability_sum(self, sample_returns):
        """Test that probabilities sum to ~1."""
        result = MonteCarloSimulator.simulate_returns_bootstrap(
            sample_returns, num_simulations=5000, random_seed=42
        )

        analysis = MonteCarloSimulator.analyze_return_distribution(result)

        prob_sum = (
            analysis["probability_of_profit"]
            + analysis["probability_of_loss"]
            + analysis["probability_of_breakeven"]
        )

        assert abs(prob_sum - 1.0) < 0.001

    def test_calculate_probability_of_target(self, sample_returns):
        """Test probability of achieving target return."""
        result = MonteCarloSimulator.simulate_returns_bootstrap(
            sample_returns, num_simulations=5000, random_seed=42
        )

        prob = MonteCarloSimulator.calculate_probability_of_target(
            result, target_return=0.10
        )

        assert "probability" in prob
        assert "target_return" in prob
        assert 0 <= prob["probability"] <= 1

    def test_higher_target_lower_probability(self, sample_returns):
        """Test that higher targets have lower probability."""
        result = MonteCarloSimulator.simulate_returns_bootstrap(
            sample_returns, num_simulations=5000, random_seed=42
        )

        prob_10 = MonteCarloSimulator.calculate_probability_of_target(
            result, target_return=0.10
        )["probability"]
        prob_20 = MonteCarloSimulator.calculate_probability_of_target(
            result, target_return=0.20
        )["probability"]
        prob_50 = MonteCarloSimulator.calculate_probability_of_target(
            result, target_return=0.50
        )["probability"]

        assert prob_10 >= prob_20 >= prob_50


# =============================================================================
# Tests for Drawdown Analysis
# =============================================================================


class TestDrawdownAnalysis:
    """Tests for drawdown distribution analysis."""

    def test_analyze_drawdown_distribution(self, sample_returns):
        """Test drawdown distribution analysis."""
        result = MonteCarloSimulator.simulate_returns_bootstrap(
            sample_returns, num_simulations=1000, random_seed=42
        )

        dd_analysis = MonteCarloSimulator.analyze_drawdown_distribution(result)

        assert "mean_max_drawdown" in dd_analysis
        assert "median_max_drawdown" in dd_analysis
        assert "worst_drawdown" in dd_analysis
        assert "percentiles" in dd_analysis
        assert "probability_dd_exceeds_10pct" in dd_analysis
        assert "probability_dd_exceeds_20pct" in dd_analysis

    def test_drawdown_values_negative(self, sample_returns):
        """Test that drawdown values are negative or zero."""
        result = MonteCarloSimulator.simulate_returns_bootstrap(
            sample_returns, num_simulations=1000, random_seed=42
        )

        dd_analysis = MonteCarloSimulator.analyze_drawdown_distribution(result)

        assert dd_analysis["mean_max_drawdown"] <= 0
        assert dd_analysis["worst_drawdown"] <= 0


# =============================================================================
# Tests for Utility Methods
# =============================================================================


class TestUtilityMethods:
    """Tests for utility methods."""

    def test_paths_to_dataframe(self, sample_returns):
        """Test conversion of paths to DataFrame."""
        result = MonteCarloSimulator.simulate_returns_bootstrap(
            sample_returns, num_simulations=100, random_seed=42
        )

        df = MonteCarloSimulator.paths_to_dataframe(result, sample_paths=50)

        assert isinstance(df, pd.DataFrame)
        assert df.shape[1] == 50  # 50 sampled paths
        assert "path_0" in df.columns

    def test_summary_report(self, sample_returns):
        """Test comprehensive summary report generation."""
        result = MonteCarloSimulator.simulate_returns_bootstrap(
            sample_returns, num_simulations=1000, random_seed=42
        )

        report = MonteCarloSimulator.summary_report(result)

        assert "simulation_parameters" in report
        assert "basic_statistics" in report
        assert "distribution_analysis" in report
        assert "drawdown_analysis" in report
        assert "confidence_intervals" in report
        assert "var_estimates" in report


# =============================================================================
# Tests for Statistical Properties
# =============================================================================


class TestStatisticalProperties:
    """Tests for statistical correctness of simulations."""

    def test_gbm_log_normal_returns(self, sample_returns):
        """Test that GBM produces approximately log-normal distribution."""
        result = MonteCarloSimulator.simulate_returns_gbm(
            sample_returns, num_simulations=10000, num_periods=252, random_seed=42
        )

        # Log of final values should be approximately normal
        log_finals = np.log(result.final_values)

        # Use Shapiro-Wilk test (sample of 5000 for performance)
        from scipy import stats

        sample = np.random.choice(
            log_finals, size=min(5000, len(log_finals)), replace=False
        )
        _, p_value = stats.shapiro(sample)

        # With many simulations, log returns should be close to normal
        # p-value > 0.01 suggests normality is reasonable
        assert p_value > 0.001  # Very loose threshold due to sampling

    def test_more_simulations_smaller_standard_error(self, sample_returns):
        """Test that more simulations reduce standard error."""
        metric_func = lambda path: path[-1] / path[0] - 1

        result_small = MonteCarloSimulator.simulate_returns_bootstrap(
            sample_returns, num_simulations=100, random_seed=42
        )
        result_large = MonteCarloSimulator.simulate_returns_bootstrap(
            sample_returns, num_simulations=5000, random_seed=42
        )

        ci_small = MonteCarloSimulator.calculate_metric_confidence_interval(
            result_small.simulated_paths, metric_func
        )
        ci_large = MonteCarloSimulator.calculate_metric_confidence_interval(
            result_large.simulated_paths, metric_func
        )

        # Larger sample should have smaller standard error
        assert ci_large.standard_error < ci_small.standard_error


# =============================================================================
# Tests for Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_minimum_valid_data(self, small_returns):
        """Test with minimum valid data size."""
        result = MonteCarloSimulator.simulate_returns_gbm(
            small_returns, num_simulations=100, random_seed=42
        )

        assert isinstance(result, SimulationResult)

    def test_single_simulation(self, sample_returns):
        """Test with single simulation."""
        result = MonteCarloSimulator.simulate_returns_bootstrap(
            sample_returns, num_simulations=1, random_seed=42
        )

        assert result.simulated_paths.shape[0] == 1

    def test_single_period(self, sample_returns):
        """Test with single period simulation."""
        result = MonteCarloSimulator.simulate_returns_bootstrap(
            sample_returns, num_periods=1, random_seed=42
        )

        assert result.simulated_paths.shape[1] == 2  # Initial + 1 period

    def test_returns_with_zero_volatility(self):
        """Test handling of near-zero volatility returns."""
        # Constant returns (zero volatility)
        constant_returns = pd.Series([0.001] * 100)

        # Should still work but with warning
        result = MonteCarloSimulator.simulate_returns_gbm(
            constant_returns, num_simulations=100, random_seed=42
        )

        assert isinstance(result, SimulationResult)

    def test_empty_after_dropna_raises_error(self):
        """Test that all-NaN data raises error."""
        nan_returns = pd.Series([np.nan] * 100)

        with pytest.raises(InsufficientDataError):
            MonteCarloSimulator.simulate_returns_gbm(nan_returns)


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for Monte Carlo simulation workflow."""

    def test_complete_analysis_workflow(self, sample_returns):
        """Test complete analysis workflow from simulation to report."""
        # Step 1: Run simulation
        result = MonteCarloSimulator.simulate_returns_bootstrap(
            sample_returns, num_simulations=1000, num_periods=252, random_seed=42
        )

        # Step 2: Calculate confidence intervals
        cis = MonteCarloSimulator.calculate_all_confidence_intervals(result)

        # Step 3: Estimate VaR
        var_metrics = MonteCarloSimulator.estimate_var_monte_carlo(result)

        # Step 4: Analyze distribution
        dist_analysis = MonteCarloSimulator.analyze_return_distribution(result)

        # Step 5: Analyze drawdowns
        dd_analysis = MonteCarloSimulator.analyze_drawdown_distribution(result)

        # Step 6: Generate report
        report = MonteCarloSimulator.summary_report(result)

        # Verify all components work together
        assert len(cis) >= 5
        assert var_metrics["var"] < 0
        assert dist_analysis["probability_of_profit"] >= 0
        assert dd_analysis["mean_max_drawdown"] <= 0
        assert len(report) >= 6

    def test_comparing_simulation_methods(self, sample_returns):
        """Test comparing GBM and bootstrap methods."""
        gbm_result = MonteCarloSimulator.simulate_returns_gbm(
            sample_returns, num_simulations=5000, num_periods=252, random_seed=42
        )

        bootstrap_result = MonteCarloSimulator.simulate_returns_bootstrap(
            sample_returns, num_simulations=5000, num_periods=252, random_seed=42
        )

        # Both should have similar characteristics (rough check)
        gbm_mean = gbm_result.statistics["mean"]
        bootstrap_mean = bootstrap_result.statistics["mean"]

        # Means should be in the same ballpark (not necessarily equal)
        assert abs(gbm_mean - bootstrap_mean) / bootstrap_mean < 0.5


# =============================================================================
# Performance Tests
# =============================================================================


class TestPerformance:
    """Performance tests for Monte Carlo simulation."""

    def test_large_simulation_completes(self, sample_returns):
        """Test that large simulation completes in reasonable time."""
        import time

        start = time.time()
        result = MonteCarloSimulator.simulate_returns_bootstrap(
            sample_returns, num_simulations=10000, num_periods=252, random_seed=42
        )
        elapsed = time.time() - start

        assert elapsed < 10.0  # Should complete in under 10 seconds
        assert result.simulated_paths.shape == (10000, 253)
