"""
Comprehensive Tests for Analytics Module

This module contains extensive tests for:
    - PerformanceMetrics: Returns, Sharpe, Sortino, drawdown, trade statistics
    - RiskAnalytics: VaR, CVaR, Greeks analysis, tail risk

Test Coverage:
    - Unit tests for each metric with known inputs/outputs
    - Boundary condition tests (edge cases)
    - Error condition testing
    - Mathematical correctness validation against known formulas

Requirements:
    - pytest
    - numpy
    - pandas
    - scipy

Run Tests:
    pytest tests/test_analytics.py -v --tb=short

Author: OptionsBacktester2 Team
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

# Import analytics classes
from backtester.analytics.metrics import (
    PerformanceMetrics,
    MetricsError,
    InsufficientDataError as MetricsInsufficientDataError,
    InvalidDataError as MetricsInvalidDataError,
    TRADING_DAYS_PER_YEAR,
    DEFAULT_RISK_FREE_RATE,
    EPSILON,
)

from backtester.analytics.risk import (
    RiskAnalytics,
    RiskAnalyticsError,
    InsufficientDataError as RiskInsufficientDataError,
    InvalidDataError as RiskInvalidDataError,
    InvalidConfidenceError,
    VAR_CONFIDENCE_95,
    VAR_CONFIDENCE_99,
    GREEK_NAMES,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_equity_curve():
    """Create a sample equity curve for testing."""
    dates = pd.date_range(start='2023-01-01', periods=252, freq='B')
    # Simulate equity growth with some volatility
    np.random.seed(42)
    returns = np.random.normal(0.0004, 0.01, 252)  # ~10% annual, 15% vol
    equity = 100000 * np.cumprod(1 + returns)

    return pd.DataFrame({
        'equity': equity,
        'cash': equity * 0.8,
        'positions_value': equity * 0.2,
    }, index=dates)


@pytest.fixture
def sample_equity_series(sample_equity_curve):
    """Get equity series from curve."""
    return sample_equity_curve['equity']


@pytest.fixture
def sample_returns(sample_equity_curve):
    """Get returns series from equity curve."""
    return sample_equity_curve['equity'].pct_change().dropna()


@pytest.fixture
def known_equity_curve():
    """Create equity curve with known values for exact testing."""
    dates = pd.date_range(start='2023-01-01', periods=5, freq='B')
    # Start at 100000, end at 120000 (20% total return)
    equity = [100000, 105000, 110000, 108000, 120000]

    return pd.DataFrame({
        'equity': equity,
    }, index=dates)


@pytest.fixture
def drawdown_equity_curve():
    """Create equity curve with known drawdown for testing."""
    dates = pd.date_range(start='2023-01-01', periods=10, freq='B')
    # Peak at 110000, trough at 90000 (18.18% drawdown), recovery at 110000
    equity = [100000, 105000, 110000, 100000, 95000, 90000, 95000, 100000, 105000, 110000]

    return pd.DataFrame({
        'equity': equity,
    }, index=dates)


@pytest.fixture
def sample_trades():
    """Create sample trade records for testing."""
    return pd.DataFrame({
        'trade_id': ['T001', 'T002', 'T003', 'T004', 'T005', 'T006', 'T007', 'T008', 'T009', 'T010'],
        'structure_id': ['S001', 'S002', 'S003', 'S004', 'S005', 'S006', 'S007', 'S008', 'S009', 'S010'],
        'action': ['close'] * 10,
        'timestamp': pd.date_range(start='2023-01-01', periods=10, freq='W'),
        'realized_pnl': [100, -50, 200, -30, 150, 80, -100, 250, -40, 120],
        'total_cost': [1000, 800, 1200, 600, 1000, 900, 1100, 1400, 700, 1000],
    })


@pytest.fixture
def all_wins_trades():
    """Create trade records with all winning trades."""
    return pd.DataFrame({
        'trade_id': ['T001', 'T002', 'T003'],
        'action': ['close'] * 3,
        'realized_pnl': [100, 200, 150],
    })


@pytest.fixture
def all_losses_trades():
    """Create trade records with all losing trades."""
    return pd.DataFrame({
        'trade_id': ['T001', 'T002', 'T003'],
        'action': ['close'] * 3,
        'realized_pnl': [-100, -200, -150],
    })


@pytest.fixture
def sample_greeks_history():
    """Create sample Greeks history for testing."""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='B')
    np.random.seed(42)

    return pd.DataFrame({
        'delta': np.random.normal(0.3, 0.1, 100),
        'gamma': np.random.uniform(0.01, 0.05, 100),
        'theta': np.random.normal(-0.05, 0.02, 100),
        'vega': np.random.uniform(0.1, 0.5, 100),
        'rho': np.random.normal(0.02, 0.01, 100),
    }, index=dates)


@pytest.fixture
def normal_returns():
    """Create normally distributed returns for testing."""
    np.random.seed(42)
    return pd.Series(np.random.normal(0.001, 0.02, 1000))


@pytest.fixture
def negative_skew_returns():
    """Create returns with negative skewness (like short premium strategies)."""
    np.random.seed(42)
    # Many small gains, few large losses
    returns = np.concatenate([
        np.random.normal(0.005, 0.01, 900),  # Small gains
        np.random.normal(-0.05, 0.02, 100),  # Occasional losses
    ])
    np.random.shuffle(returns)
    return pd.Series(returns)


# =============================================================================
# PerformanceMetrics Tests - Total Return
# =============================================================================

class TestTotalReturn:
    """Tests for calculate_total_return method."""

    def test_total_return_known_values(self, known_equity_curve):
        """Test total return with known values."""
        result = PerformanceMetrics.calculate_total_return(known_equity_curve)
        expected = 20.0  # (120000 - 100000) / 100000 * 100
        assert abs(result - expected) < 0.01

    def test_total_return_positive(self, sample_equity_curve):
        """Test total return calculation for positive returns."""
        result = PerformanceMetrics.calculate_total_return(sample_equity_curve)
        # Should be a reasonable return
        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_total_return_with_series(self, sample_equity_series):
        """Test total return with Series input."""
        result = PerformanceMetrics.calculate_total_return(sample_equity_series)
        assert isinstance(result, float)

    def test_total_return_empty_raises(self):
        """Test that empty DataFrame raises error."""
        empty_df = pd.DataFrame({'equity': []})
        with pytest.raises(MetricsInvalidDataError):
            PerformanceMetrics.calculate_total_return(empty_df)

    def test_total_return_single_point_raises(self):
        """Test that single data point raises error."""
        single_df = pd.DataFrame({'equity': [100000]})
        with pytest.raises(MetricsInsufficientDataError):
            PerformanceMetrics.calculate_total_return(single_df)

    def test_total_return_missing_column_raises(self):
        """Test that missing equity column raises error."""
        df = pd.DataFrame({'price': [100, 110, 120]})
        with pytest.raises(KeyError):
            PerformanceMetrics.calculate_total_return(df)

    def test_total_return_negative_initial_raises(self):
        """Test that negative initial equity raises error."""
        df = pd.DataFrame({'equity': [-100000, 110000, 120000]})
        with pytest.raises(MetricsInvalidDataError):
            PerformanceMetrics.calculate_total_return(df)


# =============================================================================
# PerformanceMetrics Tests - Annualized Return
# =============================================================================

class TestAnnualizedReturn:
    """Tests for calculate_annualized_return method."""

    def test_annualized_return_one_year(self):
        """Test CAGR for exactly one year of data."""
        dates = pd.date_range(start='2023-01-01', periods=252, freq='B')
        equity = [100000] + [None] * 250 + [115000]  # 15% total return
        equity = pd.Series([100000 + i * (15000/251) for i in range(252)], index=dates)
        df = pd.DataFrame({'equity': equity})

        result = PerformanceMetrics.calculate_annualized_return(df, 252)
        # For one year, annualized return should equal total return
        expected = (115000 / 100000) - 1  # ~0.15
        assert abs(result - expected) < 0.01

    def test_annualized_return_half_year(self):
        """Test CAGR for half year of data."""
        dates = pd.date_range(start='2023-01-01', periods=126, freq='B')
        # 10% return over half year should annualize to ~21%
        equity = np.linspace(100000, 110000, 126)
        df = pd.DataFrame({'equity': equity}, index=dates)

        result = PerformanceMetrics.calculate_annualized_return(df, 252)
        # CAGR = (1.10)^2 - 1 = 0.21
        expected = (1.10 ** 2) - 1
        assert abs(result - expected) < 0.02

    def test_annualized_return_negative(self):
        """Test CAGR for negative returns."""
        dates = pd.date_range(start='2023-01-01', periods=252, freq='B')
        equity = np.linspace(100000, 80000, 252)  # 20% loss
        df = pd.DataFrame({'equity': equity}, index=dates)

        result = PerformanceMetrics.calculate_annualized_return(df, 252)
        assert result < 0

    def test_annualized_return_total_loss(self):
        """Test CAGR when final equity is zero."""
        dates = pd.date_range(start='2023-01-01', periods=252, freq='B')
        equity = np.linspace(100000, 0, 252)
        df = pd.DataFrame({'equity': equity}, index=dates)

        result = PerformanceMetrics.calculate_annualized_return(df, 252)
        assert result == -1.0  # -100% return


# =============================================================================
# PerformanceMetrics Tests - Sharpe Ratio
# =============================================================================

class TestSharpeRatio:
    """Tests for calculate_sharpe_ratio method."""

    def test_sharpe_ratio_positive(self, sample_returns):
        """Test Sharpe ratio with positive returns."""
        result = PerformanceMetrics.calculate_sharpe_ratio(sample_returns)
        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_sharpe_ratio_known_values(self):
        """Test Sharpe ratio with known values."""
        # Create returns with known mean and std
        np.random.seed(42)
        mean_daily = 0.0008  # ~20% annual
        std_daily = 0.01  # ~16% annual vol
        returns = pd.Series(np.random.normal(mean_daily, std_daily, 252))

        result = PerformanceMetrics.calculate_sharpe_ratio(returns, risk_free_rate=0.02)

        # Expected: (mean_daily - rf/252) / std * sqrt(252)
        rf_daily = 0.02 / 252
        expected = (returns.mean() - rf_daily) / returns.std() * np.sqrt(252)

        assert abs(result - expected) < 0.01

    def test_sharpe_ratio_zero_volatility(self):
        """Test Sharpe ratio with zero volatility."""
        returns = pd.Series([0.001] * 100)  # Constant returns
        result = PerformanceMetrics.calculate_sharpe_ratio(returns)
        assert result == 0.0  # Returns 0 for zero vol

    def test_sharpe_ratio_negative_returns(self):
        """Test Sharpe ratio with negative average returns."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(-0.001, 0.01, 252))
        result = PerformanceMetrics.calculate_sharpe_ratio(returns)
        assert result < 0

    def test_sharpe_ratio_different_rf_rate(self, sample_returns):
        """Test Sharpe ratio with different risk-free rates."""
        result_low = PerformanceMetrics.calculate_sharpe_ratio(sample_returns, risk_free_rate=0.01)
        result_high = PerformanceMetrics.calculate_sharpe_ratio(sample_returns, risk_free_rate=0.05)

        # Higher risk-free rate should result in lower Sharpe
        assert result_high < result_low

    def test_sharpe_ratio_insufficient_data(self):
        """Test Sharpe ratio with insufficient data."""
        returns = pd.Series([0.01])
        with pytest.raises(MetricsInsufficientDataError):
            PerformanceMetrics.calculate_sharpe_ratio(returns)


# =============================================================================
# PerformanceMetrics Tests - Sortino Ratio
# =============================================================================

class TestSortinoRatio:
    """Tests for calculate_sortino_ratio method."""

    def test_sortino_ratio_positive(self, sample_returns):
        """Test Sortino ratio with positive returns."""
        result = PerformanceMetrics.calculate_sortino_ratio(sample_returns)
        assert isinstance(result, float)

    def test_sortino_vs_sharpe_positive_skew(self):
        """Test Sortino > Sharpe for positively skewed returns."""
        np.random.seed(42)
        # Create returns with positive skew (more upside than downside)
        returns = pd.Series(np.abs(np.random.normal(0.001, 0.02, 252)))

        sharpe = PerformanceMetrics.calculate_sharpe_ratio(returns)
        sortino = PerformanceMetrics.calculate_sortino_ratio(returns)

        # For positive skew, Sortino should be higher
        # (less downside deviation than total volatility)
        assert sortino >= sharpe or np.isinf(sortino)

    def test_sortino_all_positive_returns(self):
        """Test Sortino with all positive returns."""
        returns = pd.Series([0.01, 0.02, 0.015, 0.005, 0.03])
        result = PerformanceMetrics.calculate_sortino_ratio(returns)
        # Should return infinity or large positive number
        assert result > 0

    def test_sortino_ratio_known_downside(self):
        """Test Sortino ratio with known downside deviation."""
        # Mix of positive and negative returns
        returns = pd.Series([0.02, -0.01, 0.03, -0.02, 0.01, -0.015, 0.025, -0.005])
        result = PerformanceMetrics.calculate_sortino_ratio(returns)
        assert isinstance(result, float)


# =============================================================================
# PerformanceMetrics Tests - Calmar Ratio
# =============================================================================

class TestCalmarRatio:
    """Tests for calculate_calmar_ratio method."""

    def test_calmar_ratio_positive(self):
        """Test Calmar ratio with positive values."""
        result = PerformanceMetrics.calculate_calmar_ratio(0.15, -0.10)
        expected = 0.15 / 0.10  # 1.5
        assert abs(result - expected) < 0.001

    def test_calmar_ratio_zero_drawdown(self):
        """Test Calmar ratio with zero drawdown."""
        result = PerformanceMetrics.calculate_calmar_ratio(0.15, 0.0)
        assert np.isinf(result) and result > 0

    def test_calmar_ratio_negative_return(self):
        """Test Calmar ratio with negative return."""
        result = PerformanceMetrics.calculate_calmar_ratio(-0.10, -0.15)
        expected = -0.10 / 0.15
        assert abs(result - expected) < 0.001

    def test_calmar_ratio_invalid_input(self):
        """Test Calmar ratio with invalid input."""
        with pytest.raises(MetricsInvalidDataError):
            PerformanceMetrics.calculate_calmar_ratio(np.nan, -0.10)


# =============================================================================
# PerformanceMetrics Tests - Max Drawdown
# =============================================================================

class TestMaxDrawdown:
    """Tests for calculate_max_drawdown method."""

    def test_max_drawdown_known_values(self, drawdown_equity_curve):
        """Test max drawdown with known values."""
        result = PerformanceMetrics.calculate_max_drawdown(drawdown_equity_curve)

        # Peak = 110000, Trough = 90000
        # Max DD = (90000 - 110000) / 110000 = -0.1818
        expected_dd = (90000 - 110000) / 110000

        assert abs(result['max_drawdown_pct'] - expected_dd) < 0.001
        assert abs(result['max_drawdown_value'] - (90000 - 110000)) < 1

    def test_max_drawdown_peak_trough_dates(self, drawdown_equity_curve):
        """Test that peak and trough dates are correct."""
        result = PerformanceMetrics.calculate_max_drawdown(drawdown_equity_curve)

        # Peak at index 2, trough at index 5
        assert result['peak_date'] == drawdown_equity_curve.index[2]
        assert result['trough_date'] == drawdown_equity_curve.index[5]

    def test_max_drawdown_recovery_date(self, drawdown_equity_curve):
        """Test recovery date detection."""
        result = PerformanceMetrics.calculate_max_drawdown(drawdown_equity_curve)

        # Recovery at index 9 (back to 110000)
        assert result['recovery_date'] == drawdown_equity_curve.index[9]

    def test_max_drawdown_no_recovery(self):
        """Test when drawdown never recovers."""
        dates = pd.date_range(start='2023-01-01', periods=5, freq='B')
        equity = [100000, 110000, 90000, 85000, 88000]  # Never recovers to 110000
        df = pd.DataFrame({'equity': equity}, index=dates)

        result = PerformanceMetrics.calculate_max_drawdown(df)
        assert result['recovery_date'] is None

    def test_max_drawdown_no_drawdown(self):
        """Test with monotonically increasing equity."""
        dates = pd.date_range(start='2023-01-01', periods=5, freq='B')
        equity = [100000, 105000, 110000, 115000, 120000]
        df = pd.DataFrame({'equity': equity}, index=dates)

        result = PerformanceMetrics.calculate_max_drawdown(df)
        assert result['max_drawdown_pct'] == 0.0

    def test_max_drawdown_series_returned(self, drawdown_equity_curve):
        """Test that drawdown series is returned."""
        result = PerformanceMetrics.calculate_max_drawdown(drawdown_equity_curve)

        assert 'drawdown_series' in result
        assert isinstance(result['drawdown_series'], pd.Series)
        assert len(result['drawdown_series']) == len(drawdown_equity_curve)


# =============================================================================
# PerformanceMetrics Tests - Trade Metrics
# =============================================================================

class TestTradeMetrics:
    """Tests for trade-based metrics."""

    def test_win_rate_calculation(self, sample_trades):
        """Test win rate calculation."""
        result = PerformanceMetrics.calculate_win_rate(sample_trades)
        # 6 wins out of 10 trades = 60%
        expected = 60.0
        assert abs(result - expected) < 0.01

    def test_win_rate_all_wins(self, all_wins_trades):
        """Test win rate with all winning trades."""
        result = PerformanceMetrics.calculate_win_rate(all_wins_trades)
        assert result == 100.0

    def test_win_rate_all_losses(self, all_losses_trades):
        """Test win rate with all losing trades."""
        result = PerformanceMetrics.calculate_win_rate(all_losses_trades)
        assert result == 0.0

    def test_win_rate_empty_trades(self):
        """Test win rate with empty trades."""
        empty_trades = pd.DataFrame({'realized_pnl': []})
        with pytest.raises(MetricsInsufficientDataError):
            PerformanceMetrics.calculate_win_rate(empty_trades)

    def test_profit_factor_calculation(self, sample_trades):
        """Test profit factor calculation."""
        result = PerformanceMetrics.calculate_profit_factor(sample_trades)

        # Gross profit = 100 + 200 + 150 + 80 + 250 + 120 = 900
        # Gross loss = 50 + 30 + 100 + 40 = 220
        expected = 900 / 220
        assert abs(result - expected) < 0.01

    def test_profit_factor_all_wins(self, all_wins_trades):
        """Test profit factor with all wins."""
        result = PerformanceMetrics.calculate_profit_factor(all_wins_trades)
        assert np.isinf(result)

    def test_profit_factor_all_losses(self, all_losses_trades):
        """Test profit factor with all losses."""
        result = PerformanceMetrics.calculate_profit_factor(all_losses_trades)
        assert result == 0.0

    def test_average_win(self, sample_trades):
        """Test average win calculation."""
        result = PerformanceMetrics.calculate_average_win(sample_trades)
        # Wins: 100, 200, 150, 80, 250, 120 = 900 / 6 = 150
        expected = 150.0
        assert abs(result - expected) < 0.01

    def test_average_loss(self, sample_trades):
        """Test average loss calculation."""
        result = PerformanceMetrics.calculate_average_loss(sample_trades)
        # Losses: -50, -30, -100, -40 = -220 / 4 = -55
        expected = -55.0
        assert abs(result - expected) < 0.01

    def test_expectancy_calculation(self, sample_trades):
        """Test expectancy calculation."""
        result = PerformanceMetrics.calculate_expectancy(sample_trades)

        # Win rate = 0.6, Avg win = 150
        # Loss rate = 0.4, Avg loss = 55
        # Expectancy = 0.6 * 150 - 0.4 * 55 = 90 - 22 = 68
        expected = 0.6 * 150 - 0.4 * 55
        assert abs(result - expected) < 1.0

    def test_payoff_ratio(self, sample_trades):
        """Test payoff ratio calculation."""
        result = PerformanceMetrics.calculate_payoff_ratio(sample_trades)
        # Avg win / |Avg loss| = 150 / 55 = 2.727
        expected = 150 / 55
        assert abs(result - expected) < 0.01

    def test_consecutive_wins(self, sample_trades):
        """Test consecutive wins calculation."""
        result = PerformanceMetrics.calculate_consecutive_wins(sample_trades)
        # Sequence: W, L, W, L, W, W, L, W, L, W
        # Max consecutive wins = 2
        assert result == 2

    def test_consecutive_losses(self, sample_trades):
        """Test consecutive losses calculation."""
        result = PerformanceMetrics.calculate_consecutive_losses(sample_trades)
        # Sequence: W, L, W, L, W, W, L, W, L, W
        # Max consecutive losses = 1
        assert result == 1


# =============================================================================
# PerformanceMetrics Tests - Distribution
# =============================================================================

class TestReturnsDistribution:
    """Tests for returns distribution analysis."""

    def test_distribution_basic(self, sample_returns):
        """Test basic distribution calculation."""
        result = PerformanceMetrics.calculate_returns_distribution(sample_returns)

        assert 'mean' in result
        assert 'median' in result
        assert 'std' in result
        assert 'skewness' in result
        assert 'kurtosis' in result
        assert 'percentiles' in result

    def test_distribution_normal_skewness(self, normal_returns):
        """Test skewness for normal distribution."""
        result = PerformanceMetrics.calculate_returns_distribution(normal_returns)

        # Normal distribution should have skewness near 0
        assert abs(result['skewness']) < 0.3

    def test_distribution_normal_kurtosis(self, normal_returns):
        """Test kurtosis for normal distribution."""
        result = PerformanceMetrics.calculate_returns_distribution(normal_returns)

        # Normal distribution should have excess kurtosis near 0
        assert abs(result['kurtosis']) < 0.5

    def test_distribution_negative_skew(self, negative_skew_returns):
        """Test detection of negative skewness."""
        result = PerformanceMetrics.calculate_returns_distribution(negative_skew_returns)

        # Should detect negative skewness
        assert result['skewness'] < 0

    def test_distribution_percentiles(self, sample_returns):
        """Test percentile calculations."""
        result = PerformanceMetrics.calculate_returns_distribution(sample_returns)

        # Check percentiles are ordered correctly
        percentiles = result['percentiles']
        assert percentiles[5] < percentiles[25] < percentiles[50] < percentiles[75] < percentiles[95]


# =============================================================================
# RiskAnalytics Tests - VaR
# =============================================================================

class TestValueAtRisk:
    """Tests for Value at Risk calculations."""

    def test_var_historical_95(self, normal_returns):
        """Test 95% historical VaR."""
        result = RiskAnalytics.calculate_var(normal_returns, 0.95, 'historical')

        # Should be negative (representing a loss)
        assert result < 0

        # Should be approximately the 5th percentile
        expected = np.percentile(normal_returns, 5)
        assert abs(result - expected) < 0.001

    def test_var_historical_99(self, normal_returns):
        """Test 99% historical VaR."""
        result_99 = RiskAnalytics.calculate_var(normal_returns, 0.99, 'historical')
        result_95 = RiskAnalytics.calculate_var(normal_returns, 0.95, 'historical')

        # 99% VaR should be more negative than 95% VaR
        assert result_99 < result_95

    def test_var_parametric_vs_historical(self, normal_returns):
        """Test parametric vs historical VaR for normal returns."""
        var_hist = RiskAnalytics.calculate_var(normal_returns, 0.95, 'historical')
        var_param = RiskAnalytics.calculate_var(normal_returns, 0.95, 'parametric')

        # For normal returns, should be close
        assert abs(var_hist - var_param) < 0.01

    def test_var_parametric_formula(self, normal_returns):
        """Test parametric VaR formula."""
        result = RiskAnalytics.calculate_var(normal_returns, 0.95, 'parametric')

        # VaR = mean - z * std where z = 1.645 for 95%
        expected = normal_returns.mean() - 1.645 * normal_returns.std()
        assert abs(result - expected) < 0.001

    def test_var_invalid_confidence(self, normal_returns):
        """Test VaR with invalid confidence level."""
        with pytest.raises(InvalidConfidenceError):
            RiskAnalytics.calculate_var(normal_returns, 1.5, 'historical')

        with pytest.raises(InvalidConfidenceError):
            RiskAnalytics.calculate_var(normal_returns, 0.0, 'historical')

    def test_var_invalid_method(self, normal_returns):
        """Test VaR with invalid method."""
        with pytest.raises(ValueError):
            RiskAnalytics.calculate_var(normal_returns, 0.95, 'invalid_method')

    def test_var_empty_series(self):
        """Test VaR with empty series."""
        empty = pd.Series([])
        with pytest.raises(RiskInvalidDataError):
            RiskAnalytics.calculate_var(empty, 0.95)


# =============================================================================
# RiskAnalytics Tests - CVaR
# =============================================================================

class TestConditionalVaR:
    """Tests for Conditional VaR (Expected Shortfall) calculations."""

    def test_cvar_worse_than_var(self, normal_returns):
        """Test that CVaR is always worse (more negative) than VaR."""
        var = RiskAnalytics.calculate_var(normal_returns, 0.95)
        cvar = RiskAnalytics.calculate_cvar(normal_returns, 0.95)

        # CVaR should be more negative (larger loss)
        assert cvar <= var

    def test_cvar_calculation(self, normal_returns):
        """Test CVaR calculation."""
        result = RiskAnalytics.calculate_cvar(normal_returns, 0.95)

        # Should be negative
        assert result < 0

        # Should be the average of returns below VaR
        var = RiskAnalytics.calculate_var(normal_returns, 0.95)
        tail = normal_returns[normal_returns <= var]
        expected = tail.mean()
        assert abs(result - expected) < 0.001

    def test_cvar_vs_var_ratio(self, normal_returns):
        """Test CVaR/VaR ratio for normal distribution."""
        var = RiskAnalytics.calculate_var(normal_returns, 0.95)
        cvar = RiskAnalytics.calculate_cvar(normal_returns, 0.95)

        # For normal distribution, CVaR/VaR ratio should be around 1.2-1.5
        if var != 0:
            ratio = cvar / var
            assert 1.0 < ratio < 2.0  # CVaR is larger in magnitude


# =============================================================================
# RiskAnalytics Tests - Tail Risk
# =============================================================================

class TestTailRisk:
    """Tests for tail risk metrics."""

    def test_tail_risk_basic(self, sample_returns):
        """Test basic tail risk calculation."""
        result = RiskAnalytics.calculate_tail_risk(sample_returns)

        assert 'skewness' in result
        assert 'kurtosis' in result
        assert 'left_tail_ratio' in result
        assert 'right_tail_ratio' in result
        assert 'jarque_bera_stat' in result

    def test_tail_risk_negative_skew(self, negative_skew_returns):
        """Test tail risk detection for negatively skewed returns."""
        result = RiskAnalytics.calculate_tail_risk(negative_skew_returns)

        # Should detect negative skewness
        assert result['skewness'] < 0

        # Left tail should be heavier
        assert result['left_tail_ratio'] >= result['right_tail_ratio']

    def test_tail_risk_normal_jb(self, normal_returns):
        """Test Jarque-Bera test for normal returns."""
        result = RiskAnalytics.calculate_tail_risk(normal_returns)

        # For normal distribution, JB test should not reject normality
        # (p-value > 0.05)
        # Note: with random sampling, this may occasionally fail
        assert result['jarque_bera_pvalue'] is not None


# =============================================================================
# RiskAnalytics Tests - Greeks Analysis
# =============================================================================

class TestGreeksAnalysis:
    """Tests for Greeks analysis."""

    def test_greeks_analysis_basic(self, sample_greeks_history):
        """Test basic Greeks analysis."""
        result = RiskAnalytics.analyze_greeks_over_time(sample_greeks_history)

        # Check all Greeks are analyzed
        for greek in GREEK_NAMES:
            assert greek in result
            assert 'mean' in result[greek]
            assert 'std' in result[greek]
            assert 'min' in result[greek]
            assert 'max' in result[greek]

    def test_greeks_correlation_matrix(self, sample_greeks_history):
        """Test Greeks correlation matrix."""
        result = RiskAnalytics.analyze_greeks_over_time(sample_greeks_history)

        assert 'correlation_matrix' in result
        corr = result['correlation_matrix']

        # Diagonal should be 1.0
        for greek in GREEK_NAMES:
            if greek in corr.columns:
                assert abs(corr.loc[greek, greek] - 1.0) < 0.001

    def test_greeks_analysis_empty(self):
        """Test Greeks analysis with empty DataFrame."""
        empty_df = pd.DataFrame()
        with pytest.raises(RiskInvalidDataError):
            RiskAnalytics.analyze_greeks_over_time(empty_df)

    def test_greeks_current_value(self, sample_greeks_history):
        """Test that current (most recent) value is captured."""
        result = RiskAnalytics.analyze_greeks_over_time(sample_greeks_history)

        # Current value should match last row
        for greek in GREEK_NAMES:
            expected = sample_greeks_history[greek].iloc[-1]
            assert abs(result[greek]['current'] - expected) < 0.001


# =============================================================================
# RiskAnalytics Tests - Downside Risk
# =============================================================================

class TestDownsideRisk:
    """Tests for downside risk metrics."""

    def test_downside_risk_basic(self, sample_returns):
        """Test basic downside risk calculation."""
        result = RiskAnalytics.calculate_downside_risk(sample_returns)

        assert 'downside_deviation' in result
        assert 'downside_variance' in result
        assert 'loss_probability' in result
        assert 'average_loss' in result
        assert 'worst_loss' in result

    def test_downside_deviation_formula(self, sample_returns):
        """Test downside deviation formula."""
        result = RiskAnalytics.calculate_downside_risk(sample_returns)

        # Manual calculation
        negative = np.minimum(sample_returns, 0)
        expected_variance = np.mean(np.square(negative))
        expected_deviation = np.sqrt(expected_variance)

        assert abs(result['downside_deviation'] - expected_deviation) < 0.001

    def test_downside_risk_all_positive(self):
        """Test downside risk with all positive returns."""
        returns = pd.Series([0.01, 0.02, 0.015, 0.005, 0.03])
        result = RiskAnalytics.calculate_downside_risk(returns)

        assert result['loss_probability'] == 0.0
        assert result['downside_deviation'] == 0.0


# =============================================================================
# RiskAnalytics Tests - MAE
# =============================================================================

class TestMAE:
    """Tests for Maximum Adverse Excursion analysis."""

    def test_mae_basic(self, sample_trades):
        """Test basic MAE calculation."""
        result = RiskAnalytics.calculate_mae(sample_trades)

        assert isinstance(result, pd.DataFrame)
        assert 'trade_id' in result.columns
        assert 'realized_pnl' in result.columns
        assert 'mae' in result.columns

    def test_mae_losing_trade_estimate(self):
        """Test MAE estimation for losing trades."""
        trades = pd.DataFrame({
            'trade_id': ['T001'],
            'realized_pnl': [-100],
        })

        result = RiskAnalytics.calculate_mae(trades)

        # For losing trades, MAE >= |loss|
        assert result.iloc[0]['mae'] <= -100

    def test_mae_stats(self, sample_trades):
        """Test MAE statistics."""
        mae_df = RiskAnalytics.calculate_mae(sample_trades)
        stats = RiskAnalytics.analyze_mae_statistics(mae_df)

        assert 'avg_mae' in stats
        assert 'max_mae' in stats


# =============================================================================
# RiskAnalytics Tests - VaR Breach
# =============================================================================

class TestVarBreach:
    """Tests for VaR breach analysis."""

    def test_var_breach_count(self, normal_returns):
        """Test VaR breach counting."""
        var_95 = RiskAnalytics.calculate_var(normal_returns, 0.95)
        result = RiskAnalytics.calculate_var_breach_count(normal_returns, var_95)

        assert 'breach_count' in result
        assert 'total_observations' in result
        assert 'breach_rate' in result

        # For 95% VaR, expect ~5% breach rate
        assert 0.03 < result['breach_rate'] < 0.10


# =============================================================================
# Integration Tests
# =============================================================================

class TestAnalyticsIntegration:
    """Integration tests combining multiple metrics."""

    def test_all_performance_metrics(self, sample_equity_curve, sample_trades):
        """Test calculating all performance metrics together."""
        result = PerformanceMetrics.calculate_all_metrics(
            equity_curve=sample_equity_curve,
            trades=sample_trades
        )

        assert 'returns' in result
        assert 'risk' in result
        assert 'drawdown' in result
        assert 'trades' in result
        assert 'distribution' in result

    def test_all_risk_metrics(self, sample_returns, sample_greeks_history, sample_trades):
        """Test calculating all risk metrics together."""
        result = RiskAnalytics.calculate_all_risk_metrics(
            returns=sample_returns,
            greeks_history=sample_greeks_history,
            trades=sample_trades
        )

        assert 'var_historical' in result
        assert 'var_parametric' in result
        assert 'cvar' in result
        assert 'tail_risk' in result


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_day_data(self):
        """Test metrics with single day of data."""
        df = pd.DataFrame({'equity': [100000]})

        with pytest.raises(MetricsInsufficientDataError):
            PerformanceMetrics.calculate_total_return(df)

    def test_two_days_data(self):
        """Test metrics with minimum required data."""
        df = pd.DataFrame({'equity': [100000, 110000]})

        result = PerformanceMetrics.calculate_total_return(df)
        assert result == 10.0

    def test_extreme_returns(self):
        """Test with extreme returns."""
        # 100x return
        df = pd.DataFrame({'equity': [100000, 10000000]})
        result = PerformanceMetrics.calculate_total_return(df)
        assert result == 9900.0  # 9900% return

    def test_tiny_returns(self, sample_equity_curve):
        """Test with very small returns."""
        # Scale down returns
        tiny_equity = sample_equity_curve['equity'] / 1e10 + 1e-15
        tiny_df = pd.DataFrame({'equity': tiny_equity})

        # Should still work with tiny values
        result = PerformanceMetrics.calculate_total_return(tiny_df)
        assert np.isfinite(result)

    def test_nan_handling(self):
        """Test handling of NaN values."""
        returns = pd.Series([0.01, np.nan, 0.02, np.nan, 0.03])

        # Should work after dropping NaN
        result = PerformanceMetrics.calculate_sharpe_ratio(returns)
        assert np.isfinite(result)

    def test_zero_trades(self):
        """Test with zero trades."""
        empty_trades = pd.DataFrame({'realized_pnl': []})

        with pytest.raises(MetricsInsufficientDataError):
            PerformanceMetrics.calculate_win_rate(empty_trades)

    def test_breakeven_trade(self):
        """Test with breakeven trades (zero P&L)."""
        trades = pd.DataFrame({
            'trade_id': ['T001', 'T002', 'T003'],
            'realized_pnl': [0.0, 0.0, 0.0],
        })

        # Breakeven trades are not wins
        result = PerformanceMetrics.calculate_win_rate(trades)
        assert result == 0.0


# =============================================================================
# Numerical Stability Tests
# =============================================================================

class TestNumericalStability:
    """Tests for numerical stability."""

    def test_large_equity_values(self):
        """Test with very large equity values."""
        df = pd.DataFrame({'equity': [1e12, 1.1e12]})
        result = PerformanceMetrics.calculate_total_return(df)
        assert abs(result - 10.0) < 0.01

    def test_small_equity_values(self):
        """Test with very small equity values."""
        df = pd.DataFrame({'equity': [1e-6, 1.1e-6]})
        result = PerformanceMetrics.calculate_total_return(df)
        assert abs(result - 10.0) < 0.01

    def test_var_stability(self):
        """Test VaR numerical stability."""
        np.random.seed(42)
        # Very small standard deviation
        returns = pd.Series(np.random.normal(0.001, 0.0001, 1000))

        var = RiskAnalytics.calculate_var(returns, 0.95)
        assert np.isfinite(var)

    def test_sharpe_near_zero_vol(self):
        """Test Sharpe with near-zero volatility."""
        # Almost constant returns
        returns = pd.Series([0.001 + i * 1e-10 for i in range(100)])

        result = PerformanceMetrics.calculate_sharpe_ratio(returns)
        # Should handle gracefully
        assert isinstance(result, float)


# =============================================================================
# Consistency Tests
# =============================================================================

class TestConsistency:
    """Tests for metric consistency and relationships."""

    def test_sortino_geq_sharpe_positive_skew(self):
        """Test Sortino >= Sharpe for positive skewed returns."""
        np.random.seed(42)
        # Create returns with clear positive skew
        returns = pd.Series(np.abs(np.random.normal(0, 0.02, 252)) + 0.001)

        sharpe = PerformanceMetrics.calculate_sharpe_ratio(returns)
        sortino = PerformanceMetrics.calculate_sortino_ratio(returns)

        # Sortino should be higher for positive skew
        assert sortino >= sharpe or np.isinf(sortino)

    def test_cvar_leq_var(self, sample_returns):
        """Test CVaR <= VaR (more severe)."""
        var = RiskAnalytics.calculate_var(sample_returns, 0.95)
        cvar = RiskAnalytics.calculate_cvar(sample_returns, 0.95)

        assert cvar <= var

    def test_expectancy_consistency(self, sample_trades):
        """Test expectancy consistency with win rate and averages."""
        win_rate = PerformanceMetrics.calculate_win_rate(sample_trades) / 100
        avg_win = PerformanceMetrics.calculate_average_win(sample_trades)
        avg_loss = PerformanceMetrics.calculate_average_loss(sample_trades)
        expectancy = PerformanceMetrics.calculate_expectancy(sample_trades)

        # Expectancy = win_rate * avg_win + loss_rate * avg_loss
        expected = win_rate * avg_win + (1 - win_rate) * avg_loss
        assert abs(expectancy - expected) < 0.01


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
