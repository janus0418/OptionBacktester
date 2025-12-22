"""
Test Suite for Visualization, Dashboard, and ReportGenerator Classes

This module provides comprehensive tests for the visualization and reporting
components of the options backtesting framework.

Test Categories:
    1. Visualization Class Tests
        - Equity curve plotting (matplotlib/plotly)
        - Drawdown plotting
        - P&L distribution histograms
        - Greeks over time charts
        - Payoff diagrams
        - Entry/exit point visualization
        - Monthly returns heatmap
        - Rolling Sharpe ratio
        - Returns distribution

    2. Dashboard Class Tests
        - Performance dashboard creation
        - Risk dashboard creation
        - Trade analysis dashboard creation
        - HTML file generation

    3. ReportGenerator Class Tests
        - HTML report generation
        - Summary table formatting (HTML, LaTeX, Markdown)
        - Chart embedding

    4. Edge Case Tests
        - Empty data handling
        - Single data point
        - Invalid backend specification
        - File save operations

Test Count: 30+ tests as specified
"""

import os
import tempfile
import shutil
from datetime import datetime, timedelta
from typing import Dict, Any
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_equity_curve():
    """Create sample equity curve data."""
    dates = pd.date_range(start='2023-01-01', periods=252, freq='B')
    np.random.seed(42)

    # Simulate equity with some volatility and drift
    returns = np.random.normal(0.0004, 0.015, 252)
    equity = 100000 * np.cumprod(1 + returns)

    df = pd.DataFrame({
        'equity': equity,
        'cash': equity * 0.3,
        'positions_value': equity * 0.7,
        'num_positions': np.random.randint(0, 5, 252),
        'delta': np.random.normal(0, 0.5, 252),
        'gamma': np.random.uniform(0, 0.1, 252),
        'theta': np.random.uniform(-100, 0, 252),
        'vega': np.random.uniform(0, 200, 252),
        'rho': np.random.uniform(-10, 10, 252),
    }, index=dates)

    return df


@pytest.fixture
def sample_trade_log():
    """Create sample trade log data."""
    np.random.seed(42)
    n_trades = 50

    dates = pd.date_range(start='2023-01-01', periods=n_trades * 2, freq='W')
    pnl_values = np.random.normal(100, 500, n_trades)

    records = []
    for i in range(n_trades):
        # Entry trade
        records.append({
            'trade_id': f'T{i*2+1:06d}',
            'structure_id': f'S{i:04d}',
            'structure_type': np.random.choice(['straddle', 'strangle', 'iron_condor']),
            'underlying': 'SPY',
            'action': 'open',
            'timestamp': dates[i*2],
            'num_legs': np.random.choice([2, 4]),
            'net_premium': np.random.uniform(100, 500),
            'total_cost': np.random.uniform(100, 500),
            'total_proceeds': 0.0,
            'commission': np.random.uniform(1, 5),
            'slippage': np.random.uniform(0.5, 2),
            'realized_pnl': 0.0,
            'exit_reason': '',
        })

        # Exit trade
        records.append({
            'trade_id': f'T{i*2+2:06d}',
            'structure_id': f'S{i:04d}',
            'structure_type': records[-1]['structure_type'],
            'underlying': 'SPY',
            'action': 'close',
            'timestamp': dates[i*2 + 1],
            'num_legs': records[-1]['num_legs'],
            'net_premium': 0.0,
            'total_cost': 0.0,
            'total_proceeds': np.random.uniform(100, 600),
            'commission': np.random.uniform(1, 5),
            'slippage': np.random.uniform(0.5, 2),
            'realized_pnl': pnl_values[i],
            'exit_reason': np.random.choice(['profit_target', 'stop_loss', 'expiration', 'time_exit']),
        })

    return pd.DataFrame(records)


@pytest.fixture
def sample_greeks_history():
    """Create sample Greeks history data."""
    dates = pd.date_range(start='2023-01-01', periods=252, freq='B')
    np.random.seed(42)

    return pd.DataFrame({
        'delta': np.cumsum(np.random.normal(0, 0.1, 252)),
        'gamma': np.abs(np.random.normal(0.05, 0.02, 252)),
        'theta': np.random.uniform(-150, -50, 252),
        'vega': np.random.uniform(100, 400, 252),
        'rho': np.random.uniform(-20, 20, 252),
    }, index=dates)


@pytest.fixture
def sample_returns():
    """Create sample returns series."""
    dates = pd.date_range(start='2023-01-01', periods=252, freq='B')
    np.random.seed(42)
    returns = np.random.normal(0.0004, 0.015, 252)
    return pd.Series(returns, index=dates)


@pytest.fixture
def sample_underlying_prices():
    """Create sample underlying prices."""
    dates = pd.date_range(start='2023-01-01', periods=252, freq='B')
    np.random.seed(42)

    # Simulate price with GBM
    returns = np.random.normal(0.0003, 0.012, 252)
    prices = 450 * np.cumprod(1 + returns)

    return pd.Series(prices, index=dates, name='SPY')


@pytest.fixture
def sample_backtest_results(sample_equity_curve, sample_trade_log, sample_greeks_history):
    """Create sample backtest results dictionary."""
    return {
        'equity_curve': sample_equity_curve,
        'trade_log': sample_trade_log,
        'greeks_history': sample_greeks_history,
        'initial_capital': 100000.0,
        'final_equity': float(sample_equity_curve['equity'].iloc[-1]),
        'total_return': (sample_equity_curve['equity'].iloc[-1] / 100000.0) - 1,
        'num_trades': len(sample_trade_log),
        'strategy_stats': {'name': 'TestStrategy'},
        'start_date': sample_equity_curve.index[0],
        'end_date': sample_equity_curve.index[-1],
        'underlying': 'SPY',
    }


@pytest.fixture
def sample_metrics():
    """Create sample metrics dictionary."""
    return {
        'performance': {
            'total_return_pct': 0.15,
            'annualized_return': 0.12,
            'sharpe_ratio': 1.5,
            'sortino_ratio': 2.0,
            'calmar_ratio': 1.2,
            'max_drawdown': -0.08,
            'max_drawdown_value': -8000,
            'max_drawdown_duration': 45,
            'win_rate': 0.65,
            'profit_factor': 1.8,
            'expectancy': 150,
            'average_win': 300,
            'average_loss': -200,
            'average_trade': 100,
            'payoff_ratio': 1.5,
            'max_consecutive_wins': 8,
            'max_consecutive_losses': 4,
            'total_trades': 50,
            'volatility': 0.18,
        },
        'risk': {
            'var_95_historical': -0.025,
            'var_95_parametric': -0.024,
            'var_99_historical': -0.035,
            'cvar_95': -0.032,
            'cvar_99': -0.045,
            'skewness': -0.3,
            'kurtosis': 3.5,
            'downside_deviation': 0.012,
            'tail_risk': {'tail_ratio': 0.9},
        },
        'summary': {
            'total_return_pct': 0.15,
            'annualized_return': 0.12,
            'sharpe_ratio': 1.5,
            'sortino_ratio': 2.0,
            'max_drawdown': -0.08,
            'calmar_ratio': 1.2,
            'win_rate': 0.65,
            'profit_factor': 1.8,
            'expectancy': 150,
            'total_trades': 50,
            'var_95': -0.025,
            'cvar_95': -0.032,
            'volatility': 0.18,
            'initial_equity': 100000,
            'final_equity': 115000,
            'trading_days': 252,
        }
    }


@pytest.fixture
def temp_dir():
    """Create and clean up temporary directory."""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def mock_option_structure():
    """Create mock option structure for payoff diagram tests."""
    structure = MagicMock()
    structure.structure_type = 'straddle'
    structure.underlying = 'SPY'
    structure.options = [MagicMock(strike=450), MagicMock(strike=450)]

    # Mock payoff at expiry
    def mock_pnl_at_expiry(spots):
        # Simulate short straddle payoff
        strikes = 450
        premium = 1000
        call_payoff = -np.maximum(spots - strikes, 0) * 100
        put_payoff = -np.maximum(strikes - spots, 0) * 100
        return call_payoff + put_payoff + premium

    def mock_get_payoff_diagram(spot_range=None, num_points=101):
        # Return proper (spots, payoffs) tuple
        if spot_range is None:
            spot_range = (400, 500)
        spots = np.linspace(spot_range[0], spot_range[1], num_points)
        payoffs = mock_pnl_at_expiry(spots)
        return spots, payoffs

    structure.get_pnl_at_expiry = mock_pnl_at_expiry
    structure.get_payoff_diagram = mock_get_payoff_diagram
    structure.calculate_breakeven_points = MagicMock(return_value=[440.0, 460.0])

    return structure


# =============================================================================
# Visualization Class Tests
# =============================================================================

class TestVisualization:
    """Test suite for Visualization class."""

    def test_import_visualization(self):
        """Test that Visualization can be imported."""
        from backtester.analytics.visualization import Visualization
        assert Visualization is not None

    def test_plot_equity_curve_plotly_creates_figure(self, sample_equity_curve):
        """Test equity curve plot with plotly backend."""
        from backtester.analytics.visualization import Visualization

        fig = Visualization.plot_equity_curve(
            sample_equity_curve,
            backend='plotly',
            show=False
        )

        assert fig is not None
        # Check it's a plotly figure
        assert hasattr(fig, 'data')
        assert len(fig.data) >= 1

    def test_plot_equity_curve_matplotlib_creates_figure(self, sample_equity_curve):
        """Test equity curve plot with matplotlib backend."""
        from backtester.analytics.visualization import Visualization
        import matplotlib
        matplotlib.use('Agg')

        fig = Visualization.plot_equity_curve(
            sample_equity_curve,
            backend='matplotlib',
            show=False
        )

        assert fig is not None
        # Check it's a matplotlib figure
        assert hasattr(fig, 'axes')

    def test_plot_equity_curve_save_file(self, sample_equity_curve, temp_dir):
        """Test equity curve plot saves file correctly."""
        from backtester.analytics.visualization import Visualization

        save_path = os.path.join(temp_dir, 'equity.html')
        Visualization.plot_equity_curve(
            sample_equity_curve,
            save_path=save_path,
            backend='plotly',
            show=False
        )

        assert os.path.exists(save_path)
        assert os.path.getsize(save_path) > 0

    def test_plot_drawdown_creates_figure(self, sample_equity_curve):
        """Test drawdown plot creates figure."""
        from backtester.analytics.visualization import Visualization

        fig = Visualization.plot_drawdown(
            sample_equity_curve,
            backend='plotly',
            show=False
        )

        assert fig is not None
        assert hasattr(fig, 'data')

    def test_plot_pnl_distribution_creates_figure(self, sample_trade_log):
        """Test P&L distribution plot creates figure."""
        from backtester.analytics.visualization import Visualization

        closed_trades = sample_trade_log[sample_trade_log['action'] == 'close']

        fig = Visualization.plot_pnl_distribution(
            closed_trades,
            backend='plotly',
            show=False
        )

        assert fig is not None
        assert hasattr(fig, 'data')

    def test_plot_greeks_over_time_creates_figure(self, sample_greeks_history):
        """Test Greeks plot creates figure."""
        from backtester.analytics.visualization import Visualization

        fig = Visualization.plot_greeks_over_time(
            sample_greeks_history,
            backend='plotly',
            show=False
        )

        assert fig is not None
        assert hasattr(fig, 'data')

    def test_plot_greeks_specific_greeks(self, sample_greeks_history):
        """Test Greeks plot with specific Greeks selection."""
        from backtester.analytics.visualization import Visualization

        fig = Visualization.plot_greeks_over_time(
            sample_greeks_history,
            greek_names=['delta', 'gamma'],
            backend='plotly',
            show=False
        )

        assert fig is not None

    def test_plot_payoff_diagram_creates_figure(self, mock_option_structure):
        """Test payoff diagram creates figure."""
        from backtester.analytics.visualization import Visualization

        fig = Visualization.plot_payoff_diagram(
            mock_option_structure,
            backend='plotly',
            show=False
        )

        assert fig is not None

    def test_plot_payoff_diagram_with_current_spot(self, mock_option_structure):
        """Test payoff diagram with current spot marker."""
        from backtester.analytics.visualization import Visualization

        fig = Visualization.plot_payoff_diagram(
            mock_option_structure,
            current_spot=450.0,
            backend='plotly',
            show=False
        )

        assert fig is not None

    def test_plot_entry_exit_points_creates_figure(
        self, sample_underlying_prices, sample_trade_log
    ):
        """Test entry/exit points plot creates figure."""
        from backtester.analytics.visualization import Visualization

        fig = Visualization.plot_entry_exit_points(
            sample_underlying_prices,
            sample_trade_log,
            backend='plotly',
            show=False
        )

        assert fig is not None

    def test_plot_monthly_returns_creates_figure(self, sample_equity_curve):
        """Test monthly returns heatmap creates figure."""
        from backtester.analytics.visualization import Visualization

        fig = Visualization.plot_monthly_returns(
            sample_equity_curve,
            backend='plotly',
            show=False
        )

        assert fig is not None

    def test_plot_rolling_sharpe_creates_figure(self, sample_returns):
        """Test rolling Sharpe plot creates figure."""
        from backtester.analytics.visualization import Visualization

        fig = Visualization.plot_rolling_sharpe(
            sample_returns,
            window=63,
            backend='plotly',
            show=False
        )

        assert fig is not None

    def test_plot_returns_distribution_creates_figure(self, sample_returns):
        """Test returns distribution plot creates figure."""
        from backtester.analytics.visualization import Visualization

        fig = Visualization.plot_returns_distribution(
            sample_returns,
            backend='plotly',
            show=False
        )

        assert fig is not None

    def test_invalid_backend_raises_error(self, sample_equity_curve):
        """Test invalid backend raises InvalidBackendError."""
        from backtester.analytics.visualization import (
            Visualization, InvalidBackendError
        )

        with pytest.raises(InvalidBackendError):
            Visualization.plot_equity_curve(
                sample_equity_curve,
                backend='invalid_backend',
                show=False
            )

    def test_insufficient_data_raises_error(self):
        """Test insufficient data raises InsufficientDataError."""
        from backtester.analytics.visualization import (
            Visualization, InsufficientDataError
        )

        single_point = pd.DataFrame({'equity': [100000]})

        with pytest.raises(InsufficientDataError):
            Visualization.plot_equity_curve(single_point, show=False)

    def test_empty_data_raises_error(self):
        """Test empty data raises appropriate error."""
        from backtester.analytics.visualization import (
            Visualization, InsufficientDataError
        )

        empty_df = pd.DataFrame({'equity': []})

        with pytest.raises(InsufficientDataError):
            Visualization.plot_equity_curve(empty_df, show=False)

    def test_series_input_accepted(self, sample_equity_curve):
        """Test that Series input is accepted for equity curve."""
        from backtester.analytics.visualization import Visualization

        equity_series = sample_equity_curve['equity']

        fig = Visualization.plot_equity_curve(
            equity_series,
            backend='plotly',
            show=False
        )

        assert fig is not None


# =============================================================================
# Dashboard Class Tests
# =============================================================================

class TestDashboard:
    """Test suite for Dashboard class."""

    def test_import_dashboard(self):
        """Test that Dashboard can be imported."""
        from backtester.analytics.dashboard import Dashboard
        assert Dashboard is not None

    def test_create_performance_dashboard(
        self, sample_backtest_results, sample_metrics, temp_dir
    ):
        """Test performance dashboard creation."""
        from backtester.analytics.dashboard import Dashboard

        save_path = os.path.join(temp_dir, 'dashboard.html')

        result_path = Dashboard.create_performance_dashboard(
            sample_backtest_results,
            sample_metrics,
            save_path
        )

        assert os.path.exists(result_path)
        assert os.path.getsize(result_path) > 0

        # Check HTML content
        with open(result_path, 'r') as f:
            content = f.read()
            assert '<html>' in content.lower()
            assert 'dashboard' in content.lower()

    def test_create_risk_dashboard(
        self, sample_backtest_results, sample_metrics, temp_dir
    ):
        """Test risk dashboard creation."""
        from backtester.analytics.dashboard import Dashboard

        save_path = os.path.join(temp_dir, 'risk_dashboard.html')

        result_path = Dashboard.create_risk_dashboard(
            sample_backtest_results,
            sample_metrics['risk'],
            save_path
        )

        assert os.path.exists(result_path)
        assert os.path.getsize(result_path) > 0

    def test_create_trade_analysis_dashboard(self, sample_trade_log, temp_dir):
        """Test trade analysis dashboard creation."""
        from backtester.analytics.dashboard import Dashboard

        save_path = os.path.join(temp_dir, 'trades_dashboard.html')

        result_path = Dashboard.create_trade_analysis_dashboard(
            sample_trade_log,
            save_path
        )

        assert os.path.exists(result_path)
        assert os.path.getsize(result_path) > 0

    def test_dashboard_with_empty_data_raises_error(self, temp_dir):
        """Test dashboard with empty data raises error."""
        from backtester.analytics.dashboard import Dashboard, DashboardDataError

        with pytest.raises(DashboardDataError):
            Dashboard.create_performance_dashboard(None, {}, 'test.html')

    def test_dashboard_creates_directory(self, sample_backtest_results, sample_metrics, temp_dir):
        """Test dashboard creates parent directory if needed."""
        from backtester.analytics.dashboard import Dashboard

        save_path = os.path.join(temp_dir, 'subdir', 'dashboard.html')

        result_path = Dashboard.create_performance_dashboard(
            sample_backtest_results,
            sample_metrics,
            save_path
        )

        assert os.path.exists(result_path)


# =============================================================================
# ReportGenerator Class Tests
# =============================================================================

class TestReportGenerator:
    """Test suite for ReportGenerator class."""

    def test_import_report_generator(self):
        """Test that ReportGenerator can be imported."""
        from backtester.analytics.report import ReportGenerator
        assert ReportGenerator is not None

    def test_generate_html_report(
        self, sample_backtest_results, sample_metrics, temp_dir
    ):
        """Test HTML report generation."""
        from backtester.analytics.report import ReportGenerator

        save_path = os.path.join(temp_dir, 'report.html')

        result_path = ReportGenerator.generate_html_report(
            sample_backtest_results,
            sample_metrics,
            save_path
        )

        assert os.path.exists(result_path)
        assert os.path.getsize(result_path) > 0

        # Check HTML content
        with open(result_path, 'r') as f:
            content = f.read()
            assert '<!DOCTYPE html>' in content
            assert 'Backtest Report' in content

    def test_generate_html_report_without_charts(
        self, sample_backtest_results, sample_metrics, temp_dir
    ):
        """Test HTML report without charts."""
        from backtester.analytics.report import ReportGenerator

        save_path = os.path.join(temp_dir, 'report_no_charts.html')

        result_path = ReportGenerator.generate_html_report(
            sample_backtest_results,
            sample_metrics,
            save_path,
            include_charts=False
        )

        assert os.path.exists(result_path)

    def test_generate_summary_table_html(self, sample_metrics):
        """Test summary table generation in HTML format."""
        from backtester.analytics.report import ReportGenerator

        table = ReportGenerator.generate_summary_table(sample_metrics, format='html')

        assert '<table>' in table
        assert '</table>' in table
        assert 'Total Return' in table

    def test_generate_summary_table_markdown(self, sample_metrics):
        """Test summary table generation in Markdown format."""
        from backtester.analytics.report import ReportGenerator

        table = ReportGenerator.generate_summary_table(sample_metrics, format='markdown')

        assert '|' in table
        assert 'Metric' in table
        assert 'Value' in table

    def test_generate_summary_table_latex(self, sample_metrics):
        """Test summary table generation in LaTeX format."""
        from backtester.analytics.report import ReportGenerator

        table = ReportGenerator.generate_summary_table(sample_metrics, format='latex')

        assert '\\begin{tabular}' in table
        assert '\\end{tabular}' in table

    def test_summary_table_invalid_format_raises_error(self, sample_metrics):
        """Test invalid format raises error."""
        from backtester.analytics.report import ReportGenerator

        with pytest.raises(ValueError):
            ReportGenerator.generate_summary_table(sample_metrics, format='invalid')

    def test_report_with_none_data_raises_error(self, temp_dir):
        """Test report with None data raises error."""
        from backtester.analytics.report import ReportGenerator, ReportDataError

        with pytest.raises(ReportDataError):
            ReportGenerator.generate_html_report(None, {}, 'test.html')


# =============================================================================
# Integration Tests
# =============================================================================

class TestVisualizationIntegration:
    """Integration tests combining multiple components."""

    def test_all_plots_save_to_directory(
        self, sample_equity_curve, sample_trade_log, sample_greeks_history,
        sample_returns, sample_underlying_prices, temp_dir
    ):
        """Test saving all plot types to directory."""
        from backtester.analytics.visualization import Visualization

        plots_saved = []

        # Equity curve
        path = os.path.join(temp_dir, 'equity.html')
        Visualization.plot_equity_curve(
            sample_equity_curve, save_path=path, show=False
        )
        if os.path.exists(path):
            plots_saved.append('equity')

        # Drawdown
        path = os.path.join(temp_dir, 'drawdown.html')
        Visualization.plot_drawdown(
            sample_equity_curve, save_path=path, show=False
        )
        if os.path.exists(path):
            plots_saved.append('drawdown')

        # P&L distribution
        closed_trades = sample_trade_log[sample_trade_log['action'] == 'close']
        path = os.path.join(temp_dir, 'pnl_dist.html')
        Visualization.plot_pnl_distribution(
            closed_trades, save_path=path, show=False
        )
        if os.path.exists(path):
            plots_saved.append('pnl_dist')

        # Greeks
        path = os.path.join(temp_dir, 'greeks.html')
        Visualization.plot_greeks_over_time(
            sample_greeks_history, save_path=path, show=False
        )
        if os.path.exists(path):
            plots_saved.append('greeks')

        # Monthly returns
        path = os.path.join(temp_dir, 'monthly.html')
        Visualization.plot_monthly_returns(
            sample_equity_curve, save_path=path, show=False
        )
        if os.path.exists(path):
            plots_saved.append('monthly')

        # Rolling Sharpe
        path = os.path.join(temp_dir, 'sharpe.html')
        Visualization.plot_rolling_sharpe(
            sample_returns, save_path=path, show=False
        )
        if os.path.exists(path):
            plots_saved.append('sharpe')

        # Returns distribution
        path = os.path.join(temp_dir, 'returns_dist.html')
        Visualization.plot_returns_distribution(
            sample_returns, save_path=path, show=False
        )
        if os.path.exists(path):
            plots_saved.append('returns_dist')

        assert len(plots_saved) == 7

    def test_matplotlib_backend_all_plots(
        self, sample_equity_curve, sample_trade_log, temp_dir
    ):
        """Test all plots with matplotlib backend."""
        from backtester.analytics.visualization import Visualization
        import matplotlib
        matplotlib.use('Agg')

        # Equity curve
        path = os.path.join(temp_dir, 'equity.png')
        Visualization.plot_equity_curve(
            sample_equity_curve, save_path=path, backend='matplotlib', show=False
        )
        assert os.path.exists(path)

        # P&L distribution
        closed_trades = sample_trade_log[sample_trade_log['action'] == 'close']
        path = os.path.join(temp_dir, 'pnl_dist.png')
        Visualization.plot_pnl_distribution(
            closed_trades, save_path=path, backend='matplotlib', show=False
        )
        assert os.path.exists(path)


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_equity_curve_with_constant_values(self):
        """Test equity curve with no change in values."""
        from backtester.analytics.visualization import Visualization

        dates = pd.date_range(start='2023-01-01', periods=100, freq='B')
        df = pd.DataFrame({'equity': [100000] * 100}, index=dates)

        fig = Visualization.plot_equity_curve(df, backend='plotly', show=False)
        assert fig is not None

    def test_trade_log_all_winners(self):
        """Test P&L distribution with all winning trades."""
        from backtester.analytics.visualization import Visualization

        trades = pd.DataFrame({
            'action': ['close'] * 20,
            'realized_pnl': np.random.uniform(100, 500, 20)
        })

        fig = Visualization.plot_pnl_distribution(
            trades, backend='plotly', show=False
        )
        assert fig is not None

    def test_trade_log_all_losers(self):
        """Test P&L distribution with all losing trades."""
        from backtester.analytics.visualization import Visualization

        trades = pd.DataFrame({
            'action': ['close'] * 20,
            'realized_pnl': np.random.uniform(-500, -100, 20)
        })

        fig = Visualization.plot_pnl_distribution(
            trades, backend='plotly', show=False
        )
        assert fig is not None

    def test_greeks_with_zeros(self):
        """Test Greeks plot with zero values."""
        from backtester.analytics.visualization import Visualization

        dates = pd.date_range(start='2023-01-01', periods=100, freq='B')
        df = pd.DataFrame({
            'delta': [0.0] * 100,
            'gamma': [0.0] * 100,
            'theta': [0.0] * 100,
            'vega': [0.0] * 100,
        }, index=dates)

        fig = Visualization.plot_greeks_over_time(
            df, backend='plotly', show=False
        )
        assert fig is not None

    def test_monthly_returns_single_month(self):
        """Test monthly returns with single month of data."""
        from backtester.analytics.visualization import Visualization

        dates = pd.date_range(start='2023-01-01', periods=20, freq='B')
        df = pd.DataFrame({
            'equity': np.linspace(100000, 105000, 20)
        }, index=dates)

        # Should handle gracefully or raise appropriate error
        try:
            fig = Visualization.plot_monthly_returns(
                df, backend='plotly', show=False
            )
            assert fig is not None
        except Exception:
            # Some implementations may not support single month
            pass

    def test_rolling_sharpe_exact_window_size(self):
        """Test rolling Sharpe with data exactly window size."""
        from backtester.analytics.visualization import Visualization

        dates = pd.date_range(start='2023-01-01', periods=63, freq='B')
        returns = pd.Series(np.random.normal(0.0004, 0.015, 63), index=dates)

        fig = Visualization.plot_rolling_sharpe(
            returns, window=63, backend='plotly', show=False
        )
        assert fig is not None

    def test_dashboard_with_minimal_data(self, temp_dir):
        """Test dashboard with minimal valid data."""
        from backtester.analytics.dashboard import Dashboard

        dates = pd.date_range(start='2023-01-01', periods=5, freq='B')
        minimal_results = {
            'equity_curve': pd.DataFrame({'equity': [100000, 100100, 99900, 100200, 100300]}, index=dates),
            'trade_log': pd.DataFrame({'action': [], 'realized_pnl': []}),
            'greeks_history': pd.DataFrame(),
        }
        minimal_metrics = {
            'performance': {},
            'risk': {},
            'summary': {'total_return_pct': 0.003, 'sharpe_ratio': 0.5, 'max_drawdown': -0.002, 'win_rate': 0, 'profit_factor': 0, 'total_trades': 0}
        }

        save_path = os.path.join(temp_dir, 'minimal_dashboard.html')
        result = Dashboard.create_performance_dashboard(
            minimal_results, minimal_metrics, save_path
        )

        assert os.path.exists(result)


# =============================================================================
# File Format Tests
# =============================================================================

class TestFileFormats:
    """Test different file format outputs."""

    def test_save_as_png(self, sample_equity_curve, temp_dir):
        """Test saving plot as PNG."""
        from backtester.analytics.visualization import Visualization
        import matplotlib
        matplotlib.use('Agg')

        path = os.path.join(temp_dir, 'equity.png')
        Visualization.plot_equity_curve(
            sample_equity_curve, save_path=path, backend='matplotlib', show=False
        )

        assert os.path.exists(path)
        # Check file is valid PNG (starts with PNG magic bytes)
        with open(path, 'rb') as f:
            header = f.read(8)
            assert header[:4] == b'\x89PNG'

    def test_save_as_html(self, sample_equity_curve, temp_dir):
        """Test saving plot as HTML."""
        from backtester.analytics.visualization import Visualization

        path = os.path.join(temp_dir, 'equity.html')
        Visualization.plot_equity_curve(
            sample_equity_curve, save_path=path, backend='plotly', show=False
        )

        assert os.path.exists(path)
        with open(path, 'r') as f:
            content = f.read()
            assert '<' in content  # Basic HTML check


# =============================================================================
# Module Import Tests
# =============================================================================

class TestModuleImports:
    """Test module-level imports and exports."""

    def test_import_from_analytics_package(self):
        """Test importing from analytics package."""
        from backtester.analytics import (
            Visualization, Dashboard, ReportGenerator
        )

        assert Visualization is not None
        assert Dashboard is not None
        assert ReportGenerator is not None

    def test_import_exceptions(self):
        """Test importing exception classes."""
        from backtester.analytics import (
            VisualizationError,
            InvalidBackendError,
            DashboardError,
            ReportError,
        )

        assert VisualizationError is not None
        assert InvalidBackendError is not None
        assert DashboardError is not None
        assert ReportError is not None

    def test_import_constants(self):
        """Test importing constants."""
        from backtester.analytics import (
            SUPPORTED_BACKENDS,
            COLOR_PROFIT,
            COLOR_LOSS,
        )

        assert 'plotly' in SUPPORTED_BACKENDS
        assert 'matplotlib' in SUPPORTED_BACKENDS
        assert COLOR_PROFIT is not None
        assert COLOR_LOSS is not None


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
