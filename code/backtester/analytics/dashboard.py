"""
Dashboard Class for Options Backtesting Analytics

This module provides the Dashboard class for creating interactive HTML dashboards
using Plotly to visualize comprehensive backtest results.

Key Features:
    - Performance dashboard with equity, drawdown, P&L distribution, and Greeks
    - Risk dashboard with VaR, tail risk, and Greeks exposure
    - Trade analysis dashboard with entry/exit analysis

Design Philosophy:
    Dashboards are designed to be comprehensive, interactive, and self-contained
    HTML files that can be viewed in any modern web browser without dependencies.

Output:
    All dashboards are saved as standalone HTML files with embedded JavaScript.
    They support interactive features: zoom, pan, hover tooltips, and downloads.

Usage:
    from backtester.analytics.dashboard import Dashboard

    # Create performance dashboard
    path = Dashboard.create_performance_dashboard(
        backtest_results, metrics, 'dashboard.html'
    )
    print(f"Dashboard saved to {path}")

References:
    - Plotly Dash: https://plotly.com/python/
    - HTML/CSS best practices for reports
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Configure module logger
logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Dashboard styling
DASHBOARD_COLORS = {
    'background': '#F5F5F5',
    'card_bg': '#FFFFFF',
    'primary': '#1976D2',
    'success': '#2E7D32',
    'danger': '#C62828',
    'warning': '#F57C00',
    'text': '#212121',
    'text_secondary': '#757575',
    'border': '#E0E0E0',
}

# Default dashboard dimensions
DEFAULT_DASHBOARD_WIDTH = 1400
DEFAULT_CHART_HEIGHT = 350

# CSS template for dashboard styling
DASHBOARD_CSS = """
<style>
    body {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
        background-color: #F5F5F5;
        margin: 0;
        padding: 20px;
        color: #212121;
    }
    .dashboard-container {
        max-width: 1400px;
        margin: 0 auto;
    }
    .dashboard-header {
        text-align: center;
        margin-bottom: 30px;
        padding: 20px;
        background-color: #FFFFFF;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .dashboard-title {
        font-size: 28px;
        font-weight: 600;
        color: #1976D2;
        margin: 0 0 10px 0;
    }
    .dashboard-subtitle {
        font-size: 14px;
        color: #757575;
        margin: 0;
    }
    .metrics-row {
        display: flex;
        flex-wrap: wrap;
        gap: 15px;
        margin-bottom: 25px;
    }
    .metric-card {
        flex: 1;
        min-width: 180px;
        background-color: #FFFFFF;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .metric-label {
        font-size: 12px;
        color: #757575;
        text-transform: uppercase;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: 600;
        color: #212121;
    }
    .metric-value.positive {
        color: #2E7D32;
    }
    .metric-value.negative {
        color: #C62828;
    }
    .chart-row {
        display: flex;
        flex-wrap: wrap;
        gap: 20px;
        margin-bottom: 25px;
    }
    .chart-container {
        flex: 1;
        min-width: 45%;
        background-color: #FFFFFF;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .chart-container.full-width {
        flex: 100%;
        min-width: 100%;
    }
    .chart-title {
        font-size: 16px;
        font-weight: 600;
        color: #212121;
        margin-bottom: 10px;
        padding-bottom: 10px;
        border-bottom: 1px solid #E0E0E0;
    }
    .table-container {
        background-color: #FFFFFF;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 25px;
        overflow-x: auto;
    }
    .table-title {
        font-size: 18px;
        font-weight: 600;
        color: #212121;
        margin-bottom: 15px;
    }
    table {
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
    }
    th {
        background-color: #F5F5F5;
        color: #212121;
        padding: 12px;
        text-align: left;
        border-bottom: 2px solid #E0E0E0;
    }
    td {
        padding: 10px 12px;
        border-bottom: 1px solid #E0E0E0;
    }
    tr:hover {
        background-color: #F5F5F5;
    }
    .footer {
        text-align: center;
        padding: 20px;
        color: #757575;
        font-size: 12px;
    }
</style>
"""


# =============================================================================
# Exceptions
# =============================================================================

class DashboardError(Exception):
    """Base exception for dashboard errors."""
    pass


class DashboardDataError(DashboardError):
    """Exception raised for missing or invalid data."""
    pass


class DashboardSaveError(DashboardError):
    """Exception raised when saving dashboard fails."""
    pass


# =============================================================================
# Helper Functions
# =============================================================================

def _ensure_directory(path: str) -> None:
    """Create directory for save path if it doesn't exist."""
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def _format_number(value: float, fmt: str = 'general') -> str:
    """Format number for display."""
    if pd.isna(value) or not np.isfinite(value):
        return 'N/A'

    if fmt == 'percent':
        return f'{value:.2%}'
    elif fmt == 'currency':
        return f'${value:,.2f}'
    elif fmt == 'currency_int':
        return f'${value:,.0f}'
    elif fmt == 'ratio':
        return f'{value:.2f}'
    elif fmt == 'integer':
        return f'{int(value):,}'
    else:
        if abs(value) >= 1000:
            return f'{value:,.2f}'
        return f'{value:.4f}'


def _get_value_class(value: float, threshold: float = 0) -> str:
    """Get CSS class based on value sign."""
    if pd.isna(value):
        return ''
    return 'positive' if value > threshold else ('negative' if value < threshold else '')


# =============================================================================
# Dashboard Class
# =============================================================================

class Dashboard:
    """
    Create interactive dashboards for backtest results.

    This class provides static methods for creating comprehensive HTML dashboards
    using Plotly for interactive visualizations.

    Methods:
        create_performance_dashboard: Main performance metrics dashboard
        create_risk_dashboard: Risk analysis dashboard
        create_trade_analysis_dashboard: Trade-by-trade analysis

    Example:
        >>> path = Dashboard.create_performance_dashboard(
        ...     backtest_results, metrics, 'dashboard.html'
        ... )
        >>> print(f"Dashboard saved to: {path}")
    """

    # =========================================================================
    # Performance Dashboard
    # =========================================================================

    @staticmethod
    def create_performance_dashboard(
        backtest_results: Dict[str, Any],
        metrics: Dict[str, Any],
        save_path: str = 'dashboard.html',
        title: str = 'Backtest Performance Dashboard'
    ) -> str:
        """
        Create comprehensive performance dashboard using Plotly.

        Creates an interactive HTML dashboard with:
        - Summary metrics cards (total return, Sharpe, max DD)
        - Equity curve and drawdown chart
        - P&L distribution and monthly returns heatmap
        - Greeks over time and rolling metrics
        - Trade statistics table

        Args:
            backtest_results: Dictionary from BacktestEngine.run() containing:
                - 'equity_curve': DataFrame with equity time series
                - 'trade_log': DataFrame with trade records
                - 'greeks_history': DataFrame with Greeks over time
            metrics: Dictionary from BacktestEngine.calculate_metrics() containing:
                - 'performance': Performance metrics
                - 'risk': Risk metrics
                - 'summary': Summary metrics
            save_path: Path to save HTML dashboard. Default 'dashboard.html'.
            title: Dashboard title. Default 'Backtest Performance Dashboard'.

        Returns:
            Absolute path to saved dashboard file.

        Raises:
            DashboardDataError: If required data is missing
            DashboardSaveError: If saving fails

        Example:
            >>> results = engine.run()
            >>> metrics = engine.calculate_metrics()
            >>> path = Dashboard.create_performance_dashboard(
            ...     results, metrics, 'performance.html'
            ... )
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # Validate inputs
        if backtest_results is None:
            raise DashboardDataError("backtest_results cannot be None")
        if metrics is None:
            raise DashboardDataError("metrics cannot be None")

        # Extract data
        equity_curve = backtest_results.get('equity_curve', pd.DataFrame())
        trade_log = backtest_results.get('trade_log', pd.DataFrame())
        greeks_history = backtest_results.get('greeks_history', pd.DataFrame())

        # Get metrics
        perf = metrics.get('performance', {})
        risk = metrics.get('risk', {})
        summary = metrics.get('summary', {})

        # Build HTML sections
        html_parts = []

        # Header
        html_parts.append(f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    {DASHBOARD_CSS}
</head>
<body>
    <div class="dashboard-container">
        <div class="dashboard-header">
            <h1 class="dashboard-title">{title}</h1>
            <p class="dashboard-subtitle">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
""")

        # Metrics cards row
        metrics_html = Dashboard._create_metrics_cards(summary, perf, risk)
        html_parts.append(metrics_html)

        # Equity curve chart
        if not equity_curve.empty and 'equity' in equity_curve.columns:
            equity_chart = Dashboard._create_equity_chart_html(equity_curve)
            html_parts.append(f"""
        <div class="chart-row">
            <div class="chart-container full-width">
                <div class="chart-title">Equity Curve & Drawdown</div>
                {equity_chart}
            </div>
        </div>
""")

        # P&L distribution and monthly returns
        if not trade_log.empty:
            pnl_chart = Dashboard._create_pnl_distribution_html(trade_log)
        else:
            pnl_chart = '<p>No trade data available</p>'

        if not equity_curve.empty and 'equity' in equity_curve.columns:
            monthly_chart = Dashboard._create_monthly_returns_html(equity_curve)
        else:
            monthly_chart = '<p>No equity data available</p>'

        html_parts.append(f"""
        <div class="chart-row">
            <div class="chart-container">
                <div class="chart-title">P&L Distribution</div>
                {pnl_chart}
            </div>
            <div class="chart-container">
                <div class="chart-title">Monthly Returns</div>
                {monthly_chart}
            </div>
        </div>
""")

        # Greeks and rolling Sharpe
        if not greeks_history.empty:
            greeks_chart = Dashboard._create_greeks_chart_html(greeks_history)
        else:
            greeks_chart = '<p>No Greeks data available</p>'

        if not equity_curve.empty and 'equity' in equity_curve.columns:
            rolling_chart = Dashboard._create_rolling_sharpe_html(equity_curve)
        else:
            rolling_chart = '<p>No equity data available</p>'

        html_parts.append(f"""
        <div class="chart-row">
            <div class="chart-container">
                <div class="chart-title">Portfolio Greeks Over Time</div>
                {greeks_chart}
            </div>
            <div class="chart-container">
                <div class="chart-title">Rolling Sharpe Ratio (63-day)</div>
                {rolling_chart}
            </div>
        </div>
""")

        # Trade statistics table
        if not trade_log.empty:
            trade_table = Dashboard._create_trade_stats_table(trade_log, perf)
            html_parts.append(f"""
        <div class="table-container">
            <div class="table-title">Trade Statistics</div>
            {trade_table}
        </div>
""")

        # Performance metrics table
        metrics_table = Dashboard._create_metrics_table(perf, risk)
        html_parts.append(f"""
        <div class="table-container">
            <div class="table-title">Detailed Metrics</div>
            {metrics_table}
        </div>
""")

        # Footer
        html_parts.append("""
        <div class="footer">
            Generated by Options Backtester - Visualization Module
        </div>
    </div>
</body>
</html>
""")

        # Combine and save
        html_content = ''.join(html_parts)

        try:
            _ensure_directory(save_path)
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            abs_path = os.path.abspath(save_path)
            logger.info(f"Performance dashboard saved to {abs_path}")
            return abs_path
        except Exception as e:
            raise DashboardSaveError(f"Failed to save dashboard: {e}")

    # =========================================================================
    # Risk Dashboard
    # =========================================================================

    @staticmethod
    def create_risk_dashboard(
        backtest_results: Dict[str, Any],
        risk_metrics: Dict[str, Any],
        save_path: str = 'risk_dashboard.html',
        title: str = 'Risk Analysis Dashboard'
    ) -> str:
        """
        Create risk analysis dashboard.

        Creates an interactive HTML dashboard with:
        - VaR/CVaR visualization
        - Greeks exposure over time
        - Tail risk analysis
        - Drawdown statistics

        Args:
            backtest_results: Dictionary from BacktestEngine.run().
            risk_metrics: Risk metrics dictionary (from calculate_metrics).
            save_path: Path to save HTML dashboard.
            title: Dashboard title.

        Returns:
            Absolute path to saved dashboard file.

        Example:
            >>> metrics = engine.calculate_metrics()
            >>> path = Dashboard.create_risk_dashboard(
            ...     results, metrics['risk'], 'risk.html'
            ... )
        """
        import plotly.graph_objects as go

        # Validate inputs
        if backtest_results is None:
            raise DashboardDataError("backtest_results cannot be None")

        # Extract data
        equity_curve = backtest_results.get('equity_curve', pd.DataFrame())
        greeks_history = backtest_results.get('greeks_history', pd.DataFrame())

        # Build HTML
        html_parts = []

        # Header
        html_parts.append(f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    {DASHBOARD_CSS}
</head>
<body>
    <div class="dashboard-container">
        <div class="dashboard-header">
            <h1 class="dashboard-title">{title}</h1>
            <p class="dashboard-subtitle">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
""")

        # Risk metrics cards
        risk_cards = Dashboard._create_risk_metrics_cards(risk_metrics)
        html_parts.append(risk_cards)

        # VaR/CVaR visualization
        if not equity_curve.empty and 'equity' in equity_curve.columns:
            var_chart = Dashboard._create_var_chart_html(equity_curve, risk_metrics)
            html_parts.append(f"""
        <div class="chart-row">
            <div class="chart-container full-width">
                <div class="chart-title">Value at Risk Analysis</div>
                {var_chart}
            </div>
        </div>
""")

        # Greeks exposure
        if not greeks_history.empty:
            greeks_exposure = Dashboard._create_greeks_exposure_html(greeks_history)
            html_parts.append(f"""
        <div class="chart-row">
            <div class="chart-container full-width">
                <div class="chart-title">Greeks Exposure Over Time</div>
                {greeks_exposure}
            </div>
        </div>
""")

        # Tail risk table
        tail_risk_table = Dashboard._create_tail_risk_table(risk_metrics)
        html_parts.append(f"""
        <div class="table-container">
            <div class="table-title">Tail Risk Analysis</div>
            {tail_risk_table}
        </div>
""")

        # Footer
        html_parts.append("""
        <div class="footer">
            Generated by Options Backtester - Visualization Module
        </div>
    </div>
</body>
</html>
""")

        # Combine and save
        html_content = ''.join(html_parts)

        try:
            _ensure_directory(save_path)
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            abs_path = os.path.abspath(save_path)
            logger.info(f"Risk dashboard saved to {abs_path}")
            return abs_path
        except Exception as e:
            raise DashboardSaveError(f"Failed to save dashboard: {e}")

    # =========================================================================
    # Trade Analysis Dashboard
    # =========================================================================

    @staticmethod
    def create_trade_analysis_dashboard(
        trades: pd.DataFrame,
        save_path: str = 'trades_dashboard.html',
        title: str = 'Trade Analysis Dashboard'
    ) -> str:
        """
        Create trade-by-trade analysis dashboard.

        Creates an interactive HTML dashboard with:
        - Trade timeline
        - Win/loss breakdown
        - P&L by trade sequence
        - Trade duration distribution

        Args:
            trades: DataFrame with trade records.
            save_path: Path to save HTML dashboard.
            title: Dashboard title.

        Returns:
            Absolute path to saved dashboard file.

        Example:
            >>> trades = results['trade_log']
            >>> path = Dashboard.create_trade_analysis_dashboard(
            ...     trades, 'trades.html'
            ... )
        """
        if trades is None or trades.empty:
            raise DashboardDataError("trades DataFrame is empty or None")

        # Build HTML
        html_parts = []

        # Header
        html_parts.append(f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    {DASHBOARD_CSS}
</head>
<body>
    <div class="dashboard-container">
        <div class="dashboard-header">
            <h1 class="dashboard-title">{title}</h1>
            <p class="dashboard-subtitle">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
""")

        # Trade summary cards
        trade_cards = Dashboard._create_trade_summary_cards(trades)
        html_parts.append(trade_cards)

        # Cumulative P&L chart
        cum_pnl_chart = Dashboard._create_cumulative_pnl_html(trades)
        html_parts.append(f"""
        <div class="chart-row">
            <div class="chart-container full-width">
                <div class="chart-title">Cumulative P&L by Trade</div>
                {cum_pnl_chart}
            </div>
        </div>
""")

        # Win/Loss breakdown and P&L distribution
        win_loss_chart = Dashboard._create_win_loss_chart_html(trades)
        pnl_by_trade = Dashboard._create_pnl_by_trade_html(trades)

        html_parts.append(f"""
        <div class="chart-row">
            <div class="chart-container">
                <div class="chart-title">Win/Loss Breakdown</div>
                {win_loss_chart}
            </div>
            <div class="chart-container">
                <div class="chart-title">P&L by Trade</div>
                {pnl_by_trade}
            </div>
        </div>
""")

        # Trade log table
        trade_table = Dashboard._create_trade_log_table(trades)
        html_parts.append(f"""
        <div class="table-container">
            <div class="table-title">Trade Log</div>
            {trade_table}
        </div>
""")

        # Footer
        html_parts.append("""
        <div class="footer">
            Generated by Options Backtester - Visualization Module
        </div>
    </div>
</body>
</html>
""")

        # Combine and save
        html_content = ''.join(html_parts)

        try:
            _ensure_directory(save_path)
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            abs_path = os.path.abspath(save_path)
            logger.info(f"Trade analysis dashboard saved to {abs_path}")
            return abs_path
        except Exception as e:
            raise DashboardSaveError(f"Failed to save dashboard: {e}")

    # =========================================================================
    # Helper Methods - Metrics Cards
    # =========================================================================

    @staticmethod
    def _create_metrics_cards(
        summary: Dict,
        perf: Dict,
        risk: Dict
    ) -> str:
        """Create HTML for summary metrics cards."""
        total_return = summary.get('total_return_pct', 0)
        sharpe = summary.get('sharpe_ratio', 0)
        max_dd = summary.get('max_drawdown', 0)
        win_rate = summary.get('win_rate', 0)
        profit_factor = summary.get('profit_factor', 0)
        total_trades = summary.get('total_trades', 0)

        return f"""
        <div class="metrics-row">
            <div class="metric-card">
                <div class="metric-label">Total Return</div>
                <div class="metric-value {_get_value_class(total_return)}">{_format_number(total_return, 'percent')}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Sharpe Ratio</div>
                <div class="metric-value {_get_value_class(sharpe, 1)}">{_format_number(sharpe, 'ratio')}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Max Drawdown</div>
                <div class="metric-value negative">{_format_number(max_dd, 'percent')}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Win Rate</div>
                <div class="metric-value {_get_value_class(win_rate - 0.5)}">{_format_number(win_rate, 'percent')}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Profit Factor</div>
                <div class="metric-value {_get_value_class(profit_factor, 1)}">{_format_number(profit_factor, 'ratio')}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Total Trades</div>
                <div class="metric-value">{_format_number(total_trades, 'integer')}</div>
            </div>
        </div>
"""

    @staticmethod
    def _create_risk_metrics_cards(risk_metrics: Dict) -> str:
        """Create HTML for risk metrics cards."""
        var_95 = risk_metrics.get('var_95_historical', 0)
        cvar_95 = risk_metrics.get('cvar_95', 0)
        skewness = risk_metrics.get('skewness', 0)
        kurtosis = risk_metrics.get('kurtosis', 0)
        dd_dev = risk_metrics.get('downside_deviation', 0)

        return f"""
        <div class="metrics-row">
            <div class="metric-card">
                <div class="metric-label">VaR (95%)</div>
                <div class="metric-value negative">{_format_number(var_95 * 100, 'ratio')}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">CVaR (95%)</div>
                <div class="metric-value negative">{_format_number(cvar_95 * 100, 'ratio')}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Skewness</div>
                <div class="metric-value {_get_value_class(skewness)}">{_format_number(skewness, 'ratio')}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Kurtosis</div>
                <div class="metric-value">{_format_number(kurtosis, 'ratio')}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Downside Dev</div>
                <div class="metric-value">{_format_number(dd_dev * 100, 'ratio')}%</div>
            </div>
        </div>
"""

    @staticmethod
    def _create_trade_summary_cards(trades: pd.DataFrame) -> str:
        """Create HTML for trade summary cards."""
        closed = trades[trades['action'] == 'close']
        total = len(closed)
        wins = len(closed[closed['realized_pnl'] > 0])
        losses = len(closed[closed['realized_pnl'] < 0])
        total_pnl = closed['realized_pnl'].sum() if 'realized_pnl' in closed.columns else 0
        avg_pnl = closed['realized_pnl'].mean() if 'realized_pnl' in closed.columns else 0

        win_rate = wins / total if total > 0 else 0

        return f"""
        <div class="metrics-row">
            <div class="metric-card">
                <div class="metric-label">Total Trades</div>
                <div class="metric-value">{total}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Wins</div>
                <div class="metric-value positive">{wins}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Losses</div>
                <div class="metric-value negative">{losses}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Win Rate</div>
                <div class="metric-value {_get_value_class(win_rate - 0.5)}">{_format_number(win_rate, 'percent')}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Total P&L</div>
                <div class="metric-value {_get_value_class(total_pnl)}">{_format_number(total_pnl, 'currency')}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Avg P&L</div>
                <div class="metric-value {_get_value_class(avg_pnl)}">{_format_number(avg_pnl, 'currency')}</div>
            </div>
        </div>
"""

    # =========================================================================
    # Helper Methods - Charts
    # =========================================================================

    @staticmethod
    def _create_equity_chart_html(equity_curve: pd.DataFrame) -> str:
        """Create equity curve with drawdown chart as HTML."""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        equity = equity_curve['equity']

        # Calculate drawdown
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max * 100

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.7, 0.3]
        )

        # Equity trace
        fig.add_trace(go.Scatter(
            x=equity.index.astype(str) if hasattr(equity.index, 'astype') else list(range(len(equity))),
            y=equity.values,
            mode='lines',
            name='Equity',
            line=dict(color='#1976D2', width=2),
            hovertemplate='Date: %{x}<br>Equity: $%{y:,.2f}<extra></extra>'
        ), row=1, col=1)

        # Drawdown trace
        fig.add_trace(go.Scatter(
            x=drawdown.index.astype(str) if hasattr(drawdown.index, 'astype') else list(range(len(drawdown))),
            y=drawdown.values,
            mode='lines',
            name='Drawdown',
            fill='tozeroy',
            line=dict(color='#C62828', width=1),
            fillcolor='rgba(198, 40, 40, 0.3)',
            hovertemplate='Date: %{x}<br>Drawdown: %{y:.2f}%<extra></extra>'
        ), row=2, col=1)

        fig.update_layout(
            height=DEFAULT_CHART_HEIGHT + 150,
            margin=dict(l=50, r=20, t=20, b=40),
            showlegend=True,
            legend=dict(yanchor='top', y=0.99, xanchor='left', x=0.01),
            template='plotly_white',
            hovermode='x unified'
        )

        fig.update_yaxes(title_text='Equity ($)', tickformat='$,.0f', row=1, col=1)
        fig.update_yaxes(title_text='Drawdown (%)', tickformat='.1f', row=2, col=1)

        return fig.to_html(full_html=False, include_plotlyjs=False)

    @staticmethod
    def _create_pnl_distribution_html(trades: pd.DataFrame) -> str:
        """Create P&L distribution histogram as HTML."""
        import plotly.graph_objects as go

        closed = trades[trades['action'] == 'close']
        if closed.empty or 'realized_pnl' not in closed.columns:
            return '<p>No closed trades available</p>'

        pnl = closed['realized_pnl'].dropna()

        fig = go.Figure()

        fig.add_trace(go.Histogram(
            x=pnl.values,
            nbinsx=30,
            name='P&L',
            marker=dict(
                color=['#2E7D32' if x >= 0 else '#C62828' for x in pnl.values],
                line=dict(color='white', width=1)
            ),
            hovertemplate='P&L: $%{x:,.0f}<br>Count: %{y}<extra></extra>'
        ))

        # Add mean line
        mean_pnl = pnl.mean()
        fig.add_vline(x=mean_pnl, line=dict(color='blue', width=2, dash='dash'),
                     annotation_text=f'Mean: ${mean_pnl:,.0f}')

        fig.update_layout(
            height=DEFAULT_CHART_HEIGHT,
            margin=dict(l=50, r=20, t=20, b=40),
            xaxis=dict(title='P&L ($)', tickformat='$,.0f'),
            yaxis=dict(title='Count'),
            template='plotly_white',
            showlegend=False
        )

        return fig.to_html(full_html=False, include_plotlyjs=False)

    @staticmethod
    def _create_monthly_returns_html(equity_curve: pd.DataFrame) -> str:
        """Create monthly returns heatmap as HTML."""
        import plotly.graph_objects as go

        if 'equity' not in equity_curve.columns:
            return '<p>No equity data available</p>'

        equity = equity_curve['equity']

        # Ensure datetime index
        if not isinstance(equity.index, pd.DatetimeIndex):
            try:
                equity.index = pd.to_datetime(equity.index)
            except Exception:
                return '<p>Could not parse dates</p>'

        # Calculate monthly returns
        # Use 'M' for compatibility with older pandas versions (< 2.0)
        try:
            monthly = equity.resample('ME').last()
        except ValueError:
            monthly = equity.resample('M').last()
        monthly_ret = monthly.pct_change().dropna() * 100

        if len(monthly_ret) < 2:
            return '<p>Insufficient data for monthly returns</p>'

        # Create pivot
        monthly_df = pd.DataFrame({
            'year': monthly_ret.index.year,
            'month': monthly_ret.index.month,
            'return': monthly_ret.values
        })

        pivot = monthly_df.pivot(index='year', columns='month', values='return')
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        # Rename columns
        pivot.columns = [month_names[m-1] for m in pivot.columns]

        # Create text
        text = [[f'{val:.1f}%' if pd.notna(val) else ''
                for val in row] for row in pivot.values]

        vmax = max(abs(pivot.min().min()), abs(pivot.max().max()), 1)

        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=list(pivot.columns),
            y=[str(y) for y in pivot.index],
            text=text,
            texttemplate='%{text}',
            textfont=dict(size=10),
            colorscale=[[0, '#C62828'], [0.5, 'white'], [1, '#2E7D32']],
            zmin=-vmax,
            zmax=vmax,
            colorbar=dict(title='Return %'),
            hovertemplate='%{y} %{x}<br>Return: %{z:.2f}%<extra></extra>'
        ))

        fig.update_layout(
            height=DEFAULT_CHART_HEIGHT,
            margin=dict(l=50, r=20, t=20, b=40),
            xaxis=dict(tickangle=0),
            yaxis=dict(autorange='reversed'),
            template='plotly_white'
        )

        return fig.to_html(full_html=False, include_plotlyjs=False)

    @staticmethod
    def _create_greeks_chart_html(greeks_history: pd.DataFrame) -> str:
        """Create Greeks over time chart as HTML."""
        import plotly.graph_objects as go

        greek_names = ['delta', 'gamma', 'theta', 'vega']
        available = [g for g in greek_names if g in greeks_history.columns]

        if not available:
            return '<p>No Greeks data available</p>'

        fig = go.Figure()

        colors = ['#1976D2', '#7B1FA2', '#F57C00', '#388E3C']

        for i, greek in enumerate(available):
            values = greeks_history[greek]
            fig.add_trace(go.Scatter(
                x=values.index.astype(str) if hasattr(values.index, 'astype') else list(range(len(values))),
                y=values.values,
                mode='lines',
                name=greek.capitalize(),
                line=dict(color=colors[i % len(colors)], width=1.5),
                hovertemplate=f'{greek.capitalize()}: %{{y:.4f}}<extra></extra>'
            ))

        fig.add_hline(y=0, line=dict(color='black', width=0.5))

        fig.update_layout(
            height=DEFAULT_CHART_HEIGHT,
            margin=dict(l=50, r=20, t=20, b=40),
            xaxis=dict(title='Date'),
            yaxis=dict(title='Greek Value'),
            template='plotly_white',
            hovermode='x unified',
            showlegend=True
        )

        return fig.to_html(full_html=False, include_plotlyjs=False)

    @staticmethod
    def _create_rolling_sharpe_html(equity_curve: pd.DataFrame) -> str:
        """Create rolling Sharpe ratio chart as HTML."""
        import plotly.graph_objects as go

        equity = equity_curve['equity']
        returns = equity.pct_change().dropna()

        if len(returns) < 63:
            return '<p>Insufficient data for rolling Sharpe (need 63+ days)</p>'

        window = 63
        rf = 0.02 / 252

        rolling_mean = returns.rolling(window).mean()
        rolling_std = returns.rolling(window).std()

        rolling_sharpe = np.where(
            rolling_std > 1e-10,
            ((rolling_mean - rf) / rolling_std) * np.sqrt(252),
            0
        )
        rolling_sharpe = pd.Series(rolling_sharpe, index=returns.index)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=rolling_sharpe.index.astype(str) if hasattr(rolling_sharpe.index, 'astype') else list(range(len(rolling_sharpe))),
            y=rolling_sharpe.values,
            mode='lines',
            name='63-day Rolling Sharpe',
            line=dict(color='#1976D2', width=2),
            hovertemplate='Date: %{x}<br>Sharpe: %{y:.2f}<extra></extra>'
        ))

        fig.add_hline(y=0, line=dict(color='black', width=0.5))
        fig.add_hline(y=1, line=dict(color='#757575', width=1, dash='dash'))
        fig.add_hline(y=2, line=dict(color='#757575', width=1, dash='dot'))

        fig.update_layout(
            height=DEFAULT_CHART_HEIGHT,
            margin=dict(l=50, r=20, t=20, b=40),
            xaxis=dict(title='Date'),
            yaxis=dict(title='Sharpe Ratio'),
            template='plotly_white',
            showlegend=False
        )

        return fig.to_html(full_html=False, include_plotlyjs=False)

    @staticmethod
    def _create_var_chart_html(equity_curve: pd.DataFrame, risk_metrics: Dict) -> str:
        """Create VaR visualization as HTML."""
        import plotly.graph_objects as go

        equity = equity_curve['equity']
        returns = equity.pct_change().dropna() * 100

        fig = go.Figure()

        # Histogram
        fig.add_trace(go.Histogram(
            x=returns.values,
            nbinsx=50,
            name='Daily Returns',
            marker=dict(color='#1976D2', line=dict(color='white', width=1)),
            hovertemplate='Return: %{x:.2f}%<br>Count: %{y}<extra></extra>'
        ))

        # VaR lines
        var_95 = risk_metrics.get('var_95_historical', 0) * 100
        cvar_95 = risk_metrics.get('cvar_95', 0) * 100

        fig.add_vline(x=var_95, line=dict(color='#F57C00', width=2),
                     annotation_text=f'VaR 95%: {var_95:.2f}%')
        fig.add_vline(x=cvar_95, line=dict(color='#C62828', width=2, dash='dash'),
                     annotation_text=f'CVaR 95%: {cvar_95:.2f}%')

        fig.update_layout(
            height=DEFAULT_CHART_HEIGHT,
            margin=dict(l=50, r=20, t=20, b=40),
            xaxis=dict(title='Daily Return (%)', tickformat='.1f'),
            yaxis=dict(title='Frequency'),
            template='plotly_white',
            showlegend=False
        )

        return fig.to_html(full_html=False, include_plotlyjs=False)

    @staticmethod
    def _create_greeks_exposure_html(greeks_history: pd.DataFrame) -> str:
        """Create Greeks exposure chart as HTML."""
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go

        greek_names = ['delta', 'gamma', 'theta', 'vega']
        available = [g for g in greek_names if g in greeks_history.columns]

        if not available:
            return '<p>No Greeks data available</p>'

        n = len(available)
        fig = make_subplots(rows=n, cols=1, shared_xaxes=True,
                           subplot_titles=[g.capitalize() for g in available],
                           vertical_spacing=0.08)

        colors = ['#1976D2', '#7B1FA2', '#F57C00', '#388E3C']

        for i, greek in enumerate(available):
            values = greeks_history[greek]
            fig.add_trace(go.Scatter(
                x=values.index.astype(str) if hasattr(values.index, 'astype') else list(range(len(values))),
                y=values.values,
                mode='lines',
                name=greek.capitalize(),
                fill='tozeroy',
                line=dict(color=colors[i % len(colors)], width=1),
            ), row=i+1, col=1)

            fig.add_hline(y=0, line=dict(color='black', width=0.5), row=i+1, col=1)

        fig.update_layout(
            height=150 * n + 50,
            margin=dict(l=50, r=20, t=40, b=40),
            showlegend=False,
            template='plotly_white'
        )

        return fig.to_html(full_html=False, include_plotlyjs=False)

    @staticmethod
    def _create_cumulative_pnl_html(trades: pd.DataFrame) -> str:
        """Create cumulative P&L chart as HTML."""
        import plotly.graph_objects as go

        closed = trades[trades['action'] == 'close'].copy()
        if closed.empty or 'realized_pnl' not in closed.columns:
            return '<p>No closed trades available</p>'

        closed = closed.sort_values('timestamp')
        closed['cum_pnl'] = closed['realized_pnl'].cumsum()

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=list(range(1, len(closed) + 1)),
            y=closed['cum_pnl'].values,
            mode='lines+markers',
            name='Cumulative P&L',
            line=dict(color='#1976D2', width=2),
            marker=dict(size=6),
            hovertemplate='Trade #%{x}<br>Cum P&L: $%{y:,.2f}<extra></extra>'
        ))

        fig.add_hline(y=0, line=dict(color='black', width=0.5))

        fig.update_layout(
            height=DEFAULT_CHART_HEIGHT,
            margin=dict(l=50, r=20, t=20, b=40),
            xaxis=dict(title='Trade Number'),
            yaxis=dict(title='Cumulative P&L ($)', tickformat='$,.0f'),
            template='plotly_white',
            showlegend=False
        )

        return fig.to_html(full_html=False, include_plotlyjs=False)

    @staticmethod
    def _create_win_loss_chart_html(trades: pd.DataFrame) -> str:
        """Create win/loss pie chart as HTML."""
        import plotly.graph_objects as go

        closed = trades[trades['action'] == 'close']
        if closed.empty or 'realized_pnl' not in closed.columns:
            return '<p>No closed trades available</p>'

        wins = len(closed[closed['realized_pnl'] > 0])
        losses = len(closed[closed['realized_pnl'] < 0])
        breakeven = len(closed[closed['realized_pnl'] == 0])

        fig = go.Figure(data=[go.Pie(
            labels=['Wins', 'Losses', 'Breakeven'],
            values=[wins, losses, breakeven],
            marker=dict(colors=['#2E7D32', '#C62828', '#757575']),
            hole=0.4,
            hovertemplate='%{label}: %{value} (%{percent})<extra></extra>'
        )])

        fig.update_layout(
            height=DEFAULT_CHART_HEIGHT,
            margin=dict(l=20, r=20, t=20, b=20),
            showlegend=True
        )

        return fig.to_html(full_html=False, include_plotlyjs=False)

    @staticmethod
    def _create_pnl_by_trade_html(trades: pd.DataFrame) -> str:
        """Create P&L by trade bar chart as HTML."""
        import plotly.graph_objects as go

        closed = trades[trades['action'] == 'close'].copy()
        if closed.empty or 'realized_pnl' not in closed.columns:
            return '<p>No closed trades available</p>'

        closed = closed.sort_values('timestamp').reset_index(drop=True)

        colors = ['#2E7D32' if x >= 0 else '#C62828' for x in closed['realized_pnl']]

        fig = go.Figure(data=[go.Bar(
            x=list(range(1, len(closed) + 1)),
            y=closed['realized_pnl'].values,
            marker=dict(color=colors),
            hovertemplate='Trade #%{x}<br>P&L: $%{y:,.2f}<extra></extra>'
        )])

        fig.add_hline(y=0, line=dict(color='black', width=0.5))

        fig.update_layout(
            height=DEFAULT_CHART_HEIGHT,
            margin=dict(l=50, r=20, t=20, b=40),
            xaxis=dict(title='Trade Number'),
            yaxis=dict(title='P&L ($)', tickformat='$,.0f'),
            template='plotly_white',
            showlegend=False
        )

        return fig.to_html(full_html=False, include_plotlyjs=False)

    # =========================================================================
    # Helper Methods - Tables
    # =========================================================================

    @staticmethod
    def _create_trade_stats_table(trades: pd.DataFrame, perf: Dict) -> str:
        """Create trade statistics table HTML."""
        closed = trades[trades['action'] == 'close']

        if closed.empty:
            return '<p>No closed trades</p>'

        total = len(closed)
        wins = len(closed[closed['realized_pnl'] > 0])
        losses = len(closed[closed['realized_pnl'] < 0])

        total_pnl = closed['realized_pnl'].sum()
        gross_profit = closed[closed['realized_pnl'] > 0]['realized_pnl'].sum()
        gross_loss = closed[closed['realized_pnl'] < 0]['realized_pnl'].sum()

        avg_win = closed[closed['realized_pnl'] > 0]['realized_pnl'].mean() if wins > 0 else 0
        avg_loss = closed[closed['realized_pnl'] < 0]['realized_pnl'].mean() if losses > 0 else 0
        avg_trade = closed['realized_pnl'].mean()

        largest_win = closed['realized_pnl'].max()
        largest_loss = closed['realized_pnl'].min()

        return f"""
        <table>
            <tr><th>Metric</th><th>Value</th><th>Metric</th><th>Value</th></tr>
            <tr>
                <td>Total Trades</td><td>{total}</td>
                <td>Win Rate</td><td>{_format_number(wins/total if total > 0 else 0, 'percent')}</td>
            </tr>
            <tr>
                <td>Winning Trades</td><td class="{_get_value_class(1)}">{wins}</td>
                <td>Losing Trades</td><td class="{_get_value_class(-1)}">{losses}</td>
            </tr>
            <tr>
                <td>Total P&L</td><td class="{_get_value_class(total_pnl)}">{_format_number(total_pnl, 'currency')}</td>
                <td>Average Trade</td><td class="{_get_value_class(avg_trade)}">{_format_number(avg_trade, 'currency')}</td>
            </tr>
            <tr>
                <td>Gross Profit</td><td class="positive">{_format_number(gross_profit, 'currency')}</td>
                <td>Gross Loss</td><td class="negative">{_format_number(gross_loss, 'currency')}</td>
            </tr>
            <tr>
                <td>Average Win</td><td class="positive">{_format_number(avg_win, 'currency')}</td>
                <td>Average Loss</td><td class="negative">{_format_number(avg_loss, 'currency')}</td>
            </tr>
            <tr>
                <td>Largest Win</td><td class="positive">{_format_number(largest_win, 'currency')}</td>
                <td>Largest Loss</td><td class="negative">{_format_number(largest_loss, 'currency')}</td>
            </tr>
            <tr>
                <td>Profit Factor</td><td>{_format_number(abs(gross_profit/gross_loss) if gross_loss != 0 else 0, 'ratio')}</td>
                <td>Payoff Ratio</td><td>{_format_number(abs(avg_win/avg_loss) if avg_loss != 0 else 0, 'ratio')}</td>
            </tr>
        </table>
"""

    @staticmethod
    def _create_metrics_table(perf: Dict, risk: Dict) -> str:
        """Create detailed metrics table HTML."""
        rows = []

        # Performance metrics
        metrics = [
            ('Total Return', perf.get('total_return_pct', np.nan), 'percent'),
            ('Annualized Return', perf.get('annualized_return', np.nan), 'percent'),
            ('Sharpe Ratio', perf.get('sharpe_ratio', np.nan), 'ratio'),
            ('Sortino Ratio', perf.get('sortino_ratio', np.nan), 'ratio'),
            ('Calmar Ratio', perf.get('calmar_ratio', np.nan), 'ratio'),
            ('Max Drawdown', perf.get('max_drawdown', np.nan), 'percent'),
            ('Volatility (Ann.)', perf.get('volatility', np.nan), 'percent'),
            ('Win Rate', perf.get('win_rate', np.nan), 'percent'),
            ('Profit Factor', perf.get('profit_factor', np.nan), 'ratio'),
            ('Expectancy', perf.get('expectancy', np.nan), 'currency'),
            ('VaR (95%)', risk.get('var_95_historical', np.nan), 'percent'),
            ('CVaR (95%)', risk.get('cvar_95', np.nan), 'percent'),
            ('Skewness', risk.get('skewness', np.nan), 'ratio'),
            ('Kurtosis', risk.get('kurtosis', np.nan), 'ratio'),
        ]

        # Build rows
        for i in range(0, len(metrics), 2):
            m1 = metrics[i]
            m2 = metrics[i+1] if i+1 < len(metrics) else ('', np.nan, 'general')

            val1 = _format_number(m1[1], m1[2])
            val2 = _format_number(m2[1], m2[2])

            rows.append(f"""
            <tr>
                <td>{m1[0]}</td><td>{val1}</td>
                <td>{m2[0]}</td><td>{val2}</td>
            </tr>
""")

        return f"""
        <table>
            <tr><th>Metric</th><th>Value</th><th>Metric</th><th>Value</th></tr>
            {''.join(rows)}
        </table>
"""

    @staticmethod
    def _create_tail_risk_table(risk_metrics: Dict) -> str:
        """Create tail risk analysis table HTML."""
        tail = risk_metrics.get('tail_risk', {})

        return f"""
        <table>
            <tr><th>Metric</th><th>Value</th><th>Description</th></tr>
            <tr>
                <td>VaR (95%, Historical)</td>
                <td>{_format_number(risk_metrics.get('var_95_historical', np.nan) * 100, 'ratio')}%</td>
                <td>Max expected daily loss at 95% confidence</td>
            </tr>
            <tr>
                <td>VaR (99%, Historical)</td>
                <td>{_format_number(risk_metrics.get('var_99_historical', np.nan) * 100, 'ratio')}%</td>
                <td>Max expected daily loss at 99% confidence</td>
            </tr>
            <tr>
                <td>CVaR (95%)</td>
                <td>{_format_number(risk_metrics.get('cvar_95', np.nan) * 100, 'ratio')}%</td>
                <td>Expected loss when VaR is breached</td>
            </tr>
            <tr>
                <td>Skewness</td>
                <td>{_format_number(risk_metrics.get('skewness', np.nan), 'ratio')}</td>
                <td>Negative = fat left tail (more downside risk)</td>
            </tr>
            <tr>
                <td>Kurtosis</td>
                <td>{_format_number(risk_metrics.get('kurtosis', np.nan), 'ratio')}</td>
                <td>Higher = fatter tails vs normal distribution</td>
            </tr>
            <tr>
                <td>Tail Ratio</td>
                <td>{_format_number(tail.get('tail_ratio', np.nan), 'ratio')}</td>
                <td>Right tail / Left tail (>1 = positive skew)</td>
            </tr>
        </table>
"""

    @staticmethod
    def _create_trade_log_table(trades: pd.DataFrame) -> str:
        """Create trade log table HTML."""
        closed = trades[trades['action'] == 'close'].copy()

        if closed.empty:
            return '<p>No closed trades</p>'

        # Select and format columns
        display_cols = ['timestamp', 'structure_type', 'underlying',
                       'realized_pnl', 'exit_reason']
        available_cols = [c for c in display_cols if c in closed.columns]

        if not available_cols:
            return '<p>No trade data available</p>'

        rows = []
        for _, trade in closed.iterrows():
            ts = trade.get('timestamp', '')
            if hasattr(ts, 'strftime'):
                ts = ts.strftime('%Y-%m-%d')

            pnl = trade.get('realized_pnl', 0)
            pnl_class = _get_value_class(pnl)

            rows.append(f"""
            <tr>
                <td>{ts}</td>
                <td>{trade.get('structure_type', 'N/A')}</td>
                <td>{trade.get('underlying', 'N/A')}</td>
                <td class="{pnl_class}">{_format_number(pnl, 'currency')}</td>
                <td>{trade.get('exit_reason', 'N/A')}</td>
            </tr>
""")

        return f"""
        <table>
            <tr>
                <th>Date</th>
                <th>Structure</th>
                <th>Underlying</th>
                <th>P&L</th>
                <th>Exit Reason</th>
            </tr>
            {''.join(rows[:50])}
        </table>
        <p style="color: #757575; font-size: 12px;">Showing first 50 trades</p>
"""


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Main class
    'Dashboard',

    # Exceptions
    'DashboardError',
    'DashboardDataError',
    'DashboardSaveError',
]
