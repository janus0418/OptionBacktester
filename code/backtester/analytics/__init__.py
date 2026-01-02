"""
Analytics Module for Options Backtesting

This module provides comprehensive performance and risk analytics for evaluating
options trading strategies. It integrates with the BacktestEngine to provide
industry-standard quantitative finance metrics.

Components:
    - PerformanceMetrics: Returns, Sharpe, Sortino, drawdown, trade statistics
    - RiskAnalytics: VaR, CVaR, Greeks analysis, tail risk, MAE
    - Visualization: Charts and plots (matplotlib/plotly)
    - Dashboard: Interactive HTML dashboards
    - ReportGenerator: HTML/PDF report generation

Usage:
    from backtester.analytics import (
        PerformanceMetrics, RiskAnalytics,
        Visualization, Dashboard, ReportGenerator
    )

    # After running a backtest
    results = engine.run()
    equity_curve = results['equity_curve']
    trade_log = results['trade_log']
    returns = equity_curve['equity'].pct_change().dropna()

    # Performance metrics
    sharpe = PerformanceMetrics.calculate_sharpe_ratio(returns)
    max_dd = PerformanceMetrics.calculate_max_drawdown(equity_curve)
    win_rate = PerformanceMetrics.calculate_win_rate(trade_log)

    # Risk metrics
    var_95 = RiskAnalytics.calculate_var(returns, 0.95)
    cvar_95 = RiskAnalytics.calculate_cvar(returns, 0.95)
    tail_risk = RiskAnalytics.calculate_tail_risk(returns)

    # Visualization
    Visualization.plot_equity_curve(equity_curve, backend='plotly')
    Visualization.plot_pnl_distribution(trade_log)

    # Dashboards and Reports
    Dashboard.create_performance_dashboard(results, metrics, 'dashboard.html')
    ReportGenerator.generate_html_report(results, metrics, 'report.html')

Metric Categories:

    Returns-Based Metrics:
        - Total Return: (Final - Initial) / Initial
        - Annualized Return (CAGR): Compound Annual Growth Rate
        - Sharpe Ratio: Risk-adjusted return vs. volatility
        - Sortino Ratio: Risk-adjusted return vs. downside volatility
        - Calmar Ratio: Return vs. maximum drawdown

    Drawdown Metrics:
        - Maximum Drawdown: Largest peak-to-trough decline
        - Drawdown Duration: Time from peak to trough
        - Recovery Time: Time from trough to new peak
        - Ulcer Index: RMS of drawdown over time

    Trade-Based Metrics:
        - Win Rate: Percentage of profitable trades
        - Profit Factor: Gross profit / Gross loss
        - Expectancy: Expected value per trade
        - Payoff Ratio: Average win / Average loss
        - Consecutive Wins/Losses: Streak analysis

    Risk Metrics:
        - Value at Risk (VaR): Maximum expected loss at confidence level
        - Conditional VaR (CVaR): Expected loss when VaR exceeded
        - Tail Risk: Skewness, kurtosis, tail ratios
        - Greeks Analysis: Delta, Gamma, Theta, Vega over time
        - Maximum Adverse Excursion: Worst intra-trade drawdown

Industry Benchmarks:
    Sharpe Ratio:
        - < 0: Poor
        - 0-1: Acceptable
        - 1-2: Good
        - > 2: Excellent

    Sortino Ratio:
        - Similar to Sharpe but using downside deviation
        - Typically higher than Sharpe for strategies with positive skew

    Calmar Ratio:
        - < 0.5: Poor
        - 0.5-1.0: Acceptable
        - 1.0-3.0: Good
        - > 3.0: Excellent

    Win Rate (options strategies):
        - Credit spreads: 60-80%
        - Directional trades: 40-60%
        - High win rate != high profitability

    Profit Factor:
        - < 1.0: Unprofitable
        - 1.0-1.5: Marginal
        - 1.5-2.0: Good
        - > 2.0: Excellent

References:
    - Sharpe, W.F. (1994). The Sharpe Ratio. Journal of Portfolio Management.
    - Sortino, F.A. (1994). Performance Measurement in a Downside Risk Framework.
    - Jorion, P. (2007). Value at Risk.
    - Hull, J.C. (2018). Options, Futures, and Other Derivatives.
"""

from backtester.analytics.metrics import (
    # Main class
    PerformanceMetrics,
    # Exceptions
    MetricsError,
    InsufficientDataError as MetricsInsufficientDataError,
    InvalidDataError as MetricsInvalidDataError,
    # Constants
    TRADING_DAYS_PER_YEAR,
    DEFAULT_RISK_FREE_RATE,
    EPSILON as METRICS_EPSILON,
)

from backtester.analytics.risk import (
    # Main class
    RiskAnalytics,
    # Exceptions
    RiskAnalyticsError,
    InsufficientDataError as RiskInsufficientDataError,
    InvalidDataError as RiskInvalidDataError,
    InvalidConfidenceError,
    # Constants
    VAR_CONFIDENCE_95,
    VAR_CONFIDENCE_99,
    Z_SCORES,
    GREEK_NAMES,
)

from backtester.analytics.visualization import (
    # Main class
    Visualization,
    # Exceptions
    VisualizationError,
    InvalidBackendError,
    InsufficientDataError as VisualizationInsufficientDataError,
    SaveError as VisualizationSaveError,
    # Constants
    SUPPORTED_BACKENDS,
    COLOR_PROFIT,
    COLOR_LOSS,
    COLOR_NEUTRAL,
    COLOR_EQUITY,
    COLOR_DRAWDOWN,
)

from backtester.analytics.dashboard import (
    # Main class
    Dashboard,
    # Exceptions
    DashboardError,
    DashboardDataError,
    DashboardSaveError,
)

from backtester.analytics.report import (
    # Main class
    ReportGenerator,
    # Exceptions
    ReportError,
    ReportDataError,
    ReportSaveError,
)

from backtester.analytics.monte_carlo import (
    # Main class
    MonteCarloSimulator,
    # Data classes
    SimulationResult,
    ConfidenceInterval,
    # Exceptions
    MonteCarloError,
    InsufficientDataError as MonteCarloInsufficientDataError,
    InvalidParameterError,
    SimulationError,
    # Constants
    DEFAULT_NUM_SIMULATIONS,
    DEFAULT_NUM_PERIODS,
    DEFAULT_CONFIDENCE_LEVEL,
)

from backtester.analytics.scenario_testing import (
    # Main class
    ScenarioTester,
    # Data classes
    Scenario,
    ScenarioResult,
    SensitivityResult,
    # Enums
    ScenarioType,
    # Exceptions
    ScenarioError,
    InvalidScenarioError,
    InsufficientDataError as ScenarioInsufficientDataError,
    # Predefined scenarios
    STRESS_SCENARIOS,
    HISTORICAL_SCENARIOS,
    # Utility functions
    get_predefined_scenario,
    list_available_scenarios,
)


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "PerformanceMetrics",
    "MetricsError",
    "MetricsInsufficientDataError",
    "MetricsInvalidDataError",
    "RiskAnalytics",
    "RiskAnalyticsError",
    "RiskInsufficientDataError",
    "RiskInvalidDataError",
    "InvalidConfidenceError",
    "Visualization",
    "VisualizationError",
    "InvalidBackendError",
    "VisualizationInsufficientDataError",
    "VisualizationSaveError",
    "Dashboard",
    "DashboardError",
    "DashboardDataError",
    "DashboardSaveError",
    "ReportGenerator",
    "ReportError",
    "ReportDataError",
    "ReportSaveError",
    "MonteCarloSimulator",
    "SimulationResult",
    "ConfidenceInterval",
    "MonteCarloError",
    "MonteCarloInsufficientDataError",
    "InvalidParameterError",
    "SimulationError",
    "ScenarioTester",
    "Scenario",
    "ScenarioResult",
    "SensitivityResult",
    "ScenarioType",
    "ScenarioError",
    "InvalidScenarioError",
    "ScenarioInsufficientDataError",
    "STRESS_SCENARIOS",
    "HISTORICAL_SCENARIOS",
    "get_predefined_scenario",
    "list_available_scenarios",
    "TRADING_DAYS_PER_YEAR",
    "DEFAULT_RISK_FREE_RATE",
    "VAR_CONFIDENCE_95",
    "VAR_CONFIDENCE_99",
    "Z_SCORES",
    "GREEK_NAMES",
    "SUPPORTED_BACKENDS",
    "COLOR_PROFIT",
    "COLOR_LOSS",
    "COLOR_NEUTRAL",
    "COLOR_EQUITY",
    "COLOR_DRAWDOWN",
    "DEFAULT_NUM_SIMULATIONS",
    "DEFAULT_NUM_PERIODS",
    "DEFAULT_CONFIDENCE_LEVEL",
]
