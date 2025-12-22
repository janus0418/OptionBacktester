#!/usr/bin/env python3
"""
Example 5: Complete Analysis Workflow

This example demonstrates the complete workflow:
    - Run backtest
    - Calculate comprehensive metrics
    - Generate visualizations
    - Create dashboard
    - Generate HTML/PDF report

This is a comprehensive example showing all capabilities of the system.

Difficulty: Advanced
Time to run: ~30 seconds
"""

import sys
from pathlib import Path
code_dir = Path(__file__).parent.parent / 'code'
sys.path.insert(0, str(code_dir))

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import tempfile
import os
from unittest.mock import Mock

from backtester.strategies import ShortStraddleHighIVStrategy, IronCondorStrategy
from backtester.engine.backtest_engine import BacktestEngine
from backtester.engine.data_stream import DataStream
from backtester.engine.execution import ExecutionModel
from backtester.analytics import (
    PerformanceMetrics, RiskAnalytics, Visualization,
    Dashboard, ReportGenerator
)
from backtester.data.dolt_adapter import DoltAdapter
from backtester.data.market_data import MarketData, OptionChain
from backtester.core.pricing import BlackScholesPricer


# Mock data functions (reuse from previous examples)
def create_mock_market_data(start_date, end_date, initial_price=450.0):
    dates = pd.bdate_range(start=start_date, end=end_date)
    np.random.seed(42)
    returns = np.random.normal(0.0008, 0.018, len(dates))  # Slightly positive drift
    prices = initial_price * np.exp(np.cumsum(returns))
    market_data = []
    for date, close in zip(dates, prices):
        daily_vol = close * 0.018
        open_price = close + np.random.normal(0, daily_vol * 0.5)
        high = max(open_price, close) + abs(np.random.normal(0, daily_vol * 0.3))
        low = min(open_price, close) - abs(np.random.normal(0, daily_vol * 0.3))
        market_data.append({
            'date': date, 'open': open_price, 'high': high,
            'low': low, 'close': close,
            'volume': int(np.random.uniform(50_000_000, 100_000_000))
        })
    return pd.DataFrame(market_data)


def create_mock_option_chain(underlying_price, date, expiration, iv=0.20):
    pricer = BlackScholesPricer()
    T = (expiration - date).days / 365.25
    r = 0.04
    strikes = [round(underlying_price * pct / 100, 2) for pct in range(85, 116, 1)]
    calls, puts = [], []
    for K in strikes:
        call_price = pricer.price(underlying_price, K, T, r, iv, 'call')
        put_price = pricer.price(underlying_price, K, T, r, iv, 'put')
        call_greeks = pricer.calculate_greeks(underlying_price, K, T, r, iv, 'call')
        put_greeks = pricer.calculate_greeks(underlying_price, K, T, r, iv, 'put')
        calls.append({
            'underlying': 'SPY', 'strike': K, 'expiration': expiration,
            'option_type': 'call', 'bid': call_price * 0.98, 'ask': call_price * 1.02,
            'mid': call_price, 'volume': 1000, 'open_interest': 5000,
            'implied_volatility': iv, **call_greeks
        })
        puts.append({
            'underlying': 'SPY', 'strike': K, 'expiration': expiration,
            'option_type': 'put', 'bid': put_price * 0.98, 'ask': put_price * 1.02,
            'mid': put_price, 'volume': 1000, 'open_interest': 5000,
            'implied_volatility': iv, **put_greeks
        })
    return OptionChain(underlying='SPY', date=date, expiration=expiration, calls=calls, puts=puts)


def setup_mock_data_stream(start_date, end_date):
    mock_adapter = Mock(spec=DoltAdapter)
    mock_adapter.is_connected.return_value = True
    data_stream = DataStream(mock_adapter, start_date, end_date, 'SPY')
    market_data_df = create_mock_market_data(start_date, end_date)

    def mock_get_market_data(date):
        row = market_data_df[market_data_df['date'] == pd.Timestamp(date)]
        if row.empty: return None
        return MarketData('SPY', date, row['open'].iloc[0], row['high'].iloc[0],
                         row['low'].iloc[0], row['close'].iloc[0], int(row['volume'].iloc[0]))

    def mock_get_option_chain(date, expiration=None):
        market_data = mock_get_market_data(date)
        if market_data is None: return None
        if expiration is None: expiration = date + timedelta(days=30)
        return create_mock_option_chain(market_data.close, date, expiration, 0.22)

    data_stream.get_market_data = mock_get_market_data
    data_stream.get_option_chain = mock_get_option_chain
    return data_stream


def run_backtest(strategy_class, strategy_params, start_date, end_date, initial_capital):
    """Run a backtest and return results."""
    strategy = strategy_class(**strategy_params)
    data_stream = setup_mock_data_stream(start_date, end_date)
    execution_model = ExecutionModel(commission_per_contract=0.65, slippage_pct=0.01, fill_on='mid')

    engine = BacktestEngine(strategy, data_stream, execution_model, initial_capital)
    return engine.run()


def calculate_all_metrics(results):
    """Calculate comprehensive performance and risk metrics."""
    equity_curve = results['equity_curve']
    trade_log = results['trade_log']
    returns = equity_curve['equity'].pct_change().dropna()

    metrics = {}

    # Performance metrics
    if len(returns) > 1:
        metrics['total_return'] = PerformanceMetrics.calculate_total_return(equity_curve)
        metrics['annualized_return'] = PerformanceMetrics.calculate_annualized_return(equity_curve)
        metrics['sharpe_ratio'] = PerformanceMetrics.calculate_sharpe_ratio(returns)
        metrics['sortino_ratio'] = PerformanceMetrics.calculate_sortino_ratio(returns)
        metrics['calmar_ratio'] = PerformanceMetrics.calculate_calmar_ratio(equity_curve, returns)
        metrics['max_drawdown'] = PerformanceMetrics.calculate_max_drawdown(equity_curve)

        # Risk metrics
        metrics['var_95'] = RiskAnalytics.calculate_var(returns, 0.95)
        metrics['cvar_95'] = RiskAnalytics.calculate_cvar(returns, 0.95)
        metrics['var_99'] = RiskAnalytics.calculate_var(returns, 0.99)
        metrics['cvar_99'] = RiskAnalytics.calculate_cvar(returns, 0.99)

    # Trade metrics
    if len(trade_log) > 0:
        metrics['num_trades'] = len(trade_log)
        metrics['win_rate'] = PerformanceMetrics.calculate_win_rate(trade_log)
        metrics['profit_factor'] = PerformanceMetrics.calculate_profit_factor(trade_log)
        metrics['avg_win'] = PerformanceMetrics.calculate_average_win(trade_log)
        metrics['avg_loss'] = PerformanceMetrics.calculate_average_loss(trade_log)
        metrics['expectancy'] = PerformanceMetrics.calculate_expectancy(trade_log)
        metrics['payoff_ratio'] = PerformanceMetrics.calculate_payoff_ratio(trade_log)

    return metrics


def main():
    print("="*70)
    print("Example 5: Complete Analysis Workflow")
    print("="*70)
    print()

    # Setup
    start_date = datetime(2024, 1, 2)
    end_date = datetime(2024, 6, 28)  # 6 months
    initial_capital = 100000.0

    print("Running multiple strategies for comparison...")
    print()

    # Run multiple strategies
    strategies_to_test = [
        (ShortStraddleHighIVStrategy, {
            'name': 'Short Straddle',
            'initial_capital': initial_capital,
            'iv_rank_threshold': 50,
            'profit_target_pct': 0.50,
            'stop_loss_pct': 2.0
        }),
        (IronCondorStrategy, {
            'name': 'Iron Condor',
            'initial_capital': initial_capital,
            'target_delta': 0.10,
            'wing_width': 5.0,
            'profit_target_pct': 0.50,
            'stop_loss_pct': 2.0
        })
    ]

    all_results = []
    all_metrics = []

    for i, (strategy_class, params) in enumerate(strategies_to_test, 1):
        print(f"[{i}/{len(strategies_to_test)}] Running {params['name']}...")
        results = run_backtest(strategy_class, params, start_date, end_date, initial_capital)
        metrics = calculate_all_metrics(results)

        all_results.append({
            'name': params['name'],
            'results': results,
            'metrics': metrics
        })
        all_metrics.append(metrics)

        print(f"    Final Equity: ${results['final_equity']:,.2f}")
        print(f"    Total Return: {metrics.get('total_return', 0)*100:.2f}%")
        print()

    # Comparison table
    print("="*70)
    print("STRATEGY COMPARISON")
    print("="*70)
    print()

    print(f"{'Metric':<25} {'Short Straddle':>20} {'Iron Condor':>20}")
    print("-" * 70)

    comparison_metrics = [
        ('Final Equity', 'final_equity', lambda x: f"${x:,.2f}"),
        ('Total Return', 'total_return', lambda x: f"{x*100:.2f}%"),
        ('Sharpe Ratio', 'sharpe_ratio', lambda x: f"{x:.2f}" if x else "N/A"),
        ('Max Drawdown', 'max_drawdown', lambda x: f"{x*100:.2f}%"),
        ('Win Rate', 'win_rate', lambda x: f"{x*100:.1f}%"),
        ('Num Trades', 'num_trades', lambda x: f"{x}"),
    ]

    for metric_name, metric_key, formatter in comparison_metrics:
        values = []
        for result in all_results:
            if metric_key == 'final_equity':
                val = result['results'][metric_key]
            else:
                val = result['metrics'].get(metric_key)

            if val is not None:
                values.append(formatter(val))
            else:
                values.append("N/A")

        print(f"{metric_name:<25} {values[0]:>20} {values[1]:>20}")

    print()

    # Generate comprehensive visualizations and reports
    print("="*70)
    print("GENERATING COMPREHENSIVE ANALYSIS")
    print("="*70)
    print()

    output_dir = tempfile.mkdtemp()
    print(f"Output directory: {output_dir}")
    print()

    for result_data in all_results:
        name = result_data['name'].replace(' ', '_').lower()
        results = result_data['results']
        metrics = result_data['metrics']

        print(f"Generating analysis for {result_data['name']}...")

        # 1. Equity curve
        equity_path = os.path.join(output_dir, f'{name}_equity.png')
        Visualization.plot_equity_curve(
            results['equity_curve'],
            backend='matplotlib',
            save_path=equity_path
        )
        print(f"  ✓ Equity curve: {equity_path}")

        # 2. Drawdown chart
        if len(results['equity_curve']) > 1:
            dd_path = os.path.join(output_dir, f'{name}_drawdown.png')
            Visualization.plot_drawdown(
                results['equity_curve'],
                backend='matplotlib',
                save_path=dd_path
            )
            print(f"  ✓ Drawdown chart: {dd_path}")

        # 3. P&L distribution
        if len(results['trade_log']) > 0:
            pnl_path = os.path.join(output_dir, f'{name}_pnl_dist.png')
            Visualization.plot_pnl_distribution(
                results['trade_log'],
                backend='matplotlib',
                save_path=pnl_path
            )
            print(f"  ✓ P&L distribution: {pnl_path}")

        # 4. Interactive dashboard
        dashboard_path = os.path.join(output_dir, f'{name}_dashboard.html')
        Dashboard.create_performance_dashboard(
            results,
            metrics,
            save_path=dashboard_path
        )
        print(f"  ✓ Interactive dashboard: {dashboard_path}")

        # 5. HTML report
        report_path = os.path.join(output_dir, f'{name}_report.html')
        ReportGenerator.generate_html_report(
            results,
            metrics,
            save_path=report_path
        )
        print(f"  ✓ HTML report: {report_path}")
        print()

    print("="*70)
    print("COMPLETE ANALYSIS GENERATED")
    print("="*70)
    print()
    print(f"All files saved to: {output_dir}")
    print()
    print("Generated files:")
    print("  - Equity curves (PNG)")
    print("  - Drawdown charts (PNG)")
    print("  - P&L distributions (PNG)")
    print("  - Interactive dashboards (HTML)")
    print("  - Comprehensive reports (HTML)")
    print()
    print("Open dashboards in your browser:")
    for result_data in all_results:
        name = result_data['name'].replace(' ', '_').lower()
        dashboard_path = os.path.join(output_dir, f'{name}_dashboard.html')
        print(f"  file://{dashboard_path}")
    print()
    print("="*70)
    print("Example complete!")
    print("="*70)


if __name__ == '__main__':
    main()
