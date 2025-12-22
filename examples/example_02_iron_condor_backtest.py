#!/usr/bin/env python3
"""
Example 2: Iron Condor Strategy Backtest

This example demonstrates running a backtest with the Iron Condor strategy
and generating a visual dashboard of results.

What this example demonstrates:
    - Using IronCondorStrategy
    - Delta-based strike selection
    - Generating performance dashboards
    - Advanced result analysis

Difficulty: Intermediate
Time to run: ~15 seconds
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

from backtester.strategies import IronCondorStrategy
from backtester.engine.backtest_engine import BacktestEngine
from backtester.engine.data_stream import DataStream
from backtester.engine.execution import ExecutionModel
from backtester.analytics import PerformanceMetrics, RiskAnalytics, Dashboard
from backtester.data.dolt_adapter import DoltAdapter
from backtester.data.market_data import MarketData, OptionChain
from backtester.core.pricing import BlackScholesPricer
from unittest.mock import Mock


# Reuse mock data functions from example_01
def create_mock_market_data(start_date, end_date, initial_price=450.0):
    dates = pd.bdate_range(start=start_date, end=end_date)
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.015, len(dates))
    prices = initial_price * np.exp(np.cumsum(returns))
    market_data = []
    for date, close in zip(dates, prices):
        daily_vol = close * 0.015
        open_price = close + np.random.normal(0, daily_vol * 0.5)
        high = max(open_price, close) + abs(np.random.normal(0, daily_vol * 0.3))
        low = min(open_price, close) - abs(np.random.normal(0, daily_vol * 0.3))
        market_data.append({'date': date, 'open': open_price, 'high': high,
                           'low': low, 'close': close, 'volume': int(np.random.uniform(50_000_000, 100_000_000))})
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
        calls.append({'underlying': 'SPY', 'strike': K, 'expiration': expiration, 'option_type': 'call',
                     'bid': call_price * 0.98, 'ask': call_price * 1.02, 'mid': call_price,
                     'volume': int(1000 * np.exp(-0.5 * ((K - underlying_price) / (0.1 * underlying_price)) ** 2)),
                     'open_interest': int(5000 * np.exp(-0.5 * ((K - underlying_price) / (0.1 * underlying_price)) ** 2)),
                     'implied_volatility': iv, **call_greeks})
        puts.append({'underlying': 'SPY', 'strike': K, 'expiration': expiration, 'option_type': 'put',
                    'bid': put_price * 0.98, 'ask': put_price * 1.02, 'mid': put_price,
                    'volume': int(1000 * np.exp(-0.5 * ((K - underlying_price) / (0.1 * underlying_price)) ** 2)),
                    'open_interest': int(5000 * np.exp(-0.5 * ((K - underlying_price) / (0.1 * underlying_price)) ** 2)),
                    'implied_volatility': iv, **put_greeks})
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
        return create_mock_option_chain(market_data.close, date, expiration, 0.20)

    data_stream.get_market_data = mock_get_market_data
    data_stream.get_option_chain = mock_get_option_chain
    return data_stream


def main():
    print("="*70)
    print("Example 2: Iron Condor Strategy Backtest")
    print("="*70)
    print()

    # Setup
    start_date = datetime(2024, 1, 2)
    end_date = datetime(2024, 3, 29)
    initial_capital = 100000.0

    # Create Iron Condor strategy
    print("Creating Iron Condor strategy...")
    strategy = IronCondorStrategy(
        name='Iron Condor Delta 10',
        initial_capital=initial_capital,
        target_delta=0.10,        # Sell 10-delta options
        wing_width=5.0,           # $5 wide wings
        profit_target_pct=0.50,   # 50% profit target
        stop_loss_pct=2.0,        # 200% stop loss
        max_dte=45,
        min_dte=7
    )
    print(f"  Target Delta: {strategy.target_delta}")
    print(f"  Wing Width: ${strategy.wing_width}")
    print()

    # Setup data and execution
    data_stream = setup_mock_data_stream(start_date, end_date)
    execution_model = ExecutionModel(commission_per_contract=0.65, slippage_pct=0.01, fill_on='mid')

    # Run backtest
    print("Running backtest...")
    engine = BacktestEngine(strategy, data_stream, execution_model, initial_capital)
    results = engine.run()
    print("Backtest complete!")
    print()

    # Calculate metrics
    print("="*70)
    print("RESULTS")
    print("="*70)
    print()

    print("Performance:")
    print(f"  Final Equity:   ${results['final_equity']:,.2f}")
    print(f"  Total Return:   {results['total_return']*100:,.2f}%")
    print(f"  Total P&L:      ${results['final_equity'] - initial_capital:,.2f}")
    print()

    trade_log = results['trade_log']
    if len(trade_log) > 0:
        winning_trades = trade_log[trade_log['pnl'] > 0]
        win_rate = len(winning_trades) / len(trade_log) * 100
        print(f"  Total Trades:   {len(trade_log)}")
        print(f"  Win Rate:       {win_rate:.1f}%")
        print(f"  Avg P&L:        ${trade_log['pnl'].mean():,.2f}")
        print(f"  Best Trade:     ${trade_log['pnl'].max():,.2f}")
        print(f"  Worst Trade:    ${trade_log['pnl'].min():,.2f}")
        print()

    equity_curve = results['equity_curve']
    returns = equity_curve['equity'].pct_change().dropna()

    if len(returns) > 1:
        print("Risk Metrics:")
        sharpe = PerformanceMetrics.calculate_sharpe_ratio(returns)
        sortino = PerformanceMetrics.calculate_sortino_ratio(returns)
        max_dd = PerformanceMetrics.calculate_max_drawdown(equity_curve)
        var_95 = RiskAnalytics.calculate_var(returns, 0.95)
        cvar_95 = RiskAnalytics.calculate_cvar(returns, 0.95)

        if sharpe: print(f"  Sharpe Ratio:   {sharpe:.2f}")
        if sortino: print(f"  Sortino Ratio:  {sortino:.2f}")
        print(f"  Max Drawdown:   {max_dd*100:.2f}%")
        if var_95: print(f"  VaR (95%):      {var_95*100:.2f}%")
        if cvar_95: print(f"  CVaR (95%):     {cvar_95*100:.2f}%")
        print()

    # Generate dashboard
    print("Generating dashboard...")
    metrics = {
        'total_return': results['total_return'],
        'sharpe_ratio': sharpe if 'sharpe' in locals() else None,
        'max_drawdown': max_dd if 'max_dd' in locals() else None,
        'win_rate': win_rate if 'win_rate' in locals() else None,
    }

    dashboard_path = tempfile.mktemp(suffix='.html')
    Dashboard.create_performance_dashboard(results, metrics, dashboard_path)
    print(f"Dashboard saved to: {dashboard_path}")
    print(f"Open in browser: file://{dashboard_path}")
    print()

    print("="*70)
    print("Example complete!")
    print("="*70)


if __name__ == '__main__':
    main()
