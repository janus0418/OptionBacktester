#!/usr/bin/env python3
"""
Example 1: Simple Backtest with Short Straddle Strategy

This is a beginner-friendly example showing the basic workflow for running
a backtest with the Options Backtesting System.

What this example demonstrates:
    - Creating mock data for testing
    - Setting up a simple strategy (Short Straddle)
    - Running a backtest
    - Displaying results

Difficulty: Beginner
Time to run: < 10 seconds
"""

import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

# Add the code directory to path so we can import backtester
code_dir = Path(__file__).parent.parent / 'code'
sys.path.insert(0, str(code_dir))

import pandas as pd
import numpy as np

# Import backtester components
from backtester.strategies import ShortStraddleHighIVStrategy
from backtester.engine.backtest_engine import BacktestEngine
from backtester.engine.data_stream import DataStream
from backtester.engine.execution import ExecutionModel
from backtester.analytics import PerformanceMetrics
from backtester.data.dolt_adapter import DoltAdapter
from backtester.data.market_data import MarketData, OptionChain
from backtester.core.pricing import BlackScholesPricer

# Mock data creation utilities
from unittest.mock import Mock


def create_mock_market_data(start_date, end_date, initial_price=450.0):
    """
    Create mock market data for testing.

    In a real application, this would come from your database or data provider.
    """
    dates = pd.bdate_range(start=start_date, end=end_date)

    # Generate realistic price movement (geometric Brownian motion)
    np.random.seed(42)  # For reproducibility
    returns = np.random.normal(0.0005, 0.015, len(dates))
    prices = initial_price * np.exp(np.cumsum(returns))

    market_data = []
    for date, close in zip(dates, prices):
        daily_vol = close * 0.015
        open_price = close + np.random.normal(0, daily_vol * 0.5)
        high = max(open_price, close) + abs(np.random.normal(0, daily_vol * 0.3))
        low = min(open_price, close) - abs(np.random.normal(0, daily_vol * 0.3))

        market_data.append({
            'date': date,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': int(np.random.uniform(50_000_000, 100_000_000))
        })

    return pd.DataFrame(market_data)


def create_mock_option_chain(underlying_price, date, expiration, iv=0.20):
    """
    Create a mock option chain with realistic pricing.

    In a real application, this would come from your database or market data provider.
    """
    pricer = BlackScholesPricer()
    T = (expiration - date).days / 365.25
    r = 0.04

    # Generate strikes around current price
    strikes = []
    for pct in range(85, 116, 1):  # 85% to 115% of spot
        strike = round(underlying_price * pct / 100, 2)
        strikes.append(strike)

    calls = []
    puts = []

    for K in strikes:
        # Calculate prices using Black-Scholes
        call_price = pricer.price(underlying_price, K, T, r, iv, 'call')
        put_price = pricer.price(underlying_price, K, T, r, iv, 'put')

        # Calculate Greeks
        call_greeks = pricer.calculate_greeks(underlying_price, K, T, r, iv, 'call')
        put_greeks = pricer.calculate_greeks(underlying_price, K, T, r, iv, 'put')

        # Create call data
        calls.append({
            'underlying': 'SPY',
            'strike': K,
            'expiration': expiration,
            'option_type': 'call',
            'bid': call_price * 0.98,
            'ask': call_price * 1.02,
            'mid': call_price,
            'volume': int(1000 * np.exp(-0.5 * ((K - underlying_price) / (0.1 * underlying_price)) ** 2)),
            'open_interest': int(5000 * np.exp(-0.5 * ((K - underlying_price) / (0.1 * underlying_price)) ** 2)),
            'implied_volatility': iv,
            'delta': call_greeks['delta'],
            'gamma': call_greeks['gamma'],
            'theta': call_greeks['theta'],
            'vega': call_greeks['vega'],
            'rho': call_greeks['rho'],
        })

        # Create put data
        puts.append({
            'underlying': 'SPY',
            'strike': K,
            'expiration': expiration,
            'option_type': 'put',
            'bid': put_price * 0.98,
            'ask': put_price * 1.02,
            'mid': put_price,
            'volume': int(1000 * np.exp(-0.5 * ((K - underlying_price) / (0.1 * underlying_price)) ** 2)),
            'open_interest': int(5000 * np.exp(-0.5 * ((K - underlying_price) / (0.1 * underlying_price)) ** 2)),
            'implied_volatility': iv,
            'delta': put_greeks['delta'],
            'gamma': put_greeks['gamma'],
            'theta': put_greeks['theta'],
            'vega': put_greeks['vega'],
            'rho': put_greeks['rho'],
        })

    return OptionChain(
        underlying='SPY',
        date=date,
        expiration=expiration,
        calls=calls,
        puts=puts
    )


def setup_mock_data_stream(start_date, end_date):
    """
    Set up a mock data stream for testing.

    This creates a DataStream that provides mock market data instead of
    querying a real database.
    """
    # Create mock adapter
    mock_adapter = Mock(spec=DoltAdapter)
    mock_adapter.is_connected.return_value = True

    # Create data stream
    data_stream = DataStream(
        adapter=mock_adapter,
        start_date=start_date,
        end_date=end_date,
        underlying='SPY'
    )

    # Generate mock market data
    market_data_df = create_mock_market_data(start_date, end_date)

    def mock_get_market_data(date):
        """Return market data for a specific date."""
        row = market_data_df[market_data_df['date'] == pd.Timestamp(date)]
        if row.empty:
            return None

        return MarketData(
            underlying='SPY',
            date=date,
            open=row['open'].iloc[0],
            high=row['high'].iloc[0],
            low=row['low'].iloc[0],
            close=row['close'].iloc[0],
            volume=int(row['volume'].iloc[0])
        )

    def mock_get_option_chain(date, expiration=None):
        """Return option chain for a specific date."""
        market_data = mock_get_market_data(date)
        if market_data is None:
            return None

        # Default expiration is 30 days out
        if expiration is None:
            expiration = date + timedelta(days=30)

        return create_mock_option_chain(
            underlying_price=market_data.close,
            date=date,
            expiration=expiration,
            iv=0.20  # 20% implied volatility
        )

    # Patch the data stream methods
    data_stream.get_market_data = mock_get_market_data
    data_stream.get_option_chain = mock_get_option_chain

    return data_stream


def main():
    """
    Run a simple backtest example.
    """
    print("="*70)
    print("Options Backtesting System - Example 1: Simple Backtest")
    print("="*70)
    print()

    # Step 1: Define backtest parameters
    print("Step 1: Defining backtest parameters...")
    start_date = datetime(2024, 1, 2)
    end_date = datetime(2024, 3, 29)
    initial_capital = 100000.0
    print(f"  Start Date: {start_date.date()}")
    print(f"  End Date: {end_date.date()}")
    print(f"  Initial Capital: ${initial_capital:,.2f}")
    print()

    # Step 2: Create the strategy
    print("Step 2: Creating Short Straddle strategy...")
    strategy = ShortStraddleHighIVStrategy(
        name='Short Straddle Example',
        initial_capital=initial_capital,
        iv_rank_threshold=50,      # Enter when IV rank > 50%
        profit_target_pct=0.50,    # Take profit at 50% of max profit
        stop_loss_pct=2.0,         # Stop loss at 200% of max profit
        max_dte=45,                # Maximum days to expiration
        min_dte=7                  # Minimum days to expiration
    )
    print(f"  Strategy: {strategy.name}")
    print(f"  IV Rank Threshold: {strategy.iv_rank_threshold}%")
    print(f"  Profit Target: {strategy.profit_target_pct*100}%")
    print()

    # Step 3: Set up data stream
    print("Step 3: Setting up mock data stream...")
    data_stream = setup_mock_data_stream(start_date, end_date)
    print("  Mock data stream created with realistic option prices")
    print()

    # Step 4: Create execution model
    print("Step 4: Creating execution model...")
    execution_model = ExecutionModel(
        commission_per_contract=0.65,  # $0.65 per contract
        slippage_pct=0.01,             # 1% slippage
        fill_on='mid'                  # Fill at mid price
    )
    print(f"  Commission: ${execution_model.commission_per_contract}/contract")
    print(f"  Slippage: {execution_model.slippage_pct*100}%")
    print()

    # Step 5: Create and run backtest engine
    print("Step 5: Running backtest...")
    engine = BacktestEngine(
        strategy=strategy,
        data_stream=data_stream,
        execution_model=execution_model,
        initial_capital=initial_capital
    )

    results = engine.run()
    print("  Backtest complete!")
    print()

    # Step 6: Display results
    print("="*70)
    print("BACKTEST RESULTS")
    print("="*70)
    print()

    # Overall performance
    print("Overall Performance:")
    print(f"  Initial Capital:    ${initial_capital:,.2f}")
    print(f"  Final Equity:       ${results['final_equity']:,.2f}")
    print(f"  Total Return:       {results['total_return']*100:,.2f}%")
    print(f"  Total P&L:          ${results['final_equity'] - initial_capital:,.2f}")
    print()

    # Trade statistics
    trade_log = results['trade_log']
    if len(trade_log) > 0:
        print("Trade Statistics:")
        print(f"  Total Trades:       {len(trade_log)}")

        # Calculate win rate
        winning_trades = trade_log[trade_log['pnl'] > 0]
        win_rate = len(winning_trades) / len(trade_log) * 100 if len(trade_log) > 0 else 0
        print(f"  Win Rate:           {win_rate:.1f}%")

        # Average P&L
        avg_pnl = trade_log['pnl'].mean()
        print(f"  Average P&L:        ${avg_pnl:,.2f}")

        # Best and worst trades
        best_trade = trade_log['pnl'].max()
        worst_trade = trade_log['pnl'].min()
        print(f"  Best Trade:         ${best_trade:,.2f}")
        print(f"  Worst Trade:        ${worst_trade:,.2f}")
        print()
    else:
        print("Trade Statistics:")
        print("  No trades were executed during this backtest period.")
        print("  Try adjusting strategy parameters to generate trades.")
        print()

    # Performance metrics
    equity_curve = results['equity_curve']
    returns = equity_curve['equity'].pct_change().dropna()

    if len(returns) > 1:
        print("Performance Metrics:")

        # Sharpe Ratio
        sharpe = PerformanceMetrics.calculate_sharpe_ratio(returns)
        if sharpe is not None:
            print(f"  Sharpe Ratio:       {sharpe:.2f}")

        # Maximum Drawdown
        max_dd = PerformanceMetrics.calculate_max_drawdown(equity_curve)
        print(f"  Max Drawdown:       {max_dd*100:.2f}%")

        # Sortino Ratio
        sortino = PerformanceMetrics.calculate_sortino_ratio(returns)
        if sortino is not None:
            print(f"  Sortino Ratio:      {sortino:.2f}")

        print()

    # Greeks summary
    greeks_history = results['greeks_history']
    if len(greeks_history) > 0:
        print("Greeks Summary (Final Values):")
        final_greeks = greeks_history.iloc[-1]
        print(f"  Delta:              {final_greeks['delta']:.2f}")
        print(f"  Gamma:              {final_greeks['gamma']:.4f}")
        print(f"  Theta:              {final_greeks['theta']:.2f}")
        print(f"  Vega:               {final_greeks['vega']:.2f}")
        print()

    print("="*70)
    print("Backtest complete! Check the results above.")
    print()
    print("Next steps:")
    print("  - Try adjusting strategy parameters")
    print("  - Run example_02_iron_condor_backtest.py for a different strategy")
    print("  - See example_05_full_analysis.py for comprehensive reporting")
    print("="*70)


if __name__ == '__main__':
    main()
