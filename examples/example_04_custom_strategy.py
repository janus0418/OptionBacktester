#!/usr/bin/env python3
"""
Example 4: Creating a Custom Strategy

This example shows how to create your own custom trading strategy by
inheriting from the Strategy base class.

What this example demonstrates:
    - Inheriting from Strategy
    - Implementing custom entry logic
    - Implementing custom exit logic
    - Position sizing
    - Risk management

Difficulty: Advanced
Time to run: ~15 seconds
"""

import sys
from pathlib import Path
code_dir = Path(__file__).parent.parent / 'code'
sys.path.insert(0, str(code_dir))

from datetime import datetime, timedelta
from typing import Optional, List
import pandas as pd
import numpy as np
from unittest.mock import Mock

from backtester.strategies import Strategy
from backtester.core.option_structure import OptionStructure
from backtester.structures import ShortStrangle
from backtester.engine.backtest_engine import BacktestEngine
from backtester.engine.data_stream import DataStream
from backtester.engine.execution import ExecutionModel
from backtester.analytics import PerformanceMetrics
from backtester.data.dolt_adapter import DoltAdapter
from backtester.data.market_data import MarketData, OptionChain
from backtester.core.pricing import BlackScholesPricer


class CustomMeanReversionStrategy(Strategy):
    """
    Custom strategy that sells premium when the underlying is overextended.

    Entry Criteria:
        - Underlying is more than 2 standard deviations from 20-day SMA
        - IV rank > 50%
        - No existing positions

    Exit Criteria:
        - 50% profit target reached
        - Price returns to SMA (mean reversion complete)
        - 7 DTE or less remaining
        - Stop loss at 200% of credit received
    """

    def __init__(
        self,
        name: str,
        initial_capital: float,
        lookback_period: int = 20,
        entry_std_devs: float = 2.0,
        iv_rank_threshold: float = 50.0,
        profit_target_pct: float = 0.50,
        stop_loss_pct: float = 2.0,
        min_dte: int = 7
    ):
        """Initialize custom strategy with specific parameters."""
        super().__init__(name, initial_capital)

        # Custom parameters
        self.lookback_period = lookback_period
        self.entry_std_devs = entry_std_devs
        self.iv_rank_threshold = iv_rank_threshold
        self.profit_target_pct = profit_target_pct
        self.stop_loss_pct = stop_loss_pct
        self.min_dte = min_dte

        # Track price history for mean reversion
        self.price_history: List[float] = []

    def should_enter(
        self,
        current_date: datetime,
        market_data: MarketData,
        option_chain: OptionChain,
        available_capital: float
    ) -> bool:
        """
        Determine if we should enter a new position.

        Custom logic: Enter when price is overextended from SMA.
        """
        # Update price history
        self.price_history.append(market_data.close)

        # Need enough history
        if len(self.price_history) < self.lookback_period:
            return False

        # Keep only lookback period
        self.price_history = self.price_history[-self.lookback_period:]

        # Calculate SMA and std dev
        sma = np.mean(self.price_history)
        std = np.std(self.price_history)

        # Check if price is overextended
        z_score = (market_data.close - sma) / std if std > 0 else 0
        is_overextended = abs(z_score) > self.entry_std_devs

        # Check IV rank (simplified - using current IV as proxy)
        avg_iv = np.mean([opt['implied_volatility'] for opt in option_chain.calls[:10]])
        iv_rank_proxy = (avg_iv - 0.15) / (0.35 - 0.15) * 100  # Map IV to 0-100 scale
        iv_high = iv_rank_proxy > self.iv_rank_threshold

        # Don't enter if already in a position
        has_positions = len(self.positions) > 0

        # Enter when overextended and high IV
        return is_overextended and iv_high and not has_positions

    def should_exit(
        self,
        position: OptionStructure,
        current_date: datetime,
        market_data: MarketData,
        option_chain: Optional[OptionChain]
    ) -> bool:
        """
        Determine if we should exit an existing position.

        Custom logic: Exit on profit target, mean reversion, or risk limits.
        """
        # Exit if near expiration
        if position.days_to_expiration <= self.min_dte:
            return True

        # Exit on profit target
        max_profit = position.max_profit
        if max_profit > 0:
            profit_target = max_profit * self.profit_target_pct
            if position.pnl >= profit_target:
                return True

        # Exit on stop loss
        stop_loss = -abs(max_profit) * self.stop_loss_pct
        if position.pnl <= stop_loss:
            return True

        # Exit if price returned to SMA (mean reversion complete)
        if len(self.price_history) >= self.lookback_period:
            sma = np.mean(self.price_history)
            std = np.std(self.price_history)
            z_score = (market_data.close - sma) / std if std > 0 else 0

            # If we're back within 0.5 std devs, mean reversion occurred
            if abs(z_score) < 0.5:
                # Only exit if we have profit
                if position.pnl > 0:
                    return True

        return False

    def create_structure(
        self,
        current_date: datetime,
        market_data: MarketData,
        option_chain: OptionChain,
        available_capital: float
    ) -> Optional[OptionStructure]:
        """
        Create a short strangle structure.

        Custom logic: Use wider strikes when more overextended.
        """
        # Find ATM strike
        atm_strike = min(
            option_chain.calls,
            key=lambda x: abs(x['strike'] - market_data.close)
        )['strike']

        # Calculate how overextended we are
        sma = np.mean(self.price_history)
        std = np.std(self.price_history)
        z_score = abs((market_data.close - sma) / std) if std > 0 else 2.0

        # Use wider strikes when more overextended (more safety margin)
        strike_offset_pct = min(0.05 + (z_score - 2.0) * 0.02, 0.15)  # 5-15%

        # Find strikes for short strangle
        call_strike_target = market_data.close * (1 + strike_offset_pct)
        put_strike_target = market_data.close * (1 - strike_offset_pct)

        call_strike_data = min(
            [c for c in option_chain.calls if c['strike'] >= call_strike_target],
            key=lambda x: abs(x['strike'] - call_strike_target),
            default=None
        )

        put_strike_data = min(
            [p for p in option_chain.puts if p['strike'] <= put_strike_target],
            key=lambda x: abs(x['strike'] - put_strike_target),
            default=None
        )

        if call_strike_data is None or put_strike_data is None:
            return None

        # Calculate position size (risk-based)
        target_risk = available_capital * 0.05  # Risk 5% of capital
        estimated_risk = 1000  # Simplified
        quantity = max(1, int(target_risk / estimated_risk))

        # Create short strangle
        try:
            expiration = option_chain.expiration
            return ShortStrangle.create(
                underlying='SPY',
                call_strike=call_strike_data['strike'],
                put_strike=put_strike_data['strike'],
                expiration=expiration,
                call_price=call_strike_data['mid'],
                put_price=put_strike_data['mid'],
                quantity=quantity,
                entry_date=current_date,
                underlying_price=market_data.close
            )
        except Exception as e:
            print(f"Failed to create structure: {e}")
            return None


# Helper functions (same as previous examples)
def create_mock_market_data(start_date, end_date, initial_price=450.0):
    dates = pd.bdate_range(start=start_date, end=end_date)
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.02, len(dates))  # Higher volatility for mean reversion
    prices = initial_price * np.exp(np.cumsum(returns))
    market_data = []
    for date, close in zip(dates, prices):
        daily_vol = close * 0.02
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
                     'volume': 1000, 'open_interest': 5000, 'implied_volatility': iv, **call_greeks})
        puts.append({'underlying': 'SPY', 'strike': K, 'expiration': expiration, 'option_type': 'put',
                    'bid': put_price * 0.98, 'ask': put_price * 1.02, 'mid': put_price,
                    'volume': 1000, 'open_interest': 5000, 'implied_volatility': iv, **put_greeks})
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


def main():
    print("="*70)
    print("Example 4: Custom Mean Reversion Strategy")
    print("="*70)
    print()

    print("This example demonstrates creating a custom strategy:")
    print("  - Inherits from Strategy base class")
    print("  - Custom entry: Price >2σ from 20-day SMA + high IV")
    print("  - Custom exit: Mean reversion, profit target, or stop loss")
    print("  - Dynamic position sizing based on risk")
    print()

    start_date = datetime(2024, 1, 2)
    end_date = datetime(2024, 3, 29)
    initial_capital = 100000.0

    # Create our custom strategy
    strategy = CustomMeanReversionStrategy(
        name='Mean Reversion Custom',
        initial_capital=initial_capital,
        lookback_period=20,
        entry_std_devs=2.0,
        iv_rank_threshold=50.0,
        profit_target_pct=0.50,
        stop_loss_pct=2.0,
        min_dte=7
    )

    data_stream = setup_mock_data_stream(start_date, end_date)
    execution_model = ExecutionModel(commission_per_contract=0.65, slippage_pct=0.01, fill_on='mid')

    print("Running backtest with custom strategy...")
    engine = BacktestEngine(strategy, data_stream, execution_model, initial_capital)
    results = engine.run()
    print()

    print("="*70)
    print("RESULTS")
    print("="*70)
    print()

    print(f"Final Equity:    ${results['final_equity']:,.2f}")
    print(f"Total Return:    {results['total_return']*100:,.2f}%")
    print(f"Total Trades:    {len(results['trade_log'])}")
    print()

    if len(results['trade_log']) > 0:
        trade_log = results['trade_log']
        winning_trades = trade_log[trade_log['pnl'] > 0]
        win_rate = len(winning_trades) / len(trade_log) * 100
        print(f"Win Rate:        {win_rate:.1f}%")
        print(f"Avg P&L:         ${trade_log['pnl'].mean():,.2f}")
        print()

    print("="*70)
    print("Custom strategy template complete!")
    print()
    print("To create your own strategy:")
    print("  1. Inherit from Strategy base class")
    print("  2. Implement should_enter() with your entry logic")
    print("  3. Implement should_exit() with your exit logic")
    print("  4. Implement create_structure() to build your positions")
    print("="*70)


if __name__ == '__main__':
    main()
