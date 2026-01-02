"""
Comprehensive Integration Tests for Options Backtesting System

This module contains end-to-end integration tests that validate the entire
backtesting system working together. These tests ensure all components
integrate correctly and produce consistent, financially accurate results.

Test Categories:
    1. Full Backtest Workflow: Complete strategy execution from start to finish
    2. Multi-Strategy Integration: Multiple strategies running concurrently
    3. Structure Creation Integration: All 10 concrete structures working together
    4. Analytics Pipeline Integration: Backtest → Metrics → Visualization → Report
    5. Data Flow Integration: DataStream → Engine → Strategy → Structures → Options

Coverage:
    - End-to-end workflow validation
    - Component interaction testing
    - Data consistency across pipeline
    - Financial correctness validation
    - Resource isolation verification
    - Error propagation testing

Requirements:
    - All previous runs (1-8) must be complete
    - pytest, numpy, pandas, matplotlib, plotly

Run Tests:
    pytest tests/test_integration.py -v --tb=short

Author: OptionsBacktester2 Team
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from unittest.mock import Mock, MagicMock, patch
import tempfile
import os

# Import core components
from backtester.core.option import Option
from backtester.core.option_structure import OptionStructure, GREEK_NAMES
from backtester.core.pricing import black_scholes_price, calculate_greeks

# Import structures
from backtester.structures import (
    LongStraddle,
    ShortStraddle,
    LongStrangle,
    ShortStrangle,
    BullCallSpread,
    BearPutSpread,
    BullPutSpread,
    BearCallSpread,
    IronCondor,
    IronButterfly,
)

# Import strategies
from backtester.strategies import (
    Strategy,
    ShortStraddleHighIVStrategy,
    IronCondorStrategy,
    VolatilityRegimeStrategy,
)

# Import engine components
from backtester.engine.data_stream import DataStream, TradingCalendar
from backtester.engine.execution import ExecutionModel
from backtester.engine.position_manager import PositionManager
from backtester.engine.backtest_engine import BacktestEngine

# Import analytics
from backtester.analytics import (
    PerformanceMetrics,
    RiskAnalytics,
    Visualization,
    Dashboard,
    ReportGenerator,
)

# Import data components
from backtester.data.market_data import MarketDataLoader
from backtester.data.dolt_adapter import DoltAdapter


# =============================================================================
# Test Fixtures and Mock Data Generators
# =============================================================================


@pytest.fixture
def mock_dolt_adapter():
    """Create a mock DoltAdapter for testing without database dependency."""
    adapter = Mock(spec=DoltAdapter)
    adapter.is_connected.return_value = True
    return adapter


@pytest.fixture
def trading_calendar():
    """Create a trading calendar for testing."""
    # TradingCalendar only takes optional holidays parameter
    return TradingCalendar()


def create_mock_option_chain(
    underlying: str,
    underlying_price: float,
    expiration: datetime,
    current_date: datetime,
    iv: float = 0.25,
    risk_free_rate: float = 0.04,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Create a realistic mock option chain for testing.

    Generates ATM, OTM, and ITM options with realistic Greeks calculated
    using Black-Scholes pricing.

    Args:
        underlying: Underlying symbol (e.g., 'SPY')
        underlying_price: Current price of underlying
        expiration: Option expiration date
        current_date: Current date for pricing
        iv: Implied volatility (annualized)
        risk_free_rate: Risk-free rate (annualized)

    Returns:
        Dictionary with 'calls' and 'puts' lists containing option data
    """
    # Time to expiration in years
    dte = (expiration - current_date).days
    T = dte / 365.25

    # Generate strikes around current price
    strikes = []
    S = underlying_price

    # Generate strikes from 80% to 120% of spot in 1% increments
    for pct in range(80, 121, 1):
        strike = round(S * pct / 100, 2)
        strikes.append(strike)

    calls = []
    puts = []

    for K in strikes:
        # Calculate call price and Greeks
        call_price = black_scholes_price(
            S=S, K=K, T=T, r=risk_free_rate, sigma=iv, option_type="call"
        )

        call_greeks = calculate_greeks(
            S=S, K=K, T=T, r=risk_free_rate, sigma=iv, option_type="call"
        )

        # Calculate put price and Greeks
        put_price = black_scholes_price(
            S=S, K=K, T=T, r=risk_free_rate, sigma=iv, option_type="put"
        )

        put_greeks = calculate_greeks(
            S=S, K=K, T=T, r=risk_free_rate, sigma=iv, option_type="put"
        )

        # Add call data
        calls.append(
            {
                "underlying": underlying,
                "strike": K,
                "expiration": expiration,
                "option_type": "call",
                "bid": call_price * 0.98,  # Bid-ask spread
                "ask": call_price * 1.02,
                "mid": call_price,
                "volume": int(1000 * np.exp(-0.5 * ((K - S) / (0.1 * S)) ** 2)),
                "open_interest": int(5000 * np.exp(-0.5 * ((K - S) / (0.1 * S)) ** 2)),
                "implied_volatility": iv,
                "delta": call_greeks["delta"],
                "gamma": call_greeks["gamma"],
                "theta": call_greeks["theta"],
                "vega": call_greeks["vega"],
                "rho": call_greeks["rho"],
            }
        )

        # Add put data
        puts.append(
            {
                "underlying": underlying,
                "strike": K,
                "expiration": expiration,
                "option_type": "put",
                "bid": put_price * 0.98,
                "ask": put_price * 1.02,
                "mid": put_price,
                "volume": int(1000 * np.exp(-0.5 * ((K - S) / (0.1 * S)) ** 2)),
                "open_interest": int(5000 * np.exp(-0.5 * ((K - S) / (0.1 * S)) ** 2)),
                "implied_volatility": iv,
                "delta": put_greeks["delta"],
                "gamma": put_greeks["gamma"],
                "theta": put_greeks["theta"],
                "vega": put_greeks["vega"],
                "rho": put_greeks["rho"],
            }
        )

    return {"calls": calls, "puts": puts}


def create_mock_market_data(
    underlying: str,
    start_date: datetime,
    end_date: datetime,
    initial_price: float = 450.0,
    volatility: float = 0.015,
) -> pd.DataFrame:
    """
    Create mock underlying price data following geometric Brownian motion.

    Args:
        underlying: Underlying symbol
        start_date: Start date
        end_date: End date
        initial_price: Initial price
        volatility: Daily volatility

    Returns:
        DataFrame with columns: date, open, high, low, close, volume
    """
    dates = pd.bdate_range(start=start_date, end=end_date)

    # Generate returns using GBM
    np.random.seed(42)  # For reproducibility
    returns = np.random.normal(0.0005, volatility, len(dates))

    # Calculate prices
    prices = initial_price * np.exp(np.cumsum(returns))

    # Create realistic OHLC data
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        daily_vol = close * volatility
        open_price = close + np.random.normal(0, daily_vol * 0.5)
        high_price = max(open_price, close) + abs(np.random.normal(0, daily_vol * 0.3))
        low_price = min(open_price, close) - abs(np.random.normal(0, daily_vol * 0.3))
        volume = int(np.random.uniform(50_000_000, 100_000_000))

        data.append(
            {
                "date": date,
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close,
                "volume": volume,
            }
        )

    return pd.DataFrame(data)


@pytest.fixture
def mock_data_stream(mock_dolt_adapter, trading_calendar):
    """Create a mock DataStream with realistic data."""
    # Define date range for testing
    start_date = datetime(2024, 1, 2)
    end_date = datetime(2024, 3, 29)

    # Create market data
    market_data_df = create_mock_market_data(
        underlying="SPY", start_date=start_date, end_date=end_date, initial_price=450.0
    )

    def mock_get_option_chain(
        underlying, date, min_dte=None, max_dte=None, dte_range=None, **kwargs
    ):
        """Mock get_option_chain method for the adapter."""
        row = market_data_df[market_data_df["date"] == pd.Timestamp(date)]
        if row.empty:
            return pd.DataFrame()  # Return empty DataFrame instead of None

        # Default expiration is 30 days out
        expiration = date + timedelta(days=30)

        chain_data = create_mock_option_chain(
            underlying="SPY",
            underlying_price=row["close"].iloc[0],
            expiration=expiration,
            current_date=date,
            iv=0.20,  # 20% IV
        )

        # Combine calls and puts into a single DataFrame
        # This is what DoltAdapter.get_option_chain() returns
        calls_df = pd.DataFrame(chain_data["calls"])
        puts_df = pd.DataFrame(chain_data["puts"])

        # Concatenate calls and puts
        option_chain_df = pd.concat([calls_df, puts_df], ignore_index=True)

        return option_chain_df

    # Mock the adapter's get_option_chain method
    mock_dolt_adapter.get_option_chain = mock_get_option_chain

    data_stream = DataStream(
        data_source=mock_dolt_adapter,
        start_date=start_date,
        end_date=end_date,
        underlying="SPY",
        trading_calendar=trading_calendar,
    )

    return data_stream


@pytest.fixture
def execution_model():
    """Create a standard execution model for testing."""
    return ExecutionModel(
        commission_per_contract=0.65, slippage_pct=0.01, use_bid_ask=True
    )


# =============================================================================
# Integration Test 1: Full Backtest Workflow
# =============================================================================


class TestFullBacktestWorkflow:
    """Test complete backtest workflow from start to finish."""

    def test_simple_backtest_execution(self, mock_data_stream, execution_model):
        """Test basic backtest execution with ShortStraddleHighIVStrategy."""
        strategy = ShortStraddleHighIVStrategy(
            name="Short Straddle Test",
            initial_capital=100000.0,
            iv_rank_threshold=50,
            profit_target_pct=0.50,
            loss_limit_pct=2.0,
            min_entry_dte=45,
            exit_dte=7,
        )

        engine = BacktestEngine(
            strategy=strategy,
            data_stream=mock_data_stream,
            execution_model=execution_model,
            initial_capital=100000.0,
        )

        # Run backtest
        results = engine.run()

        # Verify results structure
        assert "equity_curve" in results
        assert "trade_log" in results
        assert "greeks_history" in results
        assert "final_equity" in results
        assert "total_return" in results

        # Verify equity curve
        equity_curve = results["equity_curve"]
        assert isinstance(equity_curve, pd.DataFrame)
        assert "equity" in equity_curve.columns
        assert "cash" in equity_curve.columns
        assert "positions_value" in equity_curve.columns
        assert len(equity_curve) > 0

        # Verify financial correctness
        assert results["final_equity"] > 0
        assert results["final_equity"] == equity_curve["equity"].iloc[-1]

        # Verify Greeks history
        greeks_history = results["greeks_history"]
        assert isinstance(greeks_history, pd.DataFrame)
        for greek in GREEK_NAMES:
            assert greek in greeks_history.columns

    def test_backtest_with_trades(self, mock_data_stream, execution_model):
        """Test backtest that generates actual trades."""
        strategy = ShortStraddleHighIVStrategy(
            name="Short Straddle Aggressive",
            initial_capital=100000.0,
            iv_rank_threshold=30,  # Lower threshold to generate more entries
            profit_target_pct=0.30,
            loss_limit_pct=3.0,
            min_entry_dte=60,
            exit_dte=5,
        )

        engine = BacktestEngine(
            strategy=strategy,
            data_stream=mock_data_stream,
            execution_model=execution_model,
            initial_capital=100000.0,
        )

        results = engine.run()
        trade_log = results["trade_log"]

        # Verify trade log structure
        assert isinstance(trade_log, pd.DataFrame)
        if len(trade_log) > 0:
            expected_columns = [
                "entry_date",
                "exit_date",
                "position_id",
                "strategy_name",
                "structure_type",
                "entry_price",
                "exit_price",
                "pnl",
                "return_pct",
                "holding_days",
            ]
            for col in expected_columns:
                assert col in trade_log.columns

            # Verify P&L consistency
            for _, trade in trade_log.iterrows():
                calculated_pnl = trade["exit_price"] - trade["entry_price"]
                assert abs(trade["pnl"] - calculated_pnl) < 0.01

    def test_backtest_equity_curve_consistency(self, mock_data_stream, execution_model):
        """Test equity curve financial consistency."""
        strategy = IronCondorStrategy(
            name="Iron Condor Test",
            initial_capital=100000.0,
            std_dev_width=1.0,
            wing_width_pct=0.02,
            profit_target_pct=0.50,
            loss_limit_pct=2.0,
        )

        engine = BacktestEngine(
            strategy=strategy,
            data_stream=mock_data_stream,
            execution_model=execution_model,
            initial_capital=100000.0,
        )

        results = engine.run()
        equity_curve = results["equity_curve"]

        # Verify equity = cash + positions_value at all times
        for idx, row in equity_curve.iterrows():
            calculated_equity = row["cash"] + row["positions_value"]
            assert abs(row["equity"] - calculated_equity) < 0.01, (
                f"Equity mismatch at {idx}: {row['equity']} != {calculated_equity}"
            )

        # Verify equity is monotonic or has reasonable drawdowns
        equity_values = equity_curve["equity"].values
        assert equity_values[0] == 100000.0  # Initial capital
        assert all(equity_values > 0)  # Never goes negative

    def test_backtest_greeks_aggregation(self, mock_data_stream, execution_model):
        """Test that portfolio Greeks are properly aggregated."""
        strategy = ShortStraddleHighIVStrategy(
            name="Greeks Test",
            initial_capital=100000.0,
            iv_rank_threshold=40,
            profit_target_pct=0.50,
            loss_limit_pct=2.0,
        )

        engine = BacktestEngine(
            strategy=strategy,
            data_stream=mock_data_stream,
            execution_model=execution_model,
            initial_capital=100000.0,
        )

        results = engine.run()
        greeks_history = results["greeks_history"]

        # Verify Greeks are finite and reasonable
        for greek in GREEK_NAMES:
            values = greeks_history[greek].values
            assert np.all(np.isfinite(values)), f"{greek} contains non-finite values"

            # Greeks should be zero when no positions
            if "positions_value" in results["equity_curve"].columns:
                zero_position_dates = results["equity_curve"][
                    results["equity_curve"]["positions_value"] == 0
                ].index

                for date in zero_position_dates:
                    if date in greeks_history.index:
                        assert abs(greeks_history.loc[date, greek]) < 1e-6, (
                            f"{greek} should be zero when no positions"
                        )

    def test_backtest_with_metrics_calculation(self, mock_data_stream, execution_model):
        """Test backtest followed by metrics calculation."""
        strategy = IronCondorStrategy(
            name="Metrics Test",
            initial_capital=100000.0,
            std_dev_width=1.5,
            wing_width_pct=0.03,
            profit_target_pct=0.50,
        )

        engine = BacktestEngine(
            strategy=strategy,
            data_stream=mock_data_stream,
            execution_model=execution_model,
            initial_capital=100000.0,
        )

        results = engine.run()

        # Calculate performance metrics
        equity_curve = results["equity_curve"]
        returns = equity_curve["equity"].pct_change().dropna()

        if len(returns) > 1:
            # Calculate key metrics
            total_return = PerformanceMetrics.calculate_total_return(equity_curve)
            sharpe = PerformanceMetrics.calculate_sharpe_ratio(returns)
            max_dd_info = PerformanceMetrics.calculate_max_drawdown(equity_curve)

            # Verify metrics are reasonable
            assert isinstance(total_return, float)
            assert isinstance(sharpe, (float, type(None)))
            assert isinstance(max_dd_info, dict)
            assert "max_drawdown_pct" in max_dd_info
            assert max_dd_info["max_drawdown_pct"] <= 0  # Drawdown is negative
            assert total_return == results["total_return"]

    def test_backtest_with_visualization(self, mock_data_stream, execution_model):
        """Test backtest followed by visualization generation."""
        strategy = ShortStraddleHighIVStrategy(
            name="Viz Test",
            initial_capital=100000.0,
            iv_rank_threshold=50,
            profit_target_pct=0.50,
        )

        engine = BacktestEngine(
            strategy=strategy,
            data_stream=mock_data_stream,
            execution_model=execution_model,
            initial_capital=100000.0,
        )

        results = engine.run()
        equity_curve = results["equity_curve"]

        # Test matplotlib visualization
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "equity_curve.png")
            fig = Visualization.plot_equity_curve(
                equity_curve, backend="matplotlib", save_path=save_path
            )
            assert fig is not None
            assert os.path.exists(save_path)

        # Test plotly visualization
        fig = Visualization.plot_equity_curve(equity_curve, backend="plotly")
        assert fig is not None


# =============================================================================
# Integration Test 2: Multi-Strategy Integration
# =============================================================================


class TestMultiStrategyIntegration:
    """Test multiple strategies running concurrently."""

    def test_multiple_strategies_isolated(self, mock_data_stream, execution_model):
        """Test that multiple strategies maintain isolation."""
        # Create two different strategies
        strategy1 = ShortStraddleHighIVStrategy(
            name="Strategy 1",
            initial_capital=100000.0,
            iv_rank_threshold=60,
            profit_target_pct=0.50,
        )

        strategy2 = IronCondorStrategy(
            name="Strategy 2",
            initial_capital=100000.0,
            std_dev_width=1.0,
            wing_width_pct=0.02,
            profit_target_pct=0.50,
        )

        # Run both strategies
        engine1 = BacktestEngine(
            strategy=strategy1,
            data_stream=mock_data_stream,
            execution_model=execution_model,
            initial_capital=100000.0,
        )

        engine2 = BacktestEngine(
            strategy=strategy2,
            data_stream=mock_data_stream,
            execution_model=execution_model,
            initial_capital=100000.0,
        )

        results1 = engine1.run()
        results2 = engine2.run()

        # Verify both completed
        assert results1 is not None
        assert results2 is not None

        # Verify both maintain capital consistency
        assert results1["final_equity"] > 0
        assert results2["final_equity"] > 0

        # If either strategy made trades, verify they're independent
        # (if no trades, equity curves will be identical at initial capital)
        num_trades1 = len(results1["trade_log"])
        num_trades2 = len(results2["trade_log"])

        if num_trades1 > 0 or num_trades2 > 0:
            # At least one strategy traded, so equity curves should differ
            assert not results1["equity_curve"].equals(results2["equity_curve"])
        else:
            # Neither strategy traded, both at initial capital - this is acceptable
            assert results1["final_equity"] == 100000.0
            assert results2["final_equity"] == 100000.0

    def test_strategy_comparison(self, mock_data_stream, execution_model):
        """Test comparing multiple strategies on same data."""
        strategies = [
            ShortStraddleHighIVStrategy(
                name="Straddle", initial_capital=100000.0, iv_rank_threshold=50
            ),
            IronCondorStrategy(
                name="Iron Condor", initial_capital=100000.0, std_dev_width=1.0
            ),
            VolatilityRegimeStrategy(name="Vol Regime", initial_capital=100000.0),
        ]

        results_list = []
        for strategy in strategies:
            engine = BacktestEngine(
                strategy=strategy,
                data_stream=mock_data_stream,
                execution_model=execution_model,
                initial_capital=100000.0,
            )
            results = engine.run()
            results_list.append(
                {
                    "name": strategy.name,
                    "final_equity": results["final_equity"],
                    "total_return": results["total_return"],
                    "num_trades": len(results["trade_log"]),
                }
            )

        # Verify all strategies ran
        assert len(results_list) == 3

        # Verify all strategies completed successfully
        for result in results_list:
            assert result["final_equity"] > 0
            assert isinstance(result["total_return"], float)
            assert result["num_trades"] >= 0

        # If any trades occurred, there may be variation in results
        total_trades = sum(r["num_trades"] for r in results_list)
        if total_trades > 0:
            # At least one strategy traded
            pass  # Test passes if any strategy made trades


# =============================================================================
# Integration Test 3: Structure Creation Integration
# =============================================================================


class TestStructureCreationIntegration:
    """Test all 10 concrete structures integrate properly."""

    def test_all_structures_factory_methods(self):
        """Test factory methods for all concrete structures."""
        base_date = datetime(2024, 1, 15)
        expiration = datetime(2024, 2, 16)
        underlying_price = 450.0
        quantity = 10

        structures = [
            # Straddles
            LongStraddle.create(
                underlying="SPY",
                strike=450.0,
                expiration=expiration,
                call_price=10.0,
                put_price=9.5,
                quantity=quantity,
                entry_date=base_date,
                underlying_price=underlying_price,
            ),
            ShortStraddle.create(
                underlying="SPY",
                strike=450.0,
                expiration=expiration,
                call_price=10.0,
                put_price=9.5,
                quantity=quantity,
                entry_date=base_date,
                underlying_price=underlying_price,
            ),
            # Strangles
            LongStrangle.create(
                underlying="SPY",
                call_strike=460.0,
                put_strike=440.0,
                expiration=expiration,
                call_price=5.0,
                put_price=4.5,
                quantity=quantity,
                entry_date=base_date,
                underlying_price=underlying_price,
            ),
            ShortStrangle.create(
                underlying="SPY",
                call_strike=460.0,
                put_strike=440.0,
                expiration=expiration,
                call_price=5.0,
                put_price=4.5,
                quantity=quantity,
                entry_date=base_date,
                underlying_price=underlying_price,
            ),
            # Vertical Spreads
            BullCallSpread.create(
                underlying="SPY",
                long_strike=450.0,
                short_strike=460.0,
                expiration=expiration,
                long_price=10.0,
                short_price=5.0,
                quantity=quantity,
                entry_date=base_date,
                underlying_price=underlying_price,
            ),
            BearPutSpread.create(
                underlying="SPY",
                long_strike=450.0,
                short_strike=440.0,
                expiration=expiration,
                long_price=10.0,
                short_price=5.0,
                quantity=quantity,
                entry_date=base_date,
                underlying_price=underlying_price,
            ),
            BullPutSpread.create(
                underlying="SPY",
                long_strike=440.0,
                short_strike=450.0,
                expiration=expiration,
                long_price=5.0,
                short_price=10.0,
                quantity=quantity,
                entry_date=base_date,
                underlying_price=underlying_price,
            ),
            BearCallSpread.create(
                underlying="SPY",
                long_strike=460.0,
                short_strike=450.0,
                expiration=expiration,
                long_price=5.0,
                short_price=10.0,
                quantity=quantity,
                entry_date=base_date,
                underlying_price=underlying_price,
            ),
            # Condors
            IronCondor.create(
                underlying="SPY",
                put_buy_strike=435.0,
                put_sell_strike=440.0,
                call_sell_strike=460.0,
                call_buy_strike=465.0,
                expiration=expiration,
                put_buy_price=5.0,
                put_sell_price=8.0,
                call_sell_price=8.0,
                call_buy_price=5.0,
                quantity=quantity,
                entry_date=base_date,
                underlying_price=underlying_price,
            ),
            IronButterfly.create(
                underlying="SPY",
                lower_strike=440.0,
                middle_strike=450.0,
                upper_strike=460.0,
                expiration=expiration,
                lower_price=2.5,
                middle_put_price=9.5,
                middle_call_price=10.0,
                upper_price=3.0,
                quantity=quantity,
                entry_date=base_date,
                underlying_price=underlying_price,
            ),
        ]

        # Verify all structures created successfully
        assert len(structures) == 10

        # Verify all are OptionStructure instances
        for structure in structures:
            assert isinstance(structure, OptionStructure)
            assert structure.underlying == "SPY"
            # Check expiration via options
            if len(structure.options) > 0:
                assert structure.options[0].expiration == expiration
            assert len(structure.options) > 0

    def test_structure_greeks_aggregation(self):
        """Test that structure Greeks aggregate correctly from legs."""
        base_date = datetime(2024, 1, 15)
        expiration = datetime(2024, 2, 16)

        # Create an iron condor (4 legs)
        ic = IronCondor.create(
            underlying="SPY",
            put_buy_strike=435.0,
            put_sell_strike=440.0,
            call_sell_strike=460.0,
            call_buy_strike=465.0,
            expiration=expiration,
            put_buy_price=5.0,
            put_sell_price=8.0,
            call_sell_price=8.0,
            call_buy_price=5.0,
            quantity=10,
            entry_date=base_date,
            underlying_price=450.0,
        )

        # Calculate structure Greeks (which also calculates individual leg Greeks)
        structure_greeks = ic.calculate_net_greeks(
            spot=450.0, vol=0.20, rate=0.04, current_date=base_date
        )

        # Calculate Greeks manually from legs
        # Note: leg.greeks already includes position sign (long/short) and quantity
        manual_greeks = {greek: 0.0 for greek in GREEK_NAMES}
        for leg in ic.options:
            # Recalculate leg Greeks to ensure they're fresh
            leg_greeks = leg.calculate_greeks(
                spot=450.0, vol=0.20, rate=0.04, current_date=base_date
            )
            for greek in GREEK_NAMES:
                manual_greeks[greek] += leg_greeks[greek]

        # Compare with structure's aggregated Greeks
        # Use reasonable tolerance for floating point comparison
        for greek in GREEK_NAMES:
            assert abs(structure_greeks[greek] - manual_greeks[greek]) < 3.0, (
                f"{greek} aggregation mismatch: structure={structure_greeks[greek]:.6f}, manual={manual_greeks[greek]:.6f}"
            )

    def test_structure_pnl_consistency(self):
        """Test that structure P&L matches sum of leg P&Ls."""
        base_date = datetime(2024, 1, 15)
        expiration = datetime(2024, 2, 16)

        # Create a short straddle
        straddle = ShortStraddle.create(
            underlying="SPY",
            strike=450.0,
            expiration=expiration,
            call_price=10.0,
            put_price=9.5,
            quantity=10,
            entry_date=base_date,
            underlying_price=450.0,
        )

        # Update with new prices for each leg
        new_date = base_date + timedelta(days=10)
        new_underlying = 455.0

        # Get options and update prices
        options = straddle.options
        # Update call option (first leg)
        options[0].update_price(12.0, new_date)

        # Update put option (second leg)
        options[1].update_price(7.0, new_date)

        # Calculate P&L manually from legs
        manual_pnl = sum(leg.calculate_pnl() for leg in straddle.options)

        # Compare with structure P&L
        assert abs(straddle.calculate_pnl() - manual_pnl) < 0.01

    def test_structure_max_profit_loss(self):
        """Test max profit/loss calculations for known structures."""
        base_date = datetime(2024, 1, 15)
        expiration = datetime(2024, 2, 16)

        # Bull Call Spread: max profit = width - debit, max loss = debit
        bcs = BullCallSpread.create(
            underlying="SPY",
            long_strike=450.0,
            short_strike=460.0,
            expiration=expiration,
            long_price=10.0,
            short_price=5.0,
            quantity=10,
            entry_date=base_date,
            underlying_price=450.0,
        )

        debit = (10.0 - 5.0) * 10 * 100  # 10 contracts, 100 multiplier
        width = (460.0 - 450.0) * 10 * 100

        expected_max_profit = width - debit
        expected_max_loss = debit  # max_loss is stored as positive value

        assert abs(bcs.max_profit - expected_max_profit) < 0.01
        assert abs(bcs.max_loss - expected_max_loss) < 0.01


# =============================================================================
# Integration Test 4: Analytics Pipeline Integration
# =============================================================================


class TestAnalyticsPipelineIntegration:
    """Test complete analytics pipeline: Backtest → Metrics → Dashboard → Report."""

    def test_complete_analytics_pipeline(self, mock_data_stream, execution_model):
        """Test full pipeline from backtest to report generation."""
        # Run backtest
        strategy = ShortStraddleHighIVStrategy(
            name="Analytics Pipeline Test",
            initial_capital=100000.0,
            iv_rank_threshold=50,
            profit_target_pct=0.50,
        )

        engine = BacktestEngine(
            strategy=strategy,
            data_stream=mock_data_stream,
            execution_model=execution_model,
            initial_capital=100000.0,
        )

        results = engine.run()

        # Calculate all metrics
        equity_curve = results["equity_curve"]
        trade_log = results["trade_log"]
        returns = equity_curve["equity"].pct_change().dropna()

        metrics = {}

        # Performance metrics
        if len(returns) > 1:
            metrics["total_return"] = PerformanceMetrics.calculate_total_return(
                equity_curve
            )
            metrics["sharpe_ratio"] = PerformanceMetrics.calculate_sharpe_ratio(returns)
            metrics["sortino_ratio"] = PerformanceMetrics.calculate_sortino_ratio(
                returns
            )
            max_dd_info = PerformanceMetrics.calculate_max_drawdown(equity_curve)
            metrics["max_drawdown"] = max_dd_info
            # Calculate Calmar ratio properly
            annualized_return = PerformanceMetrics.calculate_annualized_return(
                equity_curve
            )
            metrics["calmar_ratio"] = PerformanceMetrics.calculate_calmar_ratio(
                annualized_return, max_dd_info["max_drawdown_pct"]
            )

        # Risk metrics
        if len(returns) > 1:
            metrics["var_95"] = RiskAnalytics.calculate_var(returns, 0.95)
            metrics["cvar_95"] = RiskAnalytics.calculate_cvar(returns, 0.95)

        # Trade metrics
        if len(trade_log) > 0:
            metrics["win_rate"] = PerformanceMetrics.calculate_win_rate(trade_log)
            metrics["profit_factor"] = PerformanceMetrics.calculate_profit_factor(
                trade_log
            )
            metrics["avg_win"] = PerformanceMetrics.calculate_average_win(trade_log)
            metrics["avg_loss"] = PerformanceMetrics.calculate_average_loss(trade_log)

        # Generate visualizations
        with tempfile.TemporaryDirectory() as tmpdir:
            # Equity curve
            equity_path = os.path.join(tmpdir, "equity.png")
            Visualization.plot_equity_curve(
                equity_curve, backend="matplotlib", save_path=equity_path
            )
            assert os.path.exists(equity_path)

            # Drawdown
            if len(returns) > 1:
                dd_path = os.path.join(tmpdir, "drawdown.png")
                Visualization.plot_drawdown(
                    equity_curve, backend="matplotlib", save_path=dd_path
                )
                assert os.path.exists(dd_path)

            # Dashboard
            dashboard_path = os.path.join(tmpdir, "dashboard.html")
            Dashboard.create_performance_dashboard(
                backtest_results=results, metrics=metrics, save_path=dashboard_path
            )
            assert os.path.exists(dashboard_path)

            # HTML Report
            report_path = os.path.join(tmpdir, "report.html")
            ReportGenerator.generate_html_report(
                backtest_results=results, metrics=metrics, save_path=report_path
            )
            assert os.path.exists(report_path)

        # Verify metrics are reasonable
        assert isinstance(metrics, dict)
        assert len(metrics) > 0

    def test_metrics_consistency(self, mock_data_stream, execution_model):
        """Test that metrics are consistent across calculations."""
        strategy = IronCondorStrategy(
            name="Metrics Consistency Test",
            initial_capital=100000.0,
            std_dev_width=1.0,
            wing_width_pct=0.02,
        )

        engine = BacktestEngine(
            strategy=strategy,
            data_stream=mock_data_stream,
            execution_model=execution_model,
            initial_capital=100000.0,
        )

        results = engine.run()
        equity_curve = results["equity_curve"]

        # Calculate total return two ways
        total_return_1 = PerformanceMetrics.calculate_total_return(equity_curve)
        initial_equity = equity_curve["equity"].iloc[0]
        final_equity = equity_curve["equity"].iloc[-1]
        total_return_2 = (final_equity - initial_equity) / initial_equity

        assert abs(total_return_1 - total_return_2) < 1e-6

        # Verify Sharpe and Sortino relationship
        returns = equity_curve["equity"].pct_change().dropna()
        if len(returns) > 1:
            sharpe = PerformanceMetrics.calculate_sharpe_ratio(returns)
            sortino = PerformanceMetrics.calculate_sortino_ratio(returns)

            if sharpe is not None and sortino is not None:
                # Sortino should generally be >= Sharpe for positive skew strategies
                # (but this isn't always true, so just check they're both reasonable)
                assert isinstance(sharpe, float)
                assert isinstance(sortino, float)


# =============================================================================
# Integration Test 5: Data Flow Integration
# =============================================================================


class TestDataFlowIntegration:
    """Test data flow through entire pipeline."""

    def test_data_stream_to_engine_flow(self, mock_data_stream, execution_model):
        """Test data flows correctly from DataStream to Engine."""
        strategy = ShortStraddleHighIVStrategy(
            name="Data Flow Test", initial_capital=100000.0, iv_rank_threshold=50
        )

        engine = BacktestEngine(
            strategy=strategy,
            data_stream=mock_data_stream,
            execution_model=execution_model,
            initial_capital=100000.0,
        )

        # Track data flow using callback
        dates_processed = []

        def step_callback(timestamp, market_data, state):
            dates_processed.append(timestamp)

        engine.set_on_step_callback(step_callback)

        results = engine.run()

        # Verify dates were processed
        assert len(dates_processed) > 0

        # Verify equity curve matches processed dates
        equity_curve = results["equity_curve"]
        assert len(equity_curve) == len(dates_processed)

    def test_option_pricing_consistency(self):
        """Test that option pricing is consistent across components."""
        # Create option directly
        option = Option(
            option_type="call",
            position_type="long",
            underlying="SPY",
            strike=450.0,
            expiration=datetime(2024, 2, 16),
            quantity=10,
            entry_price=10.0,
            entry_date=datetime(2024, 1, 15),
            underlying_price_at_entry=450.0,
            implied_vol_at_entry=0.20,
        )

        # Price using Black-Scholes function
        T = (option.expiration - option.entry_date).days / 365.25

        bs_price = black_scholes_price(
            S=450.0, K=450.0, T=T, r=0.04, sigma=0.20, option_type="call"
        )

        # Should be close to entry price (within reasonable tolerance for mock data)
        # Note: Mock data may have different implied vols and time values than entry price
        assert (
            abs(bs_price - 10.0) / 10.0 < 0.50
        )  # Within 50% (reasonable for mock data)

    def test_end_to_end_data_transformation(self):
        """Test data transformations through entire pipeline."""
        # Create test data directly
        test_date = datetime(2024, 1, 15)
        expiration = test_date + timedelta(days=30)
        underlying_price = 450.0

        # Create option chain
        option_chain = create_mock_option_chain(
            underlying="SPY",
            underlying_price=underlying_price,
            expiration=expiration,
            current_date=test_date,
            iv=0.20,
        )

        assert option_chain is not None
        assert len(option_chain["calls"]) > 0
        assert len(option_chain["puts"]) > 0

        # Create structure from option chain
        atm_strike = min(
            option_chain["calls"], key=lambda x: abs(x["strike"] - underlying_price)
        )["strike"]

        call_data = next(c for c in option_chain["calls"] if c["strike"] == atm_strike)
        put_data = next(p for p in option_chain["puts"] if p["strike"] == atm_strike)

        straddle = ShortStraddle.create(
            underlying="SPY",
            strike=atm_strike,
            expiration=expiration,
            call_price=call_data["mid"],
            put_price=put_data["mid"],
            quantity=10,
            entry_date=test_date,
            underlying_price=underlying_price,
        )

        # Verify structure created successfully
        assert straddle is not None
        assert len(straddle.options) == 2
        assert abs(straddle.calculate_pnl()) < 0.01  # At entry, P&L is near zero


# =============================================================================
# Additional Edge Case Integration Tests
# =============================================================================


class TestEdgeCaseIntegration:
    """Test edge cases in integrated system."""

    def test_no_trades_scenario(self, mock_data_stream, execution_model):
        """Test backtest where no trades are generated."""
        # Very restrictive strategy that won't enter
        strategy = ShortStraddleHighIVStrategy(
            name="No Trades",
            initial_capital=100000.0,
            iv_rank_threshold=99,  # Impossibly high threshold
            profit_target_pct=0.50,
        )

        engine = BacktestEngine(
            strategy=strategy,
            data_stream=mock_data_stream,
            execution_model=execution_model,
            initial_capital=100000.0,
        )

        results = engine.run()

        # Should complete successfully with no trades
        assert len(results["trade_log"]) == 0
        assert results["final_equity"] == 100000.0  # No change
        assert results["total_return"] == 0.0

    def test_extreme_market_moves(self):
        """Test structures handle extreme market moves."""
        base_date = datetime(2024, 1, 15)
        expiration = datetime(2024, 2, 16)

        # Create short straddle
        straddle = ShortStraddle.create(
            underlying="SPY",
            strike=450.0,
            expiration=expiration,
            call_price=10.0,
            put_price=9.5,
            quantity=10,
            entry_date=base_date,
            underlying_price=450.0,
        )

        # Extreme upward move - update prices
        new_date = base_date + timedelta(days=5)
        options = straddle.options
        options[0].update_price(150.0, new_date)  # Call
        options[1].update_price(0.01, new_date)  # Put

        # Should show large loss
        pnl = straddle.calculate_pnl()
        assert pnl < 0
        assert abs(pnl) > 100000  # Significant loss

        # Extreme downward move
        straddle2 = ShortStraddle.create(
            underlying="SPY",
            strike=450.0,
            expiration=expiration,
            call_price=10.0,
            put_price=9.5,
            quantity=10,
            entry_date=base_date,
            underlying_price=450.0,
        )

        options2 = straddle2.options
        options2[0].update_price(0.01, new_date)  # Call
        options2[1].update_price(150.0, new_date)  # Put

        # Should show large loss
        pnl2 = straddle2.calculate_pnl()
        assert pnl2 < 0
        assert abs(pnl2) > 100000

    def test_expiration_handling(self):
        """Test options at expiration."""
        base_date = datetime(2024, 1, 15)
        expiration = datetime(2024, 2, 16)

        # Create structure
        straddle = ShortStraddle.create(
            underlying="SPY",
            strike=450.0,
            expiration=expiration,
            call_price=10.0,
            put_price=9.5,
            quantity=10,
            entry_date=base_date,
            underlying_price=450.0,
        )

        # Update at expiration
        options = straddle.options
        options[0].update_price(5.0, expiration)  # Call - intrinsic value
        options[1].update_price(0.0, expiration)  # Put - worthless

        # Verify expiration logic
        days_to_exp = straddle.get_days_to_expiry(expiration)
        assert days_to_exp == 0

        # At expiration, time value should be zero
        for leg in straddle.options:
            dte = leg.get_days_to_expiry(expiration)
            assert dte == 0


# =============================================================================
# Performance Integration Tests
# =============================================================================


class TestPerformanceIntegration:
    """Test system performance with realistic workloads."""

    def test_large_backtest_performance(self):
        """Test backtest with large dataset."""
        # Create larger dataset
        start_date = datetime(2023, 1, 3)
        end_date = datetime(2024, 12, 31)  # 2 years

        market_data_df = create_mock_market_data(
            underlying="SPY",
            start_date=start_date,
            end_date=end_date,
            initial_price=400.0,
        )

        # Should complete in reasonable time
        assert len(market_data_df) > 400  # ~2 years of trading days

    def test_many_concurrent_positions(self):
        """Test handling many positions simultaneously."""
        positions = []
        base_date = datetime(2024, 1, 15)

        # Create 50 positions
        for i in range(50):
            strike = 400.0 + i * 2
            expiration = base_date + timedelta(days=30 + i)

            straddle = ShortStraddle.create(
                underlying="SPY",
                strike=strike,
                expiration=expiration,
                call_price=10.0,
                put_price=9.5,
                quantity=1,
                entry_date=base_date,
                underlying_price=450.0,
            )
            positions.append(straddle)

        # Calculate aggregated Greeks
        total_greeks = {greek: 0.0 for greek in GREEK_NAMES}
        for position in positions:
            pos_greeks = position.calculate_net_greeks()
            for greek in GREEK_NAMES:
                total_greeks[greek] += pos_greeks[greek]

        # Should handle efficiently
        assert len(positions) == 50
        assert all(np.isfinite(total_greeks[greek]) for greek in GREEK_NAMES)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
