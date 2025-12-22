"""
Backtesting Engine Module

This module provides the core backtesting infrastructure for options strategies.
It includes components for data streaming, position management, order execution,
and the main backtesting engine.

Components:
    - DataStream: Iterator over time-series market data
    - PositionManager: Multi-strategy position tracking
    - ExecutionModel: Realistic order execution simulation
    - BacktestEngine: Main orchestrator for backtests

Architecture:
    The engine follows an event-driven architecture where:
    1. DataStream provides market data day by day
    2. Strategy defines entry/exit logic
    3. ExecutionModel simulates realistic order fills
    4. BacktestEngine coordinates all components

Usage:
    from backtester.engine import (
        BacktestEngine,
        DataStream,
        ExecutionModel,
        PositionManager,
    )

    # Create components
    data_stream = DataStream(adapter, start, end, 'SPY')
    execution = ExecutionModel(commission=0.65)

    # Create engine
    engine = BacktestEngine(
        strategy=my_strategy,
        data_stream=data_stream,
        execution_model=execution,
        initial_capital=100000.0
    )

    # Run backtest
    results = engine.run()
    print(f"Final Equity: ${results['final_equity']:,.2f}")

Example Backtest Flow:
    1. Initialize DataStream with DoltAdapter and date range
    2. Initialize ExecutionModel with commission settings
    3. Create Strategy subclass with entry/exit logic
    4. Create BacktestEngine with all components
    5. Call engine.run() to execute backtest
    6. Analyze results DataFrame

Financial Correctness:
    - Equity = Cash + Mark-to-Market Value of Positions
    - P&L = Realized P&L (closed) + Unrealized P&L (open)
    - Commission = Per-contract fee x Number of contracts
    - Fill prices respect bid/ask spread
"""

# DataStream - Market data iteration
from backtester.engine.data_stream import (
    DataStream,
    TradingCalendar,
    DataStreamError,
    DataStreamConfigError,
    DataNotAvailableError,
    DataStreamExhaustedError,
    DEFAULT_MIN_DTE,
    DEFAULT_MAX_DTE,
)

# PositionManager - Position tracking
from backtester.engine.position_manager import (
    PositionManager,
    PositionRecord,
    PositionManagerError,
    PositionNotFoundError,
    DuplicatePositionError,
    DEFAULT_MARGIN_PER_CONTRACT,
)

# ExecutionModel - Order execution simulation
from backtester.engine.execution import (
    ExecutionModel,
    ExecutionResult,
    ExecutionError,
    ExecutionConfigError,
    PriceNotAvailableError,
    FillError,
    DEFAULT_COMMISSION_PER_CONTRACT,
    DEFAULT_SLIPPAGE_PCT,
    DEFAULT_SPREAD_PCT,
)

# BacktestEngine - Main orchestrator
from backtester.engine.backtest_engine import (
    BacktestEngine,
    BacktestState,
    TradeRecord,
    BacktestError,
    BacktestConfigError,
    BacktestExecutionError,
    DEFAULT_INITIAL_CAPITAL,
    DEFAULT_RISK_FREE_RATE,
)


__all__ = [
    # Main classes
    'BacktestEngine',
    'DataStream',
    'ExecutionModel',
    'PositionManager',

    # Supporting classes
    'BacktestState',
    'TradeRecord',
    'TradingCalendar',
    'PositionRecord',
    'ExecutionResult',

    # Exceptions - DataStream
    'DataStreamError',
    'DataStreamConfigError',
    'DataNotAvailableError',
    'DataStreamExhaustedError',

    # Exceptions - PositionManager
    'PositionManagerError',
    'PositionNotFoundError',
    'DuplicatePositionError',

    # Exceptions - ExecutionModel
    'ExecutionError',
    'ExecutionConfigError',
    'PriceNotAvailableError',
    'FillError',

    # Exceptions - BacktestEngine
    'BacktestError',
    'BacktestConfigError',
    'BacktestExecutionError',

    # Constants
    'DEFAULT_MIN_DTE',
    'DEFAULT_MAX_DTE',
    'DEFAULT_MARGIN_PER_CONTRACT',
    'DEFAULT_COMMISSION_PER_CONTRACT',
    'DEFAULT_SLIPPAGE_PCT',
    'DEFAULT_SPREAD_PCT',
    'DEFAULT_INITIAL_CAPITAL',
    'DEFAULT_RISK_FREE_RATE',
]
