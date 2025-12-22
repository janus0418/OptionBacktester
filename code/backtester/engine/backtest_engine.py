"""
BacktestEngine Class for Options Backtesting

This module provides the BacktestEngine class that orchestrates the backtesting
process. It coordinates the DataStream, ExecutionModel, PositionManager, and
Strategy to simulate trading over historical data.

Key Features:
    - Clean event loop architecture
    - Strategy integration (entry/exit signals)
    - Trade logging and tracking
    - Equity curve generation
    - Greeks history tracking
    - Comprehensive results reporting

Design Philosophy:
    The BacktestEngine is the central coordinator that ties all components
    together. It implements a clean event loop that processes each trading day,
    updates positions, checks exit/entry conditions, and records state.

Event Loop Architecture:
    For each timestep:
    1. Update all position prices from market data
    2. Check exit conditions for each open position
    3. Execute exits for triggered positions
    4. Check entry conditions
    5. Execute entries for new positions
    6. Record equity and Greeks state

Financial Correctness:
    - P&L calculated consistently across all positions
    - Greeks aggregated at portfolio level
    - Commission and slippage properly accounted
    - Equity curve reflects mark-to-market values

Usage:
    from backtester.engine.backtest_engine import BacktestEngine
    from backtester.engine.data_stream import DataStream
    from backtester.engine.execution import ExecutionModel

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

References:
    - Backtesting methodologies: https://www.quantopian.com/
    - Event-driven backtesting: Jansen, M. (2018)
"""

import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backtester.strategies.strategy import Strategy
from backtester.core.option_structure import (
    OptionStructure,
    EmptyStructureError,
    GREEK_NAMES,
)
from backtester.engine.data_stream import DataStream
from backtester.engine.execution import ExecutionModel
from backtester.engine.position_manager import PositionManager

# Configure module logger
logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Default initial capital
DEFAULT_INITIAL_CAPITAL = 100000.0

# Risk-free rate for Greek calculations
DEFAULT_RISK_FREE_RATE = 0.04


# =============================================================================
# Exceptions
# =============================================================================

class BacktestError(Exception):
    """Base exception for backtest errors."""
    pass


class BacktestConfigError(BacktestError):
    """Exception raised for configuration errors."""
    pass


class BacktestExecutionError(BacktestError):
    """Exception raised during backtest execution."""
    pass


# =============================================================================
# BacktestState Class
# =============================================================================

class BacktestState:
    """
    Container for backtest state at a single point in time.

    Used for recording the equity curve and position state.
    """

    __slots__ = (
        'timestamp',
        'equity',
        'cash',
        'positions_value',
        'num_positions',
        'greeks',
        'realized_pnl',
        'unrealized_pnl',
    )

    def __init__(
        self,
        timestamp: datetime,
        equity: float,
        cash: float,
        positions_value: float,
        num_positions: int,
        greeks: Dict[str, float],
        realized_pnl: float,
        unrealized_pnl: float
    ) -> None:
        """Initialize backtest state."""
        self.timestamp = timestamp
        self.equity = equity
        self.cash = cash
        self.positions_value = positions_value
        self.num_positions = num_positions
        self.greeks = greeks.copy()
        self.realized_pnl = realized_pnl
        self.unrealized_pnl = unrealized_pnl

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            'timestamp': self.timestamp,
            'equity': self.equity,
            'cash': self.cash,
            'positions_value': self.positions_value,
            'num_positions': self.num_positions,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.unrealized_pnl,
        }
        # Add Greeks
        for name in GREEK_NAMES:
            result[name] = self.greeks.get(name, 0.0)
        return result


# =============================================================================
# TradeRecord Class
# =============================================================================

class TradeRecord:
    """
    Record of a single trade (open or close).
    """

    __slots__ = (
        'trade_id',
        'structure_id',
        'structure_type',
        'underlying',
        'action',
        'timestamp',
        'num_legs',
        'net_premium',
        'total_cost',
        'total_proceeds',
        'commission',
        'slippage',
        'realized_pnl',
        'exit_reason',
        'fills',
    )

    def __init__(
        self,
        trade_id: str,
        structure_id: str,
        structure_type: str,
        underlying: str,
        action: str,
        timestamp: datetime,
        num_legs: int,
        net_premium: float,
        total_cost: float = 0.0,
        total_proceeds: float = 0.0,
        commission: float = 0.0,
        slippage: float = 0.0,
        realized_pnl: float = 0.0,
        exit_reason: str = '',
        fills: Optional[List[Dict]] = None
    ) -> None:
        """Initialize trade record."""
        self.trade_id = trade_id
        self.structure_id = structure_id
        self.structure_type = structure_type
        self.underlying = underlying
        self.action = action
        self.timestamp = timestamp
        self.num_legs = num_legs
        self.net_premium = net_premium
        self.total_cost = total_cost
        self.total_proceeds = total_proceeds
        self.commission = commission
        self.slippage = slippage
        self.realized_pnl = realized_pnl
        self.exit_reason = exit_reason
        self.fills = fills or []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'trade_id': self.trade_id,
            'structure_id': self.structure_id,
            'structure_type': self.structure_type,
            'underlying': self.underlying,
            'action': self.action,
            'timestamp': self.timestamp,
            'num_legs': self.num_legs,
            'net_premium': self.net_premium,
            'total_cost': self.total_cost,
            'total_proceeds': self.total_proceeds,
            'commission': self.commission,
            'slippage': self.slippage,
            'realized_pnl': self.realized_pnl,
            'exit_reason': self.exit_reason,
        }


# =============================================================================
# BacktestEngine Class
# =============================================================================

class BacktestEngine:
    """
    Main backtesting engine that orchestrates the simulation.

    The engine coordinates:
    - DataStream: provides market data day by day
    - Strategy: defines entry/exit logic
    - ExecutionModel: simulates order execution
    - PositionManager: tracks all positions

    Attributes:
        strategy (Strategy): Trading strategy to backtest
        data_stream (DataStream): Market data iterator
        execution_model (ExecutionModel): Order execution simulator
        position_manager (PositionManager): Position tracker
        initial_capital (float): Starting capital
        equity_curve (List[BacktestState]): Time series of portfolio state
        trade_log (List[TradeRecord]): Record of all trades

    Example:
        >>> engine = BacktestEngine(
        ...     strategy=my_strategy,
        ...     data_stream=data_stream,
        ...     execution_model=execution,
        ...     initial_capital=100000.0
        ... )
        >>> results = engine.run()
        >>> print(f"Return: {results['total_return']:.2%}")
    """

    __slots__ = (
        '_strategy',
        '_data_stream',
        '_execution_model',
        '_position_manager',
        '_initial_capital',
        '_current_capital',
        '_equity_curve',
        '_trade_log',
        '_greeks_history',
        '_trade_counter',
        '_is_running',
        '_current_timestamp',
        '_risk_free_rate',
        '_on_step_callback',
        '_on_trade_callback',
    )

    def __init__(
        self,
        strategy: Strategy,
        data_stream: DataStream,
        execution_model: ExecutionModel,
        initial_capital: float = DEFAULT_INITIAL_CAPITAL,
        risk_free_rate: float = DEFAULT_RISK_FREE_RATE
    ) -> None:
        """
        Initialize the BacktestEngine.

        Args:
            strategy: Trading strategy to backtest. Must implement
                     should_enter() and should_exit() methods.
            data_stream: DataStream providing market data iteration.
            execution_model: ExecutionModel for order simulation.
            initial_capital: Starting capital in dollars. Default $100,000.
            risk_free_rate: Risk-free rate for Greek calculations.

        Raises:
            BacktestConfigError: If parameters are invalid

        Example:
            >>> engine = BacktestEngine(
            ...     strategy=my_strategy,
            ...     data_stream=data_stream,
            ...     execution_model=execution,
            ...     initial_capital=100000.0
            ... )
        """
        # Validate strategy
        if strategy is None:
            raise BacktestConfigError("strategy cannot be None")
        if not isinstance(strategy, Strategy):
            raise BacktestConfigError(
                f"strategy must be a Strategy instance, got {type(strategy).__name__}"
            )
        self._strategy = strategy

        # Validate data stream
        if data_stream is None:
            raise BacktestConfigError("data_stream cannot be None")
        if not isinstance(data_stream, DataStream):
            raise BacktestConfigError(
                f"data_stream must be a DataStream instance, "
                f"got {type(data_stream).__name__}"
            )
        self._data_stream = data_stream

        # Validate execution model
        if execution_model is None:
            raise BacktestConfigError("execution_model cannot be None")
        if not isinstance(execution_model, ExecutionModel):
            raise BacktestConfigError(
                f"execution_model must be an ExecutionModel instance, "
                f"got {type(execution_model).__name__}"
            )
        self._execution_model = execution_model

        # Validate initial capital
        if initial_capital is None or initial_capital <= 0:
            raise BacktestConfigError(
                f"initial_capital must be positive, got {initial_capital}"
            )
        if not np.isfinite(initial_capital):
            raise BacktestConfigError(
                f"initial_capital must be finite, got {initial_capital}"
            )
        self._initial_capital = float(initial_capital)
        self._current_capital = float(initial_capital)

        # Set risk-free rate
        self._risk_free_rate = float(risk_free_rate)

        # Initialize position manager
        self._position_manager = PositionManager()

        # Initialize tracking
        self._equity_curve: List[BacktestState] = []
        self._trade_log: List[TradeRecord] = []
        self._greeks_history: List[Dict[str, Any]] = []
        self._trade_counter = 0

        # State flags
        self._is_running = False
        self._current_timestamp: Optional[datetime] = None

        # Callbacks
        self._on_step_callback: Optional[Callable] = None
        self._on_trade_callback: Optional[Callable] = None

        logger.info(
            f"BacktestEngine initialized: strategy={strategy.name}, "
            f"capital=${initial_capital:,.2f}, "
            f"trading_days={data_stream.num_trading_days}"
        )

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def strategy(self) -> Strategy:
        """Get the strategy being backtested."""
        return self._strategy

    @property
    def data_stream(self) -> DataStream:
        """Get the data stream."""
        return self._data_stream

    @property
    def execution_model(self) -> ExecutionModel:
        """Get the execution model."""
        return self._execution_model

    @property
    def position_manager(self) -> PositionManager:
        """Get the position manager."""
        return self._position_manager

    @property
    def initial_capital(self) -> float:
        """Get initial capital."""
        return self._initial_capital

    @property
    def current_capital(self) -> float:
        """Get current cash capital."""
        return self._current_capital

    @property
    def is_running(self) -> bool:
        """Check if backtest is currently running."""
        return self._is_running

    @property
    def current_timestamp(self) -> Optional[datetime]:
        """Get current timestamp in the backtest."""
        return self._current_timestamp

    @property
    def num_trades(self) -> int:
        """Get total number of trades."""
        return len(self._trade_log)

    # =========================================================================
    # Main Run Method
    # =========================================================================

    def run(self) -> Dict[str, Any]:
        """
        Run the full backtest.

        Iterates through all trading days in the data stream, executing
        the event loop at each step.

        Returns:
            Dictionary with backtest results:
                - 'equity_curve': DataFrame with time series of equity
                - 'trade_log': DataFrame with all trades
                - 'final_equity': Final portfolio value
                - 'total_return': Percentage return
                - 'num_trades': Total number of trades
                - 'greeks_history': DataFrame with Greeks over time
                - 'strategy_stats': Strategy statistics
                - 'execution_stats': Execution statistics

        Raises:
            BacktestExecutionError: If backtest fails

        Example:
            >>> results = engine.run()
            >>> print(f"Final Equity: ${results['final_equity']:,.2f}")
            >>> print(f"Total Return: {results['total_return']:.2%}")
        """
        self._is_running = True
        self._reset()

        logger.info(
            f"Starting backtest: {self._strategy.name}, "
            f"{self._data_stream.num_trading_days} trading days"
        )

        try:
            # Main event loop
            for timestamp, market_data in self._data_stream:
                self._current_timestamp = timestamp

                try:
                    self.step(timestamp, market_data)
                except Exception as e:
                    logger.error(f"Error in step at {timestamp}: {e}")
                    if not self._handle_step_error(timestamp, e):
                        raise BacktestExecutionError(
                            f"Backtest failed at {timestamp}: {e}"
                        ) from e

            # Final state recording
            self._record_final_state()

            # Generate results
            results = self._generate_results()

            logger.info(
                f"Backtest completed: {self._strategy.name}, "
                f"Final Equity=${results['final_equity']:,.2f}, "
                f"Return={results['total_return']:.2%}, "
                f"Trades={results['num_trades']}"
            )

            return results

        finally:
            self._is_running = False

    def step(self, timestamp: datetime, market_data: Dict[str, Any]) -> None:
        """
        Execute a single timestep in the backtest.

        Event loop order:
        1. Update all position prices from market data
        2. Check exit conditions for each position
        3. Execute exits
        4. Check entry conditions
        5. Execute entries
        6. Record state

        Args:
            timestamp: Current timestamp
            market_data: Market data dictionary containing spot, option_chain, iv

        Example:
            >>> engine.step(datetime(2023, 6, 15), market_data)
        """
        # 1. Update all position prices
        self._update_positions(market_data, timestamp)

        # 2. Check and execute exits
        self._process_exits(market_data, timestamp)

        # 3. Check and execute entries
        self._process_entries(market_data, timestamp)

        # 4. Record state
        self._record_state(timestamp, market_data)

        # 5. Call step callback if set
        if self._on_step_callback:
            self._on_step_callback(timestamp, market_data, self._get_current_state())

    # =========================================================================
    # Event Loop Components
    # =========================================================================

    def _update_positions(
        self,
        market_data: Dict[str, Any],
        timestamp: datetime
    ) -> None:
        """
        Update all position prices from market data.

        Args:
            market_data: Current market data
            timestamp: Current timestamp
        """
        # Update strategy positions
        self._strategy.update_positions(market_data, timestamp)

        # Update position manager
        self._position_manager.update_all_positions(market_data, timestamp)

    def _process_exits(
        self,
        market_data: Dict[str, Any],
        timestamp: datetime
    ) -> None:
        """
        Check exit conditions and execute exits.

        Args:
            market_data: Current market data
            timestamp: Current timestamp
        """
        # Get list of structures to check (copy to allow modification during iteration)
        structures_to_check = list(self._strategy.structures)

        for structure in structures_to_check:
            try:
                # Check if strategy wants to exit
                should_exit = self._strategy.should_exit(structure, market_data)

                if should_exit:
                    self._execute_exit(structure, market_data, timestamp)

            except Exception as e:
                logger.warning(
                    f"Error checking exit for {structure.structure_id}: {e}"
                )

    def _process_entries(
        self,
        market_data: Dict[str, Any],
        timestamp: datetime
    ) -> None:
        """
        Check entry conditions and execute entries.

        Args:
            market_data: Current market data
            timestamp: Current timestamp
        """
        try:
            # Check if strategy wants to enter
            should_enter = self._strategy.should_enter(market_data)

            if should_enter:
                # Get structure to enter (strategy must implement create_structure)
                structure = self._create_structure_from_strategy(market_data, timestamp)

                if structure is not None:
                    self._execute_entry(structure, market_data, timestamp)

        except Exception as e:
            logger.warning(f"Error processing entry: {e}")

    def _create_structure_from_strategy(
        self,
        market_data: Dict[str, Any],
        timestamp: datetime
    ) -> Optional[OptionStructure]:
        """
        Create structure from strategy.

        Strategies should implement a create_structure method that returns
        the OptionStructure to enter. If not implemented, this returns None.

        Args:
            market_data: Current market data
            timestamp: Current timestamp

        Returns:
            OptionStructure to enter or None
        """
        # Check if strategy has create_structure method
        if hasattr(self._strategy, 'create_structure'):
            try:
                return self._strategy.create_structure(market_data)
            except Exception as e:
                logger.warning(f"Error creating structure: {e}")
                return None
        else:
            logger.debug(
                "Strategy does not implement create_structure method. "
                "Override this method in your strategy to enable entry execution."
            )
            return None

    def _execute_entry(
        self,
        structure: OptionStructure,
        market_data: Dict[str, Any],
        timestamp: datetime
    ) -> None:
        """
        Execute entry for a structure.

        Args:
            structure: Structure to enter
            market_data: Current market data
            timestamp: Entry timestamp
        """
        try:
            # Execute through execution model (calculates costs)
            entry_result = self._execution_model.execute_entry(
                structure=structure,
                market_data=market_data,
                timestamp=timestamp
            )

            # Open position in strategy (validates limits, margin, etc.)
            # This must happen BEFORE capital update so validation errors
            # don't leave capital in an inconsistent state
            self._strategy.open_position(structure, validate_limits=True)

            # Update capital AFTER successful validation
            # (subtract cost for debit, add for credit)
            self._current_capital -= entry_result['total_cost']

            # Add to position manager
            self._position_manager.add_position(
                structure=structure,
                strategy_name=self._strategy.name,
                open_timestamp=timestamp
            )

            # Record trade
            self._record_trade(
                structure=structure,
                action='open',
                timestamp=timestamp,
                entry_result=entry_result
            )

            logger.debug(
                f"Opened position: {structure.structure_type.upper()} on "
                f"{structure.underlying}, cost=${entry_result['total_cost']:,.2f}"
            )

        except Exception as e:
            logger.error(f"Failed to execute entry: {e}")

    def _execute_exit(
        self,
        structure: OptionStructure,
        market_data: Dict[str, Any],
        timestamp: datetime,
        exit_reason: str = 'strategy_signal'
    ) -> None:
        """
        Execute exit for a structure.

        Args:
            structure: Structure to exit
            market_data: Current market data
            timestamp: Exit timestamp
            exit_reason: Reason for exit
        """
        try:
            # Execute through execution model
            exit_result = self._execution_model.execute_exit(
                structure=structure,
                market_data=market_data,
                timestamp=timestamp
            )

            # Update capital (add proceeds)
            self._current_capital += exit_result['total_proceeds']

            # Calculate realized P&L
            realized_pnl = exit_result['total_proceeds']

            # Build exit data
            exit_data = {
                'exit_reason': exit_reason,
                'exit_timestamp': timestamp,
                'exit_prices': exit_result.get('exit_prices', {}),
            }

            # Close position in strategy
            close_result = self._strategy.close_position(structure, exit_data)
            realized_pnl = close_result.get('realized_pnl', realized_pnl)

            # Remove from position manager
            self._position_manager.remove_position(
                structure_id=structure.structure_id,
                close_timestamp=timestamp,
                realized_pnl=realized_pnl
            )

            # Record trade
            self._record_trade(
                structure=structure,
                action='close',
                timestamp=timestamp,
                exit_result=exit_result,
                realized_pnl=realized_pnl,
                exit_reason=exit_reason
            )

            logger.debug(
                f"Closed position: {structure.structure_type.upper()} on "
                f"{structure.underlying}, P&L=${realized_pnl:,.2f}"
            )

        except Exception as e:
            logger.error(f"Failed to execute exit: {e}")

    # =========================================================================
    # Recording Methods
    # =========================================================================

    def _record_state(
        self,
        timestamp: datetime,
        market_data: Dict[str, Any]
    ) -> None:
        """
        Record portfolio state at current timestamp.

        Args:
            timestamp: Current timestamp
            market_data: Current market data
        """
        # Calculate portfolio value
        positions_value = self._position_manager.calculate_portfolio_value(market_data)
        equity = self._current_capital + positions_value

        # Calculate Greeks
        greeks = self._position_manager.get_portfolio_greeks(
            market_data=market_data,
            rate=self._risk_free_rate
        )

        # Calculate P&L
        realized_pnl = self._position_manager.total_realized_pnl
        unrealized_pnl = self._position_manager.calculate_unrealized_pnl()

        # Create state record
        state = BacktestState(
            timestamp=timestamp,
            equity=equity,
            cash=self._current_capital,
            positions_value=positions_value,
            num_positions=self._position_manager.num_positions,
            greeks=greeks,
            realized_pnl=realized_pnl,
            unrealized_pnl=unrealized_pnl
        )

        self._equity_curve.append(state)

        # Record Greeks history
        greeks_record = {'timestamp': timestamp}
        greeks_record.update(greeks)
        self._greeks_history.append(greeks_record)

    def _record_trade(
        self,
        structure: OptionStructure,
        action: str,
        timestamp: datetime,
        entry_result: Optional[Dict] = None,
        exit_result: Optional[Dict] = None,
        realized_pnl: float = 0.0,
        exit_reason: str = ''
    ) -> None:
        """
        Record trade details.

        Args:
            structure: Structure traded
            action: 'open' or 'close'
            timestamp: Trade timestamp
            entry_result: Entry execution result
            exit_result: Exit execution result
            realized_pnl: Realized P&L for closes
            exit_reason: Reason for exit
        """
        self._trade_counter += 1
        trade_id = f"T{self._trade_counter:06d}"

        result = entry_result or exit_result or {}

        record = TradeRecord(
            trade_id=trade_id,
            structure_id=structure.structure_id,
            structure_type=structure.structure_type,
            underlying=structure.underlying or '',
            action=action,
            timestamp=timestamp,
            num_legs=structure.num_legs,
            net_premium=structure.net_premium,
            total_cost=result.get('total_cost', 0.0),
            total_proceeds=result.get('total_proceeds', 0.0),
            commission=result.get('commissions', 0.0),
            slippage=result.get('slippage', 0.0),
            realized_pnl=realized_pnl,
            exit_reason=exit_reason,
            fills=result.get('fills', [])
        )

        self._trade_log.append(record)

        # Call trade callback if set
        if self._on_trade_callback:
            self._on_trade_callback(record)

    def _record_final_state(self) -> None:
        """Record final state after backtest completion."""
        if self._equity_curve:
            # Final state is already recorded in last step
            pass

    # =========================================================================
    # Results Generation
    # =========================================================================

    def _generate_results(self) -> Dict[str, Any]:
        """
        Generate backtest results.

        Returns:
            Dictionary with comprehensive results
        """
        # Generate equity curve DataFrame
        equity_curve_df = self.generate_equity_curve()

        # Generate trade log DataFrame
        trade_log_df = self.get_trade_log()

        # Generate Greeks history DataFrame
        greeks_history_df = self._generate_greeks_history()

        # Calculate final metrics
        final_equity = self._get_final_equity()
        total_return = (final_equity - self._initial_capital) / self._initial_capital

        # Count trades
        num_entries = len([t for t in self._trade_log if t.action == 'open'])
        num_exits = len([t for t in self._trade_log if t.action == 'close'])

        # Get strategy and execution statistics
        strategy_stats = self._strategy.get_statistics()
        execution_stats = self._execution_model.get_execution_summary()

        results = {
            # Core results
            'equity_curve': equity_curve_df,
            'trade_log': trade_log_df,
            'greeks_history': greeks_history_df,

            # Summary metrics
            'initial_capital': self._initial_capital,
            'final_equity': final_equity,
            'total_return': total_return,
            'total_pnl': final_equity - self._initial_capital,

            # Trade statistics
            'num_trades': len(self._trade_log),
            'num_entries': num_entries,
            'num_exits': num_exits,

            # Component statistics
            'strategy_stats': strategy_stats,
            'execution_stats': execution_stats,

            # Metadata
            'underlying': self._data_stream.underlying,
            'start_date': self._data_stream.start_date,
            'end_date': self._data_stream.end_date,
            'trading_days': self._data_stream.num_trading_days,
        }

        return results

    def generate_equity_curve(self) -> pd.DataFrame:
        """
        Generate equity curve as DataFrame.

        Returns:
            DataFrame with columns for timestamp, equity, greeks, etc.

        Example:
            >>> equity_df = engine.generate_equity_curve()
            >>> plt.plot(equity_df['timestamp'], equity_df['equity'])
        """
        if not self._equity_curve:
            return pd.DataFrame()

        records = [state.to_dict() for state in self._equity_curve]
        df = pd.DataFrame(records)

        # Set timestamp as index
        if 'timestamp' in df.columns:
            df.set_index('timestamp', inplace=True)

        return df

    def get_trade_log(self) -> pd.DataFrame:
        """
        Get trade log as DataFrame.

        Returns:
            DataFrame with all trade records

        Example:
            >>> trades_df = engine.get_trade_log()
            >>> print(trades_df[['timestamp', 'action', 'realized_pnl']])
        """
        if not self._trade_log:
            return pd.DataFrame()

        records = [trade.to_dict() for trade in self._trade_log]
        return pd.DataFrame(records)

    def _generate_greeks_history(self) -> pd.DataFrame:
        """
        Generate Greeks history as DataFrame.

        Returns:
            DataFrame with Greeks over time
        """
        if not self._greeks_history:
            return pd.DataFrame()

        df = pd.DataFrame(self._greeks_history)

        if 'timestamp' in df.columns:
            df.set_index('timestamp', inplace=True)

        return df

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _reset(self) -> None:
        """Reset engine state for a new run."""
        self._current_capital = self._initial_capital
        self._equity_curve.clear()
        self._trade_log.clear()
        self._greeks_history.clear()
        self._trade_counter = 0
        self._current_timestamp = None

        # Reset data stream
        self._data_stream.reset()

        # Clear position manager
        self._position_manager.clear()

        # Clear execution log
        self._execution_model.clear_log()

        logger.debug("BacktestEngine reset")

    def _get_final_equity(self) -> float:
        """Get final equity value."""
        if self._equity_curve:
            return self._equity_curve[-1].equity
        return self._initial_capital

    def _get_current_state(self) -> Dict[str, Any]:
        """Get current state as dictionary."""
        if self._equity_curve:
            return self._equity_curve[-1].to_dict()
        return {
            'equity': self._initial_capital,
            'cash': self._initial_capital,
            'positions_value': 0.0,
            'num_positions': 0,
        }

    def _handle_step_error(self, timestamp: datetime, error: Exception) -> bool:
        """
        Handle error during step execution.

        Args:
            timestamp: Timestamp where error occurred
            error: The exception that was raised

        Returns:
            True if error was handled and backtest should continue,
            False if backtest should abort
        """
        logger.warning(f"Error at {timestamp}: {error}")
        # By default, log and continue
        return True

    # =========================================================================
    # Callback Methods
    # =========================================================================

    def set_on_step_callback(
        self,
        callback: Callable[[datetime, Dict, Dict], None]
    ) -> None:
        """
        Set callback to be called after each step.

        Args:
            callback: Function taking (timestamp, market_data, state)
        """
        self._on_step_callback = callback

    def set_on_trade_callback(
        self,
        callback: Callable[[TradeRecord], None]
    ) -> None:
        """
        Set callback to be called after each trade.

        Args:
            callback: Function taking (trade_record)
        """
        self._on_trade_callback = callback

    # =========================================================================
    # Analytics Methods
    # =========================================================================

    def calculate_metrics(
        self,
        risk_free_rate: float = 0.02
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive performance and risk metrics.

        This method integrates with the analytics module to provide
        industry-standard metrics for evaluating backtest performance.

        Metrics calculated:
            Performance:
                - Total return (%)
                - Annualized return (CAGR)
                - Sharpe ratio
                - Sortino ratio
                - Calmar ratio
                - Max drawdown
                - Win rate
                - Profit factor
                - Expectancy
                - Consecutive wins/losses

            Risk:
                - Value at Risk (95%)
                - Conditional VaR (Expected Shortfall)
                - Tail risk (skewness, kurtosis)
                - Greeks analysis
                - Downside risk metrics

        Args:
            risk_free_rate: Annual risk-free rate for ratio calculations.
                           Default 0.02 (2%)

        Returns:
            Dictionary with metrics organized by category:
                {
                    'performance': {...},  # From PerformanceMetrics
                    'risk': {...},         # From RiskAnalytics
                    'summary': {...}       # Key metrics summary
                }

        Raises:
            BacktestError: If backtest has not been run yet

        Example:
            >>> results = engine.run()
            >>> metrics = engine.calculate_metrics(risk_free_rate=0.02)
            >>> print(f"Sharpe Ratio: {metrics['performance']['sharpe_ratio']:.2f}")
            >>> print(f"Max Drawdown: {metrics['performance']['max_drawdown']:.2%}")

        Note:
            This method should be called after run() completes.
            It uses the equity curve, trade log, and Greeks history
            generated during the backtest.
        """
        # Import analytics classes
        from backtester.analytics import PerformanceMetrics, RiskAnalytics

        # Get data from backtest
        equity_curve = self.generate_equity_curve()
        trade_log = self.get_trade_log()
        greeks_history = self._generate_greeks_history()

        # Validate we have data
        if equity_curve.empty:
            raise BacktestError(
                "No equity curve data available. "
                "Please run the backtest first."
            )

        # Calculate returns series
        equity_series = equity_curve['equity']
        returns = equity_series.pct_change().dropna()

        # Initialize results
        performance = {}
        risk = {}

        # =================================================================
        # Performance Metrics
        # =================================================================

        # Returns-based metrics
        try:
            performance['total_return_pct'] = PerformanceMetrics.calculate_total_return(
                equity_curve
            )
        except Exception as e:
            logger.warning(f"Could not calculate total return: {e}")
            performance['total_return_pct'] = np.nan

        try:
            performance['annualized_return'] = PerformanceMetrics.calculate_annualized_return(
                equity_curve
            )
        except Exception as e:
            logger.warning(f"Could not calculate annualized return: {e}")
            performance['annualized_return'] = np.nan

        try:
            performance['sharpe_ratio'] = PerformanceMetrics.calculate_sharpe_ratio(
                returns, risk_free_rate
            )
        except Exception as e:
            logger.warning(f"Could not calculate Sharpe ratio: {e}")
            performance['sharpe_ratio'] = np.nan

        try:
            performance['sortino_ratio'] = PerformanceMetrics.calculate_sortino_ratio(
                returns, risk_free_rate
            )
        except Exception as e:
            logger.warning(f"Could not calculate Sortino ratio: {e}")
            performance['sortino_ratio'] = np.nan

        # Drawdown metrics
        try:
            dd_info = PerformanceMetrics.calculate_max_drawdown(equity_curve)
            performance['max_drawdown'] = dd_info['max_drawdown_pct']
            performance['max_drawdown_value'] = dd_info['max_drawdown_value']
            performance['max_drawdown_duration'] = dd_info['duration_days']
            performance['peak_date'] = dd_info['peak_date']
            performance['trough_date'] = dd_info['trough_date']
            performance['recovery_date'] = dd_info['recovery_date']
            performance['drawdown_series'] = dd_info['drawdown_series']
        except Exception as e:
            logger.warning(f"Could not calculate max drawdown: {e}")
            performance['max_drawdown'] = np.nan
            performance['max_drawdown_value'] = np.nan
            performance['max_drawdown_duration'] = np.nan

        # Calmar ratio (needs annualized return and max drawdown)
        try:
            ann_return = performance.get('annualized_return', np.nan)
            max_dd = performance.get('max_drawdown', np.nan)
            if not np.isnan(ann_return) and not np.isnan(max_dd):
                performance['calmar_ratio'] = PerformanceMetrics.calculate_calmar_ratio(
                    ann_return, max_dd
                )
            else:
                performance['calmar_ratio'] = np.nan
        except Exception as e:
            logger.warning(f"Could not calculate Calmar ratio: {e}")
            performance['calmar_ratio'] = np.nan

        # Ulcer Index
        try:
            performance['ulcer_index'] = PerformanceMetrics.calculate_ulcer_index(
                equity_curve
            )
        except Exception as e:
            logger.warning(f"Could not calculate Ulcer Index: {e}")
            performance['ulcer_index'] = np.nan

        # Trade-based metrics (only if we have trades)
        if not trade_log.empty:
            # Filter to closed trades
            closed_trades = trade_log[trade_log['action'] == 'close'].copy()

            if not closed_trades.empty:
                try:
                    performance['win_rate'] = PerformanceMetrics.calculate_win_rate(
                        closed_trades
                    )
                except Exception as e:
                    logger.warning(f"Could not calculate win rate: {e}")
                    performance['win_rate'] = np.nan

                try:
                    performance['profit_factor'] = PerformanceMetrics.calculate_profit_factor(
                        closed_trades
                    )
                except Exception as e:
                    logger.warning(f"Could not calculate profit factor: {e}")
                    performance['profit_factor'] = np.nan

                try:
                    performance['average_win'] = PerformanceMetrics.calculate_average_win(
                        closed_trades
                    )
                except Exception as e:
                    performance['average_win'] = np.nan

                try:
                    performance['average_loss'] = PerformanceMetrics.calculate_average_loss(
                        closed_trades
                    )
                except Exception as e:
                    performance['average_loss'] = np.nan

                try:
                    performance['expectancy'] = PerformanceMetrics.calculate_expectancy(
                        closed_trades
                    )
                except Exception as e:
                    logger.warning(f"Could not calculate expectancy: {e}")
                    performance['expectancy'] = np.nan

                try:
                    performance['average_trade'] = PerformanceMetrics.calculate_average_trade(
                        closed_trades
                    )
                except Exception as e:
                    performance['average_trade'] = np.nan

                try:
                    performance['payoff_ratio'] = PerformanceMetrics.calculate_payoff_ratio(
                        closed_trades
                    )
                except Exception as e:
                    performance['payoff_ratio'] = np.nan

                performance['max_consecutive_wins'] = PerformanceMetrics.calculate_consecutive_wins(
                    closed_trades
                )
                performance['max_consecutive_losses'] = PerformanceMetrics.calculate_consecutive_losses(
                    closed_trades
                )
                performance['total_trades'] = len(closed_trades)

        # Distribution statistics
        try:
            dist_stats = PerformanceMetrics.calculate_returns_distribution(returns)
            performance['returns_distribution'] = dist_stats
            performance['volatility'] = dist_stats['std'] * np.sqrt(252)  # Annualized
        except Exception as e:
            logger.warning(f"Could not calculate return distribution: {e}")
            performance['returns_distribution'] = {}
            performance['volatility'] = np.nan

        # =================================================================
        # Risk Metrics
        # =================================================================

        # Value at Risk
        try:
            risk['var_95_historical'] = RiskAnalytics.calculate_var(
                returns, 0.95, 'historical'
            )
        except Exception as e:
            logger.warning(f"Could not calculate VaR (historical): {e}")
            risk['var_95_historical'] = np.nan

        try:
            risk['var_95_parametric'] = RiskAnalytics.calculate_var(
                returns, 0.95, 'parametric'
            )
        except Exception as e:
            logger.warning(f"Could not calculate VaR (parametric): {e}")
            risk['var_95_parametric'] = np.nan

        try:
            risk['var_99_historical'] = RiskAnalytics.calculate_var(
                returns, 0.99, 'historical'
            )
        except Exception as e:
            risk['var_99_historical'] = np.nan

        # Conditional VaR (Expected Shortfall)
        try:
            risk['cvar_95'] = RiskAnalytics.calculate_cvar(returns, 0.95)
        except Exception as e:
            logger.warning(f"Could not calculate CVaR: {e}")
            risk['cvar_95'] = np.nan

        try:
            risk['cvar_99'] = RiskAnalytics.calculate_cvar(returns, 0.99)
        except Exception as e:
            risk['cvar_99'] = np.nan

        # Tail risk metrics
        try:
            tail_risk = RiskAnalytics.calculate_tail_risk(returns)
            risk['tail_risk'] = tail_risk
            risk['skewness'] = tail_risk['skewness']
            risk['kurtosis'] = tail_risk['kurtosis']
        except Exception as e:
            logger.warning(f"Could not calculate tail risk: {e}")
            risk['tail_risk'] = {}
            risk['skewness'] = np.nan
            risk['kurtosis'] = np.nan

        # Downside risk
        try:
            downside = RiskAnalytics.calculate_downside_risk(returns)
            risk['downside_risk'] = downside
            risk['downside_deviation'] = downside['downside_deviation']
        except Exception as e:
            logger.warning(f"Could not calculate downside risk: {e}")
            risk['downside_risk'] = {}
            risk['downside_deviation'] = np.nan

        # Greeks analysis (if available)
        if not greeks_history.empty:
            try:
                greeks_analysis = RiskAnalytics.analyze_greeks_over_time(greeks_history)
                risk['greeks_analysis'] = greeks_analysis
            except Exception as e:
                logger.warning(f"Could not analyze Greeks: {e}")
                risk['greeks_analysis'] = {}
        else:
            risk['greeks_analysis'] = {}

        # MAE analysis (if trades available)
        if not trade_log.empty:
            closed_trades = trade_log[trade_log['action'] == 'close'].copy()
            if not closed_trades.empty:
                try:
                    mae_df = RiskAnalytics.calculate_mae(closed_trades)
                    mae_stats = RiskAnalytics.analyze_mae_statistics(mae_df)
                    risk['mae_analysis'] = mae_stats
                except Exception as e:
                    logger.warning(f"Could not calculate MAE: {e}")
                    risk['mae_analysis'] = {}

        # =================================================================
        # Summary
        # =================================================================

        summary = {
            # Key performance metrics
            'total_return_pct': performance.get('total_return_pct', np.nan),
            'annualized_return': performance.get('annualized_return', np.nan),
            'sharpe_ratio': performance.get('sharpe_ratio', np.nan),
            'sortino_ratio': performance.get('sortino_ratio', np.nan),
            'max_drawdown': performance.get('max_drawdown', np.nan),
            'calmar_ratio': performance.get('calmar_ratio', np.nan),

            # Key trade metrics
            'win_rate': performance.get('win_rate', np.nan),
            'profit_factor': performance.get('profit_factor', np.nan),
            'expectancy': performance.get('expectancy', np.nan),
            'total_trades': performance.get('total_trades', 0),

            # Key risk metrics
            'var_95': risk.get('var_95_historical', np.nan),
            'cvar_95': risk.get('cvar_95', np.nan),
            'volatility': performance.get('volatility', np.nan),

            # Equity info
            'initial_equity': float(equity_series.iloc[0]),
            'final_equity': float(equity_series.iloc[-1]),
            'trading_days': len(equity_curve),
        }

        return {
            'performance': performance,
            'risk': risk,
            'summary': summary,
        }

    def get_greeks_history(self) -> pd.DataFrame:
        """
        Get Greeks history as DataFrame.

        Public method to access Greeks history for external analysis.

        Returns:
            DataFrame with Greeks over time

        Example:
            >>> greeks = engine.get_greeks_history()
            >>> greeks[['delta', 'gamma']].plot()
        """
        return self._generate_greeks_history()

    # =========================================================================
    # Visualization Methods
    # =========================================================================

    def plot_results(
        self,
        save_dir: Optional[str] = None,
        backend: str = 'plotly',
        show: bool = False
    ) -> Dict[str, str]:
        """
        Generate all standard plots for backtest results.

        Creates equity curve, drawdown, P&L distribution, and Greeks
        plots using the specified backend.

        Args:
            save_dir: Directory to save plots. If None, plots are not saved.
            backend: 'matplotlib' or 'plotly'. Default 'plotly'.
            show: If True, display plots interactively. Default False.

        Returns:
            Dictionary mapping plot names to file paths (if saved).

        Example:
            >>> results = engine.run()
            >>> plots = engine.plot_results(save_dir='./plots', backend='plotly')
            >>> print(plots['equity'])  # Path to equity curve plot
        """
        from backtester.analytics import Visualization

        plots = {}
        equity_curve = self.generate_equity_curve()
        trade_log = self.get_trade_log()
        greeks_history = self.get_greeks_history()

        # Determine save paths
        equity_path = None
        drawdown_path = None
        pnl_dist_path = None
        greeks_path = None
        monthly_path = None

        if save_dir:
            import os
            os.makedirs(save_dir, exist_ok=True)
            ext = 'html' if backend == 'plotly' else 'png'
            equity_path = os.path.join(save_dir, f'equity_curve.{ext}')
            drawdown_path = os.path.join(save_dir, f'drawdown.{ext}')
            pnl_dist_path = os.path.join(save_dir, f'pnl_distribution.{ext}')
            greeks_path = os.path.join(save_dir, f'greeks.{ext}')
            monthly_path = os.path.join(save_dir, f'monthly_returns.{ext}')

        # Generate equity curve plot
        if not equity_curve.empty and 'equity' in equity_curve.columns:
            try:
                Visualization.plot_equity_curve(
                    equity_curve,
                    save_path=equity_path,
                    show=show,
                    backend=backend,
                    initial_capital=self._initial_capital
                )
                if equity_path:
                    plots['equity'] = equity_path
            except Exception as e:
                logger.warning(f"Failed to create equity curve plot: {e}")

            # Generate drawdown plot
            try:
                Visualization.plot_drawdown(
                    equity_curve,
                    save_path=drawdown_path,
                    show=show,
                    backend=backend
                )
                if drawdown_path:
                    plots['drawdown'] = drawdown_path
            except Exception as e:
                logger.warning(f"Failed to create drawdown plot: {e}")

            # Generate monthly returns heatmap
            try:
                Visualization.plot_monthly_returns(
                    equity_curve,
                    save_path=monthly_path,
                    show=show,
                    backend=backend
                )
                if monthly_path:
                    plots['monthly_returns'] = monthly_path
            except Exception as e:
                logger.warning(f"Failed to create monthly returns plot: {e}")

        # Generate P&L distribution plot
        if not trade_log.empty and 'realized_pnl' in trade_log.columns:
            closed_trades = trade_log[trade_log['action'] == 'close']
            if not closed_trades.empty:
                try:
                    Visualization.plot_pnl_distribution(
                        closed_trades,
                        save_path=pnl_dist_path,
                        show=show,
                        backend=backend
                    )
                    if pnl_dist_path:
                        plots['pnl_distribution'] = pnl_dist_path
                except Exception as e:
                    logger.warning(f"Failed to create P&L distribution plot: {e}")

        # Generate Greeks plot
        if not greeks_history.empty:
            try:
                Visualization.plot_greeks_over_time(
                    greeks_history,
                    save_path=greeks_path,
                    show=show,
                    backend=backend
                )
                if greeks_path:
                    plots['greeks'] = greeks_path
            except Exception as e:
                logger.warning(f"Failed to create Greeks plot: {e}")

        logger.info(f"Generated {len(plots)} plots")
        return plots

    def create_dashboard(
        self,
        save_path: str = 'dashboard.html'
    ) -> str:
        """
        Create interactive dashboard for backtest results.

        Generates a comprehensive HTML dashboard with equity curve,
        drawdown, P&L distribution, Greeks, and trade statistics.

        Args:
            save_path: Path to save HTML dashboard. Default 'dashboard.html'.

        Returns:
            Absolute path to saved dashboard file.

        Example:
            >>> results = engine.run()
            >>> path = engine.create_dashboard('my_dashboard.html')
            >>> print(f"Dashboard saved to: {path}")
        """
        from backtester.analytics import Dashboard

        results = {
            'equity_curve': self.generate_equity_curve(),
            'trade_log': self.get_trade_log(),
            'greeks_history': self.get_greeks_history(),
            'strategy_stats': self._strategy.get_statistics(),
            'start_date': self._data_stream.start_date,
            'end_date': self._data_stream.end_date,
            'underlying': self._data_stream.underlying,
        }

        metrics = self.calculate_metrics()

        return Dashboard.create_performance_dashboard(
            results, metrics, save_path
        )

    def generate_report(
        self,
        save_path: str = 'backtest_report.html',
        include_charts: bool = True,
        format: str = 'html'
    ) -> str:
        """
        Generate comprehensive backtest report.

        Creates a professional report with executive summary, metrics tables,
        charts, and trade log.

        Args:
            save_path: Path to save report.
            include_charts: If True, include embedded charts.
            format: 'html' or 'pdf'. Default 'html'.

        Returns:
            Absolute path to saved report file.

        Example:
            >>> results = engine.run()
            >>> path = engine.generate_report('report.html')
        """
        from backtester.analytics import ReportGenerator

        results = {
            'equity_curve': self.generate_equity_curve(),
            'trade_log': self.get_trade_log(),
            'greeks_history': self.get_greeks_history(),
            'strategy_stats': self._strategy.get_statistics(),
            'start_date': self._data_stream.start_date,
            'end_date': self._data_stream.end_date,
            'underlying': self._data_stream.underlying,
        }

        metrics = self.calculate_metrics()

        if format.lower() == 'pdf':
            return ReportGenerator.generate_pdf_report(
                results, metrics, save_path
            )
        else:
            return ReportGenerator.generate_html_report(
                results, metrics, save_path, include_charts
            )

    # =========================================================================
    # Special Methods
    # =========================================================================

    def __repr__(self) -> str:
        """Return detailed string representation."""
        return (
            f"BacktestEngine("
            f"strategy={self._strategy.name!r}, "
            f"capital=${self._initial_capital:,.2f}, "
            f"trading_days={self._data_stream.num_trading_days}, "
            f"trades={self.num_trades}"
            f")"
        )

    def __str__(self) -> str:
        """Return human-readable string."""
        return (
            f"BacktestEngine: {self._strategy.name}, "
            f"${self._initial_capital:,.2f} capital, "
            f"{self._data_stream.num_trading_days} days"
        )


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Main class
    'BacktestEngine',

    # Supporting classes
    'BacktestState',
    'TradeRecord',

    # Exceptions
    'BacktestError',
    'BacktestConfigError',
    'BacktestExecutionError',

    # Constants
    'DEFAULT_INITIAL_CAPITAL',
    'DEFAULT_RISK_FREE_RATE',
]
