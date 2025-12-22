# Backtesting Engine Documentation

## Overview

The backtesting engine provides the core infrastructure for simulating options trading strategies over historical data. It consists of four main components:

1. **DataStream** - Iterator over time-series market data
2. **PositionManager** - Multi-strategy position tracking
3. **ExecutionModel** - Realistic order execution simulation
4. **BacktestEngine** - Main orchestrator coordinating all components

## Architecture

```
                    +------------------+
                    |  BacktestEngine  |
                    +--------+---------+
                             |
            +----------------+----------------+
            |                |                |
    +-------v-------+ +------v------+ +-------v-------+
    |  DataStream   | | Execution   | |  Position     |
    |               | | Model       | |  Manager      |
    +-------+-------+ +------+------+ +-------+-------+
            |                |                |
    +-------v-------+        |         +------v-------+
    | DoltAdapter   |        |         |  Option      |
    |  (Data)       |        |         |  Structure   |
    +---------------+        |         +--------------+
                             |
                    +--------v--------+
                    |    Strategy     |
                    | (User-defined)  |
                    +-----------------+
```

## Event Loop

The backtest engine follows an event-driven architecture:

```python
for timestamp, market_data in data_stream:
    # 1. Update all position prices from market data
    strategy.update_positions(market_data)

    # 2. Check exit conditions for each open position
    for structure in list(strategy.structures):
        if strategy.should_exit(structure, market_data):
            exit_data = execution_model.execute_exit(structure, market_data)
            strategy.close_position(structure, exit_data)
            record_trade(...)

    # 3. Check entry conditions
    if strategy.should_enter(market_data):
        structure = strategy.create_structure(market_data)
        entry_data = execution_model.execute_entry(structure, market_data)
        strategy.open_position(structure)
        record_trade(...)

    # 4. Record state (equity, Greeks, etc.)
    equity = strategy.get_equity()
    greeks = strategy.calculate_portfolio_greeks(...)
    record_state(timestamp, equity, greeks)
```

---

## Component Details

### 1. DataStream

**File**: `backtester/engine/data_stream.py`

The DataStream provides an iterator interface over trading days, fetching market data from the data source (DoltAdapter) and handling weekends/holidays.

#### Key Features
- Iterator protocol (`__iter__`, `__next__`)
- Skip weekends and market holidays
- Data caching for efficiency
- Optional preloading of all data
- Graceful handling of missing data

#### Initialization

```python
from backtester.engine import DataStream
from backtester.data.dolt_adapter import DoltAdapter

adapter = DoltAdapter('/path/to/dolt/options')
adapter.connect()

stream = DataStream(
    data_source=adapter,           # DoltAdapter instance
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2023, 12, 31),
    underlying='SPY',
    dte_range=(7, 45),             # Min/max DTE for option chain
    cache_enabled=True,            # Cache fetched data
    preload=False,                 # Pre-load all data upfront
    skip_missing_data=True         # Skip days with no data
)
```

#### Market Data Dictionary

Each iteration returns a tuple of `(timestamp, market_data)`:

```python
market_data = {
    'date': datetime,              # Trading date
    'spot': float,                 # Estimated spot price
    'option_chain': pd.DataFrame,  # Available options
    'iv': float,                   # ATM implied volatility
    'vix': Optional[float]         # VIX level (if available)
}
```

#### Usage

```python
# Iterate through trading days
for timestamp, market_data in stream:
    print(f"Date: {timestamp.date()}, Spot: ${market_data['spot']:.2f}")

# Reset to start over
stream.reset()

# Skip days
stream.skip(5)

# Check progress
print(f"Progress: {stream.progress:.1%}")

# Clear cache to free memory
stream.clear_cache()
```

---

### 2. PositionManager

**File**: `backtester/engine/position_manager.py`

The PositionManager tracks all positions across multiple strategies, providing portfolio-level metrics and position lookup functionality.

#### Key Features
- Multi-strategy position tracking
- Position-level and portfolio-level metrics
- Margin requirement calculation
- Portfolio Greeks aggregation
- Position history tracking

#### Initialization

```python
from backtester.engine import PositionManager

manager = PositionManager()
```

#### Adding/Removing Positions

```python
# Add a position
structure_id = manager.add_position(
    structure=iron_condor,         # OptionStructure
    strategy_name='IronCondor',    # Strategy name
    open_timestamp=datetime.now()  # When opened
)

# Remove a position
closed_structure = manager.remove_position(
    structure_id='abc123',
    close_timestamp=datetime.now(),
    realized_pnl=500.0
)
```

#### Position Queries

```python
# Get all active positions
all_positions = manager.get_all_positions()

# Get positions for a specific strategy
condor_positions = manager.get_positions_by_strategy('IronCondor')

# Get positions for a specific underlying
spy_positions = manager.get_positions_by_underlying('SPY')

# Check if position exists
if manager.has_position('abc123'):
    pos = manager.get_position('abc123')
```

#### Portfolio Metrics

```python
# Total margin requirement
margin = manager.calculate_total_margin()

# Mark-to-market portfolio value
value = manager.calculate_portfolio_value(market_data)

# Portfolio Greeks
greeks = manager.get_portfolio_greeks(market_data, rate=0.04)
print(f"Delta: {greeks['delta']:.2f}")
print(f"Gamma: {greeks['gamma']:.4f}")
print(f"Theta: {greeks['theta']:.2f}")
print(f"Vega: {greeks['vega']:.2f}")

# P&L
unrealized = manager.calculate_unrealized_pnl()
total_pnl = manager.calculate_total_pnl()
```

#### Statistics

```python
stats = manager.get_statistics()
print(f"Active positions: {stats['num_active']}")
print(f"Closed positions: {stats['num_closed']}")
print(f"Total P&L: ${stats['total_pnl']:,.2f}")

# Position summary as DataFrame
summary_df = manager.get_position_summary()
```

---

### 3. ExecutionModel

**File**: `backtester/engine/execution.py`

The ExecutionModel simulates realistic order execution, accounting for bid/ask spreads, commissions, and slippage.

#### Key Features
- Bid/ask spread modeling (buy at ask, sell at bid)
- Configurable commission per contract
- Optional slippage percentage
- Transaction cost breakdown
- Execution logging

#### Initialization

```python
from backtester.engine import ExecutionModel

execution = ExecutionModel(
    commission_per_contract=0.65,  # $0.65 per contract
    slippage_pct=0.001,            # 0.1% slippage
    use_bid_ask=True,              # Use bid/ask prices
    default_spread_pct=0.02        # Default 2% spread if bid/ask not available
)
```

#### Executing Entries

```python
entry_result = execution.execute_entry(
    structure=iron_condor,
    market_data=market_data,
    timestamp=datetime.now()
)

print(f"Entry prices: {entry_result['entry_prices']}")
print(f"Total cost: ${entry_result['total_cost']:,.2f}")
print(f"Commission: ${entry_result['commissions']:.2f}")
print(f"Slippage: ${entry_result['slippage']:.2f}")
```

#### Executing Exits

```python
exit_result = execution.execute_exit(
    structure=iron_condor,
    market_data=market_data,
    timestamp=datetime.now()
)

print(f"Exit prices: {exit_result['exit_prices']}")
print(f"Total proceeds: ${exit_result['total_proceeds']:,.2f}")
print(f"Commission: ${exit_result['commissions']:.2f}")
```

#### Fill Price Logic

```python
# Get fill price for a single option
fill_price = execution.get_fill_price(
    option=call_option,
    market_data=market_data,
    side='buy'   # 'buy' = ask price, 'sell' = bid price
)
```

Fill price determination:
1. Look up option in market data option chain
2. If `use_bid_ask=True` and bid/ask available: use ask for buy, bid for sell
3. If mid available: apply default spread percentage
4. Fall back to option's current price with spread
5. Fall back to option's entry price with spread

#### Execution Summary

```python
summary = execution.get_execution_summary()
print(f"Total executions: {summary['num_executions']}")
print(f"Total commissions: ${summary['total_commissions']:,.2f}")
print(f"Total slippage: ${summary['total_slippage']:,.2f}")

# Clear execution log
execution.clear_log()
```

---

### 4. BacktestEngine

**File**: `backtester/engine/backtest_engine.py`

The BacktestEngine is the main orchestrator that coordinates all components and runs the simulation.

#### Key Features
- Clean event loop architecture
- Strategy integration (entry/exit signals)
- Trade logging and tracking
- Equity curve generation
- Greeks history tracking
- Callback support for custom logic

#### Initialization

```python
from backtester.engine import BacktestEngine, DataStream, ExecutionModel

engine = BacktestEngine(
    strategy=my_strategy,          # Strategy instance
    data_stream=data_stream,       # DataStream instance
    execution_model=execution,     # ExecutionModel instance
    initial_capital=100000.0,      # Starting capital
    risk_free_rate=0.04            # For Greek calculations
)
```

#### Running a Backtest

```python
results = engine.run()

# Access results
print(f"Initial Capital: ${results['initial_capital']:,.2f}")
print(f"Final Equity: ${results['final_equity']:,.2f}")
print(f"Total Return: {results['total_return']:.2%}")
print(f"Total P&L: ${results['total_pnl']:,.2f}")
print(f"Number of Trades: {results['num_trades']}")

# Access DataFrames
equity_curve = results['equity_curve']      # Time series of equity
trade_log = results['trade_log']            # All trades
greeks_history = results['greeks_history']  # Greeks over time

# Component statistics
strategy_stats = results['strategy_stats']
execution_stats = results['execution_stats']
```

#### Equity Curve

The equity curve DataFrame contains:

| Column | Description |
|--------|-------------|
| equity | Total portfolio value |
| cash | Cash on hand |
| positions_value | Mark-to-market value of positions |
| num_positions | Number of active positions |
| realized_pnl | Cumulative realized P&L |
| unrealized_pnl | Current unrealized P&L |
| delta | Portfolio delta |
| gamma | Portfolio gamma |
| theta | Portfolio theta |
| vega | Portfolio vega |
| rho | Portfolio rho |

#### Trade Log

The trade log DataFrame contains:

| Column | Description |
|--------|-------------|
| trade_id | Unique trade identifier |
| structure_id | Structure identifier |
| structure_type | Type of structure |
| underlying | Underlying symbol |
| action | 'open' or 'close' |
| timestamp | Trade timestamp |
| num_legs | Number of legs |
| net_premium | Net premium at entry |
| total_cost | Total cost of entry |
| total_proceeds | Proceeds from exit |
| commission | Commission paid |
| slippage | Slippage cost |
| realized_pnl | P&L (for closes) |
| exit_reason | Why position was closed |

#### Callbacks

```python
# Called after each step
def on_step(timestamp, market_data, state):
    print(f"{timestamp}: Equity = ${state['equity']:,.2f}")

engine.set_on_step_callback(on_step)

# Called after each trade
def on_trade(trade_record):
    print(f"Trade: {trade_record.action} {trade_record.structure_type}")

engine.set_on_trade_callback(on_trade)

# Run backtest with callbacks
results = engine.run()
```

---

## Financial Correctness

### Commission Calculation

```
Total Commission = commission_per_contract x num_contracts
```

For multi-leg structures:
```
num_contracts = sum(leg.quantity for leg in structure.options)
```

### Bid/Ask Spread

- **Buy orders**: Fill at ask price (mid + half spread)
- **Sell orders**: Fill at bid price (mid - half spread)

### P&L Calculation

```
Unrealized P&L = Current Value - Entry Cost
Realized P&L = Exit Proceeds - Entry Cost - Commissions
Total P&L = Realized P&L + Unrealized P&L
```

### Equity Curve

```
Equity = Cash + Mark-to-Market Value of All Positions
       = Initial Capital + Total P&L
```

### Portfolio Greeks

```
Portfolio Delta = sum(structure.net_delta for structure in positions)
Portfolio Gamma = sum(structure.net_gamma for structure in positions)
Portfolio Theta = sum(structure.net_theta for structure in positions)
Portfolio Vega = sum(structure.net_vega for structure in positions)
Portfolio Rho = sum(structure.net_rho for structure in positions)
```

---

## Integration with Strategy

Strategies must implement the abstract methods from the Strategy base class:

```python
from backtester.strategies.strategy import Strategy

class MyStrategy(Strategy):
    def should_enter(self, market_data: Dict) -> bool:
        """Determine if entry conditions are met."""
        # Example: Enter when IV is high
        return market_data.get('iv', 0) > 0.25

    def should_exit(self, structure: OptionStructure, market_data: Dict) -> bool:
        """Determine if exit conditions are met for a position."""
        # Example: Exit at 50% profit or 100% loss
        pnl_pct = structure.calculate_pnl_percent()
        return pnl_pct >= 0.50 or pnl_pct <= -1.0

    def create_structure(self, market_data: Dict) -> OptionStructure:
        """Create the structure to enter (optional)."""
        # Build and return the OptionStructure
        return self._build_iron_condor(market_data)
```

---

## Usage Example

```python
from datetime import datetime
from backtester.data.dolt_adapter import DoltAdapter
from backtester.engine import (
    BacktestEngine,
    DataStream,
    ExecutionModel,
)

# 1. Set up data source
adapter = DoltAdapter('/path/to/dolt/options')
adapter.connect()

# 2. Create data stream
data_stream = DataStream(
    data_source=adapter,
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2023, 12, 31),
    underlying='SPY',
    dte_range=(7, 45)
)

# 3. Create execution model
execution = ExecutionModel(
    commission_per_contract=0.65,
    slippage_pct=0.001,
    use_bid_ask=True
)

# 4. Create strategy (user-defined, see Run 8)
strategy = MyStrategy(
    name='IronCondor',
    initial_capital=100000.0
)

# 5. Create backtest engine
engine = BacktestEngine(
    strategy=strategy,
    data_stream=data_stream,
    execution_model=execution,
    initial_capital=100000.0
)

# 6. Run backtest
results = engine.run()

# 7. Analyze results
print(f"Final Equity: ${results['final_equity']:,.2f}")
print(f"Total Return: {results['total_return']:.2%}")
print(f"Trades: {results['num_trades']}")

# Plot equity curve
import matplotlib.pyplot as plt
equity_df = results['equity_curve']
plt.figure(figsize=(12, 6))
plt.plot(equity_df.index, equity_df['equity'])
plt.xlabel('Date')
plt.ylabel('Equity ($)')
plt.title('Equity Curve')
plt.show()

# Clean up
adapter.close()
```

---

## File Locations

| Component | File Path |
|-----------|-----------|
| DataStream | `backtester/engine/data_stream.py` |
| PositionManager | `backtester/engine/position_manager.py` |
| ExecutionModel | `backtester/engine/execution.py` |
| BacktestEngine | `backtester/engine/backtest_engine.py` |
| Module Exports | `backtester/engine/__init__.py` |
| Tests | `tests/test_engine.py` |

---

## Exception Classes

| Exception | Description |
|-----------|-------------|
| `DataStreamError` | Base exception for DataStream |
| `DataStreamConfigError` | Invalid configuration |
| `DataNotAvailableError` | Required data not available |
| `PositionManagerError` | Base exception for PositionManager |
| `PositionNotFoundError` | Position not in active positions |
| `DuplicatePositionError` | Adding already-tracked position |
| `ExecutionError` | Base exception for ExecutionModel |
| `ExecutionConfigError` | Invalid configuration |
| `PriceNotAvailableError` | Cannot determine fill price |
| `BacktestError` | Base exception for BacktestEngine |
| `BacktestConfigError` | Invalid configuration |
| `BacktestExecutionError` | Error during backtest execution |

---

## Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `DEFAULT_MIN_DTE` | 7 | Minimum days to expiration |
| `DEFAULT_MAX_DTE` | 60 | Maximum days to expiration |
| `DEFAULT_COMMISSION_PER_CONTRACT` | 0.65 | Commission per contract |
| `DEFAULT_SLIPPAGE_PCT` | 0.0 | Default slippage |
| `DEFAULT_SPREAD_PCT` | 0.02 | Default bid/ask spread |
| `DEFAULT_INITIAL_CAPITAL` | 100000.0 | Default starting capital |
| `DEFAULT_RISK_FREE_RATE` | 0.04 | Risk-free rate for Greeks |
| `DEFAULT_MARGIN_PER_CONTRACT` | 2000.0 | Fallback margin |

---

## Dependencies

- **Internal**:
  - `backtester.data.dolt_adapter.DoltAdapter`
  - `backtester.core.option.Option`
  - `backtester.core.option_structure.OptionStructure`
  - `backtester.strategies.strategy.Strategy`

- **External**:
  - `numpy`: Numerical operations
  - `pandas`: DataFrame handling
  - `scipy.optimize`: Breakeven calculations
