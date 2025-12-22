# Architecture

This document describes the high-level system architecture, data flow, and component relationships for Options Backtester.

## System Overview

Options Backtester is a Python-based framework for backtesting systematic options trading strategies. The system provides:

- **Historical Options Data Access**: Query option chains from a Dolt database
- **Realistic Execution Simulation**: Model bid/ask spreads, commissions, and slippage
- **Multi-leg Position Management**: Support for complex structures (straddles, spreads, iron condors)
- **Industry-Standard Analytics**: Sharpe, Sortino, drawdown, VaR, Greeks tracking
- **Extensible Strategy Framework**: Abstract base class for custom strategy implementations

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           OPTIONS BACKTESTER                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐     │
│  │    Data     │   │   Engine    │   │  Strategies │   │  Analytics  │     │
│  │  Layer      │──▶│   Layer     │──▶│   Layer     │──▶│   Layer     │     │
│  └─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘     │
│        │                 │                  │                 │             │
│        ▼                 ▼                  ▼                 ▼             │
│   DoltAdapter      BacktestEngine       Strategy        PerformanceMetrics  │
│                    DataStream           Concrete        RiskAnalytics       │
│                    ExecutionModel       Strategies      Visualization       │
│                    PositionManager                      Dashboard           │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Module Structure

```
code/backtester/
├── __init__.py
├── core/                    # Core data structures
│   ├── option.py           # Option class with Greeks
│   ├── option_structure.py # Multi-leg positions
│   └── pricing.py          # Black-Scholes model
├── engine/                  # Backtesting engine
│   ├── backtest_engine.py  # Main orchestrator
│   ├── data_stream.py      # Data iteration
│   ├── execution.py        # Order execution
│   └── position_manager.py # Position tracking
├── strategies/              # Strategy implementations
│   ├── strategy.py         # Abstract base class
│   ├── short_straddle_strategy.py
│   ├── iron_condor_strategy.py
│   └── volatility_regime_strategy.py
├── structures/              # Pre-built option structures
│   └── straddle.py         # Straddle factory
├── analytics/               # Performance analysis
│   ├── metrics.py          # Performance metrics
│   ├── risk.py             # Risk analytics (VaR, CVaR)
│   ├── visualization.py    # Charting
│   ├── dashboard.py        # Interactive dashboards
│   └── report.py           # Report generation
├── data/                    # Data access
│   └── dolt_adapter.py     # Dolt database adapter
└── utils/                   # Utilities
```

## Data Flow / Function Work Flow

```
                            ┌──────────────────┐
                            │   Dolt Database  │
                            │  (option_chain,  │
                            │   volatility)    │
                            └────────┬─────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                              DoltAdapter                                    │
│  - get_option_chain(symbol, date, expiration_range)                        │
│  - get_implied_volatility(symbol, date)                                    │
│  - get_date_range(symbol) → (min_date, max_date)                          │
└────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                              DataStream                                     │
│  - Iterates through trading days chronologically                           │
│  - Yields (timestamp, market_data) tuples                                  │
│  - market_data includes: spot, option_chain, iv_rank, volatility           │
└────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                            BacktestEngine.run()                             │
│                                                                             │
│   FOR each (timestamp, market_data) in DataStream:                         │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  1. UPDATE POSITIONS                                                 │  │
│   │     strategy.update_positions(market_data, timestamp)               │  │
│   │     position_manager.update_all_positions(market_data, timestamp)   │  │
│   ├─────────────────────────────────────────────────────────────────────┤  │
│   │  2. PROCESS EXITS                                                    │  │
│   │     for structure in strategy.structures:                           │  │
│   │         if strategy.should_exit(structure, market_data):            │  │
│   │             execution_model.execute_exit(structure, market_data)    │  │
│   │             strategy.close_position(structure)                      │  │
│   ├─────────────────────────────────────────────────────────────────────┤  │
│   │  3. PROCESS ENTRIES                                                  │  │
│   │     if strategy.should_enter(market_data):                          │  │
│   │         structure = strategy.create_structure(market_data)          │  │
│   │         execution_model.execute_entry(structure, market_data)       │  │
│   │         strategy.open_position(structure)                           │  │
│   ├─────────────────────────────────────────────────────────────────────┤  │
│   │  4. RECORD STATE                                                     │  │
│   │     equity, cash, positions_value, greeks → equity_curve            │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   RETURN results: equity_curve, trade_log, greeks_history, metrics         │
└────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                              Analytics                                      │
│  - PerformanceMetrics: Sharpe, Sortino, Calmar, drawdown, win rate         │
│  - RiskAnalytics: VaR, CVaR, tail risk, Greeks analysis                    │
│  - Visualization: Equity curves, drawdown charts, P&L distribution         │
│  - Dashboard: Interactive HTML dashboards                                   │
│  - ReportGenerator: Comprehensive backtest reports                          │
└────────────────────────────────────────────────────────────────────────────┘
```

## Component Architecture

### Core Functions and Data Structures

#### Option (`core/option.py`)

Represents a single options contract with real-time Greek calculations.

| Attribute | Type | Purpose |
|-----------|------|---------|
| `symbol` | str | Options symbol (e.g., "SPY240621C00450000") |
| `underlying` | str | Underlying ticker (e.g., "SPY") |
| `strike` | float | Strike price |
| `expiration` | datetime | Expiration date |
| `option_type` | str | "call" or "put" |
| `position_type` | str | "long" or "short" |
| `quantity` | int | Number of contracts |
| `entry_price` | float | Price at entry (per share) |
| `current_price` | float | Current market price |
| `iv` | float | Implied volatility (decimal) |

**Key Methods:**
- `calculate_greeks(spot, rate, div_yield) → dict` - Returns delta, gamma, theta, vega, rho
- `calculate_pnl() → float` - Unrealized P&L including multiplier
- `update_price(price)` - Update current market price

**Constants:**
- `CONTRACT_MULTIPLIER = 100` - Standard options multiplier

#### OptionStructure (`core/option_structure.py`)

Container for multi-leg option positions (straddles, spreads, iron condors).

| Attribute | Type | Purpose |
|-----------|------|---------|
| `structure_id` | str | Unique identifier |
| `structure_type` | str | Type name (e.g., "short_straddle") |
| `underlying` | str | Underlying ticker |
| `legs` | List[Option] | Component options |
| `entry_date` | datetime | Position open date |
| `net_premium` | float | Net premium collected/paid |

**Key Methods:**
- `add_leg(option)` - Add option to structure
- `calculate_pnl() → float` - Aggregate P&L across all legs
- `get_net_greeks() → dict` - Sum Greeks across legs
- `update_prices(market_data)` - Update all leg prices

**Greek Aggregation:**
```python
net_delta = sum(leg.delta for leg in legs)
net_gamma = sum(leg.gamma for leg in legs)
# ... etc for theta, vega, rho
```

#### Pricing (`core/pricing.py`)

Black-Scholes option pricing model implementation.

**Functions:**
- `black_scholes_price(S, K, T, r, sigma, option_type) → float`
- `calculate_delta(S, K, T, r, sigma, option_type) → float`
- `calculate_gamma(S, K, T, r, sigma) → float`
- `calculate_theta(S, K, T, r, sigma, option_type) → float`
- `calculate_vega(S, K, T, r, sigma) → float`
- `calculate_rho(S, K, T, r, sigma, option_type) → float`

**Parameters:**
| Symbol | Description |
|--------|-------------|
| S | Spot price |
| K | Strike price |
| T | Time to expiration (years) |
| r | Risk-free rate (annual) |
| sigma | Implied volatility (annual) |

### Engine Components

#### BacktestEngine (`engine/backtest_engine.py`)

Central orchestrator that coordinates all backtesting components.

```
┌─────────────────────────────────────────────────────────────────┐
│                        BacktestEngine                            │
├─────────────────────────────────────────────────────────────────┤
│  Inputs:                                                         │
│    - strategy: Strategy instance                                 │
│    - data_stream: DataStream instance                           │
│    - execution_model: ExecutionModel instance                   │
│    - initial_capital: float (default $100,000)                  │
├─────────────────────────────────────────────────────────────────┤
│  Outputs (from run()):                                           │
│    - equity_curve: DataFrame (timestamp, equity, greeks, etc.)  │
│    - trade_log: DataFrame (entries, exits, P&L)                 │
│    - greeks_history: DataFrame (portfolio Greeks over time)     │
│    - final_equity: float                                        │
│    - total_return: float (percentage)                           │
│    - strategy_stats: dict                                       │
│    - execution_stats: dict                                      │
└─────────────────────────────────────────────────────────────────┘
```

**Supporting Classes:**
- `BacktestState` - Snapshot of portfolio state at a timestamp
- `TradeRecord` - Details of a single trade (open/close)

#### DataStream (`engine/data_stream.py`)

Iterator providing market data day-by-day.

| Property | Type | Purpose |
|----------|------|---------|
| `start_date` | datetime | Backtest start |
| `end_date` | datetime | Backtest end |
| `underlying` | str | Ticker symbol |
| `num_trading_days` | int | Total days in period |

**Yields:**
```python
(timestamp: datetime, market_data: dict)

market_data = {
    'spot': float,           # Underlying price
    'option_chain': DataFrame,# Available options
    'iv_rank': float,        # IV percentile (0-100)
    'volatility': float,     # Current implied volatility
    'dte': int,              # Days to target expiration
}
```

#### ExecutionModel (`engine/execution.py`)

Simulates realistic order execution with market microstructure.

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `commission_per_contract` | $0.65 | Per-contract fee |
| `slippage_pct` | 0.01 | Slippage as % of mid price |
| `use_bid_ask` | True | Use bid/ask vs mid price |

**Methods:**
- `execute_entry(structure, market_data, timestamp) → dict`
  - Returns: total_cost, commissions, slippage, fills
- `execute_exit(structure, market_data, timestamp) → dict`
  - Returns: total_proceeds, commissions, slippage, fills

**Execution Logic:**
```
Entry (buying): pay at ask + slippage + commission
Entry (selling): receive at bid - slippage - commission
Exit: reverse of entry positions
```

#### PositionManager (`engine/position_manager.py`)

Tracks all open positions and calculates portfolio-level metrics.

**Methods:**
- `add_position(structure, strategy_name, timestamp)` - Add new position
- `remove_position(structure_id, timestamp, realized_pnl)` - Close position
- `update_all_positions(market_data, timestamp)` - Mark-to-market
- `calculate_portfolio_value(market_data) → float` - Total positions value
- `get_portfolio_greeks(market_data, rate) → dict` - Aggregate Greeks
- `calculate_unrealized_pnl() → float` - Open position P&L

### Strategy Layer

#### Strategy Base Class (`strategies/strategy.py`)

Abstract base class that all strategies must inherit from.

```python
class Strategy(ABC):
    """
    Required implementations:
    - should_enter(market_data) → bool
    - should_exit(structure, market_data) → bool

    Optional override:
    - create_structure(market_data) → OptionStructure
    """
```

| Attribute | Type | Purpose |
|-----------|------|---------|
| `name` | str | Strategy identifier |
| `initial_capital` | float | Starting capital |
| `position_limits` | dict | Risk constraints |
| `structures` | List[OptionStructure] | Open positions |
| `trade_history` | List[dict] | Completed trades |

**Position Management:**
- `open_position(structure, validate_limits=True)` - Add position with validation
- `close_position(structure, exit_data) → dict` - Close and record P&L
- `update_positions(market_data, timestamp)` - Update all positions

**Risk Limits (position_limits dict):**
```python
{
    'max_positions': 10,          # Max concurrent positions
    'max_delta': 100,             # Portfolio delta limit
    'max_gamma': 50,              # Portfolio gamma limit
    'max_capital_per_trade': 0.10 # 10% of capital per trade
}
```

#### Concrete Strategy Example: ShortStraddleHighIVStrategy

```python
class ShortStraddleHighIVStrategy(Strategy):
    """
    Entry: IV rank > threshold (default 70%)
    Exit: Profit target (50%) OR Loss limit (2x) OR DTE <= 7
    """

    Parameters:
    - iv_rank_threshold: float (0-100)
    - profit_target_pct: float (0-1)
    - loss_limit_pct: float (multiplier)
    - exit_dte: int (days)
    - min_entry_dte: int (days)
```

### Analytics Components

#### PerformanceMetrics (`analytics/metrics.py`)

Industry-standard performance calculations (all static methods).

**Returns-Based Metrics:**
| Metric | Formula | Interpretation |
|--------|---------|----------------|
| Total Return | (Final - Initial) / Initial | Overall performance |
| CAGR | (Final/Initial)^(252/days) - 1 | Annualized return |
| Sharpe | (R - Rf) / σ × √252 | Risk-adjusted return |
| Sortino | (R - Rf) / σ_down × √252 | Downside risk-adjusted |
| Calmar | CAGR / |Max Drawdown| | Return per drawdown |

**Drawdown Metrics:**
| Metric | Description |
|--------|-------------|
| Max Drawdown | Largest peak-to-trough decline |
| Duration | Days from peak to trough |
| Recovery | Days from trough to new high |
| Ulcer Index | RMS of drawdown values |

**Trade-Based Metrics:**
| Metric | Formula |
|--------|---------|
| Win Rate | Winning trades / Total trades |
| Profit Factor | Gross profit / Gross loss |
| Expectancy | (Win% × Avg Win) - (Loss% × Avg Loss) |
| Payoff Ratio | Avg Win / |Avg Loss| |

#### RiskAnalytics (`analytics/risk.py`)

Risk measurement and analysis tools.

**Value at Risk (VaR):**
- Historical: Percentile of actual returns
- Parametric: μ - z × σ (assumes normal distribution)

**Conditional VaR (CVaR / Expected Shortfall):**
- Average loss when loss exceeds VaR threshold

**Tail Risk:**
- Skewness: Asymmetry of return distribution (negative = left tail)
- Kurtosis: Fat tails (positive = more extreme events)

**Greeks Analysis:**
- Track delta, gamma, theta, vega over time
- Calculate max/min/average exposure

### Data Layer

#### DoltAdapter (`data/dolt_adapter.py`)

Interface to Dolt version-controlled SQL database.

**Connection:**
```python
adapter = DoltAdapter('dolt_data/options')
adapter.connect()
```

**Key Methods:**
| Method | Returns |
|--------|---------|
| `get_option_chain(symbol, date, exp_range)` | DataFrame of options |
| `get_implied_volatility(symbol, date)` | Float IV value |
| `get_date_range(symbol)` | (min_date, max_date) tuple |
| `get_available_symbols()` | List of tickers |
| `query_custom(sql)` | DataFrame result |

**Database Tables:**
- `option_chain`: strike, expiration, bid, ask, last, iv, delta, gamma, etc.
- `volatility_history`: date, symbol, iv, iv_rank, hv_20, hv_60

### Library Modules

#### Structures (`structures/`)

Pre-built option structure factories for common strategies.

**ShortStraddle:**
```python
straddle = ShortStraddle.create(
    underlying='SPY',
    strike=450.0,
    expiration=datetime(2024, 7, 19),
    call_price=6.50,
    put_price=6.25,
    quantity=1
)

Properties:
- net_premium: Total premium collected
- max_profit: Same as net_premium
- lower_breakeven: strike - total_premium
- upper_breakeven: strike + total_premium
```

#### Utils (`utils/`)

Utility functions for dates, validation, and common operations.

## Usage Example

```python
from datetime import datetime
from backtester.data.dolt_adapter import DoltAdapter
from backtester.engine.data_stream import DataStream
from backtester.engine.execution import ExecutionModel
from backtester.engine.backtest_engine import BacktestEngine
from backtester.strategies.short_straddle_strategy import ShortStraddleHighIVStrategy

# 1. Connect to data
adapter = DoltAdapter('dolt_data/options')
adapter.connect()

# 2. Create data stream
data_stream = DataStream(
    adapter=adapter,
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2023, 12, 31),
    underlying='SPY'
)

# 3. Create execution model
execution = ExecutionModel(
    commission_per_contract=0.65,
    slippage_pct=0.01
)

# 4. Create strategy
strategy = ShortStraddleHighIVStrategy(
    name='Short Straddle IV70',
    initial_capital=100000,
    iv_rank_threshold=70,
    profit_target_pct=0.50,
    loss_limit_pct=2.0
)

# 5. Create and run engine
engine = BacktestEngine(
    strategy=strategy,
    data_stream=data_stream,
    execution_model=execution,
    initial_capital=100000
)

results = engine.run()

# 6. Analyze results
metrics = engine.calculate_metrics()
print(f"Total Return: {metrics['summary']['total_return_pct']:.2f}%")
print(f"Sharpe Ratio: {metrics['summary']['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {metrics['summary']['max_drawdown']:.2%}")

# 7. Generate reports
engine.create_dashboard('results/dashboard.html')
```

## Key Design Principles

1. **Separation of Concerns**: Each module has a single responsibility
2. **Event-Driven Architecture**: Clean timestep-based simulation loop
3. **Extensibility**: Abstract Strategy class allows custom implementations
4. **Financial Correctness**: Proper handling of multipliers, Greeks, execution costs
5. **Comprehensive Analytics**: Industry-standard metrics for evaluation
