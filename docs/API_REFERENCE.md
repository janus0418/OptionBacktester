# API Reference

Complete API reference for the Options Backtesting System.

## Core Module (`backtester.core`)

### Option Class

```python
from backtester.core.option import Option

option = Option(
    option_type: str,           # 'call' or 'put'
    position_type: str,         # 'long' or 'short'
    underlying: str,            # e.g., 'SPY'
    strike: float,
    expiration: datetime,
    quantity: int,
    entry_price: float,
    entry_date: datetime,
    underlying_price_at_entry: float,
    implied_vol_at_entry: float
)
```

**Properties:**
- `current_value: float` - Mark-to-market value
- `pnl: float` - Profit/loss
- `return_pct: float` - Return percentage
- `greeks: Dict[str, float]` - Delta, gamma, theta, vega, rho
- `days_to_expiration: int`

**Methods:**
- `update_price(current_price, current_date, underlying_price, risk_free_rate)`
- `calculate_greeks(underlying_price, risk_free_rate) -> Dict[str, float]`

### OptionStructure Class

```python
from backtester.core.option_structure import OptionStructure

# Use concrete implementations
from backtester.structures import ShortStraddle, IronCondor, etc.
```

**Properties:**
- `legs: List[Option]`
- `current_value: float`
- `entry_value: float`
- `pnl: float`
- `greeks: Dict[str, float]`
- `max_profit: float`
- `max_loss: float`
- `expiration: datetime`
- `days_to_expiration: int`

**Methods:**
- `update_prices(...)`
- `add_leg(option: Option)`

### BlackScholesPricer

```python
from backtester.core.pricing import BlackScholesPricer

pricer = BlackScholesPricer()

price = pricer.price(
    S: float,              # Spot price
    K: float,              # Strike
    T: float,              # Time to expiration (years)
    r: float,              # Risk-free rate
    sigma: float,          # Volatility (annualized)
    option_type: str       # 'call' or 'put'
) -> float

greeks = pricer.calculate_greeks(
    S, K, T, r, sigma, option_type
) -> Dict[str, float]  # delta, gamma, theta, vega, rho
```

## Structures Module (`backtester.structures`)

All structures have `.create()` factory methods:

```python
from backtester.structures import *

# Straddles
straddle = ShortStraddle.create(underlying, strike, expiration,
                                call_price, put_price, quantity,
                                entry_date, underlying_price)

# Strangles
strangle = ShortStrangle.create(underlying, call_strike, put_strike,
                                expiration, call_price, put_price,
                                quantity, entry_date, underlying_price)

# Vertical Spreads
bull_call = BullCallSpread.create(underlying, long_strike, short_strike,
                                  expiration, long_call_price, short_call_price,
                                  quantity, entry_date, underlying_price)

# Condors
iron_condor = IronCondor.create(underlying, put_short_strike, put_long_strike,
                                call_short_strike, call_long_strike, expiration,
                                put_short_price, put_long_price, call_short_price,
                                call_long_price, quantity, entry_date, underlying_price)
```

## Strategies Module (`backtester.strategies`)

### Strategy Base Class

```python
from backtester.strategies import Strategy

class MyStrategy(Strategy):
    def __init__(self, name: str, initial_capital: float, **kwargs):
        super().__init__(name, initial_capital)

    def should_enter(self, current_date, market_data, option_chain,
                    available_capital) -> bool:
        pass

    def should_exit(self, position, current_date, market_data,
                    option_chain) -> bool:
        pass

    def create_structure(self, current_date, market_data, option_chain,
                        available_capital) -> Optional[OptionStructure]:
        pass
```

### Built-in Strategies

```python
from backtester.strategies import (
    ShortStraddleHighIVStrategy,
    IronCondorStrategy,
    VolatilityRegimeStrategy
)

strategy = ShortStraddleHighIVStrategy(
    name: str,
    initial_capital: float,
    iv_rank_threshold: float = 70.0,
    profit_target_pct: float = 0.50,
    stop_loss_pct: float = 2.0,
    max_dte: int = 45,
    min_dte: int = 7,
    max_positions: int = 5
)
```

## Engine Module (`backtester.engine`)

### BacktestEngine

```python
from backtester.engine.backtest_engine import BacktestEngine

engine = BacktestEngine(
    strategy: Strategy,
    data_stream: DataStream,
    execution_model: ExecutionModel,
    initial_capital: float,
    risk_free_rate: float = 0.04
)

results = engine.run() -> Dict[str, Any]
```

**Results Dictionary:**
```python
{
    'final_equity': float,
    'total_return': float,
    'equity_curve': pd.DataFrame,      # Columns: date, equity, cash, positions_value
    'trade_log': pd.DataFrame,         # Trade-by-trade results
    'greeks_history': pd.DataFrame,    # Portfolio Greeks over time
    'positions_history': List[Dict]
}
```

### DataStream

```python
from backtester.engine.data_stream import DataStream

data_stream = DataStream(
    adapter: DoltAdapter,
    start_date: datetime,
    end_date: datetime,
    underlying: str
)

# Methods
market_data = data_stream.get_market_data(date: datetime) -> MarketData
option_chain = data_stream.get_option_chain(date: datetime,
                                            expiration: datetime) -> OptionChain
```

### ExecutionModel

```python
from backtester.engine.execution import ExecutionModel

execution = ExecutionModel(
    commission_per_contract: float = 0.65,
    slippage_pct: float = 0.01,
    fill_on: str = 'mid'  # 'mid', 'bid', 'ask'
)

result = execution.execute_entry(structure: OptionStructure) -> ExecutionResult
result = execution.execute_exit(structure: OptionStructure) -> ExecutionResult
```

## Analytics Module (`backtester.analytics`)

### PerformanceMetrics

```python
from backtester.analytics import PerformanceMetrics

# All methods are static
PerformanceMetrics.calculate_total_return(equity_curve: pd.DataFrame) -> float
PerformanceMetrics.calculate_annualized_return(equity_curve: pd.DataFrame) -> float
PerformanceMetrics.calculate_sharpe_ratio(returns: pd.Series,
                                          risk_free_rate: float = 0.04) -> float
PerformanceMetrics.calculate_sortino_ratio(returns: pd.Series,
                                           risk_free_rate: float = 0.04) -> float
PerformanceMetrics.calculate_calmar_ratio(equity_curve: pd.DataFrame,
                                          returns: pd.Series) -> float
PerformanceMetrics.calculate_max_drawdown(equity_curve: pd.DataFrame) -> float
PerformanceMetrics.calculate_win_rate(trade_log: pd.DataFrame) -> float
PerformanceMetrics.calculate_profit_factor(trade_log: pd.DataFrame) -> float
PerformanceMetrics.calculate_expectancy(trade_log: pd.DataFrame) -> float
PerformanceMetrics.calculate_payoff_ratio(trade_log: pd.DataFrame) -> float
PerformanceMetrics.calculate_average_win(trade_log: pd.DataFrame) -> float
PerformanceMetrics.calculate_average_loss(trade_log: pd.DataFrame) -> float
```

### RiskAnalytics

```python
from backtester.analytics import RiskAnalytics

RiskAnalytics.calculate_var(returns: pd.Series, confidence: float = 0.95) -> float
RiskAnalytics.calculate_cvar(returns: pd.Series, confidence: float = 0.95) -> float
RiskAnalytics.calculate_tail_risk(returns: pd.Series) -> Dict[str, float]
```

### Visualization

```python
from backtester.analytics import Visualization

fig = Visualization.plot_equity_curve(
    equity_curve: pd.DataFrame,
    backend: str = 'matplotlib',  # or 'plotly'
    save_path: Optional[str] = None,
    show: bool = True
)

fig = Visualization.plot_drawdown(equity_curve, backend, save_path, show)
fig = Visualization.plot_pnl_distribution(trade_log, backend, save_path, show)
fig = Visualization.plot_greeks_over_time(greeks_history, backend, save_path, show)
fig = Visualization.plot_returns_distribution(returns, backend, save_path, show)
```

### Dashboard

```python
from backtester.analytics import Dashboard

Dashboard.create_performance_dashboard(
    results: Dict[str, Any],
    metrics: Dict[str, Any],
    save_path: str,
    title: str = "Strategy Performance Dashboard"
)
```

### ReportGenerator

```python
from backtester.analytics import ReportGenerator

ReportGenerator.generate_html_report(
    results: Dict[str, Any],
    metrics: Dict[str, Any],
    save_path: str,
    title: str = "Backtest Report",
    include_charts: bool = True
)
```

## Data Module (`backtester.data`)

### DoltAdapter

```python
from backtester.data.dolt_adapter import DoltAdapter

adapter = DoltAdapter(
    database: str,
    host: str = 'localhost',
    port: int = 3306,
    user: str = 'root',
    password: str = ''
)

# Methods
adapter.connect()
adapter.disconnect()
is_connected = adapter.is_connected() -> bool
```

### MarketData

```python
from backtester.data.market_data import MarketData

market_data = MarketData(
    underlying: str,
    date: datetime,
    open: float,
    high: float,
    low: float,
    close: float,
    volume: int
)
```

### OptionChain

```python
from backtester.data.market_data import OptionChain

option_chain = OptionChain(
    underlying: str,
    date: datetime,
    expiration: datetime,
    calls: List[Dict],  # List of call option data
    puts: List[Dict]    # List of put option data
)

# Option data dictionaries contain:
{
    'strike': float,
    'bid': float,
    'ask': float,
    'mid': float,
    'implied_volatility': float,
    'delta': float,
    'gamma': float,
    'theta': float,
    'vega': float,
    'rho': float,
    'volume': int,
    'open_interest': int
}
```

## Constants

```python
from backtester.analytics import (
    TRADING_DAYS_PER_YEAR,      # 252
    DEFAULT_RISK_FREE_RATE,     # 0.04
    VAR_CONFIDENCE_95,          # 0.95
    VAR_CONFIDENCE_99,          # 0.99
)

from backtester.core.option_structure import (
    GREEK_NAMES,  # ['delta', 'gamma', 'theta', 'vega', 'rho']
)

from backtester.strategies.strategy import (
    DEFAULT_MAX_POSITIONS,              # 5
    DEFAULT_MAX_TOTAL_DELTA,           # 100.0
    DEFAULT_MAX_TOTAL_VEGA,            # 1000.0
    DEFAULT_MAX_CAPITAL_UTILIZATION,   # 0.80
    DEFAULT_MARGIN_PER_CONTRACT,       # 5000.0
)
```

## Exceptions

```python
# Core
from backtester.core.option import OptionError, OptionValidationError
from backtester.core.option_structure import (
    StructureError, EmptyStructureError
)
from backtester.core.pricing import PricingError, InvalidParameterError

# Strategies
from backtester.strategies.strategy import (
    StrategyError, StrategyValidationError, PositionError,
    RiskLimitError, InsufficientCapitalError
)

# Engine
from backtester.engine.backtest_engine import (
    BacktestError, BacktestConfigError, BacktestExecutionError
)
from backtester.engine.data_stream import (
    DataStreamError, DataNotAvailableError
)
from backtester.engine.execution import (
    ExecutionError, PriceNotAvailableError
)

# Analytics
from backtester.analytics import (
    MetricsError, RiskAnalyticsError, VisualizationError,
    DashboardError, ReportError
)
```

## Type Hints

```python
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np

# Common type aliases used throughout codebase
PriceData = Dict[str, float]
GreeksDict = Dict[str, float]
TradeLogRow = Dict[str, Any]
```

## Complete Example

```python
from datetime import datetime
from backtester.strategies import ShortStraddleHighIVStrategy
from backtester.engine.backtest_engine import BacktestEngine
from backtester.engine.data_stream import DataStream
from backtester.engine.execution import ExecutionModel
from backtester.analytics import PerformanceMetrics, Visualization, Dashboard
from backtester.data.dolt_adapter import DoltAdapter

# Setup
adapter = DoltAdapter(database='options_data')
data_stream = DataStream(adapter, datetime(2024, 1, 1),
                         datetime(2024, 12, 31), 'SPY')
execution = ExecutionModel(commission_per_contract=0.65)

# Create strategy
strategy = ShortStraddleHighIVStrategy(
    name='High IV Straddle',
    initial_capital=100000.0,
    iv_rank_threshold=70
)

# Run backtest
engine = BacktestEngine(strategy, data_stream, execution, 100000.0)
results = engine.run()

# Calculate metrics
equity_curve = results['equity_curve']
returns = equity_curve['equity'].pct_change().dropna()

metrics = {
    'sharpe': PerformanceMetrics.calculate_sharpe_ratio(returns),
    'max_dd': PerformanceMetrics.calculate_max_drawdown(equity_curve),
    'win_rate': PerformanceMetrics.calculate_win_rate(results['trade_log'])
}

# Visualize
Visualization.plot_equity_curve(equity_curve, save_path='equity.png')
Dashboard.create_performance_dashboard(results, metrics, 'dashboard.html')

print(f"Sharpe: {metrics['sharpe']:.2f}")
print(f"Max DD: {metrics['max_dd']*100:.2f}%")
```
