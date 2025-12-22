# Options Backtesting System - User Guide

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Core Concepts](#core-concepts)
5. [Common Workflows](#common-workflows)
6. [Troubleshooting](#troubleshooting)

## Introduction

The Options Backtesting System is a professional-grade framework for backtesting options trading strategies. It provides:

- **Accurate Options Pricing:** Black-Scholes model with full Greeks calculations
- **Flexible Strategy Framework:** Easy-to-extend base classes for custom strategies
- **Comprehensive Analytics:** 30+ performance and risk metrics
- **Professional Reporting:** Interactive dashboards and detailed HTML/PDF reports
- **Data Integration:** Built-in support for Dolt databases and external data sources

## Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Install Dependencies

```bash
cd /Users/janussuk/Desktop/OptionsBacktester2/code
pip install -r requirements.txt
```

### Required Packages

- `numpy>=2.0.2` - Numerical computations
- `pandas>=2.3.3` - Data manipulation
- `scipy>=1.13.1` - Statistical functions
- `matplotlib>=3.9.4` - Static visualizations
- `plotly>=6.5.0` - Interactive visualizations
- `seaborn>=0.13.2` - Statistical plotting
- `doltpy>=2.0.0` - Database integration
- `pytest>=7.4.0` - Testing framework

### Verify Installation

```bash
cd /Users/janussuk/Desktop/OptionsBacktester2/code
pytest tests/ -v
```

All 730+ tests should pass.

## Quick Start

### Minimal Example

```python
from datetime import datetime
from backtester.strategies import ShortStraddleHighIVStrategy
from backtester.engine.backtest_engine import BacktestEngine
from backtester.engine.data_stream import DataStream
from backtester.engine.execution import ExecutionModel
from backtester.data.dolt_adapter import DoltAdapter

# Create strategy
strategy = ShortStraddleHighIVStrategy(
    name='My First Strategy',
    initial_capital=100000.0,
    iv_rank_threshold=70,
    profit_target_pct=0.50
)

# Setup data (use mock data for testing, real database for production)
adapter = DoltAdapter(database='options_data')
data_stream = DataStream(
    adapter=adapter,
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 12, 31),
    underlying='SPY'
)

# Create execution model
execution = ExecutionModel(commission_per_contract=0.65)

# Run backtest
engine = BacktestEngine(strategy, data_stream, execution, 100000.0)
results = engine.run()

# Display results
print(f"Final Equity: ${results['final_equity']:,.2f}")
print(f"Total Return: {results['total_return']*100:.2f}%")
```

### Run an Example

```bash
cd /Users/janussuk/Desktop/OptionsBacktester2/examples
python example_01_simple_backtest.py
```

See `/examples/README.md` for all available examples.

## Core Concepts

### 1. Option

The `Option` class represents a single option contract (call or put, long or short).

```python
from backtester.core.option import Option
from datetime import datetime

option = Option(
    option_type='call',        # 'call' or 'put'
    position_type='long',      # 'long' or 'short'
    underlying='SPY',
    strike=450.0,
    expiration=datetime(2024, 3, 15),
    quantity=10,               # Number of contracts
    entry_price=5.50,         # Price paid/received per contract
    entry_date=datetime(2024, 1, 15),
    underlying_price_at_entry=448.0,
    implied_vol_at_entry=0.20
)

# Access properties
print(f"Current Value: ${option.current_value:,.2f}")
print(f"P&L: ${option.pnl:,.2f}")
print(f"Delta: {option.greeks['delta']:.4f}")
```

**Key Properties:**
- `current_value`: Mark-to-market value
- `pnl`: Profit/loss since entry
- `greeks`: Dictionary with delta, gamma, theta, vega, rho
- `days_to_expiration`: Days until expiration

### 2. OptionStructure

The `OptionStructure` class represents multi-leg option positions (spreads, straddles, condors, etc.).

```python
from backtester.structures import ShortStraddle
from datetime import datetime

# Create using factory method
straddle = ShortStraddle.create(
    underlying='SPY',
    strike=450.0,
    expiration=datetime(2024, 3, 15),
    call_price=10.0,
    put_price=9.5,
    quantity=10,
    entry_date=datetime(2024, 1, 15),
    underlying_price=450.0
)

# Access structure properties
print(f"Max Profit: ${straddle.max_profit:,.2f}")
print(f"Max Loss: ${straddle.max_loss:,.2f}")
print(f"Breakevens: {straddle.lower_breakeven:.2f}, {straddle.upper_breakeven:.2f}")
print(f"Current P&L: ${straddle.pnl:,.2f}")

# Update prices
straddle.update_prices(
    current_date=datetime(2024, 2, 1),
    underlying_price=455.0,
    call_price=12.0,
    put_price=7.0,
    risk_free_rate=0.04
)
```

**Available Structures:**
- `LongStraddle`, `ShortStraddle`
- `LongStrangle`, `ShortStrangle`
- `BullCallSpread`, `BearPutSpread`, `BullPutSpread`, `BearCallSpread`
- `IronCondor`, `IronButterfly`

### 3. Strategy

The `Strategy` base class defines trading logic. You can use built-in strategies or create custom ones.

**Built-in Strategies:**

```python
from backtester.strategies import (
    ShortStraddleHighIVStrategy,
    IronCondorStrategy,
    VolatilityRegimeStrategy
)

# Short Straddle when IV is high
strategy = ShortStraddleHighIVStrategy(
    name='High IV Straddle',
    initial_capital=100000.0,
    iv_rank_threshold=70,      # Enter when IV rank > 70%
    profit_target_pct=0.50,    # Exit at 50% max profit
    stop_loss_pct=2.0,         # Stop at 200% max profit
    max_dte=45,                # Max days to expiration
    min_dte=7                  # Min days to expiration (roll/close)
)
```

**Custom Strategy:**

```python
from backtester.strategies import Strategy

class MyCustomStrategy(Strategy):
    def should_enter(self, current_date, market_data, option_chain, available_capital):
        """Return True if we should enter a new position."""
        # Your entry logic here
        return False

    def should_exit(self, position, current_date, market_data, option_chain):
        """Return True if we should exit this position."""
        # Your exit logic here
        return False

    def create_structure(self, current_date, market_data, option_chain, available_capital):
        """Create and return an OptionStructure."""
        # Your structure creation logic here
        return None
```

See `examples/example_04_custom_strategy.py` for a complete template.

### 4. BacktestEngine

The `BacktestEngine` orchestrates the backtest by coordinating the strategy, data, and execution.

```python
from backtester.engine.backtest_engine import BacktestEngine

engine = BacktestEngine(
    strategy=my_strategy,
    data_stream=data_stream,
    execution_model=execution_model,
    initial_capital=100000.0
)

# Run backtest
results = engine.run()

# Results dictionary contains:
results = {
    'final_equity': float,
    'total_return': float,
    'equity_curve': pd.DataFrame,
    'trade_log': pd.DataFrame,
    'greeks_history': pd.DataFrame,
    'positions_history': list
}
```

### 5. Analytics

Calculate performance and risk metrics:

```python
from backtester.analytics import PerformanceMetrics, RiskAnalytics

# Performance metrics
total_return = PerformanceMetrics.calculate_total_return(equity_curve)
sharpe_ratio = PerformanceMetrics.calculate_sharpe_ratio(returns)
max_drawdown = PerformanceMetrics.calculate_max_drawdown(equity_curve)
win_rate = PerformanceMetrics.calculate_win_rate(trade_log)

# Risk metrics
var_95 = RiskAnalytics.calculate_var(returns, 0.95)
cvar_95 = RiskAnalytics.calculate_cvar(returns, 0.95)
```

### 6. Visualization and Reporting

Generate charts, dashboards, and reports:

```python
from backtester.analytics import Visualization, Dashboard, ReportGenerator

# Create charts
Visualization.plot_equity_curve(equity_curve, save_path='equity.png')
Visualization.plot_drawdown(equity_curve, save_path='drawdown.png')

# Create interactive dashboard
Dashboard.create_performance_dashboard(
    results=results,
    metrics=metrics,
    save_path='dashboard.html'
)

# Generate comprehensive report
ReportGenerator.generate_html_report(
    results=results,
    metrics=metrics,
    save_path='report.html'
)
```

## Common Workflows

### Workflow 1: Run Single Strategy Backtest

```python
# 1. Create strategy
strategy = ShortStraddleHighIVStrategy(
    name='My Strategy',
    initial_capital=100000.0,
    iv_rank_threshold=70
)

# 2. Setup data
data_stream = setup_data_stream(start_date, end_date)

# 3. Run backtest
engine = BacktestEngine(strategy, data_stream, execution_model, 100000.0)
results = engine.run()

# 4. Analyze
print(f"Final Equity: ${results['final_equity']:,.2f}")
```

### Workflow 2: Compare Multiple Strategies

```python
strategies = [
    ShortStraddleHighIVStrategy(...),
    IronCondorStrategy(...),
    VolatilityRegimeStrategy(...)
]

for strategy in strategies:
    engine = BacktestEngine(strategy, data_stream, execution_model, 100000.0)
    results = engine.run()
    print(f"{strategy.name}: {results['total_return']*100:.2f}%")
```

### Workflow 3: Generate Comprehensive Report

```python
# Run backtest
results = engine.run()

# Calculate metrics
equity_curve = results['equity_curve']
returns = equity_curve['equity'].pct_change().dropna()

metrics = {
    'total_return': PerformanceMetrics.calculate_total_return(equity_curve),
    'sharpe_ratio': PerformanceMetrics.calculate_sharpe_ratio(returns),
    'max_drawdown': PerformanceMetrics.calculate_max_drawdown(equity_curve),
    # ... more metrics
}

# Generate visualizations
Visualization.plot_equity_curve(equity_curve, save_path='equity.png')
Visualization.plot_drawdown(equity_curve, save_path='drawdown.png')

# Generate dashboard and report
Dashboard.create_performance_dashboard(results, metrics, 'dashboard.html')
ReportGenerator.generate_html_report(results, metrics, 'report.html')
```

## Troubleshooting

### Common Issues

**Issue: "No module named 'backtester'"**

Solution:
```bash
# Add code directory to PYTHONPATH
export PYTHONPATH="/Users/janussuk/Desktop/OptionsBacktester2/code:$PYTHONPATH"

# Or install in development mode
cd /Users/janussuk/Desktop/OptionsBacktester2/code
pip install -e .
```

**Issue: "No trades generated during backtest"**

Causes:
- Entry criteria too restrictive
- Insufficient data
- Strategy parameters need adjustment

Solutions:
- Lower `iv_rank_threshold`
- Extend date range
- Add debug logging to `should_enter()`

**Issue: "Database connection failed"**

Solution:
- Verify Dolt is running: `dolt version`
- Check connection parameters
- Use mock data for testing (see examples)

**Issue: "Pricing errors or invalid Greeks"**

Causes:
- Negative time to expiration
- Invalid volatility
- Extreme parameters

Solution:
- Validate input data
- Check for edge cases in option chain
- Add data validation in strategy

### Performance Issues

**Slow backtest execution:**

- Reduce data frequency (use daily instead of intraday)
- Limit option chain size
- Use vectorized calculations where possible
- Profile code to identify bottlenecks

**High memory usage:**

- Process data in chunks
- Clear old positions from history
- Use appropriate data types (float32 vs float64)

### Getting Help

1. Check documentation in `/docs`
2. Review examples in `/examples`
3. Run tests to verify installation: `pytest tests/ -v`
4. Check test files for usage patterns
5. Review source code docstrings

## Next Steps

- **Run Examples:** Start with `example_01_simple_backtest.py`
- **Create Custom Strategy:** Use `example_04_custom_strategy.py` as template
- **Read API Reference:** See `API_REFERENCE.md` for detailed API documentation
- **Strategy Development:** See `STRATEGY_DEVELOPMENT_GUIDE.md` for best practices
- **Data Integration:** See `DATA_INTEGRATION_GUIDE.md` for data sources

## Additional Resources

- **Test Files:** `/code/tests/` - Extensive test coverage with usage examples
- **Source Code:** `/code/backtester/` - Well-documented implementation
- **Examples:** `/examples/` - Five complete example scripts

For detailed metric explanations, see `ANALYTICS_GUIDE.md`.
