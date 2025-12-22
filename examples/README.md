# Options Backtesting System - Examples

This directory contains example scripts demonstrating how to use the Options Backtesting System.

## Prerequisites

Before running these examples, ensure you have:

1. Installed all dependencies:
   ```bash
   cd /Users/janussuk/Desktop/OptionsBacktester2/code
   pip install -r requirements.txt
   ```

2. The backtester package is accessible (examples handle path setup automatically)

## Examples Overview

### Example 1: Simple Backtest (`example_01_simple_backtest.py`)

**Difficulty:** Beginner
**Time:** < 10 seconds
**Demonstrates:**
- Basic backtest workflow
- Short Straddle strategy
- Mock data creation
- Result analysis

**Run:**
```bash
cd /Users/janussuk/Desktop/OptionsBacktester2/examples
python example_01_simple_backtest.py
```

**Output:**
- Console results with performance metrics
- Trade statistics
- Greeks summary

**Learn:**
This is the best starting point. It shows the minimal code needed to run a backtest and understand results.

---

### Example 2: Iron Condor Backtest (`example_02_iron_condor_backtest.py`)

**Difficulty:** Intermediate
**Time:** ~15 seconds
**Demonstrates:**
- Iron Condor strategy with delta-based strike selection
- Advanced performance metrics (Sharpe, Sortino, VaR, CVaR)
- Dashboard generation

**Run:**
```bash
python example_02_iron_condor_backtest.py
```

**Output:**
- Console results with risk metrics
- Interactive HTML dashboard (opens in browser)

**Learn:**
- How to use different strategy types
- Risk metrics calculation
- Dashboard visualization

---

### Example 3: Volatility Regime Strategy (`example_03_volatility_regime_backtest.py`)

**Difficulty:** Advanced
**Time:** ~20 seconds
**Demonstrates:**
- Adaptive strategy based on volatility regimes
- Changing IV environments
- Comprehensive HTML report generation

**Run:**
```bash
python example_03_volatility_regime_backtest.py
```

**Output:**
- Console results
- Comprehensive HTML report with charts and analysis

**Learn:**
- Adaptive strategy implementation
- Volatility regime detection
- Report generation

---

### Example 4: Custom Strategy Template (`example_04_custom_strategy.py`)

**Difficulty:** Advanced
**Time:** ~15 seconds
**Demonstrates:**
- Creating custom strategies by inheriting from Strategy base class
- Mean reversion logic
- Custom entry/exit conditions
- Dynamic position sizing

**Run:**
```bash
python example_04_custom_strategy.py
```

**Output:**
- Console results from custom strategy

**Learn:**
- How to create your own trading strategies
- Implementing custom entry logic (mean reversion example)
- Implementing custom exit logic
- Position sizing based on risk

**Use as Template:**
This example serves as a template for creating your own strategies. Copy and modify the `CustomMeanReversionStrategy` class to implement your own ideas.

---

### Example 5: Complete Analysis Workflow (`example_05_full_analysis.py`)

**Difficulty:** Advanced
**Time:** ~30 seconds
**Demonstrates:**
- Running multiple strategies for comparison
- Calculating all available metrics
- Generating all visualization types
- Creating dashboards and reports
- Complete workflow from backtest to deliverables

**Run:**
```bash
python example_05_full_analysis.py
```

**Output:**
- Console comparison table
- PNG charts (equity curves, drawdown, P&L distribution)
- Interactive HTML dashboards for each strategy
- Comprehensive HTML reports

**Files Generated:**
- `{strategy}_equity.png` - Equity curve chart
- `{strategy}_drawdown.png` - Drawdown over time
- `{strategy}_pnl_dist.png` - P&L distribution histogram
- `{strategy}_dashboard.html` - Interactive performance dashboard
- `{strategy}_report.html` - Multi-page comprehensive report

**Learn:**
- Complete end-to-end workflow
- Strategy comparison
- Professional report generation
- All visualization capabilities

---

## Common Patterns

### Basic Backtest Structure

All examples follow this pattern:

```python
# 1. Import components
from backtester.strategies import SomeStrategy
from backtester.engine.backtest_engine import BacktestEngine
from backtester.engine.data_stream import DataStream
from backtester.engine.execution import ExecutionModel
from backtester.analytics import PerformanceMetrics

# 2. Create strategy
strategy = SomeStrategy(
    name='My Strategy',
    initial_capital=100000.0,
    # strategy-specific parameters...
)

# 3. Setup data stream
data_stream = setup_data_stream(start_date, end_date)

# 4. Create execution model
execution_model = ExecutionModel(
    commission_per_contract=0.65,
    slippage_pct=0.01,
    fill_on='mid'
)

# 5. Create and run backtest engine
engine = BacktestEngine(
    strategy=strategy,
    data_stream=data_stream,
    execution_model=execution_model,
    initial_capital=100000.0
)

results = engine.run()

# 6. Analyze results
print(f"Final Equity: ${results['final_equity']:,.2f}")
```

### Mock Data Creation

These examples use mock data for demonstration. In production:

```python
# Replace mock data with real data:
from backtester.data.dolt_adapter import DoltAdapter

# Connect to actual database
adapter = DoltAdapter(
    database='options_data',
    host='localhost',
    port=3306
)

# Create real data stream
data_stream = DataStream(
    adapter=adapter,
    start_date=start_date,
    end_date=end_date,
    underlying='SPY'
)
```

### Metrics Calculation

```python
# Calculate common metrics
equity_curve = results['equity_curve']
trade_log = results['trade_log']
returns = equity_curve['equity'].pct_change().dropna()

# Performance
total_return = PerformanceMetrics.calculate_total_return(equity_curve)
sharpe = PerformanceMetrics.calculate_sharpe_ratio(returns)
max_dd = PerformanceMetrics.calculate_max_drawdown(equity_curve)

# Risk
var_95 = RiskAnalytics.calculate_var(returns, 0.95)
cvar_95 = RiskAnalytics.calculate_cvar(returns, 0.95)

# Trades
win_rate = PerformanceMetrics.calculate_win_rate(trade_log)
profit_factor = PerformanceMetrics.calculate_profit_factor(trade_log)
```

### Visualization

```python
# Create charts
Visualization.plot_equity_curve(equity_curve, save_path='equity.png')
Visualization.plot_drawdown(equity_curve, save_path='drawdown.png')
Visualization.plot_pnl_distribution(trade_log, save_path='pnl.png')

# Create dashboard
Dashboard.create_performance_dashboard(
    results=results,
    metrics=metrics,
    save_path='dashboard.html'
)

# Create report
ReportGenerator.generate_html_report(
    results=results,
    metrics=metrics,
    save_path='report.html'
)
```

## Troubleshooting

### Import Errors

If you get import errors:
```bash
# Ensure you're running from the examples directory
cd /Users/janussuk/Desktop/OptionsBacktester2/examples

# Or set PYTHONPATH
export PYTHONPATH="/Users/janussuk/Desktop/OptionsBacktester2/code:$PYTHONPATH"
```

### No Trades Generated

If a backtest generates no trades:
- Lower the `iv_rank_threshold` parameter
- Increase the date range
- Adjust entry criteria parameters
- Check that mock data is being generated correctly

### Visualization Errors

If visualizations fail to generate:
```bash
# Install visualization dependencies
pip install matplotlib plotly seaborn
```

## Next Steps

After running these examples:

1. **Modify Parameters:** Adjust strategy parameters to see how results change
2. **Create Custom Strategy:** Use Example 4 as a template
3. **Use Real Data:** Replace mock data with your database connection
4. **Run Longer Backtests:** Extend date ranges for more comprehensive results
5. **Compare Strategies:** Use Example 5 pattern to compare multiple approaches

## Documentation

For more detailed information:
- User Guide: `/Users/janussuk/Desktop/OptionsBacktester2/docs/USER_GUIDE.md`
- API Reference: `/Users/janussuk/Desktop/OptionsBacktester2/docs/API_REFERENCE.md`
- Strategy Development Guide: `/Users/janussuk/Desktop/OptionsBacktester2/docs/STRATEGY_DEVELOPMENT_GUIDE.md`

## Support

For questions or issues:
1. Check the documentation in `/docs`
2. Review test files in `/code/tests` for more examples
3. Examine the source code in `/code/backtester`

## License

See main project README for license information.
