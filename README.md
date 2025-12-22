# Options Backtesting System

Professional-grade options strategy backtesting framework with accurate pricing, comprehensive analytics, and institutional-quality reporting.

## Features

- **Accurate Options Pricing:** Black-Scholes model with full Greeks calculations (delta, gamma, theta, vega, rho)
- **Flexible Strategy Framework:** Easy-to-extend base classes for custom strategies
- **Comprehensive Analytics:** 30+ performance and risk metrics (Sharpe, Sortino, Calmar, VaR, CVaR, drawdowns, etc.)
- **Professional Reporting:** Interactive HTML dashboards and detailed multi-page reports
- **Data Integration:** Built-in support for Dolt databases and external data sources
- **10 Pre-built Structures:** Straddles, strangles, spreads, condors, butterflies
- **3 Example Strategies:** High IV straddle, iron condor, volatility regime
- **Production-Ready:** 730+ comprehensive tests, full type hints, extensive error handling

## Quick Start

### Installation

```bash
cd /Users/janussuk/Desktop/OptionsBacktester2/code
pip install -r requirements.txt
```

### Run a Simple Backtest

```python
from datetime import datetime
from backtester.strategies import ShortStraddleHighIVStrategy
from backtester.engine.backtest_engine import BacktestEngine
from backtester.engine.data_stream import DataStream
from backtester.engine.execution import ExecutionModel
from backtester.data.dolt_adapter import DoltAdapter

# Create strategy
strategy = ShortStraddleHighIVStrategy(
    name='High IV Straddle',
    initial_capital=100000.0,
    iv_rank_threshold=70,
    profit_target_pct=0.50
)

# Setup data
adapter = DoltAdapter(database='options_data')
data_stream = DataStream(adapter, datetime(2024, 1, 1), datetime(2024, 12, 31), 'SPY')
execution = ExecutionModel(commission_per_contract=0.65)

# Run backtest
engine = BacktestEngine(strategy, data_stream, execution, 100000.0)
results = engine.run()

# Display results
print(f"Final Equity: ${results['final_equity']:,.2f}")
print(f"Total Return: {results['total_return']*100:.2f}%")
```

### Run Examples

```bash
cd /Users/janussuk/Desktop/OptionsBacktester2/examples
python example_01_simple_backtest.py
```

## Project Structure

```
OptionsBacktester2/
├── code/
│   ├── backtester/              # Main package
│   │   ├── core/                # Core classes (Option, OptionStructure, Pricing)
│   │   ├── strategies/          # Strategy framework and implementations
│   │   ├── structures/          # Pre-built option structures
│   │   ├── engine/              # Backtest engine, data stream, execution
│   │   ├── analytics/           # Metrics, visualizations, reports
│   │   ├── data/                # Data adapters and market data classes
│   │   └── utils/               # Utilities and conditions
│   ├── tests/                   # 730+ comprehensive tests
│   └── setup.py                 # Package installation
├── examples/                    # 5 complete example scripts
├── docs/                        # Comprehensive documentation
├── notebooks/                   # Jupyter notebooks
└── README.md                    # This file
```

## Core Components

### 1. Option & OptionStructure

```python
from backtester.core.option import Option
from backtester.structures import ShortStraddle

# Single option
option = Option(
    option_type='call', position_type='long',
    underlying='SPY', strike=450.0,
    expiration=datetime(2024, 3, 15),
    quantity=10, entry_price=5.50,
    entry_date=datetime(2024, 1, 15),
    underlying_price_at_entry=448.0,
    implied_vol_at_entry=0.20
)

# Multi-leg structure
straddle = ShortStraddle.create(
    underlying='SPY', strike=450.0,
    expiration=datetime(2024, 3, 15),
    call_price=10.0, put_price=9.5,
    quantity=10, entry_date=datetime(2024, 1, 15),
    underlying_price=450.0
)
```

### 2. Strategies

**Built-in Strategies:**
- `ShortStraddleHighIVStrategy` - Sell ATM straddles when IV rank is high
- `IronCondorStrategy` - Delta-based iron condor with defined risk
- `VolatilityRegimeStrategy` - Adaptive strategy based on VIX levels

**Create Custom Strategy:**

```python
from backtester.strategies import Strategy

class MyStrategy(Strategy):
    def should_enter(self, current_date, market_data, option_chain, available_capital):
        # Your entry logic
        return True

    def should_exit(self, position, current_date, market_data, option_chain):
        # Your exit logic
        return False

    def create_structure(self, current_date, market_data, option_chain, available_capital):
        # Create your structure
        return some_option_structure
```

### 3. Analytics

```python
from backtester.analytics import PerformanceMetrics, RiskAnalytics, Visualization

# Performance metrics
sharpe = PerformanceMetrics.calculate_sharpe_ratio(returns)
max_dd = PerformanceMetrics.calculate_max_drawdown(equity_curve)
win_rate = PerformanceMetrics.calculate_win_rate(trade_log)

# Risk metrics
var_95 = RiskAnalytics.calculate_var(returns, 0.95)
cvar_95 = RiskAnalytics.calculate_cvar(returns, 0.95)

# Visualizations
Visualization.plot_equity_curve(equity_curve, save_path='equity.png')
Visualization.plot_drawdown(equity_curve, save_path='drawdown.png')
```

### 4. Dashboards & Reports

```python
from backtester.analytics import Dashboard, ReportGenerator

# Interactive dashboard
Dashboard.create_performance_dashboard(results, metrics, 'dashboard.html')

# Comprehensive report
ReportGenerator.generate_html_report(results, metrics, 'report.html')
```

## Available Option Structures

- **Straddles:** `LongStraddle`, `ShortStraddle`
- **Strangles:** `LongStrangle`, `ShortStrangle`
- **Vertical Spreads:** `BullCallSpread`, `BearPutSpread`, `BullPutSpread`, `BearCallSpread`
- **Condors:** `IronCondor`, `IronButterfly`

All structures provide:
- Automatic Greeks aggregation
- Max profit/loss calculations
- Breakeven prices
- P&L tracking

## Performance Metrics

**Returns-Based:**
- Total Return, Annualized Return (CAGR)
- Sharpe Ratio, Sortino Ratio, Calmar Ratio
- Maximum Drawdown, Drawdown Duration, Recovery Time

**Trade-Based:**
- Win Rate, Profit Factor, Expectancy, Payoff Ratio
- Average Win/Loss, Consecutive Wins/Losses

**Risk Metrics:**
- Value at Risk (VaR 95%, 99%)
- Conditional VaR (CVaR/Expected Shortfall)
- Tail Risk (Skewness, Kurtosis, Tail Ratios)
- Greeks Analysis (Portfolio Delta, Gamma, Theta, Vega exposure over time)

## Examples

Five complete example scripts demonstrate different aspects:

1. **`example_01_simple_backtest.py`** - Basic workflow (Beginner)
2. **`example_02_iron_condor_backtest.py`** - Iron condor with dashboard (Intermediate)
3. **`example_03_volatility_regime_backtest.py`** - Adaptive strategy with reports (Advanced)
4. **`example_04_custom_strategy.py`** - Custom strategy template (Advanced)
5. **`example_05_full_analysis.py`** - Complete workflow with all features (Advanced)

## Documentation

Comprehensive documentation in `/docs`:

- **`USER_GUIDE.md`** - Installation, quick start, core concepts, common workflows
- **`API_REFERENCE.md`** - Complete API documentation for all classes and methods
- **`STRATEGY_DEVELOPMENT_GUIDE.md`** - How to create custom strategies with best practices
- **`ANALYTICS_GUIDE.md`** - Explanation of all 30+ metrics with interpretation guidelines
- **`DATA_INTEGRATION_GUIDE.md`** - Working with Dolt, CSV, APIs, and data quality

## Testing

Comprehensive test suite with 730+ tests:

```bash
cd /Users/janussuk/Desktop/OptionsBacktester2/code
pytest tests/ -v
```

**Test Coverage:**
- Unit tests for all core components
- Integration tests for end-to-end workflows
- Performance benchmarks
- Financial correctness validation
- Edge case handling

**Test Categories:**
- `test_option.py` - Option class (80+ tests)
- `test_option_structure.py` - OptionStructure base (70+ tests)
- `test_concrete_structures.py` - All 10 structures (100+ tests)
- `test_pricing.py` - Black-Scholes pricing (50+ tests)
- `test_strategy.py` - Strategy framework (80+ tests)
- `test_example_strategies.py` - Built-in strategies (90+ tests)
- `test_engine.py` - Backtest engine (100+ tests)
- `test_analytics.py` - Metrics and visualizations (150+ tests)
- `test_integration.py` - End-to-end integration (40+ tests)
- `test_performance.py` - Performance benchmarks (15+ tests)
- `test_system_validation.py` - Financial correctness (30+ tests)

## Requirements

- Python 3.9+
- NumPy >= 2.0.2
- pandas >= 2.3.3
- SciPy >= 1.13.1
- Matplotlib >= 3.9.4
- Plotly >= 6.5.0
- Seaborn >= 0.13.2
- Doltpy >= 2.0.0 (optional, for database integration)
- pytest >= 7.4.0 (for testing)

## Installation from Source

```bash
cd /Users/janussuk/Desktop/OptionsBacktester2/code
pip install -e .
```

## Financial Accuracy

The system implements industry-standard financial models:

- **Black-Scholes Pricing:** Exact implementation with numerical stability
- **Greeks Calculations:** Analytical formulas for all first-order and second-order Greeks
- **Put-Call Parity:** Validated across all pricing operations
- **P&L Calculations:** Consistent mark-to-market valuation
- **Max Profit/Loss:** Correct formulas for all structure types
- **Breakevens:** Accurate calculation for complex multi-leg structures

All calculations validated against:
- Closed-form solutions
- Numerical methods (finite differences)
- Industry standard values
- Academic references (Hull, McDonald, Jansen)

## Performance

System performance benchmarks:

- **Pricing Speed:** >10,000 options/second
- **Structure Creation:** <10ms per structure
- **Greeks Calculation:** <2ms per option
- **Backtest Speed:** >1000 data points/second

## Development Status

**Version:** 1.0.0
**Status:** Production-Ready Beta

**Completed:**
- Core option pricing engine
- 10 pre-built option structures
- Strategy framework with 3 example strategies
- Complete backtest engine
- 30+ analytics metrics
- Visualization and reporting
- 730+ comprehensive tests
- Full documentation
- Example scripts

**Future Enhancements:**
- Additional option structures (calendars, diagonals)
- More built-in strategies
- Machine learning integration
- Real-time data feeds
- Portfolio optimization
- CLI tools
- Web interface

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Support

- **Documentation:** `/docs` directory
- **Examples:** `/examples` directory
- **Tests:** `/code/tests` directory for usage patterns
- **Issues:** GitHub Issues (if publicly hosted)

## Authors

Options Backtester Team

## Acknowledgments

Built with:
- NumPy for numerical computations
- pandas for data manipulation
- SciPy for statistical functions
- Matplotlib/Plotly for visualizations
- Dolt for version-controlled database

Inspired by industry-standard quantitative finance practices and academic research in options pricing and risk management.

## Citation

If you use this system in research, please cite:

```
@software{options_backtester_2025,
  title={Options Backtesting System: Professional Options Strategy Backtesting Framework},
  author={Options Backtester Team},
  year={2025},
  version={1.0.0},
  url={https://github.com/yourusername/options-backtester}
}
```

---

**Note:** This system is for educational and research purposes. Always validate strategies with paper trading before using real capital. Past performance does not guarantee future results.
