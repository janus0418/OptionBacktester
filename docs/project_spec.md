# Project Specification

## Project Requirement

You are a quantitative trader and developer specialized in systematic options strategies with 2 major goals:
- Develop and maintain the options backtester, making it as efficient, accurate, and error free as possible
- When given strategy ideas, research, implement, and backtest the strategy using the backtester

### Project Objectives

1. **Accurate Options Pricing**: Implement Black-Scholes model with full Greeks calculations (delta, gamma, theta, vega, rho)
2. **Flexible Strategy Framework**: Provide easy-to-extend base classes for custom strategy implementations
3. **Comprehensive Analytics**: Deliver 30+ performance and risk metrics (Sharpe, Sortino, Calmar, VaR, etc.)
4. **Professional Reporting**: Generate interactive dashboards and detailed HTML/PDF reports
5. **Data Integration**: Support Dolt databases, CSV files, and external API data sources
6. **Realistic Execution**: Model bid/ask spreads, commissions, and slippage

---

# Engineering Requirement

## Tech Stack

| Component | Technology | Version |
|-----------|------------|---------|
| **Language** | Python | >= 3.9 |
| **Data Processing** | NumPy, Pandas | >= 2.0.2, >= 2.3.3 |
| **Scientific Computing** | SciPy | >= 1.13.1 |
| **Visualization** | Matplotlib, Plotly, Seaborn | >= 3.9.4, >= 6.5.0, >= 0.13.2 |
| **Database** | Dolt (via doltpy) | >= 2.0.0 |
| **Performance** | Numba (optional) | >= 0.58.0 |
| **Testing** | pytest | >= 7.4.0 |
| **Code Quality** | Black, Flake8 | >= 23.0.0, >= 6.0.0 |
| **Documentation** | Jupyter, Notebook | >= 1.0.0, >= 7.0.0 |

## Required Libraries

### Core Dependencies
```
numpy>=2.0.2          # Numerical computations, array operations
pandas>=2.3.3         # DataFrames, time series, data manipulation
scipy>=1.13.1         # Statistical functions, distributions, optimization
```

### Visualization Dependencies
```
matplotlib>=3.9.4     # Static plots, chart generation
plotly>=6.5.0         # Interactive visualizations, dashboards
seaborn>=0.13.2       # Statistical plotting, heatmaps
```

### Data Dependencies
```
doltpy>=2.0.0         # Dolt database integration
pyyaml>=6.0           # Configuration file parsing
```

### Development Dependencies
```
pytest>=7.4.0         # Testing framework
pytest-cov>=4.0.0     # Coverage reporting
pytest-benchmark>=4.0.0  # Performance benchmarking
black>=23.0.0         # Code formatting
flake8>=6.0.0         # Linting
mypy>=1.0.0           # Type checking
jupyter>=1.0.0        # Interactive development
notebook>=7.0.0       # Notebook support
ipykernel>=6.25.0     # Jupyter kernel
```

### Optional Performance Dependencies
```
numba>=0.58.0         # JIT compilation for numerical functions
```

## Project Structure

```
OptionsBacktester2/
├── CLAUDE.md                    # Project context for Claude Code
├── README.md                    # Project overview
├── docs/                        # Documentation
│   ├── project_spec.md         # This file - requirements specification
│   ├── architecture.md         # System design and data flow
│   ├── changelog.md            # Version history
│   ├── project_status.md       # Current progress
│   ├── USER_GUIDE.md           # User documentation
│   ├── STRATEGY_DEVELOPMENT_GUIDE.md
│   ├── ANALYTICS_GUIDE.md
│   ├── DATA_INTEGRATION_GUIDE.md
│   └── API_REFERENCE.md
├── code/                        # Source code
│   ├── requirements.txt        # Python dependencies
│   ├── setup.py                # Package configuration
│   ├── backtester/             # Main package
│   │   ├── __init__.py
│   │   ├── core/               # Core data structures
│   │   │   ├── option.py       # Option class
│   │   │   ├── option_structure.py  # Multi-leg positions
│   │   │   └── pricing.py      # Black-Scholes model
│   │   ├── engine/             # Backtesting engine
│   │   │   ├── backtest_engine.py   # Main orchestrator
│   │   │   ├── data_stream.py       # Data iteration
│   │   │   ├── execution.py         # Order execution
│   │   │   └── position_manager.py  # Position tracking
│   │   ├── strategies/         # Strategy implementations
│   │   │   ├── strategy.py     # Abstract base class
│   │   │   ├── short_straddle_strategy.py
│   │   │   ├── iron_condor_strategy.py
│   │   │   └── volatility_regime_strategy.py
│   │   ├── structures/         # Option structure factories
│   │   │   ├── straddle.py
│   │   │   ├── strangle.py
│   │   │   ├── spread.py
│   │   │   └── condor.py
│   │   ├── analytics/          # Performance analysis
│   │   │   ├── metrics.py      # Performance metrics
│   │   │   ├── risk.py         # Risk analytics
│   │   │   ├── visualization.py
│   │   │   ├── dashboard.py
│   │   │   └── report.py
│   │   ├── data/               # Data access layer
│   │   │   ├── dolt_adapter.py
│   │   │   ├── market_data.py
│   │   │   └── data_validator.py
│   │   └── utils/              # Utilities
│   │       └── conditions.py
│   └── tests/                  # Test suite (730+ tests)
│       ├── test_option.py
│       ├── test_pricing.py
│       ├── test_option_structure.py
│       ├── test_strategy.py
│       ├── test_engine.py
│       ├── test_analytics.py
│       ├── test_visualization.py
│       ├── test_concrete_structures.py
│       ├── test_example_strategies.py
│       ├── test_system_validation.py
│       ├── test_performance.py
│       └── test_integration.py
├── examples/                    # Example scripts
│   ├── example_01_simple_backtest.py
│   ├── example_02_iron_condor_backtest.py
│   ├── example_03_volatility_regime_backtest.py
│   ├── example_04_custom_strategy.py
│   └── example_05_full_analysis.py
├── knowledgeBase/               # Strategy research and notes
└── dolt_data/                   # Local Dolt database
```

## Python Version

- **Minimum**: Python 3.9
- **Supported**: Python 3.9, 3.10, 3.11, 3.12
- **Development**: Python 3.9+ recommended

---

# Feature Requirements

## Core Features

### 1. Option Modeling
| Feature | Description | Status |
|---------|-------------|--------|
| Option Class | Single option contract representation | Complete |
| Greeks Calculation | Delta, gamma, theta, vega, rho | Complete |
| Black-Scholes Pricing | European option pricing model | Complete |
| Position Types | Long/short calls and puts | Complete |
| Contract Multiplier | Standard 100x multiplier | Complete |

### 2. Option Structures
| Structure | Description | Status |
|-----------|-------------|--------|
| Straddle | Long/Short ATM call + put | Complete |
| Strangle | Long/Short OTM call + put | Complete |
| Vertical Spread | Bull/Bear call/put spreads | Complete |
| Iron Condor | Short strangle + long wings | Complete |
| Iron Butterfly | ATM short straddle + wings | Complete |
| Custom | User-defined multi-leg | Complete |

### 3. Backtesting Engine
| Feature | Description | Status |
|---------|-------------|--------|
| Event Loop | Clean timestep iteration | Complete |
| Entry/Exit Signals | Strategy-driven decisions | Complete |
| Position Management | Track open positions | Complete |
| Equity Tracking | Mark-to-market valuation | Complete |
| Greeks History | Portfolio Greeks over time | Complete |
| Trade Logging | Complete trade records | Complete |

### 4. Execution Modeling
| Feature | Description | Default |
|---------|-------------|---------|
| Commission | Per-contract fees | $0.65 |
| Slippage | Percentage of mid price | 1% |
| Bid/Ask Spread | Realistic fill prices | Enabled |
| Fill Simulation | Entry at ask, exit at bid | Enabled |

## Strategy Features

### Abstract Strategy Interface
```python
class Strategy(ABC):
    @abstractmethod
    def should_enter(self, market_data: Dict) -> bool:
        """Determine if entry conditions are met."""
        pass

    @abstractmethod
    def should_exit(self, structure: OptionStructure, market_data: Dict) -> bool:
        """Determine if exit conditions are met."""
        pass

    def create_structure(self, market_data: Dict) -> Optional[OptionStructure]:
        """Create option structure to enter."""
        pass
```

### Built-in Strategies
| Strategy | Entry Trigger | Exit Conditions |
|----------|---------------|-----------------|
| ShortStraddleHighIV | IV rank > threshold | Profit target, stop loss, DTE |
| IronCondor | IV conditions | Profit target, stop loss, delta breach |
| VolatilityRegime | Regime detection | Regime change, time exit |

### Risk Limits
```python
position_limits = {
    'max_positions': 10,           # Max concurrent positions
    'max_delta': 100,              # Portfolio delta limit
    'max_gamma': 50,               # Portfolio gamma limit
    'max_vega': 500,               # Portfolio vega limit
    'max_capital_per_trade': 0.10  # 10% per trade
}
```

## Analytics Features

### Performance Metrics (30+)
| Category | Metrics |
|----------|---------|
| **Returns** | Total return, CAGR, annualized return |
| **Risk-Adjusted** | Sharpe ratio, Sortino ratio, Calmar ratio |
| **Drawdown** | Max drawdown, duration, recovery time, Ulcer Index |
| **Trade** | Win rate, profit factor, expectancy, payoff ratio |
| **Distribution** | Mean, std, skewness, kurtosis, percentiles |

### Risk Metrics
| Metric | Description |
|--------|-------------|
| VaR (95%, 99%) | Value at Risk (historical & parametric) |
| CVaR / ES | Conditional VaR / Expected Shortfall |
| Tail Risk | Skewness, kurtosis analysis |
| Downside Deviation | Volatility of negative returns |
| Greeks Exposure | Delta, gamma, theta, vega over time |

### Visualization
| Chart Type | Description |
|------------|-------------|
| Equity Curve | Portfolio value over time |
| Drawdown | Peak-to-trough decline |
| P&L Distribution | Histogram of trade returns |
| Monthly Returns | Heatmap by year/month |
| Greeks Over Time | Line charts of exposure |
| Trade Scatter | Entry/exit visualization |

### Reporting
| Output | Format |
|--------|--------|
| Interactive Dashboard | HTML (Plotly) |
| Static Report | HTML, PDF |
| Metrics Summary | JSON, CSV |

## Data Integration Features

### Supported Data Sources
| Source | Type | Support |
|--------|------|---------|
| Dolt Database | SQL | Full |
| CSV Files | Flat file | Full |
| Mock Data | Generated | Full |
| External APIs | REST | Template |

### Required Data Schema

**Market Data:**
```sql
symbol, date, open, high, low, close, volume
```

**Options Data:**
```sql
symbol, quote_date, expiration, strike, option_type,
bid, ask, last, volume, open_interest,
implied_volatility, delta, gamma, theta, vega, rho
```

---

# Code Quality Standards

## Testing Requirements

- **Minimum Coverage**: 80%
- **Test Types**: Unit, integration, system validation, performance
- **Test Count**: 730+ tests
- **Test Command**: `pytest tests/ -v`

## Code Style

- **Formatter**: Black (line length 88)
- **Linter**: Flake8
- **Type Hints**: Required for public APIs
- **Docstrings**: Google style required

## Documentation Requirements

- All public classes and methods must have docstrings
- Module-level docstrings explaining purpose
- Type hints for all function signatures
- Usage examples in docstrings

---

# Design Principles

1. **Separation of Concerns**
   - Each module has a single responsibility
   - Clear boundaries between layers

2. **Event-Driven Architecture**
   - Clean timestep-based simulation
   - Predictable execution order

3. **Extensibility**
   - Abstract base classes for customization
   - Plugin-style strategy system

4. **Financial Correctness**
   - Proper handling of contract multipliers
   - Accurate P&L and Greeks calculations
   - Realistic execution simulation

5. **Comprehensive Analytics**
   - Industry-standard metrics
   - Professional-grade reporting

---

# Performance Requirements

## Backtest Speed
| Metric | Target |
|--------|--------|
| Single day processing | < 10ms |
| 1 year backtest (252 days) | < 5 seconds |
| Full analytics calculation | < 2 seconds |

## Memory Usage
| Data Size | Max Memory |
|-----------|------------|
| 1 year daily data | < 500 MB |
| 5 years daily data | < 2 GB |
| Greeks history tracking | Proportional to days |

## Optimization Strategies
- Vectorized calculations with NumPy
- Efficient data structures (slots, typed containers)
- Optional Numba JIT compilation
- Lazy evaluation where appropriate

---

# Development Workflow

## Git Workflow

1. **Branch Creation**: `git checkout -b feature/description`
2. **Development**: Commit frequently with clear messages
3. **Testing**: Run full test suite before push
4. **Push**: `git push -u origin feature/description`
5. **PR**: Create pull request to merge into `main`
6. **Documentation**: Use `/update-docs-and-commit` command

## Commit Guidelines

- Use conventional commit format: `type: description`
  - `feat:` New feature
  - `fix:` Bug fix
  - `docs:` Documentation
  - `refactor:` Code refactoring
  - `test:` Test additions
- Keep commits focused on single changes
- Reference issues/PRs where applicable

## Testing Before Commit

1. Run all tests: `pytest tests/ -v`
2. Verify P&L accuracy with simple trades
3. Check strategy behavior matches expectations
4. Document and fix any errors found

---

# API Quick Reference

## Core Usage
```python
from backtester.core.option import Option
from backtester.core.option_structure import OptionStructure
from backtester.strategies import ShortStraddleHighIVStrategy
from backtester.engine.backtest_engine import BacktestEngine
from backtester.engine.data_stream import DataStream
from backtester.engine.execution import ExecutionModel
from backtester.analytics import PerformanceMetrics, RiskAnalytics
```

## Typical Workflow
```python
# 1. Create strategy
strategy = ShortStraddleHighIVStrategy(
    name='My Strategy',
    initial_capital=100000,
    iv_rank_threshold=70
)

# 2. Setup data
data_stream = DataStream(adapter, start_date, end_date, 'SPY')

# 3. Configure execution
execution = ExecutionModel(commission_per_contract=0.65)

# 4. Run backtest
engine = BacktestEngine(strategy, data_stream, execution, 100000)
results = engine.run()

# 5. Analyze
metrics = engine.calculate_metrics()

# 6. Report
engine.create_dashboard('dashboard.html')
```

---

# Version Information

- **Current Version**: 1.0.0
- **Status**: Beta
- **License**: MIT
