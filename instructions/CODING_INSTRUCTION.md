# CODING_INSTRUCTION.md - Options Strategy Backtester Implementation Plan

## Executive Summary

This document provides a comprehensive implementation plan for building a Python-based options strategy backtester. The backtester will support complex options strategies through a hierarchical architecture: **Option** → **OptionStructure** → **Strategy** → **Backtesting Engine**. The implementation is designed to be built incrementally across multiple development runs to maximize context efficiency.

### Key Implementation Principles

1. **Agent-Driven Development**:
   - Use **quant-options-dev agent** for all financial/options code implementation
   - Use **code-quality-auditor agent** after each implementation for bug detection
   - Follow: Plan → Implement (with quant agent) → Audit (with quality agent) → Test → Document

2. **Core-First Approach**:
   - Build core libraries (Option, OptionStructure, Strategy, Engine) FIRST
   - Implement concrete examples (structures, strategies) ONLY AFTER core is complete
   - Prevents premature optimization and ensures solid foundation

3. **Data Schema Verification**:
   - **CRITICAL**: Verify actual Dolt database schema before implementing data layer
   - Do NOT assume schema structure
   - Document exact table/column names from actual database
   - Ensures all available data is properly utilized

4. **Incremental Validation**:
   - Each run includes comprehensive testing
   - Quality audits after each component
   - Build on validated components only

---

## 1. Project Architecture Overview

### 1.1 Core Philosophy

The backtester follows a **compositional design pattern** where simple building blocks combine to create complex strategies. Each component is self-contained, testable, and integrates seamlessly with others.

### 1.2 Technology Stack

- **Language**: Python 3.9+
- **Environment Management**: `uv` (fast package manager)
- **Database**: Dolt (version-controlled SQL database)
- **Data Source**: DoltHub repository `post-no-preference/options`
- **Core Libraries**: numpy, pandas, scipy, matplotlib, plotly, seaborn
- **Optional Performance**: numba (JIT compilation)

### 1.3 Agent Usage Guidelines

**CRITICAL**: This project uses specialized Claude Code agents for development and quality assurance.

#### 1.3.1 quant-options-dev Agent
**Purpose**: Develop quantitative options trading code with financial correctness

**Use this agent for**:
- Implementing pricing models (Black-Scholes, binomial, etc.)
- Greeks calculations (delta, gamma, theta, vega, rho)
- Option class and OptionStructure implementations
- Strategy logic and risk management
- Any financial mathematics or options-specific algorithms
- Performance-critical numerical computations

**When to invoke**:
- At the start of each implementation task involving options mathematics
- When implementing core financial logic
- Before writing any pricing or Greeks calculation code

#### 1.3.2 code-quality-auditor Agent
**Purpose**: Verify code quality, fix bugs, ensure optimal codebase integration

**Use this agent for**:
- Reviewing code after implementation
- Finding and fixing bugs
- Checking package dependencies and configurations
- Ensuring proper integration with existing codebase
- Identifying redundancies and optimization opportunities

**When to invoke**:
- IMMEDIATELY after completing any implementation task
- Before marking a run as complete
- After refactoring or significant code changes
- When encountering unexpected errors

#### 1.3.3 Implementation Workflow
For each development run:
1. **Plan**: Review context summaries and plan the implementation
2. **Implement**: Use **quant-options-dev agent** for core financial/options code
3. **Audit**: Use **code-quality-auditor agent** to review and fix issues
4. **Test**: Run all tests and verify functionality
5. **Document**: Update context summaries

### 1.4 Project Directory Structure

```
OptionsBacktester2/
├── code/
│   ├── backtester/
│   │   ├── __init__.py
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── option.py              # Option class
│   │   │   ├── option_structure.py    # OptionStructure class
│   │   │   ├── strategy.py            # Strategy class
│   │   │   └── pricing.py             # Pricing models (Black-Scholes, Greeks)
│   │   ├── structures/
│   │   │   ├── __init__.py
│   │   │   ├── base.py                # Base structure definitions
│   │   │   ├── straddle.py            # Long/Short Straddle
│   │   │   ├── strangle.py            # Long/Short Strangle
│   │   │   ├── spread.py              # Vertical/Calendar spreads
│   │   │   └── condor.py              # Iron Condor/Butterfly
│   │   ├── strategies/
│   │   │   ├── __init__.py
│   │   │   ├── examples/
│   │   │   │   ├── short_straddle_daily.py  # Based on knowledgeBase PDF
│   │   │   │   └── volatility_selling.py
│   │   ├── engine/
│   │   │   ├── __init__.py
│   │   │   ├── backtest_engine.py     # Main backtesting engine
│   │   │   ├── data_stream.py         # Time-series data streaming
│   │   │   ├── position_manager.py    # Track open positions
│   │   │   └── execution.py           # Order execution simulation
│   │   ├── data/
│   │   │   ├── __init__.py
│   │   │   ├── dolt_adapter.py        # Dolt database connector
│   │   │   ├── market_data.py         # Market data loader
│   │   │   └── data_validator.py      # Data quality checks
│   │   ├── analytics/
│   │   │   ├── __init__.py
│   │   │   ├── metrics.py             # Performance metrics
│   │   │   ├── risk.py                # Risk calculations (Greeks, VaR)
│   │   │   └── visualization.py       # Plotting utilities
│   │   └── utils/
│   │       ├── __init__.py
│   │       ├── config.py              # Configuration management
│   │       ├── logger.py              # Logging utilities
│   │       └── helpers.py             # General helper functions
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── test_option.py
│   │   ├── test_structures.py
│   │   ├── test_strategy.py
│   │   └── test_engine.py
│   ├── notebooks/
│   │   ├── 01_data_exploration.ipynb
│   │   ├── 02_option_pricing.ipynb
│   │   ├── 03_strategy_development.ipynb
│   │   └── 04_backtest_analysis.ipynb
│   └── examples/
│       ├── simple_backtest.py
│       └── advanced_backtest.py
├── contextSummary/
│   ├── ARCHITECTURE.md
│   ├── OPTION_CLASS.md
│   ├── STRUCTURE_CLASS.md
│   ├── STRATEGY_CLASS.md
│   ├── ENGINE_CLASS.md
│   ├── DATA_SCHEMA.md
│   └── INTEGRATION_GUIDE.md
├── instructions/
│   ├── CODING_INSTRUCTION.md (this file)
│   ├── inctruction_guideline.md
│   ├── Install.md
│   └── Using_Dolt.md
├── knowledgeBase/
│   └── Daily Short Straddle Strategy for SPY (ATM).pdf
├── .venv/                              # Virtual environment (created by uv)
├── pyproject.toml                      # Project configuration
├── requirements.txt                    # Dependency list
└── README.md                           # Project documentation
```

---

## 2. Component Specifications

### 2.1 Component 1: Option Class

**File**: `/Users/janussuk/Desktop/OptionsBacktester2/code/backtester/core/option.py`

**Purpose**: Represents a single option position (long call, short call, long put, short put)

**Key Attributes**:
```python
class Option:
    - option_type: str              # 'call' or 'put'
    - position_type: str            # 'long' or 'short'
    - underlying: str               # 'SPY', 'SPX', etc.
    - strike: float                 # Strike price
    - expiration: datetime          # Expiration date
    - quantity: int                 # Number of contracts
    - entry_price: float            # Premium paid/received at entry
    - entry_date: datetime          # Trade entry timestamp
    - underlying_price_at_entry: float  # Spot price at entry
    - implied_vol_at_entry: float   # IV at entry
    - greeks: dict                  # Delta, Gamma, Theta, Vega, Rho
    - current_price: float          # Current mark price (updated)
    - pnl: float                    # Current P&L
```

**Key Methods**:
```python
- calculate_greeks(spot, vol, rate, time_to_expiry) -> dict
- update_price(new_price, timestamp) -> None
- calculate_pnl() -> float
- get_payoff_at_expiry(spot_price) -> float
- is_itm(spot_price) -> bool
- is_atm(spot_price, threshold=0.02) -> bool
- get_intrinsic_value(spot_price) -> float
- get_time_value() -> float
- __repr__() -> str
```

**Dependencies**:
- Black-Scholes pricing model
- Greeks calculation (analytical formulas)
- scipy.stats for normal distribution

**Context Summary**: Store in `/Users/janussuk/Desktop/OptionsBacktester2/contextSummary/OPTION_CLASS.md`

---

### 2.2 Component 2: OptionStructure Class

**File**: `/Users/janussuk/Desktop/OptionsBacktester2/code/backtester/core/option_structure.py`

**Purpose**: Aggregates multiple Option objects into common structures (straddles, spreads, etc.)

**Key Attributes**:
```python
class OptionStructure:
    - structure_id: str             # Unique identifier
    - structure_type: str           # 'straddle', 'strangle', 'spread', etc.
    - options: List[Option]         # Component options
    - entry_date: datetime          # When structure was opened
    - underlying: str               # Underlying asset
    - net_premium: float            # Total credit/debit at entry
    - max_profit: float             # Theoretical max profit
    - max_loss: float               # Theoretical max loss
    - breakeven_points: List[float] # Breakeven prices
    - current_value: float          # Current mark value
    - pnl: float                    # Current P&L
```

**Key Methods**:
```python
- add_option(option: Option) -> None
- calculate_net_greeks() -> dict
- calculate_pnl() -> float
- calculate_max_profit() -> float
- calculate_max_loss() -> float
- calculate_breakevens() -> List[float]
- get_payoff_diagram(price_range) -> tuple
- is_delta_neutral(threshold=0.1) -> bool
- update_all_prices(market_data) -> None
- close_structure(exit_prices, timestamp) -> dict
- __repr__() -> str
```

**Concrete Implementations** (in `/structures/`):
- `LongStraddle`, `ShortStraddle`
- `LongStrangle`, `ShortStrangle`
- `BullCallSpread`, `BearPutSpread`
- `IronCondor`, `IronButterfly`

Each concrete class inherits from `OptionStructure` and implements:
```python
@classmethod
def create(cls, underlying, strike, expiration, quantity, ...) -> OptionStructure
```

**Context Summary**: Store in `/Users/janussuk/Desktop/OptionsBacktester2/contextSummary/STRUCTURE_CLASS.md`

---

### 2.3 Component 3: Strategy Class

**File**: `/Users/janussuk/Desktop/OptionsBacktester2/code/backtester/core/strategy.py`

**Purpose**: Defines entry/exit conditions and manages a collection of OptionStructures

**Key Attributes**:
```python
class Strategy:
    - name: str                     # Strategy name
    - description: str              # Strategy description
    - structures: List[OptionStructure]  # Active structures
    - closed_structures: List[OptionStructure]  # Closed positions
    - entry_conditions: List[Callable]  # Functions returning bool
    - exit_conditions: List[Callable]   # Functions returning bool
    - position_limits: dict         # Max positions, exposure limits
    - capital: float                # Available capital
    - initial_capital: float        # Starting capital
    - pnl_history: List[dict]       # Historical P&L
    - metrics: dict                 # Performance metrics
```

**Key Methods**:
```python
- should_enter(market_data, indicators) -> bool
- should_exit(structure, market_data, indicators) -> bool
- on_data(market_data, timestamp) -> None  # Called each time step
- open_position(structure: OptionStructure) -> None
- close_position(structure: OptionStructure, exit_data) -> dict
- update_positions(market_data) -> None
- calculate_portfolio_greeks() -> dict
- get_total_exposure() -> float
- get_margin_requirement() -> float
- validate_risk_limits() -> bool
- generate_signals(market_data) -> dict
- __repr__() -> str
```

**Example Entry/Exit Conditions** (for Short Straddle based on PDF):
```python
entry_conditions = [
    lambda data: data['iv_percentile'] > 50,  # High IV
    lambda data: data['vix'] > 20,            # VIX above threshold
    lambda data: not has_major_event(data['date']),  # No major events
    lambda data: get_open_positions() < max_positions
]

exit_conditions = [
    lambda structure: structure.pnl / structure.net_premium >= 0.25,  # 25% profit
    lambda structure: structure.pnl / structure.net_premium <= -1.0,  # 100% loss (stop)
    lambda structure: days_to_expiry(structure) <= 1,  # Time-based exit
]
```

**Context Summary**: Store in `/Users/janussuk/Desktop/OptionsBacktester2/contextSummary/STRATEGY_CLASS.md`

---

### 2.4 Component 4: Backtesting Engine Class

**File**: `/Users/janussuk/Desktop/OptionsBacktester2/code/backtester/engine/backtest_engine.py`

**Purpose**: Orchestrates the backtest, streams data, manages execution, collects metrics

**Key Attributes**:
```python
class BacktestEngine:
    - strategies: List[Strategy]     # Strategies to backtest
    - data_stream: DataStream        # Market data provider
    - start_date: datetime           # Backtest start
    - end_date: datetime             # Backtest end
    - initial_capital: float         # Starting capital per strategy
    - position_manager: PositionManager  # Track all positions
    - execution_model: ExecutionModel    # Slippage, commissions
    - results: dict                  # Backtest results
    - equity_curve: List[float]      # Daily portfolio values
    - trade_log: List[dict]          # All executed trades
```

**Key Methods**:
```python
- run() -> dict                     # Main backtest loop
- step(timestamp, market_data) -> None  # Single time step
- execute_orders(orders) -> List[dict]  # Execute pending orders
- update_portfolio(market_data) -> None
- calculate_metrics() -> dict
- generate_report() -> dict
- plot_results() -> None
- export_results(filepath) -> None
- validate_data_quality() -> bool
- handle_corporate_actions(event) -> None
- __repr__() -> str
```

**Backtesting Loop** (pseudo-code):
```python
def run():
    for timestamp, market_data in data_stream:
        # 1. Update all positions with current prices
        update_portfolio(market_data)

        # 2. Check exit conditions for open positions
        for strategy in strategies:
            for structure in strategy.structures:
                if strategy.should_exit(structure, market_data):
                    close_position(structure, market_data)

        # 3. Check entry conditions for new positions
        for strategy in strategies:
            if strategy.should_enter(market_data):
                new_structure = strategy.create_structure(market_data)
                open_position(new_structure)

        # 4. Record state
        record_metrics(timestamp)

    # 5. Generate final report
    return calculate_metrics()
```

**Context Summary**: Store in `/Users/janussuk/Desktop/OptionsBacktester2/contextSummary/ENGINE_CLASS.md`

---

### 2.5 Component 5: Dolt Data Adapter

**File**: `/Users/janussuk/Desktop/OptionsBacktester2/code/backtester/data/dolt_adapter.py`

**Purpose**: Interface with Dolt database to access options data

**Key Methods**:
```python
class DoltAdapter:
    - connect(db_path) -> Connection
    - get_option_chain(underlying, date, min_dte, max_dte) -> DataFrame
    - get_historical_options(underlying, start, end, strikes) -> DataFrame
    - get_underlying_prices(symbol, start, end) -> DataFrame
    - get_implied_volatility(symbol, date) -> float
    - get_vix_history(start, end) -> DataFrame
    - query_custom(sql) -> DataFrame
    - close() -> None
```

**Dolt Integration** (based on Using_Dolt.md):
```python
# Connect to cloned Dolt database
import doltpy
from doltpy.core import Dolt

# Initialize connection
repo = Dolt('/path/to/dolt_data/options')

# Query options data
sql = """
SELECT * FROM option_chain
WHERE underlying_symbol = 'SPY'
  AND quote_date = '2023-01-03'
  AND expiration >= '2023-01-10'
  AND expiration <= '2023-01-17'
"""
data = repo.sql(sql, result_format='pandas')
```

**Expected Database Schema** (to be verified):
```sql
-- Likely tables in post-no-preference/options
option_chain (
    quote_date DATE,
    underlying_symbol VARCHAR,
    expiration DATE,
    strike DECIMAL,
    option_type VARCHAR,  -- 'call' or 'put'
    bid DECIMAL,
    ask DECIMAL,
    mid DECIMAL,
    volume INT,
    open_interest INT,
    implied_volatility DECIMAL,
    delta DECIMAL,
    gamma DECIMAL,
    theta DECIMAL,
    vega DECIMAL,
    rho DECIMAL
)

underlying_prices (
    date DATE,
    symbol VARCHAR,
    open DECIMAL,
    high DECIMAL,
    low DECIMAL,
    close DECIMAL,
    volume BIGINT
)
```

**Context Summary**: Store schema and usage patterns in `/Users/janussuk/Desktop/OptionsBacktester2/contextSummary/DATA_SCHEMA.md`

---

## 3. Implementation Sequence (Multi-Run Strategy)

### Implementation Sequence Overview

| Run | Focus | Agent Usage | Deliverables |
|-----|-------|-------------|--------------|
| 1 | Data Layer & Schema Verification | quant-dev, quality-audit | Verified Dolt schema, DoltAdapter, MarketDataLoader |
| 2 | Option & Pricing Core | quant-dev, quality-audit | Option class, Black-Scholes pricing, Greeks |
| 3 | OptionStructure Base Class | quant-dev, quality-audit | Base class for multi-leg structures |
| 4 | Strategy Framework | quant-dev, quality-audit | Strategy base class, condition helpers |
| 5 | Backtesting Engine | quant-dev, quality-audit | DataStream, PositionManager, BacktestEngine |
| 6 | Analytics & Metrics | quant-dev, quality-audit | Performance metrics, risk analytics |
| 7 | Visualization & Reporting | quality-audit | Plotting, dashboards, reports |
| 8 | Concrete Implementations | quant-dev, quality-audit | Structure types (straddle, etc.), example strategies |
| 9 | Final Integration & Validation | quality-audit | Integration tests, example scripts, documentation |

**Key Principle**: Core libraries (Runs 1-7) are built and validated BEFORE concrete examples (Run 8).

---

### Run 1: Foundation & Data Layer
**Objective**: Set up project structure, environment, and verify actual Dolt database schema

**Tasks**:
1. Initialize project with `uv`:
   ```bash
   cd /Users/janussuk/Desktop/OptionsBacktester2/code
   uv venv
   source .venv/bin/activate
   uv pip install numpy pandas scipy matplotlib plotly seaborn pytest pyyaml doltpy
   ```

2. Clone Dolt database:
   ```bash
   cd /Users/janussuk/Desktop/OptionsBacktester2
   mkdir -p dolt_data && cd dolt_data
   dolt clone post-no-preference/options
   ```

3. **CRITICAL: Verify Actual Database Schema**:
   - Connect to the Dolt database
   - Run `SHOW TABLES;` to see all available tables
   - For each table, run `DESCRIBE table_name;` or `SHOW CREATE TABLE table_name;`
   - Run sample queries to verify data format and available columns
   - **Document ACTUAL schema** (not assumed schema) including:
     - Exact table names
     - All column names and types
     - Which columns contain Greeks (if any)
     - Which columns contain pricing data
     - Date range coverage
     - Available underlying symbols
   - Identify any discrepancies from the assumed schema in section 2.5

4. Create basic project structure (directories only):
   ```bash
   cd /Users/janussuk/Desktop/OptionsBacktester2/code
   mkdir -p backtester/{core,structures,strategies/examples,engine,data,analytics,utils}
   mkdir -p tests notebooks examples
   touch backtester/__init__.py
   touch backtester/{core,structures,strategies,engine,data,analytics,utils}/__init__.py
   ```

5. Implement `DoltAdapter` class based on ACTUAL schema:
   - Connection management
   - Query methods using ACTUAL column names
   - Data validation based on ACTUAL data format
   - **Use quant-options-dev agent** for this implementation

6. Implement `MarketDataLoader` based on verified schema:
   - Load underlying prices (using actual table/column names)
   - Load option chains (using actual structure)
   - Calculate implied volatility percentiles (if IV data exists)
   - **Use quant-options-dev agent** for this implementation

7. **Run code-quality-auditor agent** to review data layer implementation:
   - Check for bugs
   - Verify package dependencies
   - Ensure proper error handling

8. Create notebook `01_data_exploration.ipynb`:
   - Connect to Dolt and explore schema
   - Query sample data for SPY
   - Verify data quality and coverage
   - Visualize data availability
   - Document any data quirks or limitations

9. Write tests for data layer

10. **Context Summary**: Document ACTUAL database schema, data quirks, and usage patterns in `DATA_SCHEMA.md`:
    - Include exact SQL schema
    - Document all available columns
    - Note which fields need to be calculated vs. provided
    - List data coverage by symbol and date range

**Success Criteria**:
- Dolt database schema fully documented with ACTUAL structure
- Can successfully query options data from Dolt
- Can load SPY option chains for specific dates
- Data quality validated (no major gaps)
- All tests pass
- code-quality-auditor confirms no bugs or issues

---

### Run 2: Option & Pricing Core
**Objective**: Build Option class with pricing and Greeks (CORE LIBRARY - NO EXAMPLES YET)

**Tasks**:
1. **Use quant-options-dev agent** to implement `pricing.py` module:
   - Black-Scholes call/put pricing (European style)
   - Analytical Greeks calculations (delta, gamma, theta, vega, rho)
   - Implied volatility calculation (Newton-Raphson or Brent's method)
   - American option approximations (Barone-Adesi-Whaley if needed)
   - Proper handling of edge cases (T=0, sigma=0, etc.)
   - Numerical stability considerations

2. **Use quant-options-dev agent** to implement `Option` class (`option.py`):
   - All attributes and initialization
   - `calculate_greeks()` - using pricing module
   - `update_price()` - update current price and recalculate P&L
   - `calculate_pnl()` - account for long/short positions correctly
   - `get_payoff_at_expiry()` - intrinsic value at expiration
   - `is_itm()`, `is_atm()`, `is_otm()` - moneyness checks
   - `get_intrinsic_value()` - intrinsic value calculation
   - `get_time_value()` - extrinsic value calculation
   - Proper `__repr__()` and `__str__()` methods

3. **Run code-quality-auditor agent** to review pricing and Option implementation:
   - Verify financial correctness
   - Check for numerical stability issues
   - Ensure proper error handling
   - Verify edge cases are handled
   - Check for bugs or package issues

4. Write comprehensive tests (before auditing):
   - Test pricing edge cases (at expiry, deep ITM/OTM, zero volatility)
   - Verify Greeks calculations against known values
   - Test P&L tracking for long/short positions
   - Validate moneyness functions
   - Test American vs European pricing (if applicable)

5. Create notebook `02_option_pricing.ipynb`:
   - Load real market data from Dolt
   - Compare theoretical prices to market mid-prices
   - Visualize Greeks behavior across strikes
   - Validate P&L calculations with real scenarios
   - Document any discrepancies

6. **Context Summary**: Document in `OPTION_CLASS.md`:
   - Complete Option class API
   - Pricing model assumptions and limitations
   - Greeks calculation formulas used
   - Edge cases and how they're handled
   - Validation results from market data
   - Example usage patterns

**Success Criteria**:
- Option class fully functional
- Pricing accurate to within 1-2% of market mid-prices
- Greeks calculated correctly (validated against market Greeks if available)
- All tests pass
- code-quality-auditor confirms no bugs, proper numerical handling
- No hard-coded examples or concrete strategies yet (those come later)

---

### Run 3: OptionStructure Base Class
**Objective**: Build base structure aggregation class (CORE LIBRARY ONLY - NO CONCRETE IMPLEMENTATIONS)

**Tasks**:
1. **Use quant-options-dev agent** to implement `OptionStructure` base class (`option_structure.py`):
   - Container for multiple Option objects
   - `add_option(option: Option)` - add legs to structure
   - `remove_option(index)` - remove legs
   - `calculate_net_greeks()` - aggregate Greeks across all legs
   - `calculate_pnl()` - total P&L accounting for all positions
   - `calculate_total_cost()` - net debit/credit at entry
   - `update_all_prices(market_data)` - update all leg prices
   - `get_payoff_at_expiry(spot_price)` - total payoff at expiration
   - `get_payoff_diagram(price_range)` - payoff over price range
   - `close_structure(exit_prices, timestamp)` - close all legs and return stats
   - Properties: `net_delta`, `net_gamma`, `net_theta`, `net_vega`, `net_rho`
   - Proper validation (all options same underlying, etc.)
   - `__repr__()` and `__str__()` methods

2. **Run code-quality-auditor agent** to review OptionStructure implementation:
   - Verify financial correctness of aggregations
   - Check that P&L correctly handles long/short across multiple legs
   - Ensure proper error handling for invalid structures
   - Check for bugs or integration issues with Option class

3. Write comprehensive tests for OptionStructure base class:
   - Test adding/removing options
   - Verify net Greeks calculations (manual calculation vs. method)
   - Test P&L for complex multi-leg positions
   - Validate payoff diagrams
   - Test edge cases (empty structure, single leg, etc.)

4. Create notebook `03_option_structures.ipynb`:
   - Manually create simple 2-leg and 4-leg structures
   - Visualize combined payoff diagrams
   - Validate Greeks aggregation
   - Test with real market data from Dolt

5. **Context Summary**: Document in `STRUCTURE_CLASS.md`:
   - OptionStructure base class API
   - How to create custom structures
   - Greeks aggregation methodology
   - P&L calculation approach
   - Example usage for building multi-leg positions
   - **Note**: Concrete implementations (straddle, strangle, etc.) will come in Run 7

**Success Criteria**:
- OptionStructure base class fully functional
- Can manually create any multi-leg structure
- Aggregate calculations correct (validated against manual calculations)
- All tests pass
- code-quality-auditor confirms no bugs
- **NO concrete structure implementations yet** (those come after core engine is built)

---

### Run 4: Strategy Framework
**Objective**: Build Strategy class with condition system (CORE LIBRARY - NO EXAMPLE STRATEGIES YET)

**Tasks**:
1. **Use quant-options-dev agent** to implement `Strategy` base class (`strategy.py`):
   - `name`, `description` attributes
   - `structures: List[OptionStructure]` - active positions
   - `closed_structures: List[OptionStructure]` - trade history
   - `capital`, `initial_capital` - capital tracking
   - `position_limits` - risk limits dictionary
   - `open_position(structure: OptionStructure)` - add to active positions
   - `close_position(structure: OptionStructure, exit_data)` - close and record
   - `update_positions(market_data)` - update all active position prices
   - `calculate_portfolio_greeks()` - aggregate across all positions
   - `get_total_exposure()` - total capital at risk
   - `get_margin_requirement()` - calculate margin needed
   - `validate_risk_limits()` - check if within limits
   - `calculate_total_pnl()` - unrealized + realized P&L
   - Entry/exit condition framework (to be overridden by subclasses)
   - `should_enter(market_data)` - abstract method
   - `should_exit(structure, market_data)` - abstract method

2. Create condition helper utilities (`utils/conditions.py`):
   - `calculate_iv_percentile(current_iv, historical_ivs, window)`
   - `is_major_event_date(date, event_calendar)` - check earnings, FOMC, etc.
   - `check_position_limit(current_count, max_count)`
   - `days_to_expiry(expiration, current_date)`
   - `calculate_profit_pct(pnl, initial_premium)`
   - Generic condition builders for reusability

3. **Run code-quality-auditor agent** to review Strategy implementation:
   - Verify position tracking logic
   - Check capital and P&L calculations
   - Ensure proper integration with OptionStructure
   - Verify risk limit calculations
   - Check for bugs or issues

4. Write comprehensive tests for Strategy base class:
   - Test position opening/closing
   - Verify capital tracking
   - Test portfolio Greeks aggregation
   - Validate margin calculations
   - Test risk limit enforcement

5. Create notebook `04_strategy_framework.ipynb`:
   - Create a simple test strategy (manual entry/exit)
   - Test position management
   - Visualize portfolio Greeks over time
   - Validate capital tracking

6. **Context Summary**: Document in `STRATEGY_CLASS.md`:
   - Strategy base class API
   - How to implement custom strategies
   - Condition helper utilities
   - Risk management framework
   - Capital and P&L tracking methodology
   - **Note**: Concrete example strategies (short straddle, etc.) will come in Run 8

**Success Criteria**:
- Strategy base class fully functional
- Can create custom strategy subclasses
- Position management works correctly
- All tests pass
- code-quality-auditor confirms no bugs
- **NO concrete example strategies yet** (those come after engine is working)

---

### Run 5: Backtesting Engine
**Objective**: Build core backtesting infrastructure (ENGINE CORE - NO EXAMPLE BACKTESTS YET)

**Tasks**:
1. **Use quant-options-dev agent** to implement `DataStream` class (`data_stream.py`):
   - Iterator interface over time-series data
   - `__init__(start_date, end_date, data_source)` - configure date range
   - `__iter__()` and `__next__()` - iteration protocol
   - `get_current_timestamp()` - current time in backtest
   - `get_option_chain(underlying, date, dte_range)` - get available options
   - `get_underlying_price(symbol, date)` - get spot price
   - Handle market hours/calendar (skip weekends, holidays)
   - Pre-load or stream data efficiently

2. **Use quant-options-dev agent** to implement `PositionManager` (`position_manager.py`):
   - `add_position(structure)` - track new position
   - `remove_position(structure_id)` - remove closed position
   - `get_all_positions()` - list all active positions
   - `calculate_total_margin()` - margin requirement across all positions
   - `calculate_portfolio_value(market_data)` - mark-to-market value
   - `get_portfolio_greeks()` - aggregate Greeks across all positions
   - Position-level risk tracking

3. **Use quant-options-dev agent** to implement `ExecutionModel` (`execution.py`):
   - `execute_entry(structure, market_data)` - simulate opening position
   - `execute_exit(structure, market_data)` - simulate closing position
   - Apply bid/ask spread (buy at ask, sell at bid)
   - Transaction costs (commissions per contract)
   - Slippage modeling (optional)
   - Fill price calculation

4. **Use quant-options-dev agent** to implement `BacktestEngine` (`backtest_engine.py`):
   - `__init__(strategies, data_stream, initial_capital, execution_model)`
   - `run()` - main backtest loop
   - `step(timestamp, market_data)` - single time step
   - Event loop: update positions → check exits → check entries → record state
   - `record_trade(trade_data)` - log all trades
   - `record_state(timestamp, portfolio_value)` - equity curve
   - `generate_equity_curve()` - return time series of portfolio value
   - `get_trade_log()` - return all trades as DataFrame
   - Error handling and validation

5. **Run code-quality-auditor agent** to review engine implementation:
   - Verify event loop logic
   - Check data flow between components
   - Ensure proper integration with Strategy and OptionStructure
   - Verify execution logic (bid/ask, commissions)
   - Check for bugs, race conditions, or edge cases

6. Write comprehensive tests for engine components:
   - Test DataStream iteration and data retrieval
   - Test PositionManager tracking
   - Test ExecutionModel fill prices and costs
   - Test BacktestEngine event loop with mock data
   - Integration test: simple end-to-end backtest with manual strategy

7. **Context Summary**: Document in `ENGINE_CLASS.md`:
   - Engine architecture and data flow
   - Event loop mechanics
   - Execution model details
   - How components integrate
   - **Note**: Example backtests will come in Run 9

**Success Criteria**:
- Engine successfully runs through historical data
- Positions tracked correctly
- Equity curve generated
- Trade log captured
- All tests pass
- code-quality-auditor confirms no bugs
- **NO example backtests yet** (those require concrete strategies in Run 8)

---

### Run 6: Analytics & Metrics
**Objective**: Calculate performance metrics and risk analytics

**Tasks**:
1. **Use quant-options-dev agent** to implement `PerformanceMetrics` (`analytics/metrics.py`):
   - `calculate_total_return(equity_curve)`
   - `calculate_annualized_return(equity_curve, trading_days)`
   - `calculate_sharpe_ratio(returns, risk_free_rate)`
   - `calculate_sortino_ratio(returns, risk_free_rate)`
   - `calculate_max_drawdown(equity_curve)` - return max DD and duration
   - `calculate_win_rate(trades)` - % profitable trades
   - `calculate_profit_factor(trades)` - gross profit / gross loss
   - `calculate_average_win(trades)` and `calculate_average_loss(trades)`
   - `calculate_calmar_ratio(returns, max_drawdown)`
   - All standard quantitative metrics

2. **Use quant-options-dev agent** to implement `RiskAnalytics` (`analytics/risk.py`):
   - `calculate_var(returns, confidence_level)` - Value at Risk
   - `calculate_cvar(returns, confidence_level)` - Conditional VaR (Expected Shortfall)
   - `analyze_greeks_over_time(backtest_results)` - time series of portfolio Greeks
   - `calculate_margin_utilization(backtest_results)`
   - `calculate_tail_risk(returns)` - skewness, kurtosis
   - `calculate_maximum_adverse_excursion(trades)`

3. Integrate metrics into `BacktestEngine`:
   - Add `calculate_metrics()` method that uses PerformanceMetrics and RiskAnalytics
   - Store results in results dictionary
   - Generate summary statistics

4. **Run code-quality-auditor agent** to review analytics implementation:
   - Verify correctness of metric formulas
   - Check for numerical stability issues
   - Ensure proper handling of edge cases (empty trades, zero returns, etc.)
   - Check for bugs or calculation errors

5. Write comprehensive tests for metrics:
   - Test each metric with known inputs/outputs
   - Verify against manual calculations
   - Test edge cases (no trades, all wins, all losses, etc.)
   - Validate VaR/CVaR calculations

6. **Context Summary**: Update `ENGINE_CLASS.md`:
   - Document all available metrics
   - Explain calculation methodologies
   - Note assumptions and limitations
   - Add examples of metric interpretation

**Success Criteria**:
- All standard metrics calculated correctly
- Risk analytics functional and validated
- Results match manual calculations
- All tests pass
- code-quality-auditor confirms correctness

---

### Run 7: Visualization & Reporting
**Objective**: Build visualization and reporting capabilities

**Tasks**:
1. Implement `Visualization` class (`analytics/visualization.py`):
   - `plot_equity_curve(equity_curve, save_path)` - matplotlib/plotly
   - `plot_drawdown(equity_curve, save_path)` - drawdown over time
   - `plot_pnl_distribution(trades, save_path)` - histogram of trade P&L
   - `plot_greeks_over_time(greek_history, save_path)` - time series
   - `plot_payoff_diagram(structure, price_range, save_path)` - option payoff
   - `plot_entry_exit_points(underlying_prices, trades, save_path)` - mark trades on price chart
   - Support both static (matplotlib) and interactive (plotly) outputs

2. Create interactive dashboards using Plotly:
   - `create_performance_dashboard(backtest_results)` - comprehensive dashboard
   - Strategy metrics summary
   - Position-level details table
   - Risk exposure over time
   - Returns distribution
   - Save as HTML for viewing

3. Implement report generation (`analytics/reporting.py`):
   - `generate_summary_report(backtest_results)` - text/markdown summary
   - `export_trade_log(trades, filepath)` - CSV export
   - `create_metrics_table(metrics)` - formatted table
   - Optional: PDF report generation (if time permits)

4. **Run code-quality-auditor agent** to review visualization implementation:
   - Check for bugs in plotting code
   - Verify data handling for edge cases
   - Ensure proper error handling
   - Check package dependencies (matplotlib, plotly, seaborn)

5. Write tests for visualization functions:
   - Test with mock data
   - Verify plots generate without errors
   - Test export functionality

6. Create notebook `05_backtest_analysis.ipynb`:
   - Load sample backtest results
   - Generate all visualizations
   - Create interactive dashboard
   - Export reports

7. **Context Summary**: Update `ENGINE_CLASS.md`:
   - Document visualization capabilities
   - Provide examples of each plot type
   - Explain dashboard usage

**Success Criteria**:
- Can generate all visualization types
- Interactive dashboards functional
- Reports exportable to CSV/HTML
- All tests pass
- code-quality-auditor confirms no issues

---

### Run 8: Concrete Structures & Example Strategies
**Objective**: Implement concrete option structures and example strategies (NOW THAT CORE IS COMPLETE)

**Tasks**:
1. **Use quant-options-dev agent** to implement concrete structure classes (`structures/`):
   - `structures/straddle.py`:
     - `LongStraddle` - buy ATM call + put
     - `ShortStraddle` - sell ATM call + put
   - `structures/strangle.py`:
     - `LongStrangle` - buy OTM call + OTM put
     - `ShortStrangle` - sell OTM call + OTM put
   - `structures/spread.py`:
     - `BullCallSpread` - buy lower strike call, sell higher strike call
     - `BearPutSpread` - buy higher strike put, sell lower strike put
     - `CalendarSpread` - buy far expiry, sell near expiry (same strike)
   - `structures/condor.py`:
     - `IronCondor` - OTM put spread + OTM call spread
     - `IronButterfly` - ATM put spread + ATM call spread

   Each class should:
   - Inherit from `OptionStructure`
   - Implement `@classmethod create(...)` factory method
   - Calculate theoretical max profit/loss
   - Calculate breakeven points
   - Validate structure parameters

2. **Use quant-options-dev agent** to implement example strategies (`strategies/examples/`):
   - `short_straddle_daily.py` - based on knowledgeBase PDF:
     - Entry: IV > 50th percentile, no major events, DTE = 1-3 days
     - Exit: 25% profit OR 100% loss OR DTE <= 1
     - Position sizing: 1-2% of capital
   - `volatility_selling.py` - generic short premium strategy:
     - Entry: VIX > threshold, IV rank > 50
     - Exit: profit target, stop loss, or time

   Each strategy should:
   - Inherit from `Strategy` base class
   - Implement `should_enter()` and `should_exit()`
   - Use condition helper utilities from `utils/conditions.py`
   - Include docstrings explaining the logic

3. **Run code-quality-auditor agent** to review all concrete implementations:
   - Verify structure factory methods create valid positions
   - Check strategy entry/exit logic for bugs
   - Ensure proper use of base classes
   - Verify calculations (max profit, breakevens, etc.)
   - Check for integration issues

4. Write tests for concrete implementations:
   - Test each structure's `create()` method
   - Verify max profit/loss calculations
   - Test breakeven calculations
   - Test strategy entry/exit conditions with various scenarios
   - Integration tests with engine

5. Create example usage notebooks:
   - `06_concrete_structures.ipynb`:
     - Create each structure type
     - Visualize payoff diagrams
     - Compare theoretical vs actual values
   - `07_example_strategies.ipynb`:
     - Demonstrate short straddle strategy
     - Show entry/exit signal generation
     - Backtest on sample data

6. **Context Summary**: Update documentation:
   - Update `STRUCTURE_CLASS.md` with all concrete implementations
   - Update `STRATEGY_CLASS.md` with example strategies
   - Document usage patterns and best practices

**Success Criteria**:
- All concrete structures implemented and tested
- Example strategies functional
- Factory methods work correctly
- All tests pass
- code-quality-auditor confirms no issues
- Can now run real backtests with example strategies

---

### Run 9: Testing, Validation & Final Integration
**Objective**: Comprehensive testing, validation, and create end-to-end examples

**Tasks**:
1. Write comprehensive integration tests (`tests/test_integration.py`):
   - End-to-end backtest with short straddle strategy
   - Multi-strategy portfolio backtest
   - Edge cases:
     - Market crash scenarios (rapid price movements)
     - Early assignment (for American options if applicable)
     - Zero volume options
     - Expiration handling
   - Data edge cases (missing data, holidays, etc.)

2. **Run code-quality-auditor agent** for final system review:
   - Full codebase audit for bugs
   - Check all package dependencies are correct
   - Verify proper integration across all components
   - Identify any redundancies or unused code
   - Check for performance bottlenecks

3. Performance optimization (if needed):
   - Profile slow operations using cProfile
   - Vectorize calculations where possible
   - Consider numba JIT for Greeks calculations if performance is insufficient
   - Optimize database queries
   - Cache repeated calculations

4. Validate against known results:
   - Run short straddle strategy per PDF specifications
   - Compare results to published expectations (win rate ~75-90%)
   - Manually verify sample calculations
   - Cross-check metrics with alternative calculations

5. Create example scripts (`examples/`):
   - `simple_backtest.py`:
     - Load data
     - Create simple strategy
     - Run backtest
     - Display results
     - < 50 lines, well-commented
   - `advanced_backtest.py`:
     - Multi-strategy portfolio
     - Custom entry/exit logic
     - Risk management
     - Generate reports and visualizations
     - ~100-150 lines with detailed comments

6. Create final demonstration notebook (`08_full_backtest_demo.ipynb`):
   - Complete end-to-end workflow
   - Load Dolt data
   - Implement short straddle strategy
   - Run backtest for 1+ years
   - Generate all visualizations
   - Export results
   - Analyze performance

7. Documentation:
   - Ensure all classes/methods have docstrings
   - Create comprehensive README.md:
     - Project overview
     - Installation instructions
     - Quick start guide
     - Example usage
     - Links to documentation
   - Update all context summaries with any changes
   - Create API reference documentation

8. **Context Summary**: Create `INTEGRATION_GUIDE.md`:
   - How all components work together
   - Common workflows
   - Best practices for custom strategies
   - Troubleshooting guide
   - Performance tips
   - Extension points for future development

**Success Criteria**:
- All tests pass (unit + integration)
- code-quality-auditor confirms system is bug-free and optimized
- Performance acceptable (< 1 min for 1-year SPY backtest on modern hardware)
- Validation results match expected outcomes from PDF (within ±10%)
- Documentation complete and comprehensive
- Example scripts work out-of-the-box
- System ready for production use

---

## 4. Context Summary Strategy

To maintain coherence across runs, create these context summaries:

### 4.1 ARCHITECTURE.md
**Purpose**: High-level system design
**Contents**:
- Component diagram
- Data flow
- Dependency graph
- Key design decisions
- Integration points

### 4.2 OPTION_CLASS.md
**Purpose**: Option class reference
**Contents**:
- Complete API documentation
- Pricing model assumptions
- Greeks calculation formulas
- Edge cases and limitations
- Example usage

### 4.3 STRUCTURE_CLASS.md
**Purpose**: OptionStructure reference
**Contents**:
- Base class API
- All concrete implementations
- How to add new structures
- Payoff calculations
- Example usage

### 4.4 STRATEGY_CLASS.md
**Purpose**: Strategy framework reference
**Contents**:
- Strategy API
- Condition system design
- Example strategies
- How to implement custom strategies
- Risk management patterns

### 4.5 ENGINE_CLASS.md
**Purpose**: Backtesting engine reference
**Contents**:
- Engine architecture
- Event loop mechanics
- Execution model
- Metrics calculations
- Visualization capabilities
- Performance considerations

### 4.6 DATA_SCHEMA.md
**Purpose**: Data layer reference
**Contents**:
- Dolt database schema
- Query patterns
- Data quality notes
- Available date ranges
- Data preprocessing steps

### 4.7 INTEGRATION_GUIDE.md
**Purpose**: Cross-component usage
**Contents**:
- How components interact
- Common workflows
- Best practices
- Troubleshooting
- Extension points

---

## 5. Dolt Database Setup & Usage

### 5.1 Installation (macOS)
```bash
# Install Dolt
brew install dolt

# Verify installation
dolt version
```

### 5.2 Clone Options Database
```bash
cd /Users/janussuk/Desktop/OptionsBacktester2
mkdir dolt_data && cd dolt_data

# Clone the repository (this downloads historical options data)
dolt clone post-no-preference/options

# Navigate into the cloned database
cd options

# Verify data
dolt sql -q "SHOW TABLES;"
dolt sql -q "SELECT COUNT(*) FROM option_chain LIMIT 10;"
```

### 5.3 Using Dolt in Python
```python
import doltpy
from doltpy.core import Dolt
import pandas as pd

# Connect to local Dolt database
repo = Dolt('/Users/janussuk/Desktop/OptionsBacktester2/dolt_data/options')

# Query data
sql = """
SELECT
    quote_date,
    strike,
    option_type,
    bid,
    ask,
    implied_volatility,
    delta,
    gamma,
    theta,
    vega
FROM option_chain
WHERE underlying_symbol = 'SPY'
  AND quote_date BETWEEN '2023-01-01' AND '2023-01-31'
  AND expiration = '2023-02-17'
ORDER BY strike
"""

df = repo.sql(sql, result_format='pandas')
```

### 5.4 Data Quality Considerations
- **Check for gaps**: Some dates may be missing (holidays, data issues)
- **Verify pricing**: Compare mid-price to theoretical values
- **Filter bad data**: Remove options with zero bid/ask or extreme IVs
- **Handle splits**: Corporate actions may affect option contracts

---

## 6. Key Implementation Details

### 6.1 Option Pricing & Greeks

**Black-Scholes Implementation**:
```python
from scipy.stats import norm
import numpy as np

def black_scholes_call(S, K, T, r, sigma):
    """Black-Scholes call option price"""
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

def calculate_greeks(S, K, T, r, sigma, option_type='call'):
    """Calculate all Greeks analytically"""
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)

    # Delta
    if option_type == 'call':
        delta = norm.cdf(d1)
    else:
        delta = -norm.cdf(-d1)

    # Gamma (same for call and put)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))

    # Theta
    if option_type == 'call':
        theta = (-S*norm.pdf(d1)*sigma/(2*np.sqrt(T))
                 - r*K*np.exp(-r*T)*norm.cdf(d2)) / 365
    else:
        theta = (-S*norm.pdf(d1)*sigma/(2*np.sqrt(T))
                 + r*K*np.exp(-r*T)*norm.cdf(-d2)) / 365

    # Vega (same for call and put)
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100

    # Rho
    if option_type == 'call':
        rho = K * T * np.exp(-r*T) * norm.cdf(d2) / 100
    else:
        rho = -K * T * np.exp(-r*T) * norm.cdf(-d2) / 100

    return {
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'vega': vega,
        'rho': rho
    }
```

### 6.2 Short Straddle Strategy (from PDF)

**Entry Conditions**:
```python
def should_enter_short_straddle(market_data, portfolio_state):
    """Entry logic for daily short straddle (per PDF)"""

    # 1. IV must be above 50th percentile
    if market_data['iv_percentile'] < 50:
        return False

    # 2. No major events scheduled
    if has_major_event(market_data['date']):
        return False

    # 3. Position limits
    if len(portfolio_state['open_positions']) >= MAX_POSITIONS:
        return False

    # 4. VIX above threshold (optional enhancement)
    if market_data['vix'] < 20:
        return False

    return True
```

**Exit Conditions**:
```python
def should_exit_short_straddle(structure, market_data):
    """Exit logic for short straddle (per PDF)"""

    # Calculate current P&L as % of premium collected
    pnl_pct = structure.pnl / structure.net_premium

    # 1. Profit target: 25% of premium
    if pnl_pct >= 0.25:
        return True, 'profit_target'

    # 2. Stop loss: 100% of premium (doubling)
    if pnl_pct <= -1.0:
        return True, 'stop_loss'

    # 3. Time-based: Close by 1 DTE
    days_to_expiry = (structure.expiration - market_data['date']).days
    if days_to_expiry <= 1:
        return True, 'time_exit'

    return False, None
```

**Position Sizing** (per PDF risk management):
```python
def calculate_position_size(account_equity, market_data):
    """Size position at 1-2% of capital"""

    # Base size: 1 contract per $50k
    base_size = max(1, int(account_equity / 50000))

    # Scale up slightly if IV is very high (optional)
    if market_data['iv_percentile'] > 75:
        return min(base_size * 1.5, base_size + 1)

    return base_size
```

### 6.3 Performance Metrics

**Key Metrics to Calculate**:
```python
class PerformanceMetrics:
    @staticmethod
    def sharpe_ratio(returns, risk_free_rate=0.04):
        """Annualized Sharpe ratio"""
        excess_returns = returns - risk_free_rate/252
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

    @staticmethod
    def max_drawdown(equity_curve):
        """Maximum drawdown from peak"""
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / running_max
        return drawdown.min()

    @staticmethod
    def win_rate(trades):
        """Percentage of profitable trades"""
        profitable = sum(1 for t in trades if t['pnl'] > 0)
        return profitable / len(trades) if trades else 0

    @staticmethod
    def profit_factor(trades):
        """Gross profit / Gross loss"""
        gross_profit = sum(t['pnl'] for t in trades if t['pnl'] > 0)
        gross_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
        return gross_profit / gross_loss if gross_loss > 0 else float('inf')
```

---

## 7. Testing Strategy

### 7.1 Unit Tests
Each component should have comprehensive unit tests:

```python
# tests/test_option.py
def test_option_creation():
    """Test Option instantiation"""
    opt = Option(
        option_type='call',
        position_type='long',
        underlying='SPY',
        strike=400.0,
        expiration=datetime(2024, 1, 19),
        quantity=1,
        entry_price=10.50,
        entry_date=datetime(2024, 1, 3)
    )
    assert opt.strike == 400.0
    assert opt.option_type == 'call'

def test_greeks_calculation():
    """Test Greeks accuracy"""
    opt = Option(...)
    greeks = opt.calculate_greeks(
        spot=400, vol=0.20, rate=0.05, time_to_expiry=0.0438
    )
    assert 0 < greeks['delta'] < 1  # Call delta
    assert greeks['gamma'] > 0
    assert greeks['theta'] < 0
```

### 7.2 Integration Tests
Test component interactions:

```python
# tests/test_integration.py
def test_full_backtest():
    """Test complete backtest workflow"""
    # Setup
    engine = BacktestEngine(...)
    strategy = ShortStraddleStrategy(...)
    engine.add_strategy(strategy)

    # Run
    results = engine.run(
        start_date='2023-01-01',
        end_date='2023-12-31'
    )

    # Validate
    assert results['total_trades'] > 0
    assert 'sharpe_ratio' in results
    assert results['final_equity'] > 0
```

### 7.3 Validation Tests
Compare to known benchmarks:

```python
def test_against_published_results():
    """Validate against PDF strategy results"""
    # Implement short straddle per PDF spec
    # Compare win rate, profit factor, etc.
    # Should be ~75-90% win rate, positive expectancy
    pass
```

---

## 8. Dependencies & Environment

### 8.1 Core Dependencies (requirements.txt)
```
# Numerical & Data
numpy>=2.0.2
pandas>=2.3.3
scipy>=1.13.1

# Visualization
matplotlib>=3.9.4
plotly>=6.5.0
seaborn>=0.13.2

# Database
doltpy>=2.0.0

# Optional Performance
numba>=0.58.0

# Development
pytest>=7.4.0
black>=23.0.0
flake8>=6.0.0
jupyter>=1.0.0
notebook>=7.0.0
ipykernel>=6.25.0
```

### 8.2 Environment Setup
```bash
# Navigate to code directory
cd /Users/janussuk/Desktop/OptionsBacktester2/code

# Create virtual environment with uv
uv venv

# Activate
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt

# Install in editable mode (for development)
uv pip install -e .
```

---

## 9. Development Workflow

### 9.1 For Each Run

**Before Starting**:
1. Read relevant context summaries from `/contextSummary/`
2. Review previous component implementations
3. Understand integration points

**During Development**:
1. Follow TDD where appropriate (write tests first)
2. Document as you go (docstrings)
3. Run tests frequently
4. Use notebooks for exploratory work

**After Completion**:
1. Update context summaries with new learnings
2. Document any deviations from plan
3. Note integration points for next run
4. Run full test suite

### 9.2 Code Quality Standards

**Style**:
- Use Black for formatting
- Follow PEP 8
- Type hints where helpful
- Descriptive variable names

**Documentation**:
- Docstrings for all public methods
- Inline comments for complex logic
- Examples in docstrings

**Testing**:
- Minimum 80% coverage
- Test edge cases
- Integration tests for workflows

---

## 10. Expected Challenges & Solutions

### 10.1 Data Quality Issues
**Challenge**: Missing data, bad pricing, corporate actions
**Solution**:
- Implement data validation layer
- Filter anomalies
- Document data quirks in DATA_SCHEMA.md

### 10.2 Performance
**Challenge**: Slow backtests on large datasets
**Solution**:
- Vectorize calculations
- Use numba for Greeks
- Cache repeated calculations
- Consider downsampling for development

### 10.3 Memory Usage
**Challenge**: Loading large option chains
**Solution**:
- Stream data rather than load all at once
- Use generators for data iteration
- Clear unused data periodically

### 10.4 Complexity Management
**Challenge**: Strategy logic becomes complex
**Solution**:
- Keep components modular
- Use composition over inheritance
- Create helper functions for common patterns
- Document thoroughly

---

## 11. Success Metrics

### 11.1 Per-Run Metrics
- All tests pass
- Component fully functional
- Context summary updated
- Integration verified

### 11.2 Final Project Metrics
- Complete backtest runs successfully
- Reproduces PDF strategy results (±10%)
- Performance acceptable (< 1 min for 1-year SPY backtest)
- All documentation complete
- Example scripts work end-to-end

---

## 12. Extension Points

After core implementation, consider:

### 12.1 Additional Features
- Real-time data integration (live trading)
- Multi-asset support (SPX, QQQ, etc.)
- Advanced order types (trailing stops)
- Portfolio optimization
- Machine learning for signal generation

### 12.2 Additional Strategies
- Iron Condor weekly
- Calendar spreads
- Ratio spreads
- Long volatility strategies
- Earnings plays

### 12.3 Advanced Analytics
- Monte Carlo simulation
- Regime detection
- Correlation analysis
- Factor attribution

---

## Critical Files for Implementation

Based on this plan, the most critical files for initial implementation are:

1. **/Users/janussuk/Desktop/OptionsBacktester2/code/backtester/data/dolt_adapter.py**
   - Reason: Foundation for all data access; must understand Dolt schema first

2. **/Users/janussuk/Desktop/OptionsBacktester2/code/backtester/core/option.py**
   - Reason: Building block for everything; pricing and Greeks are core

3. **/Users/janussuk/Desktop/OptionsBacktester2/code/backtester/core/option_structure.py**
   - Reason: Aggregates Options; enables strategy implementation

4. **/Users/janussuk/Desktop/OptionsBacktester2/code/backtester/core/strategy.py**
   - Reason: Orchestrates entry/exit logic; where strategy rules live

5. **/Users/janussuk/Desktop/OptionsBacktester2/code/backtester/engine/backtest_engine.py**
   - Reason: Main execution loop; coordinates all components

---

## Appendix: Quick Reference

### Option Types
- **Long Call**: Bullish, limited risk, unlimited upside
- **Short Call**: Bearish/neutral, unlimited risk, limited profit
- **Long Put**: Bearish, limited risk, large downside potential
- **Short Put**: Bullish/neutral, large risk, limited profit

### Common Structures
- **Straddle**: Same strike call + put (bet on volatility)
- **Strangle**: OTM call + OTM put (bet on big move)
- **Vertical Spread**: Buy one, sell another at different strike
- **Iron Condor**: OTM put spread + OTM call spread (bet on range)

### Greek Interpretation
- **Delta**: Directional exposure (±0.5 for ATM)
- **Gamma**: Rate of delta change (highest ATM)
- **Theta**: Time decay (negative for longs, positive for shorts)
- **Vega**: IV sensitivity (positive for longs, negative for shorts)
- **Rho**: Interest rate sensitivity (usually minor)

### Risk Management Rules (from PDF)
- Position size: 1-2% of capital per trade
- Profit target: 25% of premium
- Stop loss: 100% of premium (doubling)
- Time exit: Close by 1 DTE
- Entry filter: IV > 50th percentile
- Event avoidance: No major scheduled catalysts

---

**End of Implementation Plan**

This plan provides a comprehensive roadmap for building the options strategy backtester incrementally. Each run is self-contained yet contributes to the larger system. Context summaries ensure coherence across runs, enabling efficient development by different agents or sessions.