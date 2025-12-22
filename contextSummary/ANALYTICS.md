# Analytics Module Documentation

## Overview

The Analytics module provides comprehensive performance and risk metrics for evaluating options trading strategies. It integrates seamlessly with the BacktestEngine to deliver industry-standard quantitative finance metrics.

## Module Components

### PerformanceMetrics Class
**Location**: `/code/backtester/analytics/metrics.py`

Provides static methods for calculating performance metrics.

### RiskAnalytics Class
**Location**: `/code/backtester/analytics/risk.py`

Provides static methods for calculating risk metrics.

---

## Metrics Reference

### Returns-Based Metrics

#### Total Return
```
Total Return = (Final Equity - Initial Equity) / Initial Equity * 100
```

**Interpretation**: Simple percentage gain/loss from start to end.

#### Annualized Return (CAGR)
```
CAGR = (Final / Initial)^(252 / num_days) - 1
```

**Interpretation**: Compound annual growth rate. Allows comparison across different time periods.

**Benchmarks**:
- S&P 500 long-term average: ~10%
- Options premium selling: 15-25%
- High-frequency strategies: >50%

#### Sharpe Ratio
```
Sharpe = (mean(returns) - rf/252) / std(returns) * sqrt(252)
```

**Interpretation**: Risk-adjusted return measuring excess return per unit of total volatility.

**Benchmarks**:
| Sharpe | Quality |
|--------|---------|
| < 0 | Poor - losing money |
| 0 - 1.0 | Below average |
| 1.0 - 2.0 | Good |
| 2.0 - 3.0 | Very good |
| > 3.0 | Excellent (rare) |

**Note**: Institutional strategies typically target Sharpe > 1.0.

#### Sortino Ratio
```
Sortino = (mean(returns) - rf/252) / downside_deviation * sqrt(252)
```

**Interpretation**: Like Sharpe but uses only downside volatility. Better for asymmetric return distributions common in options strategies.

**When Sortino > Sharpe**: Strategy has positive skew (more upside than downside volatility).

#### Calmar Ratio
```
Calmar = Annualized Return / |Max Drawdown|
```

**Interpretation**: Return relative to worst-case drawdown risk.

**Benchmarks**:
| Calmar | Quality |
|--------|---------|
| < 0.5 | Poor |
| 0.5 - 1.0 | Acceptable |
| 1.0 - 3.0 | Good |
| > 3.0 | Excellent |

---

### Drawdown Metrics

#### Maximum Drawdown
```
Drawdown(t) = (Equity(t) - Peak(t)) / Peak(t)
Max Drawdown = min(Drawdown(t)) for all t
```

**Interpretation**: Largest peak-to-trough decline. Represents worst-case loss an investor could have experienced.

**Components**:
- **Peak Date**: When equity reached its high before the drawdown
- **Trough Date**: When equity hit its low point
- **Recovery Date**: When equity returned to the peak level
- **Duration**: Days from peak to trough
- **Recovery Days**: Days from trough to recovery

**Benchmarks** (options strategies):
| Max DD | Risk Level |
|--------|------------|
| < -10% | Low risk |
| -10% to -20% | Moderate |
| -20% to -30% | High |
| > -30% | Very high |

#### Ulcer Index
```
UI = sqrt(mean(drawdown^2))
```

**Interpretation**: RMS of drawdowns. Lower is better. Captures both depth and duration of drawdowns.

---

### Trade-Based Metrics

#### Win Rate
```
Win Rate = (Winning Trades / Total Trades) * 100
```

**Interpretation**: Percentage of trades that were profitable.

**Benchmarks by Strategy Type**:
| Strategy | Typical Win Rate |
|----------|------------------|
| Credit spreads | 60-80% |
| Iron condors | 70-85% |
| Long options | 30-45% |
| Straddles/strangles (long) | 25-40% |

**Important**: High win rate does NOT guarantee profitability. Must consider payoff ratio.

#### Profit Factor
```
Profit Factor = Gross Profit / Gross Loss
```

**Interpretation**: How much profit generated for each dollar lost.

**Benchmarks**:
| Profit Factor | Quality |
|--------------|---------|
| < 1.0 | Losing strategy |
| 1.0 - 1.5 | Marginal |
| 1.5 - 2.0 | Good |
| > 2.0 | Excellent |

#### Expectancy (Edge per Trade)
```
Expectancy = Win Rate * Average Win - Loss Rate * |Average Loss|
```

**Interpretation**: Expected dollar amount per trade. Must be positive for long-term profitability.

**Example**:
- Win Rate: 60%
- Average Win: $150
- Average Loss: $200
- Expectancy = 0.60 * $150 - 0.40 * $200 = $90 - $80 = $10 per trade

#### Payoff Ratio
```
Payoff Ratio = Average Win / |Average Loss|
```

**Interpretation**: Risk/reward relationship.

**Kelly Criterion Connection**:
```
Kelly % = Win Rate - (Loss Rate / Payoff Ratio)
```

---

### Risk Metrics

#### Value at Risk (VaR)
```
Historical VaR: (1-alpha) percentile of returns
Parametric VaR: mu - z_alpha * sigma
```

**Interpretation**: Maximum expected loss at a given confidence level.

**Example**: 95% VaR of -2% means there is a 5% chance of losing more than 2% in a single period.

**Methods**:
- **Historical**: Uses actual return distribution (non-parametric)
- **Parametric**: Assumes normal distribution

**Benchmarks** (daily VaR for options portfolios):
| 95% Daily VaR | Risk Level |
|---------------|------------|
| > -1% | Low |
| -1% to -2% | Moderate |
| -2% to -3% | High |
| < -3% | Very high |

#### Conditional VaR (CVaR / Expected Shortfall)
```
CVaR = E[R | R <= VaR] = Average of returns below VaR
```

**Interpretation**: Expected loss when VaR is exceeded. More conservative than VaR.

**Why CVaR is preferred**:
- Coherent risk measure (satisfies subadditivity)
- Captures tail risk better than VaR
- Required by some regulators

#### Tail Risk Metrics

**Skewness**:
```
Skewness = E[(R - mu)^3] / sigma^3
```

**Interpretation**:
- Negative skew: Heavier left tail (more extreme losses) - COMMON for short premium strategies
- Positive skew: Heavier right tail (more extreme gains)
- Zero: Symmetric distribution

**Kurtosis** (excess):
```
Kurtosis = E[(R - mu)^4] / sigma^4 - 3
```

**Interpretation**:
- Positive: Fat tails (more extreme events than normal)
- Zero: Normal distribution
- Negative: Thin tails

**Options Strategy Typical Values**:
| Strategy | Skewness | Kurtosis |
|----------|----------|----------|
| Short puts | Negative | Positive |
| Iron condors | Negative | Positive |
| Long straddles | Positive | Positive |
| Covered calls | Slightly negative | Near zero |

---

### Greeks Analysis

The analytics module tracks portfolio Greeks over time:

| Greek | Measures | Typical Range |
|-------|----------|---------------|
| Delta | Directional exposure | -1 to +1 per option |
| Gamma | Delta sensitivity | 0 to 0.1 typically |
| Theta | Time decay (daily) | Negative for long, positive for short |
| Vega | Volatility sensitivity | 0.1 to 0.5 typically |
| Rho | Interest rate sensitivity | Usually small |

**Correlation Matrix**: Shows relationships between Greeks. Important for understanding risk interactions.

---

## Usage Guide

### Basic Usage

```python
from backtester.analytics import PerformanceMetrics, RiskAnalytics

# After running backtest
results = engine.run()
equity_curve = results['equity_curve']
trade_log = results['trade_log']

# Calculate returns
returns = equity_curve['equity'].pct_change().dropna()

# Performance metrics
sharpe = PerformanceMetrics.calculate_sharpe_ratio(returns)
sortino = PerformanceMetrics.calculate_sortino_ratio(returns)
max_dd = PerformanceMetrics.calculate_max_drawdown(equity_curve)

# Risk metrics
var_95 = RiskAnalytics.calculate_var(returns, 0.95)
cvar_95 = RiskAnalytics.calculate_cvar(returns, 0.95)
tail_risk = RiskAnalytics.calculate_tail_risk(returns)

# Trade metrics (filter to closed trades)
closed_trades = trade_log[trade_log['action'] == 'close']
win_rate = PerformanceMetrics.calculate_win_rate(closed_trades)
profit_factor = PerformanceMetrics.calculate_profit_factor(closed_trades)
```

### Using BacktestEngine Integration

```python
# Run backtest
engine = BacktestEngine(strategy, data_stream, execution_model)
results = engine.run()

# Get comprehensive metrics
metrics = engine.calculate_metrics(risk_free_rate=0.02)

# Access results
print(f"Sharpe Ratio: {metrics['performance']['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {metrics['performance']['max_drawdown']:.2%}")
print(f"95% VaR: {metrics['risk']['var_95_historical']:.2%}")
print(f"Win Rate: {metrics['performance']['win_rate']:.1f}%")

# Quick summary
summary = metrics['summary']
print(f"Total Return: {summary['total_return_pct']:.2f}%")
print(f"Annualized: {summary['annualized_return']:.2%}")
```

### Calculating All Metrics at Once

```python
# Performance metrics summary
perf_summary = PerformanceMetrics.calculate_all_metrics(
    equity_curve=equity_curve,
    trades=trade_log,
    risk_free_rate=0.02
)

# Risk metrics summary
risk_summary = RiskAnalytics.calculate_all_risk_metrics(
    returns=returns,
    greeks_history=greeks_history,
    trades=trade_log,
    confidence_level=0.95
)
```

---

## Analysis Workflow

### 1. Initial Assessment

```python
# Quick health check
total_return = PerformanceMetrics.calculate_total_return(equity_curve)
sharpe = PerformanceMetrics.calculate_sharpe_ratio(returns)
max_dd = PerformanceMetrics.calculate_max_drawdown(equity_curve)

print(f"Return: {total_return:.2f}%")
print(f"Sharpe: {sharpe:.2f}")
print(f"Max DD: {max_dd['max_drawdown_pct']:.2%}")

# Is strategy viable?
if sharpe > 1.0 and max_dd['max_drawdown_pct'] > -0.20:
    print("Strategy shows promise")
```

### 2. Risk Assessment

```python
# Tail risk analysis
tail = RiskAnalytics.calculate_tail_risk(returns)
print(f"Skewness: {tail['skewness']:.2f}")
print(f"Kurtosis: {tail['kurtosis']:.2f}")

# If negative skew, check VaR/CVaR
if tail['skewness'] < -0.5:
    var = RiskAnalytics.calculate_var(returns, 0.99)
    cvar = RiskAnalytics.calculate_cvar(returns, 0.99)
    print(f"99% VaR: {var:.2%}")
    print(f"99% CVaR: {cvar:.2%}")
```

### 3. Trade Analysis

```python
# Trade quality assessment
closed = trade_log[trade_log['action'] == 'close']

win_rate = PerformanceMetrics.calculate_win_rate(closed)
profit_factor = PerformanceMetrics.calculate_profit_factor(closed)
expectancy = PerformanceMetrics.calculate_expectancy(closed)
payoff = PerformanceMetrics.calculate_payoff_ratio(closed)

print(f"Win Rate: {win_rate:.1f}%")
print(f"Profit Factor: {profit_factor:.2f}")
print(f"Expectancy: ${expectancy:.2f}")
print(f"Payoff Ratio: {payoff:.2f}")

# Kelly criterion
kelly = win_rate/100 - ((1 - win_rate/100) / payoff)
print(f"Kelly Fraction: {kelly:.2%}")
```

### 4. Greeks Risk Analysis

```python
# Analyze Greeks exposure
greeks_history = engine.get_greeks_history()
greeks_analysis = RiskAnalytics.analyze_greeks_over_time(greeks_history)

# Check delta exposure
delta_stats = greeks_analysis['delta']
print(f"Average Delta: {delta_stats['mean']:.4f}")
print(f"Max Delta: {delta_stats['max']:.4f}")
print(f"Min Delta: {delta_stats['min']:.4f}")

# Check theta decay
theta_stats = greeks_analysis['theta']
print(f"Average Daily Theta: ${theta_stats['mean'] * 100:.2f}")
```

---

## Interpretation Guidelines

### Good Strategy Characteristics

1. **Sharpe Ratio > 1.0**: Decent risk-adjusted returns
2. **Max Drawdown > -25%**: Manageable risk
3. **Profit Factor > 1.5**: Profitable with margin
4. **Positive Expectancy**: Edge per trade
5. **Calmar Ratio > 1.0**: Good return/risk trade-off

### Warning Signs

1. **Negative Skewness with Low Win Rate**: Potentially disastrous tail events
2. **High Kurtosis (>5)**: Fat tails, VaR may underestimate risk
3. **Win Rate > 90% with Low Profit Factor**: Wins small, losses catastrophic
4. **Large CVaR/VaR Gap**: Extreme tail risk
5. **High Consecutive Losses**: May trigger margin calls

### Options-Specific Considerations

1. **Short Premium Strategies**:
   - Expect negative skewness
   - Focus on CVaR, not just VaR
   - Monitor tail risk metrics closely
   - Win rate can be misleading

2. **Long Premium Strategies**:
   - Expect positive skewness
   - Win rate typically 30-45%
   - Payoff ratio should be > 2.0
   - Theta decay is your enemy

3. **Delta-Neutral Strategies**:
   - Monitor vega and gamma exposure
   - Theta is primary profit source
   - Watch for large moves

---

## API Reference

### PerformanceMetrics Methods

| Method | Input | Output |
|--------|-------|--------|
| `calculate_total_return(equity_curve)` | DataFrame/Series | float (%) |
| `calculate_annualized_return(equity_curve)` | DataFrame/Series | float (decimal) |
| `calculate_sharpe_ratio(returns, rf, periods)` | Series | float |
| `calculate_sortino_ratio(returns, rf, periods)` | Series | float |
| `calculate_calmar_ratio(cagr, max_dd)` | float, float | float |
| `calculate_max_drawdown(equity_curve)` | DataFrame/Series | Dict |
| `calculate_win_rate(trades)` | DataFrame | float (%) |
| `calculate_profit_factor(trades)` | DataFrame | float |
| `calculate_expectancy(trades)` | DataFrame | float ($) |
| `calculate_returns_distribution(returns)` | Series | Dict |

### RiskAnalytics Methods

| Method | Input | Output |
|--------|-------|--------|
| `calculate_var(returns, conf, method)` | Series | float (decimal) |
| `calculate_cvar(returns, conf)` | Series | float (decimal) |
| `calculate_tail_risk(returns)` | Series | Dict |
| `calculate_downside_risk(returns)` | Series | Dict |
| `analyze_greeks_over_time(greeks_history)` | DataFrame | Dict |
| `calculate_mae(trades)` | DataFrame | DataFrame |

---

## Testing

Run analytics tests:
```bash
pytest tests/test_analytics.py -v
```

Test coverage includes:
- Known value validation
- Edge cases (empty data, single point, all wins/losses)
- Numerical stability
- Mathematical correctness
- Boundary conditions

---

## References

1. Sharpe, W.F. (1994). "The Sharpe Ratio." Journal of Portfolio Management.
2. Sortino, F.A. & van der Meer, R. (1991). "Downside Risk." Journal of Portfolio Management.
3. Jorion, P. (2007). "Value at Risk: The New Benchmark for Managing Financial Risk."
4. Acerbi, C. & Tasche, D. (2002). "Expected Shortfall: A Natural Coherent Alternative to VaR."
5. Hull, J.C. (2018). "Options, Futures, and Other Derivatives."
