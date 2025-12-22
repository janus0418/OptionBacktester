# Analytics Guide

## Overview

This guide explains all 30+ performance and risk metrics available in the Options Backtesting System.

## Performance Metrics

### Total Return
```python
total_return = PerformanceMetrics.calculate_total_return(equity_curve)
```
**Formula:** (Final Equity - Initial Equity) / Initial Equity

**Interpretation:**
- 0.10 = 10% return
- Negative values indicate losses

### Annualized Return (CAGR)
```python
annualized_return = PerformanceMetrics.calculate_annualized_return(equity_curve)
```
**Formula:** (Final / Initial)^(252 / num_days) - 1

**Interpretation:** Compound annual growth rate

### Sharpe Ratio
```python
sharpe = PerformanceMetrics.calculate_sharpe_ratio(returns, risk_free_rate=0.04)
```
**Formula:** (Mean Return - Risk Free Rate) / Std Dev of Returns × √252

**Interpretation:**
- < 0: Poor (losing money)
- 0-1: Acceptable
- 1-2: Good
- > 2: Excellent

**Typical Values:**
- S&P 500: 0.5-0.8
- Hedge funds: 1.0-1.5
- Top quant strategies: 2.0+

### Sortino Ratio
```python
sortino = PerformanceMetrics.calculate_sortino_ratio(returns, risk_free_rate=0.04)
```
**Formula:** (Mean Return - Risk Free Rate) / Downside Deviation × √252

**Interpretation:** Like Sharpe but only penalizes downside volatility
- Generally higher than Sharpe for strategies with positive skew
- More relevant for options strategies (asymmetric payoffs)

### Calmar Ratio
```python
calmar = PerformanceMetrics.calculate_calmar_ratio(equity_curve, returns)
```
**Formula:** Annualized Return / Absolute Max Drawdown

**Interpretation:**
- < 0.5: Poor
- 0.5-1.0: Acceptable
- 1.0-3.0: Good
- > 3.0: Excellent

### Maximum Drawdown
```python
max_dd = PerformanceMetrics.calculate_max_drawdown(equity_curve)
```
**Formula:** (Trough - Peak) / Peak (largest peak-to-trough decline)

**Interpretation:**
- -0.10 = 10% drawdown
- More negative = worse
- Key risk metric for capital preservation

**Typical Values:**
- Conservative strategies: -10% to -20%
- Aggressive strategies: -30% to -50%
- S&P 500 (2008): -55%

## Trade-Based Metrics

### Win Rate
```python
win_rate = PerformanceMetrics.calculate_win_rate(trade_log)
```
**Formula:** Winning Trades / Total Trades

**Interpretation:**
- 0.60 = 60% win rate
- High win rate ≠ profitability!
- Options strategies often have 60-80% win rates but small wins

**Typical Values by Strategy:**
- Credit spreads: 60-80%
- Directional trades: 40-60%
- Straddles/strangles: 50-70%

### Profit Factor
```python
profit_factor = PerformanceMetrics.calculate_profit_factor(trade_log)
```
**Formula:** Gross Profit / Absolute Gross Loss

**Interpretation:**
- < 1.0: Unprofitable (losing more than gaining)
- 1.0-1.5: Marginal
- 1.5-2.0: Good
- > 2.0: Excellent

**Example:** $10,000 wins / $5,000 losses = 2.0 profit factor

### Expectancy
```python
expectancy = PerformanceMetrics.calculate_expectancy(trade_log)
```
**Formula:** (Win Rate × Avg Win) - (Loss Rate × Avg Loss)

**Interpretation:** Expected value per trade
- Positive = profitable system
- $100 expectancy = expect to make $100 per trade on average

### Payoff Ratio
```python
payoff_ratio = PerformanceMetrics.calculate_payoff_ratio(trade_log)
```
**Formula:** Average Win / Average Loss

**Interpretation:**
- 2.0 = average win is 2× average loss
- Can be low for high win rate strategies
- Low payoff ratio + high win rate = premium selling
- High payoff ratio + low win rate = directional/long options

### Average Win/Loss
```python
avg_win = PerformanceMetrics.calculate_average_win(trade_log)
avg_loss = PerformanceMetrics.calculate_average_loss(trade_log)
```
**Usage:** Understanding P&L distribution

## Risk Metrics

### Value at Risk (VaR)
```python
var_95 = RiskAnalytics.calculate_var(returns, confidence=0.95)
var_99 = RiskAnalytics.calculate_var(returns, confidence=0.99)
```
**Interpretation:**
- VaR(95%) = -0.02 means "95% confident daily loss won't exceed 2%"
- VaR(99%) = -0.04 means "99% confident daily loss won't exceed 4%"

**Methods:**
- Historical: Uses actual return distribution
- Parametric: Assumes normal distribution

### Conditional VaR (CVaR / Expected Shortfall)
```python
cvar_95 = RiskAnalytics.calculate_cvar(returns, confidence=0.95)
```
**Interpretation:** Expected loss when VaR is exceeded
- CVaR(95%) = -0.05 means "when worst 5% of days occur, expect 5% loss"
- Always worse (more negative) than VaR
- Better risk measure than VaR (tail risk)

### Tail Risk Metrics
```python
tail_metrics = RiskAnalytics.calculate_tail_risk(returns)
# Returns: {'skewness': float, 'kurtosis': float, 'tail_ratio': float}
```

**Skewness:**
- 0 = symmetric distribution
- < 0 = negative skew (long left tail, bad for long positions)
- > 0 = positive skew (long right tail, good for long positions)

**Kurtosis:**
- 3 = normal distribution
- > 3 = fat tails (more extreme events than normal)
- < 3 = thin tails

**Options strategies typically have:**
- Short premium: Negative skew (small wins, occasional large losses)
- Long premium: Positive skew (small losses, occasional large wins)

## Greeks Analysis

### Delta Exposure
```python
delta_history = results['greeks_history']['delta']
avg_delta = delta_history.mean()
max_delta = delta_history.max()
```
**Interpretation:**
- Positive delta: Long exposure (profit from up moves)
- Negative delta: Short exposure (profit from down moves)
- Zero delta: Market neutral

**Typical Ranges:**
- Neutral strategies: -10 to +10
- Directional strategies: ±50 to ±100

### Gamma Exposure
```python
gamma_history = results['greeks_history']['gamma']
```
**Interpretation:**
- Positive gamma: Delta increases as market moves (long options)
- Negative gamma: Delta decreases as market moves (short options)

**Risks:**
- Negative gamma = larger losses on big moves
- Requires more active management

### Theta (Time Decay)
```python
theta_history = results['greeks_history']['theta']
```
**Interpretation:**
- Negative theta: Lose money from time decay (long options)
- Positive theta: Make money from time decay (short options)

**Daily P&L from theta:**
- Theta = 100 means making ~$100/day from time decay
- Theta = -100 means losing ~$100/day from time decay

### Vega Exposure
```python
vega_history = results['greeks_history']['vega']
```
**Interpretation:**
- Positive vega: Profit from IV increase (long options)
- Negative vega: Profit from IV decrease (short options)

**Typical Strategies:**
- Short premium: Negative vega (hurt by IV spikes)
- Long premium: Positive vega (benefit from IV spikes)

## Advanced Metrics

### Drawdown Duration
```python
dd_duration = PerformanceMetrics.calculate_drawdown_duration(equity_curve)
```
**Interpretation:** Time from peak to trough
- Longer durations = harder to hold psychologically
- Important for strategy persistence

### Recovery Time
```python
recovery = PerformanceMetrics.calculate_recovery_time(equity_curve)
```
**Interpretation:** Time from trough back to peak
- Measures how quickly strategy recovers from losses

### Consecutive Wins/Losses
```python
max_consecutive_wins = PerformanceMetrics.calculate_consecutive_wins(trade_log)
max_consecutive_losses = PerformanceMetrics.calculate_consecutive_losses(trade_log)
```
**Interpretation:** Longest winning/losing streaks
- Helps understand variance and psychological difficulty
- Longer losing streaks = harder to trade

### Ulcer Index
```python
ulcer_index = PerformanceMetrics.calculate_ulcer_index(equity_curve)
```
**Formula:** RMS of all drawdowns
**Interpretation:** Measures depth and duration of drawdowns
- Lower is better
- Alternative to standard deviation for downside risk

## Benchmarking

### Compare to Benchmarks
```python
# Compare strategy to buy-and-hold
strategy_sharpe = 1.5
spy_sharpe = 0.7  # Typical S&P 500 Sharpe

# Strategy is 1.5/0.7 = 2.14× better on risk-adjusted basis
```

### Industry Benchmarks

**Options Premium Selling:**
- Win rate: 60-80%
- Sharpe: 0.8-1.5
- Max DD: -20% to -40%
- Profit factor: 1.3-2.0

**Directional Options:**
- Win rate: 35-50%
- Sharpe: 0.5-1.2
- Max DD: -30% to -60%
- Profit factor: 1.5-3.0

**Market Neutral:**
- Win rate: 50-65%
- Sharpe: 1.0-2.0
- Max DD: -15% to -30%
- Profit factor: 1.5-2.5

## Usage Examples

### Calculate All Metrics
```python
from backtester.analytics import PerformanceMetrics, RiskAnalytics

# After running backtest
results = engine.run()
equity_curve = results['equity_curve']
trade_log = results['trade_log']
returns = equity_curve['equity'].pct_change().dropna()

# Performance
metrics = {
    'total_return': PerformanceMetrics.calculate_total_return(equity_curve),
    'sharpe_ratio': PerformanceMetrics.calculate_sharpe_ratio(returns),
    'sortino_ratio': PerformanceMetrics.calculate_sortino_ratio(returns),
    'calmar_ratio': PerformanceMetrics.calculate_calmar_ratio(equity_curve, returns),
    'max_drawdown': PerformanceMetrics.calculate_max_drawdown(equity_curve),
}

# Risk
metrics.update({
    'var_95': RiskAnalytics.calculate_var(returns, 0.95),
    'cvar_95': RiskAnalytics.calculate_cvar(returns, 0.95),
    'tail_risk': RiskAnalytics.calculate_tail_risk(returns),
})

# Trades
metrics.update({
    'win_rate': PerformanceMetrics.calculate_win_rate(trade_log),
    'profit_factor': PerformanceMetrics.calculate_profit_factor(trade_log),
    'expectancy': PerformanceMetrics.calculate_expectancy(trade_log),
})
```

### Print Metrics Report
```python
print(f"Performance:")
print(f"  Total Return: {metrics['total_return']*100:.2f}%")
print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"  Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
print(f"\nTrade Statistics:")
print(f"  Win Rate: {metrics['win_rate']*100:.1f}%")
print(f"  Profit Factor: {metrics['profit_factor']:.2f}")
```

## Interpreting Results

### Good Strategy Profile
- Sharpe ratio > 1.0
- Profit factor > 1.5
- Max drawdown < -30%
- Consistent performance (low variance of monthly returns)
- Recovery time < drawdown duration

### Warning Signs
- High win rate but low profit factor (few large losses)
- High Sharpe but large max drawdown (infrequent disasters)
- Extreme positive skew in short periods (lucky run)
- Deteriorating metrics over time (regime change)

## Additional Resources

- See `/examples/example_05_full_analysis.py` for comprehensive metrics calculation
- See `/code/tests/test_analytics.py` for metric validation tests
- API Reference: `API_REFERENCE.md`
