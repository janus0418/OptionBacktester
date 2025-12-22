# Strategy Development Guide

## Table of Contents

1. [Overview](#overview)
2. [Strategy Base Class](#strategy-base-class)
3. [Implementing Custom Strategies](#implementing-custom-strategies)
4. [Best Practices](#best-practices)
5. [Common Patterns](#common-patterns)
6. [Testing Strategies](#testing-strategies)

## Overview

This guide teaches you how to develop custom options trading strategies using the Options Backtesting System.

## Strategy Base Class

All strategies inherit from the `Strategy` base class:

```python
from backtester.strategies import Strategy
from backtester.core.option_structure import OptionStructure
from datetime import datetime
from typing import Optional

class MyStrategy(Strategy):
    def __init__(self, name: str, initial_capital: float, **kwargs):
        super().__init__(name, initial_capital)
        # Add custom parameters
        self.custom_param = kwargs.get('custom_param', default_value)

    def should_enter(
        self,
        current_date: datetime,
        market_data: MarketData,
        option_chain: OptionChain,
        available_capital: float
    ) -> bool:
        """
        Determine if we should enter a new position.

        Args:
            current_date: Current trading date
            market_data: Current market data (open, high, low, close, volume)
            option_chain: Available options with prices and Greeks
            available_capital: Capital available for new positions

        Returns:
            True if we should enter, False otherwise
        """
        # Implement your entry logic
        return False

    def should_exit(
        self,
        position: OptionStructure,
        current_date: datetime,
        market_data: MarketData,
        option_chain: Optional[OptionChain]
    ) -> bool:
        """
        Determine if we should exit an existing position.

        Args:
            position: The OptionStructure to potentially exit
            current_date: Current trading date
            market_data: Current market data
            option_chain: Current option chain (may be None)

        Returns:
            True if we should exit, False otherwise
        """
        # Implement your exit logic
        return False

    def create_structure(
        self,
        current_date: datetime,
        market_data: MarketData,
        option_chain: OptionChain,
        available_capital: float
    ) -> Optional[OptionStructure]:
        """
        Create the option structure to enter.

        Args:
            current_date: Current trading date
            market_data: Current market data
            option_chain: Available options
            available_capital: Capital available

        Returns:
            OptionStructure to enter, or None if unable to create
        """
        # Implement your structure creation logic
        return None
```

## Implementing Custom Strategies

### Example: High IV Straddle Seller

```python
from backtester.strategies import Strategy
from backtester.structures import ShortStraddle
import numpy as np

class HighIVStraddleSeller(Strategy):
    def __init__(
        self,
        name: str,
        initial_capital: float,
        iv_percentile_threshold: float = 75.0,
        profit_target_pct: float = 0.50,
        stop_loss_pct: float = 2.0,
        max_positions: int = 3
    ):
        super().__init__(name, initial_capital)
        self.iv_percentile_threshold = iv_percentile_threshold
        self.profit_target_pct = profit_target_pct
        self.stop_loss_pct = stop_loss_pct
        self.max_positions = max_positions

        # Track IV history for percentile calculation
        self.iv_history = []

    def should_enter(self, current_date, market_data, option_chain, available_capital):
        # Don't enter if at position limit
        if len(self.positions) >= self.max_positions:
            return False

        # Calculate current IV (average of ATM options)
        atm_calls = sorted(
            option_chain.calls,
            key=lambda x: abs(x['strike'] - market_data.close)
        )[:3]

        current_iv = np.mean([c['implied_volatility'] for c in atm_calls])

        # Update IV history
        self.iv_history.append(current_iv)
        if len(self.iv_history) > 252:  # Keep 1 year
            self.iv_history = self.iv_history[-252:]

        # Check if IV is in high percentile
        if len(self.iv_history) < 60:  # Need history
            return False

        iv_percentile = (
            sum(1 for iv in self.iv_history if iv < current_iv) /
            len(self.iv_history) * 100
        )

        # Enter when IV is high
        return iv_percentile >= self.iv_percentile_threshold

    def should_exit(self, position, current_date, market_data, option_chain):
        # Exit near expiration
        if position.days_to_expiration <= 7:
            return True

        # Profit target
        max_profit = position.max_profit
        if max_profit > 0:
            profit_target = max_profit * self.profit_target_pct
            if position.pnl >= profit_target:
                return True

        # Stop loss
        stop_loss = -abs(max_profit) * self.stop_loss_pct
        if position.pnl <= stop_loss:
            return True

        return False

    def create_structure(self, current_date, market_data, option_chain, available_capital):
        # Find ATM strike
        atm_strike = min(
            option_chain.calls,
            key=lambda x: abs(x['strike'] - market_data.close)
        )['strike']

        # Get call and put prices
        call_data = next(
            (c for c in option_chain.calls if c['strike'] == atm_strike),
            None
        )
        put_data = next(
            (p for p in option_chain.puts if p['strike'] == atm_strike),
            None
        )

        if not call_data or not put_data:
            return None

        # Calculate position size
        estimated_margin = 5000  # Simplified
        contracts = max(1, int(available_capital * 0.20 / estimated_margin))

        # Create short straddle
        return ShortStraddle.create(
            underlying=market_data.underlying,
            strike=atm_strike,
            expiration=option_chain.expiration,
            call_price=call_data['mid'],
            put_price=put_data['mid'],
            quantity=contracts,
            entry_date=current_date,
            underlying_price=market_data.close
        )
```

## Best Practices

### 1. Entry Conditions

**DO:**
- Use multiple confirmation signals
- Check capital availability
- Validate data quality
- Respect position limits

**DON'T:**
- Enter on single indicator
- Ignore risk limits
- Skip data validation
- Over-leverage

```python
def should_enter(self, current_date, market_data, option_chain, available_capital):
    # Multiple confirmations
    if not self._check_iv_condition(option_chain):
        return False
    if not self._check_price_condition(market_data):
        return False
    if not self._check_capital_available(available_capital):
        return False
    if not self._check_position_limits():
        return False

    return True
```

### 2. Exit Conditions

**DO:**
- Always have maximum holding period
- Implement profit targets AND stop losses
- Handle edge cases (expiration, illiquid options)
- Consider time decay

**DON'T:**
- Rely on single exit criterion
- Hold to expiration without check
- Ignore Greeks changes

```python
def should_exit(self, position, current_date, market_data, option_chain):
    # Time-based exits
    if position.days_to_expiration <= self.min_dte:
        return True

    # P&L-based exits
    if self._hit_profit_target(position):
        return True
    if self._hit_stop_loss(position):
        return True

    # Greeks-based exits
    if self._greeks_limit_breached(position):
        return True

    return False
```

### 3. Position Sizing

**Risk-Based Sizing:**

```python
def calculate_position_size(self, available_capital, max_risk_pct=0.05):
    """Size position based on maximum acceptable risk."""
    max_risk = available_capital * max_risk_pct

    # Estimate maximum loss for the structure
    estimated_max_loss = self._estimate_max_loss()

    # Calculate contracts
    contracts = max(1, int(max_risk / abs(estimated_max_loss)))

    return contracts
```

**Capital-Based Sizing:**

```python
def calculate_position_size(self, available_capital, allocation_pct=0.20):
    """Size position based on capital allocation."""
    target_allocation = available_capital * allocation_pct

    # Estimate margin requirement
    estimated_margin = self._estimate_margin()

    contracts = max(1, int(target_allocation / estimated_margin))

    return contracts
```

### 4. State Management

```python
class StatefulStrategy(Strategy):
    def __init__(self, name, initial_capital):
        super().__init__(name, initial_capital)

        # Track strategy state
        self.market_regime = None
        self.price_history = []
        self.volatility_history = []
        self.signal_history = []

    def update_state(self, market_data, option_chain):
        """Update strategy state with new data."""
        self.price_history.append(market_data.close)
        self.volatility_history.append(self._calc_iv(option_chain))

        # Keep limited history
        if len(self.price_history) > 252:
            self.price_history = self.price_history[-252:]
            self.volatility_history = self.volatility_history[-252:]

        # Update regime
        self.market_regime = self._determine_regime()

    def should_enter(self, current_date, market_data, option_chain, available_capital):
        # Update state first
        self.update_state(market_data, option_chain)

        # Make decision based on state
        if self.market_regime == 'high_vol':
            return self._high_vol_entry_logic()
        elif self.market_regime == 'low_vol':
            return self._low_vol_entry_logic()

        return False
```

## Common Patterns

### Pattern 1: Mean Reversion

```python
def should_enter(self, current_date, market_data, option_chain, available_capital):
    # Calculate z-score
    sma = np.mean(self.price_history)
    std = np.std(self.price_history)
    z_score = (market_data.close - sma) / std if std > 0 else 0

    # Enter when price is extended
    return abs(z_score) > 2.0
```

### Pattern 2: Trend Following

```python
def should_enter(self, current_date, market_data, option_chain, available_capital):
    # Calculate moving averages
    sma_short = np.mean(self.price_history[-20:])
    sma_long = np.mean(self.price_history[-50:])

    # Enter on trend confirmation
    return sma_short > sma_long  # Bullish trend
```

### Pattern 3: Volatility Regime

```python
def should_enter(self, current_date, market_data, option_chain, available_capital):
    current_iv = self._calc_current_iv(option_chain)

    # Determine regime
    if current_iv < 0.15:
        return False  # Don't sell premium when IV too low
    elif current_iv > 0.30:
        return True  # Sell premium when IV high
    else:
        # Medium regime - additional filters
        return self._additional_checks()
```

### Pattern 4: Multi-Timeframe

```python
def should_enter(self, current_date, market_data, option_chain, available_capital):
    # Long-term trend
    long_term_bullish = self._check_long_term_trend() > 0

    # Medium-term momentum
    medium_term_up = self._check_medium_term_momentum() > 0

    # Short-term setup
    short_term_setup = self._check_short_term_setup()

    # All timeframes must align
    return long_term_bullish and medium_term_up and short_term_setup
```

## Testing Strategies

### Unit Testing

```python
import pytest
from datetime import datetime

def test_entry_conditions():
    strategy = MyStrategy(
        name='Test',
        initial_capital=100000.0,
        iv_threshold=70
    )

    # Create mock data
    market_data = create_mock_market_data()
    option_chain = create_mock_option_chain()

    # Test entry logic
    should_enter = strategy.should_enter(
        current_date=datetime(2024, 1, 15),
        market_data=market_data,
        option_chain=option_chain,
        available_capital=100000.0
    )

    assert isinstance(should_enter, bool)
```

### Backtest Validation

```python
def validate_strategy():
    # Run backtest
    strategy = MyStrategy(...)
    engine = BacktestEngine(strategy, data_stream, execution, 100000.0)
    results = engine.run()

    # Validate results
    assert results['final_equity'] > 0
    assert len(results['trade_log']) > 0
    assert results['total_return'] != 0

    # Check all positions were closed
    assert len(strategy.positions) == 0
```

## Performance Optimization

### 1. Vectorize Calculations

```python
# Slow
for option in option_chain.calls:
    if self._check_condition(option):
        selected.append(option)

# Fast
strikes = np.array([o['strike'] for o in option_chain.calls])
mask = (strikes >= lower_bound) & (strikes <= upper_bound)
selected = [o for o, m in zip(option_chain.calls, mask) if m]
```

### 2. Cache Expensive Calculations

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def _calculate_expensive_metric(self, date_key):
    # Expensive calculation
    return result
```

### 3. Limit Historical Data

```python
# Keep only necessary history
MAX_HISTORY = 252  # 1 year

self.price_history.append(price)
if len(self.price_history) > MAX_HISTORY:
    self.price_history = self.price_history[-MAX_HISTORY:]
```

## Examples

See `/examples/example_04_custom_strategy.py` for a complete working example of a custom mean reversion strategy.

## Additional Resources

- **API Reference:** See `API_REFERENCE.md` for full API documentation
- **User Guide:** See `USER_GUIDE.md` for basic concepts
- **Test Files:** See `/code/tests/test_example_strategies.py` for strategy tests
- **Built-in Strategies:** See `/code/backtester/strategies/` for examples
