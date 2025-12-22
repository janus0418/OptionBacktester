# Data Integration Guide

## Overview

This guide explains how to integrate various data sources with the Options Backtesting System.

## Table of Contents

1. [Dolt Database](#dolt-database)
2. [Mock Data (Testing)](#mock-data-testing)
3. [CSV Files](#csv-files)
4. [External APIs](#external-apis)
5. [Data Quality](#data-quality)
6. [Handling Missing Data](#handling-missing-data)

## Dolt Database

### Setup Dolt

```bash
# Install Dolt
curl -L https://github.com/dolthub/dolt/releases/latest/download/install.sh | bash

# Create database
dolt init
dolt sql -q "CREATE DATABASE options_data"

# Start SQL server
dolt sql-server --host 0.0.0.0 --port 3306
```

### Connect to Dolt

```python
from backtester.data.dolt_adapter import DoltAdapter

adapter = DoltAdapter(
    database='options_data',
    host='localhost',
    port=3306,
    user='root',
    password=''
)

# Connect
adapter.connect()

# Verify connection
if adapter.is_connected():
    print("Connected to Dolt!")

# Use in DataStream
from backtester.engine.data_stream import DataStream
from datetime import datetime

data_stream = DataStream(
    adapter=adapter,
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 12, 31),
    underlying='SPY'
)
```

### Required Database Schema

**Market Data Table:**
```sql
CREATE TABLE market_data (
    symbol VARCHAR(10),
    date DATE,
    open DECIMAL(10, 2),
    high DECIMAL(10, 2),
    low DECIMAL(10, 2),
    close DECIMAL(10, 2),
    volume BIGINT,
    PRIMARY KEY (symbol, date)
);
```

**Options Data Table:**
```sql
CREATE TABLE options_data (
    symbol VARCHAR(10),
    quote_date DATE,
    expiration DATE,
    strike DECIMAL(10, 2),
    option_type VARCHAR(4),  -- 'call' or 'put'
    bid DECIMAL(10, 4),
    ask DECIMAL(10, 4),
    last DECIMAL(10, 4),
    volume INT,
    open_interest INT,
    implied_volatility DECIMAL(6, 4),
    delta DECIMAL(6, 4),
    gamma DECIMAL(8, 6),
    theta DECIMAL(10, 4),
    vega DECIMAL(10, 4),
    rho DECIMAL(10, 4),
    PRIMARY KEY (symbol, quote_date, expiration, strike, option_type)
);
```

## Mock Data (Testing)

For testing without a database, use mock data:

```python
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from unittest.mock import Mock

from backtester.engine.data_stream import DataStream
from backtester.data.market_data import MarketData, OptionChain
from backtester.core.pricing import BlackScholesPricer

def create_mock_data_stream(start_date, end_date):
    """Create DataStream with mock data."""
    # Create mock adapter
    mock_adapter = Mock()
    mock_adapter.is_connected.return_value = True

    # Create DataStream
    data_stream = DataStream(mock_adapter, start_date, end_date, 'SPY')

    # Generate price data
    dates = pd.bdate_range(start_date, end_date)
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.015, len(dates))
    prices = 450.0 * np.exp(np.cumsum(returns))

    price_df = pd.DataFrame({'date': dates, 'close': prices})

    def mock_get_market_data(date):
        row = price_df[price_df['date'] == pd.Timestamp(date)]
        if row.empty:
            return None
        close = row['close'].iloc[0]
        return MarketData('SPY', date, close*0.999, close*1.001,
                         close*0.998, close, 50_000_000)

    def mock_get_option_chain(date, expiration=None):
        market_data = mock_get_market_data(date)
        if not market_data:
            return None
        if not expiration:
            expiration = date + timedelta(days=30)

        # Generate option chain
        pricer = BlackScholesPricer()
        S = market_data.close
        T = (expiration - date).days / 365.25
        r = 0.04
        sigma = 0.20

        calls, puts = [], []
        for pct in range(90, 111, 1):
            K = round(S * pct / 100, 2)

            call_price = pricer.price(S, K, T, r, sigma, 'call')
            put_price = pricer.price(S, K, T, r, sigma, 'put')

            call_greeks = pricer.calculate_greeks(S, K, T, r, sigma, 'call')
            put_greeks = pricer.calculate_greeks(S, K, T, r, sigma, 'put')

            calls.append({
                'strike': K, 'mid': call_price, 'bid': call_price*0.98,
                'ask': call_price*1.02, 'implied_volatility': sigma,
                'volume': 1000, 'open_interest': 5000, **call_greeks
            })
            puts.append({
                'strike': K, 'mid': put_price, 'bid': put_price*0.98,
                'ask': put_price*1.02, 'implied_volatility': sigma,
                'volume': 1000, 'open_interest': 5000, **put_greeks
            })

        return OptionChain('SPY', date, expiration, calls, puts)

    # Patch methods
    data_stream.get_market_data = mock_get_market_data
    data_stream.get_option_chain = mock_get_option_chain

    return data_stream

# Usage
data_stream = create_mock_data_stream(
    datetime(2024, 1, 1),
    datetime(2024, 12, 31)
)
```

See `/examples/example_01_simple_backtest.py` for complete mock data implementation.

## CSV Files

### Load from CSV

```python
import pandas as pd
from datetime import datetime

def load_market_data_from_csv(csv_path):
    """Load market data from CSV file."""
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    return df

def create_csv_data_stream(market_csv, options_csv, underlying, start, end):
    """Create DataStream from CSV files."""
    from unittest.mock import Mock

    market_df = load_market_data_from_csv(market_csv)
    options_df = pd.read_csv(options_csv)
    options_df['quote_date'] = pd.to_datetime(options_df['quote_date'])
    options_df['expiration'] = pd.to_datetime(options_df['expiration'])

    mock_adapter = Mock()
    mock_adapter.is_connected.return_value = True

    data_stream = DataStream(mock_adapter, start, end, underlying)

    def get_market_data(date):
        row = market_df[
            (market_df['symbol'] == underlying) &
            (market_df['date'] == pd.Timestamp(date))
        ]
        if row.empty:
            return None
        return MarketData(
            underlying, date, row['open'].iloc[0], row['high'].iloc[0],
            row['low'].iloc[0], row['close'].iloc[0], row['volume'].iloc[0]
        )

    def get_option_chain(date, expiration=None):
        market_data = get_market_data(date)
        if not market_data:
            return None

        # Filter options data
        chain_df = options_df[
            (options_df['symbol'] == underlying) &
            (options_df['quote_date'] == pd.Timestamp(date))
        ]
        if expiration:
            chain_df = chain_df[chain_df['expiration'] == pd.Timestamp(expiration)]
        else:
            # Use first available expiration
            if len(chain_df) == 0:
                return None
            expiration = chain_df['expiration'].iloc[0]

        calls = chain_df[chain_df['option_type'] == 'call'].to_dict('records')
        puts = chain_df[chain_df['option_type'] == 'put'].to_dict('records')

        return OptionChain(underlying, date, expiration, calls, puts)

    data_stream.get_market_data = get_market_data
    data_stream.get_option_chain = get_option_chain

    return data_stream
```

## External APIs

### Yahoo Finance (yfinance)

```python
import yfinance as yf
from datetime import datetime

def fetch_market_data_yfinance(symbol, start_date, end_date):
    """Fetch market data from Yahoo Finance."""
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start_date, end=end_date)

    # Convert to required format
    df = df.reset_index()
    df.columns = df.columns.str.lower()
    df = df.rename(columns={'date': 'date', 'volume': 'volume'})

    return df

# Note: Yahoo Finance doesn't provide options historical data easily
# For options data, consider:
# - QuantConnect
# - Interactive Brokers API
# - ThetaData
# - CBOE DataShop
```

### Custom API Integration

```python
import requests

def fetch_from_api(endpoint, symbol, date):
    """Fetch data from custom API."""
    response = requests.get(
        f"{endpoint}/market_data",
        params={'symbol': symbol, 'date': date.strftime('%Y-%m-%d')}
    )

    if response.status_code == 200:
        return response.json()
    return None

def create_api_data_stream(api_endpoint, underlying, start, end):
    """Create DataStream using API."""
    from unittest.mock import Mock

    mock_adapter = Mock()
    mock_adapter.is_connected.return_value = True
    data_stream = DataStream(mock_adapter, start, end, underlying)

    def get_market_data(date):
        data = fetch_from_api(api_endpoint, underlying, date)
        if not data:
            return None
        return MarketData(
            underlying, date, data['open'], data['high'],
            data['low'], data['close'], data['volume']
        )

    data_stream.get_market_data = get_market_data
    # Implement get_option_chain similarly

    return data_stream
```

## Data Quality

### Validation Checks

```python
from backtester.data.data_validator import DataValidator

def validate_market_data(market_data):
    """Validate market data quality."""
    checks = []

    # Price sanity checks
    if market_data.close <= 0:
        checks.append("ERROR: Close price is non-positive")

    if market_data.high < market_data.low:
        checks.append("ERROR: High < Low")

    if market_data.close > market_data.high or market_data.close < market_data.low:
        checks.append("WARNING: Close outside High-Low range")

    if market_data.volume < 0:
        checks.append("ERROR: Negative volume")

    return checks

def validate_option_chain(option_chain):
    """Validate option chain data."""
    checks = []

    # Check for required fields
    for call in option_chain.calls:
        if call['bid'] > call['ask']:
            checks.append(f"ERROR: Bid > Ask for call strike {call['strike']}")

        if call['implied_volatility'] <= 0:
            checks.append(f"ERROR: Invalid IV for call strike {call['strike']}")

        # Delta range check
        if not (0 <= call['delta'] <= 1):
            checks.append(f"WARNING: Delta out of range for call {call['strike']}")

    for put in option_chain.puts:
        if put['bid'] > put['ask']:
            checks.append(f"ERROR: Bid > Ask for put strike {put['strike']}")

        if not (-1 <= put['delta'] <= 0):
            checks.append(f"WARNING: Delta out of range for put {put['strike']}")

    return checks
```

### Data Cleaning

```python
def clean_option_chain(option_chain):
    """Remove invalid options from chain."""
    valid_calls = []
    for call in option_chain.calls:
        # Skip if bid/ask spread > 100%
        if call['ask'] > 0 and (call['ask'] - call['bid']) / call['mid'] < 1.0:
            # Skip if no volume or OI
            if call['volume'] > 0 or call['open_interest'] > 0:
                valid_calls.append(call)

    valid_puts = []
    for put in option_chain.puts:
        if put['ask'] > 0 and (put['ask'] - put['bid']) / put['mid'] < 1.0:
            if put['volume'] > 0 or put['open_interest'] > 0:
                valid_puts.append(put)

    return OptionChain(
        option_chain.underlying,
        option_chain.date,
        option_chain.expiration,
        valid_calls,
        valid_puts
    )
```

## Handling Missing Data

### Forward Fill

```python
def handle_missing_dates(data_stream, current_date):
    """Handle missing data by forward-filling last known value."""
    market_data = data_stream.get_market_data(current_date)

    if market_data is None:
        # Try previous trading day
        from datetime import timedelta
        prev_date = current_date - timedelta(days=1)
        market_data = data_stream.get_market_data(prev_date)

        if market_data:
            # Use previous day's close as current day's prices
            return MarketData(
                market_data.underlying,
                current_date,
                market_data.close,
                market_data.close,
                market_data.close,
                market_data.close,
                0  # No volume
            )

    return market_data
```

### Skip Missing Days

```python
def run_backtest_skip_missing(engine):
    """Run backtest, skipping days with missing data."""
    results = []

    for date in engine.trading_calendar:
        market_data = engine.data_stream.get_market_data(date)

        if market_data is None:
            # Skip this day
            continue

        # Process day
        # ...

    return results
```

## Best Practices

1. **Validate Data Quality:**
   - Check for gaps in data
   - Verify price/volume sanity
   - Validate Greeks ranges
   - Check bid-ask spreads

2. **Handle Corporate Actions:**
   - Adjust for splits
   - Adjust for dividends
   - Handle symbol changes

3. **Time Zones:**
   - Use consistent timezone (UTC or market timezone)
   - Handle market holidays correctly

4. **Data Storage:**
   - Use efficient storage (Parquet, HDF5 for large datasets)
   - Index by date and symbol
   - Compress historical data

5. **Performance:**
   - Cache frequently accessed data
   - Use database indexes
   - Batch queries when possible

## Example: Complete Data Pipeline

```python
from datetime import datetime
from backtester.data.dolt_adapter import DoltAdapter
from backtester.engine.data_stream import DataStream

# 1. Connect to database
adapter = DoltAdapter(database='options_data')
adapter.connect()

# 2. Verify connection
if not adapter.is_connected():
    raise RuntimeError("Database connection failed")

# 3. Create data stream
data_stream = DataStream(
    adapter=adapter,
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 12, 31),
    underlying='SPY'
)

# 4. Validate data
test_date = datetime(2024, 1, 15)
market_data = data_stream.get_market_data(test_date)

if market_data:
    validation_issues = validate_market_data(market_data)
    if validation_issues:
        for issue in validation_issues:
            print(f"Data quality issue: {issue}")

# 5. Use in backtest
from backtester.engine.backtest_engine import BacktestEngine

engine = BacktestEngine(strategy, data_stream, execution, 100000.0)
results = engine.run()

# 6. Cleanup
adapter.disconnect()
```

## Additional Resources

- See `/examples/` for complete data integration examples
- See `/code/backtester/data/` for data module source code
- Dolt documentation: https://docs.dolthub.com/
