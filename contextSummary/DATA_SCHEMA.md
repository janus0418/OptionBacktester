# DATA_SCHEMA.md - Dolt Database Schema and Data Layer Documentation

**Last Updated**: December 15, 2025
**Database Source**: DoltHub `post-no-preference/options`
**Status**: ✅ Schema Verified, ⚠️ Implementation Has Issues

---

## 1. Actual Database Schema (VERIFIED)

### 1.1 Database Information
- **Source**: `post-no-preference/options` from DoltHub
- **Data Range**: February 9, 2019 to December 12, 2025
- **Total Symbols**: 2,227 (including SPY, AAPL, QQQ, etc.)
- **Database Size**: ~5 million chunks (very large)
- **Access Method**: Dolt CLI with doltpy Python library

### 1.2 Tables

#### Table 1: `option_chain`
Historical options chain data with quotes and Greeks.

**Columns** (verified from implementation):
```sql
- date                 DATE           # Quote date
- act_symbol          VARCHAR        # Underlying ticker (e.g., 'SPY')
- expiration          DATE           # Option expiration date
- strike              DECIMAL        # Strike price
- call_put            VARCHAR        # 'call' or 'put'
- bid                 DECIMAL        # Bid price
- ask                 DECIMAL        # Ask price
- vol                 DECIMAL        # Implied volatility (IV)
- delta               DECIMAL        # Delta Greek
- gamma               DECIMAL        # Gamma Greek
- theta               DECIMAL        # Theta Greek
- vega                DECIMAL        # Vega Greek
- rho                 DECIMAL        # Rho Greek (assumed based on standard schema)
- volume              INT            # Trading volume (assumed)
- open_interest       INT            # Open interest (assumed)
```

**Notes**:
- Column names use **lowercase with underscores**
- `act_symbol` is the underlying symbol (not `underlying_symbol`)
- `call_put` indicates option type (not `option_type`)
- `vol` contains implied volatility (not `implied_volatility`)
- **mid price is NOT stored** - must be calculated as `(bid + ask) / 2`

#### Table 2: `volatility_history`
Historical volatility metrics for underlying securities.

**Columns** (inferred from implementation):
```sql
- date                DATE           # Date
- symbol              VARCHAR        # Underlying ticker
- hv10                DECIMAL        # 10-day historical volatility
- hv20                DECIMAL        # 20-day historical volatility
- hv30                DECIMAL        # 30-day historical volatility
- hv60                DECIMAL        # 60-day historical volatility
- hv90                DECIMAL        # 90-day historical volatility
- iv_mean             DECIMAL        # Mean implied volatility
- iv30                DECIMAL        # 30-day implied volatility
```

**IMPORTANT**: This table does **NOT** contain OHLCV price data. Despite the method name `get_underlying_prices()`, there is no price data in this database.

### 1.3 Missing Data

**What's NOT in the database**:
- ❌ Underlying stock OHLCV prices (open, high, low, close, volume)
- ❌ VIX index historical data
- ❌ Earnings dates / event calendar
- ❌ Dividend information
- ❌ Stock split information

**Implications**:
- External data source required for underlying prices
- VIX data must come from another source
- Event detection (earnings, FOMC) requires external calendar

---

## 2. Column Name Mapping

### 2.1 Database → Standard Mapping

The implementation attempts to standardize column names but has inconsistencies:

| Database Column | Standard Name | Type | Notes |
|----------------|---------------|------|-------|
| `date` | `quote_date` | DATE | Quote/trade date |
| `act_symbol` | `underlying_symbol` | VARCHAR | Underlying ticker |
| `expiration` | `expiration` | DATE | Expiration date |
| `strike` | `strike` | DECIMAL | Strike price |
| `call_put` | `option_type` | VARCHAR | 'call' or 'put' |
| `bid` | `bid` | DECIMAL | Bid price |
| `ask` | `ask` | DECIMAL | Ask price |
| `vol` | `implied_volatility` | DECIMAL | IV |
| `delta` | `delta` | DECIMAL | Delta |
| `gamma` | `gamma` | DECIMAL | Gamma |
| `theta` | `theta` | DECIMAL | Theta |
| `vega` | `vega` | DECIMAL | Vega |
| `rho` | `rho` | DECIMAL | Rho |
| N/A | `mid` | DECIMAL | **Calculated**: `(bid + ask) / 2` |

### 2.2 Inconsistencies Found

⚠️ **CRITICAL**: The data layer implementation has multiple column name mapping bugs:
1. Code expects `implied_volatility` but database has `vol`
2. Code expects `option_type` but database has `call_put`
3. Code expects `underlying_symbol` but database has `act_symbol`
4. Standardization to lowercase is inconsistent across modules

---

## 3. Data Quality Characteristics

### 3.1 Known Issues (from testing)

1. **Missing Option Chain Data**:
   - Test query for SPY on 2023-01-03 returned 0 rows
   - This could indicate:
     - Market holiday (Jan 2, 2023 was a Monday but markets were closed for New Year's observance)
     - Data gap in database
     - Query date/DTE logic issue

2. **No OHLCV Price Data**:
   - `get_underlying_prices()` method is misleading
   - Actually returns volatility metrics, not prices
   - External price source is REQUIRED

3. **Column Name Mismatches**:
   - Multiple KeyError risks due to column naming inconsistencies

### 3.2 Data Coverage

✅ **Verified Coverage**:
- Symbols: 2,227 tickers available
- Date Range: 2019-02-09 to 2025-12-12 (nearly 7 years)
- SPY data confirmed available in range

### 3.3 Data Quality Filters Implemented

The `DataValidator` class implements filters for:
- Zero or negative bid/ask prices
- Excessive bid/ask spreads (>50% of mid price)
- Extreme implied volatilities (< 1% or > 500%)
- Invalid option types
- Out-of-range dates
- Duplicate records

---

## 4. Implementation Status

### 4.1 Files Implemented

| File | Size | Status | Issues |
|------|------|--------|--------|
| `dolt_adapter.py` | 37 KB | ⚠️ Has Bugs | 7 critical SQL injection vulnerabilities |
| `market_data.py` | 35 KB | ⚠️ Has Bugs | Column mismatch bugs, broken price loading |
| `data_validator.py` | 32 KB | ⚠️ Has Bugs | IV filter logic error |
| `__init__.py` | 3 KB | ✅ OK | Exports correct |

### 4.2 Critical Issues Found (Code Quality Audit)

**CRITICAL (Must Fix Before Use)**:
1. **SQL Injection Vulnerabilities**: All queries use f-strings instead of parameterized queries (7 locations)
2. **Column Name Bug**: Line 804 in `market_data.py` accesses `'implied_volatility'` column that doesn't exist (should be `iv_col` parameter)
3. **Broken Price Loading**: `get_underlying_prices()` returns volatility data, not OHLCV prices - fundamentally broken
4. **IV Filter Logic Error**: Line 495 in `data_validator.py` has incorrect boolean operator

**HIGH PRIORITY**:
- Date parsing errors not properly caught
- Division by zero potential in spread calculations
- Missing error handling for empty IV history
- Inconsistent column naming throughout

**Full Audit Report**: See agent e5bcce44 output (23 issues total)

### 4.3 Production Readiness

**STATUS**: ❌ **NOT PRODUCTION READY**

**Blockers**:
1. SQL injection security vulnerability
2. Column name mismatches will cause runtime KeyErrors
3. Price loading completely broken
4. IV percentile calculation has correctness issues

**Required Before Use**:
- Fix all CRITICAL and HIGH priority bugs
- Add comprehensive unit tests
- Integration tests with real database
- Security review of SQL queries

---

## 5. Usage Patterns

### 5.1 Connecting to Database

```python
from backtester.data import DoltAdapter
from datetime import datetime

# Database path (after cloning)
dolt_path = '/Users/janussuk/Desktop/OptionsBacktester2/dolt_data/options'

# Use context manager for automatic cleanup
with DoltAdapter(dolt_path) as adapter:
    # Get available symbols
    symbols = adapter.get_available_symbols()

    # Get date range
    date_range = adapter.get_date_range('SPY')

    # Query option chain
    chain = adapter.get_option_chain(
        underlying='SPY',
        date=datetime(2023, 1, 4),  # Use trading day
        min_dte=7,
        max_dte=30
    )
```

### 5.2 Loading Market Data

```python
from backtester.data import MarketDataLoader, DoltAdapter
from datetime import datetime

dolt_path = '/path/to/dolt_data/options'

with DoltAdapter(dolt_path) as adapter:
    loader = MarketDataLoader(adapter)

    # Load option chain
    chain = loader.load_option_chain(
        underlying='SPY',
        date=datetime(2023, 1, 4),
        dte_range=(7, 30)
    )

    # Calculate IV percentile (⚠️ May have issues)
    iv_pct = loader.calculate_iv_percentile(
        symbol='SPY',
        date=datetime(2023, 1, 4),
        window=252
    )
```

### 5.3 Data Validation

```python
from backtester.data import DataValidator

# Validate option data
is_valid, errors = DataValidator.validate_option_data(chain)
if not is_valid:
    print(f"Validation errors: {errors}")

# Filter bad quotes
clean_chain = DataValidator.filter_bad_quotes(chain)

# Check for gaps
gaps = DataValidator.check_for_gaps(chain, 'date')
```

---

## 6. Query Patterns

### 6.1 Get Option Chain for Specific Date

```sql
-- Internal query performed by get_option_chain()
SELECT *
FROM option_chain
WHERE act_symbol = 'SPY'
  AND date = '2023-01-04'
  AND expiration >= '2023-01-11'  -- date + min_dte
  AND expiration <= '2023-02-03'  -- date + max_dte
ORDER BY strike, call_put;
```

### 6.2 Get Volatility History

```sql
-- Internal query performed by get_underlying_prices()
SELECT *
FROM volatility_history
WHERE symbol = 'SPY'
  AND date >= '2023-01-01'
  AND date <= '2023-12-31'
ORDER BY date;
```

### 6.3 Get Implied Volatility for Date

```sql
-- Internal query for IV calculation
SELECT AVG(vol) as avg_iv
FROM option_chain
WHERE act_symbol = 'SPY'
  AND date = '2023-01-04'
  AND vol IS NOT NULL
  AND vol > 0;
```

---

## 7. Performance Considerations

### 7.1 Query Performance

- **Option chain queries**: Moderate performance
  - Filtering by symbol and date is indexed
  - DTE range filter requires expiration calculation
  - Large result sets (100-1000s of rows per query)

- **Volatility history queries**: Fast
  - Simple date range filter
  - Small result sets (typically < 1000 rows)

### 7.2 Memory Usage

- **Full option chain**: Can be large (>10K rows for liquid stocks)
- **Historical data**: Moderate (252-500 rows for annual data)
- **Recommendation**: Use date filtering, avoid loading entire history

### 7.3 Caching Strategy

The implementation includes caching for:
- IV history (by symbol, date range, DTE target)
- Historical volatility calculations

⚠️ **Bug**: Cache keys don't include all parameters (e.g., `use_atm` flag)

---

## 8. External Data Requirements

### 8.1 Required External Sources

**For full backtesting capability, you need**:

1. **Underlying Prices** (OHLCV data)
   - Suggested sources: yfinance, Alpha Vantage, Polygon.io
   - Required for: spot price at any timestamp, P&L calculations
   - Frequency: Daily or intraday

2. **VIX Index**
   - Source: CBOE historical data or financial API
   - Required for: volatility regime detection, entry conditions
   - Frequency: Daily

3. **Event Calendar** (Optional but recommended)
   - Earnings dates: Earnings Whispers API, Alpha Vantage
   - FOMC dates: Federal Reserve calendar
   - Required for: avoiding major events in strategies
   - Frequency: Updated monthly

### 8.2 Integration Approach

```python
# Recommended pattern: Separate data source for prices
from backtester.data import DoltAdapter
import yfinance as yf  # External library

# Get options data from Dolt
with DoltAdapter(dolt_path) as adapter:
    option_chain = adapter.get_option_chain('SPY', date, 7, 30)

# Get price data from yfinance
spy = yf.Ticker('SPY')
price_data = spy.history(start=start_date, end=end_date)
spot_price = price_data.loc[date, 'Close']
```

---

## 9. Known Limitations

### 9.1 Database Limitations

1. ❌ No intraday data (daily quotes only)
2. ❌ No underlying price data (OHLCV)
3. ❌ No bid/ask size or depth of book
4. ❌ No last trade price or volume by strike
5. ❌ Greeks may be calculated (not actual exchange values)
6. ⚠️ Data gaps exist (holidays, low-volume strikes)

### 9.2 Implementation Limitations

1. ❌ SQL injection vulnerabilities (CRITICAL)
2. ❌ Column name mapping inconsistencies
3. ❌ Price loading broken
4. ⚠️ Incomplete error handling
5. ⚠️ No transaction support
6. ⚠️ Cache invalidation issues

### 9.3 Workarounds

**For missing price data**:
```python
# Use yfinance or similar
import yfinance as yf

def get_spot_price(symbol: str, date: datetime) -> float:
    """Get spot price from external source."""
    ticker = yf.Ticker(symbol)
    hist = ticker.history(start=date, end=date + timedelta(days=1))
    return hist['Close'].iloc[0] if len(hist) > 0 else None
```

**For VIX data**:
```python
# Download VIX from CBOE or use API
vix = yf.Ticker('^VIX')
vix_history = vix.history(start=start_date, end=end_date)
```

---

## 10. Next Steps

### 10.1 Immediate Fixes Required

Before continuing with Run 2 (Option & Pricing Core), **MUST FIX**:

1. **Fix SQL injection vulnerabilities**
   - Replace f-strings with parameterized queries
   - Or use proper SQL escaping

2. **Fix column name mismatches**
   - Implement consistent column renaming layer
   - Update all hardcoded column references

3. **Fix or document price loading**
   - Either implement external price source
   - Or clearly document limitation and throw NotImplementedError

4. **Add critical error handling**
   - Validate inputs
   - Handle missing data gracefully

### 10.2 Recommended Additions

1. Create utility module for column mapping
2. Add comprehensive unit tests
3. Add integration tests with real database
4. Document external data integration pattern

### 10.3 Testing Strategy

```python
# Test with known-good dates
TEST_DATES = [
    datetime(2023, 1, 4),   # First trading day 2023
    datetime(2023, 6, 15),  # Mid-year
    datetime(2024, 3, 15),  # Recent (quarterly expiry)
]

# Test with liquid symbols
TEST_SYMBOLS = ['SPY', 'AAPL', 'QQQ', 'IWM']

# Validate returned data structure
assert 'strike' in chain.columns
assert 'call_put' in chain.columns  # NOT 'option_type'
assert 'vol' in chain.columns       # NOT 'implied_volatility'
```

---

## 11. Context for Next Runs

### 11.1 Schema Understanding

✅ **Verified Facts**:
- Tables: `option_chain`, `volatility_history`
- Column names: `act_symbol`, `call_put`, `vol` (not standard names)
- No OHLCV price data available
- Date range: 2019-02-09 to 2025-12-12
- 2,227 symbols available

### 11.2 Implementation Status

⚠️ **Data layer has 23 bugs** (4 critical, 8 high priority)
- **Not production ready**
- Needs fixes before Run 2
- Or document as "known issues, will fix in Run 9"

### 11.3 Decision Point

**Option A**: Fix critical bugs now before proceeding
- Pros: Clean foundation, no technical debt
- Cons: More time before seeing full system

**Option B**: Continue to Run 2, fix all bugs in Run 9
- Pros: Faster progress through implementation
- Cons: May encounter issues when integrating components

**Recommendation**: Fix at least the CRITICAL issues (SQL injection, column names) before Run 2, defer other fixes to Run 9.

---

**End of DATA_SCHEMA.md**
