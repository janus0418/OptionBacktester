# Phase B Quick Start Guide

**Ready to Begin Implementation? Start Here!**

---

## 🎯 Overview

Phase B focuses on **accuracy and realism** - the foundation for all other enhancements. You'll implement:

1. Volatility surface (SVI model)
2. American option pricing (binomial tree)
3. Advanced execution model (volume impact)
4. Multi-source data architecture
5. Streamlit web app (MVP)

**Duration:** 3 weeks
**Difficulty:** Medium-High
**Impact:** Critical foundation

---

## ✅ Prerequisites

### Knowledge Requirements
- ✅ Python proficiency (intermediate+)
- ✅ Options fundamentals (Greeks, pricing)
- ✅ NumPy/SciPy basics
- ⚠️ Helpful: Numerical methods, optimization

### Environment Setup
```bash
cd /Users/janussuk/Desktop/AI\ Projects/OptionsBacktester2/code

# Ensure all dependencies installed
pip install -r requirements.txt

# Additional for Phase B
pip install scipy>=1.13.1  # Optimization for calibration
pip install streamlit>=1.31.0  # Web app

# Run tests to ensure system working
pytest tests/ -v
# Should see: 765 tests passing
```

---

## 📅 Week 1: Volatility Surface & American Pricing

### Day 1-2: Volatility Surface Implementation

#### Step 1: Create Branch
```bash
cd /Users/janussuk/Desktop/AI\ Projects/OptionsBacktester2
git checkout -b feature/volatility-surface
```

#### Step 2: Create Files
```bash
# Create new module
touch code/backtester/core/volatility_surface.py

# Create tests
touch code/tests/test_volatility_surface.py
```

#### Step 3: Implement SVI Model

Copy code from `/IMPLEMENTATION_PLAN.md` section "Day 1-2: Volatility Surface"

**Key components:**
- `SVIParameters` class - parameter container with validation
- `VolatilitySurface` class - main surface class
- `_calibrate()` method - fit SVI to market data
- `get_iv()` method - interpolate IV for any strike/DTE

#### Step 4: Write Tests First (TDD)
```python
# test_volatility_surface.py

def test_svi_calibration_smoke():
    """Ensure calibration runs without errors"""
    market_data = create_synthetic_data()
    surface = VolatilitySurface(market_data)
    assert len(surface.svi_params) > 0

def test_iv_interpolation():
    """Test IV retrieval works"""
    surface = create_test_surface()
    iv = surface.get_iv(strike=450, dte=30, spot=450)
    assert 0.10 < iv < 0.50  # Reasonable range

def test_put_skew():
    """Verify puts more expensive than calls (skew)"""
    surface = create_test_surface()
    atm_iv = surface.get_iv(450, 30, 450)
    otm_put_iv = surface.get_iv(420, 30, 450)  # 7% OTM
    assert otm_put_iv > atm_iv  # Put skew
```

#### Step 5: Run Tests
```bash
pytest code/tests/test_volatility_surface.py -v

# Goal: All tests pass
```

#### Step 6: Integration Test
```python
# Test with real backtester
from backtester.core.volatility_surface import VolatilitySurface
from backtester.core.option import Option

# Load market IV data
market_data = load_historical_iv_data()

# Create surface
surface = VolatilitySurface(market_data)

# Use in pricing
option = Option(...)
iv = surface.get_iv(option.strike, option.days_to_expiration, spot)
option.implied_vol = iv  # Use surface IV instead of flat

# Verify pricing changed
```

**Success Criteria:**
- [ ] SVI calibration runs without errors
- [ ] IV interpolation works for any strike/DTE
- [ ] Put skew visible (OTM puts higher IV)
- [ ] All tests pass
- [ ] Pricing accuracy improved (validate against market)

---

### Day 3-5: American Option Pricing

#### Step 1: Create Files
```bash
touch code/backtester/core/american_pricing.py
touch code/tests/test_american_pricing.py
```

#### Step 2: Implement Binomial Tree

Copy code from `/IMPLEMENTATION_PLAN.md` section "Day 3-5: American Option Pricing"

**Key components:**
- `BinomialPricer` class - tree builder
- `price()` method - American option pricing
- `calculate_greeks()` - finite difference Greeks
- `AmericanOptionPricer` - unified interface

#### Step 3: Write Tests
```python
def test_american_put_early_exercise():
    """Verify American put worth more than European"""
    pricer = BinomialPricer(steps=100)
    
    # Deep ITM put should exercise early
    american_price = pricer.price(S=100, K=120, T=1.0, r=0.05, 
                                   sigma=0.20, option_type='put')
    
    # European put (for comparison)
    european_price = black_scholes_price(100, 120, 1.0, 0.05, 0.20, 'put')
    
    # American should be worth more (can exercise early)
    assert american_price > european_price

def test_american_call_no_dividends():
    """American call = European call (no dividends, no early exercise)"""
    pricer = BinomialPricer(steps=50)
    
    american = pricer.price(S=100, K=100, T=0.5, r=0.05, 
                           sigma=0.25, option_type='call')
    european = black_scholes_price(100, 100, 0.5, 0.05, 0.25, 'call')
    
    # Should be within 1% (numerical error)
    assert abs(american - european) / european < 0.01
```

#### Step 4: Benchmark Performance
```python
import time

pricer = BinomialPricer(steps=100)

# Time 1,000 price calculations
start = time.time()
for _ in range(1000):
    price = pricer.price(S=450, K=450, T=0.1, r=0.04, sigma=0.20, option_type='put')
elapsed = time.time() - start

print(f"Time per price: {elapsed/1000*1000:.2f} ms")
# Goal: < 10ms per price
```

**Success Criteria:**
- [ ] American puts worth more than European (deep ITM)
- [ ] American calls ≈ European calls (no dividends)
- [ ] Greeks calculations work
- [ ] Performance < 10ms per price (100 steps)
- [ ] All tests pass

---

## 📅 Week 2: Execution Model & Data Architecture

### Day 6-8: Enhanced Execution Model

#### Step 1: Enhance Existing File
```bash
# Edit existing file (don't create new one)
code/backtester/engine/execution.py
```

#### Step 2: Add Advanced Features

Copy code from `/IMPLEMENTATION_PLAN.md` section "Day 6-8: Advanced Execution Model"

**New features:**
- Volume impact calculation (Kyle's lambda)
- Partial fills based on volume
- Order queue management
- Limit/stop order support

#### Step 3: Test Volume Impact
```python
def test_volume_impact_small_order():
    """Small orders have minimal impact"""
    model = AdvancedExecutionModel(use_volume_impact=True)
    
    market_data = {
        'bid': 5.00,
        'ask': 5.10,
        'volume': 10000  # Large daily volume
    }
    
    # Small order (10 contracts, 0.1% of volume)
    result = model.execute_order('buy', quantity=10, market_data=market_data)
    
    # Impact should be minimal
    assert result.volume_impact < 0.01  # < 1%

def test_volume_impact_large_order():
    """Large orders have significant impact"""
    model = AdvancedExecutionModel(use_volume_impact=True)
    
    market_data = {
        'bid': 5.00,
        'ask': 5.10,
        'volume': 1000  # Low volume
    }
    
    # Large order (500 contracts, 50% of volume!)
    result = model.execute_order('buy', quantity=500, market_data=market_data)
    
    # Impact should be large
    assert result.volume_impact > 0.03  # > 3%
```

**Success Criteria:**
- [ ] Volume impact scales with √(order_size/volume)
- [ ] Partial fills work for large orders
- [ ] All tests pass
- [ ] Backtests show more conservative P&L

---

### Day 9-11: Multi-Source Data Architecture

#### Step 1: Create Plugin System
```bash
touch code/backtester/data/data_manager.py
touch code/backtester/data/csv_adapter.py
```

#### Step 2: Implement

Copy code from `/IMPLEMENTATION_PLAN.md` section "Day 9-11: Data Architecture"

**Components:**
- `BaseDataAdapter` - abstract base class
- `CSVDataAdapter` - read from CSV files
- `DataSourceRegistry` - manage multiple sources with priority

#### Step 3: Test Multi-Source
```python
def test_fallback_behavior():
    """Test fallback to secondary source if primary fails"""
    registry = DataSourceRegistry()
    
    # Register primary (will fail)
    primary = Mock(spec=BaseDataAdapter)
    primary.get_option_chain.side_effect = Exception("DB down")
    registry.register_source('primary', primary, priority=10)
    
    # Register fallback (CSV)
    fallback = CSVDataAdapter('./test_data')
    registry.register_source('fallback', fallback, priority=5)
    
    # Should use fallback
    data = registry.get_option_chain('SPY', date, expiration)
    assert data is not None  # Got data from CSV
```

**Success Criteria:**
- [ ] Can load from CSV files
- [ ] Fallback works when primary fails
- [ ] All tests pass
- [ ] Documented configuration

---

## 📅 Week 3: Streamlit Web App (MVP)

### Day 12-15: Basic Web Interface

#### Step 1: Create App Directory
```bash
mkdir -p web_app
touch web_app/streamlit_app.py
touch web_app/requirements.txt
```

#### Step 2: Implement MVP

Copy code from `/IMPLEMENTATION_PLAN.md` section "Day 12-15: Streamlit Web App"

**Features:**
- Strategy configuration sidebar
- Basic backtest execution
- Results display (metrics, charts)
- Export functionality

#### Step 3: Run Locally
```bash
cd web_app
streamlit run streamlit_app.py

# Should open browser at http://localhost:8501
```

#### Step 4: Test User Flow
1. Select strategy
2. Configure parameters
3. Click "Run Backtest"
4. View results
5. Export report

**Success Criteria:**
- [ ] App runs without errors
- [ ] Can configure strategy via forms
- [ ] Results display correctly
- [ ] Charts render properly
- [ ] Export works

---

## 🧪 Testing Strategy

### Unit Tests (Day-by-Day)
```bash
# Test individual components as you build
pytest code/tests/test_volatility_surface.py -v
pytest code/tests/test_american_pricing.py -v
pytest code/tests/test_execution.py -v -k "volume_impact"
```

### Integration Tests (End of Week)
```bash
# Test full backtest with new features
pytest code/tests/test_integration.py -v

# Should still have 765+ tests passing
pytest code/tests/ -v --tb=short
```

### Validation Tests
```python
# Validate against known values
def test_pricing_accuracy():
    """Compare to broker calculator"""
    surface = load_production_surface()
    
    # Known market price
    market_price = 5.50
    
    # Our price
    our_price = calculate_option_price_with_surface(...)
    
    # Should be within 2%
    assert abs(our_price - market_price) / market_price < 0.02
```

---

## 📊 Progress Tracking

### Week 1 Checklist
- [ ] Day 1: SVI class implemented
- [ ] Day 2: Calibration working, tests passing
- [ ] Day 3: Binomial tree implemented
- [ ] Day 4: American pricing working
- [ ] Day 5: Performance optimization, all tests pass

### Week 2 Checklist
- [ ] Day 6: Volume impact model implemented
- [ ] Day 7: Partial fills working
- [ ] Day 8: All execution tests passing
- [ ] Day 9: Data adapter base class
- [ ] Day 10: CSV adapter working
- [ ] Day 11: Multi-source registry complete

### Week 3 Checklist
- [ ] Day 12: Streamlit app skeleton
- [ ] Day 13: Strategy configuration forms
- [ ] Day 14: Results visualization
- [ ] Day 15: Polish and deploy

---

## 🚨 Common Issues & Solutions

### Issue: SVI Calibration Fails
**Symptom:** Optimization doesn't converge
**Solution:** 
```python
# Improve initial guess
x0 = [
    market_data['iv'].mean()**2,  # Better a estimate
    0.1,
    -0.4 if market_data['is_put'].mean() > 0.5 else 0.0,  # Skew direction
    0.0,
    0.3
]

# Add bounds
bounds = [
    (0, None),    # a >= 0
    (0, None),    # b >= 0
    (-0.99, 0.99), # |rho| < 1
    (-1, 1),      # m reasonable
    (0.01, 1.0)   # sigma > 0
]
```

### Issue: Binomial Tree Too Slow
**Symptom:** >10ms per price
**Solution:**
- Reduce steps (100 → 50 for testing)
- Cache repeated calculations
- Consider Numba JIT compilation

### Issue: Streamlit App Won't Start
**Symptom:** Module import errors
**Solution:**
```bash
# Ensure paths correct
export PYTHONPATH="/Users/janussuk/Desktop/AI Projects/OptionsBacktester2/code:$PYTHONPATH"

# Or in app:
import sys
sys.path.append('../code')
```

---

## 📝 Documentation

As you implement, update:

1. **API docs:** Add docstrings to all new classes/methods
2. **Examples:** Create usage examples
3. **Tests:** Comment complex test scenarios
4. **Changelog:** Track changes in `docs/changelog.md`

---

## ✅ Phase B Complete Criteria

Before moving to Phase A, ensure:

- [ ] All 765+ tests passing
- [ ] Vol surface pricing within 2% of market
- [ ] American pricing matches broker calculators
- [ ] Execution model accounts for volume
- [ ] 3+ data sources working
- [ ] Streamlit app functional
- [ ] Documentation updated
- [ ] Code reviewed and merged to main

---

## 🎯 Next Steps

After completing Phase B:

1. **Merge to main:**
```bash
git checkout main
git merge feature/volatility-surface
git push origin main
```

2. **Create Phase A branch:**
```bash
git checkout -b feature/strategy-builder-sdk
```

3. **Start Phase A Week 1:** Strategy Builder SDK

See `/docs/IMPROVEMENT_ROADMAP.md` for Phase A details.

---

**Questions? Issues?** Refer to `/IMPLEMENTATION_PLAN.md` for detailed code examples.

**Good luck! 🚀**
