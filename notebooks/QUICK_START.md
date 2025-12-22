# Quick Start Guide - Short Straddle Strategy Notebook

## Run the Notebook in 3 Steps

### Step 1: Open Jupyter
```bash
cd /Users/janussuk/Desktop/OptionsBacktester2/notebooks
jupyter notebook ShortStraddleStrategy_RealData.ipynb
```

### Step 2: Execute All Cells
In Jupyter menu: **Cell → Run All**

### Step 3: Review Results
Scroll to Section 6 (visualizations) and Section 7 (insights)

---

## What You'll Get

### Performance Metrics
- Total Return
- Sharpe Ratio
- Max Drawdown
- Win Rate
- Profit Factor

### 8 Interactive Charts
1. Equity Curve with Drawdowns
2. P&L Distribution
3. IV Rank Histogram
4. Trade Timeline
5. Rolling Sharpe Ratio
6. Greeks Over Time
7. Monthly Returns Heatmap
8. Win Rate Breakdown

### Strategy Insights
- Entry timing effectiveness
- Exit management quality
- Risk assessment
- Improvement recommendations

---

## File Locations

| File | Path |
|------|------|
| **Notebook** | `/Users/janussuk/Desktop/OptionsBacktester2/notebooks/ShortStraddleStrategy_RealData.ipynb` |
| **User Guide** | `/Users/janussuk/Desktop/OptionsBacktester2/notebooks/README_ShortStraddle.md` |
| **Summary** | `/Users/janussuk/Desktop/OptionsBacktester2/NOTEBOOK_IMPLEMENTATION_SUMMARY.md` |

---

## Strategy at a Glance

### Entry Rules (ALL must be true)
- ✓ DTE: 7-21 days
- ✓ IV Rank > 50% (30-day lookback)
- ✓ IV NOT trending up (5-day window)
- ✓ Max 3 positions
- ✓ 1 contract per trade

### Exit Rules (ANY triggers exit)
- ✓ Profit: 50% of credit
- ✓ Loss: >200% of credit
- ✓ Time: DTE ≤ 1
- ✓ Delta: |Δ| > 0.30

---

## Customization Examples

### Change Date Range
```python
# In Section 2
START_DATE = datetime(2024, 1, 1)
END_DATE = datetime(2024, 12, 31)
```

### Adjust Parameters
```python
# In Section 5
strategy = AdvancedShortStraddleStrategy(
    iv_rank_threshold=60.0,      # Require higher IV
    profit_target_pct=0.75,      # Take profit at 75%
    stop_loss_pct=1.50,          # Tighter stop loss
    max_positions=5              # Allow more positions
)
```

### Different Underlying
```python
# In Section 2
UNDERLYING = 'QQQ'  # Trade QQQ instead of SPY
```

---

## Troubleshooting

### Database Connection Failed
```python
# Verify path
import os
os.path.exists('/Users/janussuk/Desktop/OptionsBacktester2/dolt_data/options')
```

### No Trades Generated
- Lower `iv_rank_threshold` (try 40)
- Expand date range
- Check data availability for your period

### Import Errors
```python
# Verify path
import sys
sys.path.insert(0, '/Users/janussuk/Desktop/OptionsBacktester2/code')
```

---

## Next Steps

1. **Run baseline backtest** with default parameters
2. **Analyze Section 8** for improvement recommendations
3. **Optimize parameters** using grid search
4. **Validate results** with walk-forward analysis
5. **Paper trade** before going live

---

## Key Features

- ✓ Real market data (Dolt database)
- ✓ Production-grade code quality
- ✓ Comprehensive error handling
- ✓ Interactive Plotly visualizations
- ✓ Institutional-level metrics
- ✓ Actionable insights

---

## Important Notes

**Risk Warning:** Short straddles have unlimited loss potential. Always use stop losses and never risk more than you can afford to lose.

**Data Requirement:** You must have the Dolt database with SPY options data for your selected date range.

**Performance:** Past results do not guarantee future performance. Always validate thoroughly before live trading.

---

**For detailed documentation, see:** `README_ShortStraddle.md`

**For complete implementation details, see:** `NOTEBOOK_IMPLEMENTATION_SUMMARY.md`
