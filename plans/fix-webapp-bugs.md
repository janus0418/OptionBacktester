# Fix Web App Bugs (Simplified)

## Overview

Fix bugs in the web app by removing duplicated legacy code and adding missing module exports.

## Problem Statement

1. **Legacy file with bugs**: `streamlit_app.py` contains 943 lines of duplicated code and has a bug calling `adapter.disconnect()` (method doesn't exist - should be `close()`)
2. **Missing exports**: `shared/__init__.py` is missing exports for comparison functions used by Strategy Builder page

## Solution

Based on reviewer feedback (DHH, Kieran, Simplicity):
- **Delete** `streamlit_app.py` entirely rather than fixing it - it's untracked, never deployed, and duplicates `shared/` modules
- **Add** missing exports to `shared/__init__.py`

## Changes Required

### Change 1: Delete Legacy File

**File**: `web_app/streamlit_app.py`

**Action**: Delete entirely

**Rationale**:
- File is untracked in git (never committed)
- Contains 943 lines that duplicate `shared/utils.py` and `shared/charts.py`
- Has bug (`disconnect()` instead of `close()`) that proves it's unmaintained
- Multi-page app (`Home.py`) is the intended entry point

### Change 2: Add Missing Exports

**File**: `web_app/shared/__init__.py`

**Current** (lines 3-15):
```python
from .utils import (
    STRATEGY_INFO,
    DEFAULT_DB_PATH,
    check_database_connection,
    run_real_backtest,
    init_session_state,
    get_results,
    set_results,
    clear_results,
    get_all_saved_results,
    save_result,
    delete_saved_result,
)
```

**Add these imports**:
```python
    add_to_comparison,
    clear_comparison,
    get_comparison_results,
```

**Update `__all__` list** (add after line 36):
```python
    "add_to_comparison",
    "clear_comparison",
    "get_comparison_results",
```

## Acceptance Criteria

- [ ] `web_app/streamlit_app.py` is deleted
- [ ] `shared/__init__.py` exports all comparison functions
- [ ] Strategy Builder page loads without errors
- [ ] Comparison features work (add results, view chart, clear)

## Test Plan

1. Run the app:
   ```bash
   cd web_app
   streamlit run Home.py
   ```

2. Navigate to Strategy Builder page

3. Verify page loads without ImportError

4. Run a backtest with "Add to comparison" checked

5. Verify comparison chart displays

6. Click "Clear Comparison" and verify it works

## Files Changed

| File | Action | Impact |
|------|--------|--------|
| `web_app/streamlit_app.py` | DELETE | -943 lines |
| `web_app/shared/__init__.py` | EDIT | +6 lines |

**Net change**: -937 lines

## References

- `web_app/shared/utils.py:124-138` - Comparison function definitions
- `web_app/pages/3_Strategy_Builder.py:38-40` - Imports that require these exports
