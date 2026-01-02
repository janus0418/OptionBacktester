---
title: "Web App Refactoring: Remove Legacy streamlit_app.py and Fix Module Exports"
category: runtime-errors
tags:
  - web-app
  - refactoring
  - module-exports
  - legacy-code-cleanup
  - streamlit
  - database-connection
  - python-imports
component: web_app
severity: high
date_solved: "2026-01-02"
related_pr: "#4"
files_changed:
  - web_app/streamlit_app.py (deleted)
  - web_app/shared/__init__.py (modified)
---

# Web App Legacy Code Cleanup and Export Fix

## Problem Summary

The web application had multiple interconnected issues:

1. **Legacy file with API bug**: `streamlit_app.py` (943 lines) called non-existent `adapter.disconnect()` method
2. **Missing module exports**: `shared/__init__.py` was missing exports for comparison functions
3. **Perceived empty pages**: Slow database connection (57 seconds) made pages appear broken

## Symptoms

- ImportError when Strategy Builder page tried to import comparison functions
- Potential runtime error if legacy file was executed (`disconnect()` method doesn't exist)
- Users reported "empty pages" during initial load

## Root Cause Analysis

### Issue 1: Legacy File with Incorrect API Call

The old `web_app/streamlit_app.py` contained a critical bug:

```python
# WRONG - method doesn't exist in DoltAdapter
adapter.disconnect()
```

The correct DoltAdapter API uses:

```python
# CORRECT - proper cleanup method
adapter.close()
```

**Root Cause**: The legacy file was created before the DoltAdapter API was finalized and never updated. Since it was untracked in git and never deployed, it accumulated technical debt.

### Issue 2: Missing Module Exports

The `web_app/shared/__init__.py` was missing three comparison function exports:

- `add_to_comparison`
- `clear_comparison`
- `get_comparison_results`

These functions existed in `shared/utils.py` (lines 124-138) but weren't exposed through the package's `__init__.py`.

### Issue 3: Slow Database Connection

The `check_database_connection()` function performs a full database validation including:
- Database path exists
- Database tables exist
- SPY data is available
- Date range is retrievable

This takes ~57 seconds on first connection, making pages appear empty during loading.

## Solution

### Step 1: Delete Legacy File

**File**: `web_app/streamlit_app.py`

**Action**: Completely removed (943 lines deleted)

**Rationale**:
- File was untracked (never committed to git)
- Duplicated functionality from `shared/utils.py` and `shared/charts.py`
- Contained unmaintained code with API mismatches
- Multi-page app structure (`Home.py` + `pages/`) is the canonical implementation

### Step 2: Add Missing Exports

**File**: `web_app/shared/__init__.py`

**Before**:
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

**After** (added 3 imports):
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
    add_to_comparison,        # NEW
    clear_comparison,         # NEW
    get_comparison_results,   # NEW
)
```

Also updated `__all__` list with the 3 new exports.

### Step 3: Document Database Connection Timing

Added documentation that pages take ~60 seconds to load due to database connection check. This is expected behavior, not a bug.

## Verification

1. **Legacy file deleted**: `ls web_app/streamlit_app.py` returns "No such file"

2. **Exports available**: All imports work correctly:
   ```python
   from shared import add_to_comparison, clear_comparison, get_comparison_results  # OK
   ```

3. **Pages load correctly**: After waiting for database connection (~60s), all pages display full content

4. **API method correct**: Verified `adapter.close()` is used throughout codebase

## Changes Summary

| File | Action | Lines Changed |
|------|--------|---------------|
| `web_app/streamlit_app.py` | DELETE | -943 |
| `web_app/shared/__init__.py` | EDIT | +6 |
| **Net Impact** | | **-937** |

## Prevention Strategies

### 1. Prevent Legacy Code Accumulation

- Track all files in git - untracked files accumulate bugs
- Regular cleanup sprints to identify duplicate code
- Enforce Single Source of Truth (SSOT) principle

### 2. Keep Module Exports in Sync

- Add import validation tests that run on every commit
- Document which functions each page imports
- Use automated validation scripts to check `__all__` completeness

### 3. Handle Slow Database Connections

- Add connection caching with TTL (57s -> <100ms for cached)
- Show loading indicators during database initialization
- Consider lazy loading - check connection only when needed

## Test Cases

```python
# Test that all comparison functions are importable
def test_comparison_exports():
    from shared import add_to_comparison, clear_comparison, get_comparison_results
    assert callable(add_to_comparison)
    assert callable(clear_comparison)
    assert callable(get_comparison_results)

# Test that DoltAdapter uses close() not disconnect()
def test_adapter_cleanup_method():
    from backtester.data.dolt_adapter import DoltAdapter
    assert hasattr(DoltAdapter, 'close')
    assert not hasattr(DoltAdapter, 'disconnect')
```

## Related Documentation

- `docs/architecture.md` - DoltAdapter documentation (lines 405-426)
- `docs/project_spec.md` - Project structure
- `web_app/shared/utils.py:124-138` - Comparison function definitions
- `web_app/pages/3_Strategy_Builder.py:38-40` - Page imports

## Lessons Learned

1. **Track all code in version control** - Untracked files become unmaintained
2. **Keep exports synchronized** - When adding functions, update `__init__.py`
3. **Provide loading feedback** - Slow operations should show progress indicators
4. **Delete, don't deprecate** - Removing dead code is cleaner than fixing it
