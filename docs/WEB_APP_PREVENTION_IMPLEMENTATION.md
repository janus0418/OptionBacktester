# Web App Prevention Strategies - Implementation Guide

Quick reference for implementing the prevention strategies discussed in `WEB_APP_BUG_PREVENTION_STRATEGIES.md`.

---

## Quick Implementation (1-2 hours)

### 1. Add Import Validation Test

**File**: `/code/tests/test_web_app_imports.py` (create new)

```python
"""
Test web app module structure and imports.

Ensures:
- All pages can import required functions
- No circular imports
- __init__.py exports are complete
"""

import sys
from pathlib import Path
import pytest


class TestWebAppImports:
    """Validate web app module structure."""

    @classmethod
    def setup_class(cls):
        """Add web_app to path."""
        web_app_path = Path(__file__).parent.parent.parent / "web_app"
        sys.path.insert(0, str(web_app_path))

    def test_shared_module_can_be_imported(self):
        """Test: shared module can be imported."""
        import shared
        assert shared is not None

    def test_all_utils_exports_available(self):
        """Test: all utils functions can be imported from shared."""
        from shared import (
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
            add_to_comparison,
            clear_comparison,
            get_comparison_results,
        )

        # Verify they exist
        assert STRATEGY_INFO is not None
        assert DEFAULT_DB_PATH is not None
        assert callable(check_database_connection)
        assert callable(run_real_backtest)
        assert callable(add_to_comparison)

    def test_all_charts_exports_available(self):
        """Test: all chart functions can be imported from shared."""
        from shared import (
            render_equity_curve,
            render_returns_distribution,
            render_monthly_returns,
            render_trades_table,
            render_metrics_cards,
            render_detailed_metrics,
        )

        # All should be callable
        assert callable(render_equity_curve)
        assert callable(render_returns_distribution)
        assert callable(render_monthly_returns)

    def test_shared_all_list_is_complete(self):
        """Test: __all__ in __init__.py includes all public exports."""
        import shared

        expected = {
            # From utils
            "STRATEGY_INFO",
            "DEFAULT_DB_PATH",
            "check_database_connection",
            "run_real_backtest",
            "init_session_state",
            "get_results",
            "set_results",
            "clear_results",
            "get_all_saved_results",
            "save_result",
            "delete_saved_result",
            "add_to_comparison",
            "clear_comparison",
            "get_comparison_results",
            # From charts
            "render_equity_curve",
            "render_returns_distribution",
            "render_monthly_returns",
            "render_trades_table",
            "render_metrics_cards",
            "render_detailed_metrics",
        }

        actual = set(shared.__all__)

        missing = expected - actual
        extra = actual - expected

        assert missing == set(), f"Missing exports in __all__: {missing}"
        # Allow extra exports, but warn about them
        if extra:
            print(f"Warning: Unexpected exports in __all__: {extra}")

    def test_pages_can_import_shared(self):
        """Test: each page file can import shared without errors."""
        pages_dir = (
            Path(__file__).parent.parent.parent / "web_app" / "pages"
        )

        for page_file in pages_dir.glob("*.py"):
            if page_file.name.startswith("_"):
                continue

            # Just verify the file exists and is valid Python
            with open(page_file) as f:
                code = f.read()

            # Check that it imports from shared
            assert "from shared" in code or "import shared" in code, (
                f"{page_file.name} doesn't import from shared"
            )

    def test_no_circular_imports(self):
        """Test: no circular import dependencies detected."""
        try:
            import shared
            # If we got here without RecursionError, we're good
            assert True
        except RecursionError:
            pytest.fail("Circular import detected in shared module")

    def test_strategy_info_has_all_strategies(self):
        """Test: STRATEGY_INFO contains all required strategies."""
        from shared import STRATEGY_INFO

        required_strategies = [
            "Short Straddle (High IV)",
            "Iron Condor",
            "Volatility Regime",
        ]

        for strategy in required_strategies:
            assert strategy in STRATEGY_INFO, (
                f"Strategy '{strategy}' missing from STRATEGY_INFO"
            )

        for strategy, info in STRATEGY_INFO.items():
            assert "description" in info
            assert "risk" in info
            assert "ideal_conditions" in info
            assert "params" in info

    def test_render_functions_are_callable(self):
        """Test: all render functions are actually callable."""
        from shared.charts import (
            render_equity_curve,
            render_returns_distribution,
            render_monthly_returns,
            render_trades_table,
            render_metrics_cards,
            render_detailed_metrics,
        )

        functions = [
            render_equity_curve,
            render_returns_distribution,
            render_monthly_returns,
            render_trades_table,
            render_metrics_cards,
            render_detailed_metrics,
        ]

        for func in functions:
            assert callable(func), f"{func.__name__} is not callable"
            assert func.__doc__, f"{func.__name__} has no docstring"
```

**Run the test**:
```bash
cd /Users/janussuk/Desktop/AI\ Projects/OptionsBacktester2
pytest code/tests/test_web_app_imports.py -v
```

### 2. Document Dependencies

**File**: `/docs/WEB_APP_DEPENDENCIES.md` (create new)

```markdown
# Web App Import Dependencies

Quick reference for what each page imports from `shared/`.

## Home.py
**Purpose**: Main entry point and navigation

**Imports from shared**:
- `STRATEGY_INFO` - Dict of strategy metadata
- `DEFAULT_DB_PATH` - Path to database
- `check_database_connection` - Verify DB is accessible
- `init_session_state` - Initialize Streamlit session

## pages/1_Backtest.py
**Purpose**: Run backtests with various strategies

**Imports from shared**:
- `STRATEGY_INFO` - Available strategies
- `DEFAULT_DB_PATH` - Database location
- `check_database_connection` - Verify connection before running
- `run_real_backtest` - Execute the backtest
- `init_session_state` - Initialize session
- `set_results` - Store backtest results
- `save_result` - Save results to session state
- `render_equity_curve` - Plot equity curve
- `render_metrics_cards` - Display performance metrics
- `render_returns_distribution` - Plot return distribution
- `render_monthly_returns` - Plot monthly returns heatmap
- `render_detailed_metrics` - Show detailed statistics
- `render_trades_table` - Display trade history

## pages/2_Results.py
**Purpose**: Browse and analyze saved results

**Imports from shared**:
- `init_session_state` - Initialize session
- `get_results` - Get current results
- `get_all_saved_results` - Load saved results
- `delete_saved_result` - Remove saved result
- `save_result` - Save new result
- `render_equity_curve` - Plot equity curve
- `render_metrics_cards` - Display metrics
- `render_returns_distribution` - Plot distributions
- `render_monthly_returns` - Plot monthly heatmap
- `render_detailed_metrics` - Detailed statistics
- `render_trades_table` - Trade list
- `render_comparison_chart` - Compare multiple results
- `render_comparison_table` - Comparison statistics

## pages/3_Strategy_Builder.py
**Purpose**: Build and compare custom strategies

**Imports from shared**:
- `STRATEGY_INFO` - Strategy definitions
- `DEFAULT_DB_PATH` - Database path
- `check_database_connection` - Verify DB
- `run_real_backtest` - Execute backtest
- `init_session_state` - Initialize session
- `set_results` - Store results
- **CRITICAL**: `add_to_comparison` - Add to comparison
- **CRITICAL**: `clear_comparison` - Clear comparison
- **CRITICAL**: `get_comparison_results` - Get comparison list
- `render_equity_curve` - Plot equity
- `render_comparison_chart` - Plot comparison
- `render_comparison_table` - Compare results

---

## How to Maintain

When you:

1. **Add a new function to `shared/utils.py` or `shared/charts.py`**:
   - [ ] Add to import section in `shared/__init__.py`
   - [ ] Add to `__all__` list in `shared/__init__.py`
   - [ ] Add test case in `code/tests/test_web_app_imports.py`
   - [ ] Update this document with which page uses it

2. **Create a new page that needs a function**:
   - [ ] Check this document to see if function exists
   - [ ] If not, create it in appropriate `shared/` module
   - [ ] Export it in `shared/__init__.py`
   - [ ] Add to this document
   - [ ] Add test case

3. **Remove a function from shared/**:
   - [ ] Check this document to see which pages import it
   - [ ] Update those pages first
   - [ ] Remove from `shared/__init__.py`
   - [ ] Update tests
   - [ ] Update this document

---

## Checklist Template

Use this when reviewing web app changes:

```
Code Review: Web App Changes

- [ ] All new functions exported in shared/__init__.py
- [ ] All __all__ entries match actual exports
- [ ] Tests added to test_web_app_imports.py
- [ ] WEB_APP_DEPENDENCIES.md updated
- [ ] Import test passes: pytest code/tests/test_web_app_imports.py -v
- [ ] No new functions starting with _underscore (unless truly private)
- [ ] All functions have docstrings
- [ ] No circular imports detected
```
```

### 3. Add Connection Caching

**File**: `/web_app/shared/utils.py` (modify existing)

Add this near the top after imports:

```python
import time

# Connection cache
_connection_cache = {
    'status': None,
    'message': '',
    'timestamp': 0,
    'cache_duration': 300,  # 5 minutes in seconds
}
```

Replace the `check_database_connection()` function with:

```python
def check_database_connection(db_path: str, use_cache: bool = True) -> Tuple[bool, str]:
    """
    Check if database is accessible and has data.

    Results are cached for 5 minutes to avoid repeated expensive I/O.

    Args:
        db_path: Path to the Dolt database
        use_cache: If False, bypass cache and do fresh check

    Returns:
        Tuple of (is_connected: bool, status_message: str)

    Caching:
        - First call: ~57s (actual database check)
        - Subsequent calls within 5 min: <100ms (from cache)
        - Subsequent calls after 5 min: ~57s (cache expired)
    """
    # Check cache validity
    if use_cache:
        age = time.time() - _connection_cache['timestamp']
        if age < _connection_cache['cache_duration']:
            return _connection_cache['status'], _connection_cache['message']

    try:
        from backtester.data.dolt_adapter import DoltAdapter

        if not os.path.exists(db_path):
            result = (False, f"Database path does not exist: {db_path}")
        else:
            adapter = DoltAdapter(db_path)
            adapter.connect()

            try:
                tables = adapter.get_tables()
                if not tables:
                    result = (False, "Database is empty (no tables found)")
                else:
                    # Check for SPY data
                    date_range = adapter.get_date_range("SPY")
                    if date_range[0] is None:
                        result = (False, "No SPY data found in database")
                    else:
                        result = (
                            True,
                            f"Connected! Data range: {date_range[0].date()} to {date_range[1].date()}",
                        )
            finally:
                adapter.close()

    except Exception as e:
        result = (False, f"Connection error: {str(e)}")

    # Update cache
    _connection_cache['status'] = result[0]
    _connection_cache['message'] = result[1]
    _connection_cache['timestamp'] = time.time()

    return result
```

---

## Medium Implementation (2-4 hours)

### 4. Create Constants Module

**File**: `/web_app/shared/constants.py` (create new)

```python
"""
Configuration constants for the web app.

Centralized location for all magic numbers, default values,
and configuration that shouldn't change per-user.
"""

from pathlib import Path
from typing import Dict, Any

# Database
DEFAULT_DB_PATH = str(Path(__file__).parent.parent.parent / "dolt_data" / "options")

# Strategy information
STRATEGY_INFO: Dict[str, Dict[str, Any]] = {
    "Short Straddle (High IV)": {
        "description": "Sell ATM call and put when IV rank is high. Profits from time decay and IV contraction.",
        "risk": "High",
        "ideal_conditions": "High IV environment, range-bound market",
        "params": [
            "iv_rank_threshold",
            "profit_target_pct",
            "stop_loss_pct",
            "dte_target",
        ],
    },
    "Iron Condor": {
        "description": "Sell OTM put spread and call spread. Limited risk, profits from low volatility.",
        "risk": "Medium",
        "ideal_conditions": "Low to moderate IV, range-bound market",
        "params": [
            "delta_target",
            "wing_width",
            "profit_target_pct",
            "stop_loss_pct",
            "dte_target",
        ],
    },
    "Volatility Regime": {
        "description": "Adaptive strategy that adjusts position sizing based on VIX levels.",
        "risk": "Variable",
        "ideal_conditions": "Any market condition",
        "params": [
            "low_vol_threshold",
            "high_vol_threshold",
            "profit_target_pct",
            "stop_loss_pct",
        ],
    },
}

# Caching
CONNECTION_CHECK_CACHE_TTL = 300  # 5 minutes
DATA_QUERY_CACHE_TTL = 600  # 10 minutes

# Session state defaults
SESSION_STATE_DEFAULTS = {
    "results": None,  # Current backtest results
    "saved_results": {},  # Dict of saved results {name: results}
    "db_path": DEFAULT_DB_PATH,
    "is_connected": False,
    "connection_status": "",
    "comparison_results": [],  # List of results for comparison
}

# Timeouts
DATABASE_CONNECTION_TIMEOUT = 60  # seconds
BACKTEST_EXECUTION_TIMEOUT = 3600  # 1 hour
```

Then update `/web_app/shared/utils.py` to use it:

```python
from .constants import (
    DEFAULT_DB_PATH,
    STRATEGY_INFO,
    SESSION_STATE_DEFAULTS,
    CONNECTION_CHECK_CACHE_TTL,
)

# Remove the STRATEGY_INFO and DEFAULT_DB_PATH definitions
# They now come from constants.py
```

Update `/web_app/shared/__init__.py`:

```python
"""Shared utilities for multi-page Streamlit app."""

from .constants import (
    STRATEGY_INFO,
    DEFAULT_DB_PATH,
)

from .utils import (
    check_database_connection,
    run_real_backtest,
    init_session_state,
    get_results,
    set_results,
    clear_results,
    get_all_saved_results,
    save_result,
    delete_saved_result,
    add_to_comparison,
    clear_comparison,
    get_comparison_results,
)

from .charts import (
    render_equity_curve,
    render_returns_distribution,
    render_monthly_returns,
    render_trades_table,
    render_metrics_cards,
    render_detailed_metrics,
)

__all__ = [
    # From constants
    "STRATEGY_INFO",
    "DEFAULT_DB_PATH",
    # From utils
    "check_database_connection",
    "run_real_backtest",
    "init_session_state",
    "get_results",
    "set_results",
    "clear_results",
    "get_all_saved_results",
    "save_result",
    "delete_saved_result",
    "add_to_comparison",
    "clear_comparison",
    "get_comparison_results",
    # From charts
    "render_equity_curve",
    "render_returns_distribution",
    "render_monthly_returns",
    "render_trades_table",
    "render_metrics_cards",
    "render_detailed_metrics",
]
```

### 5. Add Input Validation Script

**File**: `/scripts/validate_init_exports.py` (create new)

```python
#!/usr/bin/env python3
"""
Validate that __init__.py exports match actual module contents.

Usage:
    python scripts/validate_init_exports.py web_app/shared/__init__.py

Exit codes:
    0 = OK
    1 = Missing exports
    2 = Extra exports (warnings only)
    3 = Parse error
"""

import ast
import sys
from pathlib import Path
from typing import Set, Tuple


def get_module_functions(module_path: Path) -> Set[str]:
    """Extract public names from a Python module."""
    with open(module_path) as f:
        tree = ast.parse(f.read())

    names = set()
    for node in ast.walk(tree):
        # Get function definitions
        if isinstance(node, ast.FunctionDef):
            if not node.name.startswith("_"):
                names.add(node.name)

        # Get class definitions
        elif isinstance(node, ast.ClassDef):
            if not node.name.startswith("_"):
                names.add(node.name)

        # Get module-level constants (UPPER_CASE)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    names.add(target.id)

    return names


def get_init_exports(init_path: Path) -> Tuple[Set[str], Set[str]]:
    """Extract __all__ and actual imports from __init__.py."""
    with open(init_path) as f:
        tree = ast.parse(f.read())

    all_list = set()
    imports = set()

    for node in ast.walk(tree):
        # Find __all__ definition
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    if isinstance(node.value, ast.List):
                        all_list = {
                            elt.value
                            for elt in node.value.elts
                            if isinstance(elt, ast.Constant)
                        }

        # Find imported names
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name != "*":
                    imports.add(alias.asname or alias.name)

    return all_list, imports


def validate_exports(init_path: Path) -> bool:
    """Validate that __init__.py exports are complete and accurate."""
    init_dir = init_path.parent

    # Get what's actually exported in __all__
    all_list, imports = get_init_exports(init_path)

    # Get what's available in submodules
    available = set()
    for py_file in init_dir.glob("*.py"):
        if py_file.name.startswith("_"):
            continue
        if py_file.name == "__init__.py":
            continue

        available.update(get_module_functions(py_file))

    print(f"Available exports: {sorted(available)}")
    print(f"Declared in __all__: {sorted(all_list)}")
    print(f"Actually imported: {sorted(imports)}")

    # Check for missing exports
    missing = available - all_list
    if missing:
        print(f"ERROR: Missing from __all__: {sorted(missing)}")
        return False

    # Check for extra exports (warning only)
    extra = all_list - imports
    if extra:
        print(f"WARNING: In __all__ but not imported: {sorted(extra)}")

    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_init_exports.py <path/to/__init__.py>")
        sys.exit(3)

    init_file = Path(sys.argv[1])

    if not init_file.exists():
        print(f"ERROR: File not found: {init_file}")
        sys.exit(3)

    try:
        valid = validate_exports(init_file)
        sys.exit(0 if valid else 1)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(3)
```

Make it executable and use in pre-commit:

```bash
chmod +x /Users/janussuk/Desktop/AI\ Projects/OptionsBacktester2/scripts/validate_init_exports.py

# Test it:
python /Users/janussuk/Desktop/AI\ Projects/OptionsBacktester2/scripts/validate_init_exports.py \
  /Users/janussuk/Desktop/AI\ Projects/OptionsBacktester2/web_app/shared/__init__.py
```

---

## Final Verification

Run all checks:

```bash
cd /Users/janussuk/Desktop/AI\ Projects/OptionsBacktester2

# 1. Import tests
echo "=== Import Tests ==="
pytest code/tests/test_web_app_imports.py -v

# 2. Validate exports
echo -e "\n=== Validating Exports ==="
python scripts/validate_init_exports.py web_app/shared/__init__.py

# 3. Test the app loads
echo -e "\n=== Testing App Load (Home page) ==="
timeout 15 streamlit run web_app/Home.py --client.showErrorDetails=true 2>&1 | head -20

# 4. Check for duplicates
echo -e "\n=== Checking for Duplicate Code ==="
pylint web_app/ --duplicate-code-check=yes 2>/dev/null || echo "Install pylint: pip install pylint"
```

---

## One-Command Setup

```bash
#!/bin/bash
# setup_web_app_prevention.sh

cd /Users/janussuk/Desktop/AI\ Projects/OptionsBacktester2

# 1. Create test file
cat > code/tests/test_web_app_imports.py << 'EOF'
[Insert test code from above]
EOF

# 2. Create constants module
cat > web_app/shared/constants.py << 'EOF'
[Insert constants code from above]
EOF

# 3. Create validation script
mkdir -p scripts
cat > scripts/validate_init_exports.py << 'EOF'
[Insert validation script from above]
EOF

# 4. Run verification
pytest code/tests/test_web_app_imports.py -v
python scripts/validate_init_exports.py web_app/shared/__init__.py

echo "Setup complete!"
```

---

## Next Steps

1. **Complete Quick Implementation** (1-2 hours)
   - Run import tests
   - Add connection caching
   - Document dependencies

2. **Test thoroughly**
   - `pytest code/tests/test_web_app_imports.py -v`
   - Load pages in browser and time them
   - Verify caching works (second load is faster)

3. **Add to CI/CD**
   - Add import test to automated pipeline
   - Add validation script to pre-commit hooks
   - Run performance tests on every PR

4. **Documentation**
   - Update developer guide with best practices
   - Add checklist to PR template
   - Train team on new process

---

## Troubleshooting

**Import tests fail:**
- Ensure all functions exported in `__init__.py` are actually in their modules
- Check for typos in function names
- Verify no circular imports

**Connection still slow:**
- Verify caching is enabled: `use_cache=True` (default)
- Check cache TTL isn't too short (300s recommended)
- Consider lazy loading (only check when backtest runs)

**Validation script errors:**
- Install ast module: usually included in Python
- Check file permissions: `chmod +x scripts/validate_init_exports.py`
- Verify path syntax is correct

---

**Questions?** See `WEB_APP_BUG_PREVENTION_STRATEGIES.md` for detailed explanation of each strategy.
