# Web App Bug Prevention Strategies

## Executive Summary

This document outlines prevention strategies for three critical bugs that affected the web app during development:

1. **Legacy code duplication** - `streamlit_app.py` (943 lines of duplicated code with bugs)
2. **Module export drift** - `__init__.py` exports out of sync with actual function implementations
3. **Slow database connections** - 57s+ connection times made pages appear broken during loading

---

## Bug 1: Legacy Code Duplication

### Problem Analysis

**What Happened:**
- `streamlit_app.py` was a monolithic 943-line file containing duplicated code from `shared/utils.py` and `shared/charts.py`
- It had a bug calling non-existent `adapter.disconnect()` method (should be `adapter.close()`)
- Never tracked in git but still left in codebase, creating maintenance burden
- Multi-page app structure (`Home.py + pages/`) was already the intended design but legacy file persisted

**Why This Was Problematic:**
- Duplicate code in two places = maintenance burden and risk of inconsistency
- Bug proved code was unmaintained
- Unclear which was the "canonical" version for new developers
- Wasted effort fixing bugs in legacy version when new version had same structure

**Root Causes:**
- No cleanup process when refactoring to new structure
- No automated checks for duplicate code patterns
- No clear "entry point" enforcement in documentation

### Prevention Strategies

#### 1. Enforce Single Source of Truth (SSOT) Principle
- **Guideline**: Designate one canonical location for each piece of functionality
  - Core utilities → `shared/` module only
  - Page layouts → `pages/` directory structure
  - Configuration → `shared/config.py` (to be created)

- **Implementation**:
  - Document the expected structure in `/docs/architecture.md`
  - Add a "Project Structure" section to `CLAUDE.md` with explicit file purpose
  - Mark deprecated/legacy files clearly in their docstring

- **Example docstring for clarity**:
  ```python
  """
  DEPRECATED: This file should not be used. Use shared/utils.py instead.
  Kept only for reference. Delete before production release.
  See /docs/architecture.md for current structure.
  """
  ```

#### 2. Automated Code Duplication Detection
- **Tool**: Add `vulture` or `pylint` duplicate-code detection to CI
  ```yaml
  # Add to .github/workflows/ci.yml or pre-commit hooks
  - repo: https://github.com/PyCQA/pylint
    hooks:
      - id: pylint
        args: [--duplicate-code-check=yes, --disable=all, --enable=duplicate-code]
  ```

- **Threshold**: Fail CI if >10 consecutive lines of code are duplicated across files
- **When**: Run on every commit to catch duplicates immediately

#### 3. Documentation-Driven Development
- **Rule**: Before creating a file/module, update architecture docs with:
  - File path and purpose
  - What it exports (for modules)
  - Why it exists (new feature vs legacy)
  - When it will be removed (if deprecated)

- **Template** (add to `/docs/MODULE_REGISTRY.md`):
  ```markdown
  | Module | Purpose | Status | Review Date |
  |--------|---------|--------|-------------|
  | shared/utils.py | Core utilities, database, backtest | Active | Jan 2024 |
  | shared/charts.py | Plotting functions | Active | Jan 2024 |
  | streamlit_app.py | LEGACY - use Home.py instead | Deprecated | DELETE |
  ```

#### 4. Regular Code Cleanup Sprints
- **Frequency**: Monthly 30-min review
- **Checklist**:
  - [ ] Any files not imported by entry points?
  - [ ] Any functions defined but never called?
  - [ ] Any deprecated warnings triggered?
  - [ ] Any dead code branches?

- **Process**:
  ```bash
  # Add to CI to find unused code
  vulture web_app/ --min-confidence 100
  ```

---

## Bug 2: Module Export Drift

### Problem Analysis

**What Happened:**
- `web_app/shared/__init__.py` was missing exports for comparison functions:
  - `add_to_comparison()`
  - `clear_comparison()`
  - `get_comparison_results()`
- Pages 3 and others tried to import these but got `ImportError`
- Functions existed in `shared/utils.py` but weren't listed in `__init__.py`'s `__all__`

**Why This Was Problematic:**
- Inconsistent interface between what's documented and what's actually available
- Pages break with cryptic `ImportError` instead of clear "function missing" message
- When new developers add functions, they may forget to export them
- No automatic validation that exports match implementation

**Root Causes:**
- Manual maintenance of `__init__.py` without automation
- No test validating imports work
- No documentation linking what each page needs to what's exported

### Prevention Strategies

#### 1. Generate `__init__.py` Automatically (Recommended)
- **Tool**: Use `__init__` generator or explicit import validation

- **Option A - Explicit with comments**:
  ```python
  # web_app/shared/__init__.py
  """Shared utilities for multi-page Streamlit app."""

  # Import all public functions
  from .utils import (
      # Configuration
      STRATEGY_INFO,
      DEFAULT_DB_PATH,

      # Database
      check_database_connection,

      # Backtest execution
      run_real_backtest,

      # Session state management
      init_session_state,
      get_results,
      set_results,
      clear_results,

      # Result persistence
      get_all_saved_results,
      save_result,
      delete_saved_result,

      # Result comparison (for Strategy Builder)
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
      # utils.py exports
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
      # charts.py exports
      "render_equity_curve",
      "render_returns_distribution",
      "render_monthly_returns",
      "render_trades_table",
      "render_metrics_cards",
      "render_detailed_metrics",
  ]
  ```

- **Option B - Script-based generation** (more scalable):
  ```python
  # scripts/generate_init.py
  """Auto-generate __init__.py from source modules."""

  import ast
  import sys
  from pathlib import Path

  def find_public_functions(module_path):
      """Extract all public functions from a module."""
      with open(module_path) as f:
          tree = ast.parse(f.read())

      functions = []
      for node in ast.walk(tree):
          if isinstance(node, ast.FunctionDef):
              if not node.name.startswith('_'):
                  functions.append(node.name)
          elif isinstance(node, ast.Assign):
              # Also capture module-level constants
              for target in node.targets:
                  if isinstance(target, ast.Name) and target.id.isupper():
                      functions.append(target.id)

      return sorted(functions)

  # Then generate imports and __all__ automatically
  ```

#### 2. Validate Imports in Tests
- **Test**: Verify all page imports work at startup
  ```python
  # code/tests/test_web_app_imports.py
  """Test that all web app modules can be imported successfully."""

  import sys
  from pathlib import Path

  def test_shared_module_imports():
      """Verify shared module exports are available."""
      web_app_path = Path(__file__).parent.parent.parent / "web_app"
      sys.path.insert(0, str(web_app_path))

      # These imports should NOT raise ImportError
      from shared import (
          STRATEGY_INFO,
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
          render_equity_curve,
          render_returns_distribution,
          render_monthly_returns,
          render_trades_table,
          render_metrics_cards,
          render_detailed_metrics,
      )

      # Verify they're callable
      assert callable(check_database_connection)
      assert callable(run_real_backtest)
      assert isinstance(STRATEGY_INFO, dict)

  def test_page_imports():
      """Verify each page can import what it needs."""
      # This would actually try to import each page's imports
      # without running the page itself
      pass
  ```

- **CI Integration**: Run this test on every commit:
  ```bash
  pytest code/tests/test_web_app_imports.py -v
  ```

#### 3. Import Dependency Documentation
- **Create**: `/docs/WEB_APP_DEPENDENCIES.md`
  ```markdown
  # Web App Import Dependencies

  ## Home.py
  - check_database_connection
  - init_session_state
  - STRATEGY_INFO
  - DEFAULT_DB_PATH

  ## pages/1_Backtest.py
  - STRATEGY_INFO, DEFAULT_DB_PATH, check_database_connection
  - run_real_backtest, init_session_state, set_results, save_result
  - [chart functions...]

  ## pages/3_Strategy_Builder.py
  - add_to_comparison ← THIS WAS MISSING!
  - clear_comparison ← THIS WAS MISSING!
  - get_comparison_results ← THIS WAS MISSING!
  - [other imports...]
  ```

- **Rule**: Before modifying `__init__.py`, check this document to ensure all dependencies are covered

#### 4. Linting Rule for `__all__`
- **Tool**: Add `pylint` configuration to enforce `__all__` completeness
  ```ini
  # .pylintrc
  [VARIABLES]
  # Flag if __all__ doesn't match public functions
  preferred-modules=no-private-attribute
  ```

- **Pre-commit hook**:
  ```bash
  #!/bin/bash
  # .git/hooks/pre-commit
  python scripts/validate_init_exports.py web_app/shared/__init__.py
  ```

---

## Bug 3: Slow Database Connections

### Problem Analysis

**What Happened:**
- `check_database_connection()` took 57+ seconds to return
- DoltAdapter connection + table enumeration + date range query was blocking
- Users saw blank/loading pages, thought app was broken
- No visual feedback that work was happening in background

**Why This Was Problematic:**
- Synchronous database operations blocked the entire UI
- First page load (Home.py) checked connection, causing 57s delay
- Every page transition that checked connection status added delay
- Users had no indication of progress or expected wait time
- Slow feedback loop made development/testing painful

**Root Causes:**
- DoltAdapter connection is I/O bound (database queries)
- Connection check was doing multiple queries sequentially without caching
- No async/background execution in Streamlit
- No caching of connection state between page reloads

### Prevention Strategies

#### 1. Connection Status Caching (Immediate Win)
- **Approach**: Cache connection check result to avoid repeated I/O
- **Lifetime**: 5-10 minutes (balance between freshness and performance)

- **Implementation**:
  ```python
  # web_app/shared/utils.py
  import time
  from functools import lru_cache

  # Global cache for connection status
  _connection_cache = {
      'status': None,
      'message': '',
      'timestamp': 0,
      'cache_duration': 300,  # 5 minutes
  }

  def check_database_connection(db_path: str, use_cache=True) -> Tuple[bool, str]:
      """
      Check if database is accessible and has data.

      Results are cached for 5 minutes to avoid repeated I/O.
      """
      # Check cache validity
      if use_cache:
          age = time.time() - _connection_cache['timestamp']
          if age < _connection_cache['cache_duration']:
              return _connection_cache['status'], _connection_cache['message']

      try:
          from backtester.data.dolt_adapter import DoltAdapter

          if not os.path.exists(db_path):
              result = False, f"Database path does not exist: {db_path}"
          else:
              adapter = DoltAdapter(db_path)
              adapter.connect()

              tables = adapter.get_tables()
              if not tables:
                  result = False, "Database is empty (no tables found)"
              else:
                  date_range = adapter.get_date_range("SPY")
                  if date_range[0] is None:
                      result = False, "No SPY data found in database"
                  else:
                      result = (
                          True,
                          f"Connected! Data range: {date_range[0].date()} to {date_range[1].date()}",
                      )

              adapter.close()
      except Exception as e:
          result = False, f"Connection error: {str(e)}"

      # Update cache
      _connection_cache['status'] = result[0]
      _connection_cache['message'] = result[1]
      _connection_cache['timestamp'] = time.time()

      return result
  ```

#### 2. Streamlit Caching Decorators (Best Practice)
- **Tool**: Use `@st.cache_resource` for expensive operations

- **Implementation**:
  ```python
  # web_app/shared/utils.py
  import streamlit as st

  @st.cache_resource(ttl=300)  # Cache for 5 minutes
  def get_dolt_adapter(db_path: str):
      """Get or create a DoltAdapter instance (cached)."""
      from backtester.data.dolt_adapter import DoltAdapter

      adapter = DoltAdapter(db_path)
      adapter.connect()
      return adapter

  @st.cache_data(ttl=600)  # Cache query results for 10 minutes
  def get_cached_date_range(db_path: str, symbol: str):
      """Get date range with caching."""
      adapter = get_dolt_adapter(db_path)
      return adapter.get_date_range(symbol)
  ```

#### 3. Loading Indicators & Progress Feedback
- **Current State**: App already has `st.spinner()` for backtest execution
- **Enhancement**: Add progress indicators for connection checks

- **Implementation**:
  ```python
  # Home.py (or any page doing background work)

  # Show connection status with immediate feedback
  connection_col, status_col = st.sidebar.columns([1, 2])

  with connection_col:
      status_placeholder = st.empty()

  with status_col:
      status_placeholder.write("🔄 Checking connection...")

  # Do the check
  is_connected, status_msg = check_database_connection(db_path)

  # Update UI
  with status_col:
      if is_connected:
          status_placeholder.success(f"✅ {status_msg}")
      else:
          status_placeholder.error(f"❌ {status_msg}")
  ```

- **Better approach using session state**:
  ```python
  # pages/1_Backtest.py

  @st.fragment  # Streamlit 1.30+
  def render_sidebar():
      """Sidebar that can update independently."""

      st.sidebar.title("🚀 Backtest Configuration")

      if 'db_check_done' not in st.session_state:
          with st.sidebar.status("🔄 Checking database...", expanded=True) as status:
              is_connected, status_msg = check_database_connection(db_path)
              st.session_state.is_connected = is_connected

              if is_connected:
                  status.update(label="✅ Database connected", state="complete")
              else:
                  status.update(label="❌ Connection failed", state="error")

          st.session_state.db_check_done = True
  ```

#### 4. Lazy Database Connection (Advanced)
- **Approach**: Don't check connection on page load; only check when backtest is about to run

- **Benefits**:
  - Home page loads instantly (no DB check)
  - User gets immediate feedback they're in the app
  - Connection check happens only when needed
  - Covers network failures during backtest

- **Implementation**:
  ```python
  # pages/1_Backtest.py

  if run_button:
      # Only check connection when user initiates action
      with st.spinner("Verifying database connection..."):
          is_connected, status_msg = check_database_connection(db_path)

      if not is_connected:
          st.error(f"Database connection failed: {status_msg}")
      else:
          with st.spinner(f"Running {strategy} backtest..."):
              # Run backtest
              results = run_real_backtest(...)
  ```

#### 5. Database Connection Pooling (Production)
- **For larger deployments**: Use connection pooling to avoid repeated connection overhead

- **Implementation** (future enhancement):
  ```python
  # web_app/shared/database.py (new file)
  """Connection pooling for Dolt database."""

  from queue import Queue
  import threading

  class DoltConnectionPool:
      """Thread-safe connection pool for Dolt adapters."""

      def __init__(self, db_path: str, pool_size: int = 3):
          self.db_path = db_path
          self.pool = Queue(maxsize=pool_size)
          self.lock = threading.Lock()

          # Pre-populate pool
          for _ in range(pool_size):
              adapter = self._create_adapter()
              self.pool.put(adapter)

      def _create_adapter(self):
          from backtester.data.dolt_adapter import DoltAdapter
          adapter = DoltAdapter(self.db_path)
          adapter.connect()
          return adapter

      def acquire(self):
          """Get a connection from the pool."""
          return self.pool.get()

      def release(self, adapter):
          """Return a connection to the pool."""
          self.pool.put(adapter)
  ```

---

## Best Practices for Multi-Page Streamlit Apps

### Structure & Organization

**Recommended Directory Layout:**
```
web_app/
├── Home.py                    # Entry point
├── pages/
│   ├── 1_Backtest.py
│   ├── 2_Results.py
│   └── 3_Strategy_Builder.py
├── shared/
│   ├── __init__.py           # Central exports
│   ├── utils.py              # Core utilities
│   ├── charts.py             # Visualization
│   ├── database.py           # DB operations (new)
│   ├── caching.py            # Cache utilities (new)
│   └── constants.py          # Config (new)
├── components/               # Reusable UI components (new)
│   ├── forms.py
│   ├── cards.py
│   └── tables.py
└── tests/
    ├── test_imports.py       # New: validate structure
    └── test_pages.py         # New: basic page functionality
```

### Module Organization Rules

1. **One responsibility per module**
   - `utils.py` → Backtest execution & session management
   - `charts.py` → All visualization code
   - `database.py` → Database operations (future)
   - `constants.py` → Configuration (future)

2. **Explicit is better than implicit**
   - Always use relative imports in `shared/` module
   - Always export via `__init__.py`
   - Always document public API

3. **Centralize configuration**
   ```python
   # shared/constants.py
   DEFAULT_DB_PATH = ...
   STRATEGY_INFO = ...
   CHART_COLORS = ...
   CACHE_TTL = ...
   ```

4. **Reusable components**
   ```python
   # shared/components/forms.py
   def strategy_selector() -> str:
       """Reusable strategy selection widget."""
       return st.selectbox("Strategy", list(STRATEGY_INFO.keys()))

   # Then use in any page:
   strategy = strategy_selector()
   ```

### Code Quality Rules

1. **Import validation (required)**
   - Every module change → Run import tests
   - CI should validate: `pytest code/tests/test_web_app_imports.py`

2. **Naming conventions**
   - Public functions: `snake_case`
   - Private functions: `_snake_case`
   - Classes: `PascalCase`
   - Constants: `UPPER_CASE`

3. **Documentation requirements**
   ```python
   def my_function(param: str) -> bool:
       """
       Brief description.

       Args:
           param: What this does

       Returns:
           What is returned and what it means

       Raises:
           ValueError: When this happens

       Example:
           >>> result = my_function("test")
           >>> assert result is True
       """
   ```

4. **Performance considerations**
   - Use `@st.cache_resource` for expensive connections
   - Use `@st.cache_data` for expensive computations
   - Document cache TTL in docstring
   - Test with slow database to catch bottlenecks

---

## Test Cases to Add

### 1. Import Validation Test (Critical)
```python
# code/tests/test_web_app_imports.py
def test_all_shared_exports_available():
    """Verify all public functions in shared/ can be imported."""
    # [See implementation above]

def test_no_import_errors_on_page_load():
    """Simulate loading each page without executing page code."""
    # Validate import statements in each page file work
    pass

def test_missing_exports_detected():
    """Ensure if a page tries to import non-existent function, test fails."""
    # This is the opposite: verify ImportError is raised correctly
    pass
```

### 2. Database Connection Performance Test
```python
def test_database_connection_caching():
    """Verify connection check is cached and doesn't repeat queries."""
    import time

    # First call should take ~57s
    start = time.time()
    result1, msg1 = check_database_connection(db_path, use_cache=True)
    duration1 = time.time() - start

    # Second call should be instant (cached)
    start = time.time()
    result2, msg2 = check_database_connection(db_path, use_cache=True)
    duration2 = time.time() - start

    assert result1 == result2
    assert msg1 == msg2
    assert duration2 < 0.1  # Should be cached (< 100ms)
    assert duration1 > duration2  # First was slower

def test_database_connection_timeout():
    """Verify connection doesn't hang beyond reasonable timeout."""
    import signal

    def timeout_handler(signum, frame):
        raise TimeoutError("Connection check exceeded 30 seconds")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(30)  # 30 second timeout

    try:
        is_connected, msg = check_database_connection(invalid_path)
    finally:
        signal.alarm(0)  # Cancel alarm
```

### 3. Module Structure Validation Test
```python
def test_no_duplicate_code():
    """Use pylint to detect code duplication."""
    # Should fail if duplicate code blocks > 10 lines found
    pass

def test_all_constants_exported():
    """Verify all UPPER_CASE constants in utils.py are in __all__."""
    # Parse utils.py, find constants, verify in __init__.py
    pass

def test_circular_imports():
    """Verify no circular import dependencies."""
    # Use Python's import system to detect cycles
    pass
```

### 4. Performance Regression Tests
```python
def test_home_page_loads_in_under_10_seconds():
    """Home page should load quickly (no database check on load)."""
    # Measure page load time
    pass

def test_backtest_page_startup_responsive():
    """Backtest page UI appears before connection check completes."""
    # Verify lazy connection check doesn't block sidebar rendering
    pass
```

---

## Implementation Roadmap

### Phase 1: Immediate (This Sprint)
- [ ] Add import validation test to CI
- [ ] Document all exports in `/docs/WEB_APP_DEPENDENCIES.md`
- [ ] Add connection caching to prevent 57s checks
- [ ] Review `shared/__init__.py` for completeness

### Phase 2: Short-term (Next Sprint)
- [ ] Create `shared/constants.py` for configuration
- [ ] Refactor `shared/` to follow organization rules
- [ ] Add `shared/components/` for reusable UI elements
- [ ] Implement `@st.cache_resource` for adapters
- [ ] Create `scripts/validate_init_exports.py` tool

### Phase 3: Medium-term (Q2)
- [ ] Create `shared/database.py` for DB operations
- [ ] Implement connection pooling
- [ ] Add comprehensive test suite for web app
- [ ] Update developer docs with best practices
- [ ] Create `shared/caching.py` utilities

### Phase 4: Production (Q3)
- [ ] Performance profiling of all pages
- [ ] Load testing with concurrent users
- [ ] Production monitoring/alerting
- [ ] Connection pooling in staging
- [ ] Update docs with lessons learned

---

## Checklist for Future Development

When adding new features to the web app:

- [ ] **Code Organization**
  - [ ] Function goes in appropriate `shared/` module
  - [ ] Function is documented with docstring
  - [ ] Function is exported in `shared/__init__.py`
  - [ ] Function is tested in `code/tests/test_web_app_imports.py`

- [ ] **Performance**
  - [ ] Database operations are cached with `@st.cache_resource` or `@st.cache_data`
  - [ ] Cache TTL is appropriate (5-10 min for DB, longer for data)
  - [ ] Page load time is < 5 seconds without user action
  - [ ] Long operations have loading indicators

- [ ] **Documentation**
  - [ ] Module purpose documented in docstring
  - [ ] Public API documented
  - [ ] Dependencies listed in `/docs/WEB_APP_DEPENDENCIES.md`
  - [ ] Architecture updated if structure changes

- [ ] **Testing**
  - [ ] Import validation tests pass
  - [ ] No circular imports
  - [ ] No duplicate code patterns
  - [ ] Performance tests pass

- [ ] **Deployment**
  - [ ] Code review for structure compliance
  - [ ] All tests pass in CI
  - [ ] No breaking changes to public API
  - [ ] Changelog updated

---

## References

- **Current Implementation**: `/web_app/shared/__init__.py` (export structure)
- **Related Bugs Fixed**:
  - Commit `a3774f6`: Removed legacy `streamlit_app.py`, added missing exports
  - Commit `ffb5201`: Replaced fake data with real BacktestEngine integration
- **Streamlit Caching**: https://docs.streamlit.io/library/advanced-features/caching
- **Python Import Best Practices**: PEP 8, Python docs on packages
