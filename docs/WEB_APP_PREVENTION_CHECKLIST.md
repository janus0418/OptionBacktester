# Web App Bug Prevention - Implementation Checklist

Use this checklist to track implementation of prevention strategies for the three web app bugs.

---

## Overview

This checklist covers implementation of prevention strategies for:
1. Legacy code duplication
2. Module export drift
3. Slow database connections

**Estimated Time**: 4-10 hours depending on scope
**Difficulty**: Low to Medium
**Risk**: Minimal (backward compatible)

---

## Phase 1: Quick Implementation (2-4 hours)

Essential prevention strategies for immediate impact.

### 1.1 Create Import Validation Test

- [ ] Create file: `/code/tests/test_web_app_imports.py`
  - [ ] Copy test code from WEB_APP_PREVENTION_IMPLEMENTATION.md
  - [ ] Verify file contains 7 test functions
  - [ ] Run: `pytest code/tests/test_web_app_imports.py -v`
  - [ ] All tests pass

- [ ] Add to CI/CD pipeline
  - [ ] Add to `.github/workflows/ci.yml` (if using GitHub Actions)
  - [ ] Or add to `.gitlab-ci.yml` (if using GitLab)
  - [ ] Or add to `Makefile` test target
  - [ ] Verify test runs on every commit

- [ ] Update project documentation
  - [ ] Add to `/docs/project_status.md`: "Import validation tests added"
  - [ ] Add reference to WEB_APP_PREVENTION_IMPLEMENTATION.md

**Verification**:
```bash
pytest code/tests/test_web_app_imports.py -v
# Expected: 7 passed
```

**Time**: 30-45 minutes

---

### 1.2 Add Connection Caching

- [ ] Open: `/web_app/shared/utils.py`

- [ ] Add at top of file (after imports):
  ```python
  import time

  _connection_cache = {
      'status': None,
      'message': '',
      'timestamp': 0,
      'cache_duration': 300,  # 5 minutes
  }
  ```

- [ ] Replace `check_database_connection()` function
  - [ ] Copy updated function from WEB_APP_PREVENTION_IMPLEMENTATION.md
  - [ ] Update docstring with caching info
  - [ ] Verify function has `use_cache` parameter

- [ ] Test connection caching
  ```bash
  # Time first call (should take ~57 seconds)
  time python -c "from web_app.shared.utils import check_database_connection; check_database_connection()"

  # Time second call (should be instant)
  time python -c "from web_app.shared.utils import check_database_connection; check_database_connection()"

  # Second call should be 100x faster
  ```

- [ ] Verify cache expires after 5 minutes
  - [ ] Check timestamp logic is correct
  - [ ] Verify fallback to fresh check when cache expires

**Verification**:
- [ ] First call: ~57 seconds
- [ ] Second call: <100 milliseconds
- [ ] Pages load faster on reload

**Time**: 15-30 minutes

---

### 1.3 Document Module Dependencies

- [ ] Create file: `/docs/WEB_APP_DEPENDENCIES.md`
  - [ ] Copy template from WEB_APP_PREVENTION_IMPLEMENTATION.md
  - [ ] Fill in all page imports
  - [ ] Verify Home.py section is complete
  - [ ] Verify pages/1_Backtest.py section is complete
  - [ ] Verify pages/2_Results.py section is complete
  - [ ] Verify pages/3_Strategy_Builder.py section is complete

- [ ] Add checklist section
  - [ ] Copy from implementation guide
  - [ ] Verify checklist covers all scenarios

- [ ] Link from other docs
  - [ ] Update `/docs/architecture.md` with reference
  - [ ] Update `/CLAUDE.md` with reference
  - [ ] Add to PR template as reference

**Verification**:
```bash
# Verify all imports listed actually exist
grep -r "from shared" web_app/pages/*.py | sort | uniq
# Should match entries in WEB_APP_DEPENDENCIES.md
```

**Time**: 15-20 minutes

---

## Phase 2: Medium Implementation (2-4 hours)

Additional improvements for better structure and automation.

### 2.1 Create Constants Module

- [ ] Create file: `/web_app/shared/constants.py`
  - [ ] Copy from WEB_APP_PREVENTION_IMPLEMENTATION.md
  - [ ] Verify DEFAULT_DB_PATH is correct
  - [ ] Verify STRATEGY_INFO matches current strategies
  - [ ] Add all configuration constants

- [ ] Update `/web_app/shared/utils.py`
  - [ ] Remove `DEFAULT_DB_PATH` definition
  - [ ] Remove `STRATEGY_INFO` definition
  - [ ] Add import: `from .constants import DEFAULT_DB_PATH, STRATEGY_INFO`
  - [ ] Update references to use imported versions
  - [ ] Remove duplicate docstrings

- [ ] Update `/web_app/shared/__init__.py`
  - [ ] Update import line:
    ```python
    from .constants import (
        STRATEGY_INFO,
        DEFAULT_DB_PATH,
    )
    ```
  - [ ] Verify tests still pass

- [ ] Update imports in pages (if needed)
  - [ ] Search: `from shared.utils import DEFAULT_DB_PATH`
  - [ ] Change to: `from shared import DEFAULT_DB_PATH`
  - [ ] Search: `from shared.utils import STRATEGY_INFO`
  - [ ] Change to: `from shared import STRATEGY_INFO`

- [ ] Verify no breakage
  - [ ] Run import test: `pytest code/tests/test_web_app_imports.py -v`
  - [ ] Load Home.py in browser: `streamlit run web_app/Home.py`

**Verification**:
```bash
# All imports should work
python -c "from web_app.shared import DEFAULT_DB_PATH, STRATEGY_INFO"
echo "✓ Constants import successful"

# Tests should still pass
pytest code/tests/test_web_app_imports.py -v
```

**Time**: 30-45 minutes

---

### 2.2 Create Export Validation Script

- [ ] Create directory: `/scripts/` (if not exists)
  - [ ] `mkdir -p /Users/janussuk/Desktop/AI\ Projects/OptionsBacktester2/scripts`

- [ ] Create file: `/scripts/validate_init_exports.py`
  - [ ] Copy from WEB_APP_PREVENTION_IMPLEMENTATION.md
  - [ ] Verify all functions are present
  - [ ] Make executable: `chmod +x scripts/validate_init_exports.py`

- [ ] Test the script
  ```bash
  cd /Users/janussuk/Desktop/AI\ Projects/OptionsBacktester2
  python scripts/validate_init_exports.py web_app/shared/__init__.py
  # Should show: Available exports, Declared in __all__, Actually imported
  # Should exit with code 0 (success)
  ```

- [ ] Add to version control
  - [ ] `git add scripts/validate_init_exports.py`
  - [ ] Verify in git: `git ls-files scripts/`

**Verification**:
```bash
python scripts/validate_init_exports.py web_app/shared/__init__.py
# Expected exit code: 0
# Expected: No errors about missing exports
```

**Time**: 20-30 minutes

---

### 2.3 Add Validation to CI/CD

- [ ] Check what CI system is in use
  - [ ] GitHub Actions: `.github/workflows/*.yml`
  - [ ] GitLab CI: `.gitlab-ci.yml`
  - [ ] Jenkins: `Jenkinsfile`
  - [ ] Other: specify which

- [ ] Add import test to CI
  ```yaml
  # Example for GitHub Actions
  - name: Test web app imports
    run: pytest code/tests/test_web_app_imports.py -v
  ```

- [ ] Add export validation to CI
  ```yaml
  - name: Validate module exports
    run: python scripts/validate_init_exports.py web_app/shared/__init__.py
  ```

- [ ] Test CI locally (if available)
  - [ ] Run: `act` (for GitHub Actions)
  - [ ] Or run the commands manually

- [ ] Verify CI runs on pull requests
  - [ ] Create a test PR
  - [ ] Verify tests run automatically
  - [ ] Verify tests must pass before merge

**Verification**:
- [ ] CI file updated
- [ ] Tests listed in CI config
- [ ] CI passes when running locally

**Time**: 15-30 minutes (depends on CI setup)

---

## Phase 3: Long-term Improvements (Next Month)

Additional enhancements for robustness and scalability.

### 3.1 Add Streamlit Caching Decorators

- [ ] Review: `web_app/shared/utils.py`

- [ ] Find `run_real_backtest()` function
  - [ ] Add decorator: `@st.cache_resource(ttl=3600)`
  - [ ] Update docstring with cache info
  - [ ] Test with multiple runs

- [ ] Create separate function for adapter
  ```python
  @st.cache_resource(ttl=300)
  def get_dolt_adapter(db_path: str):
      adapter = DoltAdapter(db_path)
      adapter.connect()
      return adapter
  ```

- [ ] Update `check_database_connection()` to use cached adapter
  - [ ] Use `get_dolt_adapter()`
  - [ ] Avoid duplicate connections

**Verification**:
```bash
# Run backtest twice quickly - second should be much faster
streamlit run web_app/Home.py
# Run same backtest again - should be instant if cached
```

**Time**: 1-2 hours

---

### 3.2 Implement Lazy Loading

- [ ] Update pages to not check connection on load
  - [ ] Move `check_database_connection()` from page load
  - [ ] Call only when backtest is initiated
  - [ ] Show loading status while checking

- [ ] Test page load speed
  - [ ] Before: measure time to render
  - [ ] After: should be significantly faster

**Verification**:
- [ ] Home page loads in <2 seconds
- [ ] No database check on page load
- [ ] Database check happens when backtest starts

**Time**: 1-2 hours

---

### 3.3 Implement Connection Pooling (Production)

- [ ] Create file: `/web_app/shared/database.py`
  - [ ] Implement `DoltConnectionPool` class
  - [ ] Add connection recycling
  - [ ] Add thread safety

- [ ] Integrate pooling into `run_real_backtest()`
  - [ ] Use pool instead of direct connection
  - [ ] Handle connection release

**Verification**:
- [ ] Multiple concurrent backtests work
- [ ] No connection exhaustion errors
- [ ] Performance improved under load

**Time**: 2-3 hours

---

## Verification Checklist

### All Phases Complete?
- [ ] Phase 1: Quick Implementation (2-4 hours)
- [ ] Phase 2: Medium Implementation (2-4 hours)
- [ ] Phase 3: Long-term Improvements (scheduled for later)

### Code Quality
- [ ] All tests pass: `pytest code/tests/test_web_app_imports.py -v`
- [ ] No import errors when loading pages
- [ ] No duplicate code >10 lines
- [ ] All docstrings present
- [ ] No circular imports

### Performance
- [ ] Home page loads in <5 seconds
- [ ] Connection check cached (2nd call <100ms)
- [ ] No blocking database operations on page load

### Documentation
- [ ] WEB_APP_DEPENDENCIES.md created and complete
- [ ] All new files have docstrings
- [ ] Code comments explain caching logic
- [ ] README updated with reference to prevention strategies

### Testing
- [ ] Import validation test passes
- [ ] Connection caching test passes
- [ ] No breakage to existing pages
- [ ] Backtest still works correctly

---

## Sign-Off

### Developer
- Name: _______________________
- Date: _______________________
- Items completed: _______________________

### Code Reviewer
- Name: _______________________
- Date: _______________________
- Approved: [ ] Yes [ ] No
- Comments: _______________________

### QA/Testing
- Name: _______________________
- Date: _______________________
- Tests passed: [ ] All [ ] Some [ ] None
- Issues found: _______________________

---

## Timeline Example

### Week 1
- [ ] Monday: Create import test, get passing
- [ ] Tuesday: Add connection caching
- [ ] Wednesday: Document dependencies
- [ ] Thursday: Code review and fixes
- [ ] Friday: Merge to main, deploy to staging

### Week 2
- [ ] Monday: Create constants module
- [ ] Tuesday: Add validation script
- [ ] Wednesday: Integrate to CI/CD
- [ ] Thursday: Testing and debugging
- [ ] Friday: Deploy to production

### Month 2
- [ ] Add caching decorators
- [ ] Implement lazy loading
- [ ] Performance testing

### Month 3
- [ ] Connection pooling
- [ ] Load testing
- [ ] Final optimizations

---

## Troubleshooting

### Import Test Fails
**Problem**: `ImportError: cannot import name 'add_to_comparison'`
**Solution**:
- [ ] Check it's defined in `shared/utils.py`
- [ ] Check it's imported in `shared/__init__.py`
- [ ] Check it's in `__all__` list
- [ ] Run: `python scripts/validate_init_exports.py web_app/shared/__init__.py`

### Connection Still Slow
**Problem**: Second call to `check_database_connection()` still takes 57 seconds
**Solution**:
- [ ] Verify cache duration is 300 (5 minutes)
- [ ] Verify `use_cache=True` (default)
- [ ] Check timestamp logic in caching code
- [ ] Add debug print to see if cache is being used

### CI Tests Not Running
**Problem**: Tests don't run automatically on commit
**Solution**:
- [ ] Verify CI configuration file exists
- [ ] Check pytest is installed: `pip install pytest`
- [ ] Run locally: `pytest code/tests/test_web_app_imports.py -v`
- [ ] Add to CI manually if needed

### Validation Script Errors
**Problem**: `SyntaxError` or `AttributeError` in validation script
**Solution**:
- [ ] Verify Python 3.7+ installed
- [ ] Check file paths are correct
- [ ] Run with verbose flag: `python scripts/validate_init_exports.py web_app/shared/__init__.py -v`
- [ ] Check for tabs vs spaces in indentation

---

## Rollback Plan

If something goes wrong:

### Rollback Option 1: Git
```bash
# Revert all changes
git revert HEAD~5..HEAD

# Or go back to specific commit
git checkout [commit_hash] -- web_app/shared/
```

### Rollback Option 2: Selective
```bash
# Keep some changes, revert others
git checkout HEAD -- web_app/shared/utils.py  # Keep only utils.py changes
```

### Rollback Option 3: Full Reset
```bash
# Reset to before changes
git reset --hard [known_good_commit]
```

---

## Success Metrics

### Before Prevention
- Import errors: ~1 per sprint (or per bug fix cycle)
- Page load time: 57+ seconds
- Code duplication: Present but not detected
- Test coverage: No specific import tests

### After Prevention (Target)
- Import errors: 0 (caught by CI before merge)
- Page load time: <1 second
- Code duplication: Caught immediately by CI
- Test coverage: 7 dedicated import validation tests

### How to Measure
```bash
# Test 1: Import validation
pytest code/tests/test_web_app_imports.py -v
# Expected: 7 passed

# Test 2: Connection caching
time python -c "from web_app.shared.utils import check_database_connection; check_database_connection()"
time python -c "from web_app.shared.utils import check_database_connection; check_database_connection()"
# Expected: 1st call ~57s, 2nd call <0.1s

# Test 3: Page load
time streamlit run web_app/Home.py --logger.level=error
# Expected: <5 seconds (no database check on startup)
```

---

## References

### Documentation
- `/docs/WEB_APP_PREVENTION_SUMMARY.md` - Overview
- `/docs/WEB_APP_BUG_PREVENTION_STRATEGIES.md` - Detailed strategies
- `/docs/WEB_APP_PREVENTION_IMPLEMENTATION.md` - Code examples
- `/docs/WEB_APP_DEPENDENCIES.md` - Import dependencies

### Related Files
- `/web_app/shared/__init__.py` - Module exports
- `/web_app/shared/utils.py` - Core utilities
- `/code/tests/` - Test directory
- `/CLAUDE.md` - Project instructions

### Tools Used
- `pytest` - Test framework
- `streamlit` - Web framework
- Python `ast` module - Code analysis
- `git` - Version control

---

## Final Notes

### Key Principles
1. **Automate what can be automated** - Let CI catch bugs
2. **Cache expensive operations** - Don't repeat 57s checks
3. **Document dependencies** - Know what imports where
4. **Test early, test often** - Catch issues immediately
5. **Make changes backward compatible** - No surprises

### Remember
- This is not a breaking change - all improvements are additive
- Start with Phase 1 (quick wins), add Phase 2 (structure), do Phase 3 (optimization) later
- Tests give confidence that changes work correctly
- Documentation helps future developers understand the structure

---

**Document Version**: 1.0
**Created**: January 2, 2026
**Last Updated**: 2026-01-02
**Status**: Ready for Implementation
