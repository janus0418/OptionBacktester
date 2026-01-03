# Change Log

All notable changes to the Option Backtester will be documented in this file

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0] - 2026-01-02

### Fixed
- **Web App**: Replace deprecated `use_container_width` with `width="stretch"` parameter (PR #5)
- **Web App**: Remove legacy `streamlit_app.py` file with incorrect `adapter.disconnect()` call (PR #4)
- **Web App**: Add missing exports for comparison functions in `shared/__init__.py` (PR #4)
- **Tests**: Correct Dashboard/ReportGenerator API parameter names in integration test

### Added
- **Documentation**: Add comprehensive bug prevention strategies for web app
- **Documentation**: Create `docs/solutions/` directory for indexed problem solutions
- **Documentation**: Add solution documentation for legacy code cleanup pattern

### Changed
- **Web App**: Replace fake demo data with real BacktestEngine integration
- **Notebooks**: Update API calls to match Monte Carlo and Scenario Testing modules

## [0.3.0] - 2026-01-01

### Added
- **Analytics**: Monte Carlo simulation module for portfolio risk analysis
- **Analytics**: Scenario testing module for stress testing strategies
- **CLI**: Command-line tools for running backtests and managing configurations
- **Strategies**: Strategy Validation Framework for configuration and risk validation
- **Strategies**: Pre-built strategy templates (Short Straddle, Iron Condor, Volatility Regime)
- **Docs**: IV Rank strategy notebook with implementation examples

### Changed
- **Docs**: Update project status for Phase C completion (860 tests passing)