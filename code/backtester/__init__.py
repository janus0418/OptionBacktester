"""
Options Backtester Package

A professional-grade options strategy backtesting framework with accurate pricing,
comprehensive analytics, and institutional-quality reporting.

Modules:
    core: Option pricing, Greeks, and volatility surface
    strategies: Strategy framework, builder, templates, and validation
    structures: Pre-built option structures (straddles, condors, etc.)
    engine: Backtest engine, execution model, data streaming
    analytics: Performance metrics, risk analysis, visualization
    data: Data adapters and market data classes
    cli: Command-line interface and configuration management
"""

__version__ = "1.0.0"
__author__ = "Options Backtester Team"

from backtester.cli import (
    load_config,
    load_config_string,
    StrategyConfig,
    Environment,
    get_environment,
    set_environment,
)

__all__ = [
    "__version__",
    "__author__",
    "load_config",
    "load_config_string",
    "StrategyConfig",
    "Environment",
    "get_environment",
    "set_environment",
]
