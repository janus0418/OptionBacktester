"""
CLI Package for Options Backtester

Provides command-line interface tools for running backtests,
validating configurations, and managing environments.

Usage:
    # Run a backtest
    backtest run --config strategy.yaml

    # Validate a configuration
    backtest validate --config strategy.yaml

    # List available templates
    backtest list templates

    # Generate a report
    backtest report --results results.json --output report.html
"""

from backtester.cli.config_schema import (
    # Enums
    StructureType,
    PositionSizeMethod,
    DataSourceType,
    # Config Classes
    EntryConditionConfig,
    ExitConditionConfig,
    StructureConfig,
    PositionSizeConfig,
    RiskLimitsConfig,
    BacktestConfig,
    DataSourceConfig,
    StrategyConfig,
    # Validation
    ConfigValidator,
    ConfigValidationError,
    validate_config,
)

from backtester.cli.config_loader import (
    ConfigLoader,
    load_config,
    load_config_string,
)

from backtester.cli.environment import (
    Environment,
    EnvironmentSettings,
    EnvironmentManager,
    get_environment,
    get_settings,
    set_environment,
    configure_logging,
)

__all__ = [
    # Schema Enums
    "StructureType",
    "PositionSizeMethod",
    "DataSourceType",
    # Config Classes
    "EntryConditionConfig",
    "ExitConditionConfig",
    "StructureConfig",
    "PositionSizeConfig",
    "RiskLimitsConfig",
    "BacktestConfig",
    "DataSourceConfig",
    "StrategyConfig",
    # Validation
    "ConfigValidator",
    "ConfigValidationError",
    "validate_config",
    # Loader
    "ConfigLoader",
    "load_config",
    "load_config_string",
    # Environment
    "Environment",
    "EnvironmentSettings",
    "EnvironmentManager",
    "get_environment",
    "get_settings",
    "set_environment",
    "configure_logging",
]
