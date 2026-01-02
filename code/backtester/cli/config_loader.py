"""
Configuration Loader for Strategy Definitions

Loads strategy configurations from YAML and JSON files,
validates them, and converts them to StrategyConfig objects.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging

import yaml

from backtester.cli.config_schema import (
    BacktestConfig,
    ConfigValidationError,
    ConfigValidator,
    DataSourceConfig,
    DataSourceType,
    EntryConditionConfig,
    ExitConditionConfig,
    PositionSizeConfig,
    PositionSizeMethod,
    RiskLimitsConfig,
    StrategyConfig,
    StructureConfig,
    StructureType,
)

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Loads and parses strategy configuration files."""

    @classmethod
    def load(cls, path: Union[str, Path]) -> StrategyConfig:
        """
        Load configuration from file.

        Args:
            path: Path to YAML or JSON configuration file

        Returns:
            Parsed and validated StrategyConfig

        Raises:
            FileNotFoundError: If config file doesn't exist
            ConfigValidationError: If configuration is invalid
            ValueError: If file format is unsupported
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        # Load raw data based on extension
        raw_data = cls._load_file(path)

        # Parse into StrategyConfig
        config = cls._parse_config(raw_data)

        # Validate
        errors = ConfigValidator.validate(config)
        if errors:
            raise ConfigValidationError(
                f"Configuration validation failed: {path}", errors=errors
            )

        logger.info(f"Loaded configuration '{config.name}' from {path}")
        return config

    @classmethod
    def load_from_string(cls, content: str, format: str = "yaml") -> StrategyConfig:
        """
        Load configuration from string content.

        Args:
            content: YAML or JSON string
            format: "yaml" or "json"

        Returns:
            Parsed and validated StrategyConfig
        """
        if format.lower() == "yaml":
            raw_data = yaml.safe_load(content)
        elif format.lower() == "json":
            raw_data = json.loads(content)
        else:
            raise ValueError(f"Unsupported format: {format}")

        config = cls._parse_config(raw_data)

        errors = ConfigValidator.validate(config)
        if errors:
            raise ConfigValidationError(
                "Configuration validation failed", errors=errors
            )

        return config

    @classmethod
    def _load_file(cls, path: Path) -> Dict[str, Any]:
        """Load raw data from file."""
        suffix = path.suffix.lower()

        with open(path, "r") as f:
            if suffix in (".yaml", ".yml"):
                return yaml.safe_load(f)
            elif suffix == ".json":
                return json.load(f)
            else:
                raise ValueError(f"Unsupported file format: {suffix}")

    @classmethod
    def _parse_config(cls, data: Dict[str, Any]) -> StrategyConfig:
        """Parse raw dictionary into StrategyConfig."""
        # Parse structure
        structure = None
        if "structure" in data:
            structure = cls._parse_structure(data["structure"])

        # Parse entry conditions
        entry_conditions = []
        if "entry_conditions" in data:
            for cond_data in data["entry_conditions"]:
                entry_conditions.append(cls._parse_entry_condition(cond_data))

        # Parse exit conditions
        exit_conditions = None
        if "exit_conditions" in data:
            exit_conditions = cls._parse_exit_conditions(data["exit_conditions"])

        # Parse position sizing
        position_size = None
        if "position_size" in data:
            position_size = cls._parse_position_size(data["position_size"])

        # Parse risk limits
        risk_limits = None
        if "risk_limits" in data:
            risk_limits = cls._parse_risk_limits(data["risk_limits"])

        # Parse backtest config
        backtest = None
        if "backtest" in data:
            backtest = cls._parse_backtest(data["backtest"])

        # Parse data source
        data_source = None
        if "data_source" in data:
            data_source = cls._parse_data_source(data["data_source"])

        return StrategyConfig(
            name=data.get("name", "Unnamed Strategy"),
            version=data.get("version", "1.0"),
            description=data.get("description"),
            underlying=data.get("underlying", "SPY"),
            structure=structure,
            entry_conditions=entry_conditions,
            exit_conditions=exit_conditions,
            position_size=position_size,
            risk_limits=risk_limits,
            backtest=backtest,
            data_source=data_source,
            template=data.get("template"),
            template_params=data.get("template_params", {}),
        )

    @classmethod
    def _parse_structure(cls, data: Dict[str, Any]) -> StructureConfig:
        """Parse structure configuration."""
        structure_type = data.get("type", "short_straddle")

        # Handle string enum conversion
        if isinstance(structure_type, str):
            structure_type = StructureType(structure_type.lower())

        return StructureConfig(
            type=structure_type,
            dte=data.get("dte", 30),
            delta=data.get("delta"),
            width=data.get("width"),
            quantity=data.get("quantity", 1),
        )

    @classmethod
    def _parse_entry_condition(cls, data: Dict[str, Any]) -> EntryConditionConfig:
        """Parse single entry condition."""
        # Handle compound conditions
        and_conditions = None
        if "and" in data:
            and_conditions = [cls._parse_entry_condition(c) for c in data["and"]]

        or_conditions = None
        if "or" in data:
            or_conditions = [cls._parse_entry_condition(c) for c in data["or"]]

        # Extract params (everything except type, and, or, not)
        params = {
            k: v for k, v in data.items() if k not in ("type", "and", "or", "not")
        }

        return EntryConditionConfig(
            type=data.get("type", "always_true"),
            params=params,
            and_conditions=and_conditions,
            or_conditions=or_conditions,
            negate=data.get("not", False),
        )

    @classmethod
    def _parse_exit_conditions(cls, data: Dict[str, Any]) -> ExitConditionConfig:
        """Parse exit conditions."""
        return ExitConditionConfig(
            profit_target=data.get("profit_target"),
            stop_loss=data.get("stop_loss"),
            fixed_stop_loss=data.get("fixed_stop_loss"),
            trailing_stop=data.get("trailing_stop"),
            max_holding_days=data.get("max_holding_days"),
            min_dte=data.get("min_dte"),
        )

    @classmethod
    def _parse_position_size(cls, data: Dict[str, Any]) -> PositionSizeConfig:
        """Parse position sizing configuration."""
        method = data.get("method", "fixed_contracts")

        if isinstance(method, str):
            method = PositionSizeMethod(method.lower())

        return PositionSizeConfig(
            method=method,
            value=data.get("value", 1.0),
        )

    @classmethod
    def _parse_risk_limits(cls, data: Dict[str, Any]) -> RiskLimitsConfig:
        """Parse risk limits configuration."""
        return RiskLimitsConfig(
            max_positions=data.get("max_positions", 5),
            max_daily_loss=data.get("max_daily_loss"),
            max_daily_loss_pct=data.get("max_daily_loss_pct"),
            max_drawdown=data.get("max_drawdown"),
            max_portfolio_delta=data.get("max_portfolio_delta"),
            max_portfolio_vega=data.get("max_portfolio_vega"),
            max_capital_utilization=data.get("max_capital_utilization", 0.80),
        )

    @classmethod
    def _parse_backtest(cls, data: Dict[str, Any]) -> BacktestConfig:
        """Parse backtest configuration."""
        return BacktestConfig(
            start_date=data.get("start_date"),
            end_date=data.get("end_date"),
            initial_capital=data.get("initial_capital", 100000.0),
            commission_per_contract=data.get("commission_per_contract", 0.65),
            slippage_pct=data.get("slippage_pct", 0.01),
        )

    @classmethod
    def _parse_data_source(cls, data: Dict[str, Any]) -> DataSourceConfig:
        """Parse data source configuration."""
        source_type = data.get("type", "dolt")

        if isinstance(source_type, str):
            source_type = DataSourceType(source_type.lower())

        return DataSourceConfig(
            type=source_type,
            database=data.get("database"),
            directory=data.get("directory"),
            host=data.get("host"),
            port=data.get("port"),
        )


def load_config(path: Union[str, Path]) -> StrategyConfig:
    """
    Convenience function to load a configuration file.

    Args:
        path: Path to YAML or JSON config file

    Returns:
        Validated StrategyConfig
    """
    return ConfigLoader.load(path)


def load_config_string(content: str, format: str = "yaml") -> StrategyConfig:
    """
    Convenience function to load configuration from string.

    Args:
        content: YAML or JSON string
        format: "yaml" or "json"

    Returns:
        Validated StrategyConfig
    """
    return ConfigLoader.load_from_string(content, format)
