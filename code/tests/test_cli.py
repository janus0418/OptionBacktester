"""
Tests for CLI Module

Tests configuration schema, loading, validation, environment management,
and CLI commands.
"""

import json
import os
import tempfile
from datetime import date, datetime
from pathlib import Path
from unittest.mock import patch

import pytest
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
    validate_config,
)
from backtester.cli.config_loader import (
    ConfigLoader,
    load_config,
    load_config_string,
)
from backtester.cli.environment import (
    Environment,
    EnvironmentManager,
    EnvironmentSettings,
    get_environment,
    get_settings,
    set_environment,
)


class TestConfigSchema:
    def test_structure_type_enum(self):
        assert StructureType.SHORT_STRADDLE.value == "short_straddle"
        assert StructureType.IRON_CONDOR.value == "iron_condor"

    def test_position_size_method_enum(self):
        assert PositionSizeMethod.FIXED_CONTRACTS.value == "fixed_contracts"
        assert PositionSizeMethod.RISK_PERCENT.value == "risk_percent"

    def test_data_source_type_enum(self):
        assert DataSourceType.DOLT.value == "dolt"
        assert DataSourceType.CSV.value == "csv"
        assert DataSourceType.MOCK.value == "mock"

    def test_entry_condition_config_creation(self):
        config = EntryConditionConfig(type="iv_rank_above", params={"threshold": 70})
        assert config.type == "iv_rank_above"
        assert config.params["threshold"] == 70
        assert config.negate is False

    def test_exit_condition_config_creation(self):
        config = ExitConditionConfig(profit_target=0.50, stop_loss=2.0, min_dte=7)
        assert config.profit_target == 0.50
        assert config.stop_loss == 2.0
        assert config.min_dte == 7

    def test_structure_config_creation(self):
        config = StructureConfig(type=StructureType.SHORT_STRADDLE, dte=30, delta=0.30)
        assert config.type == StructureType.SHORT_STRADDLE
        assert config.dte == 30
        assert config.delta == 0.30

    def test_position_size_config_creation(self):
        config = PositionSizeConfig(method=PositionSizeMethod.RISK_PERCENT, value=0.02)
        assert config.method == PositionSizeMethod.RISK_PERCENT
        assert config.value == 0.02

    def test_risk_limits_config_creation(self):
        config = RiskLimitsConfig(
            max_positions=5, max_drawdown=0.20, max_capital_utilization=0.80
        )
        assert config.max_positions == 5
        assert config.max_drawdown == 0.20

    def test_backtest_config_creation(self):
        config = BacktestConfig(
            start_date="2024-01-01", end_date="2024-12-31", initial_capital=100000.0
        )
        assert config.initial_capital == 100000.0

    def test_strategy_config_creation(self):
        config = StrategyConfig(name="Test Strategy", underlying="SPY", version="1.0")
        assert config.name == "Test Strategy"
        assert config.underlying == "SPY"


class TestConfigValidator:
    def test_validate_valid_config(self):
        config = StrategyConfig(
            name="Valid Strategy",
            underlying="SPY",
            structure=StructureConfig(type=StructureType.SHORT_STRADDLE, dte=30),
        )
        errors = ConfigValidator.validate(config)
        assert len(errors) == 0

    def test_validate_missing_name(self):
        config = StrategyConfig(name="", underlying="SPY")
        errors = ConfigValidator.validate(config)
        assert any("name" in e.lower() for e in errors)

    def test_validate_missing_underlying(self):
        config = StrategyConfig(name="Test", underlying="")
        errors = ConfigValidator.validate(config)
        assert any("underlying" in e.lower() for e in errors)

    def test_validate_structure_invalid_dte(self):
        config = StrategyConfig(
            name="Test",
            underlying="SPY",
            structure=StructureConfig(type=StructureType.SHORT_STRADDLE, dte=0),
        )
        errors = ConfigValidator.validate(config)
        assert any("dte" in e.lower() for e in errors)

    def test_validate_structure_excessive_dte(self):
        config = StrategyConfig(
            name="Test",
            underlying="SPY",
            structure=StructureConfig(type=StructureType.SHORT_STRADDLE, dte=400),
        )
        errors = ConfigValidator.validate(config)
        assert any("365" in e for e in errors)

    def test_validate_structure_invalid_delta(self):
        config = StrategyConfig(
            name="Test",
            underlying="SPY",
            structure=StructureConfig(
                type=StructureType.SHORT_STRADDLE, dte=30, delta=1.5
            ),
        )
        errors = ConfigValidator.validate(config)
        assert any("delta" in e.lower() for e in errors)

    def test_validate_iron_condor_requires_width(self):
        config = StrategyConfig(
            name="Test",
            underlying="SPY",
            structure=StructureConfig(
                type=StructureType.IRON_CONDOR, dte=30, width=None
            ),
        )
        errors = ConfigValidator.validate(config)
        assert any("width" in e.lower() for e in errors)

    def test_validate_entry_condition_unknown_type(self):
        config = StrategyConfig(
            name="Test",
            underlying="SPY",
            entry_conditions=[
                EntryConditionConfig(type="unknown_condition", params={})
            ],
        )
        errors = ConfigValidator.validate(config)
        assert any("unknown" in e.lower() for e in errors)

    def test_validate_entry_condition_iv_rank_missing_threshold(self):
        config = StrategyConfig(
            name="Test",
            underlying="SPY",
            entry_conditions=[EntryConditionConfig(type="iv_rank_above", params={})],
        )
        errors = ConfigValidator.validate(config)
        assert any("threshold" in e.lower() for e in errors)

    def test_validate_entry_condition_iv_rank_invalid_threshold(self):
        config = StrategyConfig(
            name="Test",
            underlying="SPY",
            entry_conditions=[
                EntryConditionConfig(type="iv_rank_above", params={"threshold": 150})
            ],
        )
        errors = ConfigValidator.validate(config)
        assert any("0-100" in e for e in errors)

    def test_validate_exit_condition_invalid_profit_target(self):
        config = StrategyConfig(
            name="Test",
            underlying="SPY",
            exit_conditions=ExitConditionConfig(profit_target=-0.5),
        )
        errors = ConfigValidator.validate(config)
        assert any("profit" in e.lower() for e in errors)

    def test_validate_exit_condition_invalid_trailing_stop(self):
        config = StrategyConfig(
            name="Test",
            underlying="SPY",
            exit_conditions=ExitConditionConfig(trailing_stop=1.5),
        )
        errors = ConfigValidator.validate(config)
        assert any("trailing" in e.lower() for e in errors)

    def test_validate_position_size_negative_value(self):
        config = StrategyConfig(
            name="Test",
            underlying="SPY",
            position_size=PositionSizeConfig(
                method=PositionSizeMethod.FIXED_CONTRACTS, value=-1
            ),
        )
        errors = ConfigValidator.validate(config)
        assert any("position" in e.lower() for e in errors)

    def test_validate_risk_limits_invalid_max_positions(self):
        config = StrategyConfig(
            name="Test", underlying="SPY", risk_limits=RiskLimitsConfig(max_positions=0)
        )
        errors = ConfigValidator.validate(config)
        assert any("max positions" in e.lower() for e in errors)

    def test_validate_backtest_invalid_dates(self):
        config = StrategyConfig(
            name="Test",
            underlying="SPY",
            backtest=BacktestConfig(
                start_date="2024-12-31", end_date="2024-01-01", initial_capital=100000
            ),
        )
        errors = ConfigValidator.validate(config)
        assert any("before" in e.lower() for e in errors)

    def test_validate_backtest_negative_capital(self):
        config = StrategyConfig(
            name="Test",
            underlying="SPY",
            backtest=BacktestConfig(
                start_date="2024-01-01", end_date="2024-12-31", initial_capital=-100000
            ),
        )
        errors = ConfigValidator.validate(config)
        assert any("capital" in e.lower() for e in errors)

    def test_validate_template_and_structure_conflict(self):
        config = StrategyConfig(
            name="Test",
            underlying="SPY",
            template="high_iv_straddle",
            structure=StructureConfig(type=StructureType.SHORT_STRADDLE, dte=30),
        )
        errors = ConfigValidator.validate(config)
        assert any("template" in e.lower() and "structure" in e.lower() for e in errors)

    def test_validate_config_function_raises(self):
        config = StrategyConfig(name="", underlying="SPY")
        with pytest.raises(ConfigValidationError):
            validate_config(config)


class TestConfigLoader:
    def test_load_yaml_config(self, tmp_path):
        config_data = {
            "name": "Test Strategy",
            "underlying": "SPY",
            "structure": {"type": "short_straddle", "dte": 30},
        }
        config_file = tmp_path / "strategy.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = load_config(config_file)
        assert config.name == "Test Strategy"
        assert config.underlying == "SPY"
        assert config.structure.type == StructureType.SHORT_STRADDLE

    def test_load_json_config(self, tmp_path):
        config_data = {
            "name": "Test Strategy",
            "underlying": "QQQ",
            "structure": {"type": "iron_condor", "dte": 45, "width": 5.0},
        }
        config_file = tmp_path / "strategy.json"
        with open(config_file, "w") as f:
            json.dump(config_data, f)

        config = load_config(config_file)
        assert config.name == "Test Strategy"
        assert config.underlying == "QQQ"
        assert config.structure.type == StructureType.IRON_CONDOR

    def test_load_config_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")

    def test_load_config_invalid_extension(self, tmp_path):
        config_file = tmp_path / "strategy.txt"
        config_file.write_text("invalid")

        with pytest.raises(ValueError, match="Unsupported"):
            load_config(config_file)

    def test_load_config_with_entry_conditions(self, tmp_path):
        config_data = {
            "name": "Test",
            "underlying": "SPY",
            "entry_conditions": [
                {"type": "iv_rank_above", "threshold": 70},
                {"type": "no_open_positions"},
            ],
        }
        config_file = tmp_path / "strategy.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = load_config(config_file)
        assert len(config.entry_conditions) == 2
        assert config.entry_conditions[0].type == "iv_rank_above"
        assert config.entry_conditions[0].params["threshold"] == 70

    def test_load_config_with_exit_conditions(self, tmp_path):
        config_data = {
            "name": "Test",
            "underlying": "SPY",
            "exit_conditions": {"profit_target": 0.50, "stop_loss": 2.0, "min_dte": 7},
        }
        config_file = tmp_path / "strategy.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = load_config(config_file)
        assert config.exit_conditions.profit_target == 0.50
        assert config.exit_conditions.stop_loss == 2.0
        assert config.exit_conditions.min_dte == 7

    def test_load_config_with_position_size(self, tmp_path):
        config_data = {
            "name": "Test",
            "underlying": "SPY",
            "position_size": {"method": "risk_percent", "value": 0.02},
        }
        config_file = tmp_path / "strategy.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = load_config(config_file)
        assert config.position_size.method == PositionSizeMethod.RISK_PERCENT
        assert config.position_size.value == 0.02

    def test_load_config_with_backtest(self, tmp_path):
        config_data = {
            "name": "Test",
            "underlying": "SPY",
            "backtest": {
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
                "initial_capital": 100000,
                "commission_per_contract": 0.65,
            },
        }
        config_file = tmp_path / "strategy.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = load_config(config_file)
        assert config.backtest.initial_capital == 100000
        assert config.backtest.commission_per_contract == 0.65

    def test_load_config_string_yaml(self):
        yaml_content = """
name: "YAML Strategy"
underlying: "SPY"
structure:
  type: short_straddle
  dte: 30
"""
        config = load_config_string(yaml_content, format="yaml")
        assert config.name == "YAML Strategy"
        assert config.structure.type == StructureType.SHORT_STRADDLE

    def test_load_config_string_json(self):
        json_content = '{"name": "JSON Strategy", "underlying": "SPY"}'
        config = load_config_string(json_content, format="json")
        assert config.name == "JSON Strategy"

    def test_load_config_validation_error(self, tmp_path):
        config_data = {"name": "", "underlying": "SPY"}
        config_file = tmp_path / "invalid.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        with pytest.raises(ConfigValidationError):
            load_config(config_file)


class TestEnvironmentManager:
    def setup_method(self):
        EnvironmentManager.reset()
        if "BACKTESTER_ENV" in os.environ:
            del os.environ["BACKTESTER_ENV"]

    def teardown_method(self):
        EnvironmentManager.reset()
        if "BACKTESTER_ENV" in os.environ:
            del os.environ["BACKTESTER_ENV"]

    def test_default_environment(self):
        env = get_environment()
        assert env == Environment.DEVELOPMENT

    def test_set_environment(self):
        set_environment(Environment.PRODUCTION)
        assert get_environment() == Environment.PRODUCTION

    def test_environment_from_env_var(self):
        EnvironmentManager.reset()
        os.environ["BACKTESTER_ENV"] = "staging"
        assert get_environment() == Environment.STAGING

    def test_environment_invalid_env_var(self):
        EnvironmentManager.reset()
        os.environ["BACKTESTER_ENV"] = "invalid_env"
        assert get_environment() == Environment.DEVELOPMENT

    def test_get_settings_development(self):
        set_environment(Environment.DEVELOPMENT)
        settings = get_settings()
        assert settings.name == Environment.DEVELOPMENT
        assert settings.log_level == "DEBUG"

    def test_get_settings_production(self):
        set_environment(Environment.PRODUCTION)
        settings = get_settings()
        assert settings.name == Environment.PRODUCTION
        assert settings.log_level == "WARNING"
        assert settings.parallel_execution is True

    def test_get_settings_test(self):
        set_environment(Environment.TEST)
        settings = get_settings()
        assert settings.name == Environment.TEST
        assert settings.data_source_type == "mock"
        assert settings.generate_reports is False

    def test_is_development(self):
        set_environment(Environment.DEVELOPMENT)
        assert EnvironmentManager.is_development() is True
        assert EnvironmentManager.is_production() is False

    def test_is_production(self):
        set_environment(Environment.PRODUCTION)
        assert EnvironmentManager.is_production() is True
        assert EnvironmentManager.is_development() is False

    def test_is_test(self):
        set_environment(Environment.TEST)
        assert EnvironmentManager.is_test() is True

    def test_environment_settings_dataclass(self):
        settings = EnvironmentSettings(
            name=Environment.DEVELOPMENT,
            log_level="INFO",
            output_directory="./output/test",
        )
        assert settings.name == Environment.DEVELOPMENT
        assert settings.commission_per_contract == 0.65
        assert settings.use_volume_impact is True


class TestCLICommands:
    @pytest.fixture
    def cli_runner(self):
        from click.testing import CliRunner

        return CliRunner()

    @pytest.fixture
    def sample_config(self, tmp_path):
        config_data = {
            "name": "CLI Test Strategy",
            "underlying": "SPY",
            "structure": {"type": "short_straddle", "dte": 30},
            "entry_conditions": [{"type": "iv_rank_above", "threshold": 70}],
            "exit_conditions": {"profit_target": 0.50, "stop_loss": 2.0},
            "backtest": {
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
                "initial_capital": 100000,
            },
        }
        config_file = tmp_path / "test_strategy.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)
        return config_file

    def test_cli_help(self, cli_runner):
        from backtester.cli.cli import cli

        result = cli_runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Options Backtester" in result.output

    def test_cli_version(self, cli_runner):
        from backtester.cli.cli import cli

        result = cli_runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "1.0.0" in result.output

    def test_validate_command_success(self, cli_runner, sample_config):
        from backtester.cli.cli import cli

        result = cli_runner.invoke(cli, ["validate", "--config", str(sample_config)])
        assert result.exit_code == 0
        assert "valid" in result.output.lower()

    def test_validate_command_invalid_config(self, cli_runner, tmp_path):
        from backtester.cli.cli import cli

        invalid_config = tmp_path / "invalid.yaml"
        with open(invalid_config, "w") as f:
            yaml.dump({"name": "", "underlying": "SPY"}, f)

        result = cli_runner.invoke(cli, ["validate", "--config", str(invalid_config)])
        assert result.exit_code == 1
        assert "error" in result.output.lower()

    def test_validate_command_file_not_found(self, cli_runner):
        from backtester.cli.cli import cli

        result = cli_runner.invoke(
            cli, ["validate", "--config", "/nonexistent/file.yaml"]
        )
        assert result.exit_code != 0

    def test_list_templates_command(self, cli_runner):
        from backtester.cli.cli import cli

        result = cli_runner.invoke(cli, ["list", "templates"])
        assert result.exit_code == 0

    def test_list_structures_command(self, cli_runner):
        from backtester.cli.cli import cli

        result = cli_runner.invoke(cli, ["list", "structures"])
        assert result.exit_code == 0
        assert "straddle" in result.output.lower()

    def test_list_conditions_command(self, cli_runner):
        from backtester.cli.cli import cli

        result = cli_runner.invoke(cli, ["list", "conditions"])
        assert result.exit_code == 0
        assert "iv_rank" in result.output.lower()
        assert "profit_target" in result.output.lower()

    def test_env_command(self, cli_runner):
        from backtester.cli.cli import cli

        result = cli_runner.invoke(cli, ["env"])
        assert result.exit_code == 0
        assert "environment" in result.output.lower()

    def test_init_command(self, cli_runner, tmp_path):
        from backtester.cli.cli import cli

        output_file = tmp_path / "new_strategy.yaml"
        result = cli_runner.invoke(
            cli, ["init", "--name", "New Strategy", "--output", str(output_file)]
        )
        assert result.exit_code == 0
        assert output_file.exists()

        with open(output_file) as f:
            content = yaml.safe_load(f)
        assert content["name"] == "New Strategy"

    def test_run_command_dry_run(self, cli_runner, sample_config):
        from backtester.cli.cli import cli

        result = cli_runner.invoke(
            cli, ["run", "--config", str(sample_config), "--dry-run"]
        )
        assert result.exit_code == 0
        assert "dry run" in result.output.lower() or "valid" in result.output.lower()

    def test_report_command_json_output(self, cli_runner, tmp_path):
        from backtester.cli.cli import cli

        results_file = tmp_path / "results.json"
        with open(results_file, "w") as f:
            json.dump(
                {"strategy_name": "Test", "total_return": 0.15, "sharpe_ratio": 1.5}, f
            )

        output_file = tmp_path / "report.json"
        result = cli_runner.invoke(
            cli,
            [
                "report",
                "--results",
                str(results_file),
                "--output",
                str(output_file),
                "--format",
                "json",
            ],
        )
        assert result.exit_code == 0
        assert output_file.exists()

    def test_cli_with_env_option(self, cli_runner):
        from backtester.cli.cli import cli

        result = cli_runner.invoke(cli, ["--env", "production", "env"])
        assert result.exit_code == 0


class TestCompleteConfigWorkflow:
    def test_full_config_lifecycle(self, tmp_path):
        yaml_content = """
name: "Complete Strategy"
version: "1.0"
description: "A complete strategy configuration"
underlying: "SPY"

structure:
  type: short_straddle
  dte: 30
  quantity: 1

entry_conditions:
  - type: iv_rank_above
    threshold: 70
  - type: no_open_positions

exit_conditions:
  profit_target: 0.50
  stop_loss: 2.0
  min_dte: 7

position_size:
  method: risk_percent
  value: 0.02

risk_limits:
  max_positions: 5
  max_capital_utilization: 0.80

backtest:
  start_date: "2024-01-01"
  end_date: "2024-12-31"
  initial_capital: 100000
  commission_per_contract: 0.65
  slippage_pct: 0.01

data_source:
  type: dolt
  database: options_data
"""
        config_file = tmp_path / "complete_strategy.yaml"
        config_file.write_text(yaml_content)

        config = load_config(config_file)

        assert config.name == "Complete Strategy"
        assert config.underlying == "SPY"
        assert config.structure.type == StructureType.SHORT_STRADDLE
        assert config.structure.dte == 30
        assert len(config.entry_conditions) == 2
        assert config.exit_conditions.profit_target == 0.50
        assert config.position_size.method == PositionSizeMethod.RISK_PERCENT
        assert config.risk_limits.max_positions == 5
        assert config.backtest.initial_capital == 100000
        assert config.data_source.type == DataSourceType.DOLT

        errors = ConfigValidator.validate(config)
        assert len(errors) == 0
