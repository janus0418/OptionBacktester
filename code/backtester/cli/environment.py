"""
Environment Management for Options Backtester

Manages different environments (development, staging, production)
with environment-specific configurations and settings.
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional
import logging

import yaml

logger = logging.getLogger(__name__)


class Environment(str, Enum):
    """Available environments."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"


@dataclass
class EnvironmentSettings:
    """Settings specific to an environment."""

    name: Environment

    # Data source settings
    data_source_type: str = "dolt"
    database_name: Optional[str] = None
    database_host: Optional[str] = None
    database_port: Optional[int] = None
    data_directory: Optional[str] = None

    # Execution settings
    commission_per_contract: float = 0.65
    slippage_pct: float = 0.01
    use_volume_impact: bool = True

    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None

    # Output settings
    output_directory: str = "./output"
    save_trades: bool = True
    save_equity_curve: bool = True
    generate_reports: bool = True

    # Performance settings
    use_caching: bool = True
    parallel_execution: bool = False

    # Additional settings
    extra: Dict[str, Any] = field(default_factory=dict)


# Default settings for each environment
DEFAULT_SETTINGS: Dict[Environment, Dict[str, Any]] = {
    Environment.DEVELOPMENT: {
        "log_level": "DEBUG",
        "output_directory": "./output/dev",
        "use_caching": False,
        "generate_reports": True,
    },
    Environment.STAGING: {
        "log_level": "INFO",
        "output_directory": "./output/staging",
        "use_caching": True,
        "generate_reports": True,
    },
    Environment.PRODUCTION: {
        "log_level": "WARNING",
        "output_directory": "./output/prod",
        "use_caching": True,
        "generate_reports": True,
        "parallel_execution": True,
    },
    Environment.TEST: {
        "log_level": "DEBUG",
        "output_directory": "./output/test",
        "use_caching": False,
        "generate_reports": False,
        "data_source_type": "mock",
    },
}


class EnvironmentManager:
    """Manages environment configuration and settings."""

    # Environment variable name for current environment
    ENV_VAR = "BACKTESTER_ENV"

    # Config file search paths
    CONFIG_PATHS = [
        Path.cwd() / "config",
        Path.cwd() / ".config",
        Path.home() / ".backtester",
        Path("/etc/backtester"),
    ]

    _current_env: Optional[Environment] = None
    _settings: Optional[EnvironmentSettings] = None

    @classmethod
    def get_environment(cls) -> Environment:
        """
        Get the current environment.

        Priority:
        1. Explicitly set via set_environment()
        2. BACKTESTER_ENV environment variable
        3. Default to DEVELOPMENT
        """
        if cls._current_env is not None:
            return cls._current_env

        env_str = os.environ.get(cls.ENV_VAR, "development").lower()

        try:
            return Environment(env_str)
        except ValueError:
            logger.warning(
                f"Unknown environment '{env_str}', defaulting to development"
            )
            return Environment.DEVELOPMENT

    @classmethod
    def set_environment(cls, env: Environment) -> None:
        """
        Set the current environment.

        Args:
            env: Environment to use
        """
        cls._current_env = env
        cls._settings = None  # Reset cached settings
        logger.info(f"Environment set to: {env.value}")

    @classmethod
    def get_settings(cls) -> EnvironmentSettings:
        """
        Get settings for the current environment.

        Loads from config file if available, otherwise uses defaults.
        """
        if cls._settings is not None:
            return cls._settings

        env = cls.get_environment()

        # Try to load from config file
        config_data = cls._load_config_file(env)

        # Merge with defaults
        defaults = DEFAULT_SETTINGS.get(env, {})
        merged = {**defaults, **(config_data or {})}

        cls._settings = EnvironmentSettings(
            name=env,
            data_source_type=merged.get("data_source_type", "dolt"),
            database_name=merged.get("database_name"),
            database_host=merged.get("database_host"),
            database_port=merged.get("database_port"),
            data_directory=merged.get("data_directory"),
            commission_per_contract=merged.get("commission_per_contract", 0.65),
            slippage_pct=merged.get("slippage_pct", 0.01),
            use_volume_impact=merged.get("use_volume_impact", True),
            log_level=merged.get("log_level", "INFO"),
            log_file=merged.get("log_file"),
            output_directory=merged.get("output_directory", "./output"),
            save_trades=merged.get("save_trades", True),
            save_equity_curve=merged.get("save_equity_curve", True),
            generate_reports=merged.get("generate_reports", True),
            use_caching=merged.get("use_caching", True),
            parallel_execution=merged.get("parallel_execution", False),
            extra=merged.get("extra", {}),
        )

        return cls._settings

    @classmethod
    def _load_config_file(cls, env: Environment) -> Optional[Dict[str, Any]]:
        """Load environment config from file."""
        config_names = [
            f"{env.value}.yaml",
            f"{env.value}.yml",
            f"{env.value}.json",
            "config.yaml",
            "config.yml",
        ]

        for base_path in cls.CONFIG_PATHS:
            for config_name in config_names:
                config_path = base_path / config_name
                if config_path.exists():
                    try:
                        with open(config_path) as f:
                            data = yaml.safe_load(f)
                            logger.debug(f"Loaded config from {config_path}")

                            # Handle nested environment config
                            if env.value in data:
                                return data[env.value]
                            return data
                    except Exception as e:
                        logger.warning(f"Failed to load {config_path}: {e}")

        return None

    @classmethod
    def reset(cls) -> None:
        """Reset environment manager state."""
        cls._current_env = None
        cls._settings = None

    @classmethod
    def is_development(cls) -> bool:
        """Check if running in development mode."""
        return cls.get_environment() == Environment.DEVELOPMENT

    @classmethod
    def is_production(cls) -> bool:
        """Check if running in production mode."""
        return cls.get_environment() == Environment.PRODUCTION

    @classmethod
    def is_test(cls) -> bool:
        """Check if running in test mode."""
        return cls.get_environment() == Environment.TEST


def get_environment() -> Environment:
    """Get current environment."""
    return EnvironmentManager.get_environment()


def get_settings() -> EnvironmentSettings:
    """Get current environment settings."""
    return EnvironmentManager.get_settings()


def set_environment(env: Environment) -> None:
    """Set current environment."""
    EnvironmentManager.set_environment(env)


def configure_logging() -> None:
    """Configure logging based on environment settings."""
    settings = get_settings()

    # Set log level
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Console handler
    if not root_logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # File handler (if configured)
    if settings.log_file:
        file_handler = logging.FileHandler(settings.log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    logger.info(f"Logging configured for {settings.name.value} environment")
