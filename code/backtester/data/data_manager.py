"""
Multi-Source Data Manager for Options Backtesting

This module provides a flexible data architecture that supports multiple data sources
with automatic fallback. It enables the backtester to seamlessly switch between or
combine data from different providers (Dolt database, CSV files, APIs, etc.).

Key Components:
    - BaseDataAdapter: Abstract base class defining the data source interface
    - CSVDataAdapter: Adapter for reading data from CSV files
    - DataSourceRegistry: Manages multiple data sources with priority-based fallback

Design Philosophy:
    The adapter pattern allows adding new data sources without changing strategy code.
    Each adapter implements a consistent interface, enabling drop-in replacements
    and automatic failover when a source is unavailable.

Usage:
    from backtester.data.data_manager import (
        DataSourceRegistry, CSVDataAdapter, DoltDataAdapter
    )

    registry = DataSourceRegistry()
    registry.register_source('csv', CSVDataAdapter('/path/to/data'), priority=1)
    registry.register_source('dolt', DoltDataAdapter('/path/to/dolt'), priority=2)

    chain = registry.get_option_chain('SPY', date)

References:
    - Adapter Pattern: https://refactoring.guru/design-patterns/adapter
"""

import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataSourceError(Exception):
    """Base exception for data source errors."""

    pass


class DataSourceConnectionError(DataSourceError):
    """Exception raised when connection to data source fails."""

    pass


class DataNotFoundError(DataSourceError):
    """Exception raised when requested data is not found."""

    pass


class AllSourcesFailedError(DataSourceError):
    """Exception raised when all data sources fail to provide data."""

    pass


class BaseDataAdapter(ABC):
    """
    Abstract base class for all data adapters.

    All data adapters must implement this interface to ensure consistent
    behavior across different data sources. The interface covers the
    essential operations needed for options backtesting.

    Subclasses must implement:
        - connect(): Establish connection to data source
        - disconnect(): Close connection
        - is_connected: Property indicating connection status
        - get_option_chain(): Retrieve option chain data
        - get_underlying_price(): Get underlying asset price
        - get_volatility_data(): Get historical/implied volatility

    Example:
        class MyCustomAdapter(BaseDataAdapter):
            def connect(self) -> bool:
                # Connect to custom source
                return True
    """

    @abstractmethod
    def connect(self) -> bool:
        """
        Establish connection to the data source.

        Returns:
            True if connection successful, False otherwise.

        Raises:
            DataSourceConnectionError: If connection fails critically.
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Close connection to the data source."""
        pass

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if adapter is connected to data source."""
        pass

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Return human-readable name of this data source."""
        pass

    @abstractmethod
    def get_option_chain(
        self,
        symbol: str,
        date: datetime,
        expiration: Optional[datetime] = None,
        min_dte: Optional[int] = None,
        max_dte: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Retrieve option chain data for a given symbol and date.

        Args:
            symbol: Underlying ticker symbol (e.g., 'SPY')
            date: Trading date for the data
            expiration: Specific expiration date (optional)
            min_dte: Minimum days to expiration filter
            max_dte: Maximum days to expiration filter

        Returns:
            DataFrame with columns:
                - strike: Strike price
                - expiration: Expiration date
                - option_type/call_put: 'call' or 'put'
                - bid: Bid price
                - ask: Ask price
                - implied_volatility/vol: Implied volatility
                - delta, gamma, theta, vega, rho: Greeks (if available)

        Raises:
            DataNotFoundError: If no data found for the request.
            DataSourceError: If query fails.
        """
        pass

    @abstractmethod
    def get_underlying_price(
        self,
        symbol: str,
        date: datetime,
    ) -> float:
        """
        Get closing price of underlying asset.

        Args:
            symbol: Ticker symbol
            date: Trading date

        Returns:
            Closing price as float.

        Raises:
            DataNotFoundError: If price not found.
        """
        pass

    def get_volatility_data(
        self,
        symbol: str,
        date: datetime,
    ) -> Dict[str, float]:
        """
        Get volatility metrics for a symbol.

        Default implementation returns empty dict. Override in subclasses
        that have volatility data available.

        Args:
            symbol: Ticker symbol
            date: Trading date

        Returns:
            Dict with keys like 'iv_current', 'hv_current', 'iv_rank', etc.
        """
        return {}

    def get_date_range(self, symbol: str) -> Tuple[datetime, datetime]:
        """
        Get available date range for a symbol.

        Default implementation raises NotImplementedError.
        Override in subclasses that support this.

        Args:
            symbol: Ticker symbol

        Returns:
            Tuple of (start_date, end_date)
        """
        raise NotImplementedError("Date range query not supported by this adapter")


class CSVDataAdapter(BaseDataAdapter):
    """
    Data adapter for reading options data from CSV files.

    Expects CSV files organized by symbol and date:
        data_directory/
            SPY/
                2024-01-02.csv
                2024-01-03.csv
                ...
            QQQ/
                ...

    Or flat structure with naming convention:
        data_directory/
            SPY_20240102.csv
            SPY_20240103.csv
            ...

    CSV files should contain columns matching the expected schema:
        strike, expiration, call_put, bid, ask, vol, delta, gamma, theta, vega, rho

    Attributes:
        data_directory: Path to directory containing CSV files
        file_pattern: Pattern for finding files ('nested' or 'flat')
        _cache: Optional cache for loaded DataFrames
    """

    REQUIRED_COLUMNS = {"strike", "expiration", "bid", "ask"}
    OPTION_TYPE_COLUMNS = {"call_put", "option_type", "type"}

    def __init__(
        self,
        data_directory: str,
        file_pattern: str = "auto",
        use_cache: bool = True,
        date_format: str = "%Y-%m-%d",
    ):
        """
        Initialize CSV data adapter.

        Args:
            data_directory: Path to directory containing CSV files
            file_pattern: 'nested' (symbol/date.csv), 'flat' (symbol_date.csv),
                         or 'auto' to detect
            use_cache: Whether to cache loaded data in memory
            date_format: Date format string for parsing dates in filenames
        """
        self.data_directory = Path(data_directory).resolve()
        self.file_pattern = file_pattern
        self.use_cache = use_cache
        self.date_format = date_format
        self._is_connected = False
        self._cache: Dict[str, pd.DataFrame] = {}
        self._detected_pattern: Optional[str] = None

    def connect(self) -> bool:
        """Verify data directory exists and contains data files."""
        if not self.data_directory.exists():
            logger.error(f"CSV data directory not found: {self.data_directory}")
            return False

        if self.file_pattern == "auto":
            self._detected_pattern = self._detect_file_pattern()
            if self._detected_pattern is None:
                logger.error(f"No CSV files found in: {self.data_directory}")
                return False
        else:
            self._detected_pattern = self.file_pattern

        self._is_connected = True
        logger.info(
            f"CSVDataAdapter connected: {self.data_directory} (pattern: {self._detected_pattern})"
        )
        return True

    def disconnect(self) -> None:
        """Clear cache and disconnect."""
        self._cache.clear()
        self._is_connected = False
        logger.debug("CSVDataAdapter disconnected")

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def source_name(self) -> str:
        return f"CSV:{self.data_directory.name}"

    def _detect_file_pattern(self) -> Optional[str]:
        """Auto-detect the file organization pattern."""
        subdirs = [d for d in self.data_directory.iterdir() if d.is_dir()]
        if subdirs:
            for subdir in subdirs:
                csvs = list(subdir.glob("*.csv"))
                if csvs:
                    return "nested"

        flat_csvs = list(self.data_directory.glob("*_*.csv"))
        if flat_csvs:
            return "flat"

        any_csvs = list(self.data_directory.glob("*.csv"))
        if any_csvs:
            return "flat"

        return None

    def _get_file_path(self, symbol: str, date: datetime) -> Path:
        """Get file path for a symbol and date."""
        if self._detected_pattern == "nested":
            date_str = date.strftime(self.date_format)
            return self.data_directory / symbol.upper() / f"{date_str}.csv"
        else:
            date_str = date.strftime("%Y%m%d")
            return self.data_directory / f"{symbol.upper()}_{date_str}.csv"

    def _load_csv(self, file_path: Path) -> pd.DataFrame:
        """Load and normalize a CSV file."""
        cache_key = str(file_path)

        if self.use_cache and cache_key in self._cache:
            return self._cache[cache_key].copy()

        if not file_path.exists():
            raise DataNotFoundError(f"CSV file not found: {file_path}")

        df = pd.read_csv(file_path)
        df.columns = df.columns.str.lower().str.strip()

        for col in self.OPTION_TYPE_COLUMNS:
            if col in df.columns:
                df["call_put"] = df[col].str.lower()
                break

        if "expiration" in df.columns:
            df["expiration"] = pd.to_datetime(df["expiration"])

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])

        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            logger.warning(f"CSV missing columns: {missing}")

        if self.use_cache:
            self._cache[cache_key] = df.copy()

        return df

    def get_option_chain(
        self,
        symbol: str,
        date: datetime,
        expiration: Optional[datetime] = None,
        min_dte: Optional[int] = None,
        max_dte: Optional[int] = None,
    ) -> pd.DataFrame:
        """Load option chain from CSV file."""
        if not self._is_connected:
            raise DataSourceError("CSVDataAdapter not connected")

        file_path = self._get_file_path(symbol, date)
        df = self._load_csv(file_path)

        if expiration is not None and "expiration" in df.columns:
            df = df[df["expiration"].dt.date == expiration.date()]

        if "expiration" in df.columns and (min_dte is not None or max_dte is not None):
            df["dte"] = (df["expiration"] - pd.Timestamp(date)).dt.days
            if min_dte is not None:
                df = df[df["dte"] >= min_dte]
            if max_dte is not None:
                df = df[df["dte"] <= max_dte]

        if len(df) == 0:
            raise DataNotFoundError(
                f"No option data found for {symbol} on {date.strftime('%Y-%m-%d')}"
            )

        result: pd.DataFrame = pd.DataFrame(df)
        return result

    def get_underlying_price(self, symbol: str, date: datetime) -> float:
        """
        Get underlying price from option chain mid-strike or separate file.
        """
        try:
            price_file = self.data_directory / f"{symbol.upper()}_prices.csv"
            if price_file.exists():
                prices_df = pd.read_csv(price_file)
                prices_df.columns = prices_df.columns.str.lower()
                prices_df["date"] = pd.to_datetime(prices_df["date"])
                row = prices_df[prices_df["date"].dt.date == date.date()]
                if len(row) > 0:
                    return float(row.iloc[0].get("close", row.iloc[0].get("price", 0)))
        except Exception:
            pass

        try:
            chain = self.get_option_chain(symbol, date)
            if "underlying_price" in chain.columns:
                return float(chain["underlying_price"].iloc[0])
            if "spot" in chain.columns:
                return float(chain["spot"].iloc[0])

            atm_strike = chain["strike"].median()
            return float(atm_strike)
        except Exception as e:
            raise DataNotFoundError(
                f"Cannot determine underlying price for {symbol} on {date}: {e}"
            )


class DoltDataAdapter(BaseDataAdapter):
    """
    Wrapper adapter for existing DoltAdapter to conform to BaseDataAdapter interface.

    This class wraps the existing DoltAdapter implementation, providing the
    standardized interface required by DataSourceRegistry.
    """

    def __init__(self, db_path: str):
        """
        Initialize wrapper around DoltAdapter.

        Args:
            db_path: Path to Dolt database directory
        """
        self.db_path = db_path
        self._adapter: Any = None
        self._is_connected = False

    def connect(self) -> bool:
        """Connect to Dolt database."""
        try:
            from backtester.data.dolt_adapter import DoltAdapter

            self._adapter = DoltAdapter(self.db_path)
            self._adapter.connect()
            self._is_connected = True
            logger.info(f"DoltDataAdapter connected: {self.db_path}")
            return True
        except Exception as e:
            logger.error(f"DoltDataAdapter connection failed: {e}")
            self._is_connected = False
            return False

    def disconnect(self) -> None:
        """Disconnect from Dolt database."""
        if self._adapter is not None:
            try:
                self._adapter.close()
            except Exception:
                pass
        self._adapter = None
        self._is_connected = False

    @property
    def is_connected(self) -> bool:
        return self._is_connected and self._adapter is not None

    @property
    def source_name(self) -> str:
        return f"Dolt:{Path(self.db_path).name}"

    def get_option_chain(
        self,
        symbol: str,
        date: datetime,
        expiration: Optional[datetime] = None,
        min_dte: Optional[int] = None,
        max_dte: Optional[int] = None,
    ) -> pd.DataFrame:
        """Get option chain from Dolt database."""
        if not self.is_connected or self._adapter is None:
            raise DataSourceError("DoltDataAdapter not connected")

        df = self._adapter.get_option_chain(
            underlying=symbol,
            date=date,
            min_dte=min_dte if min_dte is not None else 0,
            max_dte=max_dte if max_dte is not None else 60,
        )

        if df is None or len(df) == 0:
            raise DataNotFoundError(
                f"No option data found for {symbol} on {date.strftime('%Y-%m-%d')}"
            )

        result: pd.DataFrame = df
        return result

    def get_underlying_price(self, symbol: str, date: datetime) -> float:
        """
        Get underlying price from Dolt database.

        Since the Dolt options database doesn't have direct price data,
        we estimate the underlying price from ATM option strikes.
        """
        if not self.is_connected or self._adapter is None:
            raise DataSourceError("DoltDataAdapter not connected")

        try:
            chain = self._adapter.get_option_chain(
                underlying=symbol,
                date=date,
                min_dte=20,
                max_dte=40,
            )

            if chain is not None and len(chain) > 0:
                strikes = chain["strike"].unique()
                atm_strike = float(np.median(strikes))
                return atm_strike
        except Exception:
            pass

        raise DataNotFoundError(
            f"Cannot determine underlying price for {symbol} on {date.strftime('%Y-%m-%d')}"
        )

    def get_volatility_data(self, symbol: str, date: datetime) -> Dict[str, float]:
        """Get volatility data from Dolt database."""
        if not self.is_connected or self._adapter is None:
            return {}

        try:
            from datetime import timedelta

            vol_data = self._adapter.get_underlying_prices(
                symbol=symbol,
                start=date,
                end=date + timedelta(days=1),
            )
            if vol_data is not None and len(vol_data) > 0:
                row = vol_data.iloc[0]
                return {
                    "iv_current": float(row.get("iv_current", np.nan)),
                    "hv_current": float(row.get("hv_current", np.nan)),
                    "iv_year_high": float(row.get("iv_year_high", np.nan)),
                    "iv_year_low": float(row.get("iv_year_low", np.nan)),
                }
        except Exception:
            pass
        return {}

        try:
            vol_data = self._adapter.get_volatility_history(symbol, date)
            if vol_data is not None and len(vol_data) > 0:
                row = vol_data.iloc[0]
                return {
                    "iv_current": row.get("iv_current", np.nan),
                    "hv_current": row.get("hv_current", np.nan),
                    "iv_year_high": row.get("iv_year_high", np.nan),
                    "iv_year_low": row.get("iv_year_low", np.nan),
                }
        except Exception:
            pass
        return {}


class DataSourceRegistry:
    """
    Manage multiple data sources with priority-based fallback.

    The registry maintains a collection of data adapters and routes
    data requests to them in priority order. If a higher-priority source
    fails, it automatically falls back to the next source.

    Attributes:
        sources: Dictionary of registered adapters
        _priority_list: Sorted list of (name, priority) tuples

    Example:
        registry = DataSourceRegistry()
        registry.register_source('primary', DoltAdapter(...), priority=10)
        registry.register_source('backup', CSVAdapter(...), priority=5)

        chain = registry.get_option_chain('SPY', date)
    """

    def __init__(self):
        """Initialize empty registry."""
        self.sources: Dict[str, BaseDataAdapter] = {}
        self._priority_list: List[Tuple[str, int]] = []
        self._stats: Dict[str, Dict[str, int]] = {}

    def register_source(
        self,
        name: str,
        adapter: BaseDataAdapter,
        priority: int = 0,
        auto_connect: bool = True,
    ) -> bool:
        """
        Register a data source.

        Args:
            name: Unique identifier for this source
            adapter: BaseDataAdapter instance
            priority: Higher priority sources are tried first
            auto_connect: Attempt to connect immediately

        Returns:
            True if registration (and optional connection) succeeded
        """
        if auto_connect and not adapter.is_connected:
            if not adapter.connect():
                logger.warning(f"Source '{name}' failed to connect, registering anyway")

        self.sources[name] = adapter
        self._priority_list.append((name, priority))
        self._priority_list.sort(key=lambda x: x[1], reverse=True)
        self._stats[name] = {"success": 0, "failure": 0}

        logger.info(f"Registered data source: {name} (priority={priority})")
        return True

    def unregister_source(self, name: str) -> bool:
        """Remove a data source from the registry."""
        if name not in self.sources:
            return False

        adapter = self.sources.pop(name)
        adapter.disconnect()
        self._priority_list = [(n, p) for n, p in self._priority_list if n != name]
        self._stats.pop(name, None)

        logger.info(f"Unregistered data source: {name}")
        return True

    def get_source(self, name: str) -> Optional[BaseDataAdapter]:
        """Get a specific adapter by name."""
        return self.sources.get(name)

    def list_sources(self) -> List[Dict[str, Any]]:
        """List all registered sources with their status."""
        return [
            {
                "name": name,
                "priority": priority,
                "connected": self.sources[name].is_connected,
                "source_name": self.sources[name].source_name,
                "stats": self._stats.get(name, {}),
            }
            for name, priority in self._priority_list
        ]

    def get_option_chain(
        self,
        symbol: str,
        date: datetime,
        expiration: Optional[datetime] = None,
        min_dte: Optional[int] = None,
        max_dte: Optional[int] = None,
        source: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Get option chain from best available source.

        Args:
            symbol: Underlying symbol
            date: Trading date
            expiration: Specific expiration (optional)
            min_dte: Minimum days to expiration
            max_dte: Maximum days to expiration
            source: Specific source to use (bypasses priority)

        Returns:
            DataFrame with option chain data

        Raises:
            AllSourcesFailedError: If all sources fail
        """
        if source is not None:
            if source not in self.sources:
                raise DataSourceError(f"Unknown source: {source}")
            adapter = self.sources[source]
            return adapter.get_option_chain(symbol, date, expiration, min_dte, max_dte)

        errors = []
        for name, _ in self._priority_list:
            adapter = self.sources[name]
            if not adapter.is_connected:
                continue

            try:
                df = adapter.get_option_chain(
                    symbol, date, expiration, min_dte, max_dte
                )
                if df is not None and len(df) > 0:
                    self._stats[name]["success"] += 1
                    logger.debug(f"Option chain for {symbol}/{date} from {name}")
                    return df
            except Exception as e:
                self._stats[name]["failure"] += 1
                errors.append(f"{name}: {e}")
                logger.debug(f"Source {name} failed: {e}")
                continue

        raise AllSourcesFailedError(
            f"All data sources failed for {symbol} on {date}:\n" + "\n".join(errors)
        )

    def get_underlying_price(
        self,
        symbol: str,
        date: datetime,
        source: Optional[str] = None,
    ) -> float:
        """
        Get underlying price from best available source.

        Args:
            symbol: Ticker symbol
            date: Trading date
            source: Specific source to use (optional)

        Returns:
            Closing price

        Raises:
            AllSourcesFailedError: If all sources fail
        """
        if source is not None:
            if source not in self.sources:
                raise DataSourceError(f"Unknown source: {source}")
            return self.sources[source].get_underlying_price(symbol, date)

        errors = []
        for name, _ in self._priority_list:
            adapter = self.sources[name]
            if not adapter.is_connected:
                continue

            try:
                price = adapter.get_underlying_price(symbol, date)
                self._stats[name]["success"] += 1
                return price
            except Exception as e:
                self._stats[name]["failure"] += 1
                errors.append(f"{name}: {e}")
                continue

        raise AllSourcesFailedError(
            f"All data sources failed for {symbol} price on {date}:\n"
            + "\n".join(errors)
        )

    def get_statistics(self) -> Dict[str, Dict[str, int]]:
        """Get usage statistics for all sources."""
        return self._stats.copy()

    def disconnect_all(self) -> None:
        """Disconnect all data sources."""
        for name, adapter in self.sources.items():
            try:
                adapter.disconnect()
            except Exception as e:
                logger.warning(f"Error disconnecting {name}: {e}")
        logger.info("All data sources disconnected")

    def __enter__(self) -> "DataSourceRegistry":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.disconnect_all()


__all__ = [
    "BaseDataAdapter",
    "CSVDataAdapter",
    "DoltDataAdapter",
    "DataSourceRegistry",
    "DataSourceError",
    "DataSourceConnectionError",
    "DataNotFoundError",
    "AllSourcesFailedError",
]
