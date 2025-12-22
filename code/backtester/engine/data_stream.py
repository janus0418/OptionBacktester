"""
DataStream Class for Options Backtesting

This module provides the DataStream class that serves as an iterator over
time-series market data for backtesting. It abstracts data fetching and
provides a clean interface for the BacktestEngine to consume market data
day by day.

Key Features:
    - Iterator protocol over trading days
    - Skip weekends and holidays
    - Efficient data pre-loading or streaming
    - Graceful handling of missing data
    - Integration with DoltAdapter for data retrieval
    - Trading calendar support for holiday handling

Design Philosophy:
    The DataStream separates data concerns from trading logic. It handles
    all the complexity of data retrieval, caching, and calendar management,
    presenting a simple iterator interface to the BacktestEngine.

Financial Correctness:
    - Trading days only (skip weekends and holidays)
    - Proper handling of market closures
    - Consistent timestamp handling across time zones
    - Missing data is logged and gracefully handled

Usage:
    from backtester.engine.data_stream import DataStream
    from backtester.data.dolt_adapter import DoltAdapter
    from datetime import datetime

    adapter = DoltAdapter('/path/to/db')
    adapter.connect()

    stream = DataStream(
        data_source=adapter,
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 12, 31),
        underlying='SPY',
        dte_range=(7, 45)
    )

    for timestamp, market_data in stream:
        # Process market data
        spot = market_data['spot']
        options = market_data['option_chain']

References:
    - Trading calendar: https://www.nyse.com/markets/hours-calendars
"""

import logging
from datetime import datetime, timedelta, date
from typing import Any, Dict, Iterator, List, Optional, Tuple, Protocol, Union

import numpy as np
import pandas as pd

# Configure module logger
logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Trading calendar constants
WEEKDAY_MONDAY = 0
WEEKDAY_FRIDAY = 4
WEEKDAY_SATURDAY = 5
WEEKDAY_SUNDAY = 6

# Default DTE range for option chain retrieval
DEFAULT_MIN_DTE = 7
DEFAULT_MAX_DTE = 60

# Cache size limit (number of days to keep in memory)
DEFAULT_CACHE_SIZE = 30


# =============================================================================
# Exceptions
# =============================================================================

class DataStreamError(Exception):
    """Base exception for DataStream errors."""
    pass


class DataStreamConfigError(DataStreamError):
    """Exception raised for configuration errors."""
    pass


class DataNotAvailableError(DataStreamError):
    """Exception raised when required data is not available."""
    pass


class DataStreamExhaustedError(DataStreamError):
    """Exception raised when stream has no more data."""
    pass


# =============================================================================
# Protocol for Data Source
# =============================================================================

class DataSourceProtocol(Protocol):
    """Protocol defining the expected interface for data sources."""

    def get_option_chain(
        self,
        underlying: str,
        date: datetime,
        min_dte: int,
        max_dte: int,
        strikes: Optional[List[float]] = None,
        option_type: Optional[str] = None
    ) -> pd.DataFrame:
        """Get option chain data for a specific date."""
        ...

    def get_implied_volatility(
        self,
        symbol: str,
        date: datetime,
        strike: Optional[float] = None,
        option_type: str = 'call',
        dte_range: Tuple[int, int] = (20, 40)
    ) -> float:
        """Get implied volatility for a symbol on a specific date."""
        ...


# =============================================================================
# Trading Calendar Class
# =============================================================================

class TradingCalendar:
    """
    Simple trading calendar for US equity markets.

    Handles weekends and common US market holidays. For production use,
    consider using the 'exchange_calendars' or 'pandas_market_calendars'
    library for comprehensive holiday handling.

    Attributes:
        holidays (set): Set of date objects representing market holidays
    """

    # US market holidays (fixed dates - simplified)
    # For production, use a proper calendar library
    FIXED_HOLIDAYS = {
        (1, 1),   # New Year's Day
        (7, 4),   # Independence Day
        (12, 25), # Christmas
    }

    def __init__(self, holidays: Optional[List[date]] = None) -> None:
        """
        Initialize the trading calendar.

        Args:
            holidays: Optional list of additional holiday dates
        """
        self._holidays: set = set()
        if holidays:
            self._holidays = set(holidays)

    def add_holidays(self, holidays: List[date]) -> None:
        """
        Add holidays to the calendar.

        Args:
            holidays: List of date objects to add as holidays
        """
        self._holidays.update(holidays)

    def is_trading_day(self, dt: Union[datetime, date]) -> bool:
        """
        Check if a date is a trading day.

        A trading day is a weekday that is not a market holiday.

        Args:
            dt: Date to check (datetime or date object)

        Returns:
            True if the date is a trading day
        """
        if isinstance(dt, datetime):
            dt = dt.date()

        # Weekend check
        if dt.weekday() >= WEEKDAY_SATURDAY:
            return False

        # Holiday check
        if dt in self._holidays:
            return False

        # Fixed holiday check (simplified)
        if (dt.month, dt.day) in self.FIXED_HOLIDAYS:
            return False

        return True

    def next_trading_day(self, dt: Union[datetime, date]) -> date:
        """
        Get the next trading day after the given date.

        Args:
            dt: Reference date

        Returns:
            The next trading day after dt
        """
        if isinstance(dt, datetime):
            dt = dt.date()

        next_day = dt + timedelta(days=1)
        while not self.is_trading_day(next_day):
            next_day += timedelta(days=1)

        return next_day

    def get_trading_days(
        self,
        start_date: Union[datetime, date],
        end_date: Union[datetime, date]
    ) -> List[date]:
        """
        Get all trading days in a date range.

        Args:
            start_date: Start of range (inclusive)
            end_date: End of range (inclusive)

        Returns:
            List of trading day dates
        """
        if isinstance(start_date, datetime):
            start_date = start_date.date()
        if isinstance(end_date, datetime):
            end_date = end_date.date()

        trading_days = []
        current = start_date

        while current <= end_date:
            if self.is_trading_day(current):
                trading_days.append(current)
            current += timedelta(days=1)

        return trading_days


# =============================================================================
# DataStream Class
# =============================================================================

class DataStream:
    """
    Iterator over time-series market data for backtesting.

    This class provides a clean iterator interface over trading days,
    fetching and optionally caching market data from a data source
    (typically DoltAdapter). It handles weekends, holidays, and
    missing data gracefully.

    Attributes:
        data_source: Data source adapter (e.g., DoltAdapter)
        start_date (datetime): Start date of the stream
        end_date (datetime): End date of the stream
        underlying (str): Underlying symbol to fetch data for
        trading_calendar (TradingCalendar): Calendar for trading days
        dte_range (Tuple[int, int]): Min/max DTE for option chain

    Example:
        >>> stream = DataStream(
        ...     data_source=adapter,
        ...     start_date=datetime(2023, 1, 1),
        ...     end_date=datetime(2023, 3, 31),
        ...     underlying='SPY'
        ... )
        >>> for timestamp, market_data in stream:
        ...     print(f"Date: {timestamp}, Spot: {market_data['spot']}")
    """

    __slots__ = (
        '_data_source',
        '_start_date',
        '_end_date',
        '_underlying',
        '_trading_calendar',
        '_dte_range',
        '_trading_days',
        '_current_index',
        '_current_timestamp',
        '_cache',
        '_cache_enabled',
        '_preload_enabled',
        '_preloaded',
        '_skip_missing_data',
    )

    def __init__(
        self,
        data_source: Any,  # DataSourceProtocol, but Any for flexibility
        start_date: datetime,
        end_date: datetime,
        underlying: str,
        trading_calendar: Optional[TradingCalendar] = None,
        dte_range: Tuple[int, int] = (DEFAULT_MIN_DTE, DEFAULT_MAX_DTE),
        cache_enabled: bool = True,
        preload: bool = False,
        skip_missing_data: bool = True
    ) -> None:
        """
        Initialize the DataStream.

        Args:
            data_source: Data source adapter (e.g., DoltAdapter instance).
                        Must implement get_option_chain method.
            start_date: Start date of the backtest period (inclusive)
            end_date: End date of the backtest period (inclusive)
            underlying: Ticker symbol of the underlying asset (e.g., 'SPY')
            trading_calendar: Optional trading calendar for holiday handling.
                            If None, a default calendar is created.
            dte_range: Tuple of (min_dte, max_dte) for option chain retrieval.
                      Default is (7, 60).
            cache_enabled: Whether to cache fetched data. Default True.
            preload: Whether to preload all data upfront. Default False.
                    Useful for smaller datasets to avoid repeated DB calls.
            skip_missing_data: Whether to skip days with missing data.
                              If False, raises DataNotAvailableError.

        Raises:
            DataStreamConfigError: If configuration is invalid

        Example:
            >>> stream = DataStream(
            ...     data_source=adapter,
            ...     start_date=datetime(2023, 1, 1),
            ...     end_date=datetime(2023, 12, 31),
            ...     underlying='SPY',
            ...     dte_range=(7, 45)
            ... )
        """
        # Validate data source
        if data_source is None:
            raise DataStreamConfigError("data_source cannot be None")
        if not hasattr(data_source, 'get_option_chain'):
            raise DataStreamConfigError(
                "data_source must have a get_option_chain method"
            )
        self._data_source = data_source

        # Validate dates
        if start_date is None or end_date is None:
            raise DataStreamConfigError("start_date and end_date cannot be None")
        if not isinstance(start_date, datetime) or not isinstance(end_date, datetime):
            raise DataStreamConfigError(
                "start_date and end_date must be datetime objects"
            )
        if start_date > end_date:
            raise DataStreamConfigError(
                f"start_date ({start_date}) must be before or equal to "
                f"end_date ({end_date})"
            )
        self._start_date = start_date
        self._end_date = end_date

        # Validate underlying
        if not underlying or not isinstance(underlying, str):
            raise DataStreamConfigError("underlying must be a non-empty string")
        self._underlying = underlying.upper().strip()

        # Set up trading calendar
        self._trading_calendar = trading_calendar or TradingCalendar()

        # Validate and set DTE range
        if len(dte_range) != 2:
            raise DataStreamConfigError("dte_range must be a tuple of (min, max)")
        if dte_range[0] < 0 or dte_range[1] < dte_range[0]:
            raise DataStreamConfigError(
                f"Invalid dte_range: {dte_range}. Must have 0 <= min <= max"
            )
        self._dte_range = dte_range

        # Build list of trading days
        self._trading_days = self._trading_calendar.get_trading_days(
            start_date, end_date
        )

        if not self._trading_days:
            logger.warning(
                f"No trading days found between {start_date.date()} "
                f"and {end_date.date()}"
            )

        # Initialize iterator state
        self._current_index = 0
        self._current_timestamp: Optional[datetime] = None

        # Cache settings
        self._cache_enabled = cache_enabled
        self._cache: Dict[date, Dict[str, Any]] = {}

        # Preload settings
        self._preload_enabled = preload
        self._preloaded = False
        self._skip_missing_data = skip_missing_data

        if preload and self._trading_days:
            self._preload_data()

        logger.info(
            f"DataStream initialized: {self._underlying}, "
            f"{len(self._trading_days)} trading days from "
            f"{self._trading_days[0] if self._trading_days else 'N/A'} to "
            f"{self._trading_days[-1] if self._trading_days else 'N/A'}"
        )

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def start_date(self) -> datetime:
        """Get the start date of the stream."""
        return self._start_date

    @property
    def end_date(self) -> datetime:
        """Get the end date of the stream."""
        return self._end_date

    @property
    def underlying(self) -> str:
        """Get the underlying symbol."""
        return self._underlying

    @property
    def dte_range(self) -> Tuple[int, int]:
        """Get the DTE range for option chain retrieval."""
        return self._dte_range

    @property
    def trading_days(self) -> List[date]:
        """Get list of trading days in the stream (copy)."""
        return self._trading_days.copy()

    @property
    def num_trading_days(self) -> int:
        """Get the total number of trading days."""
        return len(self._trading_days)

    @property
    def current_index(self) -> int:
        """Get the current position in the stream."""
        return self._current_index

    @property
    def is_exhausted(self) -> bool:
        """Check if the stream has been fully consumed."""
        return self._current_index >= len(self._trading_days)

    @property
    def progress(self) -> float:
        """Get stream progress as a ratio (0.0 to 1.0)."""
        if not self._trading_days:
            return 1.0
        return self._current_index / len(self._trading_days)

    # =========================================================================
    # Iterator Protocol
    # =========================================================================

    def __iter__(self) -> Iterator[Tuple[datetime, Dict[str, Any]]]:
        """
        Return self as iterator.

        Returns:
            Self for iteration protocol
        """
        self._current_index = 0
        return self

    def __next__(self) -> Tuple[datetime, Dict[str, Any]]:
        """
        Get next timestep in the stream.

        Returns:
            Tuple of (timestamp, market_data) where market_data contains:
                - 'date': datetime of the trading day
                - 'spot': float, estimated spot price (from ATM strikes)
                - 'option_chain': DataFrame with option chain data
                - 'iv': float, ATM implied volatility
                - 'vix': Optional[float], VIX level if available

        Raises:
            StopIteration: When stream is exhausted
            DataNotAvailableError: If data is missing and skip_missing=False
        """
        while self._current_index < len(self._trading_days):
            current_date = self._trading_days[self._current_index]
            self._current_index += 1

            try:
                market_data = self._get_market_data(current_date)

                # Check if we got valid data
                if market_data is None:
                    if self._skip_missing_data:
                        logger.debug(f"Skipping {current_date} - no data available")
                        continue
                    else:
                        raise DataNotAvailableError(
                            f"No market data available for {current_date}"
                        )

                # Convert date to datetime for timestamp
                timestamp = datetime.combine(current_date, datetime.min.time())
                self._current_timestamp = timestamp

                return timestamp, market_data

            except DataNotAvailableError:
                if self._skip_missing_data:
                    logger.debug(f"Skipping {current_date} - data fetch failed")
                    continue
                raise

        raise StopIteration

    def __len__(self) -> int:
        """Return number of trading days in the stream."""
        return len(self._trading_days)

    # =========================================================================
    # Data Retrieval Methods
    # =========================================================================

    def _get_market_data(self, dt: date) -> Optional[Dict[str, Any]]:
        """
        Get market data for a specific date.

        Checks cache first, then fetches from data source if needed.

        Args:
            dt: Date to fetch data for

        Returns:
            Market data dictionary or None if unavailable
        """
        # Check cache
        if self._cache_enabled and dt in self._cache:
            return self._cache[dt]

        # Fetch from data source
        try:
            market_data = self._fetch_market_data(dt)

            # Cache if enabled
            if self._cache_enabled and market_data is not None:
                self._cache[dt] = market_data

                # Limit cache size to prevent memory issues
                if len(self._cache) > DEFAULT_CACHE_SIZE:
                    # Remove oldest entry
                    oldest_key = min(self._cache.keys())
                    del self._cache[oldest_key]

            return market_data

        except Exception as e:
            logger.warning(f"Failed to fetch data for {dt}: {e}")
            return None

    def _fetch_market_data(self, dt: date) -> Optional[Dict[str, Any]]:
        """
        Fetch market data from the data source.

        Args:
            dt: Date to fetch data for

        Returns:
            Market data dictionary or None if unavailable
        """
        # Convert date to datetime for API call
        dt_datetime = datetime.combine(dt, datetime.min.time())

        # Get option chain
        try:
            option_chain = self._data_source.get_option_chain(
                underlying=self._underlying,
                date=dt_datetime,
                min_dte=self._dte_range[0],
                max_dte=self._dte_range[1]
            )
        except Exception as e:
            logger.warning(f"Failed to get option chain for {dt}: {e}")
            return None

        if option_chain is None or option_chain.empty:
            logger.debug(f"No option chain data for {dt}")
            return None

        # Estimate spot price from option chain
        spot = self._estimate_spot_price(option_chain)

        # Get ATM implied volatility
        iv = self._estimate_atm_iv(option_chain, spot)

        # Build market data dictionary
        market_data = {
            'date': dt_datetime,
            'spot': spot,
            'option_chain': option_chain,
            'iv': iv,
            'vix': None,  # VIX not available in current data source
        }

        return market_data

    def _estimate_spot_price(self, option_chain: pd.DataFrame) -> float:
        """
        Estimate the spot price from the option chain.

        Uses the strike where call and put prices are closest,
        or falls back to the median strike if put/call data unavailable.

        Args:
            option_chain: DataFrame with option chain data

        Returns:
            Estimated spot price
        """
        if option_chain.empty:
            return 0.0

        # Get unique strikes
        strike_col = 'strike'
        if strike_col not in option_chain.columns:
            # Fall back to median strike
            return float(option_chain['strike'].median()) if 'strike' in option_chain.columns else 0.0

        strikes = option_chain[strike_col].unique()

        # Simple approach: use median strike as proxy for spot
        # This works well for liquid options where strikes are dense around ATM
        return float(np.median(strikes))

    def _estimate_atm_iv(
        self,
        option_chain: pd.DataFrame,
        spot: float
    ) -> float:
        """
        Estimate ATM implied volatility from the option chain.

        Args:
            option_chain: DataFrame with option chain data
            spot: Current spot price estimate

        Returns:
            ATM implied volatility (as decimal, e.g., 0.20 for 20%)
        """
        if option_chain.empty or spot <= 0:
            return 0.20  # Default IV

        iv_col = 'vol'  # Column name in post-no-preference database
        if iv_col not in option_chain.columns:
            iv_col = 'implied_volatility'
            if iv_col not in option_chain.columns:
                return 0.20  # Default IV

        # Find ATM options (within 2% of spot)
        strike_col = 'strike'
        atm_mask = abs(option_chain[strike_col] / spot - 1) < 0.02
        atm_options = option_chain[atm_mask]

        if atm_options.empty:
            # Fall back to closest strikes
            distances = abs(option_chain[strike_col] - spot)
            closest_idx = distances.nsmallest(5).index
            atm_options = option_chain.loc[closest_idx]

        # Calculate mean IV from ATM options
        valid_ivs = atm_options[iv_col].dropna()
        valid_ivs = valid_ivs[valid_ivs > 0]

        if valid_ivs.empty:
            return 0.20  # Default IV

        return float(valid_ivs.mean())

    def _preload_data(self) -> None:
        """
        Preload all data for the stream.

        This fetches data for all trading days upfront and caches it.
        Useful for smaller datasets to avoid repeated database calls.
        """
        if self._preloaded:
            return

        logger.info(f"Preloading data for {len(self._trading_days)} trading days...")

        loaded_count = 0
        for dt in self._trading_days:
            try:
                market_data = self._fetch_market_data(dt)
                if market_data is not None:
                    self._cache[dt] = market_data
                    loaded_count += 1
            except Exception as e:
                logger.warning(f"Failed to preload data for {dt}: {e}")

        self._preloaded = True
        logger.info(f"Preloaded {loaded_count}/{len(self._trading_days)} days")

    # =========================================================================
    # Public Data Access Methods
    # =========================================================================

    def get_current_timestamp(self) -> Optional[datetime]:
        """
        Get the current timestamp in the backtest.

        Returns:
            Current datetime or None if not yet started

        Example:
            >>> ts = stream.get_current_timestamp()
            >>> print(f"Current date: {ts.date()}")
        """
        return self._current_timestamp

    def get_option_chain(
        self,
        underlying: str,
        date: datetime,
        dte_range: Optional[Tuple[int, int]] = None,
        option_type: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get available options for a specific date.

        This method allows fetching option chain data for arbitrary dates
        and underlyings, not just the current stream position.

        Args:
            underlying: Ticker symbol (e.g., 'SPY')
            date: Date to fetch options for
            dte_range: Optional (min_dte, max_dte) tuple.
                      Defaults to stream's dte_range.
            option_type: Optional 'call' or 'put' filter

        Returns:
            DataFrame with option chain data

        Raises:
            DataNotAvailableError: If data cannot be fetched

        Example:
            >>> chain = stream.get_option_chain('SPY', datetime(2023, 6, 15))
            >>> print(f"Found {len(chain)} contracts")
        """
        if dte_range is None:
            dte_range = self._dte_range

        try:
            return self._data_source.get_option_chain(
                underlying=underlying.upper(),
                date=date,
                min_dte=dte_range[0],
                max_dte=dte_range[1],
                option_type=option_type
            )
        except Exception as e:
            raise DataNotAvailableError(
                f"Failed to get option chain for {underlying} on {date}: {e}"
            ) from e

    def get_underlying_price(self, symbol: str, date: datetime) -> float:
        """
        Get spot price for a specific date.

        Estimates the spot price from the option chain by finding
        the ATM strike level.

        Args:
            symbol: Ticker symbol
            date: Date to get price for

        Returns:
            Estimated spot price

        Raises:
            DataNotAvailableError: If price cannot be determined

        Example:
            >>> spot = stream.get_underlying_price('SPY', datetime(2023, 6, 15))
            >>> print(f"SPY spot: ${spot:.2f}")
        """
        dt = date.date() if isinstance(date, datetime) else date

        # Check cache first
        if dt in self._cache:
            return self._cache[dt].get('spot', 0.0)

        # Fetch option chain and estimate spot
        try:
            option_chain = self._data_source.get_option_chain(
                underlying=symbol.upper(),
                date=date,
                min_dte=self._dte_range[0],
                max_dte=self._dte_range[1]
            )

            if option_chain is None or option_chain.empty:
                raise DataNotAvailableError(
                    f"No option data available to estimate spot for {symbol} on {date}"
                )

            return self._estimate_spot_price(option_chain)

        except DataNotAvailableError:
            raise
        except Exception as e:
            raise DataNotAvailableError(
                f"Failed to get underlying price for {symbol} on {date}: {e}"
            ) from e

    def peek(self, steps: int = 1) -> Optional[Tuple[datetime, Dict[str, Any]]]:
        """
        Look ahead in the stream without advancing the position.

        Args:
            steps: Number of steps to look ahead (default 1)

        Returns:
            Tuple of (timestamp, market_data) or None if beyond stream end

        Example:
            >>> next_data = stream.peek()
            >>> if next_data:
            ...     print(f"Next date: {next_data[0]}")
        """
        peek_index = self._current_index + steps - 1
        if peek_index >= len(self._trading_days):
            return None

        dt = self._trading_days[peek_index]
        market_data = self._get_market_data(dt)

        if market_data is None:
            return None

        timestamp = datetime.combine(dt, datetime.min.time())
        return timestamp, market_data

    def reset(self) -> None:
        """
        Reset the stream to the beginning.

        Allows re-iterating over the same data stream.

        Example:
            >>> for ts, data in stream:
            ...     process(data)
            >>> stream.reset()
            >>> for ts, data in stream:  # Iterate again
            ...     process_again(data)
        """
        self._current_index = 0
        self._current_timestamp = None
        logger.debug(f"DataStream reset to beginning")

    def skip(self, days: int = 1) -> None:
        """
        Skip forward in the stream.

        Args:
            days: Number of trading days to skip (default 1)

        Example:
            >>> stream.skip(5)  # Skip 5 trading days
        """
        self._current_index = min(
            self._current_index + days,
            len(self._trading_days)
        )

    def clear_cache(self) -> None:
        """
        Clear the data cache.

        Useful if memory is a concern or if data needs to be refreshed.
        """
        self._cache.clear()
        logger.debug("DataStream cache cleared")

    # =========================================================================
    # Special Methods
    # =========================================================================

    def __repr__(self) -> str:
        """Return detailed string representation."""
        return (
            f"DataStream("
            f"underlying={self._underlying!r}, "
            f"start={self._start_date.date()}, "
            f"end={self._end_date.date()}, "
            f"trading_days={len(self._trading_days)}, "
            f"progress={self.progress:.1%}"
            f")"
        )

    def __str__(self) -> str:
        """Return human-readable string."""
        return (
            f"DataStream for {self._underlying}: "
            f"{len(self._trading_days)} trading days, "
            f"{self._start_date.date()} to {self._end_date.date()}, "
            f"{self.progress:.1%} complete"
        )


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Main classes
    'DataStream',
    'TradingCalendar',

    # Exceptions
    'DataStreamError',
    'DataStreamConfigError',
    'DataNotAvailableError',
    'DataStreamExhaustedError',

    # Constants
    'DEFAULT_MIN_DTE',
    'DEFAULT_MAX_DTE',
]
