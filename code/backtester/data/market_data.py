"""
Market Data Loader for Options Backtesting

This module provides a high-level interface for loading and processing market data
required for options backtesting. It abstracts away the database layer and provides
convenient methods for common data retrieval patterns used in strategy development.

The MarketDataLoader sits on top of the DoltAdapter and adds:
- Data preprocessing and standardization
- Caching for frequently accessed data
- Calculation of derived metrics (IV percentile, historical volatility, etc.)
- Data quality filtering

Usage:
    from backtester.data.dolt_adapter import DoltAdapter
    from backtester.data.market_data import MarketDataLoader

    adapter = DoltAdapter('/path/to/dolt_data/options')
    adapter.connect()

    loader = MarketDataLoader(adapter)
    prices = loader.load_underlying_prices('SPY', start_date, end_date)
    iv_pct = loader.calculate_iv_percentile('SPY', date)

References:
    - IV Rank/Percentile: https://www.tastylive.com/concepts-strategies/implied-volatility-rank
    - Historical Volatility: Standard deviation of log returns, annualized
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
from functools import lru_cache
import warnings

import numpy as np
import pandas as pd

from backtester.data.dolt_adapter import DoltAdapter, DoltQueryError

# Configure module logger
logger = logging.getLogger(__name__)


class DataLoadError(Exception):
    """Exception raised when data loading fails."""
    pass


class InsufficientDataError(Exception):
    """Exception raised when there is insufficient data for calculations."""
    pass


class MarketDataLoader:
    """
    High-level interface for loading and processing market data.

    This class provides convenient methods for loading options and underlying
    price data, with built-in data quality filtering and derived metrics
    calculation. It is designed to be the primary data interface for
    backtesting strategies.

    Attributes:
        dolt_adapter (DoltAdapter): The underlying database adapter.
        _price_cache (dict): Cache for underlying price data.
        _iv_cache (dict): Cache for implied volatility calculations.

    Features:
        - Loads and preprocesses underlying price data
        - Loads and filters option chain data
        - Calculates IV percentile/rank for volatility filtering
        - Calculates historical realized volatility
        - Caches frequently accessed data for performance
        - Handles data gaps and missing values

    Example:
        >>> adapter = DoltAdapter('/path/to/db')
        >>> adapter.connect()
        >>> loader = MarketDataLoader(adapter)
        >>>
        >>> # Load SPY prices for 2023
        >>> prices = loader.load_underlying_prices(
        ...     symbol='SPY',
        ...     start_date=datetime(2023, 1, 1),
        ...     end_date=datetime(2023, 12, 31)
        ... )
        >>>
        >>> # Get IV percentile for strategy filtering
        >>> iv_pct = loader.calculate_iv_percentile(
        ...     symbol='SPY',
        ...     date=datetime(2023, 6, 15),
        ...     window=252
        ... )
        >>> print(f"IV Percentile: {iv_pct:.1f}%")
    """

    # Trading days per year for annualization
    TRADING_DAYS_PER_YEAR = 252

    # Default parameters
    DEFAULT_IV_WINDOW = 252  # 1 year for IV percentile calculation
    DEFAULT_HV_WINDOW = 20   # 20 days for historical volatility

    # Column name mappings from database schema to standard names
    # The post-no-preference/options database uses different column names
    COLUMN_MAPPINGS = {
        # Option chain columns (database name -> standard name)
        'date': 'quote_date',
        'act_symbol': 'underlying_symbol',
        'call_put': 'option_type',
        'vol': 'implied_volatility',
        # These are the same in both
        'bid': 'bid',
        'ask': 'ask',
        'strike': 'strike',
        'expiration': 'expiration',
        'delta': 'delta',
        'gamma': 'gamma',
        'theta': 'theta',
        'vega': 'vega',
        'rho': 'rho',
        # Volatility history columns
        'hv_current': 'hv_current',
        'iv_current': 'iv_current',
    }

    # Reverse mapping (standard name -> database name)
    COLUMN_MAPPINGS_REVERSE = {v: k for k, v in COLUMN_MAPPINGS.items()}

    def __init__(
        self,
        dolt_adapter: DoltAdapter,
        cache_enabled: bool = True
    ) -> None:
        """
        Initialize the MarketDataLoader.

        Args:
            dolt_adapter: Connected DoltAdapter instance for database access.
            cache_enabled: Whether to enable caching for repeated queries.
                          Default True. Disable for memory-constrained environments.

        Raises:
            ValueError: If dolt_adapter is None.
            TypeError: If dolt_adapter is not a DoltAdapter instance.

        Example:
            >>> adapter = DoltAdapter('/path/to/db')
            >>> adapter.connect()
            >>> loader = MarketDataLoader(adapter)
        """
        if dolt_adapter is None:
            raise ValueError("dolt_adapter cannot be None")

        if not isinstance(dolt_adapter, DoltAdapter):
            raise TypeError(
                f"dolt_adapter must be a DoltAdapter instance, "
                f"got {type(dolt_adapter).__name__}"
            )

        self.dolt_adapter = dolt_adapter
        self._cache_enabled = cache_enabled

        # Initialize caches
        self._price_cache: Dict[str, pd.DataFrame] = {}
        self._iv_history_cache: Dict[str, pd.Series] = {}

        logger.info(
            f"MarketDataLoader initialized with caching "
            f"{'enabled' if cache_enabled else 'disabled'}"
        )

    def load_underlying_prices(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        columns: Optional[List[str]] = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Load historical underlying asset prices.

        ⚠️ WARNING: The post-no-preference/options database does NOT contain OHLCV
        price data. This method currently returns volatility history data instead.
        For actual price data, you must use an external data source (e.g., yfinance,
        Alpha Vantage, Polygon.io).

        This method retrieves OHLCV price data for the specified symbol
        and date range. The data is indexed by date and sorted chronologically.

        Args:
            symbol: Ticker symbol (e.g., 'SPY', 'SPX').
            start_date: Start date of the price history (inclusive).
            end_date: End date of the price history (inclusive).
            columns: Optional list of columns to return. If None, returns all.
                    Valid columns: 'open', 'high', 'low', 'close', 'volume'
                    NOTE: These columns will NOT be present in current database
            use_cache: Whether to use cached data if available. Default True.

        Returns:
            DataFrame with price data indexed by date. Contains columns:
            - open: Opening price
            - high: High price
            - low: Low price
            - close: Closing price
            - volume: Trading volume
            - Additional calculated columns may be added

        Raises:
            ValueError: If symbol is empty or dates are invalid.
            DataLoadError: If data retrieval fails.

        Example:
            >>> prices = loader.load_underlying_prices(
            ...     symbol='SPY',
            ...     start_date=datetime(2023, 1, 1),
            ...     end_date=datetime(2023, 12, 31)
            ... )
            >>> print(f"Loaded {len(prices)} trading days")
            >>> print(f"Price range: ${prices['close'].min():.2f} - ${prices['close'].max():.2f}")
        """
        # Input validation
        if not symbol:
            raise ValueError("symbol cannot be empty")
        if start_date is None or end_date is None:
            raise ValueError("start_date and end_date cannot be None")
        if start_date > end_date:
            raise ValueError("start_date cannot be after end_date")

        symbol = symbol.upper()
        cache_key = f"{symbol}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"

        # Check cache
        if use_cache and self._cache_enabled and cache_key in self._price_cache:
            logger.debug(f"Using cached price data for {symbol}")
            cached_data = self._price_cache[cache_key]
            if columns is not None:
                return cached_data[columns].copy()
            return cached_data.copy()

        # Load from database
        try:
            df = self.dolt_adapter.get_underlying_prices(
                symbol=symbol,
                start=start_date,
                end=end_date
            )

            if df.empty:
                logger.warning(
                    f"No price data found for {symbol} from "
                    f"{start_date.date()} to {end_date.date()}"
                )
                return df

            # Standardize column names to lowercase
            df.columns = df.columns.str.lower()

            # Set date as index
            date_col = 'date'
            if date_col in df.columns:
                df[date_col] = pd.to_datetime(df[date_col])
                df = df.set_index(date_col).sort_index()

            # Ensure numeric types for price columns
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # Cache the result
            if self._cache_enabled:
                self._price_cache[cache_key] = df

            logger.info(f"Loaded {len(df)} price records for {symbol}")

            # Return requested columns
            if columns is not None:
                available_cols = [c for c in columns if c in df.columns]
                if len(available_cols) < len(columns):
                    missing = set(columns) - set(available_cols)
                    logger.warning(f"Requested columns not found: {missing}")
                return df[available_cols].copy()

            return df

        except DoltQueryError as e:
            error_msg = f"Failed to load prices for {symbol}: {str(e)}"
            logger.error(error_msg)
            raise DataLoadError(error_msg) from e

    def load_option_chain(
        self,
        underlying: str,
        date: datetime,
        dte_range: Tuple[int, int] = (0, 60),
        min_volume: int = 0,
        min_open_interest: int = 0,
        iv_range: Optional[Tuple[float, float]] = None,
        moneyness_range: Optional[Tuple[float, float]] = None,
        spot_price: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Load and filter option chain data for a specific date.

        This method retrieves option chain data and applies various filters
        to remove low-quality or irrelevant options. It can filter by
        liquidity metrics, IV ranges, and moneyness.

        Args:
            underlying: Ticker symbol of the underlying asset.
            date: Quote date for the option chain.
            dte_range: Tuple of (min_dte, max_dte) for expiration filtering.
                      Default (0, 60) includes options expiring in 0-60 days.
            min_volume: Minimum trading volume filter. Default 0.
            min_open_interest: Minimum open interest filter. Default 0.
            iv_range: Optional tuple of (min_iv, max_iv) to filter extreme IVs.
                     Values should be decimals (e.g., (0.10, 2.0) for 10%-200%).
            moneyness_range: Optional tuple of (min_moneyness, max_moneyness).
                            Moneyness = Strike/Spot. Default None (no filter).
                            Example: (0.9, 1.1) for 90%-110% of spot.
            spot_price: Spot price for moneyness calculation. If None and
                       moneyness_range is provided, will attempt to estimate.

        Returns:
            DataFrame containing filtered option chain with standard columns.
            Returns empty DataFrame if no data matches criteria.

        Raises:
            ValueError: If underlying is empty or date is None.
            DataLoadError: If data retrieval fails.

        Example:
            >>> chain = loader.load_option_chain(
            ...     underlying='SPY',
            ...     date=datetime(2023, 6, 15),
            ...     dte_range=(7, 30),
            ...     min_open_interest=100,
            ...     moneyness_range=(0.95, 1.05),
            ...     spot_price=440.0
            ... )
            >>> print(f"Found {len(chain)} liquid near-the-money options")
        """
        # Input validation
        if not underlying:
            raise ValueError("underlying cannot be empty")
        if date is None:
            raise ValueError("date cannot be None")

        min_dte, max_dte = dte_range

        try:
            # Load raw option chain
            df = self.dolt_adapter.get_option_chain(
                underlying=underlying,
                date=date,
                min_dte=min_dte,
                max_dte=max_dte
            )

            if df.empty:
                logger.warning(
                    f"No option chain data for {underlying} on {date.date()}"
                )
                return df

            # Standardize column names to lowercase
            df.columns = df.columns.str.lower()

            # Apply IV filter using actual database column name 'vol'
            # Note: Database uses 'vol' not 'implied_volatility'
            iv_col = 'vol' if 'vol' in df.columns else 'implied_volatility'
            if iv_range is not None and iv_col in df.columns:
                min_iv, max_iv = iv_range
                df = df[
                    (df[iv_col] >= min_iv) &
                    (df[iv_col] <= max_iv)
                ]

            # Note: The post-no-preference/options database does not have
            # volume or open_interest columns. These filters are skipped
            # if the columns don't exist.
            # Apply volume filter (if available)
            if min_volume > 0 and 'volume' in df.columns:
                df = df[df['volume'] >= min_volume]

            # Apply open interest filter (if available)
            if min_open_interest > 0 and 'open_interest' in df.columns:
                df = df[df['open_interest'] >= min_open_interest]

            # Apply moneyness filter
            if moneyness_range is not None and 'strike' in df.columns:
                min_m, max_m = moneyness_range

                if spot_price is None:
                    # Estimate spot from ATM options
                    spot_price = self._estimate_spot_from_chain(df)

                if spot_price is not None and spot_price > 0:
                    moneyness = df['strike'] / spot_price
                    df = df[(moneyness >= min_m) & (moneyness <= max_m)]

            # Convert date columns to datetime
            # Note: Database uses 'date' column, not 'quote_date'
            date_col = 'date' if 'date' in df.columns else 'quote_date'
            for col in [date_col, 'expiration']:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])

            # Calculate DTE if not present
            if 'dte' not in df.columns and 'expiration' in df.columns and date_col in df.columns:
                df['dte'] = (df['expiration'] - df[date_col]).dt.days

            logger.info(
                f"Loaded {len(df)} options for {underlying} on {date.date()} "
                f"(DTE: {min_dte}-{max_dte})"
            )

            return df.reset_index(drop=True)

        except DoltQueryError as e:
            error_msg = f"Failed to load option chain for {underlying}: {str(e)}"
            logger.error(error_msg)
            raise DataLoadError(error_msg) from e

    def _estimate_spot_from_chain(self, chain: pd.DataFrame) -> Optional[float]:
        """
        Estimate spot price from option chain data.

        Uses put-call parity approximation: For ATM options, the strike
        where call and put prices are closest is approximately the spot.

        Args:
            chain: Option chain DataFrame.

        Returns:
            Estimated spot price, or None if estimation fails.
        """
        if chain.empty:
            return None

        try:
            # Group by strike and find where call-put spread is smallest
            # Note: Database uses 'call_put' column, not 'option_type'
            option_type_col = 'call_put' if 'call_put' in chain.columns else 'option_type'
            if option_type_col not in chain.columns or 'strike' not in chain.columns:
                return None

            if 'mid' in chain.columns:
                price_col = 'mid'
            elif 'bid' in chain.columns and 'ask' in chain.columns:
                chain = chain.copy()
                chain['mid'] = (chain['bid'] + chain['ask']) / 2
                price_col = 'mid'
            else:
                return None

            calls = chain[chain[option_type_col].str.lower() == 'call'].set_index('strike')[price_col]
            puts = chain[chain[option_type_col].str.lower() == 'put'].set_index('strike')[price_col]

            # Find common strikes
            common_strikes = calls.index.intersection(puts.index)

            if len(common_strikes) == 0:
                # Fall back to median strike
                return chain['strike'].median()

            # ATM is where call - put is closest to 0 (simplified put-call parity)
            spreads = calls.loc[common_strikes] - puts.loc[common_strikes]
            atm_strike = spreads.abs().idxmin()

            return float(atm_strike)

        except Exception as e:
            logger.warning(f"Failed to estimate spot price: {str(e)}")
            return chain['strike'].median() if 'strike' in chain.columns else None

    def calculate_iv_percentile(
        self,
        symbol: str,
        date: datetime,
        window: int = 252,
        use_atm: bool = True,
        dte_target: int = 30
    ) -> float:
        """
        Calculate implied volatility percentile for a given date.

        IV Percentile measures the current IV relative to its historical range.
        A percentile of 75 means current IV is higher than 75% of observations
        over the lookback window. This is a key metric for volatility-selling
        strategies.

        Formula:
            IV Percentile = (Number of days IV was lower than current) / Total Days * 100

        Args:
            symbol: Ticker symbol of the underlying.
            date: Date for which to calculate IV percentile.
            window: Lookback period in trading days. Default 252 (1 year).
            use_atm: If True, uses ATM IV. If False, uses average IV. Default True.
            dte_target: Target days to expiration for IV calculation. Default 30.
                       Uses options closest to this DTE for consistent comparison.

        Returns:
            IV percentile as a value between 0 and 100.
            Returns NaN if insufficient data for calculation.

        Raises:
            ValueError: If symbol is empty or date is None.
            InsufficientDataError: If less than 30 days of IV history available.

        Note:
            IV Percentile differs from IV Rank:
            - IV Percentile: % of observations below current IV
            - IV Rank: (Current - Min) / (Max - Min) * 100

            This method calculates IV Percentile, which is generally preferred
            as it's more robust to extreme values.

        Example:
            >>> iv_pct = loader.calculate_iv_percentile(
            ...     symbol='SPY',
            ...     date=datetime(2023, 6, 15),
            ...     window=252
            ... )
            >>> if iv_pct > 50:
            ...     print("IV is above median - favorable for premium selling")
        """
        # Input validation
        if not symbol:
            raise ValueError("symbol cannot be empty")
        if date is None:
            raise ValueError("date cannot be None")
        if window < 1:
            raise ValueError("window must be at least 1")

        symbol = symbol.upper()

        # Calculate start date for lookback
        # Add buffer for non-trading days
        lookback_start = date - timedelta(days=int(window * 1.5))

        # Get IV history
        iv_history = self._get_iv_history(
            symbol=symbol,
            start_date=lookback_start,
            end_date=date,
            dte_target=dte_target,
            use_atm=use_atm
        )

        if iv_history is None or len(iv_history) == 0:
            logger.warning(f"No IV history available for {symbol}")
            return np.nan

        # Need minimum data for meaningful percentile
        min_observations = 30
        if len(iv_history) < min_observations:
            logger.warning(
                f"Insufficient IV history for {symbol}: "
                f"{len(iv_history)} observations, need {min_observations}"
            )
            raise InsufficientDataError(
                f"Need at least {min_observations} IV observations, "
                f"only have {len(iv_history)}"
            )

        # Get current IV
        if date in iv_history.index:
            current_iv = iv_history.loc[date]
        else:
            # Find closest date
            closest_idx = (iv_history.index - pd.Timestamp(date)).abs().argmin()
            current_iv = iv_history.iloc[closest_idx]

        if pd.isna(current_iv) or current_iv <= 0:
            logger.warning(f"Invalid current IV for {symbol} on {date.date()}")
            return np.nan

        # Use most recent 'window' observations
        iv_history_window = iv_history.tail(window)

        # Calculate percentile
        # Count observations below current IV
        below_current = (iv_history_window < current_iv).sum()
        percentile = (below_current / len(iv_history_window)) * 100

        logger.debug(
            f"IV Percentile for {symbol} on {date.date()}: {percentile:.1f}% "
            f"(current IV: {current_iv:.4f})"
        )

        return percentile

    def calculate_iv_rank(
        self,
        symbol: str,
        date: datetime,
        window: int = 252,
        dte_target: int = 30
    ) -> float:
        """
        Calculate implied volatility rank for a given date.

        IV Rank measures where current IV falls within its historical range,
        normalized to 0-100. Unlike IV Percentile, it's sensitive to extreme
        values as it uses min/max.

        Formula:
            IV Rank = (Current IV - Min IV) / (Max IV - Min IV) * 100

        Args:
            symbol: Ticker symbol of the underlying.
            date: Date for which to calculate IV rank.
            window: Lookback period in trading days. Default 252 (1 year).
            dte_target: Target DTE for IV calculation. Default 30.

        Returns:
            IV rank as a value between 0 and 100.
            Returns NaN if insufficient data.

        Example:
            >>> iv_rank = loader.calculate_iv_rank('SPY', datetime(2023, 6, 15))
            >>> print(f"IV Rank: {iv_rank:.1f}%")
        """
        if not symbol:
            raise ValueError("symbol cannot be empty")
        if date is None:
            raise ValueError("date cannot be None")

        symbol = symbol.upper()

        lookback_start = date - timedelta(days=int(window * 1.5))

        iv_history = self._get_iv_history(
            symbol=symbol,
            start_date=lookback_start,
            end_date=date,
            dte_target=dte_target
        )

        if iv_history is None or len(iv_history) < 30:
            logger.warning(f"Insufficient IV history for {symbol}")
            return np.nan

        # Use window observations
        iv_window = iv_history.tail(window)

        # Get current IV
        if date in iv_history.index:
            current_iv = iv_history.loc[date]
        else:
            current_iv = iv_history.iloc[-1]

        if pd.isna(current_iv):
            return np.nan

        min_iv = iv_window.min()
        max_iv = iv_window.max()

        if max_iv == min_iv:
            # Avoid division by zero
            return 50.0

        iv_rank = ((current_iv - min_iv) / (max_iv - min_iv)) * 100

        return iv_rank

    def _get_iv_history(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        dte_target: int = 30,
        use_atm: bool = True
    ) -> Optional[pd.Series]:
        """
        Get historical IV time series for a symbol.

        This is an internal method that retrieves IV history by querying
        historical option chains and extracting IV values.

        Args:
            symbol: Ticker symbol.
            start_date: Start of the history period.
            end_date: End of the history period.
            dte_target: Target DTE for consistent comparison.
            use_atm: Whether to use ATM IV only.

        Returns:
            Series of IV values indexed by date, or None if unavailable.
        """
        cache_key = f"{symbol}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}_{dte_target}"

        if self._cache_enabled and cache_key in self._iv_history_cache:
            return self._iv_history_cache[cache_key]

        try:
            # Query historical options data
            # Use a DTE range around the target
            dte_min = max(dte_target - 5, 1)
            dte_max = dte_target + 5

            # This is a simplified approach - for production, would query
            # options for each trading day in the range
            historical_options = self.dolt_adapter.get_historical_options(
                underlying=symbol,
                start=start_date,
                end=end_date,
                strikes=None,
                expirations=None
            )

            if historical_options.empty:
                logger.warning(f"No historical options data for {symbol}")
                return None

            # Standardize columns
            historical_options.columns = historical_options.columns.str.lower()

            # Determine date column name (database uses 'date', not 'quote_date')
            date_col = 'date' if 'date' in historical_options.columns else 'quote_date'

            # Filter to target DTE range
            if date_col in historical_options.columns and 'expiration' in historical_options.columns:
                historical_options[date_col] = pd.to_datetime(historical_options[date_col])
                historical_options['expiration'] = pd.to_datetime(historical_options['expiration'])
                historical_options['dte'] = (
                    historical_options['expiration'] - historical_options[date_col]
                ).dt.days

                mask = (historical_options['dte'] >= dte_min) & (historical_options['dte'] <= dte_max)
                historical_options = historical_options[mask]

            if historical_options.empty:
                logger.warning(f"No options with DTE {dte_target}+-5 for {symbol}")
                return None

            # Extract IV time series
            # Database uses 'vol' column, not 'implied_volatility'
            iv_col = 'vol' if 'vol' in historical_options.columns else 'implied_volatility'
            if iv_col not in historical_options.columns:
                logger.warning("No IV column (vol or implied_volatility) found in data")
                return None

            # Group by date and calculate daily IV
            if use_atm:
                # Use average IV of options closest to ATM
                iv_series = self._calculate_daily_atm_iv(historical_options, date_col=date_col, iv_col=iv_col)
            else:
                # Use mean IV across all options
                iv_series = historical_options.groupby(date_col)[iv_col].mean()

            iv_series = iv_series.dropna()
            iv_series = iv_series[iv_series > 0]  # Remove invalid IVs

            if self._cache_enabled:
                self._iv_history_cache[cache_key] = iv_series

            return iv_series

        except Exception as e:
            logger.error(f"Failed to get IV history for {symbol}: {str(e)}")
            return None

    def _calculate_daily_atm_iv(
        self,
        options: pd.DataFrame,
        date_col: str = 'date',
        iv_col: str = 'vol'
    ) -> pd.Series:
        """
        Calculate daily ATM IV from options data.

        For each day, finds the strike closest to spot and averages
        the call and put IV at that strike.

        Args:
            options: DataFrame with options data.
            date_col: Name of the date column.
            iv_col: Name of the implied volatility column.

        Returns:
            Series of daily ATM IV values.
        """
        if options.empty:
            return pd.Series()

        results = {}

        for date, day_data in options.groupby(date_col):
            try:
                # Estimate spot from the options
                spot = self._estimate_spot_from_chain(day_data)

                if spot is None or spot <= 0:
                    continue

                # Find ATM strike (closest to spot)
                strikes = day_data['strike'].unique()
                atm_strike = strikes[np.argmin(np.abs(strikes - spot))]

                # Get ATM options
                atm_options = day_data[day_data['strike'] == atm_strike]

                # Average IV at ATM (use iv_col parameter, not hardcoded column name)
                atm_iv = atm_options[iv_col].mean()

                if pd.notna(atm_iv) and atm_iv > 0:
                    results[date] = atm_iv

            except Exception as e:
                logger.debug(f"Failed to calculate ATM IV for {date}: {str(e)}")
                continue

        return pd.Series(results)

    def calculate_historical_volatility(
        self,
        symbol: str,
        date: datetime,
        window: int = 20,
        annualize: bool = True
    ) -> float:
        """
        Calculate historical (realized) volatility for an underlying.

        Historical volatility is the standard deviation of log returns
        over the specified window. This can be compared to implied
        volatility to assess the volatility risk premium.

        Formula:
            Daily Vol = std(ln(P_t / P_{t-1}))
            Annual Vol = Daily Vol * sqrt(252)

        Args:
            symbol: Ticker symbol.
            date: Date for which to calculate HV.
            window: Lookback period in trading days. Default 20 (1 month).
            annualize: Whether to annualize the volatility. Default True.

        Returns:
            Historical volatility as a decimal (e.g., 0.15 for 15% vol).
            Returns NaN if insufficient data.

        Example:
            >>> hv = loader.calculate_historical_volatility(
            ...     symbol='SPY',
            ...     date=datetime(2023, 6, 15),
            ...     window=20
            ... )
            >>> print(f"20-day HV: {hv:.2%}")
        """
        if not symbol:
            raise ValueError("symbol cannot be empty")
        if date is None:
            raise ValueError("date cannot be None")

        # Add buffer for data retrieval
        start_date = date - timedelta(days=int(window * 2))

        try:
            prices = self.load_underlying_prices(
                symbol=symbol,
                start_date=start_date,
                end_date=date,
                columns=['close']
            )

            if prices.empty or len(prices) < window:
                logger.warning(
                    f"Insufficient price data for HV calculation: "
                    f"need {window}, have {len(prices)}"
                )
                return np.nan

            # Calculate log returns
            close_prices = prices['close']
            log_returns = np.log(close_prices / close_prices.shift(1)).dropna()

            # Use most recent 'window' returns
            recent_returns = log_returns.tail(window)

            if len(recent_returns) < window:
                logger.warning(f"Only {len(recent_returns)} returns available")
                return np.nan

            # Calculate volatility
            daily_vol = recent_returns.std()

            if annualize:
                return daily_vol * np.sqrt(self.TRADING_DAYS_PER_YEAR)
            else:
                return daily_vol

        except Exception as e:
            logger.error(f"Failed to calculate HV for {symbol}: {str(e)}")
            return np.nan

    def get_spot_price(
        self,
        symbol: str,
        date: datetime
    ) -> Optional[float]:
        """
        Get the spot price for a symbol on a specific date.

        Args:
            symbol: Ticker symbol.
            date: Date for which to get the spot price.

        Returns:
            Closing price on the specified date, or None if unavailable.

        Example:
            >>> spot = loader.get_spot_price('SPY', datetime(2023, 6, 15))
            >>> print(f"SPY spot: ${spot:.2f}")
        """
        try:
            prices = self.load_underlying_prices(
                symbol=symbol,
                start_date=date - timedelta(days=5),  # Buffer for holidays
                end_date=date,
                columns=['close']
            )

            if prices.empty:
                return None

            # Get the price on or before the requested date
            valid_prices = prices[prices.index <= pd.Timestamp(date)]

            if valid_prices.empty:
                return None

            return float(valid_prices['close'].iloc[-1])

        except Exception as e:
            logger.error(f"Failed to get spot price for {symbol}: {str(e)}")
            return None

    def get_trading_dates(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[datetime]:
        """
        Get list of trading dates for a symbol within a date range.

        This is useful for iterating through backtests and handling
        market closures (weekends, holidays).

        Args:
            symbol: Ticker symbol.
            start_date: Start of the range.
            end_date: End of the range.

        Returns:
            List of datetime objects representing trading days.

        Example:
            >>> dates = loader.get_trading_dates('SPY', start, end)
            >>> print(f"Found {len(dates)} trading days")
        """
        try:
            prices = self.load_underlying_prices(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                columns=['close']
            )

            if prices.empty:
                return []

            return [d.to_pydatetime() for d in prices.index]

        except Exception as e:
            logger.error(f"Failed to get trading dates: {str(e)}")
            return []

    def clear_cache(self) -> None:
        """
        Clear all cached data.

        Call this method to free memory or ensure fresh data is loaded.
        """
        self._price_cache.clear()
        self._iv_history_cache.clear()
        logger.info("MarketDataLoader cache cleared")

    def get_cache_stats(self) -> Dict[str, int]:
        """
        Get statistics about cached data.

        Returns:
            Dictionary with cache statistics.
        """
        return {
            'price_cache_entries': len(self._price_cache),
            'iv_cache_entries': len(self._iv_history_cache),
            'cache_enabled': self._cache_enabled
        }

    def __repr__(self) -> str:
        """Return string representation of the loader."""
        cache_status = "enabled" if self._cache_enabled else "disabled"
        return (
            f"MarketDataLoader(adapter={self.dolt_adapter!r}, "
            f"cache={cache_status})"
        )
