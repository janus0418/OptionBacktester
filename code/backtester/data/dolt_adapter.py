"""
Dolt Database Adapter for Options Data

This module provides a robust interface for connecting to and querying a Dolt database
containing historical options data. Dolt is a version-controlled SQL database that
enables Git-like operations on data.

The adapter is designed to work with the post-no-preference/options database from DoltHub,
which contains historical options chain data including pricing, Greeks, and market data.

Note: The database schema is assumed based on typical options data structure. The actual
schema should be verified once the database is fully downloaded and available.

Usage:
    from backtester.data.dolt_adapter import DoltAdapter

    adapter = DoltAdapter('/path/to/dolt_data/options')
    df = adapter.get_option_chain('SPY', datetime(2023, 1, 3), min_dte=1, max_dte=7)
    adapter.close()

References:
    - DoltHub: https://www.dolthub.com/repositories/post-no-preference/options
    - Dolt Documentation: https://docs.dolthub.com/
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Union, Any
from contextlib import contextmanager

import pandas as pd
import numpy as np

# Configure module logger
logger = logging.getLogger(__name__)


class DoltConnectionError(Exception):
    """Exception raised when connection to Dolt database fails."""
    pass


class DoltQueryError(Exception):
    """Exception raised when a query execution fails."""
    pass


class DoltAdapter:
    """
    Adapter class for connecting to and querying a Dolt database containing options data.

    This class provides a high-level interface for retrieving historical options data
    from a local Dolt repository. It handles connection management, query execution,
    and data transformation into pandas DataFrames.

    The adapter supports multiple connection methods:
    1. Direct SQL execution via doltpy library
    2. MySQL client connection when sql-server is running

    Attributes:
        db_path (str): Path to the local Dolt database directory.
        connection: Active database connection object.
        _is_connected (bool): Flag indicating whether a connection is established.

    Expected Database Schema (to be verified):
        Tables:
            - option_chain: Historical options quotes
            - underlying_prices: Historical underlying asset prices

        option_chain columns:
            - quote_date (DATE): Trading date
            - underlying_symbol (VARCHAR): Ticker symbol (e.g., 'SPY')
            - expiration (DATE): Option expiration date
            - strike (DECIMAL): Strike price
            - option_type (VARCHAR): 'call' or 'put'
            - bid (DECIMAL): Bid price
            - ask (DECIMAL): Ask price
            - mid (DECIMAL): Mid-market price
            - volume (INT): Trading volume
            - open_interest (INT): Open interest
            - implied_volatility (DECIMAL): Implied volatility
            - delta (DECIMAL): Option delta
            - gamma (DECIMAL): Option gamma
            - theta (DECIMAL): Option theta
            - vega (DECIMAL): Option vega
            - rho (DECIMAL): Option rho

        underlying_prices columns:
            - date (DATE): Trading date
            - symbol (VARCHAR): Ticker symbol
            - open (DECIMAL): Opening price
            - high (DECIMAL): High price
            - low (DECIMAL): Low price
            - close (DECIMAL): Closing price
            - volume (BIGINT): Trading volume

    Example:
        >>> adapter = DoltAdapter('/Users/janussuk/Desktop/OptionsBacktester2/dolt_data/options')
        >>> adapter.connect()
        >>> chain = adapter.get_option_chain(
        ...     underlying='SPY',
        ...     date=datetime(2023, 1, 3),
        ...     min_dte=1,
        ...     max_dte=7
        ... )
        >>> print(f"Retrieved {len(chain)} option contracts")
        >>> adapter.close()
    """

    # Column mappings based on actual database schema (post-no-preference/options)
    # The database uses different column names than standard conventions
    # These mappings translate from standard names to actual database column names
    #
    # Actual option_chain schema:
    #   date, act_symbol, expiration, strike, call_put, bid, ask, vol,
    #   delta, gamma, theta, vega, rho
    #
    # Actual volatility_history schema:
    #   date, act_symbol, hv_current, hv_week_ago, hv_month_ago, etc.

    OPTION_CHAIN_COLUMNS = {
        # Standard name -> Actual database column name
        'quote_date': 'date',
        'underlying_symbol': 'act_symbol',
        'expiration': 'expiration',
        'strike': 'strike',
        'option_type': 'call_put',
        'bid': 'bid',
        'ask': 'ask',
        'implied_volatility': 'vol',
        'delta': 'delta',
        'gamma': 'gamma',
        'theta': 'theta',
        'vega': 'vega',
        'rho': 'rho'
    }

    # Reverse mapping: actual column name -> standard name
    OPTION_CHAIN_COLUMNS_REVERSE = {v: k for k, v in OPTION_CHAIN_COLUMNS.items()}

    # Volatility history table column mappings
    VOLATILITY_HISTORY_COLUMNS = {
        'date': 'date',
        'underlying_symbol': 'act_symbol',
        'hv_current': 'hv_current',
        'iv_current': 'iv_current',
        'hv_week_ago': 'hv_week_ago',
        'iv_week_ago': 'iv_week_ago',
        'hv_month_ago': 'hv_month_ago',
        'iv_month_ago': 'iv_month_ago',
        'hv_year_high': 'hv_year_high',
        'iv_year_high': 'iv_year_high',
        'hv_year_low': 'hv_year_low',
        'iv_year_low': 'iv_year_low',
    }

    # Note: The database does not have:
    #   - mid (calculated as (bid + ask) / 2)
    #   - volume (trading volume)
    #   - open_interest
    #   - underlying_prices table (use volatility_history or external data)

    def __init__(self, db_path: str) -> None:
        """
        Initialize the DoltAdapter with a path to the database.

        Args:
            db_path: Absolute path to the local Dolt database directory.
                     This should be the directory containing the .dolt folder.

        Raises:
            ValueError: If db_path is empty or None.
            FileNotFoundError: If the specified path does not exist.

        Example:
            >>> adapter = DoltAdapter('/Users/janussuk/Desktop/OptionsBacktester2/dolt_data/options')
        """
        if not db_path:
            raise ValueError("Database path cannot be empty or None")

        self.db_path = Path(db_path).resolve()

        if not self.db_path.exists():
            raise FileNotFoundError(
                f"Database path does not exist: {self.db_path}. "
                f"Please ensure the Dolt database has been cloned using "
                f"'dolt clone post-no-preference/options'"
            )

        self._dolt = None
        self._is_connected = False
        self._schema_verified = False
        self._actual_tables = []
        self._actual_columns = {}

        logger.info(f"DoltAdapter initialized with database path: {self.db_path}")

    def connect(self) -> 'DoltAdapter':
        """
        Establish a connection to the Dolt database.

        This method attempts to connect to the Dolt database using the doltpy library.
        It verifies the database is accessible and can execute queries.

        Returns:
            self: Returns self to allow method chaining.

        Raises:
            DoltConnectionError: If connection cannot be established.
            ImportError: If doltpy library is not installed.

        Example:
            >>> adapter = DoltAdapter('/path/to/db')
            >>> adapter.connect()
            >>> # Now ready to execute queries
        """
        try:
            # doltpy library has changed its structure over versions
            # Try different import paths for compatibility
            try:
                from doltpy.cli import Dolt
            except ImportError:
                try:
                    from doltpy.core import Dolt
                except ImportError:
                    from doltpy import Dolt

            self._dolt = Dolt(str(self.db_path))
            self._is_connected = True

            # Verify connection by running a simple query
            self._verify_connection()

            logger.info(f"Successfully connected to Dolt database at {self.db_path}")
            return self

        except ImportError as e:
            error_msg = (
                "doltpy library is not installed or has incompatible structure. "
                "Please install it using: pip install doltpy"
            )
            logger.error(error_msg)
            raise ImportError(error_msg) from e

        except Exception as e:
            self._is_connected = False
            error_msg = f"Failed to connect to Dolt database at {self.db_path}: {str(e)}"
            logger.error(error_msg)
            raise DoltConnectionError(error_msg) from e

    def _verify_connection(self) -> None:
        """
        Verify the database connection is working by executing a test query.

        This internal method runs SHOW TABLES to verify the connection is valid
        and also caches the available tables for later reference.

        Raises:
            DoltConnectionError: If the verification query fails.
        """
        try:
            result = self._execute_sql("SHOW TABLES;")
            if isinstance(result, pd.DataFrame) and len(result) > 0:
                self._actual_tables = result.iloc[:, 0].tolist()
                logger.debug(f"Available tables: {self._actual_tables}")
            else:
                self._actual_tables = []

        except Exception as e:
            raise DoltConnectionError(f"Connection verification failed: {str(e)}") from e

    def _execute_sql(self, sql: str) -> pd.DataFrame:
        """
        Execute a SQL query and return results as a pandas DataFrame.

        This internal method handles the conversion from doltpy's native
        output format to pandas DataFrame. The doltpy library uses JSON
        or CSV format outputs which need to be converted.

        Args:
            sql: SQL query string to execute.

        Returns:
            DataFrame containing the query results.

        Raises:
            DoltQueryError: If the query fails.
        """
        try:
            # doltpy.cli returns dict with 'rows' key when using json format
            result = self._dolt.sql(sql, result_format='json')

            if result is None:
                return pd.DataFrame()

            # Handle json format output: {'rows': [dict, dict, ...]}
            if isinstance(result, dict):
                rows = result.get('rows', [])
                if not rows:
                    return pd.DataFrame()
                return pd.DataFrame(rows)

            # Handle csv format output: list of dicts
            if isinstance(result, list):
                if not result:
                    return pd.DataFrame()
                return pd.DataFrame(result)

            # If somehow we get a DataFrame directly (older API)
            if isinstance(result, pd.DataFrame):
                return result

            logger.warning(f"Unexpected result type from sql(): {type(result)}")
            return pd.DataFrame()

        except Exception as e:
            raise DoltQueryError(f"SQL execution failed: {str(e)}") from e

    def _ensure_connected(self) -> None:
        """
        Ensure the adapter is connected before executing queries.

        Raises:
            DoltConnectionError: If not connected.
        """
        if not self._is_connected or self._dolt is None:
            raise DoltConnectionError(
                "Not connected to database. Call connect() first."
            )

    def get_tables(self) -> List[str]:
        """
        Get list of all tables in the database.

        Returns:
            List of table names available in the database.

        Raises:
            DoltConnectionError: If not connected.

        Example:
            >>> tables = adapter.get_tables()
            >>> print(tables)
            ['option_chain', 'underlying_prices']
        """
        self._ensure_connected()
        return self._actual_tables.copy()

    def get_table_schema(self, table_name: str) -> pd.DataFrame:
        """
        Get the schema (column definitions) for a specific table.

        Args:
            table_name: Name of the table to describe.

        Returns:
            DataFrame containing column information (name, type, nullable, etc.)

        Raises:
            DoltConnectionError: If not connected.
            DoltQueryError: If the query fails.

        Example:
            >>> schema = adapter.get_table_schema('option_chain')
            >>> print(schema.columns)
        """
        self._ensure_connected()

        try:
            result = self._execute_sql(f"DESCRIBE {table_name};")
            if table_name not in self._actual_columns:
                self._actual_columns[table_name] = result['Field'].tolist() if 'Field' in result.columns else []
            return result

        except DoltQueryError:
            raise
        except Exception as e:
            error_msg = f"Failed to get schema for table '{table_name}': {str(e)}"
            logger.error(error_msg)
            raise DoltQueryError(error_msg) from e

    def get_option_chain(
        self,
        underlying: str,
        date: datetime,
        min_dte: int = 0,
        max_dte: int = 60,
        strikes: Optional[List[float]] = None,
        option_type: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Retrieve option chain data for a specific underlying and date.

        This method queries the option_chain table to retrieve all options
        matching the specified criteria. The data includes pricing, Greeks,
        and other market data.

        Args:
            underlying: Ticker symbol of the underlying asset (e.g., 'SPY', 'SPX').
            date: Quote date for which to retrieve the option chain.
            min_dte: Minimum days to expiration (inclusive). Default 0.
            max_dte: Maximum days to expiration (inclusive). Default 60.
            strikes: Optional list of specific strikes to filter. If None, returns all.
            option_type: Optional filter for 'call' or 'put'. If None, returns both.

        Returns:
            DataFrame containing the option chain with columns for pricing,
            Greeks, and market data. Returns empty DataFrame if no data found.

        Raises:
            ValueError: If underlying is empty, date is None, or invalid option_type.
            DoltConnectionError: If not connected.
            DoltQueryError: If the query fails.

        Example:
            >>> chain = adapter.get_option_chain(
            ...     underlying='SPY',
            ...     date=datetime(2023, 1, 3),
            ...     min_dte=1,
            ...     max_dte=7,
            ...     option_type='call'
            ... )
            >>> print(f"Found {len(chain)} call options expiring in 1-7 days")
        """
        # Input validation
        if not underlying:
            raise ValueError("underlying symbol cannot be empty")
        if date is None:
            raise ValueError("date cannot be None")
        if option_type is not None and option_type.lower() not in ('call', 'put'):
            raise ValueError("option_type must be 'call', 'put', or None")
        if min_dte < 0:
            raise ValueError("min_dte cannot be negative")
        if max_dte < min_dte:
            raise ValueError("max_dte cannot be less than min_dte")

        self._ensure_connected()

        # Calculate expiration date range
        min_expiry = date + timedelta(days=min_dte)
        max_expiry = date + timedelta(days=max_dte)

        # Format dates for SQL
        date_str = date.strftime('%Y-%m-%d')
        min_expiry_str = min_expiry.strftime('%Y-%m-%d')
        max_expiry_str = max_expiry.strftime('%Y-%m-%d')

        # Build SQL query using actual database column names
        # Actual schema: date, act_symbol, expiration, strike, call_put, bid, ask, vol, ...
        sql = f"""
        SELECT *
        FROM option_chain
        WHERE act_symbol = '{underlying.upper()}'
          AND date = '{date_str}'
          AND expiration >= '{min_expiry_str}'
          AND expiration <= '{max_expiry_str}'
        """

        # Add strike filter if specified
        if strikes is not None and len(strikes) > 0:
            strikes_str = ', '.join(str(s) for s in strikes)
            sql += f"  AND strike IN ({strikes_str})\n"

        # Add option type filter if specified
        # Database uses 'call_put' column with values 'Call' or 'Put' (capitalized)
        if option_type is not None:
            sql += f"  AND call_put = '{option_type.capitalize()}'\n"

        sql += "ORDER BY expiration, strike, call_put;"

        logger.debug(f"Executing query: {sql}")

        try:
            result = self._execute_sql(sql)

            if result is None or len(result) == 0:
                logger.warning(
                    f"No option chain data found for {underlying} on {date_str} "
                    f"with DTE range [{min_dte}, {max_dte}]"
                )
                return pd.DataFrame()

            logger.info(
                f"Retrieved {len(result)} option contracts for {underlying} "
                f"on {date_str}"
            )

            # Convert numeric columns to proper types
            result = self._convert_option_chain_types(result)

            return result

        except DoltQueryError:
            raise
        except Exception as e:
            error_msg = (
                f"Failed to get option chain for {underlying} on {date_str}: {str(e)}"
            )
            logger.error(error_msg)
            raise DoltQueryError(error_msg) from e

    def _convert_option_chain_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert option chain columns to appropriate data types.

        The Dolt JSON output returns decimal columns as strings.
        This method converts them to proper numeric types.

        Args:
            df: DataFrame with option chain data.

        Returns:
            DataFrame with properly typed columns.
        """
        if df.empty:
            return df

        # Numeric columns that should be float
        numeric_cols = ['bid', 'ask', 'strike', 'vol', 'delta', 'gamma', 'theta', 'vega', 'rho']

        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Date columns
        date_cols = ['date', 'expiration']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

        return df

    def get_historical_options(
        self,
        underlying: str,
        start: datetime,
        end: datetime,
        strikes: Optional[List[float]] = None,
        expirations: Optional[List[datetime]] = None
    ) -> pd.DataFrame:
        """
        Retrieve historical options data over a date range.

        This method retrieves time series of option quotes for specified
        strikes/expirations over a historical period. Useful for analyzing
        option price behavior over time.

        Args:
            underlying: Ticker symbol of the underlying asset.
            start: Start date of the historical period (inclusive).
            end: End date of the historical period (inclusive).
            strikes: Optional list of specific strikes to filter.
            expirations: Optional list of specific expiration dates to filter.

        Returns:
            DataFrame containing historical option quotes sorted by date and strike.
            Returns empty DataFrame if no data found.

        Raises:
            ValueError: If underlying is empty, dates are None, or start > end.
            DoltConnectionError: If not connected.
            DoltQueryError: If the query fails.

        Example:
            >>> history = adapter.get_historical_options(
            ...     underlying='SPY',
            ...     start=datetime(2023, 1, 1),
            ...     end=datetime(2023, 1, 31),
            ...     strikes=[400, 405, 410],
            ...     expirations=[datetime(2023, 2, 17)]
            ... )
        """
        # Input validation
        if not underlying:
            raise ValueError("underlying symbol cannot be empty")
        if start is None or end is None:
            raise ValueError("start and end dates cannot be None")
        if start > end:
            raise ValueError("start date cannot be after end date")

        self._ensure_connected()

        # Format dates
        start_str = start.strftime('%Y-%m-%d')
        end_str = end.strftime('%Y-%m-%d')

        # Build SQL query using actual database column names
        sql = f"""
        SELECT *
        FROM option_chain
        WHERE act_symbol = '{underlying.upper()}'
          AND date >= '{start_str}'
          AND date <= '{end_str}'
        """

        # Add strike filter
        if strikes is not None and len(strikes) > 0:
            strikes_str = ', '.join(str(s) for s in strikes)
            sql += f"  AND strike IN ({strikes_str})\n"

        # Add expiration filter
        if expirations is not None and len(expirations) > 0:
            exp_strs = ["'" + exp.strftime('%Y-%m-%d') + "'" for exp in expirations]
            exp_list = ', '.join(exp_strs)
            sql += f"  AND expiration IN ({exp_list})\n"

        sql += "ORDER BY date, expiration, strike, call_put;"

        logger.debug(f"Executing historical query: {sql}")

        try:
            result = self._execute_sql(sql)

            if result is None or len(result) == 0:
                logger.warning(
                    f"No historical options data found for {underlying} "
                    f"from {start_str} to {end_str}"
                )
                return pd.DataFrame()

            logger.info(
                f"Retrieved {len(result)} historical option records for {underlying}"
            )

            # Convert numeric columns to proper types
            result = self._convert_option_chain_types(result)

            return result

        except DoltQueryError:
            raise
        except Exception as e:
            error_msg = (
                f"Failed to get historical options for {underlying}: {str(e)}"
            )
            logger.error(error_msg)
            raise DoltQueryError(error_msg) from e

    def get_underlying_prices(
        self,
        symbol: str,
        start: datetime,
        end: datetime
    ) -> pd.DataFrame:
        """
        Retrieve historical underlying asset data.

        Note: The post-no-preference/options database does not have an
        underlying_prices table with OHLCV data. Instead, this method queries
        the volatility_history table which contains historical and implied
        volatility data but not price data.

        For actual price data (OHLC), consider using an external data source
        or estimating from option chain data (e.g., using ATM option strikes).

        Args:
            symbol: Ticker symbol (e.g., 'SPY').
            start: Start date of the history (inclusive).
            end: End date of the history (inclusive).

        Returns:
            DataFrame containing volatility data sorted by date from the
            volatility_history table. Returns empty DataFrame if no data found.

            Columns include: date, act_symbol, hv_current, iv_current, etc.

        Raises:
            ValueError: If symbol is empty, dates are None, or start > end.
            DoltConnectionError: If not connected.
            DoltQueryError: If the query fails.

        Example:
            >>> vol_history = adapter.get_underlying_prices(
            ...     symbol='SPY',
            ...     start=datetime(2023, 1, 1),
            ...     end=datetime(2023, 12, 31)
            ... )
            >>> print(f"Retrieved {len(vol_history)} volatility records")
        """
        # Input validation
        if not symbol:
            raise ValueError("symbol cannot be empty")
        if start is None or end is None:
            raise ValueError("start and end dates cannot be None")
        if start > end:
            raise ValueError("start date cannot be after end date")

        self._ensure_connected()

        # Format dates
        start_str = start.strftime('%Y-%m-%d')
        end_str = end.strftime('%Y-%m-%d')

        # Build SQL query for volatility_history table
        # The database does not have an underlying_prices table
        sql = f"""
        SELECT *
        FROM volatility_history
        WHERE act_symbol = '{symbol.upper()}'
          AND date >= '{start_str}'
          AND date <= '{end_str}'
        ORDER BY date;
        """

        logger.debug(f"Executing volatility history query: {sql}")

        try:
            result = self._execute_sql(sql)

            if result is None or len(result) == 0:
                logger.warning(
                    f"No volatility history found for {symbol} from {start_str} to {end_str}"
                )
                return pd.DataFrame()

            logger.info(f"Retrieved {len(result)} volatility history records for {symbol}")
            return result

        except DoltQueryError:
            raise
        except Exception as e:
            error_msg = f"Failed to get volatility history for {symbol}: {str(e)}"
            logger.error(error_msg)
            raise DoltQueryError(error_msg) from e

    def get_implied_volatility(
        self,
        symbol: str,
        date: datetime,
        strike: Optional[float] = None,
        option_type: str = 'call',
        dte_range: tuple = (20, 40)
    ) -> float:
        """
        Get implied volatility for a symbol on a specific date.

        This method retrieves IV data from the option chain. If no specific
        strike is provided, it returns the ATM IV (strike closest to spot).
        The IV is typically averaged across a DTE range to reduce noise.

        Args:
            symbol: Ticker symbol of the underlying.
            date: Date for which to retrieve IV.
            strike: Optional specific strike. If None, uses ATM strike.
            option_type: 'call' or 'put'. Default 'call'.
            dte_range: Tuple of (min_dte, max_dte) for expiration range.
                      Default (20, 40) targets ~30 DTE options.

        Returns:
            Implied volatility as a decimal (e.g., 0.25 for 25% IV).
            Returns NaN if no valid IV data found.

        Raises:
            ValueError: If symbol is empty or date is None.
            DoltConnectionError: If not connected.
            DoltQueryError: If the query fails.

        Example:
            >>> iv = adapter.get_implied_volatility(
            ...     symbol='SPY',
            ...     date=datetime(2023, 1, 3)
            ... )
            >>> print(f"30-day ATM IV: {iv:.2%}")
        """
        # Input validation
        if not symbol:
            raise ValueError("symbol cannot be empty")
        if date is None:
            raise ValueError("date cannot be None")

        self._ensure_connected()

        min_dte, max_dte = dte_range

        # Get option chain for the date
        chain = self.get_option_chain(
            underlying=symbol,
            date=date,
            min_dte=min_dte,
            max_dte=max_dte,
            option_type=option_type
        )

        if chain.empty:
            logger.warning(f"No options data available to calculate IV for {symbol} on {date}")
            return np.nan

        # Filter out invalid IVs
        iv_col = self.OPTION_CHAIN_COLUMNS.get('implied_volatility', 'implied_volatility')
        if iv_col not in chain.columns:
            logger.warning(f"implied_volatility column not found in data")
            return np.nan

        valid_iv = chain[chain[iv_col] > 0].copy()

        if valid_iv.empty:
            logger.warning(f"No valid IV data for {symbol} on {date}")
            return np.nan

        if strike is not None:
            # Use specific strike
            strike_col = self.OPTION_CHAIN_COLUMNS.get('strike', 'strike')
            strike_data = valid_iv[valid_iv[strike_col] == strike]

            if strike_data.empty:
                logger.warning(f"No IV data for strike {strike}")
                return np.nan

            return strike_data[iv_col].mean()
        else:
            # Use ATM strike (estimate spot price from option chain mid-point)
            strike_col = self.OPTION_CHAIN_COLUMNS.get('strike', 'strike')

            # Find the strike closest to the middle of all strikes as ATM proxy
            all_strikes = valid_iv[strike_col].unique()
            mid_strike = np.median(all_strikes)
            atm_strike = all_strikes[np.argmin(np.abs(all_strikes - mid_strike))]

            atm_data = valid_iv[valid_iv[strike_col] == atm_strike]

            if atm_data.empty:
                # Fall back to average of all IVs
                return valid_iv[iv_col].mean()

            return atm_data[iv_col].mean()

    def get_vix_history(
        self,
        start: datetime,
        end: datetime
    ) -> pd.DataFrame:
        """
        Retrieve VIX historical data.

        This method attempts to get VIX data from the underlying_prices table.
        VIX is commonly used for volatility filtering in options strategies.

        Args:
            start: Start date of the history (inclusive).
            end: End date of the history (inclusive).

        Returns:
            DataFrame with VIX price history. Returns empty DataFrame if
            VIX data is not available in the database.

        Raises:
            DoltConnectionError: If not connected.
            DoltQueryError: If the query fails.

        Note:
            VIX may not be available in all options databases. If VIX data
            is needed but not available, consider using a separate data source.
        """
        return self.get_underlying_prices(symbol='VIX', start=start, end=end)

    def query_custom(self, sql: str) -> pd.DataFrame:
        """
        Execute a custom SQL query against the database.

        This method allows executing arbitrary SQL queries for data exploration,
        custom aggregations, or accessing tables/columns not covered by the
        standard methods.

        Args:
            sql: SQL query string to execute.

        Returns:
            DataFrame containing the query results.

        Raises:
            ValueError: If sql is empty.
            DoltConnectionError: If not connected.
            DoltQueryError: If the query fails.

        Warning:
            Use caution with custom queries. Ensure proper SQL escaping to
            prevent injection attacks if using user-provided input.

        Example:
            >>> result = adapter.query_custom('''
            ...     SELECT underlying_symbol, COUNT(*) as contract_count
            ...     FROM option_chain
            ...     WHERE quote_date = '2023-01-03'
            ...     GROUP BY underlying_symbol
            ...     ORDER BY contract_count DESC
            ... ''')
        """
        if not sql or not sql.strip():
            raise ValueError("SQL query cannot be empty")

        self._ensure_connected()

        logger.debug(f"Executing custom query: {sql[:100]}...")

        try:
            result = self._execute_sql(sql)

            if result is None:
                return pd.DataFrame()

            return result

        except DoltQueryError:
            raise
        except Exception as e:
            error_msg = f"Custom query failed: {str(e)}"
            logger.error(error_msg)
            raise DoltQueryError(error_msg) from e

    def get_available_symbols(self, date: Optional[datetime] = None) -> List[str]:
        """
        Get list of available underlying symbols in the database.

        Args:
            date: Optional date to filter symbols available on that day.
                 If None, returns all symbols ever in the database.

        Returns:
            List of ticker symbols available.

        Raises:
            DoltConnectionError: If not connected.
            DoltQueryError: If the query fails.
        """
        self._ensure_connected()

        if date is not None:
            date_str = date.strftime('%Y-%m-%d')
            sql = f"""
            SELECT DISTINCT act_symbol
            FROM option_chain
            WHERE date = '{date_str}'
            ORDER BY act_symbol;
            """
        else:
            sql = """
            SELECT DISTINCT act_symbol
            FROM option_chain
            ORDER BY act_symbol;
            """

        try:
            result = self._execute_sql(sql)

            if result is None or result.empty:
                return []

            return result['act_symbol'].tolist()

        except DoltQueryError:
            raise
        except Exception as e:
            logger.error(f"Failed to get available symbols: {str(e)}")
            raise DoltQueryError(f"Failed to get available symbols: {str(e)}") from e

    def get_date_range(self, symbol: Optional[str] = None) -> tuple:
        """
        Get the date range of available data in the database.

        Args:
            symbol: Optional symbol to filter. If None, returns overall range.

        Returns:
            Tuple of (min_date, max_date) as datetime objects.
            Returns (None, None) if no data found.

        Raises:
            DoltConnectionError: If not connected.
            DoltQueryError: If the query fails.
        """
        self._ensure_connected()

        if symbol is not None:
            sql = f"""
            SELECT MIN(date) as min_date, MAX(date) as max_date
            FROM option_chain
            WHERE act_symbol = '{symbol.upper()}';
            """
        else:
            sql = """
            SELECT MIN(date) as min_date, MAX(date) as max_date
            FROM option_chain;
            """

        try:
            result = self._execute_sql(sql)

            if result is None or result.empty:
                return (None, None)

            min_date = result['min_date'].iloc[0]
            max_date = result['max_date'].iloc[0]

            # Convert to datetime if necessary
            if isinstance(min_date, str):
                min_date = datetime.strptime(min_date, '%Y-%m-%d')
            if isinstance(max_date, str):
                max_date = datetime.strptime(max_date, '%Y-%m-%d')

            return (min_date, max_date)

        except DoltQueryError:
            raise
        except Exception as e:
            logger.error(f"Failed to get date range: {str(e)}")
            raise DoltQueryError(f"Failed to get date range: {str(e)}") from e

    @contextmanager
    def transaction(self):
        """
        Context manager for executing queries within a transaction.

        This is primarily useful for read consistency when executing
        multiple related queries.

        Yields:
            Self for query execution within the transaction context.

        Example:
            >>> with adapter.transaction():
            ...     chain = adapter.get_option_chain(...)
            ...     prices = adapter.get_underlying_prices(...)

        Note:
            Dolt SQL Server supports standard SQL transactions. This context
            manager provides a convenient interface for transaction management.
        """
        self._ensure_connected()

        try:
            yield self
        except Exception as e:
            logger.error(f"Transaction error: {str(e)}")
            raise

    def close(self) -> None:
        """
        Close the database connection and release resources.

        This method should be called when done using the adapter to properly
        release database resources. After calling close(), the adapter cannot
        be used until connect() is called again.

        Example:
            >>> adapter = DoltAdapter('/path/to/db')
            >>> adapter.connect()
            >>> # ... use adapter ...
            >>> adapter.close()
        """
        if self._dolt is not None:
            # doltpy doesn't require explicit close, but we reset state
            self._dolt = None

        self._is_connected = False
        self._schema_verified = False
        logger.info("DoltAdapter connection closed")

    @property
    def is_connected(self) -> bool:
        """
        Check if the adapter is currently connected.

        Returns:
            True if connected, False otherwise.
        """
        return self._is_connected

    def __enter__(self) -> 'DoltAdapter':
        """
        Context manager entry - connects to database.

        Example:
            >>> with DoltAdapter('/path/to/db') as adapter:
            ...     chain = adapter.get_option_chain(...)
        """
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Context manager exit - closes connection.
        """
        self.close()

    def __repr__(self) -> str:
        """Return string representation of the adapter."""
        status = "connected" if self._is_connected else "disconnected"
        return f"DoltAdapter(db_path='{self.db_path}', status={status})"
