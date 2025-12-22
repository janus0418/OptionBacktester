"""
Data Validation Utilities for Options Backtesting

This module provides validation and data quality checking utilities for options
and underlying market data. Proper data validation is critical for backtesting
accuracy, as bad data can lead to unrealistic P&L calculations and misleading
strategy performance metrics.

Common data quality issues in options data:
- Zero bid/ask prices (illiquid options)
- Negative prices (data errors)
- Extreme implied volatilities (calculation errors or illiquid options)
- Missing Greek values
- Date gaps (holidays are expected, mid-week gaps indicate issues)
- Inverted quotes (bid > ask)
- Stale data (no price movement over multiple days)

Usage:
    from backtester.data.data_validator import DataValidator

    validator = DataValidator()
    is_valid, errors = validator.validate_option_data(option_chain_df)

    if not is_valid:
        print(f"Data quality issues: {errors}")

    # Filter bad quotes
    clean_data = validator.filter_bad_quotes(option_chain_df)

References:
    - Options data quality: https://quantlib.wordpress.com/2017/04/17/option-data-quality/
    - Market microstructure: Market data has inherent noise and errors
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Set
import warnings

import numpy as np
import pandas as pd

# Configure module logger
logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Exception raised when data validation fails critically."""
    pass


class DataValidator:
    """
    Validation utilities for options and market data quality checking.

    This class provides static methods for validating and cleaning options
    market data. It can identify common data quality issues and filter out
    problematic records that could distort backtesting results.

    Validation Categories:
        1. Price Validation: Check for zero, negative, or inverted prices
        2. IV Validation: Check for extreme or missing implied volatilities
        3. Greeks Validation: Check for missing or invalid Greeks values
        4. Date Validation: Check for gaps and consistency in time series
        5. Liquidity Validation: Check for minimum volume/open interest

    Attributes:
        MIN_IV (float): Minimum valid implied volatility (default 0.01 = 1%)
        MAX_IV (float): Maximum valid implied volatility (default 5.0 = 500%)
        MIN_PRICE (float): Minimum valid option price (default 0.01)
        MAX_SPREAD_PCT (float): Maximum bid-ask spread as % of mid (default 0.50)

    Example:
        >>> validator = DataValidator()
        >>> is_valid, errors = DataValidator.validate_option_data(df)
        >>> if not is_valid:
        ...     for error in errors:
        ...         print(f"Issue: {error}")
        >>> clean_df = DataValidator.filter_bad_quotes(df)
    """

    # Default validation thresholds
    # These can be adjusted based on specific data source characteristics
    MIN_IV: float = 0.01    # 1% - below this is likely an error
    MAX_IV: float = 5.0     # 500% - above this is likely an error or illiquid
    MIN_PRICE: float = 0.01 # Minimum valid option price
    MAX_SPREAD_PCT: float = 0.50  # 50% max spread as percentage of mid
    MIN_DELTA_ABS: float = 0.0001  # Very deep OTM options have near-zero delta
    MAX_DELTA_ABS: float = 1.0001  # Allow small numerical errors above 1.0

    # Required columns for full validation
    REQUIRED_OPTION_COLUMNS = {
        'strike', 'expiration', 'option_type'
    }

    PRICE_COLUMNS = {
        'bid', 'ask', 'mid'
    }

    GREEK_COLUMNS = {
        'delta', 'gamma', 'theta', 'vega', 'rho'
    }

    @staticmethod
    def validate_option_data(
        df: pd.DataFrame,
        strict: bool = False
    ) -> Tuple[bool, List[str]]:
        """
        Validate option chain data for quality issues.

        This method performs comprehensive validation of option chain data,
        checking for common data quality issues that could affect backtesting
        accuracy.

        Validation checks performed:
            1. Required columns exist
            2. No null values in critical columns
            3. Strike prices are positive
            4. Option types are valid ('call' or 'put')
            5. Bid/Ask prices are non-negative
            6. Bid <= Ask (no inverted quotes)
            7. Implied volatility within reasonable bounds
            8. Greeks within valid ranges

        Args:
            df: DataFrame containing option chain data.
            strict: If True, treats warnings as errors. Default False.

        Returns:
            Tuple of (is_valid, error_messages):
                - is_valid: True if data passes validation (or only has warnings)
                - error_messages: List of validation error/warning descriptions

        Raises:
            ValueError: If df is None.

        Example:
            >>> is_valid, errors = DataValidator.validate_option_data(chain_df)
            >>> if not is_valid:
            ...     print("Data quality issues found:")
            ...     for error in errors:
            ...         print(f"  - {error}")
        """
        if df is None:
            raise ValueError("DataFrame cannot be None")

        errors: List[str] = []
        warnings_list: List[str] = []

        # Empty DataFrame check
        if df.empty:
            errors.append("DataFrame is empty")
            return (False, errors)

        # Standardize column names for checking
        df_lower = df.copy()
        df_lower.columns = df_lower.columns.str.lower()

        # Check required columns
        missing_required = DataValidator.REQUIRED_OPTION_COLUMNS - set(df_lower.columns)
        if missing_required:
            errors.append(f"Missing required columns: {missing_required}")

        # Check for strike validity
        if 'strike' in df_lower.columns:
            invalid_strikes = df_lower['strike'] <= 0
            if invalid_strikes.any():
                count = invalid_strikes.sum()
                errors.append(f"Found {count} rows with non-positive strike prices")

            null_strikes = df_lower['strike'].isnull()
            if null_strikes.any():
                errors.append(f"Found {null_strikes.sum()} rows with null strikes")

        # Check option_type validity
        if 'option_type' in df_lower.columns:
            valid_types = {'call', 'put', 'c', 'p'}
            invalid_types = ~df_lower['option_type'].str.lower().isin(valid_types)
            if invalid_types.any():
                count = invalid_types.sum()
                errors.append(
                    f"Found {count} rows with invalid option_type "
                    f"(must be 'call' or 'put')"
                )

        # Check bid/ask validity
        for price_col in ['bid', 'ask']:
            if price_col in df_lower.columns:
                # Negative prices
                negative = df_lower[price_col] < 0
                if negative.any():
                    errors.append(
                        f"Found {negative.sum()} rows with negative {price_col} prices"
                    )

        # Check for inverted quotes (bid > ask)
        if 'bid' in df_lower.columns and 'ask' in df_lower.columns:
            inverted = df_lower['bid'] > df_lower['ask']
            if inverted.any():
                count = inverted.sum()
                warnings_list.append(
                    f"Found {count} rows with inverted quotes (bid > ask)"
                )

        # Check bid-ask spread
        if 'bid' in df_lower.columns and 'ask' in df_lower.columns:
            mid = (df_lower['bid'] + df_lower['ask']) / 2
            spread = df_lower['ask'] - df_lower['bid']
            # Avoid division by zero
            spread_pct = np.where(mid > 0, spread / mid, 0)
            wide_spread = spread_pct > DataValidator.MAX_SPREAD_PCT

            if wide_spread.any():
                count = wide_spread.sum()
                warnings_list.append(
                    f"Found {count} rows with wide bid-ask spread "
                    f"(>{DataValidator.MAX_SPREAD_PCT:.0%} of mid)"
                )

        # Check implied volatility
        if 'implied_volatility' in df_lower.columns:
            iv = df_lower['implied_volatility']

            # Zero IV
            zero_iv = iv == 0
            if zero_iv.any():
                warnings_list.append(f"Found {zero_iv.sum()} rows with zero IV")

            # Extreme low IV
            low_iv = (iv > 0) & (iv < DataValidator.MIN_IV)
            if low_iv.any():
                warnings_list.append(
                    f"Found {low_iv.sum()} rows with unusually low IV "
                    f"(<{DataValidator.MIN_IV:.0%})"
                )

            # Extreme high IV
            high_iv = iv > DataValidator.MAX_IV
            if high_iv.any():
                warnings_list.append(
                    f"Found {high_iv.sum()} rows with unusually high IV "
                    f"(>{DataValidator.MAX_IV:.0%})"
                )

            # Null IV
            null_iv = iv.isnull()
            if null_iv.any():
                warnings_list.append(f"Found {null_iv.sum()} rows with null IV")

        # Check delta validity
        if 'delta' in df_lower.columns:
            delta = df_lower['delta']

            # Delta out of bounds [-1, 1]
            invalid_delta = (delta.abs() > DataValidator.MAX_DELTA_ABS) & delta.notna()
            if invalid_delta.any():
                errors.append(
                    f"Found {invalid_delta.sum()} rows with delta outside [-1, 1]"
                )

        # Check gamma validity (should be non-negative)
        if 'gamma' in df_lower.columns:
            gamma = df_lower['gamma']
            negative_gamma = (gamma < 0) & gamma.notna()
            if negative_gamma.any():
                errors.append(
                    f"Found {negative_gamma.sum()} rows with negative gamma"
                )

        # Check expiration dates
        if 'expiration' in df_lower.columns:
            try:
                exp_dates = pd.to_datetime(df_lower['expiration'])
                null_exp = exp_dates.isnull()
                if null_exp.any():
                    errors.append(f"Found {null_exp.sum()} rows with null expiration")
            except Exception as e:
                warnings_list.append(f"Could not parse expiration dates: {str(e)}")

        # Combine errors and warnings
        all_issues = errors.copy()
        if strict:
            all_issues.extend(warnings_list)
        else:
            # Log warnings but don't fail validation
            for w in warnings_list:
                logger.warning(w)
                all_issues.append(f"WARNING: {w}")

        # Determine validity
        is_valid = len(errors) == 0

        return (is_valid, all_issues)

    @staticmethod
    def check_for_gaps(
        df: pd.DataFrame,
        date_column: str = 'quote_date',
        max_gap_days: int = 3,
        exclude_weekends: bool = True
    ) -> List[datetime]:
        """
        Check for unexpected gaps in time series data.

        This method identifies missing dates in the data that exceed the
        expected gap threshold. Gaps might indicate data quality issues,
        though some gaps are expected (weekends, holidays).

        Args:
            df: DataFrame with date column.
            date_column: Name of the date column. Default 'quote_date'.
            max_gap_days: Maximum allowed gap in calendar days. Default 3.
                         Gaps larger than this are flagged.
            exclude_weekends: If True, weekends don't count toward gap.
                            Default True.

        Returns:
            List of datetime objects representing the start of each gap.
            Empty list if no gaps exceed threshold.

        Raises:
            ValueError: If date_column not found in DataFrame.

        Note:
            Market holidays will appear as gaps. A gap of 3 calendar days
            is normal for a weekend, and 4 days for a weekend + holiday.
            Set max_gap_days accordingly.

        Example:
            >>> gaps = DataValidator.check_for_gaps(df, date_column='quote_date')
            >>> if gaps:
            ...     print(f"Found {len(gaps)} data gaps starting on:")
            ...     for gap_start in gaps:
            ...         print(f"  - {gap_start.date()}")
        """
        if df is None or df.empty:
            return []

        df_copy = df.copy()
        df_copy.columns = df_copy.columns.str.lower()
        date_column_lower = date_column.lower()

        if date_column_lower not in df_copy.columns:
            raise ValueError(f"Date column '{date_column}' not found in DataFrame")

        # Convert to datetime
        dates = pd.to_datetime(df_copy[date_column_lower])

        # Get unique sorted dates
        unique_dates = dates.drop_duplicates().sort_values()

        if len(unique_dates) < 2:
            return []

        gaps: List[datetime] = []

        # Check consecutive date pairs
        for i in range(len(unique_dates) - 1):
            current_date = unique_dates.iloc[i]
            next_date = unique_dates.iloc[i + 1]

            gap_days = (next_date - current_date).days

            if exclude_weekends:
                # Count business days in the gap
                business_days = np.busday_count(
                    current_date.date(),
                    next_date.date()
                )
                # A gap of 1 business day is normal (next trading day)
                # We flag if business day gap > 1 AND calendar gap > max_gap_days
                if business_days > 1 and gap_days > max_gap_days:
                    gaps.append(current_date.to_pydatetime())
            else:
                if gap_days > max_gap_days:
                    gaps.append(current_date.to_pydatetime())

        if gaps:
            logger.warning(f"Found {len(gaps)} gaps exceeding {max_gap_days} days")

        return gaps

    @staticmethod
    def filter_bad_quotes(
        df: pd.DataFrame,
        min_bid: float = 0.01,
        min_open_interest: int = 0,
        min_volume: int = 0,
        max_iv: float = 5.0,
        min_iv: float = 0.01,
        remove_inverted: bool = True,
        max_spread_pct: Optional[float] = None,
        inplace: bool = False
    ) -> pd.DataFrame:
        """
        Filter out bad or suspicious option quotes.

        This method removes records that fail quality checks and could
        distort backtesting results. It's designed to be conservative -
        removing questionable data rather than including potentially
        incorrect quotes.

        Filtering criteria (all applied if relevant columns exist):
            1. Bid price >= min_bid (removes zero bid options)
            2. Ask price > bid (removes inverted quotes if enabled)
            3. IV within [min_iv, max_iv] range
            4. Open interest >= min_open_interest
            5. Volume >= min_volume
            6. Bid-ask spread <= max_spread_pct of mid

        Args:
            df: DataFrame containing option data.
            min_bid: Minimum bid price to include. Default 0.01.
                    Set to 0 to include zero-bid options (illiquid).
            min_open_interest: Minimum open interest. Default 0.
            min_volume: Minimum trading volume. Default 0.
            max_iv: Maximum implied volatility (decimal). Default 5.0 (500%).
            min_iv: Minimum implied volatility (decimal). Default 0.01 (1%).
            remove_inverted: Remove quotes where bid > ask. Default True.
            max_spread_pct: Maximum bid-ask spread as % of mid. Default None.
                           Set to 0.5 to remove quotes with >50% spread.
            inplace: If True, modifies df in place. Default False.

        Returns:
            DataFrame with bad quotes removed.

        Note:
            The filtering is intentionally aggressive to ensure data quality.
            For illiquid options or wide-spread markets, you may need to
            relax some parameters.

        Example:
            >>> # Standard filtering for liquid equity options
            >>> clean_df = DataValidator.filter_bad_quotes(
            ...     df,
            ...     min_bid=0.05,
            ...     min_open_interest=100,
            ...     max_iv=3.0
            ... )
            >>> print(f"Removed {len(df) - len(clean_df)} bad quotes")

            >>> # Relaxed filtering for illiquid index options
            >>> clean_df = DataValidator.filter_bad_quotes(
            ...     df,
            ...     min_bid=0.01,
            ...     min_open_interest=0,
            ...     max_iv=5.0,
            ...     max_spread_pct=1.0
            ... )
        """
        if df is None or df.empty:
            return df if inplace else df.copy() if df is not None else pd.DataFrame()

        if not inplace:
            df = df.copy()

        # Standardize column names
        df.columns = df.columns.str.lower()

        initial_count = len(df)
        mask = pd.Series(True, index=df.index)

        # Filter by bid price
        if 'bid' in df.columns:
            bid_mask = df['bid'] >= min_bid
            removed = (~bid_mask).sum()
            if removed > 0:
                logger.debug(f"Removing {removed} rows with bid < {min_bid}")
            mask &= bid_mask

        # Filter inverted quotes
        if remove_inverted and 'bid' in df.columns and 'ask' in df.columns:
            inverted_mask = df['bid'] <= df['ask']
            removed = (~inverted_mask).sum()
            if removed > 0:
                logger.debug(f"Removing {removed} inverted quotes (bid > ask)")
            mask &= inverted_mask

        # Filter by spread percentage
        if max_spread_pct is not None and 'bid' in df.columns and 'ask' in df.columns:
            mid = (df['bid'] + df['ask']) / 2
            spread_pct = np.where(mid > 0, (df['ask'] - df['bid']) / mid, np.inf)
            spread_mask = spread_pct <= max_spread_pct
            removed = (~spread_mask).sum()
            if removed > 0:
                logger.debug(
                    f"Removing {removed} rows with spread > {max_spread_pct:.0%}"
                )
            mask &= spread_mask

        # Filter by IV
        if 'implied_volatility' in df.columns:
            iv = df['implied_volatility']
            iv_mask = (iv >= min_iv) & (iv <= max_iv) | iv.isnull()
            removed = (~iv_mask).sum()
            if removed > 0:
                logger.debug(
                    f"Removing {removed} rows with IV outside [{min_iv}, {max_iv}]"
                )
            mask &= iv_mask

        # Filter by open interest
        if min_open_interest > 0 and 'open_interest' in df.columns:
            oi_mask = df['open_interest'] >= min_open_interest
            removed = (~oi_mask).sum()
            if removed > 0:
                logger.debug(
                    f"Removing {removed} rows with OI < {min_open_interest}"
                )
            mask &= oi_mask

        # Filter by volume
        if min_volume > 0 and 'volume' in df.columns:
            vol_mask = df['volume'] >= min_volume
            removed = (~vol_mask).sum()
            if removed > 0:
                logger.debug(f"Removing {removed} rows with volume < {min_volume}")
            mask &= vol_mask

        # Apply filter
        result = df[mask]

        final_count = len(result)
        removed_total = initial_count - final_count

        if removed_total > 0:
            logger.info(
                f"Filtered {removed_total} bad quotes "
                f"({removed_total/initial_count:.1%} of data)"
            )

        return result

    @staticmethod
    def validate_price_data(
        df: pd.DataFrame,
        price_columns: Optional[List[str]] = None
    ) -> Tuple[bool, List[str]]:
        """
        Validate underlying price data (OHLCV).

        Checks for common issues in price data:
            - Missing values
            - Negative prices
            - High > Close > Low relationship violations
            - Zero or negative volume

        Args:
            df: DataFrame with OHLCV price data.
            price_columns: List of price columns to validate.
                          Default ['open', 'high', 'low', 'close'].

        Returns:
            Tuple of (is_valid, error_messages).

        Example:
            >>> is_valid, errors = DataValidator.validate_price_data(prices_df)
        """
        if df is None:
            raise ValueError("DataFrame cannot be None")

        if df.empty:
            return (False, ["DataFrame is empty"])

        errors: List[str] = []
        df_copy = df.copy()
        df_copy.columns = df_copy.columns.str.lower()

        if price_columns is None:
            price_columns = ['open', 'high', 'low', 'close']

        price_columns = [c.lower() for c in price_columns]

        # Check for required columns
        available_cols = [c for c in price_columns if c in df_copy.columns]
        if not available_cols:
            errors.append(f"No price columns found: {price_columns}")
            return (False, errors)

        # Check for negative prices
        for col in available_cols:
            negative = df_copy[col] < 0
            if negative.any():
                errors.append(f"Found {negative.sum()} negative values in '{col}'")

        # Check for null values
        for col in available_cols:
            nulls = df_copy[col].isnull()
            if nulls.any():
                errors.append(f"Found {nulls.sum()} null values in '{col}'")

        # Check OHLC relationship
        if all(c in df_copy.columns for c in ['open', 'high', 'low', 'close']):
            # High should be >= Open, Close
            high_violations = (
                (df_copy['high'] < df_copy['open']) |
                (df_copy['high'] < df_copy['close'])
            )
            if high_violations.any():
                errors.append(
                    f"Found {high_violations.sum()} rows where high < open or close"
                )

            # Low should be <= Open, Close
            low_violations = (
                (df_copy['low'] > df_copy['open']) |
                (df_copy['low'] > df_copy['close'])
            )
            if low_violations.any():
                errors.append(
                    f"Found {low_violations.sum()} rows where low > open or close"
                )

        # Check volume
        if 'volume' in df_copy.columns:
            neg_volume = df_copy['volume'] < 0
            if neg_volume.any():
                errors.append(f"Found {neg_volume.sum()} negative volume values")

        is_valid = len(errors) == 0
        return (is_valid, errors)

    @staticmethod
    def get_data_quality_report(
        df: pd.DataFrame,
        data_type: str = 'option'
    ) -> Dict[str, Union[int, float, str, List]]:
        """
        Generate a comprehensive data quality report.

        This method provides detailed statistics about data quality,
        useful for understanding the characteristics of your data source.

        Args:
            df: DataFrame to analyze.
            data_type: Type of data - 'option' or 'price'. Default 'option'.

        Returns:
            Dictionary containing quality metrics:
                - total_rows: Total number of records
                - null_counts: Dictionary of null counts per column
                - unique_symbols: Number of unique symbols
                - date_range: (min_date, max_date) tuple
                - issues_found: List of quality issues
                - quality_score: Overall quality score (0-100)

        Example:
            >>> report = DataValidator.get_data_quality_report(chain_df, 'option')
            >>> print(f"Quality Score: {report['quality_score']:.0f}/100")
            >>> print(f"Total Issues: {len(report['issues_found'])}")
        """
        if df is None or df.empty:
            return {
                'total_rows': 0,
                'null_counts': {},
                'unique_symbols': 0,
                'date_range': (None, None),
                'issues_found': ['DataFrame is empty'],
                'quality_score': 0
            }

        df_copy = df.copy()
        df_copy.columns = df_copy.columns.str.lower()

        report: Dict[str, Union[int, float, str, List]] = {}

        # Basic stats
        report['total_rows'] = len(df_copy)
        report['total_columns'] = len(df_copy.columns)

        # Null counts
        null_counts = df_copy.isnull().sum().to_dict()
        report['null_counts'] = {k: v for k, v in null_counts.items() if v > 0}

        # Unique symbols
        symbol_col = 'underlying_symbol' if 'underlying_symbol' in df_copy.columns else 'symbol'
        if symbol_col in df_copy.columns:
            report['unique_symbols'] = df_copy[symbol_col].nunique()
        else:
            report['unique_symbols'] = 0

        # Date range
        date_col = 'quote_date' if 'quote_date' in df_copy.columns else 'date'
        if date_col in df_copy.columns:
            dates = pd.to_datetime(df_copy[date_col])
            report['date_range'] = (dates.min(), dates.max())
        else:
            report['date_range'] = (None, None)

        # Run validation
        if data_type == 'option':
            is_valid, issues = DataValidator.validate_option_data(df, strict=True)
        else:
            is_valid, issues = DataValidator.validate_price_data(df)

        report['issues_found'] = issues
        report['is_valid'] = is_valid

        # Calculate quality score (simple heuristic)
        score = 100

        # Deduct for null values
        total_cells = len(df_copy) * len(df_copy.columns)
        total_nulls = sum(report['null_counts'].values()) if report['null_counts'] else 0
        null_pct = total_nulls / total_cells if total_cells > 0 else 0
        score -= min(30, null_pct * 100)

        # Deduct for issues
        score -= min(40, len(issues) * 5)

        report['quality_score'] = max(0, score)

        return report

    @staticmethod
    def detect_stale_prices(
        df: pd.DataFrame,
        price_column: str = 'mid',
        date_column: str = 'quote_date',
        stale_threshold: int = 3
    ) -> pd.DataFrame:
        """
        Detect potentially stale option prices.

        Stale prices occur when an option shows no price movement over
        multiple days, possibly indicating illiquidity or data issues.

        Args:
            df: DataFrame with option data.
            price_column: Column containing prices to check.
            date_column: Column containing dates.
            stale_threshold: Number of consecutive days with same price
                           to flag as stale. Default 3.

        Returns:
            DataFrame containing only records flagged as stale.

        Example:
            >>> stale_options = DataValidator.detect_stale_prices(df)
            >>> if not stale_options.empty:
            ...     print(f"Found {len(stale_options)} potentially stale quotes")
        """
        if df is None or df.empty:
            return pd.DataFrame()

        df_copy = df.copy()
        df_copy.columns = df_copy.columns.str.lower()
        price_column = price_column.lower()
        date_column = date_column.lower()

        if price_column not in df_copy.columns:
            logger.warning(f"Price column '{price_column}' not found")
            return pd.DataFrame()

        if date_column not in df_copy.columns:
            logger.warning(f"Date column '{date_column}' not found")
            return pd.DataFrame()

        # Need a grouping key for individual options
        group_cols = []
        for col in ['underlying_symbol', 'strike', 'expiration', 'option_type']:
            if col in df_copy.columns:
                group_cols.append(col)

        if not group_cols:
            logger.warning("Cannot identify individual options for stale detection")
            return pd.DataFrame()

        stale_records = []

        for group_key, group_data in df_copy.groupby(group_cols):
            group_sorted = group_data.sort_values(date_column)

            if len(group_sorted) < stale_threshold:
                continue

            # Check for consecutive same prices
            prices = group_sorted[price_column].values
            consecutive_same = 1

            for i in range(1, len(prices)):
                if prices[i] == prices[i-1] and not pd.isna(prices[i]):
                    consecutive_same += 1
                    if consecutive_same >= stale_threshold:
                        stale_records.append(group_sorted.iloc[i])
                else:
                    consecutive_same = 1

        if stale_records:
            return pd.DataFrame(stale_records)
        return pd.DataFrame()

    @staticmethod
    def check_arbitrage_violations(
        df: pd.DataFrame,
        underlying_price: Optional[float] = None
    ) -> List[str]:
        """
        Check for obvious arbitrage violations in option prices.

        This detects option prices that violate basic no-arbitrage conditions:
            1. Call price >= max(0, S - K*exp(-rT))  (lower bound)
            2. Put price >= max(0, K*exp(-rT) - S)   (lower bound)
            3. Call price <= S (upper bound)
            4. Put price <= K*exp(-rT) (upper bound, approximated as K)
            5. Put-call parity (approximately)

        Args:
            df: DataFrame with option data.
            underlying_price: Current underlying price for bounds checking.
                            If None, attempts to estimate from data.

        Returns:
            List of violation descriptions.

        Note:
            These are theoretical bounds. In practice, small violations
            may occur due to bid-ask spreads, dividends, and other factors.
        """
        violations: List[str] = []

        if df is None or df.empty:
            return violations

        df_copy = df.copy()
        df_copy.columns = df_copy.columns.str.lower()

        # Need price columns
        price_col = 'mid' if 'mid' in df_copy.columns else None
        if price_col is None and 'bid' in df_copy.columns and 'ask' in df_copy.columns:
            df_copy['mid'] = (df_copy['bid'] + df_copy['ask']) / 2
            price_col = 'mid'

        if price_col is None:
            violations.append("No price column available for arbitrage check")
            return violations

        # Get underlying price if not provided
        if underlying_price is None:
            # Try to estimate from ATM options
            if 'strike' in df_copy.columns:
                underlying_price = df_copy['strike'].median()
            else:
                violations.append("Cannot determine underlying price for arbitrage check")
                return violations

        if 'strike' not in df_copy.columns or 'option_type' not in df_copy.columns:
            violations.append("Missing strike or option_type for arbitrage check")
            return violations

        # Check call upper bound: Call <= S
        calls = df_copy[df_copy['option_type'].str.lower() == 'call']
        call_violations = calls[calls[price_col] > underlying_price * 1.05]  # 5% tolerance
        if len(call_violations) > 0:
            violations.append(
                f"Found {len(call_violations)} calls priced above underlying "
                f"(arbitrage violation)"
            )

        # Check call lower bound: Call >= max(0, S - K)
        intrinsic_call = np.maximum(0, underlying_price - calls['strike'])
        call_below_intrinsic = calls[calls[price_col] < intrinsic_call * 0.95]
        if len(call_below_intrinsic) > 0:
            violations.append(
                f"Found {len(call_below_intrinsic)} calls priced below intrinsic value"
            )

        # Check put upper bound: Put <= K
        puts = df_copy[df_copy['option_type'].str.lower() == 'put']
        put_violations = puts[puts[price_col] > puts['strike'] * 1.05]
        if len(put_violations) > 0:
            violations.append(
                f"Found {len(put_violations)} puts priced above strike "
                f"(arbitrage violation)"
            )

        # Check put lower bound: Put >= max(0, K - S)
        intrinsic_put = np.maximum(0, puts['strike'] - underlying_price)
        put_below_intrinsic = puts[puts[price_col] < intrinsic_put * 0.95]
        if len(put_below_intrinsic) > 0:
            violations.append(
                f"Found {len(put_below_intrinsic)} puts priced below intrinsic value"
            )

        return violations

    def __repr__(self) -> str:
        """Return string representation."""
        return f"DataValidator(MIN_IV={self.MIN_IV}, MAX_IV={self.MAX_IV})"
