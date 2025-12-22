"""
Data Layer Module for Options Backtesting

This module provides the data infrastructure for the options backtesting system,
including database connectivity, data loading, and validation utilities.

Components:
    - DoltAdapter: Low-level database adapter for Dolt options database
    - MarketDataLoader: High-level interface for loading market data
    - DataValidator: Data quality validation utilities

The data layer follows a layered architecture:
    1. DoltAdapter - Direct database access
    2. MarketDataLoader - Business logic and caching
    3. DataValidator - Quality assurance

Usage:
    from backtester.data import DoltAdapter, MarketDataLoader, DataValidator

    # Connect to database
    adapter = DoltAdapter('/path/to/dolt_data/options')
    adapter.connect()

    # Load data through high-level interface
    loader = MarketDataLoader(adapter)
    prices = loader.load_underlying_prices('SPY', start_date, end_date)

    # Validate data quality
    is_valid, errors = DataValidator.validate_option_data(chain_df)

    # Clean data
    clean_df = DataValidator.filter_bad_quotes(chain_df)

    # Close connection when done
    adapter.close()

Database Requirements:
    - Dolt database cloned from: post-no-preference/options
    - Local path should point to the cloned repository directory

Exceptions:
    - DoltConnectionError: Database connection failures
    - DoltQueryError: Query execution failures
    - DataLoadError: High-level data loading failures
    - InsufficientDataError: Not enough data for calculations
    - ValidationError: Critical validation failures

Example Workflow:
    >>> from backtester.data import DoltAdapter, MarketDataLoader, DataValidator
    >>> from datetime import datetime
    >>>
    >>> # Setup
    >>> adapter = DoltAdapter('/Users/janussuk/Desktop/OptionsBacktester2/dolt_data/options')
    >>> adapter.connect()
    >>> loader = MarketDataLoader(adapter)
    >>>
    >>> # Load option chain
    >>> chain = loader.load_option_chain(
    ...     underlying='SPY',
    ...     date=datetime(2023, 6, 15),
    ...     dte_range=(7, 30)
    ... )
    >>>
    >>> # Validate
    >>> is_valid, errors = DataValidator.validate_option_data(chain)
    >>> if not is_valid:
    ...     chain = DataValidator.filter_bad_quotes(chain)
    >>>
    >>> # Calculate IV percentile for strategy filtering
    >>> iv_pct = loader.calculate_iv_percentile('SPY', datetime(2023, 6, 15))
    >>> print(f"IV Percentile: {iv_pct:.1f}%")
    >>>
    >>> # Cleanup
    >>> adapter.close()
"""

# Import main classes
from backtester.data.dolt_adapter import (
    DoltAdapter,
    DoltConnectionError,
    DoltQueryError,
)

from backtester.data.market_data import (
    MarketDataLoader,
    DataLoadError,
    InsufficientDataError,
)

from backtester.data.data_validator import (
    DataValidator,
    ValidationError,
)

# Define public API
__all__ = [
    # Main classes
    'DoltAdapter',
    'MarketDataLoader',
    'DataValidator',

    # Exceptions
    'DoltConnectionError',
    'DoltQueryError',
    'DataLoadError',
    'InsufficientDataError',
    'ValidationError',
]

# Module metadata
__version__ = '0.1.0'
__author__ = 'Options Backtester Team'
