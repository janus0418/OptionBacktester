"""
PerformanceMetrics Class for Options Backtesting Analytics

This module provides the PerformanceMetrics class for calculating standard
quantitative finance performance metrics used in industry for evaluating
trading strategies and portfolios.

Key Features:
    - Returns-based metrics (total return, CAGR, Sharpe, Sortino, Calmar)
    - Drawdown analysis (max drawdown, duration, recovery)
    - Trade-based metrics (win rate, profit factor, expectancy)
    - Distribution statistics (skewness, kurtosis, percentiles)

Design Philosophy:
    All methods are static to enable easy use without instantiation.
    Methods accept pandas DataFrames/Series and return typed results.
    All formulas follow industry-standard implementations and are
    mathematically correct.

Mathematical Correctness:
    - Sharpe Ratio: (R_p - R_f) / sigma_p * sqrt(periods_per_year)
    - Sortino Ratio: (R_p - R_f) / sigma_downside * sqrt(periods_per_year)
    - Calmar Ratio: CAGR / |max_drawdown|
    - Max Drawdown: min((Equity(t) - Peak(t)) / Peak(t))
    - VaR: 5th percentile (historical) or mu - 1.645*sigma (parametric)

Usage:
    from backtester.analytics.metrics import PerformanceMetrics

    # Calculate from equity curve
    total_ret = PerformanceMetrics.calculate_total_return(equity_curve)
    sharpe = PerformanceMetrics.calculate_sharpe_ratio(returns)
    dd_info = PerformanceMetrics.calculate_max_drawdown(equity_curve)

References:
    - Sharpe, W.F. (1994). The Sharpe Ratio. Journal of Portfolio Management.
    - Sortino, F.A. (1994). Performance Measurement in a Downside Risk Framework.
    - Young, T.W. (1991). Calmar Ratio: A Smoother Tool.
    - Hull, J.C. (2018). Options, Futures, and Other Derivatives.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

# Configure module logger
logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Trading days per year (industry standard)
TRADING_DAYS_PER_YEAR = 252

# Numerical tolerance for calculations
EPSILON = 1e-10

# Minimum data points for reliable statistics
MIN_OBSERVATIONS_FOR_STATS = 2
MIN_OBSERVATIONS_FOR_SHARPE = 30  # At least 30 observations for meaningful Sharpe

# Default risk-free rate (annualized)
DEFAULT_RISK_FREE_RATE = 0.02


# =============================================================================
# Exceptions
# =============================================================================

class MetricsError(Exception):
    """Base exception for metrics calculation errors."""
    pass


class InsufficientDataError(MetricsError):
    """Exception raised when there is insufficient data for calculation."""
    pass


class InvalidDataError(MetricsError):
    """Exception raised when data is invalid for calculation."""
    pass


# =============================================================================
# PerformanceMetrics Class
# =============================================================================

class PerformanceMetrics:
    """
    Calculate portfolio performance metrics.

    This class provides static methods for calculating standard quantitative
    finance performance metrics used in industry. All methods are designed
    to work with pandas DataFrames and Series.

    Metrics Categories:
        1. Returns-based: Total return, CAGR, Sharpe, Sortino, Calmar
        2. Drawdown: Max drawdown, duration, recovery analysis
        3. Trade-based: Win rate, profit factor, expectancy
        4. Distribution: Mean, std, skewness, kurtosis, percentiles

    All methods perform input validation and handle edge cases appropriately.

    Example:
        >>> equity_curve = engine.generate_equity_curve()
        >>> returns = equity_curve['equity'].pct_change().dropna()
        >>>
        >>> total_ret = PerformanceMetrics.calculate_total_return(equity_curve)
        >>> sharpe = PerformanceMetrics.calculate_sharpe_ratio(returns)
        >>> dd = PerformanceMetrics.calculate_max_drawdown(equity_curve)
    """

    # =========================================================================
    # Returns-Based Metrics
    # =========================================================================

    @staticmethod
    def calculate_total_return(equity_curve: pd.DataFrame) -> float:
        """
        Calculate total return as percentage.

        Formula:
            Total Return = (Final Equity - Initial Equity) / Initial Equity * 100

        Args:
            equity_curve: DataFrame with 'equity' column or Series of equity values.
                         Index should be datetime.

        Returns:
            Total return as percentage (e.g., 25.0 for 25% return)

        Raises:
            InsufficientDataError: If equity_curve has fewer than 2 data points
            InvalidDataError: If equity_curve is empty or has invalid values
            KeyError: If 'equity' column not found when DataFrame provided

        Example:
            >>> total_ret = PerformanceMetrics.calculate_total_return(equity_curve)
            >>> print(f"Total Return: {total_ret:.2f}%")
        """
        # Handle Series vs DataFrame
        if isinstance(equity_curve, pd.Series):
            equity = equity_curve
        elif isinstance(equity_curve, pd.DataFrame):
            if 'equity' not in equity_curve.columns:
                raise KeyError("DataFrame must contain 'equity' column")
            equity = equity_curve['equity']
        else:
            raise InvalidDataError(
                f"Expected DataFrame or Series, got {type(equity_curve).__name__}"
            )

        # Validate data
        if equity.empty:
            raise InvalidDataError("Equity data is empty")

        if len(equity) < MIN_OBSERVATIONS_FOR_STATS:
            raise InsufficientDataError(
                f"Need at least {MIN_OBSERVATIONS_FOR_STATS} observations, "
                f"got {len(equity)}"
            )

        # Get initial and final values
        initial_equity = float(equity.iloc[0])
        final_equity = float(equity.iloc[-1])

        # Validate initial equity
        if not np.isfinite(initial_equity) or initial_equity <= 0:
            raise InvalidDataError(
                f"Initial equity must be positive and finite, got {initial_equity}"
            )

        if not np.isfinite(final_equity):
            raise InvalidDataError(
                f"Final equity must be finite, got {final_equity}"
            )

        # Calculate total return
        total_return = (final_equity - initial_equity) / initial_equity * 100.0

        return float(total_return)

    @staticmethod
    def calculate_annualized_return(
        equity_curve: pd.DataFrame,
        trading_days_per_year: int = TRADING_DAYS_PER_YEAR
    ) -> float:
        """
        Calculate Compound Annual Growth Rate (CAGR).

        Formula:
            CAGR = (Final / Initial)^(trading_days_per_year / num_days) - 1

        This formula properly handles partial years and different period lengths.

        Args:
            equity_curve: DataFrame with 'equity' column or Series of equity values
            trading_days_per_year: Number of trading days per year (default 252)

        Returns:
            Annualized return as decimal (e.g., 0.15 for 15% annualized return)

        Raises:
            InsufficientDataError: If equity_curve has fewer than 2 data points
            InvalidDataError: If equity_curve is empty or has invalid values

        Note:
            For periods less than 1 year, this extrapolates the return.
            Results for very short periods should be interpreted with caution.

        Example:
            >>> cagr = PerformanceMetrics.calculate_annualized_return(equity_curve)
            >>> print(f"CAGR: {cagr:.2%}")
        """
        # Handle Series vs DataFrame
        if isinstance(equity_curve, pd.Series):
            equity = equity_curve
        elif isinstance(equity_curve, pd.DataFrame):
            if 'equity' not in equity_curve.columns:
                raise KeyError("DataFrame must contain 'equity' column")
            equity = equity_curve['equity']
        else:
            raise InvalidDataError(
                f"Expected DataFrame or Series, got {type(equity_curve).__name__}"
            )

        # Validate data
        if equity.empty:
            raise InvalidDataError("Equity data is empty")

        if len(equity) < MIN_OBSERVATIONS_FOR_STATS:
            raise InsufficientDataError(
                f"Need at least {MIN_OBSERVATIONS_FOR_STATS} observations, "
                f"got {len(equity)}"
            )

        # Get initial and final values
        initial_equity = float(equity.iloc[0])
        final_equity = float(equity.iloc[-1])

        # Validate values
        if not np.isfinite(initial_equity) or initial_equity <= 0:
            raise InvalidDataError(
                f"Initial equity must be positive and finite, got {initial_equity}"
            )

        if not np.isfinite(final_equity) or final_equity <= 0:
            # Handle total loss scenario
            if final_equity <= 0:
                return -1.0  # -100% annualized return
            raise InvalidDataError(
                f"Final equity must be positive and finite, got {final_equity}"
            )

        # Calculate number of periods
        num_days = len(equity)

        if num_days <= 1:
            raise InsufficientDataError(
                "Need more than 1 observation for annualized return"
            )

        # Calculate CAGR
        # CAGR = (Final/Initial)^(periods_per_year/num_periods) - 1
        total_return_ratio = final_equity / initial_equity

        # Annualization factor
        annualization_exponent = trading_days_per_year / (num_days - 1)

        # Handle negative total return (loss)
        if total_return_ratio <= 0:
            return -1.0  # -100% annualized return

        cagr = np.power(total_return_ratio, annualization_exponent) - 1.0

        return float(cagr)

    @staticmethod
    def calculate_sharpe_ratio(
        returns: pd.Series,
        risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
        periods_per_year: int = TRADING_DAYS_PER_YEAR
    ) -> float:
        """
        Calculate annualized Sharpe Ratio.

        Formula:
            Sharpe = (mean(returns) - rf_per_period) / std(returns) * sqrt(periods_per_year)

        The Sharpe ratio measures risk-adjusted return by comparing excess returns
        to volatility. Higher is better; typically:
            - < 0: Bad
            - 0-1: Acceptable
            - 1-2: Good
            - > 2: Excellent

        Args:
            returns: Series of period returns (daily if periods_per_year=252)
            risk_free_rate: Annual risk-free rate as decimal (default 0.02 = 2%)
            periods_per_year: Number of periods per year (default 252 for daily)

        Returns:
            Annualized Sharpe ratio

        Raises:
            InsufficientDataError: If returns has fewer than MIN_OBSERVATIONS_FOR_SHARPE
            InvalidDataError: If returns is empty or all NaN

        Note:
            This implementation uses sample standard deviation (N-1 denominator).
            Returns a warning-logged 0.0 if volatility is zero.

        Example:
            >>> returns = equity['equity'].pct_change().dropna()
            >>> sharpe = PerformanceMetrics.calculate_sharpe_ratio(returns)
            >>> print(f"Sharpe Ratio: {sharpe:.2f}")

        Reference:
            Sharpe, W.F. (1994). The Sharpe Ratio. Journal of Portfolio Management.
        """
        # Validate input
        if not isinstance(returns, pd.Series):
            returns = pd.Series(returns)

        # Remove NaN values
        returns = returns.dropna()

        if returns.empty:
            raise InvalidDataError("Returns series is empty after removing NaN values")

        if len(returns) < MIN_OBSERVATIONS_FOR_SHARPE:
            logger.warning(
                f"Sharpe ratio calculated with only {len(returns)} observations. "
                f"Recommend at least {MIN_OBSERVATIONS_FOR_SHARPE} for reliability."
            )

        if len(returns) < MIN_OBSERVATIONS_FOR_STATS:
            raise InsufficientDataError(
                f"Need at least {MIN_OBSERVATIONS_FOR_STATS} observations, "
                f"got {len(returns)}"
            )

        # Convert annual risk-free rate to per-period rate
        rf_per_period = risk_free_rate / periods_per_year

        # Calculate excess returns
        excess_returns = returns - rf_per_period

        # Calculate mean and standard deviation
        mean_excess_return = excess_returns.mean()
        std_returns = returns.std()

        # Handle zero volatility
        if std_returns < EPSILON:
            logger.warning(
                "Zero volatility detected in Sharpe ratio calculation. "
                "Returning 0.0."
            )
            return 0.0

        # Calculate Sharpe ratio with annualization
        sharpe = (mean_excess_return / std_returns) * np.sqrt(periods_per_year)

        return float(sharpe)

    @staticmethod
    def calculate_sortino_ratio(
        returns: pd.Series,
        risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
        periods_per_year: int = TRADING_DAYS_PER_YEAR
    ) -> float:
        """
        Calculate annualized Sortino Ratio.

        Formula:
            Sortino = (mean(returns) - rf_per_period) / downside_deviation * sqrt(periods_per_year)

        The Sortino ratio is similar to Sharpe but uses downside deviation
        instead of total volatility, penalizing only negative returns.
        This is more appropriate for asymmetric return distributions.

        Args:
            returns: Series of period returns
            risk_free_rate: Annual risk-free rate as decimal (default 0.02 = 2%)
            periods_per_year: Number of periods per year (default 252 for daily)

        Returns:
            Annualized Sortino ratio

        Raises:
            InsufficientDataError: If returns has insufficient data
            InvalidDataError: If returns is empty or all NaN

        Note:
            Downside deviation is calculated as the standard deviation of
            returns below zero (or below the target, which we set to zero).
            If there are no negative returns, returns infinity (positive).

        Example:
            >>> returns = equity['equity'].pct_change().dropna()
            >>> sortino = PerformanceMetrics.calculate_sortino_ratio(returns)
            >>> print(f"Sortino Ratio: {sortino:.2f}")

        Reference:
            Sortino, F.A. & van der Meer, R. (1991). Downside Risk.
        """
        # Validate input
        if not isinstance(returns, pd.Series):
            returns = pd.Series(returns)

        # Remove NaN values
        returns = returns.dropna()

        if returns.empty:
            raise InvalidDataError("Returns series is empty after removing NaN values")

        if len(returns) < MIN_OBSERVATIONS_FOR_STATS:
            raise InsufficientDataError(
                f"Need at least {MIN_OBSERVATIONS_FOR_STATS} observations, "
                f"got {len(returns)}"
            )

        # Convert annual risk-free rate to per-period rate
        rf_per_period = risk_free_rate / periods_per_year

        # Calculate excess returns
        mean_excess_return = returns.mean() - rf_per_period

        # Calculate downside deviation
        # Using target return of 0 (minimum acceptable return)
        target = 0.0
        negative_returns = returns[returns < target]

        if len(negative_returns) < 1:
            # No negative returns - strategy never loses
            # Return a large positive number (not infinity for numerical stability)
            logger.info(
                "No negative returns found in Sortino calculation. "
                "Returning large positive value."
            )
            return float(np.inf) if mean_excess_return > 0 else 0.0

        # Downside deviation: std of returns below target
        # Using semi-deviation formula: sqrt(mean(min(r-target, 0)^2))
        downside_squared = np.square(np.minimum(returns - target, 0))
        downside_deviation = np.sqrt(downside_squared.mean())

        # Handle zero downside deviation
        if downside_deviation < EPSILON:
            logger.warning(
                "Near-zero downside deviation in Sortino calculation. "
                "Returning 0.0."
            )
            return 0.0

        # Calculate Sortino ratio with annualization
        sortino = (mean_excess_return / downside_deviation) * np.sqrt(periods_per_year)

        return float(sortino)

    @staticmethod
    def calculate_calmar_ratio(
        annualized_return: float,
        max_drawdown: float
    ) -> float:
        """
        Calculate Calmar Ratio.

        Formula:
            Calmar = Annualized Return / |Max Drawdown|

        The Calmar ratio measures return relative to drawdown risk.
        Higher is better; typically:
            - < 0.5: Poor
            - 0.5-1.0: Acceptable
            - 1.0-3.0: Good
            - > 3.0: Excellent

        Args:
            annualized_return: CAGR as decimal (e.g., 0.15 for 15%)
            max_drawdown: Maximum drawdown as decimal (e.g., -0.20 for -20%)
                         Should be negative or zero.

        Returns:
            Calmar ratio

        Raises:
            InvalidDataError: If max_drawdown is zero or inputs are invalid

        Note:
            If max_drawdown is zero (no drawdown), returns infinity or 0
            depending on the return sign.

        Example:
            >>> cagr = PerformanceMetrics.calculate_annualized_return(equity_curve)
            >>> mdd = PerformanceMetrics.calculate_max_drawdown(equity_curve)['max_drawdown_pct']
            >>> calmar = PerformanceMetrics.calculate_calmar_ratio(cagr, mdd)

        Reference:
            Young, T.W. (1991). Calmar Ratio: A Smoother Tool.
        """
        # Validate inputs
        if not np.isfinite(annualized_return):
            raise InvalidDataError(
                f"Annualized return must be finite, got {annualized_return}"
            )

        if not np.isfinite(max_drawdown):
            raise InvalidDataError(
                f"Max drawdown must be finite, got {max_drawdown}"
            )

        # Handle zero drawdown
        abs_max_drawdown = abs(max_drawdown)

        if abs_max_drawdown < EPSILON:
            # No drawdown - perfect strategy
            if annualized_return > 0:
                return float(np.inf)
            elif annualized_return < 0:
                return float(-np.inf)
            else:
                return 0.0

        # Calculate Calmar ratio
        calmar = annualized_return / abs_max_drawdown

        return float(calmar)

    # =========================================================================
    # Drawdown Metrics
    # =========================================================================

    @staticmethod
    def calculate_max_drawdown(equity_curve: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate maximum drawdown and related metrics.

        Formula:
            Drawdown(t) = (Equity(t) - Peak(t)) / Peak(t)
            Max Drawdown = min(Drawdown(t)) for all t

        Drawdown measures the decline from a historical peak. Maximum drawdown
        is the worst peak-to-trough decline, representing the largest loss
        an investor could have experienced.

        Args:
            equity_curve: DataFrame with 'equity' column or Series of equity values.
                         Index should be datetime for date tracking.

        Returns:
            Dictionary containing:
                - 'max_drawdown_pct': Maximum drawdown as decimal (negative)
                - 'max_drawdown_value': Maximum drawdown in dollars (negative)
                - 'peak_date': Date of the peak before max drawdown
                - 'trough_date': Date of the trough (lowest point)
                - 'recovery_date': Date when equity recovered to peak (None if not recovered)
                - 'duration_days': Days from peak to trough
                - 'recovery_days': Days from trough to recovery (None if not recovered)
                - 'drawdown_series': Series of drawdown percentages over time
                - 'underwater_series': Same as drawdown_series (alternative name)

        Raises:
            InsufficientDataError: If equity_curve has fewer than 2 data points
            InvalidDataError: If equity_curve is empty or has invalid values

        Example:
            >>> dd_info = PerformanceMetrics.calculate_max_drawdown(equity_curve)
            >>> print(f"Max Drawdown: {dd_info['max_drawdown_pct']:.2%}")
            >>> print(f"Peak Date: {dd_info['peak_date']}")
            >>> print(f"Trough Date: {dd_info['trough_date']}")
        """
        # Handle Series vs DataFrame
        if isinstance(equity_curve, pd.Series):
            equity = equity_curve.copy()
        elif isinstance(equity_curve, pd.DataFrame):
            if 'equity' not in equity_curve.columns:
                raise KeyError("DataFrame must contain 'equity' column")
            equity = equity_curve['equity'].copy()
        else:
            raise InvalidDataError(
                f"Expected DataFrame or Series, got {type(equity_curve).__name__}"
            )

        # Validate data
        if equity.empty:
            raise InvalidDataError("Equity data is empty")

        if len(equity) < MIN_OBSERVATIONS_FOR_STATS:
            raise InsufficientDataError(
                f"Need at least {MIN_OBSERVATIONS_FOR_STATS} observations, "
                f"got {len(equity)}"
            )

        # Calculate running maximum (peak)
        running_max = equity.expanding().max()

        # Calculate drawdown series
        drawdown_series = (equity - running_max) / running_max

        # Calculate drawdown values (in dollars)
        drawdown_values = equity - running_max

        # Find maximum drawdown
        max_dd_idx = drawdown_series.idxmin()
        max_drawdown_pct = float(drawdown_series.loc[max_dd_idx])
        max_drawdown_value = float(drawdown_values.loc[max_dd_idx])

        # Find peak date (before trough)
        # Get equity values up to and including trough
        equity_to_trough = equity.loc[:max_dd_idx]
        peak_idx = equity_to_trough.idxmax()
        peak_value = float(equity.loc[peak_idx])

        # Calculate duration
        peak_date = peak_idx
        trough_date = max_dd_idx

        # Calculate days between peak and trough
        if isinstance(peak_date, (datetime, pd.Timestamp)) and isinstance(trough_date, (datetime, pd.Timestamp)):
            duration_days = (trough_date - peak_date).days
        else:
            # If index is not datetime, use position difference
            peak_pos = equity.index.get_loc(peak_idx)
            trough_pos = equity.index.get_loc(max_dd_idx)
            duration_days = trough_pos - peak_pos

        # Find recovery date
        recovery_date = None
        recovery_days = None

        # Look for equity values after trough that exceed peak value
        equity_after_trough = equity.loc[max_dd_idx:]
        recovery_mask = equity_after_trough >= peak_value

        if recovery_mask.any():
            recovery_date = recovery_mask.idxmax()
            if isinstance(recovery_date, (datetime, pd.Timestamp)) and isinstance(trough_date, (datetime, pd.Timestamp)):
                recovery_days = (recovery_date - trough_date).days
            else:
                trough_pos = equity.index.get_loc(max_dd_idx)
                recovery_pos = equity.index.get_loc(recovery_date)
                recovery_days = recovery_pos - trough_pos

        return {
            'max_drawdown_pct': max_drawdown_pct,
            'max_drawdown_value': max_drawdown_value,
            'peak_date': peak_date,
            'peak_value': peak_value,
            'trough_date': trough_date,
            'trough_value': float(equity.loc[max_dd_idx]),
            'recovery_date': recovery_date,
            'duration_days': duration_days,
            'recovery_days': recovery_days,
            'drawdown_series': drawdown_series,
            'underwater_series': drawdown_series,  # Alternative name
        }

    @staticmethod
    def calculate_average_drawdown(equity_curve: pd.DataFrame) -> float:
        """
        Calculate average drawdown percentage.

        Args:
            equity_curve: DataFrame with 'equity' column or Series

        Returns:
            Average drawdown as decimal (negative)

        Example:
            >>> avg_dd = PerformanceMetrics.calculate_average_drawdown(equity_curve)
            >>> print(f"Average Drawdown: {avg_dd:.2%}")
        """
        dd_info = PerformanceMetrics.calculate_max_drawdown(equity_curve)
        drawdown_series = dd_info['drawdown_series']

        return float(drawdown_series.mean())

    @staticmethod
    def calculate_ulcer_index(equity_curve: pd.DataFrame) -> float:
        """
        Calculate Ulcer Index (UI).

        Formula:
            UI = sqrt(mean(drawdown^2))

        The Ulcer Index measures both depth and duration of drawdowns.
        Lower is better.

        Args:
            equity_curve: DataFrame with 'equity' column or Series

        Returns:
            Ulcer Index as decimal

        Example:
            >>> ui = PerformanceMetrics.calculate_ulcer_index(equity_curve)
            >>> print(f"Ulcer Index: {ui:.4f}")

        Reference:
            Martin, P. & McCann, B. (1989). The Investor's Guide to Fidelity Funds.
        """
        dd_info = PerformanceMetrics.calculate_max_drawdown(equity_curve)
        drawdown_series = dd_info['drawdown_series']

        # Ulcer Index is RMS of drawdowns
        ulcer_index = np.sqrt(np.mean(np.square(drawdown_series)))

        return float(ulcer_index)

    # =========================================================================
    # Trade-Based Metrics
    # =========================================================================

    @staticmethod
    def calculate_win_rate(trades: pd.DataFrame) -> float:
        """
        Calculate percentage of profitable trades.

        Formula:
            Win Rate = Number of Winning Trades / Total Trades * 100

        Args:
            trades: DataFrame with 'realized_pnl' column for closed trades

        Returns:
            Win rate as percentage (0-100)

        Raises:
            InsufficientDataError: If no trades provided
            KeyError: If 'realized_pnl' column not found

        Note:
            Trades with exactly zero P&L are not counted as wins.

        Example:
            >>> win_rate = PerformanceMetrics.calculate_win_rate(trade_log)
            >>> print(f"Win Rate: {win_rate:.1f}%")
        """
        if trades.empty:
            raise InsufficientDataError("No trades provided")

        if 'realized_pnl' not in trades.columns:
            raise KeyError("trades DataFrame must contain 'realized_pnl' column")

        # Filter to closed trades only (those with realized P&L)
        closed_trades = trades.dropna(subset=['realized_pnl'])

        if closed_trades.empty:
            raise InsufficientDataError("No closed trades with realized P&L")

        # Count winning trades (P&L > 0)
        winning_trades = (closed_trades['realized_pnl'] > 0).sum()
        total_trades = len(closed_trades)

        win_rate = (winning_trades / total_trades) * 100.0

        return float(win_rate)

    @staticmethod
    def calculate_profit_factor(trades: pd.DataFrame) -> float:
        """
        Calculate profit factor.

        Formula:
            Profit Factor = Gross Profit / Gross Loss

        Profit factor measures how much profit is made for each dollar lost.
        Values above 1.0 indicate a profitable strategy.
            - < 1.0: Losing strategy
            - 1.0-1.5: Marginal
            - 1.5-2.0: Good
            - > 2.0: Excellent

        Args:
            trades: DataFrame with 'realized_pnl' column for closed trades

        Returns:
            Profit factor (gross profit / gross loss)

        Raises:
            InsufficientDataError: If no trades or no losing trades
            KeyError: If 'realized_pnl' column not found

        Note:
            If there are no losing trades, returns infinity.
            If there are no winning trades, returns 0.

        Example:
            >>> pf = PerformanceMetrics.calculate_profit_factor(trade_log)
            >>> print(f"Profit Factor: {pf:.2f}")
        """
        if trades.empty:
            raise InsufficientDataError("No trades provided")

        if 'realized_pnl' not in trades.columns:
            raise KeyError("trades DataFrame must contain 'realized_pnl' column")

        # Filter to closed trades
        closed_trades = trades.dropna(subset=['realized_pnl'])

        if closed_trades.empty:
            raise InsufficientDataError("No closed trades with realized P&L")

        pnl = closed_trades['realized_pnl']

        # Calculate gross profit and loss
        gross_profit = pnl[pnl > 0].sum()
        gross_loss = abs(pnl[pnl < 0].sum())

        # Handle edge cases
        if gross_loss < EPSILON:
            # No losing trades
            return float(np.inf) if gross_profit > 0 else 0.0

        if gross_profit < EPSILON:
            # No winning trades
            return 0.0

        profit_factor = gross_profit / gross_loss

        return float(profit_factor)

    @staticmethod
    def calculate_average_win(trades: pd.DataFrame) -> float:
        """
        Calculate average winning trade P&L.

        Args:
            trades: DataFrame with 'realized_pnl' column

        Returns:
            Average P&L of winning trades in dollars

        Raises:
            InsufficientDataError: If no winning trades

        Example:
            >>> avg_win = PerformanceMetrics.calculate_average_win(trade_log)
            >>> print(f"Average Win: ${avg_win:,.2f}")
        """
        if trades.empty:
            raise InsufficientDataError("No trades provided")

        if 'realized_pnl' not in trades.columns:
            raise KeyError("trades DataFrame must contain 'realized_pnl' column")

        closed_trades = trades.dropna(subset=['realized_pnl'])
        winning_trades = closed_trades[closed_trades['realized_pnl'] > 0]

        if winning_trades.empty:
            raise InsufficientDataError("No winning trades found")

        return float(winning_trades['realized_pnl'].mean())

    @staticmethod
    def calculate_average_loss(trades: pd.DataFrame) -> float:
        """
        Calculate average losing trade P&L.

        Args:
            trades: DataFrame with 'realized_pnl' column

        Returns:
            Average P&L of losing trades in dollars (negative value)

        Raises:
            InsufficientDataError: If no losing trades

        Example:
            >>> avg_loss = PerformanceMetrics.calculate_average_loss(trade_log)
            >>> print(f"Average Loss: ${avg_loss:,.2f}")
        """
        if trades.empty:
            raise InsufficientDataError("No trades provided")

        if 'realized_pnl' not in trades.columns:
            raise KeyError("trades DataFrame must contain 'realized_pnl' column")

        closed_trades = trades.dropna(subset=['realized_pnl'])
        losing_trades = closed_trades[closed_trades['realized_pnl'] < 0]

        if losing_trades.empty:
            raise InsufficientDataError("No losing trades found")

        return float(losing_trades['realized_pnl'].mean())

    @staticmethod
    def calculate_expectancy(trades: pd.DataFrame) -> float:
        """
        Calculate expected value per trade.

        Formula:
            Expectancy = Win Rate * Average Win - Loss Rate * |Average Loss|

        Also known as "mathematical expectation" or "edge per trade".
        Represents the average amount you expect to make per trade.

        Args:
            trades: DataFrame with 'realized_pnl' column

        Returns:
            Expected P&L per trade in dollars

        Raises:
            InsufficientDataError: If insufficient trade data

        Example:
            >>> exp = PerformanceMetrics.calculate_expectancy(trade_log)
            >>> print(f"Expectancy per Trade: ${exp:,.2f}")
        """
        if trades.empty:
            raise InsufficientDataError("No trades provided")

        if 'realized_pnl' not in trades.columns:
            raise KeyError("trades DataFrame must contain 'realized_pnl' column")

        closed_trades = trades.dropna(subset=['realized_pnl'])

        if closed_trades.empty:
            raise InsufficientDataError("No closed trades with realized P&L")

        pnl = closed_trades['realized_pnl']
        winning = pnl[pnl > 0]
        losing = pnl[pnl < 0]

        total_trades = len(closed_trades)

        # Calculate win rate as decimal
        win_rate = len(winning) / total_trades if total_trades > 0 else 0.0
        loss_rate = 1.0 - win_rate

        # Calculate average win and loss
        avg_win = winning.mean() if len(winning) > 0 else 0.0
        avg_loss = abs(losing.mean()) if len(losing) > 0 else 0.0

        # Calculate expectancy
        expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)

        return float(expectancy)

    @staticmethod
    def calculate_average_trade(trades: pd.DataFrame) -> float:
        """
        Calculate average P&L per trade.

        Args:
            trades: DataFrame with 'realized_pnl' column

        Returns:
            Average P&L per trade in dollars

        Example:
            >>> avg_trade = PerformanceMetrics.calculate_average_trade(trade_log)
            >>> print(f"Average Trade P&L: ${avg_trade:,.2f}")
        """
        if trades.empty:
            raise InsufficientDataError("No trades provided")

        if 'realized_pnl' not in trades.columns:
            raise KeyError("trades DataFrame must contain 'realized_pnl' column")

        closed_trades = trades.dropna(subset=['realized_pnl'])

        if closed_trades.empty:
            raise InsufficientDataError("No closed trades with realized P&L")

        return float(closed_trades['realized_pnl'].mean())

    @staticmethod
    def calculate_payoff_ratio(trades: pd.DataFrame) -> float:
        """
        Calculate payoff ratio (reward/risk ratio).

        Formula:
            Payoff Ratio = Average Win / |Average Loss|

        Also known as "risk/reward ratio" or "average win/loss ratio".

        Args:
            trades: DataFrame with 'realized_pnl' column

        Returns:
            Payoff ratio

        Example:
            >>> payoff = PerformanceMetrics.calculate_payoff_ratio(trade_log)
            >>> print(f"Payoff Ratio: {payoff:.2f}")
        """
        try:
            avg_win = PerformanceMetrics.calculate_average_win(trades)
            avg_loss = PerformanceMetrics.calculate_average_loss(trades)
        except InsufficientDataError:
            return 0.0

        if abs(avg_loss) < EPSILON:
            return float(np.inf) if avg_win > 0 else 0.0

        return abs(avg_win / avg_loss)

    @staticmethod
    def calculate_consecutive_wins(trades: pd.DataFrame) -> int:
        """
        Calculate maximum consecutive winning trades.

        Args:
            trades: DataFrame with 'realized_pnl' column, sorted by time

        Returns:
            Maximum number of consecutive wins

        Example:
            >>> max_wins = PerformanceMetrics.calculate_consecutive_wins(trade_log)
            >>> print(f"Max Consecutive Wins: {max_wins}")
        """
        if trades.empty:
            return 0

        if 'realized_pnl' not in trades.columns:
            raise KeyError("trades DataFrame must contain 'realized_pnl' column")

        closed_trades = trades.dropna(subset=['realized_pnl'])

        if closed_trades.empty:
            return 0

        # Create win/loss sequence
        is_win = (closed_trades['realized_pnl'] > 0).astype(int)

        # Find consecutive wins
        max_consecutive = 0
        current_consecutive = 0

        for win in is_win:
            if win == 1:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0

        return max_consecutive

    @staticmethod
    def calculate_consecutive_losses(trades: pd.DataFrame) -> int:
        """
        Calculate maximum consecutive losing trades.

        Args:
            trades: DataFrame with 'realized_pnl' column, sorted by time

        Returns:
            Maximum number of consecutive losses

        Example:
            >>> max_losses = PerformanceMetrics.calculate_consecutive_losses(trade_log)
            >>> print(f"Max Consecutive Losses: {max_losses}")
        """
        if trades.empty:
            return 0

        if 'realized_pnl' not in trades.columns:
            raise KeyError("trades DataFrame must contain 'realized_pnl' column")

        closed_trades = trades.dropna(subset=['realized_pnl'])

        if closed_trades.empty:
            return 0

        # Create win/loss sequence
        is_loss = (closed_trades['realized_pnl'] < 0).astype(int)

        # Find consecutive losses
        max_consecutive = 0
        current_consecutive = 0

        for loss in is_loss:
            if loss == 1:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0

        return max_consecutive

    # =========================================================================
    # Distribution Metrics
    # =========================================================================

    @staticmethod
    def calculate_returns_distribution(returns: pd.Series) -> Dict[str, Any]:
        """
        Calculate comprehensive return distribution statistics.

        Provides statistical characterization of the return distribution
        including moments (mean, variance, skewness, kurtosis) and
        percentiles for tail analysis.

        Args:
            returns: Series of period returns

        Returns:
            Dictionary containing:
                - 'mean': Average return
                - 'median': Median return
                - 'std': Standard deviation
                - 'variance': Variance
                - 'skewness': Skewness (0 = symmetric, negative = left tail)
                - 'kurtosis': Excess kurtosis (0 = normal, positive = fat tails)
                - 'min': Minimum return
                - 'max': Maximum return
                - 'range': Max - Min
                - 'percentiles': Dict with percentile values

        Raises:
            InsufficientDataError: If insufficient data points

        Note:
            - Negative skewness indicates left tail (more extreme losses)
            - Positive kurtosis indicates fat tails (more extreme events)
            - Options strategies often have negative skewness

        Example:
            >>> dist = PerformanceMetrics.calculate_returns_distribution(returns)
            >>> print(f"Mean: {dist['mean']:.4%}")
            >>> print(f"Skewness: {dist['skewness']:.2f}")
        """
        if not isinstance(returns, pd.Series):
            returns = pd.Series(returns)

        returns = returns.dropna()

        if len(returns) < MIN_OBSERVATIONS_FOR_STATS:
            raise InsufficientDataError(
                f"Need at least {MIN_OBSERVATIONS_FOR_STATS} observations, "
                f"got {len(returns)}"
            )

        # Basic statistics
        mean_return = float(returns.mean())
        median_return = float(returns.median())
        std_return = float(returns.std())
        variance = float(returns.var())
        min_return = float(returns.min())
        max_return = float(returns.max())

        # Higher moments
        # Use scipy for more robust skewness/kurtosis calculations
        if len(returns) >= 3:
            skewness = float(stats.skew(returns))
        else:
            skewness = 0.0

        if len(returns) >= 4:
            # Fisher's kurtosis (excess kurtosis, normal = 0)
            kurtosis = float(stats.kurtosis(returns, fisher=True))
        else:
            kurtosis = 0.0

        # Percentiles
        percentile_levels = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        percentiles = {}
        for p in percentile_levels:
            percentiles[p] = float(np.percentile(returns, p))

        return {
            'mean': mean_return,
            'median': median_return,
            'std': std_return,
            'variance': variance,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'min': min_return,
            'max': max_return,
            'range': max_return - min_return,
            'count': len(returns),
            'percentiles': percentiles,
        }

    @staticmethod
    def calculate_rolling_sharpe(
        returns: pd.Series,
        window: int = 63,  # ~3 months of trading days
        risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
        periods_per_year: int = TRADING_DAYS_PER_YEAR
    ) -> pd.Series:
        """
        Calculate rolling Sharpe ratio.

        Args:
            returns: Series of period returns
            window: Rolling window size in periods
            risk_free_rate: Annual risk-free rate
            periods_per_year: Periods per year for annualization

        Returns:
            Series of rolling Sharpe ratios

        Example:
            >>> rolling_sharpe = PerformanceMetrics.calculate_rolling_sharpe(returns)
            >>> plt.plot(rolling_sharpe)
        """
        if not isinstance(returns, pd.Series):
            returns = pd.Series(returns)

        returns = returns.dropna()

        if len(returns) < window:
            raise InsufficientDataError(
                f"Need at least {window} observations for rolling calculation, "
                f"got {len(returns)}"
            )

        rf_per_period = risk_free_rate / periods_per_year

        # Calculate rolling mean and std
        rolling_mean = returns.rolling(window).mean()
        rolling_std = returns.rolling(window).std()

        # Calculate rolling Sharpe
        rolling_sharpe = ((rolling_mean - rf_per_period) / rolling_std) * np.sqrt(periods_per_year)

        return rolling_sharpe

    @staticmethod
    def calculate_monthly_returns(equity_curve: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate monthly returns from equity curve.

        Args:
            equity_curve: DataFrame with 'equity' column, datetime index

        Returns:
            DataFrame with monthly returns by year/month

        Example:
            >>> monthly = PerformanceMetrics.calculate_monthly_returns(equity_curve)
            >>> print(monthly)
        """
        # Handle Series vs DataFrame
        if isinstance(equity_curve, pd.Series):
            equity = equity_curve.copy()
        elif isinstance(equity_curve, pd.DataFrame):
            if 'equity' not in equity_curve.columns:
                raise KeyError("DataFrame must contain 'equity' column")
            equity = equity_curve['equity'].copy()
        else:
            raise InvalidDataError(
                f"Expected DataFrame or Series, got {type(equity_curve).__name__}"
            )

        # Ensure datetime index
        if not isinstance(equity.index, pd.DatetimeIndex):
            try:
                equity.index = pd.to_datetime(equity.index)
            except Exception:
                raise InvalidDataError("Cannot convert index to datetime")

        # Resample to monthly (end of month) and calculate returns
        monthly_equity = equity.resample('ME').last()
        monthly_returns = monthly_equity.pct_change()

        # Create pivot table with years as rows, months as columns
        monthly_returns_df = pd.DataFrame({
            'year': monthly_returns.index.year,
            'month': monthly_returns.index.month,
            'return': monthly_returns.values
        })

        pivot = monthly_returns_df.pivot(index='year', columns='month', values='return')
        pivot.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][:len(pivot.columns)]

        return pivot

    # =========================================================================
    # Summary Methods
    # =========================================================================

    @staticmethod
    def calculate_all_metrics(
        equity_curve: pd.DataFrame,
        trades: Optional[pd.DataFrame] = None,
        risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
        periods_per_year: int = TRADING_DAYS_PER_YEAR
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive performance metrics summary.

        This is a convenience method that calculates all available metrics
        and returns them in a structured dictionary.

        Args:
            equity_curve: DataFrame with 'equity' column
            trades: Optional DataFrame with trade records
            risk_free_rate: Annual risk-free rate
            periods_per_year: Periods per year for annualization

        Returns:
            Dictionary with all calculated metrics organized by category:
                - 'returns': Return-based metrics
                - 'risk': Risk-adjusted metrics
                - 'drawdown': Drawdown metrics
                - 'trades': Trade-based metrics (if trades provided)
                - 'distribution': Distribution statistics

        Example:
            >>> metrics = PerformanceMetrics.calculate_all_metrics(
            ...     equity_curve=equity_curve,
            ...     trades=trade_log
            ... )
            >>> print(f"Sharpe Ratio: {metrics['risk']['sharpe_ratio']:.2f}")
        """
        # Get equity and returns
        if isinstance(equity_curve, pd.Series):
            equity = equity_curve
        elif isinstance(equity_curve, pd.DataFrame):
            equity = equity_curve['equity']
        else:
            raise InvalidDataError("Invalid equity_curve format")

        returns = equity.pct_change().dropna()

        # Calculate return metrics
        try:
            total_return = PerformanceMetrics.calculate_total_return(equity_curve)
        except Exception as e:
            logger.warning(f"Could not calculate total return: {e}")
            total_return = np.nan

        try:
            annualized_return = PerformanceMetrics.calculate_annualized_return(
                equity_curve, periods_per_year
            )
        except Exception as e:
            logger.warning(f"Could not calculate CAGR: {e}")
            annualized_return = np.nan

        # Calculate risk metrics
        try:
            sharpe = PerformanceMetrics.calculate_sharpe_ratio(
                returns, risk_free_rate, periods_per_year
            )
        except Exception as e:
            logger.warning(f"Could not calculate Sharpe: {e}")
            sharpe = np.nan

        try:
            sortino = PerformanceMetrics.calculate_sortino_ratio(
                returns, risk_free_rate, periods_per_year
            )
        except Exception as e:
            logger.warning(f"Could not calculate Sortino: {e}")
            sortino = np.nan

        # Calculate drawdown metrics
        try:
            dd_info = PerformanceMetrics.calculate_max_drawdown(equity_curve)
            max_drawdown = dd_info['max_drawdown_pct']
        except Exception as e:
            logger.warning(f"Could not calculate max drawdown: {e}")
            max_drawdown = np.nan
            dd_info = {}

        try:
            calmar = PerformanceMetrics.calculate_calmar_ratio(
                annualized_return if not np.isnan(annualized_return) else 0.0,
                max_drawdown if not np.isnan(max_drawdown) else -0.01
            )
        except Exception as e:
            logger.warning(f"Could not calculate Calmar: {e}")
            calmar = np.nan

        try:
            ulcer = PerformanceMetrics.calculate_ulcer_index(equity_curve)
        except Exception as e:
            logger.warning(f"Could not calculate Ulcer Index: {e}")
            ulcer = np.nan

        # Calculate distribution metrics
        try:
            dist = PerformanceMetrics.calculate_returns_distribution(returns)
        except Exception as e:
            logger.warning(f"Could not calculate distribution: {e}")
            dist = {}

        # Build results
        results = {
            'returns': {
                'total_return_pct': total_return,
                'annualized_return': annualized_return,
                'final_equity': float(equity.iloc[-1]) if len(equity) > 0 else np.nan,
                'initial_equity': float(equity.iloc[0]) if len(equity) > 0 else np.nan,
            },
            'risk': {
                'sharpe_ratio': sharpe,
                'sortino_ratio': sortino,
                'calmar_ratio': calmar,
                'volatility': float(returns.std() * np.sqrt(periods_per_year)) if len(returns) > 0 else np.nan,
            },
            'drawdown': {
                'max_drawdown_pct': max_drawdown,
                'max_drawdown_value': dd_info.get('max_drawdown_value', np.nan),
                'peak_date': dd_info.get('peak_date'),
                'trough_date': dd_info.get('trough_date'),
                'recovery_date': dd_info.get('recovery_date'),
                'duration_days': dd_info.get('duration_days', np.nan),
                'ulcer_index': ulcer,
            },
            'distribution': dist,
        }

        # Add trade metrics if trades provided
        if trades is not None and not trades.empty:
            trade_metrics = {}

            try:
                trade_metrics['win_rate'] = PerformanceMetrics.calculate_win_rate(trades)
            except Exception:
                trade_metrics['win_rate'] = np.nan

            try:
                trade_metrics['profit_factor'] = PerformanceMetrics.calculate_profit_factor(trades)
            except Exception:
                trade_metrics['profit_factor'] = np.nan

            try:
                trade_metrics['average_win'] = PerformanceMetrics.calculate_average_win(trades)
            except Exception:
                trade_metrics['average_win'] = np.nan

            try:
                trade_metrics['average_loss'] = PerformanceMetrics.calculate_average_loss(trades)
            except Exception:
                trade_metrics['average_loss'] = np.nan

            try:
                trade_metrics['expectancy'] = PerformanceMetrics.calculate_expectancy(trades)
            except Exception:
                trade_metrics['expectancy'] = np.nan

            try:
                trade_metrics['average_trade'] = PerformanceMetrics.calculate_average_trade(trades)
            except Exception:
                trade_metrics['average_trade'] = np.nan

            try:
                trade_metrics['payoff_ratio'] = PerformanceMetrics.calculate_payoff_ratio(trades)
            except Exception:
                trade_metrics['payoff_ratio'] = np.nan

            trade_metrics['max_consecutive_wins'] = PerformanceMetrics.calculate_consecutive_wins(trades)
            trade_metrics['max_consecutive_losses'] = PerformanceMetrics.calculate_consecutive_losses(trades)
            trade_metrics['total_trades'] = len(trades)

            results['trades'] = trade_metrics

        return results


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Main class
    'PerformanceMetrics',

    # Exceptions
    'MetricsError',
    'InsufficientDataError',
    'InvalidDataError',

    # Constants
    'TRADING_DAYS_PER_YEAR',
    'DEFAULT_RISK_FREE_RATE',
    'EPSILON',
]
