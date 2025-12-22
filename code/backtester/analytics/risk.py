"""
RiskAnalytics Class for Options Backtesting

This module provides the RiskAnalytics class for calculating risk metrics
used in quantitative finance for evaluating portfolio and strategy risk.

Key Features:
    - Value at Risk (VaR): Historical and parametric methods
    - Conditional VaR (CVaR/Expected Shortfall)
    - Greeks analysis and evolution over time
    - Tail risk metrics (skewness, kurtosis, tail ratios)
    - Margin utilization analysis
    - Maximum Adverse Excursion (MAE) analysis

Design Philosophy:
    All methods are static to enable easy use without instantiation.
    Methods accept pandas DataFrames/Series and return typed results.
    All formulas follow industry-standard implementations.

Mathematical Correctness:
    - VaR (Historical): alpha-percentile of return distribution
    - VaR (Parametric): mu - z_alpha * sigma (assuming normal distribution)
    - CVaR: E[R | R <= VaR] = average of returns below VaR threshold
    - Greeks correlation calculated using Pearson correlation

Usage:
    from backtester.analytics.risk import RiskAnalytics

    # Calculate VaR
    var_95 = RiskAnalytics.calculate_var(returns, confidence_level=0.95)
    cvar_95 = RiskAnalytics.calculate_cvar(returns, confidence_level=0.95)

    # Analyze Greeks
    greeks_analysis = RiskAnalytics.analyze_greeks_over_time(greeks_history)

References:
    - Jorion, P. (2007). Value at Risk: The New Benchmark for Managing Financial Risk.
    - Acerbi, C. & Tasche, D. (2002). Expected Shortfall: A Natural Coherent Alternative to VaR.
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

# Standard confidence levels for VaR
VAR_CONFIDENCE_95 = 0.95
VAR_CONFIDENCE_99 = 0.99

# Z-scores for common confidence levels (standard normal)
Z_SCORES = {
    0.90: 1.282,
    0.95: 1.645,
    0.99: 2.326,
    0.995: 2.576,
}

# Numerical tolerance
EPSILON = 1e-10

# Minimum observations for reliable VaR
MIN_OBSERVATIONS_FOR_VAR = 30

# Greeks names
GREEK_NAMES = ['delta', 'gamma', 'theta', 'vega', 'rho']

# Sigma thresholds for tail ratio calculations
SIGMA_THRESHOLD = 2.0


# =============================================================================
# Exceptions
# =============================================================================

class RiskAnalyticsError(Exception):
    """Base exception for risk analytics errors."""
    pass


class InsufficientDataError(RiskAnalyticsError):
    """Exception raised when there is insufficient data for calculation."""
    pass


class InvalidDataError(RiskAnalyticsError):
    """Exception raised when data is invalid for calculation."""
    pass


class InvalidConfidenceError(RiskAnalyticsError):
    """Exception raised when confidence level is invalid."""
    pass


# =============================================================================
# RiskAnalytics Class
# =============================================================================

class RiskAnalytics:
    """
    Calculate risk metrics for portfolio analysis.

    This class provides static methods for calculating standard risk metrics
    used in quantitative finance. All methods are designed to work with
    pandas DataFrames and Series.

    Metrics Categories:
        1. Value at Risk (VaR): Historical and parametric approaches
        2. Conditional VaR (CVaR): Expected Shortfall
        3. Greeks Analysis: Time series analysis of option Greeks
        4. Tail Risk: Skewness, kurtosis, and tail ratios
        5. Margin: Utilization analysis
        6. MAE: Maximum Adverse Excursion per trade

    All methods perform input validation and handle edge cases appropriately.

    Example:
        >>> returns = equity['equity'].pct_change().dropna()
        >>> var_95 = RiskAnalytics.calculate_var(returns, 0.95, 'historical')
        >>> cvar_95 = RiskAnalytics.calculate_cvar(returns, 0.95)
        >>> greeks = RiskAnalytics.analyze_greeks_over_time(greeks_history)
    """

    # =========================================================================
    # Value at Risk (VaR)
    # =========================================================================

    @staticmethod
    def calculate_var(
        returns: pd.Series,
        confidence_level: float = 0.95,
        method: str = 'historical'
    ) -> float:
        """
        Calculate Value at Risk at given confidence level.

        VaR represents the worst expected loss over a given time horizon
        at a given confidence level. For example, 95% daily VaR of -2%
        means there is a 5% chance of losing more than 2% in a day.

        Formula:
            Historical VaR: (1-alpha) percentile of return distribution
            Parametric VaR: mu - z_alpha * sigma

        Args:
            returns: Series of period returns (e.g., daily returns)
            confidence_level: Confidence level (e.g., 0.95 for 95% VaR)
            method: 'historical' or 'parametric'
                   - 'historical': Uses empirical percentile (non-parametric)
                   - 'parametric': Assumes normal distribution

        Returns:
            VaR as a decimal (negative number representing loss).
            For example, -0.02 means 2% loss at the given confidence level.

        Raises:
            InsufficientDataError: If insufficient data points
            InvalidConfidenceError: If confidence level not in (0, 1)
            ValueError: If invalid method specified

        Note:
            The returned value is typically negative (representing a loss).
            95% VaR = worst loss exceeded only 5% of the time.

        Example:
            >>> var_95 = RiskAnalytics.calculate_var(returns, 0.95, 'historical')
            >>> print(f"95% VaR: {var_95:.2%}")
            >>> # If var_95 = -0.025, then there's 5% chance of losing > 2.5%

        Reference:
            Jorion, P. (2007). Value at Risk: The New Benchmark.
        """
        # Validate confidence level
        if not 0 < confidence_level < 1:
            raise InvalidConfidenceError(
                f"Confidence level must be between 0 and 1, got {confidence_level}"
            )

        # Validate method
        valid_methods = ['historical', 'parametric']
        if method.lower() not in valid_methods:
            raise ValueError(
                f"Method must be one of {valid_methods}, got {method}"
            )

        # Convert to Series if needed
        if not isinstance(returns, pd.Series):
            returns = pd.Series(returns)

        # Remove NaN values
        returns = returns.dropna()

        if returns.empty:
            raise InvalidDataError("Returns series is empty after removing NaN values")

        if len(returns) < MIN_OBSERVATIONS_FOR_VAR:
            logger.warning(
                f"VaR calculated with only {len(returns)} observations. "
                f"Recommend at least {MIN_OBSERVATIONS_FOR_VAR} for reliability."
            )

        method = method.lower()

        if method == 'historical':
            # Historical VaR: Use empirical percentile
            # For 95% confidence, we want the 5th percentile (left tail)
            percentile = (1 - confidence_level) * 100
            var = float(np.percentile(returns, percentile))

        elif method == 'parametric':
            # Parametric VaR: Assume normal distribution
            # VaR = mu - z_alpha * sigma
            mean_return = returns.mean()
            std_return = returns.std()

            # Get z-score for confidence level
            if confidence_level in Z_SCORES:
                z_score = Z_SCORES[confidence_level]
            else:
                # Calculate z-score from standard normal
                z_score = stats.norm.ppf(confidence_level)

            var = float(mean_return - z_score * std_return)

        return var

    @staticmethod
    def calculate_cvar(
        returns: pd.Series,
        confidence_level: float = 0.95
    ) -> float:
        """
        Calculate Conditional VaR (Expected Shortfall).

        CVaR represents the expected loss given that the loss exceeds VaR.
        It is a more conservative risk measure than VaR and is considered
        a "coherent" risk measure (satisfies subadditivity).

        Formula:
            CVaR_alpha = E[R | R <= VaR_alpha]
            = Average of all returns worse than VaR

        Args:
            returns: Series of period returns
            confidence_level: Confidence level (e.g., 0.95 for 95% CVaR)

        Returns:
            CVaR as a decimal (negative number representing expected loss
            in the tail). Always less than or equal to VaR (more negative).

        Raises:
            InsufficientDataError: If insufficient data points
            InvalidConfidenceError: If confidence level not in (0, 1)

        Note:
            CVaR is also called:
            - Expected Shortfall (ES)
            - Average Value at Risk (AVaR)
            - Tail Value at Risk (TVaR)

            CVaR is always more conservative than VaR (larger loss).

        Example:
            >>> cvar_95 = RiskAnalytics.calculate_cvar(returns, 0.95)
            >>> print(f"95% CVaR: {cvar_95:.2%}")
            >>> # If cvar_95 = -0.04, expected loss is 4% when VaR is exceeded

        Reference:
            Acerbi, C. & Tasche, D. (2002). Expected Shortfall.
        """
        # Validate confidence level
        if not 0 < confidence_level < 1:
            raise InvalidConfidenceError(
                f"Confidence level must be between 0 and 1, got {confidence_level}"
            )

        # Convert to Series if needed
        if not isinstance(returns, pd.Series):
            returns = pd.Series(returns)

        # Remove NaN values
        returns = returns.dropna()

        if returns.empty:
            raise InvalidDataError("Returns series is empty after removing NaN values")

        if len(returns) < MIN_OBSERVATIONS_FOR_VAR:
            logger.warning(
                f"CVaR calculated with only {len(returns)} observations. "
                f"Recommend at least {MIN_OBSERVATIONS_FOR_VAR} for reliability."
            )

        # Calculate VaR first (using historical method)
        var = RiskAnalytics.calculate_var(returns, confidence_level, 'historical')

        # CVaR is the mean of returns below VaR threshold
        tail_returns = returns[returns <= var]

        if tail_returns.empty:
            # No returns below VaR - use VaR as CVaR
            logger.warning(
                "No returns below VaR threshold. Using VaR as CVaR."
            )
            return var

        cvar = float(tail_returns.mean())

        return cvar

    @staticmethod
    def calculate_var_breach_count(
        returns: pd.Series,
        var_threshold: float
    ) -> Dict[str, Any]:
        """
        Count VaR breaches (days when return exceeded VaR).

        Used for backtesting VaR models. The number of breaches should
        be consistent with the confidence level.

        Args:
            returns: Series of period returns
            var_threshold: VaR threshold (negative number)

        Returns:
            Dictionary containing:
                - 'breach_count': Number of VaR breaches
                - 'total_observations': Total number of observations
                - 'breach_rate': Percentage of breaches
                - 'breach_dates': Index values where breaches occurred

        Example:
            >>> var_95 = RiskAnalytics.calculate_var(returns, 0.95)
            >>> breaches = RiskAnalytics.calculate_var_breach_count(returns, var_95)
            >>> print(f"Breach Rate: {breaches['breach_rate']:.1%}")
            >>> # For 95% VaR, expect ~5% breach rate
        """
        if not isinstance(returns, pd.Series):
            returns = pd.Series(returns)

        returns = returns.dropna()

        if returns.empty:
            raise InvalidDataError("Returns series is empty")

        # Count breaches (returns worse than VaR)
        breaches = returns < var_threshold
        breach_count = int(breaches.sum())
        total = len(returns)
        breach_rate = breach_count / total if total > 0 else 0.0

        # Get breach dates/indices
        breach_dates = returns.index[breaches].tolist()

        return {
            'breach_count': breach_count,
            'total_observations': total,
            'breach_rate': breach_rate,
            'breach_dates': breach_dates,
        }

    @staticmethod
    def calculate_marginal_var(
        portfolio_returns: pd.Series,
        position_returns: pd.Series,
        confidence_level: float = 0.95
    ) -> float:
        """
        Calculate Marginal VaR for a position.

        Marginal VaR measures the change in portfolio VaR per unit change
        in position size.

        Args:
            portfolio_returns: Series of portfolio returns
            position_returns: Series of returns for the specific position
            confidence_level: Confidence level

        Returns:
            Marginal VaR

        Example:
            >>> mvar = RiskAnalytics.calculate_marginal_var(
            ...     portfolio_returns, position_returns, 0.95
            ... )
        """
        # Simple implementation: beta * portfolio VaR / portfolio value
        # More sophisticated: use delta-VaR approach

        port = portfolio_returns.dropna()
        pos = position_returns.dropna()

        # Align returns
        aligned = pd.concat([port, pos], axis=1).dropna()
        if len(aligned) < MIN_OBSERVATIONS_FOR_VAR:
            raise InsufficientDataError(
                f"Need at least {MIN_OBSERVATIONS_FOR_VAR} aligned observations"
            )

        port_aligned = aligned.iloc[:, 0]
        pos_aligned = aligned.iloc[:, 1]

        # Calculate beta
        covariance = np.cov(port_aligned, pos_aligned)[0, 1]
        portfolio_variance = np.var(port_aligned)

        if portfolio_variance < EPSILON:
            return 0.0

        beta = covariance / portfolio_variance

        # Portfolio VaR
        port_var = RiskAnalytics.calculate_var(port_aligned, confidence_level)

        # Marginal VaR
        marginal_var = beta * port_var

        return float(marginal_var)

    # =========================================================================
    # Greeks Analysis
    # =========================================================================

    @staticmethod
    def analyze_greeks_over_time(
        greeks_history: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Analyze Greeks evolution over time.

        Provides statistical summary and correlation analysis of portfolio
        Greeks throughout the backtest period.

        Args:
            greeks_history: DataFrame with columns for each Greek
                           (delta, gamma, theta, vega, rho) and datetime index

        Returns:
            Dictionary containing:
                - 'delta': Statistics dict for delta
                - 'gamma': Statistics dict for gamma
                - 'theta': Statistics dict for theta
                - 'vega': Statistics dict for vega
                - 'rho': Statistics dict for rho
                - 'correlation_matrix': DataFrame of Greek correlations
                - 'summary': Overall summary statistics

            Each Greek's statistics dict contains:
                - 'mean': Average value
                - 'std': Standard deviation
                - 'min': Minimum value
                - 'max': Maximum value
                - 'range': Max - Min
                - 'median': Median value
                - 'current': Most recent value

        Raises:
            InvalidDataError: If greeks_history is empty or missing columns

        Example:
            >>> greeks_analysis = RiskAnalytics.analyze_greeks_over_time(greeks_history)
            >>> print(f"Average Delta: {greeks_analysis['delta']['mean']:.4f}")
            >>> print(greeks_analysis['correlation_matrix'])
        """
        if greeks_history.empty:
            raise InvalidDataError("Greeks history is empty")

        result = {}

        # Analyze each Greek
        for greek in GREEK_NAMES:
            if greek not in greeks_history.columns:
                logger.warning(f"Greek '{greek}' not found in history")
                result[greek] = {
                    'mean': 0.0,
                    'std': 0.0,
                    'min': 0.0,
                    'max': 0.0,
                    'range': 0.0,
                    'median': 0.0,
                    'current': 0.0,
                }
                continue

            values = greeks_history[greek].dropna()

            if values.empty:
                result[greek] = {
                    'mean': 0.0,
                    'std': 0.0,
                    'min': 0.0,
                    'max': 0.0,
                    'range': 0.0,
                    'median': 0.0,
                    'current': 0.0,
                }
                continue

            result[greek] = {
                'mean': float(values.mean()),
                'std': float(values.std()),
                'min': float(values.min()),
                'max': float(values.max()),
                'range': float(values.max() - values.min()),
                'median': float(values.median()),
                'current': float(values.iloc[-1]),
                'count': len(values),
            }

        # Calculate correlation matrix
        available_greeks = [g for g in GREEK_NAMES if g in greeks_history.columns]

        if len(available_greeks) >= 2:
            greek_data = greeks_history[available_greeks].dropna()
            if len(greek_data) >= 2:
                correlation_matrix = greek_data.corr()
            else:
                correlation_matrix = pd.DataFrame()
        else:
            correlation_matrix = pd.DataFrame()

        result['correlation_matrix'] = correlation_matrix

        # Calculate summary statistics
        result['summary'] = {
            'num_observations': len(greeks_history),
            'start_date': greeks_history.index[0] if len(greeks_history) > 0 else None,
            'end_date': greeks_history.index[-1] if len(greeks_history) > 0 else None,
            'greeks_available': available_greeks,
        }

        return result

    @staticmethod
    def calculate_greeks_pnl_attribution(
        greeks_history: pd.DataFrame,
        returns: pd.Series,
        spot_changes: Optional[pd.Series] = None,
        vol_changes: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """
        Estimate P&L attribution to each Greek.

        This is a simplified attribution that estimates how much of the
        P&L can be explained by each Greek exposure.

        Args:
            greeks_history: DataFrame with Greeks over time
            returns: Series of portfolio returns
            spot_changes: Optional series of underlying price changes
            vol_changes: Optional series of implied volatility changes

        Returns:
            Dictionary with estimated P&L contribution from each Greek

        Note:
            This is an approximation. True P&L attribution requires
            integration of Greeks over the path.
        """
        attribution = {
            'delta_pnl': 0.0,
            'gamma_pnl': 0.0,
            'theta_pnl': 0.0,
            'vega_pnl': 0.0,
            'unexplained': 0.0,
        }

        if greeks_history.empty or returns.empty:
            return attribution

        total_return = returns.sum()

        # If spot and vol changes provided, estimate attribution
        if spot_changes is not None and 'delta' in greeks_history.columns:
            delta_contrib = (greeks_history['delta'].shift(1) * spot_changes).sum()
            attribution['delta_pnl'] = float(delta_contrib)

        if spot_changes is not None and 'gamma' in greeks_history.columns:
            gamma_contrib = 0.5 * (greeks_history['gamma'].shift(1) * np.square(spot_changes)).sum()
            attribution['gamma_pnl'] = float(gamma_contrib)

        if 'theta' in greeks_history.columns:
            # Theta accrues daily
            theta_contrib = greeks_history['theta'].sum()
            attribution['theta_pnl'] = float(theta_contrib)

        if vol_changes is not None and 'vega' in greeks_history.columns:
            vega_contrib = (greeks_history['vega'].shift(1) * vol_changes).sum()
            attribution['vega_pnl'] = float(vega_contrib)

        explained = sum([
            attribution['delta_pnl'],
            attribution['gamma_pnl'],
            attribution['theta_pnl'],
            attribution['vega_pnl'],
        ])

        attribution['unexplained'] = total_return - explained

        return attribution

    # =========================================================================
    # Tail Risk Metrics
    # =========================================================================

    @staticmethod
    def calculate_tail_risk(returns: pd.Series) -> Dict[str, float]:
        """
        Calculate tail risk metrics.

        Provides measures of distribution asymmetry and tail fatness
        that are important for options strategies.

        Args:
            returns: Series of period returns

        Returns:
            Dictionary containing:
                - 'skewness': Asymmetry measure (negative = left tail heavier)
                - 'kurtosis': Excess kurtosis (0 = normal, positive = fat tails)
                - 'left_tail_ratio': Proportion of returns < -2 sigma
                - 'right_tail_ratio': Proportion of returns > +2 sigma
                - 'tail_ratio': Left tail / Right tail ratio
                - 'jarque_bera_stat': JB test statistic for normality
                - 'jarque_bera_pvalue': p-value for JB test

        Note:
            - Negative skewness is common for short premium strategies
            - High kurtosis indicates fat tails (more extreme events)
            - Options sellers typically have negative skewness

        Example:
            >>> tail_risk = RiskAnalytics.calculate_tail_risk(returns)
            >>> print(f"Skewness: {tail_risk['skewness']:.2f}")
            >>> print(f"Kurtosis: {tail_risk['kurtosis']:.2f}")
        """
        if not isinstance(returns, pd.Series):
            returns = pd.Series(returns)

        returns = returns.dropna()

        if len(returns) < 4:
            raise InsufficientDataError(
                f"Need at least 4 observations for tail risk, got {len(returns)}"
            )

        mean_ret = returns.mean()
        std_ret = returns.std()

        # Calculate moments
        skewness = float(stats.skew(returns))

        # Excess kurtosis (normal = 0)
        kurtosis = float(stats.kurtosis(returns, fisher=True))

        # Calculate tail ratios (proportion beyond +/- 2 sigma)
        if std_ret > EPSILON:
            left_threshold = mean_ret - SIGMA_THRESHOLD * std_ret
            right_threshold = mean_ret + SIGMA_THRESHOLD * std_ret

            left_tail_count = (returns < left_threshold).sum()
            right_tail_count = (returns > right_threshold).sum()

            left_tail_ratio = left_tail_count / len(returns)
            right_tail_ratio = right_tail_count / len(returns)

            # Tail ratio: left/right
            if right_tail_count > 0:
                tail_ratio = left_tail_count / right_tail_count
            else:
                tail_ratio = float(np.inf) if left_tail_count > 0 else 1.0
        else:
            left_tail_ratio = 0.0
            right_tail_ratio = 0.0
            tail_ratio = 1.0

        # Jarque-Bera test for normality
        if len(returns) >= 8:
            jb_stat, jb_pvalue = stats.jarque_bera(returns)
        else:
            jb_stat = np.nan
            jb_pvalue = np.nan

        return {
            'skewness': skewness,
            'kurtosis': kurtosis,
            'left_tail_ratio': float(left_tail_ratio),
            'right_tail_ratio': float(right_tail_ratio),
            'tail_ratio': float(tail_ratio),
            'jarque_bera_stat': float(jb_stat),
            'jarque_bera_pvalue': float(jb_pvalue),
            'is_normal': float(jb_pvalue) > 0.05 if not np.isnan(jb_pvalue) else None,
        }

    @staticmethod
    def calculate_downside_risk(
        returns: pd.Series,
        target: float = 0.0
    ) -> Dict[str, float]:
        """
        Calculate downside risk metrics.

        Focuses on negative returns relative to a target (usually 0).

        Args:
            returns: Series of period returns
            target: Target return (default 0)

        Returns:
            Dictionary containing:
                - 'downside_deviation': Std of returns below target
                - 'downside_variance': Variance of returns below target
                - 'loss_probability': Probability of returns below target
                - 'average_loss': Average return when below target
                - 'worst_loss': Minimum return

        Example:
            >>> dr = RiskAnalytics.calculate_downside_risk(returns)
            >>> print(f"Downside Deviation: {dr['downside_deviation']:.4f}")
        """
        if not isinstance(returns, pd.Series):
            returns = pd.Series(returns)

        returns = returns.dropna()

        if returns.empty:
            raise InvalidDataError("Returns series is empty")

        # Calculate downside deviation using semi-variance approach
        downside_returns = np.minimum(returns - target, 0)
        downside_variance = float(np.mean(np.square(downside_returns)))
        downside_deviation = float(np.sqrt(downside_variance))

        # Loss statistics
        losses = returns[returns < target]
        loss_probability = len(losses) / len(returns) if len(returns) > 0 else 0.0
        average_loss = float(losses.mean()) if len(losses) > 0 else 0.0
        worst_loss = float(returns.min())

        return {
            'downside_deviation': downside_deviation,
            'downside_variance': downside_variance,
            'loss_probability': float(loss_probability),
            'average_loss': average_loss,
            'worst_loss': worst_loss,
            'loss_count': len(losses),
            'total_observations': len(returns),
        }

    # =========================================================================
    # Margin Analysis
    # =========================================================================

    @staticmethod
    def calculate_margin_utilization(
        equity_curve: pd.DataFrame,
        margin_history: pd.Series
    ) -> Dict[str, Any]:
        """
        Calculate margin usage statistics.

        Margin utilization measures how much of available capital is
        being used as margin for positions.

        Args:
            equity_curve: DataFrame with 'equity' column
            margin_history: Series of margin requirements over time

        Returns:
            Dictionary containing:
                - 'avg_utilization_pct': Average margin/equity ratio
                - 'max_utilization_pct': Maximum margin/equity ratio
                - 'min_utilization_pct': Minimum margin/equity ratio
                - 'utilization_series': Series of utilization percentages
                - 'high_utilization_days': Days with >80% utilization
                - 'margin_call_risk': Days with >100% utilization

        Note:
            High margin utilization increases risk of margin calls.
            Typically want to stay below 50-70% utilization.

        Example:
            >>> margin_stats = RiskAnalytics.calculate_margin_utilization(
            ...     equity_curve, margin_history
            ... )
            >>> print(f"Avg Utilization: {margin_stats['avg_utilization_pct']:.1%}")
        """
        # Get equity series
        if isinstance(equity_curve, pd.DataFrame):
            if 'equity' not in equity_curve.columns:
                raise InvalidDataError("DataFrame must contain 'equity' column")
            equity = equity_curve['equity']
        else:
            equity = equity_curve

        if equity.empty or margin_history.empty:
            raise InvalidDataError("Equity or margin history is empty")

        # Align data
        aligned = pd.concat([equity, margin_history], axis=1).dropna()
        if aligned.empty:
            raise InvalidDataError("No aligned data between equity and margin")

        equity_aligned = aligned.iloc[:, 0]
        margin_aligned = aligned.iloc[:, 1]

        # Calculate utilization
        # Avoid division by zero
        utilization = np.where(
            equity_aligned > EPSILON,
            margin_aligned / equity_aligned,
            0.0
        )
        utilization_series = pd.Series(utilization, index=aligned.index)

        # Calculate statistics
        avg_utilization = float(utilization_series.mean())
        max_utilization = float(utilization_series.max())
        min_utilization = float(utilization_series.min())

        # Risk metrics
        high_util_threshold = 0.80
        margin_call_threshold = 1.00

        high_util_days = (utilization_series > high_util_threshold).sum()
        margin_call_days = (utilization_series > margin_call_threshold).sum()

        return {
            'avg_utilization_pct': avg_utilization,
            'max_utilization_pct': max_utilization,
            'min_utilization_pct': min_utilization,
            'utilization_series': utilization_series,
            'high_utilization_days': int(high_util_days),
            'margin_call_risk_days': int(margin_call_days),
            'total_days': len(utilization_series),
            'utilization_std': float(utilization_series.std()),
        }

    # =========================================================================
    # Maximum Adverse Excursion (MAE)
    # =========================================================================

    @staticmethod
    def calculate_mae(trades: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Maximum Adverse Excursion for each trade.

        MAE measures the worst intra-trade drawdown for each trade.
        This helps identify if trades are held too long or if stop
        losses could improve performance.

        Args:
            trades: DataFrame with trade records. Should contain:
                   - 'structure_id' or 'trade_id': Trade identifier
                   - 'realized_pnl': Final P&L
                   Additional optional columns for tracking:
                   - 'max_adverse_excursion': Pre-calculated MAE if available
                   - 'entry_timestamp': Entry time
                   - 'exit_timestamp': Exit time

        Returns:
            DataFrame with MAE analysis for each trade:
                - 'trade_id': Trade identifier
                - 'realized_pnl': Final P&L
                - 'mae': Maximum adverse excursion (worst drawdown)
                - 'mae_pct': MAE as percentage of entry value
                - 'mfe': Maximum favorable excursion (best unrealized profit)
                - 'efficiency': realized_pnl / mfe (how much profit captured)

        Note:
            MAE analysis requires intra-trade P&L tracking which may not
            be available in all trade records. If not available, returns
            simplified analysis based on final P&L only.

        Example:
            >>> mae_df = RiskAnalytics.calculate_mae(trade_log)
            >>> # Identify trades where MAE was large but trade ended profitable
            >>> recovered = mae_df[
            ...     (mae_df['mae'] < -100) & (mae_df['realized_pnl'] > 0)
            ... ]
        """
        if trades.empty:
            return pd.DataFrame()

        # Identify trade ID column
        id_col = None
        for col in ['trade_id', 'structure_id']:
            if col in trades.columns:
                id_col = col
                break

        if id_col is None:
            # Create synthetic IDs
            trades = trades.copy()
            trades['trade_id'] = range(len(trades))
            id_col = 'trade_id'

        # Check for P&L column
        if 'realized_pnl' not in trades.columns:
            raise InvalidDataError(
                "trades DataFrame must contain 'realized_pnl' column"
            )

        result_records = []

        for idx, trade in trades.iterrows():
            record = {
                'trade_id': trade[id_col],
                'realized_pnl': trade['realized_pnl'],
            }

            # Check if MAE is pre-calculated
            if 'max_adverse_excursion' in trades.columns:
                record['mae'] = trade['max_adverse_excursion']
            else:
                # Without intra-trade tracking, estimate MAE
                # For losing trades, MAE >= |loss|
                # For winning trades, MAE is unknown (could be 0)
                if trade['realized_pnl'] < 0:
                    record['mae'] = trade['realized_pnl']  # At least this much
                else:
                    record['mae'] = 0.0  # Unknown, assume 0

            # Check for MFE
            if 'max_favorable_excursion' in trades.columns:
                record['mfe'] = trade['max_favorable_excursion']
            else:
                # For winning trades, MFE >= profit
                if trade['realized_pnl'] > 0:
                    record['mfe'] = trade['realized_pnl']
                else:
                    record['mfe'] = 0.0

            # Calculate MAE percentage if entry value available
            if 'total_cost' in trades.columns and abs(trade.get('total_cost', 0)) > EPSILON:
                record['mae_pct'] = record['mae'] / abs(trade['total_cost'])
            else:
                record['mae_pct'] = np.nan

            # Calculate efficiency (profit capture ratio)
            if record['mfe'] > EPSILON:
                record['efficiency'] = trade['realized_pnl'] / record['mfe']
            else:
                record['efficiency'] = np.nan

            result_records.append(record)

        return pd.DataFrame(result_records)

    @staticmethod
    def analyze_mae_statistics(mae_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate aggregate MAE statistics.

        Args:
            mae_df: DataFrame from calculate_mae()

        Returns:
            Dictionary with MAE statistics

        Example:
            >>> mae_df = RiskAnalytics.calculate_mae(trades)
            >>> stats = RiskAnalytics.analyze_mae_statistics(mae_df)
        """
        if mae_df.empty:
            return {}

        stats = {
            'avg_mae': float(mae_df['mae'].mean()) if 'mae' in mae_df else np.nan,
            'max_mae': float(mae_df['mae'].min()) if 'mae' in mae_df else np.nan,
            'avg_mfe': float(mae_df['mfe'].mean()) if 'mfe' in mae_df else np.nan,
            'max_mfe': float(mae_df['mfe'].max()) if 'mfe' in mae_df else np.nan,
        }

        if 'efficiency' in mae_df.columns:
            valid_efficiency = mae_df['efficiency'].dropna()
            if len(valid_efficiency) > 0:
                stats['avg_efficiency'] = float(valid_efficiency.mean())
            else:
                stats['avg_efficiency'] = np.nan

        return stats

    # =========================================================================
    # Summary Methods
    # =========================================================================

    @staticmethod
    def calculate_all_risk_metrics(
        returns: pd.Series,
        greeks_history: Optional[pd.DataFrame] = None,
        trades: Optional[pd.DataFrame] = None,
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive risk metrics summary.

        Convenience method that calculates all available risk metrics
        and returns them in a structured dictionary.

        Args:
            returns: Series of period returns
            greeks_history: Optional DataFrame with Greeks over time
            trades: Optional DataFrame with trade records
            confidence_level: Confidence level for VaR/CVaR

        Returns:
            Dictionary with all calculated metrics organized by category

        Example:
            >>> risk = RiskAnalytics.calculate_all_risk_metrics(
            ...     returns=returns,
            ...     greeks_history=greeks_history,
            ...     trades=trade_log
            ... )
        """
        results = {}

        # VaR metrics
        try:
            results['var_historical'] = RiskAnalytics.calculate_var(
                returns, confidence_level, 'historical'
            )
        except Exception as e:
            logger.warning(f"Could not calculate historical VaR: {e}")
            results['var_historical'] = np.nan

        try:
            results['var_parametric'] = RiskAnalytics.calculate_var(
                returns, confidence_level, 'parametric'
            )
        except Exception as e:
            logger.warning(f"Could not calculate parametric VaR: {e}")
            results['var_parametric'] = np.nan

        try:
            results['cvar'] = RiskAnalytics.calculate_cvar(returns, confidence_level)
        except Exception as e:
            logger.warning(f"Could not calculate CVaR: {e}")
            results['cvar'] = np.nan

        # Tail risk
        try:
            results['tail_risk'] = RiskAnalytics.calculate_tail_risk(returns)
        except Exception as e:
            logger.warning(f"Could not calculate tail risk: {e}")
            results['tail_risk'] = {}

        # Downside risk
        try:
            results['downside_risk'] = RiskAnalytics.calculate_downside_risk(returns)
        except Exception as e:
            logger.warning(f"Could not calculate downside risk: {e}")
            results['downside_risk'] = {}

        # Greeks analysis
        if greeks_history is not None and not greeks_history.empty:
            try:
                results['greeks_analysis'] = RiskAnalytics.analyze_greeks_over_time(
                    greeks_history
                )
            except Exception as e:
                logger.warning(f"Could not analyze Greeks: {e}")
                results['greeks_analysis'] = {}

        # MAE analysis
        if trades is not None and not trades.empty:
            try:
                mae_df = RiskAnalytics.calculate_mae(trades)
                results['mae_stats'] = RiskAnalytics.analyze_mae_statistics(mae_df)
            except Exception as e:
                logger.warning(f"Could not calculate MAE: {e}")
                results['mae_stats'] = {}

        results['confidence_level'] = confidence_level

        return results


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Main class
    'RiskAnalytics',

    # Exceptions
    'RiskAnalyticsError',
    'InsufficientDataError',
    'InvalidDataError',
    'InvalidConfidenceError',

    # Constants
    'VAR_CONFIDENCE_95',
    'VAR_CONFIDENCE_99',
    'Z_SCORES',
    'GREEK_NAMES',
]
