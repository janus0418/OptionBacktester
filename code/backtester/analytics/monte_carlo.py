"""
Monte Carlo Simulation Module for Options Backtesting Analytics

This module provides Monte Carlo simulation capabilities for analyzing
strategy performance under various market conditions and estimating
risk metrics with statistical confidence.

Key Features:
    - Return path simulation using historical distribution parameters
    - Bootstrap resampling of historical returns
    - Confidence intervals for performance metrics
    - VaR/CVaR estimation via simulation
    - Strategy outcome distribution analysis
    - Parameter sensitivity analysis through simulation

Design Philosophy:
    All methods are static to enable easy use without instantiation.
    Simulations are reproducible via random seed control.
    Methods accept pandas DataFrames/Series and return typed results.

Mathematical Background:
    1. Geometric Brownian Motion (GBM):
       dS = mu*S*dt + sigma*S*dW
       S(t) = S(0) * exp((mu - 0.5*sigma^2)*t + sigma*W(t))

    2. Bootstrap Resampling:
       Draw with replacement from historical returns to generate
       synthetic return paths that preserve empirical distribution.

    3. Block Bootstrap:
       Preserve autocorrelation by resampling blocks of consecutive
       returns rather than individual observations.

Usage:
    from backtester.analytics.monte_carlo import MonteCarloSimulator

    # Simulate future returns
    simulated_paths = MonteCarloSimulator.simulate_returns_gbm(
        returns, num_simulations=10000, num_periods=252
    )

    # Calculate confidence intervals
    ci = MonteCarloSimulator.calculate_confidence_intervals(
        simulated_paths, metrics=['total_return', 'max_drawdown']
    )

    # Estimate VaR via simulation
    var_sim = MonteCarloSimulator.estimate_var_monte_carlo(
        simulated_paths, confidence_level=0.95
    )

References:
    - Glasserman, P. (2003). Monte Carlo Methods in Financial Engineering.
    - Efron, B. & Tibshirani, R. (1993). An Introduction to the Bootstrap.
    - Hull, J.C. (2018). Options, Futures, and Other Derivatives.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

# Configure module logger
logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Default simulation parameters
DEFAULT_NUM_SIMULATIONS = 10000
DEFAULT_NUM_PERIODS = 252  # One trading year
DEFAULT_CONFIDENCE_LEVEL = 0.95

# Random seed for reproducibility
DEFAULT_RANDOM_SEED = 42

# Numerical tolerance
EPSILON = 1e-10

# Minimum observations for reliable statistics
MIN_OBSERVATIONS_FOR_SIMULATION = 30

# Trading days per year
TRADING_DAYS_PER_YEAR = 252

# Confidence interval levels
CI_LEVELS = [0.90, 0.95, 0.99]


# =============================================================================
# Exceptions
# =============================================================================


class MonteCarloError(Exception):
    """Base exception for Monte Carlo simulation errors."""

    pass


class InsufficientDataError(MonteCarloError):
    """Exception raised when there is insufficient data for simulation."""

    pass


class InvalidParameterError(MonteCarloError):
    """Exception raised when simulation parameters are invalid."""

    pass


class SimulationError(MonteCarloError):
    """Exception raised when simulation fails."""

    pass


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class SimulationResult:
    """Container for Monte Carlo simulation results."""

    simulated_paths: np.ndarray  # Shape: (num_simulations, num_periods)
    final_values: np.ndarray  # Shape: (num_simulations,)
    parameters: Dict[str, Any]
    statistics: Dict[str, float]


@dataclass
class ConfidenceInterval:
    """Container for confidence interval results."""

    metric_name: str
    point_estimate: float
    lower_bound: float
    upper_bound: float
    confidence_level: float
    standard_error: float


# =============================================================================
# MonteCarloSimulator Class
# =============================================================================


class MonteCarloSimulator:
    """
    Monte Carlo simulation for strategy analysis.

    This class provides static methods for running Monte Carlo simulations
    to analyze strategy performance and estimate risk metrics with
    statistical confidence.

    Simulation Methods:
        1. GBM (Geometric Brownian Motion): Parametric simulation assuming
           log-normal returns with constant drift and volatility.
        2. Bootstrap: Non-parametric resampling from historical returns.
        3. Block Bootstrap: Resampling that preserves autocorrelation.

    All methods are designed to work with pandas DataFrames and Series,
    and support reproducibility through random seed control.

    Example:
        >>> returns = equity['equity'].pct_change().dropna()
        >>> paths = MonteCarloSimulator.simulate_returns_bootstrap(
        ...     returns, num_simulations=10000, num_periods=252
        ... )
        >>> ci = MonteCarloSimulator.calculate_metric_confidence_interval(
        ...     paths, metric_func=lambda x: (x[-1]/x[0] - 1), confidence=0.95
        ... )
    """

    # =========================================================================
    # Parametric Simulation (GBM)
    # =========================================================================

    @staticmethod
    def simulate_returns_gbm(
        returns: pd.Series,
        num_simulations: int = DEFAULT_NUM_SIMULATIONS,
        num_periods: int = DEFAULT_NUM_PERIODS,
        initial_value: float = 1.0,
        random_seed: Optional[int] = DEFAULT_RANDOM_SEED,
    ) -> SimulationResult:
        """
        Simulate return paths using Geometric Brownian Motion.

        Uses the historical returns to estimate drift (mu) and volatility
        (sigma), then generates synthetic price paths assuming log-normal
        returns.

        Formula:
            S(t+dt) = S(t) * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
            where Z ~ N(0,1)

        Args:
            returns: Series of historical period returns
            num_simulations: Number of simulation paths to generate
            num_periods: Number of periods per path
            initial_value: Starting value for each path (default 1.0)
            random_seed: Random seed for reproducibility (None for random)

        Returns:
            SimulationResult containing:
                - simulated_paths: Array of shape (num_simulations, num_periods+1)
                - final_values: Array of final path values
                - parameters: Dict with mu, sigma, etc.
                - statistics: Dict with mean, std, percentiles

        Raises:
            InsufficientDataError: If insufficient historical data
            InvalidParameterError: If parameters are invalid

        Example:
            >>> result = MonteCarloSimulator.simulate_returns_gbm(
            ...     returns, num_simulations=10000, num_periods=252
            ... )
            >>> print(f"Mean final value: {result.statistics['mean']:.4f}")
        """
        # Validate inputs
        if not isinstance(returns, pd.Series):
            returns = pd.Series(returns)

        returns = returns.dropna()

        if len(returns) < MIN_OBSERVATIONS_FOR_SIMULATION:
            raise InsufficientDataError(
                f"Need at least {MIN_OBSERVATIONS_FOR_SIMULATION} observations, "
                f"got {len(returns)}"
            )

        if num_simulations < 1:
            raise InvalidParameterError(
                f"num_simulations must be positive, got {num_simulations}"
            )

        if num_periods < 1:
            raise InvalidParameterError(
                f"num_periods must be positive, got {num_periods}"
            )

        if initial_value <= 0:
            raise InvalidParameterError(
                f"initial_value must be positive, got {initial_value}"
            )

        # Set random seed for reproducibility
        if random_seed is not None:
            np.random.seed(random_seed)

        # Estimate parameters from historical returns
        # Using log returns for GBM
        log_returns = np.log(1 + returns)
        mu = float(log_returns.mean())  # Daily drift
        sigma = float(log_returns.std())  # Daily volatility

        # Handle zero volatility
        if sigma < EPSILON:
            logger.warning("Near-zero volatility detected. Using minimum volatility.")
            sigma = EPSILON

        # Generate random shocks
        # Shape: (num_simulations, num_periods)
        Z = np.random.standard_normal((num_simulations, num_periods))

        # Calculate log returns for each path
        # S(t+1)/S(t) = exp(drift + vol*Z) where drift = mu - 0.5*sigma^2
        drift_per_period = mu - 0.5 * sigma**2
        simulated_log_returns = drift_per_period + sigma * Z

        # Calculate cumulative log returns
        cumulative_log_returns = np.cumsum(simulated_log_returns, axis=1)

        # Prepend zeros for initial value
        cumulative_log_returns = np.hstack(
            [np.zeros((num_simulations, 1)), cumulative_log_returns]
        )

        # Convert to price paths
        simulated_paths = initial_value * np.exp(cumulative_log_returns)

        # Extract final values
        final_values = simulated_paths[:, -1]

        # Calculate statistics
        statistics = {
            "mean": float(np.mean(final_values)),
            "median": float(np.median(final_values)),
            "std": float(np.std(final_values)),
            "min": float(np.min(final_values)),
            "max": float(np.max(final_values)),
            "percentile_5": float(np.percentile(final_values, 5)),
            "percentile_25": float(np.percentile(final_values, 25)),
            "percentile_75": float(np.percentile(final_values, 75)),
            "percentile_95": float(np.percentile(final_values, 95)),
            "skewness": float(stats.skew(final_values)),
            "kurtosis": float(stats.kurtosis(final_values)),
        }

        # Store parameters
        parameters = {
            "method": "gbm",
            "mu": mu,
            "sigma": sigma,
            "num_simulations": num_simulations,
            "num_periods": num_periods,
            "initial_value": initial_value,
            "random_seed": random_seed,
            "annualized_return": mu * TRADING_DAYS_PER_YEAR,
            "annualized_volatility": sigma * np.sqrt(TRADING_DAYS_PER_YEAR),
        }

        return SimulationResult(
            simulated_paths=simulated_paths,
            final_values=final_values,
            parameters=parameters,
            statistics=statistics,
        )

    # =========================================================================
    # Bootstrap Simulation
    # =========================================================================

    @staticmethod
    def simulate_returns_bootstrap(
        returns: pd.Series,
        num_simulations: int = DEFAULT_NUM_SIMULATIONS,
        num_periods: int = DEFAULT_NUM_PERIODS,
        initial_value: float = 1.0,
        random_seed: Optional[int] = DEFAULT_RANDOM_SEED,
    ) -> SimulationResult:
        """
        Simulate return paths using bootstrap resampling.

        Draws returns with replacement from the historical distribution,
        preserving the empirical distribution including fat tails and
        skewness.

        This is a non-parametric approach that makes no assumptions about
        the underlying return distribution.

        Args:
            returns: Series of historical period returns
            num_simulations: Number of simulation paths to generate
            num_periods: Number of periods per path
            initial_value: Starting value for each path (default 1.0)
            random_seed: Random seed for reproducibility

        Returns:
            SimulationResult with simulated paths and statistics

        Note:
            Bootstrap resampling does not preserve autocorrelation.
            For strategies with autocorrelated returns, use block bootstrap.

        Example:
            >>> result = MonteCarloSimulator.simulate_returns_bootstrap(
            ...     returns, num_simulations=10000, num_periods=252
            ... )
        """
        # Validate inputs
        if not isinstance(returns, pd.Series):
            returns = pd.Series(returns)

        returns = returns.dropna()
        returns_array = np.asarray(returns.values, dtype=np.float64)

        if len(returns) < MIN_OBSERVATIONS_FOR_SIMULATION:
            raise InsufficientDataError(
                f"Need at least {MIN_OBSERVATIONS_FOR_SIMULATION} observations, "
                f"got {len(returns)}"
            )

        if num_simulations < 1:
            raise InvalidParameterError(
                f"num_simulations must be positive, got {num_simulations}"
            )

        if num_periods < 1:
            raise InvalidParameterError(
                f"num_periods must be positive, got {num_periods}"
            )

        # Set random seed
        if random_seed is not None:
            np.random.seed(random_seed)

        # Generate random indices for resampling
        # Shape: (num_simulations, num_periods)
        indices = np.random.randint(
            0, len(returns_array), size=(num_simulations, num_periods)
        )

        # Resample returns
        simulated_returns = returns_array[indices]

        # Convert returns to price paths
        # P(t+1) = P(t) * (1 + r(t))
        cumulative_returns = np.cumprod(1.0 + simulated_returns, axis=1)

        # Prepend initial value
        simulated_paths = np.hstack(
            [
                np.full((num_simulations, 1), initial_value),
                initial_value * cumulative_returns,
            ]
        )

        # Extract final values
        final_values = simulated_paths[:, -1]

        # Calculate statistics
        statistics = {
            "mean": float(np.mean(final_values)),
            "median": float(np.median(final_values)),
            "std": float(np.std(final_values)),
            "min": float(np.min(final_values)),
            "max": float(np.max(final_values)),
            "percentile_5": float(np.percentile(final_values, 5)),
            "percentile_25": float(np.percentile(final_values, 25)),
            "percentile_75": float(np.percentile(final_values, 75)),
            "percentile_95": float(np.percentile(final_values, 95)),
            "skewness": float(stats.skew(final_values)),
            "kurtosis": float(stats.kurtosis(final_values)),
        }

        # Store parameters
        parameters = {
            "method": "bootstrap",
            "historical_mean": float(returns.mean()),
            "historical_std": float(returns.std()),
            "historical_skew": float(stats.skew(returns)),
            "num_simulations": num_simulations,
            "num_periods": num_periods,
            "initial_value": initial_value,
            "random_seed": random_seed,
            "sample_size": len(returns),
        }

        return SimulationResult(
            simulated_paths=simulated_paths,
            final_values=final_values,
            parameters=parameters,
            statistics=statistics,
        )

    @staticmethod
    def simulate_returns_block_bootstrap(
        returns: pd.Series,
        num_simulations: int = DEFAULT_NUM_SIMULATIONS,
        num_periods: int = DEFAULT_NUM_PERIODS,
        block_size: int = 20,
        initial_value: float = 1.0,
        random_seed: Optional[int] = DEFAULT_RANDOM_SEED,
    ) -> SimulationResult:
        """
        Simulate return paths using block bootstrap resampling.

        Resamples blocks of consecutive returns to preserve autocorrelation
        structure in the data. Useful for strategies where returns exhibit
        serial correlation (trending or mean-reverting behavior).

        Args:
            returns: Series of historical period returns
            num_simulations: Number of simulation paths to generate
            num_periods: Number of periods per path
            block_size: Size of blocks to resample (default 20 ~ 1 month)
            initial_value: Starting value for each path
            random_seed: Random seed for reproducibility

        Returns:
            SimulationResult with simulated paths and statistics

        Note:
            Block size should be chosen based on the autocorrelation
            structure of the returns. Common choices:
            - 5 days (1 week)
            - 20 days (1 month)
            - 63 days (1 quarter)

        Example:
            >>> result = MonteCarloSimulator.simulate_returns_block_bootstrap(
            ...     returns, num_simulations=10000, block_size=20
            ... )
        """
        # Validate inputs
        if not isinstance(returns, pd.Series):
            returns = pd.Series(returns)

        returns = returns.dropna()
        returns_array = np.asarray(returns.values, dtype=np.float64)
        n = len(returns_array)

        if n < MIN_OBSERVATIONS_FOR_SIMULATION:
            raise InsufficientDataError(
                f"Need at least {MIN_OBSERVATIONS_FOR_SIMULATION} observations, got {n}"
            )

        if block_size < 1:
            raise InvalidParameterError(
                f"block_size must be positive, got {block_size}"
            )

        if block_size > n:
            logger.warning(
                f"Block size ({block_size}) exceeds data length ({n}). "
                f"Reducing block size to {n // 2}."
            )
            block_size = max(1, n // 2)

        # Set random seed
        if random_seed is not None:
            np.random.seed(random_seed)

        # Number of blocks needed per simulation
        num_blocks_needed = int(np.ceil(num_periods / block_size))

        # Generate all simulated paths
        simulated_returns_list = []

        for _ in range(num_simulations):
            # Select random starting indices for blocks
            # Ensure we can extract full blocks
            max_start_idx = n - block_size
            if max_start_idx < 0:
                max_start_idx = 0

            block_starts = np.random.randint(
                0, max_start_idx + 1, size=num_blocks_needed
            )

            # Extract and concatenate blocks
            path_returns = []
            for start in block_starts:
                end = min(start + block_size, n)
                path_returns.extend(returns_array[start:end])

            # Trim to exact length
            path_returns = np.array(path_returns[:num_periods])
            simulated_returns_list.append(path_returns)

        simulated_returns = np.array(simulated_returns_list)

        # Convert returns to price paths
        cumulative_returns = np.cumprod(1 + simulated_returns, axis=1)

        simulated_paths = np.hstack(
            [
                np.full((num_simulations, 1), initial_value),
                initial_value * cumulative_returns,
            ]
        )

        # Extract final values
        final_values = simulated_paths[:, -1]

        # Calculate statistics
        statistics = {
            "mean": float(np.mean(final_values)),
            "median": float(np.median(final_values)),
            "std": float(np.std(final_values)),
            "min": float(np.min(final_values)),
            "max": float(np.max(final_values)),
            "percentile_5": float(np.percentile(final_values, 5)),
            "percentile_25": float(np.percentile(final_values, 25)),
            "percentile_75": float(np.percentile(final_values, 75)),
            "percentile_95": float(np.percentile(final_values, 95)),
            "skewness": float(stats.skew(final_values)),
            "kurtosis": float(stats.kurtosis(final_values)),
        }

        # Store parameters
        parameters = {
            "method": "block_bootstrap",
            "block_size": block_size,
            "num_blocks_per_path": num_blocks_needed,
            "historical_mean": float(returns.mean()),
            "historical_std": float(returns.std()),
            "num_simulations": num_simulations,
            "num_periods": num_periods,
            "initial_value": initial_value,
            "random_seed": random_seed,
            "sample_size": len(returns),
        }

        return SimulationResult(
            simulated_paths=simulated_paths,
            final_values=final_values,
            parameters=parameters,
            statistics=statistics,
        )

    # =========================================================================
    # Confidence Intervals
    # =========================================================================

    @staticmethod
    def calculate_metric_confidence_interval(
        simulated_paths: np.ndarray,
        metric_func: Callable[[np.ndarray], float],
        confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
        metric_name: str = "metric",
    ) -> ConfidenceInterval:
        """
        Calculate confidence interval for a metric using simulation results.

        Applies the metric function to each simulated path and calculates
        the confidence interval from the resulting distribution.

        Args:
            simulated_paths: Array of shape (num_simulations, num_periods+1)
            metric_func: Function that takes a price path and returns a metric value
            confidence_level: Confidence level (e.g., 0.95 for 95% CI)
            metric_name: Name of the metric for the result

        Returns:
            ConfidenceInterval with bounds and point estimate

        Example:
            >>> # Total return CI
            >>> total_return_func = lambda path: (path[-1] / path[0]) - 1
            >>> ci = MonteCarloSimulator.calculate_metric_confidence_interval(
            ...     simulated_paths, total_return_func, 0.95, 'total_return'
            ... )
            >>> print(f"95% CI: [{ci.lower_bound:.2%}, {ci.upper_bound:.2%}]")
        """
        if not 0 < confidence_level < 1:
            raise InvalidParameterError(
                f"confidence_level must be between 0 and 1, got {confidence_level}"
            )

        # Calculate metric for each path
        metric_values = np.array([metric_func(path) for path in simulated_paths])

        # Remove any NaN or infinite values
        valid_values = metric_values[np.isfinite(metric_values)]

        if len(valid_values) < 100:
            logger.warning(
                f"Only {len(valid_values)} valid metric values. "
                "Confidence interval may be unreliable."
            )

        if len(valid_values) == 0:
            raise SimulationError("No valid metric values calculated")

        # Calculate percentiles for confidence interval
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        point_estimate = float(np.mean(valid_values))
        lower_bound = float(np.percentile(valid_values, lower_percentile))
        upper_bound = float(np.percentile(valid_values, upper_percentile))
        standard_error = float(np.std(valid_values) / np.sqrt(len(valid_values)))

        return ConfidenceInterval(
            metric_name=metric_name,
            point_estimate=point_estimate,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            confidence_level=confidence_level,
            standard_error=standard_error,
        )

    @staticmethod
    def calculate_all_confidence_intervals(
        simulation_result: SimulationResult,
        confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    ) -> Dict[str, ConfidenceInterval]:
        """
        Calculate confidence intervals for common performance metrics.

        Computes CIs for total return, max drawdown, Sharpe ratio, and
        other standard metrics.

        Args:
            simulation_result: Result from simulate_returns_* methods
            confidence_level: Confidence level for all intervals

        Returns:
            Dictionary mapping metric names to ConfidenceInterval objects

        Example:
            >>> result = MonteCarloSimulator.simulate_returns_bootstrap(returns)
            >>> cis = MonteCarloSimulator.calculate_all_confidence_intervals(result)
            >>> for name, ci in cis.items():
            ...     print(f"{name}: [{ci.lower_bound:.4f}, {ci.upper_bound:.4f}]")
        """
        paths = simulation_result.simulated_paths
        results = {}

        # Total Return
        def total_return_func(path):
            return (path[-1] / path[0]) - 1 if path[0] > 0 else 0

        results["total_return"] = (
            MonteCarloSimulator.calculate_metric_confidence_interval(
                paths, total_return_func, confidence_level, "total_return"
            )
        )

        # Maximum Drawdown
        def max_drawdown_func(path):
            running_max = np.maximum.accumulate(path)
            drawdown = (path - running_max) / running_max
            return float(np.min(drawdown))

        results["max_drawdown"] = (
            MonteCarloSimulator.calculate_metric_confidence_interval(
                paths, max_drawdown_func, confidence_level, "max_drawdown"
            )
        )

        # Volatility (annualized)
        def volatility_func(path):
            returns = np.diff(path) / path[:-1]
            return float(np.std(returns) * np.sqrt(TRADING_DAYS_PER_YEAR))

        results["volatility"] = (
            MonteCarloSimulator.calculate_metric_confidence_interval(
                paths, volatility_func, confidence_level, "volatility"
            )
        )

        # Sharpe Ratio (annualized, assuming 0 risk-free rate)
        def sharpe_func(path):
            returns = np.diff(path) / path[:-1]
            if len(returns) < 2:
                return 0.0
            mean_ret = np.mean(returns)
            std_ret = np.std(returns)
            if std_ret < EPSILON:
                return 0.0
            return float((mean_ret / std_ret) * np.sqrt(TRADING_DAYS_PER_YEAR))

        results["sharpe_ratio"] = (
            MonteCarloSimulator.calculate_metric_confidence_interval(
                paths, sharpe_func, confidence_level, "sharpe_ratio"
            )
        )

        # Final Value
        def final_value_func(path):
            return float(path[-1])

        results["final_value"] = (
            MonteCarloSimulator.calculate_metric_confidence_interval(
                paths, final_value_func, confidence_level, "final_value"
            )
        )

        return results

    # =========================================================================
    # VaR/CVaR via Simulation
    # =========================================================================

    @staticmethod
    def estimate_var_monte_carlo(
        simulation_result: SimulationResult,
        confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
        time_horizon: int = 1,
    ) -> Dict[str, float]:
        """
        Estimate Value at Risk using Monte Carlo simulation.

        Uses the simulated return paths to estimate VaR at the specified
        confidence level and time horizon.

        Args:
            simulation_result: Result from simulate_returns_* methods
            confidence_level: Confidence level (e.g., 0.95 for 95% VaR)
            time_horizon: Number of periods for VaR calculation

        Returns:
            Dictionary containing:
                - 'var': Value at Risk (as negative return)
                - 'var_dollar': VaR in dollar terms (per unit invested)
                - 'cvar': Conditional VaR (Expected Shortfall)
                - 'cvar_dollar': CVaR in dollar terms

        Example:
            >>> result = MonteCarloSimulator.simulate_returns_bootstrap(returns)
            >>> var_metrics = MonteCarloSimulator.estimate_var_monte_carlo(
            ...     result, confidence_level=0.95, time_horizon=1
            ... )
            >>> print(f"1-day 95% VaR: {var_metrics['var']:.2%}")
        """
        if not 0 < confidence_level < 1:
            raise InvalidParameterError(
                f"confidence_level must be between 0 and 1, got {confidence_level}"
            )

        paths = simulation_result.simulated_paths
        num_periods = paths.shape[1] - 1

        if time_horizon > num_periods:
            raise InvalidParameterError(
                f"time_horizon ({time_horizon}) exceeds simulation periods ({num_periods})"
            )

        # Calculate returns over the time horizon
        if time_horizon == 1:
            # Single period returns
            horizon_returns = paths[:, 1] / paths[:, 0] - 1
        else:
            # Multi-period returns
            horizon_returns = paths[:, time_horizon] / paths[:, 0] - 1

        # Calculate VaR (percentile of loss distribution)
        # For 95% VaR, we want the 5th percentile of returns
        var_percentile = (1 - confidence_level) * 100
        var = float(np.percentile(horizon_returns, var_percentile))

        # Calculate CVaR (expected shortfall)
        # Average of returns worse than VaR
        tail_returns = horizon_returns[horizon_returns <= var]
        if len(tail_returns) > 0:
            cvar = float(np.mean(tail_returns))
        else:
            cvar = var

        # Calculate dollar amounts (per unit invested)
        initial_value = simulation_result.parameters.get("initial_value", 1.0)
        var_dollar = abs(var) * initial_value
        cvar_dollar = abs(cvar) * initial_value

        return {
            "var": var,
            "var_dollar": var_dollar,
            "cvar": cvar,
            "cvar_dollar": cvar_dollar,
            "confidence_level": confidence_level,
            "time_horizon": time_horizon,
            "num_simulations": len(horizon_returns),
        }

    @staticmethod
    def estimate_var_at_multiple_levels(
        simulation_result: SimulationResult,
        confidence_levels: Optional[List[float]] = None,
        time_horizon: int = 1,
    ) -> pd.DataFrame:
        """
        Estimate VaR at multiple confidence levels.

        Args:
            simulation_result: Result from simulate_returns_* methods
            confidence_levels: List of confidence levels (default: [0.90, 0.95, 0.99])
            time_horizon: Number of periods for VaR calculation

        Returns:
            DataFrame with VaR and CVaR at each confidence level

        Example:
            >>> var_table = MonteCarloSimulator.estimate_var_at_multiple_levels(result)
            >>> print(var_table)
        """
        if confidence_levels is None:
            confidence_levels = CI_LEVELS

        records = []
        for level in confidence_levels:
            var_metrics = MonteCarloSimulator.estimate_var_monte_carlo(
                simulation_result, level, time_horizon
            )
            records.append(
                {
                    "confidence_level": level,
                    "var": var_metrics["var"],
                    "cvar": var_metrics["cvar"],
                    "var_dollar": var_metrics["var_dollar"],
                    "cvar_dollar": var_metrics["cvar_dollar"],
                }
            )

        return pd.DataFrame(records)

    # =========================================================================
    # Distribution Analysis
    # =========================================================================

    @staticmethod
    def analyze_return_distribution(
        simulation_result: SimulationResult,
    ) -> Dict[str, Any]:
        """
        Analyze the distribution of simulated final returns.

        Provides comprehensive statistics about the distribution of
        outcomes including moments, percentiles, and probability of
        profit/loss.

        Args:
            simulation_result: Result from simulate_returns_* methods

        Returns:
            Dictionary containing:
                - 'moments': Mean, std, skewness, kurtosis
                - 'percentiles': Various percentile values
                - 'probability_of_profit': Prob(return > 0)
                - 'probability_of_loss': Prob(return < 0)
                - 'expected_profit_given_profit': E[return | return > 0]
                - 'expected_loss_given_loss': E[return | return < 0]
                - 'tail_statistics': Tail risk metrics

        Example:
            >>> analysis = MonteCarloSimulator.analyze_return_distribution(result)
            >>> print(f"Probability of Profit: {analysis['probability_of_profit']:.1%}")
        """
        paths = simulation_result.simulated_paths
        initial_value = simulation_result.parameters.get("initial_value", 1.0)

        # Calculate total returns for each simulation
        final_values = paths[:, -1]
        total_returns = (final_values / initial_value) - 1

        # Basic moments
        moments = {
            "mean": float(np.mean(total_returns)),
            "median": float(np.median(total_returns)),
            "std": float(np.std(total_returns)),
            "variance": float(np.var(total_returns)),
            "skewness": float(stats.skew(total_returns)),
            "kurtosis": float(stats.kurtosis(total_returns)),
        }

        # Percentiles
        percentile_values = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        percentiles = {
            f"p{p}": float(np.percentile(total_returns, p)) for p in percentile_values
        }

        # Probability analysis
        num_simulations = len(total_returns)
        num_profit = np.sum(total_returns > 0)
        num_loss = np.sum(total_returns < 0)

        probability_of_profit = num_profit / num_simulations
        probability_of_loss = num_loss / num_simulations

        # Conditional expectations
        profit_returns = total_returns[total_returns > 0]
        loss_returns = total_returns[total_returns < 0]

        expected_profit_given_profit = (
            float(np.mean(profit_returns)) if len(profit_returns) > 0 else 0.0
        )
        expected_loss_given_loss = (
            float(np.mean(loss_returns)) if len(loss_returns) > 0 else 0.0
        )

        # Tail statistics
        sigma = np.std(total_returns)
        mean = np.mean(total_returns)

        tail_statistics = {
            "left_tail_prob_2sigma": float(np.mean(total_returns < mean - 2 * sigma)),
            "right_tail_prob_2sigma": float(np.mean(total_returns > mean + 2 * sigma)),
            "left_tail_prob_3sigma": float(np.mean(total_returns < mean - 3 * sigma)),
            "right_tail_prob_3sigma": float(np.mean(total_returns > mean + 3 * sigma)),
            "worst_return": float(np.min(total_returns)),
            "best_return": float(np.max(total_returns)),
        }

        return {
            "moments": moments,
            "percentiles": percentiles,
            "probability_of_profit": float(probability_of_profit),
            "probability_of_loss": float(probability_of_loss),
            "probability_of_breakeven": float(
                1 - probability_of_profit - probability_of_loss
            ),
            "expected_profit_given_profit": expected_profit_given_profit,
            "expected_loss_given_loss": expected_loss_given_loss,
            "expected_value": moments["mean"],
            "tail_statistics": tail_statistics,
            "num_simulations": num_simulations,
        }

    @staticmethod
    def calculate_probability_of_target(
        simulation_result: SimulationResult, target_return: float
    ) -> Dict[str, float]:
        """
        Calculate probability of achieving a target return.

        Args:
            simulation_result: Result from simulate_returns_* methods
            target_return: Target return as decimal (e.g., 0.10 for 10%)

        Returns:
            Dictionary with probability and related statistics

        Example:
            >>> prob = MonteCarloSimulator.calculate_probability_of_target(
            ...     result, target_return=0.10
            ... )
            >>> print(f"Probability of 10% return: {prob['probability']:.1%}")
        """
        paths = simulation_result.simulated_paths
        initial_value = simulation_result.parameters.get("initial_value", 1.0)

        final_values = paths[:, -1]
        total_returns = (final_values / initial_value) - 1

        num_achieving_target = np.sum(total_returns >= target_return)
        probability = num_achieving_target / len(total_returns)

        # Calculate expected return given target achieved
        returns_above_target = total_returns[total_returns >= target_return]
        expected_given_achieved = (
            float(np.mean(returns_above_target))
            if len(returns_above_target) > 0
            else target_return
        )

        return {
            "target_return": target_return,
            "probability": float(probability),
            "num_achieving": int(num_achieving_target),
            "num_simulations": len(total_returns),
            "expected_return_given_achieved": expected_given_achieved,
        }

    # =========================================================================
    # Drawdown Analysis
    # =========================================================================

    @staticmethod
    def analyze_drawdown_distribution(
        simulation_result: SimulationResult,
    ) -> Dict[str, Any]:
        """
        Analyze the distribution of maximum drawdowns across simulations.

        Args:
            simulation_result: Result from simulate_returns_* methods

        Returns:
            Dictionary with drawdown distribution statistics

        Example:
            >>> dd_analysis = MonteCarloSimulator.analyze_drawdown_distribution(result)
            >>> print(f"Expected Max DD: {dd_analysis['mean_max_drawdown']:.2%}")
        """
        paths = simulation_result.simulated_paths

        # Calculate max drawdown for each path
        max_drawdowns = []
        for path in paths:
            running_max = np.maximum.accumulate(path)
            drawdown = (path - running_max) / running_max
            max_dd = float(np.min(drawdown))
            max_drawdowns.append(max_dd)

        max_drawdowns = np.array(max_drawdowns)

        # Statistics
        percentile_values = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        percentiles = {
            f"p{p}": float(np.percentile(max_drawdowns, p)) for p in percentile_values
        }

        return {
            "mean_max_drawdown": float(np.mean(max_drawdowns)),
            "median_max_drawdown": float(np.median(max_drawdowns)),
            "std_max_drawdown": float(np.std(max_drawdowns)),
            "worst_drawdown": float(np.min(max_drawdowns)),
            "best_drawdown": float(np.max(max_drawdowns)),
            "percentiles": percentiles,
            "probability_dd_exceeds_10pct": float(np.mean(max_drawdowns < -0.10)),
            "probability_dd_exceeds_20pct": float(np.mean(max_drawdowns < -0.20)),
            "probability_dd_exceeds_30pct": float(np.mean(max_drawdowns < -0.30)),
        }

    # =========================================================================
    # Utility Methods
    # =========================================================================

    @staticmethod
    def paths_to_dataframe(
        simulation_result: SimulationResult, sample_paths: int = 100
    ) -> pd.DataFrame:
        """
        Convert simulation paths to DataFrame for visualization.

        Args:
            simulation_result: Result from simulate_returns_* methods
            sample_paths: Number of paths to include (for performance)

        Returns:
            DataFrame with columns for each sampled path

        Example:
            >>> df = MonteCarloSimulator.paths_to_dataframe(result, sample_paths=50)
            >>> df.plot(legend=False, alpha=0.3)
        """
        paths = simulation_result.simulated_paths
        num_simulations = paths.shape[0]

        # Sample paths if needed
        if sample_paths < num_simulations:
            indices = np.random.choice(num_simulations, sample_paths, replace=False)
            sampled_paths = paths[indices]
        else:
            sampled_paths = paths

        # Create DataFrame
        df = pd.DataFrame(sampled_paths.T)
        df.columns = [f"path_{i}" for i in range(df.shape[1])]
        df.index.name = "period"

        return df

    @staticmethod
    def summary_report(
        simulation_result: SimulationResult,
        confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive summary report of simulation results.

        Args:
            simulation_result: Result from simulate_returns_* methods
            confidence_level: Confidence level for intervals

        Returns:
            Dictionary with complete analysis results

        Example:
            >>> report = MonteCarloSimulator.summary_report(result)
            >>> print(json.dumps(report, indent=2, default=str))
        """
        # Distribution analysis
        dist_analysis = MonteCarloSimulator.analyze_return_distribution(
            simulation_result
        )

        # Drawdown analysis
        dd_analysis = MonteCarloSimulator.analyze_drawdown_distribution(
            simulation_result
        )

        # Confidence intervals
        confidence_intervals = MonteCarloSimulator.calculate_all_confidence_intervals(
            simulation_result, confidence_level
        )

        # VaR estimates
        var_estimates = MonteCarloSimulator.estimate_var_at_multiple_levels(
            simulation_result
        )

        return {
            "simulation_parameters": simulation_result.parameters,
            "basic_statistics": simulation_result.statistics,
            "distribution_analysis": dist_analysis,
            "drawdown_analysis": dd_analysis,
            "confidence_intervals": {
                name: {
                    "point_estimate": ci.point_estimate,
                    "lower_bound": ci.lower_bound,
                    "upper_bound": ci.upper_bound,
                    "confidence_level": ci.confidence_level,
                }
                for name, ci in confidence_intervals.items()
            },
            "var_estimates": var_estimates.to_dict("records"),
        }


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Main class
    "MonteCarloSimulator",
    # Data classes
    "SimulationResult",
    "ConfidenceInterval",
    # Exceptions
    "MonteCarloError",
    "InsufficientDataError",
    "InvalidParameterError",
    "SimulationError",
    # Constants
    "DEFAULT_NUM_SIMULATIONS",
    "DEFAULT_NUM_PERIODS",
    "DEFAULT_CONFIDENCE_LEVEL",
    "DEFAULT_RANDOM_SEED",
    "TRADING_DAYS_PER_YEAR",
]
