"""
Volatility Surface Implementation using SVI Parameterization

This module provides a volatility surface class that models the implied volatility
smile/skew across different strikes and expirations. It uses the SVI (Stochastic
Volatility Inspired) model which is an industry standard for volatility surface
fitting.

Key Features:
    - SVI parameterization (industry standard, no-arbitrage)
    - Strike and time interpolation
    - Calibration from market IV data
    - Smooth volatility smile extraction

Usage:
    >>> import pandas as pd
    >>> from backtester.core.volatility_surface import VolatilitySurface
    >>>
    >>> # Load market data with columns: strike, dte, iv, spot
    >>> market_data = pd.read_csv('historical_iv.csv')
    >>>
    >>> # Create surface
    >>> surface = VolatilitySurface(market_data)
    >>>
    >>> # Get IV for any strike/DTE combination
    >>> iv = surface.get_iv(strike=450, dte=30, spot=450)
    >>> print(f"ATM 30-day IV: {iv:.2%}")

References:
    - Gatheral, J. (2004). "A parsimonious arbitrage-free implied volatility
      parameterization with application to the valuation of volatility derivatives"
    - CBOE VIX methodology
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, Tuple, Optional, List
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Bounds for SVI parameters
SVI_PARAM_BOUNDS = [
    (0.001, 1.0),  # a: ATM variance level (positive)
    (0.001, 1.0),  # b: vol of vol (positive)
    (-0.999, 0.999),  # rho: skew direction (correlation, |rho| < 1)
    (-2.0, 2.0),  # m: ATM level shift
    (0.01, 2.0),  # sigma: smile width (positive)
]


# =============================================================================
# Exceptions
# =============================================================================


class VolatilitySurfaceError(Exception):
    """Base exception for volatility surface errors"""

    pass


class CalibrationError(VolatilitySurfaceError):
    """Exception raised when calibration fails"""

    pass


class InterpolationError(VolatilitySurfaceError):
    """Exception raised when interpolation fails"""

    pass


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class SVIParameters:
    """
    SVI model parameters for a single DTE slice

    The SVI formula: σ²(k) = a + b(ρ(k - m) + √((k - m)² + σ²))

    Attributes:
        a: ATM variance level
        b: Volatility of volatility (controls smile amplitude)
        rho: Skew direction (-1 to 1, negative for put skew)
        m: ATM level (moneyness shift)
        sigma: Smile width
        dte: Days to expiration this slice represents
    """

    a: float
    b: float
    rho: float
    m: float
    sigma: float
    dte: int

    def validate(self) -> bool:
        """
        Check no-arbitrage conditions

        Returns:
            True if parameters satisfy no-arbitrage constraints
        """
        # b >= 0 (no calendar arbitrage)
        if self.b < 0:
            return False

        # |rho| < 1 (valid correlation)
        if abs(self.rho) >= 1:
            return False

        # a + b*sigma*sqrt(1-rho^2) >= 0 (non-negative variance)
        min_variance = self.a + self.b * self.sigma * np.sqrt(1 - self.rho**2)
        if min_variance < 0:
            return False

        return True

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "a": self.a,
            "b": self.b,
            "rho": self.rho,
            "m": self.m,
            "sigma": self.sigma,
            "dte": self.dte,
        }


# =============================================================================
# Volatility Surface Class
# =============================================================================


class VolatilitySurface:
    """
    2D volatility surface using SVI parameterization

    Models implied volatility as a function of strike (moneyness) and
    time to expiration. Calibrates separate SVI parameter sets for each
    DTE slice in the market data, then interpolates between slices.

    Attributes:
        market_data: DataFrame with calibration data
        svi_params: Dictionary mapping DTE -> SVIParameters
        calibration_quality: Dictionary of calibration metrics per DTE

    Example:
        >>> surface = VolatilitySurface(market_iv_data)
        >>> iv_30d_atm = surface.get_iv(strike=450, dte=30, spot=450)
        >>> smile_df = surface.get_smile_dataframe(dte=30, spot=450)
    """

    def __init__(self, market_data: pd.DataFrame, min_points_per_slice: int = 5):
        """
        Initialize and calibrate volatility surface

        Args:
            market_data: DataFrame with columns:
                - strike: Strike price
                - dte: Days to expiration
                - iv: Implied volatility (decimal, e.g., 0.20 for 20%)
                - spot: Underlying price
            min_points_per_slice: Minimum data points required per DTE slice

        Raises:
            VolatilitySurfaceError: If data format is invalid
            CalibrationError: If calibration fails for all slices
        """
        # Validate input data
        required_columns = ["strike", "dte", "iv", "spot"]
        missing = set(required_columns) - set(market_data.columns)
        if missing:
            raise VolatilitySurfaceError(f"Missing required columns: {missing}")

        # Remove invalid data points
        self.market_data = market_data[
            (market_data["iv"] > 0)  # Positive IV
            & (market_data["iv"] < 2.0)  # Reasonable upper bound (200%)
            & (market_data["strike"] > 0)  # Positive strikes
            & (market_data["spot"] > 0)  # Positive spot
        ].copy()

        if len(self.market_data) == 0:
            raise VolatilitySurfaceError("No valid data points after filtering")

        self.min_points_per_slice = min_points_per_slice
        self.svi_params: Dict[int, SVIParameters] = {}
        self.calibration_quality: Dict[int, Dict] = {}

        # Calibrate surface
        self._calibrate()

        if len(self.svi_params) == 0:
            raise CalibrationError("Failed to calibrate any DTE slices")

        logger.info(
            f"Volatility surface calibrated with {len(self.svi_params)} DTE slices"
        )

    def _calibrate(self):
        """Calibrate SVI parameters for each DTE slice"""
        unique_dtes = sorted(self.market_data["dte"].unique())

        for dte in unique_dtes:
            try:
                slice_data = self.market_data[self.market_data["dte"] == dte]

                # Skip if insufficient data points
                if len(slice_data) < self.min_points_per_slice:
                    logger.warning(
                        f"Skipping DTE={dte}: only {len(slice_data)} points "
                        f"(min={self.min_points_per_slice})"
                    )
                    continue

                params = self._fit_svi_slice(slice_data)

                # Validate parameters
                if params.validate():
                    self.svi_params[dte] = params
                    logger.debug(f"Calibrated DTE={dte}: {params.to_dict()}")
                else:
                    logger.warning(f"DTE={dte} parameters failed validation, skipping")

            except Exception as e:
                logger.warning(f"Failed to calibrate DTE={dte}: {e}")
                continue

    def _fit_svi_slice(self, data: pd.DataFrame) -> SVIParameters:
        """
        Fit SVI parameters to a single DTE slice

        Args:
            data: Slice data for single DTE

        Returns:
            Calibrated SVIParameters

        Raises:
            CalibrationError: If optimization fails
        """
        dte = int(data["dte"].iloc[0])
        spot = data["spot"].mean()  # Should be constant, but average just in case

        strikes = data["strike"].values
        ivs = data["iv"].values

        # Calculate moneyness: k = ln(K/S)
        k = np.log(strikes / spot)

        # Target: variance (σ²)
        target_variance = ivs**2

        # Initial guess based on data
        atm_idx = np.argmin(np.abs(k))
        atm_iv = ivs[atm_idx]

        # Detect skew direction from data
        left_ivs = ivs[k < 0]  # OTM puts
        right_ivs = ivs[k > 0]  # OTM calls

        if len(left_ivs) > 0 and len(right_ivs) > 0:
            # Typical equity skew: puts more expensive
            skew_direction = -0.4 if left_ivs.mean() > right_ivs.mean() else 0.2
        else:
            skew_direction = -0.3  # Default to put skew

        x0 = np.array(
            [
                atm_iv**2,  # a: ATM variance
                0.1,  # b: vol of vol
                skew_direction,  # rho: skew direction
                0.0,  # m: ATM level
                0.3,  # sigma: smile width
            ]
        )

        # Objective function: minimize sum of squared errors
        def objective(params):
            a, b, rho, m, sigma = params
            model_variance = self._svi_formula(k, a, b, rho, m, sigma)
            return np.sum((model_variance - target_variance) ** 2)

        # Constraints for no-arbitrage
        def constraint_b_positive(params):
            return params[1]  # b >= 0

        def constraint_rho_magnitude(params):
            return 1 - params[2] ** 2  # rho² < 1

        constraints = [
            {"type": "ineq", "fun": constraint_b_positive},
            {"type": "ineq", "fun": constraint_rho_magnitude},
        ]

        # Optimize
        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=SVI_PARAM_BOUNDS,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-9},
        )

        if not result.success:
            # Try again with different initial guess
            x0_alt = np.array([0.04, 0.2, -0.5, 0.0, 0.5])
            result = minimize(
                objective,
                x0_alt,
                method="SLSQP",
                bounds=SVI_PARAM_BOUNDS,
                constraints=constraints,
                options={"maxiter": 1000},
            )

        if not result.success:
            raise CalibrationError(f"SVI optimization failed: {result.message}")

        a, b, rho, m, sigma = result.x

        # Calculate calibration quality
        model_variance = self._svi_formula(k, a, b, rho, m, sigma)
        rmse = np.sqrt(np.mean((np.sqrt(model_variance) - ivs) ** 2))

        self.calibration_quality[dte] = {
            "rmse": rmse,
            "num_points": len(data),
            "objective_value": result.fun,
        }

        return SVIParameters(a=a, b=b, rho=rho, m=m, sigma=sigma, dte=dte)

    @staticmethod
    def _svi_formula(
        k: np.ndarray, a: float, b: float, rho: float, m: float, sigma: float
    ) -> np.ndarray:
        """
        SVI variance formula

        σ²(k) = a + b(ρ(k - m) + √((k - m)² + σ²))

        Args:
            k: Log-moneyness array
            a, b, rho, m, sigma: SVI parameters

        Returns:
            Variance (σ²) array
        """
        k_shifted = k - m
        return a + b * (rho * k_shifted + np.sqrt(k_shifted**2 + sigma**2))

    def get_iv(self, strike: float, dte: float, spot: float) -> float:
        """
        Get implied volatility for any strike/DTE combination

        Interpolates between calibrated DTE slices if necessary.

        Args:
            strike: Strike price
            dte: Days to expiration (can be fractional)
            spot: Current spot price

        Returns:
            Implied volatility (decimal, e.g., 0.20 for 20%)

        Raises:
            InterpolationError: If unable to interpolate
        """
        if len(self.svi_params) == 0:
            raise InterpolationError("No calibrated parameters available")

        # Calculate moneyness
        k = np.log(strike / spot)

        # Get available DTEs
        dte_keys = sorted(self.svi_params.keys())

        # Find surrounding DTEs for interpolation
        if dte <= dte_keys[0]:
            # Extrapolate to shorter DTE using nearest slice
            params = self.svi_params[dte_keys[0]]
            variance = self._svi_formula(
                np.array([k]), params.a, params.b, params.rho, params.m, params.sigma
            )[0]

        elif dte >= dte_keys[-1]:
            # Extrapolate to longer DTE using nearest slice
            params = self.svi_params[dte_keys[-1]]
            variance = self._svi_formula(
                np.array([k]), params.a, params.b, params.rho, params.m, params.sigma
            )[0]

        else:
            # Interpolate between two DTEs
            dte_lower = max([d for d in dte_keys if d <= dte])
            dte_upper = min([d for d in dte_keys if d > dte])

            params_lower = self.svi_params[dte_lower]
            params_upper = self.svi_params[dte_upper]

            # Calculate variance at both DTEs
            var_lower = self._svi_formula(
                np.array([k]),
                params_lower.a,
                params_lower.b,
                params_lower.rho,
                params_lower.m,
                params_lower.sigma,
            )[0]

            var_upper = self._svi_formula(
                np.array([k]),
                params_upper.a,
                params_upper.b,
                params_upper.rho,
                params_upper.m,
                params_upper.sigma,
            )[0]

            # Linear interpolation of variance
            weight = (dte - dte_lower) / (dte_upper - dte_lower)
            variance = (1 - weight) * var_lower + weight * var_upper

        # Convert variance to volatility
        # Ensure non-negative and reasonable bounds
        variance = np.clip(variance, 0.0001, 4.0)  # Min 1% IV, max 200% IV
        iv = np.sqrt(variance)

        return float(iv)

    def get_smile_dataframe(
        self,
        dte: int,
        spot: float,
        strike_range: Tuple[float, float] = (0.8, 1.2),
        num_points: int = 50,
    ) -> pd.DataFrame:
        """
        Get volatility smile for a specific DTE (for visualization)

        Args:
            dte: Days to expiration
            spot: Current spot price
            strike_range: (min_pct, max_pct) of spot for strike range
            num_points: Number of points in smile

        Returns:
            DataFrame with columns: strike, moneyness, iv
        """
        strikes = np.linspace(
            spot * strike_range[0], spot * strike_range[1], num_points
        )

        ivs = [self.get_iv(K, dte, spot) for K in strikes]

        return pd.DataFrame(
            {
                "strike": strikes,
                "moneyness": strikes / spot,
                "log_moneyness": np.log(strikes / spot),
                "iv": ivs,
            }
        )

    def get_available_dtes(self) -> List[int]:
        """Get list of calibrated DTEs"""
        return [int(d) for d in sorted(self.svi_params.keys())]

    def get_calibration_report(self) -> pd.DataFrame:
        """
        Get calibration quality report

        Returns:
            DataFrame with calibration metrics for each DTE
        """
        report_data = []
        for dte in sorted(self.svi_params.keys()):
            params = self.svi_params[dte]
            quality = self.calibration_quality.get(dte, {})

            report_data.append(
                {
                    "dte": dte,
                    "a": params.a,
                    "b": params.b,
                    "rho": params.rho,
                    "m": params.m,
                    "sigma": params.sigma,
                    "rmse": quality.get("rmse", np.nan),
                    "num_points": quality.get("num_points", 0),
                }
            )

        return pd.DataFrame(report_data)


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "VolatilitySurface",
    "SVIParameters",
    "VolatilitySurfaceError",
    "CalibrationError",
    "InterpolationError",
]
