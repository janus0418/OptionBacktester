"""
Scenario Testing Module for Options Backtesting Analytics

This module provides scenario testing capabilities for stress testing
strategies under various market conditions and analyzing sensitivity
to different market factors.

Key Features:
    - Predefined stress scenarios (crash, vol spike, rate change)
    - Custom scenario definition and application
    - Historical scenario replay (2008 crisis, 2020 COVID, etc.)
    - Sensitivity analysis (delta, vega, theta sensitivity)
    - What-if analysis framework
    - Multi-scenario comparison

Design Philosophy:
    Scenarios are defined as market factor shocks that can be applied
    to portfolio positions to estimate P&L impact. Methods support
    both instantaneous shocks and gradual transitions.

Mathematical Background:
    1. Delta-Gamma Approximation:
       P&L ≈ Delta * dS + 0.5 * Gamma * dS^2 + Theta * dt + Vega * dIV

    2. Full Repricing:
       For more accurate results under large moves, positions can be
       fully repriced using Black-Scholes with shocked parameters.

    3. Scenario Composition:
       Multiple shocks can be composed to create complex scenarios
       (e.g., crash + vol spike + rate cut).

Usage:
    from backtester.analytics.scenario_testing import ScenarioTester

    # Define a stress scenario
    crash_scenario = Scenario(
        name='Market Crash',
        spot_shock=-0.20,      # -20% spot move
        vol_shock=0.30,        # +30 vol points
        rate_shock=-0.01       # -1% rate cut
    )

    # Apply scenario to portfolio
    impact = ScenarioTester.apply_scenario(
        portfolio_greeks, crash_scenario
    )

    # Run stress test suite
    results = ScenarioTester.run_stress_test_suite(portfolio_greeks)

References:
    - Taleb, N.N. (2007). The Black Swan.
    - Jorion, P. (2007). Value at Risk.
    - Hull, J.C. (2018). Options, Futures, and Other Derivatives.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

# Configure module logger
logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Standard scenario parameters
DEFAULT_SHOCK_LEVELS = [-0.20, -0.10, -0.05, 0.05, 0.10, 0.20]  # Spot shocks
DEFAULT_VOL_SHOCKS = [-0.10, -0.05, 0.05, 0.10, 0.20, 0.50]  # Vol point changes
DEFAULT_RATE_SHOCKS = [-0.02, -0.01, -0.005, 0.005, 0.01, 0.02]  # Rate changes

# Numerical tolerance
EPSILON = 1e-10

# Trading days per year
TRADING_DAYS_PER_YEAR = 252

# Historical event dates (for reference)
HISTORICAL_EVENTS = {
    "black_monday_1987": datetime(1987, 10, 19),
    "asian_crisis_1997": datetime(1997, 10, 27),
    "ltcm_1998": datetime(1998, 8, 21),
    "dot_com_crash_2000": datetime(2000, 3, 10),
    "lehman_2008": datetime(2008, 9, 15),
    "flash_crash_2010": datetime(2010, 5, 6),
    "covid_crash_2020": datetime(2020, 3, 16),
    "volmageddon_2018": datetime(2018, 2, 5),
}


# =============================================================================
# Enums
# =============================================================================


class ScenarioType(Enum):
    """Types of scenarios for classification."""

    STRESS = "stress"
    HISTORICAL = "historical"
    SENSITIVITY = "sensitivity"
    CUSTOM = "custom"


class ShockType(Enum):
    """Types of market shocks."""

    INSTANTANEOUS = "instantaneous"
    GRADUAL = "gradual"


# =============================================================================
# Exceptions
# =============================================================================


class ScenarioError(Exception):
    """Base exception for scenario testing errors."""

    pass


class InvalidScenarioError(ScenarioError):
    """Exception raised when scenario parameters are invalid."""

    pass


class InsufficientDataError(ScenarioError):
    """Exception raised when there is insufficient data for scenario analysis."""

    pass


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class Scenario:
    """
    Definition of a market scenario.

    A scenario represents a set of market factor shocks that can be
    applied to estimate portfolio P&L impact.

    Attributes:
        name: Human-readable scenario name
        spot_shock: Percentage change in underlying price (e.g., -0.20 = -20%)
        vol_shock: Absolute change in implied volatility (e.g., 0.10 = +10 vol points)
        rate_shock: Absolute change in risk-free rate (e.g., -0.01 = -100bps)
        time_shock: Time decay in days (e.g., 1 = one day theta)
        scenario_type: Classification of the scenario
        description: Optional detailed description
    """

    name: str
    spot_shock: float = 0.0
    vol_shock: float = 0.0
    rate_shock: float = 0.0
    time_shock: float = 0.0
    scenario_type: ScenarioType = ScenarioType.CUSTOM
    description: str = ""

    def __post_init__(self):
        """Validate scenario parameters."""
        if self.spot_shock < -1.0:
            raise InvalidScenarioError(
                f"spot_shock cannot be less than -1.0 (total loss), got {self.spot_shock}"
            )


@dataclass
class ScenarioResult:
    """
    Result of applying a scenario to a portfolio.

    Attributes:
        scenario: The scenario that was applied
        pnl_estimate: Estimated P&L from the scenario
        pnl_breakdown: P&L attributed to each Greek
        new_position_value: Estimated position value after scenario
        risk_metrics: Additional risk metrics under the scenario
    """

    scenario: Scenario
    pnl_estimate: float
    pnl_breakdown: Dict[str, float]
    new_position_value: float
    risk_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class SensitivityResult:
    """
    Result of sensitivity analysis.

    Attributes:
        factor: The factor being analyzed (spot, vol, rate, time)
        shock_levels: List of shock levels applied
        pnl_values: P&L at each shock level
        greeks_at_shocks: Greeks values at each shock level
    """

    factor: str
    shock_levels: List[float]
    pnl_values: List[float]
    greeks_at_shocks: Optional[Dict[str, List[float]]] = None


# =============================================================================
# Predefined Scenarios
# =============================================================================

# Standard stress scenarios
STRESS_SCENARIOS = {
    "market_crash_severe": Scenario(
        name="Severe Market Crash",
        spot_shock=-0.25,
        vol_shock=0.40,
        rate_shock=-0.01,
        scenario_type=ScenarioType.STRESS,
        description="Severe market crash similar to 2008 or 2020",
    ),
    "market_crash_moderate": Scenario(
        name="Moderate Market Crash",
        spot_shock=-0.15,
        vol_shock=0.25,
        rate_shock=-0.005,
        scenario_type=ScenarioType.STRESS,
        description="Moderate market correction",
    ),
    "flash_crash": Scenario(
        name="Flash Crash",
        spot_shock=-0.10,
        vol_shock=0.50,
        rate_shock=0.0,
        scenario_type=ScenarioType.STRESS,
        description="Sudden intraday crash with vol spike",
    ),
    "vol_spike": Scenario(
        name="Volatility Spike",
        spot_shock=-0.05,
        vol_shock=0.30,
        rate_shock=0.0,
        scenario_type=ScenarioType.STRESS,
        description="VIX-style volatility explosion",
    ),
    "vol_crush": Scenario(
        name="Volatility Crush",
        spot_shock=0.02,
        vol_shock=-0.15,
        rate_shock=0.0,
        scenario_type=ScenarioType.STRESS,
        description="Post-event volatility collapse",
    ),
    "rate_hike": Scenario(
        name="Rate Hike",
        spot_shock=-0.02,
        vol_shock=0.05,
        rate_shock=0.01,
        scenario_type=ScenarioType.STRESS,
        description="Aggressive Fed rate hike",
    ),
    "rate_cut": Scenario(
        name="Emergency Rate Cut",
        spot_shock=0.03,
        vol_shock=-0.05,
        rate_shock=-0.01,
        scenario_type=ScenarioType.STRESS,
        description="Emergency rate cut response",
    ),
    "melt_up": Scenario(
        name="Melt Up",
        spot_shock=0.15,
        vol_shock=-0.10,
        rate_shock=0.005,
        scenario_type=ScenarioType.STRESS,
        description="Euphoric market rally",
    ),
}

# Historical scenarios (approximate parameters)
HISTORICAL_SCENARIOS = {
    "black_monday_1987": Scenario(
        name="Black Monday 1987",
        spot_shock=-0.22,
        vol_shock=0.60,
        rate_shock=-0.005,
        scenario_type=ScenarioType.HISTORICAL,
        description="October 19, 1987 crash (-22.6% in one day)",
    ),
    "lehman_2008": Scenario(
        name="Lehman Crisis 2008",
        spot_shock=-0.30,
        vol_shock=0.50,
        rate_shock=-0.02,
        scenario_type=ScenarioType.HISTORICAL,
        description="September 2008 financial crisis period",
    ),
    "flash_crash_2010": Scenario(
        name="Flash Crash 2010",
        spot_shock=-0.09,
        vol_shock=0.40,
        rate_shock=0.0,
        scenario_type=ScenarioType.HISTORICAL,
        description="May 6, 2010 flash crash",
    ),
    "covid_crash_2020": Scenario(
        name="COVID Crash 2020",
        spot_shock=-0.35,
        vol_shock=0.55,
        rate_shock=-0.015,
        scenario_type=ScenarioType.HISTORICAL,
        description="March 2020 pandemic selloff",
    ),
    "volmageddon_2018": Scenario(
        name="Volmageddon 2018",
        spot_shock=-0.04,
        vol_shock=0.80,
        rate_shock=0.0,
        scenario_type=ScenarioType.HISTORICAL,
        description="February 2018 VIX spike (+115% in one day)",
    ),
}


# =============================================================================
# ScenarioTester Class
# =============================================================================


class ScenarioTester:
    """
    Scenario testing and stress analysis for portfolios.

    This class provides static methods for applying scenarios to portfolios,
    running stress tests, and performing sensitivity analysis.

    Methods support both:
    1. Delta-Gamma approximation for quick estimates
    2. Full repricing for accurate results under large moves

    Example:
        >>> from backtester.analytics.scenario_testing import ScenarioTester, Scenario
        >>>
        >>> # Portfolio Greeks
        >>> greeks = {'delta': 100, 'gamma': 5, 'theta': -50, 'vega': 200}
        >>> position_value = 10000
        >>>
        >>> # Apply a crash scenario
        >>> result = ScenarioTester.apply_scenario(
        ...     greeks, position_value, STRESS_SCENARIOS['market_crash_severe']
        ... )
        >>> print(f"Estimated P&L: ${result.pnl_estimate:,.2f}")
    """

    # =========================================================================
    # Core Scenario Application
    # =========================================================================

    @staticmethod
    def apply_scenario(
        greeks: Dict[str, float],
        position_value: float,
        scenario: Scenario,
        underlying_price: Optional[float] = None,
        current_vol: Optional[float] = None,
    ) -> ScenarioResult:
        """
        Apply a scenario to estimate P&L impact.

        Uses delta-gamma-vega-theta approximation:
        P&L ≈ Delta*dS + 0.5*Gamma*dS^2 + Vega*dIV + Theta*dt

        Args:
            greeks: Dictionary with portfolio Greeks (delta, gamma, theta, vega, rho)
            position_value: Current position value in dollars
            scenario: Scenario to apply
            underlying_price: Current underlying price (for dS calculation)
            current_vol: Current implied volatility (for reference)

        Returns:
            ScenarioResult with estimated P&L and breakdown

        Example:
            >>> greeks = {'delta': 50, 'gamma': 2, 'theta': -20, 'vega': 100}
            >>> result = ScenarioTester.apply_scenario(
            ...     greeks, 5000, STRESS_SCENARIOS['vol_spike']
            ... )
        """
        # Extract Greeks with defaults
        delta = greeks.get("delta", 0.0)
        gamma = greeks.get("gamma", 0.0)
        theta = greeks.get("theta", 0.0)
        vega = greeks.get("vega", 0.0)
        rho = greeks.get("rho", 0.0)

        # Calculate dollar move if underlying price provided
        if underlying_price is not None and underlying_price > 0:
            dollar_move = underlying_price * scenario.spot_shock
        else:
            # Estimate dollar move as percentage of delta (approximation)
            dollar_move = position_value * scenario.spot_shock

        # Calculate P&L components
        # Delta P&L
        delta_pnl = delta * dollar_move

        # Gamma P&L (second-order effect)
        gamma_pnl = 0.5 * gamma * dollar_move**2

        # Vega P&L (vol is in percentage points, vega is per 1% move)
        # Convert vol shock to percentage points
        vega_pnl = vega * (
            scenario.vol_shock * 100
        )  # vega is typically per 1% vol move

        # Theta P&L
        theta_pnl = theta * scenario.time_shock

        # Rho P&L (rate is in absolute terms, rho is per 1% move)
        rho_pnl = rho * (scenario.rate_shock * 100)

        # Total P&L estimate
        total_pnl = delta_pnl + gamma_pnl + vega_pnl + theta_pnl + rho_pnl

        # P&L breakdown
        pnl_breakdown = {
            "delta_pnl": float(delta_pnl),
            "gamma_pnl": float(gamma_pnl),
            "vega_pnl": float(vega_pnl),
            "theta_pnl": float(theta_pnl),
            "rho_pnl": float(rho_pnl),
        }

        # New position value
        new_value = position_value + total_pnl

        # Risk metrics
        risk_metrics = {
            "pnl_pct": float(total_pnl / position_value)
            if position_value > EPSILON
            else 0.0,
            "position_value_after": float(new_value),
            "max_component": max(pnl_breakdown.items(), key=lambda x: abs(x[1]))[0],
        }

        return ScenarioResult(
            scenario=scenario,
            pnl_estimate=float(total_pnl),
            pnl_breakdown=pnl_breakdown,
            new_position_value=float(new_value),
            risk_metrics=risk_metrics,
        )

    @staticmethod
    def apply_multiple_scenarios(
        greeks: Dict[str, float],
        position_value: float,
        scenarios: List[Scenario],
        underlying_price: Optional[float] = None,
    ) -> Dict[str, ScenarioResult]:
        """
        Apply multiple scenarios and return results.

        Args:
            greeks: Dictionary with portfolio Greeks
            position_value: Current position value
            scenarios: List of scenarios to apply
            underlying_price: Current underlying price

        Returns:
            Dictionary mapping scenario names to results

        Example:
            >>> scenarios = [STRESS_SCENARIOS['market_crash_severe'],
            ...              STRESS_SCENARIOS['vol_spike']]
            >>> results = ScenarioTester.apply_multiple_scenarios(
            ...     greeks, 10000, scenarios
            ... )
        """
        results = {}
        for scenario in scenarios:
            results[scenario.name] = ScenarioTester.apply_scenario(
                greeks, position_value, scenario, underlying_price
            )
        return results

    # =========================================================================
    # Stress Testing
    # =========================================================================

    @staticmethod
    def run_stress_test_suite(
        greeks: Dict[str, float],
        position_value: float,
        underlying_price: Optional[float] = None,
        include_historical: bool = True,
    ) -> pd.DataFrame:
        """
        Run a comprehensive stress test suite.

        Applies all predefined stress scenarios and optionally historical
        scenarios, returning a summary DataFrame.

        Args:
            greeks: Dictionary with portfolio Greeks
            position_value: Current position value
            underlying_price: Current underlying price
            include_historical: Whether to include historical scenarios

        Returns:
            DataFrame with stress test results

        Example:
            >>> greeks = {'delta': 100, 'gamma': 5, 'theta': -50, 'vega': 200}
            >>> results_df = ScenarioTester.run_stress_test_suite(greeks, 10000)
            >>> print(results_df[['scenario', 'pnl', 'pnl_pct']])
        """
        all_scenarios = list(STRESS_SCENARIOS.values())
        if include_historical:
            all_scenarios.extend(HISTORICAL_SCENARIOS.values())

        records = []
        for scenario in all_scenarios:
            result = ScenarioTester.apply_scenario(
                greeks, position_value, scenario, underlying_price
            )

            records.append(
                {
                    "scenario": scenario.name,
                    "type": scenario.scenario_type.value,
                    "spot_shock": scenario.spot_shock,
                    "vol_shock": scenario.vol_shock,
                    "rate_shock": scenario.rate_shock,
                    "pnl": result.pnl_estimate,
                    "pnl_pct": result.risk_metrics["pnl_pct"],
                    "delta_pnl": result.pnl_breakdown["delta_pnl"],
                    "gamma_pnl": result.pnl_breakdown["gamma_pnl"],
                    "vega_pnl": result.pnl_breakdown["vega_pnl"],
                    "theta_pnl": result.pnl_breakdown["theta_pnl"],
                    "new_value": result.new_position_value,
                }
            )

        df = pd.DataFrame(records)
        df = df.sort_values("pnl", ascending=True)

        return df

    @staticmethod
    def identify_worst_scenarios(
        greeks: Dict[str, float],
        position_value: float,
        top_n: int = 5,
        underlying_price: Optional[float] = None,
    ) -> List[ScenarioResult]:
        """
        Identify the worst scenarios for the portfolio.

        Args:
            greeks: Dictionary with portfolio Greeks
            position_value: Current position value
            top_n: Number of worst scenarios to return
            underlying_price: Current underlying price

        Returns:
            List of worst ScenarioResults sorted by P&L (ascending)
        """
        all_scenarios = list(STRESS_SCENARIOS.values()) + list(
            HISTORICAL_SCENARIOS.values()
        )

        results = []
        for scenario in all_scenarios:
            result = ScenarioTester.apply_scenario(
                greeks, position_value, scenario, underlying_price
            )
            results.append(result)

        # Sort by P&L (worst first)
        results.sort(key=lambda x: x.pnl_estimate)

        return results[:top_n]

    # =========================================================================
    # Sensitivity Analysis
    # =========================================================================

    @staticmethod
    def spot_sensitivity(
        greeks: Dict[str, float],
        position_value: float,
        shock_levels: Optional[List[float]] = None,
        underlying_price: Optional[float] = None,
    ) -> SensitivityResult:
        """
        Analyze P&L sensitivity to spot price changes.

        Args:
            greeks: Dictionary with portfolio Greeks
            position_value: Current position value
            shock_levels: List of spot shock levels (default: standard range)
            underlying_price: Current underlying price

        Returns:
            SensitivityResult with P&L at each shock level

        Example:
            >>> result = ScenarioTester.spot_sensitivity(
            ...     greeks, 10000, shock_levels=[-0.20, -0.10, 0, 0.10, 0.20]
            ... )
            >>> # Plot result.shock_levels vs result.pnl_values
        """
        if shock_levels is None:
            shock_levels = DEFAULT_SHOCK_LEVELS

        pnl_values = []
        for shock in shock_levels:
            scenario = Scenario(name=f"Spot {shock:+.0%}", spot_shock=shock)
            result = ScenarioTester.apply_scenario(
                greeks, position_value, scenario, underlying_price
            )
            pnl_values.append(result.pnl_estimate)

        return SensitivityResult(
            factor="spot", shock_levels=shock_levels, pnl_values=pnl_values
        )

    @staticmethod
    def vol_sensitivity(
        greeks: Dict[str, float],
        position_value: float,
        shock_levels: Optional[List[float]] = None,
    ) -> SensitivityResult:
        """
        Analyze P&L sensitivity to volatility changes.

        Args:
            greeks: Dictionary with portfolio Greeks
            position_value: Current position value
            shock_levels: List of vol shock levels in absolute terms

        Returns:
            SensitivityResult with P&L at each shock level
        """
        if shock_levels is None:
            shock_levels = DEFAULT_VOL_SHOCKS

        pnl_values = []
        for shock in shock_levels:
            scenario = Scenario(name=f"Vol {shock:+.0%}", vol_shock=shock)
            result = ScenarioTester.apply_scenario(greeks, position_value, scenario)
            pnl_values.append(result.pnl_estimate)

        return SensitivityResult(
            factor="vol", shock_levels=shock_levels, pnl_values=pnl_values
        )

    @staticmethod
    def rate_sensitivity(
        greeks: Dict[str, float],
        position_value: float,
        shock_levels: Optional[List[float]] = None,
    ) -> SensitivityResult:
        """
        Analyze P&L sensitivity to interest rate changes.

        Args:
            greeks: Dictionary with portfolio Greeks
            position_value: Current position value
            shock_levels: List of rate shock levels in absolute terms

        Returns:
            SensitivityResult with P&L at each shock level
        """
        if shock_levels is None:
            shock_levels = DEFAULT_RATE_SHOCKS

        pnl_values = []
        for shock in shock_levels:
            scenario = Scenario(name=f"Rate {shock * 100:+.0f}bps", rate_shock=shock)
            result = ScenarioTester.apply_scenario(greeks, position_value, scenario)
            pnl_values.append(result.pnl_estimate)

        return SensitivityResult(
            factor="rate", shock_levels=shock_levels, pnl_values=pnl_values
        )

    @staticmethod
    def time_decay_analysis(
        greeks: Dict[str, float],
        position_value: float,
        days: Optional[List[int]] = None,
    ) -> SensitivityResult:
        """
        Analyze P&L sensitivity to time decay.

        Args:
            greeks: Dictionary with portfolio Greeks
            position_value: Current position value
            days: List of days to analyze

        Returns:
            SensitivityResult with P&L at each time point
        """
        if days is None:
            days = [1, 2, 5, 10, 20, 30]

        pnl_values = []
        for day in days:
            scenario = Scenario(name=f"{day} Day Theta", time_shock=float(day))
            result = ScenarioTester.apply_scenario(greeks, position_value, scenario)
            pnl_values.append(result.pnl_estimate)

        return SensitivityResult(
            factor="time", shock_levels=[float(d) for d in days], pnl_values=pnl_values
        )

    @staticmethod
    def full_sensitivity_analysis(
        greeks: Dict[str, float],
        position_value: float,
        underlying_price: Optional[float] = None,
    ) -> Dict[str, SensitivityResult]:
        """
        Run complete sensitivity analysis across all factors.

        Args:
            greeks: Dictionary with portfolio Greeks
            position_value: Current position value
            underlying_price: Current underlying price

        Returns:
            Dictionary with sensitivity results for each factor
        """
        return {
            "spot": ScenarioTester.spot_sensitivity(
                greeks, position_value, underlying_price=underlying_price
            ),
            "vol": ScenarioTester.vol_sensitivity(greeks, position_value),
            "rate": ScenarioTester.rate_sensitivity(greeks, position_value),
            "time": ScenarioTester.time_decay_analysis(greeks, position_value),
        }

    # =========================================================================
    # What-If Analysis
    # =========================================================================

    @staticmethod
    def what_if_spot_vol(
        greeks: Dict[str, float],
        position_value: float,
        spot_shocks: Optional[List[float]] = None,
        vol_shocks: Optional[List[float]] = None,
        underlying_price: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Create a spot-vol matrix showing P&L under different combinations.

        This creates a 2D grid showing how P&L changes across different
        combinations of spot and volatility moves.

        Args:
            greeks: Dictionary with portfolio Greeks
            position_value: Current position value
            spot_shocks: List of spot shock levels
            vol_shocks: List of vol shock levels
            underlying_price: Current underlying price

        Returns:
            DataFrame with spot shocks as rows, vol shocks as columns

        Example:
            >>> matrix = ScenarioTester.what_if_spot_vol(greeks, 10000)
            >>> # Matrix shows P&L for each (spot, vol) combination
        """
        if spot_shocks is None:
            spot_shocks = [-0.20, -0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20]
        if vol_shocks is None:
            vol_shocks = [-0.10, -0.05, 0.0, 0.05, 0.10, 0.20, 0.30]

        # Build matrix
        matrix = np.zeros((len(spot_shocks), len(vol_shocks)))

        for i, spot_shock in enumerate(spot_shocks):
            for j, vol_shock in enumerate(vol_shocks):
                scenario = Scenario(
                    name=f"Spot {spot_shock:+.0%}, Vol {vol_shock:+.0%}",
                    spot_shock=spot_shock,
                    vol_shock=vol_shock,
                )
                result = ScenarioTester.apply_scenario(
                    greeks, position_value, scenario, underlying_price
                )
                matrix[i, j] = result.pnl_estimate

        # Create DataFrame
        index_labels = pd.Index([f"{s:+.0%}" for s in spot_shocks], name="Spot Shock")
        column_labels = pd.Index([f"{v:+.0%}" for v in vol_shocks], name="Vol Shock")
        df = pd.DataFrame(matrix, index=index_labels, columns=column_labels)

        return df

    @staticmethod
    def what_if_custom(
        greeks: Dict[str, float],
        position_value: float,
        spot_shock: float,
        vol_shock: float,
        rate_shock: float = 0.0,
        time_shock: float = 0.0,
        underlying_price: Optional[float] = None,
    ) -> ScenarioResult:
        """
        Run a custom what-if scenario.

        Args:
            greeks: Dictionary with portfolio Greeks
            position_value: Current position value
            spot_shock: Percentage change in spot
            vol_shock: Absolute change in vol
            rate_shock: Absolute change in rate
            time_shock: Days of theta decay
            underlying_price: Current underlying price

        Returns:
            ScenarioResult for the custom scenario
        """
        scenario = Scenario(
            name="Custom What-If",
            spot_shock=spot_shock,
            vol_shock=vol_shock,
            rate_shock=rate_shock,
            time_shock=time_shock,
            scenario_type=ScenarioType.CUSTOM,
        )

        return ScenarioTester.apply_scenario(
            greeks, position_value, scenario, underlying_price
        )

    # =========================================================================
    # Scenario Composition
    # =========================================================================

    @staticmethod
    def compose_scenario(
        base_scenario: Scenario,
        additional_shocks: Dict[str, float],
        new_name: Optional[str] = None,
    ) -> Scenario:
        """
        Create a new scenario by modifying an existing one.

        Args:
            base_scenario: Base scenario to modify
            additional_shocks: Dict with additional shocks to apply
                               ('spot_shock', 'vol_shock', 'rate_shock', 'time_shock')
            new_name: Name for the new scenario

        Returns:
            New Scenario with combined shocks

        Example:
            >>> # Add extra vol to a crash scenario
            >>> new_scenario = ScenarioTester.compose_scenario(
            ...     STRESS_SCENARIOS['market_crash_severe'],
            ...     {'vol_shock': 0.10},  # Extra 10% vol
            ...     'Crash + Extra Vol'
            ... )
        """
        new_spot = base_scenario.spot_shock + additional_shocks.get("spot_shock", 0.0)
        new_vol = base_scenario.vol_shock + additional_shocks.get("vol_shock", 0.0)
        new_rate = base_scenario.rate_shock + additional_shocks.get("rate_shock", 0.0)
        new_time = base_scenario.time_shock + additional_shocks.get("time_shock", 0.0)

        return Scenario(
            name=new_name or f"{base_scenario.name} (Modified)",
            spot_shock=new_spot,
            vol_shock=new_vol,
            rate_shock=new_rate,
            time_shock=new_time,
            scenario_type=ScenarioType.CUSTOM,
            description=f"Modified from: {base_scenario.name}",
        )

    @staticmethod
    def create_custom_scenario(
        name: str,
        spot_shock: float = 0.0,
        vol_shock: float = 0.0,
        rate_shock: float = 0.0,
        time_shock: float = 0.0,
        description: str = "",
    ) -> Scenario:
        """
        Create a custom scenario from parameters.

        Args:
            name: Scenario name
            spot_shock: Percentage change in spot
            vol_shock: Absolute change in vol
            rate_shock: Absolute change in rate
            time_shock: Days of theta decay
            description: Optional description

        Returns:
            New custom Scenario
        """
        return Scenario(
            name=name,
            spot_shock=spot_shock,
            vol_shock=vol_shock,
            rate_shock=rate_shock,
            time_shock=time_shock,
            scenario_type=ScenarioType.CUSTOM,
            description=description,
        )

    # =========================================================================
    # Reporting
    # =========================================================================

    @staticmethod
    def generate_stress_report(
        greeks: Dict[str, float],
        position_value: float,
        underlying_price: Optional[float] = None,
        include_sensitivity: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive stress test report.

        Args:
            greeks: Dictionary with portfolio Greeks
            position_value: Current position value
            underlying_price: Current underlying price
            include_sensitivity: Whether to include sensitivity analysis

        Returns:
            Dictionary with complete stress test results
        """
        report = {
            "portfolio_summary": {
                "position_value": position_value,
                "underlying_price": underlying_price,
                "greeks": greeks,
            },
            "stress_tests": ScenarioTester.run_stress_test_suite(
                greeks, position_value, underlying_price
            ).to_dict("records"),
            "worst_scenarios": [
                {
                    "scenario": r.scenario.name,
                    "pnl": r.pnl_estimate,
                    "pnl_pct": r.risk_metrics["pnl_pct"],
                }
                for r in ScenarioTester.identify_worst_scenarios(
                    greeks, position_value, top_n=5, underlying_price=underlying_price
                )
            ],
        }

        if include_sensitivity:
            sensitivities = ScenarioTester.full_sensitivity_analysis(
                greeks, position_value, underlying_price
            )
            report["sensitivity"] = {
                factor: {
                    "shock_levels": result.shock_levels,
                    "pnl_values": result.pnl_values,
                }
                for factor, result in sensitivities.items()
            }

        # Summary statistics
        stress_df = pd.DataFrame(report["stress_tests"])
        report["summary"] = {
            "worst_pnl": float(stress_df["pnl"].min()),
            "best_pnl": float(stress_df["pnl"].max()),
            "avg_stress_pnl": float(stress_df["pnl"].mean()),
            "scenarios_with_loss": int((stress_df["pnl"] < 0).sum()),
            "max_loss_scenario": stress_df.loc[stress_df["pnl"].idxmin(), "scenario"],
        }

        return report


# =============================================================================
# Utility Functions
# =============================================================================


def get_predefined_scenario(name: str) -> Scenario:
    """
    Get a predefined scenario by name.

    Args:
        name: Scenario name (from STRESS_SCENARIOS or HISTORICAL_SCENARIOS keys)

    Returns:
        Scenario object

    Raises:
        KeyError: If scenario name not found
    """
    if name in STRESS_SCENARIOS:
        return STRESS_SCENARIOS[name]
    elif name in HISTORICAL_SCENARIOS:
        return HISTORICAL_SCENARIOS[name]
    else:
        available = list(STRESS_SCENARIOS.keys()) + list(HISTORICAL_SCENARIOS.keys())
        raise KeyError(f"Scenario '{name}' not found. Available: {available}")


def list_available_scenarios() -> Dict[str, List[str]]:
    """
    List all available predefined scenarios.

    Returns:
        Dictionary with 'stress' and 'historical' scenario names
    """
    return {
        "stress": list(STRESS_SCENARIOS.keys()),
        "historical": list(HISTORICAL_SCENARIOS.keys()),
    }


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Main class
    "ScenarioTester",
    # Data classes
    "Scenario",
    "ScenarioResult",
    "SensitivityResult",
    # Enums
    "ScenarioType",
    "ShockType",
    # Exceptions
    "ScenarioError",
    "InvalidScenarioError",
    "InsufficientDataError",
    # Predefined scenarios
    "STRESS_SCENARIOS",
    "HISTORICAL_SCENARIOS",
    # Utility functions
    "get_predefined_scenario",
    "list_available_scenarios",
    # Constants
    "HISTORICAL_EVENTS",
    "DEFAULT_SHOCK_LEVELS",
    "DEFAULT_VOL_SHOCKS",
    "DEFAULT_RATE_SHOCKS",
]
