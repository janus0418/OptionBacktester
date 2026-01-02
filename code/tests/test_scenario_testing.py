"""
Comprehensive Tests for Scenario Testing Module

This module contains extensive tests for:
    - ScenarioTester: Stress testing and scenario application
    - Predefined scenarios (stress and historical)
    - Sensitivity analysis
    - What-if analysis
    - Scenario composition

Test Coverage:
    - Unit tests for each method
    - Scenario validation tests
    - P&L calculation accuracy tests
    - Edge case handling

Requirements:
    - pytest
    - numpy
    - pandas

Run Tests:
    pytest tests/test_scenario_testing.py -v --tb=short

Author: OptionsBacktester2 Team
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Any, Dict, List

# Import scenario testing classes
from backtester.analytics.scenario_testing import (
    ScenarioTester,
    Scenario,
    ScenarioResult,
    SensitivityResult,
    ScenarioType,
    ShockType,
    ScenarioError,
    InvalidScenarioError,
    InsufficientDataError,
    STRESS_SCENARIOS,
    HISTORICAL_SCENARIOS,
    DEFAULT_SHOCK_LEVELS,
    DEFAULT_VOL_SHOCKS,
    DEFAULT_RATE_SHOCKS,
    get_predefined_scenario,
    list_available_scenarios,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_greeks():
    """Create sample portfolio Greeks for testing."""
    return {
        "delta": 100.0,
        "gamma": 5.0,
        "theta": -50.0,
        "vega": 200.0,
        "rho": 25.0,
    }


@pytest.fixture
def long_delta_greeks():
    """Create long delta biased Greeks."""
    return {
        "delta": 500.0,
        "gamma": 10.0,
        "theta": -100.0,
        "vega": 150.0,
        "rho": 50.0,
    }


@pytest.fixture
def short_vega_greeks():
    """Create short vega (short premium) Greeks."""
    return {
        "delta": 20.0,
        "gamma": 2.0,
        "theta": 100.0,  # Positive theta (collecting time decay)
        "vega": -300.0,  # Short vega
        "rho": 10.0,
    }


@pytest.fixture
def zero_greeks():
    """Create zero Greeks (flat position)."""
    return {
        "delta": 0.0,
        "gamma": 0.0,
        "theta": 0.0,
        "vega": 0.0,
        "rho": 0.0,
    }


@pytest.fixture
def sample_position_value():
    """Sample position value for testing."""
    return 10000.0


@pytest.fixture
def sample_underlying_price():
    """Sample underlying price for testing."""
    return 450.0


# =============================================================================
# Tests for Scenario Data Class
# =============================================================================


class TestScenarioDataClass:
    """Tests for the Scenario dataclass."""

    def test_create_basic_scenario(self):
        """Test creating a basic scenario."""
        scenario = Scenario(name="Test Scenario", spot_shock=-0.10, vol_shock=0.05)

        assert scenario.name == "Test Scenario"
        assert scenario.spot_shock == -0.10
        assert scenario.vol_shock == 0.05
        assert scenario.rate_shock == 0.0
        assert scenario.time_shock == 0.0

    def test_scenario_with_all_parameters(self):
        """Test scenario with all parameters."""
        scenario = Scenario(
            name="Full Scenario",
            spot_shock=-0.15,
            vol_shock=0.20,
            rate_shock=-0.01,
            time_shock=5.0,
            scenario_type=ScenarioType.STRESS,
            description="Test description",
        )

        assert scenario.spot_shock == -0.15
        assert scenario.vol_shock == 0.20
        assert scenario.rate_shock == -0.01
        assert scenario.time_shock == 5.0
        assert scenario.scenario_type == ScenarioType.STRESS
        assert scenario.description == "Test description"

    def test_invalid_spot_shock_raises_error(self):
        """Test that spot_shock < -1.0 raises error."""
        with pytest.raises(InvalidScenarioError):
            Scenario(name="Invalid", spot_shock=-1.5)  # Can't lose > 100%


# =============================================================================
# Tests for Apply Scenario
# =============================================================================


class TestApplyScenario:
    """Tests for applying scenarios to portfolios."""

    def test_apply_basic_scenario(self, sample_greeks, sample_position_value):
        """Test applying a basic scenario."""
        scenario = Scenario(name="Test", spot_shock=-0.10, vol_shock=0.05)

        result = ScenarioTester.apply_scenario(
            sample_greeks, sample_position_value, scenario
        )

        assert isinstance(result, ScenarioResult)
        assert result.scenario.name == "Test"
        assert "delta_pnl" in result.pnl_breakdown
        assert "gamma_pnl" in result.pnl_breakdown
        assert "vega_pnl" in result.pnl_breakdown
        assert "theta_pnl" in result.pnl_breakdown
        assert "rho_pnl" in result.pnl_breakdown

    def test_delta_pnl_calculation(
        self, sample_position_value, sample_underlying_price
    ):
        """Test delta P&L calculation accuracy."""
        greeks = {"delta": 100, "gamma": 0, "theta": 0, "vega": 0, "rho": 0}
        scenario = Scenario(name="Delta Test", spot_shock=-0.10)

        result = ScenarioTester.apply_scenario(
            greeks, sample_position_value, scenario, sample_underlying_price
        )

        # Delta P&L = delta * price_move = 100 * (-0.10 * 450) = -4500
        expected_delta_pnl = 100 * (-0.10 * sample_underlying_price)
        assert abs(result.pnl_breakdown["delta_pnl"] - expected_delta_pnl) < 1e-6

    def test_gamma_pnl_calculation(
        self, sample_position_value, sample_underlying_price
    ):
        """Test gamma P&L calculation accuracy."""
        greeks = {"delta": 0, "gamma": 10, "theta": 0, "vega": 0, "rho": 0}
        scenario = Scenario(name="Gamma Test", spot_shock=-0.10)

        result = ScenarioTester.apply_scenario(
            greeks, sample_position_value, scenario, sample_underlying_price
        )

        # Gamma P&L = 0.5 * gamma * dS^2
        dollar_move = -0.10 * sample_underlying_price
        expected_gamma_pnl = 0.5 * 10 * dollar_move**2
        assert abs(result.pnl_breakdown["gamma_pnl"] - expected_gamma_pnl) < 1e-6

    def test_vega_pnl_calculation(self, sample_position_value):
        """Test vega P&L calculation accuracy."""
        greeks = {"delta": 0, "gamma": 0, "theta": 0, "vega": 100, "rho": 0}
        scenario = Scenario(name="Vega Test", vol_shock=0.10)

        result = ScenarioTester.apply_scenario(greeks, sample_position_value, scenario)

        # Vega P&L = vega * vol_change_pct = 100 * 10 = 1000
        expected_vega_pnl = 100 * 10  # 10% vol move = 10 vol points
        assert abs(result.pnl_breakdown["vega_pnl"] - expected_vega_pnl) < 1e-6

    def test_theta_pnl_calculation(self, sample_position_value):
        """Test theta P&L calculation accuracy."""
        greeks = {"delta": 0, "gamma": 0, "theta": -50, "vega": 0, "rho": 0}
        scenario = Scenario(name="Theta Test", time_shock=5.0)

        result = ScenarioTester.apply_scenario(greeks, sample_position_value, scenario)

        # Theta P&L = theta * days = -50 * 5 = -250
        expected_theta_pnl = -50 * 5
        assert abs(result.pnl_breakdown["theta_pnl"] - expected_theta_pnl) < 1e-6

    def test_combined_pnl(self, sample_greeks, sample_position_value):
        """Test combined P&L equals sum of components."""
        scenario = Scenario(
            name="Combined", spot_shock=-0.10, vol_shock=0.10, time_shock=1.0
        )

        result = ScenarioTester.apply_scenario(
            sample_greeks, sample_position_value, scenario
        )

        # Total P&L should equal sum of components
        component_sum = sum(result.pnl_breakdown.values())
        assert abs(result.pnl_estimate - component_sum) < 1e-6

    def test_zero_greeks_zero_pnl(self, zero_greeks, sample_position_value):
        """Test that zero Greeks produce zero P&L."""
        scenario = Scenario(
            name="Zero Test",
            spot_shock=-0.20,
            vol_shock=0.30,
            rate_shock=-0.01,
            time_shock=10,
        )

        result = ScenarioTester.apply_scenario(
            zero_greeks, sample_position_value, scenario
        )

        assert result.pnl_estimate == 0.0

    def test_result_contains_risk_metrics(self, sample_greeks, sample_position_value):
        """Test that result contains risk metrics."""
        scenario = Scenario(name="Risk Test", spot_shock=-0.10)

        result = ScenarioTester.apply_scenario(
            sample_greeks, sample_position_value, scenario
        )

        assert "pnl_pct" in result.risk_metrics
        assert "position_value_after" in result.risk_metrics
        assert "max_component" in result.risk_metrics


# =============================================================================
# Tests for Multiple Scenarios
# =============================================================================


class TestApplyMultipleScenarios:
    """Tests for applying multiple scenarios."""

    def test_apply_multiple_scenarios(self, sample_greeks, sample_position_value):
        """Test applying multiple scenarios at once."""
        scenarios = [
            Scenario(name="Scenario 1", spot_shock=-0.10),
            Scenario(name="Scenario 2", vol_shock=0.20),
            Scenario(name="Scenario 3", spot_shock=-0.05, vol_shock=0.10),
        ]

        results = ScenarioTester.apply_multiple_scenarios(
            sample_greeks, sample_position_value, scenarios
        )

        assert len(results) == 3
        assert "Scenario 1" in results
        assert "Scenario 2" in results
        assert "Scenario 3" in results

    def test_results_are_scenario_results(self, sample_greeks, sample_position_value):
        """Test that all results are ScenarioResult objects."""
        scenarios = [
            Scenario(name="Test 1", spot_shock=-0.10),
            Scenario(name="Test 2", vol_shock=0.10),
        ]

        results = ScenarioTester.apply_multiple_scenarios(
            sample_greeks, sample_position_value, scenarios
        )

        for name, result in results.items():
            assert isinstance(result, ScenarioResult)


# =============================================================================
# Tests for Stress Test Suite
# =============================================================================


class TestStressTestSuite:
    """Tests for the stress test suite."""

    def test_run_stress_test_suite(self, sample_greeks, sample_position_value):
        """Test running the complete stress test suite."""
        df = ScenarioTester.run_stress_test_suite(sample_greeks, sample_position_value)

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "scenario" in df.columns
        assert "pnl" in df.columns
        assert "pnl_pct" in df.columns

    def test_stress_test_includes_all_scenarios(
        self, sample_greeks, sample_position_value
    ):
        """Test that stress test includes both stress and historical scenarios."""
        df = ScenarioTester.run_stress_test_suite(
            sample_greeks, sample_position_value, include_historical=True
        )

        # Should have stress + historical scenarios
        total_scenarios = len(STRESS_SCENARIOS) + len(HISTORICAL_SCENARIOS)
        assert len(df) == total_scenarios

    def test_stress_test_without_historical(self, sample_greeks, sample_position_value):
        """Test stress test without historical scenarios."""
        df = ScenarioTester.run_stress_test_suite(
            sample_greeks, sample_position_value, include_historical=False
        )

        assert len(df) == len(STRESS_SCENARIOS)

    def test_stress_test_sorted_by_pnl(self, sample_greeks, sample_position_value):
        """Test that results are sorted by P&L ascending."""
        df = ScenarioTester.run_stress_test_suite(sample_greeks, sample_position_value)

        # First row should have worst (lowest) P&L
        assert df["pnl"].iloc[0] <= df["pnl"].iloc[-1]

    def test_identify_worst_scenarios(self, sample_greeks, sample_position_value):
        """Test identifying worst scenarios."""
        worst = ScenarioTester.identify_worst_scenarios(
            sample_greeks, sample_position_value, top_n=3
        )

        assert len(worst) == 3
        # Should be sorted worst first
        assert worst[0].pnl_estimate <= worst[1].pnl_estimate
        assert worst[1].pnl_estimate <= worst[2].pnl_estimate


# =============================================================================
# Tests for Sensitivity Analysis
# =============================================================================


class TestSensitivityAnalysis:
    """Tests for sensitivity analysis methods."""

    def test_spot_sensitivity(self, sample_greeks, sample_position_value):
        """Test spot sensitivity analysis."""
        result = ScenarioTester.spot_sensitivity(sample_greeks, sample_position_value)

        assert isinstance(result, SensitivityResult)
        assert result.factor == "spot"
        assert len(result.shock_levels) == len(DEFAULT_SHOCK_LEVELS)
        assert len(result.pnl_values) == len(result.shock_levels)

    def test_spot_sensitivity_custom_shocks(self, sample_greeks, sample_position_value):
        """Test spot sensitivity with custom shock levels."""
        custom_shocks = [-0.30, -0.15, 0.0, 0.15, 0.30]

        result = ScenarioTester.spot_sensitivity(
            sample_greeks, sample_position_value, shock_levels=custom_shocks
        )

        assert result.shock_levels == custom_shocks
        assert len(result.pnl_values) == len(custom_shocks)

    def test_vol_sensitivity(self, sample_greeks, sample_position_value):
        """Test volatility sensitivity analysis."""
        result = ScenarioTester.vol_sensitivity(sample_greeks, sample_position_value)

        assert isinstance(result, SensitivityResult)
        assert result.factor == "vol"
        assert len(result.shock_levels) == len(DEFAULT_VOL_SHOCKS)

    def test_rate_sensitivity(self, sample_greeks, sample_position_value):
        """Test rate sensitivity analysis."""
        result = ScenarioTester.rate_sensitivity(sample_greeks, sample_position_value)

        assert isinstance(result, SensitivityResult)
        assert result.factor == "rate"
        assert len(result.shock_levels) == len(DEFAULT_RATE_SHOCKS)

    def test_time_decay_analysis(self, sample_greeks, sample_position_value):
        """Test time decay analysis."""
        result = ScenarioTester.time_decay_analysis(
            sample_greeks, sample_position_value
        )

        assert isinstance(result, SensitivityResult)
        assert result.factor == "time"
        assert len(result.pnl_values) > 0

    def test_full_sensitivity_analysis(self, sample_greeks, sample_position_value):
        """Test running full sensitivity analysis."""
        results = ScenarioTester.full_sensitivity_analysis(
            sample_greeks, sample_position_value
        )

        assert "spot" in results
        assert "vol" in results
        assert "rate" in results
        assert "time" in results

        for factor, result in results.items():
            assert isinstance(result, SensitivityResult)

    def test_long_delta_spot_sensitivity(
        self, sample_position_value, sample_underlying_price
    ):
        """Test that long delta has expected P&L pattern on spot moves."""
        # Use zero gamma to ensure delta effect dominates
        long_delta = {"delta": 500, "gamma": 0, "theta": -100, "vega": 150}

        result = ScenarioTester.spot_sensitivity(
            long_delta,
            sample_position_value,
            shock_levels=[-0.10, 0.10],
            underlying_price=sample_underlying_price,
        )

        # With positive delta and underlying_price, down move = negative delta P&L
        # Up move = positive delta P&L
        # The direction of P&L should change between down and up moves
        assert result.pnl_values[0] < result.pnl_values[1]

    def test_short_vega_vol_sensitivity(self, short_vega_greeks, sample_position_value):
        """Test that short vega loses on vol spike."""
        result = ScenarioTester.vol_sensitivity(
            short_vega_greeks, sample_position_value, shock_levels=[-0.10, 0.10]
        )

        # With negative vega, vol down = positive P&L
        assert result.pnl_values[0] > 0
        # With negative vega, vol up = negative P&L
        assert result.pnl_values[1] < 0


# =============================================================================
# Tests for What-If Analysis
# =============================================================================


class TestWhatIfAnalysis:
    """Tests for what-if analysis methods."""

    def test_what_if_spot_vol_matrix(self, sample_greeks, sample_position_value):
        """Test spot-vol what-if matrix."""
        df = ScenarioTester.what_if_spot_vol(sample_greeks, sample_position_value)

        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] > 0  # Has rows
        assert df.shape[1] > 0  # Has columns

    def test_what_if_custom_shocks(self, sample_greeks, sample_position_value):
        """Test what-if with custom shock levels."""
        spot_shocks = [-0.10, 0.0, 0.10]
        vol_shocks = [0.0, 0.10, 0.20]

        df = ScenarioTester.what_if_spot_vol(
            sample_greeks,
            sample_position_value,
            spot_shocks=spot_shocks,
            vol_shocks=vol_shocks,
        )

        assert df.shape[0] == 3  # 3 spot levels
        assert df.shape[1] == 3  # 3 vol levels

    def test_what_if_custom(self, sample_greeks, sample_position_value):
        """Test custom what-if scenario."""
        result = ScenarioTester.what_if_custom(
            sample_greeks,
            sample_position_value,
            spot_shock=-0.15,
            vol_shock=0.25,
            rate_shock=-0.01,
            time_shock=2.0,
        )

        assert isinstance(result, ScenarioResult)
        assert result.scenario.spot_shock == -0.15
        assert result.scenario.vol_shock == 0.25
        assert result.scenario.rate_shock == -0.01
        assert result.scenario.time_shock == 2.0


# =============================================================================
# Tests for Scenario Composition
# =============================================================================


class TestScenarioComposition:
    """Tests for scenario composition methods."""

    def test_compose_scenario(self):
        """Test composing a new scenario from existing."""
        base = STRESS_SCENARIOS["market_crash_moderate"]

        new_scenario = ScenarioTester.compose_scenario(
            base,
            {"vol_shock": 0.10},  # Add extra vol
            "Modified Crash",
        )

        # Vol should be base + extra
        expected_vol = base.vol_shock + 0.10
        assert new_scenario.vol_shock == expected_vol
        # Other parameters unchanged
        assert new_scenario.spot_shock == base.spot_shock
        assert new_scenario.rate_shock == base.rate_shock

    def test_create_custom_scenario(self):
        """Test creating custom scenario."""
        scenario = ScenarioTester.create_custom_scenario(
            name="My Scenario",
            spot_shock=-0.12,
            vol_shock=0.15,
            rate_shock=-0.005,
            description="Custom test",
        )

        assert scenario.name == "My Scenario"
        assert scenario.spot_shock == -0.12
        assert scenario.vol_shock == 0.15
        assert scenario.rate_shock == -0.005
        assert scenario.scenario_type == ScenarioType.CUSTOM


# =============================================================================
# Tests for Predefined Scenarios
# =============================================================================


class TestPredefinedScenarios:
    """Tests for predefined scenario access."""

    def test_get_predefined_stress_scenario(self):
        """Test getting predefined stress scenario."""
        scenario = get_predefined_scenario("market_crash_severe")

        assert scenario.name == "Severe Market Crash"
        assert scenario.spot_shock < 0  # Should be negative (crash)
        assert scenario.vol_shock > 0  # Should be positive (vol spike)

    def test_get_predefined_historical_scenario(self):
        """Test getting predefined historical scenario."""
        scenario = get_predefined_scenario("lehman_2008")

        assert "Lehman" in scenario.name
        assert scenario.scenario_type == ScenarioType.HISTORICAL

    def test_get_invalid_scenario_raises_error(self):
        """Test that invalid scenario name raises error."""
        with pytest.raises(KeyError):
            get_predefined_scenario("nonexistent_scenario")

    def test_list_available_scenarios(self):
        """Test listing available scenarios."""
        available = list_available_scenarios()

        assert "stress" in available
        assert "historical" in available
        assert len(available["stress"]) > 0
        assert len(available["historical"]) > 0

    def test_all_stress_scenarios_valid(self, sample_greeks, sample_position_value):
        """Test that all stress scenarios can be applied."""
        for name, scenario in STRESS_SCENARIOS.items():
            result = ScenarioTester.apply_scenario(
                sample_greeks, sample_position_value, scenario
            )
            assert isinstance(result, ScenarioResult)

    def test_all_historical_scenarios_valid(self, sample_greeks, sample_position_value):
        """Test that all historical scenarios can be applied."""
        for name, scenario in HISTORICAL_SCENARIOS.items():
            result = ScenarioTester.apply_scenario(
                sample_greeks, sample_position_value, scenario
            )
            assert isinstance(result, ScenarioResult)


# =============================================================================
# Tests for Reporting
# =============================================================================


class TestReporting:
    """Tests for stress report generation."""

    def test_generate_stress_report(self, sample_greeks, sample_position_value):
        """Test generating stress report."""
        report = ScenarioTester.generate_stress_report(
            sample_greeks, sample_position_value
        )

        assert "portfolio_summary" in report
        assert "stress_tests" in report
        assert "worst_scenarios" in report
        assert "summary" in report

    def test_report_includes_sensitivity(self, sample_greeks, sample_position_value):
        """Test that report includes sensitivity analysis."""
        report = ScenarioTester.generate_stress_report(
            sample_greeks, sample_position_value, include_sensitivity=True
        )

        assert "sensitivity" in report
        assert "spot" in report["sensitivity"]
        assert "vol" in report["sensitivity"]

    def test_report_without_sensitivity(self, sample_greeks, sample_position_value):
        """Test report without sensitivity analysis."""
        report = ScenarioTester.generate_stress_report(
            sample_greeks, sample_position_value, include_sensitivity=False
        )

        assert "sensitivity" not in report

    def test_report_summary_statistics(self, sample_greeks, sample_position_value):
        """Test that report contains summary statistics."""
        report = ScenarioTester.generate_stress_report(
            sample_greeks, sample_position_value
        )

        summary = report["summary"]
        assert "worst_pnl" in summary
        assert "best_pnl" in summary
        assert "avg_stress_pnl" in summary
        assert "max_loss_scenario" in summary


# =============================================================================
# Tests for Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_very_large_shock(self, sample_greeks, sample_position_value):
        """Test handling of very large but valid shock."""
        scenario = Scenario(name="Large Shock", spot_shock=-0.99)

        result = ScenarioTester.apply_scenario(
            sample_greeks, sample_position_value, scenario
        )

        assert isinstance(result, ScenarioResult)

    def test_zero_position_value(self, sample_greeks):
        """Test handling of zero position value."""
        result = ScenarioTester.apply_scenario(
            sample_greeks, 0.0, Scenario(name="Zero", spot_shock=-0.10)
        )

        # P&L percentage should be 0 (not error)
        assert result.risk_metrics["pnl_pct"] == 0.0

    def test_partial_greeks(self, sample_position_value):
        """Test with only some Greeks provided."""
        partial_greeks = {"delta": 100}  # Only delta

        result = ScenarioTester.apply_scenario(
            partial_greeks,
            sample_position_value,
            Scenario(name="Partial", spot_shock=-0.10, vol_shock=0.10),
        )

        # Should still work, missing Greeks treated as 0
        assert isinstance(result, ScenarioResult)
        assert result.pnl_breakdown["vega_pnl"] == 0.0

    def test_empty_greeks(self, sample_position_value):
        """Test with empty Greeks dictionary."""
        result = ScenarioTester.apply_scenario(
            {}, sample_position_value, Scenario(name="Empty", spot_shock=-0.20)
        )

        # All P&L should be zero
        assert result.pnl_estimate == 0.0


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for scenario testing workflow."""

    def test_complete_stress_workflow(self, sample_greeks, sample_position_value):
        """Test complete stress testing workflow."""
        # Step 1: Run stress test suite
        stress_df = ScenarioTester.run_stress_test_suite(
            sample_greeks, sample_position_value
        )

        # Step 2: Identify worst scenarios
        worst = ScenarioTester.identify_worst_scenarios(
            sample_greeks, sample_position_value, top_n=3
        )

        # Step 3: Run sensitivity analysis
        sensitivities = ScenarioTester.full_sensitivity_analysis(
            sample_greeks, sample_position_value
        )

        # Step 4: Generate report
        report = ScenarioTester.generate_stress_report(
            sample_greeks, sample_position_value
        )

        # Verify all components work together
        assert len(stress_df) > 0
        assert len(worst) == 3
        assert len(sensitivities) == 4
        assert "summary" in report

    def test_scenario_comparison(self, sample_position_value, sample_underlying_price):
        """Test comparing multiple strategy profiles under same scenarios."""
        # Compare long delta vs short vega strategies
        # Use zero gamma to isolate delta effect
        long_delta = {"delta": 500, "gamma": 0, "theta": -100, "vega": 50}
        short_vega = {"delta": 20, "gamma": 0, "theta": 100, "vega": -300}

        crash_scenario = STRESS_SCENARIOS["market_crash_severe"]

        result_long = ScenarioTester.apply_scenario(
            long_delta, sample_position_value, crash_scenario, sample_underlying_price
        )
        result_short = ScenarioTester.apply_scenario(
            short_vega, sample_position_value, crash_scenario, sample_underlying_price
        )

        # Long delta should lose more on crash (due to large positive delta)
        # Short vega should also lose (due to vol spike)
        assert result_long.pnl_breakdown["delta_pnl"] < 0  # Delta loss
        assert result_short.pnl_breakdown["vega_pnl"] < 0  # Vega loss from vol spike
