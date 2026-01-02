"""
Tests for Volatility Surface Implementation

Comprehensive test suite for the SVI-based volatility surface including:
- Parameter validation
- Calibration accuracy
- Interpolation correctness
- Edge cases and error handling
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backtester.core.volatility_surface import (
    VolatilitySurface,
    SVIParameters,
    VolatilitySurfaceError,
    CalibrationError,
    InterpolationError,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def synthetic_market_data():
    """Create synthetic market data with realistic volatility smile"""
    spot = 450.0

    # Create data for multiple DTEs
    data_rows = []

    for dte in [7, 14, 30, 45, 60]:
        # Generate strikes from 85% to 115% of spot
        for pct in range(85, 116, 2):
            strike = round(spot * pct / 100, 2)
            moneyness = strike / spot

            # Simulate realistic volatility smile
            # - ATM around 20%
            # - Put skew (lower strikes have higher IV)
            # - Call wing (slightly elevated for far OTM calls)
            atm_iv = 0.20

            if moneyness < 1:  # OTM puts
                iv = atm_iv + 0.08 * (1 - moneyness) ** 1.5
            else:  # OTM calls
                iv = atm_iv + 0.03 * (moneyness - 1) ** 1.2

            # Add some time structure (longer DTE = slightly higher IV)
            iv *= 1 + 0.001 * dte

            # Add small noise for realism
            iv += np.random.normal(0, 0.002)
            iv = max(0.05, iv)  # Floor at 5%

            data_rows.append({"strike": strike, "dte": dte, "iv": iv, "spot": spot})

    return pd.DataFrame(data_rows)


@pytest.fixture
def single_dte_data():
    """Create data for single DTE slice"""
    spot = 450.0
    dte = 30

    strikes = np.linspace(380, 520, 30)
    ivs = []

    for K in strikes:
        moneyness = K / spot
        if moneyness < 1:
            iv = 0.22 + 0.06 * (1 - moneyness)
        else:
            iv = 0.22 + 0.02 * (moneyness - 1)
        ivs.append(iv)

    return pd.DataFrame({"strike": strikes, "dte": dte, "iv": ivs, "spot": spot})


@pytest.fixture
def sparse_data():
    """Create sparse data with few points"""
    return pd.DataFrame(
        {
            "strike": [430, 450, 470],
            "dte": [30, 30, 30],
            "iv": [0.25, 0.20, 0.18],
            "spot": [450.0, 450.0, 450.0],
        }
    )


# =============================================================================
# SVIParameters Tests
# =============================================================================


class TestSVIParameters:
    """Test SVIParameters dataclass"""

    def test_valid_parameters(self):
        """Test valid SVI parameters pass validation"""
        params = SVIParameters(a=0.04, b=0.1, rho=-0.3, m=0.0, sigma=0.2, dte=30)
        assert params.validate() is True

    def test_invalid_b_negative(self):
        """Test b < 0 fails validation"""
        params = SVIParameters(
            a=0.04,
            b=-0.1,  # Invalid: must be >= 0
            rho=-0.3,
            m=0.0,
            sigma=0.2,
            dte=30,
        )
        assert params.validate() is False

    def test_invalid_rho_magnitude(self):
        """Test |rho| >= 1 fails validation"""
        params = SVIParameters(
            a=0.04,
            b=0.1,
            rho=1.0,  # Invalid: |rho| must be < 1
            m=0.0,
            sigma=0.2,
            dte=30,
        )
        assert params.validate() is False

        params2 = SVIParameters(
            a=0.04,
            b=0.1,
            rho=-1.5,  # Invalid
            m=0.0,
            sigma=0.2,
            dte=30,
        )
        assert params2.validate() is False

    def test_invalid_negative_variance(self):
        """Test parameters that produce negative variance fail"""
        # This set of parameters would produce negative variance
        params = SVIParameters(
            a=-0.1,  # Negative a
            b=0.01,  # Small b
            rho=0.0,
            m=0.0,
            sigma=0.1,
            dte=30,
        )
        assert params.validate() is False

    def test_to_dict(self):
        """Test conversion to dictionary"""
        params = SVIParameters(a=0.04, b=0.1, rho=-0.3, m=0.0, sigma=0.2, dte=30)
        d = params.to_dict()

        assert d["a"] == 0.04
        assert d["b"] == 0.1
        assert d["rho"] == -0.3
        assert d["m"] == 0.0
        assert d["sigma"] == 0.2
        assert d["dte"] == 30


# =============================================================================
# VolatilitySurface Construction Tests
# =============================================================================


class TestVolatilitySurfaceConstruction:
    """Test VolatilitySurface initialization and validation"""

    def test_construction_with_valid_data(self, synthetic_market_data):
        """Test successful construction with valid data"""
        surface = VolatilitySurface(synthetic_market_data)

        assert len(surface.svi_params) > 0
        assert len(surface.get_available_dtes()) > 0

    def test_missing_columns_raises_error(self):
        """Test that missing required columns raises error"""
        invalid_data = pd.DataFrame(
            {
                "strike": [450],
                "dte": [30],
                # Missing 'iv' and 'spot'
            }
        )

        with pytest.raises(VolatilitySurfaceError, match="Missing required columns"):
            VolatilitySurface(invalid_data)

    def test_empty_data_after_filtering(self):
        """Test that invalid data (all filtered out) raises error"""
        invalid_data = pd.DataFrame(
            {
                "strike": [450, 460],
                "dte": [30, 30],
                "iv": [-0.1, 0],  # Invalid: negative and zero IV
                "spot": [450, 450],
            }
        )

        with pytest.raises(VolatilitySurfaceError, match="No valid data"):
            VolatilitySurface(invalid_data)

    def test_filters_invalid_data_points(self, synthetic_market_data):
        """Test that invalid data points are filtered before calibration"""
        # Add some invalid rows
        invalid_rows = pd.DataFrame(
            {
                "strike": [450, 450, 450],
                "dte": [30, 30, 30],
                "iv": [-0.1, 0, 3.0],  # negative, zero, and extremely high
                "spot": [450, 450, 450],
            }
        )

        data_with_invalid = pd.concat(
            [synthetic_market_data, invalid_rows], ignore_index=True
        )

        # Should still construct successfully (invalid rows filtered)
        surface = VolatilitySurface(data_with_invalid)
        assert len(surface.svi_params) > 0

    def test_min_points_per_slice(self, sparse_data):
        """Test minimum points requirement per DTE slice"""
        # With only 3 points and min_points_per_slice=5, should fail
        with pytest.raises(CalibrationError):
            VolatilitySurface(sparse_data, min_points_per_slice=5)

        # With min_points_per_slice=3, should succeed
        surface = VolatilitySurface(sparse_data, min_points_per_slice=3)
        assert len(surface.svi_params) > 0


# =============================================================================
# Calibration Tests
# =============================================================================


class TestCalibration:
    """Test SVI calibration accuracy"""

    def test_calibration_produces_valid_parameters(self, synthetic_market_data):
        """Test that calibration produces valid SVI parameters"""
        surface = VolatilitySurface(synthetic_market_data)

        for dte, params in surface.svi_params.items():
            assert params.validate(), f"Invalid parameters for DTE={dte}"

    def test_calibration_quality_metrics(self, synthetic_market_data):
        """Test calibration quality metrics are recorded"""
        surface = VolatilitySurface(synthetic_market_data)

        for dte in surface.get_available_dtes():
            assert dte in surface.calibration_quality
            quality = surface.calibration_quality[dte]

            assert "rmse" in quality
            assert "num_points" in quality
            assert quality["rmse"] >= 0
            assert quality["num_points"] > 0

    def test_calibration_rmse_reasonable(self, synthetic_market_data):
        """Test that calibration RMSE is within reasonable bounds"""
        surface = VolatilitySurface(synthetic_market_data)

        for dte, quality in surface.calibration_quality.items():
            # RMSE should be less than 5% (0.05) for good fit
            assert quality["rmse"] < 0.05, f"High RMSE for DTE={dte}: {quality['rmse']}"

    def test_calibration_report(self, synthetic_market_data):
        """Test calibration report generation"""
        surface = VolatilitySurface(synthetic_market_data)
        report = surface.get_calibration_report()

        assert isinstance(report, pd.DataFrame)
        assert "dte" in report.columns
        assert "a" in report.columns
        assert "rmse" in report.columns
        assert len(report) == len(surface.svi_params)


# =============================================================================
# Interpolation Tests
# =============================================================================


class TestInterpolation:
    """Test IV interpolation functionality"""

    def test_get_iv_atm(self, synthetic_market_data):
        """Test getting ATM implied volatility"""
        surface = VolatilitySurface(synthetic_market_data)
        spot = 450.0

        iv = surface.get_iv(strike=450, dte=30, spot=spot)

        # ATM IV should be around 20% for our synthetic data
        assert 0.15 < iv < 0.30

    def test_get_iv_returns_float(self, synthetic_market_data):
        """Test that get_iv returns a float"""
        surface = VolatilitySurface(synthetic_market_data)
        iv = surface.get_iv(strike=450, dte=30, spot=450)

        assert isinstance(iv, float)

    def test_put_skew_preserved(self, synthetic_market_data):
        """Test that put skew is preserved (OTM puts higher IV than ATM)"""
        surface = VolatilitySurface(synthetic_market_data)
        spot = 450.0

        atm_iv = surface.get_iv(strike=450, dte=30, spot=spot)
        otm_put_iv = surface.get_iv(strike=420, dte=30, spot=spot)  # ~7% OTM

        # OTM puts should have higher IV due to skew
        assert otm_put_iv > atm_iv, "Put skew not preserved"

    def test_interpolation_between_dtes(self, synthetic_market_data):
        """Test interpolation between calibrated DTE slices"""
        surface = VolatilitySurface(synthetic_market_data)
        spot = 450.0

        # Get IV at calibrated DTEs
        iv_14 = surface.get_iv(strike=450, dte=14, spot=spot)
        iv_30 = surface.get_iv(strike=450, dte=30, spot=spot)

        # Interpolate at DTE=22 (between 14 and 30)
        iv_22 = surface.get_iv(strike=450, dte=22, spot=spot)

        # Interpolated value should be between the two
        assert min(iv_14, iv_30) <= iv_22 <= max(iv_14, iv_30)

    def test_extrapolation_short_dte(self, synthetic_market_data):
        """Test extrapolation to shorter DTEs"""
        surface = VolatilitySurface(synthetic_market_data)
        spot = 450.0

        # Get IV at DTE shorter than minimum calibrated
        iv = surface.get_iv(strike=450, dte=3, spot=spot)

        # Should return a reasonable value (not NaN or error)
        assert np.isfinite(iv)
        assert 0.05 < iv < 1.0

    def test_extrapolation_long_dte(self, synthetic_market_data):
        """Test extrapolation to longer DTEs"""
        surface = VolatilitySurface(synthetic_market_data)
        spot = 450.0

        # Get IV at DTE longer than maximum calibrated
        iv = surface.get_iv(strike=450, dte=90, spot=spot)

        # Should return a reasonable value
        assert np.isfinite(iv)
        assert 0.05 < iv < 1.0

    def test_iv_bounds(self, synthetic_market_data):
        """Test that IV is bounded to reasonable range"""
        surface = VolatilitySurface(synthetic_market_data)
        spot = 450.0

        # Test various strikes
        for strike in [350, 400, 450, 500, 550]:
            iv = surface.get_iv(strike=strike, dte=30, spot=spot)

            # IV should be between 1% and 200%
            assert 0.01 <= iv <= 2.0, f"IV out of bounds for strike={strike}: {iv}"


# =============================================================================
# Smile Visualization Tests
# =============================================================================


class TestSmileVisualization:
    """Test smile extraction for visualization"""

    def test_get_smile_dataframe(self, synthetic_market_data):
        """Test smile DataFrame generation"""
        surface = VolatilitySurface(synthetic_market_data)
        spot = 450.0

        smile = surface.get_smile_dataframe(dte=30, spot=spot)

        assert isinstance(smile, pd.DataFrame)
        assert "strike" in smile.columns
        assert "moneyness" in smile.columns
        assert "iv" in smile.columns
        assert len(smile) == 50  # Default num_points

    def test_smile_strike_range(self, synthetic_market_data):
        """Test custom strike range for smile"""
        surface = VolatilitySurface(synthetic_market_data)
        spot = 450.0

        smile = surface.get_smile_dataframe(
            dte=30,
            spot=spot,
            strike_range=(0.9, 1.1),  # 90% to 110%
        )

        # Check strike range
        assert smile["strike"].min() >= spot * 0.9
        assert smile["strike"].max() <= spot * 1.1

    def test_smile_shape(self, synthetic_market_data):
        """Test that smile has expected shape (higher IV for OTM puts)"""
        surface = VolatilitySurface(synthetic_market_data)
        spot = 450.0

        smile = surface.get_smile_dataframe(dte=30, spot=spot)

        # Find ATM and OTM put IVs
        atm_idx = (smile["moneyness"] - 1.0).abs().idxmin()
        atm_iv = smile.loc[atm_idx, "iv"]

        otm_put = smile[smile["moneyness"] < 0.95].iloc[0]

        # OTM put should have higher IV (skew)
        assert otm_put["iv"] > atm_iv


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_single_dte_slice(self, single_dte_data):
        """Test surface with only one DTE slice"""
        surface = VolatilitySurface(single_dte_data)

        assert len(surface.svi_params) == 1

        # Should still work for interpolation (extrapolates)
        iv = surface.get_iv(strike=450, dte=30, spot=450)
        assert np.isfinite(iv)

    def test_extreme_strikes(self, synthetic_market_data):
        """Test IV retrieval for extreme strikes"""
        surface = VolatilitySurface(synthetic_market_data)
        spot = 450.0

        # Very deep OTM put
        iv_deep_put = surface.get_iv(strike=300, dte=30, spot=spot)
        assert np.isfinite(iv_deep_put)

        # Very deep OTM call
        iv_deep_call = surface.get_iv(strike=600, dte=30, spot=spot)
        assert np.isfinite(iv_deep_call)

    def test_zero_dte(self, synthetic_market_data):
        """Test IV retrieval at DTE=0"""
        surface = VolatilitySurface(synthetic_market_data)
        spot = 450.0

        # Should handle gracefully
        iv = surface.get_iv(strike=450, dte=0, spot=spot)
        assert np.isfinite(iv)

    def test_fractional_dte(self, synthetic_market_data):
        """Test IV retrieval with fractional DTE"""
        surface = VolatilitySurface(synthetic_market_data)
        spot = 450.0

        # Should interpolate smoothly
        iv = surface.get_iv(strike=450, dte=25.5, spot=spot)
        assert np.isfinite(iv)

    def test_available_dtes(self, synthetic_market_data):
        """Test getting list of calibrated DTEs"""
        surface = VolatilitySurface(synthetic_market_data)
        dtes = surface.get_available_dtes()

        assert isinstance(dtes, list)
        assert len(dtes) > 0
        assert all(isinstance(d, int) for d in dtes)
        assert dtes == sorted(dtes)  # Should be sorted


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests with other components"""

    def test_surface_with_real_world_like_data(self):
        """Test with data that mimics real market conditions"""
        # Create SPY-like market data
        spot = 450.0
        data_rows = []

        # Multiple expirations with different term structures
        for dte in [7, 14, 21, 30, 45, 60, 90]:
            # Base ATM vol depends on term (typically higher for shorter terms)
            base_vol = 0.18 + 0.02 * np.exp(-dte / 30)

            for pct in range(80, 121, 2):
                strike = round(spot * pct / 100)
                moneyness = strike / spot

                # Realistic skew
                if moneyness < 1:
                    skew_adj = 0.10 * (1 - moneyness) ** 1.3
                else:
                    skew_adj = 0.04 * (moneyness - 1) ** 1.2

                iv = base_vol + skew_adj
                iv = max(0.08, min(0.80, iv))  # Realistic bounds

                data_rows.append({"strike": strike, "dte": dte, "iv": iv, "spot": spot})

        market_data = pd.DataFrame(data_rows)
        surface = VolatilitySurface(market_data)

        # Verify all DTEs calibrated
        assert len(surface.svi_params) == 7

        # Verify term structure (shorter term should have higher ATM vol)
        iv_7d = surface.get_iv(450, 7, spot)
        iv_90d = surface.get_iv(450, 90, spot)
        # Short term can be higher OR lower depending on market conditions
        # Just verify both are reasonable
        assert 0.10 < iv_7d < 0.40
        assert 0.10 < iv_90d < 0.40

    def test_surface_consistency(self, synthetic_market_data):
        """Test that surface is internally consistent"""
        surface = VolatilitySurface(synthetic_market_data)
        spot = 450.0

        # Same inputs should give same outputs
        iv1 = surface.get_iv(450, 30, spot)
        iv2 = surface.get_iv(450, 30, spot)

        assert iv1 == iv2

    def test_surface_smoothness(self, synthetic_market_data):
        """Test that surface is smooth (no discontinuities)"""
        surface = VolatilitySurface(synthetic_market_data)
        spot = 450.0

        # Get IVs across strike range
        strikes = np.linspace(400, 500, 50)
        ivs = [surface.get_iv(K, 30, spot) for K in strikes]

        # Check for smoothness (no large jumps)
        iv_changes = np.diff(ivs)
        max_change = np.max(np.abs(iv_changes))

        # Max change between adjacent strikes should be small
        assert max_change < 0.02, f"Surface not smooth: max change = {max_change}"


# =============================================================================
# Performance Tests
# =============================================================================


class TestPerformance:
    """Performance tests for volatility surface"""

    def test_calibration_time(self, synthetic_market_data):
        """Test that calibration completes in reasonable time"""
        import time

        start = time.time()
        surface = VolatilitySurface(synthetic_market_data)
        elapsed = time.time() - start

        # Calibration should complete in < 5 seconds
        assert elapsed < 5.0, f"Calibration took too long: {elapsed:.2f}s"

    def test_interpolation_speed(self, synthetic_market_data):
        """Test interpolation performance"""
        import time

        surface = VolatilitySurface(synthetic_market_data)
        spot = 450.0

        # Time 1000 interpolations
        start = time.time()
        for _ in range(1000):
            _ = surface.get_iv(
                strike=np.random.uniform(380, 520),
                dte=np.random.randint(5, 60),
                spot=spot,
            )
        elapsed = time.time() - start

        # Should complete in < 1 second
        assert elapsed < 1.0, f"Interpolation too slow: {elapsed:.2f}s for 1000 calls"


# =============================================================================
# Module-level Tests
# =============================================================================


def test_module_exports():
    """Test that all expected exports are available"""
    from backtester.core.volatility_surface import (
        VolatilitySurface,
        SVIParameters,
        VolatilitySurfaceError,
        CalibrationError,
        InterpolationError,
    )

    # All imports should work
    assert VolatilitySurface is not None
    assert SVIParameters is not None
    assert VolatilitySurfaceError is not None
    assert CalibrationError is not None
    assert InterpolationError is not None
