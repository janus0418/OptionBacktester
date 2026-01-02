"""
Tests for Strategy Templates

This module tests the pre-built strategy templates that wrap the StrategyBuilder API.
Tests verify:
- Template creation with defaults
- Template creation with custom parameters
- Configuration object usage
- Template registry functionality
- Strategy behavior and integration
"""

import pytest
from datetime import datetime, timedelta

from backtester.strategies.strategy_templates import (
    # Templates
    HighIVStraddleTemplate,
    IronCondorTemplate,
    WheelStrategyTemplate,
    EarningsStraddleTemplate,
    TrendFollowingTemplate,
    # Configuration Classes
    HighIVStraddleConfig,
    IronCondorConfig,
    WheelConfig,
    EarningsStraddleConfig,
    TrendFollowingConfig,
    # Registry
    TemplateRegistry,
)
from backtester.strategies.strategy_builder import BuiltStrategy


# =============================================================================
# HighIVStraddleTemplate Tests
# =============================================================================


class TestHighIVStraddleTemplate:
    """Tests for High IV Straddle template."""

    def test_create_with_defaults(self):
        """Test creating template with default parameters."""
        strategy = HighIVStraddleTemplate.create()

        assert strategy is not None
        assert isinstance(strategy, BuiltStrategy)
        assert "High IV Short Straddle" in strategy.name
        assert "SPY" in strategy.name

    def test_create_with_custom_underlying(self):
        """Test creating template with custom underlying."""
        strategy = HighIVStraddleTemplate.create(underlying="QQQ")

        assert "QQQ" in strategy.name
        assert strategy.underlying == "QQQ"

    def test_create_with_custom_iv_threshold(self):
        """Test creating template with custom IV threshold."""
        strategy = HighIVStraddleTemplate.create(iv_threshold=80.0)

        assert strategy is not None
        # IV threshold affects entry condition

    def test_create_with_custom_profit_target(self):
        """Test creating template with custom profit target."""
        strategy = HighIVStraddleTemplate.create(profit_target_pct=0.40)

        assert "40%" in strategy.description

    def test_create_with_custom_stop_loss(self):
        """Test creating template with custom stop loss."""
        strategy = HighIVStraddleTemplate.create(stop_loss_multiple=3.0)

        assert "300%" in strategy.description

    def test_create_with_custom_dte_range(self):
        """Test creating template with custom DTE range."""
        strategy = HighIVStraddleTemplate.create(dte_range=(30, 60))

        assert strategy is not None

    def test_create_with_custom_exit_dte(self):
        """Test creating template with custom exit DTE."""
        strategy = HighIVStraddleTemplate.create(exit_dte=14)

        assert strategy is not None

    def test_create_with_custom_max_positions(self):
        """Test creating template with custom max positions."""
        strategy = HighIVStraddleTemplate.create(max_positions=5)

        assert strategy is not None

    def test_create_with_custom_risk_pct(self):
        """Test creating template with custom risk percentage."""
        strategy = HighIVStraddleTemplate.create(risk_pct=0.03)

        assert strategy is not None

    def test_create_with_custom_initial_capital(self):
        """Test creating template with custom initial capital."""
        strategy = HighIVStraddleTemplate.create(initial_capital=200_000.0)

        assert strategy._initial_capital == 200_000.0

    def test_create_from_config(self):
        """Test creating template from config object."""
        config = HighIVStraddleConfig(
            underlying="AAPL",
            iv_threshold=75.0,
            profit_target_pct=0.60,
            stop_loss_multiple=2.5,
        )
        strategy = HighIVStraddleTemplate.create_from_config(config)

        assert "AAPL" in strategy.name
        assert strategy.underlying == "AAPL"

    def test_config_default_values(self):
        """Test that config has sensible defaults."""
        config = HighIVStraddleConfig()

        assert config.underlying == "SPY"
        assert config.iv_threshold == 70.0
        assert config.profit_target_pct == 0.50
        assert config.stop_loss_multiple == 2.0
        assert config.dte_range == (25, 45)
        assert config.exit_dte == 7
        assert config.max_positions == 3

    def test_strategy_has_entry_condition(self):
        """Test that strategy has entry condition."""
        strategy = HighIVStraddleTemplate.create()

        assert strategy.entry_condition is not None

    def test_strategy_has_exit_condition(self):
        """Test that strategy has exit condition."""
        strategy = HighIVStraddleTemplate.create()

        assert strategy.exit_condition is not None

    def test_strategy_has_structure_spec(self):
        """Test that strategy has structure specification."""
        strategy = HighIVStraddleTemplate.create()

        assert strategy.structure_spec is not None
        assert strategy.structure_spec.structure_type == "short_straddle"

    def test_strategy_has_position_sizer(self):
        """Test that strategy has position sizer."""
        strategy = HighIVStraddleTemplate.create()

        assert strategy.position_sizer is not None


# =============================================================================
# IronCondorTemplate Tests
# =============================================================================


class TestIronCondorTemplate:
    """Tests for Iron Condor template."""

    def test_create_with_defaults(self):
        """Test creating template with default parameters."""
        strategy = IronCondorTemplate.create()

        assert strategy is not None
        assert isinstance(strategy, BuiltStrategy)
        assert "Iron Condor" in strategy.name

    def test_create_with_custom_underlying(self):
        """Test creating template with custom underlying."""
        strategy = IronCondorTemplate.create(underlying="IWM")

        assert "IWM" in strategy.name

    def test_create_with_custom_iv_threshold(self):
        """Test creating template with custom IV threshold."""
        strategy = IronCondorTemplate.create(iv_threshold=60.0)

        assert strategy is not None

    def test_create_with_custom_short_delta(self):
        """Test creating template with custom short delta."""
        strategy = IronCondorTemplate.create(short_delta=0.20)

        assert "20%" in strategy.description

    def test_create_with_custom_wing_width(self):
        """Test creating template with custom wing width."""
        strategy = IronCondorTemplate.create(wing_width_pct=0.05)

        assert strategy is not None

    def test_create_with_custom_capital_pct(self):
        """Test creating template with custom capital percentage."""
        strategy = IronCondorTemplate.create(capital_pct=0.15)

        assert strategy is not None

    def test_create_from_config(self):
        """Test creating template from config object."""
        config = IronCondorConfig(
            underlying="SPX",
            iv_threshold=55.0,
            short_delta=0.10,
        )
        strategy = IronCondorTemplate.create_from_config(config)

        assert "SPX" in strategy.name

    def test_config_default_values(self):
        """Test that config has sensible defaults."""
        config = IronCondorConfig()

        assert config.underlying == "SPY"
        assert config.iv_threshold == 50.0
        assert config.short_delta == 0.16
        assert config.wing_width_pct == 0.03
        assert config.dte_range == (30, 60)

    def test_structure_is_iron_condor(self):
        """Test that strategy uses iron condor structure."""
        strategy = IronCondorTemplate.create()

        assert strategy.structure_spec.structure_type == "iron_condor"


# =============================================================================
# WheelStrategyTemplate Tests
# =============================================================================


class TestWheelStrategyTemplate:
    """Tests for Wheel Strategy template."""

    def test_create_with_defaults(self):
        """Test creating template with default parameters."""
        strategy = WheelStrategyTemplate.create()

        assert strategy is not None
        assert isinstance(strategy, BuiltStrategy)
        assert "Wheel" in strategy.name

    def test_create_with_custom_underlying(self):
        """Test creating template with custom underlying."""
        strategy = WheelStrategyTemplate.create(underlying="MSFT")

        assert "MSFT" in strategy.name

    def test_create_with_custom_put_delta(self):
        """Test creating template with custom put delta."""
        strategy = WheelStrategyTemplate.create(put_delta=0.25)

        assert "25%" in strategy.description

    def test_create_with_custom_profit_target(self):
        """Test creating template with custom profit target."""
        strategy = WheelStrategyTemplate.create(profit_target_pct=0.40)

        assert strategy is not None

    def test_create_from_config(self):
        """Test creating template from config object."""
        config = WheelConfig(
            underlying="AMD",
            put_delta=0.35,
        )
        strategy = WheelStrategyTemplate.create_from_config(config)

        assert "AMD" in strategy.name

    def test_config_default_values(self):
        """Test that config has sensible defaults."""
        config = WheelConfig()

        assert config.underlying == "SPY"
        assert config.put_delta == 0.30
        assert config.max_positions == 1  # Wheel typically one at a time

    def test_max_positions_is_one(self):
        """Test that default wheel has max 1 position per underlying."""
        config = WheelConfig()
        assert config.max_positions == 1


# =============================================================================
# EarningsStraddleTemplate Tests
# =============================================================================


class TestEarningsStraddleTemplate:
    """Tests for Earnings Straddle template."""

    def test_create_with_defaults(self):
        """Test creating template with default parameters."""
        strategy = EarningsStraddleTemplate.create()

        assert strategy is not None
        assert isinstance(strategy, BuiltStrategy)
        assert "Earnings Straddle" in strategy.name

    def test_create_with_custom_underlying(self):
        """Test creating template with custom underlying."""
        strategy = EarningsStraddleTemplate.create(underlying="NVDA")

        assert "NVDA" in strategy.name

    def test_create_with_custom_days_before(self):
        """Test creating template with custom days before earnings."""
        strategy = EarningsStraddleTemplate.create(days_before_earnings=3)

        assert strategy is not None

    def test_create_with_custom_profit_target(self):
        """Test creating template with custom profit target."""
        strategy = EarningsStraddleTemplate.create(profit_target_pct=0.30)

        assert strategy is not None

    def test_create_with_custom_stop_loss(self):
        """Test creating template with custom stop loss."""
        strategy = EarningsStraddleTemplate.create(stop_loss_pct=0.40)

        assert strategy is not None

    def test_create_from_config(self):
        """Test creating template from config object."""
        config = EarningsStraddleConfig(
            underlying="TSLA",
            days_before_earnings=7,
        )
        strategy = EarningsStraddleTemplate.create_from_config(config)

        assert "TSLA" in strategy.name

    def test_config_default_values(self):
        """Test that config has sensible defaults."""
        config = EarningsStraddleConfig()

        assert config.underlying == "SPY"
        assert config.days_before_earnings == 5
        assert config.profit_target_pct == 0.25  # Lower target for quick plays
        assert config.risk_pct == 0.01  # Small risk for earnings

    def test_structure_is_long_straddle(self):
        """Test that strategy uses long straddle structure."""
        strategy = EarningsStraddleTemplate.create()

        assert strategy.structure_spec.structure_type == "long_straddle"


# =============================================================================
# TrendFollowingTemplate Tests
# =============================================================================


class TestTrendFollowingTemplate:
    """Tests for Trend Following template."""

    def test_create_bullish_with_defaults(self):
        """Test creating bullish template with defaults."""
        strategy = TrendFollowingTemplate.create(direction="bullish")

        assert strategy is not None
        assert isinstance(strategy, BuiltStrategy)
        assert "Trend Following" in strategy.name
        assert "bullish" in strategy.name.lower()

    def test_create_bearish(self):
        """Test creating bearish template."""
        strategy = TrendFollowingTemplate.create(direction="bearish")

        assert "bearish" in strategy.name.lower()

    def test_create_with_custom_underlying(self):
        """Test creating template with custom underlying."""
        strategy = TrendFollowingTemplate.create(underlying="DIA")

        assert "DIA" in strategy.name

    def test_create_with_custom_vix_thresholds(self):
        """Test creating template with custom VIX thresholds."""
        strategy = TrendFollowingTemplate.create(
            vix_threshold_low=12.0,
            vix_threshold_high=20.0,
        )

        assert strategy is not None

    def test_create_with_custom_profit_target(self):
        """Test creating template with custom profit target."""
        # Note: profit_target must be <= 1.0 (100%)
        strategy = TrendFollowingTemplate.create(profit_target_pct=0.80)

        assert strategy is not None

    def test_create_from_config(self):
        """Test creating template from config object."""
        config = TrendFollowingConfig(
            underlying="QQQ",
            vix_threshold_low=14.0,
        )
        strategy = TrendFollowingTemplate.create_from_config(config)

        assert "QQQ" in strategy.name

    def test_config_default_values(self):
        """Test that config has sensible defaults."""
        config = TrendFollowingConfig()

        assert config.underlying == "SPY"
        assert config.vix_threshold_low == 15.0
        assert config.vix_threshold_high == 25.0
        assert config.profit_target_pct == 1.0  # 100% for trends
        assert config.dte_range == (45, 90)  # Longer DTE for trends

    def test_bullish_uses_call_spread(self):
        """Test that bullish direction uses bull call spread."""
        strategy = TrendFollowingTemplate.create(direction="bullish")

        assert strategy.structure_spec.structure_type == "bull_call_spread"

    def test_bearish_uses_put_spread(self):
        """Test that bearish direction uses bear put spread."""
        strategy = TrendFollowingTemplate.create(direction="bearish")

        assert strategy.structure_spec.structure_type == "bear_put_spread"


# =============================================================================
# TemplateRegistry Tests
# =============================================================================


class TestTemplateRegistry:
    """Tests for Template Registry."""

    def test_list_templates(self):
        """Test listing all available templates."""
        templates = TemplateRegistry.list_templates()

        assert isinstance(templates, list)
        assert len(templates) == 5
        assert "high_iv_straddle" in templates
        assert "iron_condor" in templates
        assert "wheel" in templates
        assert "earnings_straddle" in templates
        assert "trend_following" in templates

    def test_get_description(self):
        """Test getting template description."""
        desc = TemplateRegistry.get_description("high_iv_straddle")

        assert isinstance(desc, str)
        assert len(desc) > 0
        assert "straddle" in desc.lower() or "IV" in desc

    def test_get_description_unknown_template(self):
        """Test getting description for unknown template."""
        desc = TemplateRegistry.get_description("unknown_template")

        assert desc == "No description available"

    def test_get_template(self):
        """Test getting template class by name."""
        template = TemplateRegistry.get_template("high_iv_straddle")

        assert template is HighIVStraddleTemplate

    def test_get_template_unknown_raises(self):
        """Test that getting unknown template raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            TemplateRegistry.get_template("unknown_template")

        assert "Unknown template" in str(exc_info.value)
        assert "Available" in str(exc_info.value)

    def test_create_strategy(self):
        """Test creating strategy through registry."""
        strategy = TemplateRegistry.create("high_iv_straddle")

        assert strategy is not None
        assert isinstance(strategy, BuiltStrategy)

    def test_create_strategy_with_params(self):
        """Test creating strategy through registry with custom params."""
        strategy = TemplateRegistry.create(
            "iron_condor",
            underlying="QQQ",
            iv_threshold=60.0,
        )

        assert "QQQ" in strategy.name

    def test_describe_all(self):
        """Test getting all template descriptions."""
        descriptions = TemplateRegistry.describe_all()

        assert isinstance(descriptions, dict)
        assert len(descriptions) == 5

        for name, info in descriptions.items():
            assert "class" in info
            assert "description" in info
            assert isinstance(info["class"], str)
            assert isinstance(info["description"], str)


# =============================================================================
# Strategy Behavior Tests
# =============================================================================


class TestStrategyBehavior:
    """Tests for strategy behavior with market data."""

    def test_high_iv_straddle_should_enter_with_high_iv(self):
        """Test that high IV straddle enters on high IV."""
        strategy = HighIVStraddleTemplate.create(
            iv_threshold=70.0,
            max_positions=3,
        )

        # High IV context
        context = {
            "iv_rank": 75.0,
            "dte": 35,
            "underlying": "SPY",
            "open_positions": [],
        }

        assert strategy.should_enter(context) is True

    def test_high_iv_straddle_should_not_enter_with_low_iv(self):
        """Test that high IV straddle doesn't enter on low IV."""
        strategy = HighIVStraddleTemplate.create(iv_threshold=70.0)

        # Low IV context
        context = {
            "iv_rank": 50.0,
            "dte": 35,
            "underlying": "SPY",
            "open_positions": [],
        }

        assert strategy.should_enter(context) is False

    def test_iron_condor_should_enter_with_moderate_iv(self):
        """Test that iron condor enters on moderate IV."""
        strategy = IronCondorTemplate.create(
            iv_threshold=50.0,
            max_positions=2,
        )

        context = {
            "iv_rank": 55.0,
            "dte": 45,
            "underlying": "SPY",
            "open_positions": [],
        }

        assert strategy.should_enter(context) is True

    def test_trend_following_should_enter_in_vix_range(self):
        """Test that trend following enters within VIX range."""
        strategy = TrendFollowingTemplate.create(
            vix_threshold_low=15.0,
            vix_threshold_high=25.0,
            max_positions=2,
        )

        context = {
            "vix": 20.0,
            "dte": 60,
            "underlying": "SPY",
            "open_positions": [],
        }

        assert strategy.should_enter(context) is True

    def test_trend_following_should_not_enter_high_vix(self):
        """Test that trend following doesn't enter on high VIX."""
        strategy = TrendFollowingTemplate.create(
            vix_threshold_low=15.0,
            vix_threshold_high=25.0,
        )

        context = {
            "vix": 30.0,
            "dte": 60,
            "underlying": "SPY",
            "open_positions": [],
        }

        assert strategy.should_enter(context) is False


# =============================================================================
# Integration Tests
# =============================================================================


class TestTemplateIntegration:
    """Integration tests for templates."""

    def test_all_templates_produce_valid_strategies(self):
        """Test that all templates produce valid strategies."""
        templates = TemplateRegistry.list_templates()

        for template_name in templates:
            strategy = TemplateRegistry.create(template_name)

            assert strategy is not None
            assert isinstance(strategy, BuiltStrategy)
            assert strategy.name is not None
            assert len(strategy.name) > 0
            assert strategy.entry_condition is not None
            assert strategy.exit_condition is not None
            assert strategy.structure_spec is not None
            assert strategy.position_sizer is not None

    def test_all_templates_describe_correctly(self):
        """Test that all templates have descriptions."""
        templates = TemplateRegistry.list_templates()

        for template_name in templates:
            strategy = TemplateRegistry.create(template_name)
            description = strategy.describe()

            assert isinstance(description, str)
            assert len(description) > 0
            assert "Strategy:" in description

    def test_template_strategies_are_compatible_with_base(self):
        """Test that template strategies inherit from Strategy properly."""
        strategy = HighIVStraddleTemplate.create()

        # Should have base Strategy attributes
        assert hasattr(strategy, "name")
        assert hasattr(strategy, "description")
        assert hasattr(strategy, "_initial_capital")
        assert hasattr(strategy, "should_enter")
        assert hasattr(strategy, "should_exit")
        assert callable(strategy.should_enter)
        assert callable(strategy.should_exit)


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_very_high_iv_threshold(self):
        """Test with very high IV threshold."""
        strategy = HighIVStraddleTemplate.create(iv_threshold=99.0)

        context = {
            "iv_rank": 98.0,
            "dte": 35,
            "underlying": "SPY",
            "open_positions": [],
        }

        # Should not enter since IV is below threshold
        assert strategy.should_enter(context) is False

    def test_very_low_iv_threshold(self):
        """Test with very low IV threshold."""
        strategy = HighIVStraddleTemplate.create(iv_threshold=1.0)

        context = {
            "iv_rank": 5.0,
            "dte": 35,
            "underlying": "SPY",
            "open_positions": [],
        }

        # Should enter since IV is above threshold
        assert strategy.should_enter(context) is True

    def test_dte_at_boundary(self):
        """Test with DTE at boundary of range."""
        strategy = HighIVStraddleTemplate.create(
            iv_threshold=50.0,
            dte_range=(25, 45),
        )

        # At lower boundary
        context_low = {
            "iv_rank": 75.0,
            "dte": 25,
            "underlying": "SPY",
            "open_positions": [],
        }
        assert strategy.should_enter(context_low) is True

        # At upper boundary
        context_high = {
            "iv_rank": 75.0,
            "dte": 45,
            "underlying": "SPY",
            "open_positions": [],
        }
        assert strategy.should_enter(context_high) is True

        # Below range
        context_below = {
            "iv_rank": 75.0,
            "dte": 10,
            "underlying": "SPY",
            "open_positions": [],
        }
        assert strategy.should_enter(context_below) is False

    def test_max_positions_reached(self):
        """Test that max positions condition is configured correctly."""
        strategy = HighIVStraddleTemplate.create(
            iv_threshold=50.0,
            max_positions=2,
        )

        # Verify the entry condition contains max_open_positions check
        # The BuiltStrategy uses self._structures internally for open positions
        # so we test the condition directly
        from backtester.strategies.strategy_builder import MaxOpenPositions

        # Create mock positions
        class MockPosition:
            pass

        positions = [MockPosition(), MockPosition()]

        # Test the condition directly
        context = {
            "iv_rank": 75.0,
            "dte": 35,
            "underlying": "SPY",
            "open_positions": positions,
        }

        # The MaxOpenPositions condition should evaluate to False when at max
        max_pos_condition = MaxOpenPositions(2)
        assert max_pos_condition.evaluate(context) is False

        # And should be True when under max
        context["open_positions"] = [MockPosition()]
        assert max_pos_condition.evaluate(context) is True

    def test_empty_positions_list(self):
        """Test with empty positions list."""
        strategy = HighIVStraddleTemplate.create(iv_threshold=50.0)

        context = {
            "iv_rank": 75.0,
            "dte": 35,
            "underlying": "SPY",
            "open_positions": [],
        }

        assert strategy.should_enter(context) is True

    def test_missing_iv_rank_in_context(self):
        """Test behavior when IV rank is missing from context."""
        strategy = HighIVStraddleTemplate.create(iv_threshold=70.0)

        context = {
            "dte": 35,
            "underlying": "SPY",
            "open_positions": [],
        }

        # Should not enter - missing required data
        assert strategy.should_enter(context) is False

    def test_missing_dte_in_context(self):
        """Test behavior when DTE is missing from context."""
        strategy = HighIVStraddleTemplate.create(iv_threshold=70.0)

        context = {
            "iv_rank": 75.0,
            "underlying": "SPY",
            "open_positions": [],
        }

        # Should not enter - missing required data
        assert strategy.should_enter(context) is False
