"""
Tests for Strategy Builder - Fluent API for Declarative Strategy Creation

Tests cover:
- Condition classes and composition (AND, OR, NOT)
- Entry and exit condition factories
- Structure specification and creation
- Position sizing methods
- StrategyBuilder fluent API
- Built strategy behavior
"""

import pytest
from datetime import datetime, timedelta
from typing import Dict, Any

from backtester.strategies.strategy_builder import (
    # Builder
    StrategyBuilder,
    BuiltStrategy,
    # Condition base
    Condition,
    AndCondition,
    OrCondition,
    NotCondition,
    AlwaysTrue,
    AlwaysFalse,
    # Entry conditions
    IVRankAbove,
    IVRankBelow,
    IVRankBetween,
    VIXAbove,
    VIXBelow,
    VIXBetween,
    DTEAbove,
    DTEBelow,
    DTEBetween,
    DayOfWeek,
    NoOpenPositions,
    MaxOpenPositions,
    # Exit conditions
    ProfitTarget,
    StopLoss,
    FixedStopLoss,
    TrailingStop,
    HoldingPeriod,
    ExpirationApproaching,
    # Condition factories
    iv_rank_above,
    iv_rank_below,
    iv_rank_between,
    vix_above,
    vix_below,
    vix_between,
    dte_above,
    dte_below,
    dte_between,
    day_of_week,
    no_open_positions,
    max_open_positions,
    profit_target,
    stop_loss,
    fixed_stop_loss,
    trailing_stop,
    holding_period,
    expiration_approaching,
    # Structure spec
    StructureSpec,
    short_straddle,
    long_straddle,
    short_strangle,
    long_strangle,
    iron_condor,
    bull_call_spread,
    bear_put_spread,
    # Position sizing
    PositionSizer,
    fixed_contracts,
    risk_percent,
    capital_percent,
    delta_target,
    premium_target,
    # Exceptions
    StrategyBuilderError,
    BuilderValidationError,
    ConditionError,
)


class TestConditionComposition:
    def test_always_true(self):
        cond = AlwaysTrue()
        assert cond.evaluate({}) is True
        assert cond.describe() == "always"

    def test_always_false(self):
        cond = AlwaysFalse()
        assert cond.evaluate({}) is False
        assert cond.describe() == "never"

    def test_and_condition(self):
        true_cond = AlwaysTrue()
        false_cond = AlwaysFalse()

        combined = AndCondition(true_cond, true_cond)
        assert combined.evaluate({}) is True

        combined = AndCondition(true_cond, false_cond)
        assert combined.evaluate({}) is False

        combined = AndCondition(false_cond, false_cond)
        assert combined.evaluate({}) is False

    def test_or_condition(self):
        true_cond = AlwaysTrue()
        false_cond = AlwaysFalse()

        combined = OrCondition(true_cond, false_cond)
        assert combined.evaluate({}) is True

        combined = OrCondition(false_cond, false_cond)
        assert combined.evaluate({}) is False

    def test_not_condition(self):
        true_cond = AlwaysTrue()
        false_cond = AlwaysFalse()

        assert NotCondition(true_cond).evaluate({}) is False
        assert NotCondition(false_cond).evaluate({}) is True

    def test_operator_and(self):
        cond1 = iv_rank_above(50)
        cond2 = dte_above(20)
        combined = cond1 & cond2

        assert isinstance(combined, AndCondition)
        assert combined.evaluate({"iv_rank": 60, "dte": 30}) is True
        assert combined.evaluate({"iv_rank": 60, "dte": 10}) is False

    def test_operator_or(self):
        cond1 = iv_rank_above(80)
        cond2 = vix_above(25)
        combined = cond1 | cond2

        assert isinstance(combined, OrCondition)
        assert combined.evaluate({"iv_rank": 85, "vix": 20}) is True
        assert combined.evaluate({"iv_rank": 70, "vix": 30}) is True
        assert combined.evaluate({"iv_rank": 70, "vix": 20}) is False

    def test_operator_invert(self):
        cond = iv_rank_above(50)
        inverted = ~cond

        assert isinstance(inverted, NotCondition)
        assert inverted.evaluate({"iv_rank": 60}) is False
        assert inverted.evaluate({"iv_rank": 40}) is True

    def test_complex_composition(self):
        cond = (iv_rank_above(70) & dte_between(25, 45)) | vix_above(30)

        assert cond.evaluate({"iv_rank": 75, "dte": 30, "vix": 20}) is True
        assert cond.evaluate({"iv_rank": 60, "dte": 30, "vix": 35}) is True
        assert cond.evaluate({"iv_rank": 60, "dte": 30, "vix": 20}) is False


class TestEntryConditions:
    def test_iv_rank_above(self):
        cond = iv_rank_above(70)
        assert cond.evaluate({"iv_rank": 75}) is True
        assert cond.evaluate({"iv_rank": 65}) is False
        assert cond.evaluate({"iv_rank": 70}) is False
        assert cond.evaluate({}) is False

    def test_iv_rank_above_uses_percentile_fallback(self):
        cond = iv_rank_above(70)
        assert cond.evaluate({"iv_percentile": 75}) is True

    def test_iv_rank_below(self):
        cond = iv_rank_below(30)
        assert cond.evaluate({"iv_rank": 25}) is True
        assert cond.evaluate({"iv_rank": 35}) is False

    def test_iv_rank_between(self):
        cond = iv_rank_between(40, 60)
        assert cond.evaluate({"iv_rank": 50}) is True
        assert cond.evaluate({"iv_rank": 40}) is True
        assert cond.evaluate({"iv_rank": 60}) is True
        assert cond.evaluate({"iv_rank": 39}) is False
        assert cond.evaluate({"iv_rank": 61}) is False

    def test_iv_rank_validation(self):
        with pytest.raises(ConditionError):
            iv_rank_above(150)
        with pytest.raises(ConditionError):
            iv_rank_above(-10)
        with pytest.raises(ConditionError):
            iv_rank_between(60, 40)

    def test_vix_above(self):
        cond = vix_above(20)
        assert cond.evaluate({"vix": 25}) is True
        assert cond.evaluate({"vix": 15}) is False

    def test_vix_below(self):
        cond = vix_below(15)
        assert cond.evaluate({"vix": 12}) is True
        assert cond.evaluate({"vix": 18}) is False

    def test_vix_between(self):
        cond = vix_between(15, 25)
        assert cond.evaluate({"vix": 20}) is True
        assert cond.evaluate({"vix": 30}) is False

    def test_dte_above(self):
        cond = dte_above(30)
        assert cond.evaluate({"dte": 45}) is True
        assert cond.evaluate({"dte": 25}) is False

    def test_dte_below(self):
        cond = dte_below(10)
        assert cond.evaluate({"dte": 5}) is True
        assert cond.evaluate({"dte": 15}) is False

    def test_dte_between(self):
        cond = dte_between(25, 45)
        assert cond.evaluate({"dte": 30}) is True
        assert cond.evaluate({"dte": 50}) is False

    def test_day_of_week(self):
        cond = day_of_week("monday", "wednesday", "friday")

        monday = datetime(2024, 1, 1)  # Monday
        tuesday = datetime(2024, 1, 2)  # Tuesday
        wednesday = datetime(2024, 1, 3)  # Wednesday

        assert cond.evaluate({"date": monday}) is True
        assert cond.evaluate({"date": tuesday}) is False
        assert cond.evaluate({"date": wednesday}) is True

    def test_day_of_week_invalid(self):
        with pytest.raises(ConditionError):
            day_of_week("invalid_day")

    def test_no_open_positions(self):
        cond = no_open_positions()
        assert cond.evaluate({"open_positions": []}) is True
        assert cond.evaluate({}) is True

        class MockPosition:
            def __init__(self, underlying):
                self.underlying = underlying

        assert cond.evaluate({"open_positions": [MockPosition("SPY")]}) is False

    def test_no_open_positions_specific_underlying(self):
        cond = no_open_positions("SPY")

        class MockPosition:
            def __init__(self, underlying):
                self.underlying = underlying

        assert cond.evaluate({"open_positions": [MockPosition("QQQ")]}) is True
        assert cond.evaluate({"open_positions": [MockPosition("SPY")]}) is False

    def test_max_open_positions(self):
        cond = max_open_positions(3)
        assert cond.evaluate({"open_positions": [1, 2]}) is True
        assert cond.evaluate({"open_positions": [1, 2, 3]}) is False


class TestExitConditions:
    def test_profit_target(self):
        cond = profit_target(0.50)
        assert cond.evaluate({"pnl_pct": 0.55}) is True
        assert cond.evaluate({"pnl_pct": 0.50}) is True
        assert cond.evaluate({"pnl_pct": 0.40}) is False

    def test_profit_target_validation(self):
        with pytest.raises(ConditionError):
            profit_target(1.5)
        with pytest.raises(ConditionError):
            profit_target(0)

    def test_stop_loss(self):
        cond = stop_loss(2.0)
        assert cond.evaluate({"pnl_pct": -2.5}) is True
        assert cond.evaluate({"pnl_pct": -2.0}) is True
        assert cond.evaluate({"pnl_pct": -1.5}) is False

    def test_fixed_stop_loss(self):
        cond = fixed_stop_loss(500)
        assert cond.evaluate({"pnl": -600}) is True
        assert cond.evaluate({"pnl": -500}) is True
        assert cond.evaluate({"pnl": -400}) is False

    def test_trailing_stop(self):
        cond = trailing_stop(0.25)
        assert cond.evaluate({"pnl": 750, "peak_pnl": 1000}) is True
        assert cond.evaluate({"pnl": 800, "peak_pnl": 1000}) is False

    def test_holding_period(self):
        cond = holding_period(21)
        entry = datetime(2024, 1, 1)
        current_short = datetime(2024, 1, 15)
        current_long = datetime(2024, 1, 25)

        assert cond.evaluate({"entry_date": entry, "date": current_short}) is False
        assert cond.evaluate({"entry_date": entry, "date": current_long}) is True

    def test_expiration_approaching(self):
        cond = expiration_approaching(7)
        assert cond.evaluate({"dte": 5}) is True
        assert cond.evaluate({"dte": 7}) is True
        assert cond.evaluate({"dte": 10}) is False


class TestStructureSpec:
    def test_short_straddle_spec(self):
        spec = short_straddle(dte=30, quantity=5)
        assert spec.structure_type == "short_straddle"
        assert spec.target_dte == 30
        assert spec.quantity == 5

    def test_long_straddle_spec(self):
        spec = long_straddle(dte=45)
        assert spec.structure_type == "long_straddle"
        assert spec.target_dte == 45

    def test_short_strangle_spec(self):
        spec = short_strangle(dte=30, width=20)
        assert spec.structure_type == "short_strangle"
        assert spec.width == 20

    def test_iron_condor_spec(self):
        spec = iron_condor(dte=45, width=25, wing_width=10)
        assert spec.structure_type == "iron_condor"
        assert spec.width == 25
        assert spec.extra_params.get("wing_width") == 10

    def test_bull_call_spread_spec(self):
        spec = bull_call_spread(dte=30, width=10)
        assert spec.structure_type == "bull_call_spread"

    def test_bear_put_spread_spec(self):
        spec = bear_put_spread(dte=30, width=10)
        assert spec.structure_type == "bear_put_spread"

    def test_structure_describe(self):
        spec = short_straddle(dte=30)
        desc = spec.describe()
        assert "Short Straddle" in desc
        assert "30 DTE" in desc


class TestPositionSizing:
    def test_fixed_contracts(self):
        sizer = fixed_contracts(10)
        result = sizer.calculate(100000, short_straddle(), {})
        assert result == 10

    def test_risk_percent(self):
        sizer = risk_percent(0.02)
        result = sizer.calculate(
            available_capital=100000,
            structure_spec=short_straddle(),
            market_data={"estimated_max_loss": 500},
        )
        assert result == 4

    def test_capital_percent(self):
        sizer = capital_percent(0.10)
        result = sizer.calculate(
            available_capital=100000,
            structure_spec=short_straddle(),
            market_data={"margin_per_contract": 2000},
        )
        assert result == 5

    def test_delta_target(self):
        sizer = delta_target(50)
        result = sizer.calculate(
            available_capital=100000,
            structure_spec=short_straddle(),
            market_data={"delta_per_contract": 10},
        )
        assert result == 5

    def test_premium_target(self):
        sizer = premium_target(1000)
        result = sizer.calculate(
            available_capital=100000,
            structure_spec=short_straddle(),
            market_data={"premium_per_contract": 200},
        )
        assert result == 5

    def test_max_contracts_limit(self):
        sizer = risk_percent(0.50, max_contracts=10)
        result = sizer.calculate(
            available_capital=100000,
            structure_spec=short_straddle(),
            market_data={"estimated_max_loss": 100},
        )
        assert result == 10

    def test_min_contracts(self):
        sizer = risk_percent(0.001)
        result = sizer.calculate(
            100000, short_straddle(), {"estimated_max_loss": 10000}
        )
        assert result >= 1

    def test_sizer_describe(self):
        assert "10 contracts" in fixed_contracts(10).describe()
        assert "risk" in risk_percent(0.02).describe()
        assert "allocate" in capital_percent(0.10).describe()


class TestStrategyBuilder:
    def test_builder_basic(self):
        strategy = (
            StrategyBuilder()
            .name("Test Strategy")
            .entry_condition(iv_rank_above(70))
            .exit_condition(profit_target(0.50))
            .structure(short_straddle(dte=30))
            .build()
        )

        assert strategy.name == "Test Strategy"
        assert isinstance(strategy, BuiltStrategy)

    def test_builder_full_config(self):
        strategy = (
            StrategyBuilder()
            .name("Full Config Strategy")
            .description("A fully configured strategy")
            .underlying("SPY")
            .initial_capital(200000)
            .entry_condition(iv_rank_above(70) & dte_between(25, 45))
            .exit_condition(profit_target(0.50) | stop_loss(2.0) | dte_below(7))
            .structure(short_straddle(dte=30))
            .position_size(risk_percent(0.02))
            .max_positions(5)
            .max_delta(100)
            .max_capital_utilization(0.80)
            .build()
        )

        assert strategy.name == "Full Config Strategy"
        assert strategy.underlying == "SPY"
        assert strategy.initial_capital == 200000
        assert strategy.position_limits.get("max_positions") == 5

    def test_builder_missing_name(self):
        with pytest.raises(BuilderValidationError, match="name is required"):
            (
                StrategyBuilder()
                .entry_condition(iv_rank_above(70))
                .exit_condition(profit_target(0.50))
                .structure(short_straddle())
                .build()
            )

    def test_builder_missing_entry_condition(self):
        with pytest.raises(BuilderValidationError, match="entry_condition is required"):
            (
                StrategyBuilder()
                .name("Test")
                .exit_condition(profit_target(0.50))
                .structure(short_straddle())
                .build()
            )

    def test_builder_missing_exit_condition(self):
        with pytest.raises(BuilderValidationError, match="exit_condition is required"):
            (
                StrategyBuilder()
                .name("Test")
                .entry_condition(iv_rank_above(70))
                .structure(short_straddle())
                .build()
            )

    def test_builder_missing_structure(self):
        with pytest.raises(BuilderValidationError, match="structure is required"):
            (
                StrategyBuilder()
                .name("Test")
                .entry_condition(iv_rank_above(70))
                .exit_condition(profit_target(0.50))
                .build()
            )

    def test_builder_invalid_type(self):
        with pytest.raises(BuilderValidationError):
            StrategyBuilder().entry_condition("not a condition")

    def test_builder_validates_before_build(self):
        builder = StrategyBuilder().name("Test")
        errors = builder.validate()
        assert "entry_condition is required" in errors
        assert "exit_condition is required" in errors
        assert "structure is required" in errors


class TestBuiltStrategy:
    @pytest.fixture
    def sample_strategy(self):
        return (
            StrategyBuilder()
            .name("Sample Strategy")
            .underlying("SPY")
            .entry_condition(iv_rank_above(70) & dte_above(25))
            .exit_condition(profit_target(0.50) | stop_loss(2.0))
            .structure(short_straddle(dte=30))
            .position_size(fixed_contracts(5))
            .build()
        )

    def test_should_enter_true(self, sample_strategy):
        market_data = {"underlying": "SPY", "iv_rank": 75, "dte": 30, "spot": 450.0}
        assert sample_strategy.should_enter(market_data) is True

    def test_should_enter_false_low_iv(self, sample_strategy):
        market_data = {"underlying": "SPY", "iv_rank": 60, "dte": 30}
        assert sample_strategy.should_enter(market_data) is False

    def test_should_enter_false_wrong_underlying(self, sample_strategy):
        market_data = {"underlying": "QQQ", "iv_rank": 75, "dte": 30}
        assert sample_strategy.should_enter(market_data) is False

    def test_describe(self, sample_strategy):
        desc = sample_strategy.describe()
        assert "Sample Strategy" in desc
        assert "SPY" in desc
        assert "short straddle" in desc.lower()

    def test_inherits_from_strategy(self, sample_strategy):
        from backtester.strategies.strategy import Strategy

        assert isinstance(sample_strategy, Strategy)


class TestConditionDescriptions:
    def test_iv_rank_above_describe(self):
        assert "IV rank > 70%" in iv_rank_above(70).describe()

    def test_profit_target_describe(self):
        assert "profit >= 50%" in profit_target(0.50).describe()

    def test_stop_loss_describe(self):
        assert "loss >= 200%" in stop_loss(2.0).describe()

    def test_combined_describe(self):
        cond = iv_rank_above(70) & dte_above(30)
        desc = cond.describe()
        assert "AND" in desc
        assert "IV rank" in desc
        assert "DTE" in desc

    def test_or_describe(self):
        cond = profit_target(0.50) | stop_loss(2.0)
        desc = cond.describe()
        assert "OR" in desc


class TestEdgeCases:
    def test_condition_with_missing_data(self):
        cond = iv_rank_above(70)
        assert cond.evaluate({}) is False
        assert cond.evaluate({"other_field": 100}) is False

    def test_zero_dte(self):
        cond = dte_below(1)
        assert cond.evaluate({"dte": 0}) is True

    def test_condition_repr(self):
        cond = iv_rank_above(70)
        repr_str = repr(cond)
        assert "IVRankAbove" in repr_str

    def test_builder_method_chaining(self):
        builder = StrategyBuilder()
        result = builder.name("Test")
        assert result is builder

        result = builder.underlying("SPY")
        assert result is builder


class TestStructureCreation:
    def test_short_straddle_creation(self):
        spec = short_straddle(dte=30, quantity=5, call_price=10.0, put_price=9.5)

        structure = spec.create(
            underlying="SPY",
            spot=450.0,
            option_chain=None,
            entry_date=datetime(2024, 1, 15),
        )

        assert structure is not None
        assert structure.underlying == "SPY"
        assert structure.num_legs == 2

    def test_iron_condor_creation(self):
        spec = iron_condor(
            dte=45,
            width=20,
            wing_width=10,
            put_buy_price=1.0,
            put_sell_price=2.0,
            call_sell_price=2.0,
            call_buy_price=1.0,
        )

        structure = spec.create(
            underlying="SPY",
            spot=450.0,
            option_chain=None,
            entry_date=datetime(2024, 1, 15),
        )

        assert structure is not None
        assert structure.underlying == "SPY"
        assert structure.num_legs == 4
