"""
Tests for Advanced Execution Model.

Tests the volume-aware execution simulation including:
- Kyle's lambda market impact model
- Partial fills based on available volume
- Limit and stop order support
- Execution quality metrics
"""

import pytest
import numpy as np
from datetime import datetime
from backtester.engine.execution import (
    VolumeImpactModel,
    AdvancedExecutionResult,
    AdvancedExecutionModel,
    ExecutionConfigError,
    PriceNotAvailableError,
    EmptyStructureError,
    KYLE_LAMBDA_COEFFICIENT,
    MAX_VOLUME_IMPACT,
    DEFAULT_DAILY_VOLUME,
    MAX_FILL_RATIO,
    DEFAULT_COMMISSION_PER_CONTRACT,
)
from backtester.core.option import Option, CONTRACT_MULTIPLIER
from backtester.structures import ShortStraddle


# =============================================================================
# Test VolumeImpactModel
# =============================================================================


class TestVolumeImpactModelConstruction:
    def test_default_construction(self):
        model = VolumeImpactModel()
        assert model.lambda_coefficient == KYLE_LAMBDA_COEFFICIENT
        assert model.max_impact == MAX_VOLUME_IMPACT

    def test_custom_lambda(self):
        model = VolumeImpactModel(lambda_coefficient=0.8)
        assert model.lambda_coefficient == 0.8

    def test_custom_max_impact(self):
        model = VolumeImpactModel(max_impact=0.05)
        assert model.max_impact == 0.05


class TestVolumeImpactCalculation:
    @pytest.fixture
    def model(self):
        return VolumeImpactModel(lambda_coefficient=0.5, max_impact=0.10)

    def test_zero_volume_returns_max_impact(self, model):
        impact = model.calculate_impact(order_size=10, daily_volume=0)
        assert impact == model.max_impact

    def test_negative_volume_returns_max_impact(self, model):
        impact = model.calculate_impact(order_size=10, daily_volume=-100)
        assert impact == model.max_impact

    def test_small_order_small_impact(self, model):
        # 10 contracts out of 10000 volume = 0.1% of volume
        impact = model.calculate_impact(order_size=10, daily_volume=10000)
        # sqrt(0.001) * 0.5 = 0.0158
        expected = 0.5 * np.sqrt(10 / 10000)
        assert impact == pytest.approx(expected, rel=1e-6)

    def test_large_order_larger_impact(self, model):
        # 1000 contracts out of 10000 volume = 10% of volume
        # sqrt(0.10) * 0.5 = 0.158 which exceeds max_impact of 0.10
        impact = model.calculate_impact(order_size=1000, daily_volume=10000)
        # Result is capped at max_impact
        assert impact == model.max_impact

    def test_impact_capped_at_max(self, model):
        # Very large order should hit cap
        impact = model.calculate_impact(order_size=10000, daily_volume=100)
        assert impact == model.max_impact

    def test_kyles_lambda_formula(self, model):
        """Verify Kyle's lambda formula: impact = lambda * sqrt(order/volume)"""
        order_size = 10
        daily_volume = 10000
        expected = 0.5 * np.sqrt(10 / 10000)
        actual = model.calculate_impact(order_size, daily_volume)
        assert actual == pytest.approx(expected, rel=1e-6)

    def test_impact_increases_with_order_size(self, model):
        impact_small = model.calculate_impact(order_size=10, daily_volume=1000)
        impact_large = model.calculate_impact(order_size=100, daily_volume=1000)
        assert impact_large > impact_small

    def test_impact_decreases_with_volume(self, model):
        impact_low_vol = model.calculate_impact(order_size=100, daily_volume=500)
        impact_high_vol = model.calculate_impact(order_size=100, daily_volume=5000)
        assert impact_low_vol > impact_high_vol


class TestVolumeImpactCost:
    @pytest.fixture
    def model(self):
        return VolumeImpactModel(lambda_coefficient=0.5, max_impact=0.10)

    def test_impact_cost_calculation(self, model):
        order_size = 100
        daily_volume = 1000
        mid_price = 5.00

        impact_pct = model.calculate_impact(order_size, daily_volume)
        expected_cost = mid_price * impact_pct
        actual_cost = model.calculate_impact_cost(order_size, daily_volume, mid_price)

        assert actual_cost == pytest.approx(expected_cost, rel=1e-6)

    def test_impact_cost_scales_with_price(self, model):
        cost_low = model.calculate_impact_cost(100, 1000, 1.00)
        cost_high = model.calculate_impact_cost(100, 1000, 10.00)
        assert cost_high == pytest.approx(cost_low * 10, rel=1e-6)


# =============================================================================
# Test AdvancedExecutionResult
# =============================================================================


class TestAdvancedExecutionResult:
    def test_basic_result(self):
        result = AdvancedExecutionResult(
            fill_price=5.00,
            total_cost=5200.00,
            total_proceeds=0.0,
            commission=6.50,
            slippage=50.00,
            volume_impact=0.02,
            volume_impact_cost=100.00,
            timestamp=datetime(2024, 1, 15),
            filled_quantity=10,
            unfilled_quantity=0,
            order_type="buy",
            order_style="market",
            is_partial_fill=False,
        )

        assert result.fill_price == 5.00
        assert result.filled_quantity == 10
        assert result.unfilled_quantity == 0
        assert result.fill_rate == 1.0
        assert not result.is_partial_fill

    def test_partial_fill_result(self):
        result = AdvancedExecutionResult(
            fill_price=5.00,
            total_cost=2600.00,
            total_proceeds=0.0,
            commission=3.25,
            slippage=25.00,
            volume_impact=0.05,
            volume_impact_cost=125.00,
            timestamp=datetime(2024, 1, 15),
            filled_quantity=5,
            unfilled_quantity=5,
            order_type="buy",
            order_style="market",
            is_partial_fill=True,
        )

        assert result.filled_quantity == 5
        assert result.unfilled_quantity == 5
        assert result.fill_rate == 0.5
        assert result.is_partial_fill

    def test_fill_rate_no_quantity(self):
        result = AdvancedExecutionResult(
            fill_price=0.0,
            total_cost=0.0,
            total_proceeds=0.0,
            commission=0.0,
            slippage=0.0,
            volume_impact=0.0,
            volume_impact_cost=0.0,
            timestamp=datetime(2024, 1, 15),
            filled_quantity=0,
            unfilled_quantity=0,
            order_type="buy",
            order_style="limit",
            is_partial_fill=True,
        )
        assert result.fill_rate == 0.0

    def test_to_dict(self):
        result = AdvancedExecutionResult(
            fill_price=5.00,
            total_cost=5200.00,
            total_proceeds=0.0,
            commission=6.50,
            slippage=50.00,
            volume_impact=0.02,
            volume_impact_cost=100.00,
            timestamp=datetime(2024, 1, 15),
            filled_quantity=10,
            unfilled_quantity=0,
            order_type="buy",
            order_style="market",
            is_partial_fill=False,
        )

        d = result.to_dict()
        assert d["fill_price"] == 5.00
        assert d["total_cost"] == 5200.00
        assert d["commission"] == 6.50
        assert d["order_type"] == "buy"

    def test_execution_quality(self):
        result = AdvancedExecutionResult(
            fill_price=5.00,
            total_cost=5200.00,
            total_proceeds=0.0,
            commission=6.50,
            slippage=50.00,
            volume_impact=0.02,
            volume_impact_cost=100.00,
            timestamp=datetime(2024, 1, 15),
            filled_quantity=10,
            unfilled_quantity=0,
            order_type="buy",
            order_style="market",
            is_partial_fill=False,
        )
        # Quality should be between 0 and 1
        assert 0 <= result.execution_quality <= 1


# =============================================================================
# Test AdvancedExecutionModel Construction
# =============================================================================


class TestAdvancedExecutionModelConstruction:
    def test_default_construction(self):
        model = AdvancedExecutionModel()
        assert model.commission_per_contract == DEFAULT_COMMISSION_PER_CONTRACT
        assert model.base_slippage_pct == 0.01
        assert model.use_volume_impact is True
        assert model.allow_partial_fills is False

    def test_custom_construction(self):
        model = AdvancedExecutionModel(
            commission_per_contract=1.00,
            base_slippage_pct=0.02,
            use_volume_impact=False,
            allow_partial_fills=True,
            max_fill_ratio=0.20,
            kyle_lambda=0.8,
        )
        assert model.commission_per_contract == 1.00
        assert model.base_slippage_pct == 0.02
        assert model.use_volume_impact is False
        assert model.allow_partial_fills is True

    def test_invalid_negative_commission(self):
        with pytest.raises(ExecutionConfigError, match="commission"):
            AdvancedExecutionModel(commission_per_contract=-1.0)

    def test_invalid_slippage_negative(self):
        with pytest.raises(ExecutionConfigError, match="slippage"):
            AdvancedExecutionModel(base_slippage_pct=-0.01)

    def test_invalid_slippage_over_one(self):
        with pytest.raises(ExecutionConfigError, match="slippage"):
            AdvancedExecutionModel(base_slippage_pct=1.5)

    def test_invalid_fill_ratio_zero(self):
        with pytest.raises(ExecutionConfigError, match="fill_ratio"):
            AdvancedExecutionModel(max_fill_ratio=0)

    def test_invalid_fill_ratio_over_one(self):
        with pytest.raises(ExecutionConfigError, match="fill_ratio"):
            AdvancedExecutionModel(max_fill_ratio=1.5)


# =============================================================================
# Test AdvancedExecutionModel.execute_order - Market Orders
# =============================================================================


class TestExecuteMarketOrders:
    @pytest.fixture
    def model(self):
        return AdvancedExecutionModel(
            commission_per_contract=0.65,
            base_slippage_pct=0.01,
            use_volume_impact=True,
            allow_partial_fills=False,
        )

    @pytest.fixture
    def market_data(self):
        return {
            "bid": 5.00,
            "ask": 5.10,
            "mid": 5.05,
            "volume": 1000,
            "timestamp": datetime(2024, 1, 15, 10, 0, 0),
        }

    def test_buy_market_order(self, model, market_data):
        result = model.execute_order(
            order_type="buy",
            quantity=10,
            market_data=market_data,
            order_style="market",
        )

        assert result.filled_quantity == 10
        assert result.unfilled_quantity == 0
        assert result.order_type == "buy"
        assert result.order_style == "market"
        assert result.fill_price > market_data["ask"]  # Due to slippage/impact
        assert result.total_cost > 0
        assert result.total_proceeds == 0
        assert result.commission == 0.65 * 10

    def test_sell_market_order(self, model, market_data):
        result = model.execute_order(
            order_type="sell",
            quantity=10,
            market_data=market_data,
            order_style="market",
        )

        assert result.filled_quantity == 10
        assert result.order_type == "sell"
        assert result.fill_price < market_data["bid"]  # Due to slippage/impact
        assert result.total_proceeds > 0
        assert result.total_cost == 0

    def test_invalid_order_type(self, model, market_data):
        with pytest.raises(ValueError, match="order_type"):
            model.execute_order(
                order_type="invalid",
                quantity=10,
                market_data=market_data,
            )

    def test_invalid_order_style(self, model, market_data):
        with pytest.raises(ValueError, match="order_style"):
            model.execute_order(
                order_type="buy",
                quantity=10,
                market_data=market_data,
                order_style="invalid",
            )

    def test_invalid_quantity(self, model, market_data):
        with pytest.raises(ValueError, match="quantity"):
            model.execute_order(
                order_type="buy",
                quantity=0,
                market_data=market_data,
            )

    def test_invalid_bid_ask(self, model):
        bad_data = {"bid": 0, "ask": 0, "volume": 1000}
        with pytest.raises(PriceNotAvailableError):
            model.execute_order(
                order_type="buy",
                quantity=10,
                market_data=bad_data,
            )


# =============================================================================
# Test AdvancedExecutionModel.execute_order - Limit Orders
# =============================================================================


class TestExecuteLimitOrders:
    @pytest.fixture
    def model(self):
        return AdvancedExecutionModel(
            commission_per_contract=0.65,
            base_slippage_pct=0.01,
            use_volume_impact=True,
        )

    @pytest.fixture
    def market_data(self):
        return {
            "bid": 5.00,
            "ask": 5.10,
            "mid": 5.05,
            "volume": 1000,
            "timestamp": datetime(2024, 1, 15, 10, 0, 0),
        }

    def test_buy_limit_order_fills_when_ask_at_limit(self, model, market_data):
        # Limit at 5.15, ask is 5.10 - should fill
        result = model.execute_order(
            order_type="buy",
            quantity=10,
            market_data=market_data,
            order_style="limit",
            limit_price=5.15,
        )
        assert result.filled_quantity == 10

    def test_buy_limit_order_no_fill_when_ask_above_limit(self, model, market_data):
        # Limit at 5.05, ask is 5.10 - should NOT fill
        result = model.execute_order(
            order_type="buy",
            quantity=10,
            market_data=market_data,
            order_style="limit",
            limit_price=5.05,
        )
        assert result.filled_quantity == 0
        assert result.unfilled_quantity == 10

    def test_sell_limit_order_fills_when_bid_at_limit(self, model, market_data):
        # Limit at 4.90, bid is 5.00 - should fill
        result = model.execute_order(
            order_type="sell",
            quantity=10,
            market_data=market_data,
            order_style="limit",
            limit_price=4.90,
        )
        assert result.filled_quantity == 10

    def test_sell_limit_order_no_fill_when_bid_below_limit(self, model, market_data):
        # Limit at 5.05, bid is 5.00 - should NOT fill
        result = model.execute_order(
            order_type="sell",
            quantity=10,
            market_data=market_data,
            order_style="limit",
            limit_price=5.05,
        )
        assert result.filled_quantity == 0

    def test_limit_order_requires_limit_price(self, model, market_data):
        with pytest.raises(ValueError, match="limit_price"):
            model.execute_order(
                order_type="buy",
                quantity=10,
                market_data=market_data,
                order_style="limit",
            )

    def test_buy_limit_uses_limit_as_cap(self, model, market_data):
        # When limit is below ask, but order fills, use ask
        result = model.execute_order(
            order_type="buy",
            quantity=10,
            market_data=market_data,
            order_style="limit",
            limit_price=5.50,  # Above ask
        )
        # Fill price should be based on ask, not limit
        base_expected = market_data["ask"]
        assert result.filled_quantity == 10


# =============================================================================
# Test AdvancedExecutionModel.execute_order - Stop Orders
# =============================================================================


class TestExecuteStopOrders:
    @pytest.fixture
    def model(self):
        return AdvancedExecutionModel(
            commission_per_contract=0.65,
            base_slippage_pct=0.01,
            use_volume_impact=True,
        )

    @pytest.fixture
    def market_data(self):
        return {
            "bid": 5.00,
            "ask": 5.10,
            "mid": 5.05,
            "volume": 1000,
            "timestamp": datetime(2024, 1, 15, 10, 0, 0),
        }

    def test_buy_stop_triggers_when_price_above_stop(self, model, market_data):
        # Stop at 5.00, mid is 5.05 - should trigger
        result = model.execute_order(
            order_type="buy",
            quantity=10,
            market_data=market_data,
            order_style="stop",
            stop_price=5.00,
        )
        assert result.filled_quantity == 10

    def test_buy_stop_no_trigger_when_price_below_stop(self, model, market_data):
        # Stop at 5.10, mid is 5.05 - should NOT trigger
        result = model.execute_order(
            order_type="buy",
            quantity=10,
            market_data=market_data,
            order_style="stop",
            stop_price=5.10,
        )
        assert result.filled_quantity == 0

    def test_sell_stop_triggers_when_price_below_stop(self, model, market_data):
        # Stop at 5.10, mid is 5.05 - should trigger (sell stop)
        result = model.execute_order(
            order_type="sell",
            quantity=10,
            market_data=market_data,
            order_style="stop",
            stop_price=5.10,
        )
        assert result.filled_quantity == 10

    def test_sell_stop_no_trigger_when_price_above_stop(self, model, market_data):
        # Stop at 5.00, mid is 5.05 - should NOT trigger
        result = model.execute_order(
            order_type="sell",
            quantity=10,
            market_data=market_data,
            order_style="stop",
            stop_price=5.00,
        )
        assert result.filled_quantity == 0

    def test_stop_order_requires_stop_price(self, model, market_data):
        with pytest.raises(ValueError, match="stop_price"):
            model.execute_order(
                order_type="buy",
                quantity=10,
                market_data=market_data,
                order_style="stop",
            )


# =============================================================================
# Test AdvancedExecutionModel - Volume Impact
# =============================================================================


class TestVolumeImpactExecution:
    @pytest.fixture
    def model_with_impact(self):
        return AdvancedExecutionModel(
            commission_per_contract=0.65,
            base_slippage_pct=0.01,
            use_volume_impact=True,
            kyle_lambda=0.5,
        )

    @pytest.fixture
    def model_no_impact(self):
        return AdvancedExecutionModel(
            commission_per_contract=0.65,
            base_slippage_pct=0.01,
            use_volume_impact=False,
        )

    @pytest.fixture
    def market_data(self):
        return {
            "bid": 5.00,
            "ask": 5.10,
            "mid": 5.05,
            "volume": 1000,
            "timestamp": datetime(2024, 1, 15, 10, 0, 0),
        }

    def test_volume_impact_increases_buy_cost(
        self, model_with_impact, model_no_impact, market_data
    ):
        result_with = model_with_impact.execute_order(
            order_type="buy",
            quantity=100,  # Larger order for noticeable impact
            market_data=market_data,
        )
        result_without = model_no_impact.execute_order(
            order_type="buy",
            quantity=100,
            market_data=market_data,
        )

        assert result_with.fill_price > result_without.fill_price
        assert result_with.volume_impact > 0
        assert result_without.volume_impact == 0

    def test_volume_impact_decreases_sell_proceeds(
        self, model_with_impact, model_no_impact, market_data
    ):
        result_with = model_with_impact.execute_order(
            order_type="sell",
            quantity=100,
            market_data=market_data,
        )
        result_without = model_no_impact.execute_order(
            order_type="sell",
            quantity=100,
            market_data=market_data,
        )

        assert result_with.fill_price < result_without.fill_price
        assert result_with.total_proceeds < result_without.total_proceeds

    def test_large_order_more_impact(self, model_with_impact, market_data):
        result_small = model_with_impact.execute_order(
            order_type="buy",
            quantity=10,
            market_data=market_data,
        )
        result_large = model_with_impact.execute_order(
            order_type="buy",
            quantity=100,
            market_data=market_data,
        )

        assert result_large.volume_impact > result_small.volume_impact

    def test_low_volume_more_impact(self, model_with_impact):
        low_vol_data = {
            "bid": 5.00,
            "ask": 5.10,
            "mid": 5.05,
            "volume": 100,  # Low volume
            "timestamp": datetime(2024, 1, 15),
        }
        high_vol_data = {
            "bid": 5.00,
            "ask": 5.10,
            "mid": 5.05,
            "volume": 10000,  # High volume
            "timestamp": datetime(2024, 1, 15),
        }

        result_low_vol = model_with_impact.execute_order(
            order_type="buy",
            quantity=50,
            market_data=low_vol_data,
        )
        result_high_vol = model_with_impact.execute_order(
            order_type="buy",
            quantity=50,
            market_data=high_vol_data,
        )

        assert result_low_vol.volume_impact > result_high_vol.volume_impact


# =============================================================================
# Test AdvancedExecutionModel - Partial Fills
# =============================================================================


class TestPartialFills:
    @pytest.fixture
    def model_partial(self):
        return AdvancedExecutionModel(
            commission_per_contract=0.65,
            base_slippage_pct=0.01,
            use_volume_impact=True,
            allow_partial_fills=True,
            max_fill_ratio=0.10,  # Can only fill 10% of daily volume
        )

    @pytest.fixture
    def model_no_partial(self):
        return AdvancedExecutionModel(
            commission_per_contract=0.65,
            base_slippage_pct=0.01,
            use_volume_impact=True,
            allow_partial_fills=False,
        )

    @pytest.fixture
    def market_data(self):
        return {
            "bid": 5.00,
            "ask": 5.10,
            "mid": 5.05,
            "volume": 100,  # Low volume to trigger partial fills
            "timestamp": datetime(2024, 1, 15, 10, 0, 0),
        }

    def test_partial_fill_when_order_exceeds_volume(self, model_partial, market_data):
        # Order for 50 contracts, but only 10% of 100 = 10 can fill
        result = model_partial.execute_order(
            order_type="buy",
            quantity=50,
            market_data=market_data,
        )

        assert result.filled_quantity == 10  # 10% of 100
        assert result.unfilled_quantity == 40
        assert result.is_partial_fill is True

    def test_full_fill_when_order_within_volume(self, model_partial, market_data):
        # Order for 5 contracts, 10% of 100 = 10 available
        result = model_partial.execute_order(
            order_type="buy",
            quantity=5,
            market_data=market_data,
        )

        assert result.filled_quantity == 5
        assert result.unfilled_quantity == 0
        assert result.is_partial_fill is False

    def test_no_partial_fills_mode(self, model_no_partial, market_data):
        # Should fill fully even if order is large
        result = model_no_partial.execute_order(
            order_type="buy",
            quantity=50,
            market_data=market_data,
        )

        assert result.filled_quantity == 50
        assert result.unfilled_quantity == 0


# =============================================================================
# Test AdvancedExecutionModel - Structure Execution
# =============================================================================


class TestStructureExecution:
    @pytest.fixture
    def model(self):
        return AdvancedExecutionModel(
            commission_per_contract=0.65,
            base_slippage_pct=0.01,
            use_volume_impact=True,
        )

    @pytest.fixture
    def straddle(self):
        return ShortStraddle.create(
            underlying="SPY",
            strike=450.0,
            expiration=datetime(2024, 3, 15),
            call_price=10.00,
            put_price=9.50,
            quantity=10,
            entry_date=datetime(2024, 1, 15),
            underlying_price=450.0,
        )

    @pytest.fixture
    def market_data(self):
        return {
            "timestamp": datetime(2024, 1, 15),
            "option_chain": None,  # Will use option's entry price
        }

    def test_structure_entry_short_straddle(self, model, straddle, market_data):
        result = model.execute_structure_entry(straddle, market_data)

        assert "entry_results" in result
        assert "total_cost" in result
        assert "total_premium" in result
        assert "total_commission" in result
        assert "total_slippage" in result
        assert "total_volume_impact" in result
        assert "filled_contracts" in result
        assert "execution_quality" in result

        # Short straddle = selling, so should have proceeds
        assert result["filled_contracts"] > 0

    def test_structure_entry_empty_structure_raises(self, model, market_data):
        from backtester.core.option_structure import OptionStructure

        empty = OptionStructure(structure_type="empty")
        with pytest.raises(EmptyStructureError):
            model.execute_structure_entry(empty, market_data)


# =============================================================================
# Test AdvancedExecutionModel - Statistics
# =============================================================================


class TestExecutionStatistics:
    @pytest.fixture
    def model(self):
        return AdvancedExecutionModel(
            commission_per_contract=0.65,
            base_slippage_pct=0.01,
            use_volume_impact=True,
        )

    @pytest.fixture
    def market_data(self):
        return {
            "bid": 5.00,
            "ask": 5.10,
            "mid": 5.05,
            "volume": 1000,
            "timestamp": datetime(2024, 1, 15, 10, 0, 0),
        }

    def test_empty_statistics(self, model):
        stats = model.get_execution_statistics()
        assert stats["num_executions"] == 0
        assert stats["avg_fill_rate"] == 0.0
        assert stats["avg_volume_impact"] == 0.0

    def test_statistics_after_executions(self, model, market_data):
        # Execute some orders
        model.execute_order("buy", 10, market_data)
        model.execute_order("sell", 20, market_data)
        model.execute_order("buy", 15, market_data)

        stats = model.get_execution_statistics()
        assert stats["num_executions"] == 3
        assert stats["avg_fill_rate"] == 1.0  # All fully filled
        assert stats["total_commission"] == 0.65 * (10 + 20 + 15)

    def test_execution_history(self, model, market_data):
        model.execute_order("buy", 10, market_data)
        model.execute_order("sell", 20, market_data)

        history = model.execution_history
        assert len(history) == 2
        assert history[0].order_type == "buy"
        assert history[1].order_type == "sell"

    def test_clear_history(self, model, market_data):
        model.execute_order("buy", 10, market_data)
        model.execute_order("sell", 20, market_data)

        assert len(model.execution_history) == 2

        model.clear_history()

        assert len(model.execution_history) == 0


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    @pytest.fixture
    def model(self):
        return AdvancedExecutionModel()

    def test_minimum_fill_price(self):
        """Ensure fill price never goes negative"""
        model = AdvancedExecutionModel(
            base_slippage_pct=0.50,  # Very high slippage
            use_volume_impact=True,
        )
        market_data = {
            "bid": 0.05,  # Very low price
            "ask": 0.06,
            "mid": 0.055,
            "volume": 100,
        }

        result = model.execute_order("sell", 100, market_data)
        assert result.fill_price >= 0.01  # Minimum floor

    def test_custom_timestamp(self, model):
        market_data = {
            "bid": 5.00,
            "ask": 5.10,
            "volume": 1000,
        }
        custom_ts = datetime(2024, 6, 15, 14, 30, 0)

        result = model.execute_order("buy", 10, market_data, timestamp=custom_ts)

        assert result.timestamp == custom_ts

    def test_repr(self, model):
        repr_str = repr(model)
        assert "AdvancedExecutionModel" in repr_str
        assert "commission" in repr_str


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    def test_full_workflow(self):
        """Test complete buy-sell workflow with statistics"""
        model = AdvancedExecutionModel(
            commission_per_contract=0.65,
            base_slippage_pct=0.01,
            use_volume_impact=True,
            allow_partial_fills=False,
        )

        market_data = {
            "bid": 5.00,
            "ask": 5.10,
            "mid": 5.05,
            "volume": 1000,
            "timestamp": datetime(2024, 1, 15),
        }

        # Buy order
        buy_result = model.execute_order("buy", 10, market_data)
        assert buy_result.filled_quantity == 10
        assert buy_result.total_cost > 0

        # Sell order
        sell_result = model.execute_order("sell", 10, market_data)
        assert sell_result.filled_quantity == 10
        assert sell_result.total_proceeds > 0

        # Check statistics
        stats = model.get_execution_statistics()
        assert stats["num_executions"] == 2
        assert stats["total_commission"] == 0.65 * 20

    def test_volume_impact_vs_base_model(self):
        """Compare advanced model with volume impact to base model"""
        advanced = AdvancedExecutionModel(
            commission_per_contract=0.65,
            base_slippage_pct=0.01,
            use_volume_impact=True,
        )

        market_data = {
            "bid": 5.00,
            "ask": 5.10,
            "mid": 5.05,
            "volume": 500,  # Moderate volume
            "timestamp": datetime(2024, 1, 15),
        }

        # Large order - should show significant impact
        result = advanced.execute_order("buy", 100, market_data)

        # Volume impact should be tracked
        assert result.volume_impact > 0
        assert result.volume_impact_cost > 0
        assert result.execution_quality < 1.0
