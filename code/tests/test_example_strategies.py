"""
Comprehensive tests for example trading strategies.

Tests all example strategies:
    - ShortStraddleHighIVStrategy
    - IronCondorStrategy
    - VolatilityRegimeStrategy

Each strategy is tested for:
    - Initialization and parameter validation
    - Entry logic
    - Exit logic
    - Position creation
    - Edge cases
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock

from backtester.strategies.short_straddle_strategy import ShortStraddleHighIVStrategy
from backtester.strategies.iron_condor_strategy import IronCondorStrategy
from backtester.strategies.volatility_regime_strategy import (
    VolatilityRegimeStrategy,
    VolatilityRegime,
)
from backtester.structures.straddle import ShortStraddle
from backtester.structures.condor import IronCondor


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def base_market_data():
    """Common market data for testing."""
    return {
        'underlying': 'SPY',
        'spot': 450.0,
        'dte': 45,
        'date': datetime(2024, 3, 1),
        'iv_rank': 60,
        'vix': 20.0,
        'atm_strike': 450.0,
        'atm_call_price': 6.50,
        'atm_put_price': 6.25,
    }


# =============================================================================
# ShortStraddleHighIVStrategy Tests
# =============================================================================

class TestShortStraddleHighIVStrategy:
    """Tests for ShortStraddleHighIVStrategy."""

    def test_initialization(self):
        """Test strategy initialization."""
        strategy = ShortStraddleHighIVStrategy(
            name='Test Straddle',
            initial_capital=100000,
            iv_rank_threshold=70,
            profit_target_pct=0.50
        )

        assert strategy.name == 'Test Straddle'
        assert strategy.initial_capital == 100000
        assert strategy.iv_rank_threshold == 70
        assert strategy.profit_target_pct == 0.50

    def test_invalid_iv_rank_threshold(self):
        """Test validation of IV rank threshold."""
        with pytest.raises(ValueError):
            ShortStraddleHighIVStrategy(iv_rank_threshold=150)

        with pytest.raises(ValueError):
            ShortStraddleHighIVStrategy(iv_rank_threshold=-10)

    def test_invalid_profit_target(self):
        """Test validation of profit target."""
        with pytest.raises(ValueError):
            ShortStraddleHighIVStrategy(profit_target_pct=1.5)

        with pytest.raises(ValueError):
            ShortStraddleHighIVStrategy(profit_target_pct=0)

    def test_should_enter_high_iv(self, base_market_data):
        """Test entry when IV rank is high."""
        strategy = ShortStraddleHighIVStrategy(iv_rank_threshold=50)
        base_market_data['iv_rank'] = 75

        assert strategy.should_enter(base_market_data) is True

    def test_should_not_enter_low_iv(self, base_market_data):
        """Test no entry when IV rank is low."""
        strategy = ShortStraddleHighIVStrategy(iv_rank_threshold=70)
        base_market_data['iv_rank'] = 50

        assert strategy.should_enter(base_market_data) is False

    def test_should_not_enter_low_dte(self, base_market_data):
        """Test no entry when DTE too low."""
        strategy = ShortStraddleHighIVStrategy(min_entry_dte=45)
        base_market_data['dte'] = 20

        assert strategy.should_enter(base_market_data) is False

    def test_should_not_enter_missing_data(self):
        """Test no entry with missing required data."""
        strategy = ShortStraddleHighIVStrategy()

        # Missing iv_rank
        assert strategy.should_enter({'dte': 45, 'underlying': 'SPY'}) is False

        # Missing dte
        assert strategy.should_enter({'iv_rank': 75, 'underlying': 'SPY'}) is False

    def test_should_exit_profit_target(self, base_market_data):
        """Test exit on profit target."""
        strategy = ShortStraddleHighIVStrategy(profit_target_pct=0.50)

        # Create mock structure with profit
        structure = Mock()
        structure.structure_id = 'test_id'
        structure.calculate_pnl.return_value = 5000  # $5,000 profit
        structure.max_profit = 10000  # $10,000 max profit
        structure.net_premium = 10000

        assert strategy.should_exit(structure, base_market_data) is True

    def test_should_exit_loss_limit(self, base_market_data):
        """Test exit on loss limit."""
        strategy = ShortStraddleHighIVStrategy(loss_limit_pct=2.0)

        # Create mock structure with loss
        structure = Mock()
        structure.structure_id = 'test_id'
        structure.calculate_pnl.return_value = -20000  # $20,000 loss
        structure.max_profit = 10000  # $10,000 max profit
        structure.net_premium = 10000

        assert strategy.should_exit(structure, base_market_data) is True

    def test_should_exit_low_dte(self, base_market_data):
        """Test exit on low DTE."""
        strategy = ShortStraddleHighIVStrategy(exit_dte=7)
        base_market_data['dte'] = 5

        structure = Mock()
        structure.structure_id = 'test_id'
        structure.calculate_pnl.return_value = 0
        structure.max_profit = 10000
        structure.net_premium = 10000

        assert strategy.should_exit(structure, base_market_data) is True

    def test_create_position(self):
        """Test position creation."""
        strategy = ShortStraddleHighIVStrategy()

        position = strategy.create_position(
            underlying='SPY',
            strike=450.0,
            expiration=datetime(2024, 3, 15),
            call_price=6.50,
            put_price=6.25,
            quantity=10,
            entry_date=datetime(2024, 3, 1),
            spot_price=450.0
        )

        assert isinstance(position, ShortStraddle)
        assert position.strike == 450.0
        assert position.num_legs == 2

    def test_max_positions_per_underlying(self, base_market_data):
        """Test max positions limit per underlying."""
        strategy = ShortStraddleHighIVStrategy(max_positions_per_underlying=1)

        # Add a mock position for SPY
        mock_structure = Mock()
        mock_structure.underlying = 'SPY'
        strategy._structures.append(mock_structure)

        # Should not enter with existing position
        base_market_data['iv_rank'] = 80
        assert strategy.should_enter(base_market_data) is False


# =============================================================================
# IronCondorStrategy Tests
# =============================================================================

class TestIronCondorStrategy:
    """Tests for IronCondorStrategy."""

    def test_initialization(self):
        """Test strategy initialization."""
        strategy = IronCondorStrategy(
            name='Test IC',
            initial_capital=100000,
            iv_rank_threshold=50
        )

        assert strategy.name == 'Test IC'
        assert strategy.initial_capital == 100000
        assert strategy._iv_rank_threshold == 50

    def test_should_enter_moderate_iv(self, base_market_data):
        """Test entry when IV rank is moderate."""
        strategy = IronCondorStrategy(iv_rank_threshold=50)
        base_market_data['iv_rank'] = 60

        assert strategy.should_enter(base_market_data) is True

    def test_should_not_enter_low_iv(self, base_market_data):
        """Test no entry when IV rank is low."""
        strategy = IronCondorStrategy(iv_rank_threshold=60)
        base_market_data['iv_rank'] = 40

        assert strategy.should_enter(base_market_data) is False

    def test_calculate_strikes(self):
        """Test strike calculation based on standard deviation."""
        strategy = IronCondorStrategy(std_dev_width=1.0, wing_width_pct=0.02)

        strikes = strategy.calculate_strikes(
            spot=450.0,
            vol=0.20,
            dte=45
        )

        # Verify all strikes are present
        assert 'put_buy' in strikes
        assert 'put_sell' in strikes
        assert 'call_sell' in strikes
        assert 'call_buy' in strikes

        # Verify ordering
        assert strikes['put_buy'] < strikes['put_sell']
        assert strikes['put_sell'] < strikes['call_sell']
        assert strikes['call_sell'] < strikes['call_buy']

    def test_strike_rounding(self):
        """Test strike rounding to nearest $5."""
        strategy = IronCondorStrategy()

        assert strategy._round_strike(452.3) == 450.0
        assert strategy._round_strike(453.7) == 455.0
        assert strategy._round_strike(450.0) == 450.0

    def test_create_position(self):
        """Test position creation."""
        strategy = IronCondorStrategy()

        strikes = {
            'put_buy': 430.0,
            'put_sell': 440.0,
            'call_sell': 460.0,
            'call_buy': 470.0
        }

        prices = {
            'put_buy': 1.50,
            'put_sell': 3.00,
            'call_sell': 3.25,
            'call_buy': 1.75
        }

        position = strategy.create_position(
            underlying='SPY',
            strikes=strikes,
            expiration=datetime(2024, 3, 15),
            prices=prices,
            quantity=10,
            entry_date=datetime(2024, 3, 1),
            spot_price=450.0
        )

        assert isinstance(position, IronCondor)
        assert position.num_legs == 4

    def test_should_exit_profit_target(self, base_market_data):
        """Test exit on profit target."""
        strategy = IronCondorStrategy(profit_target_pct=0.50)

        structure = Mock()
        structure.structure_id = 'test_id'
        structure.calculate_pnl.return_value = 2500
        structure.max_profit = 5000
        structure.net_premium = 5000

        assert strategy.should_exit(structure, base_market_data) is True


# =============================================================================
# VolatilityRegimeStrategy Tests
# =============================================================================

class TestVolatilityRegimeStrategy:
    """Tests for VolatilityRegimeStrategy."""

    def test_initialization(self):
        """Test strategy initialization."""
        strategy = VolatilityRegimeStrategy(
            name='Test Regime',
            high_vix_threshold=25.0,
            low_vix_threshold=15.0
        )

        assert strategy.name == 'Test Regime'
        assert strategy.high_vix_threshold == 25.0
        assert strategy.low_vix_threshold == 15.0

    def test_invalid_thresholds(self):
        """Test validation of VIX thresholds."""
        with pytest.raises(ValueError):
            VolatilityRegimeStrategy(
                high_vix_threshold=15.0,
                low_vix_threshold=25.0  # Should be lower
            )

        with pytest.raises(ValueError):
            VolatilityRegimeStrategy(
                high_vix_threshold=-5.0,
                low_vix_threshold=10.0
            )

    def test_get_current_regime_high(self, base_market_data):
        """Test regime classification for high VIX."""
        strategy = VolatilityRegimeStrategy(
            high_vix_threshold=25.0,
            low_vix_threshold=15.0
        )

        base_market_data['vix'] = 30.0
        regime = strategy.get_current_regime(base_market_data)

        assert regime == VolatilityRegime.HIGH

    def test_get_current_regime_medium(self, base_market_data):
        """Test regime classification for medium VIX."""
        strategy = VolatilityRegimeStrategy(
            high_vix_threshold=25.0,
            low_vix_threshold=15.0
        )

        base_market_data['vix'] = 20.0
        regime = strategy.get_current_regime(base_market_data)

        assert regime == VolatilityRegime.MEDIUM

    def test_get_current_regime_low(self, base_market_data):
        """Test regime classification for low VIX."""
        strategy = VolatilityRegimeStrategy(
            high_vix_threshold=25.0,
            low_vix_threshold=15.0
        )

        base_market_data['vix'] = 12.0
        regime = strategy.get_current_regime(base_market_data)

        assert regime == VolatilityRegime.LOW

    def test_get_current_regime_no_vix(self, base_market_data):
        """Test regime classification without VIX data."""
        strategy = VolatilityRegimeStrategy()
        del base_market_data['vix']

        regime = strategy.get_current_regime(base_market_data)
        assert regime is None

    def test_update_regime_history(self):
        """Test regime history tracking."""
        strategy = VolatilityRegimeStrategy()

        date1 = datetime(2024, 3, 1)
        date2 = datetime(2024, 3, 2)

        strategy.update_regime_history(date1, VolatilityRegime.HIGH)
        strategy.update_regime_history(date2, VolatilityRegime.MEDIUM)

        assert strategy._regime_history[date1] == VolatilityRegime.HIGH
        assert strategy._regime_history[date2] == VolatilityRegime.MEDIUM

    def test_get_regime_duration(self):
        """Test regime duration calculation."""
        strategy = VolatilityRegimeStrategy()

        # Add 5 consecutive days of HIGH regime
        for i in range(5):
            date = datetime(2024, 3, 1) + timedelta(days=i)
            strategy.update_regime_history(date, VolatilityRegime.HIGH)

        current_date = datetime(2024, 3, 5)
        duration = strategy.get_regime_duration(current_date, VolatilityRegime.HIGH)

        assert duration == 5

    def test_get_regime_duration_broken(self):
        """Test regime duration with broken sequence."""
        strategy = VolatilityRegimeStrategy()

        # Add HIGH, HIGH, MEDIUM, HIGH sequence
        strategy.update_regime_history(datetime(2024, 3, 1), VolatilityRegime.HIGH)
        strategy.update_regime_history(datetime(2024, 3, 2), VolatilityRegime.HIGH)
        strategy.update_regime_history(datetime(2024, 3, 3), VolatilityRegime.MEDIUM)
        strategy.update_regime_history(datetime(2024, 3, 4), VolatilityRegime.HIGH)

        # Duration from latest date should be 1 (only the last HIGH)
        duration = strategy.get_regime_duration(datetime(2024, 3, 4), VolatilityRegime.HIGH)
        assert duration == 1

    def test_should_enter_sufficient_duration(self, base_market_data):
        """Test entry requires minimum regime duration."""
        strategy = VolatilityRegimeStrategy(
            min_regime_duration=3,
            high_vix_threshold=25.0,
            low_vix_threshold=15.0
        )

        # Add 3 days of HIGH regime
        for i in range(3):
            date = datetime(2024, 3, 1) + timedelta(days=i)
            strategy.update_regime_history(date, VolatilityRegime.HIGH)

        base_market_data['vix'] = 30.0
        base_market_data['date'] = datetime(2024, 3, 3)

        assert strategy.should_enter(base_market_data) is True

    def test_should_not_enter_insufficient_duration(self, base_market_data):
        """Test no entry with insufficient regime duration."""
        strategy = VolatilityRegimeStrategy(
            min_regime_duration=5,
            high_vix_threshold=25.0,
            low_vix_threshold=15.0
        )

        # Only 2 days of HIGH regime
        for i in range(2):
            date = datetime(2024, 3, 1) + timedelta(days=i)
            strategy.update_regime_history(date, VolatilityRegime.HIGH)

        base_market_data['vix'] = 30.0
        base_market_data['date'] = datetime(2024, 3, 2)

        assert strategy.should_enter(base_market_data) is False

    def test_get_recommended_structure_high_vix(self, base_market_data):
        """Test structure recommendation for high VIX."""
        strategy = VolatilityRegimeStrategy(
            high_vix_threshold=25.0,
            low_vix_threshold=15.0
        )

        base_market_data['vix'] = 30.0
        recommended = strategy.get_recommended_structure(base_market_data)

        assert recommended == 'short_straddle'

    def test_get_recommended_structure_medium_vix(self, base_market_data):
        """Test structure recommendation for medium VIX."""
        strategy = VolatilityRegimeStrategy(
            high_vix_threshold=25.0,
            low_vix_threshold=15.0
        )

        base_market_data['vix'] = 20.0
        recommended = strategy.get_recommended_structure(base_market_data)

        assert recommended == 'iron_condor'

    def test_get_recommended_structure_low_vix(self, base_market_data):
        """Test structure recommendation for low VIX."""
        strategy = VolatilityRegimeStrategy(
            high_vix_threshold=25.0,
            low_vix_threshold=15.0
        )

        base_market_data['vix'] = 12.0
        recommended = strategy.get_recommended_structure(base_market_data)

        assert recommended == 'long_straddle'

    def test_should_exit_regime_change_long_position(self, base_market_data):
        """Test exit when regime changes for long position."""
        strategy = VolatilityRegimeStrategy(
            high_vix_threshold=25.0,
            low_vix_threshold=15.0
        )

        # Create mock long structure
        structure = Mock()
        structure.structure_id = 'test_id'
        structure.structure_type = 'long_straddle'
        structure.calculate_pnl.return_value = 0
        structure.max_profit = float('inf')
        structure.net_premium = -10000

        # VIX spikes to high regime
        base_market_data['vix'] = 35.0

        assert strategy.should_exit(structure, base_market_data) is True

    def test_should_exit_regime_change_short_position(self, base_market_data):
        """Test exit when regime changes for short position."""
        strategy = VolatilityRegimeStrategy(
            high_vix_threshold=25.0,
            low_vix_threshold=15.0
        )

        # Create mock short structure
        structure = Mock()
        structure.structure_id = 'test_id'
        structure.structure_type = 'short_straddle'
        structure.calculate_pnl.return_value = 0
        structure.max_profit = 10000
        structure.net_premium = 10000

        # VIX drops to low regime
        base_market_data['vix'] = 12.0

        assert strategy.should_exit(structure, base_market_data) is True

    def test_get_strategy_stats(self):
        """Test strategy statistics retrieval."""
        strategy = VolatilityRegimeStrategy()

        # Add some regime history
        strategy.update_regime_history(datetime(2024, 3, 1), VolatilityRegime.HIGH)
        strategy.update_regime_history(datetime(2024, 3, 2), VolatilityRegime.MEDIUM)

        stats = strategy.get_strategy_stats()

        assert 'strategy_type' in stats
        assert stats['strategy_type'] == 'volatility_regime'
        assert 'regime_distribution' in stats
        assert stats['high_vix_threshold'] == strategy.high_vix_threshold


# =============================================================================
# Integration Tests
# =============================================================================

class TestStrategyIntegration:
    """Integration tests for strategies."""

    def test_straddle_strategy_full_lifecycle(self, base_market_data):
        """Test full lifecycle of straddle strategy."""
        strategy = ShortStraddleHighIVStrategy(
            initial_capital=250000,  # Increase capital for test
            iv_rank_threshold=60,
            position_limits={
                'max_single_position_size': 200000,  # Increase for this test
                'max_capital_utilization': 2.0  # Allow high utilization for test
            }
        )

        # Check entry
        base_market_data['iv_rank'] = 75
        assert strategy.should_enter(base_market_data)

        # Create position (smaller size)
        position = strategy.create_position(
            underlying='SPY',
            strike=450.0,
            expiration=datetime(2024, 3, 15),
            call_price=6.50,
            put_price=6.25,
            quantity=5,  # Reduced from 10 to 5
            entry_date=datetime(2024, 3, 1),
            spot_price=450.0
        )

        # Open position
        strategy.open_position(position)
        assert strategy.num_open_positions == 1

        # Check exit (simulate profit - 50% of max profit should trigger exit)
        position_mock = Mock()
        position_mock.structure_id = position.structure_id
        # Max profit for 5 contracts = (6.50 + 6.25) * 5 * 100 = 6375
        position_mock.max_profit = 6375
        position_mock.net_premium = 6375
        # Set profit to 51% of max to trigger exit
        position_mock.calculate_pnl.return_value = 3300

        assert strategy.should_exit(position_mock, {'dte': 20})


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
