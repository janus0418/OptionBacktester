"""
Comprehensive Tests for Backtesting Engine Components

This module contains extensive tests for:
    - DataStream: Market data iteration and caching
    - PositionManager: Position tracking and portfolio metrics
    - ExecutionModel: Order execution and fill prices
    - BacktestEngine: Event loop and integration

Test Coverage:
    - Unit tests for each component
    - Integration tests combining components
    - Edge case handling
    - Error condition testing
    - Financial correctness validation

Requirements:
    - pytest
    - numpy
    - pandas

Run Tests:
    pytest tests/test_engine.py -v --tb=short

Author: OptionsBacktester2 Team
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date
from typing import Any, Dict, List, Optional
from unittest.mock import Mock, MagicMock, patch

# Import engine components
from backtester.engine.data_stream import (
    DataStream,
    TradingCalendar,
    DataStreamError,
    DataStreamConfigError,
    DataNotAvailableError,
)
from backtester.engine.position_manager import (
    PositionManager,
    PositionRecord,
    PositionManagerError,
    PositionNotFoundError,
    DuplicatePositionError,
)
from backtester.engine.execution import (
    ExecutionModel,
    ExecutionResult,
    ExecutionError,
    ExecutionConfigError,
    PriceNotAvailableError,
)
from backtester.engine.backtest_engine import (
    BacktestEngine,
    BacktestState,
    TradeRecord,
    BacktestError,
    BacktestConfigError,
    BacktestExecutionError,
)

# Import supporting classes
from backtester.core.option import Option
from backtester.core.option_structure import OptionStructure
from backtester.strategies.strategy import Strategy


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_option():
    """Create a sample option for testing."""
    return Option(
        option_type='call',
        position_type='long',
        underlying='SPY',
        strike=450.0,
        expiration=datetime(2024, 3, 15),
        quantity=10,
        entry_price=5.50,
        entry_date=datetime(2024, 1, 15),
        underlying_price_at_entry=448.0,
        implied_vol_at_entry=0.20
    )


@pytest.fixture
def sample_short_option():
    """Create a sample short option for testing."""
    return Option(
        option_type='put',
        position_type='short',
        underlying='SPY',
        strike=440.0,
        expiration=datetime(2024, 3, 15),
        quantity=10,
        entry_price=4.50,
        entry_date=datetime(2024, 1, 15),
        underlying_price_at_entry=448.0,
        implied_vol_at_entry=0.22
    )


@pytest.fixture
def sample_structure(sample_option, sample_short_option):
    """Create a sample structure for testing."""
    structure = OptionStructure(
        structure_type='vertical_spread',
        underlying='SPY',
        entry_date=datetime(2024, 1, 15)
    )
    structure.add_option(sample_option)
    structure.add_option(sample_short_option)
    return structure


@pytest.fixture
def small_structure():
    """Create a small structure (1 contract) for testing with position limits."""
    call = Option(
        option_type='call',
        position_type='long',
        underlying='SPY',
        strike=450.0,
        expiration=datetime(2024, 3, 15),
        quantity=1,  # Small quantity to pass position limits
        entry_price=5.50,
        entry_date=datetime(2024, 1, 15),
        underlying_price_at_entry=448.0,
        implied_vol_at_entry=0.20
    )
    put = Option(
        option_type='put',
        position_type='short',
        underlying='SPY',
        strike=440.0,
        expiration=datetime(2024, 3, 15),
        quantity=1,  # Small quantity to pass position limits
        entry_price=4.50,
        entry_date=datetime(2024, 1, 15),
        underlying_price_at_entry=448.0,
        implied_vol_at_entry=0.22
    )
    structure = OptionStructure(
        structure_type='vertical_spread',
        underlying='SPY',
        entry_date=datetime(2024, 1, 15)
    )
    structure.add_option(call)
    structure.add_option(put)
    return structure


@pytest.fixture
def sample_option_chain():
    """Create a sample option chain DataFrame."""
    return pd.DataFrame({
        'date': pd.to_datetime(['2024-01-15'] * 10),
        'act_symbol': ['SPY'] * 10,
        'strike': [440.0, 445.0, 450.0, 455.0, 460.0] * 2,
        'call_put': ['Call', 'Call', 'Call', 'Call', 'Call',
                     'Put', 'Put', 'Put', 'Put', 'Put'],
        'expiration': pd.to_datetime(['2024-03-15'] * 10),
        'bid': [12.5, 9.2, 5.4, 3.1, 1.5, 1.2, 2.3, 4.4, 7.2, 10.8],
        'ask': [12.8, 9.5, 5.6, 3.3, 1.7, 1.4, 2.5, 4.6, 7.4, 11.0],
        'vol': [0.18, 0.19, 0.20, 0.21, 0.22] * 2,
        'delta': [0.75, 0.60, 0.50, 0.35, 0.20,
                  -0.25, -0.40, -0.50, -0.65, -0.80],
        'gamma': [0.02] * 10,
        'theta': [-0.05] * 10,
        'vega': [0.30] * 10,
        'rho': [0.10] * 10,
    })


@pytest.fixture
def mock_data_source(sample_option_chain):
    """Create a mock data source."""
    mock = Mock()
    mock.get_option_chain = Mock(return_value=sample_option_chain)
    mock.get_implied_volatility = Mock(return_value=0.20)
    return mock


@pytest.fixture
def sample_market_data(sample_option_chain):
    """Create sample market data dictionary."""
    return {
        'date': datetime(2024, 1, 15),
        'spot': 448.0,
        'option_chain': sample_option_chain,
        'iv': 0.20,
        'vix': 15.5,
    }


class MockStrategy(Strategy):
    """Mock strategy for testing."""

    def __init__(
        self,
        name: str = 'MockStrategy',
        initial_capital: float = 100000.0,
        should_enter_result: bool = False,
        should_exit_result: bool = False
    ):
        super().__init__(name=name, initial_capital=initial_capital)
        self._should_enter_result = should_enter_result
        self._should_exit_result = should_exit_result
        self._structures_to_create: List[OptionStructure] = []

    def should_enter(self, market_data: Dict[str, Any]) -> bool:
        return self._should_enter_result

    def should_exit(
        self,
        structure: OptionStructure,
        market_data: Dict[str, Any]
    ) -> bool:
        return self._should_exit_result

    def set_should_enter(self, value: bool) -> None:
        self._should_enter_result = value

    def set_should_exit(self, value: bool) -> None:
        self._should_exit_result = value

    def add_structure_to_create(self, structure: OptionStructure) -> None:
        self._structures_to_create.append(structure)

    def create_structure(self, market_data: Dict[str, Any]) -> Optional[OptionStructure]:
        if self._structures_to_create:
            return self._structures_to_create.pop(0)
        return None


# =============================================================================
# TradingCalendar Tests
# =============================================================================

class TestTradingCalendar:
    """Tests for TradingCalendar class."""

    def test_init_default(self):
        """Test default initialization."""
        calendar = TradingCalendar()
        assert calendar is not None

    def test_init_with_holidays(self):
        """Test initialization with holidays."""
        holidays = [date(2024, 1, 15), date(2024, 2, 19)]
        calendar = TradingCalendar(holidays=holidays)
        assert not calendar.is_trading_day(date(2024, 1, 15))
        assert not calendar.is_trading_day(date(2024, 2, 19))

    def test_is_trading_day_weekday(self):
        """Test weekday is trading day."""
        calendar = TradingCalendar()
        # Monday 2024-01-08
        assert calendar.is_trading_day(date(2024, 1, 8))

    def test_is_trading_day_weekend_saturday(self):
        """Test Saturday is not trading day."""
        calendar = TradingCalendar()
        # Saturday 2024-01-06
        assert not calendar.is_trading_day(date(2024, 1, 6))

    def test_is_trading_day_weekend_sunday(self):
        """Test Sunday is not trading day."""
        calendar = TradingCalendar()
        # Sunday 2024-01-07
        assert not calendar.is_trading_day(date(2024, 1, 7))

    def test_is_trading_day_new_year(self):
        """Test New Year's Day is not trading day."""
        calendar = TradingCalendar()
        assert not calendar.is_trading_day(date(2024, 1, 1))

    def test_is_trading_day_christmas(self):
        """Test Christmas is not trading day."""
        calendar = TradingCalendar()
        assert not calendar.is_trading_day(date(2024, 12, 25))

    def test_is_trading_day_with_datetime(self):
        """Test is_trading_day with datetime input."""
        calendar = TradingCalendar()
        dt = datetime(2024, 1, 8, 10, 30)
        assert calendar.is_trading_day(dt)

    def test_add_holidays(self):
        """Test adding holidays."""
        calendar = TradingCalendar()
        calendar.add_holidays([date(2024, 1, 8)])
        assert not calendar.is_trading_day(date(2024, 1, 8))

    def test_next_trading_day(self):
        """Test getting next trading day."""
        calendar = TradingCalendar()
        # Friday 2024-01-05 -> Monday 2024-01-08
        next_day = calendar.next_trading_day(date(2024, 1, 5))
        assert next_day == date(2024, 1, 8)

    def test_next_trading_day_from_saturday(self):
        """Test next trading day from Saturday."""
        calendar = TradingCalendar()
        # Saturday 2024-01-06 -> Monday 2024-01-08
        next_day = calendar.next_trading_day(date(2024, 1, 6))
        assert next_day == date(2024, 1, 8)

    def test_get_trading_days(self):
        """Test getting trading days in range."""
        calendar = TradingCalendar()
        # Week of 2024-01-08 to 2024-01-12 (Mon-Fri)
        trading_days = calendar.get_trading_days(
            date(2024, 1, 8), date(2024, 1, 12)
        )
        assert len(trading_days) == 5
        assert trading_days[0] == date(2024, 1, 8)
        assert trading_days[-1] == date(2024, 1, 12)

    def test_get_trading_days_with_weekend(self):
        """Test getting trading days spanning weekend."""
        calendar = TradingCalendar()
        # 2024-01-05 (Fri) to 2024-01-08 (Mon)
        trading_days = calendar.get_trading_days(
            date(2024, 1, 5), date(2024, 1, 8)
        )
        assert len(trading_days) == 2
        assert date(2024, 1, 5) in trading_days
        assert date(2024, 1, 8) in trading_days


# =============================================================================
# DataStream Tests
# =============================================================================

class TestDataStream:
    """Tests for DataStream class."""

    def test_init_valid(self, mock_data_source):
        """Test valid initialization."""
        stream = DataStream(
            data_source=mock_data_source,
            start_date=datetime(2024, 1, 8),
            end_date=datetime(2024, 1, 12),
            underlying='SPY'
        )
        assert stream.underlying == 'SPY'
        assert stream.num_trading_days == 5

    def test_init_invalid_data_source(self):
        """Test initialization with None data source."""
        with pytest.raises(DataStreamConfigError, match="cannot be None"):
            DataStream(
                data_source=None,
                start_date=datetime(2024, 1, 8),
                end_date=datetime(2024, 1, 12),
                underlying='SPY'
            )

    def test_init_invalid_dates(self, mock_data_source):
        """Test initialization with end before start."""
        with pytest.raises(DataStreamConfigError, match="must be before"):
            DataStream(
                data_source=mock_data_source,
                start_date=datetime(2024, 1, 12),
                end_date=datetime(2024, 1, 8),
                underlying='SPY'
            )

    def test_init_invalid_underlying(self, mock_data_source):
        """Test initialization with empty underlying."""
        with pytest.raises(DataStreamConfigError, match="non-empty string"):
            DataStream(
                data_source=mock_data_source,
                start_date=datetime(2024, 1, 8),
                end_date=datetime(2024, 1, 12),
                underlying=''
            )

    def test_init_invalid_dte_range(self, mock_data_source):
        """Test initialization with invalid DTE range."""
        with pytest.raises(DataStreamConfigError, match="Invalid dte_range"):
            DataStream(
                data_source=mock_data_source,
                start_date=datetime(2024, 1, 8),
                end_date=datetime(2024, 1, 12),
                underlying='SPY',
                dte_range=(30, 10)  # max < min
            )

    def test_iteration(self, mock_data_source, sample_option_chain):
        """Test iterator protocol."""
        stream = DataStream(
            data_source=mock_data_source,
            start_date=datetime(2024, 1, 8),
            end_date=datetime(2024, 1, 9),
            underlying='SPY'
        )

        # Mock returns data
        mock_data_source.get_option_chain.return_value = sample_option_chain

        count = 0
        for timestamp, market_data in stream:
            count += 1
            assert isinstance(timestamp, datetime)
            assert 'spot' in market_data
            assert 'option_chain' in market_data
            assert 'iv' in market_data

        assert count == 2  # Two trading days

    def test_properties(self, mock_data_source):
        """Test stream properties."""
        start = datetime(2024, 1, 8)
        end = datetime(2024, 1, 12)
        stream = DataStream(
            data_source=mock_data_source,
            start_date=start,
            end_date=end,
            underlying='SPY',
            dte_range=(7, 45)
        )

        assert stream.start_date == start
        assert stream.end_date == end
        assert stream.underlying == 'SPY'
        assert stream.dte_range == (7, 45)
        assert stream.num_trading_days == 5
        assert not stream.is_exhausted

    def test_reset(self, mock_data_source, sample_option_chain):
        """Test stream reset."""
        stream = DataStream(
            data_source=mock_data_source,
            start_date=datetime(2024, 1, 8),
            end_date=datetime(2024, 1, 9),
            underlying='SPY'
        )
        mock_data_source.get_option_chain.return_value = sample_option_chain

        # Consume stream
        for _ in stream:
            pass

        assert stream.is_exhausted

        # Reset and iterate again
        stream.reset()
        assert not stream.is_exhausted

        count = 0
        for _ in stream:
            count += 1
        assert count == 2

    def test_skip(self, mock_data_source):
        """Test skipping days."""
        stream = DataStream(
            data_source=mock_data_source,
            start_date=datetime(2024, 1, 8),
            end_date=datetime(2024, 1, 12),
            underlying='SPY'
        )

        stream.skip(3)
        assert stream.current_index == 3

    def test_progress(self, mock_data_source, sample_option_chain):
        """Test progress tracking."""
        stream = DataStream(
            data_source=mock_data_source,
            start_date=datetime(2024, 1, 8),
            end_date=datetime(2024, 1, 12),
            underlying='SPY'
        )
        mock_data_source.get_option_chain.return_value = sample_option_chain

        assert stream.progress == 0.0

        for i, _ in enumerate(stream, 1):
            pass

        assert stream.progress == 1.0

    def test_cache_enabled(self, mock_data_source, sample_option_chain):
        """Test caching is working."""
        stream = DataStream(
            data_source=mock_data_source,
            start_date=datetime(2024, 1, 8),
            end_date=datetime(2024, 1, 8),
            underlying='SPY',
            cache_enabled=True
        )
        mock_data_source.get_option_chain.return_value = sample_option_chain

        # First iteration
        for _ in stream:
            pass

        # Reset and iterate again
        stream.reset()
        for _ in stream:
            pass

        # Should have called get_option_chain only once due to cache
        assert mock_data_source.get_option_chain.call_count == 1

    def test_clear_cache(self, mock_data_source, sample_option_chain):
        """Test clearing cache."""
        stream = DataStream(
            data_source=mock_data_source,
            start_date=datetime(2024, 1, 8),
            end_date=datetime(2024, 1, 8),
            underlying='SPY',
            cache_enabled=True
        )
        mock_data_source.get_option_chain.return_value = sample_option_chain

        for _ in stream:
            pass

        stream.clear_cache()
        stream.reset()

        for _ in stream:
            pass

        # Should have called get_option_chain twice after cache clear
        assert mock_data_source.get_option_chain.call_count == 2


# =============================================================================
# PositionManager Tests
# =============================================================================

class TestPositionManager:
    """Tests for PositionManager class."""

    def test_init(self):
        """Test initialization."""
        manager = PositionManager()
        assert manager.num_positions == 0
        assert manager.num_closed_positions == 0
        assert manager.total_realized_pnl == 0.0

    def test_add_position(self, sample_structure):
        """Test adding a position."""
        manager = PositionManager()

        structure_id = manager.add_position(
            structure=sample_structure,
            strategy_name='TestStrategy'
        )

        assert manager.num_positions == 1
        assert structure_id == sample_structure.structure_id
        assert manager.has_position(structure_id)

    def test_add_position_invalid(self):
        """Test adding None structure."""
        manager = PositionManager()

        with pytest.raises(ValueError, match="cannot be None"):
            manager.add_position(None, 'TestStrategy')

    def test_add_duplicate_position(self, sample_structure):
        """Test adding duplicate position."""
        manager = PositionManager()

        manager.add_position(sample_structure, 'TestStrategy')

        with pytest.raises(DuplicatePositionError):
            manager.add_position(sample_structure, 'TestStrategy')

    def test_remove_position(self, sample_structure):
        """Test removing a position."""
        manager = PositionManager()

        structure_id = manager.add_position(sample_structure, 'TestStrategy')

        removed = manager.remove_position(
            structure_id=structure_id,
            realized_pnl=500.0
        )

        assert manager.num_positions == 0
        assert manager.num_closed_positions == 1
        assert removed == sample_structure
        assert manager.total_realized_pnl == 500.0

    def test_remove_position_not_found(self):
        """Test removing non-existent position."""
        manager = PositionManager()

        with pytest.raises(PositionNotFoundError):
            manager.remove_position('nonexistent')

    def test_get_position(self, sample_structure):
        """Test getting a position."""
        manager = PositionManager()

        structure_id = manager.add_position(sample_structure, 'TestStrategy')

        retrieved = manager.get_position(structure_id)
        assert retrieved == sample_structure

        not_found = manager.get_position('nonexistent')
        assert not_found is None

    def test_get_all_positions(self, sample_structure, sample_option):
        """Test getting all positions."""
        manager = PositionManager()

        structure2 = OptionStructure(
            structure_type='single_leg',
            underlying='SPY'
        )
        structure2.add_option(sample_option)

        manager.add_position(sample_structure, 'Strategy1')
        manager.add_position(structure2, 'Strategy2')

        all_positions = manager.get_all_positions()
        assert len(all_positions) == 2

    def test_get_positions_by_strategy(self, sample_structure, sample_option):
        """Test getting positions by strategy."""
        manager = PositionManager()

        structure2 = OptionStructure(
            structure_type='single_leg',
            underlying='SPY'
        )
        structure2.add_option(sample_option)

        manager.add_position(sample_structure, 'Strategy1')
        manager.add_position(structure2, 'Strategy2')

        strat1_positions = manager.get_positions_by_strategy('Strategy1')
        assert len(strat1_positions) == 1
        assert strat1_positions[0] == sample_structure

        strat2_positions = manager.get_positions_by_strategy('Strategy2')
        assert len(strat2_positions) == 1

    def test_get_positions_by_underlying(self, sample_structure, sample_option):
        """Test getting positions by underlying."""
        manager = PositionManager()

        manager.add_position(sample_structure, 'TestStrategy')

        spy_positions = manager.get_positions_by_underlying('SPY')
        assert len(spy_positions) == 1

        qqq_positions = manager.get_positions_by_underlying('QQQ')
        assert len(qqq_positions) == 0

    def test_calculate_total_margin(self, sample_structure):
        """Test margin calculation."""
        manager = PositionManager()

        manager.add_position(sample_structure, 'TestStrategy')

        margin = manager.calculate_total_margin()
        assert margin > 0

    def test_calculate_portfolio_value(self, sample_structure, sample_market_data):
        """Test portfolio value calculation."""
        manager = PositionManager()

        manager.add_position(sample_structure, 'TestStrategy')

        value = manager.calculate_portfolio_value(sample_market_data)
        # Value depends on P&L calculation
        assert isinstance(value, float)

    def test_get_portfolio_greeks(self, sample_structure, sample_market_data):
        """Test portfolio Greeks calculation."""
        manager = PositionManager()

        manager.add_position(sample_structure, 'TestStrategy')

        greeks = manager.get_portfolio_greeks(sample_market_data)

        assert 'delta' in greeks
        assert 'gamma' in greeks
        assert 'theta' in greeks
        assert 'vega' in greeks
        assert 'rho' in greeks

    def test_calculate_unrealized_pnl(self, sample_structure):
        """Test unrealized P&L calculation."""
        manager = PositionManager()

        manager.add_position(sample_structure, 'TestStrategy')

        pnl = manager.calculate_unrealized_pnl()
        assert isinstance(pnl, float)

    def test_calculate_total_pnl(self, sample_structure):
        """Test total P&L calculation."""
        manager = PositionManager()

        manager.add_position(sample_structure, 'TestStrategy')

        total_pnl = manager.calculate_total_pnl()
        assert isinstance(total_pnl, float)

    def test_get_statistics(self, sample_structure):
        """Test getting statistics."""
        manager = PositionManager()

        manager.add_position(sample_structure, 'TestStrategy')

        stats = manager.get_statistics()

        assert stats['num_active'] == 1
        assert stats['num_closed'] == 0
        assert 'realized_pnl' in stats
        assert 'unrealized_pnl' in stats

    def test_get_position_summary(self, sample_structure):
        """Test getting position summary."""
        manager = PositionManager()

        manager.add_position(sample_structure, 'TestStrategy')

        summary = manager.get_position_summary()

        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 1
        assert 'structure_id' in summary.columns

    def test_clear(self, sample_structure):
        """Test clearing manager."""
        manager = PositionManager()

        manager.add_position(sample_structure, 'TestStrategy')
        manager.clear()

        assert manager.num_positions == 0
        assert manager.num_closed_positions == 0

    def test_iterator(self, sample_structure):
        """Test iterator protocol."""
        manager = PositionManager()

        manager.add_position(sample_structure, 'TestStrategy')

        count = 0
        for pos in manager:
            count += 1
            assert isinstance(pos, OptionStructure)

        assert count == 1

    def test_len(self, sample_structure, sample_option):
        """Test len function."""
        manager = PositionManager()

        assert len(manager) == 0

        manager.add_position(sample_structure, 'TestStrategy')
        assert len(manager) == 1

        structure2 = OptionStructure(structure_type='single', underlying='SPY')
        structure2.add_option(sample_option)
        manager.add_position(structure2, 'TestStrategy')
        assert len(manager) == 2

    def test_contains(self, sample_structure):
        """Test contains operator."""
        manager = PositionManager()

        structure_id = manager.add_position(sample_structure, 'TestStrategy')

        assert structure_id in manager
        assert 'nonexistent' not in manager


# =============================================================================
# PositionRecord Tests
# =============================================================================

class TestPositionRecord:
    """Tests for PositionRecord class."""

    def test_init(self, sample_structure):
        """Test initialization."""
        record = PositionRecord(
            structure=sample_structure,
            strategy_name='TestStrategy'
        )

        assert record.structure == sample_structure
        assert record.strategy_name == 'TestStrategy'
        assert record.is_active
        assert record.realized_pnl is None

    def test_close(self, sample_structure):
        """Test closing a record."""
        record = PositionRecord(
            structure=sample_structure,
            strategy_name='TestStrategy'
        )

        close_time = datetime.now()
        record.close(close_timestamp=close_time, realized_pnl=500.0)

        assert not record.is_active
        assert record.close_timestamp == close_time
        assert record.realized_pnl == 500.0

    def test_to_dict(self, sample_structure):
        """Test dictionary conversion."""
        record = PositionRecord(
            structure=sample_structure,
            strategy_name='TestStrategy'
        )

        d = record.to_dict()

        assert 'record_id' in d
        assert 'structure_id' in d
        assert 'strategy_name' in d
        assert 'is_active' in d


# =============================================================================
# ExecutionModel Tests
# =============================================================================

class TestExecutionModel:
    """Tests for ExecutionModel class."""

    def test_init_default(self):
        """Test default initialization."""
        execution = ExecutionModel()

        assert execution.commission_per_contract == 0.65
        assert execution.slippage_pct == 0.0
        assert execution.use_bid_ask is True

    def test_init_custom(self):
        """Test custom initialization."""
        execution = ExecutionModel(
            commission_per_contract=1.0,
            slippage_pct=0.005,
            use_bid_ask=False
        )

        assert execution.commission_per_contract == 1.0
        assert execution.slippage_pct == 0.005
        assert execution.use_bid_ask is False

    def test_init_invalid_commission(self):
        """Test initialization with negative commission."""
        with pytest.raises(ExecutionConfigError, match="non-negative"):
            ExecutionModel(commission_per_contract=-1.0)

    def test_init_invalid_slippage(self):
        """Test initialization with invalid slippage."""
        with pytest.raises(ExecutionConfigError, match="between 0 and 1"):
            ExecutionModel(slippage_pct=1.5)

    def test_execute_entry(self, sample_structure, sample_market_data):
        """Test entry execution."""
        execution = ExecutionModel(
            commission_per_contract=0.65,
            slippage_pct=0.0
        )

        result = execution.execute_entry(
            structure=sample_structure,
            market_data=sample_market_data
        )

        assert 'entry_prices' in result
        assert 'commissions' in result
        assert 'total_cost' in result
        assert 'fills' in result
        assert len(result['fills']) == sample_structure.num_legs

    def test_execute_exit(self, sample_structure, sample_market_data):
        """Test exit execution."""
        execution = ExecutionModel(
            commission_per_contract=0.65,
            slippage_pct=0.0
        )

        result = execution.execute_exit(
            structure=sample_structure,
            market_data=sample_market_data
        )

        assert 'exit_prices' in result
        assert 'commissions' in result
        assert 'total_proceeds' in result
        assert 'fills' in result

    def test_execute_entry_with_slippage(self, sample_structure, sample_market_data):
        """Test entry with slippage."""
        execution_no_slip = ExecutionModel(slippage_pct=0.0)
        execution_slip = ExecutionModel(slippage_pct=0.01)

        result_no_slip = execution_no_slip.execute_entry(
            sample_structure, sample_market_data
        )
        result_slip = execution_slip.execute_entry(
            sample_structure, sample_market_data
        )

        # Slippage should increase cost
        assert result_slip['slippage'] > result_no_slip['slippage']

    def test_get_fill_price_buy(self, sample_option, sample_market_data):
        """Test fill price for buy order."""
        execution = ExecutionModel(use_bid_ask=True)

        price = execution.get_fill_price(
            option=sample_option,
            market_data=sample_market_data,
            side='buy'
        )

        assert price > 0

    def test_get_fill_price_sell(self, sample_option, sample_market_data):
        """Test fill price for sell order."""
        execution = ExecutionModel(use_bid_ask=True)

        price = execution.get_fill_price(
            option=sample_option,
            market_data=sample_market_data,
            side='sell'
        )

        assert price > 0

    def test_get_fill_price_invalid_side(self, sample_option, sample_market_data):
        """Test fill price with invalid side."""
        execution = ExecutionModel()

        with pytest.raises(ValueError, match="must be 'buy' or 'sell'"):
            execution.get_fill_price(
                option=sample_option,
                market_data=sample_market_data,
                side='invalid'
            )

    def test_commission_calculation(self, sample_structure, sample_market_data):
        """Test commission calculation."""
        commission_rate = 1.0
        execution = ExecutionModel(commission_per_contract=commission_rate)

        result = execution.execute_entry(sample_structure, sample_market_data)

        expected_commission = commission_rate * result['num_contracts']
        assert result['commissions'] == expected_commission

    def test_set_commission(self):
        """Test updating commission."""
        execution = ExecutionModel(commission_per_contract=0.65)

        execution.set_commission(1.0)
        assert execution.commission_per_contract == 1.0

        with pytest.raises(ExecutionConfigError):
            execution.set_commission(-1.0)

    def test_set_slippage(self):
        """Test updating slippage."""
        execution = ExecutionModel(slippage_pct=0.0)

        execution.set_slippage(0.01)
        assert execution.slippage_pct == 0.01

        with pytest.raises(ExecutionConfigError):
            execution.set_slippage(1.5)

    def test_get_execution_summary(self, sample_structure, sample_market_data):
        """Test execution summary."""
        execution = ExecutionModel()

        # Perform some executions
        execution.execute_entry(sample_structure, sample_market_data)
        execution.execute_exit(sample_structure, sample_market_data)

        summary = execution.get_execution_summary()

        assert summary['num_executions'] == 2
        assert summary['num_entries'] == 1
        assert summary['num_exits'] == 1
        assert 'total_commissions' in summary

    def test_clear_log(self, sample_structure, sample_market_data):
        """Test clearing execution log."""
        execution = ExecutionModel()

        execution.execute_entry(sample_structure, sample_market_data)
        assert len(execution.execution_log) == 1

        execution.clear_log()
        assert len(execution.execution_log) == 0


# =============================================================================
# BacktestEngine Tests
# =============================================================================

class TestBacktestEngine:
    """Tests for BacktestEngine class."""

    def test_init_valid(self, mock_data_source, sample_option_chain):
        """Test valid initialization."""
        mock_data_source.get_option_chain.return_value = sample_option_chain

        strategy = MockStrategy()
        data_stream = DataStream(
            data_source=mock_data_source,
            start_date=datetime(2024, 1, 8),
            end_date=datetime(2024, 1, 12),
            underlying='SPY'
        )
        execution = ExecutionModel()

        engine = BacktestEngine(
            strategy=strategy,
            data_stream=data_stream,
            execution_model=execution,
            initial_capital=100000.0
        )

        assert engine.strategy == strategy
        assert engine.initial_capital == 100000.0
        assert not engine.is_running

    def test_init_invalid_strategy(self, mock_data_source):
        """Test initialization with None strategy."""
        data_stream = DataStream(
            data_source=mock_data_source,
            start_date=datetime(2024, 1, 8),
            end_date=datetime(2024, 1, 12),
            underlying='SPY'
        )
        execution = ExecutionModel()

        with pytest.raises(BacktestConfigError, match="cannot be None"):
            BacktestEngine(
                strategy=None,
                data_stream=data_stream,
                execution_model=execution
            )

    def test_init_invalid_capital(self, mock_data_source):
        """Test initialization with invalid capital."""
        strategy = MockStrategy()
        data_stream = DataStream(
            data_source=mock_data_source,
            start_date=datetime(2024, 1, 8),
            end_date=datetime(2024, 1, 12),
            underlying='SPY'
        )
        execution = ExecutionModel()

        with pytest.raises(BacktestConfigError, match="must be positive"):
            BacktestEngine(
                strategy=strategy,
                data_stream=data_stream,
                execution_model=execution,
                initial_capital=-1000.0
            )

    def test_run_no_trades(self, mock_data_source, sample_option_chain):
        """Test running backtest with no trades."""
        mock_data_source.get_option_chain.return_value = sample_option_chain

        strategy = MockStrategy(should_enter_result=False)
        data_stream = DataStream(
            data_source=mock_data_source,
            start_date=datetime(2024, 1, 8),
            end_date=datetime(2024, 1, 9),
            underlying='SPY'
        )
        execution = ExecutionModel()

        engine = BacktestEngine(
            strategy=strategy,
            data_stream=data_stream,
            execution_model=execution,
            initial_capital=100000.0
        )

        results = engine.run()

        assert results['initial_capital'] == 100000.0
        assert results['final_equity'] == 100000.0
        assert results['total_return'] == 0.0
        assert results['num_trades'] == 0

    def test_run_basic(self, mock_data_source, sample_option_chain, small_structure):
        """Test basic backtest run with trades."""
        mock_data_source.get_option_chain.return_value = sample_option_chain

        strategy = MockStrategy(
            should_enter_result=True,
            should_exit_result=False
        )
        strategy.add_structure_to_create(small_structure)

        data_stream = DataStream(
            data_source=mock_data_source,
            start_date=datetime(2024, 1, 8),
            end_date=datetime(2024, 1, 9),
            underlying='SPY'
        )
        execution = ExecutionModel(commission_per_contract=0.0)

        engine = BacktestEngine(
            strategy=strategy,
            data_stream=data_stream,
            execution_model=execution,
            initial_capital=100000.0
        )

        results = engine.run()

        assert 'equity_curve' in results
        assert 'trade_log' in results
        assert isinstance(results['equity_curve'], pd.DataFrame)

    def test_generate_equity_curve(self, mock_data_source, sample_option_chain):
        """Test equity curve generation."""
        mock_data_source.get_option_chain.return_value = sample_option_chain

        strategy = MockStrategy()
        data_stream = DataStream(
            data_source=mock_data_source,
            start_date=datetime(2024, 1, 8),
            end_date=datetime(2024, 1, 9),
            underlying='SPY'
        )
        execution = ExecutionModel()

        engine = BacktestEngine(
            strategy=strategy,
            data_stream=data_stream,
            execution_model=execution,
            initial_capital=100000.0
        )

        engine.run()

        equity_df = engine.generate_equity_curve()

        assert isinstance(equity_df, pd.DataFrame)
        assert 'equity' in equity_df.columns
        assert len(equity_df) == 2  # Two trading days

    def test_get_trade_log(self, mock_data_source, sample_option_chain, small_structure):
        """Test trade log generation."""
        mock_data_source.get_option_chain.return_value = sample_option_chain

        strategy = MockStrategy(should_enter_result=True)
        strategy.add_structure_to_create(small_structure)

        data_stream = DataStream(
            data_source=mock_data_source,
            start_date=datetime(2024, 1, 8),
            end_date=datetime(2024, 1, 8),
            underlying='SPY'
        )
        execution = ExecutionModel()

        engine = BacktestEngine(
            strategy=strategy,
            data_stream=data_stream,
            execution_model=execution
        )

        engine.run()

        trade_log = engine.get_trade_log()

        assert isinstance(trade_log, pd.DataFrame)
        # Should have at least the entry trade
        assert len(trade_log) >= 1

    def test_properties(self, mock_data_source, sample_option_chain):
        """Test engine properties."""
        mock_data_source.get_option_chain.return_value = sample_option_chain

        strategy = MockStrategy()
        data_stream = DataStream(
            data_source=mock_data_source,
            start_date=datetime(2024, 1, 8),
            end_date=datetime(2024, 1, 12),
            underlying='SPY'
        )
        execution = ExecutionModel()

        engine = BacktestEngine(
            strategy=strategy,
            data_stream=data_stream,
            execution_model=execution,
            initial_capital=100000.0
        )

        assert engine.strategy == strategy
        assert engine.data_stream == data_stream
        assert engine.execution_model == execution
        assert engine.initial_capital == 100000.0
        assert engine.current_capital == 100000.0
        assert not engine.is_running
        assert engine.num_trades == 0

    def test_callbacks(self, mock_data_source, sample_option_chain):
        """Test callback functionality."""
        mock_data_source.get_option_chain.return_value = sample_option_chain

        strategy = MockStrategy()
        data_stream = DataStream(
            data_source=mock_data_source,
            start_date=datetime(2024, 1, 8),
            end_date=datetime(2024, 1, 9),
            underlying='SPY'
        )
        execution = ExecutionModel()

        engine = BacktestEngine(
            strategy=strategy,
            data_stream=data_stream,
            execution_model=execution
        )

        step_count = [0]

        def on_step(timestamp, market_data, state):
            step_count[0] += 1

        engine.set_on_step_callback(on_step)
        engine.run()

        assert step_count[0] == 2  # Two trading days


# =============================================================================
# BacktestState Tests
# =============================================================================

class TestBacktestState:
    """Tests for BacktestState class."""

    def test_init(self):
        """Test initialization."""
        state = BacktestState(
            timestamp=datetime(2024, 1, 15),
            equity=100000.0,
            cash=50000.0,
            positions_value=50000.0,
            num_positions=2,
            greeks={'delta': 10.0, 'gamma': 0.5},
            realized_pnl=1000.0,
            unrealized_pnl=500.0
        )

        assert state.timestamp == datetime(2024, 1, 15)
        assert state.equity == 100000.0
        assert state.cash == 50000.0
        assert state.positions_value == 50000.0
        assert state.num_positions == 2
        assert state.realized_pnl == 1000.0
        assert state.unrealized_pnl == 500.0

    def test_to_dict(self):
        """Test dictionary conversion."""
        state = BacktestState(
            timestamp=datetime(2024, 1, 15),
            equity=100000.0,
            cash=50000.0,
            positions_value=50000.0,
            num_positions=2,
            greeks={'delta': 10.0, 'gamma': 0.5},
            realized_pnl=1000.0,
            unrealized_pnl=500.0
        )

        d = state.to_dict()

        assert d['timestamp'] == datetime(2024, 1, 15)
        assert d['equity'] == 100000.0
        assert d['delta'] == 10.0
        assert d['gamma'] == 0.5


# =============================================================================
# TradeRecord Tests
# =============================================================================

class TestTradeRecord:
    """Tests for TradeRecord class."""

    def test_init(self):
        """Test initialization."""
        record = TradeRecord(
            trade_id='T000001',
            structure_id='S123',
            structure_type='straddle',
            underlying='SPY',
            action='open',
            timestamp=datetime(2024, 1, 15),
            num_legs=2,
            net_premium=500.0,
            total_cost=100.0,
            commission=1.30
        )

        assert record.trade_id == 'T000001'
        assert record.structure_id == 'S123'
        assert record.action == 'open'
        assert record.num_legs == 2
        assert record.commission == 1.30

    def test_to_dict(self):
        """Test dictionary conversion."""
        record = TradeRecord(
            trade_id='T000001',
            structure_id='S123',
            structure_type='straddle',
            underlying='SPY',
            action='close',
            timestamp=datetime(2024, 1, 15),
            num_legs=2,
            net_premium=500.0,
            realized_pnl=250.0,
            exit_reason='profit_target'
        )

        d = record.to_dict()

        assert d['trade_id'] == 'T000001'
        assert d['action'] == 'close'
        assert d['realized_pnl'] == 250.0
        assert d['exit_reason'] == 'profit_target'


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_backtest_workflow(self, mock_data_source, sample_option_chain):
        """Test complete backtest workflow."""
        mock_data_source.get_option_chain.return_value = sample_option_chain

        # Create components
        strategy = MockStrategy(
            name='TestIntegration',
            initial_capital=100000.0
        )
        data_stream = DataStream(
            data_source=mock_data_source,
            start_date=datetime(2024, 1, 8),
            end_date=datetime(2024, 1, 12),
            underlying='SPY'
        )
        execution = ExecutionModel(
            commission_per_contract=0.65,
            slippage_pct=0.001
        )

        # Create engine
        engine = BacktestEngine(
            strategy=strategy,
            data_stream=data_stream,
            execution_model=execution,
            initial_capital=100000.0
        )

        # Run backtest
        results = engine.run()

        # Verify results structure
        assert 'equity_curve' in results
        assert 'trade_log' in results
        assert 'greeks_history' in results
        assert 'final_equity' in results
        assert 'total_return' in results
        assert 'strategy_stats' in results
        assert 'execution_stats' in results

    def test_position_manager_with_execution(
        self,
        sample_structure,
        sample_market_data
    ):
        """Test position manager with execution model."""
        # Create components
        manager = PositionManager()
        execution = ExecutionModel(commission_per_contract=0.65)

        # Execute entry
        entry_result = execution.execute_entry(
            sample_structure,
            sample_market_data
        )

        # Add to position manager
        manager.add_position(sample_structure, 'TestStrategy')

        assert manager.num_positions == 1

        # Execute exit
        exit_result = execution.execute_exit(
            sample_structure,
            sample_market_data
        )

        # Calculate P&L
        realized_pnl = exit_result['total_proceeds'] - entry_result['total_cost']

        # Remove from position manager
        manager.remove_position(
            sample_structure.structure_id,
            realized_pnl=realized_pnl
        )

        assert manager.num_positions == 0
        assert manager.num_closed_positions == 1
        assert manager.total_realized_pnl == realized_pnl

    def test_data_stream_with_engine(self, mock_data_source, sample_option_chain):
        """Test data stream integration with engine."""
        mock_data_source.get_option_chain.return_value = sample_option_chain

        strategy = MockStrategy()
        data_stream = DataStream(
            data_source=mock_data_source,
            start_date=datetime(2024, 1, 8),
            end_date=datetime(2024, 1, 12),
            underlying='SPY',
            preload=True  # Test preloading
        )
        execution = ExecutionModel()

        engine = BacktestEngine(
            strategy=strategy,
            data_stream=data_stream,
            execution_model=execution
        )

        results = engine.run()

        # Verify all trading days were processed
        equity_curve = results['equity_curve']
        assert len(equity_curve) == 5  # 5 trading days


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_option_chain(self, mock_data_source):
        """Test handling empty option chain."""
        mock_data_source.get_option_chain.return_value = pd.DataFrame()

        stream = DataStream(
            data_source=mock_data_source,
            start_date=datetime(2024, 1, 8),
            end_date=datetime(2024, 1, 9),
            underlying='SPY',
            skip_missing_data=True
        )

        count = 0
        for _ in stream:
            count += 1

        # Should skip days with no data
        assert count == 0

    def test_missing_price_data(self, sample_option, sample_market_data):
        """Test handling missing price data."""
        execution = ExecutionModel()

        # Remove option chain from market data
        market_data_no_chain = sample_market_data.copy()
        market_data_no_chain['option_chain'] = pd.DataFrame()

        # Should fall back to option's current price
        price = execution.get_fill_price(
            sample_option,
            market_data_no_chain,
            'buy'
        )

        assert price > 0  # Should use entry_price as fallback

    def test_zero_quantity_option(self):
        """Test option with zero quantity."""
        with pytest.raises(Exception):  # OptionValidationError
            Option(
                option_type='call',
                position_type='long',
                underlying='SPY',
                strike=450.0,
                expiration=datetime(2024, 3, 15),
                quantity=0,  # Invalid
                entry_price=5.50,
                entry_date=datetime(2024, 1, 15),
                underlying_price_at_entry=448.0
            )

    def test_position_manager_empty_portfolio_greeks(self, sample_market_data):
        """Test getting Greeks for empty portfolio."""
        manager = PositionManager()

        greeks = manager.get_portfolio_greeks(sample_market_data)

        for greek in ['delta', 'gamma', 'theta', 'vega', 'rho']:
            assert greeks[greek] == 0.0

    def test_execution_with_extreme_slippage(
        self,
        sample_structure,
        sample_market_data
    ):
        """Test execution with high slippage."""
        # Maximum allowed slippage
        execution = ExecutionModel(slippage_pct=0.99)

        result = execution.execute_entry(sample_structure, sample_market_data)

        assert result['slippage'] > 0


# =============================================================================
# Performance Tests
# =============================================================================

class TestPerformance:
    """Performance-related tests."""

    def test_large_position_manager(self, sample_option):
        """Test position manager with many positions."""
        manager = PositionManager()

        # Add 100 positions
        for i in range(100):
            structure = OptionStructure(
                structure_type='test',
                underlying='SPY',
                structure_id=f'S{i:04d}'
            )
            structure.add_option(sample_option)
            manager.add_position(structure, f'Strategy_{i % 5}')

        assert manager.num_positions == 100

        # Test strategy grouping performance
        for i in range(5):
            positions = manager.get_positions_by_strategy(f'Strategy_{i}')
            assert len(positions) == 20

    def test_data_stream_cache_efficiency(
        self,
        mock_data_source,
        sample_option_chain
    ):
        """Test cache efficiency."""
        mock_data_source.get_option_chain.return_value = sample_option_chain

        stream = DataStream(
            data_source=mock_data_source,
            start_date=datetime(2024, 1, 8),
            end_date=datetime(2024, 1, 12),
            underlying='SPY',
            cache_enabled=True
        )

        # First pass
        for _ in stream:
            pass

        call_count_after_first = mock_data_source.get_option_chain.call_count

        # Second pass (should use cache)
        stream.reset()
        for _ in stream:
            pass

        # Should not have made additional calls
        assert mock_data_source.get_option_chain.call_count == call_count_after_first


# =============================================================================
# Representation Tests
# =============================================================================

class TestRepresentations:
    """Tests for string representations."""

    def test_trading_calendar_repr(self):
        """Test TradingCalendar has no repr issues."""
        calendar = TradingCalendar()
        assert calendar is not None

    def test_data_stream_repr(self, mock_data_source):
        """Test DataStream repr."""
        stream = DataStream(
            data_source=mock_data_source,
            start_date=datetime(2024, 1, 8),
            end_date=datetime(2024, 1, 12),
            underlying='SPY'
        )

        repr_str = repr(stream)
        assert 'DataStream' in repr_str
        assert 'SPY' in repr_str

    def test_position_manager_repr(self, sample_structure):
        """Test PositionManager repr."""
        manager = PositionManager()
        manager.add_position(sample_structure, 'Test')

        repr_str = repr(manager)
        assert 'PositionManager' in repr_str

    def test_execution_model_repr(self):
        """Test ExecutionModel repr."""
        execution = ExecutionModel(
            commission_per_contract=0.65,
            slippage_pct=0.01
        )

        repr_str = repr(execution)
        assert 'ExecutionModel' in repr_str

    def test_backtest_engine_repr(self, mock_data_source, sample_option_chain):
        """Test BacktestEngine repr."""
        mock_data_source.get_option_chain.return_value = sample_option_chain

        strategy = MockStrategy()
        data_stream = DataStream(
            data_source=mock_data_source,
            start_date=datetime(2024, 1, 8),
            end_date=datetime(2024, 1, 12),
            underlying='SPY'
        )
        execution = ExecutionModel()

        engine = BacktestEngine(
            strategy=strategy,
            data_stream=data_stream,
            execution_model=execution
        )

        repr_str = repr(engine)
        assert 'BacktestEngine' in repr_str


# =============================================================================
# Module Exports Test
# =============================================================================

class TestModuleExports:
    """Test that all expected exports are available."""

    def test_engine_module_imports(self):
        """Test engine module can be imported."""
        from backtester.engine import (
            BacktestEngine,
            DataStream,
            ExecutionModel,
            PositionManager,
            BacktestState,
            TradeRecord,
            TradingCalendar,
            PositionRecord,
            ExecutionResult,
        )

        assert BacktestEngine is not None
        assert DataStream is not None
        assert ExecutionModel is not None
        assert PositionManager is not None

    def test_exception_imports(self):
        """Test exception classes can be imported."""
        from backtester.engine import (
            DataStreamError,
            DataStreamConfigError,
            DataNotAvailableError,
            PositionManagerError,
            PositionNotFoundError,
            DuplicatePositionError,
            ExecutionError,
            ExecutionConfigError,
            PriceNotAvailableError,
            BacktestError,
            BacktestConfigError,
            BacktestExecutionError,
        )

        assert DataStreamError is not None
        assert BacktestError is not None

    def test_constant_imports(self):
        """Test constants can be imported."""
        from backtester.engine import (
            DEFAULT_MIN_DTE,
            DEFAULT_MAX_DTE,
            DEFAULT_MARGIN_PER_CONTRACT,
            DEFAULT_COMMISSION_PER_CONTRACT,
            DEFAULT_SLIPPAGE_PCT,
            DEFAULT_SPREAD_PCT,
            DEFAULT_INITIAL_CAPITAL,
            DEFAULT_RISK_FREE_RATE,
        )

        assert DEFAULT_MIN_DTE == 7
        assert DEFAULT_COMMISSION_PER_CONTRACT == 0.65
        assert DEFAULT_INITIAL_CAPITAL == 100000.0


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
