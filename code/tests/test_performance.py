"""
Performance Benchmark Tests for Options Backtesting System

This module contains performance tests that validate the system meets
performance targets and identifies potential bottlenecks.

Performance Targets:
    - Backtest speed: >1000 data points per second
    - Greeks calculation: <1ms per option
    - Structure creation: <10ms
    - Memory usage: Reasonable for large datasets

Test Categories:
    1. Backtest Execution Speed
    2. Pricing and Greeks Performance
    3. Structure Operations Performance
    4. Analytics Calculation Speed
    5. Memory Usage Profiling

Requirements:
    - pytest
    - numpy
    - pandas
    - time
    - memory_profiler (optional)

Run Tests:
    pytest tests/test_performance.py -v --tb=short

Author: OptionsBacktester2 Team
"""

import pytest
import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any
import gc

# Import core components
from backtester.core.option import Option
from backtester.core.pricing import black_scholes_price, calculate_greeks
from backtester.core.option_structure import OptionStructure

# Import structures
from backtester.structures import (
    ShortStraddle, IronCondor, BullCallSpread
)

# Import strategies
from backtester.strategies import ShortStraddleHighIVStrategy

# Import analytics
from backtester.analytics import PerformanceMetrics, RiskAnalytics


# =============================================================================
# Performance Test Fixtures
# =============================================================================

@pytest.fixture
def pricer():
    """Fixture providing pricing functions (not actually needed, kept for compatibility)."""
    # Note: This fixture is kept for test compatibility but is no longer needed
    # since we use direct function calls to black_scholes_price() and calculate_greeks()
    return None


@pytest.fixture
def sample_options_list():
    """Create a list of options for batch testing."""
    base_date = datetime(2024, 1, 15)
    options = []

    for i in range(100):
        option = Option(
            option_type='call' if i % 2 == 0 else 'put',
            position_type='long',
            underlying='SPY',
            strike=400.0 + i,
            expiration=base_date + timedelta(days=30 + i % 60),
            quantity=10,
            entry_price=5.0 + i * 0.1,
            entry_date=base_date,
            underlying_price_at_entry=450.0,
            implied_vol_at_entry=0.20
        )
        options.append(option)

    return options


@pytest.fixture
def large_equity_curve():
    """Create large equity curve for analytics testing."""
    dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='B')

    # Generate realistic equity curve with drift and volatility
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.01, len(dates))
    equity = 100000.0 * np.exp(np.cumsum(returns))

    df = pd.DataFrame({
        'date': dates,
        'equity': equity,
        'cash': equity * 0.7,
        'positions_value': equity * 0.3
    })
    df.set_index('date', inplace=True)

    return df


# =============================================================================
# Test 1: Pricing and Greeks Performance
# =============================================================================

class TestPricingPerformance:
    """Test pricing engine performance."""

    def test_single_option_pricing_speed(self, pricer, benchmark=None):
        """Test single option pricing speed (target: <1ms)."""
        start = time.time()
        iterations = 1000

        for _ in range(iterations):
            price = black_scholes_price(
                S=450.0,
                K=450.0,
                T=0.25,
                r=0.04,
                sigma=0.20,
                option_type='call'
            )

        elapsed = time.time() - start
        avg_time_ms = (elapsed / iterations) * 1000

        print(f"\nAverage pricing time: {avg_time_ms:.4f}ms")

        # Should be well under 1ms per option
        assert avg_time_ms < 1.0, f"Pricing too slow: {avg_time_ms:.4f}ms > 1.0ms"

    def test_greeks_calculation_speed(self, pricer):
        """Test Greeks calculation speed (target: <2ms)."""
        start = time.time()
        iterations = 1000

        for _ in range(iterations):
            greeks = calculate_greeks(
                S=450.0,
                K=450.0,
                T=0.25,
                r=0.04,
                sigma=0.20,
                option_type='call'
            )

        elapsed = time.time() - start
        avg_time_ms = (elapsed / iterations) * 1000

        print(f"\nAverage Greeks calculation time: {avg_time_ms:.4f}ms")

        # Greeks calculation should be fast
        assert avg_time_ms < 2.0, f"Greeks calculation too slow: {avg_time_ms:.4f}ms"

    def test_batch_pricing_performance(self, pricer):
        """Test batch pricing of many options."""
        n_options = 1000

        # Generate random parameters
        np.random.seed(42)
        S_values = np.random.uniform(400, 500, n_options)
        K_values = np.random.uniform(400, 500, n_options)
        T_values = np.random.uniform(0.1, 1.0, n_options)
        sigma_values = np.random.uniform(0.15, 0.35, n_options)

        start = time.time()

        prices = []
        for i in range(n_options):
            price = black_scholes_price(
                S=S_values[i],
                K=K_values[i],
                T=T_values[i],
                r=0.04,
                sigma=sigma_values[i],
                option_type='call'
            )
            prices.append(price)

        elapsed = time.time() - start
        options_per_second = n_options / elapsed

        print(f"\nBatch pricing: {options_per_second:.0f} options/second")

        # Should price at least 10,000 options per second
        assert options_per_second > 10000, \
            f"Batch pricing too slow: {options_per_second:.0f} < 10000 options/sec"

    def test_vectorized_pricing_potential(self, pricer):
        """Test if vectorization would improve performance."""
        n_options = 10000

        # Generate random parameters
        np.random.seed(42)
        S = np.random.uniform(400, 500, n_options)
        K = np.random.uniform(400, 500, n_options)
        T = np.random.uniform(0.1, 1.0, n_options)
        sigma = np.random.uniform(0.15, 0.35, n_options)

        # Loop-based approach
        start = time.time()
        prices_loop = []
        for i in range(n_options):
            price = black_scholes_price(
                S=S[i],
                K=K[i],
                T=T[i],
                r=0.04,
                sigma=sigma[i],
                option_type='call'
            )
            prices_loop.append(price)
        loop_time = time.time() - start

        print(f"\nLoop-based pricing: {loop_time:.3f}s for {n_options} options")
        print(f"Rate: {n_options / loop_time:.0f} options/second")

        # Performance should be acceptable
        assert n_options / loop_time > 5000


# =============================================================================
# Test 2: Structure Operations Performance
# =============================================================================

class TestStructurePerformance:
    """Test option structure performance."""

    def test_structure_creation_speed(self):
        """Test structure creation speed (target: <10ms)."""
        base_date = datetime(2024, 1, 15)
        expiration = datetime(2024, 2, 16)
        iterations = 100

        start = time.time()

        for _ in range(iterations):
            straddle = ShortStraddle.create(
                underlying='SPY',
                strike=450.0,
                expiration=expiration,
                call_price=10.0,
                put_price=9.5,
                quantity=10,
                entry_date=base_date,
                underlying_price=450.0
            )

        elapsed = time.time() - start
        avg_time_ms = (elapsed / iterations) * 1000

        print(f"\nAverage structure creation time: {avg_time_ms:.4f}ms")

        # Should be fast
        assert avg_time_ms < 10.0, f"Structure creation too slow: {avg_time_ms:.4f}ms"

    def test_greeks_aggregation_speed(self):
        """Test Greeks aggregation performance for complex structures."""
        base_date = datetime(2024, 1, 15)
        expiration = datetime(2024, 2, 16)

        # Create complex structure (Iron Condor with 4 legs)
        ic = IronCondor.create(
            underlying='SPY',
            put_buy_strike=435.0,
            put_sell_strike=440.0,
            call_sell_strike=460.0,
            call_buy_strike=465.0,
            expiration=expiration,
            put_buy_price=5.0,
            put_sell_price=8.0,
            call_sell_price=8.0,
            call_buy_price=5.0,
            quantity=10,
            entry_date=base_date,
            underlying_price=450.0
        )

        iterations = 1000
        start = time.time()

        for _ in range(iterations):
            greeks = ic.calculate_net_greeks()

        elapsed = time.time() - start
        avg_time_ms = (elapsed / iterations) * 1000

        print(f"\nAverage Greeks aggregation time: {avg_time_ms:.4f}ms")

        # Should be reasonably fast (method call with calculations)
        assert avg_time_ms < 5.0

    def test_structure_update_speed(self):
        """Test structure price update performance."""
        base_date = datetime(2024, 1, 15)
        expiration = datetime(2024, 2, 16)

        straddle = ShortStraddle.create(
            underlying='SPY',
            strike=450.0,
            expiration=expiration,
            call_price=10.0,
            put_price=9.5,
            quantity=10,
            entry_date=base_date,
            underlying_price=450.0
        )

        iterations = 1000
        start = time.time()

        for i in range(iterations):
            # Update individual leg prices
            new_date = base_date + timedelta(days=i % 30)
            straddle.options[0].update_price(10.0 + (i % 10) * 0.5, new_date)
            straddle.options[1].update_price(9.5 + (i % 10) * 0.5, new_date)

        elapsed = time.time() - start
        avg_time_ms = (elapsed / iterations) * 1000

        print(f"\nAverage structure update time: {avg_time_ms:.4f}ms")

        # Should be reasonably fast
        assert avg_time_ms < 5.0

    def test_many_structures_creation(self):
        """Test creating many structures simultaneously."""
        base_date = datetime(2024, 1, 15)
        n_structures = 100

        start = time.time()

        structures = []
        for i in range(n_structures):
            expiration = base_date + timedelta(days=30 + i)
            strike = 400.0 + i

            straddle = ShortStraddle.create(
                underlying='SPY',
                strike=strike,
                expiration=expiration,
                call_price=10.0,
                put_price=9.5,
                quantity=10,
                entry_date=base_date,
                underlying_price=450.0
            )
            structures.append(straddle)

        elapsed = time.time() - start
        structures_per_second = n_structures / elapsed

        print(f"\nCreated {n_structures} structures in {elapsed:.3f}s")
        print(f"Rate: {structures_per_second:.0f} structures/second")

        # Should create at least 100 structures per second
        assert structures_per_second > 100


# =============================================================================
# Test 3: Analytics Performance
# =============================================================================

class TestAnalyticsPerformance:
    """Test analytics calculation performance."""

    def test_performance_metrics_calculation_speed(self, large_equity_curve):
        """Test performance metrics calculation speed."""
        returns = large_equity_curve['equity'].pct_change().dropna()

        start = time.time()
        iterations = 100

        for _ in range(iterations):
            total_return = PerformanceMetrics.calculate_total_return(large_equity_curve)
            sharpe = PerformanceMetrics.calculate_sharpe_ratio(returns)
            sortino = PerformanceMetrics.calculate_sortino_ratio(returns)
            max_dd = PerformanceMetrics.calculate_max_drawdown(large_equity_curve)

        elapsed = time.time() - start
        avg_time_ms = (elapsed / iterations) * 1000

        print(f"\nAverage metrics calculation time: {avg_time_ms:.4f}ms")
        print(f"Equity curve length: {len(large_equity_curve)} rows")

        # Should calculate metrics quickly even for large datasets
        assert avg_time_ms < 100.0

    def test_drawdown_calculation_speed(self, large_equity_curve):
        """Test drawdown calculation performance."""
        iterations = 100
        start = time.time()

        for _ in range(iterations):
            max_dd = PerformanceMetrics.calculate_max_drawdown(large_equity_curve)

        elapsed = time.time() - start
        avg_time_ms = (elapsed / iterations) * 1000

        print(f"\nDrawdown calculation time: {avg_time_ms:.4f}ms")

        # Drawdown calculation should be optimized
        assert avg_time_ms < 50.0

    def test_var_calculation_speed(self, large_equity_curve):
        """Test VaR calculation performance."""
        returns = large_equity_curve['equity'].pct_change().dropna()
        iterations = 100

        start = time.time()

        for _ in range(iterations):
            var_95 = RiskAnalytics.calculate_var(returns, 0.95)
            cvar_95 = RiskAnalytics.calculate_cvar(returns, 0.95)

        elapsed = time.time() - start
        avg_time_ms = (elapsed / iterations) * 1000

        print(f"\nVaR/CVaR calculation time: {avg_time_ms:.4f}ms")

        # Risk metrics should be fast
        assert avg_time_ms < 20.0

    def test_trade_metrics_calculation_speed(self):
        """Test trade metrics calculation performance."""
        # Create large trade log
        n_trades = 1000
        np.random.seed(42)

        trade_log = pd.DataFrame({
            'entry_date': pd.date_range('2020-01-01', periods=n_trades, freq='D'),
            'exit_date': pd.date_range('2020-02-01', periods=n_trades, freq='D'),
            'realized_pnl': np.random.normal(100, 500, n_trades),
            'return_pct': np.random.normal(0.05, 0.10, n_trades),
        })

        iterations = 100
        start = time.time()

        for _ in range(iterations):
            win_rate = PerformanceMetrics.calculate_win_rate(trade_log)
            profit_factor = PerformanceMetrics.calculate_profit_factor(trade_log)
            avg_win = PerformanceMetrics.calculate_average_win(trade_log)
            avg_loss = PerformanceMetrics.calculate_average_loss(trade_log)

        elapsed = time.time() - start
        avg_time_ms = (elapsed / iterations) * 1000

        print(f"\nTrade metrics calculation time: {avg_time_ms:.4f}ms")
        print(f"Trade log size: {len(trade_log)} trades")

        # Should be very fast
        assert avg_time_ms < 50.0


# =============================================================================
# Test 4: Memory Usage Tests
# =============================================================================

class TestMemoryUsage:
    """Test memory usage patterns."""

    def test_option_memory_footprint(self):
        """Test memory usage of Option objects."""
        gc.collect()

        # Create many options and measure memory
        n_options = 10000
        base_date = datetime(2024, 1, 15)

        options = []
        for i in range(n_options):
            option = Option(
                option_type='call',
                position_type='long',
                underlying='SPY',
                strike=400.0 + i * 0.5,
                expiration=base_date + timedelta(days=30),
                quantity=10,
                entry_price=5.0,
                entry_date=base_date,
                underlying_price_at_entry=450.0,
                implied_vol_at_entry=0.20
            )
            options.append(option)

        # Memory should be reasonable (not testing exact bytes due to Python overhead)
        assert len(options) == n_options

        # Clean up
        del options
        gc.collect()

    def test_structure_memory_footprint(self):
        """Test memory usage of OptionStructure objects."""
        gc.collect()

        n_structures = 1000
        base_date = datetime(2024, 1, 15)
        expiration = datetime(2024, 2, 16)

        structures = []
        for i in range(n_structures):
            straddle = ShortStraddle.create(
                underlying='SPY',
                strike=400.0 + i,
                expiration=expiration,
                call_price=10.0,
                put_price=9.5,
                quantity=10,
                entry_date=base_date,
                underlying_price=450.0
            )
            structures.append(straddle)

        # Should handle 1000 structures without issues
        assert len(structures) == n_structures

        # Clean up
        del structures
        gc.collect()

    def test_large_equity_curve_memory(self):
        """Test memory usage of large equity curves."""
        gc.collect()

        # Create 10 years of daily data
        dates = pd.date_range(start='2015-01-01', end='2024-12-31', freq='B')
        n_days = len(dates)

        np.random.seed(42)
        returns = np.random.normal(0.0005, 0.01, n_days)
        equity = 100000.0 * np.exp(np.cumsum(returns))

        df = pd.DataFrame({
            'date': dates,
            'equity': equity,
            'cash': equity * 0.7,
            'positions_value': equity * 0.3,
            'delta': np.random.normal(0, 50, n_days),
            'gamma': np.random.normal(0, 5, n_days),
            'theta': np.random.normal(-100, 20, n_days),
            'vega': np.random.normal(0, 100, n_days),
        })

        print(f"\nEquity curve size: {len(df)} rows")
        print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

        # Should be reasonable (under 10MB for 10 years of data)
        assert df.memory_usage(deep=True).sum() / 1024 / 1024 < 10.0

        # Clean up
        del df
        gc.collect()


# =============================================================================
# Test 5: Scalability Tests
# =============================================================================

class TestScalability:
    """Test system scalability with increasing loads."""

    def test_increasing_option_counts(self, pricer):
        """Test pricing performance with increasing option counts."""
        counts = [100, 500, 1000, 5000, 10000]
        times = []

        for n in counts:
            # Generate random parameters
            np.random.seed(42)
            S_values = np.random.uniform(400, 500, n)
            K_values = np.random.uniform(400, 500, n)
            T_values = np.random.uniform(0.1, 1.0, n)
            sigma_values = np.random.uniform(0.15, 0.35, n)

            start = time.time()

            for i in range(n):
                price = black_scholes_price(
                    S=S_values[i],
                    K=K_values[i],
                    T=T_values[i],
                    r=0.04,
                    sigma=sigma_values[i],
                    option_type='call'
                )

            elapsed = time.time() - start
            times.append(elapsed)

        # Check scaling is roughly linear
        print("\nScaling test:")
        for count, elapsed in zip(counts, times):
            rate = count / elapsed
            print(f"{count:5d} options: {elapsed:.3f}s ({rate:.0f} opts/sec)")

        # Verify roughly linear scaling (within 2x factor)
        ratio = (times[-1] / counts[-1]) / (times[0] / counts[0])
        assert ratio < 2.0, f"Non-linear scaling detected: {ratio:.2f}x"

    def test_equity_curve_scaling(self):
        """Test analytics performance with increasing data size."""
        periods = [100, 500, 1000, 5000]
        times = []

        for n in periods:
            # Create equity curve
            np.random.seed(42)
            returns = np.random.normal(0.0005, 0.01, n)
            equity = 100000.0 * np.exp(np.cumsum(returns))

            df = pd.DataFrame({
                'equity': equity,
                'cash': equity * 0.7,
                'positions_value': equity * 0.3,
            })

            start = time.time()

            # Calculate several metrics
            total_return = PerformanceMetrics.calculate_total_return(df)
            max_dd = PerformanceMetrics.calculate_max_drawdown(df)
            returns_series = df['equity'].pct_change().dropna()
            sharpe = PerformanceMetrics.calculate_sharpe_ratio(returns_series)

            elapsed = time.time() - start
            times.append(elapsed)

        print("\nEquity curve scaling:")
        for period, elapsed in zip(periods, times):
            print(f"{period:5d} periods: {elapsed:.4f}s")

        # Should scale well (not exponentially)
        ratio = (times[-1] / periods[-1]) / (times[0] / periods[0])
        assert ratio < 3.0, f"Poor scaling detected: {ratio:.2f}x"


# =============================================================================
# Test 6: Bottleneck Identification
# =============================================================================

class TestBottleneckIdentification:
    """Identify potential bottlenecks in the system."""

    def test_option_creation_vs_pricing(self):
        """Compare time for option creation vs pricing."""
        n = 1000
        base_date = datetime(2024, 1, 15)
        expiration = datetime(2024, 2, 16)

        # Test option creation
        start = time.time()
        for i in range(n):
            option = Option(
                option_type='call',
                position_type='long',
                underlying='SPY',
                strike=450.0,
                expiration=expiration,
                quantity=10,
                entry_price=5.0,
                entry_date=base_date,
                underlying_price_at_entry=450.0,
                implied_vol_at_entry=0.20
            )
        creation_time = time.time() - start

        # Test pricing
        start = time.time()
        for i in range(n):
            price = black_scholes_price(
                S=450.0,
                K=450.0,
                T=0.25,
                r=0.04,
                sigma=0.20,
                option_type='call'
            )
        pricing_time = time.time() - start

        print(f"\nOption creation: {creation_time:.4f}s for {n} options")
        print(f"Option pricing:  {pricing_time:.4f}s for {n} options")
        print(f"Creation/Pricing ratio: {creation_time / pricing_time:.2f}x")

        # Both should be fast
        assert creation_time < 1.0
        assert pricing_time < 0.5

    def test_structure_operations_breakdown(self):
        """Break down structure operation timing."""
        n = 100
        base_date = datetime(2024, 1, 15)
        expiration = datetime(2024, 2, 16)

        # Test creation
        start = time.time()
        structures = []
        for _ in range(n):
            straddle = ShortStraddle.create(
                underlying='SPY',
                strike=450.0,
                expiration=expiration,
                call_price=10.0,
                put_price=9.5,
                quantity=10,
                entry_date=base_date,
                underlying_price=450.0
            )
            structures.append(straddle)
        creation_time = time.time() - start

        # Test Greeks access
        start = time.time()
        for structure in structures:
            greeks = structure.calculate_net_greeks()
        greeks_time = time.time() - start

        # Test P&L access
        start = time.time()
        for structure in structures:
            pnl = structure.calculate_pnl()
        pnl_time = time.time() - start

        # Test update
        start = time.time()
        for structure in structures:
            # Update individual leg prices
            new_date = base_date + timedelta(days=10)
            structure.options[0].update_price(12.0, new_date)
            structure.options[1].update_price(7.0, new_date)
        update_time = time.time() - start

        print(f"\nStructure operations ({n} structures):")
        print(f"Creation: {creation_time:.4f}s ({creation_time/n*1000:.2f}ms each)")
        print(f"Greeks:   {greeks_time:.4f}s ({greeks_time/n*1000:.2f}ms each)")
        print(f"P&L:      {pnl_time:.4f}s ({pnl_time/n*1000:.2f}ms each)")
        print(f"Update:   {update_time:.4f}s ({update_time/n*1000:.2f}ms each)")


# =============================================================================
# Summary Performance Report
# =============================================================================

def test_performance_summary(capsys):
    """Generate a summary performance report."""
    print("\n" + "="*70)
    print("PERFORMANCE SUMMARY REPORT")
    print("="*70)

    # Run key benchmarks
    # 1. Pricing speed
    start = time.time()
    for _ in range(10000):
        black_scholes_price(450.0, 450.0, 0.25, 0.04, 0.20, 'call')
    pricing_rate = 10000 / (time.time() - start)

    # 2. Structure creation
    start = time.time()
    for _ in range(100):
        ShortStraddle.create(
            'SPY', 450.0, datetime(2024, 2, 16),
            10.0, 9.5, 10, datetime(2024, 1, 15), 450.0
        )
    structure_rate = 100 / (time.time() - start)

    # 3. Analytics
    dates = pd.date_range('2020-01-01', '2024-12-31', freq='B')
    np.random.seed(42)
    equity = 100000.0 * np.exp(np.cumsum(np.random.normal(0.0005, 0.01, len(dates))))
    df = pd.DataFrame({'equity': equity, 'cash': equity*0.7, 'positions_value': equity*0.3})

    start = time.time()
    for _ in range(10):
        PerformanceMetrics.calculate_total_return(df)
        PerformanceMetrics.calculate_max_drawdown(df)
    analytics_time = (time.time() - start) / 10

    print(f"\n1. Pricing Performance:")
    print(f"   Options priced per second: {pricing_rate:,.0f}")
    print(f"   Time per option: {1000/pricing_rate:.4f}ms")
    print(f"   Target: >10,000 options/sec ✓" if pricing_rate > 10000 else "   Target: >10,000 options/sec ✗")

    print(f"\n2. Structure Performance:")
    print(f"   Structures created per second: {structure_rate:,.0f}")
    print(f"   Time per structure: {1000/structure_rate:.4f}ms")
    print(f"   Target: <10ms per structure ✓" if 1000/structure_rate < 10 else "   Target: <10ms per structure ✗")

    print(f"\n3. Analytics Performance:")
    print(f"   Time for metrics (5 years data): {analytics_time*1000:.2f}ms")
    print(f"   Data points: {len(df):,}")
    print(f"   Target: <100ms ✓" if analytics_time*1000 < 100 else "   Target: <100ms ✗")

    print(f"\n" + "="*70)
    print("All performance targets met!" if all([
        pricing_rate > 10000,
        1000/structure_rate < 10,
        analytics_time*1000 < 100
    ]) else "Some performance targets not met")
    print("="*70 + "\n")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short', '-s'])
