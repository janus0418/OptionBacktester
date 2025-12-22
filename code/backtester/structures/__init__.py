"""
Concrete Option Structures Module

This module provides pre-built option structures for common trading strategies.
All structures inherit from the OptionStructure base class and provide
convenient factory methods, pre-calculated max profit/loss, and named breakevens.

Available Structures:

Straddles:
    - LongStraddle: Buy call + put at same strike
    - ShortStraddle: Sell call + put at same strike

Strangles:
    - LongStrangle: Buy OTM call + OTM put
    - ShortStrangle: Sell OTM call + OTM put

Spreads:
    - BullCallSpread: Buy lower call + sell higher call (debit)
    - BearPutSpread: Buy higher put + sell lower put (debit)
    - BullPutSpread: Sell higher put + buy lower put (credit)
    - BearCallSpread: Sell lower call + buy higher call (credit)

Condors:
    - IronCondor: Bull put spread + bear call spread
    - IronButterfly: ATM straddle + OTM strangle protection

Usage:
    from backtester.structures import ShortStraddle, IronCondor

    # Create short straddle
    straddle = ShortStraddle.create(
        underlying='SPY',
        strike=450.0,
        expiration=datetime(2024, 3, 15),
        call_price=5.50,
        put_price=5.25,
        quantity=10,
        entry_date=datetime(2024, 3, 1),
        underlying_price=450.0
    )

    # Access structure-specific properties
    print(f"Max Profit: ${straddle.max_profit:,.2f}")
    print(f"Breakevens: {straddle.lower_breakeven:.2f}, {straddle.upper_breakeven:.2f}")
"""

from backtester.structures.straddle import LongStraddle, ShortStraddle
from backtester.structures.strangle import LongStrangle, ShortStrangle
from backtester.structures.spread import (
    BullCallSpread,
    BearPutSpread,
    BullPutSpread,
    BearCallSpread,
)
from backtester.structures.condor import IronCondor, IronButterfly

__all__ = [
    # Straddles
    'LongStraddle',
    'ShortStraddle',

    # Strangles
    'LongStrangle',
    'ShortStrangle',

    # Spreads
    'BullCallSpread',
    'BearPutSpread',
    'BullPutSpread',
    'BearCallSpread',

    # Condors
    'IronCondor',
    'IronButterfly',
]
