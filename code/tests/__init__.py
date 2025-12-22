"""
Test Suite for Options Backtester

This package contains comprehensive unit tests for the backtesting system,
organized by module.

Test modules:
    - test_pricing: Black-Scholes pricing and Greeks validation
    - test_option: Option class functionality and P&L calculations
    - test_option_structure: OptionStructure class and multi-leg positions
    - test_strategy: Strategy base class for portfolio management
    - test_conditions: Condition helper utilities for strategy logic

Run all tests:
    pytest tests/ -v

Run specific test file:
    pytest tests/test_strategy.py -v

Run with coverage:
    pytest tests/ --cov=backtester --cov-report=term-missing
"""
