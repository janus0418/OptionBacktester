"""
Strategy Templates - Pre-built Strategy Configurations

This module provides ready-to-use strategy templates that leverage the StrategyBuilder API.
Each template encapsulates a complete trading strategy with sensible defaults that can be
customized through parameters.

Templates Available:
    HighIVStraddleTemplate: Premium selling on elevated implied volatility
    IronCondorTemplate: Range-bound income strategy with defined risk
    WheelStrategyTemplate: Cash-secured puts cycling to covered calls
    EarningsStraddleTemplate: Pre-earnings volatility capture
    TrendFollowingTemplate: Directional momentum with options

Usage:
    from backtester.strategies.strategy_templates import HighIVStraddleTemplate

    # Create with defaults
    strategy = HighIVStraddleTemplate.create()

    # Or customize parameters
    strategy = HighIVStraddleTemplate.create(
        underlying="QQQ",
        iv_threshold=80,
        profit_target_pct=0.40,
        dte_range=(20, 35)
    )

References:
    - Tastytrade research on premium selling
    - CBOE volatility studies
    - Academic literature on options strategies
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple, List

from backtester.strategies.strategy_builder import (
    StrategyBuilder,
    BuiltStrategy,
    # Entry conditions
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
    # Exit conditions
    profit_target as make_profit_target,
    stop_loss as make_stop_loss,
    fixed_stop_loss,
    trailing_stop,
    holding_period,
    expiration_approaching,
    # Structures
    short_straddle,
    long_straddle,
    short_strangle,
    long_strangle,
    iron_condor,
    bull_call_spread,
    bear_put_spread,
    # Position sizing
    fixed_contracts,
    risk_percent,
    capital_percent,
    delta_target,
    premium_target,
    # Condition classes for composition
    Condition,
    AlwaysTrue,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Template Configuration Classes
# =============================================================================


@dataclass
class HighIVStraddleConfig:
    """Configuration for High IV Straddle strategy."""

    underlying: str = "SPY"
    iv_threshold: float = 70.0
    profit_target_pct: float = 0.50
    stop_loss_multiple: float = 2.0
    dte_range: Tuple[int, int] = (25, 45)
    exit_dte: int = 7
    max_positions: int = 3
    risk_pct: float = 0.02
    initial_capital: float = 100_000.0


@dataclass
class IronCondorConfig:
    """Configuration for Iron Condor strategy."""

    underlying: str = "SPY"
    iv_threshold: float = 50.0
    short_delta: float = 0.16
    wing_width_pct: float = 0.03
    profit_target_pct: float = 0.50
    stop_loss_multiple: float = 2.0
    dte_range: Tuple[int, int] = (30, 60)
    exit_dte: int = 14
    max_positions: int = 2
    capital_pct: float = 0.10
    initial_capital: float = 100_000.0


@dataclass
class WheelConfig:
    """Configuration for Wheel strategy."""

    underlying: str = "SPY"
    put_delta: float = 0.30
    profit_target_pct: float = 0.50
    dte_range: Tuple[int, int] = (30, 45)
    exit_dte: int = 7
    max_positions: int = 1
    capital_pct: float = 0.20
    initial_capital: float = 100_000.0


@dataclass
class EarningsStraddleConfig:
    """Configuration for Earnings Straddle strategy."""

    underlying: str = "SPY"
    days_before_earnings: int = 5
    profit_target_pct: float = 0.25
    stop_loss_pct: float = 0.50
    dte_range: Tuple[int, int] = (7, 21)
    max_positions: int = 5
    risk_pct: float = 0.01
    initial_capital: float = 100_000.0


@dataclass
class TrendFollowingConfig:
    """Configuration for Trend Following strategy."""

    underlying: str = "SPY"
    vix_threshold_low: float = 15.0
    vix_threshold_high: float = 25.0
    profit_target_pct: float = 1.0
    stop_loss_pct: float = 0.50
    dte_range: Tuple[int, int] = (45, 90)
    exit_dte: int = 21
    max_positions: int = 2
    risk_pct: float = 0.03
    initial_capital: float = 100_000.0


# =============================================================================
# Strategy Templates
# =============================================================================


class HighIVStraddleTemplate:
    """
    High IV Short Straddle Strategy Template.

    This strategy sells ATM straddles when implied volatility is elevated,
    betting that IV will revert to the mean and the underlying will stay
    within the breakeven range.

    Entry Criteria:
        - IV rank above threshold (default 70%)
        - DTE within target range (default 25-45 days)
        - Below max position limit

    Exit Criteria:
        - Profit target reached (default 50% of premium)
        - Stop loss triggered (default 200% of premium)
        - DTE below minimum (default 7 days)

    Risk Management:
        - Position sized by risk percentage of capital
        - Max concurrent positions limit
        - Time-based exit before expiration

    Research Basis:
        - Tastytrade studies show 45 DTE optimal for premium decay
        - IV rank >70 historically leads to IV contraction
        - Managing winners at 50% improves risk-adjusted returns
    """

    @staticmethod
    def create(
        underlying: str = "SPY",
        iv_threshold: float = 70.0,
        profit_target_pct: float = 0.50,
        stop_loss_multiple: float = 2.0,
        dte_range: Tuple[int, int] = (25, 45),
        exit_dte: int = 7,
        max_positions: int = 3,
        risk_pct: float = 0.02,
        initial_capital: float = 100_000.0,
    ) -> BuiltStrategy:
        """
        Create a High IV Short Straddle strategy.

        Args:
            underlying: Ticker symbol (default: SPY)
            iv_threshold: IV rank threshold for entry (default: 70)
            profit_target_pct: Profit target as decimal (default: 0.50 = 50%)
            stop_loss_multiple: Stop loss as multiple of premium (default: 2.0 = 200%)
            dte_range: (min, max) DTE for entry (default: (25, 45))
            exit_dte: Exit when DTE falls below this (default: 7)
            max_positions: Maximum concurrent positions (default: 3)
            risk_pct: Risk per trade as decimal (default: 0.02 = 2%)
            initial_capital: Starting capital (default: 100,000)

        Returns:
            BuiltStrategy ready for backtesting
        """
        target_dte = (dte_range[0] + dte_range[1]) // 2

        strategy = (
            StrategyBuilder()
            .name(f"High IV Short Straddle - {underlying}")
            .description(
                f"Sell ATM straddles when IV rank > {iv_threshold}%, "
                f"target {profit_target_pct:.0%} profit, {stop_loss_multiple:.0%} stop"
            )
            .underlying(underlying)
            .initial_capital(initial_capital)
            .entry_condition(
                iv_rank_above(iv_threshold)
                & dte_between(dte_range[0], dte_range[1])
                & max_open_positions(max_positions)
            )
            .structure(short_straddle(dte=target_dte))
            .exit_condition(
                make_profit_target(profit_target_pct)
                | make_stop_loss(stop_loss_multiple)
                | dte_below(exit_dte)
            )
            .position_size(risk_percent(risk_pct))
            .max_positions(max_positions)
            .build()
        )

        logger.info(f"Created HighIVStraddleTemplate for {underlying}")
        return strategy

    @staticmethod
    def create_from_config(config: HighIVStraddleConfig) -> BuiltStrategy:
        """Create strategy from configuration object."""
        return HighIVStraddleTemplate.create(
            underlying=config.underlying,
            iv_threshold=config.iv_threshold,
            profit_target_pct=config.profit_target_pct,
            stop_loss_multiple=config.stop_loss_multiple,
            dte_range=config.dte_range,
            exit_dte=config.exit_dte,
            max_positions=config.max_positions,
            risk_pct=config.risk_pct,
            initial_capital=config.initial_capital,
        )


class IronCondorTemplate:
    """
    Iron Condor Strategy Template.

    This strategy sells iron condors in neutral to slightly elevated IV
    environments, profiting from time decay while defining maximum risk
    with protective wings.

    Entry Criteria:
        - IV rank above threshold (default 50%)
        - DTE within target range (default 30-60 days)
        - Below max position limit

    Exit Criteria:
        - Profit target reached (default 50% of credit)
        - Stop loss triggered (default 200% of credit)
        - DTE below minimum (default 14 days)

    Structure:
        - Short call spread (OTM)
        - Short put spread (OTM)
        - Wings at configurable width

    Risk Management:
        - Position sized by capital percentage
        - Max loss is width minus credit received
        - Time-based exit for gamma risk management

    Research Basis:
        - Iron condors benefit from theta decay and IV contraction
        - 30-45 DTE balances theta/gamma ratio
        - Managing at 50% profit improves consistency
    """

    @staticmethod
    def create(
        underlying: str = "SPY",
        iv_threshold: float = 50.0,
        short_delta: float = 0.16,
        wing_width_pct: float = 0.03,
        profit_target_pct: float = 0.50,
        stop_loss_multiple: float = 2.0,
        dte_range: Tuple[int, int] = (30, 60),
        exit_dte: int = 14,
        max_positions: int = 2,
        capital_pct: float = 0.10,
        initial_capital: float = 100_000.0,
    ) -> BuiltStrategy:
        """
        Create an Iron Condor strategy.

        Args:
            underlying: Ticker symbol (default: SPY)
            iv_threshold: IV rank threshold for entry (default: 50)
            short_delta: Target delta for short strikes (default: 0.16)
            wing_width_pct: Wing width as % of spot (default: 0.03 = 3%)
            profit_target_pct: Profit target as decimal (default: 0.50)
            stop_loss_multiple: Stop loss as multiple of credit (default: 2.0)
            dte_range: (min, max) DTE for entry (default: (30, 60))
            exit_dte: Exit when DTE falls below this (default: 14)
            max_positions: Maximum concurrent positions (default: 2)
            capital_pct: Capital per trade as decimal (default: 0.10)
            initial_capital: Starting capital (default: 100,000)

        Returns:
            BuiltStrategy ready for backtesting
        """
        target_dte = (dte_range[0] + dte_range[1]) // 2

        strategy = (
            StrategyBuilder()
            .name(f"Iron Condor - {underlying}")
            .description(
                f"Sell iron condors with {short_delta:.0%} delta shorts, "
                f"target {profit_target_pct:.0%} profit"
            )
            .underlying(underlying)
            .initial_capital(initial_capital)
            .entry_condition(
                iv_rank_above(iv_threshold)
                & dte_between(dte_range[0], dte_range[1])
                & max_open_positions(max_positions)
            )
            .structure(
                iron_condor(
                    dte=target_dte,
                    width=wing_width_pct * 450,  # Approximate SPY price
                    wing_width=wing_width_pct * 450 * 0.5,
                )
            )
            .exit_condition(
                make_profit_target(profit_target_pct)
                | make_stop_loss(stop_loss_multiple)
                | dte_below(exit_dte)
            )
            .position_size(capital_percent(capital_pct))
            .max_positions(max_positions)
            .build()
        )

        logger.info(f"Created IronCondorTemplate for {underlying}")
        return strategy

    @staticmethod
    def create_from_config(config: IronCondorConfig) -> BuiltStrategy:
        """Create strategy from configuration object."""
        return IronCondorTemplate.create(
            underlying=config.underlying,
            iv_threshold=config.iv_threshold,
            short_delta=config.short_delta,
            wing_width_pct=config.wing_width_pct,
            profit_target_pct=config.profit_target_pct,
            stop_loss_multiple=config.stop_loss_multiple,
            dte_range=config.dte_range,
            exit_dte=config.exit_dte,
            max_positions=config.max_positions,
            capital_pct=config.capital_pct,
            initial_capital=config.initial_capital,
        )


class WheelStrategyTemplate:
    """
    Wheel Strategy Template.

    The Wheel is a systematic approach to income generation that cycles
    between cash-secured puts and covered calls. When puts are assigned,
    the position converts to covered calls on the acquired shares.

    Entry Criteria:
        - DTE within target range (default 30-45 days)
        - No existing position on the underlying
        - Sufficient capital for cash-secured put

    Exit Criteria:
        - Profit target reached (default 50% of premium)
        - DTE below minimum (default 7 days)
        - Assignment (handled separately)

    Wheel Cycle:
        1. Sell cash-secured put (OTM, ~30 delta)
        2. If assigned: sell covered call on shares
        3. If called away: return to step 1
        4. If not assigned: collect premium, repeat

    Risk Management:
        - Position sized by available margin
        - One underlying at a time
        - Rolling before expiration

    Research Basis:
        - Wheel generates income in sideways/bullish markets
        - 30-delta puts balance premium vs assignment risk
        - Systematic approach removes emotional decisions
    """

    @staticmethod
    def create(
        underlying: str = "SPY",
        put_delta: float = 0.30,
        profit_target_pct: float = 0.50,
        dte_range: Tuple[int, int] = (30, 45),
        exit_dte: int = 7,
        max_positions: int = 1,
        capital_pct: float = 0.20,
        initial_capital: float = 100_000.0,
    ) -> BuiltStrategy:
        """
        Create a Wheel strategy (cash-secured put phase).

        Note: This template implements the put-selling phase of the wheel.
        Assignment handling and covered call phase require additional logic.

        Args:
            underlying: Ticker symbol (default: SPY)
            put_delta: Target delta for short puts (default: 0.30)
            profit_target_pct: Profit target as decimal (default: 0.50)
            dte_range: (min, max) DTE for entry (default: (30, 45))
            exit_dte: Exit when DTE falls below this (default: 7)
            max_positions: Maximum concurrent positions (default: 1)
            capital_pct: Capital per trade as decimal (default: 0.20)
            initial_capital: Starting capital (default: 100,000)

        Returns:
            BuiltStrategy ready for backtesting
        """
        target_dte = (dte_range[0] + dte_range[1]) // 2

        # Wheel uses short strangle with put-only emphasis
        # In practice, this is a cash-secured put
        strategy = (
            StrategyBuilder()
            .name(f"Wheel Strategy - {underlying}")
            .description(
                f"Cash-secured puts at {put_delta:.0%} delta, "
                f"target {profit_target_pct:.0%} profit"
            )
            .underlying(underlying)
            .initial_capital(initial_capital)
            .entry_condition(
                dte_between(dte_range[0], dte_range[1]) & no_open_positions(underlying)
            )
            .structure(
                bear_put_spread(  # Approximating cash-secured put
                    dte=target_dte,
                    width=10.0,  # Small width to approximate naked put
                )
            )
            .exit_condition(make_profit_target(profit_target_pct) | dte_below(exit_dte))
            .position_size(capital_percent(capital_pct))
            .max_positions(max_positions)
            .build()
        )

        logger.info(f"Created WheelStrategyTemplate for {underlying}")
        return strategy

    @staticmethod
    def create_from_config(config: WheelConfig) -> BuiltStrategy:
        """Create strategy from configuration object."""
        return WheelStrategyTemplate.create(
            underlying=config.underlying,
            put_delta=config.put_delta,
            profit_target_pct=config.profit_target_pct,
            dte_range=config.dte_range,
            exit_dte=config.exit_dte,
            max_positions=config.max_positions,
            capital_pct=config.capital_pct,
            initial_capital=config.initial_capital,
        )


class EarningsStraddleTemplate:
    """
    Earnings Straddle Strategy Template.

    This strategy buys straddles before earnings announcements to profit
    from the expected volatility expansion and large price moves.

    Entry Criteria:
        - N days before earnings (configurable)
        - DTE within target range (default 7-21 days)
        - Below max position limit

    Exit Criteria:
        - Profit target reached (default 25%)
        - Stop loss triggered (default 50%)
        - After earnings release (holding period)

    Strategy Logic:
        - Buy ATM straddle before earnings
        - Profit if stock moves more than straddle cost
        - Exit after earnings move, win or lose

    Risk Management:
        - Small position size (high risk trade)
        - Strict stop loss
        - Limited holding period

    Research Basis:
        - Earnings often cause outsized moves
        - IV typically inflated pre-earnings (IV crush risk)
        - Net profitable if big movers offset IV crush losers
    """

    @staticmethod
    def create(
        underlying: str = "SPY",
        days_before_earnings: int = 5,
        profit_target_pct: float = 0.25,
        stop_loss_pct: float = 0.50,
        dte_range: Tuple[int, int] = (7, 21),
        max_positions: int = 5,
        risk_pct: float = 0.01,
        initial_capital: float = 100_000.0,
    ) -> BuiltStrategy:
        """
        Create an Earnings Straddle strategy.

        Note: Actual earnings date detection requires external data.
        This template provides the structure for earnings plays.

        Args:
            underlying: Ticker symbol (default: SPY)
            days_before_earnings: Days before earnings to enter (default: 5)
            profit_target_pct: Profit target as decimal (default: 0.25)
            stop_loss_pct: Stop loss as decimal (default: 0.50)
            dte_range: (min, max) DTE for entry (default: (7, 21))
            max_positions: Maximum concurrent positions (default: 5)
            risk_pct: Risk per trade as decimal (default: 0.01)
            initial_capital: Starting capital (default: 100,000)

        Returns:
            BuiltStrategy ready for backtesting
        """
        target_dte = (dte_range[0] + dte_range[1]) // 2

        strategy = (
            StrategyBuilder()
            .name(f"Earnings Straddle - {underlying}")
            .description(
                f"Buy straddles {days_before_earnings} days before earnings, "
                f"target {profit_target_pct:.0%} profit"
            )
            .underlying(underlying)
            .initial_capital(initial_capital)
            .entry_condition(
                dte_between(dte_range[0], dte_range[1])
                & max_open_positions(max_positions)
            )
            .structure(long_straddle(dte=target_dte))
            .exit_condition(
                make_profit_target(profit_target_pct)
                | make_stop_loss(stop_loss_pct)
                | holding_period(days_before_earnings + 2)
            )
            .position_size(risk_percent(risk_pct))
            .max_positions(max_positions)
            .build()
        )

        logger.info(f"Created EarningsStraddleTemplate for {underlying}")
        return strategy

    @staticmethod
    def create_from_config(config: EarningsStraddleConfig) -> BuiltStrategy:
        """Create strategy from configuration object."""
        return EarningsStraddleTemplate.create(
            underlying=config.underlying,
            days_before_earnings=config.days_before_earnings,
            profit_target_pct=config.profit_target_pct,
            stop_loss_pct=config.stop_loss_pct,
            dte_range=config.dte_range,
            max_positions=config.max_positions,
            risk_pct=config.risk_pct,
            initial_capital=config.initial_capital,
        )


class TrendFollowingTemplate:
    """
    Trend Following Strategy Template.

    This strategy uses options for directional bets based on trend signals,
    buying call spreads in uptrends and put spreads in downtrends.

    Entry Criteria:
        - VIX within target range (calm to moderate volatility)
        - DTE within target range (default 45-90 days)
        - Below max position limit

    Exit Criteria:
        - Profit target reached (default 100%)
        - Stop loss triggered (default 50%)
        - DTE below minimum (default 21 days)

    Strategy Logic:
        - Buy debit spreads in direction of trend
        - Longer DTE for trend to develop
        - Wide spreads for leverage

    Risk Management:
        - Defined risk via spread structure
        - Position sized by risk percentage
        - Time-based exit for theta decay

    Research Basis:
        - Trends tend to persist in medium-term
        - Lower VIX = cheaper options for directional bets
        - Debit spreads reduce cost vs naked options
    """

    @staticmethod
    def create(
        underlying: str = "SPY",
        vix_threshold_low: float = 15.0,
        vix_threshold_high: float = 25.0,
        profit_target_pct: float = 1.0,
        stop_loss_pct: float = 0.50,
        dte_range: Tuple[int, int] = (45, 90),
        exit_dte: int = 21,
        max_positions: int = 2,
        risk_pct: float = 0.03,
        initial_capital: float = 100_000.0,
        direction: str = "bullish",
    ) -> BuiltStrategy:
        """
        Create a Trend Following strategy.

        Args:
            underlying: Ticker symbol (default: SPY)
            vix_threshold_low: Min VIX for entry (default: 15)
            vix_threshold_high: Max VIX for entry (default: 25)
            profit_target_pct: Profit target as decimal (default: 1.0 = 100%)
            stop_loss_pct: Stop loss as decimal (default: 0.50)
            dte_range: (min, max) DTE for entry (default: (45, 90))
            exit_dte: Exit when DTE falls below this (default: 21)
            max_positions: Maximum concurrent positions (default: 2)
            risk_pct: Risk per trade as decimal (default: 0.03)
            initial_capital: Starting capital (default: 100,000)
            direction: "bullish" or "bearish" (default: "bullish")

        Returns:
            BuiltStrategy ready for backtesting
        """
        target_dte = (dte_range[0] + dte_range[1]) // 2

        # Choose structure based on direction
        if direction.lower() == "bullish":
            structure = bull_call_spread(dte=target_dte, width=10.0)
            direction_desc = "bullish"
        else:
            structure = bear_put_spread(dte=target_dte, width=10.0)
            direction_desc = "bearish"

        strategy = (
            StrategyBuilder()
            .name(f"Trend Following ({direction_desc}) - {underlying}")
            .description(
                f"{direction_desc.title()} spread when VIX {vix_threshold_low}-{vix_threshold_high}, "
                f"target {profit_target_pct:.0%} profit"
            )
            .underlying(underlying)
            .initial_capital(initial_capital)
            .entry_condition(
                vix_between(vix_threshold_low, vix_threshold_high)
                & dte_between(dte_range[0], dte_range[1])
                & max_open_positions(max_positions)
            )
            .structure(structure)
            .exit_condition(
                make_profit_target(profit_target_pct)
                | make_stop_loss(stop_loss_pct)
                | dte_below(exit_dte)
            )
            .position_size(risk_percent(risk_pct))
            .max_positions(max_positions)
            .build()
        )

        logger.info(f"Created TrendFollowingTemplate ({direction}) for {underlying}")
        return strategy

    @staticmethod
    def create_from_config(config: TrendFollowingConfig) -> BuiltStrategy:
        """Create strategy from configuration object."""
        return TrendFollowingTemplate.create(
            underlying=config.underlying,
            vix_threshold_low=config.vix_threshold_low,
            vix_threshold_high=config.vix_threshold_high,
            profit_target_pct=config.profit_target_pct,
            stop_loss_pct=config.stop_loss_pct,
            dte_range=config.dte_range,
            exit_dte=config.exit_dte,
            max_positions=config.max_positions,
            risk_pct=config.risk_pct,
            initial_capital=config.initial_capital,
        )


# =============================================================================
# Template Registry
# =============================================================================


class TemplateRegistry:
    """
    Registry for discovering and instantiating strategy templates.

    Provides a centralized way to:
    - List available templates
    - Get template descriptions
    - Create strategies by name
    """

    _templates = {
        "high_iv_straddle": HighIVStraddleTemplate,
        "iron_condor": IronCondorTemplate,
        "wheel": WheelStrategyTemplate,
        "earnings_straddle": EarningsStraddleTemplate,
        "trend_following": TrendFollowingTemplate,
    }

    _descriptions = {
        "high_iv_straddle": "Sell ATM straddles when IV rank is elevated",
        "iron_condor": "Sell iron condors for range-bound income",
        "wheel": "Cash-secured puts cycling to covered calls",
        "earnings_straddle": "Buy straddles before earnings announcements",
        "trend_following": "Directional spreads based on trend signals",
    }

    @classmethod
    def list_templates(cls) -> List[str]:
        """List all available template names."""
        return list(cls._templates.keys())

    @classmethod
    def get_description(cls, name: str) -> str:
        """Get description for a template."""
        return cls._descriptions.get(name, "No description available")

    @classmethod
    def get_template(cls, name: str):
        """Get template class by name."""
        if name not in cls._templates:
            available = ", ".join(cls._templates.keys())
            raise ValueError(f"Unknown template: {name}. Available: {available}")
        return cls._templates[name]

    @classmethod
    def create(cls, name: str, **kwargs) -> BuiltStrategy:
        """Create a strategy from template name with custom parameters."""
        template = cls.get_template(name)
        return template.create(**kwargs)

    @classmethod
    def describe_all(cls) -> dict:
        """Get all templates with descriptions."""
        return {
            name: {
                "class": cls._templates[name].__name__,
                "description": cls._descriptions[name],
            }
            for name in cls._templates
        }


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Templates
    "HighIVStraddleTemplate",
    "IronCondorTemplate",
    "WheelStrategyTemplate",
    "EarningsStraddleTemplate",
    "TrendFollowingTemplate",
    # Configuration Classes
    "HighIVStraddleConfig",
    "IronCondorConfig",
    "WheelConfig",
    "EarningsStraddleConfig",
    "TrendFollowingConfig",
    # Registry
    "TemplateRegistry",
]
