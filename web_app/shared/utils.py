"""
Shared utilities for Options Strategy Backtester web app.

Contains:
- Configuration constants
- Database connection helpers
- Backtest execution
- Session state management
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple, List
import logging
import os

import streamlit as st
import pandas as pd
import numpy as np

# Add backtester to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "code"))

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_DB_PATH = str(Path(__file__).parent.parent.parent / "dolt_data" / "options")

STRATEGY_INFO = {
    "Short Straddle (High IV)": {
        "description": "Sell ATM call and put when IV rank is high. Profits from time decay and IV contraction.",
        "risk": "High",
        "ideal_conditions": "High IV environment, range-bound market",
        "params": [
            "iv_rank_threshold",
            "profit_target_pct",
            "stop_loss_pct",
            "dte_target",
        ],
    },
    "Iron Condor": {
        "description": "Sell OTM put spread and call spread. Limited risk, profits from low volatility.",
        "risk": "Medium",
        "ideal_conditions": "Low to moderate IV, range-bound market",
        "params": [
            "delta_target",
            "wing_width",
            "profit_target_pct",
            "stop_loss_pct",
            "dte_target",
        ],
    },
    "Volatility Regime": {
        "description": "Adaptive strategy that adjusts position sizing based on VIX levels.",
        "risk": "Variable",
        "ideal_conditions": "Any market condition",
        "params": [
            "low_vol_threshold",
            "high_vol_threshold",
            "profit_target_pct",
            "stop_loss_pct",
        ],
    },
}


# =============================================================================
# Session State Management
# =============================================================================


def init_session_state() -> None:
    """Initialize session state with default values."""
    defaults = {
        "results": None,  # Current backtest results
        "saved_results": {},  # Dict of saved results {name: results}
        "db_path": DEFAULT_DB_PATH,
        "is_connected": False,
        "connection_status": "",
        "comparison_results": [],  # List of results for comparison
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_results() -> Optional[Dict[str, Any]]:
    """Get current backtest results from session state."""
    return st.session_state.get("results")


def set_results(results: Dict[str, Any]) -> None:
    """Set current backtest results in session state."""
    st.session_state.results = results


def clear_results() -> None:
    """Clear current backtest results."""
    st.session_state.results = None


def save_result(name: str, results: Dict[str, Any]) -> None:
    """Save a backtest result with a name."""
    if "saved_results" not in st.session_state:
        st.session_state.saved_results = {}
    st.session_state.saved_results[name] = results


def delete_saved_result(name: str) -> None:
    """Delete a saved backtest result."""
    if "saved_results" in st.session_state and name in st.session_state.saved_results:
        del st.session_state.saved_results[name]


def get_all_saved_results() -> Dict[str, Dict[str, Any]]:
    """Get all saved backtest results."""
    return st.session_state.get("saved_results", {})


def add_to_comparison(results: Dict[str, Any]) -> None:
    """Add results to comparison list."""
    if "comparison_results" not in st.session_state:
        st.session_state.comparison_results = []
    st.session_state.comparison_results.append(results)


def clear_comparison() -> None:
    """Clear comparison list."""
    st.session_state.comparison_results = []


def get_comparison_results() -> List[Dict[str, Any]]:
    """Get comparison results list."""
    return st.session_state.get("comparison_results", [])


# =============================================================================
# Database Connection
# =============================================================================


def check_database_connection(db_path: str) -> Tuple[bool, str]:
    """Check if database is accessible and has data."""
    try:
        from backtester.data.dolt_adapter import DoltAdapter

        if not os.path.exists(db_path):
            return False, f"Database path does not exist: {db_path}"

        adapter = DoltAdapter(db_path)
        adapter.connect()

        tables = adapter.get_tables()
        if not tables:
            return False, "Database is empty (no tables found)"

        # Check for SPY data
        date_range = adapter.get_date_range("SPY")
        if date_range[0] is None:
            return False, "No SPY data found in database"

        adapter.close()
        return (
            True,
            f"Connected! Data range: {date_range[0].date()} to {date_range[1].date()}",
        )

    except Exception as e:
        return False, f"Connection error: {str(e)}"


# =============================================================================
# Backtest Execution
# =============================================================================


def run_real_backtest(
    strategy_name: str,
    start_date: datetime,
    end_date: datetime,
    initial_capital: float,
    params: Dict[str, Any],
    db_path: str,
    progress_callback=None,
) -> Dict[str, Any]:
    """
    Run an actual backtest using the BacktestEngine.

    This is the REAL backtest - not fake data!
    """
    from backtester.data.dolt_adapter import DoltAdapter
    from backtester.engine.backtest_engine import BacktestEngine
    from backtester.engine.data_stream import DataStream
    from backtester.engine.execution import ExecutionModel
    from backtester.strategies import (
        ShortStraddleHighIVStrategy,
        IronCondorStrategy,
        VolatilityRegimeStrategy,
    )

    # Connect to database
    adapter = DoltAdapter(db_path)
    adapter.connect()

    try:
        # Create strategy based on selection
        if strategy_name == "Short Straddle (High IV)":
            strategy = ShortStraddleHighIVStrategy(
                name="Short Straddle High IV",
                initial_capital=initial_capital,
                iv_rank_threshold=params.get("iv_rank_threshold", 70),
                profit_target_pct=params.get("profit_target_pct", 0.50),
                loss_limit_pct=params.get("stop_loss_pct", 2.0),
                min_entry_dte=params.get("dte_target", 30),
                exit_dte=7,
            )
            dte_range = (
                params.get("dte_target", 30),
                params.get("dte_target", 30) + 30,
            )

        elif strategy_name == "Iron Condor":
            strategy = IronCondorStrategy(
                name="Iron Condor",
                initial_capital=initial_capital,
                iv_rank_threshold=50,  # Iron condors work in moderate IV
                profit_target_pct=params.get("profit_target_pct", 0.50),
                loss_limit_pct=params.get("stop_loss_pct", 2.0),
                min_entry_dte=params.get("dte_target", 45),
                exit_dte=7,
                wing_width_pct=params.get("wing_width", 5) / 100,
            )
            dte_range = (
                params.get("dte_target", 45),
                params.get("dte_target", 45) + 15,
            )

        elif strategy_name == "Volatility Regime":
            strategy = VolatilityRegimeStrategy(
                name="Volatility Regime",
                initial_capital=initial_capital,
                high_vix_threshold=params.get("high_vol_threshold", 25),
                low_vix_threshold=params.get("low_vol_threshold", 15),
                profit_target_pct=params.get("profit_target_pct", 0.50),
                loss_limit_pct=params.get("stop_loss_pct", 2.0),
                exit_dte=7,
            )
            dte_range = (30, 60)
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        # Create data stream
        data_stream = DataStream(
            data_source=adapter,
            start_date=start_date,
            end_date=end_date,
            underlying="SPY",
            dte_range=dte_range,
            cache_enabled=True,
            skip_missing_data=True,
        )

        # Create execution model
        execution = ExecutionModel(commission_per_contract=0.65, slippage_pct=0.001)

        # Create and run backtest engine
        engine = BacktestEngine(
            strategy=strategy,
            data_stream=data_stream,
            execution_model=execution,
            initial_capital=initial_capital,
        )

        # Run the backtest
        results = engine.run()

        # Calculate metrics
        metrics = engine.calculate_metrics()

        # Get equity curve and trade log
        equity_curve = results["equity_curve"]
        trade_log = results["trade_log"]

        # Calculate daily returns
        if not equity_curve.empty and "equity" in equity_curve.columns:
            returns = equity_curve["equity"].pct_change().dropna()
        else:
            returns = pd.Series(dtype=float)

        # Calculate drawdown
        if not equity_curve.empty and "equity" in equity_curve.columns:
            rolling_max = equity_curve["equity"].expanding().max()
            drawdown = (equity_curve["equity"] - rolling_max) / rolling_max
        else:
            drawdown = pd.Series(dtype=float)

        # Build trades DataFrame in expected format
        if not trade_log.empty:
            closed_trades = trade_log[trade_log["action"] == "close"].copy()
            trades_list = []
            for i, row in closed_trades.iterrows():
                trades_list.append(
                    {
                        "trade_id": i + 1,
                        "entry_date": row["timestamp"]
                        - timedelta(days=30),  # Approximate
                        "exit_date": row["timestamp"],
                        "strategy": strategy_name,
                        "pnl": row["realized_pnl"],
                        "return_pct": row["realized_pnl"] / initial_capital * 100,
                        "holding_days": 30,  # Approximate
                        "result": "Win" if row["realized_pnl"] > 0 else "Loss",
                    }
                )
            trades_df = pd.DataFrame(trades_list)
        else:
            trades_df = pd.DataFrame(
                columns=[
                    "trade_id",
                    "entry_date",
                    "exit_date",
                    "strategy",
                    "pnl",
                    "return_pct",
                    "holding_days",
                    "result",
                ]
            )

        # Extract summary metrics
        summary = metrics.get("summary", {})
        performance = metrics.get("performance", {})
        risk = metrics.get("risk", {})

        final_equity = results.get("final_equity", initial_capital)
        total_return = results.get("total_return", 0)

        # Calculate additional metrics
        if not trades_df.empty:
            win_rate = len(trades_df[trades_df["pnl"] > 0]) / len(trades_df)
            avg_win = (
                trades_df[trades_df["pnl"] > 0]["pnl"].mean()
                if len(trades_df[trades_df["pnl"] > 0]) > 0
                else 0
            )
            avg_loss = (
                abs(trades_df[trades_df["pnl"] < 0]["pnl"].mean())
                if len(trades_df[trades_df["pnl"] < 0]) > 0
                else 1
            )
            profit_factor = (
                (avg_win * win_rate) / (avg_loss * (1 - win_rate))
                if avg_loss > 0 and win_rate < 1
                else 0
            )
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0

        return {
            "equity_curve": equity_curve["equity"]
            if not equity_curve.empty and "equity" in equity_curve.columns
            else pd.Series([initial_capital]),
            "returns": returns,
            "drawdown": drawdown,
            "trades": trades_df,
            "metrics": {
                "initial_capital": initial_capital,
                "final_equity": final_equity,
                "total_return": total_return,
                "annualized_return": summary.get("annualized_return", 0),
                "max_drawdown": summary.get("max_drawdown", 0),
                "sharpe_ratio": summary.get("sharpe_ratio", 0),
                "sortino_ratio": summary.get("sortino_ratio", 0),
                "calmar_ratio": summary.get("calmar_ratio", 0),
                "win_rate": win_rate,
                "profit_factor": profit_factor,
                "total_trades": len(trades_df),
                "avg_trade": trades_df["pnl"].mean() if len(trades_df) > 0 else 0,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "var_95": risk.get("var_95_historical", 0),
                "cvar_95": risk.get("cvar_95", 0),
                "best_trade": trades_df["pnl"].max() if len(trades_df) > 0 else 0,
                "worst_trade": trades_df["pnl"].min() if len(trades_df) > 0 else 0,
            },
            "params": params,
            "strategy": strategy_name,
            "start_date": start_date,
            "end_date": end_date,
            "trading_days": results.get("trading_days", 0),
            "timestamp": datetime.now(),  # When the backtest was run
        }

    finally:
        adapter.close()
