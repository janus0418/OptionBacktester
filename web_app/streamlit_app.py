"""
Streamlit Web Application for Options Strategy Backtester

A user-friendly interface for running options strategy backtests.
Now uses the REAL BacktestEngine for actual backtesting with Dolt database.

Usage:
    streamlit run streamlit_app.py

Requirements:
    pip install streamlit plotly pandas numpy
"""

import sys
from pathlib import Path

# Add backtester to path
sys.path.insert(0, str(Path(__file__).parent.parent / "code"))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
import logging
import os

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Options Strategy Backtester",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# Configuration
# =============================================================================

# Default database path - update this to your Dolt database location
DEFAULT_DB_PATH = str(Path(__file__).parent.parent / "dolt_data" / "options")

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
# Backtester Integration
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

        adapter.disconnect()
        return (
            True,
            f"Connected! Data range: {date_range[0].date()} to {date_range[1].date()}",
        )

    except Exception as e:
        return False, f"Connection error: {str(e)}"


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
        }

    finally:
        adapter.disconnect()


# =============================================================================
# UI Components
# =============================================================================


def render_sidebar() -> Tuple[str, datetime, datetime, float, Dict[str, Any], str]:
    """Render sidebar configuration and return user selections."""
    st.sidebar.title("Strategy Configuration")

    # Database configuration
    st.sidebar.header("Database")
    db_path = st.sidebar.text_input(
        "Dolt Database Path",
        value=DEFAULT_DB_PATH,
        help="Path to your Dolt options database",
    )

    # Check connection
    is_connected, status_msg = check_database_connection(db_path)
    if is_connected:
        st.sidebar.success(f"✅ {status_msg}")
    else:
        st.sidebar.error(f"❌ {status_msg}")

    st.sidebar.header("Strategy Selection")
    strategy = st.sidebar.selectbox(
        "Strategy Type",
        list(STRATEGY_INFO.keys()),
        help="Select the options strategy to backtest",
    )

    with st.sidebar.expander("Strategy Info", expanded=False):
        info = STRATEGY_INFO[strategy]
        st.write(f"**Description:** {info['description']}")
        st.write(f"**Risk Level:** {info['risk']}")
        st.write(f"**Ideal Conditions:** {info['ideal_conditions']}")

    st.sidebar.header("Backtest Period")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime(2023, 1, 1),
            min_value=datetime(2019, 1, 1),
            max_value=datetime.now() - timedelta(days=30),
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime(2024, 1, 1),
            min_value=start_date + timedelta(days=30),
            max_value=datetime.now(),
        )

    st.sidebar.header("Capital")
    initial_capital = st.sidebar.number_input(
        "Initial Capital ($)",
        min_value=10000,
        max_value=10000000,
        value=100000,
        step=10000,
        help="Starting capital for the backtest",
    )

    st.sidebar.header("Strategy Parameters")
    params = {}

    if strategy == "Short Straddle (High IV)":
        params["iv_rank_threshold"] = st.sidebar.slider(
            "IV Rank Threshold (%)",
            min_value=30,
            max_value=90,
            value=70,
            help="Minimum IV rank to enter a trade",
        )
        params["dte_target"] = st.sidebar.slider(
            "Target DTE (days)",
            min_value=7,
            max_value=60,
            value=30,
            help="Days to expiration for options",
        )

    elif strategy == "Iron Condor":
        params["delta_target"] = st.sidebar.slider(
            "Short Strike Delta",
            min_value=0.05,
            max_value=0.30,
            value=0.16,
            step=0.01,
            help="Delta for short strikes",
        )
        params["wing_width"] = st.sidebar.slider(
            "Wing Width ($)",
            min_value=1,
            max_value=20,
            value=5,
            help="Width between short and long strikes",
        )
        params["dte_target"] = st.sidebar.slider(
            "Target DTE (days)",
            min_value=21,
            max_value=60,
            value=45,
        )

    elif strategy == "Volatility Regime":
        params["low_vol_threshold"] = st.sidebar.slider(
            "Low Vol VIX Threshold",
            min_value=10,
            max_value=20,
            value=15,
        )
        params["high_vol_threshold"] = st.sidebar.slider(
            "High Vol VIX Threshold",
            min_value=20,
            max_value=40,
            value=25,
        )

    params["profit_target_pct"] = (
        st.sidebar.slider(
            "Profit Target (%)",
            min_value=10,
            max_value=80,
            value=50 if strategy == "Short Straddle (High IV)" else 25,
            help="Close position when this % of max profit is reached",
        )
        / 100
    )

    params["stop_loss_pct"] = (
        st.sidebar.slider(
            "Stop Loss (%)",
            min_value=50,
            max_value=300,
            value=200 if strategy == "Short Straddle (High IV)" else 100,
            help="Close position when loss exceeds this % of credit received",
        )
        / 100
    )

    return (
        strategy,
        datetime.combine(start_date, datetime.min.time()),
        datetime.combine(end_date, datetime.min.time()),
        initial_capital,
        params,
        db_path,
    )


def render_metrics_cards(metrics: Dict[str, float]) -> None:
    """Render key metrics as cards."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_return = metrics["total_return"]
        st.metric(
            "Total Return",
            f"{total_return:.1%}",
            delta=f"{metrics['annualized_return']:.1%} ann."
            if metrics["annualized_return"]
            else None,
        )

    with col2:
        sharpe = metrics.get("sharpe_ratio", 0)
        if sharpe and not np.isnan(sharpe):
            st.metric(
                "Sharpe Ratio",
                f"{sharpe:.2f}",
                delta="Good" if sharpe > 1 else "Low",
                delta_color="normal" if sharpe > 1 else "inverse",
            )
        else:
            st.metric("Sharpe Ratio", "N/A")

    with col3:
        max_dd = metrics.get("max_drawdown", 0)
        if max_dd and not np.isnan(max_dd):
            st.metric(
                "Max Drawdown",
                f"{max_dd:.1%}",
                delta_color="inverse",
            )
        else:
            st.metric("Max Drawdown", "N/A")

    with col4:
        st.metric(
            "Win Rate",
            f"{metrics['win_rate']:.1%}" if metrics["win_rate"] else "N/A",
            delta=f"{metrics['total_trades']} trades",
        )


def render_equity_curve(results: Dict[str, Any]) -> None:
    """Render interactive equity curve with drawdown."""
    equity = results["equity_curve"]
    drawdown = results["drawdown"]

    if equity.empty:
        st.warning("No equity curve data available")
        return

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=("Equity Curve", "Drawdown"),
    )

    fig.add_trace(
        go.Scatter(
            x=equity.index,
            y=equity.values,
            mode="lines",
            name="Equity",
            line=dict(color="#2E86AB", width=2),
            fill="tozeroy",
            fillcolor="rgba(46, 134, 171, 0.1)",
        ),
        row=1,
        col=1,
    )

    initial = results["metrics"]["initial_capital"]
    fig.add_hline(
        y=initial,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"Initial: ${initial:,.0f}",
        row=1,
        col=1,
    )

    if not drawdown.empty:
        fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown.values * 100,
                mode="lines",
                name="Drawdown",
                line=dict(color="#E94F37", width=1.5),
                fill="tozeroy",
                fillcolor="rgba(233, 79, 55, 0.2)",
            ),
            row=2,
            col=1,
        )

    fig.update_layout(
        height=500,
        showlegend=False,
        hovermode="x unified",
        margin=dict(l=0, r=0, t=30, b=0),
    )

    fig.update_yaxes(title_text="Equity ($)", tickformat="$,.0f", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", tickformat=".1f", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)


def render_returns_distribution(results: Dict[str, Any]) -> None:
    """Render returns distribution histogram."""
    returns = results["returns"]

    if returns.empty:
        st.info("No returns data available")
        return

    returns_pct = returns * 100

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=returns_pct,
            nbinsx=50,
            name="Daily Returns",
            marker_color="#2E86AB",
            opacity=0.7,
        )
    )

    fig.add_vline(
        x=0,
        line_dash="dash",
        line_color="gray",
    )

    fig.add_vline(
        x=returns_pct.mean(),
        line_dash="solid",
        line_color="green",
        annotation_text=f"Mean: {returns_pct.mean():.2f}%",
    )

    var_95 = results["metrics"].get("var_95", 0)
    if var_95 and not np.isnan(var_95):
        fig.add_vline(
            x=var_95 * 100,
            line_dash="dot",
            line_color="red",
            annotation_text=f"VaR 95%: {var_95 * 100:.2f}%",
        )

    fig.update_layout(
        title="Daily Returns Distribution",
        xaxis_title="Return (%)",
        yaxis_title="Frequency",
        height=350,
        showlegend=False,
        margin=dict(l=0, r=0, t=40, b=0),
    )

    st.plotly_chart(fig, use_container_width=True)


def render_trades_table(results: Dict[str, Any]) -> None:
    """Render trades table with formatting."""
    trades = results["trades"].copy()

    if trades.empty:
        st.info("No trades executed during the backtest period")
        return

    trades["entry_date"] = pd.to_datetime(trades["entry_date"]).dt.strftime("%Y-%m-%d")
    trades["exit_date"] = pd.to_datetime(trades["exit_date"]).dt.strftime("%Y-%m-%d")
    trades["pnl"] = trades["pnl"].apply(lambda x: f"${x:,.2f}")
    trades["return_pct"] = trades["return_pct"].apply(lambda x: f"{x:.2f}%")

    st.dataframe(
        trades[
            [
                "trade_id",
                "entry_date",
                "exit_date",
                "holding_days",
                "pnl",
                "return_pct",
                "result",
            ]
        ],
        use_container_width=True,
        hide_index=True,
        column_config={
            "trade_id": st.column_config.NumberColumn("Trade #", format="%d"),
            "entry_date": "Entry Date",
            "exit_date": "Exit Date",
            "holding_days": st.column_config.NumberColumn("Days Held", format="%d"),
            "pnl": "P&L",
            "return_pct": "Return",
            "result": st.column_config.TextColumn("Result"),
        },
    )


def render_detailed_metrics(metrics: Dict[str, float]) -> None:
    """Render detailed metrics in columns."""
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Performance")
        st.write(f"**Initial Capital:** ${metrics['initial_capital']:,.2f}")
        st.write(f"**Final Equity:** ${metrics['final_equity']:,.2f}")
        st.write(f"**Total Return:** {metrics['total_return']:.2%}")
        ann_ret = metrics.get("annualized_return", 0)
        st.write(
            f"**Annualized Return:** {ann_ret:.2%}"
            if ann_ret and not np.isnan(ann_ret)
            else "**Annualized Return:** N/A"
        )
        sharpe = metrics.get("sharpe_ratio", 0)
        st.write(
            f"**Sharpe Ratio:** {sharpe:.3f}"
            if sharpe and not np.isnan(sharpe)
            else "**Sharpe Ratio:** N/A"
        )
        sortino = metrics.get("sortino_ratio", 0)
        st.write(
            f"**Sortino Ratio:** {sortino:.3f}"
            if sortino and not np.isnan(sortino)
            else "**Sortino Ratio:** N/A"
        )
        calmar = metrics.get("calmar_ratio", 0)
        st.write(
            f"**Calmar Ratio:** {calmar:.3f}"
            if calmar and not np.isnan(calmar)
            else "**Calmar Ratio:** N/A"
        )

    with col2:
        st.subheader("Risk")
        max_dd = metrics.get("max_drawdown", 0)
        st.write(
            f"**Max Drawdown:** {max_dd:.2%}"
            if max_dd and not np.isnan(max_dd)
            else "**Max Drawdown:** N/A"
        )
        var_95 = metrics.get("var_95", 0)
        st.write(
            f"**VaR (95%):** {var_95:.2%}"
            if var_95 and not np.isnan(var_95)
            else "**VaR (95%):** N/A"
        )
        cvar_95 = metrics.get("cvar_95", 0)
        st.write(
            f"**CVaR (95%):** {cvar_95:.2%}"
            if cvar_95 and not np.isnan(cvar_95)
            else "**CVaR (95%):** N/A"
        )
        st.write(f"**Best Trade:** ${metrics['best_trade']:,.2f}")
        st.write(f"**Worst Trade:** ${metrics['worst_trade']:,.2f}")

    with col3:
        st.subheader("Trading")
        st.write(f"**Total Trades:** {metrics['total_trades']}")
        st.write(f"**Win Rate:** {metrics['win_rate']:.1%}")
        st.write(f"**Profit Factor:** {metrics['profit_factor']:.2f}")
        st.write(f"**Avg Trade:** ${metrics['avg_trade']:,.2f}")
        st.write(f"**Avg Win:** ${metrics['avg_win']:,.2f}")
        st.write(f"**Avg Loss:** ${metrics['avg_loss']:,.2f}")


def render_monthly_returns(results: Dict[str, Any]) -> None:
    """Render monthly returns heatmap."""
    returns = results["returns"]

    if returns.empty:
        st.info("No returns data for monthly analysis")
        return

    monthly = returns.resample("M").apply(lambda x: (1 + x).prod() - 1) * 100
    monthly_df = pd.DataFrame(
        {
            "Year": monthly.index.year,
            "Month": monthly.index.month,
            "Return": monthly.values,
        }
    )

    if monthly_df.empty:
        return

    pivot = monthly_df.pivot(index="Year", columns="Month", values="Return")
    month_names = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    pivot.columns = [month_names[m - 1] for m in pivot.columns]

    fig = px.imshow(
        pivot,
        color_continuous_scale=["#E94F37", "#FFFFFF", "#2E86AB"],
        color_continuous_midpoint=0,
        aspect="auto",
        labels=dict(color="Return (%)"),
    )

    fig.update_layout(
        title="Monthly Returns (%)",
        height=250,
        margin=dict(l=0, r=0, t=40, b=0),
    )

    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# Main Application
# =============================================================================


def main():
    st.title("Options Strategy Backtester")
    st.markdown(
        "*Professional-grade options strategy backtesting with real market data*"
    )

    strategy, start_date, end_date, capital, params, db_path = render_sidebar()

    # Check if database is connected
    is_connected, _ = check_database_connection(db_path)

    run_button = st.sidebar.button(
        "🚀 Run Backtest",
        type="primary",
        use_container_width=True,
        disabled=not is_connected,
    )

    if "results" not in st.session_state:
        st.session_state.results = None

    if run_button:
        if not is_connected:
            st.error("Please configure a valid database connection first")
            return

        with st.spinner(f"Running {strategy} backtest... This may take a few minutes."):
            try:
                st.session_state.results = run_real_backtest(
                    strategy, start_date, end_date, capital, params, db_path
                )
                st.success(
                    f"✅ Backtest completed! Processed {st.session_state.results.get('trading_days', 0)} trading days."
                )
            except Exception as e:
                st.error(f"Backtest failed: {str(e)}")
                logger.exception("Backtest error")
                return

    if st.session_state.results is None:
        st.info(
            "Configure your strategy in the sidebar and click 'Run Backtest' to begin."
        )

        st.subheader("Available Strategies")
        for name, info in STRATEGY_INFO.items():
            with st.expander(name):
                st.write(f"**Description:** {info['description']}")
                st.write(f"**Risk Level:** {info['risk']}")
                st.write(f"**Ideal Conditions:** {info['ideal_conditions']}")

        st.markdown("---")
        st.markdown("""
        ### Getting Started
        
        1. **Configure Database**: Enter the path to your Dolt options database
        2. **Select Strategy**: Choose from Short Straddle, Iron Condor, or Volatility Regime
        3. **Set Parameters**: Adjust the backtest period, capital, and strategy parameters
        4. **Run Backtest**: Click the button to run the actual backtest
        
        ⚠️ **Note**: This runs REAL backtests using the BacktestEngine - not simulated data!
        """)
        return

    results = st.session_state.results

    st.header(f"📊 {results['strategy']} Results")
    st.caption(
        f"Period: {results['start_date'].strftime('%Y-%m-%d')} to {results['end_date'].strftime('%Y-%m-%d')} | "
        f"Initial Capital: ${results['metrics']['initial_capital']:,.0f} | "
        f"Trading Days: {results.get('trading_days', 'N/A')}"
    )

    render_metrics_cards(results["metrics"])

    st.subheader("Equity Curve")
    render_equity_curve(results)

    col1, col2 = st.columns(2)
    with col1:
        render_returns_distribution(results)
    with col2:
        render_monthly_returns(results)

    st.subheader("Detailed Metrics")
    render_detailed_metrics(results["metrics"])

    st.subheader("Trade History")
    render_trades_table(results)

    st.subheader("Strategy Parameters Used")
    params_df = pd.DataFrame([results["params"]])
    st.dataframe(params_df, use_container_width=True, hide_index=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        **About**
        
        This backtester uses the actual BacktestEngine 
        with real options data from your Dolt database.
        
        Results are based on actual historical option chains,
        not simulated data.
        
        [View Documentation](../docs/USER_GUIDE.md)
        """
    )


if __name__ == "__main__":
    main()
