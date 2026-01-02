"""
Streamlit Web Application for Options Strategy Backtester

A user-friendly interface for running options strategy backtests without programming.
Supports strategy selection, parameter configuration, and interactive results visualization.

Usage:
    streamlit run streamlit_app.py

Requirements:
    pip install streamlit plotly pandas numpy
"""

import sys
from pathlib import Path

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

logging.basicConfig(level=logging.WARNING)

st.set_page_config(
    page_title="Options Strategy Backtester",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


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


def generate_sample_results(
    strategy_name: str,
    start_date: datetime,
    end_date: datetime,
    initial_capital: float,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Generate sample backtest results for demonstration.
    In production, this would call the actual backtest engine.
    """
    np.random.seed(42)

    dates = pd.date_range(start=start_date, end=end_date, freq="B")
    n_days = len(dates)

    if strategy_name == "Short Straddle (High IV)":
        base_return = 0.0003
        volatility = 0.015
        win_rate = 0.72
    elif strategy_name == "Iron Condor":
        base_return = 0.0002
        volatility = 0.008
        win_rate = 0.78
    else:
        base_return = 0.00025
        volatility = 0.012
        win_rate = 0.68

    daily_returns = np.random.normal(base_return, volatility, n_days)

    n_trades = int(n_days / 5)
    trade_dates = np.random.choice(dates, n_trades, replace=False)
    trade_dates = sorted(trade_dates)

    equity = [initial_capital]
    for r in daily_returns:
        equity.append(equity[-1] * (1 + r))
    equity = equity[1:]

    trades = []
    for i, entry_date in enumerate(trade_dates):
        exit_idx = min(dates.get_loc(entry_date) + np.random.randint(3, 15), n_days - 1)
        exit_date = dates[exit_idx]

        is_winner = np.random.random() < win_rate
        if is_winner:
            pnl = np.random.uniform(
                100, params.get("profit_target_pct", 0.25) * initial_capital * 0.1
            )
        else:
            pnl = -np.random.uniform(
                100, params.get("stop_loss_pct", 1.0) * initial_capital * 0.05
            )

        trades.append(
            {
                "trade_id": i + 1,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "strategy": strategy_name,
                "pnl": pnl,
                "return_pct": pnl / initial_capital * 100,
                "holding_days": (exit_date - entry_date).days,
                "result": "Win" if pnl > 0 else "Loss",
            }
        )

    trades_df = pd.DataFrame(trades)
    equity_series = pd.Series(equity, index=dates)
    returns_series = equity_series.pct_change().dropna()

    total_return = (equity[-1] - initial_capital) / initial_capital
    annual_factor = 252 / n_days
    annualized_return = (1 + total_return) ** annual_factor - 1

    rolling_max = equity_series.expanding().max()
    drawdown = (equity_series - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    sharpe = (
        (returns_series.mean() / returns_series.std()) * np.sqrt(252)
        if returns_series.std() > 0
        else 0
    )

    downside = returns_series[returns_series < 0]
    sortino = (
        (returns_series.mean() / downside.std()) * np.sqrt(252)
        if len(downside) > 0 and downside.std() > 0
        else 0
    )

    calmar = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

    actual_win_rate = (
        len(trades_df[trades_df["pnl"] > 0]) / len(trades_df)
        if len(trades_df) > 0
        else 0
    )
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
        (avg_win * actual_win_rate) / (avg_loss * (1 - actual_win_rate))
        if avg_loss > 0 and actual_win_rate < 1
        else 0
    )

    var_95 = np.percentile(returns_series, 5)
    cvar_95 = returns_series[returns_series <= var_95].mean()

    return {
        "equity_curve": equity_series,
        "returns": returns_series,
        "drawdown": drawdown,
        "trades": trades_df,
        "metrics": {
            "initial_capital": initial_capital,
            "final_equity": equity[-1],
            "total_return": total_return,
            "annualized_return": annualized_return,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "calmar_ratio": calmar,
            "win_rate": actual_win_rate,
            "profit_factor": profit_factor,
            "total_trades": len(trades_df),
            "avg_trade": trades_df["pnl"].mean() if len(trades_df) > 0 else 0,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "var_95": var_95,
            "cvar_95": cvar_95,
            "best_trade": trades_df["pnl"].max() if len(trades_df) > 0 else 0,
            "worst_trade": trades_df["pnl"].min() if len(trades_df) > 0 else 0,
        },
        "params": params,
        "strategy": strategy_name,
        "start_date": start_date,
        "end_date": end_date,
    }


def render_sidebar() -> Tuple[str, datetime, datetime, float, Dict[str, Any]]:
    """Render sidebar configuration and return user selections."""
    st.sidebar.title("Strategy Configuration")

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
            min_value=datetime(2020, 1, 1),
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
    )


def render_metrics_cards(metrics: Dict[str, float]) -> None:
    """Render key metrics as cards."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_return = metrics["total_return"]
        st.metric(
            "Total Return",
            f"{total_return:.1%}",
            delta=f"{metrics['annualized_return']:.1%} ann.",
        )

    with col2:
        st.metric(
            "Sharpe Ratio",
            f"{metrics['sharpe_ratio']:.2f}",
            delta="Good" if metrics["sharpe_ratio"] > 1 else "Low",
            delta_color="normal" if metrics["sharpe_ratio"] > 1 else "inverse",
        )

    with col3:
        st.metric(
            "Max Drawdown",
            f"{metrics['max_drawdown']:.1%}",
            delta_color="inverse",
        )

    with col4:
        st.metric(
            "Win Rate",
            f"{metrics['win_rate']:.1%}",
            delta=f"{metrics['total_trades']} trades",
        )


def render_equity_curve(results: Dict[str, Any]) -> None:
    """Render interactive equity curve with drawdown."""
    equity = results["equity_curve"]
    drawdown = results["drawdown"]

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
    returns = results["returns"] * 100

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=returns,
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
        x=returns.mean(),
        line_dash="solid",
        line_color="green",
        annotation_text=f"Mean: {returns.mean():.2f}%",
    )

    var_95 = results["metrics"]["var_95"] * 100
    fig.add_vline(
        x=var_95,
        line_dash="dot",
        line_color="red",
        annotation_text=f"VaR 95%: {var_95:.2f}%",
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
        st.write(f"**Annualized Return:** {metrics['annualized_return']:.2%}")
        st.write(f"**Sharpe Ratio:** {metrics['sharpe_ratio']:.3f}")
        st.write(f"**Sortino Ratio:** {metrics['sortino_ratio']:.3f}")
        st.write(f"**Calmar Ratio:** {metrics['calmar_ratio']:.3f}")

    with col2:
        st.subheader("Risk")
        st.write(f"**Max Drawdown:** {metrics['max_drawdown']:.2%}")
        st.write(f"**VaR (95%):** {metrics['var_95']:.2%}")
        st.write(f"**CVaR (95%):** {metrics['cvar_95']:.2%}")
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

    monthly = returns.resample("M").apply(lambda x: (1 + x).prod() - 1) * 100
    monthly_df = pd.DataFrame(
        {
            "Year": monthly.index.year,
            "Month": monthly.index.month,
            "Return": monthly.values,
        }
    )

    pivot = monthly_df.pivot(index="Year", columns="Month", values="Return")
    pivot.columns = [
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
    ][: len(pivot.columns)]

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


def main():
    st.title("Options Strategy Backtester")
    st.markdown("*Professional-grade options strategy backtesting platform*")

    strategy, start_date, end_date, capital, params = render_sidebar()

    run_button = st.sidebar.button(
        "Run Backtest",
        type="primary",
        use_container_width=True,
    )

    if "results" not in st.session_state:
        st.session_state.results = None

    if run_button:
        with st.spinner(f"Running {strategy} backtest..."):
            import time

            time.sleep(1)

            st.session_state.results = generate_sample_results(
                strategy, start_date, end_date, capital, params
            )
        st.success("Backtest completed!")

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

        return

    results = st.session_state.results

    st.header(f"{results['strategy']} Results")
    st.caption(
        f"Period: {results['start_date'].strftime('%Y-%m-%d')} to {results['end_date'].strftime('%Y-%m-%d')} | "
        f"Initial Capital: ${results['metrics']['initial_capital']:,.0f}"
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
        
        This is an MVP demo of the Options Strategy Backtester. 
        Results shown are simulated for demonstration purposes.
        
        [View Documentation](../docs/USER_GUIDE.md)
        """
    )


if __name__ == "__main__":
    main()
