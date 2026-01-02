"""
Chart and visualization components for Options Strategy Backtester.

Contains reusable Plotly chart components.
"""

from typing import Dict, Any
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


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


def render_comparison_chart(results_list: list) -> None:
    """Render comparison chart for multiple backtests."""
    if not results_list:
        st.info("No results to compare")
        return

    fig = go.Figure()

    for i, results in enumerate(results_list):
        equity = results["equity_curve"]
        if not equity.empty:
            # Normalize to starting value of 100
            normalized = (equity / equity.iloc[0]) * 100
            fig.add_trace(
                go.Scatter(
                    x=normalized.index,
                    y=normalized.values,
                    mode="lines",
                    name=f"{results['strategy']} ({results['start_date'].strftime('%Y-%m-%d')})",
                    line=dict(width=2),
                )
            )

    fig.add_hline(y=100, line_dash="dash", line_color="gray")

    fig.update_layout(
        title="Strategy Comparison (Normalized to 100)",
        xaxis_title="Date",
        yaxis_title="Normalized Value",
        height=500,
        hovermode="x unified",
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    st.plotly_chart(fig, use_container_width=True)


def render_comparison_table(results_list: list) -> None:
    """Render comparison table for multiple backtests."""
    if not results_list:
        return

    data = []
    for results in results_list:
        m = results["metrics"]
        data.append(
            {
                "Strategy": results["strategy"],
                "Period": f"{results['start_date'].strftime('%Y-%m-%d')} to {results['end_date'].strftime('%Y-%m-%d')}",
                "Total Return": f"{m['total_return']:.1%}",
                "Ann. Return": f"{m['annualized_return']:.1%}"
                if m.get("annualized_return")
                else "N/A",
                "Sharpe": f"{m['sharpe_ratio']:.2f}"
                if m.get("sharpe_ratio")
                else "N/A",
                "Max DD": f"{m['max_drawdown']:.1%}"
                if m.get("max_drawdown")
                else "N/A",
                "Win Rate": f"{m['win_rate']:.1%}" if m.get("win_rate") else "N/A",
                "Trades": m["total_trades"],
            }
        )

    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True, hide_index=True)
