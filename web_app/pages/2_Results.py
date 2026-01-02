"""
Options Strategy Backtester - Results Page

View and analyze backtest results in detail.
"""

import streamlit as st

# Page configuration MUST be first Streamlit command
st.set_page_config(
    page_title="Results - Options Backtester",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import plotly.express as px

# Add paths for imports
web_app_path = Path(__file__).parent.parent
sys.path.insert(0, str(web_app_path))
sys.path.insert(0, str(web_app_path.parent / "code"))

# Import shared utilities
from shared.utils import (
    init_session_state,
    get_results,
    get_all_saved_results,
    delete_saved_result,
    save_result,
)
from shared.charts import (
    render_equity_curve,
    render_metrics_cards,
    render_returns_distribution,
    render_monthly_returns,
    render_detailed_metrics,
    render_trades_table,
    render_comparison_chart,
    render_comparison_table,
)

# Initialize session state
init_session_state()


def render_result_selector():
    """Render result selection sidebar."""
    st.sidebar.title("📈 Results Browser")

    # Current result
    current = get_results()
    saved = get_all_saved_results()

    st.sidebar.header("Current Result")
    if current:
        st.sidebar.success(f"✅ {current['strategy']}")
        st.sidebar.caption(
            f"{current['start_date'].strftime('%Y-%m-%d')} to {current['end_date'].strftime('%Y-%m-%d')}"
        )
    else:
        st.sidebar.info("No current result")

    # Saved results
    st.sidebar.header("Saved Results")
    if saved:
        selected = st.sidebar.selectbox(
            "Select saved result",
            options=["Current"] + list(saved.keys()),
            help="Choose a saved result to view",
        )

        if selected != "Current" and selected in saved:
            if st.sidebar.button("Load Selected"):
                st.session_state.results = saved[selected]
                st.rerun()

            if st.sidebar.button("🗑️ Delete Selected", type="secondary"):
                delete_saved_result(selected)
                st.rerun()
    else:
        st.sidebar.info("No saved results yet")

    # Export section
    st.sidebar.header("Export")
    export_format = st.sidebar.selectbox(
        "Export Format",
        options=["CSV", "JSON"],
        help="Select format for export",
    )

    if st.sidebar.button("📥 Export Results"):
        if current:
            if export_format == "CSV":
                # Export trades as CSV
                trades_csv = current["trades"].to_csv(index=False)
                st.sidebar.download_button(
                    "Download Trades CSV",
                    trades_csv,
                    f"trades_{current['strategy'].replace(' ', '_')}.csv",
                    "text/csv",
                )
            else:
                # Export metrics as JSON
                import json

                metrics_json = json.dumps(current["metrics"], indent=2, default=str)
                st.sidebar.download_button(
                    "Download Metrics JSON",
                    metrics_json,
                    f"metrics_{current['strategy'].replace(' ', '_')}.json",
                    "application/json",
                )
        else:
            st.sidebar.warning("No results to export")


def render_summary_tab(results):
    """Render summary tab content."""
    st.header(f"📊 {results['strategy']} Summary")
    st.caption(
        f"Period: {results['start_date'].strftime('%Y-%m-%d')} to {results['end_date'].strftime('%Y-%m-%d')} | "
        f"Trading Days: {results.get('trading_days', 'N/A')}"
    )

    # Key metrics
    render_metrics_cards(results["metrics"])

    # Equity curve
    st.subheader("Equity Curve")
    render_equity_curve(results)

    # Two column charts
    col1, col2 = st.columns(2)
    with col1:
        render_returns_distribution(results)
    with col2:
        render_monthly_returns(results)


def render_metrics_tab(results):
    """Render detailed metrics tab content."""
    st.header("📊 Detailed Metrics")

    render_detailed_metrics(results["metrics"])

    # Additional metrics table
    st.subheader("All Metrics")
    metrics = results["metrics"]

    # Format metrics for display
    metrics_display = []
    for key, value in metrics.items():
        if isinstance(value, float):
            if (
                "pct" in key.lower()
                or "return" in key.lower()
                or "rate" in key.lower()
                or "drawdown" in key.lower()
            ):
                formatted = f"{value:.2%}" if not np.isnan(value) else "N/A"
            elif "ratio" in key.lower():
                formatted = f"{value:.3f}" if not np.isnan(value) else "N/A"
            else:
                formatted = f"${value:,.2f}" if abs(value) > 1 else f"{value:.4f}"
        else:
            formatted = str(value)

        metrics_display.append(
            {
                "Metric": key.replace("_", " ").title(),
                "Value": formatted,
            }
        )

    st.dataframe(
        pd.DataFrame(metrics_display),
        width="stretch",
        hide_index=True,
    )


def render_trades_tab(results):
    """Render trades tab content."""
    st.header("📝 Trade History")

    trades = results["trades"]

    if trades.empty:
        st.info("No trades executed during the backtest period")
        return

    # Trade statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Trades", len(trades))
    with col2:
        wins = len(trades[trades["pnl"] > 0])
        st.metric("Winning Trades", wins)
    with col3:
        losses = len(trades[trades["pnl"] < 0])
        st.metric("Losing Trades", losses)
    with col4:
        total_pnl = trades["pnl"].sum()
        st.metric("Total P&L", f"${total_pnl:,.2f}")

    # Trade filters
    st.subheader("Filter Trades")
    col1, col2 = st.columns(2)
    with col1:
        show_winners = st.checkbox("Show Winners", value=True)
    with col2:
        show_losers = st.checkbox("Show Losers", value=True)

    # Filter trades
    filtered = trades.copy()
    if not show_winners:
        filtered = filtered[filtered["pnl"] <= 0]
    if not show_losers:
        filtered = filtered[filtered["pnl"] >= 0]

    # Display trades table
    render_trades_table({"trades": filtered})

    # P&L distribution
    st.subheader("P&L Distribution")

    fig = px.histogram(
        trades,
        x="pnl",
        nbins=20,
        title="Trade P&L Distribution",
        labels={"pnl": "P&L ($)"},
        color_discrete_sequence=["#2E86AB"],
    )
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig, width="stretch")


def render_params_tab(results):
    """Render parameters tab content."""
    st.header("⚙️ Strategy Parameters")

    # Parameters table
    st.subheader("Parameters Used")
    params_df = pd.DataFrame([results["params"]])
    st.dataframe(params_df, width="stretch", hide_index=True)

    # Backtest configuration
    st.subheader("Backtest Configuration")
    config_data = {
        "Setting": [
            "Strategy",
            "Start Date",
            "End Date",
            "Initial Capital",
            "Trading Days",
        ],
        "Value": [
            results["strategy"],
            results["start_date"].strftime("%Y-%m-%d"),
            results["end_date"].strftime("%Y-%m-%d"),
            f"${results['metrics']['initial_capital']:,.2f}",
            results.get("trading_days", "N/A"),
        ],
    }
    st.dataframe(pd.DataFrame(config_data), width="stretch", hide_index=True)

    # Timestamp
    if "timestamp" in results:
        st.caption(
            f"Backtest run at: {results['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}"
        )


# Render sidebar
render_result_selector()

# Main page content
st.title("📈 Backtest Results")
st.markdown("*Analyze your backtest results in detail*")

# Get current results
results = get_results()

if results is None:
    st.info("No backtest results to display. Run a backtest first!")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚀 Go to Backtest"):
            st.switch_page("pages/1_Backtest.py")
    with col2:
        saved = get_all_saved_results()
        if saved:
            st.write(
                f"You have {len(saved)} saved result(s). Select one from the sidebar."
            )
else:
    # Tab navigation
    tabs = st.tabs(["📊 Summary", "📈 Metrics", "📝 Trades", "⚙️ Parameters"])

    with tabs[0]:
        render_summary_tab(results)

    with tabs[1]:
        render_metrics_tab(results)

    with tabs[2]:
        render_trades_tab(results)

    with tabs[3]:
        render_params_tab(results)

    # Save option
    st.markdown("---")
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        save_name = st.text_input(
            "Save this result as:",
            value=f"{results['strategy']}_{results['start_date'].strftime('%Y%m%d')}",
        )
    with col2:
        if st.button("💾 Save Result"):
            save_result(save_name, results)
            st.success(f"Saved as '{save_name}'")
    with col3:
        if st.button("🔧 Compare Strategies"):
            st.switch_page("pages/3_Strategy_Builder.py")
