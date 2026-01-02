"""
Options Strategy Backtester - Strategy Builder Page

Build, configure, and compare options strategies.
"""

import streamlit as st

# Page configuration MUST be first Streamlit command
st.set_page_config(
    page_title="Strategy Builder - Options Backtester",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any
import pandas as pd
import numpy as np

# Add paths for imports
web_app_path = Path(__file__).parent.parent
sys.path.insert(0, str(web_app_path))
sys.path.insert(0, str(web_app_path.parent / "code"))

# Import shared utilities
from shared.utils import (
    STRATEGY_INFO,
    DEFAULT_DB_PATH,
    check_database_connection,
    run_real_backtest,
    init_session_state,
    get_all_saved_results,
    save_result,
    add_to_comparison,
    clear_comparison,
    get_comparison_results,
)
from shared.charts import (
    render_comparison_chart,
    render_comparison_table,
    render_equity_curve,
    render_metrics_cards,
)

# Initialize session state
init_session_state()


def render_strategy_wizard():
    """Render the strategy configuration wizard."""
    st.header("🔧 Strategy Configuration Wizard")

    # Step 1: Select Strategy
    st.subheader("Step 1: Select Strategy")

    strategy = st.selectbox(
        "Choose a strategy template",
        options=list(STRATEGY_INFO.keys()),
        help="Select the base strategy to configure",
    )

    info = STRATEGY_INFO[strategy]
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"**Risk Level:** {info['risk']}")
    with col2:
        st.info(f"**Ideal Conditions:** {info['ideal_conditions']}")
    with col3:
        st.info(f"**Parameters:** {len(info['params'])}")

    st.markdown(f"*{info['description']}*")

    # Step 2: Configure Parameters
    st.subheader("Step 2: Configure Parameters")

    params = {}
    col1, col2 = st.columns(2)

    with col1:
        if strategy == "Short Straddle (High IV)":
            params["iv_rank_threshold"] = st.slider(
                "IV Rank Threshold (%)",
                min_value=30,
                max_value=90,
                value=70,
                help="Minimum IV rank to enter a trade. Higher = more selective.",
                key="wizard_iv_rank",
            )
            params["dte_target"] = st.slider(
                "Target DTE (days)",
                min_value=7,
                max_value=60,
                value=30,
                help="Days to expiration. Longer = more premium but more risk.",
                key="wizard_dte",
            )

        elif strategy == "Iron Condor":
            params["delta_target"] = st.slider(
                "Short Strike Delta",
                min_value=0.05,
                max_value=0.30,
                value=0.16,
                step=0.01,
                help="Delta for short strikes. Lower = wider but less premium.",
                key="wizard_delta",
            )
            params["wing_width"] = st.slider(
                "Wing Width ($)",
                min_value=1,
                max_value=20,
                value=5,
                help="Width between short and long strikes. Wider = more risk.",
                key="wizard_wing",
            )
            params["dte_target"] = st.slider(
                "Target DTE (days)",
                min_value=21,
                max_value=60,
                value=45,
                key="wizard_ic_dte",
            )

        elif strategy == "Volatility Regime":
            params["low_vol_threshold"] = st.slider(
                "Low Vol VIX Threshold",
                min_value=10,
                max_value=20,
                value=15,
                help="VIX level below which market is considered low volatility.",
                key="wizard_low_vol",
            )
            params["high_vol_threshold"] = st.slider(
                "High Vol VIX Threshold",
                min_value=20,
                max_value=40,
                value=25,
                help="VIX level above which market is considered high volatility.",
                key="wizard_high_vol",
            )

    with col2:
        params["profit_target_pct"] = (
            st.slider(
                "Profit Target (%)",
                min_value=10,
                max_value=80,
                value=50 if strategy == "Short Straddle (High IV)" else 25,
                help="Close position when this % of max profit is reached.",
                key="wizard_profit",
            )
            / 100
        )

        params["stop_loss_pct"] = (
            st.slider(
                "Stop Loss (%)",
                min_value=50,
                max_value=300,
                value=200 if strategy == "Short Straddle (High IV)" else 100,
                help="Close position when loss exceeds this % of credit received.",
                key="wizard_stop",
            )
            / 100
        )

    # Step 3: Backtest Settings
    st.subheader("Step 3: Backtest Settings")

    col1, col2, col3 = st.columns(3)

    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime(2023, 1, 1),
            min_value=datetime(2019, 1, 1),
            max_value=datetime.now() - timedelta(days=30),
            key="wizard_start",
        )

    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime(2024, 1, 1),
            min_value=start_date + timedelta(days=30),
            max_value=datetime.now(),
            key="wizard_end",
        )

    with col3:
        initial_capital = st.number_input(
            "Initial Capital ($)",
            min_value=10000,
            max_value=10000000,
            value=100000,
            step=10000,
            key="wizard_capital",
        )

    return {
        "strategy": strategy,
        "params": params,
        "start_date": datetime.combine(start_date, datetime.min.time()),
        "end_date": datetime.combine(end_date, datetime.min.time()),
        "initial_capital": initial_capital,
    }


def render_comparison_section():
    """Render the strategy comparison section."""
    st.header("📊 Strategy Comparison")

    comparison_results = get_comparison_results()
    saved_results = get_all_saved_results()

    # Add to comparison
    col1, col2 = st.columns([3, 1])
    with col1:
        if saved_results:
            selected = st.multiselect(
                "Select results to compare",
                options=list(saved_results.keys()),
                default=[],
                help="Choose saved results to add to comparison",
            )
        else:
            st.info("No saved results available. Run and save some backtests first!")
            selected = []

    with col2:
        if st.button("Add to Comparison", disabled=not selected):
            for name in selected:
                add_to_comparison(saved_results[name])
            st.rerun()

    # Current comparison
    if comparison_results:
        st.subheader(f"Comparing {len(comparison_results)} Strategies")

        # Clear button
        if st.button("🗑️ Clear Comparison"):
            clear_comparison()
            st.rerun()

        # Comparison chart
        render_comparison_chart(comparison_results)

        # Comparison table
        render_comparison_table(comparison_results)

        # Detailed comparison
        st.subheader("Detailed Comparison")

        comparison_data = []
        for results in comparison_results:
            m = results["metrics"]
            comparison_data.append(
                {
                    "Strategy": results["strategy"],
                    "Period": f"{results['start_date'].strftime('%Y-%m-%d')} to {results['end_date'].strftime('%Y-%m-%d')}",
                    "Initial Capital": f"${m['initial_capital']:,.0f}",
                    "Final Equity": f"${m['final_equity']:,.0f}",
                    "Total Return": f"{m['total_return']:.2%}",
                    "Annualized Return": f"{m.get('annualized_return', 0):.2%}"
                    if m.get("annualized_return")
                    else "N/A",
                    "Sharpe Ratio": f"{m.get('sharpe_ratio', 0):.2f}"
                    if m.get("sharpe_ratio")
                    else "N/A",
                    "Sortino Ratio": f"{m.get('sortino_ratio', 0):.2f}"
                    if m.get("sortino_ratio")
                    else "N/A",
                    "Max Drawdown": f"{m.get('max_drawdown', 0):.2%}"
                    if m.get("max_drawdown")
                    else "N/A",
                    "Win Rate": f"{m.get('win_rate', 0):.1%}"
                    if m.get("win_rate")
                    else "N/A",
                    "Profit Factor": f"{m.get('profit_factor', 0):.2f}",
                    "Total Trades": m["total_trades"],
                    "Avg Trade": f"${m.get('avg_trade', 0):,.2f}",
                }
            )

        df = pd.DataFrame(comparison_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("Add strategies to compare them side by side.")


# Sidebar - Database connection
st.sidebar.title("Database")
db_path = st.sidebar.text_input(
    "Dolt Database Path",
    value=st.session_state.get("db_path", DEFAULT_DB_PATH),
)
st.session_state.db_path = db_path

is_connected, status_msg = check_database_connection(db_path)
if is_connected:
    st.sidebar.success(f"✅ {status_msg}")
else:
    st.sidebar.error(f"❌ {status_msg}")

# Saved results
st.sidebar.header("Saved Results")
saved = get_all_saved_results()
st.sidebar.metric("Saved Results", len(saved))

if saved:
    for name in list(saved.keys())[:5]:  # Show first 5
        st.sidebar.text(f"• {name}")
    if len(saved) > 5:
        st.sidebar.text(f"... and {len(saved) - 5} more")

# Main page content
st.title("🔧 Strategy Builder")
st.markdown("*Build, configure, and compare options strategies*")

# Main content tabs
tab1, tab2 = st.tabs(["🔧 Configure Strategy", "📊 Compare Strategies"])

with tab1:
    config = render_strategy_wizard()

    # Run button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        run_button = st.button(
            "🚀 Run Backtest",
            type="primary",
            use_container_width=True,
            disabled=not is_connected,
        )

    with col2:
        add_to_compare = st.checkbox("Add to comparison", value=True)

    if run_button:
        with st.spinner(f"Running {config['strategy']} backtest..."):
            try:
                results = run_real_backtest(
                    config["strategy"],
                    config["start_date"],
                    config["end_date"],
                    config["initial_capital"],
                    config["params"],
                    db_path,
                )

                # Save results
                st.session_state.results = results

                # Add to comparison if checked
                if add_to_compare:
                    add_to_comparison(results)

                st.success(
                    f"✅ Backtest completed! {results.get('trading_days', 0)} trading days processed."
                )

                # Show quick results
                st.subheader("Quick Results")
                render_metrics_cards(results["metrics"])

                # Option to save
                col1, col2 = st.columns([3, 1])
                with col1:
                    save_name = st.text_input(
                        "Save as:",
                        value=f"{config['strategy']}_{config['start_date'].strftime('%Y%m%d')}",
                    )
                with col2:
                    if st.button("💾 Save"):
                        save_result(save_name, results)
                        st.success(f"Saved as '{save_name}'")

            except Exception as e:
                st.error(f"Backtest failed: {str(e)}")

with tab2:
    render_comparison_section()
