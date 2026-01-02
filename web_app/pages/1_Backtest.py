"""
Options Strategy Backtester - Backtest Page

Configure and run backtests with various options strategies.
"""

import streamlit as st

# Page configuration MUST be first Streamlit command
st.set_page_config(
    page_title="Run Backtest - Options Backtester",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple
import pandas as pd
import logging

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
    set_results,
    save_result,
)
from shared.charts import (
    render_equity_curve,
    render_metrics_cards,
    render_returns_distribution,
    render_monthly_returns,
    render_detailed_metrics,
    render_trades_table,
)

# Initialize session state
init_session_state()


def render_sidebar() -> Tuple[str, datetime, datetime, float, Dict[str, Any], str]:
    """Render sidebar configuration and return user selections."""
    st.sidebar.title("🚀 Backtest Configuration")

    # Database configuration
    st.sidebar.header("Database")
    db_path = st.sidebar.text_input(
        "Dolt Database Path",
        value=st.session_state.get("db_path", DEFAULT_DB_PATH),
        help="Path to your Dolt options database",
    )
    st.session_state.db_path = db_path

    # Check connection
    is_connected, status_msg = check_database_connection(db_path)
    st.session_state.is_connected = is_connected
    st.session_state.connection_status = status_msg

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


# Main page content
st.title("🚀 Run Backtest")
st.markdown("*Configure and execute options strategy backtests*")

# Render sidebar and get configuration
strategy, start_date, end_date, capital, params, db_path = render_sidebar()

# Check if database is connected
is_connected = st.session_state.get("is_connected", False)

# Run button
col1, col2 = st.sidebar.columns(2)
with col1:
    run_button = st.button(
        "🚀 Run Backtest",
        type="primary",
        use_container_width=True,
        disabled=not is_connected,
    )
with col2:
    clear_button = st.button(
        "🗑️ Clear",
        use_container_width=True,
    )

if clear_button:
    st.session_state.results = None
    st.rerun()

# Run backtest
if run_button:
    if not is_connected:
        st.error("Please configure a valid database connection first")
    else:
        with st.spinner(f"Running {strategy} backtest... This may take a few minutes."):
            try:
                results = run_real_backtest(
                    strategy, start_date, end_date, capital, params, db_path
                )
                set_results(results)
                st.success(
                    f"✅ Backtest completed! Processed {results.get('trading_days', 0)} trading days."
                )
            except Exception as e:
                st.error(f"Backtest failed: {str(e)}")
                logging.exception("Backtest error")

# Display results or instructions
results = st.session_state.get("results")

if results is None:
    st.info("Configure your strategy in the sidebar and click 'Run Backtest' to begin.")

    st.subheader("Quick Tips")
    st.markdown("""
    1. **Database**: Ensure your Dolt database path is correct and connected
    2. **Strategy**: Choose a strategy that matches your market outlook
    3. **Period**: Select a date range with sufficient data (at least 30 days)
    4. **Parameters**: Adjust parameters based on your risk tolerance
    
    ⚠️ **Note**: This runs REAL backtests using the BacktestEngine with actual historical data!
    """)

    # Strategy comparison
    st.subheader("Strategy Overview")
    strategy_data = []
    for name, info in STRATEGY_INFO.items():
        strategy_data.append(
            {
                "Strategy": name,
                "Risk Level": info["risk"],
                "Ideal Conditions": info["ideal_conditions"],
            }
        )
    st.dataframe(pd.DataFrame(strategy_data), use_container_width=True, hide_index=True)
else:
    # Display results
    st.header(f"📊 {results['strategy']} Results")
    st.caption(
        f"Period: {results['start_date'].strftime('%Y-%m-%d')} to {results['end_date'].strftime('%Y-%m-%d')} | "
        f"Initial Capital: ${results['metrics']['initial_capital']:,.0f} | "
        f"Trading Days: {results.get('trading_days', 'N/A')}"
    )

    # Key metrics cards
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

    # Detailed metrics
    st.subheader("Detailed Metrics")
    render_detailed_metrics(results["metrics"])

    # Trade history
    st.subheader("Trade History")
    render_trades_table(results)

    # Parameters used
    st.subheader("Strategy Parameters Used")
    params_df = pd.DataFrame([results["params"]])
    st.dataframe(params_df, use_container_width=True, hide_index=True)

    # Save results section
    st.subheader("Save Results")
    col1, col2 = st.columns([3, 1])
    with col1:
        save_name = st.text_input(
            "Result Name",
            value=f"{results['strategy']}_{results['start_date'].strftime('%Y%m%d')}",
            help="Enter a name to save this result",
        )
    with col2:
        if st.button("💾 Save", use_container_width=True):
            save_result(save_name, results)
            st.success(f"Saved as '{save_name}'")

    # Navigation
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📈 View Detailed Results →"):
            st.switch_page("pages/2_Results.py")
    with col2:
        if st.button("🔧 Compare Strategies →"):
            st.switch_page("pages/3_Strategy_Builder.py")
