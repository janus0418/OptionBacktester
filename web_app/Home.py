"""
Options Strategy Backtester - Home Page

Main entry point for the multi-page Streamlit application.
"""

import streamlit as st

# Page configuration MUST be first Streamlit command
st.set_page_config(
    page_title="Options Strategy Backtester",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

import sys
from pathlib import Path

# Add paths for imports
web_app_path = Path(__file__).parent
sys.path.insert(0, str(web_app_path))
sys.path.insert(0, str(web_app_path.parent / "code"))

# Now import shared utilities
from shared.utils import (
    STRATEGY_INFO,
    DEFAULT_DB_PATH,
    check_database_connection,
    init_session_state,
)

# Initialize session state
init_session_state()

# Sidebar - Database connection
st.sidebar.title("Database Connection")
db_path = st.sidebar.text_input(
    "Dolt Database Path",
    value=st.session_state.get("db_path", DEFAULT_DB_PATH),
    help="Path to your Dolt options database",
)
st.session_state.db_path = db_path

is_connected, status_msg = check_database_connection(db_path)
st.session_state.is_connected = is_connected
st.session_state.connection_status = status_msg

if is_connected:
    st.sidebar.success(f"✅ {status_msg}")
else:
    st.sidebar.error(f"❌ {status_msg}")

# Main content
st.title("📊 Options Strategy Backtester")
st.markdown("*Professional-grade options strategy backtesting with real market data*")

st.header("Welcome")
st.markdown("""
This application provides a professional-grade interface for backtesting options strategies 
using real historical market data.

**Features:**
- Real backtesting with historical options data
- Multiple pre-built strategies (Short Straddle, Iron Condor, Volatility Regime)
- Comprehensive performance analytics
- Interactive charts and visualizations
- Strategy comparison tools
""")

# Quick navigation cards
st.header("Quick Start")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🚀 Run Backtest")
    st.markdown("""
    Configure and run a backtest with your chosen strategy.
    
    - Select strategy type
    - Set date range and capital
    - Adjust strategy parameters
    - View real-time results
    """)
    if st.button("Go to Backtest →", key="goto_backtest"):
        st.switch_page("pages/1_Backtest.py")

with col2:
    st.subheader("📈 View Results")
    st.markdown("""
    Analyze your backtest results in detail.
    
    - Equity curves and drawdown
    - Performance metrics
    - Trade history
    - Monthly returns heatmap
    """)
    if st.button("Go to Results →", key="goto_results"):
        st.switch_page("pages/2_Results.py")

with col3:
    st.subheader("🔧 Strategy Builder")
    st.markdown("""
    Build and compare custom strategies.
    
    - Parameter wizard
    - Strategy comparison
    - Save configurations
    - Export results
    """)
    if st.button("Go to Strategy Builder →", key="goto_builder"):
        st.switch_page("pages/3_Strategy_Builder.py")

# Available strategies section
st.header("Available Strategies")

for name, info in STRATEGY_INFO.items():
    with st.expander(f"**{name}** - Risk: {info['risk']}"):
        st.write(f"**Description:** {info['description']}")
        st.write(f"**Ideal Conditions:** {info['ideal_conditions']}")
        st.write(f"**Parameters:** {', '.join(info['params'])}")

# Current session status
st.header("Session Status")

col1, col2, col3 = st.columns(3)

with col1:
    current_results = st.session_state.get("results")
    if current_results:
        st.success(f"✅ Active backtest: {current_results['strategy']}")
        st.caption(
            f"Period: {current_results['start_date'].strftime('%Y-%m-%d')} to {current_results['end_date'].strftime('%Y-%m-%d')}"
        )
    else:
        st.info("No active backtest results")

with col2:
    saved_results = st.session_state.get("saved_results", {})
    st.metric("Saved Results", len(saved_results))

with col3:
    comparison_results = st.session_state.get("comparison_results", [])
    st.metric("In Comparison", len(comparison_results))

# Footer
st.markdown("---")
st.markdown("""
**Documentation:** [User Guide](../docs/USER_GUIDE.md) | [API Reference](../docs/API_REFERENCE.md) | [Strategy Guide](../docs/STRATEGY_DEVELOPMENT_GUIDE.md)

*Built with real options data from Dolt database.*
""")
