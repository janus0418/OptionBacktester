"""Shared utilities for multi-page Streamlit app."""

from .utils import (
    STRATEGY_INFO,
    DEFAULT_DB_PATH,
    check_database_connection,
    run_real_backtest,
    init_session_state,
    get_results,
    set_results,
    clear_results,
    get_all_saved_results,
    save_result,
    delete_saved_result,
    add_to_comparison,
    clear_comparison,
    get_comparison_results,
)
from .charts import (
    render_equity_curve,
    render_returns_distribution,
    render_monthly_returns,
    render_trades_table,
    render_metrics_cards,
    render_detailed_metrics,
)

__all__ = [
    "STRATEGY_INFO",
    "DEFAULT_DB_PATH",
    "check_database_connection",
    "run_real_backtest",
    "init_session_state",
    "get_results",
    "set_results",
    "clear_results",
    "get_all_saved_results",
    "save_result",
    "delete_saved_result",
    "add_to_comparison",
    "clear_comparison",
    "get_comparison_results",
    "render_equity_curve",
    "render_returns_distribution",
    "render_monthly_returns",
    "render_trades_table",
    "render_metrics_cards",
    "render_detailed_metrics",
]
