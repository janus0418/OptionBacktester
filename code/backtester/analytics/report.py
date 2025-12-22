"""
ReportGenerator Class for Options Backtesting Analytics

This module provides the ReportGenerator class for creating comprehensive
backtest reports in various formats (HTML, PDF, Markdown).

Key Features:
    - HTML reports with embedded charts and interactive tables
    - PDF reports using static matplotlib charts
    - Formatted summary tables (HTML, LaTeX, Markdown)
    - Executive summary generation
    - Monthly returns tables
    - Detailed trade logs

Design Philosophy:
    Reports are designed to be comprehensive, professional, and self-contained.
    They combine metrics, charts, and tables into a single document suitable
    for sharing and archival.

Output Formats:
    - HTML: Interactive, can be viewed in browser
    - PDF: Static, suitable for printing and sharing (requires weasyprint)
    - Markdown: For documentation and README files

Usage:
    from backtester.analytics.report import ReportGenerator

    # Generate HTML report
    path = ReportGenerator.generate_html_report(
        backtest_results, metrics, 'report.html'
    )

    # Generate summary table
    table = ReportGenerator.generate_summary_table(metrics, format='markdown')

References:
    - HTML/CSS report best practices
    - Financial report standards
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
import base64
from io import BytesIO

import numpy as np
import pandas as pd

# Configure module logger
logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Report styling
REPORT_CSS = """
<style>
    @page {
        size: A4;
        margin: 20mm;
    }
    * {
        box-sizing: border-box;
    }
    body {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
                     'Helvetica Neue', Arial, sans-serif;
        font-size: 14px;
        line-height: 1.6;
        color: #333;
        background-color: #fff;
        margin: 0;
        padding: 30px;
        max-width: 1200px;
        margin: 0 auto;
    }
    h1 {
        font-size: 28px;
        color: #1a1a2e;
        border-bottom: 3px solid #16213e;
        padding-bottom: 15px;
        margin-bottom: 25px;
    }
    h2 {
        font-size: 20px;
        color: #16213e;
        border-bottom: 1px solid #e0e0e0;
        padding-bottom: 10px;
        margin-top: 35px;
        margin-bottom: 20px;
    }
    h3 {
        font-size: 16px;
        color: #333;
        margin-top: 25px;
        margin-bottom: 15px;
    }
    .header {
        text-align: center;
        margin-bottom: 40px;
        padding: 30px;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        color: white;
        border-radius: 8px;
    }
    .header h1 {
        color: white;
        border: none;
        margin: 0 0 10px 0;
        font-size: 32px;
    }
    .header p {
        margin: 5px 0;
        opacity: 0.9;
    }
    .summary-box {
        background-color: #f8f9fa;
        border-left: 4px solid #16213e;
        padding: 20px;
        margin: 25px 0;
        border-radius: 0 8px 8px 0;
    }
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 15px;
        margin: 20px 0;
    }
    .metric-box {
        background-color: #fff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
    }
    .metric-label {
        font-size: 12px;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: 600;
        color: #1a1a2e;
        margin-top: 5px;
    }
    .metric-value.positive {
        color: #2e7d32;
    }
    .metric-value.negative {
        color: #c62828;
    }
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 20px 0;
        font-size: 13px;
    }
    th {
        background-color: #16213e;
        color: white;
        padding: 12px;
        text-align: left;
        font-weight: 600;
    }
    td {
        padding: 10px 12px;
        border-bottom: 1px solid #e0e0e0;
    }
    tr:nth-child(even) {
        background-color: #f8f9fa;
    }
    tr:hover {
        background-color: #f0f0f0;
    }
    .text-right {
        text-align: right;
    }
    .text-center {
        text-align: center;
    }
    .positive {
        color: #2e7d32;
    }
    .negative {
        color: #c62828;
    }
    .chart-container {
        margin: 25px 0;
        text-align: center;
    }
    .chart-container img {
        max-width: 100%;
        height: auto;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
    }
    .section {
        margin: 30px 0;
        page-break-inside: avoid;
    }
    .footer {
        margin-top: 50px;
        padding-top: 20px;
        border-top: 1px solid #e0e0e0;
        text-align: center;
        font-size: 12px;
        color: #666;
    }
    .two-column {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 30px;
    }
    @media print {
        body {
            padding: 0;
        }
        .header {
            background: #16213e !important;
            -webkit-print-color-adjust: exact;
            print-color-adjust: exact;
        }
        th {
            background-color: #16213e !important;
            -webkit-print-color-adjust: exact;
            print-color-adjust: exact;
        }
    }
</style>
"""


# =============================================================================
# Exceptions
# =============================================================================

class ReportError(Exception):
    """Base exception for report generation errors."""
    pass


class ReportDataError(ReportError):
    """Exception raised for missing or invalid data."""
    pass


class ReportSaveError(ReportError):
    """Exception raised when saving report fails."""
    pass


# =============================================================================
# Helper Functions
# =============================================================================

def _ensure_directory(path: str) -> None:
    """Create directory for save path if it doesn't exist."""
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def _format_number(value: float, fmt: str = 'general') -> str:
    """Format number for display."""
    if pd.isna(value) or not np.isfinite(value):
        return 'N/A'

    if fmt == 'percent':
        return f'{value:.2%}'
    elif fmt == 'currency':
        return f'${value:,.2f}'
    elif fmt == 'currency_int':
        return f'${value:,.0f}'
    elif fmt == 'ratio':
        return f'{value:.2f}'
    elif fmt == 'integer':
        return f'{int(value):,}'
    else:
        if abs(value) >= 1000:
            return f'{value:,.2f}'
        return f'{value:.4f}'


def _get_value_class(value: float, threshold: float = 0) -> str:
    """Get CSS class based on value sign."""
    if pd.isna(value):
        return ''
    return 'positive' if value > threshold else ('negative' if value < threshold else '')


def _fig_to_base64(fig) -> str:
    """Convert matplotlib figure to base64 encoded PNG."""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return img_base64


# =============================================================================
# ReportGenerator Class
# =============================================================================

class ReportGenerator:
    """
    Generate comprehensive backtest reports.

    This class provides static methods for creating professional reports
    in various formats including HTML, PDF, and formatted tables.

    Methods:
        generate_html_report: Full HTML report with charts
        generate_pdf_report: PDF version of the report
        generate_summary_table: Formatted metrics table

    Example:
        >>> path = ReportGenerator.generate_html_report(
        ...     results, metrics, 'report.html'
        ... )
        >>> table = ReportGenerator.generate_summary_table(
        ...     metrics, format='markdown'
        ... )
    """

    # =========================================================================
    # HTML Report
    # =========================================================================

    @staticmethod
    def generate_html_report(
        backtest_results: Dict[str, Any],
        metrics: Dict[str, Any],
        save_path: str = 'backtest_report.html',
        include_charts: bool = True,
        title: str = 'Backtest Report'
    ) -> str:
        """
        Generate comprehensive HTML report.

        Creates a professional HTML report with:
        1. Executive Summary with key metrics
        2. Performance Metrics Table
        3. Risk Metrics Table
        4. Charts (equity, drawdown, P&L distribution, Greeks)
        5. Trade Log Table
        6. Monthly Returns Table

        Args:
            backtest_results: Dictionary from BacktestEngine.run() containing:
                - 'equity_curve': DataFrame with equity time series
                - 'trade_log': DataFrame with trade records
                - 'greeks_history': DataFrame with Greeks over time
            metrics: Dictionary from BacktestEngine.calculate_metrics():
                - 'performance': Performance metrics
                - 'risk': Risk metrics
                - 'summary': Summary metrics
            save_path: Path to save HTML report. Default 'backtest_report.html'.
            include_charts: If True, include embedded charts. Default True.
            title: Report title.

        Returns:
            Absolute path to saved HTML file.

        Raises:
            ReportDataError: If required data is missing
            ReportSaveError: If saving fails

        Example:
            >>> results = engine.run()
            >>> metrics = engine.calculate_metrics()
            >>> path = ReportGenerator.generate_html_report(
            ...     results, metrics, 'report.html'
            ... )
        """
        # Validate inputs
        if backtest_results is None:
            raise ReportDataError("backtest_results cannot be None")
        if metrics is None:
            raise ReportDataError("metrics cannot be None")

        # Extract data
        equity_curve = backtest_results.get('equity_curve', pd.DataFrame())
        trade_log = backtest_results.get('trade_log', pd.DataFrame())
        greeks_history = backtest_results.get('greeks_history', pd.DataFrame())

        # Get metrics
        perf = metrics.get('performance', {})
        risk = metrics.get('risk', {})
        summary = metrics.get('summary', {})

        # Build HTML sections
        html_parts = []

        # Document header
        html_parts.append(f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{title}</title>
    {REPORT_CSS}
</head>
<body>
""")

        # Report header
        strategy_name = backtest_results.get('strategy_stats', {}).get('name', 'Strategy')
        start_date = backtest_results.get('start_date', '')
        end_date = backtest_results.get('end_date', '')
        underlying = backtest_results.get('underlying', '')

        if hasattr(start_date, 'strftime'):
            start_date = start_date.strftime('%Y-%m-%d')
        if hasattr(end_date, 'strftime'):
            end_date = end_date.strftime('%Y-%m-%d')

        html_parts.append(f"""
    <div class="header">
        <h1>{title}</h1>
        <p><strong>Strategy:</strong> {strategy_name} | <strong>Underlying:</strong> {underlying}</p>
        <p><strong>Period:</strong> {start_date} to {end_date}</p>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
""")

        # Executive Summary
        exec_summary = ReportGenerator._create_executive_summary(
            backtest_results, summary, perf
        )
        html_parts.append(exec_summary)

        # Key Metrics Grid
        metrics_grid = ReportGenerator._create_metrics_grid(summary, perf, risk)
        html_parts.append(metrics_grid)

        # Charts section
        if include_charts and not equity_curve.empty:
            charts_html = ReportGenerator._create_charts_section(
                equity_curve, trade_log, greeks_history
            )
            html_parts.append(charts_html)

        # Performance Metrics Table
        perf_table = ReportGenerator._create_performance_table(perf)
        html_parts.append(f"""
    <div class="section">
        <h2>Performance Metrics</h2>
        {perf_table}
    </div>
""")

        # Risk Metrics Table
        risk_table = ReportGenerator._create_risk_table(risk)
        html_parts.append(f"""
    <div class="section">
        <h2>Risk Metrics</h2>
        {risk_table}
    </div>
""")

        # Monthly Returns Table
        if not equity_curve.empty and 'equity' in equity_curve.columns:
            monthly_table = ReportGenerator._create_monthly_returns_table(equity_curve)
            html_parts.append(f"""
    <div class="section">
        <h2>Monthly Returns</h2>
        {monthly_table}
    </div>
""")

        # Trade Log Table
        if not trade_log.empty:
            trade_table = ReportGenerator._create_trade_log_table(trade_log)
            html_parts.append(f"""
    <div class="section">
        <h2>Trade Log</h2>
        {trade_table}
    </div>
""")

        # Footer
        html_parts.append("""
    <div class="footer">
        <p>Generated by Options Backtester - Analytics Module</p>
        <p>This report is for informational purposes only and does not constitute financial advice.</p>
    </div>
</body>
</html>
""")

        # Combine and save
        html_content = ''.join(html_parts)

        try:
            _ensure_directory(save_path)
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            abs_path = os.path.abspath(save_path)
            logger.info(f"HTML report saved to {abs_path}")
            return abs_path
        except Exception as e:
            raise ReportSaveError(f"Failed to save HTML report: {e}")

    # =========================================================================
    # PDF Report
    # =========================================================================

    @staticmethod
    def generate_pdf_report(
        backtest_results: Dict[str, Any],
        metrics: Dict[str, Any],
        save_path: str = 'backtest_report.pdf',
        title: str = 'Backtest Report'
    ) -> str:
        """
        Generate PDF report (static version).

        Uses matplotlib for charts and weasyprint for PDF generation.
        If weasyprint is not available, generates HTML and provides
        instructions for manual conversion.

        Args:
            backtest_results: Dictionary from BacktestEngine.run().
            metrics: Dictionary from BacktestEngine.calculate_metrics().
            save_path: Path to save PDF report.
            title: Report title.

        Returns:
            Absolute path to saved PDF file.

        Raises:
            ReportDataError: If required data is missing
            ReportSaveError: If saving fails

        Example:
            >>> path = ReportGenerator.generate_pdf_report(
            ...     results, metrics, 'report.pdf'
            ... )
        """
        # First generate HTML with embedded matplotlib charts
        html_path = save_path.replace('.pdf', '_temp.html')

        # Generate HTML report with matplotlib charts instead of plotly
        html_path = ReportGenerator.generate_html_report(
            backtest_results, metrics, html_path,
            include_charts=True, title=title
        )

        # Try to convert to PDF using weasyprint
        try:
            from weasyprint import HTML
            _ensure_directory(save_path)
            HTML(filename=html_path).write_pdf(save_path)
            abs_path = os.path.abspath(save_path)
            logger.info(f"PDF report saved to {abs_path}")

            # Clean up temp HTML file
            try:
                os.remove(html_path)
            except Exception:
                pass

            return abs_path

        except ImportError:
            logger.warning(
                "weasyprint not installed. PDF generation requires weasyprint. "
                "Install with: pip install weasyprint"
            )
            logger.info(f"HTML report available at: {html_path}")
            return html_path

        except Exception as e:
            raise ReportSaveError(f"Failed to generate PDF: {e}")

    # =========================================================================
    # Summary Table
    # =========================================================================

    @staticmethod
    def generate_summary_table(
        metrics: Dict[str, Any],
        format: str = 'html'
    ) -> str:
        """
        Generate formatted summary table of all metrics.

        Creates a table with key performance and risk metrics in
        the specified format (HTML, LaTeX, or Markdown).

        Args:
            metrics: Metrics dictionary from calculate_metrics().
            format: Output format - 'html', 'latex', or 'markdown'.

        Returns:
            Formatted table string.

        Raises:
            ValueError: If format is not supported

        Example:
            >>> table = ReportGenerator.generate_summary_table(
            ...     metrics, format='markdown'
            ... )
            >>> print(table)
        """
        format = format.lower()
        if format not in ['html', 'latex', 'markdown']:
            raise ValueError(f"Format must be 'html', 'latex', or 'markdown', got '{format}'")

        # Extract metrics
        summary = metrics.get('summary', {})
        perf = metrics.get('performance', {})
        risk = metrics.get('risk', {})

        # Define metrics to include
        metric_rows = [
            ('Total Return', summary.get('total_return_pct', np.nan), 'percent'),
            ('Annualized Return', perf.get('annualized_return', np.nan), 'percent'),
            ('Sharpe Ratio', summary.get('sharpe_ratio', np.nan), 'ratio'),
            ('Sortino Ratio', perf.get('sortino_ratio', np.nan), 'ratio'),
            ('Calmar Ratio', perf.get('calmar_ratio', np.nan), 'ratio'),
            ('Max Drawdown', summary.get('max_drawdown', np.nan), 'percent'),
            ('Volatility (Ann.)', perf.get('volatility', np.nan), 'percent'),
            ('Win Rate', summary.get('win_rate', np.nan), 'percent'),
            ('Profit Factor', summary.get('profit_factor', np.nan), 'ratio'),
            ('Expectancy', summary.get('expectancy', np.nan), 'currency'),
            ('Total Trades', summary.get('total_trades', 0), 'integer'),
            ('VaR (95%)', risk.get('var_95_historical', np.nan), 'percent'),
            ('CVaR (95%)', risk.get('cvar_95', np.nan), 'percent'),
            ('Skewness', risk.get('skewness', np.nan), 'ratio'),
            ('Kurtosis', risk.get('kurtosis', np.nan), 'ratio'),
        ]

        if format == 'html':
            return ReportGenerator._format_table_html(metric_rows)
        elif format == 'latex':
            return ReportGenerator._format_table_latex(metric_rows)
        else:  # markdown
            return ReportGenerator._format_table_markdown(metric_rows)

    @staticmethod
    def _format_table_html(rows: List[Tuple]) -> str:
        """Format metrics as HTML table."""
        html = '<table>\n<tr><th>Metric</th><th>Value</th></tr>\n'
        for name, value, fmt in rows:
            formatted = _format_number(value, fmt)
            html += f'<tr><td>{name}</td><td>{formatted}</td></tr>\n'
        html += '</table>'
        return html

    @staticmethod
    def _format_table_latex(rows: List[Tuple]) -> str:
        """Format metrics as LaTeX table."""
        latex = '\\begin{tabular}{lr}\n\\hline\n'
        latex += '\\textbf{Metric} & \\textbf{Value} \\\\\n\\hline\n'
        for name, value, fmt in rows:
            formatted = _format_number(value, fmt)
            # Escape special LaTeX characters
            formatted = formatted.replace('%', '\\%').replace('$', '\\$')
            latex += f'{name} & {formatted} \\\\\n'
        latex += '\\hline\n\\end{tabular}'
        return latex

    @staticmethod
    def _format_table_markdown(rows: List[Tuple]) -> str:
        """Format metrics as Markdown table."""
        md = '| Metric | Value |\n|--------|-------|\n'
        for name, value, fmt in rows:
            formatted = _format_number(value, fmt)
            md += f'| {name} | {formatted} |\n'
        return md

    # =========================================================================
    # Helper Methods - Sections
    # =========================================================================

    @staticmethod
    def _create_executive_summary(
        results: Dict,
        summary: Dict,
        perf: Dict
    ) -> str:
        """Create executive summary section."""
        total_return = summary.get('total_return_pct', 0)
        sharpe = summary.get('sharpe_ratio', 0)
        max_dd = summary.get('max_drawdown', 0)
        win_rate = summary.get('win_rate', 0)
        total_trades = summary.get('total_trades', 0)
        initial_equity = summary.get('initial_equity', 100000)
        final_equity = summary.get('final_equity', 100000)

        return_sign = 'gained' if total_return >= 0 else 'lost'
        return_color = 'positive' if total_return >= 0 else 'negative'

        return f"""
    <div class="section">
        <h2>Executive Summary</h2>
        <div class="summary-box">
            <p>
                The strategy <span class="{return_color}">{return_sign} {abs(total_return):.2%}</span>
                over the backtest period, transforming an initial investment of
                <strong>{_format_number(initial_equity, 'currency')}</strong> into
                <strong>{_format_number(final_equity, 'currency')}</strong>.
            </p>
            <p>
                With a Sharpe ratio of <strong>{sharpe:.2f}</strong> and maximum drawdown of
                <span class="negative">{max_dd:.2%}</span>, the strategy completed
                <strong>{total_trades:,}</strong> trades with a win rate of
                <strong>{win_rate:.1%}</strong>.
            </p>
        </div>
    </div>
"""

    @staticmethod
    def _create_metrics_grid(summary: Dict, perf: Dict, risk: Dict) -> str:
        """Create key metrics grid."""
        metrics = [
            ('Total Return', summary.get('total_return_pct', 0), 'percent', True),
            ('Sharpe Ratio', summary.get('sharpe_ratio', 0), 'ratio', True),
            ('Max Drawdown', summary.get('max_drawdown', 0), 'percent', False),
            ('Win Rate', summary.get('win_rate', 0), 'percent', True),
            ('Profit Factor', summary.get('profit_factor', 0), 'ratio', True),
            ('VaR (95%)', risk.get('var_95_historical', 0), 'percent', False),
        ]

        boxes = []
        for name, value, fmt, is_higher_better in metrics:
            formatted = _format_number(value, fmt)
            if fmt == 'percent':
                css_class = _get_value_class(value if is_higher_better else -value)
            else:
                css_class = _get_value_class(value - 1 if fmt == 'ratio' else value)

            boxes.append(f"""
            <div class="metric-box">
                <div class="metric-label">{name}</div>
                <div class="metric-value {css_class}">{formatted}</div>
            </div>
""")

        return f"""
    <div class="section">
        <h2>Key Metrics</h2>
        <div class="metrics-grid">
            {''.join(boxes)}
        </div>
    </div>
"""

    @staticmethod
    def _create_charts_section(
        equity_curve: pd.DataFrame,
        trade_log: pd.DataFrame,
        greeks_history: pd.DataFrame
    ) -> str:
        """Create charts section with matplotlib figures embedded as base64."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        charts_html = '<div class="section"><h2>Performance Charts</h2>'

        # Equity Curve
        if 'equity' in equity_curve.columns:
            fig, axes = plt.subplots(2, 1, figsize=(10, 6),
                                     gridspec_kw={'height_ratios': [3, 1]})

            # Equity
            ax1 = axes[0]
            equity = equity_curve['equity']
            ax1.plot(equity.index, equity.values, color='#1976D2', linewidth=1.5)
            ax1.fill_between(equity.index, equity.values, equity.values.min(),
                           alpha=0.1, color='#1976D2')
            ax1.set_ylabel('Equity ($)')
            ax1.set_title('Equity Curve', fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, p: f'${x:,.0f}')
            )

            # Drawdown
            ax2 = axes[1]
            running_max = equity.expanding().max()
            drawdown = (equity - running_max) / running_max * 100
            ax2.fill_between(drawdown.index, drawdown.values, 0,
                           color='#C62828', alpha=0.5)
            ax2.set_ylabel('Drawdown (%)')
            ax2.set_xlabel('Date')
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            img_base64 = _fig_to_base64(fig)
            plt.close(fig)

            charts_html += f"""
        <div class="chart-container">
            <img src="data:image/png;base64,{img_base64}" alt="Equity Curve">
        </div>
"""

        # P&L Distribution
        if not trade_log.empty and 'realized_pnl' in trade_log.columns:
            closed = trade_log[trade_log['action'] == 'close']
            if not closed.empty:
                pnl = closed['realized_pnl'].dropna()
                if len(pnl) > 1:
                    fig, ax = plt.subplots(figsize=(10, 4))

                    colors = ['#2E7D32' if x >= 0 else '#C62828' for x in pnl.values]
                    ax.hist(pnl.values, bins=30, color='#1976D2', alpha=0.7,
                           edgecolor='white')
                    ax.axvline(pnl.mean(), color='orange', linestyle='--',
                              linewidth=2, label=f'Mean: ${pnl.mean():,.0f}')
                    ax.axvline(0, color='black', linewidth=0.5)
                    ax.set_xlabel('P&L ($)')
                    ax.set_ylabel('Frequency')
                    ax.set_title('Trade P&L Distribution', fontsize=12, fontweight='bold')
                    ax.legend()
                    ax.grid(True, alpha=0.3)

                    plt.tight_layout()
                    img_base64 = _fig_to_base64(fig)
                    plt.close(fig)

                    charts_html += f"""
        <div class="chart-container">
            <img src="data:image/png;base64,{img_base64}" alt="P&L Distribution">
        </div>
"""

        charts_html += '</div>'
        return charts_html

    @staticmethod
    def _create_performance_table(perf: Dict) -> str:
        """Create performance metrics table."""
        metrics = [
            ('Total Return', perf.get('total_return_pct', np.nan), 'percent'),
            ('Annualized Return (CAGR)', perf.get('annualized_return', np.nan), 'percent'),
            ('Volatility (Annualized)', perf.get('volatility', np.nan), 'percent'),
            ('Sharpe Ratio', perf.get('sharpe_ratio', np.nan), 'ratio'),
            ('Sortino Ratio', perf.get('sortino_ratio', np.nan), 'ratio'),
            ('Calmar Ratio', perf.get('calmar_ratio', np.nan), 'ratio'),
            ('Max Drawdown', perf.get('max_drawdown', np.nan), 'percent'),
            ('Max DD Duration (days)', perf.get('max_drawdown_duration', np.nan), 'integer'),
            ('Win Rate', perf.get('win_rate', np.nan), 'percent'),
            ('Profit Factor', perf.get('profit_factor', np.nan), 'ratio'),
            ('Payoff Ratio', perf.get('payoff_ratio', np.nan), 'ratio'),
            ('Average Win', perf.get('average_win', np.nan), 'currency'),
            ('Average Loss', perf.get('average_loss', np.nan), 'currency'),
            ('Average Trade', perf.get('average_trade', np.nan), 'currency'),
            ('Expectancy', perf.get('expectancy', np.nan), 'currency'),
            ('Max Consecutive Wins', perf.get('max_consecutive_wins', np.nan), 'integer'),
            ('Max Consecutive Losses', perf.get('max_consecutive_losses', np.nan), 'integer'),
            ('Total Trades', perf.get('total_trades', np.nan), 'integer'),
        ]

        rows = []
        for name, value, fmt in metrics:
            formatted = _format_number(value, fmt)
            rows.append(f'<tr><td>{name}</td><td class="text-right">{formatted}</td></tr>')

        return f"""
        <table>
            <tr><th>Metric</th><th class="text-right">Value</th></tr>
            {''.join(rows)}
        </table>
"""

    @staticmethod
    def _create_risk_table(risk: Dict) -> str:
        """Create risk metrics table."""
        metrics = [
            ('VaR (95%, Historical)', risk.get('var_95_historical', np.nan), 'percent'),
            ('VaR (95%, Parametric)', risk.get('var_95_parametric', np.nan), 'percent'),
            ('VaR (99%, Historical)', risk.get('var_99_historical', np.nan), 'percent'),
            ('CVaR / Expected Shortfall (95%)', risk.get('cvar_95', np.nan), 'percent'),
            ('CVaR / Expected Shortfall (99%)', risk.get('cvar_99', np.nan), 'percent'),
            ('Skewness', risk.get('skewness', np.nan), 'ratio'),
            ('Kurtosis', risk.get('kurtosis', np.nan), 'ratio'),
            ('Downside Deviation', risk.get('downside_deviation', np.nan), 'percent'),
        ]

        rows = []
        for name, value, fmt in metrics:
            formatted = _format_number(value, fmt)
            rows.append(f'<tr><td>{name}</td><td class="text-right">{formatted}</td></tr>')

        return f"""
        <table>
            <tr><th>Metric</th><th class="text-right">Value</th></tr>
            {''.join(rows)}
        </table>
"""

    @staticmethod
    def _create_monthly_returns_table(equity_curve: pd.DataFrame) -> str:
        """Create monthly returns table."""
        if 'equity' not in equity_curve.columns:
            return '<p>No equity data available</p>'

        equity = equity_curve['equity']

        # Ensure datetime index
        if not isinstance(equity.index, pd.DatetimeIndex):
            try:
                equity.index = pd.to_datetime(equity.index)
            except Exception:
                return '<p>Could not parse dates</p>'

        # Calculate monthly returns
        # Use 'M' for compatibility with older pandas versions (< 2.0)
        try:
            monthly = equity.resample('ME').last()
        except ValueError:
            monthly = equity.resample('M').last()
        monthly_ret = monthly.pct_change().dropna() * 100

        if len(monthly_ret) < 1:
            return '<p>Insufficient data for monthly returns</p>'

        # Create pivot
        monthly_df = pd.DataFrame({
            'year': monthly_ret.index.year,
            'month': monthly_ret.index.month,
            'return': monthly_ret.values
        })

        pivot = monthly_df.pivot(index='year', columns='month', values='return')

        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        # Build table
        header = '<tr><th>Year</th>'
        for m in range(1, 13):
            header += f'<th class="text-center">{month_names[m-1]}</th>'
        header += '<th class="text-center">Annual</th></tr>'

        rows = []
        for year in pivot.index:
            row = f'<tr><td><strong>{year}</strong></td>'
            annual_ret = 0
            for m in range(1, 13):
                val = pivot.loc[year, m] if m in pivot.columns else np.nan
                if pd.notna(val):
                    css_class = _get_value_class(val)
                    row += f'<td class="text-center {css_class}">{val:.1f}%</td>'
                    annual_ret += val / 100  # Approximate
                else:
                    row += '<td class="text-center">-</td>'

            # Annual (compounded approximation)
            css_class = _get_value_class(annual_ret)
            row += f'<td class="text-center {css_class}"><strong>{annual_ret*100:.1f}%</strong></td>'
            row += '</tr>'
            rows.append(row)

        return f"""
        <table>
            {header}
            {''.join(rows)}
        </table>
        <p style="font-size: 12px; color: #666;">Values shown as percentage returns</p>
"""

    @staticmethod
    def _create_trade_log_table(trade_log: pd.DataFrame, max_rows: int = 50) -> str:
        """Create trade log table."""
        closed = trade_log[trade_log['action'] == 'close'].copy()

        if closed.empty:
            return '<p>No closed trades</p>'

        # Sort by timestamp
        if 'timestamp' in closed.columns:
            closed = closed.sort_values('timestamp')

        rows = []
        for i, (_, trade) in enumerate(closed.iterrows()):
            if i >= max_rows:
                break

            ts = trade.get('timestamp', '')
            if hasattr(ts, 'strftime'):
                ts = ts.strftime('%Y-%m-%d')

            pnl = trade.get('realized_pnl', 0)
            pnl_class = _get_value_class(pnl)

            rows.append(f"""
            <tr>
                <td>{trade.get('trade_id', '-')}</td>
                <td>{ts}</td>
                <td>{trade.get('structure_type', '-')}</td>
                <td>{trade.get('underlying', '-')}</td>
                <td class="text-right">{trade.get('num_legs', '-')}</td>
                <td class="text-right {pnl_class}">{_format_number(pnl, 'currency')}</td>
                <td>{trade.get('exit_reason', '-')}</td>
            </tr>
""")

        return f"""
        <table>
            <tr>
                <th>ID</th>
                <th>Date</th>
                <th>Structure</th>
                <th>Underlying</th>
                <th class="text-right">Legs</th>
                <th class="text-right">P&L</th>
                <th>Exit Reason</th>
            </tr>
            {''.join(rows)}
        </table>
        <p style="font-size: 12px; color: #666;">Showing first {min(len(closed), max_rows)} of {len(closed)} trades</p>
"""


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Main class
    'ReportGenerator',

    # Exceptions
    'ReportError',
    'ReportDataError',
    'ReportSaveError',
]
