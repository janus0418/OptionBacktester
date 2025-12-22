"""
Visualization Class for Options Backtesting Analytics

This module provides the Visualization class for creating professional-grade
charts and plots for backtest analysis. It supports both static (matplotlib)
and interactive (plotly) outputs.

Key Features:
    - Equity curve plotting with drawdown shading
    - Drawdown analysis charts
    - P&L distribution histograms
    - Greeks evolution over time
    - Option payoff diagrams
    - Trade entry/exit visualization
    - Monthly returns heatmap
    - Rolling Sharpe ratio
    - Returns distribution analysis

Design Philosophy:
    All methods are static to enable easy use without instantiation.
    Methods support dual backends (matplotlib and plotly) for flexibility.
    Output can be displayed, saved to file, or returned as Figure objects.

Styling Standards:
    - Matplotlib: Uses seaborn style for professional appearance
    - Plotly: Interactive with hover tooltips, zoom/pan, and export
    - Color schemes: Colorblind-friendly (green/red for profit/loss)

Usage:
    from backtester.analytics.visualization import Visualization

    # Plot equity curve
    fig = Visualization.plot_equity_curve(
        equity_curve, backend='plotly', show=True
    )

    # Plot P&L distribution
    fig = Visualization.plot_pnl_distribution(
        trades, backend='matplotlib', save_path='pnl_dist.png'
    )

References:
    - Matplotlib: https://matplotlib.org/stable/
    - Plotly: https://plotly.com/python/
    - Seaborn: https://seaborn.pydata.org/
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

# Configure module logger
logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Default figure sizes
DEFAULT_FIGSIZE = (12, 6)
DEFAULT_FIGSIZE_WIDE = (14, 6)
DEFAULT_FIGSIZE_TALL = (12, 8)
DEFAULT_FIGSIZE_HEATMAP = (14, 8)

# Color schemes (colorblind-friendly)
COLOR_PROFIT = '#2E7D32'       # Dark green
COLOR_LOSS = '#C62828'         # Dark red
COLOR_NEUTRAL = '#1565C0'      # Blue
COLOR_EQUITY = '#1976D2'       # Blue
COLOR_DRAWDOWN = '#D32F2F'     # Red
COLOR_BENCHMARK = '#757575'    # Gray
COLOR_HIGHLIGHT = '#FF9800'    # Orange
COLOR_GRID = '#E0E0E0'         # Light gray

# Plotly color scheme
PLOTLY_COLORS = {
    'profit': 'rgb(46, 125, 50)',
    'loss': 'rgb(198, 40, 40)',
    'neutral': 'rgb(21, 101, 192)',
    'equity': 'rgb(25, 118, 210)',
    'drawdown': 'rgb(211, 47, 47)',
    'benchmark': 'rgb(117, 117, 117)',
    'highlight': 'rgb(255, 152, 0)',
}

# Supported backends
SUPPORTED_BACKENDS = ['matplotlib', 'plotly']

# Default rolling window for Sharpe
DEFAULT_ROLLING_WINDOW = 63  # ~3 months

# File format mapping
FILE_FORMATS = {
    '.png': 'png',
    '.svg': 'svg',
    '.pdf': 'pdf',
    '.jpg': 'jpeg',
    '.jpeg': 'jpeg',
    '.html': 'html',
}


# =============================================================================
# Exceptions
# =============================================================================

class VisualizationError(Exception):
    """Base exception for visualization errors."""
    pass


class InvalidBackendError(VisualizationError):
    """Exception raised for invalid backend specification."""
    pass


class InsufficientDataError(VisualizationError):
    """Exception raised when there is insufficient data for visualization."""
    pass


class SaveError(VisualizationError):
    """Exception raised when saving figure fails."""
    pass


# =============================================================================
# Helper Functions
# =============================================================================

def _ensure_directory(path: str) -> None:
    """Create directory for save path if it doesn't exist."""
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def _validate_backend(backend: str) -> str:
    """Validate and normalize backend string."""
    backend = backend.lower().strip()
    if backend not in SUPPORTED_BACKENDS:
        raise InvalidBackendError(
            f"Backend must be one of {SUPPORTED_BACKENDS}, got '{backend}'"
        )
    return backend


def _get_save_format(save_path: str) -> str:
    """Determine file format from save path extension."""
    ext = os.path.splitext(save_path)[1].lower()
    return FILE_FORMATS.get(ext, 'png')


def _setup_matplotlib_style() -> None:
    """Setup matplotlib with professional styling."""
    import matplotlib.pyplot as plt
    try:
        import seaborn as sns
        sns.set_style("whitegrid")
        sns.set_context("notebook", font_scale=1.1)
    except ImportError:
        plt.style.use('seaborn-v0_8-whitegrid')


# =============================================================================
# Visualization Class
# =============================================================================

class Visualization:
    """
    Create charts and plots for backtest results.

    This class provides static methods for creating professional-grade
    visualizations for analyzing options trading backtest results. All
    methods support both matplotlib (static) and plotly (interactive)
    backends.

    Methods:
        plot_equity_curve: Plot portfolio equity over time
        plot_drawdown: Plot drawdown percentage over time
        plot_pnl_distribution: Histogram of trade P&L
        plot_greeks_over_time: Greeks evolution chart
        plot_payoff_diagram: Option structure payoff at expiry
        plot_entry_exit_points: Trade markers on price chart
        plot_monthly_returns: Monthly returns heatmap
        plot_rolling_sharpe: Rolling Sharpe ratio over time
        plot_returns_distribution: Daily returns histogram with Q-Q

    Example:
        >>> from backtester.analytics.visualization import Visualization
        >>> fig = Visualization.plot_equity_curve(equity_curve, backend='plotly')
        >>> fig = Visualization.plot_pnl_distribution(trades, save_path='pnl.png')
    """

    # =========================================================================
    # Equity Curve
    # =========================================================================

    @staticmethod
    def plot_equity_curve(
        equity_curve: pd.DataFrame,
        save_path: Optional[str] = None,
        show: bool = True,
        backend: str = 'plotly',
        title: str = 'Portfolio Equity Curve',
        show_drawdown: bool = True,
        show_benchmark: bool = True,
        initial_capital: Optional[float] = None
    ) -> Optional[Any]:
        """
        Plot equity curve over time.

        Creates a line chart showing portfolio equity evolution with optional
        drawdown shading and initial capital reference line.

        Args:
            equity_curve: DataFrame with 'equity' column and datetime index.
                         Can also be a Series of equity values.
            save_path: Path to save figure. If None, figure is not saved.
            show: If True, display the figure. Default True.
            backend: 'matplotlib' or 'plotly'. Default 'plotly'.
            title: Chart title. Default 'Portfolio Equity Curve'.
            show_drawdown: If True, shade drawdown periods in red.
            show_benchmark: If True, show initial capital as horizontal line.
            initial_capital: Initial capital for benchmark line. If None,
                           uses first equity value.

        Returns:
            Figure object (matplotlib.Figure or plotly.graph_objects.Figure),
            or None if only showing/saving.

        Raises:
            InsufficientDataError: If equity_curve has fewer than 2 points
            InvalidBackendError: If backend is not supported

        Example:
            >>> fig = Visualization.plot_equity_curve(
            ...     equity_curve, backend='plotly', show=True
            ... )
        """
        backend = _validate_backend(backend)

        # Handle Series vs DataFrame
        if isinstance(equity_curve, pd.Series):
            equity = equity_curve.copy()
            equity_df = pd.DataFrame({'equity': equity})
        elif isinstance(equity_curve, pd.DataFrame):
            if 'equity' not in equity_curve.columns:
                raise ValueError("DataFrame must contain 'equity' column")
            equity = equity_curve['equity'].copy()
            equity_df = equity_curve.copy()
        else:
            raise ValueError(
                f"Expected DataFrame or Series, got {type(equity_curve).__name__}"
            )

        if len(equity) < 2:
            raise InsufficientDataError(
                f"Need at least 2 data points, got {len(equity)}"
            )

        # Set initial capital
        if initial_capital is None:
            initial_capital = float(equity.iloc[0])

        # Calculate drawdown series
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max

        if backend == 'matplotlib':
            return Visualization._plot_equity_curve_matplotlib(
                equity, drawdown, initial_capital, save_path, show, title,
                show_drawdown, show_benchmark
            )
        else:  # plotly
            return Visualization._plot_equity_curve_plotly(
                equity, drawdown, initial_capital, save_path, show, title,
                show_drawdown, show_benchmark
            )

    @staticmethod
    def _plot_equity_curve_matplotlib(
        equity: pd.Series,
        drawdown: pd.Series,
        initial_capital: float,
        save_path: Optional[str],
        show: bool,
        title: str,
        show_drawdown: bool,
        show_benchmark: bool
    ) -> Optional[Any]:
        """Matplotlib implementation of equity curve plot."""
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        _setup_matplotlib_style()

        fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)

        # Plot equity line
        ax.plot(
            equity.index, equity.values,
            color=COLOR_EQUITY, linewidth=1.5, label='Equity'
        )

        # Shade drawdown periods
        if show_drawdown:
            ax.fill_between(
                equity.index,
                equity.values,
                running_max := equity.expanding().max().values,
                where=(equity.values < running_max),
                color=COLOR_DRAWDOWN,
                alpha=0.3,
                label='Drawdown'
            )

        # Initial capital benchmark
        if show_benchmark:
            ax.axhline(
                y=initial_capital,
                color=COLOR_BENCHMARK,
                linestyle='--',
                linewidth=1,
                alpha=0.7,
                label=f'Initial Capital (${initial_capital:,.0f})'
            )

        # Formatting
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=11)
        ax.set_ylabel('Equity ($)', fontsize=11)
        ax.legend(loc='upper left', framealpha=0.9)
        ax.grid(True, alpha=0.3)

        # Format y-axis with dollar signs
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, p: f'${x:,.0f}')
        )

        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.xticks(rotation=45)

        plt.tight_layout()

        # Save
        if save_path:
            _ensure_directory(save_path)
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved equity curve to {save_path}")

        # Show
        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    @staticmethod
    def _plot_equity_curve_plotly(
        equity: pd.Series,
        drawdown: pd.Series,
        initial_capital: float,
        save_path: Optional[str],
        show: bool,
        title: str,
        show_drawdown: bool,
        show_benchmark: bool
    ) -> Optional[Any]:
        """Plotly implementation of equity curve plot."""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # Create figure with secondary y-axis for drawdown
        if show_drawdown:
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.7, 0.3]
            )
        else:
            fig = go.Figure()

        # Equity line
        equity_trace = go.Scatter(
            x=equity.index,
            y=equity.values,
            mode='lines',
            name='Equity',
            line=dict(color=PLOTLY_COLORS['equity'], width=2),
            hovertemplate='Date: %{x}<br>Equity: $%{y:,.2f}<extra></extra>'
        )

        if show_drawdown:
            fig.add_trace(equity_trace, row=1, col=1)
        else:
            fig.add_trace(equity_trace)

        # Benchmark line
        if show_benchmark:
            benchmark_trace = go.Scatter(
                x=[equity.index[0], equity.index[-1]],
                y=[initial_capital, initial_capital],
                mode='lines',
                name=f'Initial Capital (${initial_capital:,.0f})',
                line=dict(color=PLOTLY_COLORS['benchmark'], dash='dash', width=1),
                hoverinfo='skip'
            )
            if show_drawdown:
                fig.add_trace(benchmark_trace, row=1, col=1)
            else:
                fig.add_trace(benchmark_trace)

        # Drawdown area
        if show_drawdown:
            dd_trace = go.Scatter(
                x=drawdown.index,
                y=drawdown.values * 100,
                mode='lines',
                name='Drawdown %',
                fill='tozeroy',
                line=dict(color=PLOTLY_COLORS['drawdown'], width=1),
                fillcolor='rgba(211, 47, 47, 0.3)',
                hovertemplate='Date: %{x}<br>Drawdown: %{y:.2f}%<extra></extra>'
            )
            fig.add_trace(dd_trace, row=2, col=1)

        # Layout
        layout_kwargs = dict(
            title=dict(text=title, x=0.5, xanchor='center', font=dict(size=16)),
            hovermode='x unified',
            showlegend=True,
            legend=dict(yanchor='top', y=0.99, xanchor='left', x=0.01),
            template='plotly_white'
        )

        if show_drawdown:
            layout_kwargs.update(
                yaxis=dict(title='Equity ($)', tickformat='$,.0f'),
                yaxis2=dict(title='Drawdown (%)', tickformat='.1f'),
                xaxis2=dict(title='Date')
            )
        else:
            layout_kwargs.update(
                xaxis=dict(title='Date'),
                yaxis=dict(title='Equity ($)', tickformat='$,.0f')
            )

        fig.update_layout(**layout_kwargs)

        # Save
        if save_path:
            _ensure_directory(save_path)
            fmt = _get_save_format(save_path)
            if fmt == 'html':
                fig.write_html(save_path)
            else:
                fig.write_image(save_path, scale=2)
            logger.info(f"Saved equity curve to {save_path}")

        # Show
        if show:
            fig.show()

        return fig

    # =========================================================================
    # Drawdown Plot
    # =========================================================================

    @staticmethod
    def plot_drawdown(
        equity_curve: pd.DataFrame,
        save_path: Optional[str] = None,
        show: bool = True,
        backend: str = 'plotly',
        title: str = 'Portfolio Drawdown'
    ) -> Optional[Any]:
        """
        Plot drawdown over time.

        Creates an area chart showing drawdown percentage with the maximum
        drawdown period highlighted.

        Args:
            equity_curve: DataFrame with 'equity' column and datetime index.
            save_path: Path to save figure. If None, figure is not saved.
            show: If True, display the figure. Default True.
            backend: 'matplotlib' or 'plotly'. Default 'plotly'.
            title: Chart title. Default 'Portfolio Drawdown'.

        Returns:
            Figure object or None if only showing/saving.

        Raises:
            InsufficientDataError: If equity_curve has fewer than 2 points

        Example:
            >>> fig = Visualization.plot_drawdown(equity_curve, show=True)
        """
        backend = _validate_backend(backend)

        # Handle Series vs DataFrame
        if isinstance(equity_curve, pd.Series):
            equity = equity_curve.copy()
        elif isinstance(equity_curve, pd.DataFrame):
            if 'equity' not in equity_curve.columns:
                raise ValueError("DataFrame must contain 'equity' column")
            equity = equity_curve['equity'].copy()
        else:
            raise ValueError(
                f"Expected DataFrame or Series, got {type(equity_curve).__name__}"
            )

        if len(equity) < 2:
            raise InsufficientDataError(
                f"Need at least 2 data points, got {len(equity)}"
            )

        # Calculate drawdown
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max * 100  # As percentage

        # Find max drawdown point
        max_dd_idx = drawdown.idxmin()
        max_dd_value = float(drawdown.loc[max_dd_idx])

        if backend == 'matplotlib':
            return Visualization._plot_drawdown_matplotlib(
                drawdown, max_dd_idx, max_dd_value, save_path, show, title
            )
        else:
            return Visualization._plot_drawdown_plotly(
                drawdown, max_dd_idx, max_dd_value, save_path, show, title
            )

    @staticmethod
    def _plot_drawdown_matplotlib(
        drawdown: pd.Series,
        max_dd_idx: Any,
        max_dd_value: float,
        save_path: Optional[str],
        show: bool,
        title: str
    ) -> Optional[Any]:
        """Matplotlib implementation of drawdown plot."""
        import matplotlib.pyplot as plt
        _setup_matplotlib_style()

        fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)

        # Area plot
        ax.fill_between(
            drawdown.index, drawdown.values, 0,
            color=COLOR_DRAWDOWN, alpha=0.5
        )
        ax.plot(
            drawdown.index, drawdown.values,
            color=COLOR_DRAWDOWN, linewidth=1
        )

        # Mark max drawdown
        ax.scatter(
            [max_dd_idx], [max_dd_value],
            color=COLOR_HIGHLIGHT, s=100, zorder=5,
            label=f'Max DD: {max_dd_value:.2f}%'
        )

        # Zero line
        ax.axhline(y=0, color='black', linewidth=0.5)

        # Formatting
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=11)
        ax.set_ylabel('Drawdown (%)', fontsize=11)
        ax.legend(loc='lower left', framealpha=0.9)
        ax.grid(True, alpha=0.3)

        # Format y-axis
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, p: f'{x:.1f}%')
        )

        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_path:
            _ensure_directory(save_path)
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved drawdown plot to {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    @staticmethod
    def _plot_drawdown_plotly(
        drawdown: pd.Series,
        max_dd_idx: Any,
        max_dd_value: float,
        save_path: Optional[str],
        show: bool,
        title: str
    ) -> Optional[Any]:
        """Plotly implementation of drawdown plot."""
        import plotly.graph_objects as go

        fig = go.Figure()

        # Area trace
        fig.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            mode='lines',
            name='Drawdown',
            fill='tozeroy',
            line=dict(color=PLOTLY_COLORS['drawdown'], width=1),
            fillcolor='rgba(211, 47, 47, 0.4)',
            hovertemplate='Date: %{x}<br>Drawdown: %{y:.2f}%<extra></extra>'
        ))

        # Max drawdown marker
        fig.add_trace(go.Scatter(
            x=[max_dd_idx],
            y=[max_dd_value],
            mode='markers+text',
            name=f'Max DD: {max_dd_value:.2f}%',
            marker=dict(color=PLOTLY_COLORS['highlight'], size=12),
            text=[f'{max_dd_value:.2f}%'],
            textposition='bottom center',
            hoverinfo='skip'
        ))

        # Zero line
        fig.add_hline(y=0, line=dict(color='black', width=0.5))

        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor='center', font=dict(size=16)),
            xaxis=dict(title='Date'),
            yaxis=dict(title='Drawdown (%)', tickformat='.1f'),
            hovermode='x unified',
            showlegend=True,
            template='plotly_white'
        )

        if save_path:
            _ensure_directory(save_path)
            fmt = _get_save_format(save_path)
            if fmt == 'html':
                fig.write_html(save_path)
            else:
                fig.write_image(save_path, scale=2)
            logger.info(f"Saved drawdown plot to {save_path}")

        if show:
            fig.show()

        return fig

    # =========================================================================
    # P&L Distribution
    # =========================================================================

    @staticmethod
    def plot_pnl_distribution(
        trades: pd.DataFrame,
        save_path: Optional[str] = None,
        show: bool = True,
        backend: str = 'plotly',
        title: str = 'Trade P&L Distribution',
        bins: int = 30,
        show_normal: bool = True,
        show_stats: bool = True
    ) -> Optional[Any]:
        """
        Plot P&L distribution histogram.

        Creates a histogram of trade P&L with optional normal distribution
        overlay and statistical markers (mean, median, VaR).

        Args:
            trades: DataFrame with 'realized_pnl' column for closed trades.
            save_path: Path to save figure. If None, figure is not saved.
            show: If True, display the figure. Default True.
            backend: 'matplotlib' or 'plotly'. Default 'plotly'.
            title: Chart title. Default 'Trade P&L Distribution'.
            bins: Number of histogram bins. Default 30.
            show_normal: If True, overlay normal distribution curve.
            show_stats: If True, show mean, median, and VaR markers.

        Returns:
            Figure object or None if only showing/saving.

        Raises:
            InsufficientDataError: If no trade data available
            KeyError: If 'realized_pnl' column not found

        Example:
            >>> fig = Visualization.plot_pnl_distribution(
            ...     trades, backend='plotly', show_normal=True
            ... )
        """
        backend = _validate_backend(backend)

        if trades.empty:
            raise InsufficientDataError("No trade data provided")

        if 'realized_pnl' not in trades.columns:
            raise KeyError("trades DataFrame must contain 'realized_pnl' column")

        # Get P&L values
        pnl = trades['realized_pnl'].dropna()

        if len(pnl) < 2:
            raise InsufficientDataError(
                f"Need at least 2 trades, got {len(pnl)}"
            )

        # Calculate statistics
        mean_pnl = float(pnl.mean())
        median_pnl = float(pnl.median())
        std_pnl = float(pnl.std())
        var_95 = float(np.percentile(pnl, 5))

        if backend == 'matplotlib':
            return Visualization._plot_pnl_dist_matplotlib(
                pnl, mean_pnl, median_pnl, std_pnl, var_95,
                save_path, show, title, bins, show_normal, show_stats
            )
        else:
            return Visualization._plot_pnl_dist_plotly(
                pnl, mean_pnl, median_pnl, std_pnl, var_95,
                save_path, show, title, bins, show_normal, show_stats
            )

    @staticmethod
    def _plot_pnl_dist_matplotlib(
        pnl: pd.Series,
        mean_pnl: float,
        median_pnl: float,
        std_pnl: float,
        var_95: float,
        save_path: Optional[str],
        show: bool,
        title: str,
        bins: int,
        show_normal: bool,
        show_stats: bool
    ) -> Optional[Any]:
        """Matplotlib implementation of P&L distribution plot."""
        import matplotlib.pyplot as plt
        _setup_matplotlib_style()

        fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)

        # Histogram with win/loss coloring
        n, bin_edges, patches = ax.hist(
            pnl.values, bins=bins, density=True,
            alpha=0.7, edgecolor='white'
        )

        # Color bins by profit/loss
        for patch, left_edge in zip(patches, bin_edges[:-1]):
            if left_edge >= 0:
                patch.set_facecolor(COLOR_PROFIT)
            else:
                patch.set_facecolor(COLOR_LOSS)

        # Normal distribution overlay
        if show_normal and std_pnl > 0:
            x = np.linspace(pnl.min(), pnl.max(), 100)
            normal_pdf = stats.norm.pdf(x, mean_pnl, std_pnl)
            ax.plot(x, normal_pdf, color='black', linestyle='--',
                   linewidth=2, label='Normal Distribution')

        # Statistical markers
        if show_stats:
            ax.axvline(mean_pnl, color='blue', linestyle='-', linewidth=1.5,
                      label=f'Mean: ${mean_pnl:,.0f}')
            ax.axvline(median_pnl, color='orange', linestyle='-', linewidth=1.5,
                      label=f'Median: ${median_pnl:,.0f}')
            ax.axvline(var_95, color='red', linestyle=':', linewidth=1.5,
                      label=f'5% VaR: ${var_95:,.0f}')

        # Zero line
        ax.axvline(0, color='black', linewidth=0.5)

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('P&L ($)', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.legend(loc='upper right', framealpha=0.9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            _ensure_directory(save_path)
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved P&L distribution to {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    @staticmethod
    def _plot_pnl_dist_plotly(
        pnl: pd.Series,
        mean_pnl: float,
        median_pnl: float,
        std_pnl: float,
        var_95: float,
        save_path: Optional[str],
        show: bool,
        title: str,
        bins: int,
        show_normal: bool,
        show_stats: bool
    ) -> Optional[Any]:
        """Plotly implementation of P&L distribution plot."""
        import plotly.graph_objects as go

        fig = go.Figure()

        # Histogram
        fig.add_trace(go.Histogram(
            x=pnl.values,
            nbinsx=bins,
            histnorm='probability density',
            name='P&L Distribution',
            marker=dict(
                color=[PLOTLY_COLORS['profit'] if x >= 0 else PLOTLY_COLORS['loss']
                       for x in pnl.values],
                line=dict(color='white', width=1)
            ),
            hovertemplate='P&L: $%{x:,.0f}<br>Density: %{y:.4f}<extra></extra>'
        ))

        # Normal curve
        if show_normal and std_pnl > 0:
            x = np.linspace(float(pnl.min()), float(pnl.max()), 100)
            normal_pdf = stats.norm.pdf(x, mean_pnl, std_pnl)
            fig.add_trace(go.Scatter(
                x=x, y=normal_pdf,
                mode='lines',
                name='Normal Distribution',
                line=dict(color='black', dash='dash', width=2)
            ))

        # Statistical markers
        if show_stats:
            fig.add_vline(x=mean_pnl, line=dict(color='blue', width=2),
                         annotation_text=f'Mean: ${mean_pnl:,.0f}',
                         annotation_position='top left')
            fig.add_vline(x=median_pnl, line=dict(color='orange', width=2),
                         annotation_text=f'Median: ${median_pnl:,.0f}',
                         annotation_position='top right')
            fig.add_vline(x=var_95, line=dict(color='red', width=2, dash='dot'),
                         annotation_text=f'5% VaR: ${var_95:,.0f}',
                         annotation_position='bottom left')

        # Zero line
        fig.add_vline(x=0, line=dict(color='black', width=0.5))

        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor='center', font=dict(size=16)),
            xaxis=dict(title='P&L ($)', tickformat='$,.0f'),
            yaxis=dict(title='Density'),
            barmode='overlay',
            showlegend=True,
            template='plotly_white'
        )

        if save_path:
            _ensure_directory(save_path)
            fmt = _get_save_format(save_path)
            if fmt == 'html':
                fig.write_html(save_path)
            else:
                fig.write_image(save_path, scale=2)
            logger.info(f"Saved P&L distribution to {save_path}")

        if show:
            fig.show()

        return fig

    # =========================================================================
    # Greeks Over Time
    # =========================================================================

    @staticmethod
    def plot_greeks_over_time(
        greeks_history: pd.DataFrame,
        greek_names: Optional[List[str]] = None,
        save_path: Optional[str] = None,
        show: bool = True,
        backend: str = 'plotly',
        title: str = 'Portfolio Greeks Over Time'
    ) -> Optional[Any]:
        """
        Plot Greeks evolution over time.

        Creates a multi-line chart showing the evolution of portfolio Greeks
        (delta, gamma, theta, vega, rho) over the backtest period.

        Args:
            greeks_history: DataFrame with columns for each Greek and datetime index.
                          Standard columns: 'delta', 'gamma', 'theta', 'vega', 'rho'.
            greek_names: List of Greeks to plot. If None, plots all available.
            save_path: Path to save figure. If None, figure is not saved.
            show: If True, display the figure. Default True.
            backend: 'matplotlib' or 'plotly'. Default 'plotly'.
            title: Chart title. Default 'Portfolio Greeks Over Time'.

        Returns:
            Figure object or None if only showing/saving.

        Raises:
            InsufficientDataError: If greeks_history is empty
            KeyError: If specified greek_names not found

        Example:
            >>> fig = Visualization.plot_greeks_over_time(
            ...     greeks_history, greek_names=['delta', 'gamma']
            ... )
        """
        backend = _validate_backend(backend)

        if greeks_history.empty:
            raise InsufficientDataError("Greeks history is empty")

        # Determine which Greeks to plot
        available_greeks = ['delta', 'gamma', 'theta', 'vega', 'rho']
        if greek_names is None:
            greek_names = [g for g in available_greeks if g in greeks_history.columns]
        else:
            # Validate requested Greeks exist
            for g in greek_names:
                if g not in greeks_history.columns:
                    raise KeyError(f"Greek '{g}' not found in greeks_history")

        if not greek_names:
            raise InsufficientDataError("No Greeks columns found in data")

        if backend == 'matplotlib':
            return Visualization._plot_greeks_matplotlib(
                greeks_history, greek_names, save_path, show, title
            )
        else:
            return Visualization._plot_greeks_plotly(
                greeks_history, greek_names, save_path, show, title
            )

    @staticmethod
    def _plot_greeks_matplotlib(
        greeks_history: pd.DataFrame,
        greek_names: List[str],
        save_path: Optional[str],
        show: bool,
        title: str
    ) -> Optional[Any]:
        """Matplotlib implementation of Greeks plot."""
        import matplotlib.pyplot as plt
        _setup_matplotlib_style()

        n_greeks = len(greek_names)
        fig, axes = plt.subplots(n_greeks, 1, figsize=(12, 3 * n_greeks),
                                  sharex=True)

        if n_greeks == 1:
            axes = [axes]

        colors = ['#1976D2', '#7B1FA2', '#F57C00', '#388E3C', '#5D4037']

        for i, (greek, ax) in enumerate(zip(greek_names, axes)):
            values = greeks_history[greek]
            color = colors[i % len(colors)]

            ax.plot(values.index, values.values, color=color, linewidth=1.5)
            ax.fill_between(values.index, values.values, 0, alpha=0.2, color=color)
            ax.axhline(y=0, color='black', linewidth=0.5, linestyle='-')

            ax.set_ylabel(greek.capitalize(), fontsize=11)
            ax.grid(True, alpha=0.3)

        axes[0].set_title(title, fontsize=14, fontweight='bold')
        axes[-1].set_xlabel('Date', fontsize=11)

        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_path:
            _ensure_directory(save_path)
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved Greeks plot to {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    @staticmethod
    def _plot_greeks_plotly(
        greeks_history: pd.DataFrame,
        greek_names: List[str],
        save_path: Optional[str],
        show: bool,
        title: str
    ) -> Optional[Any]:
        """Plotly implementation of Greeks plot."""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        n_greeks = len(greek_names)
        fig = make_subplots(
            rows=n_greeks, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=[g.capitalize() for g in greek_names]
        )

        colors = ['#1976D2', '#7B1FA2', '#F57C00', '#388E3C', '#5D4037']

        for i, greek in enumerate(greek_names):
            values = greeks_history[greek]
            color = colors[i % len(colors)]

            fig.add_trace(go.Scatter(
                x=values.index,
                y=values.values,
                mode='lines',
                name=greek.capitalize(),
                line=dict(color=color, width=1.5),
                fill='tozeroy',
                fillcolor=f'rgba{tuple(list(int(color[i:i+2], 16) for i in (1, 3, 5)) + [0.2])}',
                hovertemplate=f'{greek.capitalize()}: %{{y:.4f}}<extra></extra>'
            ), row=i+1, col=1)

            # Zero line
            fig.add_hline(y=0, line=dict(color='black', width=0.5), row=i+1, col=1)

        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor='center', font=dict(size=16)),
            height=200 * n_greeks + 100,
            hovermode='x unified',
            showlegend=True,
            template='plotly_white'
        )

        fig.update_xaxes(title_text='Date', row=n_greeks, col=1)

        if save_path:
            _ensure_directory(save_path)
            fmt = _get_save_format(save_path)
            if fmt == 'html':
                fig.write_html(save_path)
            else:
                fig.write_image(save_path, scale=2)
            logger.info(f"Saved Greeks plot to {save_path}")

        if show:
            fig.show()

        return fig

    # =========================================================================
    # Payoff Diagram
    # =========================================================================

    @staticmethod
    def plot_payoff_diagram(
        structure: Any,
        spot_range: Optional[Tuple[float, float]] = None,
        save_path: Optional[str] = None,
        show: bool = True,
        backend: str = 'plotly',
        title: Optional[str] = None,
        current_spot: Optional[float] = None,
        num_points: int = 201
    ) -> Optional[Any]:
        """
        Plot option structure payoff at expiration.

        Creates a P&L vs spot price diagram showing the payoff profile
        of an option structure at expiration, with breakeven points and
        max profit/loss regions marked.

        Args:
            structure: OptionStructure instance with option legs.
            spot_range: Tuple of (min_spot, max_spot) for x-axis. If None,
                       auto-calculated based on strikes.
            save_path: Path to save figure. If None, figure is not saved.
            show: If True, display the figure. Default True.
            backend: 'matplotlib' or 'plotly'. Default 'plotly'.
            title: Chart title. If None, auto-generated from structure type.
            current_spot: Current spot price to mark. If None, not shown.
            num_points: Number of points for payoff curve. Default 201.

        Returns:
            Figure object or None if only showing/saving.

        Example:
            >>> fig = Visualization.plot_payoff_diagram(
            ...     short_straddle, current_spot=450, show=True
            ... )
        """
        backend = _validate_backend(backend)

        # Get payoff data from structure
        if hasattr(structure, 'get_payoff_diagram'):
            spots, payoffs = structure.get_payoff_diagram(spot_range, num_points)
        elif hasattr(structure, 'get_pnl_at_expiry'):
            if spot_range is None:
                strikes = [opt.strike for opt in structure.options]
                margin = max(strikes) * 0.2
                spot_range = (min(strikes) - margin, max(strikes) + margin)
            spots = np.linspace(spot_range[0], spot_range[1], num_points)
            payoffs = structure.get_pnl_at_expiry(spots)
        else:
            raise ValueError(
                "structure must have get_payoff_diagram or get_pnl_at_expiry method"
            )

        # Get breakeven points
        breakevens = []
        if hasattr(structure, 'calculate_breakeven_points'):
            try:
                breakevens = structure.calculate_breakeven_points(spot_range)
            except Exception:
                pass

        # Auto-generate title
        if title is None:
            struct_type = getattr(structure, 'structure_type', 'custom').upper()
            underlying = getattr(structure, 'underlying', 'N/A')
            title = f'{struct_type} Payoff Diagram - {underlying}'

        if backend == 'matplotlib':
            return Visualization._plot_payoff_matplotlib(
                spots, payoffs, breakevens, current_spot,
                save_path, show, title
            )
        else:
            return Visualization._plot_payoff_plotly(
                spots, payoffs, breakevens, current_spot,
                save_path, show, title
            )

    @staticmethod
    def _plot_payoff_matplotlib(
        spots: np.ndarray,
        payoffs: np.ndarray,
        breakevens: List[float],
        current_spot: Optional[float],
        save_path: Optional[str],
        show: bool,
        title: str
    ) -> Optional[Any]:
        """Matplotlib implementation of payoff diagram."""
        import matplotlib.pyplot as plt
        _setup_matplotlib_style()

        fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)

        # Profit/loss coloring
        profit_mask = payoffs >= 0
        loss_mask = payoffs < 0

        ax.fill_between(spots, payoffs, 0, where=profit_mask,
                       color=COLOR_PROFIT, alpha=0.3, label='Profit')
        ax.fill_between(spots, payoffs, 0, where=loss_mask,
                       color=COLOR_LOSS, alpha=0.3, label='Loss')

        # Payoff line
        ax.plot(spots, payoffs, color='black', linewidth=2)

        # Zero line
        ax.axhline(y=0, color='black', linewidth=0.5)

        # Breakeven points
        for be in breakevens:
            ax.axvline(x=be, color=COLOR_BENCHMARK, linestyle='--', linewidth=1)
            ax.annotate(f'BE: ${be:.2f}', xy=(be, 0),
                       xytext=(5, 10), textcoords='offset points',
                       fontsize=9, alpha=0.8)

        # Current spot
        if current_spot is not None:
            idx = np.abs(spots - current_spot).argmin()
            current_pnl = payoffs[idx]
            ax.scatter([current_spot], [current_pnl], color=COLOR_HIGHLIGHT,
                      s=100, zorder=5, label=f'Current: ${current_spot:.2f}')
            ax.axvline(x=current_spot, color=COLOR_HIGHLIGHT, linestyle=':',
                      linewidth=1, alpha=0.7)

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Spot Price ($)', fontsize=11)
        ax.set_ylabel('P&L ($)', fontsize=11)
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            _ensure_directory(save_path)
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved payoff diagram to {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    @staticmethod
    def _plot_payoff_plotly(
        spots: np.ndarray,
        payoffs: np.ndarray,
        breakevens: List[float],
        current_spot: Optional[float],
        save_path: Optional[str],
        show: bool,
        title: str
    ) -> Optional[Any]:
        """Plotly implementation of payoff diagram."""
        import plotly.graph_objects as go

        fig = go.Figure()

        # Profit region
        profit_y = np.where(payoffs >= 0, payoffs, 0)
        fig.add_trace(go.Scatter(
            x=spots, y=profit_y,
            mode='lines',
            name='Profit',
            fill='tozeroy',
            line=dict(color=PLOTLY_COLORS['profit'], width=0),
            fillcolor='rgba(46, 125, 50, 0.3)',
            hoverinfo='skip'
        ))

        # Loss region
        loss_y = np.where(payoffs < 0, payoffs, 0)
        fig.add_trace(go.Scatter(
            x=spots, y=loss_y,
            mode='lines',
            name='Loss',
            fill='tozeroy',
            line=dict(color=PLOTLY_COLORS['loss'], width=0),
            fillcolor='rgba(198, 40, 40, 0.3)',
            hoverinfo='skip'
        ))

        # Payoff line
        fig.add_trace(go.Scatter(
            x=spots, y=payoffs,
            mode='lines',
            name='P&L at Expiry',
            line=dict(color='black', width=2),
            hovertemplate='Spot: $%{x:.2f}<br>P&L: $%{y:,.2f}<extra></extra>'
        ))

        # Zero line
        fig.add_hline(y=0, line=dict(color='black', width=0.5))

        # Breakeven points
        for be in breakevens:
            fig.add_vline(x=be, line=dict(color=PLOTLY_COLORS['benchmark'],
                         dash='dash', width=1),
                         annotation_text=f'BE: ${be:.2f}',
                         annotation_position='top')

        # Current spot
        if current_spot is not None:
            idx = np.abs(spots - current_spot).argmin()
            current_pnl = payoffs[idx]
            fig.add_trace(go.Scatter(
                x=[current_spot], y=[current_pnl],
                mode='markers',
                name=f'Current: ${current_spot:.2f}',
                marker=dict(color=PLOTLY_COLORS['highlight'], size=12),
            ))
            fig.add_vline(x=current_spot,
                         line=dict(color=PLOTLY_COLORS['highlight'],
                                   dash='dot', width=1))

        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor='center', font=dict(size=16)),
            xaxis=dict(title='Spot Price ($)', tickformat='$,.0f'),
            yaxis=dict(title='P&L ($)', tickformat='$,.0f'),
            hovermode='x unified',
            showlegend=True,
            template='plotly_white'
        )

        if save_path:
            _ensure_directory(save_path)
            fmt = _get_save_format(save_path)
            if fmt == 'html':
                fig.write_html(save_path)
            else:
                fig.write_image(save_path, scale=2)
            logger.info(f"Saved payoff diagram to {save_path}")

        if show:
            fig.show()

        return fig

    # =========================================================================
    # Entry/Exit Points
    # =========================================================================

    @staticmethod
    def plot_entry_exit_points(
        underlying_prices: pd.Series,
        trades: pd.DataFrame,
        save_path: Optional[str] = None,
        show: bool = True,
        backend: str = 'plotly',
        title: str = 'Trade Entry/Exit Points'
    ) -> Optional[Any]:
        """
        Plot trade entry/exit points on underlying price chart.

        Creates a line chart of underlying price with markers showing
        trade entries (green arrows) and exits (red arrows), color-coded
        by profit/loss.

        Args:
            underlying_prices: Series of underlying prices with datetime index.
            trades: DataFrame with trade records. Expected columns:
                   - 'timestamp': Trade timestamp
                   - 'action': 'open' or 'close'
                   - 'realized_pnl': P&L for closed trades (optional)
            save_path: Path to save figure. If None, figure is not saved.
            show: If True, display the figure. Default True.
            backend: 'matplotlib' or 'plotly'. Default 'plotly'.
            title: Chart title. Default 'Trade Entry/Exit Points'.

        Returns:
            Figure object or None if only showing/saving.

        Example:
            >>> fig = Visualization.plot_entry_exit_points(
            ...     spy_prices, trades, show=True
            ... )
        """
        backend = _validate_backend(backend)

        if underlying_prices.empty:
            raise InsufficientDataError("Underlying prices is empty")

        if trades.empty:
            raise InsufficientDataError("No trade data provided")

        # Separate entries and exits
        entries = trades[trades['action'] == 'open'].copy()
        exits = trades[trades['action'] == 'close'].copy()

        if backend == 'matplotlib':
            return Visualization._plot_entry_exit_matplotlib(
                underlying_prices, entries, exits, save_path, show, title
            )
        else:
            return Visualization._plot_entry_exit_plotly(
                underlying_prices, entries, exits, save_path, show, title
            )

    @staticmethod
    def _plot_entry_exit_matplotlib(
        prices: pd.Series,
        entries: pd.DataFrame,
        exits: pd.DataFrame,
        save_path: Optional[str],
        show: bool,
        title: str
    ) -> Optional[Any]:
        """Matplotlib implementation of entry/exit plot."""
        import matplotlib.pyplot as plt
        _setup_matplotlib_style()

        fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE_WIDE)

        # Price line
        ax.plot(prices.index, prices.values, color=COLOR_NEUTRAL,
               linewidth=1, alpha=0.8, label='Underlying Price')

        # Entry markers
        for _, trade in entries.iterrows():
            ts = trade['timestamp']
            if ts in prices.index or (hasattr(prices.index, 'get_loc')):
                try:
                    price = prices.loc[ts] if ts in prices.index else prices.iloc[prices.index.get_indexer([ts], method='nearest')[0]]
                    ax.scatter([ts], [price], marker='^', s=100,
                              color=COLOR_PROFIT, zorder=5, edgecolors='white')
                except (KeyError, IndexError):
                    pass

        # Exit markers (color by P&L)
        for _, trade in exits.iterrows():
            ts = trade['timestamp']
            pnl = trade.get('realized_pnl', 0)
            color = COLOR_PROFIT if pnl >= 0 else COLOR_LOSS

            if ts in prices.index or (hasattr(prices.index, 'get_loc')):
                try:
                    price = prices.loc[ts] if ts in prices.index else prices.iloc[prices.index.get_indexer([ts], method='nearest')[0]]
                    ax.scatter([ts], [price], marker='v', s=100,
                              color=color, zorder=5, edgecolors='white')
                except (KeyError, IndexError):
                    pass

        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color=COLOR_NEUTRAL, linewidth=2, label='Price'),
            Line2D([0], [0], marker='^', color='w', markerfacecolor=COLOR_PROFIT,
                  markersize=10, label='Entry'),
            Line2D([0], [0], marker='v', color='w', markerfacecolor=COLOR_PROFIT,
                  markersize=10, label='Exit (Profit)'),
            Line2D([0], [0], marker='v', color='w', markerfacecolor=COLOR_LOSS,
                  markersize=10, label='Exit (Loss)'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', framealpha=0.9)

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=11)
        ax.set_ylabel('Price ($)', fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_path:
            _ensure_directory(save_path)
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved entry/exit plot to {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    @staticmethod
    def _plot_entry_exit_plotly(
        prices: pd.Series,
        entries: pd.DataFrame,
        exits: pd.DataFrame,
        save_path: Optional[str],
        show: bool,
        title: str
    ) -> Optional[Any]:
        """Plotly implementation of entry/exit plot."""
        import plotly.graph_objects as go

        fig = go.Figure()

        # Price line
        fig.add_trace(go.Scatter(
            x=prices.index, y=prices.values,
            mode='lines',
            name='Underlying Price',
            line=dict(color=PLOTLY_COLORS['neutral'], width=1.5),
            hovertemplate='Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
        ))

        # Entry markers
        entry_prices = []
        entry_times = []
        for _, trade in entries.iterrows():
            ts = trade['timestamp']
            try:
                if ts in prices.index:
                    price = prices.loc[ts]
                else:
                    idx = prices.index.get_indexer([ts], method='nearest')[0]
                    price = prices.iloc[idx]
                entry_times.append(ts)
                entry_prices.append(price)
            except (KeyError, IndexError):
                pass

        if entry_times:
            fig.add_trace(go.Scatter(
                x=entry_times, y=entry_prices,
                mode='markers',
                name='Entry',
                marker=dict(symbol='triangle-up', size=14,
                           color=PLOTLY_COLORS['profit'],
                           line=dict(color='white', width=1)),
                hovertemplate='Entry<br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
            ))

        # Exit markers (profit)
        exit_profit_prices = []
        exit_profit_times = []
        exit_loss_prices = []
        exit_loss_times = []

        for _, trade in exits.iterrows():
            ts = trade['timestamp']
            pnl = trade.get('realized_pnl', 0)
            try:
                if ts in prices.index:
                    price = prices.loc[ts]
                else:
                    idx = prices.index.get_indexer([ts], method='nearest')[0]
                    price = prices.iloc[idx]

                if pnl >= 0:
                    exit_profit_times.append(ts)
                    exit_profit_prices.append(price)
                else:
                    exit_loss_times.append(ts)
                    exit_loss_prices.append(price)
            except (KeyError, IndexError):
                pass

        if exit_profit_times:
            fig.add_trace(go.Scatter(
                x=exit_profit_times, y=exit_profit_prices,
                mode='markers',
                name='Exit (Profit)',
                marker=dict(symbol='triangle-down', size=14,
                           color=PLOTLY_COLORS['profit'],
                           line=dict(color='white', width=1)),
                hovertemplate='Exit (Profit)<br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
            ))

        if exit_loss_times:
            fig.add_trace(go.Scatter(
                x=exit_loss_times, y=exit_loss_prices,
                mode='markers',
                name='Exit (Loss)',
                marker=dict(symbol='triangle-down', size=14,
                           color=PLOTLY_COLORS['loss'],
                           line=dict(color='white', width=1)),
                hovertemplate='Exit (Loss)<br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
            ))

        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor='center', font=dict(size=16)),
            xaxis=dict(title='Date'),
            yaxis=dict(title='Price ($)', tickformat='$,.2f'),
            hovermode='x unified',
            showlegend=True,
            template='plotly_white'
        )

        if save_path:
            _ensure_directory(save_path)
            fmt = _get_save_format(save_path)
            if fmt == 'html':
                fig.write_html(save_path)
            else:
                fig.write_image(save_path, scale=2)
            logger.info(f"Saved entry/exit plot to {save_path}")

        if show:
            fig.show()

        return fig

    # =========================================================================
    # Monthly Returns Heatmap
    # =========================================================================

    @staticmethod
    def plot_monthly_returns(
        equity_curve: pd.DataFrame,
        save_path: Optional[str] = None,
        show: bool = True,
        backend: str = 'plotly',
        title: str = 'Monthly Returns Heatmap'
    ) -> Optional[Any]:
        """
        Plot monthly returns heatmap.

        Creates a heatmap with months as columns and years as rows,
        showing monthly return percentages with color intensity.

        Args:
            equity_curve: DataFrame with 'equity' column and datetime index.
            save_path: Path to save figure. If None, figure is not saved.
            show: If True, display the figure. Default True.
            backend: 'matplotlib' or 'plotly'. Default 'plotly'.
            title: Chart title. Default 'Monthly Returns Heatmap'.

        Returns:
            Figure object or None if only showing/saving.

        Example:
            >>> fig = Visualization.plot_monthly_returns(equity_curve, show=True)
        """
        backend = _validate_backend(backend)

        # Handle Series vs DataFrame
        if isinstance(equity_curve, pd.Series):
            equity = equity_curve.copy()
        elif isinstance(equity_curve, pd.DataFrame):
            if 'equity' not in equity_curve.columns:
                raise ValueError("DataFrame must contain 'equity' column")
            equity = equity_curve['equity'].copy()
        else:
            raise ValueError(
                f"Expected DataFrame or Series, got {type(equity_curve).__name__}"
            )

        if len(equity) < 2:
            raise InsufficientDataError(
                f"Need at least 2 data points, got {len(equity)}"
            )

        # Ensure datetime index
        if not isinstance(equity.index, pd.DatetimeIndex):
            equity.index = pd.to_datetime(equity.index)

        # Resample to monthly end-of-month and calculate returns
        # Use 'M' for compatibility with older pandas versions (< 2.0)
        try:
            monthly_equity = equity.resample('ME').last()
        except ValueError:
            monthly_equity = equity.resample('M').last()
        monthly_returns = monthly_equity.pct_change().dropna() * 100  # Percentage

        # Create pivot table
        monthly_df = pd.DataFrame({
            'year': monthly_returns.index.year,
            'month': monthly_returns.index.month,
            'return': monthly_returns.values
        })

        pivot = monthly_df.pivot(index='year', columns='month', values='return')

        # Rename columns to month names
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        pivot.columns = [month_names[m-1] for m in pivot.columns]

        if backend == 'matplotlib':
            return Visualization._plot_monthly_returns_matplotlib(
                pivot, save_path, show, title
            )
        else:
            return Visualization._plot_monthly_returns_plotly(
                pivot, save_path, show, title
            )

    @staticmethod
    def _plot_monthly_returns_matplotlib(
        pivot: pd.DataFrame,
        save_path: Optional[str],
        show: bool,
        title: str
    ) -> Optional[Any]:
        """Matplotlib implementation of monthly returns heatmap."""
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        _setup_matplotlib_style()

        fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE_HEATMAP)

        # Create diverging colormap (red-white-green)
        colors = [COLOR_LOSS, 'white', COLOR_PROFIT]
        n_bins = 100
        cmap = mcolors.LinearSegmentedColormap.from_list('RdWtGn', colors, N=n_bins)

        # Determine color range
        vmax = max(abs(pivot.min().min()), abs(pivot.max().max()))
        vmin = -vmax

        # Heatmap
        im = ax.imshow(pivot.values, cmap=cmap, aspect='auto',
                      vmin=vmin, vmax=vmax)

        # Labels
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, fontsize=10)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=10)

        # Add text annotations
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.iloc[i, j]
                if pd.notna(val):
                    text_color = 'white' if abs(val) > vmax * 0.5 else 'black'
                    ax.text(j, i, f'{val:.1f}%', ha='center', va='center',
                           fontsize=9, color=text_color)

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Return (%)', fontsize=11)

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Month', fontsize=11)
        ax.set_ylabel('Year', fontsize=11)

        plt.tight_layout()

        if save_path:
            _ensure_directory(save_path)
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved monthly returns heatmap to {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    @staticmethod
    def _plot_monthly_returns_plotly(
        pivot: pd.DataFrame,
        save_path: Optional[str],
        show: bool,
        title: str
    ) -> Optional[Any]:
        """Plotly implementation of monthly returns heatmap."""
        import plotly.graph_objects as go

        # Create text annotations
        text_matrix = [[f'{val:.1f}%' if pd.notna(val) else ''
                       for val in row] for row in pivot.values]

        # Determine color range
        vmax = max(abs(pivot.min().min()), abs(pivot.max().max()))

        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=list(pivot.columns),
            y=[str(y) for y in pivot.index],
            text=text_matrix,
            texttemplate='%{text}',
            textfont=dict(size=11),
            colorscale=[[0, 'rgb(198, 40, 40)'], [0.5, 'white'],
                       [1, 'rgb(46, 125, 50)']],
            zmin=-vmax,
            zmax=vmax,
            colorbar=dict(title='Return (%)', tickformat='.1f'),
            hovertemplate='%{y} %{x}<br>Return: %{z:.2f}%<extra></extra>'
        ))

        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor='center', font=dict(size=16)),
            xaxis=dict(title='Month', tickangle=0),
            yaxis=dict(title='Year', autorange='reversed'),
            template='plotly_white'
        )

        if save_path:
            _ensure_directory(save_path)
            fmt = _get_save_format(save_path)
            if fmt == 'html':
                fig.write_html(save_path)
            else:
                fig.write_image(save_path, scale=2)
            logger.info(f"Saved monthly returns heatmap to {save_path}")

        if show:
            fig.show()

        return fig

    # =========================================================================
    # Rolling Sharpe
    # =========================================================================

    @staticmethod
    def plot_rolling_sharpe(
        returns: pd.Series,
        window: int = DEFAULT_ROLLING_WINDOW,
        save_path: Optional[str] = None,
        show: bool = True,
        backend: str = 'plotly',
        title: str = 'Rolling Sharpe Ratio',
        risk_free_rate: float = 0.02
    ) -> Optional[Any]:
        """
        Plot rolling Sharpe ratio over time.

        Creates a line chart showing the rolling Sharpe ratio with
        benchmark lines at 0 and 1.

        Args:
            returns: Series of period returns with datetime index.
            window: Rolling window size in periods. Default 63 (~3 months).
            save_path: Path to save figure. If None, figure is not saved.
            show: If True, display the figure. Default True.
            backend: 'matplotlib' or 'plotly'. Default 'plotly'.
            title: Chart title. Default 'Rolling Sharpe Ratio'.
            risk_free_rate: Annual risk-free rate. Default 0.02 (2%).

        Returns:
            Figure object or None if only showing/saving.

        Example:
            >>> fig = Visualization.plot_rolling_sharpe(
            ...     returns, window=63, show=True
            ... )
        """
        backend = _validate_backend(backend)

        if not isinstance(returns, pd.Series):
            returns = pd.Series(returns)

        returns = returns.dropna()

        if len(returns) < window:
            raise InsufficientDataError(
                f"Need at least {window} data points, got {len(returns)}"
            )

        # Calculate rolling Sharpe
        rf_per_period = risk_free_rate / 252
        rolling_mean = returns.rolling(window).mean()
        rolling_std = returns.rolling(window).std()

        # Avoid division by zero
        rolling_sharpe = np.where(
            rolling_std > 1e-10,
            ((rolling_mean - rf_per_period) / rolling_std) * np.sqrt(252),
            0
        )
        rolling_sharpe = pd.Series(rolling_sharpe, index=returns.index)

        if backend == 'matplotlib':
            return Visualization._plot_rolling_sharpe_matplotlib(
                rolling_sharpe, window, save_path, show, title
            )
        else:
            return Visualization._plot_rolling_sharpe_plotly(
                rolling_sharpe, window, save_path, show, title
            )

    @staticmethod
    def _plot_rolling_sharpe_matplotlib(
        rolling_sharpe: pd.Series,
        window: int,
        save_path: Optional[str],
        show: bool,
        title: str
    ) -> Optional[Any]:
        """Matplotlib implementation of rolling Sharpe plot."""
        import matplotlib.pyplot as plt
        _setup_matplotlib_style()

        fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)

        # Rolling Sharpe line
        ax.plot(rolling_sharpe.index, rolling_sharpe.values,
               color=COLOR_NEUTRAL, linewidth=1.5, label=f'{window}-day Rolling Sharpe')

        # Fill regions
        ax.fill_between(rolling_sharpe.index, rolling_sharpe.values, 0,
                       where=(rolling_sharpe.values >= 0),
                       color=COLOR_PROFIT, alpha=0.2)
        ax.fill_between(rolling_sharpe.index, rolling_sharpe.values, 0,
                       where=(rolling_sharpe.values < 0),
                       color=COLOR_LOSS, alpha=0.2)

        # Benchmark lines
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.axhline(y=1, color=COLOR_BENCHMARK, linestyle='--', linewidth=1,
                  label='Sharpe = 1.0')
        ax.axhline(y=2, color=COLOR_BENCHMARK, linestyle=':', linewidth=1,
                  alpha=0.7, label='Sharpe = 2.0')

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=11)
        ax.set_ylabel('Sharpe Ratio', fontsize=11)
        ax.legend(loc='upper left', framealpha=0.9)
        ax.grid(True, alpha=0.3)

        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_path:
            _ensure_directory(save_path)
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved rolling Sharpe plot to {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    @staticmethod
    def _plot_rolling_sharpe_plotly(
        rolling_sharpe: pd.Series,
        window: int,
        save_path: Optional[str],
        show: bool,
        title: str
    ) -> Optional[Any]:
        """Plotly implementation of rolling Sharpe plot."""
        import plotly.graph_objects as go

        fig = go.Figure()

        # Main line
        fig.add_trace(go.Scatter(
            x=rolling_sharpe.index,
            y=rolling_sharpe.values,
            mode='lines',
            name=f'{window}-day Rolling Sharpe',
            line=dict(color=PLOTLY_COLORS['neutral'], width=2),
            hovertemplate='Date: %{x}<br>Sharpe: %{y:.2f}<extra></extra>'
        ))

        # Benchmark lines
        fig.add_hline(y=0, line=dict(color='black', width=0.5))
        fig.add_hline(y=1, line=dict(color=PLOTLY_COLORS['benchmark'],
                     dash='dash', width=1),
                     annotation_text='Sharpe = 1.0',
                     annotation_position='top right')
        fig.add_hline(y=2, line=dict(color=PLOTLY_COLORS['benchmark'],
                     dash='dot', width=1),
                     annotation_text='Sharpe = 2.0',
                     annotation_position='top right')

        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor='center', font=dict(size=16)),
            xaxis=dict(title='Date'),
            yaxis=dict(title='Sharpe Ratio'),
            hovermode='x unified',
            showlegend=True,
            template='plotly_white'
        )

        if save_path:
            _ensure_directory(save_path)
            fmt = _get_save_format(save_path)
            if fmt == 'html':
                fig.write_html(save_path)
            else:
                fig.write_image(save_path, scale=2)
            logger.info(f"Saved rolling Sharpe plot to {save_path}")

        if show:
            fig.show()

        return fig

    # =========================================================================
    # Returns Distribution
    # =========================================================================

    @staticmethod
    def plot_returns_distribution(
        returns: pd.Series,
        save_path: Optional[str] = None,
        show: bool = True,
        backend: str = 'plotly',
        title: str = 'Daily Returns Distribution',
        bins: int = 50
    ) -> Optional[Any]:
        """
        Plot daily returns distribution.

        Creates a histogram with kernel density estimate and Q-Q plot
        to check normality, marking VaR and CVaR levels.

        Args:
            returns: Series of period returns.
            save_path: Path to save figure. If None, figure is not saved.
            show: If True, display the figure. Default True.
            backend: 'matplotlib' or 'plotly'. Default 'plotly'.
            title: Chart title. Default 'Daily Returns Distribution'.
            bins: Number of histogram bins. Default 50.

        Returns:
            Figure object or None if only showing/saving.

        Example:
            >>> fig = Visualization.plot_returns_distribution(returns, show=True)
        """
        backend = _validate_backend(backend)

        if not isinstance(returns, pd.Series):
            returns = pd.Series(returns)

        returns = returns.dropna()

        if len(returns) < 10:
            raise InsufficientDataError(
                f"Need at least 10 data points, got {len(returns)}"
            )

        # Calculate statistics
        mean_ret = float(returns.mean())
        std_ret = float(returns.std())
        var_95 = float(np.percentile(returns, 5))
        cvar_95 = float(returns[returns <= var_95].mean())

        if backend == 'matplotlib':
            return Visualization._plot_returns_dist_matplotlib(
                returns, mean_ret, std_ret, var_95, cvar_95,
                save_path, show, title, bins
            )
        else:
            return Visualization._plot_returns_dist_plotly(
                returns, mean_ret, std_ret, var_95, cvar_95,
                save_path, show, title, bins
            )

    @staticmethod
    def _plot_returns_dist_matplotlib(
        returns: pd.Series,
        mean_ret: float,
        std_ret: float,
        var_95: float,
        cvar_95: float,
        save_path: Optional[str],
        show: bool,
        title: str,
        bins: int
    ) -> Optional[Any]:
        """Matplotlib implementation of returns distribution plot."""
        import matplotlib.pyplot as plt
        from scipy.stats import probplot
        _setup_matplotlib_style()

        fig, axes = plt.subplots(1, 2, figsize=DEFAULT_FIGSIZE_WIDE)

        # Histogram
        ax1 = axes[0]
        n, bin_edges, patches = ax1.hist(
            returns.values * 100, bins=bins, density=True,
            alpha=0.7, color=COLOR_NEUTRAL, edgecolor='white'
        )

        # Normal curve
        x = np.linspace(returns.min() * 100, returns.max() * 100, 100)
        normal_pdf = stats.norm.pdf(x, mean_ret * 100, std_ret * 100)
        ax1.plot(x, normal_pdf, color='black', linestyle='--', linewidth=2,
                label='Normal Distribution')

        # Statistical markers
        ax1.axvline(var_95 * 100, color='red', linestyle='-', linewidth=1.5,
                   label=f'5% VaR: {var_95*100:.2f}%')
        ax1.axvline(cvar_95 * 100, color='darkred', linestyle=':', linewidth=1.5,
                   label=f'5% CVaR: {cvar_95*100:.2f}%')

        ax1.set_title('Returns Histogram', fontsize=12)
        ax1.set_xlabel('Return (%)', fontsize=11)
        ax1.set_ylabel('Density', fontsize=11)
        ax1.legend(loc='upper left', fontsize=9)
        ax1.grid(True, alpha=0.3)

        # Q-Q plot
        ax2 = axes[1]
        probplot(returns.values, dist='norm', plot=ax2)
        ax2.set_title('Q-Q Plot (Normal)', fontsize=12)
        ax2.grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        if save_path:
            _ensure_directory(save_path)
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved returns distribution to {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    @staticmethod
    def _plot_returns_dist_plotly(
        returns: pd.Series,
        mean_ret: float,
        std_ret: float,
        var_95: float,
        cvar_95: float,
        save_path: Optional[str],
        show: bool,
        title: str,
        bins: int
    ) -> Optional[Any]:
        """Plotly implementation of returns distribution plot."""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Returns Histogram', 'Q-Q Plot (Normal)']
        )

        # Histogram
        fig.add_trace(go.Histogram(
            x=returns.values * 100,
            nbinsx=bins,
            histnorm='probability density',
            name='Returns',
            marker=dict(color=PLOTLY_COLORS['neutral'],
                       line=dict(color='white', width=1)),
            hovertemplate='Return: %{x:.2f}%<br>Density: %{y:.4f}<extra></extra>'
        ), row=1, col=1)

        # Normal curve
        x = np.linspace(float(returns.min() * 100), float(returns.max() * 100), 100)
        normal_pdf = stats.norm.pdf(x, mean_ret * 100, std_ret * 100)
        fig.add_trace(go.Scatter(
            x=x, y=normal_pdf,
            mode='lines',
            name='Normal',
            line=dict(color='black', dash='dash', width=2)
        ), row=1, col=1)

        # VaR and CVaR
        fig.add_vline(x=var_95 * 100, line=dict(color='red', width=2),
                     row=1, col=1,
                     annotation_text=f'5% VaR: {var_95*100:.2f}%',
                     annotation_position='top left')
        fig.add_vline(x=cvar_95 * 100, line=dict(color='darkred', width=2, dash='dot'),
                     row=1, col=1,
                     annotation_text=f'5% CVaR: {cvar_95*100:.2f}%',
                     annotation_position='bottom left')

        # Q-Q plot
        sorted_returns = np.sort(returns.values)
        theoretical_quantiles = stats.norm.ppf(
            np.linspace(0.01, 0.99, len(sorted_returns))
        )

        fig.add_trace(go.Scatter(
            x=theoretical_quantiles,
            y=sorted_returns,
            mode='markers',
            name='Q-Q Points',
            marker=dict(color=PLOTLY_COLORS['neutral'], size=4),
            hovertemplate='Theoretical: %{x:.3f}<br>Sample: %{y:.4f}<extra></extra>'
        ), row=1, col=2)

        # Reference line
        min_val = min(theoretical_quantiles.min(), sorted_returns.min())
        max_val = max(theoretical_quantiles.max(), sorted_returns.max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[mean_ret + std_ret * min_val, mean_ret + std_ret * max_val],
            mode='lines',
            name='Reference Line',
            line=dict(color='red', dash='dash', width=1)
        ), row=1, col=2)

        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor='center', font=dict(size=16)),
            showlegend=True,
            template='plotly_white'
        )

        fig.update_xaxes(title_text='Return (%)', row=1, col=1)
        fig.update_yaxes(title_text='Density', row=1, col=1)
        fig.update_xaxes(title_text='Theoretical Quantiles', row=1, col=2)
        fig.update_yaxes(title_text='Sample Quantiles', row=1, col=2)

        if save_path:
            _ensure_directory(save_path)
            fmt = _get_save_format(save_path)
            if fmt == 'html':
                fig.write_html(save_path)
            else:
                fig.write_image(save_path, scale=2)
            logger.info(f"Saved returns distribution to {save_path}")

        if show:
            fig.show()

        return fig


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Main class
    'Visualization',

    # Exceptions
    'VisualizationError',
    'InvalidBackendError',
    'InsufficientDataError',
    'SaveError',

    # Constants
    'SUPPORTED_BACKENDS',
    'COLOR_PROFIT',
    'COLOR_LOSS',
    'COLOR_NEUTRAL',
    'COLOR_EQUITY',
    'COLOR_DRAWDOWN',
]
