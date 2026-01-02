"""
Command-Line Interface for Options Backtester

Provides CLI commands for running backtests, validating configurations,
listing templates, and generating reports.

Usage:
    backtest run --config strategy.yaml
    backtest validate --config strategy.yaml
    backtest list templates
    backtest report --results results.json
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import click

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from backtester.cli.config_loader import ConfigLoader, load_config
from backtester.cli.config_schema import (
    ConfigValidationError,
    ConfigValidator,
    StrategyConfig,
    StructureType,
)
from backtester.cli.environment import (
    Environment,
    EnvironmentManager,
    configure_logging,
    get_settings,
    set_environment,
)

console = Console() if RICH_AVAILABLE else None


def echo(message: str, style: Optional[str] = None, err: bool = False) -> None:
    """Output message using rich if available, otherwise click.echo."""
    if RICH_AVAILABLE and console:
        if err:
            from rich.console import Console

            err_console = Console(stderr=True)
            err_console.print(message, style=style)
        else:
            console.print(message, style=style)
    else:
        click.echo(message, err=err)


def echo_error(message: str) -> None:
    """Output error message."""
    echo(
        f"[red]Error:[/red] {message}" if RICH_AVAILABLE else f"Error: {message}",
        err=True,
    )


def echo_success(message: str) -> None:
    """Output success message."""
    echo(f"[green]{message}[/green]" if RICH_AVAILABLE else message)


def echo_warning(message: str) -> None:
    """Output warning message."""
    echo(
        f"[yellow]Warning:[/yellow] {message}"
        if RICH_AVAILABLE
        else f"Warning: {message}"
    )


@click.group()
@click.option(
    "--env",
    "-e",
    type=click.Choice(["development", "staging", "production", "test"]),
    default="development",
    help="Environment to use",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option("--quiet", "-q", is_flag=True, help="Suppress non-essential output")
@click.version_option(version="1.0.0", prog_name="Options Backtester")
@click.pass_context
def cli(ctx: click.Context, env: str, verbose: bool, quiet: bool) -> None:
    """Options Backtester CLI - Run backtests and manage strategies."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["quiet"] = quiet

    set_environment(Environment(env))
    if verbose:
        configure_logging()


@cli.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to strategy configuration file (YAML or JSON)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output directory for results",
)
@click.option(
    "--start-date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="Override start date from config",
)
@click.option(
    "--end-date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="Override end date from config",
)
@click.option("--capital", type=float, help="Override initial capital from config")
@click.option(
    "--dry-run", is_flag=True, help="Validate config without running backtest"
)
@click.pass_context
def run(
    ctx: click.Context,
    config: Path,
    output: Optional[Path],
    start_date: Optional[datetime],
    end_date: Optional[datetime],
    capital: Optional[float],
    dry_run: bool,
) -> None:
    """Run a backtest using configuration file."""
    verbose = ctx.obj.get("verbose", False)
    quiet = ctx.obj.get("quiet", False)

    try:
        if not quiet:
            echo(
                f"Loading configuration from [cyan]{config}[/cyan]..."
                if RICH_AVAILABLE
                else f"Loading configuration from {config}..."
            )

        strategy_config = load_config(config)

        if verbose:
            echo(f"Strategy: {strategy_config.name}")
            echo(f"Underlying: {strategy_config.underlying}")

        if dry_run:
            echo_success("Configuration is valid. Dry run complete.")
            return

        if not quiet:
            echo("Running backtest...")

        result = _execute_backtest(
            strategy_config,
            start_date=start_date,
            end_date=end_date,
            capital=capital,
            output_dir=output,
            verbose=verbose,
        )

        if result:
            _display_results(result, verbose=verbose)
            echo_success("Backtest completed successfully!")

    except ConfigValidationError as e:
        echo_error(f"Configuration validation failed: {e}")
        for error in e.errors:
            echo(f"  - {error}")
        sys.exit(1)
    except FileNotFoundError as e:
        echo_error(str(e))
        sys.exit(1)
    except Exception as e:
        echo_error(f"Backtest failed: {e}")
        if verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to strategy configuration file",
)
@click.option("--strict", is_flag=True, help="Treat warnings as errors")
@click.pass_context
def validate(ctx: click.Context, config: Path, strict: bool) -> None:
    """Validate a strategy configuration file."""
    verbose = ctx.obj.get("verbose", False)

    try:
        echo(
            f"Validating [cyan]{config}[/cyan]..."
            if RICH_AVAILABLE
            else f"Validating {config}..."
        )

        strategy_config = load_config(config)
        errors = ConfigValidator.validate(strategy_config)

        if errors:
            echo_error("Validation failed with the following errors:")
            for error in errors:
                echo(f"  [red]x[/red] {error}" if RICH_AVAILABLE else f"  x {error}")
            sys.exit(1)

        echo_success(f"Configuration '{strategy_config.name}' is valid!")

        if verbose:
            _display_config_summary(strategy_config)

    except ConfigValidationError as e:
        echo_error(str(e))
        for error in e.errors:
            echo(f"  - {error}")
        sys.exit(1)
    except Exception as e:
        echo_error(f"Validation error: {e}")
        sys.exit(1)


@cli.group()
def list() -> None:
    """List available resources."""
    pass


@list.command(name="templates")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed information")
def list_templates(verbose: bool) -> None:
    """List available strategy templates."""
    try:
        from backtester.strategies import TemplateRegistry

        templates = TemplateRegistry.list_templates()

        if RICH_AVAILABLE and console:
            table = Table(title="Available Strategy Templates")
            table.add_column("Name", style="cyan")
            table.add_column("Description", style="white")

            for name in templates:
                desc = TemplateRegistry.get_description(name)
                table.add_row(name, desc)

            console.print(table)
        else:
            echo("Available Strategy Templates:")
            echo("-" * 40)
            for name in templates:
                desc = TemplateRegistry.get_description(name)
                echo(f"  {name}: {desc}")

    except ImportError:
        echo_warning("Strategy templates not available. Install backtester package.")


@list.command(name="structures")
def list_structures() -> None:
    """List available option structure types."""
    if RICH_AVAILABLE and console:
        table = Table(title="Available Option Structures")
        table.add_column("Type", style="cyan")
        table.add_column("Description", style="white")

        structures = [
            ("short_straddle", "Sell ATM call and put"),
            ("long_straddle", "Buy ATM call and put"),
            ("short_strangle", "Sell OTM call and put"),
            ("long_strangle", "Buy OTM call and put"),
            ("iron_condor", "Sell OTM strangle, buy further OTM strangle"),
            ("bull_call_spread", "Buy call, sell higher strike call"),
            ("bear_put_spread", "Buy put, sell lower strike put"),
            ("bull_put_spread", "Sell put, buy lower strike put"),
            ("bear_call_spread", "Sell call, buy higher strike call"),
        ]

        for struct_type, desc in structures:
            table.add_row(struct_type, desc)

        console.print(table)
    else:
        echo("Available Option Structures:")
        echo("-" * 40)
        for st in StructureType:
            echo(f"  {st.value}")


@list.command(name="conditions")
def list_conditions() -> None:
    """List available entry and exit conditions."""
    entry_conditions = [
        ("iv_rank_above", "IV rank exceeds threshold"),
        ("iv_rank_below", "IV rank below threshold"),
        ("iv_rank_between", "IV rank within range"),
        ("vix_above", "VIX exceeds level"),
        ("vix_below", "VIX below level"),
        ("vix_between", "VIX within range"),
        ("dte_above", "Days to expiration above value"),
        ("dte_below", "Days to expiration below value"),
        ("dte_between", "Days to expiration within range"),
        ("day_of_week", "Specific days of week"),
        ("no_open_positions", "No existing positions"),
        ("max_open_positions", "Below position limit"),
    ]

    exit_conditions = [
        ("profit_target", "Profit reaches target percentage"),
        ("stop_loss", "Loss exceeds stop percentage"),
        ("fixed_stop_loss", "Loss exceeds fixed dollar amount"),
        ("trailing_stop", "Profit drops from peak"),
        ("max_holding_days", "Position held for N days"),
        ("min_dte", "DTE falls below threshold"),
    ]

    if RICH_AVAILABLE and console:
        console.print(Panel("[bold]Entry Conditions[/bold]"))
        table = Table()
        table.add_column("Condition", style="cyan")
        table.add_column("Description")
        for name, desc in entry_conditions:
            table.add_row(name, desc)
        console.print(table)

        console.print(Panel("[bold]Exit Conditions[/bold]"))
        table = Table()
        table.add_column("Condition", style="cyan")
        table.add_column("Description")
        for name, desc in exit_conditions:
            table.add_row(name, desc)
        console.print(table)
    else:
        echo("\nEntry Conditions:")
        echo("-" * 40)
        for name, desc in entry_conditions:
            echo(f"  {name}: {desc}")

        echo("\nExit Conditions:")
        echo("-" * 40)
        for name, desc in exit_conditions:
            echo(f"  {name}: {desc}")


@cli.command()
@click.option(
    "--results",
    "-r",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to backtest results JSON file",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output path for report (default: report.html)",
)
@click.option(
    "--format",
    "-f",
    type=click.Choice(["html", "json", "text"]),
    default="html",
    help="Report output format",
)
@click.pass_context
def report(
    ctx: click.Context,
    results: Path,
    output: Optional[Path],
    format: str,
) -> None:
    """Generate a report from backtest results."""
    verbose = ctx.obj.get("verbose", False)

    try:
        echo(
            f"Loading results from [cyan]{results}[/cyan]..."
            if RICH_AVAILABLE
            else f"Loading results from {results}..."
        )

        with open(results) as f:
            results_data = json.load(f)

        output_path = output or Path(f"report.{format}")

        if format == "html":
            _generate_html_report(results_data, output_path)
        elif format == "json":
            _generate_json_report(results_data, output_path)
        else:
            _generate_text_report(results_data, output_path)

        echo_success(f"Report generated: {output_path}")

    except Exception as e:
        echo_error(f"Report generation failed: {e}")
        if verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.option("--name", "-n", required=True, help="Strategy name")
@click.option("--template", "-t", help="Use a template as starting point")
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output file path (default: {name}.yaml)",
)
def init(name: str, template: Optional[str], output: Optional[Path]) -> None:
    """Initialize a new strategy configuration file."""
    output_path = output or Path(f"{name.lower().replace(' ', '_')}.yaml")

    if output_path.exists():
        if not click.confirm(f"{output_path} already exists. Overwrite?"):
            echo("Aborted.")
            return

    if template:
        try:
            from backtester.strategies import TemplateRegistry

            if template not in TemplateRegistry.list_templates():
                echo_error(f"Unknown template: {template}")
                echo(
                    f"Available templates: {', '.join(TemplateRegistry.list_templates())}"
                )
                sys.exit(1)
            config_content = _generate_template_config(name, template)
        except ImportError:
            echo_error("Templates not available")
            sys.exit(1)
    else:
        config_content = _generate_default_config(name)

    with open(output_path, "w") as f:
        f.write(config_content)

    echo_success(f"Created strategy configuration: {output_path}")


@cli.command()
def env() -> None:
    """Show current environment configuration."""
    settings = get_settings()

    if RICH_AVAILABLE and console:
        table = Table(title=f"Environment: {settings.name.value}")
        table.add_column("Setting", style="cyan")
        table.add_column("Value")

        table.add_row("Data Source Type", settings.data_source_type)
        table.add_row("Database", settings.database_name or "Not set")
        table.add_row("Log Level", settings.log_level)
        table.add_row("Output Directory", settings.output_directory)
        table.add_row("Commission", f"${settings.commission_per_contract}")
        table.add_row("Slippage", f"{settings.slippage_pct:.2%}")
        table.add_row("Use Caching", str(settings.use_caching))
        table.add_row("Parallel Execution", str(settings.parallel_execution))

        console.print(table)
    else:
        echo(f"\nEnvironment: {settings.name.value}")
        echo("-" * 40)
        echo(f"  Data Source: {settings.data_source_type}")
        echo(f"  Log Level: {settings.log_level}")
        echo(f"  Output: {settings.output_directory}")


def _execute_backtest(
    config: StrategyConfig,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    capital: Optional[float] = None,
    output_dir: Optional[Path] = None,
    verbose: bool = False,
) -> Optional[dict]:
    """Execute backtest with given configuration."""
    try:
        from backtester.strategies import StrategyBuilder, TemplateRegistry
        from backtester.engine.backtest_engine import BacktestEngine

        if config.template:
            strategy = TemplateRegistry.create(
                config.template,
                underlying=config.underlying,
                **config.template_params,
            )
        else:
            strategy = _build_strategy_from_config(config)

        if strategy is None:
            echo_error("Failed to build strategy from configuration")
            return None

        backtest_config = config.backtest
        if backtest_config:
            final_capital = capital or backtest_config.initial_capital
            final_start = start_date or backtest_config.start_date
            final_end = end_date or backtest_config.end_date
        else:
            final_capital = capital or 100000.0
            final_start = start_date or datetime(2024, 1, 1)
            final_end = end_date or datetime(2024, 12, 31)

        if verbose:
            echo(f"Capital: ${final_capital:,.2f}")
            echo(f"Period: {final_start} to {final_end}")

        echo_warning("Backtest execution requires data source configuration.")
        echo("This is a placeholder - full implementation requires data integration.")

        return {
            "strategy_name": config.name,
            "underlying": config.underlying,
            "initial_capital": final_capital,
            "start_date": str(final_start),
            "end_date": str(final_end),
            "status": "placeholder",
        }

    except ImportError as e:
        echo_error(f"Missing required module: {e}")
        return None


def _build_strategy_from_config(config: StrategyConfig):
    """Build strategy from configuration using StrategyBuilder."""
    try:
        from backtester.strategies import StrategyBuilder
        from backtester.strategies.strategy_builder import (
            iv_rank_above,
            iv_rank_below,
            iv_rank_between,
            vix_above,
            vix_below,
            vix_between,
            dte_above,
            dte_below,
            dte_between,
            day_of_week,
            no_open_positions,
            max_open_positions,
            profit_target,
            stop_loss,
            fixed_stop_loss,
            trailing_stop,
            holding_period,
            expiration_approaching,
            short_straddle,
            long_straddle,
            short_strangle,
            long_strangle,
            iron_condor,
            bull_call_spread,
            bear_put_spread,
            fixed_contracts,
            risk_percent,
            capital_percent,
        )

        builder = StrategyBuilder().name(config.name).underlying(config.underlying)

        condition_builders = {
            "iv_rank_above": lambda p: iv_rank_above(p.get("threshold", 70)),
            "iv_rank_below": lambda p: iv_rank_below(p.get("threshold", 30)),
            "iv_rank_between": lambda p: iv_rank_between(
                p.get("low", 30), p.get("high", 70)
            ),
            "vix_above": lambda p: vix_above(p.get("level", 20)),
            "vix_below": lambda p: vix_below(p.get("level", 15)),
            "vix_between": lambda p: vix_between(p.get("low", 15), p.get("high", 25)),
            "dte_above": lambda p: dte_above(p.get("days", 30)),
            "dte_below": lambda p: dte_below(p.get("days", 7)),
            "dte_between": lambda p: dte_between(p.get("low", 20), p.get("high", 45)),
            "day_of_week": lambda p: day_of_week(p.get("days", [0, 1, 2, 3, 4])),
            "no_open_positions": lambda p: no_open_positions(),
            "max_open_positions": lambda p: max_open_positions(p.get("max", 5)),
        }

        combined_entry = None
        for cond_config in config.entry_conditions:
            if cond_config.type in condition_builders:
                condition = condition_builders[cond_config.type](cond_config.params)
                if combined_entry is None:
                    combined_entry = condition
                else:
                    combined_entry = combined_entry & condition

        if combined_entry:
            builder = builder.entry_condition(combined_entry)

        if config.structure:
            structure_builders = {
                StructureType.SHORT_STRADDLE: short_straddle,
                StructureType.LONG_STRADDLE: long_straddle,
                StructureType.SHORT_STRANGLE: short_strangle,
                StructureType.LONG_STRANGLE: long_strangle,
                StructureType.IRON_CONDOR: iron_condor,
                StructureType.BULL_CALL_SPREAD: bull_call_spread,
                StructureType.BEAR_PUT_SPREAD: bear_put_spread,
            }

            struct_builder = structure_builders.get(config.structure.type)
            if struct_builder:
                struct_kwargs = {"dte": config.structure.dte}
                if config.structure.delta:
                    struct_kwargs["delta"] = config.structure.delta
                if config.structure.width:
                    struct_kwargs["width"] = config.structure.width
                builder = builder.structure(struct_builder(**struct_kwargs))

        if config.exit_conditions:
            exit_conds = []
            if config.exit_conditions.profit_target:
                exit_conds.append(profit_target(config.exit_conditions.profit_target))
            if config.exit_conditions.stop_loss:
                exit_conds.append(stop_loss(config.exit_conditions.stop_loss))
            if config.exit_conditions.fixed_stop_loss:
                exit_conds.append(
                    fixed_stop_loss(config.exit_conditions.fixed_stop_loss)
                )
            if config.exit_conditions.trailing_stop:
                exit_conds.append(trailing_stop(config.exit_conditions.trailing_stop))
            if config.exit_conditions.max_holding_days:
                exit_conds.append(
                    holding_period(config.exit_conditions.max_holding_days)
                )
            if config.exit_conditions.min_dte:
                exit_conds.append(
                    expiration_approaching(config.exit_conditions.min_dte)
                )

            if exit_conds:
                combined_exit = exit_conds[0]
                for cond in exit_conds[1:]:
                    combined_exit = combined_exit | cond
                builder = builder.exit_condition(combined_exit)

        if config.position_size:
            from backtester.cli.config_schema import PositionSizeMethod

            sizer_builders = {
                PositionSizeMethod.FIXED_CONTRACTS: fixed_contracts,
                PositionSizeMethod.RISK_PERCENT: risk_percent,
                PositionSizeMethod.CAPITAL_PERCENT: capital_percent,
            }
            sizer = sizer_builders.get(config.position_size.method)
            if sizer:
                builder = builder.position_size(sizer(config.position_size.value))

        if config.risk_limits:
            builder = builder.max_positions(config.risk_limits.max_positions)

        return builder.build()

    except Exception as e:
        echo_error(f"Failed to build strategy: {e}")
        return None


def _display_results(results: dict, verbose: bool = False) -> None:
    """Display backtest results."""
    if RICH_AVAILABLE and console:
        console.print(
            Panel(
                f"[bold]Backtest Results: {results.get('strategy_name', 'Unknown')}[/bold]"
            )
        )

        table = Table()
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")

        for key, value in results.items():
            if key not in ("trades", "equity_curve"):
                table.add_row(key.replace("_", " ").title(), str(value))

        console.print(table)
    else:
        echo(f"\nResults: {results.get('strategy_name', 'Unknown')}")
        echo("-" * 40)
        for key, value in results.items():
            if key not in ("trades", "equity_curve"):
                echo(f"  {key}: {value}")


def _display_config_summary(config: StrategyConfig) -> None:
    """Display configuration summary."""
    if RICH_AVAILABLE and console:
        table = Table(title="Configuration Summary")
        table.add_column("Setting", style="cyan")
        table.add_column("Value")

        table.add_row("Name", config.name)
        table.add_row("Underlying", config.underlying)
        table.add_row("Version", config.version)
        if config.structure:
            table.add_row("Structure", config.structure.type.value)
            table.add_row("DTE", str(config.structure.dte))
        table.add_row("Entry Conditions", str(len(config.entry_conditions)))

        console.print(table)
    else:
        echo(f"\nConfiguration: {config.name}")
        echo(f"  Underlying: {config.underlying}")
        if config.structure:
            echo(f"  Structure: {config.structure.type.value}")


def _generate_default_config(name: str) -> str:
    """Generate default strategy configuration YAML."""
    return f"""# Strategy Configuration
name: "{name}"
version: "1.0"
description: "Custom strategy"

underlying: "SPY"

structure:
  type: short_straddle
  dte: 30

entry_conditions:
  - type: iv_rank_above
    threshold: 70
  - type: no_open_positions

exit_conditions:
  profit_target: 0.50
  stop_loss: 2.0
  min_dte: 7

position_size:
  method: fixed_contracts
  value: 1

risk_limits:
  max_positions: 5
  max_capital_utilization: 0.80

backtest:
  start_date: "2024-01-01"
  end_date: "2024-12-31"
  initial_capital: 100000
  commission_per_contract: 0.65
  slippage_pct: 0.01
"""


def _generate_template_config(name: str, template: str) -> str:
    """Generate configuration based on a template."""
    try:
        from backtester.strategies import TemplateRegistry

        info = TemplateRegistry.get_template_info(template)
        defaults = info.get("defaults", {}) if info else {}
    except ImportError:
        defaults = {}

    return f"""# Strategy Configuration - Based on {template} template
name: "{name}"
version: "1.0"
description: "Strategy based on {template} template"

template: "{template}"
template_params:
  # Add template-specific parameters here
  underlying: "SPY"

backtest:
  start_date: "2024-01-01"
  end_date: "2024-12-31"
  initial_capital: 100000
"""


def _generate_html_report(results: dict, output_path: Path) -> None:
    """Generate HTML report from results."""
    try:
        from backtester.analytics import Dashboard

        Dashboard.create_performance_dashboard(
            results, results.get("metrics", {}), str(output_path)
        )
    except ImportError:
        html = f"""<!DOCTYPE html>
<html>
<head><title>Backtest Report</title></head>
<body>
<h1>Backtest Results</h1>
<pre>{json.dumps(results, indent=2, default=str)}</pre>
</body>
</html>"""
        with open(output_path, "w") as f:
            f.write(html)


def _generate_json_report(results: dict, output_path: Path) -> None:
    """Generate JSON report from results."""
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)


def _generate_text_report(results: dict, output_path: Path) -> None:
    """Generate text report from results."""
    lines = ["Backtest Results", "=" * 40]
    for key, value in results.items():
        if key not in ("trades", "equity_curve"):
            lines.append(f"{key}: {value}")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def main() -> None:
    """Main entry point for CLI."""
    cli(obj={})


if __name__ == "__main__":
    main()
