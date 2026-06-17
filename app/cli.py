"""Command-line interface for the stock backtester.

Five commands:
    backtester strategies          List available strategies.
    backtester run -t T -s S [...] Run a structured backtest.
    backtester ask "PROMPT"        Run a backtest via the LLM agent
                                   (requires ANTHROPIC_API_KEY in env or .env).
    backtester history             Show past runs (table).
    backtester show ID             Print full metrics for a stored run.

Installed via the [project.scripts] entry in pyproject.toml — after
`pip install -e .`, `backtester` is available anywhere in this venv.
"""
from __future__ import annotations

import json
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from app.api.schemas import BacktestRequest
from app.api.services import run_and_persist
from app.db.database import Base, SessionLocal, engine
from app.db.models import BacktestRun
from app.strategies import STRATEGY_REGISTRY


app = typer.Typer(
    name="backtester",
    add_completion=False,
    no_args_is_help=True,
    help="Backtest trading strategies from the command line.",
    rich_markup_mode="rich",
)
console = Console()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _ensure_tables() -> None:
    """Mirror the FastAPI lifespan create_all() so the CLI works on a fresh
    checkout without requiring `alembic upgrade head` first."""
    Base.metadata.create_all(engine)


def _print_run(row: BacktestRun) -> None:
    """Render a stored BacktestRun as a tidy metrics panel + trade table."""
    params_str = (
        ", ".join(f"{k}={v}" for k, v in row.strategy_params.items())
        if row.strategy_params else "defaults"
    )

    header = f"[bold]{row.ticker}[/bold] · {row.strategy_name}({params_str}) · {row.period}"

    pos_neg = lambda v: "green" if v >= 0 else "red"
    pct = lambda v: f"{v * 100:+.2f}%"

    metrics = Table.grid(padding=(0, 2))
    metrics.add_column(style="dim")
    metrics.add_column(justify="right")
    metrics.add_row("Total Return", f"[{pos_neg(row.total_return)}]{pct(row.total_return)}[/]")
    metrics.add_row("CAGR",         f"[{pos_neg(row.cagr)}]{pct(row.cagr)}[/]")
    metrics.add_row("Sharpe Ratio", f"{row.sharpe_ratio:.4f}")
    metrics.add_row("Max Drawdown", f"[red]{row.max_drawdown * 100:.2f}%[/red]")
    metrics.add_row("Win Rate",     f"{row.win_rate * 100:.1f}%" if row.win_rate is not None else "n/a")
    metrics.add_row("Num Trades",   str(row.num_trades))

    console.print()
    console.print(Panel(metrics, title=header, title_align="left", border_style="blue"))

    if row.trades:
        tbl = Table(show_header=True, header_style="bold dim", box=None, padding=(0, 1))
        tbl.add_column("Entry")
        tbl.add_column("Exit")
        tbl.add_column("Entry $",  justify="right")
        tbl.add_column("Exit $",   justify="right")
        tbl.add_column("PnL",      justify="right")
        tbl.add_column("Return",   justify="right")
        for t in row.trades:
            ret_style = pos_neg(t["return_pct"])
            tbl.add_row(
                t["entry_date"], t["exit_date"],
                f"{t['entry_price']:.2f}", f"{t['exit_price']:.2f}",
                f"[{pos_neg(t['pnl'])}]{t['pnl']:+.2f}[/]",
                f"[{ret_style}]{t['return_pct']:+.2f}%[/]",
            )
        console.print(tbl)
    console.print()


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
@app.command()
def strategies() -> None:
    """List all available strategies."""
    console.print()
    console.print("[bold]Available strategies:[/bold]")
    for name in sorted(STRATEGY_REGISTRY):
        console.print(f"  • {name}")
    console.print()


@app.command()
def run(
    ticker:   str   = typer.Option(..., "-t", "--ticker",   help="Ticker symbol (e.g. AAPL)."),
    strategy: str   = typer.Option(..., "-s", "--strategy", help="Strategy name (see `backtester strategies`)."),
    period:   str   = typer.Option("1y",  "-p", "--period",   help="Lookback period (e.g. 1y, 5y, max)."),
    interval: str   = typer.Option("1d",  "-i", "--interval", help="Bar interval."),
    capital:  float = typer.Option(10_000.0, "-c", "--capital", help="Initial capital."),
    params:   Optional[str] = typer.Option(None, "--params", help='Strategy params as JSON, e.g. \'{"fast_period": 50}\''),
) -> None:
    """Run a structured backtest and persist the result."""
    if strategy not in STRATEGY_REGISTRY:
        console.print(f"[red]Unknown strategy '{strategy}'.[/red]")
        console.print(f"Available: {', '.join(sorted(STRATEGY_REGISTRY))}")
        raise typer.Exit(1)

    strategy_params: dict = {}
    if params:
        try:
            strategy_params = json.loads(params)
        except json.JSONDecodeError as e:
            console.print(f"[red]Invalid --params JSON: {e}[/red]")
            raise typer.Exit(1)

    _ensure_tables()
    req = BacktestRequest(
        ticker=ticker, strategy_name=strategy,
        strategy_params=strategy_params, period=period,
        interval=interval, initial_capital=capital,
    )

    with SessionLocal() as db:
        with console.status(f"[blue]Running {strategy} on {ticker}[/blue]"):
            try:
                row = run_and_persist(req, db)
            except Exception as e:  # noqa: BLE001 — surface any failure with a clean message
                detail = getattr(e, "detail", None) or str(e)
                console.print(f"[red]{detail}[/red]")
                raise typer.Exit(1)

    _print_run(row)


@app.command()
def ask(
    prompt: str = typer.Argument(..., help="Natural-language description of the backtest."),
) -> None:
    """Run a backtest via the Claude agent. Requires ANTHROPIC_API_KEY."""
    from dotenv import load_dotenv

    load_dotenv()
    _ensure_tables()

    # Lazy import — keeps the other CLI commands fast and lets the agent
    # module's anthropic import happen only when actually needed.
    from app.agent.parser import MissingAPIKeyError
    from app.agent.runner import run_agent

    with SessionLocal() as db:
        try:
            with console.status("[blue]Thinking…[/blue]"):
                summary_text, row = run_agent(prompt, db)
        except MissingAPIKeyError as e:
            console.print(f"[red]{e}[/red]")
            console.print(
                "[dim]Set [bold]ANTHROPIC_API_KEY[/bold] in your shell or in a .env "
                "file (see [bold].env.example[/bold]).[/dim]"
            )
            raise typer.Exit(2)
        except Exception as e:  # noqa: BLE001
            detail = getattr(e, "detail", None) or str(e)
            console.print(f"[red]{detail}[/red]")
            raise typer.Exit(1)

    _print_run(row)
    console.print(Panel(summary_text.strip(), title="Summary", title_align="left", border_style="green"))
    console.print()


@app.command()
def history(
    limit: int = typer.Option(20, "-n", "--limit", min=1, max=200, help="Max rows to show."),
) -> None:
    """List past backtest runs (most recent first)."""
    from sqlalchemy import select

    _ensure_tables()
    with SessionLocal() as db:
        rows = db.execute(
            select(BacktestRun)
            .order_by(BacktestRun.created_at.desc(), BacktestRun.id.desc())
            .limit(limit)
        ).scalars().all()

    if not rows:
        console.print("[yellow]No past runs yet. Try `backtester run` to create one.[/yellow]")
        return

    table = Table(title=f"Past backtest runs (latest {len(rows)})", header_style="bold")
    table.add_column("ID",          justify="right")
    table.add_column("Ticker")
    table.add_column("Strategy")
    table.add_column("Return",      justify="right")
    table.add_column("Sharpe",      justify="right")
    table.add_column("Trades",      justify="right")
    table.add_column("When")

    for r in rows:
        ret_color = "green" if r.total_return >= 0 else "red"
        table.add_row(
            str(r.id), r.ticker, r.strategy_name,
            f"[{ret_color}]{r.total_return * 100:+.2f}%[/]",
            f"{r.sharpe_ratio:.2f}", str(r.num_trades),
            r.created_at.strftime("%Y-%m-%d %H:%M"),
        )

    console.print()
    console.print(table)
    console.print()


@app.command()
def show(run_id: int = typer.Argument(..., help="Run ID from `backtester history`.")) -> None:
    """Print the full metrics + trade log for a stored run."""
    _ensure_tables()
    with SessionLocal() as db:
        row = db.get(BacktestRun, run_id)
        if row is None:
            console.print(f"[red]Run {run_id} not found.[/red]")
            raise typer.Exit(1)
    _print_run(row)


if __name__ == "__main__":
    app()
