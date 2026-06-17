"""CLI smoke tests via typer's CliRunner.

These confirm each command's wiring without making real network calls:
  - `strategies` and `history` hit the registry / DB only.
  - `show` is exercised on the no-such-id path.
  - `run` is patched so the underlying run_and_persist never calls yfinance.
  - `ask` is exercised on both the no-API-key and mocked-agent paths.
"""
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from app.cli import app


runner = CliRunner()


# ---------------------------------------------------------------------------
# Trivial wiring tests
# ---------------------------------------------------------------------------
def test_help_lists_every_command():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("strategies", "run", "ask", "history", "show"):
        assert cmd in result.stdout


def test_strategies_lists_every_registered_name():
    from app.strategies import STRATEGY_REGISTRY
    result = runner.invoke(app, ["strategies"])
    assert result.exit_code == 0
    for name in STRATEGY_REGISTRY:
        assert name in result.stdout


# ---------------------------------------------------------------------------
# `show` — DB lookup, no network
# ---------------------------------------------------------------------------
def test_show_missing_id_exits_one():
    result = runner.invoke(app, ["show", "999999"])
    assert result.exit_code == 1
    assert "not found" in result.stdout


# ---------------------------------------------------------------------------
# `history` — DB-only, must not crash on either empty or populated state
# ---------------------------------------------------------------------------
def test_history_renders_without_crashing():
    """Whether the DB has rows or not, history must return cleanly."""
    result = runner.invoke(app, ["history", "--limit", "5"])
    assert result.exit_code == 0
    # Either the empty-state hint or the table header should appear.
    assert "No past runs" in result.stdout or "Past backtest runs" in result.stdout


# ---------------------------------------------------------------------------
# `run` — patch run_and_persist so we don't hit yfinance
# ---------------------------------------------------------------------------
def test_run_invokes_service_and_prints_metrics():
    fake_row = MagicMock()
    fake_row.ticker = "AAPL"
    fake_row.strategy_name = "SMACrossover"
    fake_row.strategy_params = {"fast_period": 50, "slow_period": 200}
    fake_row.period = "1y"
    fake_row.total_return = 0.12
    fake_row.cagr = 0.12
    fake_row.sharpe_ratio = 1.5
    fake_row.max_drawdown = -0.08
    fake_row.win_rate = 0.6
    fake_row.num_trades = 4
    fake_row.trades = []

    with patch("app.cli.run_and_persist", return_value=fake_row):
        result = runner.invoke(app, [
            "run", "-t", "AAPL", "-s", "SMACrossover",
            "-p", "1y", "--params", '{"fast_period": 50, "slow_period": 200}',
        ])
    assert result.exit_code == 0
    assert "AAPL" in result.stdout
    assert "+12.00%" in result.stdout


def test_run_unknown_strategy_exits_one():
    result = runner.invoke(app, ["run", "-t", "AAPL", "-s", "DoesNotExist"])
    assert result.exit_code == 1
    assert "Unknown strategy" in result.stdout


def test_run_invalid_params_json_exits_one():
    result = runner.invoke(app, [
        "run", "-t", "AAPL", "-s", "SMACrossover",
        "--params", "{not valid json",
    ])
    assert result.exit_code == 1
    assert "Invalid --params JSON" in result.stdout


# ---------------------------------------------------------------------------
# `ask` — API-key error path + mocked happy path
# ---------------------------------------------------------------------------
def test_ask_without_api_key_exits_two(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    # load_dotenv() in the command may re-populate from a .env file — patch it
    # out so the test is hermetic.
    with patch("dotenv.load_dotenv", return_value=False):
        result = runner.invoke(app, ["ask", "Backtest SMA on AAPL"])
    assert result.exit_code == 2
    assert "ANTHROPIC_API_KEY" in result.stdout


def test_ask_with_mocked_agent_succeeds(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake-for-test")

    fake_row = MagicMock()
    fake_row.ticker = "AAPL"
    fake_row.strategy_name = "SMACrossover"
    fake_row.strategy_params = {}
    fake_row.period = "5y"
    fake_row.total_return = 0.34
    fake_row.cagr = 0.06
    fake_row.sharpe_ratio = 0.8
    fake_row.max_drawdown = -0.18
    fake_row.win_rate = 0.55
    fake_row.num_trades = 12
    fake_row.trades = []

    with patch("app.agent.runner.run_agent",
               return_value=("AAPL with SMA returned 34%, a respectable result.", fake_row)):
        result = runner.invoke(app, ["ask", "Backtest SMA on AAPL for 5 years"])
    assert result.exit_code == 0
    assert "AAPL" in result.stdout
    assert "+34.00%" in result.stdout
    assert "respectable" in result.stdout  # the LLM summary panel
