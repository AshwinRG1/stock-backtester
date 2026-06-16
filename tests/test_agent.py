"""Phase 5 agent tests. All Anthropic calls are mocked — zero cost, no network.

Covers three scenarios:
  - happy path: tool_use turn followed by end_turn → summary + persisted run
  - missing API key: route returns 503 cleanly
  - prompt validation: Pydantic rejects empty / oversized prompts before
    the Anthropic client is even constructed
"""
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.api.schemas import BacktestRequest


# ---------------------------------------------------------------------------
# Fake Anthropic response builders
# ---------------------------------------------------------------------------
def _tool_use_response(tool_input: dict):
    """Mimic a stop_reason='tool_use' response from anthropic.messages.create."""
    tool_block = SimpleNamespace(
        type="tool_use",
        id="toolu_fake_id",
        name="run_backtest",
        input=tool_input,
    )
    return SimpleNamespace(stop_reason="tool_use", content=[tool_block])


def _end_turn_response(summary_text: str):
    """Mimic a stop_reason='end_turn' response with a single text block."""
    text_block = SimpleNamespace(type="text", text=summary_text)
    return SimpleNamespace(stop_reason="end_turn", content=[text_block])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_agent_route_503_when_api_key_missing(monkeypatch):
    """Without ANTHROPIC_API_KEY the endpoint must return 503, not crash."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client = TestClient(app)
    response = client.post("/api/agent", json={"prompt": "Backtest SMA on AAPL"})
    assert response.status_code == 503
    assert "ANTHROPIC_API_KEY" in response.json()["detail"]


def test_agent_route_422_on_empty_prompt():
    """Pydantic rejects empty prompts before any Anthropic call is made."""
    client = TestClient(app)
    response = client.post("/api/agent", json={"prompt": ""})
    assert response.status_code == 422


def test_agent_route_422_on_oversized_prompt():
    """Pydantic rejects prompts over 2000 chars."""
    client = TestClient(app)
    response = client.post("/api/agent", json={"prompt": "x" * 2001})
    assert response.status_code == 422


def test_agent_tool_use_loop_with_mocked_anthropic(monkeypatch):
    """The full tool-use loop returns (summary, run) without any real LLM call.

    Strategy: patch app.agent.parser.get_client to return a MagicMock whose
    messages.create returns two scripted responses — first the tool_use, then
    the end_turn. Also patch run_and_persist so we don't need yfinance.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake-for-test")

    fake_client = MagicMock()
    fake_client.messages.create.side_effect = [
        _tool_use_response({
            "ticker": "AAPL",
            "strategy_name": "SMACrossover",
            "strategy_params": {"fast_period": 50, "slow_period": 200},
            "period": "1y",
        }),
        _end_turn_response("AAPL with a 50/200 SMA crossover returned 12.3% over the year."),
    ]

    # Build a fake persisted BacktestRun with enough attributes for the
    # runner to construct its trimmed tool_result dict.
    fake_run = MagicMock()
    fake_run.id = 999
    fake_run.ticker = "AAPL"
    fake_run.strategy_name = "SMACrossover"
    fake_run.strategy_params = {"fast_period": 50, "slow_period": 200}
    fake_run.period = "1y"
    fake_run.total_return = 0.123
    fake_run.cagr = 0.123
    fake_run.sharpe_ratio = 1.5
    fake_run.max_drawdown = -0.08
    fake_run.win_rate = 0.6
    fake_run.num_trades = 4
    fake_run.equity_curve = [{"date": "2025-01-01", "value": 10000.0},
                             {"date": "2026-01-01", "value": 11230.0}]

    with patch("app.agent.runner.get_client", return_value=fake_client), \
         patch("app.agent.runner.run_and_persist", return_value=fake_run):
        from app.agent.runner import run_agent

        summary, run = run_agent("Test prompt", db=MagicMock())

    assert "AAPL" in summary
    assert run is fake_run
    assert fake_client.messages.create.call_count == 2


def test_runner_propagates_engine_errors_to_claude(monkeypatch):
    """When run_and_persist raises HTTPException, the runner should hand the
    error back to Claude as a tool_result rather than aborting — so Claude
    can explain or retry. Verified by counting that messages.create is called
    a second time after the engine failure."""
    from fastapi import HTTPException
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake-for-test")

    fake_client = MagicMock()
    fake_client.messages.create.side_effect = [
        _tool_use_response({"ticker": "BADTICKER", "strategy_name": "SMACrossover"}),
        _end_turn_response("Sorry, BADTICKER returned no data."),
    ]

    def _raise_404(*args, **kwargs):
        raise HTTPException(status_code=404, detail="No data for 'BADTICKER'")

    with patch("app.agent.runner.get_client", return_value=fake_client), \
         patch("app.agent.runner.run_and_persist", side_effect=_raise_404):
        from app.agent.runner import run_agent

        # No tool result means run_agent should raise after end_turn
        # because persisted_run is None.
        with pytest.raises(RuntimeError, match="ended the turn"):
            run_agent("Test bad ticker", db=MagicMock())

    # But Claude was still given the tool_result error and got a chance
    # to respond — both calls happened.
    assert fake_client.messages.create.call_count == 2
