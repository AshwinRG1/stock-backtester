"""Orchestration for the natural-language backtest agent.

The tool-use loop:

  1. Send the user's prompt + tool definition to Claude.
  2. If Claude responds with stop_reason="tool_use", extract the
     run_backtest call, execute it via app.api.routes.run_and_persist(),
     and feed the result back as a tool_result message.
  3. Continue until Claude responds with stop_reason="end_turn", at
     which point the final text block is the user-facing summary.

We cap iterations at MAX_TURNS as a safety net — under the current prompt
Claude is expected to call run_backtest exactly once, so two turns is the
normal path. Anything beyond five suggests a prompt or schema bug.
"""
from __future__ import annotations

import json
from typing import Any

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.agent.parser import (
    DEFAULT_MODEL,
    RUN_BACKTEST_TOOL,
    SYSTEM_PROMPT,
    get_client,
)
from app.api.routes import run_and_persist
from app.api.schemas import BacktestRequest
from app.db.models import BacktestRun


MAX_TURNS = 5
MAX_TOKENS = 1024


def run_agent(prompt: str, db: Session) -> tuple[str, BacktestRun]:
    """Drive the tool-use loop and return (summary, persisted BacktestRun).

    Raises:
        MissingAPIKeyError: when ANTHROPIC_API_KEY is not set (re-raised
            from parser.get_client() so the route can map it to 503).
        HTTPException:      when the engine itself fails (bad ticker / bad
            params). Bubbles up unchanged so the route returns the same
            404/422 the /api/backtest route would.
        RuntimeError:       when Claude stops without calling the tool or
            exceeds MAX_TURNS — either is a prompt-engineering bug.
    """
    client = get_client()
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]

    persisted_run: BacktestRun | None = None

    for _turn in range(MAX_TURNS):
        response = client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=[RUN_BACKTEST_TOOL],
            messages=messages,
        )

        if response.stop_reason == "tool_use":
            tool_use = next(b for b in response.content if b.type == "tool_use")
            tool_result, persisted_run = _execute_run_backtest(tool_use.input, db)

            # Echo the assistant turn back into history, then add the tool
            # result as a user-role message — required shape for the next
            # call to messages.create().
            messages.append({"role": "assistant", "content": response.content})
            messages.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": json.dumps(tool_result),
                }],
            })
            continue

        if response.stop_reason == "end_turn":
            if persisted_run is None:
                raise RuntimeError(
                    "Claude ended the turn without calling run_backtest. "
                    "Check the system prompt and tool description."
                )
            summary = "".join(b.text for b in response.content if b.type == "text")
            return summary, persisted_run

        # max_tokens, refusal, pause_turn, or anything else unexpected
        raise RuntimeError(f"Unexpected stop_reason: {response.stop_reason!r}")

    raise RuntimeError(f"Agent exceeded MAX_TURNS={MAX_TURNS} without finishing.")


def _execute_run_backtest(
    tool_input: dict[str, Any],
    db: Session,
) -> tuple[dict[str, Any], BacktestRun | None]:
    """Run the backtest and shape the result for Claude.

    Returns (trimmed_dict_for_claude, persisted_run_or_none).

    The trimmed dict keeps the prompt token count small — Claude only needs
    the headline metrics to write a 2-3 sentence summary, not the full
    500-point equity curve. The persisted ORM row is what we return to the
    API client, so the user still gets the complete BacktestResponse.

    Errors from the engine (404 / 422) come back as HTTPException; we
    convert those to a tool_result error string so Claude can either retry
    with corrected args or explain the failure to the user.
    """
    # Defaults for fields Claude may omit per the system prompt.
    req = BacktestRequest(
        ticker=tool_input["ticker"],
        strategy_name=tool_input["strategy_name"],
        strategy_params=tool_input.get("strategy_params", {}),
        period=tool_input.get("period", "1y"),
        interval=tool_input.get("interval", "1d"),
        initial_capital=tool_input.get("initial_capital", 10_000.0),
    )

    try:
        run = run_and_persist(req, db)
    except HTTPException as e:
        # Hand the error back to Claude rather than aborting — lets the
        # model explain the failure or retry with corrected args.
        return {"error": e.detail, "status_code": e.status_code}, None

    return {
        "id": run.id,
        "ticker": run.ticker,
        "strategy_name": run.strategy_name,
        "strategy_params": run.strategy_params,
        "period": run.period,
        "total_return": run.total_return,
        "cagr": run.cagr,
        "sharpe_ratio": run.sharpe_ratio,
        "max_drawdown": run.max_drawdown,
        "win_rate": run.win_rate,
        "num_trades": run.num_trades,
        "period_start": run.equity_curve[0]["date"] if run.equity_curve else None,
        "period_end": run.equity_curve[-1]["date"] if run.equity_curve else None,
    }, run
