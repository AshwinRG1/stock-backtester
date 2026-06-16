"""LLM client, tool schema, and system prompt for the natural-language agent.

This module owns three things and nothing else:
  - get_client()        : lazy Anthropic SDK client factory (reads env var)
  - RUN_BACKTEST_TOOL   : tool schema Claude sees and must call
  - SYSTEM_PROMPT       : instructions for translating English -> tool call

Orchestration (the tool-use loop) lives in app/agent/runner.py, not here.
Keeping the "what we tell Claude" config separate from the "how we drive
the conversation" code makes the prompt and tool schema easy to iterate
on without touching control flow.
"""
import os

from anthropic import Anthropic

from app.strategies import STRATEGY_REGISTRY


# claude-sonnet-4-6 is the strongest model for structured extraction with
# tool use. Override via env var if you want to A/B against haiku for cost.
DEFAULT_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")


class MissingAPIKeyError(RuntimeError):
    """Raised when ANTHROPIC_API_KEY is not set.

    The API route catches this and returns 503 so the rest of the API
    stays usable even when the agent is unconfigured.
    """


def get_client() -> Anthropic:
    """Return a configured Anthropic client.

    Lazy on purpose: at import time we don't know whether the key is set,
    and we don't want every test that imports app.agent.* to need the key.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise MissingAPIKeyError(
            "ANTHROPIC_API_KEY is not set. "
            "Copy .env.example to .env and add your key from console.anthropic.com."
        )
    return Anthropic(api_key=api_key)


# ---------------------------------------------------------------------------
# Tool schema — what Claude is allowed to call.
# ---------------------------------------------------------------------------
# The enum on strategy_name is a hard guard rail: Claude cannot invent a
# strategy that does not exist in STRATEGY_REGISTRY. The required-fields
# list keeps the door open for sensible defaults on the rest.
RUN_BACKTEST_TOOL = {
    "name": "run_backtest",
    "description": (
        "Run a backtest against historical OHLCV data and return the "
        "performance metrics. Call this whenever the user describes a "
        "strategy they want to test."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "Yahoo Finance ticker symbol (e.g. 'AAPL', 'MSFT', '^NDX', '^GSPC').",
            },
            "strategy_name": {
                "type": "string",
                "enum": sorted(STRATEGY_REGISTRY.keys()),
                "description": "Which trading strategy to test. Must be one of the listed names.",
            },
            "strategy_params": {
                "type": "object",
                "description": (
                    "Optional kwargs passed to the strategy constructor. Omit or pass "
                    "an empty object to use the strategy's own defaults. Example for "
                    "SMACrossover: {\"fast_period\": 50, \"slow_period\": 200}."
                ),
            },
            "period": {
                "type": "string",
                "description": (
                    "Lookback window in yfinance syntax: '1mo', '6mo', '1y', '2y', "
                    "'5y', '10y', '40y', 'max'. Default '1y' if the user does not say."
                ),
            },
            "interval": {
                "type": "string",
                "description": "Bar interval: '1d' (daily) is the default and the only widely-tested option.",
            },
            "initial_capital": {
                "type": "number",
                "description": "Starting portfolio value in dollars. Default 10000.",
            },
        },
        "required": ["ticker", "strategy_name"],
    },
}


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
# Pinning the registry list into the prompt makes the model's available
# choices unambiguous even if the tool's enum is ignored. The "summary"
# instructions at the end are what produce the trailing natural-language
# response after the tool returns.
SYSTEM_PROMPT = f"""You are a quantitative trading assistant. Users describe a backtest in
plain English and you translate that into a call to the run_backtest tool.

Available strategies (this is the complete list):
{chr(10).join(f"  - {name}" for name in sorted(STRATEGY_REGISTRY))}

Rules:
  - Always call run_backtest exactly once for any backtest request.
  - If the user does not specify strategy_params, omit the field — each
    strategy has sensible defaults baked in.
  - If the user does not specify a period, default to "1y".
  - If the user does not specify initial_capital, default to 10000.
  - If the user's request is too vague to map to a single strategy (e.g.
    "test something good on AAPL"), pick the most obvious match and
    explain your choice in the final summary.

After the tool returns, write a concise 2-3 sentence summary covering:
  - The headline result (total return, CAGR)
  - Risk-adjusted performance (Sharpe ratio — over 1.0 is generally considered good)
  - Drawdown context (the worst peak-to-trough drop)

Keep the summary plain-language. No bullet points. No code blocks.
"""
