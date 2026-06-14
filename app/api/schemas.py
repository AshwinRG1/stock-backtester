"""Pydantic request/response schemas for the FastAPI layer.

Three roles:
  - BacktestRequest   : validates incoming POST /backtest bodies
  - BacktestResponse  : full POST /backtest + GET /results/{id} body
  - RunSummary        : lightweight GET /results list-view item

BacktestResponse and RunSummary use ConfigDict(from_attributes=True)
so a route handler can do `BacktestResponse.model_validate(orm_run)`
and Pydantic reads the fields straight off the BacktestRun ORM
instance. Keeps the ORM↔schema mapping in one place (Pydantic) instead
of hand-mapping in every route.
"""
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.strategies import STRATEGY_REGISTRY


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------
class BacktestRequest(BaseModel):
    """Incoming POST /backtest body."""

    ticker: str = Field(
        ..., min_length=1, max_length=32,
        description="Yahoo Finance ticker symbol, e.g. 'AAPL' or '^NDX'.",
    )
    strategy_name: str = Field(
        ...,
        description="Must be a key in app.strategies.STRATEGY_REGISTRY.",
    )
    strategy_params: dict[str, Any] = Field(
        default_factory=dict,
        description="Kwargs forwarded to the strategy constructor.",
    )
    period: str = Field(default="1y", description="yfinance period, e.g. '1y', '6mo', '40y'.")
    interval: str = Field(default="1d", description="Bar interval, e.g. '1d'.")
    initial_capital: float = Field(default=10_000.0, gt=0)

    @field_validator("strategy_name")
    @classmethod
    def _strategy_must_be_registered(cls, v: str) -> str:
        if v not in STRATEGY_REGISTRY:
            available = ", ".join(sorted(STRATEGY_REGISTRY))
            raise ValueError(f"Unknown strategy '{v}'. Available: {available}.")
        return v


# ---------------------------------------------------------------------------
# Nested response types
# ---------------------------------------------------------------------------
class EquityPoint(BaseModel):
    """One sample of the equity curve."""
    date: str
    value: float


class Trade(BaseModel):
    """One completed round-trip trade."""
    entry_date:  str
    exit_date:   str
    entry_price: float
    exit_price:  float
    pnl:         float
    return_pct:  float


# ---------------------------------------------------------------------------
# Response: full payload
# ---------------------------------------------------------------------------
class BacktestResponse(BaseModel):
    """Full backtest payload — POST /backtest and GET /results/{id}.

    Built from a BacktestRun ORM row via model_validate(orm_row).
    """
    model_config = ConfigDict(from_attributes=True)

    id:               int
    ticker:           str
    strategy_name:    str
    strategy_params:  dict[str, Any]
    period:           str
    interval:         str
    initial_capital:  float
    created_at:       datetime

    total_return:     float
    sharpe_ratio:     float
    max_drawdown:     float
    cagr:             float
    win_rate:         float | None = None
    num_trades:       int

    equity_curve:     list[EquityPoint]
    trades:           list[Trade]


# ---------------------------------------------------------------------------
# Response: list view (no equity_curve / no trades — keeps the payload
# small when paging through many runs)
# ---------------------------------------------------------------------------
class RunSummary(BaseModel):
    """Lightweight GET /results list-view item."""
    model_config = ConfigDict(from_attributes=True)

    id:            int
    ticker:        str
    strategy_name: str
    created_at:    datetime
    total_return:  float
    sharpe_ratio:  float
    num_trades:    int
