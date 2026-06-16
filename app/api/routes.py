"""FastAPI route handlers for the backtest API.

Four endpoints:
  GET  /strategies        — list strategies registered for use
  POST /backtest          — run a backtest and persist the result
  GET  /results           — paginated list of past runs (lightweight)
  GET  /results/{run_id}  — full payload for a single run

All handlers are sync `def`. FastAPI auto-offloads sync handlers to its
threadpool (~40 concurrent by default), which is the right scaling model
for this app: the bottleneck is the yfinance HTTP call + the per-bar
pandas loop in the engine, both blocking. Switching to async + asyncio.
to_thread() is a single-file change if we ever hit thread starvation.
"""
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agent.parser import MissingAPIKeyError
from app.agent.runner import run_agent
from app.api.schemas import (
    AgentRequest,
    AgentResponse,
    BacktestRequest,
    BacktestResponse,
    RunSummary,
)
from app.api.services import run_and_persist
from app.db.database import get_db
from app.db.models import BacktestRun
from app.strategies import STRATEGY_REGISTRY


router = APIRouter()


@router.get("/strategies", response_model=list[str])
def list_strategies() -> list[str]:
    """Return all strategy names registered for use in POST /backtest."""
    return sorted(STRATEGY_REGISTRY.keys())


@router.post(
    "/backtest",
    response_model=BacktestResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_backtest(
    req: BacktestRequest,
    db: Session = Depends(get_db),
) -> BacktestRun:
    """Run a backtest, persist the result, return the full payload."""
    return run_and_persist(req, db)


@router.get("/results", response_model=list[RunSummary])
def list_results(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> list[BacktestRun]:
    """Recent runs first. Pageable via ?limit=&offset="""
    stmt = (
        select(BacktestRun)
        .order_by(BacktestRun.created_at.desc(), BacktestRun.id.desc())
        .limit(limit)
        .offset(offset)
    )
    return list(db.execute(stmt).scalars())


@router.get("/results/{run_id}", response_model=BacktestResponse)
def get_result(run_id: int, db: Session = Depends(get_db)) -> BacktestRun:
    """Full payload for one run — includes the equity curve and trade log."""
    row = db.get(BacktestRun, run_id)
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Backtest run {run_id} not found.",
        )
    return row


@router.post("/agent", response_model=AgentResponse)
def create_agent_backtest(
    req: AgentRequest,
    db: Session = Depends(get_db),
) -> AgentResponse:
    """Natural-language entry point — Claude parses the prompt, calls the
    run_backtest tool, and returns a plain-English summary alongside the
    full BacktestResponse.

    Status codes:
      503  ANTHROPIC_API_KEY not set (rest of the API still works).
      404  Bad ticker — yfinance returned no data.
      422  Bad strategy_params — strategy constructor signature mismatch.
      200  Success.
    """
    try:
        summary, run = run_agent(req.prompt, db)
    except MissingAPIKeyError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        )

    return AgentResponse(
        summary=summary,
        result=BacktestResponse.model_validate(run),
    )
