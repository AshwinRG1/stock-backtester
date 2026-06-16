"""Backtest orchestration — fetch, run, persist.

Lives in a separate module so both `app/api/routes.py` (the HTTP layer)
and `app/agent/runner.py` (the LLM tool layer) can import it without
either importing the other. Keeping the cross-cutting orchestration
out of `routes.py` also prevents future tool-using callers from
pulling in the whole FastAPI route surface just to run a backtest.
"""
from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.api.schemas import BacktestRequest
from app.data import fetcher
from app.db.models import BacktestRun
from app.engine import run_engine
from app.metrics import compute_metrics
from app.strategies import STRATEGY_REGISTRY


def run_and_persist(req: BacktestRequest, db: Session) -> BacktestRun:
    """Run a backtest, persist the row, return the ORM object.

    Raises HTTPException so the FastAPI layer can surface the error directly
    when called from a route, and the agent runner can catch and report it
    back to Claude as a tool-result error.

    Failure modes handled here:
      404 — yfinance returns no data (bad ticker, delisted, bad period).
      422 — strategy_params do not match the strategy's constructor signature.
    """
    try:
        data = fetcher.fetch_ohlcv(req.ticker, period=req.period, interval=req.interval)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))

    strategy_cls = STRATEGY_REGISTRY[req.strategy_name]
    try:
        strategy = strategy_cls(**req.strategy_params)
    except TypeError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Bad strategy_params for {req.strategy_name}: {e}",
        )

    result = run_engine(data, strategy, initial_capital=req.initial_capital, ticker=req.ticker)
    metrics = compute_metrics(result, initial_capital=req.initial_capital)
    payload = result.to_dict()

    run = BacktestRun(
        ticker=req.ticker,
        strategy_name=req.strategy_name,
        strategy_params=req.strategy_params,
        period=req.period,
        interval=req.interval,
        initial_capital=req.initial_capital,
        total_return=metrics["total_return"],
        sharpe_ratio=metrics["sharpe_ratio"],
        max_drawdown=metrics["max_drawdown"],
        cagr=metrics["cagr"],
        win_rate=metrics["win_rate"],
        num_trades=metrics["num_trades"],
        equity_curve=payload["equity_curve"],
        trades=payload["trades"],
    )
    db.add(run)
    db.commit()
    db.refresh(run)  # populates id + created_at from the DB
    return run
