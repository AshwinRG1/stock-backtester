"""SQLAlchemy ORM models — one row per backtest run.

Schema reflects what an end user wants to retrieve after a run:
the reproducibility inputs (ticker, strategy, params, period), the
scalar performance metrics, and the time-series outputs (equity curve,
trade log) needed to redraw a chart without re-running the engine.

equity_curve and trades are stored as JSON columns rather than as
separate child tables. They are write-once / read-once payloads tied
to a single run, never queried row-by-row, and SQLite's JSON type
maps cleanly to PostgreSQL's native JSONB if/when we migrate.
"""
from datetime import datetime

from sqlalchemy import JSON, DateTime, Float, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base


class BacktestRun(Base):
    __tablename__ = "backtest_runs"

    id:              Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # --- run inputs (what the user asked for) ---
    ticker:          Mapped[str]  = mapped_column(String(32), index=True)
    strategy_name:   Mapped[str]  = mapped_column(String(64), index=True)
    strategy_params: Mapped[dict] = mapped_column(JSON, default=dict)
    period:          Mapped[str]  = mapped_column(String(16))
    interval:        Mapped[str]  = mapped_column(String(8))
    initial_capital: Mapped[float] = mapped_column(Float)

    # Server-side default so the DB owns the timestamp — avoids clock
    # drift between Python and the DB, important when multiple workers
    # eventually write concurrently.
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        index=True,
    )

    # --- scalar performance metrics (queryable, indexable) ---
    total_return: Mapped[float]         = mapped_column(Float)
    sharpe_ratio: Mapped[float]         = mapped_column(Float)
    max_drawdown: Mapped[float]         = mapped_column(Float)
    cagr:         Mapped[float]         = mapped_column(Float)
    win_rate:     Mapped[float | None]  = mapped_column(Float, nullable=True)
    num_trades:   Mapped[int]           = mapped_column(Integer)

    # --- time-series outputs (opaque blobs, retrieved whole) ---
    equity_curve: Mapped[list] = mapped_column(JSON, default=list)
    trades:       Mapped[list] = mapped_column(JSON, default=list)
