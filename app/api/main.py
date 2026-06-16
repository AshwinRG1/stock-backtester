"""FastAPI application entry point.

Run locally with:
    uvicorn app.api.main:app --reload

Routes:
    GET  /health                  Liveness probe (no DB).
    GET  /api/strategies          List registered strategies.
    POST /api/backtest            Run + persist a backtest.
    GET  /api/results             Paginated list of past runs (light).
    GET  /api/results/{run_id}    Full payload for one run.
    POST /api/agent               Natural-language backtest via Claude.
    GET  /docs                    Auto-generated Swagger UI.
    GET  /redoc                   Auto-generated ReDoc.
"""
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import routes
from app.db.database import Base, engine


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Startup setup:
      1. Load .env so ANTHROPIC_API_KEY (and any future secrets) land in
         os.environ for the rest of the process. Centralised here so the
         agent module stays env-agnostic and easy to test.
      2. Ensure tables exist. Alembic is the production source of truth
         (`alembic upgrade head` in CI/deploy); this call is a no-op
         against an already-migrated schema and just lets the API come up
         on a fresh machine where someone forgot the Alembic step.
    """
    load_dotenv()
    Base.metadata.create_all(engine)
    yield


app = FastAPI(
    title="Stock Backtester API",
    description=(
        "Run trading-strategy backtests against historical OHLCV data and "
        "retrieve stored results. POST /api/backtest with a strategy name "
        "and ticker; GET /api/results to browse past runs."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# Wide-open CORS for development. allow_credentials is intentionally left
# at its False default — combining "*" with credentials is rejected by
# the CORS spec, and we have no credentials to send anyway. In production,
# replace allow_origins with the actual frontend origin(s).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe — does not touch the DB or any external service."""
    return {"status": "ok"}


app.include_router(routes.router, prefix="/api")
