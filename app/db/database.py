"""SQLAlchemy engine, session factory, and declarative base.

Single source of truth for database connectivity. Both the ORM models
(app/db/models.py) and the FastAPI route handlers (app/api/routes.py)
import from here so there is exactly one engine and one Base in the
process.

SQLite is used for local development. Swap DB_URL to a PostgreSQL DSN
to switch — the ORM models and migrations remain identical.
"""
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker


# Project root = parents[2] from this file:
#   app/db/database.py → app/db → app → stock-backtester (project root)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DATA_DIR = _PROJECT_ROOT / "data"
_DATA_DIR.mkdir(exist_ok=True)  # SQLite cannot create the parent folder itself.

DB_PATH = _DATA_DIR / "backtests.db"
DB_URL = f"sqlite:///{DB_PATH}"

# check_same_thread=False is required because FastAPI may share a
# connection across threads (via dependency injection). Safe for
# SQLite when each request uses its own session, which get_db() guarantees.
engine = create_engine(DB_URL, connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    """Declarative base class for all ORM models."""
    pass


def get_db():
    """FastAPI dependency that yields a request-scoped DB session.

    Used as `db: Session = Depends(get_db)` in route handlers. The
    try/finally ensures the session is closed even if the handler raises.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
