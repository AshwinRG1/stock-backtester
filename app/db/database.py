"""SQLAlchemy engine, session factory, and declarative base.

Single source of truth for database connectivity. Both the ORM models
(app/db/models.py) and the FastAPI route handlers (app/api/routes.py)
import from here so there is exactly one engine and one Base in the
process.

SQLite is used for local development. Swap DB_URL to a PostgreSQL DSN
to switch — the ORM models and migrations remain identical.
"""
from pathlib import Path

from sqlalchemy import create_engine, event
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


@event.listens_for(engine, "connect")
def _set_sqlite_pragmas(dbapi_connection, _record):
    """Apply SQLite tuning PRAGMAs on every new connection.

    journal_mode=WAL upgrades from the default rollback journal so that
    readers no longer block on a writer (and vice versa). Critical for
    multi-user concurrency — without it, every web request that touches
    the DB serialises behind every other one. The setting is persisted
    in the DB file itself, so it sticks after the first connection.

    synchronous=NORMAL is the recommended pairing for WAL — safe against
    application crashes, and only at risk of losing the last commit on
    a hard power loss. ~10x faster than the default FULL.

    foreign_keys=ON enforces FK constraints, which SQLite leaves off by
    default for historical compatibility. Must be set per-connection.
    """
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


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
