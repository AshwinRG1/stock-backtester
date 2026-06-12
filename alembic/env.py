"""Alembic migration environment.

Customised to:
  1. Import our SQLAlchemy Base from app.db.database so autogenerate
     can detect schema changes by comparing models to the live DB.
  2. Override sqlalchemy.url with the project's single source of truth
     (app.db.database.DB_URL), so we never keep two DB locations in sync.
"""
import sys
from logging.config import fileConfig
from pathlib import Path

from sqlalchemy import engine_from_config, pool

from alembic import context

# Make `app.*` importable: env.py runs from the alembic/ subdir, so we
# add the project root (one level up) to sys.path before importing.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db.database import Base, DB_URL  # noqa: E402
from app.db import models  # noqa: E402,F401  — registers BacktestRun on Base.metadata


config = context.config

# Override the URL from alembic.ini so the live DB_URL is the only knob.
config.set_main_option("sqlalchemy.url", DB_URL)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (emit SQL without a live connection)."""
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against a live connection."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            # SQLite cannot ALTER columns in place; batch mode rewrites the
            # table behind the scenes so future schema changes work.
            render_as_batch=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
