import os
import sys
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool
from alembic import context

# ensure `app` package is on PYTHONPATH
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
PROJECT_ROOT_FOR_ALEMBIC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT_FOR_ALEMBIC)

# this is the Alembic Config object
config = context.config

# Interpret the config file for Python logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# import Base and DATABASE_URL
from app.db import DATABASE_URL, Base
# ensure models are imported so metadata is populated
import app.models as models # noqa: F401

target_metadata = Base.metadata

def run_migrations_offline():
    """Run migrations in 'offline' mode (emit SQL)."""
    context.configure(
        url=DATABASE_URL,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online():
    """Run migrations in 'online' mode (apply to DB)."""
    configuration = config.get_section(config.config_ini_section)
    configuration['sqlalchemy.url'] = DATABASE_URL
    connectable = engine_from_config(
        configuration,
        prefix='sqlalchemy.',
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()