import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# Database connection settings: use environment variables or defaults
DB_USER = os.getenv("APP_DB_USER")
DB_PASSWORD = os.getenv("APP_DB_PASSWORD")
DB_NAME = os.getenv("APP_DB_NAME")
DB_HOST = os.getenv("APP_DB_HOST", "postgres")
DB_PORT = os.getenv("APP_DB_PORT", "5432")

# Construct the database URL
DATABASE_URL = (
    f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# Create engine and session factory
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
# Base class for ORM models
Base = declarative_base()

def get_db():
    """
    Dependency for FastAPI: provides a SQLAlchemy session and ensures it's closed.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

from contextlib import contextmanager

@contextmanager
def get_session():
    """
    Context manager to provide a transactional session.
    Usage: with get_session() as db: ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()