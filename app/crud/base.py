"""Base CRUD utility for bulk operations on pipeline output tables."""
from typing import Type
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert as pg_insert
import numpy as np

def add_records(db: Session, model: Type, run_id: str, records: list) -> None:
    """
    Bulk insert records for a given run_id into the specified model.
    Uses PostgreSQL ON CONFLICT DO NOTHING to skip existing primary keys.
    Each record must include an 'id' key.
    """
    if not records:
        return
    # Clean up record values (convert numpy types) and annotate run_id
    for rec in records:
        # Normalize numpy types to Python builtins for JSON serialization
        for key, val in list(rec.items()):
            if isinstance(val, np.generic):
                rec[key] = val.item()
            elif isinstance(val, np.ndarray):
                rec[key] = val.tolist()
        rec['pipeline_run_id'] = run_id
    # Build INSERT ... ON CONFLICT DO NOTHING statement
    stmt = pg_insert(model.__table__).values(records)
    # Determine primary key columns for conflict handling (supports composite keys)
    pk_cols = [col.name for col in model.__table__.primary_key.columns]
    stmt = stmt.on_conflict_do_nothing(index_elements=pk_cols)

    db.execute(stmt)