"""
Entry point for the Pipeline API server.
"""
from fastapi import FastAPI, HTTPException

import scripts.pipeline_tasks as pt
import scripts.batch_utils as batch_utils

import os
from pathlib import Path

app = FastAPI(
    title="Airflow Pipeline API",
    description="API para expor chamadas aos scripts de pipeline",
    version="0.1.0",
)
 


@app.get("/", tags=["health"])
async def health_check():
    """
    Health check endpoint.
    """
    return {"status": "ok", "message": "API server is running"}

# --- Pipeline task endpoints ---
@app.post("/pipeline/{run_id}/prepare-origins", tags=["pipeline"])
def prepare_origins(run_id: str):
    """Endpoint para executar task_prepare_origins em contexto de run_id."""
    try:
        pt.task_prepare_origins(run_id)
        return {"status": "success", "run_id": run_id, "message": "Origins prepared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/pipeline/{run_id}/submit-uc-generation-batch", tags=["pipeline"])
def submit_uc_generation_batch(run_id: str):
    """Endpoint para executar task_submit_uc_generation_batch em contexto de run_id."""
    try:
        batch_id = pt.task_submit_uc_generation_batch(run_id)
        return {"status": "success", "run_id": run_id, "batch_id": batch_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/pipeline/{run_id}/process-uc-generation-batch/{batch_id}", tags=["pipeline"])
def process_uc_generation_batch(run_id: str, batch_id: str):
    """Endpoint para processar resultados de geração UC em contexto de run_id."""
    try:
        ok = pt.task_process_uc_generation_batch(run_id, batch_id)
        return {"status": "success", "run_id": run_id, "processed": ok}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/pipeline/{run_id}/define-relationships", tags=["pipeline"])
def define_relationships(run_id: str):
    """Endpoint para executar task_define_relationships em contexto de run_id."""
    try:
        pt.task_define_relationships(run_id)
        return {"status": "success", "run_id": run_id, "message": "Relationships defined"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/pipeline/{run_id}/submit-difficulty-batch", tags=["pipeline"])
def submit_difficulty_batch(run_id: str):
    """Endpoint para executar task_submit_difficulty_batch em contexto de run_id."""
    try:
        batch_id = pt.task_submit_difficulty_batch(run_id)
        return {"status": "success", "run_id": run_id, "batch_id": batch_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/pipeline/{run_id}/batch-status/{batch_id}", tags=["pipeline"])
def get_batch_status(run_id: str, batch_id: str):
    """Endpoint para obter status de um batch em contexto de run_id."""
    try:
        status, output_file_id, error_file_id = batch_utils.check_batch_status(batch_id)
        return {
            "status": status,
            "run_id": run_id,
            "output_file_id": output_file_id,
            "error_file_id": error_file_id,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/pipeline/{run_id}/process-difficulty-batch/{batch_id}", tags=["pipeline"])
def process_difficulty_batch(run_id: str, batch_id: str):
    """Endpoint para processar batch de dificuldade em contexto de run_id."""
    try:
        ok = pt.task_process_difficulty_batch(run_id, batch_id)
        return {"status": "success", "run_id": run_id, "processed": ok}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/pipeline/{run_id}/finalize-outputs", tags=["pipeline"])
def finalize_outputs(run_id: str):
    """Endpoint para executar task_finalize_outputs em contexto de run_id."""
    try:
        pt.task_finalize_outputs(run_id)
        return {"status": "success", "run_id": run_id, "message": "Final outputs saved"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)