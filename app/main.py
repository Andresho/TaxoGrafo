"""
Entry point for the Pipeline API server.
"""
from fastapi import FastAPI, HTTPException

import scripts.pipeline_tasks as pt
import scripts.batch_utils as batch_utils

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
@app.post("/pipeline/prepare-origins", tags=["pipeline"])
def prepare_origins():
    """Endpoint to run task_prepare_origins."""
    try:
        pt.task_prepare_origins()
        return {"status": "success", "message": "Origins prepared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/pipeline/submit-uc-generation-batch", tags=["pipeline"])
def submit_uc_generation_batch():
    """Endpoint to run task_submit_uc_generation_batch."""
    try:
        batch_id = pt.task_submit_uc_generation_batch()
        return {"status": "success", "batch_id": batch_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/pipeline/process-uc-generation-batch/{batch_id}", tags=["pipeline"])
def process_uc_generation_batch(batch_id: str):
    """Process a completed UC generation batch."""
    try:
        status, output_file_id, error_file_id = batch_utils.check_batch_status(batch_id)
        if status != 'completed':
            raise HTTPException(
                status_code=400,
                detail=f"Batch {batch_id} not completed (status: {status})",
            )
        ok = batch_utils.process_batch_results(
            batch_id,
            output_file_id,
            error_file_id,
            pt.stage2_output_ucs_dir,
            pt.GENERATED_UCS_RAW,
        )
        return {"status": "success", "processed": ok}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/pipeline/define-relationships", tags=["pipeline"])
def define_relationships():
    """Endpoint to run task_define_relationships."""
    try:
        pt.task_define_relationships()
        return {"status": "success", "message": "Relationships defined"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/pipeline/submit-difficulty-batch", tags=["pipeline"])
def submit_difficulty_batch():
    """Endpoint to run task_submit_difficulty_batch."""
    try:
        batch_id = pt.task_submit_difficulty_batch()
        return {"status": "success", "batch_id": batch_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/pipeline/batch-status/{batch_id}", tags=["pipeline"])
def get_batch_status(batch_id: str):
    """Endpoint to retrieve the status of any batch job."""
    try:
        status, output_file_id, error_file_id = batch_utils.check_batch_status(batch_id)
        return {
            "status": status,
            "output_file_id": output_file_id,
            "error_file_id": error_file_id,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/pipeline/process-difficulty-batch/{batch_id}", tags=["pipeline"])
def process_difficulty_batch(batch_id: str):
    """Process a completed difficulty evaluation batch."""
    try:
        status, output_file_id, error_file_id = batch_utils.check_batch_status(batch_id)
        if status != 'completed':
            raise HTTPException(
                status_code=400,
                detail=f"Batch {batch_id} not completed (status: {status})",
            )
        ok = batch_utils.process_batch_results(
            batch_id,
            output_file_id,
            error_file_id,
            pt.stage4_output_eval_dir,
            pt.UC_EVALUATIONS_RAW,
        )
        return {"status": "success", "processed": ok}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/pipeline/finalize-outputs", tags=["pipeline"])
def finalize_outputs():
    """Endpoint to run task_finalize_outputs."""
    try:
        pt.task_finalize_outputs()
        return {"status": "success", "message": "Final outputs saved"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)