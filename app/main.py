"""
Entry point for the Pipeline API server.
"""
from fastapi import FastAPI, HTTPException

import scripts.pipeline_tasks as pt
import scripts.batch_utils as batch_utils
from db import SessionLocal, get_session
import crud.pipeline_run as crud_runs

import os
from pathlib import Path
import shutil
import requests
import time

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

    if batch_id == "fake_id_for_indepotency":
        return {"status": "completed", "run_id": run_id, "processed": True}

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
    if batch_id == "fake_id_for_indepotency":
        return {"status": "completed", "run_id": run_id, "processed": True}

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
        # Update pipeline run status to 'success'
        with get_session() as db:
            crud_runs.update_run_status(db, run_id, status='success')
        return {"status": "success", "run_id": run_id, "message": "Final outputs saved"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
@app.post("/pipeline/{run_id}/init", tags=["pipeline"])
def init_pipeline(run_id: str):
    """Endpoint para inicializar pipeline e disparar o DAG no Airflow.
    - Cria pasta de execução e copia dados iniciais (idempotente).
    - Dispara um DagRun com dag_run_id == run_id e conf.pipeline_id == run_id.
    """
    # Define diretórios locais e variáveis de conexão Airflow
    base_dir = Path(os.getenv("AIRFLOW_DATA_DIR", "./data"))
    run_dir = base_dir / run_id
    airflow_api_url = os.getenv("AIRFLOW_API_URL", "http://airflow-webserver:8080")
    dag_id = os.getenv("AIRFLOW_DAG_ID", "knowledge_graph_pipeline")
    airflow_user = os.getenv("AIRFLOW_API_USER", "admin")
    airflow_pass = os.getenv("AIRFLOW_API_PASSWORD", "admin")
    created = False
    # Register the pipeline run in the database
    with get_session() as db:
        crud_runs.create_run(db, run_id, trigger_source='api')
    try:
        # Se não existe, cria a estrutura e copia settings e input
        if not run_dir.exists():
            run_dir.mkdir(parents=True, exist_ok=False)

            src_input = base_dir / "input"
            dst_input = run_dir / "input"
            if not src_input.exists():
                raise HTTPException(status_code=500, detail="input directory not found in base data dir")
            shutil.copytree(src_input, dst_input)
            created = True
        # Dispara o DAG no Airflow via API, informando se podemos pular o Graphrag
        trigger_url = f"{airflow_api_url}/api/v1/dags/{dag_id}/dagRuns"
        # Idempotência: verifica se os arquivos do Graphrag já existem
        required_files = [
            "communities.parquet",
            "community_reports.parquet",
            "documents.parquet",
            "entities.parquet",
            "relationships.parquet",
            "text_units.parquet",
        ]
        output_dir = run_dir / "output"
        skip_graphrag = output_dir.is_dir() and all((output_dir / f).exists() for f in required_files)
        payload = {
            "dag_run_id": run_id + "_" + str(int(time.time())),
            "conf": {
                "pipeline_id": run_id,
                "skip_graphrag": skip_graphrag,
            },
        }
        resp = requests.post(
            trigger_url,
            json=payload,
            auth=(airflow_user, airflow_pass),
            timeout=10,
        )
        if resp.status_code not in (200, 201):
            raise HTTPException(
                status_code=500,
                detail=f"Failed to trigger DAG {dag_id}: {resp.status_code} {resp.text}"
            )
        return {"status": "success", "run_id": run_id, "created": created, "dag_response": resp.json()}
    except HTTPException:
        # Propaga erros controlados
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)