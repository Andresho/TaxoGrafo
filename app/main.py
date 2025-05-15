"""
Entry point for the Pipeline API server.
"""
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
import scripts.pipeline_tasks as pt
import scripts.batch_utils as batch_utils
from db import SessionLocal, get_session
from sqlalchemy.orm import Session
import crud.pipeline_run as crud_runs
import gets
import os
from pathlib import Path
import shutil
import requests
import time
import schemas
import crud.resource as crud_resource
import crud.pipeline_run_resource as crud_pipeline_run_resource
import uuid

UPLOADS_ORIGINALS_DIR = Path(os.getenv("AIRFLOW_DATA_DIR", "./data")) / "uploads" / "originals"
PROCESSED_TXT_DIR = Path(os.getenv("AIRFLOW_DATA_DIR", "./data")) / "uploads" / "processed_txt"

app = FastAPI(
    title="Airflow Pipeline API",
    description="API para expor chamadas aos scripts de pipeline",
    version="0.1.0",
)

app.include_router(gets.router)

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
        
@app.post("/api/v1/resources/upload", response_model=schemas.ResourceResponse, tags=["resources"])
async def upload_resource(
    file: UploadFile = File(...)
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Nome do arquivo não pode ser vazio.")

    if not (file.filename.lower().endswith(".pdf") or file.filename.lower().endswith(".txt")):
        raise HTTPException(status_code=400, detail="Tipo de arquivo não suportado. Apenas PDF ou TXT.")

    resource_id = uuid.uuid4()
    original_filename = file.filename
    mime_type = file.content_type or "application/octet-stream"

    relative_original_path = Path("uploads") / "originals" / str(resource_id) / original_filename
    absolute_original_dir = UPLOADS_ORIGINALS_DIR / str(resource_id)
    absolute_original_dir.mkdir(parents=True, exist_ok=True)
    absolute_original_file_path = absolute_original_dir / original_filename

    try:
        with open(absolute_original_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Não foi possível salvar o arquivo: {e}")
    finally:
        await file.close()

    with get_session() as db:
        try:
            db_resource = crud_resource.create_resource(
                db,
                resource_id=resource_id,
                original_filename=original_filename,
                original_mime_type=mime_type,
                original_file_path=str(relative_original_path)
            )
            return db_resource
        except Exception as e_db:
            db.rollback()
            logging.error(f"Erro de banco de dados ao criar recurso: {e_db}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Erro de banco de dados ao criar recurso: {e_db}")






@app.get("/api/v1/resources/{resource_id}", response_model=schemas.ResourceResponse, tags=["resources"])
def get_resource_details(
    resource_id: uuid.UUID
):
    with get_session() as db:
        db_resource = crud_resource.get_resource(db, resource_id)
        if db_resource is None:
            raise HTTPException(status_code=404, detail="Recurso não encontrado")
        return db_resource


@app.post("/pipeline/{run_id}/init", tags=["pipeline"])
def init_pipeline(run_id: str, init_request: schemas.PipelineInitRequest):
    """
    Endpoint para inicializar pipeline e disparar o DAG no Airflow.
    - Associa recursos à run.
    - Cria pasta de execução (mas não copia inputs aqui, isso é feito no DAG).
    - Dispara um DagRun.
    """
    base_dir = Path(os.getenv("AIRFLOW_DATA_DIR", "./data"))
    run_data_dir = base_dir / run_id
    airflow_api_url = os.getenv("AIRFLOW_API_URL", "http://airflow-webserver:8080")
    dag_id = os.getenv("AIRFLOW_DAG_ID", "knowledge_graph_pipeline")
    airflow_user = os.getenv("AIRFLOW_API_USER", "admin")
    airflow_pass = os.getenv("AIRFLOW_API_PASSWORD", "admin")

    if not init_request.resource_ids:
        raise HTTPException(status_code=400, detail="Pelo menos um resource_id deve ser fornecido.")

    # Preparar o payload para o banco de dados, convertendo UUIDs para strings
    payload_for_db = {
        "resource_ids": [str(rid) for rid in init_request.resource_ids],
        "skip_graphrag": init_request.skip_graphrag
    }

    with get_session() as db:
        db_resources = crud_resource.get_resources_by_ids(db, init_request.resource_ids)
        if len(db_resources) != len(init_request.resource_ids):
            found_ids = {str(r.resource_id) for r in db_resources}
            missing_ids = [str(rid) for rid in init_request.resource_ids if str(rid) not in found_ids]
            raise HTTPException(status_code=404, detail=f"Recursos não encontrados: {', '.join(missing_ids)}")

        crud_runs.create_run(db, run_id, trigger_source='api', payload=payload_for_db)
        crud_pipeline_run_resource.link_resources_to_run(db, run_id,
                                                         init_request.resource_ids)

    try:
        if not run_data_dir.exists():
            run_data_dir.mkdir(parents=True, exist_ok=False)

        output_dir = run_data_dir / "output"
        required_files = [
            "communities.parquet", "community_reports.parquet", "documents.parquet",
            "entities.parquet", "relationships.parquet", "text_units.parquet",
        ]
        skip_graphrag_actual = init_request.skip_graphrag or \
                               (output_dir.is_dir() and all((output_dir / f).exists() for f in required_files))

        airflow_conf_payload = {
            "pipeline_id": run_id,
            "resource_ids_for_run": [str(rid) for rid in init_request.resource_ids],
            "skip_graphrag": skip_graphrag_actual,
        }

        payload_for_airflow_dag_run = {
            "dag_run_id": run_id + "_" + str(int(time.time())),
            "conf": airflow_conf_payload,
        }

        resp = requests.post(
            f"{airflow_api_url}/api/v1/dags/{dag_id}/dagRuns",
            json=payload_for_airflow_dag_run,
            auth=(airflow_user, airflow_pass),
            timeout=10,
        )
        resp.raise_for_status()
        return {"status": "success", "run_id": run_id, "dag_response": resp.json()}
    except requests.exceptions.HTTPError as http_err:
        raise HTTPException(
            status_code=http_err.response.status_code,
            detail=f"Falha ao disparar DAG {dag_id}: {http_err.response.text}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/v1/resources/{resource_id}", response_model=schemas.ResourceResponse, tags=["resources"])
def update_resource(
        resource_id: uuid.UUID,
        resource_update: schemas.ResourceUpdate
):
    with get_session() as db:
        db_resource_check = crud_resource.get_resource(db, resource_id)
        if db_resource_check is None:
            raise HTTPException(status_code=404, detail="Recurso não encontrado")

        status_to_set = resource_update.status if resource_update.status is not None else db_resource_check.status
        processed_txt_path_to_set = resource_update.processed_txt_path
        error_message_to_set = resource_update.error_message

        updated_resource = crud_resource.update_resource_status(
            db,
            resource_id=resource_id,
            status=status_to_set,
            processed_txt_path=processed_txt_path_to_set,
            error_message=error_message_to_set
        )

        if not updated_resource:
            raise HTTPException(status_code=500,
                                detail="Falha ao atualizar o recurso no banco de dados após verificação inicial.")
        return updated_resource




if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)