"""
Entry point for the Pipeline API server.
"""
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, Path as FastAPIPath

from app.scripts.pipeline_stages.task_define_relationships import task_define_relationships
from app.scripts.pipeline_stages.task_finalize_outputs import task_finalize_outputs
from app.scripts.pipeline_stages.task_prepare_origins import task_prepare_origins
from app.scripts.pipeline_stages.task_process_difficulty_batch import task_process_difficulty_batch
from app.scripts.pipeline_stages.task_process_uc_generation_batch import task_process_uc_generation_batch
from app.scripts.pipeline_stages.task_submit_difficulty_batch import task_submit_difficulty_batch
from app.scripts.pipeline_stages.task_submit_uc_generation_batch import task_submit_uc_generation_batch

import app.scripts.batch_utils as batch_utils
from app.scripts.llm_client import get_llm_strategy
from app.db import get_session, get_db
from sqlalchemy.orm import Session
import app.crud.pipeline_run as crud_runs
import app.gets as gets
import os
from pathlib import Path
import shutil
import requests
import time
import app.schemas as schemas
import app.crud.resource as crud_resource
import app.crud.pipeline_run_resource as crud_pipeline_run_resource
import app.crud.pipeline_batch_job as crud_batch_jobs
import uuid
import logging

UPLOADS_ORIGINALS_DIR = Path(os.getenv("AIRFLOW_DATA_DIR", "./data")) / "uploads" / "originals"
PROCESSED_TXT_DIR = Path(os.getenv("AIRFLOW_DATA_DIR", "./data")) / "uploads" / "processed_txt"

BATCH_TYPE_UC_GENERATION = "uc_generation"
BATCH_TYPE_DIFFICULTY_ASSESSMENT = "difficulty_assessment"

VALID_BATCH_TYPES = {BATCH_TYPE_UC_GENERATION, BATCH_TYPE_DIFFICULTY_ASSESSMENT}

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
        task_prepare_origins(run_id)
        return {"status": "success", "run_id": run_id, "message": "Origins prepared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/pipeline/{run_id}/submit-batch/{batch_type}", tags=["pipeline_batch_jobs"])
def submit_llm_batch(
    run_id: str,
    batch_type: str = FastAPIPath(..., description=f"Type of batch to submit. Valid types: {list(VALID_BATCH_TYPES)}"),
    db: Session = Depends(get_db)
):
    """
    Submits a batch job of the specified type (e.g., uc_generation, difficulty_assessment)
    to the LLM for a given pipeline run.
    Manages job state in the database for idempotency and tracking.
    """
    if batch_type not in VALID_BATCH_TYPES:
        raise HTTPException(status_code=400, detail=f"Invalid batch_type: {batch_type}. Valid types are: {list(VALID_BATCH_TYPES)}")

    db_run = crud_runs.get_run(db, run_id)
    if not db_run:
        raise HTTPException(status_code=404, detail=f"Pipeline run '{run_id}' not found.")

    job_record = crud_batch_jobs.create_or_get_pipeline_batch_job(
        db,
        pipeline_run_id=run_id,
        batch_type=batch_type,
        initial_status=crud_batch_jobs.STATUS_PENDING_SUBMISSION # Tenta resetar para este estado se falhou antes
    )
    db.flush()

    if job_record.status in [crud_batch_jobs.STATUS_SUBMITTED, crud_batch_jobs.STATUS_PENDING_PROCESSING, crud_batch_jobs.STATUS_COMPLETED]:
        logging.info(f"Batch job (id: {job_record.id}, type: {batch_type}, run: {run_id}) already in status {job_record.status}. LLM Batch ID: {job_record.llm_batch_id}. Skipping submission.")
        return {
            "status": "skipped_already_active_or_completed",
            "message": f"Batch job already in status {job_record.status}.",
            "pipeline_run_id": run_id,
            "batch_type": batch_type,
            "job_id": job_record.id,
            "llm_batch_id": job_record.llm_batch_id
        }

    if job_record.status != crud_batch_jobs.STATUS_PENDING_SUBMISSION:
        crud_batch_jobs.update_pipeline_batch_job(
            db,
            job_id=job_record.id,
            status=crud_batch_jobs.STATUS_PENDING_SUBMISSION,
            llm_batch_id="",
            last_error=""
        )
    db.commit()

    llm_batch_id_from_provider = None
    try:
        if batch_type == BATCH_TYPE_UC_GENERATION:
            logging.info(f"Submitting UC generation batch for job_id: {job_record.id}")
            llm_batch_id_from_provider = task_submit_uc_generation_batch(run_id)
                                                                      
        elif batch_type == BATCH_TYPE_DIFFICULTY_ASSESSMENT:
            logging.info(f"Submitting difficulty assessment batch for job_id: {job_record.id}")
            llm_batch_id_from_provider = task_submit_difficulty_batch(run_id)

        if not llm_batch_id_from_provider:
            if job_record.status == crud_batch_jobs.STATUS_PENDING_SUBMISSION: # Ainda pendente, mas nada foi submetido
                crud_batch_jobs.update_pipeline_batch_job(db, job_id=job_record.id, status=crud_batch_jobs.STATUS_COMPLETED, last_error="No data to submit to LLM.")
                db.commit()
                logging.warning(f"No LLM batch ID returned for job_id {job_record.id} (type: {batch_type}, run: {run_id}). Likely no data. Marked as COMPLETED.")
                return {
                    "status": "completed_no_data",
                    "message": "No data was available to submit to the LLM batch.",
                    "pipeline_run_id": run_id,
                    "batch_type": batch_type,
                    "job_id": job_record.id
                }
            if llm_batch_id_from_provider == "fake_id_for_indepotency": # Tratamento da idempotência antiga
                logging.warning(f"Legacy idempotency detected from task for job_id {job_record.id}. This should be handled by job status now.")
                
                if job_record.status == crud_batch_jobs.STATUS_PENDING_SUBMISSION:
                    crud_batch_jobs.update_pipeline_batch_job(db, job_id=job_record.id, status=crud_batch_jobs.STATUS_SUBMISSION_FAILED, last_error="Legacy idempotency conflict.")
                    db.commit()
                    raise HTTPException(status_code=500, detail="Legacy idempotency conflict during batch submission.")

        # Submissão ao LLM foi bem-sucedida
        crud_batch_jobs.update_pipeline_batch_job(
            db,
            job_id=job_record.id,
            llm_batch_id=llm_batch_id_from_provider,
            status=crud_batch_jobs.STATUS_SUBMITTED
        )
        db.commit()
        logging.info(f"Batch job (id: {job_record.id}, type: {batch_type}, run: {run_id}) submitted to LLM. LLM Batch ID: {llm_batch_id_from_provider}")
        return {
            "status": "submitted",
            "message": "Batch job successfully submitted to LLM.",
            "pipeline_run_id": run_id,
            "batch_type": batch_type,
            "job_id": job_record.id,
            "llm_batch_id": llm_batch_id_from_provider
        }

    except Exception as e:
        db.rollback()
        logging.error(f"Failed to submit batch job (type: {batch_type}, run: {run_id}): {e}", exc_info=True)
        
        if job_record and job_record.id:
            try:
                with get_session() as error_db:
                    crud_batch_jobs.update_pipeline_batch_job(
                        error_db,
                        job_id=job_record.id,
                        status=crud_batch_jobs.STATUS_SUBMISSION_FAILED,
                        last_error=str(e)[:1023]
                    )
                    error_db.commit()
            except Exception as db_err:
                logging.error(f"Additionally failed to update job status to SUBMISSION_FAILED for job_id {job_record.id}: {db_err}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to submit batch job: {e}")

@app.post("/pipeline/{run_id}/define-relationships", tags=["pipeline"])
def define_relationships(run_id: str):
    """Endpoint para executar task_define_relationships em contexto de run_id."""
    try:
        task_define_relationships(run_id)
        return {"status": "success", "run_id": run_id, "message": "Relationships defined"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/pipeline/{run_id}/batch-job-status/{batch_type}", tags=["pipeline_batch_jobs"])
def get_llm_batch_job_status(
        run_id: str,
        batch_type: str = FastAPIPath(...,
                                      description=f"Type of batch to check. Valid types: {list(VALID_BATCH_TYPES)}"),
        db: Session = Depends(get_db)
):
    """
    Checks the status of an LLM batch job associated with a pipeline_run_id and batch_type.
    Updates the internal job status if the LLM job has completed.
    """
    if batch_type not in VALID_BATCH_TYPES:
        raise HTTPException(status_code=400, detail=f"Invalid batch_type: {batch_type}")

    job_record = crud_batch_jobs.get_pipeline_batch_job(db, run_id, batch_type)

    if not job_record:
        raise HTTPException(status_code=404,
                            detail=f"No batch job found for run_id '{run_id}' and batch_type '{batch_type}'.")

    if not job_record.llm_batch_id:
        if job_record.status == crud_batch_jobs.STATUS_COMPLETED:
            return {
                "job_id": job_record.id,
                "pipeline_run_id": run_id,
                "batch_type": batch_type,
                "internal_status": job_record.status,
                "llm_status": "not_applicable",
                "message": "Job was marked as completed internally without LLM submission (e.g., no data)."
            }
        raise HTTPException(status_code=404,
                            detail=f"LLM batch ID not found for job_id '{job_record.id}'. Batch may not have been submitted successfully.")

    if job_record.status in [crud_batch_jobs.STATUS_PENDING_PROCESSING, crud_batch_jobs.STATUS_PROCESSING,
                             crud_batch_jobs.STATUS_COMPLETED]:
        logging.debug(
            f"Internal status for job {job_record.id} is {job_record.status}. Returning 'completed' as llm_status for sensor.")
        return {
            "job_id": job_record.id,
            "pipeline_run_id": run_id,
            "batch_type": batch_type,
            "internal_status": job_record.status,
            "llm_status": "completed",
            "output_file_id": None,
            "error_file_id": None, 
            "message": f"Internal job status is {job_record.status}, implying LLM completion."
        }

    if job_record.status == crud_batch_jobs.STATUS_SUBMITTED:
        try:
            llm_status, output_file_id, error_file_id = batch_utils.check_batch_status(job_record.llm_batch_id)

            current_internal_status = job_record.status
            new_internal_status = None

            if llm_status == "completed":
                new_internal_status = crud_batch_jobs.STATUS_PENDING_PROCESSING
            elif llm_status in ["failed", "cancelled", "expired"]:
                new_internal_status = crud_batch_jobs.STATUS_SUBMISSION_FAILED
                error_message_from_llm = f"LLM job {job_record.llm_batch_id} ended with status: {llm_status}."
                if error_file_id:
                    try:
                        llm_error_content_bytes = get_llm_strategy().read_file(error_file_id)
                        llm_error_content = llm_error_content_bytes.decode('utf-8', errors='replace')[:500]
                        error_message_from_llm += f" Error details: {llm_error_content}"
                    except Exception as read_err:
                        logging.warning(f"Failed to read LLM error file {error_file_id}: {read_err}")

                crud_batch_jobs.update_pipeline_batch_job(db, job_id=job_record.id, status=new_internal_status,
                                                          last_error=error_message_from_llm)

            if new_internal_status and new_internal_status != current_internal_status:
                crud_batch_jobs.update_pipeline_batch_job(db, job_id=job_record.id, status=new_internal_status)
                db.commit()
                logging.info(
                    f"LLM Batch {job_record.llm_batch_id} status: {llm_status}. Internal job {job_record.id} status updated to {new_internal_status}.")

            return {
                "job_id": job_record.id,
                "pipeline_run_id": run_id,
                "batch_type": batch_type,
                "internal_status": job_record.status,
                "llm_status": llm_status,
                "output_file_id": output_file_id,
                "error_file_id": error_file_id
            }
        except Exception as e:
            logging.error(
                f"Error checking LLM batch status for job_id {job_record.id} (LLM ID: {job_record.llm_batch_id}): {e}",
                exc_info=True)

            raise HTTPException(status_code=503, detail=f"Failed to get LLM batch status: {e}")
    else:
        # Status como PENDING_SUBMISSION, SUBMISSION_FAILED, PROCESSING_FAILED
        logging.warning(
            f"LLM batch status check requested for job_id {job_record.id} with internal status {job_record.status}, which is not SUBMITTED.")
        return {
            "job_id": job_record.id,
            "pipeline_run_id": run_id,
            "batch_type": batch_type,
            "internal_status": job_record.status,
            "llm_status": "not_queried_due_to_internal_status",
            "message": f"LLM status not queried because internal job status is '{job_record.status}'."
        }


@app.post("/pipeline/{run_id}/process-batch-results/{batch_type}", tags=["pipeline_batch_jobs"])
def process_llm_batch_results(
        run_id: str,
        batch_type: str = FastAPIPath(...,
                                      description=f"Type of batch results to process. Valid types: {list(VALID_BATCH_TYPES)}"),
        db: Session = Depends(get_db)
):
    """
    Processes the results of a completed LLM batch job.
    Downloads results from LLM provider, parses them, and saves to application database.
    Manages job state for idempotency.
    """
    if batch_type not in VALID_BATCH_TYPES:
        raise HTTPException(status_code=400, detail=f"Invalid batch_type: {batch_type}")

    job_record = crud_batch_jobs.get_pipeline_batch_job(db, run_id, batch_type)

    if not job_record:
        raise HTTPException(status_code=404,
                            detail=f"No batch job found for run_id '{run_id}' and batch_type '{batch_type}'.")

    if job_record.status == crud_batch_jobs.STATUS_COMPLETED:
        logging.info(
            f"Batch results for job_id {job_record.id} (type: {batch_type}, run: {run_id}) already processed. Skipping.")
        return {
            "status": "skipped_already_processed",
            "message": "Batch results have already been processed.",
            "job_id": job_record.id,
            "pipeline_run_id": run_id,
            "batch_type": batch_type
        }

    if job_record.status != crud_batch_jobs.STATUS_PENDING_PROCESSING:
        error_detail_msg = f"Cannot process results for job_id {job_record.id}. Current status is '{job_record.status}'. Expected '{crud_batch_jobs.STATUS_PENDING_PROCESSING}'."
        if job_record.status == crud_batch_jobs.STATUS_PROCESSING:
            error_detail_msg = f"Job_id {job_record.id} is already being processed. Concurrent processing attempt rejected."

        logging.warning(error_detail_msg)
        raise HTTPException(status_code=409, detail=error_detail_msg)  # 409 Conflict

    if not job_record.llm_batch_id:
        crud_batch_jobs.update_pipeline_batch_job(db, job_id=job_record.id,
                                                  status=crud_batch_jobs.STATUS_PROCESSING_FAILED,
                                                  last_error="LLM Batch ID missing at PENDING_PROCESSING stage.")
        db.commit()
        raise HTTPException(status_code=500,
                            detail=f"Internal error: LLM Batch ID missing for job_id {job_record.id} at PENDING_PROCESSING stage.")

    crud_batch_jobs.update_pipeline_batch_job(db, job_id=job_record.id, status=crud_batch_jobs.STATUS_PROCESSING)
    db.commit()

    processing_successful = False
    try:
        if batch_type == BATCH_TYPE_UC_GENERATION:
            logging.info(
                f"Processing UC generation results for job_id {job_record.id} (LLM ID: {job_record.llm_batch_id})")

            processing_successful = task_process_uc_generation_batch(run_id, job_record.llm_batch_id,
                                                                        db_session_from_api=db)

        elif batch_type == BATCH_TYPE_DIFFICULTY_ASSESSMENT:
            logging.info(
                f"Processing difficulty assessment results for job_id {job_record.id} (LLM ID: {job_record.llm_batch_id})")
            processing_successful = task_process_difficulty_batch(run_id, job_record.llm_batch_id,
                                                                     db_session_from_api=db)

        if processing_successful:
            crud_batch_jobs.update_pipeline_batch_job(db, job_id=job_record.id, status=crud_batch_jobs.STATUS_COMPLETED)
            db.commit()
            logging.info(f"Successfully processed and saved results for job_id {job_record.id}.")
            return {
                "status": "success_processed",
                "message": "Batch results successfully processed and saved.",
                "job_id": job_record.id
            }
        else:
            crud_batch_jobs.update_pipeline_batch_job(db, job_id=job_record.id,
                                                      status=crud_batch_jobs.STATUS_PROCESSING_FAILED,
                                                      last_error="Processing logic returned failure.")
            db.commit()
            logging.error(
                f"Processing logic returned failure for job_id {job_record.id}. Status set to PROCESSING_FAILED.")
            raise HTTPException(status_code=500, detail=f"Failed to process batch results for job_id {job_record.id}.")

    except Exception as e:
        db.rollback()
        logging.error(f"Error processing batch results for job_id {job_record.id}: {e}", exc_info=True)
        
        try:
            with get_session() as error_db:
                job_to_update_on_error = crud_batch_jobs.get_pipeline_batch_job(error_db, run_id, batch_type)
                if job_to_update_on_error and job_to_update_on_error.status == crud_batch_jobs.STATUS_PROCESSING:
                    crud_batch_jobs.update_pipeline_batch_job(
                        error_db,
                        job_id=job_to_update_on_error.id,
                        status=crud_batch_jobs.STATUS_PROCESSING_FAILED,
                        last_error=str(e)[:1023]
                    )
                    error_db.commit()
        except Exception as db_err:
            logging.error(
                f"Additionally failed to update job status to PROCESSING_FAILED for job_id {job_record.id if job_record else 'unknown'}: {db_err}",
                exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing batch results: {e}")

@app.post("/pipeline/{run_id}/finalize-outputs", tags=["pipeline"])
def finalize_outputs(run_id: str):
    """Endpoint para executar task_finalize_outputs em contexto de run_id."""
    try:
        task_finalize_outputs(run_id)

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