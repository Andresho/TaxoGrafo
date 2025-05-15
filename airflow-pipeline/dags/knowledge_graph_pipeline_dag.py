# dags/knowledge_graph_pipeline_dag.py
from __future__ import annotations

import pendulum

import os
from airflow.models.dag import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

from airflow.providers.http.operators.http import SimpleHttpOperator
from airflow.providers.http.sensors.http import HttpSensor
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.trigger_rule import TriggerRule

import logging
from pathlib import Path
import shutil
import requests
import pypdfium2 as pdfium

import ast

AIRFLOW_DATA_DIR_VOL = Path(os.getenv("AIRFLOW_DATA_DIR", "/opt/airflow/data"))
PIPELINE_API_CONN_ID = "pipeline_api"
PIPELINE_API_BASE_URL = os.getenv("AIRFLOW_CONN_PIPELINE_API", "http://pipeline-api:8000")


def _prepare_input_files_callable(run_id: str, resource_ids_for_run_input: str, **kwargs):
    logging.info(f"Iniciando _prepare_input_files_callable para run_id={run_id}")

    try:
        if isinstance(resource_ids_for_run_input, str):
            actual_resource_ids_list = ast.literal_eval(resource_ids_for_run_input)
        elif isinstance(resource_ids_for_run_input, list):
            actual_resource_ids_list = resource_ids_for_run_input
        else:
            raise ValueError(f"Tipo inesperado para resource_ids_for_run_input: {type(resource_ids_for_run_input)}")

        if not isinstance(actual_resource_ids_list, list):
            raise ValueError(
                f"resource_ids_for_run_input não pôde ser convertido para uma lista. Valor: {resource_ids_for_run_input}")

    except (ValueError, SyntaxError) as e:
        logging.error(f"Erro ao converter resource_ids_for_run_input ('{resource_ids_for_run_input}') para lista: {e}")
        actual_resource_ids_list = []

    if not actual_resource_ids_list:
        logging.warning("Nenhum resource_id para processar após conversão/validação.")
        return

    logging.info(f"Lista de resource_ids a processar: {actual_resource_ids_list}")

    run_specific_input_dir = AIRFLOW_DATA_DIR_VOL / run_id / "input"
    run_specific_input_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Diretório de input da run: {run_specific_input_dir}")

    base_originals_dir = AIRFLOW_DATA_DIR_VOL / "uploads" / "originals"
    base_processed_txt_dir = AIRFLOW_DATA_DIR_VOL / "uploads" / "processed_txt"
    base_processed_txt_dir.mkdir(parents=True, exist_ok=True)

    for resource_id_str in actual_resource_ids_list:
        logging.info(f"Processando resource_id: {resource_id_str}")
        if not resource_id_str or not isinstance(resource_id_str, str):
            logging.warning(f"Item inválido na lista de resource_ids: '{resource_id_str}'. Pulando.")
            continue
        try:
            # 1. Obter detalhes do recurso da API
            res_details_url = f"{PIPELINE_API_BASE_URL}/api/v1/resources/{resource_id_str}"
            response = requests.get(res_details_url, timeout=10)
            response.raise_for_status()
            resource_data = response.json()
            logging.info(f"Detalhes do recurso {resource_id_str}: {resource_data}")

            original_file_path_relative = Path(resource_data["original_file_path"])
            # Monta caminho absoluto DENTRO do worker Airflow, usando o volume montado
            original_file_path_absolute = AIRFLOW_DATA_DIR_VOL / original_file_path_relative

            processed_txt_path_relative_str = resource_data.get("processed_txt_path")
            current_status = resource_data.get("status")

            final_txt_to_copy_to_run_input = None

            if processed_txt_path_relative_str and current_status == "processed_txt_success":
                logging.info(f"Recurso {resource_id_str} já processado para TXT. Usando cache.")
                final_txt_to_copy_to_run_input = AIRFLOW_DATA_DIR_VOL / Path(processed_txt_path_relative_str)
            else:
                logging.info(f"Recurso {resource_id_str} precisa ser convertido para TXT.")
                update_payload = {"status": "processing_txt"}
                requests.put(f"{PIPELINE_API_BASE_URL}/api/v1/resources/{resource_id_str}", json=update_payload, timeout=5).raise_for_status()

                # Caminho para o TXT processado (relativo e absoluto)
                # Salva como <uuid_recurso>.txt para evitar conflitos e facilitar lookup
                processed_txt_filename = f"{resource_id_str}.txt"
                target_processed_txt_path_relative = Path("uploads") / "processed_txt" / resource_id_str / processed_txt_filename
                target_processed_txt_path_absolute = base_processed_txt_dir / resource_id_str / processed_txt_filename
                (base_processed_txt_dir / resource_id_str).mkdir(parents=True, exist_ok=True)


                if not original_file_path_absolute.exists():
                    raise FileNotFoundError(f"Arquivo original não encontrado em: {original_file_path_absolute}")

                if resource_data["original_mime_type"] == "text/plain" or original_file_path_absolute.suffix.lower() == ".txt":
                    logging.info(f"Copiando arquivo TXT original: {original_file_path_absolute} para {target_processed_txt_path_absolute}")
                    shutil.copy(original_file_path_absolute, target_processed_txt_path_absolute)
                elif resource_data["original_mime_type"] == "application/pdf" or original_file_path_absolute.suffix.lower() == ".pdf":
                    logging.info(f"Convertendo PDF: {original_file_path_absolute} para {target_processed_txt_path_absolute}")
                    pdf_doc = pdfium.PdfDocument(original_file_path_absolute)
                    text_content = ""
                    for i in range(len(pdf_doc)):
                        page = pdf_doc[i]
                        textpage = page.get_textpage()
                        text_content += textpage.get_text_range() + "\n"
                        textpage.close()
                        page.close()
                    pdf_doc.close()
                    with open(target_processed_txt_path_absolute, "w", encoding="utf-8") as f_out:
                        f_out.write(text_content)
                else:
                    raise ValueError(f"Tipo de arquivo não suportado para conversão: {resource_data['original_mime_type']}")

                # Atualizar status para 'processed_txt_success' via API
                update_payload_success = {
                    "status": "processed_txt_success",
                    "processed_txt_path": str(target_processed_txt_path_relative)
                }
                requests.put(f"{PIPELINE_API_BASE_URL}/api/v1/resources/{resource_id_str}", json=update_payload_success, timeout=5).raise_for_status()
                logging.info(f"Recurso {resource_id_str} convertido e status atualizado.")
                final_txt_to_copy_to_run_input = target_processed_txt_path_absolute

            # Copiar o TXT (original ou processado) para a pasta de input da run
            if final_txt_to_copy_to_run_input and final_txt_to_copy_to_run_input.exists():
                # Usar o nome original do arquivo (sanitizado) + .txt para a pasta de input da run
                sanitized_original_filename = "".join(c if c.isalnum() or c in ('.', '_') else '_' for c in Path(resource_data["original_filename"]).stem)
                destination_in_run_input = run_specific_input_dir / f"{sanitized_original_filename}.txt"
                shutil.copy(final_txt_to_copy_to_run_input, destination_in_run_input)
                logging.info(f"Copiado {final_txt_to_copy_to_run_input} para {destination_in_run_input}")
            else:
                raise FileNotFoundError(f"Arquivo TXT final não encontrado para cópia: {final_txt_to_copy_to_run_input}")

        except Exception as e:
            logging.error(f"Erro ao processar resource_id {resource_id_str}: {e}", exc_info=True)
            try:
                update_payload_error = {"status": "processed_txt_error", "error_message": str(e)[:1000]}
                requests.put(f"{PIPELINE_API_BASE_URL}/api/v1/resources/{resource_id_str}", json=update_payload_error,
                             timeout=5)
            except Exception as api_err:
                logging.error(f"Falha ao atualizar status de erro para {resource_id_str} via API: {api_err}")
            raise

def _skip_or_run_graphrag(**kwargs):
    """Retorna a tarefa a executar: 'skip_graphrag' ou 'graphrag_index'."""
    dag_run = kwargs.get('dag_run')
    if dag_run and dag_run.conf.get('skip_graphrag', False):
        return "skip_graphrag"
    return "graphrag_index"

with DAG(
    dag_id="knowledge_graph_pipeline",
    schedule=None,
    start_date=pendulum.datetime(2024, 5, 24, tz="UTC"),
    catchup=False,
    tags=["knowledge_graph", "llm", "batch_api"],
    params={'identifier': "{{ dag_run.conf.get('pipeline_id', dag_run.run_id) }}"},
    doc_md="""
    ### Knowledge Graph Pipeline DAG
    Orquestra a geração de um grafo de conhecimento educacional a partir de outputs do GraphRAG,
    utilizando a OpenAI Batch API para geração de UCs e avaliação de dificuldade.
    """,
) as dag:
    prepare_input_files = PythonOperator(
        task_id="prepare_input_files",
        python_callable=_prepare_input_files_callable,
        op_kwargs={
            "run_id": "{{ dag_run.conf.get('pipeline_id', dag_run.run_id) }}",
            "resource_ids_for_run_input": "{{ dag_run.conf.get('resource_ids_for_run', []) }}",
        },
    )

    branch_graphrag = BranchPythonOperator(
        task_id="branch_graphrag",
        python_callable=_skip_or_run_graphrag,
    )

    skip_graphrag = EmptyOperator(
        task_id="skip_graphrag",
    )

    graphrag_index = DockerOperator(
        task_id="graphrag_index",
        image="graphrag:latest",
        api_version="auto",
        auto_remove=True,
        entrypoint="/bin/bash",
        command='-c "graphrag index --root /graphrag_config --method standard --logger rich"',
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge",
        mounts=[Mount(source='pipeline_data', target='/data', type='volume')],
        environment={
            "GRAPHRAG_MODEL": os.getenv("GRAPHRAG_MODEL"),
            "GRAPHRAG_API_KEY": os.getenv("GRAPHRAG_API_KEY"),
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
            "GRAPHRAG_RUN_ID": "{{ dag_run.conf.get('pipeline_id', dag_run.run_id) }}",
        },
    )

    prepare_origins = SimpleHttpOperator(
        task_id="prepare_origins",
        http_conn_id="pipeline_api",
        method="POST",
        endpoint="/pipeline/{{ dag_run.conf.get('pipeline_id', dag_run.run_id) }}/prepare-origins",
        headers={"Content-Type": "application/json"},
        do_xcom_push=False,
        # allow this task to run if at least one upstream succeeded (branch or graphrag)
        trigger_rule=TriggerRule.ONE_SUCCESS,
    )

    # 3: Submete batch de geração UC e armazena batch_id em XCom
    submit_generation = SimpleHttpOperator(
        task_id="submit_generation_batch",
        http_conn_id="pipeline_api",
        method="POST",
        endpoint="/pipeline/{{ dag_run.conf.get('pipeline_id', dag_run.run_id) }}/submit-uc-generation-batch",
        headers={"Content-Type": "application/json"},
        response_filter=lambda response: response.json().get("batch_id"),
        do_xcom_push=True,
    )

    # 4: Aguarda término da batch de geração UC
    wait_generation = HttpSensor(
        task_id="wait_generation_batch",
        http_conn_id="pipeline_api",
        method="GET",
        # usa run_id e batch_id
        endpoint="/pipeline/{{ dag_run.conf.get('pipeline_id', dag_run.run_id) }}/batch-status/{{ ti.xcom_pull(task_ids='submit_generation_batch') }}",
        request_params={},
        response_check=lambda response: response.json().get("status") == "completed",
        poke_interval=60,
        timeout=3600,
        mode="reschedule",
    )

    # 5: Processa resultados da geração UC
    process_generation = SimpleHttpOperator(
        task_id="process_generation_batch",
        http_conn_id="pipeline_api",
        method="POST",
        endpoint="/pipeline/{{ dag_run.conf.get('pipeline_id', dag_run.run_id) }}/process-uc-generation-batch/{{ ti.xcom_pull(task_ids='submit_generation_batch') }}",
        headers={"Content-Type": "application/json"},
        do_xcom_push=False,
    )

    # 6: Define relacionamentos via API
    define_rels = SimpleHttpOperator(
        task_id="define_relationships",
        http_conn_id="pipeline_api",
        method="POST",
        endpoint="/pipeline/{{ dag_run.conf.get('pipeline_id', dag_run.run_id) }}/define-relationships",
        headers={"Content-Type": "application/json"},
        do_xcom_push=False,
    )

    # 7: Submete batch de avaliação de dificuldade e armazena batch_id em XCom
    submit_difficulty = SimpleHttpOperator(
        task_id="submit_difficulty_batch",
        http_conn_id="pipeline_api",
        method="POST",
        endpoint="/pipeline/{{ dag_run.conf.get('pipeline_id', dag_run.run_id) }}/submit-difficulty-batch",
        headers={"Content-Type": "application/json"},
        response_filter=lambda response: response.json().get("batch_id"),
        do_xcom_push=True,
    )

    # 8: Aguarda término da batch de avaliação de dificuldade
    wait_difficulty = HttpSensor(
        task_id="wait_difficulty_batch",
        http_conn_id="pipeline_api",
        method="GET",
        endpoint="/pipeline/{{ dag_run.conf.get('pipeline_id', dag_run.run_id) }}/batch-status/{{ ti.xcom_pull(task_ids='submit_difficulty_batch') }}",
        request_params={},
        response_check=lambda response: response.json().get("status") == "completed",
        poke_interval=60,
        timeout=3600,
        mode="reschedule",
    )

    # 9: Processa resultados de dificuldade
    process_difficulty = SimpleHttpOperator(
        task_id="process_difficulty_batch",
        http_conn_id="pipeline_api",
        method="POST",
        endpoint="/pipeline/{{ dag_run.conf.get('pipeline_id', dag_run.run_id) }}/process-difficulty-batch/{{ ti.xcom_pull(task_ids='submit_difficulty_batch') }}",
        headers={"Content-Type": "application/json"},
        do_xcom_push=False,
    )

    # 10: Finaliza outputs via API
    finalize = SimpleHttpOperator(
        task_id="finalize_outputs",
        http_conn_id="pipeline_api",
        method="POST",
        endpoint="/pipeline/{{ dag_run.conf.get('pipeline_id', dag_run.run_id) }}/finalize-outputs",
        headers={"Content-Type": "application/json"},
        do_xcom_push=False,
    )

    # Definindo as dependências com idempotência do Graphrag
    branch_graphrag >> [graphrag_index, skip_graphrag]
    graphrag_index >> prepare_origins
    skip_graphrag >> prepare_origins
    prepare_origins >> submit_generation >> wait_generation >> process_generation
    process_generation >> define_rels
    define_rels >> submit_difficulty >> wait_difficulty >> process_difficulty
    process_difficulty >> finalize