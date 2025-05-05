# dags/knowledge_graph_pipeline_dag.py
from __future__ import annotations

import pendulum

import os
from airflow.models.dag import DAG
# Importa operadores usados
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

from airflow.providers.http.operators.http import SimpleHttpOperator
from airflow.providers.http.sensors.http import HttpSensor
from airflow.operators.python import BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.trigger_rule import TriggerRule

def _skip_or_run_graphrag(**kwargs):
    """Retorna a tarefa a executar: 'skip_graphrag' ou 'graphrag_index'."""
    dag_run = kwargs.get('dag_run')
    if dag_run and dag_run.conf.get('skip_graphrag', False):
        return "skip_graphrag"
    return "graphrag_index"

with DAG(
    dag_id="knowledge_graph_pipeline",
    schedule=None, # Para execução manual
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

    # Decide se deve executar a etapa Graphrag index ou pular
    branch_graphrag = BranchPythonOperator(
        task_id="branch_graphrag",
        python_callable=_skip_or_run_graphrag,
    )

    skip_graphrag = EmptyOperator(
        task_id="skip_graphrag",
    )
    # Task 1: executa Graphrag index em container Docker isolado
    graphrag_index = DockerOperator(
        task_id="graphrag_index",
        image="graphrag:latest",
        api_version="auto",
        auto_remove=True,
        entrypoint="/bin/bash",
        command=(
            '-c "graphrag init --root /data/{{ dag_run.conf.get(\'pipeline_id\', dag_run.run_id) }} --force && '
            'graphrag index --root /data/{{ dag_run.conf.get(\'pipeline_id\', dag_run.run_id) }}"'
        ),
        #   ' && true && sleep infinity"'
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge",
        # Mount named volume 'pipeline_data' into container at /data
        mounts=[Mount(source='pipeline_data', target='/data', type='volume')],
        environment={
            "GRAPHRAG_MODEL": os.getenv("GRAPHRAG_MODEL"),
            "GRAPHRAG_API_KEY": os.getenv("GRAPHRAG_API_KEY"),
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        },
    )

    # 2: Prepara origens via API
    prepare_origins = SimpleHttpOperator(
        task_id="prepare_origins",
        http_conn_id="pipeline_api",
        method="POST",
        # roda para este run_id dinamicamente
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