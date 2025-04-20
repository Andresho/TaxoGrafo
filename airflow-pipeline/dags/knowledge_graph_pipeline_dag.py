# dags/knowledge_graph_pipeline_dag.py
from __future__ import annotations

import pendulum

from airflow.models.dag import DAG
from airflow.operators.python import PythonOperator
# Para sensores reais da OpenAI Batch API (se disponíveis ou customizados)
# from airflow.providers.openai.sensors.openai_batch import OpenAIBatchSensor

# Importa as funções de tarefa do nosso script
# O Airflow adiciona o diretório 'dags' e 'scripts' ao PYTHONPATH se estiverem na estrutura correta
# ou podemos ajustar o sys.path se necessário
try:
    from scripts.pipeline_tasks import (
        task_prepare_origins,
        task_submit_uc_generation_batch,
        task_wait_and_process_batch,
        task_define_relationships,
        task_submit_difficulty_batch,
        task_finalize_outputs,
        stage2_output_ucs_dir,
        stage4_output_eval_dir,
    )
except ImportError:
    # Abordagem alternativa se a importação direta falhar
    import sys
    from pathlib import Path
    # Adiciona diretório pai de 'dags' ao path (onde 'scripts' deve estar)
    sys.path.append(str(Path(__file__).parent.parent))
    from scripts.pipeline_tasks import (
        task_prepare_origins,
        task_submit_uc_generation_batch,
        task_wait_and_process_batch,
        task_define_relationships,
        task_submit_difficulty_batch,
        task_finalize_outputs,
        stage2_output_ucs_dir,
        stage4_output_eval_dir,
    )


with DAG(
    dag_id="knowledge_graph_pipeline",
    schedule=None, # Para execução manual
    start_date=pendulum.datetime(2024, 5, 24, tz="UTC"),
    catchup=False,
    tags=["knowledge_graph", "llm", "batch_api"],
    doc_md="""
    ### Knowledge Graph Pipeline DAG
    Orquestra a geração de um grafo de conhecimento educacional a partir de outputs do GraphRAG,
    utilizando a OpenAI Batch API para geração de UCs e avaliação de dificuldade.
    """,
) as dag:

    prepare_origins = PythonOperator(
        task_id="prepare_origins",
        python_callable=task_prepare_origins,
        # provide_context=True # padrão no Airflow 2+
    )

    submit_generation = PythonOperator(
        task_id="submit_generation_batch",
        python_callable=task_submit_uc_generation_batch,
    )

    # Usando nossa função combinada de wait/process
    # Passamos um ID chave para diferenciar os batches
    wait_process_generation = PythonOperator(
        task_id="wait_process_generation_results",
        python_callable=task_wait_and_process_batch,
        op_kwargs={
            'batch_id_key': 'generation',  # Key para buscar o ID do batch submetido via XCom
            'output_dir': stage2_output_ucs_dir,
            'output_filename': 'generated_ucs_raw'
        },
        trigger_rule='all_success', # Só roda se a submissão foi ok (e retornou ID)
    )

    define_rels = PythonOperator(
        task_id="define_relationships",
        python_callable=task_define_relationships,
    )

    submit_difficulty = PythonOperator(
        task_id="submit_difficulty_batch",
        python_callable=task_submit_difficulty_batch,
    )

    wait_process_difficulty = PythonOperator(
        task_id="wait_process_difficulty_results",
        python_callable=task_wait_and_process_batch,  # Reutiliza a função de wait
        op_kwargs={
            'batch_id_key': 'difficulty',  # Key para buscar o ID do batch de dificuldade via XCom
            'output_dir': stage4_output_eval_dir,
            # Salva avaliações brutas aqui para cálculo final na próxima etapa
            'output_filename': 'uc_evaluations_aggregated_raw'
        },
        trigger_rule='all_success',
    )

    finalize = PythonOperator(
        task_id="finalize_outputs",
        python_callable=task_finalize_outputs,
    )

    # Definindo as dependências
    prepare_origins >> submit_generation >> wait_process_generation
    wait_process_generation >> define_rels
    define_rels >> submit_difficulty >> wait_process_difficulty
    wait_process_difficulty >> finalize