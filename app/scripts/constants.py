import pandas as pd
from pathlib import Path
from typing import Optional
import os

# Parâmetros configuráveis (poderiam vir de variáveis de ambiente ou config do Airflow)
MAX_ORIGINS_FOR_TESTING: Optional[int] = os.environ.get('MAX_ORIGINS_FOR_TESTING', None)
if MAX_ORIGINS_FOR_TESTING: MAX_ORIGINS_FOR_TESTING = int(MAX_ORIGINS_FOR_TESTING)

BLOOM_ORDER = ["Lembrar", "Entender", "Aplicar", "Analisar", "Avaliar", "Criar"]
BLOOM_ORDER_MAP = {level: i for i, level in enumerate(BLOOM_ORDER)}
import os
# Localiza prompts no diretório do pacote 'scripts' ou via env var
DEFAULT_PROMPT_DIR = Path(__file__).parent
PROMPT_UC_GENERATION_FILE = Path(
    os.environ.get(
        'PROMPT_UC_GENERATION_FILE',
        DEFAULT_PROMPT_DIR / 'prompt_uc_generation.txt'
    )
)
PROMPT_UC_DIFFICULTY_FILE = Path(
    os.environ.get(
        'PROMPT_UC_DIFFICULTY_FILE',
        DEFAULT_PROMPT_DIR / 'prompt_uc_difficulty.txt'
    )
)
LLM_MODEL = os.environ.get('LLM_MODEL', "gpt-4o-mini")
LLM_TEMPERATURE_GENERATION = float(os.environ.get('LLM_TEMPERATURE_GENERATION', 0.2))
LLM_TEMPERATURE_DIFFICULTY = float(os.environ.get('LLM_TEMPERATURE_DIFFICULTY', 0.1))
DIFFICULTY_BATCH_SIZE = int(os.environ.get('DIFFICULTY_BATCH_SIZE', 5))
MIN_EVALUATIONS_PER_UC = int(os.environ.get('MIN_EVALUATIONS_PER_UC', 3))
# BATCH API não usa concorrência configurável do nosso lado
# UC_GENERATION_BATCH_SIZE = 20 # Não relevante para Batch API (tamanho do arquivo é o limite)

"""
Diretórios de dados via variável de ambiente ou padrão antigo.
"""
# Permite sobrescrever localização dos dados pela variável de ambiente
AIRFLOW_DATA_DIR = Path(
    os.environ.get('AIRFLOW_DATA_DIR', '/opt/airflow/data')
)
# O Graphrag escreve por padrão em 'output' (conforme settings.yaml)
BASE_INPUT_DIR = AIRFLOW_DATA_DIR / 'output'
PIPELINE_WORK_DIR = AIRFLOW_DATA_DIR / 'pipeline_workdir'
BATCH_FILES_DIR = PIPELINE_WORK_DIR / 'batch_files'
stage1_dir = PIPELINE_WORK_DIR / "1_origins"
stage2_output_ucs_dir = PIPELINE_WORK_DIR / "2_generated_ucs"
stage3_dir = PIPELINE_WORK_DIR / "3_relationships"
stage4_input_batch_dir = BATCH_FILES_DIR # Reutiliza para dificuldade
stage4_output_eval_dir = PIPELINE_WORK_DIR / "4_difficulty_evals"
stage5_dir = PIPELINE_WORK_DIR / "5_final_outputs"

# Constantes para evitar magic strings e configuração centralizada
GENERATED_UCS_RAW = "generated_ucs_raw"
UC_EVALUATIONS_RAW = "uc_evaluations_aggregated_raw"
REL_TYPE_REQUIRES = "REQUIRES"
REL_TYPE_EXPANDS = "EXPANDS"

# Nomes de arquivos intermediários e finais
REL_INTERMEDIATE = "knowledge_relationships_intermediate"
FINAL_UC_FILE = "final_knowledge_units"
FINAL_REL_FILE = "final_knowledge_relationships"