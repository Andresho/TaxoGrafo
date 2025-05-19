from typing import Optional
import logging

from sqlalchemy.orm import Session
from app.db import get_session

from app.scripts.batch_utils import check_batch_status, process_batch_results
from app.scripts.constants import GENERATED_UCS_RAW, get_dirs

def task_process_uc_generation_batch(run_id: str, llm_batch_id: str,
                                     db_session_from_api: Optional[Session] = None) -> bool:
    """
    Processa resultados de geração UC para um run_id e llm_batch_id.
    Usa db_session_from_api se fornecida, senão cria uma nova.
    NÃO faz commit/rollback se db_session_from_api for usada.
    Retorna True para sucesso, False para falha.
    """
    logging.info(f"--- LOGIC: process_uc_generation_batch (run_id={run_id}, llm_batch_id={llm_batch_id}) ---")

    def _core_logic(db: Session) -> bool:
        try:
            llm_status, output_file_id, error_file_id = check_batch_status(llm_batch_id)

            if llm_status != 'completed':
                logging.error(
                    f"LLM Batch {llm_batch_id} não está 'completed' (status atual: {llm_status}) ao tentar processar resultados.")
                return False
            if not output_file_id:
                logging.error(f"LLM Batch {llm_batch_id} está 'completed' mas não possui output_file_id.")
                return False

            logging.info(f"LLM Batch {llm_batch_id} confirmado como 'completed'. output_file_id: {output_file_id}")

            _, _, _, s1_dir, s2_dir_for_run, s3_dir, s4_dir, s5_dir = get_dirs(run_id)

            processing_ok = process_batch_results(
                batch_id=llm_batch_id,
                output_file_id=output_file_id,
                error_file_id=error_file_id,
                stage_output_dir=s2_dir_for_run,
                output_filename_key=GENERATED_UCS_RAW,
                run_id=run_id,
                db=db
            )

            if processing_ok:
                logging.info(f"Processamento dos resultados do LLM Batch {llm_batch_id} (geração UC) bem-sucedido.")
                return True
            else:
                logging.error(
                    f"Falha no processamento interno dos resultados do LLM Batch {llm_batch_id} (geração UC).")
                return False
        except Exception as e:
            logging.exception(
                f"Erro crítico durante process_uc_generation_batch (run_id={run_id}, llm_batch_id={llm_batch_id})")
            return False

    if db_session_from_api:
        return _core_logic(db_session_from_api)
    else:
        with get_session() as db:
            success = _core_logic(db)
            if success:
                db.commit()
            else:
                db.rollback()
            return success