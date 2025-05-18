from typing import List, Dict, Any
import logging

from app.db import get_session

import app.crud.generated_ucs_raw as crud_generated_ucs_raw
import app.crud.knowledge_unit_evaluations_batch as crud_knowledge_unit_evaluations_batch
import app.crud.knowledge_relationships_intermediate as crud_rel_intermediate
import app.crud.final_knowledge_units as crud_final_ucs
import app.crud.final_knowledge_relationships as crud_final_rels
import app.crud.pipeline_run as crud_runs

from app.scripts.difficulty_utils import _calculate_final_difficulty_from_raw

def task_finalize_outputs(run_id: str):
    """
    Combina UCs geradas com avaliações de dificuldade, calcula scores finais,
    salva UCs e Relações finais no banco, e atualiza o status do PipelineRun.
    A API que chama esta função gerencia a idempotência de alto nível e a transação principal.
    """
    logging.info(f"--- LOGIC: finalize_outputs (run_id={run_id}) ---")

    with get_session() as db:
        try:
            existing_final_ucs = crud_final_ucs.get_final_knowledge_units(db, run_id)
            existing_final_rels = crud_final_rels.get_final_knowledge_relationships(db, run_id)
            db_run = crud_runs.get_run(db, run_id)

            if not db_run:
                logging.error(f"PipelineRun {run_id} não encontrado no DB ao tentar finalizar outputs.")
                raise ValueError(f"PipelineRun {run_id} não existe.")

            if existing_final_ucs and existing_final_rels and db_run.status == 'success':
                logging.info(
                    f"Outputs finais já existem e PipelineRun status é 'success' para run_id={run_id}. Pulando finalização.")
                logging.info(f"--- LOGIC: finalize_outputs CONCLUÍDA (Idempotente) (run_id={run_id}) ---")
                return

            if existing_final_ucs and existing_final_rels and db_run.status != 'success':
                logging.warning(
                    f"Outputs finais existem para run_id={run_id}, mas run status é '{db_run.status}'. Tentando apenas atualizar status do run para 'success'.")
                crud_runs.update_run_status(db, run_id, status='success')
                logging.info(f"PipelineRun {run_id} status atualizado para 'success'.")
                logging.info(f"--- LOGIC: finalize_outputs CONCLUÍDA (Status do Run atualizado) (run_id={run_id}) ---")
                return

            logging.info("Carregando dados intermediários do banco para finalização...")
            generated_ucs_raw_list = crud_generated_ucs_raw.get_generated_ucs_raw(db, run_id)
            rels_intermed_list = crud_rel_intermediate.get_knowledge_relationships_intermediate(db, run_id)
            evals_raw_list = crud_knowledge_unit_evaluations_batch.get_knowledge_unit_evaluations_batch(db, run_id)

            if not generated_ucs_raw_list:
                raise ValueError(
                    f"Erro crítico: Nenhum registro de UCs geradas (generated_ucs_raw) encontrado no banco para run_id={run_id} ao finalizar.")

            if not rels_intermed_list:
                logging.warning(
                    f"Nenhum registro de relações intermediárias encontrado para run_id={run_id}. Tabela final_knowledge_relationships ficará vazia.")
                rels_intermed_list = []
            if not evals_raw_list:
                logging.warning(
                    f"Nenhuma avaliação de dificuldade bruta (batch) encontrada para run_id={run_id}. UCs finais não terão scores de dificuldade calculados nesta etapa.")
                evals_raw_list = []

            final_ucs_to_save: List[Dict[str, Any]] = []
            if evals_raw_list:
                logging.info("Calculando scores finais de dificuldade a partir das avaliações do batch...")
                final_ucs_to_save, evaluated_count, min_evals_met_count = _calculate_final_difficulty_from_raw(
                    generated_ucs_raw_list, evals_raw_list
                )
                logging.info(
                    f"Cálculo de dificuldade concluído: {evaluated_count} UCs com score, {min_evals_met_count} atingiram o mínimo de avaliações.")
            else:
                logging.info("Nenhuma avaliação de dificuldade encontrada. UCs finais não terão scores de dificuldade.")
                for uc_raw_dict in generated_ucs_raw_list:
                    uc_final_dict = uc_raw_dict.copy()
                    uc_final_dict["difficulty_score"] = None
                    uc_final_dict["difficulty_justification"] = "Não avaliado"
                    uc_final_dict["evaluation_count"] = 0
                    final_ucs_to_save.append(uc_final_dict)

            if not final_ucs_to_save and generated_ucs_raw_list:
                raise ValueError(
                    f"Erro inesperado: Lista final de UCs (final_ucs_to_save) vazia após processamento de dificuldade, mas UCs raw existiam para run_id={run_id}.")
            elif not final_ucs_to_save and not generated_ucs_raw_list:
                logging.warning(
                    f"Nenhuma UC gerada (raw) e, portanto, nenhuma UC final para salvar para run_id={run_id}.")

            if final_ucs_to_save:
                crud_final_ucs.add_final_knowledge_units(db, run_id, final_ucs_to_save)
                logging.info(f"{len(final_ucs_to_save)} UCs finais adicionadas à sessão do banco.")

            if rels_intermed_list:
                crud_final_rels.add_final_knowledge_relationships(db, run_id, rels_intermed_list)
                logging.info(
                    f"{len(rels_intermed_list)} relações finais (baseadas nas intermediárias) adicionadas à sessão do banco.")

            crud_runs.update_run_status(db, run_id, status='success')
            logging.info(f"Status do PipelineRun {run_id} definido para 'success' na sessão.")

            db.commit()
            logging.info("Commit atômico de outputs finais e status do Run bem-sucedido.")
            logging.info(f"--- LOGIC: finalize_outputs CONCLUÍDA com sucesso (run_id={run_id}) ---")

        except ValueError as ve:
            logging.error(f"Erro de valor durante finalize_outputs: {ve}", exc_info=True)
            db.rollback()
            logging.error(f"--- LOGIC: finalize_outputs FALHOU (Erro de Valor) (run_id={run_id}) ---")

            try:
                crud_runs.update_run_status(db, run_id, status='finalize_failed'); db.commit()
            except:
                logging.error(f"Falha ao tentar atualizar status do run {run_id} para 'finalize_failed'.")
            raise
        except Exception as e:
            logging.error(f"Erro geral durante finalize_outputs: {e}", exc_info=True)
            db.rollback()
            logging.error(f"--- LOGIC: finalize_outputs FALHOU (Erro Geral) (run_id={run_id}) ---")
            try:
                crud_runs.update_run_status(db, run_id, status='finalize_failed'); db.commit()
            except:
                logging.error(f"Falha ao tentar atualizar status do run {run_id} para 'finalize_failed'.")
            raise