import os
import logging
import json
from typing import List, Dict, Any, Tuple
from collections import defaultdict, Counter

# Número mínimo de avaliações por UC
MIN_EVALUATIONS_PER_UC = int(os.environ.get('MIN_EVALUATIONS_PER_UC', 1))

def _format_difficulty_prompt(
    batch_ucs_data: List[Dict[str, Any]],
    prompt_template: str
) -> str:
    """Formata o prompt de avaliação de dificuldade para um batch."""
    prompt_input_text = ""
    for uc_data in batch_ucs_data:
        uc_id = uc_data.get('uc_id', 'N/A')
        uc_text = uc_data.get('uc_text', 'N/A')
        prompt_input_text += f"- ID: {uc_id}\n  Texto: {uc_text}\n"
    return prompt_template.replace("{{BATCH_OF_UCS}}", prompt_input_text.strip())

def _calculate_final_difficulty_from_raw(
    generated_ucs: List[Dict[str, Any]],
    raw_evaluations: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], int, int]:
    """Calcula o score final a partir das avaliações brutas do batch."""
    logging.info("Calculando scores finais de dificuldade a partir dos resultados do batch...")
    uc_scores: Dict[str, List[int]] = defaultdict(list)
    uc_justifications: Dict[str, List[str]] = defaultdict(list)
    uc_evaluation_count: Counter = Counter()
    for evaluation in raw_evaluations:
        uc_id = evaluation.get("uc_id")
        score = evaluation.get("difficulty_score")
        justification = evaluation.get("justification")
        if uc_id and isinstance(score, int) and 0 <= score <= 100:
            uc_scores[uc_id].append(score)
        if justification:
            uc_justifications[uc_id].append(justification)
            uc_evaluation_count[uc_id] += 1
    updated_ucs_list: List[Dict[str, Any]] = []
    evaluated_count = 0
    min_evals_met_count = 0
    for original_uc in generated_ucs:
        uc = original_uc.copy()
        uc_id = uc.get("uc_id")
        scores = uc_scores.get(uc_id)
        eval_count = uc_evaluation_count.get(uc_id, 0)
        if scores:
            final_score = round(sum(scores) / len(scores))
            justification_text = " | ".join(uc_justifications.get(uc_id, ["N/A"]))
            uc["difficulty_score"] = final_score
            uc["difficulty_justification"] = justification_text
            uc["evaluation_count"] = eval_count
            evaluated_count += 1
        if eval_count >= MIN_EVALUATIONS_PER_UC:
            min_evals_met_count += 1
        else:
            uc["difficulty_score"] = None
            uc["difficulty_justification"] = "Não avaliado"
            uc["evaluation_count"] = 0
        updated_ucs_list.append(uc)
    logging.info(f"  {evaluated_count}/{len(generated_ucs)} UCs receberam score.")
    logging.info(f"  {min_evals_met_count}/{len(generated_ucs)} UCs atingiram {MIN_EVALUATIONS_PER_UC} avaliações.")
    return updated_ucs_list, evaluated_count, min_evals_met_count