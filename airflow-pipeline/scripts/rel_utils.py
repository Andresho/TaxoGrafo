import pandas as pd
import logging
from typing import List, Dict, Any
from collections import defaultdict

from scripts.constants import BLOOM_ORDER, BLOOM_ORDER_MAP

def _prepare_expands_lookups(
    entities_df: pd.DataFrame,
    generated_ucs: List[Dict[str, Any]]
) -> (Dict[str, str], Dict[str, Dict[str, List[str]]]):
    """Prepara os dicionários de lookup necessários para definir relações EXPANDS."""
    entity_name_to_id: Dict[str, str] = {}
    if entities_df is not None and 'title' in entities_df.columns and 'id' in entities_df.columns:
        entity_name_to_id = pd.Series(entities_df.id.values, index=entities_df.title).to_dict()
        logging.info(f"Criado mapa nome->ID ({len(entity_name_to_id)} entidades).")
    else:
        logging.warning("Não foi possível criar mapa nome->ID para EXPANDS.")
    ucs_by_origin_level: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    for uc in generated_ucs:
        origin_id = uc.get("origin_id")
        bloom_level = uc.get("bloom_level")
        uc_id = uc.get("uc_id")
        if origin_id and bloom_level and uc_id and bloom_level in BLOOM_ORDER_MAP:
            ucs_by_origin_level[origin_id][bloom_level].append(uc_id)
    logging.info(f"Criado mapa UC por origem/nível ({len(ucs_by_origin_level)} origens).")
    return entity_name_to_id, ucs_by_origin_level

def _create_expands_links(
    relationships_df: pd.DataFrame,
    entity_name_to_id: Dict[str, str],
    ucs_by_origin_level: Dict[str, Dict[str, List[str]]]
) -> List[Dict[str, Any]]:
    """Cria as relações EXPANDS com base nas relações do GraphRAG."""
    new_expands_rels = []
    processed_graphrag_rels = 0
    skipped_missing_entity = 0
    LEVELS_TO_CONNECT = ["Lembrar", "Entender"]
    logging.info(f"Processando {len(relationships_df)} relações GraphRAG para EXPANDS (Níveis: {LEVELS_TO_CONNECT})...")
    if not ('source' in relationships_df.columns and 'target' in relationships_df.columns):
        logging.error("'source'/'target' faltando em relationships.parquet.")
        return []
    for row in relationships_df.itertuples(index=False):
        s_name = row.source
        t_name = row.target
        weight_val = getattr(row, 'weight', 1.0)
        rel_weight = float(weight_val) if pd.notna(weight_val) else 1.0
        rel_desc = getattr(row, 'description', None)
        desc_clean = rel_desc if rel_desc is not None and pd.notna(rel_desc) else None
        s_id = entity_name_to_id.get(s_name)
        t_id = entity_name_to_id.get(t_name)
        if not s_id or not t_id:
            skipped_missing_entity += 1
            continue
        if s_id == t_id:
            continue
        if s_id in ucs_by_origin_level and t_id in ucs_by_origin_level:
            processed_graphrag_rels += 1
            for bloom_level in LEVELS_TO_CONNECT:
                s_ucs = ucs_by_origin_level[s_id].get(bloom_level, [])
                t_ucs = ucs_by_origin_level[t_id].get(bloom_level, [])
                if s_ucs and t_ucs:
                    for s_uc_id in s_ucs:
                        for t_uc_id in t_ucs:
                            rel = {
                                "source": s_uc_id,
                                "target": t_uc_id,
                                "type": "EXPANDS",
                                "weight": rel_weight,
                                "graphrag_rel_desc": desc_clean,
                            }
                            new_expands_rels.append(rel)
                            rev_rel = {
                                "source": t_uc_id,
                                "target": s_uc_id,
                                "type": "EXPANDS",
                                "weight": rel_weight,
                                "graphrag_rel_desc": desc_clean,
                            }
                            new_expands_rels.append(rev_rel)
    logging.info(f"Processadas {processed_graphrag_rels} relações GraphRAG com UCs.")
    if skipped_missing_entity > 0:
        logging.warning(f"{skipped_missing_entity} relações puladas (entidade não mapeada).")
    logging.info(f"Candidatas a {len(new_expands_rels)} novas relações EXPANDS.")
    return new_expands_rels

def _add_relationships_avoiding_duplicates(
    existing_rels: List[Dict[str, Any]],
    new_rels: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Adiciona novas relações a uma lista existente, evitando duplicatas."""
    if not new_rels:
        return existing_rels
    updated_rels = list(existing_rels)
    existing_rel_tuples = {(
        r.get("source"), r.get("target"), r.get("type")
    ) for r in updated_rels}
    added_count = 0
    for rel in new_rels:
        rel_tuple = (rel.get("source"), rel.get("target"), rel.get("type"))
        if rel_tuple not in existing_rel_tuples:
            updated_rels.append(rel)
            existing_rel_tuples.add(rel_tuple)
            added_count += 1
    logging.info(
        f"{added_count} novas relações adicionadas "
        f"({len(new_rels) - added_count} duplicatas evitadas)."
    )
    return updated_rels