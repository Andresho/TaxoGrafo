import pandas as pd
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set, Optional
from abc import ABC, abstractmethod
from app.scripts.constants import BLOOM_ORDER, BLOOM_ORDER_MAP
from app.scripts.io_utils import save_dataframe, load_dataframe

class OriginSelector(ABC):
    """Interface para seleção de origens de UC."""
    @abstractmethod
    def select(self, all_origins: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Retorna a lista de origens selecionadas."""
        pass

class DefaultSelector(OriginSelector):
    """Selector que retorna origens ordenadas e limitadas (ou todas)."""
    def __init__(self, max_origins: Optional[int] = None):
        self.max_origins = max_origins
    def select(self, all_origins: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self.max_origins is None or self.max_origins <= 0:
            return all_origins
        # ordena e retorna os primeiros max_origins
        return sorted(all_origins, key=_get_sort_key)[: self.max_origins]

class HubNeighborSelector(OriginSelector):
    """Selector que foca em conexões de um hub e seus vizinhos."""
    def __init__(self, max_origins: int, graphrag_output_dir: Path):
        self.max_origins = max_origins
        self.graphrag_output_dir = graphrag_output_dir
    def select(self, all_origins: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Reutiliza lógica existente de seleção para teste
        return _select_origins_for_testing(all_origins, self.graphrag_output_dir, self.max_origins)


def prepare_uc_origins(
        enriched_entities: List[Dict[str, Any]],
        community_reports_list: List[Dict[str, Any]],
        community_structures_list: List[Dict[str, Any]],
        hr_id_to_uuid_map: Dict[str, str]
) -> List[Dict[str, Any]]:
    """
    Prepara a lista de 'origens' para a geração de UCs, incluindo o parent_community_id_of_origin.
    """
    uc_origins: List[Dict[str, Any]] = []
    logging.info("Preparando origens de UC...")

    community_uuid_to_parent_uuid_map: Dict[str, Optional[str]] = {}
    for cs_rec in community_structures_list:
        cs_uuid = cs_rec.get('id')
        cs_parent_uuid = cs_rec.get('parent_community_id')
        if cs_uuid:
            community_uuid_to_parent_uuid_map[str(cs_uuid)] = str(cs_parent_uuid) if cs_parent_uuid else None

    if enriched_entities:
        logging.info(f"Processando {len(enriched_entities)} entidades enriquecidas para origens...")
        for entity_rec in enriched_entities:
            parent_community_id_for_entity_origin = entity_rec.get('parent_community_id')
            uc_origins.append({
                "origin_id": entity_rec.get("id"),
                "origin_type": "entity",
                "title": entity_rec.get("title"),
                "context": entity_rec.get("description", ""),
                "frequency": int(entity_rec.get("frequency", 0)),
                "degree": int(entity_rec.get("degree", 0)),
                "entity_type": entity_rec.get("type", "unknown"),
                "level": 0,
                "parent_community_id_of_origin": str(
                    parent_community_id_for_entity_origin) if parent_community_id_for_entity_origin else None
            })

    if community_reports_list:
        logging.info(f"Processando {len(community_reports_list)} relatórios de comunidade para origens...")
        for report_rec in community_reports_list:
            report_community_hr_id_val = report_rec.get("community")
            report_title = report_rec.get("title", "Relatório Sem Título")
            parent_id_for_this_report_origin = None
            origin_id_for_report = None  # Inicializa para evitar UnboundLocalError

            if pd.notna(report_community_hr_id_val):
                report_community_hr_id_str = str(int(report_community_hr_id_val)) if pd.api.types.is_number(
                    report_community_hr_id_val) else str(report_community_hr_id_val)
                report_community_uuid = hr_id_to_uuid_map.get(report_community_hr_id_str)

                if report_community_uuid:
                    origin_id_for_report = str(report_community_uuid)  # Define o origin_id corretamente
                    parent_id_for_this_report_origin = community_uuid_to_parent_uuid_map.get(origin_id_for_report)
                else:
                    logging.warning(
                        f"Relatório '{report_title}' tem community_hr_id '{report_community_hr_id_str}' não mapeado para UUID. Pulando origem.")
                    continue
            else:
                logging.warning(f"Relatório '{report_title}' não tem 'community' (human_readable_id). Pulando origem.")
                continue

            uc_origins.append({
                "origin_id": origin_id_for_report,
                "origin_type": "community_report",
                "title": report_title,
                "context": report_rec.get("summary", ""),
                "frequency": 0,
                "degree": 0,
                "entity_type": "community",
                "level": int(report_rec.get("level", 99)),
                "parent_community_id_of_origin": str(
                    parent_id_for_this_report_origin) if parent_id_for_this_report_origin else None
            })

    logging.info(f"Total {len(uc_origins)} origens de UC preparadas.")
    return uc_origins

def _get_sort_key(origin: Dict[str, Any]) -> Tuple[int, int]:
    """Calcula a chave de ordenação para uma origem."""
    origin_type = origin.get("origin_type")
    score = 0
    type_priority = 2
    if origin_type == "community_report":
        level = origin.get("level", 99)
        score = 10000 - level
        type_priority = 1
    elif origin_type == "entity":
        degree = origin.get("degree", 0)
        freq = origin.get("frequency", 0)
        entity_type = origin.get("entity_type", "unknown").lower()
        score = degree * 10 + freq
        if entity_type == "person":
            type_priority = 3
        elif entity_type in ["organization", "geo", "event", "unknown"]:
            type_priority = 2
        else:
            type_priority = 1
    return (type_priority, -score)

def _select_origins_for_testing(
    all_origins: List[Dict[str, Any]],
    graphrag_output_dir: Path,
    max_origins: int
) -> List[Dict[str, Any]]:
    """Seleciona origens para teste, focando em conexões."""
    logging.warning(f"--- MODO DE TESTE ATIVO (Foco em Conexões, Max: {max_origins}) ---")
    if len(all_origins) <= max_origins:
        return all_origins
    entity_origins = [o for o in all_origins if o.get("origin_type") == "entity"]
    if not entity_origins:
        logging.warning("Nenhuma origem 'entity' para teste. Usando as primeiras gerais.")
        return sorted(all_origins, key=_get_sort_key)[:max_origins]
    entity_origins.sort(key=_get_sort_key)
    hub_origin = entity_origins[0]
    hub_id = hub_origin.get("origin_id")
    logging.info(f"Hub selecionado: ID={hub_id}, Title='{hub_origin.get('title')[:50]}...'")
    neighbor_ids: Set[str] = set()
    # Carrega DataFrames dinamicamente para permitir monkeypatch em pipeline_tasks
    relationships_df = load_dataframe(graphrag_output_dir, "relationships")
    entities_df = load_dataframe(graphrag_output_dir, "entities")
    if relationships_df is not None and entities_df is not None:
        entity_name_to_id: Dict[str, str] = {}
        if 'title' in entities_df.columns and 'id' in entities_df.columns:
            entity_name_to_id = pd.Series(entities_df.id.values, index=entities_df.title).to_dict()
        if entity_name_to_id and 'source' in relationships_df.columns and 'target' in relationships_df.columns:
            logging.info(f"Buscando vizinhos do Hub (ID: {hub_id})...")
            for row in relationships_df.itertuples(index=False):
                s_id = entity_name_to_id.get(row.source)
                t_id = entity_name_to_id.get(row.target)
                if s_id == hub_id and t_id and t_id != hub_id:
                    neighbor_ids.add(t_id)
                elif t_id == hub_id and s_id and s_id != hub_id:
                    neighbor_ids.add(s_id)
        else:
            logging.warning("Não buscou vizinhos (mapa nome->ID ou colunas).")
    else:
        logging.warning("Não carregou relationships/entities para buscar vizinhos.")
    logging.info(f"Encontrados {len(neighbor_ids)} vizinhos únicos.")
    final_ids_to_process = {hub_id}
    neighbors_to_add = list(neighbor_ids)[: max_origins - 1]
    final_ids_to_process.update(neighbors_to_add)
    logging.info(f"Conjunto final teste: {len(final_ids_to_process)} IDs.")
    selected_origins = [o for o in all_origins if o.get("origin_id") in final_ids_to_process]
    selected_origins.sort(key=_get_sort_key)
    return selected_origins