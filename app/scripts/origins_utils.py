import pandas as pd
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set, Optional
from abc import ABC, abstractmethod
from scripts.constants import BLOOM_ORDER, BLOOM_ORDER_MAP

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
    entities_df: Optional[pd.DataFrame],
    reports_df: Optional[pd.DataFrame]
) -> List[Dict[str, Any]]:
    """Prepara a lista de 'origens' para a geração de UCs."""
    uc_origins = []
    logging.info("Preparando origens de UC...")
    if entities_df is not None:
        logging.info(f"Processando {len(entities_df)} entidades...")
        req_cols = ['id', 'title', 'description', 'frequency', 'degree', 'type']
        if all(c in entities_df.columns for c in req_cols):
            for r in entities_df.itertuples(index=False):
                freq = int(r.frequency) if pd.notna(r.frequency) else 0
                deg = int(r.degree) if pd.notna(r.degree) else 0
                entity_type = r.type if pd.notna(r.type) else "unknown"
                uc_origins.append({
                    "origin_id": r.id,
                    "origin_type": "entity",
                    "title": r.title,
                    "context": r.description if pd.notna(r.description) else "",
                    "frequency": freq,
                    "degree": deg,
                    "entity_type": entity_type,
                    "level": 0
                })
        else:
            missing = [c for c in req_cols if c not in entities_df.columns]
            logging.warning(f"Colunas faltando em entities.parquet: {missing}")
    if reports_df is not None:
        logging.info(f"Processando {len(reports_df)} resumos de comunidade...")
        req_cols = ['id', 'community', 'title', 'summary', 'level']
        if all(c in reports_df.columns for c in req_cols):
            for r in reports_df.itertuples(index=False):
                level = int(r.level) if pd.notna(r.level) else 99
                uc_origins.append({
                    "origin_id": r.id,
                    "origin_type": "community_report",
                    "title": r.title,
                    "context": r.summary if pd.notna(r.summary) else "",
                    "frequency": 0,
                    "degree": 0,
                    "entity_type": "community",
                    "level": level
                })
        else:
            missing = [c for c in req_cols if c not in reports_df.columns]
            logging.warning(f"Colunas faltando em community_reports.parquet: {missing}")
    logging.info(f"Total {len(uc_origins)} origens preparadas.")
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
    import scripts.pipeline_tasks as pt
    relationships_df = pt.load_dataframe(graphrag_output_dir, "relationships")
    entities_df = pt.load_dataframe(graphrag_output_dir, "entities")
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