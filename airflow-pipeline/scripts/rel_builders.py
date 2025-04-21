import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from collections import defaultdict

from scripts.constants import BLOOM_ORDER_MAP, BASE_INPUT_DIR, GENERATED_UCS_RAW, REL_INTERMEDIATE
from scripts.rel_utils import _add_relationships_avoiding_duplicates, _prepare_expands_lookups, _create_expands_links

class RelationBuilder(ABC):
    """Interface e pipeline para construir relações entre UCs."""
    def __init__(self):
        self._next = None

    def set_next(self, next_builder: 'RelationBuilder') -> 'RelationBuilder':
        self._next = next_builder
        return next_builder

    def build(self, relations: List[Dict[str, Any]], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        # Executa o handler atual
        updated = self._handle(relations, context)
        # Chama próximo se existir
        if self._next:
            return self._next.build(updated, context)
        return updated

    @abstractmethod
    def _handle(self, relations: List[Dict[str, Any]], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Implementar lógica de construção de relações."""
        pass

class RequiresBuilder(RelationBuilder):
    """Builder que gera relações do tipo REQUIRES segundo ordem de Bloom."""
    def _handle(self, relations: List[Dict[str, Any]], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        generated = context.get('generated_ucs', [])
        # Agrupa por origem
        ucs_by_origin = defaultdict(list)
        for uc in generated:
            origin_id = uc.get('origin_id')
            if origin_id:
                ucs_by_origin[origin_id].append(uc)
        new_rels = []
        for origin_id, ucs in ucs_by_origin.items():
            # Ordena pelo nível Bloom
            sorted_ucs = sorted(
                ucs,
                key=lambda u: BLOOM_ORDER_MAP.get(u.get('bloom_level'), 99)
            )
            # Cria relações sequenciais
            for i in range(len(sorted_ucs) - 1):
                s_uc = sorted_ucs[i]
                t_uc = sorted_ucs[i + 1]
                s_idx = BLOOM_ORDER_MAP.get(s_uc.get('bloom_level'))
                t_idx = BLOOM_ORDER_MAP.get(t_uc.get('bloom_level'))
                if s_idx is not None and t_idx == s_idx + 1:
                    new_rels.append({
                        'source': s_uc.get('uc_id'),
                        'target': t_uc.get('uc_id'),
                        'type': 'REQUIRES',
                        'origin_id': origin_id
                    })
        # Evita duplicadas
        return _add_relationships_avoiding_duplicates(relations, new_rels)

class ExpandsBuilder(RelationBuilder):
    """Builder que gera relações do tipo EXPANDS baseadas no GraphRAG."""
    def _handle(self, relations: List[Dict[str, Any]], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        rels_df = context.get('relationships_df')
        entities_df = context.get('entities_df')
        generated = context.get('generated_ucs', [])
        if rels_df is None or entities_df is None:
            logging.warning("Pulando EXPANDS (inputs não carregados).")
            return relations
        # Prepara lookups e UC levels
        name_to_id, ucs_by_level = _prepare_expands_lookups(entities_df, generated)
        if not name_to_id:
            logging.warning("Pulando EXPANDS (mapa nome->ID falhou).")
            return relations
        # Cria relações EXPANDS
        new_rels = _create_expands_links(rels_df, name_to_id, ucs_by_level)
        # Evita duplicatas
        return _add_relationships_avoiding_duplicates(relations, new_rels)