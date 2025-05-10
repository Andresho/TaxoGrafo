# scripts/difficulty_scheduler.py
import logging
from collections import defaultdict, Counter
import random
from typing import List, Dict, Set, Tuple, Optional, Any

DEFAULT_MAX_ASCENT_LEVELS = 1
DEFAULT_NEIGHBOR_POOL_MULTIPLIER = 2

class OriginDifficultyScheduler:
    def __init__(self,
                 all_knowledge_origins: List[Dict[str, Any]],
                 min_evaluations_per_origin: int,
                 difficulty_batch_size: int,
                 max_ascent_levels: int = DEFAULT_MAX_ASCENT_LEVELS):

        if not all_knowledge_origins:
            raise ValueError("Lista de Knowledge Origins não pode ser vazia.")
        if difficulty_batch_size <= 1:
            raise ValueError("Tamanho do batch de dificuldade deve ser maior que 1.")

        self.all_knowledge_origins = all_knowledge_origins
        self.MIN_EVALUATIONS_PER_ORIGIN = min_evaluations_per_origin
        self.DIFFICULTY_BATCH_SIZE = difficulty_batch_size
        self.MAX_ASCENT_LEVELS = max_ascent_levels

        self._initialize_internal_structures()

    def _initialize_internal_structures(self):
        """Prepara mapas e contadores internos."""
        self.origin_details: Dict[str, Dict[str, Any]] = \
            {str(o['origin_id']): o for o in self.all_knowledge_origins}

        self.origin_to_parent_map: Dict[str, Optional[str]] = {
            str(o['origin_id']): str(o['parent_community_id_of_origin']) if o.get(
                'parent_community_id_of_origin') else None
            for o in self.all_knowledge_origins
        }

        self.context_to_origins_map: Dict[str, Dict[Tuple[str, int], List[str]]] = \
            defaultdict(lambda: defaultdict(list))
        for o_data in self.all_knowledge_origins:
            origin_id = str(o_data['origin_id'])
            context_id = self.origin_to_parent_map.get(origin_id) or "__ROOT__"

            o_type = o_data.get('origin_type')
            o_level = o_data.get('level', 0)

            if not isinstance(o_type, str) or not isinstance(o_level, int):
                logging.warning(
                    f"Origem {origin_id} com tipo/nível inválido: {o_type}, {o_level}. Ignorada no mapeamento.")
                continue
            self.context_to_origins_map[context_id][(o_type, o_level)].append(origin_id)

        self.evaluation_counts: Counter = Counter()
        self.pending_origins: Set[str] = set(self.origin_details.keys())
        logging.info(f"OriginDifficultyScheduler inicializado com {len(self.pending_origins)} origens pendentes.")

    def _get_seed_origin(self) -> Optional[str]:
        """Seleciona a próxima origem semente, priorizando as menos avaliadas."""
        if not self.pending_origins:
            return None

        sorted_pending = sorted(
            list(self.pending_origins),
            key=lambda oid: (self.evaluation_counts[oid], oid)
        )
        return sorted_pending[0]

    def _get_neighbors_for_seed(self, seed_origin_id: str) -> List[str]:
        """Encontra vizinhos elegíveis para uma origem semente, subindo na hierarquia se necessário."""
        seed_details = self.origin_details.get(seed_origin_id)
        if not seed_details:
            logging.warning(f"Detalhes não encontrados para a origem semente {seed_origin_id}")
            return []

        seed_type = seed_details.get('origin_type')
        seed_level = seed_details.get('level', 0)

        collected_neighbors: Set[str] = set()
        current_search_context_id = self.origin_to_parent_map.get(seed_origin_id) or "__ROOT__"

        for ascent_level in range(self.MAX_ASCENT_LEVELS + 1):
            if not current_search_context_id: break

            origins_in_context = self.context_to_origins_map.get(current_search_context_id, {}).get(
                (seed_type, seed_level), []
            )

            for neighbor_id in origins_in_context:
                if neighbor_id != seed_origin_id:
                    collected_neighbors.add(neighbor_id)

            if len(collected_neighbors) >= self.DIFFICULTY_BATCH_SIZE * DEFAULT_NEIGHBOR_POOL_MULTIPLIER or \
                    ascent_level == self.MAX_ASCENT_LEVELS or \
                    current_search_context_id == "__ROOT__":
                break

            parent_of_current_context = self.origin_to_parent_map.get(
                current_search_context_id)
            current_search_context_id = parent_of_current_context or "__ROOT__"

        return list(collected_neighbors)

    def _select_final_neighbors(self, seed_origin_id: str, available_neighbors: List[str], num_needed: int) -> List[
        str]:
        """Seleciona os vizinhos finais da lista de elegíveis."""
        if not available_neighbors or num_needed <= 0:
            return []

        # Prioriza vizinhos com menos avaliações, depois aleatório
        # Isto ajuda a distribuir as avaliações mais uniformemente entre os vizinhos também
        random.shuffle(available_neighbors) 

        sorted_neighbors = sorted(
            available_neighbors,
            key=lambda oid: (self.evaluation_counts[oid], oid)
        )
        return sorted_neighbors[:num_needed]

    def _handle_insufficient_neighbors(self, seed_origin_id: str):
        """Lida com o caso de uma origem semente não encontrar vizinhos suficientes."""
        logging.warning(
            f"Origem semente {seed_origin_id} não formará um batch de dificuldade completo. "
            f"Incrementando contagem de 'tentativa' para evitar loop."
        )
        self.evaluation_counts[seed_origin_id] += 1
        if self.evaluation_counts[seed_origin_id] >= self.MIN_EVALUATIONS_PER_ORIGIN:
            self.pending_origins.discard(seed_origin_id)

    def _update_counts_and_pending_status(self, batch_origin_ids: Tuple[str, ...]):
        """Atualiza contadores de avaliação e o conjunto de origens pendentes."""
        for oid in batch_origin_ids:
            self.evaluation_counts[oid] += 1
            if self.evaluation_counts[oid] >= self.MIN_EVALUATIONS_PER_ORIGIN:
                self.pending_origins.discard(oid)

    def generate_origin_pairings(self) -> List[Tuple[str, ...]]:
        """
        Gera uma lista de tuplas, onde cada tupla contém origin_ids
        que devem ser comparados juntos em um batch de dificuldade.
        """
        paired_sets_for_difficulty: List[Tuple[str, ...]] = []

        while self.pending_origins:
            seed_origin_id = self._get_seed_origin()
            if not seed_origin_id: break

            available_neighbors = self._get_neighbors_for_seed(seed_origin_id)

            num_neighbors_needed = self.DIFFICULTY_BATCH_SIZE - 1

            if len(available_neighbors) < num_neighbors_needed:
                self._handle_insufficient_neighbors(seed_origin_id)
                continue

            selected_neighbors = self._select_final_neighbors(seed_origin_id, available_neighbors, num_neighbors_needed)

            current_batch_origins = tuple(
                sorted([seed_origin_id] + selected_neighbors))
            paired_sets_for_difficulty.append(current_batch_origins)
            self._update_counts_and_pending_status(current_batch_origins)

        logging.info(f"Gerados {len(paired_sets_for_difficulty)} conjuntos de origens para avaliação de dificuldade.")
        if self.pending_origins:
            logging.warning(f"{len(self.pending_origins)} origens não atingiram o mínimo de avaliações "
                            f"devido à falta de vizinhos para comparação: {list(self.pending_origins)}")
        return paired_sets_for_difficulty