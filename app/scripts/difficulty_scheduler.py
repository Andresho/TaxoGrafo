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
        self.NEIGHBOR_POOL_MULTIPLIER = DEFAULT_NEIGHBOR_POOL_MULTIPLIER

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

            if not isinstance(o_type, str) or not (
                    isinstance(o_level, int) or isinstance(o_level, float) and o_level.is_integer()):
                logging.warning(
                    f"Origem {origin_id} com tipo/nível inválido: {o_type}, {o_level}. Ignorada no mapeamento context_to_origins_map.")
                continue

            self.context_to_origins_map[context_id][(o_type, int(o_level))].append(origin_id)

        self.evaluation_counts: Counter = Counter()
        self.pending_origins: Set[str] = set(self.origin_details.keys())
        logging.info(f"OriginDifficultyScheduler inicializado com {len(self.pending_origins)} origens pendentes.")

    def _get_seed_origin(self) -> Optional[str]:
        """Seleciona a próxima origem semente, priorizando as menos avaliadas."""
        if not self.pending_origins:
            return None

        valid_pending_origins = [oid for oid in self.pending_origins if oid in self.origin_details]
        if not valid_pending_origins:
            logging.warning("_get_seed_origin: No valid pending origins found in origin_details.")
            return None

        sorted_pending = sorted(
            list(valid_pending_origins),
            key=lambda oid: (self.evaluation_counts[oid], oid)
        )
        return sorted_pending[0]

    def _get_neighbors_for_seed(self, seed_origin_id: str) -> Tuple[List[str], str]:
        """
        Encontra vizinhos elegíveis para uma origem semente, subindo na hierarquia se necessário.
        Retorna uma tupla: (lista_de_vizinhos, string_nivel_coerencia_hierarquica).
        """
        seed_details = self.origin_details.get(seed_origin_id)
        if not seed_details:
            logging.warning(
                f"Detalhes não encontrados para a origem semente {seed_origin_id} em _get_neighbors_for_seed")
            return [], "hierarchical_error_no_seed_details"

        seed_type = str(seed_details.get('origin_type'))
        seed_origin_level = int(seed_details.get('level', 0))
        logging.info(f"[NEIGHBOR_SEARCH] Seed: {seed_origin_id} (Type: {seed_type}, OriginLevel: {seed_origin_level})")

        collected_neighbors: Set[str] = set()
        current_search_context_id = self.origin_to_parent_map.get(seed_origin_id) or "__ROOT__"
        logging.info(f"[NEIGHBOR_SEARCH] Seed {seed_origin_id}: Initial search context_id: {current_search_context_id}")

        final_coherence_label = "hierarchical_no_neighbors_found"

        for ascent_count in range(self.MAX_ASCENT_LEVELS + 1):
            if not current_search_context_id:
                logging.info(
                    f"[NEIGHBOR_SEARCH] Seed {seed_origin_id}: No current_search_context_id to search in (ascent_count {ascent_count}). Breaking.")
                break

            logging.info(
                f"[NEIGHBOR_SEARCH] Seed {seed_origin_id}: Ascent count {ascent_count}, searching in context_id: {current_search_context_id} for neighbors of type '{seed_type}' and OriginLevel '{seed_origin_level}'")

            origins_in_context_for_type_level = self.context_to_origins_map.get(current_search_context_id, {}).get(
                (seed_type, seed_origin_level), []
            )

            if not origins_in_context_for_type_level:
                logging.info(
                    f"[NEIGHBOR_SEARCH] Seed {seed_origin_id}: No origins of type '{seed_type}', OriginLevel '{seed_origin_level}' found in context_id '{current_search_context_id}'.")
            else:
                logging.debug(
                    f"[NEIGHBOR_SEARCH] Seed {seed_origin_id}: Found {len(origins_in_context_for_type_level)} potential origins in context '{current_search_context_id}' for type '{seed_type}', OriginLevel '{seed_origin_level}'.")

            found_in_this_ascent = False
            for neighbor_id in origins_in_context_for_type_level:
                if neighbor_id != seed_origin_id:
                    if neighbor_id in self.pending_origins or self.evaluation_counts[
                        neighbor_id] < self.MIN_EVALUATIONS_PER_ORIGIN:
                        if neighbor_id not in collected_neighbors:
                            collected_neighbors.add(neighbor_id)
                            found_in_this_ascent = True
                            logging.debug(
                                f"[NEIGHBOR_SEARCH] Seed {seed_origin_id}: Added eligible neighbor {neighbor_id} from context {current_search_context_id} (ascent {ascent_count}).")
                    else:
                        logging.debug(
                            f"[NEIGHBOR_SEARCH] Seed {seed_origin_id}: Candidate neighbor {neighbor_id} from context {current_search_context_id} already met evaluation count. Skipping.")

            if found_in_this_ascent:
                if current_search_context_id == "__ROOT__":
                    final_coherence_label = f"hierarchical_root_context_ascent_{ascent_count}"
                else:
                    final_coherence_label = f"hierarchical_ascent_{ascent_count}"

            if len(collected_neighbors) >= self.DIFFICULTY_BATCH_SIZE * self.NEIGHBOR_POOL_MULTIPLIER:
                logging.info(
                    f"[NEIGHBOR_SEARCH] Seed {seed_origin_id}: Neighbor pool full ({len(collected_neighbors)} collected) at ascent_count {ascent_count}. Breaking.")
                break
            if ascent_count == self.MAX_ASCENT_LEVELS:
                logging.info(
                    f"[NEIGHBOR_SEARCH] Seed {seed_origin_id}: Max ascent levels reached ({self.MAX_ASCENT_LEVELS}). Breaking.")
                break
            if current_search_context_id == "__ROOT__":
                logging.info(
                    f"[NEIGHBOR_SEARCH] Seed {seed_origin_id}: Reached and searched __ROOT__ context. Breaking ascent.")
                break

            parent_of_current_context = self.origin_to_parent_map.get(current_search_context_id)
            if not parent_of_current_context and current_search_context_id != "__ROOT__":
                logging.warning(
                    f"[NEIGHBOR_SEARCH] Seed {seed_origin_id}: Context {current_search_context_id} has no parent and is not __ROOT__. Stopping ascent here.")
                break
            current_search_context_id = parent_of_current_context or "__ROOT__"
            logging.debug(
                f"[NEIGHBOR_SEARCH] Seed {seed_origin_id}: Ascending to next context: {current_search_context_id}")

        if not collected_neighbors and final_coherence_label == "hierarchical_no_neighbors_found":
            logging.info(f"[NEIGHBOR_SEARCH] Seed {seed_origin_id}: No hierarchical neighbors found after all ascents.")
        elif collected_neighbors:
            if final_coherence_label == "hierarchical_no_neighbors_found":
                final_coherence_label = "hierarchical_found_unspecified_ascent"

        logging.info(
            f"[NEIGHBOR_SEARCH] Seed {seed_origin_id}: Total collected hierarchical neighbors: {len(collected_neighbors)}. Coherence: {final_coherence_label}")
        return list(collected_neighbors), final_coherence_label

    def _get_global_fallback_neighbors(self, seed_origin_type: str, seed_origin_level: int, exclusions: Set[str],
                                       num_needed: int) -> List[str]:
        """
        Encontra vizinhos de um pool global de origens pendentes que correspondem ao tipo e nível da origem semente.
        Exclui IDs fornecidos em 'exclusions'. Retorna 'num_needed' vizinhos, priorizando os menos avaliados.
        """
        if num_needed <= 0:
            return []

        logging.info(
            f"[FALLBACK_SEARCH] Seeking {num_needed} fallback neighbors of type '{seed_origin_type}', origin_level '{seed_origin_level}', excluding {len(exclusions)} IDs.")

        candidate_fallback_neighbors: List[str] = []
        for oid in self.pending_origins:
            if oid in exclusions:
                continue

            details = self.origin_details.get(oid)
            if not details:
                logging.warning(
                    f"[FALLBACK_SEARCH] Origin {oid} in pending_origins but not in origin_details. Skipping.")
                continue

            if details.get('origin_type') == seed_origin_type and int(details.get('level', 0)) == seed_origin_level:
                candidate_fallback_neighbors.append(oid)

        logging.info(
            f"[FALLBACK_SEARCH] Found {len(candidate_fallback_neighbors)} global candidates of same type/level before sorting/selection.")

        if not candidate_fallback_neighbors:
            logging.warning(
                f"[FALLBACK_SEARCH] No global fallback candidates found for type '{seed_origin_type}', origin_level '{seed_origin_level}'.")
            return []

        shuffled_candidates = list(candidate_fallback_neighbors)
        random.shuffle(shuffled_candidates)

        sorted_fallback_neighbors = sorted(
            shuffled_candidates,
            key=lambda oid_sort: (self.evaluation_counts[oid_sort], oid_sort)
        )

        selected_fallback = sorted_fallback_neighbors[:num_needed]
        logging.info(f"[FALLBACK_SEARCH] Selected {len(selected_fallback)} fallback neighbors: {selected_fallback}")
        return selected_fallback

    def _select_final_neighbors(self, seed_origin_id: str, available_neighbors: List[str], num_needed: int) -> List[
        str]:
        """Seleciona os vizinhos finais da lista de elegíveis, priorizando os menos avaliados."""
        logging.info(
            f"[SELECT_NEIGHBORS] Seed {seed_origin_id}: Attempting to select {num_needed} neighbors from {len(available_neighbors)} available.")
        if not available_neighbors or num_needed <= 0:
            if not available_neighbors:
                logging.warning(f"[SELECT_NEIGHBORS] Seed {seed_origin_id}: No available neighbors to select from.")
            if num_needed <= 0:
                logging.warning(
                    f"[SELECT_NEIGHBORS] Seed {seed_origin_id}: Number of neighbors needed is {num_needed} or less.")
            return []

        eligible_neighbors = [
            oid for oid in available_neighbors
            if oid in self.pending_origins or self.evaluation_counts[oid] < self.MIN_EVALUATIONS_PER_ORIGIN
        ]
        logging.info(
            f"[SELECT_NEIGHBORS] Seed {seed_origin_id}: {len(eligible_neighbors)} neighbors are eligible (pending or < min_evals) out of {len(available_neighbors)} available.")

        if not eligible_neighbors:
            logging.warning(
                f"[SELECT_NEIGHBORS] Seed {seed_origin_id}: No eligible neighbors after filtering by pending/evaluation_counts.")
            return []

        shuffled_eligible_neighbors = list(eligible_neighbors)
        random.shuffle(shuffled_eligible_neighbors)

        sorted_neighbors = sorted(
            shuffled_eligible_neighbors,
            key=lambda oid_sort: (self.evaluation_counts[oid_sort], oid_sort)
        )

        selected = sorted_neighbors[:num_needed]
        logging.info(f"[SELECT_NEIGHBORS] Seed {seed_origin_id}: Selected {len(selected)} final neighbors: {selected}")
        return selected

    def _handle_insufficient_neighbors(self, seed_origin_id: str):
        """Lida com o caso de uma origem semente não encontrar vizinhos suficientes."""
        logging.warning(
            f"[INSUFFICIENT_NEIGHBORS] Seed {seed_origin_id} did not form a full difficulty batch. "
            f"Current evaluation count for seed is {self.evaluation_counts[seed_origin_id]}. "
            f"Incrementing 'attempt' count. Will be removed from pending if count reaches {self.MIN_EVALUATIONS_PER_ORIGIN}."
        )
        self.evaluation_counts[seed_origin_id] += 1
        if self.evaluation_counts[seed_origin_id] >= self.MIN_EVALUATIONS_PER_ORIGIN:
            if seed_origin_id in self.pending_origins:
                self.pending_origins.discard(seed_origin_id)
                logging.info(
                    f"[INSUFFICIENT_NEIGHBORS] Seed {seed_origin_id} reached {self.evaluation_counts[seed_origin_id]} attempts, removing from pending_origins.")
            else:
                logging.info(
                    f"[INSUFFICIENT_NEIGHBORS] Seed {seed_origin_id} reached {self.evaluation_counts[seed_origin_id]} attempts, was already removed from pending_origins.")

    def _update_counts_and_pending_status(self, batch_origin_ids: Tuple[str, ...]):
        """Atualiza contadores de avaliação e o conjunto de origens pendentes."""
        for oid in batch_origin_ids:
            self.evaluation_counts[oid] += 1
            if self.evaluation_counts[oid] >= self.MIN_EVALUATIONS_PER_ORIGIN:
                if oid in self.pending_origins:
                    self.pending_origins.discard(oid)
                    logging.info(
                        f"[UPDATE_COUNTS] Origin {oid} reached {self.evaluation_counts[oid]} evaluations, removed from pending.")

    def generate_origin_pairings(self) -> List[Dict[str, Any]]:
        """
        Gera uma lista de dicionários, onde cada dicionário contém:
        - "origin_ids": Tupla de origin_ids que devem ser comparados.
        - "coherence_level": String indicando como o grupo foi formado.
        - "seed_id_for_batch": O origin_id da semente que iniciou este grupo.
        Inclui um fallback para origens isoladas.
        """
        paired_sets_for_difficulty: List[Dict[str, Any]] = []
        iteration_count = 0

        max_iterations = len(self.all_knowledge_origins) * self.MIN_EVALUATIONS_PER_ORIGIN * 2
        if max_iterations == 0 and self.all_knowledge_origins: max_iterations = len(self.all_knowledge_origins) * 5

        while self.pending_origins and iteration_count < max_iterations:
            iteration_count += 1
            logging.info(
                f"[PAIRING_LOOP] Iteration {iteration_count}/{max_iterations}, {len(self.pending_origins)} origins pending.")

            seed_origin_id = self._get_seed_origin()
            if not seed_origin_id:
                logging.info("[PAIRING_LOOP] No more seed origins available from _get_seed_origin. Breaking.")
                break

            seed_details = self.origin_details.get(seed_origin_id)
            if not seed_details:
                logging.error(
                    f"[PAIRING_LOOP] Seed origin {seed_origin_id} has no details in self.origin_details. Discarding from pending and continuing.")
                self.pending_origins.discard(seed_origin_id)
                continue

            seed_type = str(seed_details.get('origin_type'))
            seed_origin_level = int(seed_details.get('level', 0))
            logging.info(
                f"[PAIRING_LOOP] Selected seed: {seed_origin_id} (Type: {seed_type}, OriginLevel: {seed_origin_level}, EvalCount: {self.evaluation_counts[seed_origin_id]})")

            num_neighbors_needed = self.DIFFICULTY_BATCH_SIZE - 1
            if num_neighbors_needed < 0:
                logging.error(
                    f"[PAIRING_LOOP] num_neighbors_needed is {num_neighbors_needed}, which is invalid. DIFFICULTY_BATCH_SIZE: {self.DIFFICULTY_BATCH_SIZE}")
                self._handle_insufficient_neighbors(seed_origin_id)
                continue

            selected_neighbors_list: List[str] = []
            coherence_label_from_hierarchical_search = "hierarchical_not_attempted_or_failed"

            available_hierarchical_neighbors, coherence_label_from_hierarchical_search = self._get_neighbors_for_seed(
                seed_origin_id)
            logging.info(
                f"[PAIRING_LOOP] Seed {seed_origin_id}: Found {len(available_hierarchical_neighbors)} raw hierarchical neighbors. Coherence: {coherence_label_from_hierarchical_search}")

            current_coherence_level = coherence_label_from_hierarchical_search

            if len(available_hierarchical_neighbors) >= num_neighbors_needed:
                selected_hierarchical = self._select_final_neighbors(seed_origin_id, available_hierarchical_neighbors,
                                                                     num_neighbors_needed)
                if len(selected_hierarchical) == num_neighbors_needed:
                    selected_neighbors_list.extend(selected_hierarchical)

                    logging.info(
                        f"[PAIRING_LOOP] Seed {seed_origin_id}: Successfully selected {len(selected_hierarchical)} hierarchical neighbors. Coherence: {current_coherence_level}")
                else:
                    selected_neighbors_list.extend(selected_hierarchical)
                    logging.warning(
                        f"[PAIRING_LOOP] Seed {seed_origin_id}: Hierarchical selection yielded {len(selected_hierarchical)}/{num_neighbors_needed}. Will attempt fallback to complete.")
            else:
                logging.warning(
                    f"[PAIRING_LOOP] Seed {seed_origin_id}: Not enough raw hierarchical candidates ({len(available_hierarchical_neighbors)}) to meet need ({num_neighbors_needed}). Will attempt fallback.")

            if len(selected_neighbors_list) < num_neighbors_needed:
                remaining_needed_for_fallback = num_neighbors_needed - len(selected_neighbors_list)
                exclusions_for_fallback = {seed_origin_id}.union(set(selected_neighbors_list))

                logging.info(
                    f"[PAIRING_LOOP] Seed {seed_origin_id}: Attempting fallback for {remaining_needed_for_fallback} neighbors.")
                fallback_candidates_selected = self._get_global_fallback_neighbors(
                    seed_type,
                    seed_origin_level,
                    exclusions_for_fallback,
                    remaining_needed_for_fallback
                )

                if fallback_candidates_selected:
                    selected_neighbors_list.extend(fallback_candidates_selected)

                    if current_coherence_level.startswith(
                            "hierarchical_no_neighbors") or not available_hierarchical_neighbors:
                        current_coherence_level = "global_fallback_only"
                    else:
                        current_coherence_level = f"{coherence_label_from_hierarchical_search}_then_fallback"
                    logging.info(
                        f"[PAIRING_LOOP] Seed {seed_origin_id}: Fallback search added {len(fallback_candidates_selected)} neighbors. Total selected: {len(selected_neighbors_list)}. New Coherence: {current_coherence_level}")
                else:
                    logging.warning(
                        f"[PAIRING_LOOP] Seed {seed_origin_id}: Fallback search found no additional neighbors.")

            if len(selected_neighbors_list) < num_neighbors_needed:
                logging.warning(
                    f"[PAIRING_LOOP] Seed {seed_origin_id}: Even after all attempts, only {len(selected_neighbors_list)} selected neighbors (needed {num_neighbors_needed}). Calling _handle_insufficient_neighbors.")
                self._handle_insufficient_neighbors(seed_origin_id)
                continue

            final_batch_origin_ids_list = [seed_origin_id] + selected_neighbors_list
            final_batch_origin_ids_tuple = tuple(sorted(final_batch_origin_ids_list))

            logging.info(
                f"[PAIRING_LOOP] Seed {seed_origin_id}: Formed batch (Coherence: {current_coherence_level}): {final_batch_origin_ids_tuple}")

            paired_sets_for_difficulty.append({
                "origin_ids": final_batch_origin_ids_tuple,
                "coherence_level": current_coherence_level,
                "seed_id_for_batch": seed_origin_id
            })
            self._update_counts_and_pending_status(final_batch_origin_ids_tuple)

        if iteration_count >= max_iterations and self.pending_origins:
            logging.error(
                f"[PAIRING_LOOP] Exceeded max iterations ({max_iterations}) with {len(self.pending_origins)} origins still pending. This may indicate a persistent issue in pairing logic.")

        logging.info(
            f"Generated {len(paired_sets_for_difficulty)} origin sets (dictionaries) for difficulty evaluation after {iteration_count} loop iterations.")
        if self.pending_origins:
            pending_details = {oid: self.evaluation_counts[oid] for oid in self.pending_origins}
            logging.warning(f"{len(self.pending_origins)} origins did not meet the minimum evaluation count "
                            f"and remain in pending_origins. Details (origin_id: eval_count): {pending_details}")

        return paired_sets_for_difficulty
