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

        self.all_knowledge_origins_map: Dict[str, Dict[str, Any]] = \
            {str(o['origin_id']): o for o in all_knowledge_origins}

        if not self.all_knowledge_origins_map:
            raise ValueError("Falha ao construir all_knowledge_origins_map.")

        self.MIN_EVALUATIONS_PER_ORIGIN = min_evaluations_per_origin
        self.DIFFICULTY_BATCH_SIZE = difficulty_batch_size
        self.MAX_ASCENT_LEVELS = max_ascent_levels
        self.NEIGHBOR_POOL_MULTIPLIER = DEFAULT_NEIGHBOR_POOL_MULTIPLIER

        self._initialize_internal_structures()

    def _initialize_internal_structures(self):
        """Prepara mapas e contadores internos."""

        self.origin_to_parent_map: Dict[str, Optional[str]] = {
            oid: str(o_data['parent_community_id_of_origin']) if o_data.get('parent_community_id_of_origin') else None
            for oid, o_data in self.all_knowledge_origins_map.items()
        }

        self.context_to_origins_map: Dict[str, Dict[Tuple[str, int], List[str]]] = \
            defaultdict(lambda: defaultdict(list))

        for oid, o_data in self.all_knowledge_origins_map.items():
            context_id = self.origin_to_parent_map.get(oid) or "__ROOT__"
            o_type = o_data.get('origin_type')
            o_level = o_data.get('level', 0)

            if not isinstance(o_type, str) or not (
                    isinstance(o_level, int) or (isinstance(o_level, float) and o_level.is_integer())):
                logging.warning(
                    f"Origem {oid} com tipo/nível inválido: {o_type}, {o_level}. Ignorada no mapeamento context_to_origins_map.")
                continue
            self.context_to_origins_map[context_id][(o_type, int(o_level))].append(oid)

        self.evaluation_counts: Counter = Counter()
        self.pending_eval_for_origin: Set[str] = set(self.all_knowledge_origins_map.keys())
        logging.info(
            f"OriginDifficultyScheduler inicializado com {len(self.pending_eval_for_origin)} origens necessitando de avaliação.")

    def _get_seed_origin(self) -> Optional[str]:
        """Seleciona a próxima origem semente, priorizando as menos avaliadas que ainda estão em pending_eval_for_origin."""
        if not self.pending_eval_for_origin:
            return None

        candidates_for_seed = [oid for oid in self.pending_eval_for_origin if oid in self.all_knowledge_origins_map]
        if not candidates_for_seed:
            logging.warning("_get_seed_origin: No valid pending origins found in all_knowledge_origins_map.")
            return None

        sorted_pending_seeds = sorted(
            candidates_for_seed,
            key=lambda oid: (self.evaluation_counts[oid], oid)
        )
        return sorted_pending_seeds[0]

    def _get_hierarchical_neighbor_candidates(self, seed_origin_id: str) -> Tuple[List[str], str]:
        """
        Encontra TODOS os vizinhos hierárquicos elegíveis (mesmo tipo/nível de origem da semente).
        Não filtra por contagem de avaliação aqui, apenas coleta os candidatos.
        Retorna (lista_de_vizinhos_candidatos, string_coerencia_hierarquica).
        """
        seed_details = self.all_knowledge_origins_map.get(seed_origin_id)
        if not seed_details:
            logging.warning(f"[HIERARCHICAL_CANDIDATES] Detalhes não encontrados para semente {seed_origin_id}")
            return [], "hierarchical_error_no_seed_details"

        seed_type = str(seed_details.get('origin_type'))
        seed_origin_level = int(seed_details.get('level', 0))
        logging.info(
            f"[HIERARCHICAL_CANDIDATES] Seed: {seed_origin_id} (Type: {seed_type}, OriginLevel: {seed_origin_level})")

        collected_candidate_neighbors: Set[str] = set()
        current_search_context_id = self.origin_to_parent_map.get(seed_origin_id) or "__ROOT__"
        final_coherence_label = "hierarchical_no_neighbors_found"

        max_needed_for_pool = (self.DIFFICULTY_BATCH_SIZE - 1) * self.NEIGHBOR_POOL_MULTIPLIER

        for ascent_count in range(self.MAX_ASCENT_LEVELS + 1):
            if not current_search_context_id: break
            logging.debug(
                f"[HIERARCHICAL_CANDIDATES] Seed {seed_origin_id}: Ascent {ascent_count}, context '{current_search_context_id}', type '{seed_type}', level '{seed_origin_level}'")

            origins_in_context_for_type_level = self.context_to_origins_map.get(current_search_context_id, {}).get(
                (seed_type, seed_origin_level), []
            )

            found_in_this_ascent = False
            for neighbor_id in origins_in_context_for_type_level:
                if neighbor_id != seed_origin_id:
                    if neighbor_id not in collected_candidate_neighbors:
                        collected_candidate_neighbors.add(neighbor_id)
                        found_in_this_ascent = True
                        logging.debug(
                            f"[HIERARCHICAL_CANDIDATES] Seed {seed_origin_id}: Added candidate {neighbor_id} from context {current_search_context_id}.")

            if found_in_this_ascent:
                final_coherence_label = f"hierarchical_ascent_{ascent_count}"
                if current_search_context_id == "__ROOT__":
                    final_coherence_label += "_root"

            if len(collected_candidate_neighbors) >= max_needed_for_pool:
                logging.info(
                    f"[HIERARCHICAL_CANDIDATES] Seed {seed_origin_id}: Candidate pool full ({len(collected_candidate_neighbors)}) at ascent {ascent_count}.")
                break
            if ascent_count == self.MAX_ASCENT_LEVELS or current_search_context_id == "__ROOT__":
                break

            parent_of_current_context = self.origin_to_parent_map.get(current_search_context_id)
            if not parent_of_current_context and current_search_context_id != "__ROOT__": break
            current_search_context_id = parent_of_current_context or "__ROOT__"

        logging.info(
            f"[HIERARCHICAL_CANDIDATES] Seed {seed_origin_id}: Found {len(collected_candidate_neighbors)} hierarchical candidates. Coherence: {final_coherence_label}")
        return list(collected_candidate_neighbors), final_coherence_label

    def _get_global_fallback_candidates(self, seed_origin_type: str, seed_origin_level: int, exclusions: Set[str]) -> \
    List[str]:
        """
        Retorna TODOS os candidatos globais (do mesmo tipo/nível de origem), excluindo os da lista de 'exclusions'.
        Não filtra por contagem de avaliação aqui.
        """
        logging.info(
            f"[FALLBACK_CANDIDATES] Seeking ALL global candidates of type '{seed_origin_type}', origin_level '{seed_origin_level}', excluding {len(exclusions)} IDs.")

        global_candidates: List[str] = []
        for oid, details in self.all_knowledge_origins_map.items():
            if oid in exclusions:
                continue
            if details.get('origin_type') == seed_origin_type and int(details.get('level', 0)) == seed_origin_level:
                global_candidates.append(oid)

        logging.info(f"[FALLBACK_CANDIDATES] Found {len(global_candidates)} global candidates.")
        return global_candidates

    def _select_final_neighbors_from_candidates(self, seed_origin_id: str, candidate_neighbors: List[str],
                                                num_needed: int) -> List[str]:
        """
        Seleciona os vizinhos finais da lista de CANDIDATOS fornecida.
        Prioriza aqueles com menor contagem de avaliação.
        """
        logging.debug(
            f"[SELECT_FINAL] Seed {seed_origin_id}: Selecting {num_needed} from {len(candidate_neighbors)} candidates.")
        if not candidate_neighbors or num_needed <= 0:
            return []

        shuffled_candidates = list(candidate_neighbors)
        random.shuffle(shuffled_candidates)

        sorted_candidates = sorted(
            shuffled_candidates,
            key=lambda oid: (self.evaluation_counts[oid], oid)
        )

        selected = sorted_candidates[:num_needed]
        logging.info(
            f"[SELECT_FINAL] Seed {seed_origin_id}: Selected {len(selected)} final neighbors: {selected} (Counts: {{oid: self.evaluation_counts[oid] for oid in selected}})")
        return selected

    def _handle_seed_cannot_form_batch(self, seed_origin_id: str):
        """
        Chamado quando uma semente não consegue formar um batch completo mesmo após todas as tentativas.
        Incrementa sua contagem de avaliação (como uma "tentativa de ser semente").
        Se atingir o mínimo, é removida de pending_eval_for_origin.
        """
        current_eval_count = self.evaluation_counts[seed_origin_id]
        logging.warning(
            f"[CANNOT_FORM_BATCH] Seed {seed_origin_id} (evals: {current_eval_count}) could not form a full batch. "
            f"Incrementing its eval_count as a 'seed attempt'. "
            f"Will be removed from pending if new count reaches {self.MIN_EVALUATIONS_PER_ORIGIN}."
        )
        self.evaluation_counts[seed_origin_id] += 1
        if self.evaluation_counts[seed_origin_id] >= self.MIN_EVALUATIONS_PER_ORIGIN:
            if seed_origin_id in self.pending_eval_for_origin:
                self.pending_eval_for_origin.discard(seed_origin_id)
                logging.info(
                    f"[CANNOT_FORM_BATCH] Seed {seed_origin_id} reached {self.evaluation_counts[seed_origin_id]} total eval/attempts, removed from pending_eval_for_origin.")

    def _update_eval_counts_and_pending_status(self, batch_origin_ids: Tuple[str, ...]):
        """
        Atualiza contadores de avaliação para todos os participantes do batch.
        Remove de pending_eval_for_origin se atingirem o mínimo.
        """
        for oid in batch_origin_ids:
            self.evaluation_counts[oid] += 1
            if self.evaluation_counts[oid] >= self.MIN_EVALUATIONS_PER_ORIGIN:
                if oid in self.pending_eval_for_origin:
                    self.pending_eval_for_origin.discard(oid)
                    logging.info(
                        f"[UPDATE_EVALS] Origin {oid} participated in batch, reached {self.evaluation_counts[oid]} evals, removed from pending_eval_for_origin.")

    def generate_origin_pairings(self) -> List[Dict[str, Any]]:
        paired_sets_for_difficulty: List[Dict[str, Any]] = []
        iteration_count = 0
        max_iterations = len(
            self.all_knowledge_origins_map) * self.MIN_EVALUATIONS_PER_ORIGIN * self.DIFFICULTY_BATCH_SIZE
        if max_iterations == 0 and self.all_knowledge_origins_map: max_iterations = len(
            self.all_knowledge_origins_map) * 5

        while self.pending_eval_for_origin and iteration_count < max_iterations:
            iteration_count += 1
            logging.info(
                f"[PAIRING_MAIN_LOOP] Iteration {iteration_count}/{max_iterations}, {len(self.pending_eval_for_origin)} origins in pending_eval_for_origin.")

            seed_origin_id = self._get_seed_origin()
            if not seed_origin_id:
                logging.info("[PAIRING_MAIN_LOOP] No more seed origins available. Breaking.")
                break

            seed_details = self.all_knowledge_origins_map.get(
                seed_origin_id)
            if not seed_details:
                logging.error(
                    f"[PAIRING_MAIN_LOOP] Critical: Seed {seed_origin_id} from _get_seed_origin has no details. Discarding.")
                self.pending_eval_for_origin.discard(seed_origin_id)
                continue

            seed_type = str(seed_details.get('origin_type'))
            seed_origin_level = int(seed_details.get('level', 0))
            num_neighbors_needed = self.DIFFICULTY_BATCH_SIZE - 1

            logging.info(
                f"[PAIRING_MAIN_LOOP] Seed: {seed_origin_id} (Type: {seed_type}, OriginLevel: {seed_origin_level}, Evals: {self.evaluation_counts[seed_origin_id]}). Needs {num_neighbors_needed} neighbors.")

            if num_neighbors_needed < 0:
                logging.error(
                    f"[PAIRING_MAIN_LOOP] num_neighbors_needed is < 0. Batch size: {self.DIFFICULTY_BATCH_SIZE}. Handling seed {seed_origin_id} as unable to form batch.")
                self._handle_seed_cannot_form_batch(seed_origin_id)
                continue

            selected_neighbors_final: List[str] = []
            determined_coherence_level = "unknown"

            # 1. Tentar vizinhos hierárquicos
            hierarchical_candidates, coherence_hierarchical = self._get_hierarchical_neighbor_candidates(seed_origin_id)
            determined_coherence_level = coherence_hierarchical  # Coerência base da busca hierárquica

            if hierarchical_candidates:
                # Tenta selecionar o suficiente apenas dos hierárquicos
                selected_from_hierarchical = self._select_final_neighbors_from_candidates(seed_origin_id,
                                                                                          hierarchical_candidates,
                                                                                          num_neighbors_needed)
                selected_neighbors_final.extend(selected_from_hierarchical)
                logging.info(
                    f"[PAIRING_MAIN_LOOP] Seed {seed_origin_id}: Selected {len(selected_from_hierarchical)} from {len(hierarchical_candidates)} hierarchical candidates.")

            # 2. Se não foi suficiente, tentar fallback global para completar
            if len(selected_neighbors_final) < num_neighbors_needed:
                remaining_needed = num_neighbors_needed - len(selected_neighbors_final)
                exclusions = {seed_origin_id}.union(set(selected_neighbors_final))  # Exclui semente e já selecionados

                logging.info(
                    f"[PAIRING_MAIN_LOOP] Seed {seed_origin_id}: Needs {remaining_needed} more neighbors. Attempting global fallback (excluding {len(exclusions)}).")
                global_fallback_candidates = self._get_global_fallback_candidates(seed_type, seed_origin_level,
                                                                                  exclusions)

                if global_fallback_candidates:
                    selected_from_fallback = self._select_final_neighbors_from_candidates(seed_origin_id,
                                                                                          global_fallback_candidates,
                                                                                          remaining_needed)
                    if selected_from_fallback:
                        selected_neighbors_final.extend(selected_from_fallback)
                        # Atualiza a coerência
                        if not hierarchical_candidates or coherence_hierarchical.endswith("no_neighbors_found"):
                            determined_coherence_level = "global_fallback_only"
                        else:  # Tinha alguns hierárquicos, e complementou com fallback
                            determined_coherence_level = f"{coherence_hierarchical}_then_fallback"
                        logging.info(
                            f"[PAIRING_MAIN_LOOP] Seed {seed_origin_id}: Added {len(selected_from_fallback)} from global fallback. Total selected: {len(selected_neighbors_final)}. Coherence: {determined_coherence_level}")
                else:
                    logging.info(f"[PAIRING_MAIN_LOOP] Seed {seed_origin_id}: No global fallback candidates found.")

            # 3. Verificação final e formação do batch
            if len(selected_neighbors_final) == num_neighbors_needed:
                batch_tuple = tuple(sorted([seed_origin_id] + selected_neighbors_final))
                paired_sets_for_difficulty.append({
                    "origin_ids": batch_tuple,
                    "coherence_level": determined_coherence_level,
                    "seed_id_for_batch": seed_origin_id
                })
                self._update_eval_counts_and_pending_status(
                    batch_tuple)  # Atualiza contagens para TODOS os participantes
                logging.info(
                    f"[PAIRING_MAIN_LOOP] Seed {seed_origin_id}: Successfully formed batch {batch_tuple} with coherence {determined_coherence_level}.")
            else:
                logging.warning(
                    f"[PAIRING_MAIN_LOOP] Seed {seed_origin_id}: Failed to gather enough neighbors ({len(selected_neighbors_final)}/{num_neighbors_needed}).")
                self._handle_seed_cannot_form_batch(seed_origin_id)  # Incrementa contagem da semente

        if iteration_count >= max_iterations and self.pending_eval_for_origin:
            logging.error(
                f"[PAIRING_MAIN_LOOP] Exceeded max iterations ({max_iterations}) with {len(self.pending_eval_for_origin)} origins still pending.")

        logging.info(
            f"Generated {len(paired_sets_for_difficulty)} origin sets for difficulty evaluation after {iteration_count} loop iterations.")
        if self.pending_eval_for_origin:
            pending_details = {oid: self.evaluation_counts[oid] for oid in self.pending_eval_for_origin}
            logging.warning(
                f"{len(self.pending_eval_for_origin)} origins remain in pending_eval_for_origin. Counts: {pending_details}")

        return paired_sets_for_difficulty
