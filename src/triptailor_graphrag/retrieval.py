from __future__ import annotations

from collections import defaultdict

from .config import AblationConfig, RetrievalWeights
from .graph import MultiLayerGraph
from .types import Candidate, QuerySpec, RetrievalItem, RetrievalTrace
from .utils import haversine_km, normalize_text, parse_duration_minutes, parse_opening_hours, slugify, tokenize
from .vector_index import CandidateVectorIndex

MIN_FILTERED_CANDIDATES = 6
RERANK_LAMBDA = 0.8


class GraphEnhancedRetriever:
    def __init__(self, graph: MultiLayerGraph, vector_index: CandidateVectorIndex) -> None:
        self.graph = graph
        self.vector_index = vector_index

    def retrieve(
        self,
        query: QuerySpec,
        candidate_pool: list[Candidate],
        ablation: AblationConfig,
        weights: RetrievalWeights,
    ) -> tuple[list[RetrievalItem], RetrievalTrace]:
        trace = RetrievalTrace(initial_candidate_count=len(candidate_pool))
        if not candidate_pool:
            return [], trace

        city_candidates = [c for c in candidate_pool if normalize_text(c.city) == normalize_text(query.destination_city)]
        if not city_candidates:
            city_candidates = list(candidate_pool)
            trace.notes.append("destination city missing in candidate pool; used full pid candidate pool.")
            trace.vector_scope = "pid_pool"
        else:
            trace.vector_scope = f"city:{slugify(query.destination_city)}"
        trace.city_candidate_count = len(city_candidates)

        filtered = self._hard_filter(query, city_candidates, trace)
        if not filtered:
            filtered = city_candidates
            trace.notes.append("hard filter removed everything; fell back to city candidate pool.")
        trace.filtered_candidate_count = len(filtered)

        filtered_ids = {c.candidate_id for c in filtered}
        vector_scores = (
            self.vector_index.search(
                query.query_text,
                allowed_ids=filtered_ids,
                city=query.destination_city,
            )
            if ablation.use_vector
            else {}
        )

        seeds = self._pick_seed_nodes(filtered, vector_scores, ablation.topk_vector)
        trace.seed_count = len(seeds)
        expanded = set(filtered_ids)
        if ablation.use_graph_expansion and seeds:
            expanded = self.graph.bfs_expand(seeds, hops=max(1, ablation.hops), candidate_only=True)
            expanded = expanded.intersection(filtered_ids) or set(filtered_ids)

        if ablation.use_community_retrieval and expanded:
            expanded = self._community_prune(expanded, seeds) or expanded

        route_anchor = self._route_anchor(filtered, vector_scores)
        retrieval_items: list[RetrievalItem] = []
        for c in filtered:
            if c.candidate_id not in expanded:
                continue
            v = vector_scores.get(c.candidate_id, 0.0)
            cs, notes = self._constraint_score(query, c, route_anchor)
            gs = self._graph_score(c.candidate_id, seeds)
            retrieval_items.append(
                RetrievalItem(
                    candidate=c,
                    vector_score=v,
                    constraint_score=cs,
                    graph_score=gs,
                    raw_vector_score=v,
                    raw_constraint_score=cs,
                    raw_graph_score=gs,
                    constraint_notes=notes,
                )
            )

        self._normalize_and_fuse(retrieval_items, weights)
        retrieval_items = self._diversity_rerank(retrieval_items)

        city_node = f"city:{slugify(query.destination_city)}"
        pref_nodes = [f"pref:tag:{slugify(tag)}" for tag in query.interest_tags[:3]]
        for item in retrieval_items[: ablation.topk_final]:
            paths: list[str] = []
            city_path = self.graph.shortest_path(item.candidate.candidate_id, city_node, max_hops=3)
            if city_path:
                paths.append(" -> ".join(city_path))
            for pref_node in pref_nodes:
                pref_path = self.graph.shortest_path(item.candidate.candidate_id, pref_node, max_hops=3)
                if pref_path:
                    paths.append(" -> ".join(pref_path))
            item.path_evidence = paths[:3]

        return retrieval_items[: ablation.topk_final], trace

    def _hard_filter(
        self,
        query: QuerySpec,
        candidate_pool: list[Candidate],
        trace: RetrievalTrace,
    ) -> list[Candidate]:
        strict: list[Candidate] = []
        budget_relaxed: list[Candidate] = []
        hard_relaxed: list[Candidate] = []

        for candidate in candidate_pool:
            hard_ok = self._passes_user_hard_constraints(query, candidate)
            budget_ok = self._within_budget_hint(query, candidate)
            if hard_ok and budget_ok:
                strict.append(candidate)
            elif hard_ok:
                budget_relaxed.append(candidate)
            else:
                hard_relaxed.append(candidate)

        trace.strict_candidate_count = len(strict)
        trace.budget_relaxed_candidate_count = len(budget_relaxed)
        trace.hard_relaxed_candidate_count = len(hard_relaxed)

        filtered = list(strict)
        if len(filtered) < MIN_FILTERED_CANDIDATES and budget_relaxed:
            filtered.extend(budget_relaxed)
            trace.notes.append("relaxed budget caps to keep enough candidates for retrieval.")
        if len(filtered) < MIN_FILTERED_CANDIDATES and hard_relaxed:
            filtered.extend(hard_relaxed)
            trace.notes.append("relaxed meal/hotel hard constraints because candidate pool remained too small.")

        if not filtered:
            return []
        return self._dedupe_candidates(filtered)

    def _passes_user_hard_constraints(self, query: QuerySpec, candidate: Candidate) -> bool:
        if query.meal_price_range and candidate.entity_type == "restaurant":
            lo, hi = query.meal_price_range
            if not (lo <= candidate.price <= hi):
                return False

        if query.hotel_category_pref and candidate.entity_type == "hotel":
            category = normalize_text(str(candidate.meta.get("category") or ""))
            if normalize_text(query.hotel_category_pref) not in category:
                return False

        return True

    def _within_budget_hint(self, query: QuerySpec, candidate: Candidate) -> bool:
        daily_budget = (query.budget / max(1, query.day)) if query.budget else None
        if daily_budget is None:
            return True
        if candidate.entity_type == "hotel":
            return candidate.price <= daily_budget * 0.85
        if candidate.entity_type == "restaurant":
            return candidate.price <= daily_budget * 0.35
        if candidate.entity_type == "attraction":
            return candidate.price <= daily_budget * 0.65
        return True

    def _dedupe_candidates(self, candidates: list[Candidate]) -> list[Candidate]:
        out: list[Candidate] = []
        seen: set[str] = set()
        for candidate in candidates:
            if candidate.candidate_id in seen:
                continue
            seen.add(candidate.candidate_id)
            out.append(candidate)
        return out

    def _pick_seed_nodes(
        self,
        candidates: list[Candidate],
        vector_scores: dict[str, float],
        topk_vector: int,
    ) -> set[str]:
        if vector_scores:
            ranked = sorted(vector_scores.items(), key=lambda x: x[1], reverse=True)
            return {doc_id for doc_id, _ in ranked[:topk_vector]}

        fallback = sorted(candidates, key=lambda c: (c.entity_type == "hotel", c.price))
        return {c.candidate_id for c in fallback[: min(8, len(fallback))]}

    def _community_prune(self, expanded: set[str], seeds: set[str]) -> set[str]:
        comps = self.graph.connected_components(expanded)
        if not comps:
            return expanded

        scored = []
        for comp in comps:
            overlap = len(comp.intersection(seeds))
            scored.append((overlap, len(comp), comp))
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)

        best_overlap = scored[0][0]
        kept: set[str] = set()
        for overlap, _, comp in scored:
            if overlap == 0 and kept:
                break
            if overlap < best_overlap and kept:
                break
            kept.update(comp)
        return kept or expanded

    def _route_anchor(
        self,
        candidates: list[Candidate],
        vector_scores: dict[str, float],
    ) -> tuple[float, float] | None:
        ranked = sorted(
            candidates,
            key=lambda c: (
                vector_scores.get(c.candidate_id, 0.0),
                c.entity_type == "attraction",
                -c.price,
            ),
            reverse=True,
        )
        for candidate in ranked:
            if candidate.latitude is None or candidate.longitude is None:
                continue
            return candidate.latitude, candidate.longitude
        return None

    def _constraint_score(
        self,
        query: QuerySpec,
        candidate: Candidate,
        route_anchor: tuple[float, float] | None,
    ) -> tuple[float, list[str]]:
        score = 0.3
        notes: list[str] = []

        if query.meal_price_range and candidate.entity_type == "restaurant":
            lo, hi = query.meal_price_range
            mid = (lo + hi) / 2.0
            span = max(1.0, hi - lo)
            dist = abs(candidate.price - mid)
            fit = max(0.0, 1 - dist / span)
            score += 0.28 * fit
            notes.append(f"meal_fit={fit:.2f}")
        elif candidate.entity_type == "restaurant":
            score += 0.12

        if query.hotel_category_pref and candidate.entity_type == "hotel":
            category = normalize_text(str(candidate.meta.get("category") or ""))
            matched = 1.0 if normalize_text(query.hotel_category_pref) in category else 0.0
            score += 0.2 * matched
            notes.append(f"hotel_pref_match={matched:.0f}")

        if query.interest_tags and candidate.tags:
            cand_tags = {normalize_text(t) for t in candidate.tags}
            query_tags = {normalize_text(t) for t in query.interest_tags}
            inter = len(cand_tags.intersection(query_tags))
            if inter:
                score += min(0.22, 0.08 * inter)
            notes.append(f"interest_overlap={inter}")

        if self._within_budget_hint(query, candidate):
            score += 0.1
            notes.append("budget_hint=ok")
        else:
            notes.append("budget_hint=relaxed")

        if candidate.entity_type == "attraction":
            opening = parse_opening_hours(str(candidate.meta.get("opening_hours") or ""))
            duration = parse_duration_minutes(str(candidate.meta.get("recommended_duration") or ""))
            if opening is not None:
                score += 0.05
                notes.append("opening_hours=known")
            if duration is not None:
                score += 0.05
                notes.append(f"duration_min={duration}")

        if candidate.latitude is not None and candidate.longitude is not None:
            score += 0.04
            notes.append("geo=known")
            if route_anchor is not None:
                distance = haversine_km(route_anchor[0], route_anchor[1], candidate.latitude, candidate.longitude)
                geo_fit = max(0.0, 1 - distance / 20.0)
                score += 0.12 * geo_fit
                notes.append(f"route_anchor_km={distance:.2f}")

        return min(score, 1.2), notes

    def _graph_score(self, candidate_id: str, seeds: set[str]) -> float:
        neighbors = self.graph.neighbors(candidate_id)
        if not neighbors:
            return 0.0

        seed_links = len([n for n in neighbors if n in seeds])
        type_links = len([n for n in neighbors if n.startswith("etype:")])
        pref_links = len([n for n in neighbors if n.startswith("pref:")])
        norm = max(1.0, len(neighbors))
        score = (seed_links * 1.5 + type_links + pref_links * 0.7) / norm
        if candidate_id in seeds:
            score += 0.35
        return min(score, 1.5)

    def _normalize_and_fuse(self, items: list[RetrievalItem], weights: RetrievalWeights) -> None:
        if not items:
            return

        def norm(values: list[float]) -> list[float]:
            lo = min(values)
            hi = max(values)
            if hi - lo < 1e-9:
                return [0.0 for _ in values]
            return [(v - lo) / (hi - lo) for v in values]

        raw_vector = [x.raw_vector_score for x in items]
        raw_constraint = [x.raw_constraint_score for x in items]
        raw_graph = [x.raw_graph_score for x in items]
        vnorm = norm(raw_vector)
        cnorm = norm(raw_constraint)
        gnorm = norm(raw_graph)

        for idx, item in enumerate(items):
            item.vector_score = vnorm[idx]
            item.constraint_score = cnorm[idx]
            item.graph_score = gnorm[idx]
            item.raw_fused_score = (
                weights.vector * item.raw_vector_score
                + weights.constraint * item.raw_constraint_score
                + weights.graph * item.raw_graph_score
            )
            item.fused_score = (
                weights.vector * item.vector_score
                + weights.constraint * item.constraint_score
                + weights.graph * item.graph_score
            )

    def _diversity_rerank(self, items: list[RetrievalItem]) -> list[RetrievalItem]:
        if len(items) <= 1:
            for item in items:
                item.rerank_score = item.fused_score
            return items

        remaining = sorted(items, key=lambda item: item.fused_score, reverse=True)
        selected: list[RetrievalItem] = []
        type_counts: defaultdict[str, int] = defaultdict(int)

        while remaining:
            if not selected:
                chosen = remaining.pop(0)
                chosen.diversity_penalty = 0.0
                chosen.rerank_score = chosen.fused_score
                selected.append(chosen)
                type_counts[chosen.candidate.entity_type] += 1
                continue

            best_idx = 0
            best_score = float("-inf")
            best_penalty = 0.0
            for idx, item in enumerate(remaining):
                similarity = max(
                    self._candidate_similarity(item.candidate, existing.candidate)
                    for existing in selected
                )
                type_penalty = 0.12 if item.candidate.entity_type == "hotel" and type_counts["hotel"] >= 1 else 0.0
                score = RERANK_LAMBDA * item.fused_score - (1 - RERANK_LAMBDA) * similarity - type_penalty
                if score > best_score:
                    best_idx = idx
                    best_score = score
                    best_penalty = similarity + type_penalty

            chosen = remaining.pop(best_idx)
            chosen.diversity_penalty = best_penalty
            chosen.rerank_score = best_score
            selected.append(chosen)
            type_counts[chosen.candidate.entity_type] += 1

        return selected

    def _candidate_similarity(self, left: Candidate, right: Candidate) -> float:
        if left.candidate_id == right.candidate_id:
            return 1.0

        left_tokens = set(tokenize(left.text))
        right_tokens = set(tokenize(right.text))
        text_sim = self._jaccard(left_tokens, right_tokens)

        left_tags = {normalize_text(tag) for tag in left.tags}
        right_tags = {normalize_text(tag) for tag in right.tags}
        tag_sim = self._jaccard(left_tags, right_tags)

        type_sim = 1.0 if left.entity_type == right.entity_type else 0.0
        name_sim = 1.0 if normalize_text(left.name) == normalize_text(right.name) else 0.0
        return min(1.0, 0.45 * text_sim + 0.25 * tag_sim + 0.2 * type_sim + 0.1 * name_sim)

    def _jaccard(self, left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        inter = len(left.intersection(right))
        union = len(left.union(right))
        return inter / union if union > 0 else 0.0
