from __future__ import annotations

from collections import defaultdict

from .config import AblationConfig, RetrievalWeights
from .graph import MultiLayerGraph
from .types import Candidate, QuerySpec, RetrievalItem
from .utils import normalize_text, slugify
from .vector_index import TFIDFIndex


class GraphEnhancedRetriever:
    def __init__(self, graph: MultiLayerGraph, vector_index: TFIDFIndex) -> None:
        self.graph = graph
        self.vector_index = vector_index

    def retrieve(
        self,
        query: QuerySpec,
        candidate_pool: list[Candidate],
        ablation: AblationConfig,
        weights: RetrievalWeights,
    ) -> list[RetrievalItem]:
        if not candidate_pool:
            return []

        filtered = self._hard_filter(query, candidate_pool)
        if not filtered:
            filtered = [c for c in candidate_pool if c.city == query.destination_city] or candidate_pool

        filtered_ids = {c.candidate_id for c in filtered}
        vector_scores = self.vector_index.search(query.query_text, allowed_ids=filtered_ids) if ablation.use_vector else {}

        seeds = self._pick_seed_nodes(filtered, vector_scores, ablation.topk_vector)
        expanded = set(filtered_ids)
        if ablation.use_graph_expansion and seeds:
            expanded = self.graph.bfs_expand(seeds, hops=max(1, ablation.hops), candidate_only=True)
            expanded = expanded.intersection(filtered_ids) or set(filtered_ids)

        if ablation.use_community_retrieval and expanded:
            expanded = self._community_prune(expanded, seeds) or expanded

        retrieval_items: list[RetrievalItem] = []
        for c in filtered:
            if c.candidate_id not in expanded:
                continue
            v = vector_scores.get(c.candidate_id, 0.0)
            cs = self._constraint_score(query, c)
            gs = self._graph_score(c.candidate_id, seeds)
            retrieval_items.append(
                RetrievalItem(candidate=c, vector_score=v, constraint_score=cs, graph_score=gs)
            )

        self._normalize_and_fuse(retrieval_items, weights)
        retrieval_items.sort(key=lambda x: x.fused_score, reverse=True)

        # attach path evidence for top candidates
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

        return retrieval_items[: ablation.topk_final]

    def _hard_filter(self, query: QuerySpec, candidate_pool: list[Candidate]) -> list[Candidate]:
        filtered: list[Candidate] = []
        daily_budget = (query.budget / max(1, query.day)) if query.budget else None

        for c in candidate_pool:
            if c.city != query.destination_city:
                continue

            if query.meal_price_range and c.entity_type == "restaurant":
                lo, hi = query.meal_price_range
                if not (lo <= c.price <= hi):
                    continue

            if query.hotel_category_pref and c.entity_type == "hotel":
                category = normalize_text(str(c.meta.get("category") or ""))
                if normalize_text(query.hotel_category_pref) not in category:
                    continue

            if daily_budget is not None:
                if c.entity_type == "hotel" and c.price > daily_budget * 0.85:
                    continue
                if c.entity_type == "restaurant" and c.price > daily_budget * 0.35:
                    continue
                if c.entity_type == "attraction" and c.price > daily_budget * 0.65:
                    continue

            filtered.append(c)

        # prevent over-pruning when constraints are too strict.
        if len(filtered) < 6:
            filtered = [c for c in candidate_pool if c.city == query.destination_city]
        return filtered

    def _pick_seed_nodes(
        self,
        candidates: list[Candidate],
        vector_scores: dict[str, float],
        topk_vector: int,
    ) -> set[str]:
        if vector_scores:
            ranked = sorted(vector_scores.items(), key=lambda x: x[1], reverse=True)
            return {doc_id for doc_id, _ in ranked[:topk_vector]}

        # fallback seeds: restaurants + attractions by price utility
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

    def _constraint_score(self, query: QuerySpec, candidate: Candidate) -> float:
        score = 0.6

        if query.meal_price_range and candidate.entity_type == "restaurant":
            lo, hi = query.meal_price_range
            mid = (lo + hi) / 2.0
            span = max(1.0, hi - lo)
            dist = abs(candidate.price - mid)
            score += max(0.0, 0.25 * (1 - dist / span))

        if query.hotel_category_pref and candidate.entity_type == "hotel":
            if normalize_text(query.hotel_category_pref) in normalize_text(str(candidate.meta.get("category") or "")):
                score += 0.25

        if query.interest_tags and candidate.tags:
            cand_tags = {normalize_text(t) for t in candidate.tags}
            query_tags = {normalize_text(t) for t in query.interest_tags}
            inter = len(cand_tags.intersection(query_tags))
            if inter:
                score += min(0.3, 0.1 * inter)

        return min(score, 1.2)

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

        vnorm = norm([x.vector_score for x in items])
        cnorm = norm([x.constraint_score for x in items])
        gnorm = norm([x.graph_score for x in items])

        for idx, item in enumerate(items):
            item.vector_score = vnorm[idx]
            item.constraint_score = cnorm[idx]
            item.graph_score = gnorm[idx]
            item.fused_score = (
                weights.vector * item.vector_score
                + weights.constraint * item.constraint_score
                + weights.graph * item.graph_score
            )
