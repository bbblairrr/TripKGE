from __future__ import annotations

from collections import defaultdict

from .types import EvidenceSummary, QuerySpec, RetrievalItem


class EvidenceSummarizer:
    def summarize(self, query: QuerySpec, ranked_items: list[RetrievalItem]) -> EvidenceSummary:
        by_type: dict[str, list[RetrievalItem]] = defaultdict(list)
        for item in ranked_items:
            by_type[item.candidate.entity_type].append(item)

        chosen: list[RetrievalItem] = []
        chosen.extend(by_type.get("hotel", [])[:1])
        chosen.extend(by_type.get("attraction", [])[: max(4, query.day * 2)])
        chosen.extend(by_type.get("restaurant", [])[: max(3, query.day * 2)])

        # Fill with overall top if type quotas are sparse.
        chosen_ids = {x.candidate.candidate_id for x in chosen}
        for item in ranked_items:
            if len(chosen) >= min(len(ranked_items), query.day * 6 + 2):
                break
            if item.candidate.candidate_id in chosen_ids:
                continue
            chosen.append(item)
            chosen_ids.add(item.candidate.candidate_id)

        reasons: dict[str, str] = {}
        trace_paths: dict[str, list[str]] = {}
        for item in chosen:
            cid = item.candidate.candidate_id
            reasons[cid] = (
                f"vector={item.vector_score:.3f}, constraint={item.constraint_score:.3f}, "
                f"graph={item.graph_score:.3f}, fused={item.fused_score:.3f}"
            )
            trace_paths[cid] = item.path_evidence

        budget_risk = self._estimate_budget_risk(query, chosen)
        day_suggestions = self._build_day_suggestions(query, chosen)

        return EvidenceSummary(
            query_pid=query.pid,
            chosen_ids=[x.candidate.candidate_id for x in chosen],
            reasons=reasons,
            budget_risk=budget_risk,
            day_suggestions=day_suggestions,
            trace_paths=trace_paths,
        )

    def _estimate_budget_risk(self, query: QuerySpec, chosen: list[RetrievalItem]) -> str:
        if query.budget is None:
            return "unknown"

        hotel = [x for x in chosen if x.candidate.entity_type == "hotel"]
        attrs = [x for x in chosen if x.candidate.entity_type == "attraction"]
        rests = [x for x in chosen if x.candidate.entity_type == "restaurant"]

        estimated = 0.0
        if hotel:
            estimated += hotel[0].candidate.price * query.day
        estimated += sum(x.candidate.price for x in attrs[: query.day * 2])
        estimated += sum(x.candidate.price for x in rests[: query.day * 2])

        ratio = estimated / max(1.0, query.budget)
        if ratio <= 0.8:
            return "low"
        if ratio <= 1.0:
            return "medium"
        return "high"

    def _build_day_suggestions(self, query: QuerySpec, chosen: list[RetrievalItem]) -> dict[int, list[str]]:
        attrs = [x.candidate.candidate_id for x in chosen if x.candidate.entity_type == "attraction"]
        rests = [x.candidate.candidate_id for x in chosen if x.candidate.entity_type == "restaurant"]

        day_suggestions: dict[int, list[str]] = {}
        for day in range(1, query.day + 1):
            picks: list[str] = []
            if attrs:
                picks.append(attrs[(day - 1) % len(attrs)])
            if day - 1 + query.day < len(attrs):
                picks.append(attrs[day - 1 + query.day])
            elif attrs:
                picks.append(attrs[(day + 1) % len(attrs)])
            if rests:
                picks.append(rests[(day - 1) % len(rests)])
            if len(rests) > 1:
                picks.append(rests[(day) % len(rests)])
            day_suggestions[day] = picks

        return day_suggestions
