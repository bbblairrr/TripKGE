from __future__ import annotations

import json
from collections import defaultdict

from .local_llm import LocalLLM
from .types import EvidenceSummary, QuerySpec, RetrievalItem


class EvidenceSummarizer:
    def __init__(self, llm: LocalLLM | None = None) -> None:
        self.llm = llm

    def summarize(self, query: QuerySpec, ranked_items: list[RetrievalItem]) -> EvidenceSummary:
        heuristic = self._heuristic_summary(query, ranked_items)
        if self.llm is None or not self.llm.enabled or not self.llm.config.enable_summary:
            return heuristic

        llm_summary = self._summarize_with_llm(query, ranked_items, heuristic)
        return llm_summary or heuristic

    def _heuristic_summary(self, query: QuerySpec, ranked_items: list[RetrievalItem]) -> EvidenceSummary:
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

    def _summarize_with_llm(
        self,
        query: QuerySpec,
        ranked_items: list[RetrievalItem],
        heuristic: EvidenceSummary,
    ) -> EvidenceSummary | None:
        if not ranked_items or self.llm is None:
            return None

        allowed_ids = {item.candidate.candidate_id for item in ranked_items}
        shortlist = [
            {
                "candidate_id": item.candidate.candidate_id,
                "entity_type": item.candidate.entity_type,
                "name": item.candidate.name,
                "city": item.candidate.city,
                "price": item.candidate.price,
                "tags": item.candidate.tags[:4],
                "scores": {
                    "vector": round(item.vector_score, 4),
                    "constraint": round(item.constraint_score, 4),
                    "graph": round(item.graph_score, 4),
                    "fused": round(item.fused_score, 4),
                },
                "path_evidence": item.path_evidence,
            }
            for item in ranked_items[: min(18, len(ranked_items))]
        ]

        system_prompt = (
            "You are an evidence selection module for personalized travel planning. "
            "Return only JSON with keys chosen_ids, reasons, budget_risk, day_suggestions, trace_paths. "
            "Use only candidate_id values from the shortlist."
        )
        user_prompt = (
            "Query:\n"
            + json.dumps(
                {
                    "pid": query.pid,
                    "text": query.query_text,
                    "budget": query.budget,
                    "day": query.day,
                    "meal_price_range": query.meal_price_range,
                    "hotel_category_pref": query.hotel_category_pref,
                    "intensity_pref": query.intensity_pref,
                    "interest_tags": query.interest_tags,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n\nShortlist:\n"
            + json.dumps(shortlist, ensure_ascii=False, indent=2)
            + "\n\nHeuristic baseline:\n"
            + json.dumps(
                {
                    "chosen_ids": heuristic.chosen_ids,
                    "budget_risk": heuristic.budget_risk,
                    "day_suggestions": heuristic.day_suggestions,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n\nOutput schema example:\n"
            + '{"chosen_ids":["id1"],"reasons":{"id1":"why"},"budget_risk":"low","day_suggestions":{"1":["id1"]},"trace_paths":{"id1":["path"]}}'
        )
        payload = self.llm.generate_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_new_tokens=self.llm.config.summary_max_new_tokens,
        )
        if not payload:
            return None

        chosen_ids = [
            candidate_id
            for candidate_id in payload.get("chosen_ids", [])
            if isinstance(candidate_id, str) and candidate_id in allowed_ids
        ]
        if not chosen_ids:
            chosen_ids = heuristic.chosen_ids

        reasons_raw = payload.get("reasons", {})
        reasons = dict(heuristic.reasons)
        if isinstance(reasons_raw, dict):
            for candidate_id, reason in reasons_raw.items():
                if candidate_id in allowed_ids and isinstance(reason, str) and reason.strip():
                    reasons[candidate_id] = reason.strip()

        budget_risk = str(payload.get("budget_risk", heuristic.budget_risk)).strip().lower()
        if budget_risk not in {"low", "medium", "high", "unknown"}:
            budget_risk = heuristic.budget_risk

        day_suggestions = dict(heuristic.day_suggestions)
        suggestions_raw = payload.get("day_suggestions", {})
        if isinstance(suggestions_raw, dict):
            for day_key, values in suggestions_raw.items():
                try:
                    day = int(day_key)
                except (TypeError, ValueError):
                    continue
                if not 1 <= day <= query.day or not isinstance(values, list):
                    continue
                clean_ids = [value for value in values if isinstance(value, str) and value in allowed_ids]
                if clean_ids:
                    day_suggestions[day] = clean_ids

        trace_paths = dict(heuristic.trace_paths)
        trace_raw = payload.get("trace_paths", {})
        if isinstance(trace_raw, dict):
            for candidate_id, values in trace_raw.items():
                if candidate_id not in allowed_ids or not isinstance(values, list):
                    continue
                trace_paths[candidate_id] = [str(value) for value in values if str(value).strip()]

        return EvidenceSummary(
            query_pid=query.pid,
            chosen_ids=chosen_ids,
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
