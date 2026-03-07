from __future__ import annotations

from collections import defaultdict

from .types import Candidate
from .types import EvidenceSummary, QuerySpec, RetrievalItem
from .utils import haversine_km


class EvidenceSummarizer:
    _UNKNOWN_GEO_SCORE = 0.5
    _GEO_DECAY_KM = 25.0
    _TRANSPORT_RESERVE_RATIO = 0.2
    _ATTRACTION_BUDGET_SHARE = 0.55
    _RESTAURANT_BUDGET_SHARE = 0.3
    _GATE_SKIP_AVG_KM = 4.5
    _GATE_SKIP_STEP_KM = 4.5
    _GATE_SKIP_BUDGET_PRESSURE = 0.82
    _GATE_SKIP_RANKING_FLATNESS = 0.45
    _GATE_APPLY_AVG_KM = 6.5
    _GATE_APPLY_STEP_KM = 6.0
    _GATE_APPLY_BUDGET_PRESSURE = 0.9
    _GATE_APPLY_RANKING_FLATNESS = 0.58
    _GATE_BUDGET_REGRESSION_MARGIN = 0.08

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

        baseline_items = self._baseline_items(query, ranked_items)
        gate_mode, gate_reason, gate_signals = self._decide_summary_gate(query, baseline_items, chosen)

        final_items = chosen if gate_mode == "applied" else baseline_items
        reasons, trace_paths = self._build_reason_maps(final_items)
        budget_risk = self._estimate_budget_risk(query, final_items)
        if gate_mode == "applied":
            day_suggestions = self._build_day_suggestions(query, final_items)
        else:
            day_suggestions = self._empty_day_suggestions(query)

        return EvidenceSummary(
            query_pid=query.pid,
            chosen_ids=[x.candidate.candidate_id for x in final_items],
            reasons=reasons,
            budget_risk=budget_risk,
            day_suggestions=day_suggestions,
            trace_paths=trace_paths,
            gate_mode=gate_mode,
            gate_reason=gate_reason,
            gate_signals=gate_signals,
        )

    def _build_reason_maps(
        self,
        items: list[RetrievalItem],
    ) -> tuple[dict[str, str], dict[str, list[str]]]:
        reasons: dict[str, str] = {}
        trace_paths: dict[str, list[str]] = {}
        for item in items:
            cid = item.candidate.candidate_id
            reasons[cid] = (
                f"vector={item.vector_score:.3f}, constraint={item.constraint_score:.3f}, "
                f"graph={item.graph_score:.3f}, fused={item.fused_score:.3f}, "
                f"rerank={item.rerank_score:.3f}, diversity_penalty={item.diversity_penalty:.3f}"
            )
            trace_paths[cid] = item.path_evidence
        return reasons, trace_paths

    def _baseline_items(
        self,
        query: QuerySpec,
        ranked_items: list[RetrievalItem],
    ) -> list[RetrievalItem]:
        return ranked_items[: max(8, query.day * 5)]

    def _empty_day_suggestions(self, query: QuerySpec) -> dict[int, list[str]]:
        return {day: [] for day in range(1, query.day + 1)}

    def _decide_summary_gate(
        self,
        query: QuerySpec,
        baseline_items: list[RetrievalItem],
        proposed_items: list[RetrievalItem],
    ) -> tuple[str, str, dict[str, float | int | None]]:
        baseline_attractions = [item for item in baseline_items if item.candidate.entity_type == "attraction"]
        baseline_restaurants = [item for item in baseline_items if item.candidate.entity_type == "restaurant"]
        avg_anchor_km, avg_step_km = self._baseline_geo_signals(query, baseline_attractions)
        ranking_flatness = self._ranking_flatness(baseline_items)
        baseline_budget_pressure = self._estimated_budget_ratio(query, baseline_items)
        summary_budget_pressure = self._estimated_budget_ratio(query, proposed_items)
        signals: dict[str, float | int | None] = {
            "baseline_attr_count": len(baseline_attractions),
            "baseline_rest_count": len(baseline_restaurants),
            "baseline_avg_anchor_km": avg_anchor_km,
            "baseline_avg_step_km": avg_step_km,
            "ranking_flatness": ranking_flatness,
            "baseline_budget_pressure": baseline_budget_pressure,
            "summary_budget_pressure": summary_budget_pressure,
        }

        if len(baseline_attractions) < 3:
            return "skipped", "insufficient_attractions", signals

        if (
            summary_budget_pressure is not None
            and baseline_budget_pressure is not None
            and summary_budget_pressure > baseline_budget_pressure + self._GATE_BUDGET_REGRESSION_MARGIN
            and summary_budget_pressure > self._GATE_APPLY_BUDGET_PRESSURE
        ):
            return "skipped", "summary_budget_regression", signals

        if (
            self._at_most(avg_anchor_km, self._GATE_SKIP_AVG_KM)
            and self._at_most(avg_step_km, self._GATE_SKIP_STEP_KM)
            and self._at_most(baseline_budget_pressure, self._GATE_SKIP_BUDGET_PRESSURE)
        ):
            return "skipped", "stable_rank_compact_geo", signals

        if (
            self._at_least(avg_anchor_km, self._GATE_APPLY_AVG_KM)
            or self._at_least(avg_step_km, self._GATE_APPLY_STEP_KM)
            or self._at_least(baseline_budget_pressure, self._GATE_APPLY_BUDGET_PRESSURE)
        ):
            return "applied", "route_or_budget_pressure", signals

        if (
            ranking_flatness >= self._GATE_APPLY_RANKING_FLATNESS
            and (
                self._at_least(avg_anchor_km, 5.0)
                or self._at_least(avg_step_km, 4.5)
                or query.day >= 5
            )
        ):
            return "applied", "flat_rank_needs_structuring", signals

        if query.day >= 4 and (
            self._at_least(avg_step_km, 5.0)
            or (
                ranking_flatness >= 0.5
                and self._at_least(avg_anchor_km, 4.5)
            )
        ):
            return "applied", "multi_day_moderate_pressure", signals

        return "skipped", "marginal_gain", signals

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
        attrs = [x for x in chosen if x.candidate.entity_type == "attraction"]
        rests = [x for x in chosen if x.candidate.entity_type == "restaurant"]
        relevance_scores = self._normalized_relevance_scores(attrs + rests)
        daily_activity_budget = self._estimate_daily_activity_budget(query, chosen)

        day_suggestions: dict[int, list[str]] = {}
        used_attr_ids: set[str] = set()
        used_rest_ids: set[str] = set()

        for day in range(1, query.day + 1):
            picks: list[str] = []
            anchor = self._pick_day_anchor(
                query=query,
                attractions=attrs,
                used_ids=used_attr_ids,
                relevance_scores=relevance_scores,
                daily_activity_budget=daily_activity_budget,
            )
            if anchor is not None:
                anchor_id = anchor.candidate.candidate_id
                picks.append(anchor_id)
                used_attr_ids.add(anchor_id)

                for nearby in self._pick_supporting_items(
                    query=query,
                    anchor=anchor,
                    pool=attrs,
                    used_ids=used_attr_ids,
                    relevance_scores=relevance_scores,
                    daily_activity_budget=daily_activity_budget,
                    limit=2,
                ):
                    cid = nearby.candidate.candidate_id
                    picks.append(cid)
                    used_attr_ids.add(cid)

                for restaurant in self._pick_supporting_items(
                    query=query,
                    anchor=anchor,
                    pool=rests,
                    used_ids=used_rest_ids,
                    relevance_scores=relevance_scores,
                    daily_activity_budget=daily_activity_budget,
                    limit=2,
                ):
                    cid = restaurant.candidate.candidate_id
                    picks.append(cid)
                    used_rest_ids.add(cid)
            else:
                for fallback in self._fallback_items(
                    query=query,
                    pool=attrs,
                    used_ids=used_attr_ids,
                    relevance_scores=relevance_scores,
                    daily_activity_budget=daily_activity_budget,
                    limit=2,
                ):
                    cid = fallback.candidate.candidate_id
                    if cid not in picks:
                        picks.append(cid)
                        used_attr_ids.add(cid)
                for fallback in self._fallback_items(
                    query=query,
                    pool=rests,
                    used_ids=used_rest_ids,
                    relevance_scores=relevance_scores,
                    daily_activity_budget=daily_activity_budget,
                    limit=2,
                ):
                    cid = fallback.candidate.candidate_id
                    if cid not in picks:
                        picks.append(cid)
                        used_rest_ids.add(cid)

            day_suggestions[day] = picks

        return day_suggestions

    def _pick_day_anchor(
        self,
        query: QuerySpec,
        attractions: list[RetrievalItem],
        used_ids: set[str],
        relevance_scores: dict[str, float],
        daily_activity_budget: float | None,
    ) -> RetrievalItem | None:
        available = [item for item in attractions if item.candidate.candidate_id not in used_ids]
        if not available:
            available = attractions
        if not available:
            return None
        ranked = sorted(
            available,
            key=lambda item: (
                -self._anchor_score(query, item, relevance_scores, daily_activity_budget),
                item.candidate.price,
                item.candidate.candidate_id,
            ),
        )
        return ranked[0]

    def _pick_supporting_items(
        self,
        query: QuerySpec,
        anchor: RetrievalItem,
        pool: list[RetrievalItem],
        used_ids: set[str],
        relevance_scores: dict[str, float],
        daily_activity_budget: float | None,
        limit: int,
    ) -> list[RetrievalItem]:
        available = [
            item
            for item in pool
            if item.candidate.candidate_id not in used_ids
            and item.candidate.candidate_id != anchor.candidate.candidate_id
        ]
        ranked = sorted(
            available,
            key=lambda item: (
                -self._joint_day_score(query, anchor, item, relevance_scores, daily_activity_budget),
                item.candidate.price,
                item.candidate.candidate_id,
            ),
        )
        return ranked[:limit]

    def _fallback_items(
        self,
        query: QuerySpec,
        pool: list[RetrievalItem],
        used_ids: set[str],
        relevance_scores: dict[str, float],
        daily_activity_budget: float | None,
        limit: int,
    ) -> list[RetrievalItem]:
        available = [item for item in pool if item.candidate.candidate_id not in used_ids]
        ranked = sorted(
            available,
            key=lambda item: (
                -self._anchor_score(query, item, relevance_scores, daily_activity_budget),
                item.candidate.price,
                item.candidate.candidate_id,
            ),
        )
        return ranked[:limit]

    def _anchor_score(
        self,
        query: QuerySpec,
        item: RetrievalItem,
        relevance_scores: dict[str, float],
        daily_activity_budget: float | None,
    ) -> float:
        cid = item.candidate.candidate_id
        relevance = relevance_scores.get(cid, 0.5)
        budget_fit = self._budget_fit(query, item.candidate, daily_activity_budget)
        return 0.72 * relevance + 0.28 * budget_fit

    def _joint_day_score(
        self,
        query: QuerySpec,
        anchor: RetrievalItem,
        item: RetrievalItem,
        relevance_scores: dict[str, float],
        daily_activity_budget: float | None,
    ) -> float:
        cid = item.candidate.candidate_id
        relevance = relevance_scores.get(cid, 0.5)
        geo_score = self._geo_score(anchor.candidate, item.candidate)
        budget_fit = self._budget_fit(query, item.candidate, daily_activity_budget)
        if item.candidate.entity_type == "restaurant":
            return 0.34 * relevance + 0.28 * geo_score + 0.38 * budget_fit
        return 0.38 * relevance + 0.46 * geo_score + 0.16 * budget_fit

    def _normalized_relevance_scores(self, items: list[RetrievalItem]) -> dict[str, float]:
        if not items:
            return {}
        raw_scores = {item.candidate.candidate_id: self._base_relevance(item) for item in items}
        low = min(raw_scores.values())
        high = max(raw_scores.values())
        if high - low <= 1e-9:
            return {cid: 1.0 for cid in raw_scores}
        return {cid: (score - low) / (high - low) for cid, score in raw_scores.items()}

    def _base_relevance(self, item: RetrievalItem) -> float:
        return max(item.rerank_score, item.fused_score, item.raw_fused_score, 0.0)

    def _baseline_geo_signals(
        self,
        query: QuerySpec,
        attractions: list[RetrievalItem],
    ) -> tuple[float | None, float | None]:
        sample = attractions[: max(3, min(len(attractions), query.day + 2))]
        if len(sample) < 2:
            return None, None

        anchor = sample[0].candidate
        anchor_distances = [
            self._distance(anchor, item.candidate)
            for item in sample[1:]
            if self._distance(anchor, item.candidate) < 999999.0
        ]
        step_distances = [
            self._distance(left.candidate, right.candidate)
            for left, right in zip(sample, sample[1:])
            if self._distance(left.candidate, right.candidate) < 999999.0
        ]
        avg_anchor_km = (
            sum(anchor_distances) / len(anchor_distances) if anchor_distances else None
        )
        avg_step_km = sum(step_distances) / len(step_distances) if step_distances else None
        return avg_anchor_km, avg_step_km

    def _ranking_flatness(self, items: list[RetrievalItem]) -> float:
        ranked = sorted(
            (self._base_relevance(item) for item in items if item.candidate.entity_type != "hotel"),
            reverse=True,
        )
        if len(ranked) < 2:
            return 0.0
        top = ranked[0]
        tail = sum(ranked[1 : min(5, len(ranked))]) / max(1, min(4, len(ranked) - 1))
        return max(0.0, 1.0 - max(0.0, top - tail))

    def _estimated_budget_ratio(
        self,
        query: QuerySpec,
        items: list[RetrievalItem],
    ) -> float | None:
        if query.budget is None or query.budget <= 0:
            return None
        hotel_items = [item for item in items if item.candidate.entity_type == "hotel"]
        attr_items = [item for item in items if item.candidate.entity_type == "attraction"]
        rest_items = [item for item in items if item.candidate.entity_type == "restaurant"]
        estimated = 0.0
        if hotel_items:
            estimated += hotel_items[0].candidate.price * query.day
        estimated += sum(item.candidate.price for item in attr_items[: query.day * 2])
        estimated += sum(item.candidate.price for item in rest_items[: query.day * 2])
        return estimated / max(1.0, query.budget)

    def _at_most(self, value: float | None, threshold: float) -> bool:
        return value is not None and value <= threshold

    def _at_least(self, value: float | None, threshold: float) -> bool:
        return value is not None and value >= threshold

    def _estimate_daily_activity_budget(
        self,
        query: QuerySpec,
        chosen: list[RetrievalItem],
    ) -> float | None:
        if query.budget is None or query.budget <= 0:
            return None

        hotel_items = [item for item in chosen if item.candidate.entity_type == "hotel"]
        hotel_total = hotel_items[0].candidate.price * query.day if hotel_items else 0.0
        remaining = max(0.0, query.budget - hotel_total)
        if remaining <= 0:
            return 0.0

        # Keep part of the budget for transport, which is selected downstream.
        activity_total = remaining * (1.0 - self._TRANSPORT_RESERVE_RATIO)
        return activity_total / max(1, query.day)

    def _budget_fit(
        self,
        query: QuerySpec,
        candidate: Candidate,
        daily_activity_budget: float | None,
    ) -> float:
        if candidate.entity_type == "restaurant" and query.meal_price_range:
            lo, hi = query.meal_price_range
            meal_fit = self._range_fit(candidate.price, lo, hi)
        else:
            meal_fit = 1.0

        budget_cap_fit = 1.0
        if daily_activity_budget is not None:
            item_budget = self._per_item_budget(candidate, daily_activity_budget)
            if item_budget <= 0:
                budget_cap_fit = 1.0 if candidate.price <= 0 else 0.0
            elif candidate.price <= item_budget:
                budget_cap_fit = 1.0
            else:
                budget_cap_fit = max(0.0, item_budget / max(candidate.price, 1.0))

        if candidate.entity_type == "restaurant" and query.meal_price_range:
            return 0.6 * meal_fit + 0.4 * budget_cap_fit
        return budget_cap_fit

    def _per_item_budget(self, candidate: Candidate, daily_activity_budget: float) -> float:
        if candidate.entity_type == "restaurant":
            return daily_activity_budget * self._RESTAURANT_BUDGET_SHARE
        if candidate.entity_type == "attraction":
            return daily_activity_budget * self._ATTRACTION_BUDGET_SHARE
        return daily_activity_budget

    def _range_fit(self, price: float, lo: float, hi: float) -> float:
        if hi < lo:
            lo, hi = hi, lo
        span = max(1.0, hi - lo)
        if lo <= price <= hi:
            mid = (lo + hi) / 2.0
            return max(0.7, 1.0 - abs(price - mid) / span)
        if price < lo:
            return max(0.0, 1.0 - (lo - price) / span)
        return max(0.0, 1.0 - (price - hi) / span)

    def _geo_score(self, anchor: Candidate, candidate: Candidate) -> float:
        if (
            anchor.latitude is None
            or anchor.longitude is None
            or candidate.latitude is None
            or candidate.longitude is None
        ):
            return self._UNKNOWN_GEO_SCORE
        distance = self._distance(anchor, candidate)
        return max(0.0, 1.0 - distance / self._GEO_DECAY_KM)

    def _distance(self, left: Candidate, right: Candidate) -> float:
        if (
            left.latitude is None
            or left.longitude is None
            or right.latitude is None
            or right.longitude is None
        ):
            return 999999.0
        return haversine_km(left.latitude, left.longitude, right.latitude, right.longitude)
