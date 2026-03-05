from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import ExperimentConfig
from .pattern import PatternMiner
from .types import Candidate, EvidenceSummary, PlanResult, QuerySpec


@dataclass
class CandidateIndex:
    by_id: dict[str, Candidate]
    by_type: dict[str, list[Candidate]]


class PlanGenerator:
    def __init__(self, config: ExperimentConfig, pattern_miner: PatternMiner) -> None:
        self.config = config
        self.pattern_miner = pattern_miner
        self.slot_time_map = {
            "morning": "09:00-10:30",
            "noon": "12:00-13:00",
            "afternoon": "14:00-15:30",
            "evening": "18:00-19:00",
            "unknown": "15:00-16:00",
        }
        self.day_time_sequence = [
            "09:00-10:30",
            "10:45-12:00",
            "12:00-13:00",
            "13:30-15:00",
            "15:15-17:00",
            "18:00-19:00",
            "19:15-20:30",
            "20:45-22:00",
        ]

    def generate(
        self,
        query: QuerySpec,
        summary: EvidenceSummary,
        candidate_pool: list[Candidate],
        info: dict[str, Any],
    ) -> PlanResult:
        cindex = self._build_index(candidate_pool)
        chosen = [cid for cid in summary.chosen_ids if cid in cindex.by_id]

        hotel_entry = self._pick_hotel(query, chosen, cindex)
        transportation = self._pick_transport(query, info)
        itinerary = self._build_itinerary(query, chosen, cindex, summary.day_suggestions)

        return PlanResult(
            query_pid=query.pid,
            hotel=hotel_entry,
            transportation=transportation,
            itinerary=itinerary,
            candidate_pool=[c.candidate_id for c in candidate_pool],
            evidence_ids=summary.chosen_ids,
            validator_report={},
        )

    def _build_index(self, candidate_pool: list[Candidate]) -> CandidateIndex:
        by_id = {c.candidate_id: c for c in candidate_pool}
        by_type: dict[str, list[Candidate]] = {"hotel": [], "restaurant": [], "attraction": []}
        for c in candidate_pool:
            if c.entity_type in by_type:
                by_type[c.entity_type].append(c)
        by_type["hotel"].sort(key=lambda x: x.price)
        by_type["restaurant"].sort(key=lambda x: x.price)
        by_type["attraction"].sort(key=lambda x: x.price)
        return CandidateIndex(by_id=by_id, by_type=by_type)

    def _pick_hotel(self, query: QuerySpec, chosen_ids: list[str], cindex: CandidateIndex) -> list[dict[str, Any]]:
        for cid in chosen_ids:
            c = cindex.by_id.get(cid)
            if c and c.entity_type == "hotel":
                return [{"day": 1, "name": c.name, "price_per_night": round(c.price, 2), "candidate_id": c.candidate_id}]

        fallback = cindex.by_type.get("hotel", [])
        if fallback:
            c = fallback[0]
            return [{"day": 1, "name": c.name, "price_per_night": round(c.price, 2), "candidate_id": c.candidate_id}]
        return []

    def _pick_transport(self, query: QuerySpec, info: dict[str, Any]) -> list[dict[str, Any]]:
        outbound = self._pick_one_transport(info.get("transport_otd", {}), query.departure_city, query.destination_city, day=1)
        inbound = self._pick_one_transport(
            info.get("transport_dto", {}), query.destination_city, query.departure_city, day=query.day
        )
        result = []
        if outbound:
            result.append(outbound)
        if inbound:
            result.append(inbound)
        return result

    def _pick_one_transport(
        self,
        transport_block: dict[str, Any],
        from_city: str,
        to_city: str,
        day: int,
    ) -> dict[str, Any] | None:
        train_options = transport_block.get("train_options") or []
        flight_options = transport_block.get("flight_options") or []

        options: list[tuple[str, dict[str, Any], float]] = []
        for t in train_options:
            price = float(t.get("Second_Class_Price") or 0.0)
            options.append(("Train", t, price))
        for f in flight_options:
            price = float(f.get("Price") or 0.0)
            options.append(("Flight", f, price))

        if not options:
            return None

        mode, data, price = sorted(options, key=lambda x: x[2])[0]
        number = data.get("Train_Number") or data.get("Flight Number") or data.get("Flight_Number") or "NA"
        dep = data.get("Departure_Time") or data.get("Departure Time") or ""
        arr = data.get("Arrival_Time") or data.get("Arrival Time") or ""
        return {
            "day": day,
            "mode": mode,
            "route": f"{from_city} to {to_city}",
            "number": number,
            "time": f"{dep}-{arr}".strip("-"),
            "price": round(price, 2),
        }

    def _build_itinerary(
        self,
        query: QuerySpec,
        chosen_ids: list[str],
        cindex: CandidateIndex,
        day_suggestions: dict[int, list[str]],
    ) -> dict[str, list[dict[str, Any]]]:
        pattern = self.pattern_miner.get_pattern(query.day)

        chosen_attractions = [
            cindex.by_id[cid]
            for cid in chosen_ids
            if cid in cindex.by_id and cindex.by_id[cid].entity_type == "attraction"
        ]
        chosen_restaurants = [
            cindex.by_id[cid]
            for cid in chosen_ids
            if cid in cindex.by_id and cindex.by_id[cid].entity_type == "restaurant"
        ]

        if not chosen_attractions:
            chosen_attractions = cindex.by_type.get("attraction", [])[: max(2, query.day)]
        if not chosen_restaurants:
            chosen_restaurants = cindex.by_type.get("restaurant", [])[: max(2, query.day)]

        itinerary: dict[str, list[dict[str, Any]]] = {}
        used_ids: set[str] = set()

        attr_cursor = 0
        rest_cursor = 0

        for day_idx in range(1, query.day + 1):
            day_key = f"day_{day_idx}"
            activities: list[dict[str, Any]] = []
            day_used: set[str] = set()
            day_pref_ids = day_suggestions.get(day_idx, [])

            day_tokens = pattern.signature[day_idx - 1] if day_idx - 1 < len(pattern.signature) else ()
            for token in day_tokens:
                slot, action = token.split(":", 1) if ":" in token else ("afternoon", "sightseeing")
                if action == "sightseeing":
                    blocked = used_ids.union(day_used)
                    cand = self._pick_day_preferred(day_pref_ids, "attraction", blocked, cindex)
                    if cand is None:
                        cand = self._next_unused(chosen_attractions, blocked, start_index=attr_cursor)
                    if cand is None:
                        cand = self._next_unused(chosen_attractions, day_used, start_index=attr_cursor)
                    if cand is None:
                        continue
                    attr_cursor += 1
                    used_ids.add(cand.candidate_id)
                    day_used.add(cand.candidate_id)
                    activities.append(self._activity_dict(slot, "sightseeing", cand))
                elif action == "dining":
                    blocked = used_ids.union(day_used)
                    cand = self._pick_day_preferred(day_pref_ids, "restaurant", blocked, cindex)
                    if cand is None:
                        cand = self._next_unused(chosen_restaurants, blocked, start_index=rest_cursor)
                    if cand is None:
                        cand = self._next_unused(chosen_restaurants, day_used, start_index=rest_cursor)
                    if cand is None:
                        continue
                    rest_cursor += 1
                    used_ids.add(cand.candidate_id)
                    day_used.add(cand.candidate_id)
                    activities.append(self._activity_dict(slot, "dining", cand))
                else:
                    # transport/checkin events are represented in dedicated top-level blocks.
                    continue

            if not activities:
                fallback = self._fallback_day(query, day_idx, chosen_attractions, chosen_restaurants, used_ids)
                activities.extend(fallback)

            self._normalize_day_times(activities)
            itinerary[day_key] = activities

        return itinerary

    def _next_unused(self, items: list[Candidate], used_ids: set[str], start_index: int = 0) -> Candidate | None:
        if not items:
            return None
        for i in range(len(items)):
            cand = items[(start_index + i) % len(items)]
            if cand.candidate_id not in used_ids:
                return cand
        return None

    def _fallback_day(
        self,
        query: QuerySpec,
        day_idx: int,
        attractions: list[Candidate],
        restaurants: list[Candidate],
        used_ids: set[str],
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        if attractions:
            a = self._next_unused(attractions, used_ids) or attractions[(day_idx - 1) % len(attractions)]
            out.append(self._activity_dict("morning", "sightseeing", a))
            used_ids.add(a.candidate_id)
        if restaurants:
            r = self._next_unused(restaurants, used_ids) or restaurants[(day_idx - 1) % len(restaurants)]
            out.append(self._activity_dict("noon", "dining", r))
            used_ids.add(r.candidate_id)
        if attractions:
            a2 = self._next_unused(attractions, used_ids) or attractions[(day_idx) % len(attractions)]
            out.append(self._activity_dict("afternoon", "sightseeing", a2))
            used_ids.add(a2.candidate_id)
        return out

    def _activity_dict(self, slot: str, action: str, candidate: Candidate) -> dict[str, Any]:
        return {
            "time": self.slot_time_map.get(slot, self.slot_time_map["unknown"]),
            "location": candidate.name,
            "price": round(candidate.price, 2),
            "action": action,
            "candidate_id": candidate.candidate_id,
        }

    def _normalize_day_times(self, activities: list[dict[str, Any]]) -> None:
        for idx, act in enumerate(activities):
            if idx < len(self.day_time_sequence):
                act["time"] = self.day_time_sequence[idx]

    def _pick_day_preferred(
        self,
        preferred_ids: list[str],
        entity_type: str,
        blocked_ids: set[str],
        cindex: CandidateIndex,
    ) -> Candidate | None:
        for cid in preferred_ids:
            if cid in blocked_ids:
                continue
            cand = cindex.by_id.get(cid)
            if cand and cand.entity_type == entity_type:
                return cand
        return None
