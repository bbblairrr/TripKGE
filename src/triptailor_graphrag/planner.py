from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from typing import Any

from .config import ExperimentConfig
from .local_llm import LocalLLMClient, LocalLLMError
from .pattern import PatternMiner
from .types import Candidate, EvidenceSummary, PlanResult, QuerySpec
from .utils import (
    format_time_range,
    normalize_text,
    parse_duration_minutes,
    parse_opening_hours,
    parse_time_range,
    sanitize_llm_text,
)


@dataclass
class CandidateIndex:
    by_id: dict[str, Candidate]
    by_type: dict[str, list[Candidate]]


class PlanGenerator:
    def __init__(
        self,
        config: ExperimentConfig,
        pattern_miner: PatternMiner,
        llm_client: LocalLLMClient | None = None,
    ) -> None:
        self.config = config
        self.pattern_miner = pattern_miner
        self.llm_client = llm_client
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
        self.default_day_start = 9 * 60
        self.default_day_end = 22 * 60
        self.transport_buffer_minutes = 30
        self.activity_gap_minutes = 15

    def generate(
        self,
        query: QuerySpec,
        summary: EvidenceSummary,
        candidate_pool: list[Candidate],
        info: dict[str, Any],
    ) -> PlanResult:
        cindex = self._build_index(candidate_pool)
        chosen = [cid for cid in summary.chosen_ids if cid in cindex.by_id]
        planning_error: str | None = None

        if self.llm_client is not None:
            llm_plan, planning_error = self._generate_with_llm(query, summary, candidate_pool, info, cindex)
            if llm_plan is not None:
                return llm_plan

        hotel_entry = self._pick_hotel(query, chosen, cindex)
        transportation = self._pick_transport(query, info)
        itinerary = self._build_itinerary(query, chosen, cindex, summary.day_suggestions, transportation)
        itinerary = self._schedule_itinerary(query, itinerary, transportation, cindex, chosen)

        return PlanResult(
            query_pid=query.pid,
            hotel=hotel_entry,
            transportation=transportation,
            itinerary=itinerary,
            candidate_pool=[c.candidate_id for c in candidate_pool],
            evidence_ids=summary.chosen_ids,
            validator_report={},
            planner_mode="heuristic_fallback" if self.llm_client is not None else "heuristic",
            planning_error=planning_error,
        )

    def _generate_with_llm(
        self,
        query: QuerySpec,
        summary: EvidenceSummary,
        candidate_pool: list[Candidate],
        info: dict[str, Any],
        cindex: CandidateIndex,
    ) -> tuple[PlanResult | None, str | None]:
        transportation = self._pick_transport(query, info)
        base_prompt = self._build_llm_prompt(query, summary, candidate_pool, cindex, transportation)
        last_error: Exception | None = None
        last_response = ""
        attempts = max(1, self.config.llm.generation_retries)

        for attempt in range(1, attempts + 1):
            try:
                prompt = self._build_llm_retry_prompt(base_prompt, last_response, str(last_error) if last_error else None, attempt)
                response = self.llm_client.generate(prompt, system_prompt=self._llm_system_prompt())
                last_response = response
                payload = self._parse_llm_json_with_retries(response)
                hotel_entry = self._llm_pick_hotel(payload, query, summary, cindex)
                itinerary = self._llm_build_itinerary(payload, query, summary, cindex, transportation)
                itinerary = self._schedule_itinerary(query, itinerary, transportation, cindex, summary.chosen_ids)
                return PlanResult(
                    query_pid=query.pid,
                    hotel=hotel_entry,
                    transportation=transportation,
                    itinerary=itinerary,
                    candidate_pool=[c.candidate_id for c in candidate_pool],
                    evidence_ids=summary.chosen_ids,
                    validator_report={},
                    planner_mode="llm",
                    planning_error=None,
                ), None
            except (LocalLLMError, ValueError, KeyError, SyntaxError, json.JSONDecodeError) as exc:
                last_error = exc
                self._dump_llm_failure(query.pid, attempt, last_response, str(exc))

        if not self.config.llm.fallback_to_heuristic:
            raise ValueError(
                f"LLM planning failed after {attempts} attempts for pid={query.pid}: {last_error}"
            )
        return None, str(last_error) if last_error else "LLM planning failed."

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

    def _llm_pick_hotel(
        self,
        payload: dict[str, Any],
        query: QuerySpec,
        summary: EvidenceSummary,
        cindex: CandidateIndex,
    ) -> list[dict[str, Any]]:
        hotel_ref = str(payload.get("hotel_candidate_id") or "").strip()
        hotel = self._resolve_candidate_ref(hotel_ref, cindex, entity_type="hotel") if hotel_ref else None
        if hotel is None:
            return self._pick_hotel(query, summary.chosen_ids, cindex)
        return [{"day": 1, "name": hotel.name, "price_per_night": round(hotel.price, 2), "candidate_id": hotel.candidate_id}]

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
        transportation: list[dict[str, Any]],
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
            max_items = self._max_items_for_day(query, day_idx, transportation)
            if max_items <= 0:
                itinerary[day_key] = []
                continue

            day_tokens = pattern.signature[day_idx - 1] if day_idx - 1 < len(pattern.signature) else ()
            for token in day_tokens[:max_items]:
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
                fallback = self._fallback_day(
                    query,
                    day_idx,
                    chosen_attractions,
                    chosen_restaurants,
                    used_ids,
                    max_items=max_items,
                )
                activities.extend(fallback)

            itinerary[day_key] = activities

        return itinerary

    def _llm_build_itinerary(
        self,
        payload: dict[str, Any],
        query: QuerySpec,
        summary: EvidenceSummary,
        cindex: CandidateIndex,
        transportation: list[dict[str, Any]],
    ) -> dict[str, list[dict[str, Any]]]:
        days_payload = payload.get("days", {})
        if not isinstance(days_payload, dict):
            raise ValueError("LLM output missing `days` object.")

        chosen_ids = [cid for cid in summary.chosen_ids if cid in cindex.by_id]
        chosen_attractions = [
            cindex.by_id[cid]
            for cid in chosen_ids
            if cindex.by_id[cid].entity_type == "attraction"
        ]
        chosen_restaurants = [
            cindex.by_id[cid]
            for cid in chosen_ids
            if cindex.by_id[cid].entity_type == "restaurant"
        ]

        itinerary: dict[str, list[dict[str, Any]]] = {}
        used_ids: set[str] = set()
        for day_idx in range(1, query.day + 1):
            raw_items = days_payload.get(str(day_idx), [])
            activities: list[dict[str, Any]] = []
            day_used: set[str] = set()
            max_items = self._max_items_for_day(query, day_idx, transportation)
            if max_items <= 0:
                itinerary[f"day_{day_idx}"] = []
                continue
            if isinstance(raw_items, list):
                for item in raw_items[:max_items]:
                    if not isinstance(item, dict):
                        continue
                    action = str(item.get("action") or "").strip().lower()
                    if action not in {"sightseeing", "dining"}:
                        continue
                    expected_type = "attraction" if action == "sightseeing" else "restaurant"
                    ref = str(item.get("candidate_id") or item.get("name") or "").strip()
                    cand = self._resolve_candidate_ref(ref, cindex, entity_type=expected_type)
                    if cand is None or cand.candidate_id in day_used:
                        continue
                    activities.append(self._activity_dict("unknown", action, cand))
                    day_used.add(cand.candidate_id)
                    used_ids.add(cand.candidate_id)

            if not activities:
                activities = self._fallback_day(
                    query,
                    day_idx,
                    chosen_attractions,
                    chosen_restaurants,
                    used_ids,
                    max_items=max_items,
                )
            itinerary[f"day_{day_idx}"] = activities
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
        max_items: int = 3,
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        if max_items <= 0:
            return out
        if attractions and len(out) < max_items:
            a = self._next_unused(attractions, used_ids) or attractions[(day_idx - 1) % len(attractions)]
            out.append(self._activity_dict("morning", "sightseeing", a))
            used_ids.add(a.candidate_id)
        if restaurants and len(out) < max_items:
            r = self._next_unused(restaurants, used_ids) or restaurants[(day_idx - 1) % len(restaurants)]
            out.append(self._activity_dict("noon", "dining", r))
            used_ids.add(r.candidate_id)
        if attractions and len(out) < max_items:
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

    def _schedule_itinerary(
        self,
        query: QuerySpec,
        itinerary: dict[str, list[dict[str, Any]]],
        transportation: list[dict[str, Any]],
        cindex: CandidateIndex,
        preferred_ids: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        scheduled: dict[str, list[dict[str, Any]]] = {}
        used_trip: set[str] = set()
        preferred_attractions = [
            cindex.by_id[cid]
            for cid in preferred_ids
            if cid in cindex.by_id and cindex.by_id[cid].entity_type == "attraction"
        ]
        preferred_restaurants = [
            cindex.by_id[cid]
            for cid in preferred_ids
            if cid in cindex.by_id and cindex.by_id[cid].entity_type == "restaurant"
        ]

        for day_idx in range(1, query.day + 1):
            day_key = f"day_{day_idx}"
            raw_items = itinerary.get(day_key, [])
            max_items = self._max_items_for_day(query, day_idx, transportation)
            day_items = self._schedule_day_items(
                query=query,
                day_idx=day_idx,
                items=raw_items[:max_items] if max_items > 0 else [],
                transportation=transportation,
                cindex=cindex,
                used_trip=used_trip,
            )
            if not day_items and max_items > 0:
                fallback = self._fallback_day(
                    query=query,
                    day_idx=day_idx,
                    attractions=preferred_attractions or cindex.by_type.get("attraction", []),
                    restaurants=preferred_restaurants or cindex.by_type.get("restaurant", []),
                    used_ids=set(used_trip),
                    max_items=max_items,
                )
                day_items = self._schedule_day_items(
                    query=query,
                    day_idx=day_idx,
                    items=fallback,
                    transportation=transportation,
                    cindex=cindex,
                    used_trip=used_trip,
                )
            scheduled[day_key] = day_items
            used_trip.update(
                act.get("candidate_id")
                for act in day_items
                if isinstance(act, dict) and act.get("candidate_id")
            )
        return scheduled

    def _schedule_day_items(
        self,
        query: QuerySpec,
        day_idx: int,
        items: list[dict[str, Any]],
        transportation: list[dict[str, Any]],
        cindex: CandidateIndex,
        used_trip: set[str],
    ) -> list[dict[str, Any]]:
        day_start, day_end = self._day_time_bounds(query, day_idx, transportation)
        if day_end - day_start < 45:
            return []

        current = day_start
        day_used: set[str] = set()
        scheduled: list[dict[str, Any]] = []
        for act in items:
            cid = act.get("candidate_id")
            if not cid or cid in day_used or cid in used_trip:
                continue
            candidate = cindex.by_id.get(cid)
            if candidate is None:
                continue
            interval = self._candidate_interval(candidate, str(act.get("action") or ""), current, day_end)
            if interval is None:
                continue
            start_min, end_min = interval
            scheduled.append(
                {
                    "time": format_time_range(start_min, end_min),
                    "location": candidate.name,
                    "price": round(candidate.price, 2),
                    "action": act.get("action"),
                    "candidate_id": candidate.candidate_id,
                }
            )
            current = min(day_end, end_min + self.activity_gap_minutes)
            day_used.add(candidate.candidate_id)
        return scheduled

    def _day_time_bounds(
        self,
        query: QuerySpec,
        day_idx: int,
        transportation: list[dict[str, Any]],
    ) -> tuple[int, int]:
        start = self.default_day_start
        end = self.default_day_end
        outbound = self._transport_for_direction(query, transportation, "outbound")
        inbound = self._transport_for_direction(query, transportation, "inbound")

        if day_idx == 1 and outbound is not None:
            interval = parse_time_range(str(outbound.get("time") or ""))
            if interval is not None:
                start = max(start, interval[1] + self.transport_buffer_minutes)
        if day_idx == query.day and inbound is not None:
            interval = parse_time_range(str(inbound.get("time") or ""))
            if interval is not None:
                end = min(end, interval[0] - self.transport_buffer_minutes)
        return start, max(start, end)

    def _max_items_for_day(
        self,
        query: QuerySpec,
        day_idx: int,
        transportation: list[dict[str, Any]],
    ) -> int:
        start, end = self._day_time_bounds(query, day_idx, transportation)
        available = max(0, end - start)
        if available < 60:
            return 0
        estimate = max(1, min(5, available // 105))
        if query.day > 1 and day_idx in {1, query.day}:
            estimate = min(estimate, 3)
        return int(estimate)

    def _transport_for_direction(
        self,
        query: QuerySpec,
        transportation: list[dict[str, Any]],
        direction: str,
    ) -> dict[str, Any] | None:
        outbound_route = normalize_text(f"{query.departure_city} to {query.destination_city}")
        inbound_route = normalize_text(f"{query.destination_city} to {query.departure_city}")
        for transport in transportation:
            route = normalize_text(str(transport.get("route") or ""))
            if direction == "outbound" and route == outbound_route:
                return transport
            if direction == "inbound" and route == inbound_route:
                return transport
        return None

    def _transport_constraint_lines(
        self,
        query: QuerySpec,
        transportation: list[dict[str, Any]],
    ) -> list[str]:
        lines: list[str] = []
        outbound = self._transport_for_direction(query, transportation, "outbound")
        inbound = self._transport_for_direction(query, transportation, "inbound")
        if outbound is not None:
            interval = parse_time_range(str(outbound.get("time") or ""))
            if interval is not None:
                lines.append(
                    f"day_1 earliest activity start >= {format_time_range(interval[1] + self.transport_buffer_minutes, interval[1] + self.transport_buffer_minutes)}"
                )
        if inbound is not None:
            interval = parse_time_range(str(inbound.get("time") or ""))
            if interval is not None:
                lines.append(
                    f"day_{query.day} latest activity end <= {format_time_range(interval[0] - self.transport_buffer_minutes, interval[0] - self.transport_buffer_minutes)}"
                )
        for day_idx in range(1, query.day + 1):
            start, end = self._day_time_bounds(query, day_idx, transportation)
            if end - start < 45:
                lines.append(f"day_{day_idx} has almost no sightseeing window because of transport; keep it empty or at most one short meal")
                continue
            lines.append(
                f"day_{day_idx} usable window is {format_time_range(start, end)} and should fit no more than {self._max_items_for_day(query, day_idx, transportation)} activities"
            )
        return lines

    def _candidate_interval(
        self,
        candidate: Candidate,
        action: str,
        earliest_start: int,
        latest_end: int,
    ) -> tuple[int, int] | None:
        duration = self._candidate_duration_minutes(candidate, action)
        start = earliest_start
        if action == "sightseeing":
            opening = parse_opening_hours(str(candidate.meta.get("opening_hours") if isinstance(candidate.meta, dict) else ""))
            if opening is not None:
                start = max(start, opening[0])
                if start + duration > opening[1]:
                    return None
        end = start + duration
        if end > latest_end:
            return None
        return start, end

    def _candidate_duration_minutes(self, candidate: Candidate, action: str) -> int:
        if action == "dining":
            return 60
        if isinstance(candidate.meta, dict):
            duration = parse_duration_minutes(str(candidate.meta.get("recommended_duration") or ""))
            if duration is not None:
                return max(30, min(duration, 8 * 60))
        return 90

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

    def _llm_system_prompt(self) -> str:
        return (
            "You are a travel planner. Output valid JSON only. "
            "Never invent candidate ids. Only use ids from the provided shortlist."
        )

    def _build_llm_prompt(
        self,
        query: QuerySpec,
        summary: EvidenceSummary,
        candidate_pool: list[Candidate],
        cindex: CandidateIndex,
        transportation: list[dict[str, Any]],
    ) -> str:
        preferred_ids = [cid for cid in summary.chosen_ids if cid in cindex.by_id]
        if not preferred_ids:
            preferred_ids = [c.candidate_id for c in candidate_pool[: self.config.llm.max_candidates]]
        preferred_ids = preferred_ids[: self.config.llm.max_candidates]

        candidate_lines = []
        for cid in preferred_ids:
            cand = cindex.by_id[cid]
            tag_str = ", ".join(cand.tags[:4]) if cand.tags else "none"
            extra: list[str] = []
            if cand.entity_type == "attraction" and isinstance(cand.meta, dict):
                opening = str(cand.meta.get("opening_hours") or "").strip()
                duration = str(cand.meta.get("recommended_duration") or "").strip()
                if opening:
                    extra.append(f"opening={opening}")
                if duration:
                    extra.append(f"recommended_duration={duration}")
            extra_str = "; " + "; ".join(extra) if extra else ""
            candidate_lines.append(
                f"- id={cand.candidate_id}; type={cand.entity_type}; name={cand.name}; "
                f"price={cand.price:.2f}; tags={tag_str}{extra_str}; text={cand.text}"
            )

        constraints = [
            f"trip_days={query.day}",
            f"departure_city={query.departure_city}",
            f"destination_city={query.destination_city}",
            f"budget={query.budget if query.budget is not None else 'unknown'}",
            f"meal_price_range={query.meal_price_range if query.meal_price_range else 'unknown'}",
            f"hotel_category_pref={query.hotel_category_pref or 'none'}",
            f"intensity_pref={query.intensity_pref or 'none'}",
            f"interest_tags={query.interest_tags or []}",
            f"budget_risk_hint={summary.budget_risk}",
            "keep each day geographically compact and prefer nearby attractions/restaurants together",
            "day 1 and the last day must respect the exact transport arrival/departure windows below",
            "ensure each attraction gets at least its recommended duration when provided",
            "do not schedule a sightseeing activity outside the attraction opening hours when they are known",
            "avoid repeating the same restaurant or attraction unless the shortlist is exhausted",
        ]
        transport_lines = self._transport_constraint_lines(query, transportation)
        suggestion_lines: list[str] = []
        for day, ids in summary.day_suggestions.items():
            named = []
            for cid in ids:
                cand = cindex.by_id.get(cid)
                if cand is None:
                    continue
                named.append(f"{cid}:{cand.name}")
            if named:
                suggestion_lines.append(f"day_{day} -> {', '.join(named)}")
        output_schema = {
            "hotel_candidate_id": "one hotel candidate id",
            "days": {
                "1": [
                    {"candidate_id": "attraction or restaurant id", "action": "sightseeing or dining"},
                ],
            },
        }
        return (
            "Plan a personalized itinerary from the candidate shortlist.\n"
            "Use 2 to 5 activities per day. Prefer varied attractions and restaurants. "
            "Respect budget and preferences.\n\n"
            "Query constraints:\n"
            + "\n".join(f"- {line}" for line in constraints)
            + (
                "\n\nTransport windows:\n" + "\n".join(f"- {line}" for line in transport_lines)
                if transport_lines
                else ""
            )
            + (
                "\n\nSuggested day clusters:\n" + "\n".join(f"- {line}" for line in suggestion_lines)
                if suggestion_lines
                else ""
            )
            + "\n\nCandidate shortlist:\n"
            + "\n".join(candidate_lines)
            + "\n\nReturn JSON with this schema:\n"
            + json.dumps(output_schema, ensure_ascii=False, indent=2)
        )

    def _parse_llm_json_with_retries(self, text: str) -> dict[str, Any]:
        current = sanitize_llm_text(text)
        last_error: Exception | None = None
        for _ in range(3):
            try:
                return self._parse_llm_json(current)
            except ValueError as exc:
                last_error = exc
                repair_prompt = self._build_llm_json_repair_prompt(current, error_message=str(exc))
                current = sanitize_llm_text(
                    self.llm_client.generate(repair_prompt, system_prompt=self._llm_json_repair_system_prompt())
                )
        raise ValueError(str(last_error) if last_error else "Unable to parse LLM output as JSON.")

    def _parse_llm_json(self, text: str) -> dict[str, Any]:
        cleaned = self._strip_llm_json_wrapper(text)
        candidates: list[str] = []
        if cleaned:
            candidates.append(cleaned)
        extracted = self._extract_balanced_json_object(cleaned)
        if extracted and extracted not in candidates:
            candidates.append(extracted)

        last_error: Exception | None = None
        for candidate in candidates:
            for parser in (self._parse_strict_json, self._parse_python_style_json):
                try:
                    payload = parser(candidate)
                except (ValueError, SyntaxError, json.JSONDecodeError) as exc:
                    last_error = exc
                    continue
                if isinstance(payload, dict):
                    return payload
                last_error = ValueError("LLM output root must be a JSON object.")
            try:
                payload = self._parse_planner_schema_fallback(candidate)
            except ValueError as exc:
                last_error = exc
            else:
                return payload

        error_msg = f"Unable to parse LLM output as JSON: {last_error}" if last_error else "Unable to parse LLM output as JSON."
        raise ValueError(error_msg)

    def _parse_strict_json(self, text: str) -> dict[str, Any]:
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError("LLM output root must be a JSON object.")
        return payload

    def _parse_python_style_json(self, text: str) -> dict[str, Any]:
        repaired = self._repair_json_like_text(text)
        payload = ast.literal_eval(repaired)
        if not isinstance(payload, dict):
            raise ValueError("LLM output root must be a mapping.")
        return payload

    def _strip_llm_json_wrapper(self, text: str) -> str:
        cleaned = sanitize_llm_text(text)
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE | re.DOTALL).strip()
        cleaned = cleaned.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
        return cleaned

    def _extract_balanced_json_object(self, text: str) -> str | None:
        start = text.find("{")
        if start < 0:
            return None
        depth = 0
        in_string = False
        quote_char = ""
        escaped = False
        for idx in range(start, len(text)):
            ch = text[idx]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == quote_char:
                    in_string = False
                continue
            if ch in {'"', "'"}:
                in_string = True
                quote_char = ch
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : idx + 1]
        return None

    def _repair_json_like_text(self, text: str) -> str:
        repaired = sanitize_llm_text(text)
        repaired = repaired.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
        repaired = re.sub(r"(?m)^\s*(//|#).*$", "", repaired)
        repaired = re.sub(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)", r'\1"\2"\3', repaired)
        repaired = re.sub(r",(\s*[}\]])", r"\1", repaired)
        repaired = re.sub(r"(?<![A-Za-z0-9_\"'])\btrue\b", "True", repaired)
        repaired = re.sub(r"(?<![A-Za-z0-9_\"'])\bfalse\b", "False", repaired)
        repaired = re.sub(r"(?<![A-Za-z0-9_\"'])\bnull\b", "None", repaired)

        def quote_bare_value(match: re.Match[str]) -> str:
            prefix, value, suffix = match.groups()
            value = value.strip()
            if not value:
                return f'{prefix}""{suffix}'
            if value.startswith(('"', "'", "{", "[")):
                return f"{prefix}{value}{suffix}"
            if re.fullmatch(r"-?\d+(?:\.\d+)?", value):
                return f"{prefix}{value}{suffix}"
            if value in {"True", "False", "None"}:
                return f"{prefix}{value}{suffix}"
            escaped = value.replace("\\", "\\\\").replace('"', '\\"')
            return f'{prefix}"{escaped}"{suffix}'

        repaired = re.sub(
            r'(:\s*)([^,\[\]\{\}\n][^,\}\]\n]*?)(\s*[,}\]])',
            quote_bare_value,
            repaired,
        )
        return repaired

    def _parse_planner_schema_fallback(self, text: str) -> dict[str, Any]:
        cleaned = sanitize_llm_text(text)
        hotel_match = re.search(r'"hotel_candidate_id"\s*:\s*"([^"\n]+)"', cleaned)
        hotel_candidate_id = hotel_match.group(1).strip() if hotel_match else ""

        day_pattern = re.compile(r'^\s*"(\d+)"\s*:\s*\[\s*$', re.MULTILINE)
        candidate_pattern = re.compile(r'"candidate_id"\s*:\s*"([^"\n]+)')
        action_pattern = re.compile(r'"action"\s*:\s*"([A-Za-z_ -]+)')
        standalone_action_pattern = re.compile(r'^\s*"?(sightseeing|dining)"?\s*$')
        valid_actions = {"sightseeing", "dining"}

        days: dict[str, list[dict[str, str]]] = {}
        current_day: str | None = None
        pending_candidate_id: str | None = None
        lines = cleaned.splitlines()

        for idx, line in enumerate(lines):
            day_match = day_pattern.match(line)
            if day_match:
                current_day = day_match.group(1)
                days.setdefault(current_day, [])
                pending_candidate_id = None
                continue

            if current_day is None:
                continue

            candidate_match = candidate_pattern.search(line)
            if candidate_match:
                pending_candidate_id = candidate_match.group(1).strip()
                continue

            action_match = action_pattern.search(line)
            action_value: str | None = None
            if action_match:
                raw_action = action_match.group(1).strip().lower()
                if raw_action in valid_actions:
                    action_value = raw_action
                else:
                    for look_ahead in lines[idx + 1 : min(len(lines), idx + 3)]:
                        standalone_match = standalone_action_pattern.match(look_ahead.strip().lower())
                        if standalone_match:
                            action_value = standalone_match.group(1)
                            break
                    if action_value is None:
                        if raw_action.startswith("din"):
                            action_value = "dining"
                        elif raw_action.startswith("sight"):
                            action_value = "sightseeing"

            if pending_candidate_id and action_value:
                days.setdefault(current_day, []).append(
                    {
                        "candidate_id": pending_candidate_id,
                        "action": action_value,
                    }
                )
                pending_candidate_id = None

            if line.strip().startswith("]"):
                current_day = None
                pending_candidate_id = None

        if not hotel_candidate_id and not any(days.values()):
            raise ValueError("Unable to recover planner schema from malformed LLM output.")
        return {
            "hotel_candidate_id": hotel_candidate_id,
            "days": days,
        }

    def _llm_json_repair_system_prompt(self) -> str:
        return (
            "You repair malformed JSON. Output exactly one valid JSON object and nothing else. "
            "Use double quotes for every key and every string value."
        )

    def _build_llm_retry_prompt(
        self,
        base_prompt: str,
        last_response: str,
        error_message: str | None,
        attempt: int,
    ) -> str:
        if attempt <= 1:
            return base_prompt
        error_line = f"Previous parser/planning error: {error_message}\n\n" if error_message else ""
        previous_line = f"Previous invalid output:\n{last_response}\n\n" if last_response else ""
        return (
            base_prompt
            + "\n\nIMPORTANT: Your previous answer was invalid. Retry and return exactly one valid JSON object."
            + "\nDo not include markdown fences, comments, explanations, or trailing commas.\n\n"
            + error_line
            + previous_line
        )

    def _build_llm_json_repair_prompt(self, raw_response: str, error_message: str | None = None) -> str:
        error_line = f"Parser error: {error_message}\n\n" if error_message else ""
        return (
            "Rewrite the following malformed planner response as valid JSON only.\n"
            "Requirements:\n"
            "- Output exactly one JSON object\n"
            "- Use double quotes for every key and every string value\n"
            "- Keep the same semantic content when possible\n"
            "- Do not add explanation, markdown fences, or comments\n\n"
            + error_line
            + "Response to repair:\n"
            + sanitize_llm_text(raw_response)
        )

    def _dump_llm_failure(self, pid: int, attempt: int, response: str, error_message: str) -> None:
        dump_dir = self.config.output_dir / "llm_failures"
        dump_dir.mkdir(parents=True, exist_ok=True)
        path = dump_dir / f"pid_{pid}_attempt_{attempt}.txt"
        payload = (
            f"pid={pid}\n"
            f"attempt={attempt}\n"
            f"error={error_message}\n"
            "response:\n"
            f"{sanitize_llm_text(response)}"
        )
        path.write_text(payload, encoding="utf-8")

    def _resolve_candidate_ref(
        self,
        ref: str,
        cindex: CandidateIndex,
        entity_type: str | None = None,
    ) -> Candidate | None:
        if not ref:
            return None
        if ref in cindex.by_id:
            cand = cindex.by_id[ref]
            return cand if entity_type is None or cand.entity_type == entity_type else None

        target = normalize_text(ref)
        for cand in cindex.by_id.values():
            if entity_type is not None and cand.entity_type != entity_type:
                continue
            name = normalize_text(cand.name)
            if target == name or target in name or name in target:
                return cand
        return None
