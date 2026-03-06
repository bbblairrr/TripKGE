from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .config import ExperimentConfig
from .local_llm import LocalLLMClient, LocalLLMError
from .pattern import PatternMiner
from .types import Candidate, EvidenceSummary, PlanResult, QuerySpec
from .utils import normalize_text


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

    def generate(
        self,
        query: QuerySpec,
        summary: EvidenceSummary,
        candidate_pool: list[Candidate],
        info: dict[str, Any],
    ) -> PlanResult:
        cindex = self._build_index(candidate_pool)
        chosen = [cid for cid in summary.chosen_ids if cid in cindex.by_id]

        if self.llm_client is not None:
            llm_plan = self._generate_with_llm(query, summary, candidate_pool, info, cindex)
            if llm_plan is not None:
                return llm_plan

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

    def _generate_with_llm(
        self,
        query: QuerySpec,
        summary: EvidenceSummary,
        candidate_pool: list[Candidate],
        info: dict[str, Any],
        cindex: CandidateIndex,
    ) -> PlanResult | None:
        try:
            prompt = self._build_llm_prompt(query, summary, candidate_pool, cindex)
            response = self.llm_client.generate(prompt, system_prompt=self._llm_system_prompt())
            payload = self._parse_llm_json(response)
            hotel_entry = self._llm_pick_hotel(payload, query, summary, cindex)
            transportation = self._pick_transport(query, info)
            itinerary = self._llm_build_itinerary(payload, query, summary, cindex)
            return PlanResult(
                query_pid=query.pid,
                hotel=hotel_entry,
                transportation=transportation,
                itinerary=itinerary,
                candidate_pool=[c.candidate_id for c in candidate_pool],
                evidence_ids=summary.chosen_ids,
                validator_report={},
            )
        except (LocalLLMError, ValueError, KeyError, json.JSONDecodeError):
            if not self.config.llm.fallback_to_heuristic:
                raise
            return None

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

    def _llm_build_itinerary(
        self,
        payload: dict[str, Any],
        query: QuerySpec,
        summary: EvidenceSummary,
        cindex: CandidateIndex,
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
            if isinstance(raw_items, list):
                for item in raw_items:
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
                activities = self._fallback_day(query, day_idx, chosen_attractions, chosen_restaurants, used_ids)
            self._normalize_day_times(activities)
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
    ) -> str:
        preferred_ids = [cid for cid in summary.chosen_ids if cid in cindex.by_id]
        if not preferred_ids:
            preferred_ids = [c.candidate_id for c in candidate_pool[: self.config.llm.max_candidates]]
        preferred_ids = preferred_ids[: self.config.llm.max_candidates]

        candidate_lines = []
        for cid in preferred_ids:
            cand = cindex.by_id[cid]
            tag_str = ", ".join(cand.tags[:4]) if cand.tags else "none"
            candidate_lines.append(
                f"- id={cand.candidate_id}; type={cand.entity_type}; name={cand.name}; "
                f"price={cand.price:.2f}; tags={tag_str}; text={cand.text}"
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
        ]
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
            + "\n\nCandidate shortlist:\n"
            + "\n".join(candidate_lines)
            + "\n\nReturn JSON with this schema:\n"
            + json.dumps(output_schema, ensure_ascii=False, indent=2)
        )

    def _parse_llm_json(self, text: str) -> dict[str, Any]:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE | re.DOTALL).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
            if not match:
                raise
            return json.loads(match.group(0))

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
