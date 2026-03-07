from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .types import Candidate, PlanResult, QuerySpec
from .utils import format_time_range, haversine_km, intervals_overlap, parse_duration_minutes, parse_opening_hours, parse_time_range


@dataclass
class ValidationReport:
    passed: bool
    errors: list[str]
    checks: dict[str, Any]


class PlanValidator:
    def __init__(self) -> None:
        self.default_day_start = 9 * 60
        self.default_day_end = 22 * 60
        self.transport_buffer_minutes = 30
        self.activity_gap_minutes = 15

    def validate(self, query: QuerySpec, plan: PlanResult, candidate_map: dict[str, Candidate]) -> ValidationReport:
        errors: list[str] = []

        content_ok = self._check_required_content(query, plan)
        if not content_ok:
            errors.append("missing_required_content")

        sandbox_ok = self._check_sandbox(plan, candidate_map)
        if not sandbox_ok:
            errors.append("sandbox_violation")

        budget_ok, budget_details = self._check_budget(query, plan)
        if not budget_ok:
            errors.append("budget_exceeded")

        meal_ok = self._check_meal_range(query, plan, candidate_map)
        if not meal_ok:
            errors.append("meal_range_violation")

        time_ok = self._check_temporal(plan)
        if not time_ok:
            errors.append("temporal_conflict")

        dedup_ok = self._check_dedup(plan)
        if not dedup_ok:
            errors.append("duplicate_locations")

        route_ok, route_distance = self._check_route_distance(plan, candidate_map)
        if not route_ok:
            errors.append("route_distance_unreasonable")

        opening_ok = self._check_opening_hours(plan, candidate_map)
        if not opening_ok:
            errors.append("opening_hours_violation")

        stay_ok = self._check_stay_duration(plan, candidate_map)
        if not stay_ok:
            errors.append("stay_duration_violation")

        transport_ok = self._check_transport_windows(query, plan)
        if not transport_ok:
            errors.append("transport_time_violation")

        checks = {
            "content_ok": content_ok,
            "sandbox_ok": sandbox_ok,
            "budget_ok": budget_ok,
            "meal_ok": meal_ok,
            "time_ok": time_ok,
            "dedup_ok": dedup_ok,
            "route_ok": route_ok,
            "opening_hours_ok": opening_ok,
            "stay_duration_ok": stay_ok,
            "transport_time_ok": transport_ok,
            "budget_details": budget_details,
            "route_distance_km": round(route_distance, 2),
        }
        return ValidationReport(passed=not errors, errors=errors, checks=checks)

    def repair_once(
        self,
        query: QuerySpec,
        plan: PlanResult,
        ranked_candidates: list[Candidate],
        candidate_map: dict[str, Candidate],
    ) -> PlanResult:
        ranked_by_type: dict[str, list[Candidate]] = {"attraction": [], "restaurant": [], "hotel": []}
        for c in ranked_candidates:
            if c.entity_type in ranked_by_type:
                ranked_by_type[c.entity_type].append(c)

        # 1) Meal-range repair.
        if query.meal_price_range:
            lo, hi = query.meal_price_range
            restaurant_pool = [c for c in ranked_by_type["restaurant"] if lo <= c.price <= hi]
            restaurant_pool = restaurant_pool or ranked_by_type["restaurant"]
            cursor = 0
            for day_key, items in plan.itinerary.items():
                for idx, act in enumerate(items):
                    if act.get("action") != "dining":
                        continue
                    cid = act.get("candidate_id")
                    cand = candidate_map.get(cid)
                    if cand and lo <= cand.price <= hi:
                        continue
                    if not restaurant_pool:
                        continue
                    repl = restaurant_pool[cursor % len(restaurant_pool)]
                    cursor += 1
                    items[idx] = {
                        "time": act.get("time"),
                        "location": repl.name,
                        "price": round(repl.price, 2),
                        "action": "dining",
                        "candidate_id": repl.candidate_id,
                    }

        # 2) Duplicate repair.
        for day_key, items in plan.itinerary.items():
            seen: set[str] = set()
            for idx, act in enumerate(items):
                cid = act.get("candidate_id")
                if not cid:
                    continue
                if cid not in seen:
                    seen.add(cid)
                    continue
                action = act.get("action")
                ctype = "restaurant" if action == "dining" else "attraction"
                pool = ranked_by_type.get(ctype, [])
                repl = next((c for c in pool if c.candidate_id not in seen), None)
                if repl is None:
                    continue
                items[idx] = {
                    "time": act.get("time"),
                    "location": repl.name,
                    "price": round(repl.price, 2),
                    "action": action,
                    "candidate_id": repl.candidate_id,
                }
                seen.add(repl.candidate_id)

        # 3) Reschedule around transport windows, durations, and opening hours.
        self._repair_schedule_constraints(query, plan, ranked_by_type, candidate_map)

        # 4) Budget repair by replacing expensive activities with cheaper options.
        budget_ok, _ = self._check_budget(query, plan)
        if not budget_ok and query.budget:
            for day_key, items in plan.itinerary.items():
                for idx, act in sorted(enumerate(items), key=lambda x: x[1].get("price", 0), reverse=True):
                    ctype = "restaurant" if act.get("action") == "dining" else "attraction"
                    pool = ranked_by_type.get(ctype, [])
                    cheaper = next((c for c in pool if c.price < float(act.get("price") or 0)), None)
                    if cheaper is None:
                        continue
                    items[idx] = {
                        "time": act.get("time"),
                        "location": cheaper.name,
                        "price": round(cheaper.price, 2),
                        "action": act.get("action"),
                        "candidate_id": cheaper.candidate_id,
                        }
                    budget_ok, _ = self._check_budget(query, plan)
                    if budget_ok:
                        break
                if budget_ok:
                    break

        # 5) One more schedule pass after replacements.
        self._repair_schedule_constraints(query, plan, ranked_by_type, candidate_map)

        return plan

    def _check_sandbox(self, plan: PlanResult, candidate_map: dict[str, Candidate]) -> bool:
        for day_items in plan.itinerary.values():
            for act in day_items:
                cid = act.get("candidate_id")
                if not cid or cid not in candidate_map:
                    return False
        for hotel in plan.hotel:
            cid = hotel.get("candidate_id")
            if cid and cid not in candidate_map:
                return False
        return True

    def _check_required_content(self, query: QuerySpec, plan: PlanResult) -> bool:
        has_hotel = bool(plan.hotel)
        activity_count = sum(len(day_items) for day_items in plan.itinerary.values())
        has_activity = activity_count > 0
        requires_transport = bool(query.departure_city and query.destination_city)
        if requires_transport:
            outbound = self._transport_for_direction(query, plan.transportation, "outbound")
            inbound = self._transport_for_direction(query, plan.transportation, "inbound")
            has_transport = outbound is not None and inbound is not None
        else:
            has_transport = True
        return has_hotel and has_activity and has_transport

    def _check_budget(self, query: QuerySpec, plan: PlanResult) -> tuple[bool, dict[str, float]]:
        hotel_total = sum(float(x.get("price_per_night") or 0.0) for x in plan.hotel) * max(1, query.day)
        transport_total = sum(float(x.get("price") or 0.0) for x in plan.transportation)
        activity_total = sum(
            float(act.get("price") or 0.0)
            for day_items in plan.itinerary.values()
            for act in day_items
        )
        total = hotel_total + transport_total + activity_total
        if query.budget is None:
            return True, {
                "total": total,
                "budget": -1.0,
                "hotel": hotel_total,
                "transport": transport_total,
                "activity": activity_total,
            }
        return total <= query.budget, {
            "total": total,
            "budget": query.budget,
            "hotel": hotel_total,
            "transport": transport_total,
            "activity": activity_total,
        }

    def _check_meal_range(self, query: QuerySpec, plan: PlanResult, candidate_map: dict[str, Candidate]) -> bool:
        if query.meal_price_range is None:
            return True
        lo, hi = query.meal_price_range
        for day_items in plan.itinerary.values():
            for act in day_items:
                if act.get("action") != "dining":
                    continue
                cid = act.get("candidate_id")
                if cid in candidate_map:
                    price = candidate_map[cid].price
                else:
                    price = float(act.get("price") or 0.0)
                if not (lo <= price <= hi):
                    return False
        return True

    def _check_temporal(self, plan: PlanResult) -> bool:
        for day_items in plan.itinerary.values():
            intervals: list[tuple[int, int]] = []
            for act in day_items:
                time_text = act.get("time")
                parsed = parse_time_range(time_text)
                if parsed is None:
                    continue
                for existing in intervals:
                    if intervals_overlap(parsed, existing):
                        return False
                intervals.append(parsed)
        return True

    def _check_dedup(self, plan: PlanResult) -> bool:
        for day_items in plan.itinerary.values():
            seen: set[str] = set()
            for act in day_items:
                cid = act.get("candidate_id")
                if not cid:
                    continue
                if cid in seen:
                    return False
                seen.add(cid)
        return True

    def _check_route_distance(self, plan: PlanResult, candidate_map: dict[str, Candidate]) -> tuple[bool, float]:
        total = 0.0
        day_count = 0
        for day_items in plan.itinerary.values():
            coords = []
            for act in day_items:
                cid = act.get("candidate_id")
                c = candidate_map.get(cid)
                if not c:
                    continue
                if c.latitude is None or c.longitude is None:
                    continue
                coords.append((c.latitude, c.longitude))
            if len(coords) < 2:
                continue
            day_count += 1
            for i in range(1, len(coords)):
                total += haversine_km(coords[i - 1][0], coords[i - 1][1], coords[i][0], coords[i][1])
        avg_daily = total / max(1, day_count)
        return avg_daily <= 85.0, total

    def _check_opening_hours(self, plan: PlanResult, candidate_map: dict[str, Candidate]) -> bool:
        for day_items in plan.itinerary.values():
            for act in day_items:
                if act.get("action") != "sightseeing":
                    continue
                interval = parse_time_range(str(act.get("time") or ""))
                cand = candidate_map.get(act.get("candidate_id"))
                if interval is None or cand is None:
                    continue
                opening = parse_opening_hours(str(cand.meta.get("opening_hours") if isinstance(cand.meta, dict) else ""))
                if opening is None:
                    continue
                if interval[0] < opening[0] or interval[1] > opening[1]:
                    return False
        return True

    def _check_stay_duration(self, plan: PlanResult, candidate_map: dict[str, Candidate]) -> bool:
        for day_items in plan.itinerary.values():
            for act in day_items:
                interval = parse_time_range(str(act.get("time") or ""))
                required = self._required_duration_minutes(act, candidate_map)
                if interval is None or required is None:
                    continue
                if interval[1] - interval[0] < required:
                    return False
        return True

    def _check_transport_windows(self, query: QuerySpec, plan: PlanResult) -> bool:
        checks: list[bool] = []
        outbound = self._transport_for_direction(query, plan.transportation, "outbound")
        inbound = self._transport_for_direction(query, plan.transportation, "inbound")

        first_day_items = plan.itinerary.get("day_1", [])
        first_start = min(
            (interval[0] for interval in (parse_time_range(str(act.get("time") or "")) for act in first_day_items) if interval is not None),
            default=None,
        )
        if outbound is not None and first_start is not None:
            interval = parse_time_range(str(outbound.get("time") or ""))
            if interval is not None:
                checks.append(interval[1] + self.transport_buffer_minutes <= first_start)

        last_day_items = plan.itinerary.get(f"day_{query.day}", [])
        last_end = max(
            (interval[1] for interval in (parse_time_range(str(act.get("time") or "")) for act in last_day_items) if interval is not None),
            default=None,
        )
        if inbound is not None and last_end is not None:
            interval = parse_time_range(str(inbound.get("time") or ""))
            if interval is not None:
                checks.append(last_end + self.transport_buffer_minutes <= interval[0])

        return all(checks) if checks else True

    def _repair_schedule_constraints(
        self,
        query: QuerySpec,
        plan: PlanResult,
        ranked_by_type: dict[str, list[Candidate]],
        candidate_map: dict[str, Candidate],
    ) -> None:
        used_trip: set[str] = set()
        for day_idx in range(1, query.day + 1):
            day_key = f"day_{day_idx}"
            items = plan.itinerary.get(day_key, [])
            start, end = self._day_time_bounds(query, day_idx, plan.transportation)
            if end - start < 45:
                plan.itinerary[day_key] = []
                continue

            current = start
            seen: set[str] = set()
            repaired: list[dict[str, Any]] = []
            for act in items:
                action = "dining" if act.get("action") == "dining" else "sightseeing"
                candidate = candidate_map.get(act.get("candidate_id"))
                if candidate is None or candidate.candidate_id in seen:
                    candidate = self._replacement_candidate(
                        ranked_by_type,
                        candidate_map,
                        action,
                        seen.union(used_trip),
                        current,
                        end,
                    )
                elif self._candidate_interval(candidate, action, current, end) is None:
                    blocked = seen.union(used_trip)
                    blocked.add(candidate.candidate_id)
                    candidate = self._replacement_candidate(
                        ranked_by_type,
                        candidate_map,
                        action,
                        blocked,
                        current,
                        end,
                    )
                if candidate is None:
                    continue
                interval = self._candidate_interval(candidate, action, current, end)
                if interval is None:
                    continue
                start_min, end_min = interval
                repaired.append(
                    {
                        "time": format_time_range(start_min, end_min),
                        "location": candidate.name,
                        "price": round(candidate.price, 2),
                        "action": action,
                        "candidate_id": candidate.candidate_id,
                    }
                )
                current = min(end, end_min + self.activity_gap_minutes)
                seen.add(candidate.candidate_id)
                used_trip.add(candidate.candidate_id)
            plan.itinerary[day_key] = repaired

    def _replacement_candidate(
        self,
        ranked_by_type: dict[str, list[Candidate]],
        candidate_map: dict[str, Candidate],
        action: str,
        blocked_ids: set[str],
        earliest_start: int,
        latest_end: int,
    ) -> Candidate | None:
        ctype = "restaurant" if action == "dining" else "attraction"
        for candidate in ranked_by_type.get(ctype, []):
            if candidate.candidate_id in blocked_ids:
                continue
            if candidate.candidate_id not in candidate_map:
                continue
            if self._candidate_interval(candidate, action, earliest_start, latest_end) is None:
                continue
            return candidate
        return None

    def _transport_for_direction(
        self,
        query: QuerySpec,
        transportation: list[dict[str, Any]],
        direction: str,
    ) -> dict[str, Any] | None:
        outbound_route = f"{query.departure_city} to {query.destination_city}".strip().lower()
        inbound_route = f"{query.destination_city} to {query.departure_city}".strip().lower()
        for transport in transportation:
            route = str(transport.get("route") or "").strip().lower()
            if direction == "outbound" and route == outbound_route:
                return transport
            if direction == "inbound" and route == inbound_route:
                return transport
        return None

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

    def _candidate_interval(
        self,
        candidate: Candidate,
        action: str,
        earliest_start: int,
        latest_end: int,
    ) -> tuple[int, int] | None:
        duration = self._required_duration_minutes(
            {"action": action, "candidate_id": candidate.candidate_id},
            {candidate.candidate_id: candidate},
        )
        if duration is None:
            duration = 60 if action == "dining" else 90
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

    def _required_duration_minutes(self, act: dict[str, Any], candidate_map: dict[str, Candidate]) -> int | None:
        if act.get("action") == "dining":
            return 60
        cand = candidate_map.get(act.get("candidate_id"))
        if cand is None or not isinstance(cand.meta, dict):
            return None
        duration = parse_duration_minutes(str(cand.meta.get("recommended_duration") or ""))
        if duration is None:
            return None
        return max(30, min(duration, 8 * 60))
