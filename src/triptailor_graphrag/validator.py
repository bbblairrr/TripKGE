from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .types import Candidate, PlanResult, QuerySpec
from .utils import haversine_km, intervals_overlap, parse_time_range


@dataclass
class ValidationReport:
    passed: bool
    errors: list[str]
    checks: dict[str, Any]


class PlanValidator:
    def validate(self, query: QuerySpec, plan: PlanResult, candidate_map: dict[str, Candidate]) -> ValidationReport:
        errors: list[str] = []

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

        checks = {
            "sandbox_ok": sandbox_ok,
            "budget_ok": budget_ok,
            "meal_ok": meal_ok,
            "time_ok": time_ok,
            "dedup_ok": dedup_ok,
            "route_ok": route_ok,
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

        used = {
            act.get("candidate_id")
            for day_items in plan.itinerary.values()
            for act in day_items
            if isinstance(act, dict) and act.get("candidate_id")
        }

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
                    used.add(repl.candidate_id)

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

        # 3) Budget repair by replacing expensive activities with cheaper options.
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
