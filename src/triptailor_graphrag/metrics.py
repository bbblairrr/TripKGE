from __future__ import annotations

import json
from dataclasses import dataclass
from statistics import mean
from typing import Any

from .data_loader import match_candidate_by_name
from .types import Candidate, PlanResult, QuerySpec
from .utils import haversine_km, normalize_text, tokenize


@dataclass
class SampleMetric:
    pid: int
    method: str
    values: dict[str, float]


def compute_sample_metrics(
    method: str,
    query: QuerySpec,
    plan: PlanResult,
    sample: dict[str, Any],
    candidate_pool: list[Candidate],
) -> SampleMetric:
    candidate_map = {c.candidate_id: c for c in candidate_pool}

    gt = json.loads(sample["final_plan_json"]) if sample.get("final_plan_json") else {}
    gt_ids = _extract_gt_ids(gt, candidate_pool)

    activity_ids = [
        act.get("candidate_id")
        for day_items in plan.itinerary.values()
        for act in day_items
        if isinstance(act, dict) and act.get("candidate_id")
    ]
    selected_ids = list(activity_ids)
    selected_ids.extend([h.get("candidate_id") for h in plan.hotel if h.get("candidate_id")])

    total_acts = max(1, len(activity_ids))
    in_sandbox = sum(1 for cid in activity_ids if cid in candidate_map)
    in_evidence = sum(1 for cid in activity_ids if cid in set(plan.evidence_ids))

    faithfulness = in_sandbox / total_acts
    evidence_grounding = in_evidence / total_acts

    context_precision = _safe_div(len(set(selected_ids).intersection(set(plan.evidence_ids))), len(set(selected_ids)))
    context_recall = _safe_div(len(set(plan.evidence_ids).intersection(gt_ids)), len(gt_ids))

    route_pred = _route_distance_from_plan(plan, candidate_map)
    route_gt = _route_distance_from_gt(gt, candidate_pool)
    route_distance_ratio = route_pred / route_gt if route_gt > 0 else 1.0

    personalization_proxy = _personalization_proxy(query, plan, candidate_map)
    answer_relevancy = _answer_relevancy(query, plan, candidate_map)

    values = {
        "feasibility_pass_rate": 1.0 if plan.validator_report.get("passed") else 0.0,
        "personalization_proxy": personalization_proxy,
        "route_distance_ratio": route_distance_ratio,
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_precision": context_precision,
        "context_recall": context_recall,
        "evidence_grounding_rate": evidence_grounding,
    }

    return SampleMetric(pid=query.pid, method=method, values=values)


def aggregate_metrics(rows: list[SampleMetric]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted(rows[0].values.keys())
    out: dict[str, float] = {}
    for key in keys:
        out[key] = mean(row.values[key] for row in rows)
    return out


def _extract_gt_ids(gt: dict[str, Any], candidate_pool: list[Candidate]) -> set[str]:
    ids: set[str] = set()
    for hotel in gt.get("hotel", []):
        c = match_candidate_by_name(candidate_pool, hotel.get("name", ""))
        if c:
            ids.add(c.candidate_id)
    itinerary = gt.get("itinerary", {})
    if isinstance(itinerary, dict):
        for _, acts in itinerary.items():
            for act in acts:
                c = match_candidate_by_name(candidate_pool, act.get("location", ""))
                if c:
                    ids.add(c.candidate_id)
    return ids


def _route_distance_from_plan(plan: PlanResult, candidate_map: dict[str, Candidate]) -> float:
    total = 0.0
    for _, acts in plan.itinerary.items():
        coords = []
        for act in acts:
            c = candidate_map.get(act.get("candidate_id"))
            if c and c.latitude is not None and c.longitude is not None:
                coords.append((c.latitude, c.longitude))
        for i in range(1, len(coords)):
            total += haversine_km(coords[i - 1][0], coords[i - 1][1], coords[i][0], coords[i][1])
    return total


def _route_distance_from_gt(gt: dict[str, Any], candidate_pool: list[Candidate]) -> float:
    total = 0.0
    itinerary = gt.get("itinerary", {}) if isinstance(gt, dict) else {}
    for _, acts in itinerary.items():
        coords = []
        for act in acts:
            c = match_candidate_by_name(candidate_pool, act.get("location", ""))
            if c and c.latitude is not None and c.longitude is not None:
                coords.append((c.latitude, c.longitude))
        for i in range(1, len(coords)):
            total += haversine_km(coords[i - 1][0], coords[i - 1][1], coords[i][0], coords[i][1])
    return total


def _personalization_proxy(query: QuerySpec, plan: PlanResult, candidate_map: dict[str, Candidate]) -> float:
    score = 0.0

    meal_ok = 1.0
    if query.meal_price_range:
        lo, hi = query.meal_price_range
        for day_items in plan.itinerary.values():
            for act in day_items:
                if act.get("action") != "dining":
                    continue
                c = candidate_map.get(act.get("candidate_id"))
                price = c.price if c else float(act.get("price") or 0.0)
                if not (lo <= price <= hi):
                    meal_ok = 0.0
                    break
            if meal_ok == 0.0:
                break
    score += 0.35 * meal_ok

    hotel_ok = 1.0
    if query.hotel_category_pref and plan.hotel:
        hid = plan.hotel[0].get("candidate_id")
        h = candidate_map.get(hid)
        category = normalize_text(str(h.meta.get("category") if h else ""))
        hotel_ok = 1.0 if normalize_text(query.hotel_category_pref) in category else 0.0
    score += 0.2 * hotel_ok

    # Interest coverage by tags.
    interest_cov = 0.0
    if query.interest_tags:
        wanted = {normalize_text(x) for x in query.interest_tags}
        tags = set()
        for day_items in plan.itinerary.values():
            for act in day_items:
                c = candidate_map.get(act.get("candidate_id"))
                if c:
                    tags.update(normalize_text(t) for t in c.tags)
        if wanted:
            overlap = len(wanted.intersection(tags))
            interest_cov = overlap / len(wanted)
    else:
        interest_cov = 1.0
    score += 0.3 * interest_cov

    # Intensity fit via avg activities/day.
    intensity_fit = 1.0
    avg_act = _safe_div(
        sum(len(items) for items in plan.itinerary.values()),
        len(plan.itinerary) if plan.itinerary else 1,
    )
    if query.intensity_pref == "low":
        intensity_fit = 1.0 if avg_act <= 3 else 0.5
    elif query.intensity_pref == "moderate":
        intensity_fit = 1.0 if 3 <= avg_act <= 5 else 0.6
    elif query.intensity_pref == "high":
        intensity_fit = 1.0 if avg_act >= 5 else 0.6
    score += 0.15 * intensity_fit

    return min(1.0, max(0.0, score))


def _answer_relevancy(query: QuerySpec, plan: PlanResult, candidate_map: dict[str, Candidate]) -> float:
    qtok = set(tokenize(query.query_text))
    if not qtok:
        return 0.0

    selected_text = []
    for day_items in plan.itinerary.values():
        for act in day_items:
            c = candidate_map.get(act.get("candidate_id"))
            if c:
                selected_text.append(c.text)

    dtok = set(tokenize(" ".join(selected_text)))
    inter = len(qtok.intersection(dtok))
    union = len(qtok.union(dtok))
    return _safe_div(inter, union)


def _safe_div(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return a / b
