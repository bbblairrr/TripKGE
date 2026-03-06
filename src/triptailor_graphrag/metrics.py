from __future__ import annotations

import json
import math
from dataclasses import dataclass
from statistics import mean
from typing import Any

from .data_loader import match_candidate_by_name
from .preference_judge import PreferenceJudge
from .semantic import cosine_similarity
from .types import Candidate, PlanResult, QuerySpec
from .utils import (
    haversine_km,
    normalize_text,
    parse_duration_minutes,
    parse_opening_hours,
    parse_time_range,
    tokenize,
    unique_keep_order,
)


RETRIEVAL_KS = (5, 10)
TRANSPORT_BUFFER_MINUTES = 30


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
    info: dict[str, Any] | None = None,
    preference_judge: PreferenceJudge | None = None,
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
    selected_ids = unique_keep_order([cid for cid in selected_ids if cid])

    schema_validity = _schema_validity_rate(plan)
    entity_grounding = _entity_grounding_rate(plan, candidate_map)
    transport_grounding = _transport_grounding_rate(query, plan, info or {})
    hallucination_free = _hallucination_free(query, plan, candidate_map, info or {})

    total_acts = max(1, len(activity_ids))
    in_sandbox = sum(1 for cid in activity_ids if cid in candidate_map)
    in_evidence = sum(1 for cid in activity_ids if cid in set(plan.evidence_ids))

    faithfulness = in_sandbox / total_acts
    evidence_grounding = in_evidence / total_acts

    evidence_ids = unique_keep_order([cid for cid in plan.evidence_ids if cid])
    context_precision = _safe_div(len(set(selected_ids).intersection(set(evidence_ids))), len(set(selected_ids)))
    context_recall = _safe_div(len(set(evidence_ids).intersection(gt_ids)), len(gt_ids))
    ranked_metrics = _ranking_metrics(evidence_ids, gt_ids, RETRIEVAL_KS)

    plan_day_routes = _route_distances_by_day_from_plan(plan, candidate_map)
    gt_day_routes = _route_distances_by_day_from_gt(gt, candidate_pool)
    avg_route_ratio = _average_route_distance_ratio(plan_day_routes, gt_day_routes)
    max_single_day_route = max(plan_day_routes) if plan_day_routes else 0.0

    opening_hours_compliance = _opening_hours_compliance(plan, candidate_map)
    stay_duration_feasibility = _stay_duration_feasibility(plan, candidate_map)
    transport_time_feasibility = _transport_time_feasibility(query, plan)
    personalization_proxy = _personalization_proxy(
        query=query,
        plan=plan,
        gt=gt,
        candidate_map=candidate_map,
        preference_judge=preference_judge,
    )
    answer_relevancy = _answer_relevancy(query, plan, candidate_map)

    values = {
        "feasibility_pass_rate": 1.0 if hallucination_free else 0.0,
        "constraint_satisfaction_rate": 1.0 if plan.validator_report.get("passed") else 0.0,
        "schema_validity_rate": schema_validity,
        "entity_grounding_rate": entity_grounding,
        "transport_grounding_rate": transport_grounding,
        "opening_hours_compliance": opening_hours_compliance,
        "stay_duration_feasibility": stay_duration_feasibility,
        "transport_time_feasibility": transport_time_feasibility,
        "personalization_proxy": personalization_proxy,
        "average_route_distance_ratio": avg_route_ratio,
        "max_single_day_route_km": max_single_day_route,
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_precision": context_precision,
        "context_recall": context_recall,
        "evidence_grounding_rate": evidence_grounding,
    }
    values.update(ranked_metrics)

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


def _schema_validity_rate(plan: PlanResult) -> float:
    checks: list[bool] = [
        isinstance(plan.hotel, list),
        isinstance(plan.transportation, list),
        isinstance(plan.itinerary, dict),
    ]
    checks.extend(_is_valid_hotel_item(item) for item in plan.hotel)
    checks.extend(_is_valid_transport_item(item) for item in plan.transportation)
    for day_key, items in plan.itinerary.items():
        checks.append(isinstance(day_key, str) and day_key.startswith("day_"))
        checks.append(isinstance(items, list))
        if isinstance(items, list):
            checks.extend(_is_valid_activity_item(item) for item in items)
    return _safe_div(sum(1 for check in checks if check), len(checks))


def _is_valid_hotel_item(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    required = ("day", "name", "price_per_night", "candidate_id")
    return all(item.get(key) not in (None, "") for key in required)


def _is_valid_transport_item(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    required = ("day", "mode", "route", "number", "time", "price")
    return all(item.get(key) not in (None, "") for key in required)


def _is_valid_activity_item(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    required = ("time", "location", "price", "action", "candidate_id")
    return all(item.get(key) not in (None, "") for key in required)


def _entity_grounding_rate(plan: PlanResult, candidate_map: dict[str, Candidate]) -> float:
    total = 0
    grounded = 0
    for hotel in plan.hotel:
        total += 1
        grounded += 1 if _grounded_hotel_item(hotel, candidate_map) else 0
    for day_items in plan.itinerary.values():
        for act in day_items:
            total += 1
            grounded += 1 if _grounded_activity_item(act, candidate_map) else 0
    return _safe_div(grounded, total)


def _transport_grounding_rate(query: QuerySpec, plan: PlanResult, info: dict[str, Any]) -> float:
    transport_blocks = _transport_blocks(query, info)
    expected = {key for key, value in transport_blocks.items() if _block_has_options(value["block"])}
    matched: set[str] = set()
    for tr in plan.transportation:
        direction = _transport_direction(query, tr)
        if direction is None:
            continue
        spec = transport_blocks[direction]
        if _transport_matches_block(tr, spec["block"], spec["from"], spec["to"], spec["day"]):
            matched.add(direction)
    denominator = max(len(expected), len(plan.transportation), 1)
    if not expected and not plan.transportation:
        return 1.0
    return _safe_div(len(matched), denominator)


def _opening_hours_compliance(plan: PlanResult, candidate_map: dict[str, Candidate]) -> float:
    checked = 0
    passed = 0
    for day_items in plan.itinerary.values():
        for act in day_items:
            if act.get("action") != "sightseeing":
                continue
            interval = _activity_interval(act)
            cand = candidate_map.get(act.get("candidate_id"))
            opening = parse_opening_hours(str(cand.meta.get("opening_hours") if cand else ""))
            if interval is None or opening is None:
                continue
            checked += 1
            if opening[0] <= interval[0] and interval[1] <= opening[1]:
                passed += 1
    return 1.0 if checked == 0 else _safe_div(passed, checked)


def _stay_duration_feasibility(plan: PlanResult, candidate_map: dict[str, Candidate]) -> float:
    checked = 0
    passed = 0
    for day_items in plan.itinerary.values():
        for act in day_items:
            interval = _activity_interval(act)
            required = _required_duration_minutes(act, candidate_map)
            if interval is None or required is None:
                continue
            checked += 1
            if interval[1] - interval[0] >= required:
                passed += 1
    return 1.0 if checked == 0 else _safe_div(passed, checked)


def _transport_time_feasibility(query: QuerySpec, plan: PlanResult) -> float:
    checks: list[bool] = []
    outbound = None
    inbound = None
    for tr in plan.transportation:
        direction = _transport_direction(query, tr)
        if direction == "outbound":
            outbound = tr
        elif direction == "inbound":
            inbound = tr

    first_day_items = plan.itinerary.get("day_1", [])
    first_start = min(
        (interval[0] for interval in (_activity_interval(act) for act in first_day_items) if interval is not None),
        default=None,
    )
    if outbound is not None and first_start is not None:
        transport_interval = parse_time_range(str(outbound.get("time") or ""))
        if transport_interval is not None:
            checks.append(transport_interval[1] + TRANSPORT_BUFFER_MINUTES <= first_start)

    last_day_key = f"day_{query.day}"
    last_day_items = plan.itinerary.get(last_day_key, [])
    last_end = max(
        (interval[1] for interval in (_activity_interval(act) for act in last_day_items) if interval is not None),
        default=None,
    )
    if inbound is not None and last_end is not None:
        transport_interval = parse_time_range(str(inbound.get("time") or ""))
        if transport_interval is not None:
            checks.append(last_end + TRANSPORT_BUFFER_MINUTES <= transport_interval[0])

    return 1.0 if not checks else _safe_div(sum(1 for check in checks if check), len(checks))


def _route_distances_by_day_from_plan(plan: PlanResult, candidate_map: dict[str, Candidate]) -> list[float]:
    return [_route_distance_from_acts(acts, candidate_map) for acts in plan.itinerary.values()]


def _route_distances_by_day_from_gt(gt: dict[str, Any], candidate_pool: list[Candidate]) -> list[float]:
    distances: list[float] = []
    itinerary = gt.get("itinerary", {}) if isinstance(gt, dict) else {}
    for _, acts in itinerary.items():
        candidate_map = {}
        for act in acts:
            c = match_candidate_by_name(candidate_pool, act.get("location", ""))
            if c:
                candidate_map[c.candidate_id] = c
        normalized_acts = []
        for act in acts:
            c = match_candidate_by_name(candidate_pool, act.get("location", ""))
            if c:
                normalized_acts.append({"candidate_id": c.candidate_id})
        distances.append(_route_distance_from_acts(normalized_acts, candidate_map))
    return distances


def _route_distance_from_acts(acts: list[dict[str, Any]], candidate_map: dict[str, Candidate]) -> float:
    total = 0.0
    coords = []
    for act in acts:
        c = candidate_map.get(act.get("candidate_id"))
        if c and c.latitude is not None and c.longitude is not None:
            coords.append((c.latitude, c.longitude))
    for i in range(1, len(coords)):
        total += haversine_km(coords[i - 1][0], coords[i - 1][1], coords[i][0], coords[i][1])
    return total


def _average_route_distance_ratio(plan_day_routes: list[float], gt_day_routes: list[float]) -> float:
    if not gt_day_routes:
        return 1.0
    plan_avg = sum(plan_day_routes) / max(1, len(plan_day_routes))
    gt_avg = sum(gt_day_routes) / max(1, len(gt_day_routes))
    return plan_avg / gt_avg if gt_avg > 0 else 1.0


def _personalization_proxy(
    query: QuerySpec,
    plan: PlanResult,
    gt: dict[str, Any],
    candidate_map: dict[str, Candidate],
    preference_judge: PreferenceJudge | None = None,
) -> float:
    if preference_judge is not None:
        judge_result = preference_judge.score(query, plan, gt, candidate_map)
        if judge_result is not None:
            return judge_result.score
    return _heuristic_personalization_proxy(query, plan, candidate_map)


def _heuristic_personalization_proxy(query: QuerySpec, plan: PlanResult, candidate_map: dict[str, Candidate]) -> float:
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
    plan_text = _plan_text(plan, candidate_map)
    if not query.query_text or not plan_text:
        return 0.0
    try:
        return max(0.0, min(1.0, (cosine_similarity(query.query_text, plan_text) + 1.0) / 2.0))
    except Exception:
        qtok = set(tokenize(query.query_text))
        dtok = set(tokenize(plan_text))
        inter = len(qtok.intersection(dtok))
        union = len(qtok.union(dtok))
        return _safe_div(inter, union)


def _plan_text(plan: PlanResult, candidate_map: dict[str, Candidate]) -> str:
    parts: list[str] = []
    for hotel in plan.hotel:
        cid = hotel.get("candidate_id")
        if cid and cid in candidate_map:
            parts.append(candidate_map[cid].text)
        else:
            parts.append(str(hotel.get("name") or ""))
    for tr in plan.transportation:
        parts.append(
            " ".join(
                str(tr.get(key) or "")
                for key in ("mode", "route", "number", "time", "price")
            ).strip()
        )
    for day_items in plan.itinerary.values():
        for act in day_items:
            c = candidate_map.get(act.get("candidate_id"))
            if c:
                parts.append(c.text)
            else:
                parts.append(
                    " ".join(
                        str(act.get(key) or "")
                        for key in ("time", "location", "action", "price")
                    ).strip()
                )
    return " ".join(part for part in parts if part)


def _ranking_metrics(evidence_ids: list[str], gt_ids: set[str], ks: tuple[int, ...]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for k in ks:
        topk = evidence_ids[:k]
        metrics[f"recall_at_{k}"] = _safe_div(len(set(topk).intersection(gt_ids)), len(gt_ids))
        metrics[f"ndcg_at_{k}"] = _ndcg_at_k(topk, gt_ids, k)
    return metrics


def _ndcg_at_k(ranked_ids: list[str], gt_ids: set[str], k: int) -> float:
    if not gt_ids or k <= 0:
        return 0.0
    dcg = 0.0
    for idx, cid in enumerate(ranked_ids[:k]):
        rel = 1.0 if cid in gt_ids else 0.0
        if rel > 0:
            dcg += (2**rel - 1) / math.log2(idx + 2)
    ideal_hits = min(len(gt_ids), k)
    idcg = sum(1.0 / math.log2(idx + 2) for idx in range(ideal_hits))
    return _safe_div(dcg, idcg)


def _hallucination_free(
    query: QuerySpec,
    plan: PlanResult,
    candidate_map: dict[str, Candidate],
    info: dict[str, Any],
) -> bool:
    return (
        _entity_grounding_rate(plan, candidate_map) >= 1.0
        and _grounded_transport(query, plan, info)
    )


def _grounded_hotel_item(hotel: dict[str, Any], candidate_map: dict[str, Candidate]) -> bool:
    cid = hotel.get("candidate_id")
    if not cid or cid not in candidate_map:
        return False
    cand = candidate_map[cid]
    if cand.entity_type != "hotel":
        return False
    return _text_matches_candidate(str(hotel.get("name") or ""), cand)


def _grounded_activity_item(act: dict[str, Any], candidate_map: dict[str, Candidate]) -> bool:
    cid = act.get("candidate_id")
    if not cid or cid not in candidate_map:
        return False
    cand = candidate_map[cid]
    expected_type = "restaurant" if act.get("action") == "dining" else "attraction"
    if cand.entity_type != expected_type:
        return False
    return _text_matches_candidate(str(act.get("location") or ""), cand)


def _grounded_transport(query: QuerySpec, plan: PlanResult, info: dict[str, Any]) -> bool:
    transport_blocks = _transport_blocks(query, info)
    matches = {"outbound": False, "inbound": False}
    for tr in plan.transportation:
        direction = _transport_direction(query, tr)
        if direction is None:
            return False
        spec = transport_blocks[direction]
        if not _transport_matches_block(tr, spec["block"], spec["from"], spec["to"], spec["day"]):
            return False
        matches[direction] = True

    for direction, spec in transport_blocks.items():
        if _block_has_options(spec["block"]) and not matches[direction]:
            return False
    return True


def _transport_blocks(query: QuerySpec, info: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        "outbound": {
            "from": query.departure_city,
            "to": query.destination_city,
            "day": 1,
            "block": info.get("transport_otd", {}),
        },
        "inbound": {
            "from": query.destination_city,
            "to": query.departure_city,
            "day": query.day,
            "block": info.get("transport_dto", {}),
        },
    }


def _transport_direction(query: QuerySpec, transport: dict[str, Any]) -> str | None:
    route = normalize_text(str(transport.get("route") or ""))
    outbound = normalize_text(f"{query.departure_city} to {query.destination_city}")
    inbound = normalize_text(f"{query.destination_city} to {query.departure_city}")
    if route == outbound or int(transport.get("day") or 0) == 1:
        return "outbound"
    if route == inbound or int(transport.get("day") or 0) == query.day:
        return "inbound"
    return None


def _transport_matches_block(
    transport: dict[str, Any],
    block: dict[str, Any],
    from_city: str,
    to_city: str,
    day: int,
) -> bool:
    mode = normalize_text(str(transport.get("mode") or ""))
    route = normalize_text(str(transport.get("route") or ""))
    number = normalize_text(str(transport.get("number") or ""))
    time_text = normalize_text(str(transport.get("time") or ""))
    price = float(transport.get("price") or 0.0)
    expected_route = normalize_text(f"{from_city} to {to_city}")

    if route != expected_route or int(transport.get("day") or 0) != day:
        return False

    options = []
    if mode == "train":
        options.extend(
            {
                "number": normalize_text(str(row.get("Train_Number") or "")),
                "time": normalize_text(f"{row.get('Departure_Time') or ''}-{row.get('Arrival_Time') or ''}".strip("-")),
                "price": float(row.get("Second_Class_Price") or 0.0),
            }
            for row in (block.get("train_options") or [])
        )
    elif mode == "flight":
        options.extend(
            {
                "number": normalize_text(str(row.get("Flight Number") or row.get("Flight_Number") or "")),
                "time": normalize_text(f"{row.get('Departure Time') or row.get('Departure_Time') or ''}-{row.get('Arrival Time') or row.get('Arrival_Time') or ''}".strip("-")),
                "price": float(row.get("Price") or 0.0),
            }
            for row in (block.get("flight_options") or [])
        )
    else:
        return False

    for option in options:
        if number == option["number"] and time_text == option["time"] and abs(price - option["price"]) < 1e-6:
            return True
    return False


def _block_has_options(block: dict[str, Any]) -> bool:
    return bool((block.get("train_options") or []) or (block.get("flight_options") or []))


def _text_matches_candidate(text: str, candidate: Candidate) -> bool:
    normalized = normalize_text(text)
    if not normalized:
        return False
    names = {normalize_text(candidate.name)}
    raw = candidate.meta.get("name_raw") if isinstance(candidate.meta, dict) else None
    if raw:
        names.add(normalize_text(str(raw)))
    for name in names:
        if not name:
            continue
        if normalized == name or normalized in name or name in normalized:
            return True
    return False


def _activity_interval(act: dict[str, Any]) -> tuple[int, int] | None:
    return parse_time_range(str(act.get("time") or ""))


def _required_duration_minutes(act: dict[str, Any], candidate_map: dict[str, Candidate]) -> int | None:
    if act.get("action") == "dining":
        return 60
    cand = candidate_map.get(act.get("candidate_id"))
    if not cand:
        return None
    return parse_duration_minutes(str(cand.meta.get("recommended_duration") if isinstance(cand.meta, dict) else ""))


def _safe_div(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return a / b
