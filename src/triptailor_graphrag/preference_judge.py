from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from .local_llm import LocalLLMClient, LocalLLMError
from .types import Candidate, PlanResult, QuerySpec


@dataclass(frozen=True)
class PreferenceJudgeResult:
    score: float
    winner: str
    generated_score: float
    reference_score: float
    itinerary_a_score: float
    itinerary_b_score: float
    a_is_generated: bool
    rationale: str = ""


class PreferenceJudge:
    def __init__(self, client: LocalLLMClient | None = None) -> None:
        self.client = client

    def score(
        self,
        query: QuerySpec,
        generated_plan: PlanResult,
        reference_plan: dict[str, Any],
        candidate_map: dict[str, Candidate],
    ) -> PreferenceJudgeResult | None:
        if self.client is None:
            return None
        blind_pair = self._blind_pair(query, generated_plan, reference_plan, candidate_map)
        prompt = self._build_prompt(query, blind_pair["a_text"], blind_pair["b_text"])
        try:
            raw = self.client.generate(prompt, system_prompt=self._system_prompt())
        except LocalLLMError:
            return None
        try:
            payload = self._parse_json(raw)
        except (ValueError, json.JSONDecodeError):
            return None

        winner = str(payload.get("winner") or "tie").strip().lower()
        if winner not in {"a", "b", "tie"}:
            winner = "tie"
        itinerary_a_score = self._clamp_score(payload.get("itinerary_a_score"))
        itinerary_b_score = self._clamp_score(payload.get("itinerary_b_score"))
        generated_score = itinerary_a_score if blind_pair["a_is_generated"] else itinerary_b_score
        reference_score = itinerary_b_score if blind_pair["a_is_generated"] else itinerary_a_score
        if winner == "a":
            winner = "generated" if blind_pair["a_is_generated"] else "reference"
        elif winner == "b":
            winner = "reference" if blind_pair["a_is_generated"] else "generated"
        score = max(0.0, min(1.0, 0.5 + (generated_score - reference_score) / 20.0))
        rationale = str(payload.get("rationale") or "").strip()
        return PreferenceJudgeResult(
            score=score,
            winner=winner,
            generated_score=generated_score,
            reference_score=reference_score,
            itinerary_a_score=itinerary_a_score,
            itinerary_b_score=itinerary_b_score,
            a_is_generated=blind_pair["a_is_generated"],
            rationale=rationale,
        )

    def _system_prompt(self) -> str:
        return (
            "You are an itinerary evaluator. Compare two travel plans for the same user request. "
            "Focus only on which plan better fits the user's stated preferences and constraints. "
            "Output valid JSON only."
        )

    def _build_prompt(self, query: QuerySpec, itinerary_a_text: str, itinerary_b_text: str) -> str:
        request_block = self._request_block(query)
        output_schema = {
            "winner": "A or B or tie",
            "itinerary_a_score": "0-10",
            "itinerary_b_score": "0-10",
            "rationale": "short reason",
        }
        return (
            "Evaluate which itinerary is more suitable for the user.\n"
            "Do not assume either itinerary is preferred because of order or origin.\n\n"
            "User request:\n"
            + json.dumps(request_block, ensure_ascii=False, indent=2)
            + "\n\nItinerary A:\n"
            + itinerary_a_text
            + "\n\nItinerary B:\n"
            + itinerary_b_text
            + "\n\nReturn JSON with this schema:\n"
            + json.dumps(output_schema, ensure_ascii=False, indent=2)
        )

    def _request_block(self, query: QuerySpec) -> dict[str, Any]:
        return {
            "query_text": query.query_text,
            "trip_days": query.day,
            "budget": query.budget,
            "meal_price_range": query.meal_price_range,
            "hotel_category_pref": query.hotel_category_pref,
            "intensity_pref": query.intensity_pref,
            "interest_tags": query.interest_tags,
        }

    def _plan_to_text(self, plan: PlanResult, candidate_map: dict[str, Candidate]) -> str:
        parts: list[str] = []
        if plan.hotel:
            hotel_parts = []
            for item in plan.hotel:
                cid = item.get("candidate_id")
                name = item.get("name")
                price = item.get("price_per_night")
                if cid and cid in candidate_map:
                    name = candidate_map[cid].name
                hotel_parts.append(f"day {item.get('day')}: {name} ({price}/night)")
            parts.append("hotel: " + "; ".join(hotel_parts))
        if plan.transportation:
            transport_parts = []
            for item in plan.transportation:
                transport_parts.append(
                    f"day {item.get('day')}: {item.get('mode')} {item.get('route')} "
                    f"{item.get('number')} {item.get('time')} {item.get('price')}"
                )
            parts.append("transport: " + "; ".join(transport_parts))
        for day_key, acts in plan.itinerary.items():
            activity_parts = []
            for act in acts:
                cid = act.get("candidate_id")
                name = act.get("location")
                if cid and cid in candidate_map:
                    name = candidate_map[cid].name
                activity_parts.append(
                    f"{act.get('time')} {act.get('action')} {name} ({act.get('price')})"
                )
            parts.append(f"{day_key}: " + "; ".join(activity_parts))
        return "\n".join(parts)

    def _reference_plan_to_text(self, reference_plan: dict[str, Any]) -> str:
        parts: list[str] = []
        hotels = reference_plan.get("hotel", [])
        if isinstance(hotels, list) and hotels:
            parts.append(
                "hotel: "
                + "; ".join(
                    f"day {item.get('day')}: {item.get('name')} ({item.get('price_per_night')}/night)"
                    for item in hotels
                    if isinstance(item, dict)
                )
            )
        transports = reference_plan.get("transportation", [])
        if isinstance(transports, list) and transports:
            parts.append(
                "transport: "
                + "; ".join(
                    f"day {item.get('day')}: {item.get('mode')} {item.get('route')} "
                    f"{item.get('number')} {item.get('time')} {item.get('price')}"
                    for item in transports
                    if isinstance(item, dict)
                )
            )
        itinerary = reference_plan.get("itinerary", {})
        if isinstance(itinerary, dict):
            for day_key, acts in itinerary.items():
                if not isinstance(acts, list):
                    continue
                parts.append(
                    f"{day_key}: "
                    + "; ".join(
                        f"{item.get('time')} {item.get('action')} {item.get('location')} ({item.get('price')})"
                        for item in acts
                        if isinstance(item, dict)
                    )
                )
        return "\n".join(parts)

    def _blind_pair(
        self,
        query: QuerySpec,
        generated_plan: PlanResult,
        reference_plan: dict[str, Any],
        candidate_map: dict[str, Candidate],
    ) -> dict[str, Any]:
        seed = hashlib.sha256(f"{query.pid}:{query.query_text}".encode("utf-8")).digest()[0]
        a_is_generated = (seed % 2) == 0
        generated_text = self._plan_to_text(generated_plan, candidate_map)
        reference_text = self._reference_plan_to_text(reference_plan)
        if a_is_generated:
            return {"a_text": generated_text, "b_text": reference_text, "a_is_generated": True}
        return {"a_text": reference_text, "b_text": generated_text, "a_is_generated": False}

    def _parse_json(self, text: str) -> dict[str, Any]:
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

    def _clamp_score(self, value: Any) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return 5.0
        return max(0.0, min(10.0, parsed))
