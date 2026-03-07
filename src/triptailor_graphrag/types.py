from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class QuerySpec:
    pid: int
    departure_city: str
    destination_city: str
    day: int
    budget: float | None
    meal_price_range: tuple[float, float] | None
    query_text: str
    hotel_category_pref: str | None = None
    intensity_pref: str | None = None
    interest_tags: list[str] = field(default_factory=list)


@dataclass
class Candidate:
    candidate_id: str
    entity_type: str
    city: str
    name: str
    price: float
    latitude: float | None
    longitude: float | None
    text: str
    tags: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalItem:
    candidate: Candidate
    vector_score: float = 0.0
    constraint_score: float = 0.0
    graph_score: float = 0.0
    fused_score: float = 0.0
    raw_vector_score: float = 0.0
    raw_constraint_score: float = 0.0
    raw_graph_score: float = 0.0
    raw_fused_score: float = 0.0
    diversity_penalty: float = 0.0
    rerank_score: float = 0.0
    path_evidence: list[str] = field(default_factory=list)
    constraint_notes: list[str] = field(default_factory=list)


@dataclass
class RetrievalTrace:
    initial_candidate_count: int = 0
    city_candidate_count: int = 0
    filtered_candidate_count: int = 0
    strict_candidate_count: int = 0
    budget_relaxed_candidate_count: int = 0
    hard_relaxed_candidate_count: int = 0
    seed_count: int = 0
    vector_scope: str = "global"
    notes: list[str] = field(default_factory=list)


@dataclass
class EvidenceSummary:
    query_pid: int
    chosen_ids: list[str]
    reasons: dict[str, str]
    budget_risk: str
    day_suggestions: dict[int, list[str]]
    trace_paths: dict[str, list[str]]
    gate_mode: str = "disabled"
    gate_reason: str = ""
    gate_signals: dict[str, Any] = field(default_factory=dict)


@dataclass
class PlanActivity:
    day: int
    time: str
    action: str
    candidate_id: str
    location: str
    price: float


@dataclass
class PlanResult:
    query_pid: int
    hotel: list[dict[str, Any]]
    transportation: list[dict[str, Any]]
    itinerary: dict[str, list[dict[str, Any]]]
    candidate_pool: list[str]
    evidence_ids: list[str]
    validator_report: dict[str, Any]
    planner_mode: str = "heuristic"
    planning_error: str | None = None


@dataclass
class MetricResult:
    method: str
    scores: dict[str, float]
