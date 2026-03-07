from __future__ import annotations

import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .config import AblationConfig, ExperimentConfig
from .data_loader import DataLoader
from .graph_runtime import resolve_graph
from .local_llm import build_local_llm_client
from .metrics import SampleMetric, aggregate_metrics, compute_sample_metrics
from .pattern import PatternMiner
from .planner import PlanGenerator
from .preference_judge import PreferenceJudge
from .retrieval import GraphEnhancedRetriever
from .summarizer import EvidenceSummarizer
from .types import Candidate, EvidenceSummary, PlanResult, QuerySpec, RetrievalItem, RetrievalTrace
from .utils import normalize_text, tokenize
from .validator import PlanValidator
from .vector_runtime import resolve_vector_index

METHODS = [
    "direct_llm",
    "naive_rag",
    "kg_only",
    "graphrag_no_summary",
    "graphrag_summary",
]


class TripTailorGraphRAGPipeline:
    def __init__(
        self,
        config: ExperimentConfig | None = None,
        graph_override: Any | None = None,
    ) -> None:
        self.config = config or ExperimentConfig()
        self.rng = random.Random(self.config.random_seed)

        loader = DataLoader(
            self.config.data_dir,
            train_file=self.config.train_file,
            eval_file=self.config.eval_file,
            info_file=self.config.info_file,
        )
        self.bundle = loader.load()

        self.pattern_miner = PatternMiner()
        self.pattern_miner.fit(self.bundle.train_samples)

        self.graph_resolution = resolve_graph(
            config=self.config,
            bundle=self.bundle,
            pattern_miner=self.pattern_miner,
            graph_override=graph_override,
        )
        self.graph = self.graph_resolution.graph
        all_candidates = list(self.bundle.candidates_global.values())
        self.vector_resolution = resolve_vector_index(self.config, all_candidates)
        self.vector_index = self.vector_resolution.index

        self.retriever = GraphEnhancedRetriever(self.graph, self.vector_index)
        self.summarizer = EvidenceSummarizer()
        self.llm_client = build_local_llm_client(self.config.llm)
        self.judge_llm_client = build_local_llm_client(self.config.judge_llm)
        self.preference_judge = PreferenceJudge(self.judge_llm_client)
        self.planner = PlanGenerator(self.config, self.pattern_miner, llm_client=self.llm_client)
        self.validator = PlanValidator()

        self.query_by_pid = {q.pid: q for q in self.bundle.query_specs}
        self.sample_by_pid = {int(s["pid"]): s for s in self.bundle.test_samples}

    def run_experiments(
        self,
        methods: list[str] | None = None,
        limit: int | None = None,
        show_progress: bool = False,
        progress_every: int = 10,
    ) -> dict[str, Any]:
        target_methods = methods or METHODS
        results: dict[str, Any] = {
            "methods": {},
            "meta": {
                "limit": limit,
                "train_file": self.config.train_file,
                "eval_file": self.config.eval_file,
                "info_file": self.config.info_file,
                "llm_backend": self.config.llm.backend,
                "llm_model": self.config.llm.model,
                "llm_fallback_to_heuristic": self.config.llm.fallback_to_heuristic,
                "judge_llm_backend": self.config.judge_llm.backend,
                "judge_llm_model": self.config.judge_llm.model,
                "graph_source": self.graph_resolution.source,
                "graph_action": self.graph_resolution.action,
                "graph_stats": self.graph_resolution.stats,
                "vector_backend": self.vector_resolution.backend,
                "vector_action": self.vector_resolution.action,
                "vector_embed_model": self.config.vector_store.embed_model,
            },
        }
        if self.graph_resolution.details:
            results["meta"]["graph_details"] = self.graph_resolution.details
        if self.vector_resolution.details:
            results["meta"]["vector_details"] = self.vector_resolution.details

        all_sample_metrics: dict[str, list[SampleMetric]] = {}
        all_outputs: dict[str, list[dict[str, Any]]] = {}

        for method in target_methods:
            sample_metrics, sample_outputs = self._run_method(
                method,
                limit=limit,
                show_progress=show_progress,
                progress_every=progress_every,
            )
            all_sample_metrics[method] = sample_metrics
            all_outputs[method] = sample_outputs
            results["methods"][method] = {
                "aggregated": aggregate_metrics(sample_metrics),
                "n": len(sample_metrics),
            }

        if "direct_llm" in all_sample_metrics:
            base = {x.pid: x.values.get("personalization_proxy", 0.0) for x in all_sample_metrics["direct_llm"]}
            for method, rows in all_sample_metrics.items():
                if method == "direct_llm":
                    results["methods"][method]["aggregated"]["personalization_surpassing_rate"] = 0.0
                    continue
                wins = 0
                total = 0
                for row in rows:
                    if row.pid not in base:
                        continue
                    total += 1
                    if row.values.get("personalization_proxy", 0.0) > base[row.pid]:
                        wins += 1
                rate = wins / total if total else 0.0
                results["methods"][method]["aggregated"]["personalization_surpassing_rate"] = rate

        self._write_outputs(results, all_outputs)
        return results

    def run_single(self, pid: int, method: str = "graphrag_summary") -> dict[str, Any]:
        query = self.query_by_pid[pid]
        sample = self.sample_by_pid[pid]
        plan, run_trace = self._run_one(query, sample, method)
        metric = compute_sample_metrics(
            method=method,
            query=query,
            plan=plan,
            sample=sample,
            candidate_pool=self.bundle.candidates_by_pid[pid],
            info=self.bundle.info_by_pid.get(str(pid), {}),
            preference_judge=self.preference_judge,
        )
        return {
            "pid": pid,
            "method": method,
            "plan": asdict(plan),
            "metrics": metric.values,
            "metric_details": metric.details,
            "diagnostics": self._build_sample_diagnostics(run_trace, plan, metric),
        }

    def _run_method(
        self,
        method: str,
        limit: int | None = None,
        show_progress: bool = False,
        progress_every: int = 10,
    ) -> tuple[list[SampleMetric], list[dict[str, Any]]]:
        metrics: list[SampleMetric] = []
        outputs: list[dict[str, Any]] = []

        samples = self.bundle.test_samples[:limit] if limit else self.bundle.test_samples
        total = len(samples)
        started = time.perf_counter()
        if show_progress:
            print(f"[{method}] starting {total} samples...", flush=True)
        for idx, sample in enumerate(samples, start=1):
            pid = int(sample["pid"])
            query = self.query_by_pid[pid]
            plan, run_trace = self._run_one(query, sample, method)
            metric = compute_sample_metrics(
                method=method,
                query=query,
                plan=plan,
                sample=sample,
                candidate_pool=self.bundle.candidates_by_pid[pid],
                info=self.bundle.info_by_pid.get(str(pid), {}),
                preference_judge=self.preference_judge,
            )
            metrics.append(metric)
            outputs.append(
                {
                    "pid": pid,
                    "method": method,
                    "metrics": metric.values,
                    "metric_details": metric.details,
                    "plan": asdict(plan),
                    "diagnostics": self._build_sample_diagnostics(run_trace, plan, metric),
                }
            )
            if show_progress and (idx == total or idx == 1 or idx % max(1, progress_every) == 0):
                elapsed = time.perf_counter() - started
                rate = elapsed / idx if idx else 0.0
                eta = max(0.0, rate * (total - idx))
                print(
                    f"[{method}] {idx}/{total} pid={pid} "
                    f"planner={plan.planner_mode} elapsed={elapsed:.1f}s eta={eta:.1f}s",
                    flush=True,
                )

        return metrics, outputs

    def _run_one(self, query: QuerySpec, sample: dict[str, Any], method: str) -> tuple[PlanResult, dict[str, Any]]:
        if method not in METHODS:
            raise ValueError(f"Unsupported method: {method}")

        candidate_pool = self.bundle.candidates_by_pid.get(query.pid, [])
        info = self.bundle.info_by_pid.get(str(query.pid), {})
        candidate_map = {c.candidate_id: c for c in candidate_pool}
        retrieval_items: list[RetrievalItem] = []
        retrieval_trace = RetrievalTrace(initial_candidate_count=len(candidate_pool), city_candidate_count=len(candidate_pool))

        if method == "direct_llm":
            summary = self._direct_baseline_summary(query, candidate_pool)
            retrieval_items = self._direct_retrieval_items(query, candidate_pool)
            retrieval_trace.filtered_candidate_count = len(retrieval_items)
            retrieval_trace.strict_candidate_count = len(retrieval_items)
            retrieval_trace.seed_count = min(len(retrieval_items), self.config.llm.max_candidates)
            retrieval_trace.vector_scope = "direct_heuristic"
            retrieval_trace.notes.append("direct_llm uses heuristic shortlist instead of graph retrieval.")
            ranked_candidates = [candidate_map[cid] for cid in summary.chosen_ids if cid in candidate_map]
        else:
            ablation = self._ablation_for_method(method)
            retrieval_items, retrieval_trace = self.retriever.retrieve(
                query=query,
                candidate_pool=candidate_pool,
                ablation=ablation,
                weights=self.config.retrieval_weights,
            )
            ranked_candidates = [x.candidate for x in retrieval_items]
            if ablation.use_summary_layer:
                summary = self.summarizer.summarize(query, retrieval_items)
            else:
                summary = self._summary_without_layer(query, retrieval_items)

        plan = self.planner.generate(query, summary, candidate_pool, info)
        if method == "direct_llm":
            plan.evidence_ids = []

        first_report = self.validator.validate(query, plan, candidate_map)
        plan.validator_report = {
            "passed": first_report.passed,
            "errors": first_report.errors,
            "checks": first_report.checks,
            "repaired": False,
        }

        if not first_report.passed:
            plan = self.validator.repair_once(query, plan, ranked_candidates or candidate_pool, candidate_map)
            second_report = self.validator.validate(query, plan, candidate_map)
            plan.validator_report = {
                "passed": second_report.passed,
                "errors": second_report.errors,
                "checks": second_report.checks,
                "repaired": True,
            }

        run_trace = {
            "retrieval": {
                "trace": self._serialize_retrieval_trace(retrieval_trace),
                "top_candidates": [
                    self._serialize_retrieval_item(item, selected_ids=set(summary.chosen_ids))
                    for item in retrieval_items[:10]
                ],
                "summary": self._serialize_summary(summary),
            }
        }
        return plan, run_trace

    def _ablation_for_method(self, method: str) -> AblationConfig:
        base = self.config.ablation
        if method == "naive_rag":
            return AblationConfig(
                use_vector=True,
                use_graph_expansion=False,
                use_community_retrieval=False,
                use_summary_layer=False,
                hops=base.hops,
                topk_vector=base.topk_vector,
                topk_final=base.topk_final,
            )
        if method == "kg_only":
            return AblationConfig(
                use_vector=False,
                use_graph_expansion=True,
                use_community_retrieval=True,
                use_summary_layer=False,
                hops=base.hops,
                topk_vector=base.topk_vector,
                topk_final=base.topk_final,
            )
        if method == "graphrag_no_summary":
            return AblationConfig(
                use_vector=True,
                use_graph_expansion=True,
                use_community_retrieval=True,
                use_summary_layer=False,
                hops=base.hops,
                topk_vector=base.topk_vector,
                topk_final=base.topk_final,
            )
        return base

    def _summary_without_layer(self, query: QuerySpec, retrieval_items: list[Any]) -> EvidenceSummary:
        chosen_ids = [x.candidate.candidate_id for x in retrieval_items[: max(8, query.day * 5)]]
        return EvidenceSummary(
            query_pid=query.pid,
            chosen_ids=chosen_ids,
            reasons={},
            budget_risk="unknown",
            day_suggestions={day: [] for day in range(1, query.day + 1)},
            trace_paths={},
        )

    def _direct_baseline_summary(self, query: QuerySpec, candidate_pool: list[Candidate]) -> EvidenceSummary:
        hotels = sorted([c for c in candidate_pool if c.entity_type == "hotel"], key=lambda x: self._direct_candidate_rank(query, x))
        attractions = sorted(
            [c for c in candidate_pool if c.entity_type == "attraction"],
            key=lambda x: self._direct_candidate_rank(query, x),
        )
        restaurants = sorted(
            [c for c in candidate_pool if c.entity_type == "restaurant"],
            key=lambda x: self._direct_candidate_rank(query, x),
        )

        chosen_ids: list[str] = []
        hotel_cap = 4 if self.llm_client else 1
        attr_cap = max(4, query.day * 2)
        rest_cap = max(4, query.day * 2)
        if self.llm_client:
            attr_cap = max(attr_cap, min(10, self.config.llm.max_candidates // 2))
            rest_cap = max(rest_cap, min(10, self.config.llm.max_candidates // 2))
        chosen_ids.extend([c.candidate_id for c in hotels[:hotel_cap]])
        chosen_ids.extend([c.candidate_id for c in attractions[:attr_cap]])
        chosen_ids.extend([c.candidate_id for c in restaurants[:rest_cap]])
        if self.llm_client:
            chosen_ids = chosen_ids[: self.config.llm.max_candidates]

        return EvidenceSummary(
            query_pid=query.pid,
            chosen_ids=chosen_ids,
            reasons={cid: "direct_llm_heuristic" for cid in chosen_ids},
            budget_risk="unknown",
            day_suggestions={day: [] for day in range(1, query.day + 1)},
            trace_paths={},
        )

    def _direct_candidate_rank(self, query: QuerySpec, candidate: Candidate) -> tuple[float, float]:
        query_tokens = set(tokenize(query.query_text))
        text_tokens = set(tokenize(candidate.text))
        overlap = len(query_tokens.intersection(text_tokens))
        interest_overlap = 0
        if query.interest_tags:
            wanted = {normalize_text(x) for x in query.interest_tags}
            tags = {normalize_text(x) for x in candidate.tags}
            interest_overlap = len(wanted.intersection(tags))
        budget_penalty = 0.0
        if query.meal_price_range and candidate.entity_type == "restaurant":
            lo, hi = query.meal_price_range
            budget_penalty = 0.0 if lo <= candidate.price <= hi else 1.0
        elif query.budget is not None:
            budget_penalty = candidate.price / max(query.budget, 1.0)
        return (-interest_overlap - overlap, budget_penalty)

    def _direct_retrieval_items(self, query: QuerySpec, candidate_pool: list[Candidate]) -> list[RetrievalItem]:
        ranked = sorted(candidate_pool, key=lambda candidate: self._direct_candidate_rank(query, candidate))
        items: list[RetrievalItem] = []
        for idx, candidate in enumerate(ranked[: max(10, self.config.llm.max_candidates)]):
            priority, budget_penalty = self._direct_candidate_rank(query, candidate)
            heuristic_score = max(0.0, float(-priority)) + max(0.0, 1.0 - budget_penalty)
            item = RetrievalItem(
                candidate=candidate,
                vector_score=0.0,
                constraint_score=heuristic_score,
                graph_score=0.0,
                fused_score=heuristic_score,
                raw_vector_score=0.0,
                raw_constraint_score=heuristic_score,
                raw_graph_score=0.0,
                raw_fused_score=heuristic_score,
                rerank_score=heuristic_score - idx * 1e-6,
                constraint_notes=["direct_llm_heuristic_shortlist"],
            )
            items.append(item)
        return items

    def _serialize_retrieval_trace(self, trace: RetrievalTrace) -> dict[str, Any]:
        return {
            "initial_candidate_count": trace.initial_candidate_count,
            "city_candidate_count": trace.city_candidate_count,
            "filtered_candidate_count": trace.filtered_candidate_count,
            "strict_candidate_count": trace.strict_candidate_count,
            "budget_relaxed_candidate_count": trace.budget_relaxed_candidate_count,
            "hard_relaxed_candidate_count": trace.hard_relaxed_candidate_count,
            "seed_count": trace.seed_count,
            "vector_scope": trace.vector_scope,
            "notes": trace.notes,
        }

    def _serialize_retrieval_item(self, item: RetrievalItem, selected_ids: set[str]) -> dict[str, Any]:
        candidate = item.candidate
        return {
            "candidate_id": candidate.candidate_id,
            "name": candidate.name,
            "entity_type": candidate.entity_type,
            "city": candidate.city,
            "price": round(candidate.price, 2),
            "selected_in_summary": candidate.candidate_id in selected_ids,
            "raw_scores": {
                "vector": item.raw_vector_score,
                "constraint": item.raw_constraint_score,
                "graph": item.raw_graph_score,
                "fused": item.raw_fused_score,
            },
            "normalized_scores": {
                "vector": item.vector_score,
                "constraint": item.constraint_score,
                "graph": item.graph_score,
                "fused": item.fused_score,
            },
            "rerank_score": item.rerank_score,
            "diversity_penalty": item.diversity_penalty,
            "constraint_notes": item.constraint_notes,
            "path_evidence": item.path_evidence,
        }

    def _serialize_summary(self, summary: EvidenceSummary) -> dict[str, Any]:
        return {
            "chosen_ids": summary.chosen_ids,
            "budget_risk": summary.budget_risk,
            "day_suggestions": summary.day_suggestions,
            "reasons": summary.reasons,
            "trace_paths": summary.trace_paths,
            "gate_mode": summary.gate_mode,
            "gate_reason": summary.gate_reason,
            "gate_signals": summary.gate_signals,
        }

    def _build_sample_diagnostics(
        self,
        run_trace: dict[str, Any],
        plan: PlanResult,
        metric: SampleMetric,
    ) -> dict[str, Any]:
        return {
            **run_trace,
            "planner_mode": plan.planner_mode,
            "planning_error": plan.planning_error,
            "validator_failure_reasons": list(plan.validator_report.get("errors") or []),
            "validator_checks": dict(plan.validator_report.get("checks") or {}),
            "validator_repaired": bool(plan.validator_report.get("repaired")),
            "judge": metric.details.get("personalization"),
        }

    def _write_outputs(self, summary: dict[str, Any], all_outputs: dict[str, list[dict[str, Any]]]) -> None:
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        summary_path = output_dir / "experiment_summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        for method, rows in all_outputs.items():
            path = output_dir / f"{method}_predictions.jsonl"
            with path.open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
