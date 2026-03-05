from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .config import AblationConfig, ExperimentConfig
from .data_loader import DataLoader
from .graph import GraphBuilder
from .metrics import SampleMetric, aggregate_metrics, compute_sample_metrics
from .pattern import PatternMiner
from .planner import PlanGenerator
from .retrieval import GraphEnhancedRetriever
from .summarizer import EvidenceSummarizer
from .types import Candidate, EvidenceSummary, PlanResult, QuerySpec
from .validator import PlanValidator
from .vector_index import TFIDFIndex

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

        loader = DataLoader(self.config.data_dir)
        self.bundle = loader.load()

        self.pattern_miner = PatternMiner()
        self.pattern_miner.fit(self.bundle.train_samples)

        self.graph = graph_override if graph_override is not None else GraphBuilder(self.config).build(
            self.bundle, self.pattern_miner
        )
        all_candidates = list(self.bundle.candidates_global.values())
        self.vector_index = TFIDFIndex(all_candidates)

        self.retriever = GraphEnhancedRetriever(self.graph, self.vector_index)
        self.summarizer = EvidenceSummarizer()
        self.planner = PlanGenerator(self.config, self.pattern_miner)
        self.validator = PlanValidator()

        self.query_by_pid = {q.pid: q for q in self.bundle.query_specs}
        self.sample_by_pid = {int(s["pid"]): s for s in self.bundle.test_samples}

    def run_experiments(self, methods: list[str] | None = None, limit: int | None = None) -> dict[str, Any]:
        target_methods = methods or METHODS
        results: dict[str, Any] = {"methods": {}, "meta": {"limit": limit}}

        all_sample_metrics: dict[str, list[SampleMetric]] = {}
        all_outputs: dict[str, list[dict[str, Any]]] = {}

        for method in target_methods:
            sample_metrics, sample_outputs = self._run_method(method, limit=limit)
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
        plan = self._run_one(query, sample, method)
        metric = compute_sample_metrics(
            method=method,
            query=query,
            plan=plan,
            sample=sample,
            candidate_pool=self.bundle.candidates_by_pid[pid],
        )
        return {
            "pid": pid,
            "method": method,
            "plan": asdict(plan),
            "metrics": metric.values,
        }

    def _run_method(self, method: str, limit: int | None = None) -> tuple[list[SampleMetric], list[dict[str, Any]]]:
        metrics: list[SampleMetric] = []
        outputs: list[dict[str, Any]] = []

        samples = self.bundle.test_samples[:limit] if limit else self.bundle.test_samples
        for sample in samples:
            pid = int(sample["pid"])
            query = self.query_by_pid[pid]
            plan = self._run_one(query, sample, method)
            metric = compute_sample_metrics(
                method=method,
                query=query,
                plan=plan,
                sample=sample,
                candidate_pool=self.bundle.candidates_by_pid[pid],
            )
            metrics.append(metric)
            outputs.append(
                {
                    "pid": pid,
                    "method": method,
                    "metrics": metric.values,
                    "plan": asdict(plan),
                }
            )

        return metrics, outputs

    def _run_one(self, query: QuerySpec, sample: dict[str, Any], method: str) -> PlanResult:
        if method not in METHODS:
            raise ValueError(f"Unsupported method: {method}")

        candidate_pool = self.bundle.candidates_by_pid.get(query.pid, [])
        info = self.bundle.info_by_pid.get(str(query.pid), {})
        candidate_map = {c.candidate_id: c for c in candidate_pool}

        if method == "direct_llm":
            summary = self._direct_baseline_summary(query, candidate_pool)
            ranked_candidates = [candidate_map[cid] for cid in summary.chosen_ids if cid in candidate_map]
        else:
            ablation = self._ablation_for_method(method)
            retrieval_items = self.retriever.retrieve(
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

        if not first_report.passed and method != "direct_llm":
            plan = self.validator.repair_once(query, plan, ranked_candidates or candidate_pool, candidate_map)
            second_report = self.validator.validate(query, plan, candidate_map)
            plan.validator_report = {
                "passed": second_report.passed,
                "errors": second_report.errors,
                "checks": second_report.checks,
                "repaired": True,
            }

        return plan

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
        hotels = sorted(
            [c for c in candidate_pool if c.entity_type == "hotel"],
            key=lambda x: x.price,
            reverse=True,
        )
        attractions = sorted(
            [c for c in candidate_pool if c.entity_type == "attraction"],
            key=lambda x: x.price,
            reverse=True,
        )
        restaurants = sorted(
            [c for c in candidate_pool if c.entity_type == "restaurant"],
            key=lambda x: x.price,
            reverse=True,
        )

        chosen_ids: list[str] = []
        if hotels:
            chosen_ids.append(hotels[0].candidate_id)
        chosen_ids.extend([c.candidate_id for c in attractions[: max(4, query.day * 2)]])
        chosen_ids.extend([c.candidate_id for c in restaurants[: max(4, query.day * 2)]])

        return EvidenceSummary(
            query_pid=query.pid,
            chosen_ids=chosen_ids,
            reasons={cid: "direct_llm_heuristic" for cid in chosen_ids},
            budget_risk="unknown",
            day_suggestions={day: [] for day in range(1, query.day + 1)},
            trace_paths={},
        )

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
