#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from triptailor_graphrag.config import (
    AblationConfig,
    ExperimentConfig,
    GraphStoreConfig,
    LocalLLMConfig,
    RetrievalWeights,
    VectorStoreConfig,
)
from triptailor_graphrag.graph_runtime import resolve_graph
from triptailor_graphrag.pipeline import METHODS, TripTailorGraphRAGPipeline
from triptailor_graphrag.utils import safe_float, slugify

LOWER_BETTER_METRICS = {"average_route_distance_ratio", "max_single_day_route_km"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TripTailor-GraphRAG experiments.")
    parser.add_argument("--data-dir", default="data", help="Dataset directory")
    parser.add_argument("--train-file", default="train.json", help="Training split filename relative to --data-dir")
    parser.add_argument("--eval-file", default="test.json", help="Evaluation split filename relative to --data-dir")
    parser.add_argument("--info-file", default="infomation.json", help="Info filename relative to --data-dir")
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    parser.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    parser.add_argument("--limit", type=int, default=None, help="Only run first N evaluation samples")
    parser.add_argument("--show-progress", action="store_true", help="Print per-method progress in the terminal")
    parser.add_argument("--progress-every", type=int, default=10, help="Print progress every N samples")
    parser.add_argument("--llm-backend", default=None, choices=["ollama", "transformers"])
    parser.add_argument("--llm-model", default=None, help="Single local model name or local path")
    parser.add_argument(
        "--llm-models",
        nargs="+",
        default=None,
        help="Run the same experiment for multiple local models and compare outputs",
    )
    parser.add_argument("--llm-max-candidates", type=int, default=24)
    parser.add_argument("--llm-temperature", type=float, default=0.0)
    parser.add_argument("--llm-max-new-tokens", type=int, default=768)
    parser.add_argument("--llm-timeout-seconds", type=int, default=120)
    parser.add_argument("--llm-generation-retries", type=int, default=3)
    parser.add_argument(
        "--no-llm-fallback",
        action="store_true",
        help="Fail instead of falling back to heuristic planning when local LLM planning errors.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from existing per-method prediction files in --output-dir")
    parser.add_argument("--judge-llm-backend", default=None, choices=["ollama", "transformers"])
    parser.add_argument("--judge-llm-model", default=None, help="Local judge model used for personalization comparison")
    parser.add_argument("--judge-llm-temperature", type=float, default=0.0)
    parser.add_argument("--judge-llm-max-new-tokens", type=int, default=384)
    parser.add_argument("--judge-llm-timeout-seconds", type=int, default=120)
    parser.add_argument("--vector-backend", default="auto", choices=["auto", "tfidf", "faiss"])
    parser.add_argument("--vector-cache-dir", default=".cache/faiss")
    parser.add_argument("--embed-model", default=None)
    parser.add_argument("--embed-batch-size", type=int, default=32)
    parser.add_argument("--rebuild-vector-index", action="store_true")

    parser.add_argument("--hops", type=int, default=2)
    parser.add_argument(
        "--compare-hops",
        nargs="+",
        type=int,
        default=None,
        help="Run the same experiment for multiple hop values and write cross-run comparison summaries.",
    )
    parser.add_argument("--topk-vector", type=int, default=30)
    parser.add_argument("--topk-final", type=int, default=20)
    parser.add_argument(
        "--compare-target-metric",
        default="constraint_satisfaction_rate",
        help="Ranking target metric used in comparison summaries when multiple models or hop values are evaluated.",
    )

    parser.add_argument("--w-vector", type=float, default=0.5)
    parser.add_argument("--w-constraint", type=float, default=0.25)
    parser.add_argument("--w-graph", type=float, default=0.25)

    parser.add_argument(
        "--graph-source",
        default="local",
        choices=["auto", "local", "neo4j"],
        help="Graph source. `auto` prefers Neo4j when configured, otherwise builds locally.",
    )
    parser.add_argument(
        "--neo4j-bootstrap",
        action="store_true",
        help="Deprecated: bootstrap is automatic for Neo4j",
    )
    parser.add_argument("--no-neo4j-bootstrap", action="store_true", help="Disable automatic bootstrap when Neo4j is empty")

    parser.add_argument("--export-neo4j", action="store_true", help="Persist generated graph into local Neo4j")
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687")
    parser.add_argument("--neo4j-user", default="neo4j")
    parser.add_argument("--neo4j-password", default=None)
    parser.add_argument("--neo4j-database", default="neo4j")
    parser.add_argument("--neo4j-clear", action="store_true", help="Clear existing :KGNode graph before import")
    parser.add_argument("--neo4j-batch-size", type=int, default=1000)
    parser.add_argument("--neo4j-cypher-out", default=None, help="Optional path to dump Cypher script")

    return parser.parse_args()


def build_llm_runs(args: argparse.Namespace) -> list[LocalLLMConfig]:
    models = args.llm_models or ([args.llm_model] if args.llm_model else [])
    if not models:
        return [LocalLLMConfig()]
    if not args.llm_backend:
        raise ValueError("--llm-backend is required when --llm-model or --llm-models is used")
    return [
        LocalLLMConfig(
            backend=args.llm_backend,
            model=model,
            max_candidates=args.llm_max_candidates,
            temperature=args.llm_temperature,
            max_new_tokens=args.llm_max_new_tokens,
            timeout_seconds=args.llm_timeout_seconds,
            fallback_to_heuristic=not args.no_llm_fallback,
            generation_retries=args.llm_generation_retries,
        )
        for model in models
    ]


def build_ablation_config(args: argparse.Namespace, hops: int) -> AblationConfig:
    return AblationConfig(
        use_vector=True,
        use_graph_expansion=True,
        use_community_retrieval=True,
        use_summary_layer=True,
        hops=hops,
        topk_vector=args.topk_vector,
        topk_final=args.topk_final,
    )


def comparison_sort_key(row: dict[str, Any], target_metric: str) -> tuple[float, float, float, float, float]:
    target_value = safe_float(
        row.get(target_metric),
        default=999999.0 if target_metric in LOWER_BETTER_METRICS else 0.0,
    )
    if target_metric in LOWER_BETTER_METRICS:
        target_key = target_value
    else:
        target_key = -target_value
    return (
        target_key,
        -safe_float(row.get("constraint_satisfaction_rate")),
        -safe_float(row.get("feasibility_pass_rate")),
        -safe_float(row.get("personalization_proxy")),
        safe_float(row.get("average_route_distance_ratio"), default=999999.0),
    )


def rank_rows(rows: list[dict[str, Any]], target_metric: str) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: comparison_sort_key(row, target_metric))


def build_comparison_rows(run_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in run_records:
        summary = record["summary"]
        for method, method_payload in summary["methods"].items():
            rows.append(
                {
                    "run_key": record["run_key"],
                    "llm_backend": record["llm_backend"],
                    "llm_model": record["llm_model"],
                    "hops": record["hops"],
                    "method": method,
                    "n": method_payload["n"],
                    **method_payload["aggregated"],
                    "output_dir": record["output_dir"],
                }
            )
    return rows


def write_comparison_outputs(
    rows: list[dict[str, Any]],
    output_dir: Path,
    target_metric: str,
    methods: list[str],
    llm_models: list[str],
    hops: list[int],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / "comparison_runs.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as sink:
        for row in rows:
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")

    if rows:
        csv_path = output_dir / "comparison_runs.csv"
        keys = list(rows[0].keys())
        with csv_path.open("w", encoding="utf-8", newline="") as sink:
            writer = csv.DictWriter(sink, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)

    summary_payload: dict[str, Any] = {
        "target_metric": target_metric,
        "num_runs": len({row["run_key"] for row in rows}),
        "num_rows": len(rows),
        "methods": methods,
        "llm_models": llm_models,
        "hops": hops,
        "best_by_method": {},
        "best_by_model": {},
        "best_by_hop": {},
    }

    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        ranked = rank_rows(method_rows, target_metric)
        summary_payload["best_by_method"][method] = ranked[0] if ranked else None

    for model in llm_models:
        summary_payload["best_by_model"][model] = {}
        for method in methods:
            ranked = rank_rows(
                [row for row in rows if row["llm_model"] == model and row["method"] == method],
                target_metric,
            )
            summary_payload["best_by_model"][model][method] = ranked[0] if ranked else None

    for hop in hops:
        summary_payload["best_by_hop"][str(hop)] = {}
        for method in methods:
            ranked = rank_rows(
                [row for row in rows if int(row["hops"]) == hop and row["method"] == method],
                target_metric,
            )
            summary_payload["best_by_hop"][str(hop)][method] = ranked[0] if ranked else None

    summary_path = output_dir / "comparison_summary.json"
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    llm_runs = build_llm_runs(args)
    hop_values = args.compare_hops or [args.hops]
    hop_values = list(dict.fromkeys(hop_values))

    judge_llm_cfg = LocalLLMConfig(
        backend=args.judge_llm_backend,
        model=args.judge_llm_model,
        temperature=args.judge_llm_temperature,
        max_new_tokens=args.judge_llm_max_new_tokens,
        timeout_seconds=args.judge_llm_timeout_seconds,
    )
    default_graph_store = GraphStoreConfig()
    graph_store_cfg = GraphStoreConfig(
        source=args.graph_source,
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password if args.neo4j_password is not None else default_graph_store.neo4j_password,
        neo4j_database=args.neo4j_database,
        neo4j_batch_size=args.neo4j_batch_size,
        bootstrap_if_missing=not args.no_neo4j_bootstrap,
        clear_on_bootstrap=args.neo4j_clear,
    )
    default_vector_store = VectorStoreConfig()
    vector_store_cfg = VectorStoreConfig(
        backend=args.vector_backend,
        cache_dir=Path(args.vector_cache_dir),
        embed_model=args.embed_model or default_vector_store.embed_model,
        embed_batch_size=args.embed_batch_size,
        force_rebuild=args.rebuild_vector_index,
    )

    base_kwargs = dict(
        data_dir=Path(args.data_dir),
        train_file=args.train_file,
        eval_file=args.eval_file,
        info_file=args.info_file,
        graph_store=graph_store_cfg,
        vector_store=vector_store_cfg,
        retrieval_weights=RetrievalWeights(
            vector=args.w_vector,
            constraint=args.w_constraint,
            graph=args.w_graph,
        ),
    )

    base_config = ExperimentConfig(
        output_dir=Path(args.output_dir),
        llm=LocalLLMConfig(),
        judge_llm=judge_llm_cfg,
        ablation=build_ablation_config(args, hop_values[0]),
        **base_kwargs,
    )
    graph_resolution = resolve_graph(base_config)
    if graph_resolution.stats:
        print(
            f"Graph ready via {graph_resolution.action}: "
            f"nodes={graph_resolution.stats['node_count']}, edges={graph_resolution.stats['edge_count']}"
        )
    else:
        print(f"Graph ready via {graph_resolution.action}")
    if graph_resolution.details:
        print(graph_resolution.details)

    all_runs: dict[str, dict[str, object]] = {}
    run_records: list[dict[str, Any]] = []
    export_done = False
    multiple_models = len(llm_runs) > 1
    multiple_hops = len(hop_values) > 1
    total_runs = len(llm_runs) * len(hop_values)

    for hop_index, hops in enumerate(hop_values, start=1):
        for model_index, llm_cfg in enumerate(llm_runs, start=1):
            model_label = llm_cfg.model or "heuristic"
            run_key = model_label if not multiple_hops else f"{model_label} | hop={hops}"
            run_out = Path(args.output_dir)
            if multiple_hops:
                run_out = run_out / f"hop_{hops}"
            if multiple_models:
                run_out = run_out / slugify(model_label)

            current_run = (hop_index - 1) * len(llm_runs) + model_index
            print(f"[{current_run}/{total_runs}] Running model={model_label} hops={hops}")

            config = ExperimentConfig(
                output_dir=run_out,
                llm=llm_cfg,
                judge_llm=judge_llm_cfg,
                ablation=build_ablation_config(args, hops),
                **base_kwargs,
            )
            pipeline = TripTailorGraphRAGPipeline(config=config, graph_override=graph_resolution)
            print(
                f"Vector index ready via {pipeline.vector_resolution.action}: "
                f"backend={pipeline.vector_resolution.backend}"
            )
            if pipeline.vector_resolution.details:
                print(pipeline.vector_resolution.details)

            if args.export_neo4j and not export_done:
                if not args.neo4j_password:
                    raise ValueError("--neo4j-password is required when --export-neo4j is enabled")

                from triptailor_graphrag.neo4j_store import Neo4jConnConfig, Neo4jGraphStore

                conn = Neo4jConnConfig(
                    uri=args.neo4j_uri,
                    user=args.neo4j_user,
                    password=args.neo4j_password,
                    database=args.neo4j_database,
                    batch_size=args.neo4j_batch_size,
                )
                with Neo4jGraphStore(conn) as store:
                    stats = store.persist_graph(pipeline.graph, clear_existing=args.neo4j_clear)
                    print(
                        f"Neo4j import finished: nodes={stats['node_count']}, "
                        f"edges={stats['edge_count']}, uri={args.neo4j_uri}, db={args.neo4j_database}"
                    )
                    if args.neo4j_cypher_out:
                        out = store.dump_cypher(
                            pipeline.graph,
                            args.neo4j_cypher_out,
                            clear_existing=args.neo4j_clear,
                        )
                        print(f"Neo4j cypher dump saved: {out.resolve()}")
                export_done = True

            summary = pipeline.run_experiments(
                methods=args.methods,
                limit=args.limit,
                show_progress=args.show_progress,
                progress_every=args.progress_every,
                resume=args.resume,
            )
            all_runs[run_key] = summary
            run_records.append(
                {
                    "run_key": run_key,
                    "llm_backend": args.llm_backend,
                    "llm_model": model_label,
                    "hops": hops,
                    "output_dir": str(run_out.resolve()),
                    "summary": summary,
                }
            )

    if multiple_models and not multiple_hops:
        payload: dict[str, object] = {
            "models": all_runs,
            "meta": {
                "limit": args.limit,
                "methods": args.methods,
                "hops": hop_values,
                "llm_backend": args.llm_backend,
                "llm_fallback_to_heuristic": not args.no_llm_fallback,
                "judge_llm_backend": args.judge_llm_backend,
                "judge_llm_model": args.judge_llm_model,
            },
        }
        summary_path = Path(args.output_dir) / "experiment_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    elif len(run_records) == 1:
        payload = next(iter(all_runs.values()))
    else:
        payload = {
            "runs": {
                record["run_key"]: {
                    "llm_backend": record["llm_backend"],
                    "llm_model": record["llm_model"],
                    "hops": record["hops"],
                    "output_dir": record["output_dir"],
                    "summary": record["summary"],
                }
                for record in run_records
            },
            "meta": {
                "limit": args.limit,
                "methods": args.methods,
                "hops": hop_values,
                "llm_backend": args.llm_backend,
                "llm_models": [cfg.model or "heuristic" for cfg in llm_runs],
                "llm_fallback_to_heuristic": not args.no_llm_fallback,
                "judge_llm_backend": args.judge_llm_backend,
                "judge_llm_model": args.judge_llm_model,
                "compare_target_metric": args.compare_target_metric,
            },
        }
        summary_path = Path(args.output_dir) / "experiment_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if len(run_records) > 1:
        comparison_rows = build_comparison_rows(run_records)
        write_comparison_outputs(
            rows=comparison_rows,
            output_dir=Path(args.output_dir),
            target_metric=args.compare_target_metric,
            methods=args.methods,
            llm_models=[cfg.model or "heuristic" for cfg in llm_runs],
            hops=hop_values,
        )

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Saved outputs to: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
