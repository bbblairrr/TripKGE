#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from triptailor_graphrag.config import AblationConfig, ExperimentConfig, LocalLLMConfig, RetrievalWeights
from triptailor_graphrag.data_loader import DataLoader
from triptailor_graphrag.graph import GraphBuilder
from triptailor_graphrag.pattern import PatternMiner
from triptailor_graphrag.pipeline import METHODS, TripTailorGraphRAGPipeline
from triptailor_graphrag.utils import slugify


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TripTailor-GraphRAG experiments.")
    parser.add_argument("--data-dir", default="data", help="Dataset directory")
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    parser.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    parser.add_argument("--limit", type=int, default=None, help="Only run first N test samples")
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

    parser.add_argument("--hops", type=int, default=2)
    parser.add_argument("--topk-vector", type=int, default=30)
    parser.add_argument("--topk-final", type=int, default=20)

    parser.add_argument("--w-vector", type=float, default=0.5)
    parser.add_argument("--w-constraint", type=float, default=0.25)
    parser.add_argument("--w-graph", type=float, default=0.25)

    parser.add_argument(
        "--graph-source",
        default="local",
        choices=["local", "neo4j"],
        help="Load graph from local build or existing Neo4j graph",
    )
    parser.add_argument(
        "--neo4j-bootstrap",
        action="store_true",
        help="When --graph-source neo4j and graph is empty, build once locally then write to Neo4j",
    )

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
        )
        for model in models
    ]


def build_graph_from_local(config: ExperimentConfig):
    bundle = DataLoader(config.data_dir).load()
    miner = PatternMiner()
    miner.fit(bundle.train_samples)
    return GraphBuilder(config).build(bundle, miner)


def main() -> None:
    args = parse_args()
    llm_runs = build_llm_runs(args)

    base_kwargs = dict(
        data_dir=Path(args.data_dir),
        retrieval_weights=RetrievalWeights(
            vector=args.w_vector,
            constraint=args.w_constraint,
            graph=args.w_graph,
        ),
        ablation=AblationConfig(
            use_vector=True,
            use_graph_expansion=True,
            use_community_retrieval=True,
            use_summary_layer=True,
            hops=args.hops,
            topk_vector=args.topk_vector,
            topk_final=args.topk_final,
        ),
    )

    graph_override = None
    if args.graph_source == "neo4j":
        if not args.neo4j_password:
            raise ValueError("--neo4j-password is required when --graph-source neo4j")
        from triptailor_graphrag.neo4j_store import Neo4jConnConfig, Neo4jGraphStore

        conn = Neo4jConnConfig(
            uri=args.neo4j_uri,
            user=args.neo4j_user,
            password=args.neo4j_password,
            database=args.neo4j_database,
            batch_size=args.neo4j_batch_size,
        )
        with Neo4jGraphStore(conn) as store:
            if store.graph_exists():
                stats = store.graph_stats()
                print(
                    f"Loaded graph from Neo4j: nodes={stats['node_count']}, "
                    f"edges={stats['edge_count']}, uri={args.neo4j_uri}, db={args.neo4j_database}"
                )
                graph_override = store.load_graph()
            else:
                if not args.neo4j_bootstrap:
                    raise RuntimeError(
                        "Neo4j graph is empty. Run scripts/export_neo4j.py once, "
                        "or pass --neo4j-bootstrap to build and write once."
                    )
                bootstrap_cfg = ExperimentConfig(output_dir=Path(args.output_dir), llm=LocalLLMConfig(), **base_kwargs)
                boot_graph = build_graph_from_local(bootstrap_cfg)
                stats = store.persist_graph(boot_graph, clear_existing=args.neo4j_clear)
                print(
                    f"Bootstrapped Neo4j graph: nodes={stats['node_count']}, "
                    f"edges={stats['edge_count']}, uri={args.neo4j_uri}, db={args.neo4j_database}"
                )
                graph_override = store.load_graph()

    all_runs: dict[str, dict[str, object]] = {}
    export_done = False
    multiple_models = len(llm_runs) > 1
    for llm_cfg in llm_runs:
        model_label = llm_cfg.model or "heuristic"
        run_out = Path(args.output_dir)
        if multiple_models:
            run_out = run_out / slugify(model_label)

        config = ExperimentConfig(output_dir=run_out, llm=llm_cfg, **base_kwargs)
        pipeline = TripTailorGraphRAGPipeline(config=config, graph_override=graph_override)

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

        summary = pipeline.run_experiments(methods=args.methods, limit=args.limit)
        all_runs[model_label] = summary

    if multiple_models:
        payload: dict[str, object] = {
            "models": all_runs,
            "meta": {
                "limit": args.limit,
                "methods": args.methods,
                "llm_backend": args.llm_backend,
            },
        }
        summary_path = Path(args.output_dir) / "experiment_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        payload = next(iter(all_runs.values()))

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Saved outputs to: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
