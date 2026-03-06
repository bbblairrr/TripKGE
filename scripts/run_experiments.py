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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TripTailor-GraphRAG experiments.")
    parser.add_argument("--data-dir", default="data", help="Dataset directory")
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    parser.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    parser.add_argument("--limit", type=int, default=None, help="Only run first N test samples")

    parser.add_argument("--hops", type=int, default=2)
    parser.add_argument("--topk-vector", type=int, default=30)
    parser.add_argument("--topk-final", type=int, default=20)

    parser.add_argument("--w-vector", type=float, default=0.5)
    parser.add_argument("--w-constraint", type=float, default=0.25)
    parser.add_argument("--w-graph", type=float, default=0.25)
    parser.add_argument("--local-llm-model", default=None)
    parser.add_argument("--local-llm-tokenizer", default=None)
    parser.add_argument("--local-llm-device-map", default="auto")
    parser.add_argument("--local-llm-dtype", default="auto")
    parser.add_argument("--local-llm-temperature", type=float, default=0.0)
    parser.add_argument("--local-llm-summary-tokens", type=int, default=512)
    parser.add_argument("--local-llm-planner-tokens", type=int, default=768)
    parser.add_argument("--local-llm-judge-tokens", type=int, default=160)
    parser.add_argument("--disable-local-judge", action="store_true")

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


def build_graph_from_local(config: ExperimentConfig):
    bundle = DataLoader(config.data_dir).load()
    miner = PatternMiner()
    miner.fit(bundle.train_samples)
    return GraphBuilder(config).build(bundle, miner)


def main() -> None:
    args = parse_args()

    config = ExperimentConfig(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
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
        local_llm=LocalLLMConfig(
            enabled=bool(args.local_llm_model),
            model_path=args.local_llm_model,
            tokenizer_path=args.local_llm_tokenizer,
            device_map=args.local_llm_device_map,
            torch_dtype=args.local_llm_dtype,
            temperature=args.local_llm_temperature,
            summary_max_new_tokens=args.local_llm_summary_tokens,
            planner_max_new_tokens=args.local_llm_planner_tokens,
            judge_max_new_tokens=args.local_llm_judge_tokens,
            enable_judge=not args.disable_local_judge,
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
                boot_graph = build_graph_from_local(config)
                stats = store.persist_graph(boot_graph, clear_existing=args.neo4j_clear)
                print(
                    f"Bootstrapped Neo4j graph: nodes={stats['node_count']}, "
                    f"edges={stats['edge_count']}, uri={args.neo4j_uri}, db={args.neo4j_database}"
                )
                graph_override = store.load_graph()

    pipeline = TripTailorGraphRAGPipeline(config=config, graph_override=graph_override)

    if args.export_neo4j:
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

    summary = pipeline.run_experiments(methods=args.methods, limit=args.limit)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved outputs to: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
