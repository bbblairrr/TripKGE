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

from triptailor_graphrag.config import ExperimentConfig, LocalLLMConfig
from triptailor_graphrag.data_loader import DataLoader
from triptailor_graphrag.graph import GraphBuilder
from triptailor_graphrag.pattern import PatternMiner
from triptailor_graphrag.pipeline import METHODS, TripTailorGraphRAGPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one PID with a selected method.")
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--method", default="graphrag_summary", choices=METHODS)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default=None)
    parser.add_argument("--graph-source", default="local", choices=["local", "neo4j"])
    parser.add_argument("--local-llm-model", default=None)
    parser.add_argument("--local-llm-tokenizer", default=None)
    parser.add_argument("--local-llm-device-map", default="auto")
    parser.add_argument("--local-llm-dtype", default="auto")
    parser.add_argument("--local-llm-temperature", type=float, default=0.0)
    parser.add_argument("--local-llm-summary-tokens", type=int, default=512)
    parser.add_argument("--local-llm-planner-tokens", type=int, default=768)
    parser.add_argument("--local-llm-judge-tokens", type=int, default=160)
    parser.add_argument("--disable-local-judge", action="store_true")
    parser.add_argument("--neo4j-bootstrap", action="store_true")
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687")
    parser.add_argument("--neo4j-user", default="neo4j")
    parser.add_argument("--neo4j-password", default=None)
    parser.add_argument("--neo4j-database", default="neo4j")
    parser.add_argument("--neo4j-batch-size", type=int, default=1000)
    parser.add_argument("--neo4j-clear", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig(
        data_dir=Path(args.data_dir),
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
                graph_override = store.load_graph()
            else:
                if not args.neo4j_bootstrap:
                    raise RuntimeError(
                        "Neo4j graph is empty. Run scripts/export_neo4j.py once, "
                        "or pass --neo4j-bootstrap."
                    )
                bundle = DataLoader(config.data_dir).load()
                miner = PatternMiner()
                miner.fit(bundle.train_samples)
                boot_graph = GraphBuilder(config).build(bundle, miner)
                store.persist_graph(boot_graph, clear_existing=args.neo4j_clear)
                graph_override = store.load_graph()

    pipeline = TripTailorGraphRAGPipeline(config=config, graph_override=graph_override)
    result = pipeline.run_single(pid=args.pid, method=args.method)
    payload = json.dumps(result, ensure_ascii=False, indent=2)

    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
        print(f"Saved: {Path(args.output).resolve()}")
    else:
        print(payload)


if __name__ == "__main__":
    main()
