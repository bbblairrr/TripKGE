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

from triptailor_graphrag.config import ExperimentConfig, GraphStoreConfig, LocalLLMConfig, VectorStoreConfig
from triptailor_graphrag.pipeline import METHODS, TripTailorGraphRAGPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one PID with a selected method.")
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--method", default="graphrag_summary", choices=METHODS)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--train-file", default="train.json", help="Training split filename relative to --data-dir")
    parser.add_argument("--eval-file", default="test.json", help="Evaluation split filename relative to --data-dir")
    parser.add_argument("--info-file", default="infomation.json", help="Info filename relative to --data-dir")
    parser.add_argument("--output", default=None)
    parser.add_argument("--llm-backend", default=None, choices=["ollama", "transformers"])
    parser.add_argument("--llm-model", default=None)
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
    parser.add_argument("--judge-llm-backend", default=None, choices=["ollama", "transformers"])
    parser.add_argument("--judge-llm-model", default=None)
    parser.add_argument("--judge-llm-temperature", type=float, default=0.0)
    parser.add_argument("--judge-llm-max-new-tokens", type=int, default=384)
    parser.add_argument("--judge-llm-timeout-seconds", type=int, default=120)
    parser.add_argument("--vector-backend", default="auto", choices=["auto", "tfidf", "faiss"])
    parser.add_argument("--vector-cache-dir", default=".cache/faiss")
    parser.add_argument("--embed-model", default=None)
    parser.add_argument("--embed-batch-size", type=int, default=32)
    parser.add_argument("--rebuild-vector-index", action="store_true")
    parser.add_argument("--graph-source", default="auto", choices=["auto", "local", "neo4j"])
    parser.add_argument("--neo4j-bootstrap", action="store_true", help="Deprecated: bootstrap is automatic for Neo4j")
    parser.add_argument("--no-neo4j-bootstrap", action="store_true", help="Disable automatic bootstrap when Neo4j is empty")
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687")
    parser.add_argument("--neo4j-user", default="neo4j")
    parser.add_argument("--neo4j-password", default=None)
    parser.add_argument("--neo4j-database", default="neo4j")
    parser.add_argument("--neo4j-batch-size", type=int, default=1000)
    parser.add_argument("--neo4j-clear", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    llm_cfg = LocalLLMConfig(
        backend=args.llm_backend,
        model=args.llm_model,
        max_candidates=args.llm_max_candidates,
        temperature=args.llm_temperature,
        max_new_tokens=args.llm_max_new_tokens,
        timeout_seconds=args.llm_timeout_seconds,
        fallback_to_heuristic=not args.no_llm_fallback,
        generation_retries=args.llm_generation_retries,
    )
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
    config = ExperimentConfig(
        data_dir=Path(args.data_dir),
        train_file=args.train_file,
        eval_file=args.eval_file,
        info_file=args.info_file,
        llm=llm_cfg,
        judge_llm=judge_llm_cfg,
        graph_store=graph_store_cfg,
        vector_store=vector_store_cfg,
    )

    pipeline = TripTailorGraphRAGPipeline(config=config)
    result = pipeline.run_single(pid=args.pid, method=args.method)
    payload = json.dumps(result, ensure_ascii=False, indent=2)

    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
        print(f"Saved: {Path(args.output).resolve()}")
    else:
        print(payload)


if __name__ == "__main__":
    main()
