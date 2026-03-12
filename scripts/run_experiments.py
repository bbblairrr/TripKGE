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
from triptailor_graphrag.utils import slugify


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
    parser.add_argument("--topk-vector", type=int, default=30)
    parser.add_argument("--topk-final", type=int, default=20)

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


def main() -> None:
    args = parse_args()
    llm_runs = build_llm_runs(args)
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

    base_config = ExperimentConfig(
        output_dir=Path(args.output_dir),
        llm=LocalLLMConfig(),
        judge_llm=judge_llm_cfg,
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
    export_done = False
    multiple_models = len(llm_runs) > 1
    for llm_cfg in llm_runs:
        model_label = llm_cfg.model or "heuristic"
        run_out = Path(args.output_dir)
        if multiple_models:
            run_out = run_out / slugify(model_label)

        config = ExperimentConfig(output_dir=run_out, llm=llm_cfg, judge_llm=judge_llm_cfg, **base_kwargs)
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
        all_runs[model_label] = summary

    if multiple_models:
        payload: dict[str, object] = {
            "models": all_runs,
            "meta": {
                "limit": args.limit,
                "methods": args.methods,
                "llm_backend": args.llm_backend,
                "llm_fallback_to_heuristic": not args.no_llm_fallback,
                "judge_llm_backend": args.judge_llm_backend,
                "judge_llm_model": args.judge_llm_model,
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
