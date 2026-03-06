#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from triptailor_graphrag.config import AblationConfig, ExperimentConfig, LocalLLMConfig, RetrievalWeights
from triptailor_graphrag.data_loader import DataLoader
from triptailor_graphrag.graph import GraphBuilder
from triptailor_graphrag.pattern import PatternMiner
from triptailor_graphrag.pipeline import TripTailorGraphRAGPipeline

LOWER_BETTER_METRICS = {"average_route_distance_ratio", "max_single_day_route_km"}


def parse_int_list(value: str) -> list[int]:
    out = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not out:
        raise ValueError("Empty integer list")
    return out


def parse_float_list(value: str) -> list[float]:
    out = [float(x.strip()) for x in value.split(",") if x.strip()]
    if not out:
        raise ValueError("Empty float list")
    return out


def parse_bool_list(value: str) -> list[bool]:
    mapping = {
        "1": True,
        "0": False,
        "true": True,
        "false": False,
        "yes": True,
        "no": False,
        "on": True,
        "off": False,
    }
    out: list[bool] = []
    for raw in value.split(","):
        key = raw.strip().lower()
        if not key:
            continue
        if key not in mapping:
            raise ValueError(f"Invalid bool token: {raw}")
        out.append(mapping[key])
    if not out:
        raise ValueError("Empty bool list")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ablation studies for TripTailor-GraphRAG.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs/ablation")
    parser.add_argument("--limit", type=int, default=100, help="Only evaluate first N test samples")
    parser.add_argument("--max-runs", type=int, default=None, help="Cap number of combinations")

    parser.add_argument("--hops", type=parse_int_list, default=parse_int_list("1,2"))
    parser.add_argument("--topk-vector", type=parse_int_list, default=parse_int_list("20,30"))
    parser.add_argument("--topk-final", type=parse_int_list, default=parse_int_list("15,20"))

    parser.add_argument("--w-vector", type=parse_float_list, default=parse_float_list("0.5"))
    parser.add_argument("--w-constraint", type=parse_float_list, default=parse_float_list("0.25"))
    parser.add_argument("--w-graph", type=parse_float_list, default=parse_float_list("0.25"))
    parser.add_argument("--normalize-weights", action="store_true", help="Normalize weight triples to sum to 1")

    parser.add_argument("--use-vector", type=parse_bool_list, default=parse_bool_list("1"))
    parser.add_argument("--use-graph-expansion", type=parse_bool_list, default=parse_bool_list("1,0"))
    parser.add_argument("--use-community", type=parse_bool_list, default=parse_bool_list("1,0"))
    parser.add_argument("--use-summary", type=parse_bool_list, default=parse_bool_list("1,0"))

    parser.add_argument(
        "--target-metric",
        default="feasibility_pass_rate",
        help="Ranking target metric (default: feasibility_pass_rate)",
    )
    parser.add_argument("--graph-source", default="local", choices=["local", "neo4j"])
    parser.add_argument("--judge-llm-backend", default=None, choices=["ollama", "transformers"])
    parser.add_argument("--judge-llm-model", default=None)
    parser.add_argument("--judge-llm-temperature", type=float, default=0.0)
    parser.add_argument("--judge-llm-max-new-tokens", type=int, default=384)
    parser.add_argument("--judge-llm-timeout-seconds", type=int, default=120)
    parser.add_argument("--neo4j-bootstrap", action="store_true")
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687")
    parser.add_argument("--neo4j-user", default="neo4j")
    parser.add_argument("--neo4j-password", default=None)
    parser.add_argument("--neo4j-database", default="neo4j")
    parser.add_argument("--neo4j-batch-size", type=int, default=1000)
    parser.add_argument("--neo4j-clear", action="store_true")
    return parser.parse_args()


def maybe_normalize_weights(wv: float, wc: float, wg: float, enabled: bool) -> tuple[float, float, float]:
    if not enabled:
        return wv, wc, wg
    total = wv + wc + wg
    if total <= 0:
        return 0.5, 0.25, 0.25
    return wv / total, wc / total, wg / total


def build_combinations(args: argparse.Namespace) -> list[dict[str, Any]]:
    combos = []
    for (
        hops,
        topk_vector,
        topk_final,
        wv,
        wc,
        wg,
        use_vector,
        use_graph_expansion,
        use_community,
        use_summary,
    ) in itertools.product(
        args.hops,
        args.topk_vector,
        args.topk_final,
        args.w_vector,
        args.w_constraint,
        args.w_graph,
        args.use_vector,
        args.use_graph_expansion,
        args.use_community,
        args.use_summary,
    ):
        # Community retrieval depends on graph expansion.
        if use_community and not use_graph_expansion:
            continue

        wv_n, wc_n, wg_n = maybe_normalize_weights(wv, wc, wg, args.normalize_weights)
        combos.append(
            {
                "hops": hops,
                "topk_vector": topk_vector,
                "topk_final": topk_final,
                "w_vector": round(wv_n, 6),
                "w_constraint": round(wc_n, 6),
                "w_graph": round(wg_n, 6),
                "use_vector": use_vector,
                "use_graph_expansion": use_graph_expansion,
                "use_community_retrieval": use_community,
                "use_summary_layer": use_summary,
            }
        )

    if args.max_runs is not None:
        combos = combos[: max(0, args.max_runs)]
    return combos


def run_one(
    combo: dict[str, Any],
    args: argparse.Namespace,
    run_idx: int,
    total: int,
    output_root: Path,
    graph_override: Any | None = None,
) -> dict[str, Any]:
    run_id = f"run_{run_idx:03d}"
    run_out = output_root / run_id

    cfg = ExperimentConfig(
        data_dir=Path(args.data_dir),
        output_dir=run_out,
        judge_llm=LocalLLMConfig(
            backend=args.judge_llm_backend,
            model=args.judge_llm_model,
            temperature=args.judge_llm_temperature,
            max_new_tokens=args.judge_llm_max_new_tokens,
            timeout_seconds=args.judge_llm_timeout_seconds,
        ),
        retrieval_weights=RetrievalWeights(
            vector=combo["w_vector"],
            constraint=combo["w_constraint"],
            graph=combo["w_graph"],
        ),
        ablation=AblationConfig(
            use_vector=combo["use_vector"],
            use_graph_expansion=combo["use_graph_expansion"],
            use_community_retrieval=combo["use_community_retrieval"],
            use_summary_layer=combo["use_summary_layer"],
            hops=combo["hops"],
            topk_vector=combo["topk_vector"],
            topk_final=combo["topk_final"],
        ),
    )

    pipeline = TripTailorGraphRAGPipeline(config=cfg, graph_override=graph_override)
    summary = pipeline.run_experiments(methods=["graphrag_summary"], limit=args.limit)
    agg = summary["methods"]["graphrag_summary"]["aggregated"]

    row = {
        "run_id": run_id,
        "run_index": run_idx,
        "run_total": total,
        **combo,
        **agg,
        "n": summary["methods"]["graphrag_summary"]["n"],
        "output_dir": str(run_out.resolve()),
    }
    return row


def sort_rows(rows: list[dict[str, Any]], target_metric: str) -> list[dict[str, Any]]:
    reverse = target_metric not in LOWER_BETTER_METRICS
    return sorted(
        rows,
        key=lambda x: (
            float(x.get(target_metric, 0.0)),
            float(x.get("constraint_satisfaction_rate", 0.0)),
            float(x.get("feasibility_pass_rate", 0.0)),
            float(x.get("personalization_proxy", 0.0)),
            -float(x.get("average_route_distance_ratio", 999999.0)),
        ),
        reverse=reverse,
    )


def write_outputs(rows: list[dict[str, Any]], target_metric: str, output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_root / "ablation_runs.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if rows:
        keys = list(rows[0].keys())
        csv_path = output_root / "ablation_runs.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)

    ranked = sort_rows(rows, target_metric)
    summary = {
        "target_metric": target_metric,
        "num_runs": len(rows),
        "best_run": ranked[0] if ranked else None,
        "top5": ranked[:5],
    }
    summary_path = output_root / "ablation_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_dir)

    combos = build_combinations(args)
    print(f"Total ablation combinations: {len(combos)}")

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
                        "or pass --neo4j-bootstrap."
                    )
                base_cfg = ExperimentConfig(
                    data_dir=Path(args.data_dir),
                    output_dir=Path(args.output_dir),
                    judge_llm=LocalLLMConfig(
                        backend=args.judge_llm_backend,
                        model=args.judge_llm_model,
                        temperature=args.judge_llm_temperature,
                        max_new_tokens=args.judge_llm_max_new_tokens,
                        timeout_seconds=args.judge_llm_timeout_seconds,
                    ),
                )
                bundle = DataLoader(base_cfg.data_dir).load()
                miner = PatternMiner()
                miner.fit(bundle.train_samples)
                boot_graph = GraphBuilder(base_cfg).build(bundle, miner)
                stats = store.persist_graph(boot_graph, clear_existing=args.neo4j_clear)
                print(
                    f"Bootstrapped Neo4j graph: nodes={stats['node_count']}, "
                    f"edges={stats['edge_count']}, uri={args.neo4j_uri}, db={args.neo4j_database}"
                )
                graph_override = store.load_graph()

    rows: list[dict[str, Any]] = []
    for idx, combo in enumerate(combos, start=1):
        print(f"[{idx}/{len(combos)}] Running: {combo}")
        row = run_one(combo, args, idx, len(combos), output_root, graph_override=graph_override)
        rows.append(row)
        print(
            f"  -> feasibility={row.get('feasibility_pass_rate', 0):.4f}, "
            f"personalization={row.get('personalization_proxy', 0):.4f}, "
            f"avg_route_ratio={row.get('average_route_distance_ratio', 0):.4f}, "
            f"max_day_route={row.get('max_single_day_route_km', 0):.2f}"
        )

    ranked = sort_rows(rows, args.target_metric)
    write_outputs(ranked, args.target_metric, output_root)

    print("\nTop 3 runs:")
    for row in ranked[:3]:
        print(
            f"{row['run_id']} | {args.target_metric}={row.get(args.target_metric, 0):.4f} | "
            f"hops={row['hops']} topk=({row['topk_vector']},{row['topk_final']}) "
            f"weights=({row['w_vector']},{row['w_constraint']},{row['w_graph']}) "
            f"flags=(v={int(row['use_vector'])},g={int(row['use_graph_expansion'])},"
            f"c={int(row['use_community_retrieval'])},s={int(row['use_summary_layer'])})"
        )

    print(f"\nSaved ablation outputs to: {output_root.resolve()}")


if __name__ == "__main__":
    main()
