#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from triptailor_graphrag.config import ExperimentConfig
from triptailor_graphrag.data_loader import DataLoader
from triptailor_graphrag.graph import GraphBuilder
from triptailor_graphrag.neo4j_store import Neo4jConnConfig, Neo4jGraphStore
from triptailor_graphrag.pattern import PatternMiner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export generated knowledge graph to local Neo4j.")
    parser.add_argument("--data-dir", default="data")

    parser.add_argument("--uri", default="bolt://localhost:7687")
    parser.add_argument("--user", default="neo4j")
    parser.add_argument("--password", required=True)
    parser.add_argument("--database", default="neo4j")
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--clear", action="store_true", help="Delete existing :KGNode graph before import")
    parser.add_argument(
        "--if-empty",
        action="store_true",
        help="Only build/import graph when Neo4j has no :KGNode data",
    )

    parser.add_argument(
        "--cypher-out",
        default=None,
        help="Optional path to dump Cypher import script",
    )
    return parser.parse_args()


def build_graph(data_dir: Path):
    cfg = ExperimentConfig(data_dir=data_dir)
    bundle = DataLoader(cfg.data_dir).load()
    miner = PatternMiner()
    miner.fit(bundle.train_samples)
    graph = GraphBuilder(cfg).build(bundle, miner)
    return graph


def main() -> None:
    args = parse_args()

    conn = Neo4jConnConfig(
        uri=args.uri,
        user=args.user,
        password=args.password,
        database=args.database,
        batch_size=args.batch_size,
    )

    with Neo4jGraphStore(conn) as store:
        if args.if_empty and store.graph_exists():
            stats = store.graph_stats()
            print(
                f"Neo4j already has graph data, skip rebuild: nodes={stats['node_count']}, "
                f"edges={stats['edge_count']}, uri={args.uri}, db={args.database}"
            )
            return

        graph = build_graph(Path(args.data_dir))
        stats = store.persist_graph(graph, clear_existing=args.clear)
        print(
            f"Neo4j import finished: nodes={stats['node_count']}, "
            f"edges={stats['edge_count']}, uri={args.uri}, db={args.database}"
        )

        if args.cypher_out:
            out = store.dump_cypher(graph, args.cypher_out, clear_existing=args.clear)
            print(f"Cypher script saved to: {out.resolve()}")


if __name__ == "__main__":
    main()
