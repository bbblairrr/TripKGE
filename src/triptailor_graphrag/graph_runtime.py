from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import ExperimentConfig, GraphStoreConfig
from .data_loader import DataLoader, DatasetBundle
from .graph import GraphBuilder, MultiLayerGraph
from .neo4j_store import Neo4jConnConfig, Neo4jGraphStore
from .pattern import PatternMiner


@dataclass(frozen=True)
class GraphResolution:
    graph: MultiLayerGraph
    source: str
    action: str
    stats: dict[str, int] | None = None
    details: str | None = None


def resolve_graph(
    config: ExperimentConfig,
    bundle: DatasetBundle | None = None,
    pattern_miner: PatternMiner | None = None,
    graph_override: Any | None = None,
) -> GraphResolution:
    if isinstance(graph_override, GraphResolution):
        return graph_override
    if graph_override is not None:
        return GraphResolution(
            graph=graph_override,
            source="override",
            action="override",
            stats=_graph_stats(graph_override),
        )

    graph_store = config.graph_store
    source = (graph_store.source or "auto").lower()
    if source not in {"auto", "local", "neo4j"}:
        raise ValueError(f"Unsupported graph source: {graph_store.source}")

    if source == "local":
        graph = _build_local_graph(config, bundle, pattern_miner)
        return GraphResolution(graph=graph, source="local", action="built_local", stats=_graph_stats(graph))

    if source == "auto" and not graph_store.neo4j_password:
        graph = _build_local_graph(config, bundle, pattern_miner)
        return GraphResolution(
            graph=graph,
            source="local",
            action="built_local",
            stats=_graph_stats(graph),
            details="Neo4j password not configured; fell back to local graph build.",
        )

    try:
        return _resolve_from_neo4j(config, graph_store, bundle, pattern_miner)
    except Exception as exc:
        if source != "auto":
            raise
        graph = _build_local_graph(config, bundle, pattern_miner)
        return GraphResolution(
            graph=graph,
            source="local",
            action="built_local",
            stats=_graph_stats(graph),
            details=f"Neo4j unavailable, fell back to local graph build: {exc}",
        )


def _resolve_from_neo4j(
    config: ExperimentConfig,
    graph_store: GraphStoreConfig,
    bundle: DatasetBundle | None,
    pattern_miner: PatternMiner | None,
) -> GraphResolution:
    conn = Neo4jConnConfig(
        uri=graph_store.neo4j_uri,
        user=graph_store.neo4j_user,
        password=graph_store.neo4j_password or "",
        database=graph_store.neo4j_database,
        batch_size=graph_store.neo4j_batch_size,
    )
    with Neo4jGraphStore(conn) as store:
        if store.graph_exists():
            stats = store.graph_stats()
            graph = store.load_graph()
            return GraphResolution(graph=graph, source="neo4j", action="loaded_neo4j", stats=stats)

        if not graph_store.bootstrap_if_missing:
            raise RuntimeError(
                "Neo4j graph is empty and automatic bootstrap is disabled. "
                "Enable bootstrap or import the graph first."
            )

        graph = _build_local_graph(config, bundle, pattern_miner)
        stats = store.persist_graph(graph, clear_existing=graph_store.clear_on_bootstrap)
        return GraphResolution(graph=graph, source="neo4j", action="bootstrapped_neo4j", stats=stats)


def _build_local_graph(
    config: ExperimentConfig,
    bundle: DatasetBundle | None,
    pattern_miner: PatternMiner | None,
) -> MultiLayerGraph:
    actual_bundle = bundle or DataLoader(
        config.data_dir,
        train_file=config.train_file,
        eval_file=config.eval_file,
        info_file=config.info_file,
    ).load()
    actual_pattern_miner = pattern_miner or PatternMiner()
    if pattern_miner is None:
        actual_pattern_miner.fit(actual_bundle.train_samples)
    return GraphBuilder(config).build(actual_bundle, actual_pattern_miner)


def _graph_stats(graph: Any) -> dict[str, int] | None:
    nodes = getattr(graph, "nodes", None)
    edge_rel = getattr(graph, "edge_rel", None)
    if not isinstance(nodes, dict) or not isinstance(edge_rel, dict):
        return None
    return {
        "node_count": len(nodes),
        "edge_count": len(edge_rel) // 2,
    }
