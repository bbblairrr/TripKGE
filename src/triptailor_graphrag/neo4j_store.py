from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .graph import MultiLayerGraph


@dataclass(frozen=True)
class Neo4jConnConfig:
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "neo4j"
    database: str = "neo4j"
    batch_size: int = 1000


class Neo4jGraphStore:
    """Persist MultiLayerGraph into a local Neo4j database."""

    def __init__(self, conn: Neo4jConnConfig):
        self.conn = conn
        self._driver = None

    def __enter__(self) -> "Neo4jGraphStore":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def connect(self) -> None:
        if self._driver is not None:
            return
        try:
            from neo4j import GraphDatabase  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "neo4j driver not installed. Run: python3 -m pip install 'neo4j>=5.20.0'"
            ) from exc

        self._driver = GraphDatabase.driver(
            self.conn.uri,
            auth=(self.conn.user, self.conn.password),
        )
        self._driver.verify_connectivity()

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    def persist_graph(self, graph: MultiLayerGraph, clear_existing: bool = False) -> dict[str, int]:
        self.connect()

        if self._driver is None:
            raise RuntimeError("Neo4j driver is not initialized")

        node_rows = self._node_rows(graph)
        rel_rows = self._edge_rows(graph)

        with self._driver.session(database=self.conn.database) as session:
            session.run(
                "CREATE CONSTRAINT kg_node_id IF NOT EXISTS "
                "FOR (n:KGNode) REQUIRE n.node_id IS UNIQUE"
            )

            if clear_existing:
                session.run("MATCH (n:KGNode) DETACH DELETE n")

            for batch in self._batched(node_rows, self.conn.batch_size):
                session.run(
                    "UNWIND $rows AS row "
                    "MERGE (n:KGNode {node_id: row.node_id}) "
                    "SET n += row.props",
                    rows=batch,
                )

            for label, typed_rows in self._group_node_rows_by_label(node_rows).items():
                quoted_label = self._quote_identifier(label)
                for batch in self._batched(typed_rows, self.conn.batch_size):
                    session.run(
                        "UNWIND $rows AS row "
                        "MATCH (n:KGNode {node_id: row.node_id}) "
                        f"SET n:{quoted_label}",
                        rows=batch,
                    )

            for rel_type, typed_rows in self._group_edge_rows_by_type(rel_rows).items():
                quoted_rel_type = self._quote_identifier(rel_type)
                for batch in self._batched(typed_rows, self.conn.batch_size):
                    session.run(
                        "UNWIND $rows AS row "
                        "MATCH (a:KGNode {node_id: row.src}) "
                        "MATCH (b:KGNode {node_id: row.dst}) "
                        f"MERGE (a)-[r:{quoted_rel_type} {{src: row.src, dst: row.dst}}]->(b) "
                        "SET r += row.props",
                        rows=batch,
                    )

        return {"node_count": len(node_rows), "edge_count": len(rel_rows)}

    def graph_stats(self) -> dict[str, int]:
        self.connect()
        if self._driver is None:
            raise RuntimeError("Neo4j driver is not initialized")

        with self._driver.session(database=self.conn.database) as session:
            node_count = session.run("MATCH (n:KGNode) RETURN count(n) AS c").single()["c"]
            rel_count = session.run("MATCH (:KGNode)-[r]->(:KGNode) RETURN count(r) AS c").single()["c"]
        return {"node_count": int(node_count), "edge_count": int(rel_count)}

    def graph_exists(self) -> bool:
        stats = self.graph_stats()
        return stats["node_count"] > 0

    def load_graph(self) -> MultiLayerGraph:
        self.connect()
        if self._driver is None:
            raise RuntimeError("Neo4j driver is not initialized")

        graph = MultiLayerGraph()
        with self._driver.session(database=self.conn.database) as session:
            node_rows = session.run(
                "MATCH (n:KGNode) "
                "RETURN n.node_id AS node_id, n.layer AS layer, n.node_type AS node_type, "
                "n.label AS label, n.meta_json AS meta_json, n.meta AS meta_legacy"
            )
            for row in node_rows:
                node_id = row.get("node_id")
                if not node_id:
                    continue
                graph.add_node(
                    node_id=node_id,
                    layer=int(row.get("layer") or 0),
                    node_type=str(row.get("node_type") or "unknown"),
                    label=str(row.get("label") or node_id),
                    meta=self._deserialize_meta(row.get("meta_json"), row.get("meta_legacy")),
                )

            rel_rows = session.run(
                "MATCH (a:KGNode)-[r]->(b:KGNode) "
                "RETURN a.node_id AS src, b.node_id AS dst, coalesce(r.relation, toLower(type(r))) AS relation"
            )
            for row in rel_rows:
                src = row.get("src")
                dst = row.get("dst")
                if not src or not dst:
                    continue
                relation = str(row.get("relation") or "related")
                graph.add_edge(src, dst, relation)

        return graph

    def dump_cypher(self, graph: MultiLayerGraph, path: str | Path, clear_existing: bool = False) -> Path:
        """Generate a Cypher script for manual import via cypher-shell."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)

        node_rows = self._node_rows(graph)
        rel_rows = self._edge_rows(graph)

        lines: list[str] = []
        lines.append(
            "CREATE CONSTRAINT kg_node_id IF NOT EXISTS FOR (n:KGNode) REQUIRE n.node_id IS UNIQUE;"
        )
        if clear_existing:
            lines.append("MATCH (n:KGNode) DETACH DELETE n;")

        for row in node_rows:
            props = self._to_cypher_map({"node_id": row["node_id"], **row["props"]})
            node_label = self._quote_identifier(self._node_label(str(row["props"].get("node_type") or "")))
            lines.append(
                f"MERGE (n:KGNode {{node_id: {self._quote(row['node_id'])}}}) "
                f"SET n += {props} "
                f"SET n:{node_label};"
            )

        for row in rel_rows:
            props = self._to_cypher_map(row["props"])
            rel_type = self._quote_identifier(self._relation_type(row["relation"]))
            lines.append(
                "MATCH (a:KGNode {node_id: "
                + self._quote(row["src"])
                + "}), (b:KGNode {node_id: "
                + self._quote(row["dst"])
                + "}) "
                + "MERGE (a)-[r:"
                + rel_type
                + " {src: "
                + self._quote(row["src"])
                + ", dst: "
                + self._quote(row["dst"])
                + "}]->(b) SET r += "
                + props
                + ";"
            )

        out.write_text("\n".join(lines), encoding="utf-8")
        return out

    def _node_rows(self, graph: MultiLayerGraph) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for node in graph.nodes.values():
            props = {
                "layer": node.layer,
                "node_type": node.node_type,
                "label": node.label,
                "meta_json": self._serialize_meta(node.meta),
            }
            rows.append({"node_id": node.node_id, "props": props})
        return rows

    def _edge_rows(self, graph: MultiLayerGraph) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()

        for (src, dst), relation in graph.edge_rel.items():
            if src not in graph.nodes or dst not in graph.nodes:
                continue
            a, b = sorted((src, dst))
            key = (a, b, relation)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "src": a,
                    "dst": b,
                    "relation": relation,
                    "props": {"relation": relation},
                }
            )

        return rows

    def _batched(self, rows: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
        if batch_size <= 0:
            return [rows]
        return [rows[i : i + batch_size] for i in range(0, len(rows), batch_size)]

    def _sanitize_value(self, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, list):
            return [self._sanitize_value(v) for v in value]
        if isinstance(value, tuple):
            return [self._sanitize_value(v) for v in value]
        if isinstance(value, dict):
            out: dict[str, Any] = {}
            for k, v in value.items():
                out[str(k)] = self._sanitize_value(v)
            return out
        return str(value)

    def _quote(self, value: Any) -> str:
        return json.dumps(value, ensure_ascii=False)

    def _to_cypher_map(self, payload: dict[str, Any]) -> str:
        parts: list[str] = []
        for key, val in payload.items():
            parts.append(f"{key}: {self._quote(val)}")
        return "{" + ", ".join(parts) + "}"

    def _serialize_meta(self, meta: dict[str, Any]) -> str:
        return json.dumps(self._sanitize_value(meta), ensure_ascii=False)

    def _deserialize_meta(self, payload: Any, legacy: Any = None) -> dict[str, Any]:
        if isinstance(payload, str):
            try:
                value = json.loads(payload)
            except json.JSONDecodeError:
                return {}
            return value if isinstance(value, dict) else {}
        if isinstance(legacy, dict):
            return self._sanitize_value(legacy)
        return {}

    def _group_edge_rows_by_type(self, rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            rel_type = self._relation_type(str(row.get("relation") or "related"))
            grouped.setdefault(rel_type, []).append(row)
        return grouped

    def _group_node_rows_by_label(self, rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            props = row.get("props") or {}
            node_label = self._node_label(str(props.get("node_type") or "unknown"))
            grouped.setdefault(node_label, []).append(row)
        return grouped

    def _node_label(self, node_type: str) -> str:
        aliases = {
            "city": "City",
            "transport": "Transport",
            "attraction": "Attraction",
            "restaurant": "Restaurant",
            "hotel": "Hotel",
            "etype": "EntityType",
            "preference": "Preference",
            "pattern": "Pattern",
            "pattern_slot": "PatternSlot",
        }
        if node_type in aliases:
            return aliases[node_type]
        chunks = re.split(r"[^0-9A-Za-z]+", node_type.strip())
        label = "".join(chunk[:1].upper() + chunk[1:] for chunk in chunks if chunk)
        if not label:
            return "Unknown"
        if not label[0].isalpha():
            return f"Node{label}"
        return label

    def _relation_type(self, relation: str) -> str:
        sanitized = re.sub(r"[^0-9A-Za-z_]+", "_", relation.strip())
        sanitized = re.sub(r"_+", "_", sanitized).strip("_")
        if not sanitized:
            sanitized = "RELATED"
        if not sanitized[0].isalpha():
            sanitized = f"REL_{sanitized}"
        return sanitized.upper()

    def _quote_identifier(self, value: str) -> str:
        return "`" + value.replace("`", "``") + "`"
