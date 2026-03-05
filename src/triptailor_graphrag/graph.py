from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

from .config import ExperimentConfig
from .data_loader import DatasetBundle
from .pattern import PatternMiner
from .types import Candidate
from .utils import slugify


@dataclass
class GraphNode:
    node_id: str
    layer: int
    node_type: str
    label: str
    meta: dict[str, Any] = field(default_factory=dict)


class MultiLayerGraph:
    def __init__(self) -> None:
        self.nodes: dict[str, GraphNode] = {}
        self.adj: dict[str, set[str]] = defaultdict(set)
        self.edge_rel: dict[tuple[str, str], str] = {}

    def add_node(self, node_id: str, layer: int, node_type: str, label: str, meta: dict[str, Any] | None = None) -> None:
        if node_id not in self.nodes:
            self.nodes[node_id] = GraphNode(
                node_id=node_id,
                layer=layer,
                node_type=node_type,
                label=label,
                meta=meta or {},
            )

    def add_edge(self, src: str, dst: str, relation: str) -> None:
        if src == dst:
            return
        if src not in self.nodes or dst not in self.nodes:
            return
        self.adj[src].add(dst)
        self.adj[dst].add(src)
        self.edge_rel[(src, dst)] = relation
        self.edge_rel[(dst, src)] = relation

    def neighbors(self, node_id: str) -> set[str]:
        return self.adj.get(node_id, set())

    def bfs_expand(self, seeds: set[str], hops: int, candidate_only: bool = False) -> set[str]:
        if not seeds or hops <= 0:
            return set(seeds)
        visited = set(seeds)
        frontier = deque((seed, 0) for seed in seeds if seed in self.nodes)

        while frontier:
            node, depth = frontier.popleft()
            if depth >= hops:
                continue
            for nxt in self.neighbors(node):
                if nxt in visited:
                    continue
                visited.add(nxt)
                frontier.append((nxt, depth + 1))

        if not candidate_only:
            return visited

        return {n for n in visited if self.is_candidate_node(n)}

    def connected_components(self, node_ids: set[str]) -> list[set[str]]:
        remaining = {n for n in node_ids if n in self.nodes}
        components: list[set[str]] = []

        while remaining:
            start = next(iter(remaining))
            comp = set()
            queue = deque([start])
            remaining.remove(start)
            while queue:
                cur = queue.popleft()
                comp.add(cur)
                for nxt in self.neighbors(cur):
                    if nxt in remaining:
                        remaining.remove(nxt)
                        queue.append(nxt)
            components.append(comp)

        components.sort(key=lambda c: len(c), reverse=True)
        return components

    def shortest_path(self, src: str, dst: str, max_hops: int = 4) -> list[str]:
        if src == dst:
            return [src]
        if src not in self.nodes or dst not in self.nodes:
            return []

        queue = deque([(src, [src])])
        visited = {src}
        while queue:
            node, path = queue.popleft()
            if len(path) - 1 >= max_hops:
                continue
            for nxt in self.neighbors(node):
                if nxt in visited:
                    continue
                new_path = path + [nxt]
                if nxt == dst:
                    return new_path
                visited.add(nxt)
                queue.append((nxt, new_path))
        return []

    def is_candidate_node(self, node_id: str) -> bool:
        node = self.nodes.get(node_id)
        if not node:
            return False
        return node.node_type in {"attraction", "restaurant", "hotel"}


class GraphBuilder:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    def build(self, bundle: DatasetBundle, pattern_miner: PatternMiner) -> MultiLayerGraph:
        graph = MultiLayerGraph()

        self._add_layer1_city_transport(graph, bundle)
        self._add_layer2_poi(graph, bundle)
        self._add_layer3_preferences(graph, bundle)
        self._add_layer4_patterns(graph, pattern_miner)
        self._add_soft_geo_links(graph, bundle)

        return graph

    def _add_layer1_city_transport(self, graph: MultiLayerGraph, bundle: DatasetBundle) -> None:
        cities = set()
        for query in bundle.query_specs:
            cities.add(query.departure_city)
            cities.add(query.destination_city)
        for c in bundle.candidates_global.values():
            cities.add(c.city)

        for city in cities:
            if not city:
                continue
            graph.add_node(f"city:{slugify(city)}", 1, "city", city)

        for city, records in bundle.city_transport.items():
            city_node = f"city:{slugify(city)}"
            if city_node not in graph.nodes:
                continue
            # Keep only the cheapest routes to avoid transport-dense graph noise.
            ordered = sorted(records, key=lambda r: (r.get("price") or 999999, str(r.get("number") or "")))
            for rec in ordered[: self.config.city_transport_topn]:
                number = str(rec.get("number") or "NA")
                mode = str(rec.get("mode") or "transport")
                target_city = str(rec.get("to") or "")
                target_node = f"city:{slugify(target_city)}"
                if target_node not in graph.nodes:
                    continue

                tnode = f"transport:{mode}:{slugify(number)}:{slugify(city)}:{slugify(target_city)}"
                graph.add_node(
                    tnode,
                    1,
                    "transport",
                    f"{mode.upper()} {number}",
                    meta={
                        "mode": mode,
                        "number": number,
                        "from": city,
                        "to": target_city,
                        "price": rec.get("price"),
                    },
                )
                graph.add_edge(city_node, tnode, "departs")
                graph.add_edge(tnode, target_node, "arrives")

    def _add_layer2_poi(self, graph: MultiLayerGraph, bundle: DatasetBundle) -> None:
        graph.add_node("etype:attraction", 2, "etype", "attraction")
        graph.add_node("etype:restaurant", 2, "etype", "restaurant")
        graph.add_node("etype:hotel", 2, "etype", "hotel")
        graph.add_node("etype:transport", 2, "etype", "transport")

        for candidate in bundle.candidates_global.values():
            graph.add_node(
                candidate.candidate_id,
                2,
                candidate.entity_type,
                candidate.name,
                meta={
                    "city": candidate.city,
                    "price": candidate.price,
                    "tags": candidate.tags,
                },
            )
            city_node = f"city:{slugify(candidate.city)}"
            if city_node in graph.nodes:
                graph.add_edge(candidate.candidate_id, city_node, "in_city")

            etype = f"etype:{candidate.entity_type}"
            if etype in graph.nodes:
                graph.add_edge(candidate.candidate_id, etype, "is_type")

    def _add_layer3_preferences(self, graph: MultiLayerGraph, bundle: DatasetBundle) -> None:
        preference_nodes = {
            "pref:budget:low": "budget_low",
            "pref:budget:medium": "budget_medium",
            "pref:budget:high": "budget_high",
            "pref:budget:premium": "budget_premium",
            "pref:intensity:low": "intensity_low",
            "pref:intensity:moderate": "intensity_moderate",
            "pref:intensity:high": "intensity_high",
        }
        for node_id, label in preference_nodes.items():
            graph.add_node(node_id, 3, "preference", label)

        tags_seen: set[str] = set()
        for c in bundle.candidates_global.values():
            bucket = self._price_bucket(c.price)
            graph.add_edge(c.candidate_id, f"pref:budget:{bucket}", "fits_budget")
            if c.entity_type == "attraction":
                graph.add_edge(c.candidate_id, "pref:intensity:moderate", "default_intensity")

            for tag in c.tags[:6]:
                tag_norm = slugify(tag)
                if not tag_norm:
                    continue
                node_id = f"pref:tag:{tag_norm}"
                if node_id not in tags_seen:
                    graph.add_node(node_id, 3, "preference", tag)
                    tags_seen.add(node_id)
                graph.add_edge(c.candidate_id, node_id, "has_tag")

    def _add_layer4_patterns(self, graph: MultiLayerGraph, pattern_miner: PatternMiner) -> None:
        # layer-4: day-slot-activity skeletons mined from reference itineraries.
        for day_count in range(2, 8):
            pattern = pattern_miner.get_pattern(day_count)
            pattern_id = f"pattern:days:{day_count}"
            graph.add_node(
                pattern_id,
                4,
                "pattern",
                f"{day_count}-day skeleton",
                meta={"support": pattern.support},
            )

            for day_idx, day_tokens in enumerate(pattern.signature, start=1):
                for slot_idx, token in enumerate(day_tokens, start=1):
                    slot_id = f"pattern_slot:{day_count}:{day_idx}:{slot_idx}:{slugify(token)}"
                    graph.add_node(slot_id, 4, "pattern_slot", token)
                    graph.add_edge(pattern_id, slot_id, "contains")

                    action = token.split(":")[-1]
                    if action == "dining":
                        graph.add_edge(slot_id, "etype:restaurant", "expects")
                    elif action == "sightseeing":
                        graph.add_edge(slot_id, "etype:attraction", "expects")
                    elif action == "checkin":
                        graph.add_edge(slot_id, "etype:hotel", "expects")
                    elif action == "transport":
                        graph.add_edge(slot_id, "etype:transport", "expects")

    def _add_soft_geo_links(self, graph: MultiLayerGraph, bundle: DatasetBundle) -> None:
        # Cheap geo-neighborhood links via coarse cells, avoids O(N^2) pairwise scans.
        cells: dict[tuple[str, int, int], list[Candidate]] = defaultdict(list)
        for c in bundle.candidates_global.values():
            if c.latitude is None or c.longitude is None:
                continue
            cell = (slugify(c.city), int(c.latitude * 10), int(c.longitude * 10))
            cells[cell].append(c)

        for (_, _, _), group in cells.items():
            if len(group) <= 1:
                continue
            limit = min(len(group), 18)
            for i in range(limit):
                for j in range(i + 1, limit):
                    a = group[i]
                    b = group[j]
                    graph.add_edge(a.candidate_id, b.candidate_id, "nearby")

    def _price_bucket(self, price: float) -> str:
        if price <= 50:
            return "low"
        if price <= 150:
            return "medium"
        if price <= 300:
            return "high"
        return "premium"
