from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LocalLLMConfig:
    backend: str | None = None
    model: str | None = None
    max_candidates: int = 24
    temperature: float = 0.0
    max_new_tokens: int = 768
    timeout_seconds: int = 120
    fallback_to_heuristic: bool = True


@dataclass(frozen=True)
class RetrievalWeights:
    vector: float = 0.5
    constraint: float = 0.25
    graph: float = 0.25


@dataclass(frozen=True)
class AblationConfig:
    use_vector: bool = True
    use_graph_expansion: bool = True
    use_community_retrieval: bool = True
    use_summary_layer: bool = True
    hops: int = 2
    topk_vector: int = 30
    topk_final: int = 20


@dataclass(frozen=True)
class ExperimentConfig:
    data_dir: Path = Path("data")
    output_dir: Path = Path("outputs")
    city_transport_topn: int = 3
    min_restaurants_per_day: int = 2
    min_attractions_per_day: int = 2
    retrieval_weights: RetrievalWeights = RetrievalWeights()
    ablation: AblationConfig = AblationConfig()
    random_seed: int = 42
    llm: LocalLLMConfig = LocalLLMConfig()
