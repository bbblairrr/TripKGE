from __future__ import annotations

import os
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
    generation_retries: int = 3


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
class GraphStoreConfig:
    source: str = os.getenv("TRIPTAILOR_GRAPH_SOURCE", "auto")
    neo4j_uri: str = os.getenv("TRIPTAILOR_NEO4J_URI", "bolt://localhost:7687")
    neo4j_user: str = os.getenv("TRIPTAILOR_NEO4J_USER", "neo4j")
    neo4j_password: str | None = os.getenv("TRIPTAILOR_NEO4J_PASSWORD")
    neo4j_database: str = os.getenv("TRIPTAILOR_NEO4J_DATABASE", "neo4j")
    neo4j_batch_size: int = int(os.getenv("TRIPTAILOR_NEO4J_BATCH_SIZE", "1000"))
    bootstrap_if_missing: bool = os.getenv("TRIPTAILOR_NEO4J_BOOTSTRAP", "1") not in {"0", "false", "False"}
    clear_on_bootstrap: bool = False


@dataclass(frozen=True)
class VectorStoreConfig:
    backend: str = os.getenv("TRIPTAILOR_VECTOR_BACKEND", "auto")
    cache_dir: Path = Path(os.getenv("TRIPTAILOR_VECTOR_CACHE_DIR", ".cache/faiss"))
    embed_model: str = os.getenv("TRIPTAILOR_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    embed_batch_size: int = int(os.getenv("TRIPTAILOR_EMBED_BATCH_SIZE", "32"))
    force_rebuild: bool = False


@dataclass(frozen=True)
class ExperimentConfig:
    data_dir: Path = Path("data")
    train_file: str = "train.json"
    eval_file: str = "test.json"
    info_file: str = "infomation.json"
    output_dir: Path = Path("outputs")
    city_transport_topn: int = 3
    min_restaurants_per_day: int = 2
    min_attractions_per_day: int = 2
    retrieval_weights: RetrievalWeights = RetrievalWeights()
    ablation: AblationConfig = AblationConfig()
    random_seed: int = 42
    llm: LocalLLMConfig = LocalLLMConfig()
    judge_llm: LocalLLMConfig = LocalLLMConfig()
    graph_store: GraphStoreConfig = GraphStoreConfig()
    vector_store: VectorStoreConfig = VectorStoreConfig()
