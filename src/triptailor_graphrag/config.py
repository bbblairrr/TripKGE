from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


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
class LocalLLMConfig:
    enabled: bool = False
    model_path: str | None = None
    tokenizer_path: str | None = None
    device_map: str = "auto"
    torch_dtype: str = "auto"
    trust_remote_code: bool = True
    max_input_chars: int = 20000
    summary_max_new_tokens: int = 512
    planner_max_new_tokens: int = 768
    judge_max_new_tokens: int = 160
    temperature: float = 0.0
    top_p: float = 0.9
    repetition_penalty: float = 1.0
    enable_summary: bool = True
    enable_planner: bool = True
    enable_judge: bool = True


@dataclass(frozen=True)
class ExperimentConfig:
    data_dir: Path = Path("data")
    output_dir: Path = Path("outputs")
    city_transport_topn: int = 3
    min_restaurants_per_day: int = 2
    min_attractions_per_day: int = 2
    retrieval_weights: RetrievalWeights = RetrievalWeights()
    ablation: AblationConfig = AblationConfig()
    local_llm: LocalLLMConfig = LocalLLMConfig()
    random_seed: int = 42
