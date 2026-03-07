from __future__ import annotations

from dataclasses import dataclass

from .config import ExperimentConfig
from .types import Candidate
from .vector_index import CandidateVectorIndex, FAISSIndex, TFIDFIndex


@dataclass(frozen=True)
class VectorIndexResolution:
    index: CandidateVectorIndex
    backend: str
    action: str
    details: str | None = None


def resolve_vector_index(
    config: ExperimentConfig,
    candidates: list[Candidate],
) -> VectorIndexResolution:
    vector_store = config.vector_store
    backend = (vector_store.backend or "auto").lower()
    if backend not in {"auto", "tfidf", "faiss"}:
        raise ValueError(f"Unsupported vector backend: {vector_store.backend}")

    if backend == "tfidf":
        return VectorIndexResolution(index=TFIDFIndex(candidates), backend="tfidf", action="built_tfidf")

    try:
        index = FAISSIndex(
            candidates=candidates,
            embed_model=vector_store.embed_model,
            cache_dir=vector_store.cache_dir,
            batch_size=vector_store.embed_batch_size,
            force_rebuild=vector_store.force_rebuild,
        )
        action = "loaded_faiss" if index.loaded_from_cache else "built_faiss"
        return VectorIndexResolution(index=index, backend="faiss", action=action)
    except Exception as exc:
        if backend != "auto":
            raise
        return VectorIndexResolution(
            index=TFIDFIndex(candidates),
            backend="tfidf",
            action="fallback_tfidf",
            details=f"FAISS unavailable; fell back to TF-IDF: {exc}",
        )
