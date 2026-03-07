from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .semantic import embed_texts
from .types import Candidate
from .utils import normalize_text, tokenize


class CandidateVectorIndex(Protocol):
    backend: str

    def search(
        self,
        query: str,
        allowed_ids: set[str] | None = None,
        city: str | None = None,
        top_k: int | None = None,
    ) -> dict[str, float]:
        ...


class TFIDFIndex:
    backend = "tfidf"

    def __init__(self, candidates: list[Candidate]) -> None:
        self.candidates = candidates
        self.candidate_city = {c.candidate_id: normalize_text(c.city) for c in candidates}
        self.doc_tokens: dict[str, Counter[str]] = {}
        self.doc_norm: dict[str, float] = {}
        self.idf: dict[str, float] = {}
        self.doc_freq: dict[str, int] = {}
        self._build()

    def _build(self) -> None:
        doc_freq: defaultdict[str, int] = defaultdict(int)
        for c in self.candidates:
            tokens = tokenize(c.text)
            if not tokens:
                continue
            tf = Counter(tokens)
            self.doc_tokens[c.candidate_id] = tf
            for tok in tf.keys():
                doc_freq[tok] += 1

        total_docs = max(1, len(self.doc_tokens))
        self.doc_freq = dict(doc_freq)
        self.idf = {tok: math.log((total_docs + 1) / (df + 1)) + 1.0 for tok, df in doc_freq.items()}

        for doc_id, tf in self.doc_tokens.items():
            sq = 0.0
            for term, cnt in tf.items():
                w = (1.0 + math.log(cnt)) * self.idf.get(term, 0.0)
                sq += w * w
            self.doc_norm[doc_id] = math.sqrt(sq) if sq > 0 else 1.0

    def search(
        self,
        query: str,
        allowed_ids: set[str] | None = None,
        city: str | None = None,
        top_k: int | None = None,
    ) -> dict[str, float]:
        qtokens = tokenize(query)
        if not qtokens:
            return {}
        qtf = Counter(qtokens)

        qweights: dict[str, float] = {}
        qsq = 0.0
        for term, cnt in qtf.items():
            idf = self.idf.get(term)
            if idf is None:
                continue
            w = (1.0 + math.log(cnt)) * idf
            qweights[term] = w
            qsq += w * w
        qnorm = math.sqrt(qsq) if qsq > 0 else 1.0

        scores: dict[str, float] = {}
        target_docs = self.doc_tokens.keys()
        if allowed_ids is not None:
            target_docs = [doc_id for doc_id in target_docs if doc_id in allowed_ids]
        elif city:
            normalized_city = normalize_text(city)
            target_docs = [doc_id for doc_id in target_docs if self.candidate_city.get(doc_id) == normalized_city]

        for doc_id in target_docs:
            tf = self.doc_tokens.get(doc_id)
            if not tf:
                continue
            dot = 0.0
            for term, qw in qweights.items():
                cnt = tf.get(term)
                if not cnt:
                    continue
                dw = (1.0 + math.log(cnt)) * self.idf.get(term, 0.0)
                dot += qw * dw
            if dot <= 0:
                continue
            denom = self.doc_norm.get(doc_id, 1.0) * qnorm
            scores[doc_id] = dot / denom if denom > 0 else 0.0

        if top_k is not None and top_k > 0 and len(scores) > top_k:
            ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
            return dict(ranked[:top_k])
        return scores


@dataclass(frozen=True)
class FaissCachePaths:
    index_path: Path
    metadata_path: Path
    embeddings_path: Path


class FAISSIndex:
    backend = "faiss"

    def __init__(
        self,
        candidates: list[Candidate],
        embed_model: str,
        cache_dir: Path | None = None,
        batch_size: int = 32,
        force_rebuild: bool = False,
    ) -> None:
        self.candidates = candidates
        self.embed_model = embed_model
        self.batch_size = max(1, batch_size)
        self.candidate_ids = [c.candidate_id for c in candidates]
        self.candidate_texts = [c.text for c in candidates]
        self.candidate_cities = [normalize_text(c.city) for c in candidates]
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._fingerprint = self._build_fingerprint()
        self._cache_paths = self._build_cache_paths()
        self.loaded_from_cache = False
        self._faiss = None
        self._index = None
        self._embeddings = None
        self._city_indexes: dict[str, tuple[object, list[int]]] = {}
        self._load_or_build(force_rebuild=force_rebuild)

    def search(
        self,
        query: str,
        allowed_ids: set[str] | None = None,
        city: str | None = None,
        top_k: int | None = None,
    ) -> dict[str, float]:
        if not query.strip() or not self.candidate_ids or self._index is None:
            return {}

        query_embeddings = embed_texts([query], model_name=self.embed_model, batch_size=1)
        search_index, position_map = self._resolve_search_scope(city)
        requested_k = len(position_map)
        if allowed_ids is None and top_k is not None and top_k > 0:
            requested_k = min(requested_k, top_k)
        scores, positions = search_index.search(query_embeddings, requested_k)
        allowed = set(allowed_ids) if allowed_ids is not None else None

        out: dict[str, float] = {}
        for score, pos in zip(scores[0], positions[0]):
            if pos < 0 or pos >= len(position_map):
                continue
            candidate_id = self.candidate_ids[position_map[int(pos)]]
            if allowed is not None and candidate_id not in allowed:
                continue
            out[candidate_id] = max(0.0, float(score))
        if allowed is None and top_k is not None and top_k > 0 and len(out) > top_k:
            ranked = sorted(out.items(), key=lambda item: item[1], reverse=True)
            return dict(ranked[:top_k])
        return out

    def _load_or_build(self, force_rebuild: bool) -> None:
        faiss, np = _import_faiss_dependencies()
        self._faiss = faiss

        if self._cache_paths is not None and not force_rebuild and self._cache_is_ready():
            self._index = faiss.read_index(str(self._cache_paths.index_path))
            self._embeddings = np.load(self._cache_paths.embeddings_path, allow_pickle=False)
            self.loaded_from_cache = True
            self._build_city_indexes()
            return

        if not self.candidate_texts:
            self._embeddings = np.zeros((0, 0), dtype="float32")
            self._index = faiss.IndexFlatIP(1)
            self.loaded_from_cache = False
            return

        embeddings = embed_texts(
            self.candidate_texts,
            model_name=self.embed_model,
            batch_size=self.batch_size,
        )
        embeddings = np.ascontiguousarray(embeddings, dtype="float32")
        dim = int(embeddings.shape[1])
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        self._embeddings = embeddings
        self._index = index
        self.loaded_from_cache = False
        self._build_city_indexes()
        self._write_cache()

    def _cache_is_ready(self) -> bool:
        if self._cache_paths is None:
            return False
        if not (
            self._cache_paths.index_path.exists()
            and self._cache_paths.metadata_path.exists()
            and self._cache_paths.embeddings_path.exists()
        ):
            return False
        try:
            payload = json.loads(self._cache_paths.metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return (
            payload.get("fingerprint") == self._fingerprint
            and payload.get("backend") == self.backend
            and payload.get("embed_model") == self.embed_model
            and payload.get("candidate_count") == len(self.candidate_ids)
        )

    def _write_cache(self) -> None:
        if self._cache_paths is None or self._index is None or self._embeddings is None:
            return
        self._cache_paths.index_path.parent.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(self._index, str(self._cache_paths.index_path))
        metadata = {
            "backend": self.backend,
            "fingerprint": self._fingerprint,
            "embed_model": self.embed_model,
            "candidate_count": len(self.candidate_ids),
            "dimension": int(self._embeddings.shape[1]) if self._embeddings.ndim == 2 else 0,
        }
        self._cache_paths.metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        import numpy as np

        np.save(self._cache_paths.embeddings_path, self._embeddings, allow_pickle=False)

    def _build_city_indexes(self) -> None:
        self._city_indexes = {}
        if self._embeddings is None or self._faiss is None:
            return
        if getattr(self._embeddings, "ndim", 0) != 2 or len(self._embeddings) == 0:
            return
        dim = int(self._embeddings.shape[1])
        grouped: dict[str, list[int]] = defaultdict(list)
        for idx, city in enumerate(self.candidate_cities):
            grouped[city or "__unknown__"].append(idx)
        for city, positions in grouped.items():
            if not positions:
                continue
            sub_index = self._faiss.IndexFlatIP(dim)
            sub_index.add(self._embeddings[positions])
            self._city_indexes[city] = (sub_index, positions)

    def _resolve_search_scope(self, city: str | None) -> tuple[object, list[int]]:
        normalized_city = normalize_text(city or "")
        if normalized_city and normalized_city in self._city_indexes:
            city_index, positions = self._city_indexes[normalized_city]
            return city_index, positions
        return self._index, list(range(len(self.candidate_ids)))

    def _build_fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.embed_model.encode("utf-8"))
        for candidate in self.candidates:
            digest.update(candidate.candidate_id.encode("utf-8"))
            digest.update(b"\x1f")
            digest.update(candidate.text.encode("utf-8"))
            digest.update(b"\x1e")
        return digest.hexdigest()[:16]

    def _build_cache_paths(self) -> FaissCachePaths | None:
        if self.cache_dir is None:
            return None
        base = self.cache_dir / self._fingerprint
        return FaissCachePaths(
            index_path=base.with_suffix(".faiss"),
            metadata_path=base.with_suffix(".json"),
            embeddings_path=base.with_suffix(".npy"),
        )


def _import_faiss_dependencies():
    try:
        import faiss  # type: ignore
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "FAISS retrieval requires `faiss-cpu` and `numpy`. "
            "Install them with: python -m pip install -e .[faiss]"
        ) from exc
    return faiss, np
