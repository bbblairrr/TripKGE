from __future__ import annotations

import math
from collections import Counter, defaultdict

from .types import Candidate
from .utils import tokenize


class TFIDFIndex:
    def __init__(self, candidates: list[Candidate]) -> None:
        self.candidates = candidates
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

    def search(self, query: str, allowed_ids: set[str] | None = None) -> dict[str, float]:
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

        return scores
