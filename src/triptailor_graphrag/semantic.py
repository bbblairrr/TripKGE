from __future__ import annotations

import os
from functools import lru_cache


DEFAULT_EMBED_MODEL = os.getenv("TRIPTAILOR_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DEFAULT_EMBED_BATCH_SIZE = int(os.getenv("TRIPTAILOR_EMBED_BATCH_SIZE", "32"))


def cosine_similarity(text_a: str, text_b: str) -> float:
    if not text_a or not text_b:
        return 0.0
    embeddings = embed_texts([text_a, text_b], model_name=DEFAULT_EMBED_MODEL)
    return float((embeddings[0] * embeddings[1]).sum().item())


def embed_texts(
    texts: list[str],
    model_name: str = DEFAULT_EMBED_MODEL,
    batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
):
    if not texts:
        raise ValueError("`texts` must not be empty.")
    tokenizer, model, torch = _load_embedder(model_name)
    batches = []
    actual_batch_size = max(1, batch_size)
    for start in range(0, len(texts), actual_batch_size):
        inputs = tokenizer(
            texts[start : start + actual_batch_size],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = model(**inputs)
        pooled = _mean_pool(outputs.last_hidden_state, inputs["attention_mask"], torch)
        normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
        batches.append(normalized.cpu())
    return torch.cat(batches, dim=0).numpy().astype("float32", copy=False)


@lru_cache(maxsize=1)
def _load_embedder(model_name: str):
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModel.from_pretrained(model_name, local_files_only=True)
    model.eval()
    return tokenizer, model, torch


def _mean_pool(last_hidden_state, attention_mask, torch):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    pooled = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-9)
    return pooled / denom
