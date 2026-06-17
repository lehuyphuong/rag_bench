"""
Embedding layer:
  Dense  : nomic-embed-text-v1.5 via Ollama HTTP API (768-dim)
  Sparse : BM25 via qdrant_client.models.SparseVector (no model needed)

embed_texts(texts)         → list of dense float vectors
embed_sparse(texts)        → list of SparseVector (indices + values)
embed_chunks_batched(...)  → generator yielding (chunk, dense_vec, sparse_vec)
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from typing import Generator

import httpx

from configs.settings import (
    EMBED_BATCH_SIZE,
    EMBED_MODEL,
    OLLAMA_BASE_URL,
    TEXT_EMBED_DIM,
)

logger = logging.getLogger(__name__)

_EMBED_URL = f"{OLLAMA_BASE_URL}/api/embed"
_HTTP_TIMEOUT = 120.0

# ── Dense (Ollama) ────────────────────────────────────────────────────────────

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of strings via Ollama nomic-embed-text."""
    if not texts:
        return []
    safe = [t if t.strip() else " " for t in texts]
    with httpx.Client(timeout=_HTTP_TIMEOUT) as client:
        resp = client.post(_EMBED_URL, json={"model": EMBED_MODEL, "input": safe})
        resp.raise_for_status()
    data = resp.json()
    embeddings = data.get("embeddings")
    if not embeddings or len(embeddings) != len(texts):
        raise ValueError(f"Unexpected Ollama response: got {len(embeddings) if embeddings else 0} embeddings for {len(texts)} inputs")
    return embeddings


# ── Sparse (BM25) ─────────────────────────────────────────────────────────────
# BM25 term weights: TF-IDF variant with saturation.
# We compute a simple BM25-inspired sparse vector per document:
#   - tokenize to lowercase words (no stopword removal at this stage;
#     BM25 scoring naturally down-weights common terms via IDF)
#   - map each token to an integer index via a vocabulary
#   - value = tf * idf_approx (using log(1 + tf) as tf weight)
#
# Vocabulary is built lazily from the corpus and stored as a module-level
# singleton so all calls in the same process share the same token→index map.

_vocab: dict[str, int] = {}
_df: Counter = Counter()   # document frequency per token
_n_docs: int = 0           # total documents seen

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def build_bm25_vocab(texts: list[str]) -> None:
    """
    Build/update the BM25 vocabulary from a corpus of texts.
    Must be called before embed_sparse() to ensure meaningful IDF weights.
    """
    global _n_docs
    for text in texts:
        tokens = set(_tokenize(text))
        for t in tokens:
            if t not in _vocab:
                _vocab[t] = len(_vocab)
            _df[t] += 1
        _n_docs += 1
    logger.info("BM25 vocab size: %d (from %d docs)", len(_vocab), _n_docs)


def embed_sparse(texts: list[str]) -> list[dict]:
    """
    Compute BM25-style sparse vectors for a list of texts.

    Returns list of dicts {"indices": list[int], "values": list[float]}
    compatible with qdrant_client SparseVector.
    """
    from qdrant_client.models import SparseVector

    results = []
    idf_total = max(_n_docs, 1)

    for text in texts:
        tokens = _tokenize(text)
        tf = Counter(tokens)
        indices = []
        values = []
        for token, count in tf.items():
            if token not in _vocab:
                continue
            idx = _vocab[token]
            df = _df.get(token, 1)
            idf = math.log(1 + idf_total / df)
            weight = math.log(1 + count) * idf
            indices.append(idx)
            values.append(float(weight))
        results.append(SparseVector(indices=indices, values=values))
    return results


# ── Combined batched embedding ────────────────────────────────────────────────

def embed_chunks_batched(
    chunks: list[dict],
    batch_size: int = EMBED_BATCH_SIZE,
) -> Generator[tuple[dict, list[float], object], None, None]:
    """
    Yields (chunk, dense_vector, sparse_vector) for each chunk.

    Processes chunks in batches to avoid OOM on large corpora.
    BM25 vocab must be built before calling this (via build_bm25_vocab).
    """
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c["text"] for c in batch]
        dense_vecs = embed_texts(texts)
        sparse_vecs = embed_sparse(texts)
        for chunk, dv, sv in zip(batch, dense_vecs, sparse_vecs):
            yield chunk, dv, sv
