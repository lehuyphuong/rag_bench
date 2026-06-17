"""
Chunk filtering strategies — Berdyugina et al. (arXiv:2604.24334) Section 3.5.

Implemented methods (matching paper's evaluated strategies):

  NoFilter      — pass-through baseline (no chunks removed)
  ExactNorm     — exact dedup after text normalization (Section 3.5)
  MinHashLSH    — near-duplicate detection via Jaccard / MinHash (Section 3.5)
                  threshold=0.7 (paper tests 0.6/0.7/0.8; 0.7 = balanced)
  Similarity    — cosine similarity threshold on dense embeddings (Section 3.5)
                  threshold=0.8 (paper's "most stable" value)
  NERExact      — drop chunks with identical named-entity set (Section 3.5)
                  E(cᵢ) = E(cⱼ) → remove one

Each filter class exposes:
    apply(chunks: list[dict], embed_fn=None) → list[dict]

Greedy removal strategy (paper Section 3.5):
  When two chunks are judged redundant, the LATER chunk (higher index) is
  removed — preserving the first occurrence in document order.
"""

from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)


# ── helpers ───────────────────────────────────────────────────────────────────

def _normalize_text(text: str) -> str:
    """
    Normalize text for exact dedup:
      - Unicode NFKC normalization
      - Lowercase
      - Collapse whitespace
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _cosine_matrix(vecs: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine similarity for already L2-normalized vectors.
    Returns upper-triangular matrix (diagonal excluded).
    vecs shape: (N, D), assumed unit-norm.
    """
    # dot product of unit vectors = cosine similarity
    return vecs @ vecs.T


# ── NoFilter ──────────────────────────────────────────────────────────────────

class NoFilter:
    """Pass-through — returns chunks unchanged. Used as the paper baseline."""

    def apply(
        self,
        chunks: list[dict],
        embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
    ) -> list[dict]:
        return chunks


# ── ExactNorm ─────────────────────────────────────────────────────────────────

class ExactNorm:
    """
    Exact deduplication after text normalization.

    Two chunks are duplicates when their normalized textual forms are identical.
    Paper Section 3.5: "exact deduplication is applied after text normalization."

    Complexity: O(N) using a hash set.
    """

    def apply(
        self,
        chunks: list[dict],
        embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
    ) -> list[dict]:
        seen: set[str] = set()
        kept: list[dict] = []
        for chunk in chunks:
            key = hashlib.sha256(_normalize_text(chunk["text"]).encode()).hexdigest()
            if key not in seen:
                seen.add(key)
                kept.append(chunk)

        removed = len(chunks) - len(kept)
        if removed:
            logger.debug("ExactNorm: removed %d / %d chunks", removed, len(chunks))
        return kept


# ── MinHashLSH ────────────────────────────────────────────────────────────────

class MinHashLSH:
    """
    Near-duplicate detection via MinHash + LSH (Jaccard similarity).

    Paper Section 3.5:
      "MinHash signatures are used to approximate Jaccard similarity between
       chunks. Candidate duplicates are retrieved through LSH, and chunks whose
       estimated lexical similarity exceeds a threshold are removed."

    Uses datasketch library (same algorithm as paper references Broder 1997/2000).
    threshold: Jaccard similarity above which one chunk is removed (default 0.7).
    num_perm : number of MinHash permutations (higher = more accurate).
    """

    def __init__(self, threshold: float = 0.7, num_perm: int = 128):
        self.threshold = threshold
        self.num_perm  = num_perm

    def _shingles(self, text: str, k: int = 3) -> set[str]:
        """Character k-grams as shingle set."""
        t = _normalize_text(text)
        return {t[i : i + k] for i in range(max(1, len(t) - k + 1))}

    def apply(
        self,
        chunks: list[dict],
        embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
    ) -> list[dict]:
        try:
            from datasketch import MinHash, MinHashLSH as _LSH
        except ImportError:
            logger.error(
                "datasketch not installed. Run: pip install datasketch"
            )
            return chunks

        lsh = _LSH(threshold=self.threshold, num_perm=self.num_perm)
        minhashes: list[MinHash] = []

        # Build MinHash for each chunk
        for chunk in chunks:
            m = MinHash(num_perm=self.num_perm)
            for shingle in self._shingles(chunk["text"]):
                m.update(shingle.encode("utf-8"))
            minhashes.append(m)

        # Greedy dedup: insert into LSH; if a near-duplicate already exists → skip
        kept_indices: list[int] = []
        for i, (chunk, m) in enumerate(zip(chunks, minhashes)):
            key = f"chunk_{i}"
            candidates = lsh.query(m)
            if candidates:
                # Near-duplicate of an already-kept chunk → remove
                continue
            lsh.insert(key, m)
            kept_indices.append(i)

        kept = [chunks[i] for i in kept_indices]
        removed = len(chunks) - len(kept)
        if removed:
            logger.debug(
                "MinHashLSH(%.2f): removed %d / %d chunks",
                self.threshold, removed, len(chunks),
            )
        return kept


# ── Similarity ────────────────────────────────────────────────────────────────

class Similarity:
    """
    Semantic deduplication via cosine similarity on dense embeddings.

    Paper Section 3.5, eq.:
      Redundant(cᵢ, cⱼ) iff sim(cᵢ, cⱼ) ≥ τ

    Greedy algorithm (O(N²) pairs):
      1. Embed all chunks.
      2. For each chunk (in order), check if any already-kept chunk has
         cosine similarity ≥ threshold.
      3. If yes → discard current chunk.
      4. If no  → keep it.

    threshold: cosine similarity above which chunks are considered redundant.
               Paper uses 0.8 as the primary reported value.
    """

    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold

    def apply(
        self,
        chunks: list[dict],
        embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
    ) -> list[dict]:
        if not chunks:
            return chunks
        if embed_fn is None:
            logger.warning("Similarity filter: no embed_fn provided — skipping.")
            return chunks

        texts = [c["text"] for c in chunks]
        vecs  = np.array(embed_fn(texts), dtype=np.float32)

        # Ensure unit norm (embed_fn should already normalize, but just in case)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-9, norms)
        vecs  = vecs / norms

        kept_indices: list[int] = []
        kept_vecs: list[np.ndarray] = []

        for i, vec in enumerate(vecs):
            if not kept_vecs:
                kept_indices.append(i)
                kept_vecs.append(vec)
                continue

            # Cosine similarity against all kept chunks
            kept_mat = np.stack(kept_vecs, axis=0)   # (K, D)
            sims = kept_mat @ vec                      # (K,)

            if sims.max() >= self.threshold:
                # Too similar to an existing kept chunk → discard
                continue

            kept_indices.append(i)
            kept_vecs.append(vec)

        kept = [chunks[i] for i in kept_indices]
        removed = len(chunks) - len(kept)
        if removed:
            logger.debug(
                "Similarity(%.2f): removed %d / %d chunks",
                self.threshold, removed, len(chunks),
            )
        return kept


# ── NERExact ──────────────────────────────────────────────────────────────────

class NERExact:
    """
    Entity-based deduplication: remove chunks with identical named-entity sets.

    Paper Section 3.5:
      "For exact entity-based filtering, two chunks are considered redundant if
       E(cᵢ) = E(cⱼ)."

    Entity extraction: spaCy en_core_web_sm (persons, orgs, locations, dates,
    domain-specific concepts).

    Chunks with NO recognized entities are left unchanged (paper limitation note
    Section 3.5: "NER Exact only applies to chunks containing recognized named
    entities … segments without recognized named entities remain unaffected").

    Greedy: first occurrence is kept, later duplicates are removed.
    """

    def __init__(self):
        self._nlp = None

    def _get_nlp(self):
        if self._nlp is None:
            import spacy
            try:
                # Only need NER component
                self._nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
            except OSError:
                raise RuntimeError(
                    "spaCy model not found. Run: python -m spacy download en_core_web_sm"
                )
        return self._nlp

    def _extract_entities(self, text: str) -> frozenset[str]:
        """Return frozenset of (entity_text, entity_label) pairs."""
        nlp = self._get_nlp()
        doc = nlp(text)
        return frozenset(
            (ent.text.strip().lower(), ent.label_)
            for ent in doc.ents
            if ent.text.strip()
        )

    def apply(
        self,
        chunks: list[dict],
        embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
    ) -> list[dict]:
        seen_entity_sets: set[frozenset] = set()
        kept: list[dict] = []
        no_entity_count = 0

        for chunk in chunks:
            entities = self._extract_entities(chunk["text"])

            if not entities:
                # No named entities → keep unconditionally (paper behavior)
                kept.append(chunk)
                no_entity_count += 1
                continue

            if entities in seen_entity_sets:
                # Exact same entity set as a kept chunk → remove
                continue

            seen_entity_sets.add(entities)
            kept.append(chunk)

        removed = len(chunks) - len(kept)
        if removed:
            logger.debug(
                "NERExact: removed %d / %d chunks (%d had no entities → kept)",
                removed, len(chunks), no_entity_count,
            )
        return kept


# ── Registry ──────────────────────────────────────────────────────────────────

_REGISTRY: dict[str, type] = {
    "NoFilter":   NoFilter,
    "ExactNorm":  ExactNorm,
    "MinHashLSH": MinHashLSH,
    "Similarity": Similarity,
    "NERExact":   NERExact,
}


def get_filter(method: str, **kwargs):
    """Instantiate a filter by method name."""
    if method not in _REGISTRY:
        raise ValueError(
            f"Unknown filter method '{method}'. "
            f"Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[method](**kwargs)
