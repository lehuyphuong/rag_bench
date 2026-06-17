"""
Chunking strategies following Berdyugina et al. (arXiv:2604.24334) Section 3.3.

  FixedTokenChunker     : fixed-size character windows with optional overlap.
  RecursiveTokenChunker : hierarchical splitting (paragraphs → sentences → words).
  ClusterSemanticChunker: embeds sentences, groups by cosine-similarity breakpoints.

Each chunker returns a list of chunk dicts:
  {
    "chunk_id"  : str,   # "{doc_id}_{i}"
    "doc_id"    : str,
    "title"     : str,
    "text"      : str,
    "char_start": int,
    "char_end"  : int,
  }
"""

from __future__ import annotations

import logging
import re
import time
from typing import Callable

import numpy as np
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)

logger = logging.getLogger(__name__)


# ── helpers ──────────────────────────────────────────────────────────────────

def _sentence_split(text: str) -> list[str]:
    """Simple sentence splitter (no NLTK dependency)."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p.strip()]


def _make_chunks(doc: dict, spans: list[tuple[int, int]]) -> list[dict]:
    """Convert (start, end) character spans into chunk dicts."""
    chunks = []
    text = doc["text"]
    for i, (s, e) in enumerate(spans):
        chunk_text = text[s:e].strip()
        if not chunk_text:
            continue
        chunks.append({
            "chunk_id":   f"{doc['doc_id']}_{i}",
            "doc_id":     doc["doc_id"],
            "title":      doc["title"],
            "text":       chunk_text,
            "char_start": s,
            "char_end":   e,
        })
    return chunks


# ── FixedTokenChunker ─────────────────────────────────────────────────────────

def fixed_token_chunker(
    doc: dict, chunk_size: int = 400, overlap: int = 0
) -> list[dict]:
    """
    Split document into fixed-size character windows.
    Equivalent to FixedTokenChunker in Berdyugina et al.
    """
    splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separator="",
        strip_whitespace=True,
    )
    text = doc["text"]
    raw_chunks = splitter.split_text(text)

    spans = []
    cursor = 0
    for rc in raw_chunks:
        idx = text.find(rc, cursor)
        if idx == -1:
            idx = cursor
        spans.append((idx, idx + len(rc)))
        cursor = idx + max(1, len(rc) - overlap)

    return _make_chunks(doc, spans)


# ── RecursiveTokenChunker ─────────────────────────────────────────────────────

def recursive_token_chunker(
    doc: dict, chunk_size: int = 400, overlap: int = 0
) -> list[dict]:
    """
    Recursive splitting: paragraph → sentence → word separators.
    Equivalent to RecursiveTokenChunker in Berdyugina et al.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        strip_whitespace=True,
    )
    text = doc["text"]
    raw_chunks = splitter.split_text(text)

    spans = []
    cursor = 0
    for rc in raw_chunks:
        idx = text.find(rc, cursor)
        if idx == -1:
            idx = cursor
        spans.append((idx, idx + len(rc)))
        cursor = idx + max(1, len(rc) - overlap)

    return _make_chunks(doc, spans)


# ── ClusterSemanticChunker ────────────────────────────────────────────────────

def cluster_semantic_chunker(
    doc: dict,
    chunk_size: int = 400,
    overlap: int = 0,
    embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
    breakpoint_percentile: float = 95.0,
) -> list[dict]:
    """
    Semantic chunking via sentence embedding + cosine-distance breakpoints.
    Equivalent to ClusterSemanticChunker in Berdyugina et al.

    Algorithm:
      1. Split document into sentences.
      2. Embed each sentence.
      3. Compute cosine distance between consecutive sentence embeddings.
      4. Split at distances above the breakpoint_percentile threshold.
      5. Merge consecutive semantic groups until max chunk_size is reached.
    """
    if embed_fn is None:
        logger.warning(
            "ClusterSemantic: no embed_fn — falling back to RecursiveToken."
        )
        return recursive_token_chunker(doc, chunk_size=chunk_size, overlap=overlap)

    text = doc["text"]
    sentences = _sentence_split(text)

    if len(sentences) <= 1:
        return _make_chunks(doc, [(0, len(text))])

    # Step 1: embed all sentences
    vectors = embed_fn(sentences)
    vecs = np.array(vectors, dtype=np.float32)

    # Normalize for cosine similarity
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-9, norms)
    vecs = vecs / norms

    # Step 2: pairwise distances between consecutive sentences
    dots = np.sum(vecs[:-1] * vecs[1:], axis=1)
    distances = 1.0 - dots

    # Step 3: find breakpoints
    threshold = float(np.percentile(distances, breakpoint_percentile))
    breakpoints = {i + 1 for i, d in enumerate(distances) if d >= threshold}

    # Step 4: group sentences into semantic groups
    groups: list[list[str]] = []
    current: list[str] = [sentences[0]]
    for i, sent in enumerate(sentences[1:], start=1):
        if i in breakpoints:
            groups.append(current)
            current = [sent]
        else:
            current.append(sent)
    groups.append(current)

    # Step 5: merge groups until chunk_size is respected
    final_texts: list[str] = []
    current_text = ""
    for group in groups:
        group_text = " ".join(group)
        candidate = (current_text + " " + group_text).strip() if current_text else group_text
        if len(candidate) <= chunk_size or not current_text:
            current_text = candidate
        else:
            if current_text:
                final_texts.append(current_text)
            current_text = group_text
    if current_text:
        final_texts.append(current_text)

    # Reconstruct char spans
    spans = []
    cursor = 0
    for ft in final_texts:
        idx = text.find(ft[:30], cursor)
        if idx == -1:
            idx = cursor
        spans.append((idx, idx + len(ft)))
        cursor = idx + 1

    return _make_chunks(doc, spans)


# ── Registry ──────────────────────────────────────────────────────────────────

def get_chunker(strategy: str, chunk_size: int, overlap: int, embed_fn=None):
    """Return a callable(doc) → list[chunk] for the given strategy."""
    if strategy == "FixedToken":
        return lambda doc: fixed_token_chunker(doc, chunk_size=chunk_size, overlap=overlap)
    elif strategy == "RecursiveToken":
        return lambda doc: recursive_token_chunker(doc, chunk_size=chunk_size, overlap=overlap)
    elif strategy == "ClusterSemantic":
        return lambda doc: cluster_semantic_chunker(
            doc, chunk_size=chunk_size, overlap=overlap, embed_fn=embed_fn
        )
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")


def chunk_documents(
    documents: list[dict],
    strategy: str,
    chunk_size: int,
    overlap: int,
    embed_fn=None,
) -> tuple[list[dict], float]:
    """
    Chunk all documents with the given strategy.
    Returns (chunks, elapsed_seconds).
    """
    chunker = get_chunker(strategy, chunk_size, overlap, embed_fn=embed_fn)
    t0 = time.perf_counter()
    all_chunks = []
    for doc in documents:
        all_chunks.extend(chunker(doc))
    elapsed = time.perf_counter() - t0

    logger.info(
        "Chunked %d docs → %d chunks [%s size=%d overlap=%d] in %.2fs",
        len(documents), len(all_chunks), strategy, chunk_size, overlap, elapsed,
    )
    return all_chunks, elapsed
