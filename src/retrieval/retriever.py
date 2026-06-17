"""
Hybrid retrieval: dense (nomic cosine) + sparse (BM25) via Qdrant.

Uses Qdrant's query_points API with named vectors.
Hybrid score = alpha * dense_score + (1-alpha) * sparse_score
"""

from __future__ import annotations

import logging
import time

from qdrant_client.models import FusionQuery, Prefetch, Query, SparseVector

from configs.settings import (
    DENSE_VECTOR_NAME,
    HYBRID_ALPHA,
    SPARSE_VECTOR_NAME,
    TOP_K,
    USE_SPARSE,
)
from src.ingestion.embedder import embed_sparse, embed_texts
from src.ingestion.vector_store import get_client

logger = logging.getLogger(__name__)


def retrieve(
    query: str,
    cname: str,
    top_k: int = TOP_K,
) -> tuple[list[dict], float]:
    """
    Retrieve top-k chunks for a query using hybrid search.

    Returns (chunks, latency_ms).
    Each chunk dict contains payload fields + score.
    """
    client = get_client()
    t0 = time.perf_counter()

    # Embed query
    dense_vec = embed_texts([query])[0]

    if USE_SPARSE:
        sparse_vec = embed_sparse([query])[0]

        # Qdrant hybrid: prefetch dense + sparse, then RRF fusion
        results = client.query_points(
            collection_name=cname,
            prefetch=[
                Prefetch(
                    query=dense_vec,
                    using=DENSE_VECTOR_NAME,
                    limit=top_k * 3,
                ),
                Prefetch(
                    query=SparseVector(
                        indices=sparse_vec.indices,
                        values=sparse_vec.values,
                    ),
                    using=SPARSE_VECTOR_NAME,
                    limit=top_k * 3,
                ),
            ],
            query=FusionQuery(fusion="rrf"),   # Reciprocal Rank Fusion
            limit=top_k,
            with_payload=True,
        ).points
    else:
        results = client.query_points(
            collection_name=cname,
            query=dense_vec,
            using=DENSE_VECTOR_NAME,
            limit=top_k,
            with_payload=True,
        ).points

    latency_ms = (time.perf_counter() - t0) * 1000

    chunks = []
    for hit in results:
        payload = hit.payload or {}
        chunks.append({
            "chunk_id": payload.get("chunk_id", ""),
            "doc_id": payload.get("doc_id", ""),
            "title": payload.get("title", ""),
            "text": payload.get("text", ""),
            "char_start": payload.get("char_start", 0),
            "char_end": payload.get("char_end", 0),
            "score": round(hit.score, 4),
        })

    return chunks, latency_ms
