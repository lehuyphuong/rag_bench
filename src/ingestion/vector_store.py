"""
Qdrant in-process (embedded) vector store.

Uses QdrantClient(path=...) — no Docker, no port, no root.
Each benchmark configuration gets its own collection with:
  - dense named vector "dense" (768-dim, cosine)
  - sparse named vector "bm25" (variable dim, dot product)

Collection name format: "{COLLECTION_PREFIX}_{strategy}_{size}_{overlap}"
e.g. "squad_bench_FixedToken_400_200"
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    PointStruct,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
    VectorsConfig,
)

from configs.settings import (
    COLLECTION_PREFIX,
    DENSE_VECTOR_NAME,
    QDRANT_PATH,
    SPARSE_VECTOR_NAME,
    TEXT_EMBED_DIM,
    USE_SPARSE,
)

logger = logging.getLogger(__name__)

# Module-level singleton — one client per process reused across collections
_client: QdrantClient | None = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        logger.info("Opening Qdrant embedded store at: %s", QDRANT_PATH)
        _client = QdrantClient(path=str(QDRANT_PATH))
    return _client


def collection_name(strategy: str, chunk_size: int, overlap: int) -> str:
    return f"{COLLECTION_PREFIX}_{strategy}_{chunk_size}_{overlap}"


def ensure_collection(
    strategy: str, chunk_size: int, overlap: int, recreate: bool = True
) -> str:
    """Create (or recreate) a Qdrant collection for one benchmark config."""
    client = get_client()
    cname = collection_name(strategy, chunk_size, overlap)

    existing = [c.name for c in client.get_collections().collections]
    if cname in existing:
        if recreate:
            logger.info("Dropping collection '%s'", cname)
            client.delete_collection(cname)
        else:
            logger.info("Collection '%s' already exists — skipping.", cname)
            return cname

    vectors_config: dict = {
        DENSE_VECTOR_NAME: VectorParams(
            size=TEXT_EMBED_DIM,
            distance=Distance.COSINE,
        ),
    }

    sparse_vectors_config: dict | None = None
    if USE_SPARSE:
        sparse_vectors_config = {
            SPARSE_VECTOR_NAME: SparseVectorParams(
                index=SparseIndexParams(on_disk=False),
            ),
        }

    client.create_collection(
        collection_name=cname,
        vectors_config=vectors_config,
        sparse_vectors_config=sparse_vectors_config,
        hnsw_config=HnswConfigDiff(m=16, ef_construct=100),
    )
    logger.info(
        "Created collection '%s' (dense=%d-dim%s)",
        cname, TEXT_EMBED_DIM,
        " + sparse BM25" if USE_SPARSE else "",
    )
    return cname


def upsert_chunks(
    cname: str,
    chunks: list[dict],
    dense_vecs: list[list[float]],
    sparse_vecs: list | None = None,
    batch_size: int = 256,
) -> None:
    """Upsert chunks into the collection in batches."""
    client = get_client()
    for i in range(0, len(chunks), batch_size):
        batch_c = chunks[i : i + batch_size]
        batch_d = dense_vecs[i : i + batch_size]
        batch_s = sparse_vecs[i : i + batch_size] if sparse_vecs else None

        points = []
        for j, (chunk, dvec) in enumerate(zip(batch_c, batch_d)):
            vectors: dict = {DENSE_VECTOR_NAME: dvec}
            if batch_s is not None and USE_SPARSE:
                vectors[SPARSE_VECTOR_NAME] = batch_s[j]
            points.append(
                PointStruct(
                    id=abs(hash(chunk["chunk_id"])) % (2**53),  # stable int ID
                    vector=vectors,
                    payload={
                        "chunk_id": chunk["chunk_id"],
                        "doc_id": chunk["doc_id"],
                        "title": chunk["title"],
                        "text": chunk["text"],
                        "char_start": chunk["char_start"],
                        "char_end": chunk["char_end"],
                    },
                )
            )
        client.upsert(collection_name=cname, points=points, wait=True)

    logger.info("Upserted %d points into '%s'", len(chunks), cname)


def collection_stats(cname: str) -> dict:
    """Return point count and estimated disk size for a collection."""
    client = get_client()
    info = client.get_collection(cname)

    # Embedded mode: estimate disk size from the collection directory
    col_path = QDRANT_PATH / "collection" / cname
    disk_bytes = 0
    if col_path.exists():
        disk_bytes = sum(
            f.stat().st_size
            for f in col_path.rglob("*")
            if f.is_file()
        )

    return {
        "collection": cname,
        "points_count": info.points_count,
        "disk_bytes": disk_bytes,
        "disk_mb": round(disk_bytes / 1024 / 1024, 2),
    }


def delete_collection(strategy: str, chunk_size: int, overlap: int) -> None:
    """Delete a collection to free disk space after benchmarking."""
    client = get_client()
    cname = collection_name(strategy, chunk_size, overlap)
    existing = [c.name for c in client.get_collections().collections]
    if cname in existing:
        client.delete_collection(cname)
        logger.info("Deleted collection '%s'", cname)
