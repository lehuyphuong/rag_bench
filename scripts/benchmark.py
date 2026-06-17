"""
RAG Benchmark v2 — main entry point.

Pipeline per config:
  1. Chunk documents           (chunker.py)
  2. Filter chunks             (filtering/pipeline.py — pass-through if filtering=[])
  3. Embed (dense + BM25)      (embedder.py)
  4. Ingest into Qdrant        (vector_store.py — in-process, no Docker)
  5. Retrieve top-k per query  (retriever.py)
  6. Generate answer via LLM   (generator.py)
  7. Compute metrics           (metrics.py)
  8. Record results to CSV     (results/benchmark_results.csv)

Config name format: "{strategy}_{size}_{overlap}[_{filter1}_{filter2}...]"
  No filtering:   "RecursiveToken_400_200"
  With filtering: "RecursiveToken_400_200_Similarity0.8_NER_Exact"  (future)

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --max-docs 100 --max-questions 50
    python scripts/benchmark.py --config RecursiveToken_400_0
    python scripts/benchmark.py --skip-generation
    python scripts/benchmark.py --keep-collections
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.settings import (
    CHUNKING_CONFIGS,
    EMBED_BATCH_SIZE,
    ORACLE_K,
    RESULTS_DIR,
    TOP_K,
)
from src.evaluation.generator import generate_answer
from src.evaluation.metrics import (
    compute_oracle,
    compute_retrieval_metrics,
    exact_match,
    token_f1,
)
from src.filtering.pipeline import FilteringPipeline
from src.ingestion.chunker import chunk_documents
from src.ingestion.embedder import (
    build_bm25_vocab,
    embed_chunks_batched,
    embed_texts,
)
from src.ingestion.loader import load_squad
from src.ingestion.vector_store import (
    collection_name,
    collection_stats,
    delete_collection,
    ensure_collection,
    upsert_chunks,
)
from src.retrieval.retriever import retrieve
from src.utils.logger import configure_logging

logger = logging.getLogger(__name__)


# ── Config name ───────────────────────────────────────────────────────────────

def make_config_name(strategy: str, chunk_size: int, overlap: int, filtering: list[dict]) -> str:
    """
    Human-readable config identifier used as collection name and CSV key.

    No filtering:    "RecursiveToken_400_200"
    With filtering:  "RecursiveToken_400_200__Similarity0.8__NER_Exact"
    """
    base = f"{strategy}_{chunk_size}_{overlap}"
    if not filtering:
        return base
    filter_tags = "__".join(
        step["method"] + (str(step.get("threshold", "")) if "threshold" in step else "")
        for step in filtering
    )
    return f"{base}__{filter_tags}"


# ── CSV field names ───────────────────────────────────────────────────────────

SUMMARY_FIELDS = [
    # identification
    "config_name", "strategy", "chunk_size", "overlap", "filtering",
    # ingestion
    "chunk_count_before_filter", "chunk_count_after_filter",
    "filter_reduction_pct", "ingest_time_s", "storage_mb",
    # retrieval metrics (averaged, raw tokenization)
    "precision_raw", "recall_raw", "iou_raw",
    # retrieval metrics (averaged, preprocessed tokenization)
    "precision_pre", "recall_pre", "iou_pre",
    # oracle upper bound
    "oracle_recall_raw",
    # generation metrics
    "exact_match", "token_f1",
    # latency
    "avg_retrieval_ms", "avg_generation_s",
    "n_questions",
]

PER_Q_FIELDS = [
    "config_name", "question", "doc_id", "answers",
    "precision_raw", "recall_raw", "iou_raw",
    "precision_pre", "recall_pre", "iou_pre",
    "oracle_recall_raw",
    "generated_answer", "exact_match", "token_f1",
    "retrieval_ms",
]


# ── Ingest ────────────────────────────────────────────────────────────────────

def run_ingest(
    documents: list[dict],
    strategy: str,
    chunk_size: int,
    overlap: int,
    filtering: list[dict],
    config_name: str,
    embed_fn,
) -> tuple[list[dict], list[dict], float, str, dict]:
    """
    Chunk → filter → embed → ingest one configuration.

    Returns (chunks_before_filter, chunks_after_filter, ingest_time_s, cname, stats).
    """
    t0 = time.perf_counter()

    # Step 1: Chunk
    chunks_raw, _ = chunk_documents(
        documents, strategy, chunk_size, overlap, embed_fn=embed_fn
    )
    logger.info("  %d chunks before filtering", len(chunks_raw))

    # Step 2: Filter (pass-through if filtering=[])
    filter_pipeline = FilteringPipeline(steps=filtering)
    chunks = filter_pipeline.run(chunks_raw, embed_fn=embed_fn)
    logger.info("  %d chunks after filtering", len(chunks))

    # Step 3: BM25 vocabulary (from post-filter chunks)
    build_bm25_vocab([c["text"] for c in chunks])

    # Step 4: Embed + ingest
    cname = ensure_collection(strategy, chunk_size, overlap, recreate=True)
    dense_vecs, sparse_vecs = [], []
    for chunk, dv, sv in embed_chunks_batched(chunks, batch_size=EMBED_BATCH_SIZE):
        dense_vecs.append(dv)
        sparse_vecs.append(sv)

    upsert_chunks(cname, chunks, dense_vecs, sparse_vecs)

    ingest_time = time.perf_counter() - t0
    stats = collection_stats(cname)

    logger.info(
        "  Ingest done: %.1fs | %d points | %.2f MB",
        ingest_time, stats["points_count"], stats["disk_mb"],
    )
    return chunks_raw, chunks, ingest_time, cname, stats


# ── Evaluate ──────────────────────────────────────────────────────────────────

def run_eval(
    qa_pairs: list[dict],
    documents: list[dict],
    all_chunks: list[dict],
    cname: str,
    config_name: str,
    skip_generation: bool = False,
) -> tuple[dict, list[dict]]:
    """Evaluate retrieval + generation. Returns (summary_dict, per_q_rows)."""
    doc_lookup = {d["doc_id"]: d for d in documents}
    chunks_by_doc: dict[str, list[dict]] = {}
    for c in all_chunks:
        chunks_by_doc.setdefault(c["doc_id"], []).append(c)

    acc: dict[str, list[float]] = {k: [] for k in [
        "precision_raw", "recall_raw", "iou_raw",
        "precision_pre", "recall_pre", "iou_pre",
        "oracle_recall_raw",
        "exact_match", "token_f1",
        "retrieval_ms", "generation_s",
    ]}
    per_q_rows: list[dict] = []

    for i, qa in enumerate(qa_pairs):
        doc = doc_lookup.get(qa["doc_id"])
        if doc is None:
            continue

        reference_text = doc["text"]
        retrieved, lat_ms = retrieve(qa["question"], cname, top_k=TOP_K)
        acc["retrieval_ms"].append(lat_ms)

        # Retrieval metrics — raw
        m_raw = compute_retrieval_metrics(reference_text, retrieved, mode="raw")
        acc["precision_raw"].append(m_raw["precision"])
        acc["recall_raw"].append(m_raw["recall"])
        acc["iou_raw"].append(m_raw["iou"])

        # Retrieval metrics — preprocessed
        m_pre = compute_retrieval_metrics(reference_text, retrieved, mode="preprocessed")
        acc["precision_pre"].append(m_pre["precision"])
        acc["recall_pre"].append(m_pre["recall"])
        acc["iou_pre"].append(m_pre["iou"])

        # Oracle
        doc_chunks = list(chunks_by_doc.get(qa["doc_id"], []))
        oracle = compute_oracle(reference_text, doc_chunks, k=ORACLE_K, mode="raw")
        acc["oracle_recall_raw"].append(oracle["oracle_recall"])

        # Generation
        gen_answer, em, f1, gen_time = "", 0.0, 0.0, 0.0
        if not skip_generation:
            t_gen = time.perf_counter()
            gen_answer = generate_answer(qa["question"], retrieved)
            gen_time = time.perf_counter() - t_gen
            em = exact_match(gen_answer, qa["answers"])
            f1 = token_f1(gen_answer, qa["answers"])

        acc["exact_match"].append(em)
        acc["token_f1"].append(f1)
        acc["generation_s"].append(gen_time)

        per_q_rows.append({
            "config_name": config_name,
            "question": qa["question"],
            "doc_id": qa["doc_id"],
            "answers": json.dumps(qa["answers"]),
            "precision_raw": m_raw["precision"],
            "recall_raw": m_raw["recall"],
            "iou_raw": m_raw["iou"],
            "precision_pre": m_pre["precision"],
            "recall_pre": m_pre["recall"],
            "iou_pre": m_pre["iou"],
            "oracle_recall_raw": oracle["oracle_recall"],
            "generated_answer": gen_answer,
            "exact_match": em,
            "token_f1": f1,
            "retrieval_ms": round(lat_ms, 2),
        })

        if (i + 1) % 20 == 0:
            logger.info(
                "  Evaluated %d/%d | recall_raw=%.3f | recall_pre=%.3f",
                i + 1, len(qa_pairs),
                sum(acc["recall_raw"]) / len(acc["recall_raw"]),
                sum(acc["recall_pre"]) / len(acc["recall_pre"]),
            )

    def avg(lst): return round(sum(lst) / len(lst), 4) if lst else 0.0

    summary = {k: avg(acc[k]) for k in [
        "precision_raw", "recall_raw", "iou_raw",
        "precision_pre", "recall_pre", "iou_pre",
        "oracle_recall_raw",
        "exact_match", "token_f1",
    ]}
    summary["avg_retrieval_ms"] = avg(acc["retrieval_ms"])
    summary["avg_generation_s"] = avg(acc["generation_s"])
    summary["n_questions"] = len(per_q_rows)
    return summary, per_q_rows


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    configure_logging()

    parser = argparse.ArgumentParser(description="RAG chunking benchmark on SQuAD 1.1")
    parser.add_argument("--max-docs",       type=int, default=None)
    parser.add_argument("--max-questions",  type=int, default=None)
    parser.add_argument("--config",         type=str, default=None,
                        help="Run only one config by name, e.g. 'RecursiveToken_400_0'")
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--keep-collections", action="store_true")
    args = parser.parse_args()

    # Override caps from CLI
    import configs.settings as S
    if args.max_docs:      S.MAX_DOCUMENTS = args.max_docs
    if args.max_questions: S.MAX_EVAL_QUESTIONS = args.max_questions

    documents, qa_pairs = load_squad()
    logger.info("Documents: %d | QA pairs: %d", len(documents), len(qa_pairs))

    embed_fn = lambda texts: embed_texts(texts)

    # Setup summary CSV
    summary_path = RESULTS_DIR / "benchmark_results.csv"
    is_new = not summary_path.exists()
    summary_f = open(summary_path, "a", newline="", encoding="utf-8")
    summary_writer = csv.DictWriter(summary_f, fieldnames=SUMMARY_FIELDS)
    if is_new:
        summary_writer.writeheader()

    # Filter configs if --config specified
    configs = CHUNKING_CONFIGS
    if args.config:
        configs = [
            c for c in configs
            if make_config_name(c["strategy"], c["chunk_size"],
                                c["overlap"], c["filtering"]) == args.config
        ]
        if not configs:
            logger.error("Config '%s' not found.", args.config)
            sys.exit(1)

    for cfg in configs:
        strategy  = cfg["strategy"]
        chunk_size = cfg["chunk_size"]
        overlap   = cfg["overlap"]
        filtering = cfg.get("filtering", [])
        cname_str = make_config_name(strategy, chunk_size, overlap, filtering)

        logger.info("=" * 60)
        logger.info("Config: %s", cname_str)
        logger.info("=" * 60)

        # Ingest
        chunks_raw, chunks, ingest_time, cname, stats = run_ingest(
            documents, strategy, chunk_size, overlap, filtering, cname_str, embed_fn
        )

        n_before = len(chunks_raw)
        n_after  = len(chunks)
        reduction_pct = round(100 * (n_before - n_after) / n_before, 2) if n_before else 0.0

        # Evaluate
        eval_summary, per_q = run_eval(
            qa_pairs, documents, chunks, cname, cname_str,
            skip_generation=args.skip_generation,
        )

        # Per-question CSV
        per_q_path = RESULTS_DIR / f"per_question_{cname_str}.csv"
        with open(per_q_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=PER_Q_FIELDS)
            writer.writeheader()
            writer.writerows(per_q)

        # Summary row
        row = {
            "config_name": cname_str,
            "strategy": strategy,
            "chunk_size": chunk_size,
            "overlap": overlap,
            "filtering": json.dumps(filtering),
            "chunk_count_before_filter": n_before,
            "chunk_count_after_filter": n_after,
            "filter_reduction_pct": reduction_pct,
            "ingest_time_s": round(ingest_time, 2),
            "storage_mb": stats["disk_mb"],
            **eval_summary,
        }
        summary_writer.writerow(row)
        summary_f.flush()

        logger.info(
            "  DONE | chunks=%d→%d (-%.1f%%) | %.1fs | %.2fMB | "
            "recall_raw=%.3f | recall_pre=%.3f | EM=%.3f | F1=%.3f",
            n_before, n_after, reduction_pct, ingest_time, stats["disk_mb"],
            eval_summary["recall_raw"], eval_summary["recall_pre"],
            eval_summary["exact_match"], eval_summary["token_f1"],
        )

        if not args.keep_collections:
            delete_collection(strategy, chunk_size, overlap)

    summary_f.close()
    logger.info("Results written to: %s", summary_path)


if __name__ == "__main__":
    main()
