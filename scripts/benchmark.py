"""
RAG Benchmark v4 — main entry point.

Chunk strategies : FixedToken, RecursiveToken, ClusterSemantic, Overlapping
                   (paper v3 / Berdyugina et al. Section 3.3)
                   AdaptiveEntropy, AdaptiveSentenceLen,
                   HierarchicalParentChild, Contextual, TopicBased
                   (paper v4 new strategies)
Filter methods   : NoFilter, ExactNorm, MinHashLSH(0.7), Similarity(0.8), NERExact
                   (unchanged — same 5 methods as the paper, no CACD)
Eval metrics     : Precision, Recall, IoU, Index Size (chunk count + storage MB)
                   (unchanged — same 4 metrics as the paper)

Total: 9 strategies x 2 sizes x 5 filters = 90 configs

Usage:
    # Debug (small)
    python scripts/benchmark.py --max-docs 20 --max-questions 30

    # Single strategy
    python scripts/benchmark.py --strategy FixedToken --max-docs 50 --max-questions 50

    # Single filter
    python scripts/benchmark.py --filter NERExact --max-docs 50 --max-questions 50

    # Only the 5 new v4 strategies
    python scripts/benchmark.py --v4-only

    # Only the 4 classic v3 strategies
    python scripts/benchmark.py --classic-only

    # Single config
    python scripts/benchmark.py --config "FixedToken_400_0__Similarity0.8"

    # Full benchmark (all 90 configs)
    python scripts/benchmark.py

    # Background
    nohup python scripts/benchmark.py > results/bench.log 2>&1 &
    tail -f results/bench.log
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
    RESULTS_DIR,
    TOP_K,
)
from src.evaluation.metrics import compute_retrieval_metrics
from src.filtering.pipeline import FilteringPipeline
from src.ingestion.chunker import chunk_documents
from src.ingestion.embedder import embed_chunks_batched, embed_texts
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

def make_config_name(
    strategy: str, chunk_size: int, overlap: int, filter_tag: str
) -> str:
    """
    Format: "{strategy}_{size}_{overlap}__{filter_tag}"
    Examples:
      "AdaptiveEntropy_300_0__NoFilter"
      "TopicBased_400_0__NERExact"
      "HierarchicalParentChild_200_0__Similarity0.8"
    """
    return f"{strategy}_{chunk_size}_{overlap}__{filter_tag}"


# ── CSV fields ────────────────────────────────────────────────────────────────
# Only 4 paper-core metrics + ingestion metadata

SUMMARY_FIELDS = [
    # identification
    "config_name", "strategy", "chunk_size", "overlap", "filter_method",
    # index size (metric 4)
    "chunk_count_before_filter",
    "chunk_count_after_filter",
    "filter_reduction_pct",
    "ingest_time_s",
    "storage_mb",
    "storage_du",
    # retrieval metrics — raw mode (metrics 1-3)
    "precision_raw", "recall_raw", "iou_raw",
    # retrieval metrics — preprocessed mode
    "precision_pre", "recall_pre", "iou_pre",
    # meta
    "avg_retrieval_ms",
    "n_questions",
]

PER_Q_FIELDS = [
    "config_name", "question", "doc_id",
    "precision_raw", "recall_raw", "iou_raw",
    "precision_pre", "recall_pre", "iou_pre",
    "retrieval_ms",
]


# ── Ingest ────────────────────────────────────────────────────────────────────

def run_ingest(
    documents:   list[dict],
    strategy:    str,
    chunk_size:  int,
    overlap:     int,
    filtering:   list[dict],
    filter_tag:  str,
    embed_fn,
    extra:       dict | None = None,
) -> tuple[list[dict], list[dict], float, str, dict]:
    """
    Chunk → filter → embed → ingest one configuration.
    Returns (chunks_before, chunks_after, ingest_time_s, cname, stats).
    """
    t0 = time.perf_counter()

    # Step 1: Chunk
    chunks_raw, _ = chunk_documents(
        documents, strategy, chunk_size, overlap,
        embed_fn=embed_fn, extra=extra,
    )
    logger.info("  %d chunks before filtering", len(chunks_raw))

    # Step 2: Filter
    pipeline = FilteringPipeline(steps=filtering)
    chunks   = pipeline.run(chunks_raw, embed_fn=embed_fn)
    logger.info("  %d chunks after filtering", len(chunks))

    # Step 3: Embed + ingest
    cname = collection_name(strategy, chunk_size, overlap, filter_tag)
    ensure_collection(cname, recreate=True)

    dense_vecs: list[list[float]] = []
    for chunk, dv in embed_chunks_batched(chunks, batch_size=EMBED_BATCH_SIZE):
        dense_vecs.append(dv)

    upsert_chunks(cname, chunks, dense_vecs)

    ingest_time = time.perf_counter() - t0
    stats       = collection_stats(cname)

    logger.info(
        "  Ingest done: %.1fs | %d points | %.2f MB (%s)",
        ingest_time, stats["points_count"],
        stats["disk_mb"], stats["disk_size_du"],
    )
    return chunks_raw, chunks, ingest_time, cname, stats


# ── Evaluate ──────────────────────────────────────────────────────────────────

def run_eval(
    qa_pairs:   list[dict],
    documents:  list[dict],
    cname:      str,
    config_name: str,
) -> tuple[dict, list[dict]]:
    """
    Evaluate retrieval using Precision, Recall, IoU only.
    Returns (summary_dict, per_q_rows).
    """
    doc_lookup: dict[str, dict] = {d["doc_id"]: d for d in documents}

    acc: dict[str, list[float]] = {k: [] for k in [
        "precision_raw", "recall_raw", "iou_raw",
        "precision_pre", "recall_pre", "iou_pre",
        "retrieval_ms",
    ]}
    per_q_rows: list[dict] = []

    for i, qa in enumerate(qa_pairs):
        doc = doc_lookup.get(qa["doc_id"])
        if doc is None:
            continue

        reference_text    = doc["text"]
        retrieved, lat_ms = retrieve(qa["question"], cname, top_k=TOP_K)
        acc["retrieval_ms"].append(lat_ms)

        # Metric 1-3: Precision / Recall / IoU — raw
        m_raw = compute_retrieval_metrics(reference_text, retrieved, mode="raw")
        acc["precision_raw"].append(m_raw["precision"])
        acc["recall_raw"].append(m_raw["recall"])
        acc["iou_raw"].append(m_raw["iou"])

        # Metric 1-3: Precision / Recall / IoU — preprocessed
        m_pre = compute_retrieval_metrics(reference_text, retrieved, mode="preprocessed")
        acc["precision_pre"].append(m_pre["precision"])
        acc["recall_pre"].append(m_pre["recall"])
        acc["iou_pre"].append(m_pre["iou"])

        per_q_rows.append({
            "config_name":   config_name,
            "question":      qa["question"],
            "doc_id":        qa["doc_id"],
            "precision_raw": m_raw["precision"],
            "recall_raw":    m_raw["recall"],
            "iou_raw":       m_raw["iou"],
            "precision_pre": m_pre["precision"],
            "recall_pre":    m_pre["recall"],
            "iou_pre":       m_pre["iou"],
            "retrieval_ms":  round(lat_ms, 2),
        })

        if (i + 1) % 20 == 0:
            logger.info(
                "  Evaluated %d/%d | recall_raw=%.3f | iou_raw=%.3f",
                i + 1, len(qa_pairs),
                sum(acc["recall_raw"]) / len(acc["recall_raw"]),
                sum(acc["iou_raw"])    / len(acc["iou_raw"]),
            )

    def avg(lst): return round(sum(lst) / len(lst), 4) if lst else 0.0

    summary = {k: avg(acc[k]) for k in [
        "precision_raw", "recall_raw", "iou_raw",
        "precision_pre", "recall_pre", "iou_pre",
    ]}
    summary["avg_retrieval_ms"] = avg(acc["retrieval_ms"])
    summary["n_questions"]      = len(per_q_rows)
    return summary, per_q_rows


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    configure_logging()

    parser = argparse.ArgumentParser(
        description="RAG chunk filtering benchmark v4 — new chunking strategies"
    )
    parser.add_argument("--max-docs",      type=int, default=None)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument(
        "--strategy", type=str, default=None,
        choices=[
            # Classic (paper v3)
            "FixedToken", "RecursiveToken", "ClusterSemantic", "Overlapping",
            # New (paper v4)
            "AdaptiveEntropy", "AdaptiveSentenceLen",
            "HierarchicalParentChild", "Contextual", "TopicBased",
        ],
    )
    parser.add_argument(
        "--filter", type=str, default=None,
        choices=["NoFilter", "ExactNorm", "MinHashLSH", "Similarity", "NERExact"],
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Run a single config by exact name, e.g. 'TopicBased_400_0__NERExact'",
    )
    parser.add_argument(
        "--keep-collections", action="store_true",
        help="Do not delete Qdrant collections after each config",
    )
    parser.add_argument(
        "--v4-only", action="store_true",
        help="Run only 5 new v4 strategies (50 configs)",
    )
    parser.add_argument(
        "--classic-only", action="store_true",
        help="Run only 4 classic v3 strategies (40 configs)",
    )
    args = parser.parse_args()

    import configs.settings as S
    if args.max_docs:      S.MAX_DOCUMENTS       = args.max_docs
    if args.max_questions: S.MAX_EVAL_QUESTIONS  = args.max_questions

    documents, qa_pairs = load_squad()
    logger.info("Documents: %d | QA pairs: %d", len(documents), len(qa_pairs))

    embed_fn = lambda texts: embed_texts(texts)

    def _filter_tag(filtering: list[dict]) -> str:
        return FilteringPipeline(steps=filtering).tag

    # ── Filter configs from CLI ───────────────────────────────────────────
    V4_STRATEGIES      = {"AdaptiveEntropy","AdaptiveSentenceLen",
                            "HierarchicalParentChild","Contextual","TopicBased"}
    CLASSIC_STRATEGIES = {"FixedToken","RecursiveToken",
                          "ClusterSemantic","Overlapping"}

    configs = list(CHUNKING_CONFIGS)

    if getattr(args, "v4_only", False):
        configs = [c for c in configs if c["strategy"] in V4_STRATEGIES]
    if getattr(args, "classic_only", False):
        configs = [c for c in configs if c["strategy"] in CLASSIC_STRATEGIES]

    if args.strategy:
        configs = [c for c in configs if c["strategy"] == args.strategy]

    if args.filter:
        configs = [
            c for c in configs
            if any(step["method"] == args.filter for step in c["filtering"])
        ]

    if args.config:
        def _name(c):
            ft = _filter_tag(c["filtering"])
            return make_config_name(
                c["strategy"], c["chunk_size"], c["overlap"], ft
            )
        configs = [c for c in configs if _name(c) == args.config]
        if not configs:
            logger.error("Config '%s' not found.", args.config)
            sys.exit(1)

    logger.info("Running %d configs.", len(configs))

    # ── Summary CSV ───────────────────────────────────────────────────────
    summary_path = RESULTS_DIR / "benchmark_results.csv"
    is_new       = not summary_path.exists()
    summary_f    = open(summary_path, "a", newline="", encoding="utf-8")
    writer       = csv.DictWriter(summary_f, fieldnames=SUMMARY_FIELDS)
    if is_new:
        writer.writeheader()

    # ── Run ───────────────────────────────────────────────────────────────
    for cfg in configs:
        strategy   = cfg["strategy"]
        chunk_size = cfg["chunk_size"]
        overlap    = cfg["overlap"]
        filtering  = cfg.get("filtering", [])
        extra      = cfg.get("extra", None)
        ft         = _filter_tag(filtering)
        cname_str  = make_config_name(strategy, chunk_size, overlap, ft)

        logger.info("=" * 65)
        logger.info("Config: %s", cname_str)
        logger.info("=" * 65)

        chunks_raw, chunks, ingest_time, cname, stats = run_ingest(
            documents, strategy, chunk_size, overlap,
            filtering, ft, embed_fn, extra=extra,
        )

        n_before      = len(chunks_raw)
        n_after       = len(chunks)
        reduction_pct = round(100 * (n_before - n_after) / n_before, 2) if n_before else 0.0

        eval_summary, per_q = run_eval(
            qa_pairs, documents, cname, cname_str,
        )

        # Per-question CSV
        per_q_path = RESULTS_DIR / f"per_question_{cname_str}.csv"
        with open(per_q_path, "w", newline="", encoding="utf-8") as f:
            pq_writer = csv.DictWriter(f, fieldnames=PER_Q_FIELDS)
            pq_writer.writeheader()
            pq_writer.writerows(per_q)

        # Summary row — 4 core metrics
        row = {
            "config_name":               cname_str,
            "strategy":                  strategy,
            "chunk_size":                chunk_size,
            "overlap":                   overlap,
            "filter_method":             ft,
            # Metric 4 — Index Size
            "chunk_count_before_filter": n_before,
            "chunk_count_after_filter":  n_after,
            "filter_reduction_pct":      reduction_pct,
            "ingest_time_s":             round(ingest_time, 2),
            "storage_mb":                stats["disk_mb"],
            "storage_du":                stats["disk_size_du"],
            # Metrics 1-3 — Precision / Recall / IoU
            **eval_summary,
        }
        writer.writerow(row)
        summary_f.flush()

        logger.info(
            "  DONE | chunks=%d→%d (-%.1f%%) | %.1fs | %.2f MB (%s) | "
            "P=%.3f R=%.3f IoU=%.3f",
            n_before, n_after, reduction_pct,
            ingest_time, stats["disk_mb"], stats["disk_size_du"],
            eval_summary["precision_raw"],
            eval_summary["recall_raw"],
            eval_summary["iou_raw"],
        )

        if not args.keep_collections:
            delete_collection(cname)

    summary_f.close()
    logger.info("Results written to: %s", summary_path)


if __name__ == "__main__":
    main()
