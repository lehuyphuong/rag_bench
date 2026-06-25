"""
Central configuration for rag-bench v4.

Changes from v3:
  - Chunking strategies replaced with 5 new strategies:
      AdaptiveEntropy, AdaptiveSentenceLen,
      HierarchicalParentChild, Contextual, TopicBased
  - Evaluation metrics reduced to 4 paper-core metrics:
      Precision, Recall, IoU, Index Size
  - Oracle and generation metrics removed
"""

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR    = Path(__file__).resolve().parent.parent
DATA_DIR    = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"
QDRANT_PATH = DATA_DIR / "qdrant_storage"

DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
QDRANT_PATH.mkdir(exist_ok=True)

# ── Dataset ──────────────────────────────────────────────────────────────────
DATASET_NAME       = "rajpurkar/squad"
DATASET_SPLIT      = "validation"
MAX_DOCUMENTS:       int | None = 500
MAX_EVAL_QUESTIONS:  int | None = 200

# ── Embedding — all-MiniLM-L6-v2 ─────────────────────────────────────────────
EMBED_MODEL      = "sentence-transformers/all-MiniLM-L6-v2"
TEXT_EMBED_DIM   = 384
EMBED_BATCH_SIZE = 128

# ── Qdrant (embedded, no Docker) ─────────────────────────────────────────────
COLLECTION_PREFIX = "squad_bench_v5"

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K = 5

# ── LLM (Ollama — optional, only for generation) ─────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL       = "mistral"
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS  = 256

SYSTEM_PROMPT = """\
You are a precise question-answering assistant. Answer the question using \
ONLY the provided context passages. Be concise — answer in 1-3 sentences. \
If the answer is not in the context, respond with exactly: "I don't know."\
"""

# ── Chunking + Filtering configurations ──────────────────────────────────────
#
# 9 chunking strategies × 5 filter methods = 90 configs
#   (4 classic v3: FixedToken, RecursiveToken, ClusterSemantic, Overlapping)
#   (5 new v4:     AdaptiveEntropy, AdaptiveSentenceLen, HierarchicalParentChild,
#                  Contextual, TopicBased)
#
# Strategy params:
#   chunk_size : target chunk size in characters (base unit)
#   overlap    : character overlap between chunks (0 for strategies that
#                don't use sliding windows)
#   extra      : strategy-specific overrides (optional)
#
# Filter methods (unchanged from v3):
#   NoFilter, ExactNorm, MinHashLSH(0.7), Similarity(0.8), NERExact

def _make_configs() -> list[dict]:
    chunker_configs = [
        # ── FixedToken (paper Section 3.3, baseline) ─────────────────────────
        # Paper tests: (200,0), (400,0), (400,200), (800,400)
        # Dùng 2 size phổ biến nhất để so sánh
        {"strategy": "FixedToken",      "chunk_size": 200, "overlap": 0},
        {"strategy": "FixedToken",      "chunk_size": 400, "overlap": 0},

        # ── RecursiveToken (paper Section 3.3) ───────────────────────────────
        {"strategy": "RecursiveToken",  "chunk_size": 200, "overlap": 0},
        {"strategy": "RecursiveToken",  "chunk_size": 400, "overlap": 0},

        # ── ClusterSemantic (paper Section 3.3) ──────────────────────────────
        {"strategy": "ClusterSemantic", "chunk_size": 200, "overlap": 0,
         "extra": {"threshold_percentile": 95.0}},
        {"strategy": "ClusterSemantic", "chunk_size": 400, "overlap": 0,
         "extra": {"threshold_percentile": 95.0}},

        # ── Overlapping (paper Section 3.3) — nguồn chính gây redundancy ─────
        # Paper tests: FixedToken(400,200) và FixedToken(800,400)
        {"strategy": "Overlapping",     "chunk_size": 400, "overlap": 200},
        {"strategy": "Overlapping",     "chunk_size": 800, "overlap": 400},

        # ── AdaptiveEntropy ─────────────────────────────────────────────────
        # base_size drives the lerp range; min/max derived as base//3 and base*2
        {"strategy": "AdaptiveEntropy",       "chunk_size": 300, "overlap": 0},
        {"strategy": "AdaptiveEntropy",       "chunk_size": 500, "overlap": 0},

        # ── AdaptiveSentenceLen ──────────────────────────────────────────────
        # target_sentences controls the expected group size
        {"strategy": "AdaptiveSentenceLen",   "chunk_size": 4,   "overlap": 0,
         "extra": {"target_sentences": 4, "min_sentences": 2, "max_sentences": 8}},
        {"strategy": "AdaptiveSentenceLen",   "chunk_size": 6,   "overlap": 0,
         "extra": {"target_sentences": 6, "min_sentences": 3, "max_sentences": 12}},

        # ── HierarchicalParentChild ──────────────────────────────────────────
        # chunk_size = child size; parent_size = child_size * 2 (set in extra)
        {"strategy": "HierarchicalParentChild", "chunk_size": 200, "overlap": 0,
         "extra": {"parent_size": 600}},
        {"strategy": "HierarchicalParentChild", "chunk_size": 400, "overlap": 0,
         "extra": {"parent_size": 800}},

        # ── Contextual ───────────────────────────────────────────────────────
        {"strategy": "Contextual",            "chunk_size": 300, "overlap": 0},
        {"strategy": "Contextual",            "chunk_size": 500, "overlap": 0},

        # ── TopicBased ───────────────────────────────────────────────────────
        {"strategy": "TopicBased",            "chunk_size": 200, "overlap": 0,
         "extra": {"n_topics": 4, "min_chunk_size": 100}},
        {"strategy": "TopicBased",            "chunk_size": 400, "overlap": 0,
         "extra": {"n_topics": 6, "min_chunk_size": 200}},
    ]

    filter_configs = [
        [{"method": "NoFilter"}],
        [{"method": "ExactNorm"}],
        [{"method": "MinHashLSH",  "threshold": 0.7}],
        [{"method": "Similarity",  "threshold": 0.8}],
        [{"method": "NERExact"}],
    ]

    configs = []
    for chunker in chunker_configs:
        for filtering in filter_configs:
            cfg = {
                "strategy":   chunker["strategy"],
                "chunk_size": chunker["chunk_size"],
                "overlap":    chunker["overlap"],
                "filtering":  filtering,
            }
            if "extra" in chunker:
                cfg["extra"] = chunker["extra"]
            configs.append(cfg)

    return configs


CHUNKING_CONFIGS = _make_configs()
