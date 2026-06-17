"""
Central configuration for rag-bench v3.

Reproduces Berdyugina et al. (arXiv:2604.24334) chunk filtering experiments.

Changes from v2 baseline:
  - Embedding  : all-MiniLM-L6-v2 via sentence-transformers (384-dim, no Ollama)
  - Retrieval  : dense-only cosine similarity (no BM25 hybrid)
  - Filtering  : ExactNorm, MinHashLSH(0.7), Similarity(0.8), NERExact
  - Chunking   : 3 strategies × paper configs (FixedToken, RecursiveToken, ClusterSemantic)
  - Dataset    : SQuAD 1.1 validation split

Pipeline:
    load → chunk → [filter] → embed → index → retrieve → evaluate
"""

from pathlib import Path

# ── Paths ───────────────────────────────────────────────────────────────────
ROOT_DIR    = Path(__file__).resolve().parent.parent
DATA_DIR    = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"
QDRANT_PATH = DATA_DIR / "qdrant_storage"

DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
QDRANT_PATH.mkdir(exist_ok=True)

# ── Dataset ─────────────────────────────────────────────────────────────────
DATASET_NAME        = "rajpurkar/squad"
DATASET_SPLIT       = "validation"
MAX_DOCUMENTS: int | None = 500       # None = full SQuAD val (~2,067 passages)
MAX_EVAL_QUESTIONS: int | None = 200  # questions used in evaluation

# ── Embedding — all-MiniLM-L6-v2 (sentence-transformers) ───────────────────
# Paper Section 3.4: "all-MiniLM-L6-v2 … chosen as the reference model
# because it offers a good balance between speed, reproducibility, and
# computational cost."
EMBED_MODEL     = "sentence-transformers/all-MiniLM-L6-v2"
TEXT_EMBED_DIM  = 384
EMBED_BATCH_SIZE = 128

# ── Qdrant (in-process / embedded mode — no Docker) ─────────────────────────
COLLECTION_PREFIX = "squad_bench_v3"

# ── Retrieval — dense only (paper Section 3.4) ──────────────────────────────
TOP_K = 5

# ── LLM (Ollama — generation only, not used for embedding) ──────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL       = "mistral"
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS  = 256

SYSTEM_PROMPT = """\
You are a precise question-answering assistant. Answer the question using \
ONLY the provided context passages. Be concise — answer in 1-3 sentences. \
If the answer is not in the context, respond with exactly: "I don't know."\
"""

# ── Evaluation ───────────────────────────────────────────────────────────────
ORACLE_K       = 5
EVAL_LANGUAGE  = "english"

# ── Chunking + Filtering configurations ─────────────────────────────────────
#
# Paper Section 3.3: FixedToken, RecursiveToken, ClusterSemantic
# Paper Section 3.5 filtering strategies evaluated:
#   - No filtering      (baseline)
#   - ExactNorm         (lexical exact dedup after normalization)
#   - MinHashLSH        (threshold 0.7 — paper tests 0.6/0.7/0.8, 0.7 balanced)
#   - Similarity        (cosine threshold 0.8 — paper "most stable" threshold)
#   - NERExact          (drop chunks with identical named-entity set)
#
# Each config:
#   strategy  : "FixedToken" | "RecursiveToken" | "ClusterSemantic"
#   chunk_size: int (characters)
#   overlap   : int (characters)
#   filtering : list[dict]  — pipeline steps applied after chunking
#
# Filtering step format:
#   {"method": "NoFilter"}
#   {"method": "ExactNorm"}
#   {"method": "MinHashLSH",  "threshold": 0.7}
#   {"method": "Similarity",  "threshold": 0.8}
#   {"method": "NERExact"}

def _make_configs():
    """
    Generate all (chunker × filter) combinations following the paper.
    Each chunking config is paired with each filtering strategy.
    """
    chunker_configs = [
        # ── FixedToken ────────────────────────────────────────────────────
        {"strategy": "FixedToken",      "chunk_size": 200, "overlap": 0},
        {"strategy": "FixedToken",      "chunk_size": 400, "overlap": 0},
        {"strategy": "FixedToken",      "chunk_size": 400, "overlap": 200},
        {"strategy": "FixedToken",      "chunk_size": 800, "overlap": 400},
        # ── RecursiveToken ────────────────────────────────────────────────
        {"strategy": "RecursiveToken",  "chunk_size": 200, "overlap": 0},
        {"strategy": "RecursiveToken",  "chunk_size": 400, "overlap": 0},
        {"strategy": "RecursiveToken",  "chunk_size": 400, "overlap": 200},
        {"strategy": "RecursiveToken",  "chunk_size": 800, "overlap": 400},
        # ── ClusterSemantic ───────────────────────────────────────────────
        {"strategy": "ClusterSemantic", "chunk_size": 200, "overlap": 0},
        {"strategy": "ClusterSemantic", "chunk_size": 400, "overlap": 0},
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
            configs.append({
                "strategy":   chunker["strategy"],
                "chunk_size": chunker["chunk_size"],
                "overlap":    chunker["overlap"],
                "filtering":  filtering,
            })
    return configs


CHUNKING_CONFIGS = _make_configs()
