"""
Central configuration for rag-bench v2.

Benchmarks chunking strategies (FixedToken, RecursiveToken, ClusterSemantic)
on SQuAD 1.1 with hybrid retrieval (dense nomic + sparse BM25) in Qdrant
in-process (embedded) mode — no Docker, no root required (suitable for
vast.ai / GPU instances).

Pipeline flow:
    load → chunk → [filter] → embed → index → retrieve → evaluate

The [filter] step is a placeholder for future work:
    - Exact/near-duplicate filtering (Berdyugina et al. strategies)
    - Semantic similarity filtering (cosine threshold)
    - NER-based filtering (NER_Exact, NER_Half)
    - Merge strategies (partial overlap union, NLI containment)
    - Adaptive chunking (future research)
    - CAG-style KV-cache dedup (future research idea:
        "if chunk already exists in cache → skip insert")

Currently all filtering configs are set to [] (no filtering) so the
benchmark measures raw chunking performance as the baseline.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"
QDRANT_PATH = DATA_DIR / "qdrant_storage"

DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
QDRANT_PATH.mkdir(exist_ok=True)

# ── Dataset ────────────────────────────────────────────────────────────────
DATASET_NAME = "rajpurkar/squad"
DATASET_SPLIT = "validation"
MAX_DOCUMENTS: int | None = 500       # None = full SQuAD val (~2,067 passages)
MAX_EVAL_QUESTIONS: int | None = 200  # questions used in evaluation

# ── Chunking + filtering configurations ───────────────────────────────────
# Each config is a dict with:
#   strategy  : "FixedToken" | "RecursiveToken" | "ClusterSemantic"
#   chunk_size: int (characters)
#   overlap   : int (characters)
#   filtering : list[dict]  ← pipeline of filter steps to apply after chunking
#                              Empty list = no filtering (current baseline)
#
# Future filtering step format (not yet implemented):
#   {"method": "ExactNorm"}
#   {"method": "MinHashLSH",   "threshold": 0.7}
#   {"method": "Similarity",   "threshold": 0.8}
#   {"method": "NER_Exact"}
#   {"method": "NER_Half"}
#   {"method": "Merge_Union",  "threshold": 0.85}   # partial overlap merge
#
# Following Berdyugina et al. (arXiv:2604.24334) Table configs.

CHUNKING_CONFIGS = [
    # ── FixedToken ─────────────────────────────────────────────────────────
    {"strategy": "FixedToken",      "chunk_size": 200, "overlap": 0,   "filtering": []},
    {"strategy": "FixedToken",      "chunk_size": 400, "overlap": 0,   "filtering": []},
    {"strategy": "FixedToken",      "chunk_size": 400, "overlap": 200, "filtering": []},
    {"strategy": "FixedToken",      "chunk_size": 800, "overlap": 400, "filtering": []},
    # ── RecursiveToken ──────────────────────────────────────────────────────
    {"strategy": "RecursiveToken",  "chunk_size": 200, "overlap": 0,   "filtering": []},
    {"strategy": "RecursiveToken",  "chunk_size": 400, "overlap": 0,   "filtering": []},
    {"strategy": "RecursiveToken",  "chunk_size": 400, "overlap": 200, "filtering": []},
    {"strategy": "RecursiveToken",  "chunk_size": 800, "overlap": 400, "filtering": []},
    # ── ClusterSemantic ─────────────────────────────────────────────────────
    {"strategy": "ClusterSemantic", "chunk_size": 200, "overlap": 0,   "filtering": []},
    {"strategy": "ClusterSemantic", "chunk_size": 400, "overlap": 0,   "filtering": []},
]

# ── Embedding ──────────────────────────────────────────────────────────────
EMBED_MODEL = "nomic-embed-text"
OLLAMA_BASE_URL = "http://localhost:11434"
TEXT_EMBED_DIM = 768
EMBED_BATCH_SIZE = 64

USE_SPARSE = True
SPARSE_VECTOR_NAME = "bm25"
DENSE_VECTOR_NAME = "dense"

# ── Qdrant (in-process / embedded mode) ───────────────────────────────────
COLLECTION_PREFIX = "squad_bench"

# ── Retrieval ──────────────────────────────────────────────────────────────
TOP_K = 5
HYBRID_ALPHA = 0.7

# ── LLM ───────────────────────────────────────────────────────────────────
LLM_MODEL = "mistral"
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 256

SYSTEM_PROMPT = """\
You are a precise question-answering assistant. Answer the question using \
ONLY the provided context passages. Be concise — answer in 1-3 sentences. \
If the answer is not in the context, respond with exactly: "I don't know."\
"""

# ── Evaluation ─────────────────────────────────────────────────────────────
EVAL_MODES = ["raw", "preprocessed"]
EVAL_LANGUAGE = "english"
ORACLE_K = 5
