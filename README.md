# rag-bench v3

Reproduces chunk filtering experiments from **Berdyugina et al. (arXiv:2604.24334)**  
*"Reducing Redundancy in Retrieval-Augmented Generation through Chunk Filtering"*

## What changed from v2 baseline

| Component | v2 baseline | v3 (this repo) |
|---|---|---|
| Embedding | `nomic-embed-text` via Ollama (768-dim) | `all-MiniLM-L6-v2` via sentence-transformers (384-dim) |
| Retrieval | Hybrid dense + BM25 sparse (RRF) | **Dense only** (cosine similarity) |
| Filtering | Placeholder (not implemented) | **Fully implemented**: ExactNorm, MinHashLSH, Similarity, NERExact |
| Vector store | Qdrant embedded | Qdrant embedded (unchanged) |

## Pipeline

```
load (SQuAD 1.1)
    │
    ▼
chunk (FixedToken / RecursiveToken / ClusterSemantic)
    │
    ▼
filter (NoFilter / ExactNorm / MinHashLSH(0.7) / Similarity(0.8) / NERExact)
    │
    ▼
embed (all-MiniLM-L6-v2, 384-dim, sentence-transformers)
    │
    ▼
index → Qdrant embedded (no Docker, in-process)
    │
    ▼
retrieve (dense cosine, top-k=5)
    │
    ▼
[generate] (Mistral 7B via Ollama — optional, --skip-generation to bypass)
    │
    ▼
evaluate (Precision / Recall / IoU / Oracle — paper Section 3.6)
```

## Stack

| Layer | Tool |
|---|---|
| Dataset | SQuAD 1.1 (`rajpurkar/squad`, validation split) |
| Chunking | FixedToken, RecursiveToken, ClusterSemantic |
| Filtering | ExactNorm, MinHashLSH(0.7), Similarity(0.8), NERExact |
| Embedding | `sentence-transformers/all-MiniLM-L6-v2` (384-dim) |
| Vector store | Qdrant in-process (embedded, path-based, no Docker) |
| Retrieval | Dense cosine, top-k=5 |
| LLM (optional) | Mistral 7B via Ollama |
| Eval metrics | Precision / Recall / IoU / Oracle / EM / Token F1 |

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 2. (Optional — only needed for generation metrics)
ollama pull mistral

# 3. Debug run (fast — 20 docs, 30 questions, no LLM)
python scripts/benchmark.py --max-docs 20 --max-questions 30 --skip-generation

# 4. Single strategy debug
python scripts/benchmark.py --strategy RecursiveToken --max-docs 50 --max-questions 50 --skip-generation

# 5. Single filter debug
python scripts/benchmark.py --filter NERExact --max-docs 50 --max-questions 50 --skip-generation

# 6. Single config
python scripts/benchmark.py --config "RecursiveToken_400_0__NERExact" --skip-generation

# 7. Full benchmark — all 50 configs (10 chunkers × 5 filters), no LLM
python scripts/benchmark.py --skip-generation

# 8. Full benchmark with generation
python scripts/benchmark.py

# 9. Background run
nohup python scripts/benchmark.py --skip-generation > results/bench.log 2>&1 &
tail -f results/bench.log
```

## Configs generated

**10 chunking configs × 5 filter methods = 50 total configs**

| Chunker | Configs |
|---|---|
| FixedToken | (200,0), (400,0), (400,200), (800,400) |
| RecursiveToken | (200,0), (400,0), (400,200), (800,400) |
| ClusterSemantic | (200,0), (400,0) |

| Filter | Method | Paper ref |
|---|---|---|
| NoFilter | Pass-through baseline | Section 3.7 |
| ExactNorm | Exact dedup after normalization | Section 3.5 |
| MinHashLSH(0.7) | Near-dup via Jaccard / MinHash | Section 3.5 |
| Similarity(0.8) | Cosine similarity threshold | Section 3.5 |
| NERExact | Identical named-entity set | Section 3.5 |

## Output

`results/benchmark_results.csv` — one row per config:

| Column | Description |
|---|---|
| `config_name` | e.g. `RecursiveToken_400_0__NERExact` |
| `filter_method` | Filter tag |
| `chunk_count_before_filter` | Chunks produced by chunker |
| `chunk_count_after_filter` | Chunks after filtering |
| `filter_reduction_pct` | % chunks removed |
| `ingest_time_s` | Chunking + filtering + embedding + upsert (seconds) |
| `storage_mb` | Qdrant collection disk size (MB, numeric) |
| `storage_du` | Qdrant collection disk size (du -sh output) |
| `precision_raw` / `recall_raw` / `iou_raw` | Token metrics, raw mode |
| `precision_pre` / `recall_pre` / `iou_pre` | Token metrics, preprocessed mode |
| `oracle_recall_raw` | Greedy oracle upper bound |
| `exact_match` / `token_f1` | Generation accuracy (if not --skip-generation) |

`results/per_question_{config_name}.csv` — one row per question.

## Project layout

```
rag-bench-v3/
├── configs/
│   └── settings.py              # all parameters + config generation
├── src/
│   ├── ingestion/
│   │   ├── loader.py            # SQuAD 1.1 loader
│   │   ├── chunker.py           # FixedToken, RecursiveToken, ClusterSemantic
│   │   ├── embedder.py          # all-MiniLM-L6-v2 via sentence-transformers
│   │   └── vector_store.py      # Qdrant in-process, dense only
│   ├── filtering/
│   │   ├── filters.py           # NoFilter, ExactNorm, MinHashLSH, Similarity, NERExact
│   │   └── pipeline.py          # FilteringPipeline — sequential composition
│   ├── retrieval/
│   │   └── retriever.py         # dense cosine retrieval
│   ├── evaluation/
│   │   ├── metrics.py           # Precision/Recall/IoU/Oracle/EM/F1
│   │   └── generator.py         # Ollama LLM generation (optional)
│   └── utils/
│       └── logger.py
├── scripts/
│   └── benchmark.py             # main CLI
├── data/
│   └── qdrant_storage/          # embedded Qdrant data
├── results/                     # CSV outputs
├── requirements.txt
└── README.md
```
