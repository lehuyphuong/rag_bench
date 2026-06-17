# rag-bench v2

Benchmarks RAG chunking strategies on SQuAD 1.1 with hybrid retrieval
(dense nomic-embed-text + sparse BM25) and end-to-end LLM evaluation.

Follows evaluation protocol from Berdyugina et al. (arXiv:2604.24334):
token-based Precision / Recall / IoU with raw and preprocessed tokenization,
plus oracle upper bound. Adds SQuAD Exact Match and Token F1 for generation.

**No Docker, no root required** — Qdrant runs in embedded (in-process) mode.
Suitable for vast.ai and other GPU cloud instances.

## Pipeline

```
load (SQuAD 1.1)
    │
    ▼
chunk (FixedToken / RecursiveToken / ClusterSemantic)
    │
    ▼
[filter]  ←── placeholder, not yet implemented (filtering=[] in all configs)
    │          Future: ExactNorm, MinHashLSH, Similarity, NER_Exact, Merge...
    ▼
embed (nomic dense 768-dim + BM25 sparse)
    │
    ▼
index → Qdrant embedded (no Docker)
    │
    ▼
retrieve (hybrid RRF, top-k=5)
    │
    ▼
generate (Mistral 7B via Ollama)
    │
    ▼
evaluate (Precision / Recall / IoU / Oracle / EM / F1)
```

## Stack

| Layer | Tool |
|---|---|
| Dataset | SQuAD 1.1 (`rajpurkar/squad`, validation split) |
| Chunking | FixedToken, RecursiveToken, ClusterSemantic |
| Filtering | `src/filtering/` — placeholder, to be implemented |
| Dense embedding | `nomic-embed-text-v1.5` via Ollama (768-dim) |
| Sparse embedding | BM25 (computed locally, no model) |
| Vector store | Qdrant in-process (embedded, path-based, no Docker) |
| Retrieval | Hybrid dense+sparse with RRF fusion, top-k=5 |
| LLM | Mistral 7B via Ollama |
| Eval metrics | Precision / Recall / IoU / Oracle / EM / Token F1 |

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 2. Pull Ollama models
ollama pull nomic-embed-text
ollama pull mistral

# 3. Smoke test (fast, no LLM)
python scripts/benchmark.py --max-docs 50 --max-questions 30 --skip-generation

# 4. Full benchmark (all 10 configs)
python scripts/benchmark.py

# 5. Single config
python scripts/benchmark.py --config RecursiveToken_400_0

# 6. Background run on GPU instance
nohup python scripts/benchmark.py > results/bench.log 2>&1 &
tail -f results/bench.log
```

## Chunking configurations (current baseline, no filtering)

| Config name | Strategy | chunk_size | overlap |
|---|---|---|---|
| FixedToken_200_0 | FixedToken | 200 | 0 |
| FixedToken_400_0 | FixedToken | 400 | 0 |
| FixedToken_400_200 | FixedToken | 400 | 200 |
| FixedToken_800_400 | FixedToken | 800 | 400 |
| RecursiveToken_200_0 | RecursiveToken | 200 | 0 |
| RecursiveToken_400_0 | RecursiveToken | 400 | 0 |
| RecursiveToken_400_200 | RecursiveToken | 400 | 200 |
| RecursiveToken_800_400 | RecursiveToken | 800 | 400 |
| ClusterSemantic_200_0 | ClusterSemantic | 200 | 0 |
| ClusterSemantic_400_0 | ClusterSemantic | 400 | 0 |

## Output

`results/benchmark_results.csv` — one row per config:

| Column | Description |
|---|---|
| `chunk_count_before_filter` | chunks produced by chunker |
| `chunk_count_after_filter` | chunks after filtering (= before when filtering=[]) |
| `filter_reduction_pct` | % chunks removed by filtering |
| `ingest_time_s` | chunking + filtering + embedding + upsert (seconds) |
| `storage_mb` | Qdrant collection disk size (MB) |
| `recall_raw` / `recall_pre` | token recall, raw and preprocessed |
| `oracle_recall_raw` | greedy oracle upper bound on recall |
| `exact_match` / `token_f1` | generation accuracy |

`results/per_question_{config_name}.csv` — one row per question.

## Project layout

```
rag-bench/
├── configs/
│   └── settings.py              # all parameters + chunking config list
├── src/
│   ├── ingestion/
│   │   ├── loader.py            # SQuAD 1.1 loader
│   │   ├── chunker.py           # FixedToken, RecursiveToken, ClusterSemantic
│   │   ├── embedder.py          # nomic dense + BM25 sparse
│   │   └── vector_store.py      # Qdrant in-process CRUD + disk size
│   ├── filtering/               # ← placeholder for future work
│   │   ├── __init__.py          # design doc: planned methods + merge analysis
│   │   └── pipeline.py          # FilteringPipeline (pass-through until implemented)
│   ├── retrieval/
│   │   └── retriever.py         # hybrid RRF (dense + sparse)
│   ├── evaluation/
│   │   ├── metrics.py           # Precision/Recall/IoU/Oracle/EM/F1
│   │   └── generator.py         # Ollama LLM generation
│   └── utils/
│       └── logger.py
├── scripts/
│   └── benchmark.py             # main CLI
├── data/
│   └── qdrant_storage/          # embedded Qdrant data (gitignored)
├── results/                     # CSV outputs (gitignored)
├── requirements.txt
└── README.md
```

## Adding filtering (future)

To add a new filter method:

1. Create (or add to) the appropriate file in `src/filtering/`:
   `lexical.py`, `semantic.py`, or `structural.py`

2. Implement a class with an `apply(chunks, embed_fn) → list[dict]` method.

3. Register it: `@register("MethodName")` decorator from `pipeline.py`.

4. Add configs to `CHUNKING_CONFIGS` in `settings.py`:
   ```python
   {"strategy": "RecursiveToken", "chunk_size": 400, "overlap": 200,
    "filtering": [{"method": "NER_Exact"}]}
   ```

No changes needed to `benchmark.py` or any other file.
