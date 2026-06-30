# rag-bench-v5

Benchmark framework for evaluating **chunking** and **chunk filtering** strategies in a RAG (Retrieval-Augmented Generation) pipeline on SQuAD 1.1.

Reproduces the experimental setup of **Berdyugina et al. (arXiv:2604.24334)**,
*"Reducing Redundancy in Retrieval-Augmented Generation through Chunk Filtering"*, extended with 5 additional chunking strategies for direct comparison against a separate project, `cacd-dedup`.

---

## 1. Overview

### 1.1 Goal

Evaluate the interaction between **9 chunking strategies** and **5 filtering methods** using the paper's 3 core metrics — **Precision**, **Recall**, **IoU** — plus **Index Size** (chunk count and storage MB).

### 1.2 Differences from the original paper

| Component | Original paper | rag-bench-v5 |
|---|---|---|
| Chunking strategies | FixedToken, RecursiveToken, ClusterSemantic, Overlapping | Same 4 classic strategies + 5 new ones (AdaptiveEntropy, AdaptiveSentenceLen, HierarchicalParentChild, Contextual, TopicBased) |
| Embedding model | `all-MiniLM-L6-v2` | `all-MiniLM-L6-v2` (unchanged), GPU-aware |
| Retrieval | Dense-only cosine similarity | Dense-only cosine similarity (unchanged) |
| Vector store | ChromaDB | Qdrant embedded (no Docker) |
| Dataset | Chroma corpora, SQuAD 1.1, WebFAQ | SQuAD 1.1 (validation split) |
| Eval metrics | Precision, Recall, IoU, Oracle | Precision, Recall, IoU, Index Size |

---

## 2. Pipeline

```
load (SQuAD 1.1 — rajpurkar/squad, validation split)
    |
    v
chunk (FixedToken / RecursiveToken / ClusterSemantic / Overlapping /
       AdaptiveEntropy / AdaptiveSentenceLen / HierarchicalParentChild /
       Contextual / TopicBased)
    |
    v
filter (NoFilter / ExactNorm / MinHashLSH(0.7) / Similarity(0.8) / NERExact)
    |
    v
embed (sentence-transformers/all-MiniLM-L6-v2, 384-dim, GPU-aware)
    |
    v
index => Qdrant embedded (in-process, no Docker)
    |
    v
retrieve (dense cosine similarity, top-k = 5)
    |
    v
evaluate (Precision / Recall / IoU / Index Size)
```

---

## 3. Chunking Strategies

### Classic (paper Section 3.3)

| Strategy | Mechanism | Params |
|---|---|---|
| FixedToken | Fixed-length character windows, no boundary awareness | chunk_size=200,400 |
| RecursiveToken | Hierarchical split: paragraph => sentence => space => character | chunk_size=200,400 |
| ClusterSemantic | Sequential breakpoint via cosine-distance percentile | chunk_size=200,400 |
| Overlapping | Sliding window with mandatory overlap (main source of redundancy) | size=400/overlap=200, 800/400 |

### New (paper v4 extensions)

| Strategy | Mechanism | Params |
|---|---|---|
| AdaptiveEntropy | Chunk size adapts to local Shannon entropy: `target_size = lerp(max_size, min_size, normalized_H)` | base_size=300,500 |
| AdaptiveSentenceLen | Chunk size adapts to mean sentence length in a rolling window | target_sentences=4,6 |
| HierarchicalParentChild | Two-level chunking; both parent and child chunks are indexed | child=200/parent=600, child=400/parent=800 |
| Contextual | Each chunk prepended with `[Context: {title} \| Part {i}/{n}]` | chunk_size=300,500 |
| TopicBased | Sentences embedded and k-means clustered; chunk boundaries placed where the topic label changes | n_topics=4,6 |

=> **18 configs total** (9 strategies x 2 size variants).

---

## 4. Filtering Methods

| Method | Mechanism | Signal | Paper ref |
|---|---|---|---|
| NoFilter | Pass-through, no chunks removed | — | Section 3.7 |
| ExactNorm | Removes chunks with identical text after normalization | Lexical exact | Section 3.5 |
| MinHashLSH(0.7) | Removes chunks with Jaccard similarity >= 0.7 (MinHash + LSH) | Lexical near-dup | Section 3.5 |
| Similarity(0.8) | Removes chunks with cosine similarity >= 0.8 on embeddings | Semantic | Section 3.5 |
| NERExact | Removes chunks with an identical named-entity set (spaCy) | Structural | Section 3.5 |

=> **90 configs total** (18 chunking configs x 5 filter methods).

---

## 5. Evaluation Metrics

Following Berdyugina et al. Section 3.6 (token-coverage protocol):

```
Te = token set of the reference passage (ground-truth context)
Tr = token set of the union of retrieved top-k chunks
```

| Metric | Formula |
|---|---|
| Precision | \|Te ∩ Tr\| / \|Tr\| |
| Recall | \|Te ∩ Tr\| / \|Te\| |
| IoU | \|Te ∩ Tr\| / \|Te ∪ Tr\| |
| Index Size | chunk_count_after_filter + storage_mb |

Computed under two tokenization modes: `raw` (lowercase word tokens) and `preprocessed` (stopword removal + lemmatization via spaCy).

---

## 6. Technical stack

| Layer | Tool | Detail |
|---|---|---|
| Dataset | SQuAD 1.1 | `rajpurkar/squad`, validation split |
| Chunking | 9 strategies | 4 classic + 5 new (see Section 3) |
| Filtering | 5 methods | NoFilter, ExactNorm, MinHashLSH(0.7), Similarity(0.8), NERExact |
| Embedding | all-MiniLM-L6-v2 | sentence-transformers, 384-dim, L2-normalized, GPU-aware |
| Vector store | Qdrant embedded | In-process, path-based, no Docker |
| Retrieval | Dense cosine | top-k = 5 |
| NER | spaCy | `en_core_web_sm` |
| MinHash | datasketch | 128 permutations, character 3-grams |
| Topic clustering | scikit-learn | KMeans, n_init="auto" |

---

## 7. Setup and usage

### 7.1 Setup

```bash
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 7.2 Verify installation

```bash
python -c "from sentence_transformers import SentenceTransformer; m = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); print('embed OK:', m.encode(['test']).shape)"
python -c "from qdrant_client import QdrantClient; c = QdrantClient(':memory:'); print('qdrant OK')"
python -c "import spacy; nlp = spacy.load('en_core_web_sm'); print('spacy OK')"
python -c "from sklearn.cluster import KMeans; print('sklearn OK')"
python -c "from datasketch import MinHash; print('datasketch OK')"
python -c "import torch; print('GPU available:', torch.cuda.is_available())"
```

### 7.3 Run benchmark

```bash
# Quick debug (20 docs, 30 questions)
python scripts/benchmark.py --max-docs 20 --max-questions 30

# Single strategy
python scripts/benchmark.py --strategy HierarchicalParentChild --max-docs 50 --max-questions 50

# Single filter
python scripts/benchmark.py --filter NERExact --max-docs 50 --max-questions 50

# Only the 5 new v4 strategies (50 configs)
python scripts/benchmark.py --v4-only

# Only the 4 classic v3 strategies (40 configs)
python scripts/benchmark.py --classic-only

# Single config
python scripts/benchmark.py --config "HierarchicalParentChild_400_0__NERExact"

# Full benchmark (all 90 configs)
rm -f results/benchmark_results.csv   # clear previous results first
python scripts/benchmark.py

# Background run
nohup python scripts/benchmark.py > results/bench.log 2>&1 &
tail -f results/bench.log
```

### 7.4 Inspect results

```bash
# View the results CSV
column -t -s',' results/benchmark_results.csv | less -S

# Count configs run so far
tail -n +2 results/benchmark_results.csv | wc -l

# Check Qdrant storage on disk
du -sh data/qdrant_storage/
```

---

## 8. Project structure

```
rag-bench-v5/
├── configs/
│   └── settings.py              # Parameters + auto-generates 90 configs
├── src/
│   ├── ingestion/
│   │   ├── loader.py            # SQuAD 1.1 loader (HuggingFace datasets)
│   │   ├── chunker.py           # 9 chunking strategies
│   │   ├── embedder.py          # all-MiniLM-L6-v2 via sentence-transformers (GPU-aware)
│   │   └── vector_store.py      # Qdrant embedded, unnamed vector API
│   ├── filtering/
│   │   ├── filters.py           # NoFilter, ExactNorm, MinHashLSH, Similarity, NERExact
│   │   └── pipeline.py          # FilteringPipeline — sequential composition
│   ├── retrieval/
│   │   └── retriever.py         # Dense cosine retrieval (Qdrant query_points)
│   ├── evaluation/
│   │   ├── metrics.py           # Precision / Recall / IoU (raw + preprocessed)
│   │   └── generator.py         # Ollama LLM — optional, not used in core evaluation
│   └── utils/
│       └── logger.py
├── scripts/
│   └── benchmark.py             # Main CLI: chunk => filter => embed => index => eval
├── data/
│   └── qdrant_storage/          # Qdrant collections (created on first run)
├── results/                     # benchmark_results.csv + per_question_*.csv
├── requirements.txt
└── README.md
```

---

## 9. Output format

### `results/benchmark_results.csv` — one row per config

| Column | Description |
|---|---|
| `config_name` | Unique identifier, e.g. `HierarchicalParentChild_400_0__NERExact` |
| `strategy` | Chunking strategy name |
| `chunk_size` | Target chunk size in characters |
| `overlap` | Overlap between chunks |
| `filter_method` | Filter method applied |
| `chunk_count_before_filter` | Chunk count right after chunking |
| `chunk_count_after_filter` | Chunk count after filtering |
| `filter_reduction_pct` | Percentage of chunks removed |
| `ingest_time_s` | Total time: chunk + filter + embed + upsert (seconds) |
| `storage_mb` | Qdrant collection disk size (MB, numeric) |
| `storage_du` | Qdrant collection disk size (`du -sh` output) |
| `precision_raw` / `recall_raw` / `iou_raw` | Token metrics, raw mode |
| `precision_pre` / `recall_pre` / `iou_pre` | Token metrics, preprocessed mode |
| `avg_retrieval_ms` | Mean retrieval latency (ms/query) |
| `n_questions` | Number of questions evaluated |

### `results/per_question_{config_name}.csv` — one row per question

Per-question Precision, Recall, IoU — used for distribution and outlier analysis.

---

## 10. References

- Berdyugina, D., Cohen, A., & Rioual, Y. (2026). *Reducing Redundancy in Retrieval-Augmented Generation through Chunk Filtering*. arXiv:2604.24334.
- Lewis, P. et al. (2021). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. arXiv:2005.11401.
- Anthropic. (2024). *Introducing Contextual Retrieval*.
- Rajpurkar, P. et al. (2016). *SQuAD: 100,000+ Questions for Machine Comprehension of Text*. EMNLP 2016.
- Reimers, N. & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. arXiv:1908.10084.
