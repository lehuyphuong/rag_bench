# rag-bench

Benchmark framework đánh giá các chiến lược **chunking** và **chunk filtering** trong pipeline RAG (Retrieval-Augmented Generation) trên bộ dữ liệu SQuAD 1.1.

Được phát triển dựa trên nghiên cứu của **Berdyugina et al. (arXiv:2604.24334)**:  
*"Reducing Redundancy in Retrieval-Augmented Generation through Chunk Filtering"*

---

## 1. Tổng quan

### 1.1 Mục tiêu

Đánh giá sự tương tác giữa **5 chiến lược chunking mới** và **5 phương pháp filtering** thông qua 3 metrics cốt lõi của paper: **Precision**, **Recall**, **IoU** — cùng với **Index Size** (số chunks và dung lượng storage).

### 1.2 Sự khác biệt so với paper gốc

| Thành phần | Paper gốc (Berdyugina et al.) | rag-bench v4 |
|---|---|---|
| Chunking strategies | FixedToken, RecursiveToken, ClusterSemantic | AdaptiveEntropy, AdaptiveSentenceLen, HierarchicalParentChild, Contextual, TopicBased |
| Embedding model | `all-MiniLM-L6-v2` | `all-MiniLM-L6-v2` (giữ nguyên) |
| Retrieval | Dense-only cosine similarity | Dense-only cosine similarity (giữ nguyên) |
| Vector store | ChromaDB | Qdrant embedded (no Docker) |
| Dataset | Chroma corpora, SQuAD 1.1, WebFAQ | SQuAD 1.1 (validation split) |
| Eval metrics | Precision, Recall, IoU, Oracle | Precision, Recall, IoU, Index Size |

---

## 2. Pipeline

```
load (SQuAD 1.1 — rajpurkar/squad, validation split)
    │
    ▼
chunk (AdaptiveEntropy / AdaptiveSentenceLen /
       HierarchicalParentChild / Contextual / TopicBased)
    │
    ▼
filter (NoFilter / ExactNorm / MinHashLSH(0.7) /
        Similarity(0.8) / NERExact)
    │
    ▼
embed (sentence-transformers/all-MiniLM-L6-v2, 384-dim)
    │
    ▼
index => Qdrant embedded (in-process, no Docker)
    │
    ▼
retrieve (dense cosine similarity, top-k = 5)
    │
    ▼
evaluate (Precision / Recall / IoU / Index Size)
```

---

## 3. Chunking Strategies

### 3.1 AdaptiveEntropy

Điều chỉnh kích thước chunk dựa trên **Shannon entropy** của văn bản.

- **Nguyên lý:** Đoạn văn bản có entropy cao (thông tin dày đặc) => chunk nhỏ hơn. Đoạn có entropy thấp (lặp lại, đơn giản) => chunk lớn hơn.
- **Công thức:** `target_size = lerp(max_size, min_size, normalized_H)`
- **Tham số:** `base_size` (300 hoặc 500 ký tự)

### 3.2 AdaptiveSentenceLen

Điều chỉnh số câu mỗi chunk dựa trên **độ dài trung bình của câu** trong cửa sổ cục bộ.

- **Nguyên lý:** Đoạn có câu ngắn => ít câu mỗi chunk (granularity cao). Đoạn có câu dài => nhiều câu mỗi chunk (granularity thấp).
- **Ngưỡng:** `short_threshold = 60 chars/sent`, `long_threshold = 120 chars/sent`
- **Tham số:** `target_sentences` (4 hoặc 6 câu)

### 3.3 HierarchicalParentChild

Chunking **2 tầng**: parent chunk lớn chứa nhiều child chunk nhỏ. Cả hai tầng đều được index vào vector database.

- **Nguyên lý:** Child chunks cung cấp retrieval precision cao; parent chunks cung cấp context rộng. Kết hợp cả hai giúp hệ thống tìm kiếm đa tầng.
- **Tham số:** `child_size` (200 hoặc 400) và `parent_size` (gấp 2-3 lần child)
- **Lưu ý:** Số chunks tăng gấp ~3-4 lần so với các strategy khác.

### 3.4 Contextual

Mỗi chunk được **prepend một header context** chứa tên tài liệu và vị trí tương đối.

- **Nguyên lý:** Embedding của chunk phản ánh cả nội dung cục bộ lẫn bối cảnh tài liệu. Tham khảo từ *Contextual Retrieval* (Anthropic, 2024).
- **Format header:** `[Context: {title} | Part {i}/{n}] {chunk_text}`
- **Tham số:** `chunk_size` (300 hoặc 500 ký tự)

### 3.5 TopicBased

Phân đoạn tài liệu theo **chủ đề** bằng cách embedding từng câu rồi phân cụm K-means.

- **Nguyên lý:** Câu liên tiếp thuộc cùng topic cluster => gộp vào một chunk. Chunk mới được tạo khi topic thay đổi.
- **Tham số:** `n_topics` (4 hoặc 6 clusters), `min_chunk_size`
- **Fallback:** Nếu document < 3 câu => AdaptiveSentenceLen

---

## 4. Filtering Methods

| Method | Mô tả | Tín hiệu | Ref paper |
|---|---|---|---|
| **NoFilter** | Pass-through, không loại chunk nào | — | Section 3.7 |
| **ExactNorm** | Loại chunk có text giống hệt nhau sau normalization | Lexical exact | Section 3.5 |
| **MinHashLSH(0.7)** | Loại chunk có Jaccard similarity ≥ 0.7 (MinHash + LSH) | Lexical near-dup | Section 3.5 |
| **Similarity(0.8)** | Loại chunk có cosine similarity ≥ 0.8 trên embedding | Semantic | Section 3.5 |
| **NERExact** | Loại chunk có tập named-entity giống hệt nhau (spaCy) | Structural | Section 3.5 |

---

## 5. Evaluation Metrics

Theo **Berdyugina et al. Section 3.6** — token-coverage protocol:

```
Te = tập token của reference passage (ground-truth context)
Tr = tập token của union các top-k chunks được retrieve
```

| Metric | Công thức | Ý nghĩa |
|---|---|---|
| **Precision** | \|Te ∩ Tr\| / \|Tr\| | Tỉ lệ token retrieve được là relevant |
| **Recall** | \|Te ∩ Tr\| / \|Te\| | Tỉ lệ token cần thiết được bao phủ |
| **IoU** | \|Te ∩ Tr\| / \|Te ∪ Tr\| | Jaccard similarity giữa 2 tập token |
| **Index Size** | chunk_count + storage_mb | Hiệu quả lưu trữ sau filtering |

Mỗi metric được tính trên **2 chế độ tokenization**:
- `raw`: lowercase word tokens
- `preprocessed`: bỏ stopwords + lemmatize (spaCy)

---

## 6. Kết quả thực nghiệm (500 docs, 200 questions)

### 6.1 So sánh theo Chunking Strategy

| Strategy | Precision | Recall | IoU | Chunks TB | Storage MB | Latency ms |
|---|---|---|---|---|---|---|
| **HierarchicalParentChild** | **0.374** | 0.820 | **0.345** | 2100 | 10.19 | 38.97 |
| TopicBased | 0.331 | 0.708 | 0.287 | 1303 | 6.24 | 29.77 |
| Contextual | 0.298 | 0.744 | 0.267 | 1179 | 5.82 | 27.57 |
| AdaptiveEntropy | 0.292 | 0.832 | 0.275 | 907 | 4.46 | 9.19 |
| **AdaptiveSentenceLen** | 0.242 | **0.947** | 0.240 | 545 | 2.56 | **8.53** |

### 6.2 So sánh theo Filtering Method

| Filter | Precision | Recall | IoU | Reduction% | Storage MB |
|---|---|---|---|---|---|
| NoFilter | 0.314 | 0.805 | 0.288 | 0.00% | 6.21 |
| ExactNorm | 0.308 | 0.810 | 0.284 | 1.58% | 6.07 |
| MinHashLSH(0.7) | 0.307 | 0.810 | 0.282 | 2.82% | 5.96 |
| NERExact | 0.308 | 0.810 | 0.284 | 6.77% | 5.68 |
| **Similarity(0.8)** | 0.299 | **0.816** | 0.276 | **10.16%** | **5.34** |

### 6.3 Top configs theo IoU

| Config | IoU | Recall | Precision | Reduction% |
|---|---|---|---|---|
| HierarchicalParentChild_200_0__NoFilter | **0.3749** | 0.7418 | 0.4328 | 0.0% |
| HierarchicalParentChild_400_0__NoFilter | 0.3706 | 0.8507 | 0.3849 | 0.0% |
| HierarchicalParentChild_200_0__NERExact | 0.3666 | 0.7484 | 0.4187 | 11.9% |
| HierarchicalParentChild_200_0__MinHashLSH0.7 | 0.3621 | 0.7422 | 0.4130 | 6.9% |
| TopicBased_400_0__Similarity0.8 | 0.2967 | 0.7737 | 0.3264 | 3.6% |

### 6.4 Top configs theo Recall

| Config | Recall | IoU | Precision | Reduction% |
|---|---|---|---|---|
| AdaptiveSentenceLen_6_0__Similarity0.8 | **0.9501** | 0.2389 | 0.2403 | 3.9% |
| AdaptiveSentenceLen_6_0__NoFilter | 0.9478 | 0.2373 | 0.2388 | 0.0% |
| AdaptiveSentenceLen_4_0__NoFilter | 0.9454 | 0.2429 | 0.2447 | 0.0% |
| AdaptiveSentenceLen_4_0__Similarity0.8 | 0.9441 | 0.2440 | 0.2459 | 3.3% |
| HierarchicalParentChild_400_0__MinHashLSH0.7 | 0.8968 | 0.3248 | 0.3315 | 20.9% |

### 6.5 Filter impact trên Recall (delta vs NoFilter)

| Strategy | ExactNorm | MinHashLSH | Similarity | NERExact |
|---|---|---|---|---|
| AdaptiveEntropy | +0.000 | +0.000 | +0.000 | -0.068 |
| AdaptiveSentenceLen | +0.000 | +0.000 | +0.002 | +0.000 |
| **HierarchicalParentChild** | **+0.037** | **+0.046** | **+0.069** | **+0.037** |
| Contextual | +0.000 | +0.000 | +0.001 | **-0.143** |
| TopicBased | +0.000 | +0.000 | +0.001 | **-0.131** |

### 6.6 Efficiency — IoU per MB (top 5)

| Config | IoU | Storage MB | IoU/MB | Latency ms |
|---|---|---|---|---|
| AdaptiveSentenceLen_6_0__Similarity0.8 | 0.2389 | 2.33 | **0.1025** | 7.93 |
| AdaptiveSentenceLen_6_0__NERExact | 0.2373 | 2.39 | 0.0993 | 7.93 |
| AdaptiveSentenceLen_6_0__NoFilter | 0.2373 | 2.43 | 0.0977 | 8.82 |
| AdaptiveSentenceLen_4_0__Similarity0.8 | 0.2440 | 2.65 | 0.0921 | 8.42 |
| AdaptiveSentenceLen_4_0__NERExact | 0.2429 | 2.70 | 0.0900 | 8.47 |

---

## 7. Phân tích và Nhận xét

### 7.1 Chunking Strategy

**HierarchicalParentChild** đạt IoU và Precision cao nhất nhờ index đồng thời parent + child chunks — retrieval lấy được cả context rộng lẫn chi tiết cụ thể. Tuy nhiên storage tốn gấp ~4x và latency cao nhất (39ms).

**AdaptiveSentenceLen** là đối cực — Recall 0.947 (cao nhất) nhưng Precision chỉ 0.242, chunks rất nhỏ nên bao phủ rộng nhưng noisy. Storage nhỏ nhất (2.56MB) và nhanh nhất (8.5ms). Phù hợp với use-case cần độ phủ cao và tài nguyên hạn chế.

**TopicBased và Contextual** có Recall thấp hơn kỳ vọng (~0.7) trên 500 docs. Với full dataset (2067 docs), các strategy này có thể cải thiện do clustering sẽ ổn định hơn.

### 7.2 Filtering Method

**ExactNorm và MinHashLSH loại được rất ít** (1–3%) — xác nhận đúng kết luận của paper: redundancy trong SQuAD không phải dạng lexical exact-duplicate.

**Similarity(0.8)** là filter hiệu quả nhất: loại nhiều nhất (10%) trong khi Recall tăng nhẹ (+0.011 so với NoFilter). Hiệu ứng này do lọc bỏ các chunk quá giống nhau giúp top-5 retrieved chunks đa dạng hơn, bao phủ nhiều phần của reference passage hơn.

**NERExact** không nhất quán: hoạt động tốt với HierarchicalParentChild (+0.037) nhưng gây hại nặng với Contextual (-0.143) và TopicBased (-0.131). Nguyên nhân: Contextual thêm header cố định vào mỗi chunk => nhiều chunks có cùng entity set dù nội dung khác nhau => bị loại nhầm.

### 7.3 Chỉ có HierarchicalParentChild phản ứng mạnh với Filtering

4 strategy còn lại gần như không thay đổi Recall khi áp dụng filter (delta ≈ 0). HierarchicalParentChild có delta dương đáng kể với mọi filter vì sinh ra ~3067 chunks (gấp 5–6x) — tạo ra nhiều redundancy thực sự giữa parent và child chunks chồng lấp nội dung.

### 7.4 Khuyến nghị theo use-case

| Use-case | Config được đề xuất |
|---|---|
| Chất lượng retrieval cao nhất | `HierarchicalParentChild_400_0__NoFilter` |
| Cân bằng IoU và Reduction | `HierarchicalParentChild_400_0__MinHashLSH0.7` (IoU=0.325, -20.9%) |
| Recall tối đa | `AdaptiveSentenceLen_6_0__Similarity0.8` (Recall=0.950) |
| Hiệu quả nhất (IoU/MB) | `AdaptiveSentenceLen_6_0__Similarity0.8` (IoU/MB=0.1025) |
| Realtime / tài nguyên thấp | `AdaptiveSentenceLen_6_0__NERExact` (7.93ms, 2.39MB) |

---

## 8. Stack kỹ thuật

| Layer | Tool | Chi tiết |
|---|---|---|
| Dataset | SQuAD 1.1 | `rajpurkar/squad`, validation split |
| Chunking | 5 strategies mới | AdaptiveEntropy, AdaptiveSentenceLen, HierarchicalParentChild, Contextual, TopicBased |
| Filtering | 5 methods | NoFilter, ExactNorm, MinHashLSH(0.7), Similarity(0.8), NERExact |
| Embedding | all-MiniLM-L6-v2 | sentence-transformers, 384-dim, L2-normalized |
| Vector store | Qdrant embedded | In-process, path-based, no Docker |
| Retrieval | Dense cosine | top-k = 5 |
| NER | spaCy | `en_core_web_sm` |
| MinHash | datasketch | 128 permutations, character 3-grams |
| Topic clustering | scikit-learn | KMeans, n_init="auto" |

---

## 9. Cài đặt và chạy

### 9.1 Setup

```bash
# Tạo virtual environment
python3 -m venv venv
source venv/bin/activate

# Cài dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 9.2 Verify cài đặt

```bash
python -c "from sentence_transformers import SentenceTransformer; m = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); print('embed OK:', m.encode(['test']).shape)"
python -c "from qdrant_client import QdrantClient; c = QdrantClient(':memory:'); print('qdrant OK')"
python -c "import spacy; nlp = spacy.load('en_core_web_sm'); print('spacy OK')"
python -c "from sklearn.cluster import KMeans; print('sklearn OK')"
python -c "from datasketch import MinHash; print('datasketch OK')"
```

### 9.3 Chạy benchmark

```bash
# Debug nhỏ (20 docs, 30 questions)
python scripts/benchmark.py --max-docs 20 --max-questions 30

# Một strategy cụ thể
python scripts/benchmark.py --strategy HierarchicalParentChild --max-docs 50 --max-questions 50

# Một filter cụ thể
python scripts/benchmark.py --filter NERExact --max-docs 50 --max-questions 50

# Một config cụ thể
python scripts/benchmark.py --config "HierarchicalParentChild_400_0__NERExact"

# Full benchmark (all 50 configs)
rm -f results/benchmark_results.csv   # xóa kết quả cũ trước
python scripts/benchmark.py

# Background run
nohup python scripts/benchmark.py > results/bench.log 2>&1 &
tail -f results/bench.log
```

### 9.4 Xem kết quả

```bash
# Xem CSV
column -t -s',' results/benchmark_results.csv | less -S

# Đếm số configs đã chạy
tail -n +2 results/benchmark_results.csv | wc -l

# Xem storage Qdrant
du -sh data/qdrant_storage/
```

---

## 10. Cấu trúc project

```
rag-bench-v4/
├── configs/
│   └── settings.py              # Tham số + sinh 50 configs tự động
├── src/
│   ├── ingestion/
│   │   ├── loader.py            # SQuAD 1.1 loader (HuggingFace datasets)
│   │   ├── chunker.py           # 5 chunking strategies
│   │   ├── embedder.py          # all-MiniLM-L6-v2 via sentence-transformers
│   │   └── vector_store.py      # Qdrant embedded, unnamed vector API
│   ├── filtering/
│   │   ├── filters.py           # NoFilter, ExactNorm, MinHashLSH, Similarity, NERExact
│   │   └── pipeline.py          # FilteringPipeline — sequential composition
│   ├── retrieval/
│   │   └── retriever.py         # Dense cosine retrieval (Qdrant query_points)
│   ├── evaluation/
│   │   ├── metrics.py           # Precision / Recall / IoU (raw + preprocessed)
│   │   └── generator.py         # Ollama LLM — tuỳ chọn, không dùng cho eval chính
│   └── utils/
│       └── logger.py
├── scripts/
│   └── benchmark.py             # CLI chính: chunk => filter => embed => index => eval
├── data/
│   └── qdrant_storage/          # Qdrant collections (tự tạo khi chạy)
├── results/                     # benchmark_results.csv + per_question_*.csv
├── requirements.txt
└── README.md
```

---

## 11. Output format

### `results/benchmark_results.csv` — 1 row / config

| Column | Mô tả |
|---|---|
| `config_name` | Định danh duy nhất, vd. `HierarchicalParentChild_400_0__NERExact` |
| `strategy` | Tên chunking strategy |
| `chunk_size` | Kích thước chunk (ký tự) |
| `overlap` | Overlap giữa các chunk |
| `filter_method` | Tên filter được áp dụng |
| `chunk_count_before_filter` | Số chunks sau khi chunk tài liệu |
| `chunk_count_after_filter` | Số chunks sau khi filter |
| `filter_reduction_pct` | % chunks bị loại bỏ |
| `ingest_time_s` | Tổng thời gian chunk + filter + embed + upsert (giây) |
| `storage_mb` | Dung lượng collection Qdrant (MB, numeric) |
| `storage_du` | Dung lượng collection Qdrant (`du -sh` output) |
| `precision_raw` / `recall_raw` / `iou_raw` | Token metrics — raw mode |
| `precision_pre` / `recall_pre` / `iou_pre` | Token metrics — preprocessed mode |
| `avg_retrieval_ms` | Latency retrieval trung bình (ms/query) |
| `n_questions` | Số câu hỏi được đánh giá |

### `results/per_question_{config_name}.csv` — 1 row / question

Chứa Precision, Recall, IoU cho từng câu hỏi riêng lẻ — dùng để phân tích phân phối và outlier.

---

## 12. Tham khảo

- Berdyugina, D., Cohen, A., & Rioual, Y. (2026). *Reducing Redundancy in Retrieval-Augmented Generation through Chunk Filtering*. arXiv:2604.24334.
- Lewis, P. et al. (2021). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. arXiv:2005.11401.
- Anthropic. (2024). *Introducing Contextual Retrieval*.
- Rajpurkar, P. et al. (2016). *SQuAD: 100,000+ Questions for Machine Comprehension of Text*. EMNLP 2016.
- Reimers, N. & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. arXiv:1908.10084.
