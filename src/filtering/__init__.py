"""
src/filtering — chunk filtering and deduplication pipeline.

STATUS: placeholder — not yet implemented.
The benchmark currently runs with filtering=[] (no filtering) for all configs.

═══════════════════════════════════════════════════════════════════════════════
PLANNED ARCHITECTURE
═══════════════════════════════════════════════════════════════════════════════

Each filtering config in CHUNKING_CONFIGS["filtering"] is a list of step dicts.
Steps are applied sequentially by FilteringPipeline.run(chunks) → filtered_chunks.

    chunker.py
        │
        ▼  list[chunk]
    FilteringPipeline.run()
        │  step 1 → step 2 → ... → step N
        ▼  list[chunk]  (fewer or merged chunks)
    embedder.py

═══════════════════════════════════════════════════════════════════════════════
PLANNED MODULES
═══════════════════════════════════════════════════════════════════════════════

pipeline.py
    FilteringPipeline(steps: list[dict]) — composes filter steps.

lexical.py  (Berdyugina et al. baseline methods)
    ExactNormFilter     — SHA-256 exact dedup
    MinHashLSHFilter    — near-dup by Jaccard (MinHash approximation)
    RandomFilter        — random drop (control baseline)

semantic.py  (embedding-based)
    SimilarityFilter    — cosine threshold on dense vectors
    SimilarityTopicFilter — same but only within BERTopic clusters

structural.py  (NER-based, Berdyugina et al.)
    NERExactFilter      — drop chunks with identical named-entity set
    NERHalfFilter       — drop chunks with >=50% entity overlap

merge.py  (future research — see analysis notes)
    Key open questions before implementing:
    1. Partial overlap (Type 5) is the only case where merge clearly helps.
       Other types (exact, containment) can be handled by drop.
    2. Merged chunk length may exceed embedding model context window
       → vector quality degrades → retrieval may worsen.
    3. CAG/KV-cache dedup idea: skip insert if embedding already in cache
       (cosine > threshold) — lazy semantic dedup at index time.
    Planned: LCSMerge (difflib, no model), NLIMerge (DeBERTa cross-encoder)

═══════════════════════════════════════════════════════════════════════════════
USAGE (once implemented)
═══════════════════════════════════════════════════════════════════════════════

# No filtering (current baseline)
{"filtering": []}

# NER_Exact (Berdyugina et al.)
{"filtering": [{"method": "NER_Exact"}]}

# Similarity + NER combination
{"filtering": [
    {"method": "Similarity", "threshold": 0.8},
    {"method": "NER_Exact"},
]}

# With merge (future)
{"filtering": [
    {"method": "Similarity", "threshold": 0.8},
    {"method": "LCS_Merge",  "threshold": 0.85},
]}
"""

__all__: list[str] = []
