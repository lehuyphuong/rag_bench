"""
Evaluation metrics following Berdyugina et al. (arXiv:2604.24334) Section 3.6.

Token-based evaluation:
  Given a question, the reference passage (ground-truth answer context) and
  the retrieved top-k chunks, we compute:
    - Precision = |Te ∩ Tr| / |Tr|
    - Recall    = |Te ∩ Tr| / |Te|
    - IoU       = |Te ∩ Tr| / |Te ∪ Tr|   (Jaccard similarity)

  where Te = token set of reference passage
        Tr = token set of union of retrieved chunks

Two modes (following paper Section 3.6):
  raw          : lowercase tokenization only
  preprocessed : remove English stopwords + lemmatize (spaCy)

Oracle upper bound (Section 3.6, eq. S*_k):
  Greedy selection of up to k chunks that maximizes reference token coverage.
  Measures the best achievable recall given the available chunks —
  separates retrieval model limitations from fragmentation/filtering effects.

Generation accuracy:
  Exact Match (EM): 1 if any reference answer string is a substring of
                    the generated answer (case-insensitive, stripped).
  Token F1        : standard SQuAD F1 between generated answer and
                    best-matching reference answer.
"""

from __future__ import annotations

import re
import string
from functools import lru_cache
from typing import Callable

# ── Tokenizers ───────────────────────────────────────────────────────────────

def _raw_tokens(text: str) -> set[str]:
    """Lowercase word tokens."""
    return set(re.findall(r"\b\w+\b", text.lower()))


@lru_cache(maxsize=1)
def _get_spacy():
    """Lazy-load spaCy model for lemmatization + stopword removal."""
    import spacy
    try:
        return spacy.load("en_core_web_sm", disable=["parser", "ner"])
    except OSError:
        raise RuntimeError(
            "spaCy model not found. Run: python -m spacy download en_core_web_sm"
        )


def _preprocessed_tokens(text: str) -> set[str]:
    """Lowercase, remove stopwords, lemmatize."""
    nlp = _get_spacy()
    doc = nlp(text.lower())
    return {
        token.lemma_
        for token in doc
        if not token.is_stop and not token.is_punct and token.lemma_.strip()
    }


def get_tokenizer(mode: str) -> Callable[[str], set[str]]:
    if mode == "raw":
        return _raw_tokens
    elif mode == "preprocessed":
        return _preprocessed_tokens
    else:
        raise ValueError(f"Unknown eval mode: {mode}. Choose 'raw' or 'preprocessed'.")


# ── Retrieval metrics ─────────────────────────────────────────────────────────

def compute_retrieval_metrics(
    reference_text: str,
    retrieved_chunks: list[dict],
    mode: str = "raw",
) -> dict[str, float]:
    """
    Compute token-based Precision, Recall, IoU for one query.

    Args:
        reference_text  : the ground-truth passage (SQuAD context that
                          contains the answer).
        retrieved_chunks: list of chunk dicts with "text" field.
        mode            : "raw" or "preprocessed".

    Returns dict with keys: precision, recall, iou.
    """
    tokenize = get_tokenizer(mode)

    te = tokenize(reference_text)
    if not te:
        return {"precision": 0.0, "recall": 0.0, "iou": 0.0}

    tr: set[str] = set()
    for chunk in retrieved_chunks:
        tr |= tokenize(chunk["text"])

    intersection = te & tr
    union = te | tr

    precision = len(intersection) / len(tr) if tr else 0.0
    recall = len(intersection) / len(te)
    iou = len(intersection) / len(union) if union else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "iou": round(iou, 4),
    }


# ── Oracle upper bound ────────────────────────────────────────────────────────

def compute_oracle(
    reference_text: str,
    all_chunks: list[dict],
    k: int = 5,
    mode: str = "raw",
) -> dict[str, float]:
    """
    Greedy oracle: select up to k chunks maximising coverage of reference tokens.

    Oracle score is an upper bound on retrieval recall — it shows how much
    information is *available* in the chunk index, regardless of retrieval
    model quality.

    Returns dict with keys: oracle_precision, oracle_recall, oracle_iou.
    """
    tokenize = get_tokenizer(mode)
    te = tokenize(reference_text)
    if not te:
        return {"oracle_precision": 0.0, "oracle_recall": 0.0, "oracle_iou": 0.0}

    selected_tokens: set[str] = set()
    for _ in range(k):
        best_chunk = None
        best_gain = -1
        for chunk in all_chunks:
            ct = tokenize(chunk["text"])
            gain = len((selected_tokens | ct) & te) - len(selected_tokens & te)
            if gain > best_gain:
                best_gain = gain
                best_chunk = chunk
        if best_chunk is None or best_gain == 0:
            break
        selected_tokens |= tokenize(best_chunk["text"])
        all_chunks = [c for c in all_chunks if c["chunk_id"] != best_chunk["chunk_id"]]

    union = te | selected_tokens
    intersection = te & selected_tokens

    precision = len(intersection) / len(selected_tokens) if selected_tokens else 0.0
    recall = len(intersection) / len(te)
    iou = len(intersection) / len(union) if union else 0.0

    return {
        "oracle_precision": round(precision, 4),
        "oracle_recall": round(recall, 4),
        "oracle_iou": round(iou, 4),
    }


# ── Generation accuracy ───────────────────────────────────────────────────────

def _normalize_answer(text: str) -> str:
    """Lower, strip punctuation and articles (standard SQuAD normalization)."""
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(c for c in text if c not in string.punctuation)
    return " ".join(text.split())


def exact_match(generated: str, reference_answers: list[str]) -> float:
    """1.0 if generated answer matches any reference answer (normalized)."""
    gen = _normalize_answer(generated)
    for ref in reference_answers:
        if gen == _normalize_answer(ref):
            return 1.0
    return 0.0


def token_f1(generated: str, reference_answers: list[str]) -> float:
    """
    Token-level F1 against the best-matching reference answer.
    Standard SQuAD evaluation metric.
    """
    gen_tokens = _normalize_answer(generated).split()
    best_f1 = 0.0
    for ref in reference_answers:
        ref_tokens = _normalize_answer(ref).split()
        common = Counter(gen_tokens) & Counter(ref_tokens)
        n_common = sum(common.values())
        if n_common == 0:
            continue
        precision = n_common / len(gen_tokens) if gen_tokens else 0.0
        recall = n_common / len(ref_tokens) if ref_tokens else 0.0
        f1 = 2 * precision * recall / (precision + recall)
        best_f1 = max(best_f1, f1)
    return round(best_f1, 4)


# Counter needed for token_f1
from collections import Counter
