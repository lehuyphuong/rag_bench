"""
Load SQuAD 1.1 from HuggingFace datasets.

SQuAD 1.1 structure:
  - context : Wikipedia passage (the "document")
  - question: natural language question
  - answers : dict with "text" (list of answer strings) and "answer_start"

We treat each unique context passage as a "document" to be chunked and
indexed. Questions are used only at evaluation time.

Yields:
  documents : list of {"doc_id": str, "text": str, "title": str}
  qa_pairs  : list of {"question": str, "answers": list[str], "doc_id": str}
"""

from __future__ import annotations

import hashlib
import logging
from typing import Generator

from datasets import load_dataset

from configs.settings import (
    DATASET_NAME,
    DATASET_SPLIT,
    MAX_DOCUMENTS,
    MAX_EVAL_QUESTIONS,
)

logger = logging.getLogger(__name__)


def _doc_id(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:12]


def load_squad() -> tuple[list[dict], list[dict]]:
    """
    Returns (documents, qa_pairs).

    documents : unique context passages, deduplicated by content hash.
    qa_pairs  : question + ground-truth answers + doc_id reference.
    """
    logger.info("Loading %s / %s ...", DATASET_NAME, DATASET_SPLIT)
    ds = load_dataset(DATASET_NAME, split=DATASET_SPLIT)

    seen_docs: dict[str, dict] = {}   # doc_id → document
    qa_pairs: list[dict] = []

    for row in ds:
        context = row["context"].strip()
        did = _doc_id(context)

        if did not in seen_docs:
            seen_docs[did] = {
                "doc_id": did,
                "title": row["title"],
                "text": context,
            }

        # Each row has one question + one or more answer strings
        answers = [a for a in row["answers"]["text"] if a.strip()]
        if answers:
            qa_pairs.append({
                "question": row["question"].strip(),
                "answers": answers,
                "doc_id": did,
                "title": row["title"],
            })

    documents = list(seen_docs.values())

    if MAX_DOCUMENTS is not None:
        documents = documents[:MAX_DOCUMENTS]
        # keep only QA pairs whose doc is in the kept document set
        kept_ids = {d["doc_id"] for d in documents}
        qa_pairs = [q for q in qa_pairs if q["doc_id"] in kept_ids]

    if MAX_EVAL_QUESTIONS is not None:
        qa_pairs = qa_pairs[:MAX_EVAL_QUESTIONS]

    logger.info(
        "Loaded %d unique documents, %d QA pairs.", len(documents), len(qa_pairs)
    )
    return documents, qa_pairs
