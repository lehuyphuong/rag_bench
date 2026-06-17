"""
FilteringPipeline — applies a sequence of filter steps to a list of chunks.

Usage:
    pipeline = FilteringPipeline(steps=[{"method": "NERExact"}])
    filtered = pipeline.run(chunks, embed_fn=embed_texts)

    pipeline = FilteringPipeline(steps=[])   # or [{"method": "NoFilter"}]
    filtered = pipeline.run(chunks)          # pass-through
"""

from __future__ import annotations

import logging
import time
from typing import Callable

from src.filtering.filters import get_filter

logger = logging.getLogger(__name__)


class FilteringPipeline:
    """
    Composes and runs a sequence of filtering steps sequentially.

    Args:
        steps: list of step config dicts, e.g.
               [{"method": "Similarity", "threshold": 0.8}]
               [{"method": "NERExact"}]
               [] or [{"method": "NoFilter"}] → pass-through
    """

    def __init__(self, steps: list[dict]) -> None:
        self.steps = steps
        self._filters = []

        # Normalize: empty list = NoFilter
        if not steps:
            steps = [{"method": "NoFilter"}]

        for step in steps:
            method = step["method"]
            params = {k: v for k, v in step.items() if k != "method"}
            self._filters.append(get_filter(method, **params))

    @property
    def tag(self) -> str:
        """Human-readable tag used in collection names and CSV rows."""
        if not self.steps:
            return "NoFilter"
        parts = []
        for step in self.steps:
            name = step["method"]
            thresh = step.get("threshold", "")
            parts.append(f"{name}{thresh}" if thresh else name)
        return "_".join(parts)

    def run(
        self,
        chunks: list[dict],
        embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
    ) -> list[dict]:
        """
        Apply all filter steps sequentially.

        Args:
            chunks  : list of chunk dicts from chunker.py
            embed_fn: dense embedding function (needed by Similarity filter).

        Returns filtered list of chunk dicts.
        """
        current = chunks
        original_count = len(chunks)

        for f in self._filters:
            t0 = time.perf_counter()
            current = f.apply(current, embed_fn=embed_fn)
            elapsed = time.perf_counter() - t0
            logger.info(
                "  [filter %s] %d → %d chunks (%.2fs)",
                f.__class__.__name__, original_count, len(current), elapsed,
            )
            original_count = len(current)

        return current
