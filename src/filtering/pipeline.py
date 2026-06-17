"""
FilteringPipeline: applies a sequence of filter/merge steps to a list of chunks.

Currently a pass-through (all filtering methods return [] in CHUNKING_CONFIGS).
The interface is defined here so benchmark.py can wire it in without changes
when filter implementations are added to lexical.py / semantic.py / structural.py.

Usage:
    from src.filtering.pipeline import FilteringPipeline

    pipeline = FilteringPipeline(steps=[])          # no-op (current baseline)
    filtered = pipeline.run(chunks, embed_fn=None)  # returns chunks unchanged
"""

from __future__ import annotations

import logging
from typing import Callable

logger = logging.getLogger(__name__)

# Registry: method name → filter class
# Populated as methods are implemented in lexical.py / semantic.py / structural.py
_REGISTRY: dict[str, type] = {}


def register(name: str):
    """Decorator to register a filter class under a method name."""
    def decorator(cls):
        _REGISTRY[name] = cls
        return cls
    return decorator


class FilteringPipeline:
    """
    Composes and runs a sequence of filtering/merge steps.

    Args:
        steps: list of step config dicts, e.g.
               [{"method": "Similarity", "threshold": 0.8},
                {"method": "NER_Exact"}]
               Empty list = no filtering (pass-through).
    """

    def __init__(self, steps: list[dict]) -> None:
        self.steps = steps
        self._filters = []
        for step in steps:
            method = step.get("method")
            if method not in _REGISTRY:
                raise ValueError(
                    f"Filter method '{method}' is not implemented yet. "
                    f"Available: {list(_REGISTRY.keys())} "
                    f"(see src/filtering/__init__.py for planned methods)"
                )
            params = {k: v for k, v in step.items() if k != "method"}
            self._filters.append(_REGISTRY[method](**params))

    def run(
        self,
        chunks: list[dict],
        embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
    ) -> list[dict]:
        """
        Apply all filter steps sequentially.

        Args:
            chunks  : list of chunk dicts from chunker.py
            embed_fn: dense embedding function (needed by semantic filters).
                      Pass None if no semantic filters are in the pipeline.

        Returns filtered (and possibly merged) list of chunk dicts.
        """
        if not self._filters:
            return chunks   # pass-through: no filtering configured

        original_count = len(chunks)
        for f in self._filters:
            chunks = f.apply(chunks, embed_fn=embed_fn)
            logger.info(
                "  [filter %s] %d → %d chunks",
                f.__class__.__name__, original_count, len(chunks),
            )
            original_count = len(chunks)

        return chunks
