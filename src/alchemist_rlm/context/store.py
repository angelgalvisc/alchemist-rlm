"""The context, which lives here and never in the message history.

This is the load-bearing claim of a RLM: the model is told *about* its context —
"a `str` of N characters, structured as rows, 46 segments" — and must run code to
see any of it. The manifest exists so the first decision the model makes is an
informed one rather than a guess, and so a run can be audited for whether the
evidence was ever reachable.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from alchemist_rlm.context.segmenter import (
    MAX_CHARS, MIN_CHARS, TARGET_CHARS, Segment, detect_structure, segment,
)


@dataclass
class ContextStore:
    """One text plus its structural view. Segmentation is computed once."""

    text: str
    name: str = "context"
    target_chars: int = TARGET_CHARS
    min_chars: int = MIN_CHARS
    max_chars: int = MAX_CHARS
    _segments: list[Segment] | None = field(default=None, repr=False)

    @property
    def chars(self) -> int:
        """The size of the context. Usually the first thing worth knowing about it."""
        return len(self.text)

    @property
    def sha256(self) -> str:
        """Full hash of the text, so a run can prove which context it was given."""
        return hashlib.sha256(self.text.encode()).hexdigest()

    def segments(self) -> list[Segment]:
        """The structural view, computed once and cached. Segmentation is deterministic
        for a given text and band, so every run over the same context sees the same
        refs — which is what makes coverage comparable across runs.
        """
        if self._segments is None:
            self._segments = segment(
                self.text,
                target_chars=self.target_chars,
                min_chars=self.min_chars,
                max_chars=self.max_chars,
            )
        return self._segments

    def by_ref(self, ref: str) -> Segment | None:
        """The segment with this ref, or None. Refs come back to the harness from the
        model, so an unknown one is an ordinary outcome and not an error.
        """
        return next((s for s in self.segments() if s.ref == ref), None)

    def read(self, ref_or_start: str | int, end: int | None = None) -> str:
        """`read_context("s0007")` or `read_context(1200, 4000)`."""
        if isinstance(ref_or_start, str):
            found = self.by_ref(ref_or_start)
            if found is None:
                raise KeyError(
                    f"no segment {ref_or_start!r}; refs run s0000..s{len(self.segments()) - 1:04d}"
                )
            return found.text(self.text)
        start = max(0, int(ref_or_start))
        stop = self.chars if end is None else min(self.chars, int(end))
        return self.text[start:stop]

    def manifest(self) -> dict[str, Any]:
        """A description of the context small enough to show the model. This is what it
        gets instead of the text: size, segment count and shape. Everything else it
        must go and look at.
        """
        segs = self.segments()
        sizes = [s.chars for s in segs] or [0]
        return {
            "name": self.name,
            "chars": self.chars,
            "lines": self.text.count("\n") + 1 if self.text else 0,
            "sha256": self.sha256[:16],
            "structure": detect_structure(self.text) if self.text else "empty",
            "segments": len(segs),
            "segment_chars": {
                "min": min(sizes), "max": max(sizes),
                "mean": round(sum(sizes) / len(sizes)),
            },
            "band": {"min": self.min_chars, "target": self.target_chars, "max": self.max_chars},
            "first_refs": [s.ref for s in segs[:5]],
            "last_ref": segs[-1].ref if segs else None,
            "head": self.text[:400],
        }

    # `describe()` was here and has been removed. Its docstring called it "the
    # sentence the model actually reads on turn one" and nothing called it:
    # `engine._context_line` is that sentence, and has been for as long as the
    # loop has existed. Two costs, and the dead code was the smaller one.
    #
    # It ended its ref hint at `(refs s0000...)` — first ref, then an ellipsis.
    # That is the exact defect `engine._ref_range` was written to fix: a model
    # shown `s0000` and no upper bound reads the padding as arbitrary and asks
    # for `s00010` on the tenth segment, which is the KeyError on record five
    # times. So this was a fixed bug, still spelled out, one import away from
    # being reintroduced by anyone who believed the docstring and called it.
    #
    # `manifest()` is the interface; `_ref_range` reads `last_ref` from it.
