"""`rlm_query`: a child that is the same harness, not a bigger sub-LM call.

This is the line between a RLM and a retrieval agent. `llm_query` reaches an
LM — the base case, which the paper states plainly: at max depth, the RLM is an
LM. `rlm_query` reaches a *root* that gets its own REPL, can partition its own
slice again, and can spawn its own subcalls. Without it, `search`, `read` and
`delegate` would add up to delegation, not recursion.

Three guards, all of them load-bearing:

  - **Depth.** Bounded at 2 for the first skeleton.
  - **Cycles.** A child asked the same question about the same bytes as one of
    its ancestors would recurse forever inside its budget. The lineage is
    carried down and checked by content, not by identity.
  - **The shared ledger.** The child spends the parent's budget. A fresh budget
    per node would make eight children at depth 2 licence nine times the run.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable


def signature(question: str, context: str) -> str:
    """An identity for (question, context), used to detect a cycle.

    By content, never by object identity: a child handed the same question over
    the same bytes as one of its ancestors would recurse until the budget died,
    and the two would be different objects every time.
    """
    return hashlib.sha256(
        f"{(question or '').strip()}\x00{hashlib.sha256((context or '').encode()).hexdigest()}".encode()
    ).hexdigest()[:16]


class RecursionRefused(RuntimeError):
    """Recursion was declined — depth, node budget, or a cycle in the lineage."""


@dataclass
class RecursiveCaller:
    """Bound into the REPL as `rlm_query`."""

    spawn: Callable[..., Any]            # (question, context, budget, lineage) -> answer str
    budget: Any                          # Budget
    lineage: tuple[str, ...] = ()
    trace: Any = None
    depth: int = 0
    # This node's own context, held to locate each part inside it. Without it a
    # root that delegated all 202,800 of its characters still reported
    # `coverage: 0.0`, because coverage is read from subcall sources and this
    # root made no subcalls — its children did, in their own coordinate frames.
    parent_context: str = ""
    calls: list[dict[str, Any]] = field(default_factory=list)
    # What the children did, rolled up. The batching counters live on each
    # node's own scheduler, so a root that delegated everything reported
    # `batches: 0` next to a tree-wide `subcalls: 50` — one record claiming
    # fifty subcalls, no batch and no sequential call, which cannot all be true.
    # Each child's own figures already include its descendants, so summing them
    # here is correct at every depth.
    subtree: dict[str, int] = field(
        default_factory=lambda: {"batches": 0, "sequential_calls": 0,
                                 "lazy_pulls": 0, "peak_in_flight": 0,
                                 "subcalls": 0}
    )

    def _absorb(self, child: Any) -> None:
        figures = getattr(child, "batching", None) or {}
        for key in ("batches", "sequential_calls", "lazy_pulls"):
            self.subtree[key] += int(figures.get(f"{key}_tree", 0) or 0)
        self.subtree["peak_in_flight"] = max(
            self.subtree["peak_in_flight"], int(figures.get("peak_in_flight_tree", 0) or 0)
        )
        self.subtree["subcalls"] += len(getattr(child, "subcalls", None) or []) + int(
            (getattr(child, "batching", None) or {}).get("subcalls_below", 0) or 0
        )

    def _credit(self, part: str, child: Any) -> None:
        """Record which of this node's bytes a child demonstrably read.

        Two conditions, both evidence rather than assumption: the part has to be
        a hash-verified slice of this context — so the span is exactly where the
        bytes are, not where they look like they are — and the child's own sweep
        has to report that it examined every unit of it. Handing a child a part
        proves nothing about what the child read, which is why delegation alone
        earns nothing here.
        """
        from alchemist_rlm.tracing import digest

        if not self.parent_context or self.trace is None:
            return
        if not (getattr(child, "semantic_result", None) or {}).get(
                "context_coverage_complete"):
            return
        # Content placement is only evidence when it is unique.  The generic
        # ``locate`` helper returns the first hash-verified occurrence, which
        # is useful for retrieval but would guess the frame when two delegated
        # parts have identical text.
        starts = []
        at = self.parent_context.find(part)
        while at != -1 and len(starts) < 2:
            starts.append(at)
            at = self.parent_context.find(part, at + 1)
        if len(starts) != 1:
            return
        span = (starts[0], starts[0] + len(part))
        self.trace.event("delegated_span", depth=self.depth + 1,
                         span=[span[0], span[1]], of=digest(self.parent_context))

    def invoke(self, question: str, context: Any = None, **kwargs: Any) -> Any:
        """Same guards as calling it, returning whatever spawn returns.

        `rlm_map` needs the child's episode — stop reason, its own validated
        coverage — not just the answer string. `status: ok` that only meant
        "the function returned" called a budget-starved child with an empty
        answer a success."""
        return self._guarded(question, context, **kwargs)

    def __call__(self, question: str, context: Any = None, **kwargs: Any) -> str:
        result = self.invoke(question, context, **kwargs)
        answer = getattr(result, "answer", result)
        # A child that produced nothing answers as nothing — not as the
        # four-character string "None" pretending to be text.
        if answer is None:
            return ""
        return answer if isinstance(answer, str) else str(answer)

    def _guarded(self, question: str, context: Any = None, **kwargs: Any) -> Any:
        if context is None or not str(context).strip():
            raise ValueError(
                "rlm_query(question, context) needs an explicit `context`: pass the "
                "slice the child should work on, e.g. rlm_query(q, read_context('s0003'))"
            )
        from alchemist_rlm.protocol import as_text

        text = as_text(context, what="`context` for the child")
        blocked = self.budget.may_recurse()
        if blocked:
            raise RecursionRefused(
                f"cannot recurse further ({blocked}). Use llm_query on this slice "
                "instead, or answer from what you already have."
            )
        mark = signature(question, text)
        if mark in self.lineage:
            raise RecursionRefused(
                "this exact question over this exact text is already being answered "
                "further up the recursion; split the text or change the question"
            )
        child_budget = self.budget.child()
        record = {"depth": self.depth + 1, "signature": mark,
                  "question": question, "chars": len(text)}
        self.calls.append(record)
        if self.trace is not None:
            self.trace.event("rlm_query", **record)
        result = self.spawn(
            question=question, context=text, budget=child_budget,
            lineage=self.lineage + (mark,), **kwargs,
        )
        self._absorb(result)
        self._credit(text, result)
        answer = getattr(result, "answer", result)
        if self.trace is not None:
            self.trace.event("rlm_query_return", signature=mark,
                             depth=self.depth + 1,
                             answer=answer if isinstance(answer, str) else str(answer))
        return result
