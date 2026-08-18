"""What a node can prove about its own coverage, and how a parent composes it.

A coverage certificate answers one question and refuses the neighbouring one:

    it shows   every unit was sent, came back with a well-formed value, and
               belongs to a traceable computation over *this* text
    it does not show that any of those values is semantically right

Keeping those apart is the whole point. A model can be asked whether it read
everything and will say yes; this is derived from the run instead — from spans
the scheduler verified against the bytes it sent, and from children whose own
certificates say the same about their slices.

The pieces already existed and were scattered: `provenance` on subcall events,
`spans_of` reading them back, `delegated_span` for what a child covered, and
`semantic_result` for a node's own sweep. What was missing is composition — a
parent that delegates everything made no subcalls of its own and could say
nothing, even when each child could say everything about its part.

Composition is deliberately strict. A parent's certificate is complete only if
the union of what it and its children established covers its units with no gap.
"Most of it" is not a coverage claim, and neither is "each child said it was
fine": a child's word about *its* part is evidence about that part, and the
parent still has to account for the bytes between the parts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from alchemist_rlm.tracing import digest


@dataclass
class Certificate:
    """One node's account of what it established over its own text."""

    # Which text this is about. A span means nothing without it, and a child's
    # offsets run from zero over its own slice.
    source_digest: str
    source_chars: int
    # How the text was cut into units. Two runs that unitised differently
    # cannot have their coverage compared, however similar the numbers look.
    unitization_digest: str = ""
    total_units: int = 0
    # What the schema demanded of each value, if this node ran a typed sweep.
    schema_digest: str = ""
    # Units this node established directly, and the context spans behind them.
    covered_units: list[int] = field(default_factory=list)
    covered_spans: list[list[int]] = field(default_factory=list)
    failed_units: list[int] = field(default_factory=list)
    # What children established, at their own level and in their own frame.
    children: list["Certificate"] = field(default_factory=list)
    # Where this certificate's text begins inside its parent's, or None if the
    # parent could not locate it. None is not zero: a child whose part was
    # never found in the parent's text contributes nothing, and saying so is
    # the difference between a gap and a silent overlap.
    placed_at: int | None = None

    @property
    def spans(self) -> list[tuple[int, int]]:
        """Every span this node accounts for, in *this* node's coordinates.

        A child's spans run from zero over its own slice, so they are
        translated by where the parent located that slice and dropped when it
        could not. An earlier version unioned them raw on the strength of a
        comment claiming they already shared the parent's frame — nothing
        enforced that, and two 500-character children each reporting (0, 500)
        would have been credited as one half covered twice. This is the same
        frame confusion `spans_of` refuses with `provenance_of`, and it has no
        more business here than it had there.
        """
        out = [(int(a), int(b)) for a, b in self.covered_spans]
        for child in self.children:
            if child.placed_at is None:
                continue
            out.extend((a + child.placed_at, b + child.placed_at)
                       for a, b in child.spans)
        return out

    @property
    def unplaced_children(self) -> int:
        return sum(1 for child in self.children if child.placed_at is None)

    def covers(self, threshold: float = 0.98) -> bool:
        """Is this node's text accounted for, with nothing left failing?

        Two conditions, because coverage of the bytes is not the same claim as
        every unit having produced a value. A certificate once returned True
        holding `failed_units: [1]`, on the strength of spans alone — which is
        precisely the "we sent it, so it counts" reading this whole object
        exists to refuse.

        The byte fraction is a fraction rather than an exact match because
        unitisation leaves the separators between units outside every unit, and
        calling a sweep incomplete over a blank line would be pedantry. It is a
        fraction of bytes, not of units, so a missing record cannot hide in it.
        """
        if self.failed_units or self.unplaced_children:
            return False
        if any(not child.covers(threshold) for child in self.children):
            return False
        if not self.source_chars:
            return False
        merged: list[list[int]] = []
        for start, end in sorted(self.spans):
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        reached = sum(end - start for start, end in merged)
        return reached / self.source_chars >= threshold

    def gaps(self) -> list[tuple[int, int]]:
        """The stretches of this node's text nothing accounted for. Named, so a
        partial sweep can be argued with rather than merely disbelieved."""
        merged: list[list[int]] = []
        for start, end in sorted(self.spans):
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        out, cursor = [], 0
        for start, end in merged:
            if start > cursor:
                out.append((cursor, start))
            cursor = max(cursor, end)
        if cursor < self.source_chars:
            out.append((cursor, self.source_chars))
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_digest": self.source_digest,
            "source_chars": self.source_chars,
            "unitization_digest": self.unitization_digest,
            "total_units": self.total_units,
            "schema_digest": self.schema_digest,
            "covered_units": len(self.covered_units),
            "covered_spans": len(self.covered_spans),
            "failed_units": self.failed_units,
            "placed_at": self.placed_at,
            "unplaced_children": self.unplaced_children,
            "children": [child.to_dict() for child in self.children],
            "complete": self.covers(),
            "gaps": [list(gap) for gap in self.gaps()],
            "means": ("every accounted unit was sent and answered in the "
                      "declared shape; it says nothing about whether any answer "
                      "is semantically correct"),
        }


def unitization_digest(spans: list[tuple[int, int]]) -> str:
    """An identity for how a text was cut. Two runs whose coverage numbers look
    alike but whose units differ are not comparable, and this is what says so."""
    return digest(";".join(f"{a}-{b}" for a, b in spans))


def place(child: Certificate, parent_context: str, part: str,
          span: tuple[int, int] | None = None) -> Certificate:
    """Fix where a child's slice sits in its parent's text, or leave it unplaced.

    Two ways, and the first is better. If the parent knows the span it
    delegated, it passes it and the bytes are checked: `parent_context[a:b]`
    must be the part. That is unambiguous however often the text repeats.

    Without a span, placement is by content — and only when the content occurs
    **once**. `locate` returns the first hash-verified match, which is the right
    answer when there is one and a coin flip when there are two: a fragment
    appearing twice in the parent would put the child at whichever came first
    and translate every one of its spans onto bytes it never read. Zero matches
    and several matches are the same answer here, `None`, because "I do not
    know where this went" is what both of them mean.
    """
    from alchemist_rlm.tracing import digest as _digest

    text = parent_context or ""
    if span is not None:
        start, end = int(span[0]), int(span[1])
        placed = (0 <= start < end <= len(text)
                  and _digest(text[start:end]) == _digest(part))
        child.placed_at = start if placed else None
        return child

    occurrences = []
    at = text.find(part)
    while at != -1 and len(occurrences) < 2:
        occurrences.append(at)
        at = text.find(part, at + 1)
    child.placed_at = occurrences[0] if len(occurrences) == 1 else None
    return child


def from_run(*, context: str, spans: list[tuple[int, int]],
             result: dict[str, Any] | None,
             covered_spans: list[tuple[int, int]],
             children: list[Certificate] | None = None) -> Certificate:
    """Assemble a node's certificate from what its run already recorded.

    Nothing here is asked of the model. `result` is the harness's own typed
    sweep summary, `covered_spans` are spans the scheduler verified against the
    bytes it sent, and `children` are certificates their own nodes built the
    same way.
    """
    summary = result or {}
    failed = list(summary.get("failed_items") or [])
    schema = summary.get("schema")
    return Certificate(
        source_digest=digest(context or ""),
        source_chars=len(context or ""),
        unitization_digest=unitization_digest(spans),
        total_units=len(spans),
        schema_digest=digest(repr(schema)) if schema else "",
        covered_units=sorted(int(i) for i in range(len(spans)) if i not in set(failed)
                             ) if summary else [],
        covered_spans=[[int(a), int(b)] for a, b in covered_spans],
        failed_units=failed,
        children=list(children or []),
    )
