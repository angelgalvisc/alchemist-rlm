"""Structure first, then a calibrated band — never a fixed universal size.

The measurements in this repository rule out both extremes:

  - ~79K-character prompts made the 9B fabricate counts: it returned the same
    `1089` six times for a chunk of roughly 800 lines.
  - Thousands of tiny prompts made the server materialise one task per prompt,
    swap, and take 29 minutes without producing an answer.

So a segment is built by finding the real boundaries of the text — documents,
headings, functions, paragraphs, log lines, table rows — and then packing whole
units up to a target, splitting only when a single unit exceeds the maximum.

Coverage is a property of the output, not a hope: the segments of a text tile it
exactly, with no gap and no overlap, which is what lets an episode be scored for
whether the evidence was ever handed to a subcall at all.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterator

MIN_CHARS = 400
TARGET_CHARS = 4_000
MAX_CHARS = 12_000

_DOC = re.compile(r"^(?:-{3,}|={3,}|#{1,2} |<\|doc\||\x0c)", re.M)
_HEADING = re.compile(r"^#{1,6} .*$", re.M)
_DEF = re.compile(r"^(?:def |class |async def )", re.M)
_BLANK = re.compile(r"\n[ \t]*\n")


@dataclass(frozen=True)
class Segment:
    """A named span of the context, cut on a structural edge rather than a count."""
    ref: str
    start: int
    end: int
    kind: str
    unit_count: int = 1

    @property
    def chars(self) -> int:
        """How many characters this segment spans."""
        return self.end - self.start

    def text(self, source: str) -> str:
        """This segment's text, sliced from the source it was computed over."""
        return source[self.start:self.end]

    def to_dict(self) -> dict[str, Any]:
        """The segment as plain data, for manifests and traces."""
        return {"ref": self.ref, "start": self.start, "end": self.end,
                "chars": self.chars, "kind": self.kind, "units": self.unit_count}


def detect_structure(text: str) -> str:
    """Name the finest structure the text actually has. Used for the manifest,
    so the model can see what it is working with before spending a call."""
    sample = text[:200_000]
    if _DEF.search(sample):
        return "code"
    if _HEADING.search(sample):
        return "sections"
    lines = sample.splitlines()
    if len(lines) >= 8:
        delimited = sum(1 for line in lines[:200] if line.count("|") >= 2 or line.count("\t") >= 2)
        if delimited >= min(len(lines), 200) * 0.6:
            return "rows"
    if _BLANK.search(sample):
        return "paragraphs"
    return "lines"


def _boundaries(text: str, structure: str) -> list[int]:
    """Offsets where a unit begins. Always starts at 0, always ends at len."""
    marks = {0, len(text)}
    if structure == "code":
        marks |= {m.start() for m in _DEF.finditer(text)}
        marks |= {m.end() for m in _BLANK.finditer(text)}
    elif structure == "sections":
        marks |= {m.start() for m in _HEADING.finditer(text)}
        marks |= {m.end() for m in _BLANK.finditer(text)}
    elif structure == "paragraphs":
        marks |= {m.end() for m in _BLANK.finditer(text)}
    else:                                      # rows, lines
        offset = 0
        for line in text.splitlines(keepends=True):
            offset += len(line)
            marks.add(offset)
    marks |= {m.start() for m in _DOC.finditer(text)}
    return sorted(mark for mark in marks if 0 <= mark <= len(text))


def _units(text: str, structure: str) -> Iterator[tuple[int, int]]:
    """Spans that tile the text, none of them blank.

    A boundary list marks the end of every line, so a blank line becomes a unit
    holding one newline — and the leaf then answers for it. Symmetric query 4's
    persisted rows carry
    `{"item": 1, "value": "description and abstract concept", "source": ""}`:
    a validated enum value over nothing at all, produced by a runtime whose
    purpose is refusing exactly that. Four of that context's 795 units were
    blank.

    Blank spans are merged into a neighbour, not dropped. Dropping would leave
    characters belonging to no unit, and the certificate tiles the context;
    merging keeps every byte in exactly one unit and removes the empty
    judgement. A blank run at the very start has no previous unit, so it joins
    the following one instead.

    Text that is entirely whitespace yields nothing, as empty text does: there
    is no structural unit in it to judge.
    """
    marks = _boundaries(text, structure)
    spans: list[tuple[int, int]] = []
    for start, end in zip(marks, marks[1:]):
        if end <= start:
            continue
        if spans and not text[start:end].strip():
            spans[-1] = (spans[-1][0], end)        # extend the previous unit
        else:
            spans.append((start, end))
    while len(spans) > 1 and not text[spans[0][0]:spans[0][1]].strip():
        spans[1] = (spans[0][0], spans[1][1])      # a leading blank joins forward
        spans.pop(0)
    if len(spans) == 1 and not text[spans[0][0]:spans[0][1]].strip():
        return
    yield from spans


def segment(
    text: str,
    *,
    target_chars: int = TARGET_CHARS,
    min_chars: int = MIN_CHARS,
    max_chars: int = MAX_CHARS,
    prefix: str = "s",
) -> list[Segment]:
    """Cut `text` into whole structural units packed toward `target_chars`.

    Structure first, size second. Cutting on a character count hands a sub-model
    half a record — the first real recursion run's children began mid-line, with
    'd 0320 ===' and 'work began normally.' — and a fragment that starts mid-item
    cannot be judged item by item. So the bands bound the result; they do not
    drive it.
    """
    if not text:
        return []
    if not (0 < min_chars <= target_chars <= max_chars):
        raise ValueError(
            f"band must satisfy 0 < min <= target <= max, got "
            f"{min_chars} / {target_chars} / {max_chars}"
        )
    structure = detect_structure(text)
    spans: list[tuple[int, int, int]] = []          # start, end, units
    start = end = None
    units = 0
    for unit_start, unit_end in _units(text, structure):
        # A single unit larger than the maximum is cut on the boundary; nothing
        # else can be done without dropping text, and dropping text would break
        # the coverage guarantee this module exists to provide.
        while unit_end - unit_start > max_chars:
            if start is not None:
                spans.append((start, end, units))
                start = end = None
                units = 0
            spans.append((unit_start, unit_start + max_chars, 1))
            unit_start += max_chars
        if start is None:
            start, end, units = unit_start, unit_end, 1
        else:
            end, units = unit_end, units + 1
        if end - start >= target_chars:
            spans.append((start, end, units))
            start = end = None
            units = 0
    if start is not None:
        spans.append((start, end, units))
    # A short tail is folded back rather than shipped as a fragment nobody can
    # judge: the 38%-vs-73% result says undersized prompts cost accuracy.
    if len(spans) > 1 and spans[-1][1] - spans[-1][0] < min_chars:
        last = spans.pop()
        prev = spans.pop()
        spans.append((prev[0], last[1], prev[2] + last[2]))
    return [
        Segment(ref=f"{prefix}{index:04d}", start=s, end=e, kind=structure, unit_count=u)
        for index, (s, e, u) in enumerate(spans)
    ]


def units(text: str) -> list[tuple[int, int]]:
    """The structural units of a text — rows, paragraphs, functions — as spans.

    Public because the per-item decision contract needs them: the harness
    numbers the units itself and demands a decision for each number, so
    validation compares against the harness's own numbering and requires zero
    knowledge of how the corpus happens to label its entries.
    """
    if not text:
        return []
    return list(_units(text, detect_structure(text)))


def grouped_parts(text: str, slots: int, **band: Any) -> list[str]:
    """Split into at most `slots` parts on segment boundaries only.

    Replaces the raw character cut that handed recursion children ragged
    pieces — the first real recursion run's children began with 'd 0320 ==='
    and 'work began normally.' because their parent sliced mid-record. Grouping
    whole segments guarantees every character lands in exactly one part and no
    unit is torn between two children.
    """
    if not text:
        return []
    slots = max(1, slots)
    kwargs = {k: v for k, v in band.items()
              if k in ("target_chars", "min_chars", "max_chars") and v}
    ends = [seg.end for seg in segment(text, **kwargs) if seg.end < len(text)]
    # Cut at the segment edge NEAREST each ideal proportional bound, before or
    # after it. A greedy `>= target` rule under-fills the slot count whenever a
    # segment lands just short: three 4,08x-char segments over three slots
    # produced two parts because 4,080 >= 4,086 is false by six characters.
    cuts: list[int] = []
    previous = 0
    for k in range(1, slots):
        ideal = len(text) * k / slots
        candidates = [e for e in ends if e > previous]
        if not candidates:
            break
        nearest = min(candidates, key=lambda e: abs(e - ideal))
        cuts.append(nearest)
        previous = nearest
    bounds = [0, *cuts, len(text)]
    return [text[a:b] for a, b in zip(bounds, bounds[1:]) if b > a]


def covers(segments: list[Segment], text: str) -> bool:
    """True when the segments tile the text exactly. Asserted in tests."""
    if not text:
        return not segments
    if not segments or segments[0].start != 0 or segments[-1].end != len(text):
        return False
    return all(a.end == b.start for a, b in zip(segments, segments[1:]))
