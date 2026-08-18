"""The trace, which is the only thing that makes an episode adjudicable.

Two lessons from this repository are built into the event shape.

**Read the right field.** A trajectory reader here once looked for `output` on a
result whose real key was `stdout`, saw empty strings on the decisive turns, and
nearly lost the finding that the sub-LM had fabricated its counts. Every writer
of a trace is also its reader, so the keys are declared once, here.

**Record the source, not just the answer.** The plan scores atomic failure from
the same episode: if a subcall *received* the chunk containing the evidence and
still answered wrong, the atom failed; if no subcall ever received it, retrieval
failed. That adjudication is only possible because `subcall` events carry the
source text (hashed and previewed) alongside the response. Drop that and the
episode can still be scored right-or-wrong, but never explained.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any

PREVIEW_CHARS = 600


def digest(text: str) -> str:
    """A short content hash. Short because it is written to every event and read
    by eye; sixteen hex characters is far past collision risk at this scale and
    still fits on one line of a trace."""
    return hashlib.sha256((text or "").encode()).hexdigest()[:16]


def preview(text: str, limit: int = PREVIEW_CHARS) -> str:
    """A readable excerpt that says how much it left out. The omitted count is
    part of the string on purpose: a preview that ends silently reads like a
    complete value, and this repository lost a finding that way once."""
    text = text or ""
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n...[{len(text) - limit} more chars]"


EDGE_CHARS = 240


def sourced(text: str, provenance: Any = None,
            provenance_of: str | None = None) -> dict[str, Any]:
    """A field too big to store whole that must still be locatable exactly.

    A preview alone is not enough. The plan adjudicates atomic failure by asking
    whether a subcall *received* the passage holding the evidence, and whether
    the subcalls between them covered the context — neither question can be
    answered from the first 600 characters of each source. Storing the head, the
    tail and the length lets `locate()` recover the exact span from the original
    context, and the sha256 proves the recovered span is the same bytes.

    `provenance` covers the case hash recovery cannot: a source the harness
    *built* out of context spans rather than sliced from them. It exists because
    it was measured missing. `semantic_search` renders each fragment as
    "[item N]\\n<text>" joined by blank lines, so from v5 onward no subcall
    source was a contiguous slice any more, `locate()` correctly refused to
    guess, and every run since reported `coverage: 0.0` while its own
    `semantic_result` credited 1,600 of 1,600 items examined. The metric never
    started lying; the evidence it reads stopped being emitted.
    """
    text = text or ""
    record = {
        "chars": len(text),
        "sha256": digest(text),
        "head": text[:EDGE_CHARS],
        "tail": text[-EDGE_CHARS:] if len(text) > EDGE_CHARS else "",
        "preview": preview(text),
    }
    if provenance:
        # Offsets are meaningless without the text they index. A child's spans
        # run from 0 over its own part, and every subcall in a recursive run
        # shares one trace file, so a reader measuring the root's coverage would
        # otherwise credit itself with the children's offsets read against the
        # wrong string.
        record["provenance"] = [[int(start), int(end)] for start, end in provenance]
        record["provenance_of"] = provenance_of
    return record


def locate(context: str, record: dict[str, Any]) -> tuple[int, int] | None:
    """Where in `context` this recorded source came from, or None if it was not
    a contiguous slice of it (the model may have built a source in Python).

    Verified by hash, not by the head matching: a repeated header would find the
    wrong occurrence, and a wrong span would silently corrupt every coverage and
    attribution number computed from it.
    """
    head, chars, want = record.get("head") or "", record.get("chars") or 0, record.get("sha256")
    if not head or not chars or not want:
        return None
    start = context.find(head)
    while start != -1:
        end = start + chars
        if end <= len(context) and digest(context[start:end]) == want:
            return start, end
        start = context.find(head, start + 1)
    return None


def spans_of(context: str, source: dict[str, Any]) -> list[tuple[int, int]]:
    """Every context span this source demonstrably reached.

    Recorded provenance wins, because it is exact and a rendered source has no
    hash to find. It is not taken on trust: the scheduler only records a span
    after checking that `context[start:end]` really is inside the text it sent,
    so a span here means those bytes were in front of a sub-model. Anything
    without provenance falls back to hash recovery, which is still the right
    answer for a slice the model cut itself in Python.
    """
    claimed = source.get("provenance") or []
    if claimed and source.get("provenance_of") == digest(context):
        return [(int(a), int(b)) for a, b in claimed
                if 0 <= int(a) < int(b) <= len(context)]
    found = locate(context, source)
    return [found] if found else []


def covered(context: str, spans: list[tuple[int, int]]) -> float:
    """Fraction of the context that at least one span reached."""
    if not context:
        return 0.0
    merged: list[list[int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return sum(end - start for start, end in merged) / len(context)


class Trace:
    """Append-only JSONL. Thread-safe: batched subcalls write concurrently."""

    def __init__(self, path: Path | str, *, run_id: str, meta: dict[str, Any] | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self._lock = threading.Lock()
        self._seq = 0
        self._started = time.monotonic()
        # A run id is an evidence identity, not a log category.  Appending a
        # second episode under the same id produced two run_start/run_end pairs
        # while episode.json described only the latter.  Refuse the collision
        # before any evidence is written.
        self._handle = self.path.open("x", encoding="utf-8")
        self.event("run_start", meta=meta or {})

    def event(self, kind: str, **fields: Any) -> int:
        """Append one record and return its sequence number.

        Every typed event below funnels through here, so ordering, timing and
        the run id are written in exactly one place. The lock is not decorative:
        batched subcalls write from several threads at once.
        """
        with self._lock:
            self._seq += 1
            seq = self._seq
            record = {
                "seq": seq,
                "t": round(time.monotonic() - self._started, 3),
                "run_id": self.run_id,
                "kind": kind,
                **fields,
            }
            self._handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            self._handle.flush()
            return seq

    # --- typed events ------------------------------------------------------
    def model_turn(
        self,
        *,
        depth: int,
        turn: int,
        content: str,
        reasoning: str | None,
        raw_tool_calls: Any,
        served_model: str | None,
        usage: dict[str, Any] | None,
    ) -> int:
        """`reasoning` is separate on purpose: `mlx_lm.server` emits it in its own
        field and the official client reads only `content`, so a harness that
        does not ask for it never sees why the model chose what it chose."""
        return self.event(
            "model_turn",
            depth=depth,
            turn=turn,
            content=content,
            reasoning=reasoning,
            raw_tool_calls=raw_tool_calls,
            served_model=served_model,
            usage=usage,
        )

    def tool_call(self, *, depth: int, turn: int, name: str, code: str, code_key: str) -> int:
        """The code the model asked to run, with the key the duplicate check
        uses. Both are stored: `code` is what a reader needs, `code_key` is what
        made the harness refuse a repeat, and disagreements between them are how
        a refusal bug becomes visible."""
        return self.event(
            "tool_call", depth=depth, turn=turn, name=name, code=code, code_key=code_key
        )

    def observation(self, *, depth: int, turn: int, observation: dict[str, Any]) -> int:
        """What the REPL gave back, stored whole. A trajectory reader here once
        looked for `output` on a result whose real key was `stdout` and saw
        empty strings on the decisive turns, so nothing is flattened on the way
        in — the reader is the one who has to name the right field."""
        return self.event("observation", depth=depth, turn=turn, observation=observation)

    def subcall(
        self,
        *,
        depth: int,
        parent_turn: int,
        index: int,
        instruction: str,
        source: str,
        response: str,
        reasoning: str | None = None,
        error: str | None = None,
        source_ref: str | None = None,
        provenance: Any = None,
        provenance_of: str | None = None,
    ) -> int:
        """One call to a sub-model, recorded with what it was given.

        This is the event atomic attribution rests on. Knowing a subcall
        answered wrong is worth little; knowing it answered wrong *while holding
        the passage with the evidence* separates the sub-model's own ceiling
        from a retrieval failure, and that separation is the whole method. So
        the source travels here hashed, previewed and — when the harness built
        it rather than sliced it — with the context spans it came from.
        """
        return self.event(
            "subcall",
            depth=depth,
            parent_turn=parent_turn,
            index=index,
            instruction=instruction,
            source=sourced(source, provenance, provenance_of),
            source_ref=source_ref,
            response=response,
            reasoning=reasoning,
            error=error,
        )

    def artifact(self, *, name: str, ref: str, chars: int, sha256: str) -> int:
        """A value parked on disk rather than carried through the window. The
        size and hash are recorded, never the value: an artifact exists exactly
        because it was too large to belong in the conversation."""
        return self.event("artifact", name=name, ref=ref, chars=chars, sha256=sha256)

    def finish(
        self,
        *,
        reason: str,
        answer: str | None,
        ledger: dict[str, Any],
        visible_transcript_sha256: str = "",
    ) -> int:
        """Close the trace with why it ended and what it spent.

        `reason` is the four-way termination — an answer tag, a submitted value,
        a turn with no tool call, or a budget stop — and it is written even when
        there is no answer, because "ended with nothing" is a result too.
        """
        seq = self.event(
            "run_end",
            reason=reason,
            answer=answer,
            ledger=ledger,
            visible_transcript_sha256=visible_transcript_sha256,
        )
        with self._lock:
            self._handle.close()
        return seq

    # --- reading back ------------------------------------------------------
    @staticmethod
    def read(path: Path | str) -> list[dict[str, Any]]:
        """Load a finished trace as a list of events, in order. Blank lines are
        skipped so a trace whose run was killed mid-write still reads."""
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]
