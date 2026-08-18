"""Two search modes, and a result that is never silently empty.

`probe_11` spent 23 of 25 turns on regex, ten of them re-running searches that
had already returned nothing, and three iterations returned literally empty
output. An empty result and a result that says *"0 matches for this pattern
across all 200,565 characters"* are the same fact and completely different
observations: the second one closes a hypothesis, the first one invites a
re-run. Every function here returns a count and a scope, always.

The semantic path exists because the same probe never once delegated: it kept
grepping for a name that was never written literally. When a lexical search
comes back empty the observation names the other mode explicitly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

MAX_HITS = 40
WINDOW = 160


@dataclass(frozen=True)
class Hit:
    """One literal match: where it is, and enough text to recognise it."""
    ref: str
    line: int
    start: int
    excerpt: str

    def to_dict(self) -> dict[str, Any]:
        """The hit as plain data, for the observation the model reads."""
        return {"ref": self.ref, "line": self.line, "start": self.start, "excerpt": self.excerpt}


def literal_search(
    store: Any,
    pattern: str,
    *,
    regex: bool = False,
    ignore_case: bool = True,
    max_hits: int = MAX_HITS,
) -> dict[str, Any]:
    """Deterministic search. Costs no inference, so it should be tried first."""
    flags = re.I if ignore_case else 0
    try:
        compiled = re.compile(pattern if regex else re.escape(pattern), flags)
    except re.error as error:
        return {"ok": False, "error": f"bad pattern: {error}", "pattern": pattern,
                "next_actions": ["escape the pattern or pass regex=False"]}

    text = store.text
    segments = store.segments()
    hits: list[Hit] = []
    total = 0
    for match in compiled.finditer(text):
        total += 1
        if len(hits) >= max_hits:
            continue
        start = match.start()
        seg = next((s for s in segments if s.start <= start < s.end), None)
        left = max(0, start - WINDOW // 2)
        hits.append(Hit(
            ref=seg.ref if seg else "?",
            line=text.count("\n", 0, start) + 1,
            start=start,
            excerpt=text[left:start + WINDOW].replace("\n", " ⏎ "),
        ))
    result = {
        "ok": True,
        "mode": "literal" if not regex else "regex",
        "pattern": pattern,
        "matches": total,
        "returned": len(hits),
        "searched_chars": len(text),
        "hits": [hit.to_dict() for hit in hits],
        "refs": sorted({hit.ref for hit in hits}),
    }
    if total == 0:
        result["next_actions"] = [
            "the pattern is genuinely absent from the whole context, not merely unfound",
            "try a different surface form: a substring, a different casing, a regex",
            "if the target may be described rather than named, use semantic_search "
            "or llm_query over segments instead",
        ]
    elif total > len(hits):
        result["next_actions"] = [
            f"{total - len(hits)} further matches were not returned; narrow the "
            "pattern or read the segments in `refs` directly",
        ]
    return result
