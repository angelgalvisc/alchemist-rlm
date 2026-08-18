"""Score a completed sweep against the corpus's own ground truth, per record.

A sweep reports a count. The task compares that count to a number. Between
those two there is a question neither answers: *which* records did it pick, and
were they the right ones? A count of 154 against a truth of 146 is eight over —
but eight over can be 146 right plus 8 wrong, or 154 wrong and 146 missed, and
the task score is identical either way.

This decides it offline, from the frozen corpus and the run's own
`positive_ids`. It exists because the alternative is asserting where the error
lives, and this project has been wrong every time it did that.

    ./.venv/bin/python scripts/audit_sweep_run.py [run_dir]
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from alchemist_rlm import corpus_v2                                   # noqa: E402
from alchemist_rlm.context.segmenter import units as units_of         # noqa: E402
from alchemist_rlm.suite_v2 import TASKS_V2_BY_ID                     # noqa: E402

TASK = "t08v3_semantic_sweep_neutral"


def records(context: str) -> list[str]:
    return [chunk for chunk in re.split(r"(?==== Record \d+ ===)", context)
            if chunk.strip().startswith("=== Record")]


def truth_ids(chunks: list[str]) -> set[int]:
    """Records whose note carries one of the corpus's stoppage phrases.

    Derived from the text, never from an index rule. A first attempt at this
    guessed the positives were every eleventh record; they are every eleventh
    starting at four, so the guess produced a set disjoint from the truth and
    reported zero true positives out of 154 — a result alarming enough to check,
    and wrong entirely in the checking.
    """
    phrases = [phrase.lower() for phrase in corpus_v2.STOPPAGES]
    return {index for index, chunk in enumerate(chunks)
            if any(phrase in chunk.lower() for phrase in phrases)}


def main() -> int:
    context = TASKS_V2_BY_ID[TASK].context
    chunks = records(context)
    truth = truth_ids(chunks)

    # A per-record verdict is only meaningful if a unit IS a record.
    spans = units_of(context)
    aligned = sum(1 for index, (start, end) in enumerate(spans)
                  if index < len(chunks)
                  and context[start:end].strip().startswith(f"=== Record {index:04d}"))
    print(f"units aligned to records: {aligned}/{len(spans)}")
    if aligned != len(spans):
        print("  units and records do not correspond; per-record scoring below "
              "would be comparing different things")

    run_dir = (sys.argv[1] if len(sys.argv) > 1
               else sorted(glob.glob(str(REPO / "runs" / f"*{TASK}*")),
                           key=os.path.getmtime)[-1])
    sweep = json.loads(
        (Path(run_dir) / "episode.json").read_text()).get("semantic_result") or {}
    positives = set(sweep.get("positive_ids") or [])
    if not positives:
        print(f"{run_dir}: no positive_ids recorded")
        return 1

    hit = positives & truth
    precision = len(hit) / len(positives)
    recall = len(hit) / len(truth)
    f1 = 2 * precision * recall / (precision + recall) if hit else 0.0
    print(f"\n{os.path.basename(run_dir)}")
    print(f"  answered {len(positives)} against a truth of {len(truth)}")
    print(f"  true positives {len(hit)} | false positives {len(positives - truth)} "
          f"| false negatives {len(truth - positives)}")
    print(f"  precision {precision:.3f}  recall {recall:.3f}  F1 {f1:.3f}")

    print("\n  what the false positives say:")
    for index in sorted(positives - truth)[:5]:
        note = next((line for line in chunks[index].splitlines()
                     if line.startswith("Note:")), chunks[index])
        print(f"    {note[:100]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
