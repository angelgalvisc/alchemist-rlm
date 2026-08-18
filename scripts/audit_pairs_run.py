"""Rebuild a frozen OOLONG-Pairs run per record, and say what it really covered.

No inference. Everything is keyed by the record's position in the context.
``semantic_map`` runs are reconstructed from their validated rows artifact,
whose digest and source spans are checked before a record earns coverage. A
compatibility path can also reconstruct one-label subcalls. The report
separates things a single accuracy figure hides:

    calls made          how much inference was spent
    records measured    a call about a question that occurs once
    credited by text    records sharing a question with one that was classified
    operational repeats work paid for twice
    per-record accuracy how good the labels were, once per record

    ./.venv/bin/python scripts/audit_pairs_run.py [run_dir]
"""

from __future__ import annotations

import collections
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from alchemist_rlm import oolong_pairs                                # noqa: E402

RECORD = re.compile(r"Instance: (.*?) \|\| Label:")


def latest_run() -> Path:
    runs = sorted((REPO / "runs").glob("pairs_pilot_*_t*"),
                  key=lambda p: p.stat().st_mtime)
    if not runs:
        raise SystemExit("no pairs_pilot run found")
    return runs[-1]


def gold_records(item: dict) -> list[dict]:
    """The context's records in order, each with its user, question and label."""
    out = []
    for line in item["context_window_text_with_labels"].splitlines():
        if not line.startswith("Date:"):
            continue
        question = RECORD.search(line)
        if not question:
            continue
        out.append({
            "index": len(out),
            "user": re.search(r"User: (\d+)", line).group(1),
            "question": question.group(1).strip(),
            "label": line.split("|| Label: ")[1].strip(),
            "date": datetime.strptime(line[6:line.index(" ||")], "%b %d, %Y"),
        })
    return out


def pairs_from_labels(records: list[dict], spec: dict, labels: dict[int, str],
                      *, only_complete_users: bool) -> set[tuple[str, str]]:
    """Derive the task's pairs from exactly the labels credited to the run."""
    rows = [dict(row, label=labels.get(row["index"], row["label"]))
            for row in records]
    by_user: dict[str, list[dict]] = collections.defaultdict(list)
    for row in rows:
        by_user[row["user"]].append(row)
    if only_complete_users:
        by_user = {user: mine for user, mine in by_user.items()
                   if all(row["index"] in labels for row in mine)}

    if spec["kind"] == "sym":
        wanted = set(spec["any"])
        rule = spec.get("date_rule")
        qualified = []
        for user, mine in by_user.items():
            if not any(row["label"] in wanted for row in mine):
                continue
            if rule:
                label, op, when = rule
                limit = datetime.strptime(when, "%Y-%m-%d")
                if not all(row["date"] > limit if op == "after"
                           else row["date"] < limit
                           for row in mine if row["label"] == label):
                    continue
            qualified.append(user)
        ordered = sorted(set(qualified), key=int)
        return {(a, b) for i, a in enumerate(ordered) for b in ordered[i + 1:]}

    def side(conditions):
        chosen = []
        for user, mine in by_user.items():
            counts = {label: sum(1 for row in mine if row["label"] == label)
                      for label in oolong_pairs.LABELS}
            if all(counts[label] >= n if op == ">=" else counts[label] == n
                   for label, op, n in conditions):
                chosen.append(user)
        return chosen

    return {tuple(sorted((a, b), key=int))
            for a in side(spec["a"]) for b in side(spec["b"]) if a != b}


def normalise(response: str) -> str | None:
    """The label the sub-model meant, or None if it named none.

    Syntactic only: a reply is matched against the declared labels and nothing
    else. "the label is **abbreviation**" resolves; a label we never declared
    does not become the nearest one.
    """
    said = (response or "").strip().strip("'\"").lower()
    found = [label for label in oolong_pairs.LABELS if label in said]
    if len(found) != 1:
        return None
    return found[0]


def _semantic_basis(episode: dict) -> dict | None:
    """The latest semantic sweep, preferring one grounded in the context."""
    sweeps = [s for s in episode.get("sweeps") or []
              if isinstance(s, dict) and s.get("operation") == "semantic_map"]
    grounded = [s for s in sweeps
                if (s.get("scope") if isinstance(s.get("scope"), str)
                    else (s.get("scope") or {}).get("kind")) == "context"]
    return (grounded or sweeps or [None])[-1]


def _semantic_rows(run_dir: Path, basis: dict | None) -> tuple[list[dict], dict]:
    """Load and verify the validated table written by ``semantic_map``.

    The old auditor tried to interpret a whole ``item N: value`` response as
    one label and consequently called every new-style subcall malformed.  The
    artifact is the authority now: it contains only rows that passed the
    runtime's schema validation, and its canonical-content digest is recorded
    on the sweep that produced it.
    """
    state = {"scope": None, "rows_ref": None, "rows_digest_verified": None,
             "error": None}
    if not basis:
        return [], state
    scope = basis.get("scope")
    state["scope"] = scope if isinstance(scope, str) else (scope or {}).get("kind")
    ref = basis.get("rows_ref")
    state["rows_ref"] = ref
    if not isinstance(ref, str) or not ref.startswith("artifact://"):
        state["error"] = "semantic sweep has no rows artifact"
        return [], state
    name = ref.removeprefix("artifact://")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        state["error"] = f"invalid artifact reference: {ref!r}"
        return [], state
    path = run_dir / "artifacts" / f"{name}.txt"
    try:
        rows = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        state["error"] = f"cannot read {path.name}: {error}"
        return [], state
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        state["error"] = "semantic rows artifact is not a list of objects"
        return [], state
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    actual = hashlib.sha256(canonical.encode()).hexdigest()
    expected = basis.get("rows_digest")
    state["rows_digest_verified"] = isinstance(expected, str) and actual == expected
    if state["rows_digest_verified"] is False:
        state["error"] = f"rows digest mismatch: expected {expected}, got {actual}"
        return [], state
    return rows, state


def _records_with_spans(item: dict, records: list[dict]) -> list[dict]:
    """Attach source spans to the 787 unlabelled records, preserving order."""
    found = []
    offset = 0
    for raw in item["context_window_text"].splitlines(keepends=True):
        start, end = offset, offset + len(raw)
        offset = end
        line = raw.rstrip("\r\n")
        if not line.startswith("Date:"):
            continue
        user = re.search(r"User: (\d+)", line)
        _, marker, question = line.partition(" || Instance: ")
        if not user or not marker:
            continue
        found.append({"start": start, "end": end, "user": user.group(1),
                      "question": question.strip()})
    if len(found) != len(records):
        raise ValueError(
            f"unlabelled context has {len(found)} records; gold has {len(records)}"
        )
    for index, (span, gold) in enumerate(zip(found, records)):
        if (span["user"], span["question"]) != (gold["user"], gold["question"]):
            raise ValueError(f"labelled and unlabelled record {index} disagree")
    return found


def _predictions_from_rows(item: dict, records: list[dict], rows: list[dict]
                           ) -> tuple[dict[int, str], int]:
    """Map context-grounded semantic rows to actual dataset records by span.

    One semantic unit must contain exactly one dataset record.  A preamble or
    blank unit maps to none; a unit spanning multiple records is ambiguous and
    is never silently credited to all of them.
    """
    spans = _records_with_spans(item, records)
    predicted: dict[int, str] = {}
    ambiguous = 0
    for row in rows:
        start, end, value = row.get("start"), row.get("end"), row.get("value")
        if not isinstance(start, int) or not isinstance(end, int):
            continue
        indexes = [index for index, span in enumerate(spans)
                   if start <= span["start"] and span["end"] <= end]
        if len(indexes) > 1:
            ambiguous += 1
            continue
        if len(indexes) == 1 and value in oolong_pairs.LABELS:
            predicted[indexes[0]] = value
    return predicted, ambiguous


def audit(run_dir: Path) -> dict:
    task_number = int(run_dir.name.rsplit("_t", 1)[1])
    events = [json.loads(line) for line in (run_dir / "trace.jsonl").read_text().splitlines()]
    episode = json.loads((run_dir / "episode.json").read_text())
    spec = next(t for t in oolong_pairs.load()["tasks"] if t["task"] == task_number)["spec"]
    item = json.loads((REPO / "oolong" / "sample.json").read_text())["sample"]["32768"][task_number - 1]

    records = gold_records(item)
    by_question: dict[str, list[int]] = collections.defaultdict(list)
    for record in records:
        by_question[record["question"]].append(record["index"])

    basis = _semantic_basis(episode)
    semantic_rows, artifact_state = _semantic_rows(run_dir, basis)

    # Walk the batches in order so a subcall can be attributed to one.
    batch_of: dict[int, int] = {}
    current = 0
    for event in events:
        if event["kind"] == "batch_start":
            current = event.get("batch", current + 1)
        elif event["kind"] == "subcall":
            batch_of[event["seq"]] = current

    calls = []
    seen_question: dict[str, int] = {}
    for event in events:
        if event["kind"] != "subcall":
            continue
        asked = (event["source"].get("preview") or "").strip()
        # Only the single-label compatibility protocol returns one bare label
        # per call. A
        # semantic-map response contains dozens of numbered decisions and is
        # judged through its validated artifact below, never by this parser.
        said = (None if basis else normalise(event.get("response") or ""))
        indexes = by_question.get(asked, [])
        repeat = asked in seen_question
        seen_question[asked] = seen_question.get(asked, 0) + 1
        calls.append({
            "seq": event["seq"], "batch": batch_of.get(event["seq"]),
            "parent_turn": event.get("parent_turn"),
            "question": asked[:120],
            "matched_records": indexes,
            "repeat_of_earlier_call": repeat,
            "response": (event.get("response") or "")[:80],
            "normalised": said,
            "well_formed": (said is not None) if not basis else None,
        })

    # Two kinds of prediction, and they are not the same evidence.
    #
    # A question that occurs once in the context maps to one record, so a call
    # about it *measures* that record. A question occurring on several records
    # was classified once and credited to all of them, which assumes the model
    # would have answered an identical text identically. That assumption is
    # reasonable and it is still an assumption, and an earlier version of this
    # script folded the two together and reported the total as coverage
    # "per record index" — the index was the key, but the attribution was by
    # text, and 630 of 787 was inferred rather than measured.
    predicted: dict[int, str] = {}
    measured: set[int] = set()
    unmatched = 0
    ambiguous_rows = 0
    credited_by_text = 0
    if basis:
        # A supplied list has no source spans and therefore earns no context
        # coverage.  This is exactly the distinction t20 needs: 220/220 items
        # validated, zero evidence that the 787 context records were swept.
        if artifact_state["scope"] == "context" and not artifact_state["error"]:
            predicted, ambiguous_rows = _predictions_from_rows(
                item, records, semantic_rows)
            measured = set(predicted)
    else:
        for call in calls:
            if not call["matched_records"]:
                unmatched += 1
                continue
            if call["normalised"] is None:
                continue
            for index in call["matched_records"]:
                predicted.setdefault(index, call["normalised"])
            if len(call["matched_records"]) == 1:
                measured.add(call["matched_records"][0])
        credited_by_text = len(predicted) - len(measured)

    right = sum(1 for index, label in predicted.items()
                if label == records[index]["label"])
    right_measured = sum(1 for index in measured
                         if predicted.get(index) == records[index]["label"])
    truth = oolong_pairs.gold(item["context_window_text_with_labels"], spec)

    inference = {
        "protocol": "semantic_map_artifact" if basis else "single_label",
        "calls_made": len(calls),
        "ledger_subcalls": episode["ledger"]["subcalls"],
        "batches": episode["batching"]["batches"],
        "operational_repeats": sum(1 for c in calls if c["repeat_of_earlier_call"]),
    }
    if basis:
        inference.update({
            "semantic_scope": artifact_state["scope"],
            "validated_items": basis.get("valid_items"),
            "total_items": basis.get("total_items"),
            "rows_digest_verified": artifact_state["rows_digest_verified"],
            "artifact_error": artifact_state["error"],
        })
    else:
        inference.update({
            "well_formed_replies": sum(1 for c in calls if c["well_formed"]),
            "malformed_replies": sum(1 for c in calls if not c["well_formed"]),
            "calls_matching_no_record": unmatched,
        })

    return {
        "run": run_dir.name,
        "task": task_number,
        "stop_reason": episode["stop_reason"],
        "inference": inference,
        "coverage": {
            "records_in_context": len(records),
            # One call about a question that occurs once: that record was
            # actually classified.
            "records_measured": len(measured),
            # Plus records sharing a question with one that was, credited on
            # the assumption that an identical text gets an identical answer.
            "records_credited_by_text": credited_by_text,
            "records_covered_inclusive": len(predicted),
            "fraction_measured": round(len(measured) / len(records), 4),
            "fraction_inclusive": round(len(predicted) / len(records), 4),
            "complete": len(measured) == len(records),
            "ambiguous_semantic_rows": ambiguous_rows,
        },
        "accuracy": {
            "records_with_a_label": len(predicted),
            "correct": right,
            "per_record": round(right / len(predicted), 4) if predicted else None,
            "correct_among_measured": right_measured,
            "per_record_measured": (round(right_measured / len(measured), 4)
                                    if measured else None),
            "note": (
                "over records grounded by source spans; says nothing about the rest"
                if basis else
                "over records that were reached; says nothing about the rest, "
                "and `per_record` includes records credited by matching question "
                "text rather than classified"
            ),
        },
        "pairs": {
            "gold": len(truth),
            # What the episode actually returned.
            "as_answered": oolong_pairs.f1(
                oolong_pairs.parse_answer(episode.get("answer") or ""), truth),
            "content_ignoring_format": oolong_pairs.f1(
                oolong_pairs.parse_answer_loose(episode.get("answer") or ""), truth),
            # Only users every one of whose records the run reached. Nothing
            # borrowed from the gold labels.
            "from_complete_users_only": oolong_pairs.f1(
                pairs_from_labels(records, spec, predicted,
                                  only_complete_users=True), truth),
            # The unreached records handed over as correct: the score if the
            # missing work had been free and perfect.
            "ceiling_if_the_rest_were_free": oolong_pairs.f1(
                pairs_from_labels(records, spec, predicted,
                                  only_complete_users=False), truth),
            # What answering every possible pair scores, having read nothing.
            "degenerate_floor": oolong_pairs.f1(
                oolong_pairs.every_pair(item["context_window_text_with_labels"]), truth),
        },
        "calls": calls,
    }


def main() -> int:
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else latest_run()
    report = audit(run_dir)
    out = REPO / "runs" / run_dir.name / "analysis.json"
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str))

    print(f"  {report['run']}  (task {report['task']}, {report['stop_reason']})\n")
    for section in ("inference", "coverage", "accuracy"):
        print(f"  {section}")
        for key, value in report[section].items():
            print(f"    {key:28s} {value}")
    print("  pairs")
    for key, value in report["pairs"].items():
        print(f"    {key:28s} {value}")
    print(f"\n  written: {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
