"""OOLONG-Pairs: the external task whose work is genuinely quadratic.

From `arXiv:2512.24601` Appendix D.1 — twenty queries the paper's authors wrote
by hand over the same `trec_coarse` split of OOLONG we already sampled and
froze. The queries are transcribed verbatim; nothing here rewords them.

Why this one and not the OOLONG we already ran. OOLONG asks for an aggregate
over independent per-entry labels, which is a linear map and a sum. These ask
for *all pairs of users* satisfying a joint property, so the answer itself
grows as the square of the users while the semantic work stays linear. The
paper's own numbers say what that costs a model that cannot compute: base GPT-5
scores 0.04% and base Qwen3-Coder 0.06%, against 58.0% and 23.1% for the same
models under a RLM.

Two properties of this file matter more than its size:

**The gold is derived, never written.** `context_window_text_with_labels`
carries the dataset's own label for every instance, and the model is given the
version without them. So the truth for a query is computed from the data, and
the only thing that could corrupt it is a `spec` that says something other than
its query — which is why `check_specs()` exists and a test runs it.

**The degenerate answer is named in advance.** Answering *every* pair without
reading anything scores F1 0.46-0.56 on the ten symmetric queries, because
roughly two thirds of users qualify. It scores 0.02-0.08 on the ten asymmetric
ones. A score reported without that floor beside it says almost nothing, in the
same way an answer of 488 on the V2 corpus was recognisably a keyword search.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from alchemist_rlm.output_contract import (
    OutputContract,
    PresentationBinding,
    ValidationIssue,
    ValidationResult,
)

REPO = Path(__file__).resolve().parent.parent.parent
QUERIES = REPO / "oolong" / "pairs_queries.json"

LABELS = (
    "description and abstract concept", "entity", "human being",
    "numeric value", "location", "abbreviation",
)

Pair = tuple[str, str]


def load() -> dict[str, Any]:
    """The frozen queries and their gold specs, as transcribed from the paper."""
    return json.loads(QUERIES.read_text())


def official_context(
    items: list[dict[str, Any]], frozen: dict[str, Any],
) -> dict[str, Any]:
    """Return the one context to which all official 32K pair queries bind.

    The source OOLONG split contains more than one context window at the same
    nominal length.  OOLONG-Pairs binds all twenty questions to window zero;
    indexing the source examples by question number silently switched five
    questions to window one.
    """
    binding = frozen["official_binding"]
    expected_id = binding["context_window_id"]
    candidates = [item for item in items
                  if item.get("context_window_id") == expected_id]
    if not candidates:
        raise ValueError(f"official context_window_id={expected_id} is missing")
    item = candidates[0]
    expected = {
        "context_window_text": binding["context_text_sha256"],
        "context_window_text_with_labels": binding["context_with_labels_sha256"],
    }
    for field, digest in expected.items():
        actual = hashlib.sha256(item[field].encode()).hexdigest()
        if actual != digest:
            raise ValueError(
                f"official {field} drifted: expected {digest}, got {actual}")
    return item


def pair_set_sha256(pairs: set[Pair]) -> str:
    """Digest a pair set in the official numeric order."""
    payload = "\n".join(
        f"({left}, {right})"
        for left, right in sorted(pairs, key=lambda pair: tuple(map(int, pair)))
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def check_official_binding(
    items: list[dict[str, Any]], frozen: dict[str, Any],
) -> list[str]:
    """Prove locally that context, derived gold, and published gold agree."""
    try:
        item = official_context(items, frozen)
    except (KeyError, TypeError, ValueError) as exc:
        return [str(exc)]
    questions = "\n".join(task["query"] for task in frozen["tasks"])
    questions_digest = hashlib.sha256(questions.encode()).hexdigest()
    expected_questions = frozen["official_binding"]["questions_sha256"]
    problems: list[str] = []
    if questions_digest != expected_questions:
        problems.append(
            f"official questions expected {expected_questions}, got {questions_digest}")
    expected = {
        int(task): (int(count), digest)
        for task, count, digest in frozen["official_binding"]["gold"]
    }
    for task in frozen["tasks"]:
        task_id = int(task["task"])
        truth = gold(item["context_window_text_with_labels"], task["spec"])
        count, digest = expected.get(task_id, (-1, ""))
        actual_digest = pair_set_sha256(truth)
        if len(truth) != count or actual_digest != digest:
            problems.append(
                f"task {task_id}: official gold expected {count}/{digest}, "
                f"derived {len(truth)}/{actual_digest}")
    return problems


# --- reading the data -------------------------------------------------------
_ROW = re.compile(r"User: (\d+) \|\| Instance: (.*?) \|\| Label: (.+)$")


def entries(labelled_text: str) -> list[dict[str, Any]]:
    """One record per data line: user, label, date. Lines that are not data —
    the preamble the dataset prepends — are skipped rather than guessed at."""
    out: list[dict[str, Any]] = []
    for line in labelled_text.splitlines():
        if not line.startswith("Date: "):
            continue
        match = _ROW.search(line)
        if not match:
            continue
        out.append({
            "user": match.group(1),
            "label": match.group(3).strip(),
            "date": datetime.strptime(line[6:line.index(" ||")], "%b %d, %Y"),
        })
    return out


def _pairs(users: Iterable[str]) -> set[Pair]:
    ordered = sorted(set(users), key=int)
    return {(a, b) for a, b in itertools.combinations(ordered, 2)}


def _cross(left: Iterable[str], right: Iterable[str]) -> set[Pair]:
    out: set[Pair] = set()
    for a in left:
        for b in right:
            if a != b:
                out.add(tuple(sorted((a, b), key=int)))       # type: ignore[arg-type]
    return out


def gold(labelled_text: str, spec: dict[str, Any]) -> set[Pair]:
    """The exact answer, computed from the dataset's labels."""
    rows = entries(labelled_text)
    by_user: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_user.setdefault(row["user"], []).append(row)

    if spec["kind"] == "sym":
        wanted = set(spec["any"])
        rule = spec.get("date_rule")
        qualified = []
        for user, mine in by_user.items():
            if not any(r["label"] in wanted for r in mine):
                continue
            if rule:
                label, op, when = rule
                limit = datetime.strptime(when, "%Y-%m-%d")
                # "all instances that are <label> ... must be after <date>" —
                # vacuously true for a user with no instance of that label.
                ok = all(r["date"] > limit if op == "after" else r["date"] < limit
                         for r in mine if r["label"] == label)
                if not ok:
                    continue
            qualified.append(user)
        return _pairs(qualified)

    def side(conditions: list[list[Any]]) -> list[str]:
        chosen = []
        for user, mine in by_user.items():
            counts = {label: sum(1 for r in mine if r["label"] == label)
                      for label in LABELS}
            if all(counts[label] >= n if op == ">=" else counts[label] == n
                   for label, op, n in conditions):
                chosen.append(user)
        return chosen

    return _cross(side(spec["a"]), side(spec["b"]))


def every_pair(labelled_text: str) -> set[Pair]:
    """The answer of a model that read nothing. Named in advance so a score can
    be read against it instead of admired on its own."""
    return _pairs(row["user"] for row in entries(labelled_text))


# --- scoring ----------------------------------------------------------------
# Exactly the shape the paper's query names: "(user_id_1, user_id_2)".
_PAPER = re.compile(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)")
# That, plus the shape a REPL answer naturally has when the delivered value is
# the list itself: [["a", "b"], ...].
_REPL = re.compile(r"[\(\[]\s*[\"']?(\d+)[\"']?\s*,\s*[\"']?(\d+)[\"']?\s*[\)\]]")

_PAIR_LINE = re.compile(r"^\(\s*(\d+)\s*,\s*(\d+)\s*\)$")
# A rejected line may already contain exactly the two requested numeric IDs and
# differ from the public grammar only by its outer delimiters.  Keep this
# recognition separate from parsing: it powers a precise diagnostic but never
# turns invalid text into an accepted answer.
_BARE_PAIR_LINE = re.compile(r"^\s*(\d+)\s*,\s*(\d+)\s*$")


def _validate_pair_output(text: str) -> ValidationResult:
    """Validate only the public presentation rule shared by all twenty tasks."""
    if text == "":
        return ValidationResult(True)
    body = text[:-1] if text.endswith("\n") else text
    if not body or "\n\n" in body:
        return ValidationResult(False, issues=(ValidationIssue(
            "pair_line_required", "use one non-empty pair per line"),))
    def example(number: int, line: str) -> str:
        excerpt = line if len(line) <= 120 else f"{line[:119]}…"
        return f"line {number}: {excerpt!r}"

    seen: dict[Pair, int] = {}
    missing_parentheses: list[str] = []
    invalid_shape: list[str] = []
    wrong_order: list[str] = []
    duplicates: list[str] = []
    for number, line in enumerate(body.split("\n"), 1):
        match = _PAIR_LINE.fullmatch(line)
        if match is None:
            match = _BARE_PAIR_LINE.fullmatch(line)
            if match is None:
                invalid_shape.append(example(number, line))
                continue
            missing_parentheses.append(example(number, line))
        left, right = match.groups()
        if int(left) >= int(right):
            wrong_order.append(example(number, line))
        pair = (left, right)
        if pair in seen:
            duplicates.append(
                f"{example(number, line)} repeats line {seen[pair]}")
        else:
            seen[pair] = number
    issues = tuple(
        ValidationIssue(code, message, len(lines), tuple(lines[:3]))
        for code, message, lines in (
            ("pair_parentheses_required",
             "each cited line contains two numeric IDs separated by a comma but "
             "lacks the required literal opening '(' and closing ')' delimiters; "
             "enclose every complete line as "
             "(lower_numeric_id, higher_numeric_id); other issue codes remain "
             "independent",
             missing_parentheses),
            ("invalid_pair_line",
             "each line must contain only (lower_numeric_id, higher_numeric_id); "
             "quoted excerpts are received invalid lines, not desired output",
             invalid_shape),
            ("pair_order", "each pair must put the lower numeric ID first",
             wrong_order),
            ("duplicate_pair", "pairs must not duplicate an earlier pair",
             duplicates),
        )
        if lines
    )
    return ValidationResult(not issues, issues=issues)


def _pair_presentation_equivalent(value: Any, text: str) -> ValidationResult:
    committed = pairs_from_answer_value(value)
    if isinstance(value, str):
        # A model may compute its answer directly as text.  The loose parser is
        # the adapter's pre-existing content authority for that representation;
        # it knows no gold and lets the binding prove that adding parentheses
        # changed presentation only.  A non-empty string with no recognizable
        # pair is not silently reinterpreted as the empty answer.
        parsed = parse_answer_loose(value)
        committed = parsed if parsed or value == "" else None
    elif (isinstance(value, list) and value
          and all(isinstance(item, str) for item in value)):
        committed = set()
        for item in value:
            match = _PAIR_LINE.fullmatch(item) or _PAIR_ONLY.fullmatch(item)
            if match is None:
                committed = None
                break
            committed.add(tuple(sorted(match.groups(), key=int)))  # type: ignore[arg-type]
    if committed is None:
        return ValidationResult(False, issues=(ValidationIssue(
            "unrepresentable_answer_value",
            "the committed value is not a collection of two-ID pairs",
        ),))
    presented = parse_answer(text)
    if committed == presented:
        return ValidationResult(True)
    missing = sorted(committed - presented, key=lambda pair: tuple(map(int, pair)))
    extra = sorted(presented - committed, key=lambda pair: tuple(map(int, pair)))
    issues: list[ValidationIssue] = []
    if missing:
        issues.append(ValidationIssue(
            "presentation_missing_content",
            "the presentation omits committed pairs",
            len(missing), tuple(f"({a}, {b})" for a, b in missing[:3]),
        ))
    if extra:
        issues.append(ValidationIssue(
            "presentation_added_content",
            "the presentation adds pairs absent from the committed value",
            len(extra), tuple(f"({a}, {b})" for a, b in extra[:3]),
        ))
    return ValidationResult(False, issues=tuple(issues))


def pair_output_contract() -> OutputContract:
    """The task's stated output grammar, with no access to labels or gold."""
    return OutputContract(
        name="oolong_pairs_lines",
        version="1",
        specification={
            "empty_allowed": True,
            "line": "(lower_numeric_id, higher_numeric_id)",
            "separator": "newline",
            "duplicates": False,
            "trailing_newline": True,
        },
        validator=_validate_pair_output,
        binding=PresentationBinding(
            name="oolong_pairs_value_equivalence",
            version="1",
            specification={"equivalence": "same set of numeric ID pairs"},
            equivalent=_pair_presentation_equivalent,
        ),
    )


def parse_answer(text: str) -> set[Pair]:
    """The paper's format and nothing else: `(a, b)`.

    This used to accept `[a, b]` too, and was still called the official parser.
    The widening was reasoned and written down — a RLM builds its answer as a
    Python object, so demanding it retype the object as text measures string
    formatting rather than the runtime — but the name claimed conformance the
    regex did not have, and the gap is not small: of nine v2 queries above their
    floor, four scored only because of it.

    So the reasoning keeps its parser and loses the name. This one is what the
    paper asks for, `parse_answer_repl` is the widening with the argument
    attached, and `parse_answer_loose` answers "was it computed at all". Three
    numbers, none of them standing in for another.
    """
    return {tuple(sorted((a, b), key=int))                     # type: ignore[misc]
            for a, b in _PAPER.findall(text or "")}


def parse_answer_repl(text: str) -> set[Pair]:
    """The paper's format, plus the structure a delivered Python value renders
    to.

    `submit(pairs)` crosses the channel as `[["a", "b"], ...]`, which satisfies
    no textual instruction but is the answer the model computed, in the shape a
    REPL produces. Reporting it separately says how much of the score rests on
    accepting that, rather than burying the question in the headline number.
    """
    return {tuple(sorted((a, b), key=int))                     # type: ignore[misc]
            for a, b in _REPL.findall(text or "")}


# One shape, written once: two ids and a horizontal separator, nothing else.
#
# Horizontal space only. `\s` matches a newline, and under re.MULTILINE that
# let the separator span the line break: "10\n20" parsed as the pair (10, 20),
# so a list of single ids one per line silently became pairs of neighbours. A
# t20 answer that was a bare id list produced the invented pair
# (35142, 35618). It scored zero either way — the pair was wrong — which is
# exactly why the "only three runs moved" check did not catch it.
_PAIR = r"[ \t]*(\d+)(?:[ \t]*,[ \t]*|[ \t]+)(\d+)[ \t]*"
# Anchored to a line, for an answer that is text.
_BARE = re.compile(rf"^{_PAIR}$", re.MULTILINE)
# Anchored to a whole string, for an element of a delivered list.
_PAIR_ONLY = re.compile(_PAIR)


def _from_json_list(text: str) -> set[Pair]:
    """Pairs from an answer that is, *in whole*, a JSON list of strings.

    A model that formats each pair itself and submits the list of them —
    `["26503, 92741", "26503, 90231", …]` — has done all of the work this
    parser asks about, and the only thing between it and `_BARE` is that the
    harness rendered the list onto one line.

    This reads the structure instead of the surface, which matters more than it
    looks. The first version of this was a regex for *any* quoted run holding
    two ids, and that cannot tell a delivered list from prose: it would take
    the pair (2024, 2025) out of `the relevant years are "2024, 2025"`. Over
    the 245 answers on record it fired exactly once, and that once was whole
    valid JSON — so the structured reading buys the entire measured benefit and
    none of the risk. `fullmatch` per element is what makes it safe: a list of
    single ids yields nothing, and `["a, b, c"]` yields nothing.
    """
    try:
        value = json.loads(text)
    except (ValueError, TypeError):
        return set()
    if not isinstance(value, list):
        return set()
    out: set[Pair] = set()
    for item in value:
        if not isinstance(item, str):
            continue
        match = _PAIR_ONLY.fullmatch(item)
        if match:
            out.add(tuple(sorted(match.groups(), key=int)))      # type: ignore[arg-type]
    return out


def parse_answer_loose(text: str) -> set[Pair]:
    """The same pairs, plus a line that is two ids and nothing else.

    This exists because the two failures are different and a single number
    hides which one happened. An automatic run assigned `Final` cleanly, was
    not truncated, and emitted 1,016 pairs as `31080, 89840` — one per line, no
    parentheses. Scored strictly that is zero: not one pair parsed. Scored on
    content it is 0.223 against a degenerate floor of 0.051, so the pairs were
    substantially right and the *format* was wrong.

    The separator is part of the format, not part of the content, and reading
    only the comma made this measurement fail at its own job. t17 assigned
    `Final = "\\n".join(f"{a} {b}" ...)` and scored 0.000 on both layers — the
    content layer thereby reporting that nothing had been computed, when what
    it held was 1,701 correct pairs out of 2,330 at precision 0.81 against a
    floor of 0.161. One space cost the whole difference.

    The strict parser stays the official one, because the query says "list all
    pairs in the format (user_id_1, user_id_2)" and following an instruction is
    part of the task. Reporting both is this project's two-layer scoring applied
    to the answer itself: what was computed, and whether it was returned as
    asked. Neither number is allowed to stand in for the other.

    Query 14 is the case that forced the JSON reading: 0.723 in the morning as
    `"\\n".join(...)`, 0.000 in the evening as a list of the same *kind* of
    string. What this parser could not see was 325 pairs, 217 of them right.
    Note what that does and does not say — the two runs are not the same answer
    in different wrappers. The morning run predicted 555 and hit 402; the
    evening run predicted 325 and hit 217. So the scorer was hiding real
    content and exaggerating the fall to zero, *and* the newer run genuinely
    computed less. Both, not either.
    """
    pairs = set(parse_answer_repl(text))
    pairs.update(tuple(sorted((a, b), key=int))                # type: ignore[misc]
                 for a, b in _BARE.findall(text or ""))
    pairs.update(_from_json_list(text or ""))
    return pairs


def pairs_from_answer_value(value: Any) -> set[Pair] | None:
    """Read OOLONG pairs from the canonical typed answer, if it has that shape.

    This scorer is deliberately separate from ``pair_output_contract``.  It
    may interpret answer content and later compare it with gold; the contract
    may only validate public presentation syntax. ``None`` means the value has
    no supported pair collection shape, while an empty list is a valid empty
    prediction.
    """
    if not isinstance(value, list):
        return None
    pairs: set[Pair] = set()
    for item in value:
        if not isinstance(item, list) or len(item) != 2:
            return None
        left, right = item
        if isinstance(left, bool) or isinstance(right, bool):
            return None
        if not isinstance(left, (str, int)) or not isinstance(right, (str, int)):
            return None
        left_text, right_text = str(left), str(right)
        if not left_text.isdigit() or not right_text.isdigit():
            return None
        if int(left_text) == int(right_text):
            return None
        pairs.add(tuple(sorted((left_text, right_text), key=int)))  # type: ignore[arg-type]
    return pairs


def score_answer_value(value: Any, truth: set[Pair]) -> dict[str, Any] | None:
    """Score canonical typed content without parsing presentation text."""
    predicted = pairs_from_answer_value(value)
    return None if predicted is None else f1(predicted, truth)


def f1(predicted: set[Pair], truth: set[Pair]) -> dict[str, Any]:
    """Score a predicted pair set against the truth, with its parts kept.

    Precision and recall are reported beside F1 because they fail differently
    here: answering every possible pair scores perfect recall and near-zero
    precision, and a single F1 would not say which of those happened.
    """
    hit = len(predicted & truth)
    precision = hit / len(predicted) if predicted else 0.0
    recall = hit / len(truth) if truth else 0.0
    score = (2 * precision * recall / (precision + recall)) if hit else 0.0
    if not predicted and not truth:
        precision = recall = score = 1.0
    return {"f1": round(score, 4), "precision": round(precision, 4),
            "recall": round(recall, 4), "predicted": len(predicted),
            "truth": len(truth), "hit": hit}


# --- the one thing this file could hide -------------------------------------
def check_specs() -> list[str]:
    """Every label a spec names must appear in the query beside it, and the
    query's shape must match the spec's kind. A silently wrong transcription
    would produce a gold answer that is exactly as plausible as a right one."""
    problems: list[str] = []
    for task in load()["tasks"]:
        query, spec, n = task["query"].lower(), task["spec"], task["task"]
        named = (list(spec["any"]) if spec["kind"] == "sym"
                 else [c[0] for c in spec["a"] + spec["b"]])
        for label in named:
            if label not in query:
                problems.append(f"task {n}: spec names {label!r}, the query does not")
        if spec["kind"] == "sym" and "both users have at least one" not in query:
            problems.append(f"task {n}: marked symmetric, query is not")
        if spec["kind"] == "asym" and "the other user" not in query:
            problems.append(f"task {n}: marked asymmetric, query is not")
        if spec["kind"] == "sym" and spec.get("date_rule"):
            label, op, when = spec["date_rule"]
            month = datetime.strptime(when, "%Y-%m-%d").strftime("%B %-d, %Y").lower()
            if op not in query or month not in query:
                problems.append(f"task {n}: date rule {op} {month} not in the query")
        if spec["kind"] == "sym" and not spec.get("date_rule") and (
                " must be after " in query or " must be before " in query):
            problems.append(f"task {n}: query states a date rule, the spec has none")
    return problems
