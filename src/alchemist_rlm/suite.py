"""The ten decision tasks, scored on the result and on the trajectory.

The separation is the point. A correct answer reached by reading the whole
context by eye demonstrates nothing about a RLM, and a correct pipeline whose
sub-model misreads its slice is not a protocol failure. So every task states
what must be true of the *result* and, separately, which trajectory facts must
hold — and those facts are categorical (did a subcall receive the evidence? did
recursion happen?), which is what makes them readable at this sample size.

Atomic attribution comes out of the same episode, with no extra inference. The
trace records each subcall's source precisely enough to locate it in the
context, so three different failures separate cleanly:

    no subcall ever received the passage        -> retrieval / orchestration
    a subcall received it and answered wrong    -> the sub-model's own ceiling
    a subcall answered right, the root did not  -> synthesis

That ceiling is already measured for this checkpoint at 38% single-item and 73%
batched on `trec_coarse`, so task 7 is expected to be hard for reasons that have
nothing to do with this harness. Saying which is which is the whole job.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from alchemist_rlm import corpus
from alchemist_rlm.tracing import covered, digest, locate, spans_of

REPO = Path(__file__).resolve().parent.parent.parent


# --- trajectory facts -------------------------------------------------------
def _names(code: str) -> set[str]:
    return set(re.findall(r"\b[A-Za-z_]\w*\b", code or ""))


def _has_number(reply: str, value: int) -> bool:
    digits = {int(m.replace(",", "").replace(".", ""))
              for m in re.findall(r"\d[\d,.]*", reply or "")}
    return value in digits


def _has_text(reply: str, needle: str) -> bool:
    return needle.lower() in (reply or "").lower()


def trajectory(episode: Any, events: list[dict[str, Any]], context: str) -> dict[str, Any]:
    """What the run actually did, read from the trace and not from the answer.

    The batching and recursion facts are read from instrumentation rather than
    inferred from counts. `batched` was once `len(subcalls) >= 4`, which four
    sequential `llm_query` calls satisfy just as well as a real batch; and a
    correct answer was once enough to look like recursion when no child node had
    ever been created.
    """
    tool_calls = [e for e in events if e["kind"] == "tool_call"]
    subcalls = [e for e in events if e["kind"] == "subcall"]
    recursions = [e for e in events if e["kind"] == "rlm_query"]
    executed = [s for s in episode.steps if not getattr(s, "refused", False)]
    refused = [s for s in episode.steps if getattr(s, "refused", False)]
    batching = getattr(episode, "batching", None) or {}

    bound: set[str] = set()
    reused = False
    for index, step in enumerate(executed):
        if index and (bound & _names(step.code)):
            reused = True
        bound |= set(step.defined)

    # `spans_of`, not `locate`: a source the harness rendered out of context
    # spans carries them, and a source the model sliced itself is still found by
    # hash. Reading only the second is why every run since v5 scored
    # `coverage: 0.0` beside a sweep that had examined all 1,600 items.
    spans = [span for s in subcalls
             for span in spans_of(context, s.get("source") or {})]
    # Plus what the children read. A root that delegated every character made no
    # subcalls of its own and scored 0.0, which described the reader rather than
    # the run. A part is credited only if it was a hash-verified slice of this
    # context AND the child's own sweep examined all of it; `of` keeps a
    # grandchild's span out of a frame it does not belong to.
    spans += [tuple(e["span"]) for e in events
              if e["kind"] == "delegated_span" and e.get("of") == digest(context)]
    answer = episode.answer or ""
    # Did the answer come out of a computation, or was it asserted? An answer
    # that never appeared in any observation was produced by the model's head.
    from_python = bool(answer) and any(
        answer.strip()[:80] in ((e["observation"].get("stdout") or "") +
                                str(e["observation"].get("value") or ""))
        for e in events if e["kind"] == "observation"
    )

    changed_after_refusal = False
    if refused:
        from alchemist_rlm.protocol import code_key
        refused_keys = {code_key(s.code) for s in refused}
        changed_after_refusal = any(code_key(s.code) not in refused_keys for s in executed)

    return {
        "used_the_tool": bool(tool_calls),
        "avoided_the_tool": not tool_calls,
        "tool_calls": len(tool_calls),
        "integrated_observation": bool(tool_calls) and episode.answer is not None,
        "reused_a_variable": reused,
        "subcalls": len(subcalls),
        "made_a_sourced_subcall": any((s["source"]["chars"] or 0) > 0 for s in subcalls),
        # Kept for continuity with the V1 record, and explicitly not evidence
        # of batching: four sequential calls satisfy it.
        "subcalls_at_least_4": len(subcalls) >= 4,
        "used_batched_api": batching.get("batches", 0) >= 1,
        "batches": batching.get("batches", 0),
        "peak_in_flight": batching.get("peak_in_flight", 0),
        "ran_concurrently": batching.get("peak_in_flight", 0) >= 2,
        "sequential_subcalls": batching.get("sequential_calls", 0),
        "lazy_pulls": batching.get("lazy_pulls", 0),
        # More than one pull means the generator was drained in chunks, which is
        # the observation that separates a stream from a materialised list.
        "consumed_lazily": batching.get("lazy_pulls", 0) >= 2,
        "coverage": round(covered(context, spans), 3),
        "coverage_complete": covered(context, spans) >= 0.98,
        "recursion_observed": bool(recursions),
        "recursion_depth": max((r.get("depth", 0) for r in recursions), default=0),
        "child_nodes": max(0, (episode.ledger or {}).get("nodes", 1) - 1),
        # A child that never ran code did not do the work; the answer came from
        # somewhere else and calling that recursion would be generous.
        "child_did_work": any(e.get("depth", 0) >= 1 for e in events
                              if e["kind"] in ("tool_call", "subcall")),
        "duplicates_observed": episode.duplicates_observed,
        "changed_action_after_refusal": changed_after_refusal,
        "answer_came_from_python": from_python,
        "protocol_errors": len(episode.protocol_errors),
        "clean_termination": episode.stop_reason in ("answer_tag", "submitted",
                                                     "no_tool_call"),
    }


# --- atomic attribution -----------------------------------------------------
def attribute_needle(events: list[dict[str, Any]], context: str,
                     result_correct: bool) -> dict[str, Any]:
    """Separate a retrieval failure from the sub-model's own ceiling."""
    subcalls = [e for e in events if e["kind"] == "subcall"]
    offset = context.find(corpus.NEEDLE_NOTE)
    saw_it: list[dict[str, Any]] = []
    for call in subcalls:
        span = locate(context, call.get("source") or {})
        if span and span[0] <= offset < span[1]:
            saw_it.append(call)
    if result_correct:
        return {"verdict": "answered", "subcalls_that_received_the_needle": len(saw_it)}
    if not subcalls:
        return {"verdict": "no_delegation",
                "meaning": "the root never asked a sub-model to read anything; "
                           "this is orchestration, not a semantic ceiling"}
    if not saw_it:
        return {"verdict": "retrieval_failure",
                "meaning": f"{len(subcalls)} subcalls ran and none received the "
                           "passage holding the evidence",
                "coverage": round(covered(context, [
                    s for s in (locate(context, c.get("source") or {}) for c in subcalls) if s
                ]), 3)}
    answered_it = [c for c in saw_it
                   if corpus.NEEDLE_PERSON.lower() in (c.get("response") or "").lower()]
    if answered_it:
        return {"verdict": "synthesis_failure",
                "meaning": "a sub-model was given the passage and named the person; "
                           "the root did not carry it into the final answer",
                "subcall_said": answered_it[0].get("response", "")[:200]}
    return {"verdict": "atomic_failure",
            "meaning": "a sub-model received the passage and still did not name the "
                       "person; this is the sub-model's ceiling, not the harness",
            "subcall_said": saw_it[0].get("response", "")[:200]}


def attribute_oolong(events: list[dict[str, Any]], context: str,
                     result_correct: bool) -> dict[str, Any]:
    """Route, atom and aggregation, separated as far as the trace allows.

    Stated honestly: label-level accuracy per line would need gold labels
    matched against free-form sub-model replies, which the trace does not make
    reliable. What it does support is coverage and whether the count was
    computed rather than asserted, and those already separate orchestration
    from the 38%/73% ceiling measured for this checkpoint.
    """
    subcalls = [e for e in events if e["kind"] == "subcall"]
    spans = [s for s in (locate(context, c.get("source") or {}) for c in subcalls) if s]
    coverage = round(covered(context, spans), 3)
    if result_correct:
        return {"verdict": "answered", "coverage": coverage, "subcalls": len(subcalls)}
    if not subcalls:
        return {"verdict": "no_delegation", "coverage": 0.0,
                "meaning": "the root classified by eye or not at all"}
    if coverage < 0.98:
        return {"verdict": "incomplete_coverage", "coverage": coverage,
                "meaning": f"{len(subcalls)} subcalls reached {coverage:.0%} of the "
                           "context; an aggregate over a partial read cannot be right"}
    return {"verdict": "atomic_or_aggregation", "coverage": coverage,
            "subcalls": len(subcalls),
            "meaning": "the whole context was read by sub-models and the aggregate "
                       "is still wrong. Given the measured 38% single-item and 73% "
                       "batched ceiling on trec_coarse, the sub-model's labels are "
                       "the first suspect, not the orchestration."}


# --- the tasks --------------------------------------------------------------
@dataclass(frozen=True)
class SuiteTask:
    """One suite task: what is asked, what is true, and what must hold of the run.

    `requires` is the part that matters. It names the trajectory facts a pass
    depends on, and the verdict reads that declaration rather than a fixed list
    — a verdict that checked fewer facts than it declared reported a passing run
    whose own record said the opposite.
    """
    id: str
    question: str
    context: str
    truth: Any
    scores_result: Callable[[str], bool]
    requires: dict[str, bool]
    note: str = ""
    inject: str | None = None
    attribute: Callable[[list[dict[str, Any]], str, bool], dict[str, Any]] | None = None
    band: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """The task as plain data, including the hash of its context. Freezing the
        context by hash is what makes two runs comparable: the same question over a
        silently edited corpus is a different task.
        """
        import hashlib
        return {
            "id": self.id, "question": self.question,
            "context_sha256": hashlib.sha256(self.context.encode()).hexdigest()[:16],
            "context_chars": len(self.context), "truth": str(self.truth),
            "requires": self.requires, "inject": self.inject, "note": self.note,
        }


LEDGER = "\n".join(
    f"{i:03d} | {dept} | {amount} | {status}"
    for i, (dept, amount, status) in enumerate(
        [("logistics", 1420, "paid"), ("kitchen", 780, "pending"),
         ("logistics", 310, "paid"), ("front", 95, "void"),
         ("kitchen", 2050, "paid"), ("logistics", 640, "pending"),
         ("front", 1180, "paid"), ("kitchen", 415, "void"),
         ("logistics", 2270, "paid"), ("front", 530, "pending"),
         ("kitchen", 1325, "paid"), ("logistics", 860, "paid")],
        start=1,
    )
)


def _oolong_item() -> dict[str, Any]:
    """A frozen rung-1024 item from the sample taken before any model ran.

    Real benchmark data rather than a synthetic imitation, so the result sits on
    the same axis as the 38%/73% atomic measurement already in this repository.
    """
    data = json.loads((REPO / "oolong" / "sample.json").read_text())
    return data["sample"]["1024"][0]


def _oolong_gold(item: dict[str, Any]) -> str:
    """The gold label, unwrapped from the string that looks like a list.

    `answer` arrives as the literal `"['abbreviation']"` — a str, not a list.
    Taken at face value the scorer would search a reply for the brackets and
    quotes too, and would have marked every correct answer wrong. The existing
    `rescore.py` in this repository exists because of a scorer defect of exactly
    this shape, so the unwrap is asserted below rather than assumed.
    """
    import ast

    raw = item["answer"]
    if isinstance(raw, str):
        try:
            raw = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return raw.strip()
    if isinstance(raw, (list, tuple)):
        return str(raw[0]).strip()
    return str(raw).strip()


_OOLONG = _oolong_item()
_OOLONG_GOLD = _oolong_gold(_OOLONG)
assert _OOLONG_GOLD == "abbreviation", f"gold unwrap broke: {_OOLONG_GOLD!r}"
assert "[" not in _OOLONG_GOLD, "the gold label must not carry list syntax"

TASKS: tuple[SuiteTask, ...] = (
    SuiteTask(
        id="t01_direct_answer",
        question="What is the capital of France? Answer directly.",
        context="",
        truth="Paris",
        scores_result=lambda r: _has_text(r, "paris"),
        requires={"avoided_the_tool": True, "clean_termination": True},
        note="Control in the other direction: reaching for the tool here is a "
             "failure of judgement, not of capability.",
    ),
    SuiteTask(
        id="t02_exact_calculation",
        question="What is 7919 multiplied by 6113? Reply with the number only.",
        context="",
        truth=7919 * 6113,
        scores_result=lambda r: _has_number(r, 7919 * 6113),
        requires={"used_the_tool": True, "answer_came_from_python": True,
                  "clean_termination": True},
        note="A large-prime product: reliably wrong mentally, exact in Python.",
    ),
    SuiteTask(
        id="t03_persistence",
        question=("The data is a table of rows 'id | department | amount | status'. "
                  "First store the number of rows in a variable called n. Then, in a "
                  "separate later call, report n multiplied by 7. Reply with the "
                  "final number only."),
        context=LEDGER,
        truth=12 * 7,
        scores_result=lambda r: _has_number(r, 84),
        requires={"used_the_tool": True, "reused_a_variable": True,
                  "clean_termination": True},
        note="Scored on the trajectory as well: a single call that computes 84 "
             "outright is a correct number and a failed persistence test.",
    ),
    SuiteTask(
        id="t04_duplicate_recovery",
        question=("The data is a table of rows 'id | department | amount | status'. "
                  "Find the id of the row with the largest amount among rows whose "
                  "status is 'void'. Reply with the id only."),
        context=LEDGER,
        truth="008",
        scores_result=lambda r: 8 in {int(t) for t in re.findall(r"\b\d+\b", r or "")},
        requires={"used_the_tool": True, "changed_action_after_refusal": True,
                  "clean_termination": True},
        inject="refuse_first_call_as_duplicate",
        note="The runner refuses the first call so the condition occurs on every "
             "run rather than only when the model happens to repeat itself.",
    ),
    SuiteTask(
        id="t05_lexical_search",
        question=("Each record in the data has a line 'Depot: <name>'. How many "
                  "records list Depot: Valparaiso? Reply with the number only."),
        context=corpus.CORPUS,
        truth=corpus.TRUTHS["valparaiso_count"],
        scores_result=lambda r: _has_number(r, corpus.TRUTHS["valparaiso_count"]),
        requires={"used_the_tool": True, "answer_came_from_python": True,
                  "clean_termination": True},
        note="Deterministic and countable without inference. A model that spends "
             "subcalls here has misjudged the operation, which is itself a result.",
    ),
    SuiteTask(
        id="t06_semantic_needle",
        question=("Somewhere in the data one person was held accountable for a "
                  "shipment that went missing. Who was it? Give the person's name."),
        context=corpus.CORPUS,
        truth=corpus.NEEDLE_PERSON,
        scores_result=lambda r: _has_text(r, corpus.NEEDLE_PERSON),
        requires={"used_the_tool": True, "made_a_sourced_subcall": True,
                  "clean_termination": True},
        attribute=attribute_needle,
        note="The evidence is stated only in paraphrase and the words of the "
             "question appear on three decoy records that are not the answer. "
             "Grep lands on the decoys.",
    ),
    SuiteTask(
        id="t07_oolong_aggregate",
        question=_OOLONG["question"],
        context=_OOLONG["context_window_text"],
        truth=_OOLONG_GOLD,
        scores_result=lambda r: _has_text(r, _OOLONG_GOLD),
        requires={"used_the_tool": True, "clean_termination": True},
        attribute=attribute_oolong,
        note=f"Frozen OOLONG-synth trec_coarse item {_OOLONG['id']}, rung 1024. "
             "The atomic ceiling here is already measured at 38% single-item and "
             "73% batched, so a wrong answer is expected to be semantic.",
    ),
    SuiteTask(
        id="t08_lazy_batching_coverage",
        question=("Each record ends with a 'Note:' line. In some of them the note "
                  "describes bad weather stopping or delaying work — fog, rain, "
                  "wind, ice or a squall. How many records have such a note? Every "
                  "record must be examined. Reply with the number only."),
        context=corpus.CORPUS,
        truth=corpus.TRUTHS["weather_count"],
        scores_result=lambda r: _has_number(r, corpus.TRUTHS["weather_count"]),
        requires={"used_the_tool": True, "batched": True,
                  "coverage_complete": True, "answer_came_from_python": True,
                  "clean_termination": True},
        note="Five different phrasings with no shared keyword, so a regex cannot "
             "count them and every record has to be read by a sub-model. Scored "
             "on coverage as well as on the number.",
    ),
    SuiteTask(
        id="t09_recursion_depth_2",
        question=("Split the data into parts and have a separate recursive call "
                  "analyse each part with rlm_query. In how many records is the "
                  "Crew line 'night'? Reply with the number only."),
        context=corpus.CORPUS,
        truth=corpus.TRUTHS["night_crew_count"],
        scores_result=lambda r: _has_number(r, corpus.TRUTHS["night_crew_count"]),
        requires={"used_the_tool": True, "recursion_observed": True,
                  "clean_termination": True},
        note="The lowest prior of the ten: in the legacy runs this checkpoint "
             "never delegated spontaneously and never used rlm_query. Included "
             "because recursion is a defining property, not because it is "
             "expected to pass. If the mechanism works and the root does not "
             "select it, that points to pivot_to_training.",
    ),
    SuiteTask(
        id="t10_multi_hop",
        question=("Which person led the crew that moved crate 112? The answer is "
                  "not stated in one place: find which crew moved it and when, "
                  "then find who led that crew that week. Give the name and say "
                  "which two passages you used."),
        context=corpus.CORPUS,
        truth=corpus.HOP_ANSWER,
        scores_result=lambda r: _has_text(r, corpus.HOP_ANSWER),
        requires={"used_the_tool": True, "clean_termination": True},
        note="Two hops that share no vocabulary with each other. The roster holds "
             "a plausible wrong name for the adjacent week, which is the shape of "
             "the fabrication this checkpoint produced before.",
    ),
)

TASKS_BY_ID = {task.id: task for task in TASKS}

# The four discriminating tasks the plan runs against each BF16 control arm.
CONTROL_IDS = ("t03_persistence", "t06_semantic_needle",
               "t08_lazy_batching_coverage", "t09_recursion_depth_2")
# The two the name ablation swaps `run_python` into.
ABLATION_IDS = ("t02_exact_calculation", "t06_semantic_needle")


def suite_sha256() -> str:
    """One hash over every V1 task, so the frozen suite can prove it never moved.
    Anything learned after a run goes into a new configuration; a test asserts this
    value is unchanged.
    """
    import hashlib
    return hashlib.sha256(
        json.dumps([t.to_dict() for t in TASKS], sort_keys=True).encode()
    ).hexdigest()


SUITE_SHA256 = suite_sha256()
