"""The six frozen Gate A tasks.

Gate A asks one question only: **does this model operate the Python modality
through its native tool protocol at all?** Not whether it orchestrates, not
whether it is accurate on OOLONG — those are Gate C and the atom, measured
separately. Every task here is short, mechanical, and has a truth known by
construction.

Frozen means frozen: `TASKS_SHA256` covers the whole table. If a task is edited,
the hash changes and prior Gate A results stop being comparable. That rule
exists because a scorer bug in this repository once inflated a result from 1/2
to 2/2, and a silently edited task is the same class of error.

`needs_python` marks the five tasks that cannot be answered without running
code. Task 6 is the control in the other direction: a model that reaches for
the tool to answer "what is the capital of France" has learned to call tools,
not to decide.
"""

from __future__ import annotations

# Re-exported: the loop's own copy lives in `step`, outside the evaluation
# side, and this keeps every existing `from alchemist_rlm.tasks import Step`
# working while the dependency itself runs the right way.
from alchemist_rlm.step import Step  # noqa: F401

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Callable

# A small deterministic corpus. Deliberately not OOLONG: Gate A must not be
# scoreable by the classification atom, which is already known to be 38%–73%
# for the Alchemist and would confound a protocol result with a semantic one.
LEDGER = "\n".join(
    f"{i:03d} | {dept} | {amount} | {status}"
    for i, (dept, amount, status) in enumerate(
        [
            ("logistics", 1420, "paid"), ("kitchen", 780, "pending"),
            ("logistics", 310, "paid"), ("front", 95, "void"),
            ("kitchen", 2050, "paid"), ("logistics", 640, "pending"),
            ("front", 1180, "paid"), ("kitchen", 415, "void"),
            ("logistics", 2270, "paid"), ("front", 530, "pending"),
            ("kitchen", 1325, "paid"), ("logistics", 860, "paid"),
        ],
        start=1,
    )
)

# Truths computed here once, from the same literal the model will see.
_ROWS = [line.split(" | ") for line in LEDGER.splitlines()]
_PAID_TOTAL = sum(int(r[2]) for r in _ROWS if r[3] == "paid")
_LOGISTICS_COUNT = sum(1 for r in _ROWS if r[1] == "logistics")


def _has_number(value: int) -> Callable[[str], bool]:
    def check(reply: str) -> bool:
        # Accept thousands separators; reject a substring match inside a longer
        # number, which is how a loose scorer turns a wrong answer into a pass.
        digits = {int(m.replace(",", "").replace(".", ""))
                  for m in re.findall(r"\d[\d,.]*", reply or "")}
        return value in digits
    return check


def _is_identifier(value: int) -> Callable[[str], bool]:
    """Match a whole numeric token, never a substring.

    `"008" in "1008"` is True, so a substring scorer passes a wrong row id. That
    is the same defect that once made `probe_03` match the true count inside
    `"Note 35"` and report a pass on a failed run. Compared as integers, so a
    reply of `8`, `008` or `row 008` all pass and `1008` does not.
    """
    def check(reply: str) -> bool:
        return value in {int(t) for t in re.findall(r"\b\d+\b", reply or "")}
    return check


def _has_text(needle: str) -> Callable[[str], bool]:
    def check(reply: str) -> bool:
        return needle.lower() in (reply or "").lower()
    return check


def _identifiers(code: str) -> set[str]:
    return set(re.findall(r"\b[A-Za-z_]\w*\b", code or ""))


def _reuses_an_earlier_variable(steps: list[Step]) -> bool:
    """At least two calls, and a later one reads a name an earlier one bound.

    Without this, `a4` is scored purely on its final number — and a model that
    computes the whole thing in a single call passes a task whose entire purpose
    is to prove the Python session survives between calls.
    """
    executed = [s for s in steps if not s.refused]
    if len(executed) < 2:
        return False
    bound: set[str] = set()
    for index, step in enumerate(executed):
        if index and (bound & _identifiers(step.code)):
            return True
        bound |= set(step.defined)
    return False


def _changed_action_after_refusal(steps: list[Step]) -> bool:
    """After a refusal, the next executed call must be different code.

    `probe_08` measured that a bare refusal produced no behavioural change at
    all; this is the scorer for whether the counteroffer does better.
    """
    from alchemist_rlm.protocol import code_key

    for index, step in enumerate(steps):
        if not step.refused:
            continue
        refused_key = code_key(step.code)
        return any(code_key(later.code) != refused_key for later in steps[index + 1:])
    return False


@dataclass(frozen=True)
class Task:
    """One scored task: a question, an optional context, and how to judge it.

    Result and process are judged separately and never collapsed. A right answer
    reached the wrong way is not a pass here, and a wrong answer reached the
    right way is a different failure from a wrong answer reached by guessing.
    """
    id: str
    question: str
    context: str | None
    truth: Any
    needs_python: bool
    scores: Callable[[str], bool]
    note: str
    # Some tasks cannot be judged from the reply alone. A correct final number
    # says nothing about whether the session persisted or whether the model
    # recovered from a refusal, so those are scored on the trajectory.
    scores_process: Callable[[list[Step]], bool] | None = None
    # Deterministic condition the runner must create. Waiting for the model to
    # produce a duplicate by chance would make the task untestable on the runs
    # where it simply does not repeat itself.
    inject: str | None = None

    def passed(self, reply: str, steps: list[Step] | None = None) -> bool:
        """True only if the answer is right AND the trajectory is acceptable. This
        repository has a substring scorer on record that turned a failed run into a
        pass, which is why the two halves are checked separately and both must hold.
        """
        if not self.scores(reply):
            return False
        if self.scores_process is None:
            return True
        return self.scores_process(steps or [])

    def to_dict(self) -> dict[str, Any]:
        """The task as plain data, so a suite can be hashed and frozen. What was asked
        has to be pinned before a model sees it, or a later edit silently rewrites
        the question a result was measured against.
        """
        return {
            "id": self.id,
            "question": self.question,
            "context_sha256": (
                hashlib.sha256(self.context.encode()).hexdigest() if self.context else None
            ),
            "truth": self.truth,
            "needs_python": self.needs_python,
            "scores_process": self.scores_process.__name__ if self.scores_process else None,
            "inject": self.inject,
            "note": self.note,
        }


GATE_A: tuple[Task, ...] = (
    Task(
        id="a1_exact_arithmetic",
        question="What is 7919 multiplied by 6113? Reply with the number only.",
        context=None,
        truth=7919 * 6113,
        needs_python=True,
        scores=_has_number(7919 * 6113),
        note="Large-prime product: reliably wrong mentally, exact in Python.",
    ),
    Task(
        id="a2_regex_extract",
        question=(
            "In the data, every row is 'id | department | amount | status'. "
            "How many rows have department 'logistics'? Reply with the number only."
        ),
        context=LEDGER,
        truth=_LOGISTICS_COUNT,
        needs_python=True,
        scores=_has_number(_LOGISTICS_COUNT),
        note="Deterministic filtering over external data.",
    ),
    Task(
        id="a3_group_and_sum",
        question=(
            "In the data, sum the amount column across every row whose status is "
            "'paid'. Reply with the number only."
        ),
        context=LEDGER,
        truth=_PAID_TOTAL,
        needs_python=True,
        scores=_has_number(_PAID_TOTAL),
        note="Grouping plus arithmetic; the aggregation must happen in Python.",
    ),
    Task(
        id="a4_persistence",
        question=(
            "First store the number of rows in the data in a variable called n. "
            "Then, in a later step, report n multiplied by 7. Reply with the "
            "final number only."
        ),
        context=LEDGER,
        truth=len(_ROWS) * 7,
        needs_python=True,
        scores=_has_number(len(_ROWS) * 7),
        scores_process=_reuses_an_earlier_variable,
        note=(
            "Requires the session to survive between two tool calls. Scored on "
            "the trajectory as well as the answer: a single call that computes "
            "84 outright is a correct number and a failed persistence test."
        ),
    ),
    Task(
        id="a5_duplicate_recovery",
        question=(
            "Find the id of the row with the largest amount among rows whose "
            "status is 'void'. Reply with the id only."
        ),
        context=LEDGER,
        truth="008",
        needs_python=True,
        scores=_is_identifier(8),
        scores_process=_changed_action_after_refusal,
        inject="refuse_first_call_as_duplicate",
        note=(
            "The task is easy; the point is the recovery. The runner refuses the "
            "first call with the duplicate_call observation and its next_actions "
            "menu, so the condition occurs on every run instead of only when the "
            "model happens to repeat itself. Passes only if it then emits "
            "different code and still reaches the answer."
        ),
    ),
    Task(
        id="a6_no_tool_needed",
        question="What is the capital of France? Answer directly.",
        context=None,
        truth="Paris",
        needs_python=False,
        scores=_has_text("paris"),
        note=(
            "Control in the other direction. Reaching for the tool here is a "
            "failure of judgement, not of capability."
        ),
    ),
)

TASKS_SHA256 = hashlib.sha256(
    json.dumps([t.to_dict() for t in GATE_A], sort_keys=True).encode()
).hexdigest()

PYTHON_TASKS = tuple(t for t in GATE_A if t.needs_python)
