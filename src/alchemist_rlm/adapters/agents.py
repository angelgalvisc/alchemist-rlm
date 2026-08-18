"""The reason the project exists: a RLM an agent can call as one tool.

The user's framing was that this should be *usable inside agents*, not that it
should pass the official benchmark. So the surface is a single function with an
OpenAI-shaped schema: an agent hands over a body of text too large for its own
window and a question, and gets back an answer plus a pointer to the trace that
produced it.

`evidence` is returned alongside the answer because an agent that cannot see the
work cannot decide whether to trust it, and this model's characteristic failure
mode is a confident fabricated name.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from alchemist_rlm.budgets import Budget
from alchemist_rlm.engine import RLMEngine
from alchemist_rlm.tracing import Trace, covered, digest

TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "analyze_large_context",
        "description": (
            "Answer a question about a body of text that is too large to read "
            "directly. The text is processed by a recursive language model: it is "
            "segmented, searched and read in pieces by sub-models, and the answer "
            "is assembled programmatically. Returns the answer with the evidence "
            "it rests on."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "context": {"type": "string",
                            "description": "The full text to analyse. May be very large."},
                "question": {"type": "string",
                             "description": "What to find out about it."},
                "strategy": {
                    "type": "string",
                    "enum": ["auto", "map", "recursive", "classify"],
                    "description": (
                        "auto lets the model choose its own approach. map "
                        "requires a batched semantic pass over every segment — "
                        "use for exhaustive sweeps. recursive requires "
                        "delegating parts to child analyses — use when parts "
                        "need several steps each. classify requires one label "
                        "out of a fixed set for every item — use when the "
                        "question names the categories."
                    ),
                },
            },
            "required": ["context", "question"],
        },
    },
}

# The V1 suite measured that a 4B does not reliably *select* map or recursion
# on its own even when the task calls for one. An agent caller usually knows
# which shape its question has, so letting it fix the strategy makes the tool
# dependable without asking the small model to be a planner. The directive is
# generic — it names an approach, never a task or an answer.
STRATEGY_DIRECTIVES = {
    "auto": "",
    # Only routes whose coverage the harness verifies are directed. An earlier
    # wording also sanctioned partition_context() with llm_query_batched — a
    # route that produces no validated sweep, so an episode following it to
    # the letter could never satisfy the strategy it was directed into.
    "map": (
        "Strategy requirement: this question needs an exhaustive semantic pass "
        "with verified coverage. Use semantic_search(goal) — or "
        "semantic_map(instruction, schema) when every item should yield a "
        "value of a fixed shape — so that every segment is read by a "
        "sub-model and the coverage is validated. Do not answer from a "
        "partial read."
    ),
    "recursive": (
        "Strategy requirement: this question needs delegation. Split the data "
        "into parts and hand each part to its own analysis with "
        "rlm_map(question, parts) or rlm_query(question, part), then combine "
        "the parts' answers."
    ),
    # Names the operation and stops. What the categories are, what they mean and
    # what to do with the result belong to the question — a directive supplying
    # any of those would be measuring the directive.
    "classify": (
        "Strategy requirement: this question needs one label out of a fixed set "
        "for every item. Use semantic_map(instruction, schema) with "
        "schema={'type': 'string', 'enum': [...]} listing the categories the "
        "question names, then aggregate the returned result['rows'] in "
        "Python. Do not answer from a partial read, and do not label the "
        "items yourself."
    ),
}


# What each directed strategy requires to have actually run. `auto` requires
# nothing, so its satisfaction is never asserted.
_REQUIRED_OPERATIONS = {
    "classify": ("semantic_map",),
    "map": ("semantic_map", "semantic_search"),
    "recursive": ("rlm_map", "rlm_query"),
}
# The strategies whose meaning includes exhaustiveness: their answer is only
# deliverable when the context sweep is established as complete.
_EXHAUSTIVE = ("classify", "map")


def _strategy_satisfied(strategy: str, observed: list[str],
                        sweeps: list[dict[str, Any]],
                        recursive_complete: bool | None = None,
                        ) -> tuple[bool | None, dict[str, Any] | None]:
    """One definition, because two reviews of the same run disagreed on it.

    None is "nothing was asserted", not "we do not know which": `auto`
    requires nothing, and a delegation strategy may have run without enough
    uniquely placed, child-certified spans to establish end-to-end coverage.
    False is reserved for the case that can be asserted: the required operation
    never ran, or measured exhaustiveness is incomplete.

    For an exhaustive strategy, both facts must come from ONE sweep — the
    operation and its completeness. Combining the session-wide operation list
    with the last sweep's completeness was measured to fail open: a classify
    run whose enum map covered hand-picked items, followed by a complete
    boolean search over the context, reported satisfied by two operations
    neither of which alone satisfies classify. The grounding sweep is
    returned with the verdict so the caller can report the coverage of the
    sweep that actually earned it.
    """
    if strategy == "auto":
        return None, None
    required = _REQUIRED_OPERATIONS[strategy]
    if not any(op in observed for op in required):
        return False, None
    if strategy in _EXHAUSTIVE:
        for sweep in reversed(sweeps):
            if (sweep.get("operation") in required
                    and sweep.get("context_coverage_complete") is True):
                return True, sweep
        return False, None
    return recursive_complete, None


def _recursive_certificate(context: str, trace_path: Path | str,
                           ) -> dict[str, Any] | None:
    """Coverage established by complete child sweeps in the root frame.

    ``delegated_span`` is emitted only after a child establishes complete
    coverage of its own context and that context is uniquely located in the
    parent.  Composing those spans here closes the adapter path without asking
    the model whether it covered the whole.
    """
    spans = [tuple(event["span"]) for event in Trace.read(trace_path)
             if event.get("kind") == "delegated_span"
             and event.get("of") == digest(context)]
    if not spans:
        return None
    fraction = covered(context, spans)
    merged: list[list[int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([int(start), int(end)])
    gaps, cursor = [], 0
    for start, end in merged:
        if start > cursor:
            gaps.append([cursor, start])
        cursor = max(cursor, end)
    if cursor < len(context):
        gaps.append([cursor, len(context)])
    return {
        "source_digest": digest(context), "source_chars": len(context),
        "covered_fraction": round(fraction, 6),
        "complete": fraction >= 0.98,
        "gaps": gaps,
        "means": ("each credited slice was uniquely placed and its child "
                  "established complete context coverage; this says nothing "
                  "about semantic correctness or aggregation correctness"),
    }


def analyze_large_context(
    context: str,
    question: str,
    *,
    strategy: str = "auto",
    engine: RLMEngine | None = None,
    budget: Budget | None = None,
    run_id: str | None = None,
    **engine_kwargs: Any,
) -> dict[str, Any]:
    """Run one episode and return a result an agent can act on.

    The result fails closed. The previous shape reduced everything to
    `answered: episode.answer is not None`, and the directed t14 run showed
    what that hides: a sweep that validated 755 of 795 units, an aggregation
    over the incomplete table, and a budget-forced ending would all have been
    presented to the calling agent as an answered question. The layers below
    knew — `coverage_complete: False` was computed correctly — and this was
    the layer that dropped it. So what the caller gets now is the set of
    facts, separated: whether text came back (`answer_available`), what
    actually ran (`operations_observed` vs `strategy_requested`), what each
    sweep established (`coverage`, from the sweep that grounds the verdict),
    and whether the answer is deliverable under the strategy's own meaning
    (`answer_valid`, which is the satisfaction verdict applied — never a
    parallel derivation). A partial result is never suppressed; it is
    labelled.
    """
    if strategy not in STRATEGY_DIRECTIVES:
        raise ValueError(f"strategy must be one of {sorted(STRATEGY_DIRECTIVES)}")
    directive = STRATEGY_DIRECTIVES[strategy]
    asked = f"{question}\n\n{directive}" if directive else question
    engine = engine or RLMEngine(budget=budget or Budget(), **engine_kwargs)
    episode = engine.complete(context, asked, run_id=run_id)

    sweeps = list(episode.sweeps or [])
    observed = list(dict.fromkeys(episode.operations))
    recursive_certificate = (
        _recursive_certificate(context, episode.trace_path)
        if strategy == "recursive" else None
    )
    satisfied, grounding = _strategy_satisfied(
        strategy, observed, sweeps,
        recursive_complete=(recursive_certificate or {}).get("complete"),
    )
    # Coverage is reported for the sweep that grounds the verdict when one
    # does, and for the last sweep otherwise — labelled with its operation
    # either way, so a caller never mistakes a boolean search's coverage for
    # an enum classification's.
    basis = grounding or (sweeps[-1] if sweeps else None)
    coverage = None
    if basis:
        coverage = {
            "kind": basis.get("kind"),
            "operation": basis.get("operation"),
            "scope": basis.get("scope"),
            "status": basis.get("status"),
            "valid": basis.get("valid_items"),
            "total": basis.get("total_items"),
            "complete": basis.get("coverage_complete"),
            "context_complete": basis.get("context_coverage_complete"),
            "failed_items": basis.get("failed_items"),
            "failed": basis.get("failed"),
            "sweep_id": basis.get("sweep_id"),
            "retry_exhausted": basis.get("retry_exhausted"),
            # The validated table itself, by reference and digest — hundreds
            # of rows belong in an artifact an agent can fetch and check, not
            # in every response.
            # `rows_digest` is the sha256 of the rows' canonical JSON — the
            # content — and is verified by re-canonicalising what the artifact
            # parses to. It is deliberately not the artifact's own
            # `sha256`, which covers the stored bytes under the store's own
            # serialisation. Two different questions, two different hashes.
            "rows_ref": basis.get("rows_ref"),
            "rows_digest": basis.get("rows_digest"),
        }
    answer_available = episode.answer is not None
    # A delivery cut off mid-sentence is not a delivery. The forced-final path
    # marks it in the stop reason, and t14 ended there: a complete sweep with
    # a truncated textual final was reported deliverable and complete, which
    # is the same substitution as before with the cut one step later.
    # `:submitted` is exempt on purpose — that value came out of the session,
    # so the reply's tokens running out does not touch it.
    truncated_delivery = str(episode.stop_reason or "").endswith(":truncated")
    # Deliverability is the satisfaction verdict, not a parallel derivation:
    # computing it separately from context_complete alone was measured to
    # declare a classify answer complete on the strength of a boolean sweep,
    # with strategy_satisfied False in the same payload. False blocks; None
    # (nothing was demanded, or nothing establishable) does not.
    answer_valid = (answer_available and satisfied is not False
                    and not truncated_delivery)
    if not answer_available:
        status = "failed"
    elif not answer_valid:
        status = "partial"
    elif strategy == "recursive" and satisfied is not True:
        # Delegation ran, but this trace did not establish that certified child
        # spans cover the whole. "complete" would be stronger than the evidence.
        status = "unverified"
    elif strategy == "auto" and basis is None:
        # Auto imposes no operation.  The answer remains deliverable, but a
        # direct assertion with no grounding is not thereby verified.
        status = "unverified"
    elif basis is not None and basis.get("status") != "complete":
        # From the sweep this result rests on — never from every sweep the
        # episode ever ran. An earlier failed attempt that a later grounding
        # sweep superseded was marking a fully grounded answer partial.
        status = "partial"
    else:
        status = "complete"

    return {
        "answer": episode.answer,
        # The same answer as the session held it — a list stays a list, a count
        # stays an int — so a caller that wants the data does not parse the
        # text back out. `None` here is ambiguous on its own (no delivery, or a
        # delivered `None`), which is what `answer_typed` resolves.
        "answer_value": episode.answer_value,
        "answer_typed": episode.answer_delivered,
        "answer_available": answer_available,
        "answer_valid": answer_valid,
        "status": status,
        "strategy_requested": strategy,
        "operations_observed": observed,
        "strategy_satisfied": satisfied,
        "coverage": coverage,
        # The certificate OF THE SWEEP REPORTED IN `coverage`, and of no
        # other: spans and validation composed by the harness, with its own
        # `means` disclaimer — coverage and valid shape, never semantic
        # correctness. None when that sweep has none, which is the honest
        # answer for a provided-items sweep. Other sweeps' certificates stay
        # in the episode record rather than being mixed in here.
        "certificate": (recursive_certificate if strategy == "recursive"
                        else (basis or {}).get("certificate")),
        "stop_reason": episode.stop_reason,
        "evidence": evidence(episode.trace_path),
        "cost": {
            "turns": episode.turns,
            "subcalls": episode.ledger.get("subcalls", 0),
            "semantic_cache_hits": len(episode.semantic_cache_hits),
            "seconds": round(episode.seconds, 1),
        },
        "trace": str(episode.trace_path),
    }


def evidence(trace_path: Path | str, limit: int = 8) -> list[dict[str, Any]]:
    """Which slices were actually read, and what came back from each.

    Taken from the trace rather than from the model's summary on purpose: the
    model's account of what it read is exactly the thing under suspicion.
    """
    out: list[dict[str, Any]] = []
    for event in Trace.read(trace_path):
        if event.get("kind") != "subcall" or event.get("error"):
            continue
        answer = (event.get("response") or "").strip()
        if not answer or answer.upper().startswith("NONE"):
            continue
        out.append({
            "source_ref": event.get("source_ref"),
            "source_chars": (event.get("source") or {}).get("chars"),
            "source_sha256": (event.get("source") or {}).get("sha256"),
            "excerpt": (event.get("source") or {}).get("preview", "")[:300],
            "said": answer[:300],
        })
        if len(out) >= limit:
            break
    return out
