"""V2: the two capabilities V1 could not test, on a context forty times larger.

V1 is frozen and not touched. It stands as the record of what was measured, and
a test asserts its hash is unchanged. These are new tasks under a new
configuration, which is what the plan requires of anything learned after seeing
a trace.

Each one exists because a V1 result turned out to be unfalsifiable:

  **t08v2** — V1's batching task listed its own keywords in the question, so the
  model searched for them and never batched. Here the criterion crosses
  vocabularies (weather, a power cut, a strike, a broken forklift) and the
  vocabulary crosses the criterion (208 records mention fog, rain, wind or ice
  while work carried on). A keyword search returns 488; the truth is 146. A
  lexical answer is not merely wrong, it is recognisably lexical.

  **t09v2** — V1's recursion task was answered correctly without recursing, so
  the pass condition now reads the ledger: a child node has to exist and it has
  to have run code. A right answer is not evidence that recursion happened.

Both requirements are read from instrumentation added after V1: `used_batched_api`
and `consumed_lazily` come from the scheduler and the REPL's lazy-pull counter,
not from a subcall count that four sequential calls would satisfy.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from alchemist_rlm import corpus_v2
from alchemist_rlm.suite import SuiteTask, _has_number, attribute_oolong

TASKS_V2: tuple[SuiteTask, ...] = (
    SuiteTask(
        id="t08v2_semantic_sweep_200k",
        question=(
            "Every record ends with a 'Note:' line. In some of them the note "
            "says that work was stopped, halted, held up or delayed for some "
            "reason. In others the note simply describes conditions, or says "
            "work went ahead normally. How many records report that work was "
            "actually stopped or delayed? Every record must be examined. Reply "
            "with the number only."
        ),
        context=corpus_v2.CORPUS_V2,
        truth=corpus_v2.TRUTHS_V2["stoppages"],
        scores_result=lambda r: _has_number(r, corpus_v2.TRUTHS_V2["stoppages"]),
        requires={
            "used_the_tool": True,
            "used_batched_api": True,       # the batched path, not four calls in a row
            "consumed_lazily": True,        # pulled in chunks, never materialised
            "coverage_complete": True,      # all 202,819 characters reached a sub-model
            "answer_came_from_python": True,
            "clean_termination": True,
        },
        note=(
            f"{corpus_v2.TRUTHS_V2['chars']:,} characters, "
            f"{corpus_v2.TRUTHS_V2['records']} records, truth "
            f"{corpus_v2.TRUTHS_V2['stoppages']}. A keyword search over the "
            f"weather words returns {corpus_v2.TRUTHS_V2['keyword_search_would_return']}, "
            "so a lexical answer identifies itself."
        ),
    ),
    SuiteTask(
        id="t09v2_recursion_verified",
        question=(
            "The data is far too large to read in one pass. Split it into parts "
            "and give each part to a separate recursive call with "
            "rlm_query(question, part), letting each one work out its own part's "
            "answer. Then combine them. In how many records is the Crew line "
            "'night'? Reply with the number only."
        ),
        context=corpus_v2.CORPUS_V2,
        truth=corpus_v2.TRUTHS_V2["night_crew_count"],
        scores_result=lambda r: _has_number(r, corpus_v2.TRUTHS_V2["night_crew_count"]),
        requires={
            "used_the_tool": True,
            "recursion_observed": True,
            "child_did_work": True,         # a child node that actually ran code
            "clean_termination": True,
        },
        note=(
            "V1's recursion task was answered correctly with no recursion at "
            "all, and passed its result layer. The pass condition now reads the "
            "ledger and the trace: a child node has to exist and to have run "
            "something. The prior remains the lowest of the suite."
        ),
    ),
)

# V3: same corpus, new configuration. The system prompt now carries the
# operation-choice policy, llm_query refuses oversized sources, and rlm_map
# exists. The recursion question names the *shape* of the work — delegate each
# part to its own analysis — but no function: whether the model finds rlm_map /
# rlm_query from the documented interface is exactly what is being measured.
# What a pass means is bounded and recorded: "operates recursion through a
# usable interface", not "decomposes spontaneously".
TASKS_V3: tuple[SuiteTask, ...] = (
    SuiteTask(
        id="t09v3_recursion_interface",
        question=(
            "Work depot by depot: give each depot's records to its own separate "
            "analysis, let each one work out how many of its records report "
            "that work was actually stopped or delayed, and then combine the "
            "results. How many such records are there in total? Reply with the "
            "number only."
        ),
        context=corpus_v2.CORPUS_V2,
        truth=corpus_v2.TRUTHS_V2["stoppages"],
        scores_result=lambda r: _has_number(r, corpus_v2.TRUTHS_V2["stoppages"]),
        requires={
            "used_the_tool": True,
            "recursion_observed": True,
            "child_did_work": True,
            "clean_termination": True,
        },
        note=(
            "The question names the shape (delegate each part) but no function "
            "name. The per-part work is semantic — stoppage vs carried-on notes "
            "share vocabulary — so plain counting cannot answer it, and the "
            "202,819-character context cannot be read in one pass."
        ),
    ),
)

# V4: same corpus, same truth, the literal frame removed from the question.
# t08v2 asked for notes that "say work was stopped, halted, held up or
# delayed" — and the whole pipeline answered exactly that literal question:
# the root copied the wording into its goal, the sub-models matched the words,
# and 54 is the count of notes containing them. The atomic A/B measured the
# same sub-model at 96% recall with a semantic framing, so the corrected task
# asks about interruption in substance and leaves the wording open.
TASKS_V4: tuple[SuiteTask, ...] = (
    SuiteTask(
        id="t08v3_semantic_sweep_neutral",
        question=(
            "Every record ends with a 'Note:' line. Some notes report that the "
            "work was interrupted, prevented or delayed — in whatever words, by "
            "whatever cause. Others describe conditions or normal operations. "
            "How many records report an actual interruption or delay? Every "
            "record must be examined. Reply with the number only."
        ),
        context=corpus_v2.CORPUS_V2,
        truth=corpus_v2.TRUTHS_V2["stoppages"],
        scores_result=lambda r: _has_number(r, corpus_v2.TRUTHS_V2["stoppages"]),
        requires={
            "used_the_tool": True,
            "used_batched_api": True,
            "consumed_lazily": True,
            "coverage_complete": True,
            "answer_came_from_python": True,
            "clean_termination": True,
        },
        note=(
            "Identical corpus and truth to t08v2; only the framing changed. "
            "Comparing the two runs isolates what the question's own literalism "
            "cost, with the template held fixed."
        ),
    ),
)

# t09v4 mirrors for recursion the correction t08v3 made for the sweep. t09v3's
# question said "report that work was actually stopped or delayed" -- the same
# literal frame t08v2 had -- and the root copies each question's frame into its
# goal verbatim: 143 under the neutral frame, 112 under the literal one, with
# the misses concentrating precisely on the phrasings that lack a stop-word.
# Scoring recursion against the sweep with unequal questions was our error;
# the pair t09v3/t09v4 is the ablation, exactly like t08v2/t08v3.
TASKS_V5: tuple[SuiteTask, ...] = (
    SuiteTask(
        id="t09v4_recursion_neutral",
        question=(
            "Work part by part: give each part of the data to its own separate "
            "analysis, let each one work out how many of its records report "
            "that the work was interrupted, prevented or delayed — in whatever "
            "words, by whatever cause — and then combine the results. How many "
            "such records are there in total? Every record must be examined. "
            "Reply with the number only."
        ),
        context=corpus_v2.CORPUS_V2,
        truth=corpus_v2.TRUTHS_V2["stoppages"],
        scores_result=lambda r: _has_number(r, corpus_v2.TRUTHS_V2["stoppages"]),
        requires={
            "used_the_tool": True,
            "recursion_observed": True,
            "child_did_work": True,
            "clean_termination": True,
        },
        note=(
            "Identical corpus and truth to t09v3; only the framing changed, "
            "mirroring t08v2 -> t08v3. Registered prediction before running: "
            "~140 +/- 6 if the phrasing analysis is complete; materially below "
            "that means something else remains."
        ),
    ),
)

TASKS_V2_BY_ID = {task.id: task
                  for task in TASKS_V2 + TASKS_V3 + TASKS_V4 + TASKS_V5}
SUITE_V2_SHA256 = hashlib.sha256(
    json.dumps([t.to_dict() for t in TASKS_V2], sort_keys=True).encode()
).hexdigest()


def lexical_tell(answer: str) -> dict[str, Any]:
    """Was this the answer a keyword search produces?

    Named in advance so the failure mode is legible in the result rather than
    merely wrong. The plan's whole method is separating *which* thing failed.
    """
    truths = corpus_v2.TRUTHS_V2
    if _has_number(answer or "", truths["keyword_search_would_return"]):
        return {"verdict": "lexical_answer",
                "meaning": f"answered {truths['keyword_search_would_return']}, which is "
                           "exactly what a keyword search over the weather words "
                           "returns. The records were matched, not read."}
    if _has_number(answer or "", truths["stoppages"]):
        return {"verdict": "answered"}
    return {"verdict": "other_wrong_answer",
            "meaning": "neither the truth nor the lexical number; read the coverage "
                       "and the subcall responses in the trace"}
