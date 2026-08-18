"""The outer loop, in the model's own protocol.

Everything here follows from one finding: the earlier evaluation never sent
`tools`, so the tool-calling branch of the Alchemist's chat template never
activated and the model was judged on an interface it was not trained for. This
loop sends the schema and lets `mlx_lm`'s `qwen3_coder` parser do the parsing —
there is deliberately no second regex parser, because a second parser is a
second thing to get wrong.

Termination is four-way and each branch was needed:

  - `<answer>` closes, because that is the tag this family was trained on.
  - A turn with no tool call is terminal, because the checkpoint's own README
    warns that a loop waiting only for the tag will "spin past a perfectly good
    reply".
  - A `submit()` call closes, because a RLM's answer may be built
    programmatically and may be larger than the window the model writes from.
  - The budget closes, with one forced final turn: a run that stops with no
    answer teaches nothing, and 919.9 seconds of that is already on record.
"""

from __future__ import annotations

import ast
import json
import hashlib
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from alchemist_rlm import protocol
from alchemist_rlm.budgets import Budget
from alchemist_rlm.output_contract import (
    OutputContract,
    TerminalPolicy,
    validation_feedback,
    validate_output_mode,
)
from alchemist_rlm.protocol import CallLedger, ParsedTurn, ToolCall

# One sentence per option, and only options that exist. The version this
# replaces offered a `FINAL_VARIABLE: <name>` line, which asked the model to
# *name* its answer in prose so the harness could go and fetch it — a second
# delivery channel, parsed out of tokens, with its own failure modes: a reply
# of exactly "FINAL_VARIABLE: pairs" over a missing name was once recorded as
# the episode's answer. `submit` is already the delivery channel and works here
# exactly as it works on every other turn.
COMMIT_FIRST = (
    "This is the commit phase. Do not explain or plan.\n\n"
    "Make one short Python call. Use only values already in the session and "
    "local Python — do not start another model or semantic operation. Never "
    "recreate, enumerate, copy, or inline a computed collection: reference its "
    "existing variable by name.\n\n"
    "In that same block:\n"
    "1. Finish the minimum remaining local aggregation, if necessary.\n"
    "2. If the original question asks for an exact textual shape, build that "
    "text with a short Python expression over the existing computed value; "
    "do not spell out its items.\n"
    "3. End with submit(value), or with "
    "submit(value, final_text=final_text) when you built a distinct textual "
    "presentation."
)

COMMIT_SECOND = (
    "Nothing was delivered in the previous commit turn. This is the final "
    "auxiliary turn; there will not be another one.\n\n"
    "Make one short Python call using only session state and local Python. "
    "Reference the existing result variable directly; never recreate, "
    "enumerate, copy, or inline its items. Then "
    "end it with submit(value), or submit(value, final_text=final_text) when "
    "the question requires a distinct textual presentation. If no Python value "
    "can be delivered, give the "
    "best genuinely short answer inside <answer></answer> tags. Do not explain "
    "or plan another step."
)

COMMIT_PARTIAL = (
    "This is the terminal coverage-recovery and commit phase. Do not explain "
    "or plan.\n\n"
    "The last registered semantic sweep is partial: {valid} of {total} items "
    "have validated values, and its one targeted retry is still available. "
    "Use the existing envelope bound as semantic_result. You may call "
    "retry_failed(semantic_result) exactly once; do not call semantic_map, "
    "semantic_search, llm_query or llm_query_batched.\n\n"
    "In one short Python block, retry that existing sweep if complete coverage "
    "is needed, finish only the minimum local aggregation from session state, "
    "and end with submit(value), or submit(value, final_text=final_text) when "
    "you built a distinct textual presentation. Never recreate, enumerate, "
    "copy or inline a computed collection."
)

COMMIT_PARTIAL_SECOND = (
    "Nothing was delivered. This is the final auxiliary turn. The registered "
    "partial sweep still has its one targeted retry available. In one short "
    "Python block call retry_failed(semantic_result) at most once, use the "
    "returned envelope and existing session state for the minimum local "
    "aggregation, then submit. Do not call semantic_map, semantic_search, "
    "llm_query or llm_query_batched, and do not explain."
)

# Compatibility name for code that imported the former single forced-final
# message. It now denotes the first state of the finishing machine.
FORCE_FINAL = COMMIT_FIRST

# The conformance turn. It fires whenever an episode delivered a value, on any
# exit, and it names no format — the harness does not know what shape a given
# question asks for and must not learn one. A harness that checks "is this
# `(a, b)`?" has stopped measuring the model and started scoring against a rule
# it wrote itself, which is the same defect as every other task-specific
# heuristic this project has removed.
#
# Measured reason it exists. Of twenty OOLONG-Pairs queries, four delivered
# their pairs as `10352, 12455` — one per line, no parentheses — against a
# query that asks for "the format (user_id_1, user_id_2)". Their content F1 was
# 0.952, 0.923, 0.364 and 0.164; under the paper's own parser all four scored
# 0.000. Four more delivered a list of tuples that the JSON transport rendered
# as `[["10352", "12455"]]`. The pairs were computed and the answer was
# unreadable, and that had been scoring identically to never computing them.
#
# Two properties make this measurable rather than a thumb on the scale. It
# fires *after* the answer exists and nothing follows it, so the answer it
# replaces is exactly what the run would have produced without it: one
# trajectory read at two points, not two runs to be reconciled. And it asks for
# the correction in code, so the data never passes through the generator —
# 2,330 pairs retyped by a 4B is a truncation, the same pairs reformatted by a
# comprehension are the same pairs.
# The first sentence states the turn's price, and it was removed once on the
# argument that a harness's accounting is not the model's business. That
# argument was wrong, and the removal is the only reason it is back with a
# comment instead of no comment.
#
# The decision this turn asks for *is* a budget decision — "do I spend a turn
# re-checking?" — taken by a model that has been told about scarcity all
# episode, down to "this is your last turn" one message earlier. Withhold the
# price and it decides under a false belief about the cost of an action the
# harness is offering it. An offer whose price is hidden is not an offer.
#
# The line worth holding is elsewhere and this does not cross it: the sentence
# says what an action costs, never what the answer should contain. "Make sure
# you used parentheses" would be the thumb on the scale; "this turn is free" is
# a true fact about a choice the model is being asked to make.
#
# Measured, and thin: query 1 corrected itself with the sentence present and
# declined without it, from a byte-identical delivered value, with nothing else
# differing that the model could read. One observation each way — enough to
# undo a change made on a bad argument, not enough to call the sentence the
# cause. The clean test is one query, both wordings, one ceiling.
CONFORMANCE = (
    "One more turn, and it does not spend your budget.\n\n"
    "Re-read the question you were given at the start. If it asks for the "
    "answer in a particular shape or format, check that what you delivered is "
    "already in exactly that shape.\n\n"
    "Delivery is open again. If it is not right, call submit() once more with "
    "the corrected value — and build that value with code from what you "
    "already have, do not type it out. If what you delivered is already right, "
    "call nothing and say so."
)

# Sent only when the first conformance turn ran a block cleanly and delivered
# nothing — the state a model is in when it has built the corrected value and
# stopped to look at it. It names no format either; it says only that building
# is not delivering.
# The two siblings of CONFORMANCE_FINISH, for the other two states in which
# the granted turn provably accomplished nothing. All three are facts about the
# turn that just happened, none of them names a shape, and each is sent once.
#
# `SAME` is arithmetic, not judgement: the harness renders both values and they
# are equal, so the turn changed nothing whatever the model believed. Query 7
# reached it in forty-eight tokens — "The answer is already built and verified.
# I need to submit it now." — and re-sent `submit(pairs)` without looking.
#
# `ACT` is for a turn that spent its whole ceiling reasoning and never called.
# Query 19 got the diagnosis exactly right in its own words — "the question
# asks for pairs in the format (user_id_1, user_id_2) ... however, I submitted
# a list of lists" — and then wrote seventeen thousand characters of
# deliberation, ending mid-sentence on "Wait, I need to check if the format is
# correct." It did not decline. It ran out of room while thinking, which is a
# different failure and gets a different reply.
CONFORMANCE_SAME = (
    "That delivered the same value you had already delivered, byte for byte, "
    "so nothing about the answer changed.\n\n"
    "If its shape needed changing, the value has to change: build the "
    "corrected one and submit that. If it did not need changing, you are done "
    "and can call nothing."
)

CONFORMANCE_ACT = (
    "That turn ran out of room before you called anything, so nothing "
    "happened.\n\n"
    "Answer with one short block of Python and no explanation: submit the "
    "corrected value, or call nothing if what you delivered was already right."
)

CONFORMANCE_FINISH = (
    "That ran, and nothing was delivered. Building the value and delivering it "
    "are two separate acts. If the value you just built is the answer, deliver "
    "it now with submit(). If it is not, deliver whichever variable is."
)

OUTPUT_REPAIR = (
    "The computed answer is committed and cannot be changed. Its presentation "
    "did not satisfy the predeclared output contract. "
    "Continue in this same conversation and use PythonInterpreter; do not start "
    "a new analysis of the task. "
    "Use one short Python block and commit exactly one textual presentation. "
    "If the string variable is named candidate, the only valid calls are "
    "submit(candidate) or submit(final_text=candidate). Do not use any other "
    "keyword. Do not recompute "
    "the answer, consult another model, or explain. The persistent REPL exposes "
    "copies as PRESENTATION_VALUE, PRESENTATION_TEXT and PRESENTATION_CONTRACT.\n\n"
    "Contract:\n{specification}\n\nDiagnostics:\n{errors}\n\n"
    "Committed source shape:\n{source_shape}\n\nCurrent text:\n{preview}"
    "{checker_hint}"
)

OUTPUT_REPAIR_COMPACT = (
    "PRESENTATION-ONLY TERMINAL STATE. The answer is already computed and "
    "frozen; do not resume task analysis. Use PythonInterpreter once to build "
    "one complete str from PRESENTATION_DRAFT when it is bound, otherwise from "
    "PRESENTATION_VALUE or PRESENTATION_TEXT, and call submit(candidate), "
    "submit(final_text=candidate), or submit(result=candidate). These names are "
    "direct Python globals: write "
    "PRESENTATION_DRAFT directly, never look for it in sys.modules or a file. "
    "Do not inspect context, "
    "print variables, or explain. Diagnostics describe rejected candidate text; "
    "quoted excerpts are not examples of desired output.\n\nContract:\n"
    "{specification}\n\nDiagnostics:\n{errors}\n\nCommitted source shape:\n"
    "{source_shape}\n\n{source_hint}{checker_hint}"
)

PRESENTATION_ONLY_SYSTEM = (
    "You are in a presentation-only terminal state with one persistent Python "
    "tool. The answer is already frozen. Do not solve or inspect the task. "
    "Respond only by calling PythonInterpreter with one short Python block that "
    "builds a complete str from PRESENTATION_DRAFT when a rejected candidate "
    "exists, otherwise from PRESENTATION_VALUE or PRESENTATION_TEXT, and calls "
    "submit() exactly once. They are direct Python globals, not modules or "
    "files. Never copy or spell out collection items."
)

PRESENTATION_RETRY = (
    "No presentation was committed. The previous attempt failed as: {fact}. "
    "Printing, explaining or writing a code fence does not execute or commit text. "
    "Invoke PythonInterpreter through the provided tool interface; do not reply "
    "with prose or fenced code. If PRESENTATION_DRAFT is bound, it is the exact "
    "rejected str and must be the first source you transform. Do not submit "
    "PRESENTATION_VALUE itself: it may be a list or another non-str value. Use "
    "PRESENTATION_VALUE only to build a new complete str when the diagnostics "
    "show that the draft changed or omitted content. PRESENTATION_TEXT remains "
    "available as the original text. Reference these direct globals by name; "
    "do not search sys.modules or open a file. Do not copy text from the "
    "message. Build "
    "one str variable and call submit(candidate) positionally, "
    "submit(final_text=candidate), or submit(result=candidate). No other "
    "keyword is accepted."
)

PRESENTATION_VALIDATION_RETRY = (
    "A textual presentation was committed, but the predeclared contract "
    "rejected it, so it was not promoted. The computed answer remains frozen. "
    "The exact rejected model-authored bytes are now bound immutably as "
    "PRESENTATION_DRAFT. Correct that draft locally; if binding diagnostics "
    "show missing or changed content, rebuild from the complete "
    "PRESENTATION_VALUE instead. "
    "PRESENTATION_TEXT also remains available. These are direct Python globals, "
    "not modules or files. Do not copy an excerpt from the "
    "conversation, inspect or print the value: printing cannot commit a "
    "presentation. Build and submit one new complete str candidate with "
    "Python.\n\nContract:\n{specification}"
    "\n\nDiagnostics:\n{errors}\n\nPersistent source:\n{source_hint}"
    "{checker_hint}"
)

PRESENTATION_CHECKER_HINT = (
    "\n\nA presentation specification was inferred from the public question and "
    "frozen before task execution. PRESENTATION_SPEC and the read-only local "
    "function check_presentation(text) are now direct globals. The frozen spec "
    "is:\n{inferred_specification}\nBuild a new candidate string using those "
    "literal presentation fields. Before submit(), call "
    "`report = check_presentation(candidate)` and correct the candidate unless "
    "report['valid'] is true. The checker reports format only and never rewrites "
    "or submits text."
)

PRESENTATION_RENDERER_HINT = (
    "\n\nA presentation specification was inferred only from the public question "
    "and frozen before task execution. The committed value already has the "
    "declared primitive record structure. The preferred complete transaction is "
    "the single expression `submit(render_presentation(PRESENTATION_VALUE))`; "
    "do not inspect or print the value first. render_presentation uses only the "
    "frozen grammar, validates its own output, and never submits by itself. A "
    "successful render also stays available immutably as PRESENTATION_RENDERED. "
    "check_presentation(text) is available for a locally transformed candidate. "
    "The frozen spec is:\n{inferred_specification}"
)

PRESENTATION_PROGRESS = (
    "That block completed without committing a presentation and its persistent "
    "variables remain available. It counted as clean local progress ({used} of "
    "{limit} allowed construction turns). {detail} On the next turn, use those "
    "existing variables and submit one complete string; do not spell collection "
    "items into the tool call."
)

PRESENTATION_DRAFT_RETRY = (
    "The clean stdout was preserved exactly as PRESENTATION_DRAFT, but it was "
    "not promoted and validation rejected it. Its complete bytes stay inside "
    "the REPL. Correct the presentation locally from the persistent source and "
    "submit one complete string.\n\nDiagnostics:\n{errors}"
)

PRESENTATION_STDOUT_REJECTED_DRAFT_RETAINED = (
    "The diagnostic stdout was not a submitted presentation and did not replace "
    "the stronger model-authored candidate. The exact rejected candidate remains "
    "bound as PRESENTATION_DRAFT. Transform that direct global into one complete "
    "str and submit it; do not inspect or print it.\n\nDiagnostics:\n{errors}"
)

PRESENTATION_DRAFT_READY = (
    "The clean stdout is structurally valid and content-equivalent to the frozen "
    "answer. It was preserved exactly as PRESENTATION_DRAFT but has not been "
    "promoted. Confirm the model-authored bytes now with the single call "
    "submit(PRESENTATION_DRAFT)."
)

PRESENTATION_BUILD_LIMIT = (
    "Two clean local construction turns have been used. Persistent variables "
    "remain available; this turn must build if necessary and call submit() once."
)

PRESENTATION_COMMIT_RESERVE = (
    "The ordinary presentation window ended after this block made clean local "
    "progress. One reserved commit-only turn remains. Reuse the persistent "
    "variables, build the complete string if necessary, and call submit() "
    "exactly once. If render_presentation already succeeded, use the single "
    "call `submit(PRESENTATION_RENDERED)`. A block without submit() will be "
    "refused before execution."
)


def _repeated_tail(text: str, *, window: int = 24, least: int = 20) -> str | None:
    """The phrase a generation ended up repeating, if one took it over.

    Counted over the last few thousand characters with a fixed window, which is
    crude and is meant to be: the question is only whether one short string
    dominates the tail, and any phrase that appears twenty times in three
    thousand characters has. No parsing, no knowledge of what the text is
    about — a fact about the shape of the string, available to say back.

    Measured over every root generation on record that hit the ceiling: 26 of
    55 end this way, and what repeats is the corpus. `'how many', 'how many',`
    248 times; `'what is the current rate'` 107 times. The model is writing the
    data into its code as literals and degenerating partway, which is the one
    thing the whole contract exists to make unnecessary.
    """
    tail = text[-3000:]
    if len(tail) <= window:
        return None
    counts: dict[str, int] = {}
    for index in range(len(tail) - window):
        piece = tail[index:index + window]
        counts[piece] = counts.get(piece, 0) + 1
    phrase, seen = max(counts.items(), key=lambda item: item[1])
    return phrase if seen >= least else None


def _delivery_preview(value: Any) -> str:
    """Describe the delivered text without inviting literal reconstruction.

    The complete text and typed value already exist inside the presentation
    window. Repeating even a head/tail excerpt in the message made the model
    paste a truncated 80K answer into code and parse that literal instead of
    using those bindings. Length is enough orientation; source bytes stay in
    the REPL where they are complete.
    """
    text = _render(value)
    return (
        f"The current presentation is {len(text):,} characters. Its complete bytes "
        "are bound as PRESENTATION_TEXT and the committed source value is bound as "
        "PRESENTATION_VALUE. This message intentionally contains no excerpt: do not "
        "reconstruct, paste or parse a preview."
    )


def _value_shape(value: Any, *, depth: int = 0) -> str:
    """A bounded structural description with no answer content."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        return type(value).__name__
    if isinstance(value, str):
        return f"str[{len(value):,} chars]"
    if depth >= 3:
        return type(value).__name__
    if isinstance(value, list):
        if not value:
            return "list[0]"
        shapes = sorted({_value_shape(item, depth=depth + 1)
                         for item in value[:32]})
        item_shape = shapes[0] if len(shapes) == 1 else " | ".join(shapes[:4])
        if len(shapes) > 4:
            item_shape += " | ..."
        sample_note = "" if len(value) <= 32 else " (shape sampled from first 32)"
        return f"list[{len(value):,}] of {item_shape}{sample_note}"
    if isinstance(value, dict):
        shapes = sorted({_value_shape(item, depth=depth + 1)
                         for item in list(value.values())[:32]})
        value_shape = "empty" if not shapes else (
            shapes[0] if len(shapes) == 1 else " | ".join(shapes[:4]))
        return f"dict[{len(value):,}] with value types {value_shape}"
    return type(value).__name__


def _direct_renderer_compatible(value: Any, spec: dict[str, Any] | None) -> bool:
    """Whether advertising the one-call renderer preserves the value's shape.

    Flat strings may themselves encode records, but advertising a parser for
    them changed a previously successful model trajectory into repeated
    inspection. Keep that capable runtime fallback available without steering
    toward it. The direct hint is reserved for values that already expose the
    declared record boundaries structurally.
    """
    if not isinstance(spec, dict) or spec.get("kind") != "records":
        return False
    fields = spec.get("fields")
    if not isinstance(fields, list) or not isinstance(value, (list, tuple)):
        return False
    primitive = (str, int, float)
    return all(
        isinstance(record, (list, tuple))
        and len(record) == len(fields)
        and all(not isinstance(field, bool) and isinstance(field, primitive)
                for field in record)
        for record in value
    )


def _submitted_source_name(steps: list[Any]) -> str | None:
    """The persistent name passed directly to the successful root submit."""
    for step in reversed(steps):
        try:
            tree = ast.parse(step.code)
        except SyntaxError:
            continue
        calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "submit"
        ]
        for call in reversed(calls):
            if call.args and isinstance(call.args[0], ast.Name):
                return call.args[0].id
    return None


def _source_hint(steps: list[Any]) -> tuple[str | None, str]:
    name = _submitted_source_name(steps)
    if name is None:
        return None, (
            "Use the complete immutable PRESENTATION_VALUE rather than spelling "
            "or reconstructing its items."
        )
    return name, (
        f"The committed value was submitted directly from persistent variable "
        f"`{name}`. That model-created variable still exists; reuse it or the "
        "immutable PRESENTATION_VALUE rather than spelling its items."
    )


def _render(value: Any) -> str:
    """A delivered value as text, for consumers that need a string.

    A list of pairs or a table is a natural answer, and `str()` would make the
    episode's answer a Python repr; JSON is the same information and readable.
    This is the *only* place the loop flattens, and the value it flattened
    travels beside the text in `LoopResult.answer_value` so nothing downstream
    has to parse it back.
    """
    return (json.dumps(value, ensure_ascii=False, default=str)
            if isinstance(value, (list, dict)) else str(value))


# There is no predicate here any more, and its absence is the point.
#
# The delivery test used to be a function of the *value*: is `Final` set, is it
# empty, is it a container. Two versions of that were written and both were
# wrong, because the value cannot say whether the model meant it. Query 17
# computed `Total pairs: 0` from a bad parse, wrote `Final = pairs` on turn 8
# of 14, and the run ended with an empty answer — while the same task, under
# the rule that skipped empty lists, had spent its remaining turns delivering
# 875 pairs. The repair attempted was a rule about shape (containers wait,
# scalars conclude); it sorted by emptiness rather than by intent, left a wrong
# *non-empty* list concluding exactly as before, and made the empty value
# losable down a path that never read `Final`. Withdrawn the same day.
#
# `submit(value)` makes delivery an act instead of a shape. The session records
# that the call happened; the loop asks whether it happened. No value is
# privileged, no value is excluded, and there is nothing left to infer.

# The old recovery said only "Execute one short Python statement now" — it asked
# the model to shorten the code, never to leave the strategy that made it long.
# Both cold replications (t14, t20) measured what that costs: the model wrote
# 12-15K-character tool arguments hand-enumerating per-item keyword rules, was
# cut off at the token limit, was told to be brief, and wrote another oversized
# classifier. Three times each, zero subcalls, no answer. The redirect below
# names the abstraction instead: hand-written per-item judgement is the one
# thing the bounded operations exist to replace.
TRUNCATION_RECOVERY = (
    "Your previous generation ended before the Python call was complete, so "
    "nothing ran. Do not explain or write comments. If the code was long "
    "because it spells out per-item semantic judgements by hand — keyword "
    "lists, case-by-case rules — do not rewrite it: that judgement is what the "
    "bounded operations are for. semantic_map(instruction, schema) gives one "
    "validated value per item, semantic_search(goal) reads every segment. "
    "If the result already exists in a session variable, do not rebuild, "
    "enumerate, copy, or inline it; reference that variable directly in a "
    "short submit call. Otherwise execute one short Python statement now."
)

# Recorded once, when a run has spent this many turns without a single bounded
# operation committing anything.
#
# The observation is well founded. Six of the seven episodes that ended on
# `max_turns` made **zero** sub-model calls: fifteen turns of hand-parsing the
# context, one or no errors, not stuck, simply doing the work by hand. And the
# threshold is derived, not chosen — across every episode that finished above
# its floor, the first turn on which bounded work committed was 3, 6, 6, 7, 7,
# 7 and 8, so a run still empty at turn 9 is later than every success on
# record.
#
# What was never established is the other half. For eleven firings there is no
# evidence whether the note rescued the run, changed nothing, diverted one that
# was going to work, or pushed the model at `semantic_map` where plain Python
# was enough. Saying that could not be measured was wrong — it is recorded, and
# whether the next turn ran bounded work is exactly as checkable as the
# recovery rates that were used to keep the other counteroffers. It simply had
# not been checked, and an intervention left running because the question was
# never asked is not a decision.
#
# The reminder is useful only when the absence of bounded work also coincides
# with an absence of material local preparation. A model that has just built a
# non-empty collection from the context is not idle merely because it has not
# called a sub-model yet; steering it at that exact point measurably diverted a
# successful trajectory. Scalar-only debugging has no such evidence and still
# receives the capability reminder.
NO_BOUNDED_WORK_TURN = 9
NO_BOUNDED_WORK = (
    "You have used several turns and no bounded operation has run yet, so "
    "nothing has been read by a sub-model and no coverage has been "
    "established. Reading the text yourself in Python finds what you can match "
    "literally; it cannot judge meaning, and judging meaning item by item is "
    "what the bounded operations are for. semantic_map(instruction, schema, "
    "items=None) gives one validated value per item and semantic_search(goal) "
    "reads every segment. If your plan does not need either of them, continue."
)

_NONEMPTY_COLLECTION = re.compile(
    r"^(?:list|tuple|set|dict), ([1-9][0-9]*) items$"
)


def _material_local_preparation(
    observations: list[tuple[str, dict[str, Any]]],
) -> bool:
    """Whether this turn successfully produced a non-empty local collection.

    This deliberately uses only the REPL's type-and-size state summaries.  It
    does not inspect values, variable names, stdout, the question, or any
    task-specific concept.  An exact duplicate is not fresh preparation.
    """
    for _, observation in observations:
        if (not observation.get("ok")
                or observation.get("repeated_from_turn") is not None):
            continue
        changed = observation.get("changed") or {}
        if any(
            isinstance(summary, str) and _NONEMPTY_COLLECTION.fullmatch(summary)
            for summary in changed.values()
        ):
            return True
    return False

MALFORMED_TOOL_RECOVERY = (
    "Your previous reply contained tool-call markup, but no callable tool call "
    "was parsed, so nothing ran. Do not paste XML or a code fence into prose. "
    "Call PythonInterpreter once through the provided tool interface."
)


# Everything that identifies this machine rather than the run. Built once, and
# from the interpreter rather than from a literal, so it stays true on a
# different checkout, a different user or a different Python.
_HOST_STRINGS = tuple(sorted(
    (str(p) for p in {
        Path(__file__).resolve().parents[2],          # the repository root
        Path.home(),
        Path(sys.prefix),
        Path(sys.executable).parent,
    } if str(p) not in ("", "/")),
    key=len, reverse=True))                            # longest first: nested paths


def _without_the_host(text: str) -> str:
    """Model-visible text with this machine taken out of it.

    A rule rather than three repairs, and the reason is how the three repairs
    were found. The traceback leak turned up because a run diverged; the module
    path turned up only because that fix was then attacked on purpose — and it
    was the wider of the two, 148 observations against 130 episodes. One was
    found by accident and one by luck, which is not a method.

    The principle is the one the harness already applies in the other
    direction. Names the model must not reach are held outside the namespace,
    deliberately and with tests. This is the same boundary read the other way:
    nothing crosses from the host into the model except values the harness
    built on purpose. A raw traceback is not that. It is an object that happens
    to carry the position of our own source, and any edit to that source moves
    it — which is what made two thirds of a day's comparisons unreadable.

    Redacted rather than refused. A guard that raises would end a run over a
    string, and a leak is not worth an episode. The placeholder is visible so
    that a path arriving here is a thing someone can notice and go and fix at
    its source, which is where it should have been stopped.
    """
    for host in _HOST_STRINGS:
        if host in text:
            text = text.replace(host, "<host>")
    return text


def render(observation: dict[str, Any]) -> str:
    """The observation as the model reads it.

    Compact and named, never a bare stdout. `probe_11` re-ran the same
    conditional print three times, each returning nothing, because an empty
    stdout cannot distinguish "no matches" from "did not run"; every field here
    is present even when it is empty, and `ok` is always stated.

    A repeat is announced above whatever it produced, on either branch — a
    repeated block can raise as easily as it can print, and the note is the
    context for reading the result either way.
    """
    head = [observation["repeat_note"]] if observation.get("repeat_note") else []
    if not observation.get("ok", True) and observation.get("error"):
        error = observation["error"]
        if isinstance(error, dict):                      # an exception in the REPL
            lines = head + [f"ERROR {error.get('type')}: {error.get('message')}"]
            if error.get("traceback"):
                lines.append(str(error["traceback"])[-1200:])
            # A block that died halfway still ran its first half. Reporting only
            # the exception throws away what the model printed before it and
            # what the session now holds — and that is not academic: on the
            # directed OOLONG-Pairs run the model called `semantic_map`, the
            # call succeeded, and a `result[:1000]` on the returned dict raised
            # a TypeError one line later. It never saw its own "Result length:
            # 13", never saw `semantic_rows` bound, concluded the operation did
            # not work, and spent four turns hunting for a replacement.
            partial = observation.get("stdout") or ""
            if partial:
                lines.append(f"before the error it printed:\n{partial}")
            if observation.get("changed"):
                lines.append(f"the session now holds: {observation['changed']}")
            if observation.get("operation_result"):
                lines.append(
                    "a bounded operation completed before the error:\n"
                    + json.dumps(observation["operation_result"],
                                 ensure_ascii=False, default=str)
                )
        else:                                            # a protocol refusal
            lines = head + [f"ERROR {error}: {observation.get('message', '')}".rstrip(": ")]
            # `previous_stdout` and `previous_result_ref` were rendered here for
            # the duplicate refusal, the only observation that carried them.
            # With the refusal gone nothing produces either key, so the branches
            # went too rather than sitting here looking live.
            if observation.get("available_tools"):
                lines.append(f"available_tools: {observation['available_tools']}")
        for action in observation.get("next_actions") or []:
            lines.append(f"  - {action}")
        return _without_the_host("\n".join(lines))

    if "stdout" in observation:
        stdout = observation.get("stdout") or ""
        compacted = observation.get("stdout_compacted")
        if isinstance(compacted, dict):
            if compacted.get("reason") == "presentation_stdout_not_delivery":
                lines = head + [
                    "stdout was not a presentation commit and is omitted from "
                    "this repair prompt; full bytes remain in the trace ",
                    (f"({compacted['chars']} chars, sha256 "
                     f"{str(compacted['sha256'])[:16]})."),
                ]
            else:
                lines = head + [
                    "stdout repeated byte-for-byte; full bytes remain in the trace ",
                    (f"({compacted['chars']} chars, sha256 "
                     f"{str(compacted['sha256'])[:16]}, first shown on turn "
                     f"{compacted['previous_turn']})."),
                ]
        else:
            lines = head + [f"stdout ({len(stdout)} chars):",
                            stdout if stdout else "(empty — nothing was printed)"]
        if observation.get("truncated"):
            lines.append("[stdout was truncated in the middle]")
        if observation.get("value"):
            lines.append(f"value: {observation['value']}")
        if observation.get("changed"):
            names = ", ".join(f"{k}={v}" for k, v in observation["changed"].items())
            lines.append(f"variables now bound: {names}")
        if observation.get("stderr"):
            lines.append(f"stderr: {observation['stderr'][:1000]}")
        if observation.get("operation_result"):
            lines.append(
                "bounded operation result: "
                + json.dumps(observation["operation_result"],
                             ensure_ascii=False, default=str)
            )
        # Counteroffers were rendered only on the error branch, and not every
        # thing worth saying arrives as an error. A batch that came back short
        # because the sub-call budget stopped it is a block that *ran*: `ok` is
        # true, stdout is whatever the model printed, and the shortfall lives
        # only in `next_actions`.
        #
        # So the one note in this harness whose docstring says "the harness
        # stops being the only party that knows" was, both times it has ever
        # fired, the one thing the model was not told. Query 14 today: 772 jobs
        # submitted, 600 replies, 172 never sent, the whole sub-call allowance
        # gone — and the reply that said so in those exact numbers, and named
        # `semantic_map` as the operation that covers the same items for a
        # fraction of it, was written into the observation, recorded in the
        # trace, and dropped on the way out. The model re-ran the same call on
        # the next turn.
        for action in observation.get("next_actions") or []:
            lines.append(f"  - {action}")
        return _without_the_host("\n".join(lines))

    return _without_the_host(
        json.dumps(observation, ensure_ascii=False, indent=1, default=str))[:6000]


@dataclass
class LoopResult:
    """How the conversation ended, and everything needed to judge that.

    Steps and protocol errors travel with the answer because the answer alone
    cannot distinguish a model that computed it from one that asserted it.
    """
    answer: str | None
    stop_reason: str
    turns: int
    steps: list[Any] = field(default_factory=list)          # tasks.Step
    protocol_errors: list[dict[str, Any]] = field(default_factory=list)
    duplicates_observed: int = 0
    messages: list[dict[str, Any]] = field(default_factory=list)
    # The answer as the session held it, before anything rendered it to text.
    # `answer` is a string because an episode has to be able to end with prose,
    # but when the model delivered a list of pairs or an integer, flattening it
    # at the loop boundary throws away the only typed thing in the run and
    # forces every consumer to parse it back. The reference implementation
    # flattens here — `self._last_final_answer = str(content)` — and its own
    # OOLONG scorer then splits on the last colon, strips brackets and tags the
    # result "low" confidence. That is the cost of rendering too early, in the
    # code that pays it.
    #
    # `None` means the episode ended with no typed value: prose, a truncation,
    # or nothing at all. It does NOT mean the value was empty — a delivered
    # `[]` is a delivered `[]`, which is why `answer_delivered` is separate.
    answer_value: Any = None
    answer_delivered: bool = False
    initial_final_text: str | None = None
    repair_candidate_text: str | None = None
    final_text: str | None = None
    presentation_source: str | None = None
    output_mode: str = "raw"
    contract_validation: dict[str, Any] | None = None
    output_repair: dict[str, Any] | None = None
    visible_request_sha256s: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """The loop's outcome as plain data for the episode record."""
        return {
            "answer": self.answer,
            "initial_final_text": self.initial_final_text,
            "repair_candidate_text": self.repair_candidate_text,
            "final_text": self.final_text,
            "presentation_source": self.presentation_source,
            "output_mode": self.output_mode,
            "contract_validation": self.contract_validation,
            "output_repair": self.output_repair,
            "visible_request_sha256s": self.visible_request_sha256s,
            "stop_reason": self.stop_reason,
            "turns": self.turns,
            "tool_calls": len([s for s in self.steps if not getattr(s, "refused", False)]),
            "duplicates_observed": self.duplicates_observed,
            "protocol_errors": self.protocol_errors,
        }


@dataclass(frozen=True)
class Submission:
    delivered: bool = False
    value: Any = None
    final_text_provided: bool = False
    final_text: str | None = None


@dataclass
class NativeLoop:
    """Drives one episode. Knows nothing about how the REPL is implemented."""

    client: Any                                  # Chat
    execute: Callable[[str], dict[str, Any]]     # code -> observation
    budget: Budget
    trace: Any = None
    tool_name: str = protocol.TOOL_NAME
    max_tokens: int = 4096
    # Whether the session has delivered, and what. Two values because neither
    # can be recovered from the other: `submit(None)`, `submit(0)` and
    # `submit([])` are answers, so no value can double as "nothing delivered".
    read_submission: Callable[[], Submission] | None = None
    # Grants the session one further delivery window, and by being present at
    # all, enables the conformance turn. `None` disables it — which is what the
    # scripted tests and any loop without a REPL get, and what a sub-episode
    # gets: a sub-call's answer goes back into its parent's code as a value, so
    # there is no format instruction for it to conform to and the turn would be
    # a model call spent on nothing. The caller decides, where depth is known.
    open_presentation: Callable[..., Any] | None = None
    read_presentation: Callable[[], str | None] | None = None
    output_mode: str = "raw"
    output_contract: OutputContract | None = None
    inferred_presentation_spec: dict[str, Any] | None = None
    terminal_policy: TerminalPolicy = field(default_factory=TerminalPolicy)
    inject: str | None = None                    # deterministic condition to create
    depth: int = 0
    _last_stdout: tuple[str, int, str] | None = field(
        default=None, init=False, repr=False)
    _visible_request_sha256s: list[dict[str, Any]] = field(
        default_factory=list, init=False, repr=False)

    def run(self, question: str, *, context_line: str | None = None) -> LoopResult:
        """Drive the conversation until it terminates, and say why it did.

        Four ways out, all of them explicit: a `submit()` call, an `<answer>`
        tag, a turn with no tool call, or the budget — which still gets one
        forced final turn, because a run that spent everything and was never
        asked for an answer has no result rather than a wrong one.
        """
        from alchemist_rlm.step import Step                     # local: avoids a cycle

        validate_output_mode(self.output_mode, self.output_contract)
        self._last_stdout = None
        self._visible_request_sha256s = []

        tools = [protocol.python_tool(self.tool_name)]
        opening = question if not context_line else f"{context_line}\n\n{question}"
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": protocol.system_prompt(self.tool_name)},
            {"role": "user", "content": opening},
        ]
        ledger = CallLedger()
        steps: list[Step] = []
        errors: list[dict[str, Any]] = []
        # Two counters, because these are two different failures and merging
        # them ended episodes that were still working. A SyntaxError is a wrong
        # action; a generation cut off at `max_tokens` is an incomplete one, and
        # the model may be making perfect progress in between. On the
        # OOLONG-Pairs pilot the model truncated on every other turn, so two
        # ordinary code errors — the kind any agent makes — were fatal, with a
        # truncation supplying the third strike. One episode died having made no
        # execution error at all: truncation, a refused duplicate, truncation.
        # Truncations are not free either; they have their own limit.
        consecutive_errors = 0
        consecutive_truncations = 0
        # Whether any bounded operation has ever committed work in this
        # episode, and whether the note about it has already been sent. Once.
        bounded_work = False
        pending_partial: dict[str, Any] | None = None
        said_no_bounded_work = False
        # And a third, for the same reason the second exists. See the note in
        # budgets.py: the three v2 runs that died on `consecutive_errors` were
        # the three with two or more refused duplicates.
        consecutive_duplicates = 0
        turn = 0

        while True:
            reason = self.budget.exhausted()
            stalled = (
                "consecutive_errors"
                if (self.budget.max_consecutive_errors is not None
                    and consecutive_errors >= self.budget.max_consecutive_errors)
                else "consecutive_truncations"
                if consecutive_truncations >= self.budget.max_consecutive_truncations
                else "consecutive_duplicates"
                if consecutive_duplicates >= self.budget.max_consecutive_duplicates
                else None
            )
            if reason or stalled:
                return self._commit(
                    reason or stalled or "stalled", turn, steps, errors,
                    ledger, messages, pending_partial=pending_partial,
                )

            turn += 1
            self.budget.spend_turn()
            reply = self._ask(messages, tools=tools, turn=turn)
            reply.tool_calls = self._canonical_tool_call_ids(reply.tool_calls, turn)
            parsed = protocol.normalize(
                {"content": reply.content, "tool_calls": reply.tool_calls},
                tool_name=self.tool_name,
            )
            messages.append(self._assistant_message(reply, parsed))

            tagged = protocol.answer_tag(parsed.content)
            if tagged is not None:
                return self._finish_output(
                    tagged, "answer_tag", turn, steps, errors, ledger, messages,
                    presentation_source="answer_tag",
                )

            # A `length` cut-off is a truncated turn, never a decision — and
            # that must be checked BEFORE dispatch, tool calls or not. The
            # server's parser recovery can manufacture a tool call out of a
            # generation that died mid-XML; executing it hands the model its
            # own truncated wrapper as a SyntaxError, and recording it in the
            # ledger poisons the duplicate check for the eventual complete
            # version of the same call. Measured: the best recursion plan of
            # the project ended exactly that way, twice, and ran out of turns.
            if reply.truncated:
                errors.append({"turn": turn, "kind": "truncated_generation"})
                consecutive_truncations += 1
                # A note naming the repeated phrase used to be appended here.
                # It is the only message this harness has ever measured doing
                # harm: query 19 degenerated once on turn 2 of fifteen,
                # recovered on the standing message, and delivered above its
                # floor on turn 15 — and with the note added it died on turn 6
                # with three errors and nothing delivered, twice, identically.
                #
                # It was then retimed to fire from the second degeneration in
                # an episode, which leaves query 19 untouched by construction
                # and was never measured after that. So it stands at one
                # measured harm and no measured benefit, which is the only
                # combination in this file, and it is withdrawn rather than
                # carried on reasoning. The phenomenon behind it is real — 26
                # of 55 generations that hit the ceiling end in a phrase from
                # the corpus — and `_repeated_tail` stays, recording it in the
                # trace where it costs the model nothing.
                repeated = _repeated_tail(parsed.content or "")
                if repeated is None:
                    for call in parsed.calls:
                        repeated = _repeated_tail(call.code or "")
                        if repeated:
                            break
                if repeated:
                    errors.append({"turn": turn, "kind": "degenerate_repetition"})
                note = TRUNCATION_RECOVERY
                messages.append({"role": "user", "content": note})
                continue

            if not parsed.called_a_tool:
                content = (parsed.content or "").strip()
                if "</tool_call>" in content or "<function=" in content:
                    errors.append({"turn": turn, "kind": "malformed_tool_call"})
                    consecutive_errors += 1
                    messages.append({"role": "user", "content": MALFORMED_TOOL_RECOVERY})
                    continue
                if not content:
                    return LoopResult(None, "no_tool_call", turn, steps, errors,
                                      ledger.duplicates, messages,
                                      output_mode=self.output_mode,
                                      visible_request_sha256s=list(
                                          self._visible_request_sha256s))
                return self._finish_output(
                    content, "no_tool_call", turn, steps, errors, ledger, messages,
                    presentation_source="assistant_text",
                )

            observations = self._dispatch(parsed, turn, ledger, steps, errors)
            for _, observation in observations:
                operation = observation.get("operation_result") or {}
                if operation:
                    if (operation.get("status") == "partial"
                            and not operation.get("retry_exhausted")):
                        pending_partial = operation
                    else:
                        pending_partial = None
            bounded_work = bounded_work or any(
                o.get("progress") for _, o in observations)
            # Repetition is counted apart from success, because a repeat now
            # runs and therefore reports `ok`. Reading the two off one branch
            # would let a model loop indefinitely on a block that works.
            if observations and all(o.get("repeated_from_turn") is not None
                                    for _, o in observations):
                consecutive_duplicates += 1
            else:
                consecutive_duplicates = 0
            # A block that ran is progress, and it clears both counters: the
            # code problem is fixed and the model got a call out intact.
            if any(o.get("ok") or o.get("progress") for _, o in observations):
                consecutive_errors = consecutive_truncations = 0
            else:
                consecutive_errors += 1
            for call_id, observation in observations:
                messages.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": render(self._observation_for_prompt(observation, turn)),
                })
                if self.trace is not None:
                    self.trace.observation(depth=self.depth, turn=turn, observation=observation)

            submission = self._submission()
            if submission.delivered:
                return self._deliver(submission, "submitted", turn, steps, errors,
                                     ledger, messages)

            if (not bounded_work and not said_no_bounded_work
                    and turn >= NO_BOUNDED_WORK_TURN):
                said_no_bounded_work = True
                if _material_local_preparation(observations):
                    errors.append({
                        "turn": turn,
                        "kind": "no_bounded_work_local_progress",
                    })
                else:
                    errors.append({"turn": turn, "kind": "no_bounded_work"})
                    messages.append({"role": "user", "content": NO_BOUNDED_WORK})

    # --- pieces ------------------------------------------------------------
    def _commit(
        self,
        why: str,
        turn: int,
        steps: list[Any],
        errors: list[dict[str, Any]],
        ledger: CallLedger,
        messages: list[dict[str, Any]],
        *,
        pending_partial: dict[str, Any] | None = None,
    ) -> LoopResult:
        """Use at most two auxiliary turns to turn session work into delivery.

        Every outcome of the first turn has the same meaning at this boundary:
        either a value (or a tagged short answer) was delivered, or it was not.
        Clean Python, an exception, truncation and prose therefore receive the
        same one remaining opportunity. Stdout and arbitrary prose are never
        promoted to answers.
        """
        tools = [protocol.python_tool(self.tool_name)]
        partial = dict(pending_partial) if pending_partial is not None else None
        first_prompt = (COMMIT_PARTIAL.format(
            valid=partial.get("valid_items", 0),
            total=partial.get("total_items", 0),
        ) if partial is not None else COMMIT_FIRST)
        messages.append({"role": "user", "content": first_prompt})
        last_was_truncated = False

        recovery_attempts = (
            self.terminal_policy.max_partial_recovery_attempts
            if partial is not None else 0
        )
        allowed_attempts = (
            self.terminal_policy.max_commit_attempts + recovery_attempts)
        for attempt in range(allowed_attempts):
            turn += 1
            reply = self._ask(messages, tools=tools, turn=turn)
            reply.tool_calls = self._canonical_tool_call_ids(reply.tool_calls, turn)
            parsed = protocol.normalize(
                {"content": reply.content, "tool_calls": reply.tool_calls},
                tool_name=self.tool_name,
            )
            messages.append(self._assistant_message(reply, parsed))
            last_was_truncated = bool(reply.truncated)

            if reply.truncated:
                errors.append({"turn": turn, "kind": "truncated_final"})
            else:
                # Same precedence as the ordinary loop: one reply cannot both
                # commit a tagged answer and execute a second answer channel.
                tagged = protocol.answer_tag(parsed.content)
                if tagged is not None:
                    return self._finish_output(
                        tagged, f"forced_final:{why}", turn, steps, errors,
                        ledger, messages, presentation_source="answer_tag",
                    )

            if not reply.truncated and parsed.called_a_tool:
                observations = self._dispatch(
                    parsed, turn, ledger, steps, errors,
                    terminal_partial=partial is not None,
                )
                for call_id, observation in observations:
                    errors.append({
                        "turn": turn,
                        "kind": "final_block_executed",
                        "ok": bool(observation.get("ok")),
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": render(self._observation_for_prompt(observation, turn)),
                    })
                    if self.trace is not None:
                        self.trace.observation(
                            depth=self.depth, turn=turn, observation=observation
                        )
                    operation = observation.get("operation_result") or {}
                    if operation:
                        if (operation.get("status") == "partial"
                                and not operation.get("retry_exhausted")):
                            partial = operation
                        else:
                            partial = None

            if not reply.truncated:
                submission = self._submission()
                if submission.delivered:
                    return self._deliver(
                        submission,
                        f"forced_final:{why}:submitted",
                        turn,
                        steps,
                        errors,
                        ledger,
                        messages,
                    )

            if attempt + 1 < allowed_attempts:
                if recovery_attempts and attempt == 0:
                    next_prompt = (COMMIT_PARTIAL_SECOND
                                   if partial is not None else COMMIT_FIRST)
                else:
                    next_prompt = (COMMIT_PARTIAL_SECOND
                                   if partial is not None else COMMIT_SECOND)
                messages.append({
                    "role": "user",
                    "content": next_prompt,
                })

        suffix = ":truncated" if last_was_truncated else ""
        return LoopResult(
            None,
            f"forced_final:{why}{suffix}",
            turn,
            steps,
            errors,
            ledger.duplicates,
            messages,
            visible_request_sha256s=list(self._visible_request_sha256s),
        )

    def _deliver(
        self,
        submission: Submission,
        stop_reason: str,
        turn: int,
        steps: list[Any],
        errors: list[dict[str, Any]],
        ledger: CallLedger,
        messages: list[dict[str, Any]],
    ) -> LoopResult:
        """Finish a typed delivery without letting presentation replace it."""
        initial = (submission.final_text if submission.final_text_provided
                   else _render(submission.value))
        source = ("model_final_text" if submission.final_text_provided
                  else "legacy_render")
        return self._finish_output(
            initial, stop_reason, turn, steps, errors, ledger, messages,
            presentation_source=source,
            answer_value=submission.value,
            answer_delivered=True,
        )

    def _finish_output(
        self,
        initial_text: str,
        stop_reason: str,
        turn: int,
        steps: list[Any],
        errors: list[dict[str, Any]],
        ledger: CallLedger,
        messages: list[dict[str, Any]],
        *,
        presentation_source: str,
        answer_value: Any = None,
        answer_delivered: bool = False,
    ) -> LoopResult:
        """Validate presentation and optionally spend one text-only repair."""
        if self.output_mode == "constrained":
            presentation_source = "assisted_initial"
        final_text = initial_text
        candidate: str | None = None
        validation: dict[str, Any] | None = None
        repair: dict[str, Any] | None = None

        if self.output_contract is not None:
            validation = self._presentation_validation(
                initial_text, answer_value, answer_delivered)
            hard_stop = self.budget.exhausted() in {"max_seconds", "max_output_tokens"}
            if (self.output_mode == "validate_repair"
                    and not validation["valid"]
                    and self.terminal_policy.max_presentation_attempts > 0
                    and not hard_stop
                    and self.open_presentation is not None
                    and self.read_presentation is not None):
                turn, candidate, repair = self._repair_presentation(
                    initial_text, validation, turn, steps,
                    errors, ledger, messages, answer_value=answer_value,
                    answer_delivered=answer_delivered,
                )
                if candidate is not None:
                    candidate_validation = self._presentation_validation(
                        candidate, answer_value, answer_delivered)
                    repair["candidate_validation"] = candidate_validation
                    repair["candidate_valid"] = candidate_validation["valid"]
                    if candidate_validation["valid"]:
                        final_text = candidate
                        validation = candidate_validation
                        repair["promoted"] = True

        return LoopResult(
            final_text,
            stop_reason,
            turn,
            steps,
            errors,
            ledger.duplicates,
            messages,
            answer_value=answer_value,
            answer_delivered=answer_delivered,
            initial_final_text=initial_text,
            repair_candidate_text=candidate,
            final_text=final_text,
            presentation_source=presentation_source,
            output_mode=self.output_mode,
            contract_validation=validation,
            output_repair=repair,
            visible_request_sha256s=list(self._visible_request_sha256s),
        )

    def _presentation_validation(
        self, text: str, answer_value: Any, answer_delivered: bool,
    ) -> dict[str, Any]:
        """Validate syntax and, when content exists, bind text back to it."""
        assert self.output_contract is not None
        structural = self.output_contract.validate(text)
        result = structural.to_dict()
        result["structural_valid"] = structural.valid
        binding_record: dict[str, Any] | None = None
        if answer_delivered:
            binding = self.output_contract.binding
            if binding is None:
                binding_record = {
                    "valid": False,
                    "errors": [
                        "no PresentationBinding was declared for a typed answer"
                    ],
                }
            elif structural.valid:
                binding_record = binding.validate(answer_value, text).to_dict()
            else:
                binding_record = {
                    "valid": False,
                    "errors": ["structural validation failed before equivalence"],
                    "skipped": True,
                }
        result["binding"] = binding_record
        result["valid"] = structural.valid and (
            not answer_delivered or bool(binding_record and binding_record.get("valid"))
        )
        return result

    def _repair_presentation(
        self,
        initial_text: str,
        initial_validation: dict[str, Any],
        turn: int,
        steps: list[Any],
        errors: list[dict[str, Any]],
        ledger: CallLedger,
        messages: list[dict[str, Any]],
        *,
        answer_value: Any,
        answer_delivered: bool,
    ) -> tuple[int, str | None, dict[str, Any]]:
        """Continue the root conversation in a bounded presentation-only state."""
        record: dict[str, Any] = {
            "attempted": True,
            "promoted": False,
            "initial_validation": initial_validation,
            "attempts": [],
        }
        if self.inferred_presentation_spec is not None:
            record["inferred_presentation_spec"] = self.inferred_presentation_spec
        def feedback_for(validation: dict[str, Any]) -> str:
            """Render both syntax and value-binding failures for a retry."""
            listed = validation_feedback(
                validation, max_issues=8, max_examples=3,
                max_chars=self.terminal_policy.max_feedback_chars)
            binding = validation.get("binding") or {}
            if binding and not binding.get("valid") and not binding.get("skipped"):
                binding_feedback = validation_feedback(
                    binding, max_issues=3, max_examples=3,
                    max_chars=self.terminal_policy.max_feedback_chars,
                )
                if binding_feedback:
                    binding_lines = "\n".join(
                        f"- content binding: {line.removeprefix('- ')}"
                        for line in binding_feedback.splitlines()
                    )
                    listed = f"{listed}\n{binding_lines}" if listed else binding_lines
            return (listed or "- presentation is invalid")[
                :self.terminal_policy.max_feedback_chars]

        listed = feedback_for(initial_validation)
        specification = self.output_contract.specification if self.output_contract else {}
        source_variable, source_hint = _source_hint(steps)
        if source_variable is not None:
            record["source_variable"] = source_variable
        checker_hint = (
            (PRESENTATION_RENDERER_HINT
             if _direct_renderer_compatible(
                 answer_value, self.inferred_presentation_spec)
             else PRESENTATION_CHECKER_HINT).format(
                inferred_specification=json.dumps(
                    self.inferred_presentation_spec,
                    sort_keys=True, ensure_ascii=False))
            if self.inferred_presentation_spec is not None else ""
        )
        # Keep the model on the same trained tool-calling trajectory.  Reusing
        # the history preserves the original question, the tool-call grammar
        # and the variables the model just built.  The REPL still exposes only
        # immutable-source copies during this window, so conversational
        # continuity does not reopen the committed value.
        repair_instruction = OUTPUT_REPAIR.format(
            errors=listed,
            specification=json.dumps(specification, sort_keys=True,
                                     ensure_ascii=False),
            source_shape=(_value_shape(answer_value) if answer_delivered
                          else "no typed value; use PRESENTATION_TEXT"),
            preview=_delivery_preview(initial_text),
            source_hint=source_hint,
            checker_hint=checker_hint,
        )
        compact_repair_instruction = OUTPUT_REPAIR_COMPACT.format(
            errors=listed,
            specification=json.dumps(specification, sort_keys=True,
                                     ensure_ascii=False),
            source_shape=(_value_shape(answer_value) if answer_delivered
                          else "no typed value; use PRESENTATION_TEXT"),
            source_hint=source_hint,
            checker_hint=checker_hint,
        )
        repair_messages: list[dict[str, Any]] = list(messages)
        repair_messages.append({"role": "user", "content": repair_instruction})
        tools = [protocol.python_tool(self.tool_name)]
        ordinary_allowed_calls = (
            self.terminal_policy.max_presentation_attempts
            + self.terminal_policy.max_protocol_retries
        )
        max_calls = (
            ordinary_allowed_calls
            + self.terminal_policy.max_presentation_commit_reserve
        )
        candidate_attempts = 0
        candidate_hashes: set[str] = set()
        draft_hashes: set[str] = set()
        draft_text: str | None = None
        draft_origin: str | None = None
        draft_ready = False
        clean_build_turns = 0
        commit_required = self.terminal_policy.max_presentation_build_turns == 0
        last_candidate: str | None = None
        failed_presentation_blocks: dict[str, dict[str, Any]] = {}
        commit_reserve_earned = False
        for attempt_index in range(max_calls):
            reserved_commit_turn = attempt_index >= ordinary_allowed_calls
            if reserved_commit_turn and not commit_reserve_earned:
                break
            turn += 1
            # Keep the first presentation request compatible with the proven
            # conformance trajectory: it is still the root conversation and
            # historically used the root generation ceiling.  The smaller
            # presentation ceiling applies only after a factual failed attempt,
            # when the loop has compacted to the bounded terminal grammar.
            presentation_max_tokens = (
                None if attempt_index == 0
                else self.terminal_policy.max_presentation_tokens
            )
            reply = self._ask(
                repair_messages, tools=tools, turn=turn, channel="presentation",
                max_tokens=presentation_max_tokens)
            reply.tool_calls = self._canonical_tool_call_ids(reply.tool_calls, turn)
            parsed = protocol.normalize(
                {"content": reply.content, "tool_calls": reply.tool_calls},
                tool_name=self.tool_name,
            )
            repair_messages.append(self._assistant_message(reply, parsed))
            attempt: dict[str, Any] = {
                "attempt": attempt_index + 1,
                "candidate_committed": False,
            }
            if reserved_commit_turn:
                attempt["reserved_commit_turn"] = True
                record["commit_reserve_used"] = True
            record["attempts"].append(attempt)
            if reply.truncated:
                attempt["protocol_error"] = "truncated"
            elif not parsed.called_a_tool:
                attempt["protocol_error"] = (
                    "code_fence_without_tool"
                    if "```" in (parsed.content or "") else "no_submit")
            else:
                opened = self.open_presentation(
                    initial_text, specification, draft_text,
                    commit_required=commit_required,
                    draft_ready=draft_ready,
                    source_name=source_variable,
                    inferred_spec=self.inferred_presentation_spec,
                ) if self.open_presentation else None
                if isinstance(opened, dict) and opened.get("error"):
                    attempt["protocol_error"] = "presentation_open_failed"
                    attempt["error"] = opened["error"]
                    record["error"] = opened["error"]
                    return turn, None, record
                observations = self._dispatch(
                    parsed, turn, ledger, steps, errors, allow_repeats=True)
                clean_progress = False
                clean_stdout: str | None = None
                defined_names: set[str] = set()
                block_key = (protocol.code_key(parsed.calls[0].code or "")
                             if len(parsed.calls) == 1 else None)
                for call_id, observation in observations:
                    prompt_observation = self._presentation_observation_for_prompt(
                        observation)
                    repair_messages.append({
                        "role": "tool", "tool_call_id": call_id,
                        "content": render(prompt_observation),
                    })
                    if self.trace is not None:
                        self.trace.observation(depth=self.depth, turn=turn,
                                               observation=observation)
                    if not observation.get("ok"):
                        detail = observation.get("error") or "block_failed"
                        error_type = detail.get("type") if isinstance(detail, dict) else None
                        attempt["protocol_error"] = {
                            "PresentationCommitRequired": "presentation_commit_required",
                            "PresentationDraftCommitRequired":
                                "presentation_draft_commit_required",
                            "PresentationIsolationRequired":
                                "presentation_isolation_required",
                            "PresentationSourceImmutable":
                                "presentation_source_immutable",
                            "PresentationSourceRequired":
                                "presentation_source_required",
                        }.get(str(error_type), "invalid_signature_or_block")
                        attempt["error"] = detail
                        actions = list(prompt_observation.get("next_actions") or ())
                        if actions:
                            attempt["next_actions"] = actions
                        if block_key is not None:
                            previous_failure = failed_presentation_blocks.get(block_key)
                            signature = (
                                str(error_type),
                                str(detail.get("message") if isinstance(detail, dict)
                                    else detail),
                            )
                            if (previous_failure is not None
                                    and previous_failure["signature"] == signature):
                                attempt["repeated_failed_block"] = True
                                attempt["previous_failed_attempt"] = (
                                    previous_failure["attempt"])
                                attempt.setdefault("next_actions", []).insert(0,
                                    "this exact block already produced the same error; "
                                    "change the Python before calling the tool again"
                                )
                            failed_presentation_blocks[block_key] = {
                                "attempt": attempt_index + 1,
                                "signature": signature,
                            }
                    else:
                        stdout = observation.get("stdout")
                        defined_names.update(observation.get("defined") or ())
                        clean_progress = clean_progress or any((
                            bool(stdout), bool(observation.get("defined")),
                            bool(observation.get("changed")),
                            observation.get("value") is not None,
                        ))
                        if (isinstance(stdout, str) and stdout
                                and not observation.get("truncated")):
                            clean_stdout = stdout
                candidate = self.read_presentation() if self.read_presentation else None
                if candidate is not None:
                    last_candidate = candidate
                    candidate_attempts += 1
                    candidate_sha256 = hashlib.sha256(candidate.encode()).hexdigest()
                    attempt["candidate_committed"] = True
                    attempt["source"] = "submit"
                    attempt["candidate_sha256"] = candidate_sha256
                    candidate_validation = self._presentation_validation(
                        candidate, answer_value, answer_delivered)
                    attempt["candidate_validation"] = candidate_validation
                    record["candidate_validation"] = candidate_validation
                    record["candidate_valid"] = candidate_validation["valid"]
                    if candidate_validation["valid"]:
                        return turn, candidate, record
                    attempt["validation_failed"] = True
                    # The model already authored a complete candidate.  Keep
                    # those exact rejected bytes inside the persistent REPL so
                    # the compact retry can edit them instead of reconstructing
                    # a long value or relying on the discarded root history.
                    # This is only a source binding: promotion still requires a
                    # new submit plus structural and content-binding validation.
                    draft_text = candidate
                    draft_origin = "rejected_candidate"
                    draft_ready = False
                    attempt["persisted_as_draft"] = True
                    record["rejected_candidate_draft_sha256"] = candidate_sha256
                    if candidate_sha256 in candidate_hashes:
                        attempt["protocol_error"] = "repeated_invalid_candidate"
                        record["error"] = "repeated_invalid_candidate"
                        return turn, candidate, record
                    candidate_hashes.add(candidate_sha256)
                    if (candidate_attempts
                            >= self.terminal_policy.max_presentation_attempts):
                        return turn, candidate, record
                elif "protocol_error" not in attempt:
                    if clean_progress:
                        clean_build_turns += 1
                        attempt["clean_progress"] = True
                        attempt["clean_build_turn"] = clean_build_turns
                        if defined_names:
                            attempt["defined"] = sorted(defined_names)
                        if clean_stdout is not None:
                            draft_sha256 = hashlib.sha256(
                                clean_stdout.encode()).hexdigest()
                            draft_validation = self._presentation_validation(
                                clean_stdout, answer_value, answer_delivered)
                            attempt.update({
                                "source": "stdout_draft",
                                "draft_sha256": draft_sha256,
                                "draft_chars": len(clean_stdout),
                                "draft_validation": draft_validation,
                            })
                            if draft_sha256 in draft_hashes:
                                attempt["protocol_error"] = "repeated_invalid_draft"
                                record["error"] = "repeated_invalid_draft"
                                record["clean_build_turns"] = clean_build_turns
                                return turn, last_candidate, record
                            draft_hashes.add(draft_sha256)
                            replace_draft = (
                                draft_text is None
                                or draft_origin != "rejected_candidate"
                                or bool(draft_validation["valid"])
                            )
                            if replace_draft:
                                draft_text = clean_stdout
                                draft_origin = "stdout"
                                draft_ready = bool(draft_validation["valid"])
                                attempt["draft_persisted"] = True
                                record["draft_sha256"] = draft_sha256
                                record["draft_validation"] = draft_validation
                            else:
                                attempt["retained_rejected_candidate_draft"] = True
                        commit_required = (
                            draft_ready or clean_build_turns
                            >= self.terminal_policy.max_presentation_build_turns
                        )
                        record["clean_build_turns"] = clean_build_turns
                    else:
                        attempt["protocol_error"] = "no_progress"

            reserve_earned_now = (
                attempt_index + 1 == ordinary_allowed_calls
                and self.terminal_policy.max_presentation_commit_reserve > 0
                and bool(attempt.get("clean_progress"))
                and not attempt.get("protocol_error")
                and not attempt.get("candidate_committed")
            )
            if reserve_earned_now:
                commit_reserve_earned = True
                commit_required = True
                record["commit_reserve_earned_after_attempt"] = (
                    attempt_index + 1)
            has_next_call = (
                attempt_index + 1 < ordinary_allowed_calls
                or reserve_earned_now
            )
            if has_next_call:
                # Preserve the full root trajectory for the first repair, as
                # that continuity has recovered real episodes. Once an attempt
                # has factually failed, its old deliberation and failed code
                # become anchoring ballast. Keep the persistent REPL and frozen
                # source, but retry from the compact presentation grammar.
                if reserve_earned_now:
                    repair_messages = [{
                        "role": "system", "content": PRESENTATION_ONLY_SYSTEM,
                    }, {
                        "role": "user", "content": compact_repair_instruction,
                    }]
                    record["history_compacted_for_commit_reserve_after_attempt"] = (
                        attempt_index + 1)
                elif (attempt.get("validation_failed")
                      or attempt.get("protocol_error")):
                    repair_messages = [{
                        "role": "system", "content": PRESENTATION_ONLY_SYSTEM,
                    }]
                    repair_messages.append({
                        "role": "user", "content": compact_repair_instruction,
                    })
                    record.setdefault(
                        "history_compacted_after_attempt", attempt_index + 1)
                    record.setdefault(
                        "history_compacted_after_attempts", []).append(
                            attempt_index + 1)
                if attempt.get("validation_failed"):
                    candidate_feedback = feedback_for(
                        attempt["candidate_validation"])
                    repair_messages.append({
                        "role": "user",
                        "content": PRESENTATION_VALIDATION_RETRY.format(
                            specification=json.dumps(
                                specification, sort_keys=True,
                                ensure_ascii=False),
                            errors=candidate_feedback,
                            source_hint=source_hint,
                            checker_hint=checker_hint),
                    })
                    continue
                if attempt.get("clean_progress") and not attempt.get("protocol_error"):
                    if draft_ready:
                        repair_messages.append({
                            "role": "user", "content": PRESENTATION_DRAFT_READY,
                        })
                    elif attempt.get("draft_validation") is not None:
                        draft_feedback = feedback_for(
                            attempt["draft_validation"])
                        draft_message = (
                            PRESENTATION_STDOUT_REJECTED_DRAFT_RETAINED
                            if attempt.get("retained_rejected_candidate_draft")
                            else PRESENTATION_DRAFT_RETRY
                        )
                        repair_messages.append({
                            "role": "user",
                            "content": draft_message.format(
                                errors=draft_feedback),
                        })
                    else:
                        detail = (
                            "New variables: " + ", ".join(
                                attempt.get("defined") or ()) + "."
                            if attempt.get("defined") else
                            "The local state changed without printable text."
                        )
                        repair_messages.append({
                            "role": "user",
                            "content": PRESENTATION_PROGRESS.format(
                                used=clean_build_turns,
                                limit=self.terminal_policy.max_presentation_build_turns,
                                detail=detail,
                            ),
                        })
                    if (commit_required and not reserve_earned_now
                            and not draft_ready):
                        repair_messages.append({
                            "role": "user", "content": PRESENTATION_BUILD_LIMIT,
                        })
                    if reserve_earned_now:
                        repair_messages.append({
                            "role": "user",
                            "content": PRESENTATION_COMMIT_RESERVE,
                        })
                    continue
                fact = str(attempt.get("protocol_error") or "no_candidate")
                detail = attempt.get("error")
                if isinstance(detail, dict) and detail.get("message"):
                    fact = f"{fact}: {detail['message']}"
                elif isinstance(detail, str):
                    fact = f"{fact}: {detail}"
                actions = attempt.get("next_actions") or ()
                if actions:
                    fact = f"{fact}. {actions[0]}"
                repair_messages.append({
                    "role": "user",
                    "content": PRESENTATION_RETRY.format(fact=fact[:500]),
                })

        last = record["attempts"][-1] if record["attempts"] else {}
        record["clean_build_turns"] = clean_build_turns
        record["error"] = last.get("protocol_error", "presentation_failed")
        return turn, last_candidate, record

    @staticmethod
    def _presentation_observation_for_prompt(
        observation: dict[str, Any],
    ) -> dict[str, Any]:
        """Keep printed inspection data in trace, not in the repair context."""
        compact = dict(observation)
        error = observation.get("error")
        if (isinstance(error, dict) and error.get("type") == "TypeError"
                and "unhashable type: 'list'" in str(error.get("message"))):
            compact["next_actions"] = [
                *(observation.get("next_actions") or ()),
                "set membership and dictionary keys require hashable values; "
                "when sequence identity is intended, derive an immutable key "
                "such as tuple(value) without changing the source list",
                "change the failing Python before calling the tool again",
            ]
        stdout = observation.get("stdout")
        if not isinstance(stdout, str) or len(stdout) <= 1_000:
            return compact if compact != observation else observation
        compact["stdout"] = ""
        compact["stdout_compacted"] = {
            "chars": len(stdout),
            "sha256": hashlib.sha256(stdout.encode()).hexdigest(),
            "reason": "presentation_stdout_not_delivery",
        }
        return compact

    def _submission(self) -> Submission:
        """The committed typed value and optional model-authored presentation."""
        if not self.read_submission:
            return Submission()
        submission = self.read_submission()
        # Small embedders of NativeLoop historically supplied ``(delivered,
        # value)``. Preserve that boundary while the real REPL returns the
        # richer record with independent presentation text.
        if isinstance(submission, tuple) and len(submission) == 2:
            return Submission(bool(submission[0]), submission[1])
        if not isinstance(submission, Submission):
            raise TypeError("read_submission must return Submission or (delivered, value)")
        return submission

    def _observation_for_prompt(
        self, observation: dict[str, Any], turn: int,
    ) -> dict[str, Any]:
        """Compact only a quiet, byte-identical repetition of prior stdout.

        The original object is never mutated and is what tracing receives.
        Any error or observable side effect keeps the complete observation in
        the next prompt even when its stdout happens to match.
        """
        stdout = observation.get("stdout")
        if not isinstance(stdout, str) or not stdout:
            return observation
        digest = hashlib.sha256(stdout.encode()).hexdigest()
        previous = self._last_stdout
        self._last_stdout = (stdout, turn, digest)
        has_effect = any((
            not observation.get("ok", True),
            bool(observation.get("error")),
            bool(observation.get("changed")),
            bool(observation.get("delivered")),
            bool(observation.get("presentation_candidate")),
            bool(observation.get("operation_result")),
            bool(observation.get("stderr")),
            bool(observation.get("truncated")),
        ))
        if previous is None or previous[0] != stdout or has_effect:
            return observation
        compact = dict(observation)
        compact["stdout"] = ""
        compact["stdout_compacted"] = {
            "chars": len(stdout),
            "sha256": digest,
            "previous_turn": previous[1],
        }
        return compact

    def _ask(self, messages: list[dict[str, Any]], *, tools: Any, turn: int,
             channel: str = "reasoning", max_tokens: int | None = None) -> Any:
        request_max_tokens = self.max_tokens if max_tokens is None else max_tokens
        visible = json.dumps(
            {"messages": messages, "tools": tools},
            sort_keys=True, ensure_ascii=False, separators=(",", ":"),
        )
        request_record = {
            "turn": turn,
            "channel": channel,
            "sha256": hashlib.sha256(visible.encode()).hexdigest(),
            "messages": len(messages),
            "max_tokens": request_max_tokens,
        }
        self._visible_request_sha256s.append(request_record)
        if self.trace is not None:
            self.trace.event("model_request", depth=self.depth, **request_record)
        reply = self.client.complete(
            messages, tools=tools, max_tokens=request_max_tokens)
        self.budget.ledger.add(output_tokens=reply.output_tokens)
        if self.trace is not None:
            self.trace.model_turn(
                depth=self.depth, turn=turn, content=reply.content,
                reasoning=reply.reasoning, raw_tool_calls=reply.tool_calls,
                served_model=reply.served_model, usage=reply.usage,
            )
        return reply

    @staticmethod
    def _assistant_message(reply: Any, parsed: ParsedTurn) -> dict[str, Any]:
        message: dict[str, Any] = {"role": "assistant", "content": parsed.content}
        # A truncated turn's tool calls are manufactured fragments; recording
        # them would leave an unanswered tool_call_id in the history the next
        # prompt has to render.
        if reply.tool_calls and not reply.truncated:
            message["tool_calls"] = reply.tool_calls
        return message

    @staticmethod
    def _canonical_tool_call_ids(
        tool_calls: list[dict[str, Any]] | None, turn: int,
    ) -> list[dict[str, Any]]:
        """Replace backend-generated IDs with deterministic transcript IDs."""
        normalized: list[dict[str, Any]] = []
        for index, call in enumerate(tool_calls or []):
            item = dict(call)
            item["id"] = f"call_{turn}_{index}"
            normalized.append(item)
        return normalized

    def _dispatch(
        self,
        parsed: ParsedTurn,
        turn: int,
        ledger: CallLedger,
        steps: list[Any],
        errors: list[dict[str, Any]],
        *,
        allow_repeats: bool = False,
        terminal_partial: bool = False,
    ) -> list[tuple[str, dict[str, Any]]]:
        """Answer every tool call the model made — including the ones we refuse.

        Every call gets a reply because a `tool_call_id` left unanswered breaks
        the next prompt render, and because a call that vanishes is scored as a
        decision the model never made.

        `allow_repeats` is for the presentation-repair turn and nothing else. The guard
        refuses a block whose text has run before, which is right on an ordinary
        turn and wrong on one the harness asked for: query 9 delivered with
        `submit(pairs)` on turn 15, was granted a delivery window, rebuilt
        `pairs` on turn 16, and had `submit(pairs)` refused on turn 17 as a
        duplicate of turn 15. The harness declined its own request. The guard
        keys on code text, so it could not see that the value had changed
        underneath the same two words.
        """
        from alchemist_rlm.step import Step

        out: list[tuple[str, dict[str, Any]]] = []
        for entry in parsed.unknown:
            errors.append({"turn": turn, "kind": "unknown_tool", "name": entry.get("name")})
            out.append((
                (entry.get("raw") or {}).get("id") or f"unknown_{turn}",
                protocol.unknown_tool_observation(entry.get("name"), tool_name=self.tool_name),
            ))

        for index, call in enumerate(parsed.calls):
            call_id = (call.raw or {}).get("id") or f"call_{turn}_{index}"
            if index > 0:
                # The protocol is one call per turn. Extra calls are recorded and
                # refused rather than executed, so a turn's effect stays legible.
                errors.append({"turn": turn, "kind": "multiple_calls", "extra": len(parsed.calls) - 1})
                out.append((call_id, {
                    "ok": False,
                    "error": "one_call_per_turn",
                    "message": "Only the first call in a turn runs. This one was not executed.",
                    "next_actions": ["make one call per turn and read its result first"],
                }))
                continue
            out.append((call_id, self._run(
                call, turn, ledger, steps, allow_repeats=allow_repeats,
                terminal_partial=terminal_partial,
            )))
        return out

    def _run(self, call: ToolCall, turn: int, ledger: CallLedger,
             steps: list[Any], *, allow_repeats: bool = False,
             terminal_partial: bool = False) -> dict[str, Any]:
        from alchemist_rlm.step import Step

        code = call.code or ""
        if self.trace is not None:
            self.trace.tool_call(depth=self.depth, turn=turn, name=call.name,
                                 code=code, code_key=protocol.code_key(code))
        if not code.strip():
            return {"ok": False, "error": "empty_code",
                    "message": f"{self.tool_name} was called with no code.",
                    "next_actions": ["pass the Python you want to run in `code`"]}

        if terminal_partial:
            try:
                tree = ast.parse(code)
            except SyntaxError:
                tree = None
            fresh = {
                "llm_query", "llm_query_batched", "rlm_map", "rlm_query",
                "semantic_map", "semantic_search",
            }
            called = sorted({
                node.func.id for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in fresh
            }) if tree is not None else []
            if called:
                observation = {
                    "ok": False,
                    "stdout": "", "stderr": "",
                    "error": {
                        "type": "TerminalSweepRefused",
                        "message": "a fresh semantic operation is not allowed "
                                   "in partial-sweep terminal recovery",
                    },
                    "defined": [], "changed": {}, "value": None,
                    "truncated": False, "progress": False,
                    "next_actions": [
                        "use retry_failed(semantic_result) once for the "
                        "already registered partial sweep",
                        "do not start a new semantic_map, semantic_search or "
                        "model query in the terminal recovery phase",
                    ],
                }
                steps.append(Step(code=code, defined=frozenset()))
                ledger.record(code, turn)
                return observation

        # The key keeps V1's spelling, which says "refuse", because it is
        # written into a frozen suite whose hash is pinned against a result
        # file that still cites it. Renaming it to match today's behaviour
        # failed `test_v1_stays_frozen`, and correctly: V1 is the record of
        # what was measured, and what it injected then *was* a refusal.
        forced_duplicate = (
            self.inject == "refuse_first_call_as_duplicate" and not steps
        )
        previous = None if allow_repeats else ledger.duplicate_of(code)
        if forced_duplicate and previous is None and not allow_repeats:
            # A controlled condition, not a wait for chance: the recovery task
            # is untestable on the runs where the model never repeats itself.
            ledger.record(code, turn)
            previous = ledger.duplicate_of(code)

        # A repeat is observed, not vetoed. Refusing it asserted that the same
        # code gives the same result, which a stateful session does not
        # guarantee, and the refusal recovered on 19 of 90 firings. Running it
        # costs one execution and tells the model something true: here is what
        # it produces *now*, next to the turn it produced something before.
        observation = self.execute(code)
        steps.append(Step(code=code, defined=frozenset(observation.get("defined") or ())))
        if previous is not None:
            observation = {**observation, **ledger.note_repeat(previous)}
        ledger.record(code, turn)
        return observation
