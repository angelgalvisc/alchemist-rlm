"""The wire protocol: tool schema, call ledger, termination.

Two design choices here are not arbitrary; both come from measurements in this
repository.

**The tool is named `PythonInterpreter` and takes `code`.** The checkpoint's own
README says the model "was trained and benchmarked with three specific tools"
and that "with different names it calls them far less often". Gate B tests that
claim by swapping in `run_python` with an identical schema — so the name lives
in one constant, not scattered through the code.

**A duplicate call is answered with a menu, not a wall.** `probe_08` made the
environment refuse a repeated block with "REFUSED — this exact block already ran
on turn N with the same result" and measured *zero* behavioural change: the
model kept emitting it. Separately, `probe_11` recorded 23 turns of regex in
which ten were re-runs of searches that had already returned nothing. Blocking
the bad action does not supply the good one, so every refusal here carries
`next_actions`.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

TOOL_NAME = "PythonInterpreter"
ALIAS_TOOL_NAME = "run_python"          # Gate B: identical schema, different name

# Kept deliberately short. The stock RLM system prompt is ~1,340 tokens of
# policy written for frontier models; a 4B has to hold all of it while planning.
#
# The closing paragraph is there for the same reason. It used to offer the
# delivery variable and `<answer>` as equivalents, and they are not: one is read
# out of the session, the other is written by the model and dies with its turn.
# Measured on OOLONG-Pairs — a task whose answer is thousands of pairs — the
# model computed all 3,227 of them in the session, wrote a comment about
# delivering them, and then began retyping them literally into a new list. It
# was cut off at twenty. Its own code quoted the old line back at us.
#
# Two later corrections are in the current wording. Delivery is `submit(value)`,
# an act, not an assignment whose intent the harness then had to infer from the
# value's shape — two attempts at that inference were written and both were
# wrong. And "pass the variable rather than retyping" is a claim about *token
# cost* that was reading as a claim about *shape*: three queries computed their
# pairs correctly and then submitted them in a format the benchmark rejects,
# with nothing in the prompt saying that building a required shape in code is
# not the retyping being warned against. It says so now.
#
# The operation-choice block exists because of a measured gap, not as garnish:
# in the V1 suite the model knew the *names* llm_query_batched and rlm_query —
# they were listed as bare signatures — and never once selected either. The
# only strategy it found for semantic work was llm_query over the whole
# context, which works at 30K characters and dies at 200K. Naming a tool is
# not an interface; saying when it is the cheapest correct choice is.
SYSTEM_PROMPT = """\
You answer questions about data that is too large to read directly.

The data is already loaded in a persistent Python session as `context`, a plain \
`str` variable. It is not in this conversation and you cannot see it without \
running code. Never import it: it and every helper listed below are already \
defined.

{tool_name} is the ONLY tool. semantic_search, semantic_map, retry_failed, \
llm_query, llm_query_batched, rlm_query, rlm_map and submit are Python functions \
already defined inside it — invoke them from your Python code, never as separate tools.

Call {tool_name} to run Python. Variables persist between calls. Count and \
aggregate in Python, never by eye.

Choose the cheapest operation that fits:
- Exact matching, filtering, counting or arithmetic: plain Python on `context`.
- Semantic judgement over many parts: `semantic_search()` reads every \
segment with a bounded batch of sub-models, using your question as the goal \
(pass goal=... only to narrow it), or build your own batch with \
`llm_query_batched({{'instruction': ..., 'source': part}} for part in parts)`.
- One value out of a fixed set for every item: `semantic_map(instruction, \
{{'type': 'string', 'enum': [...]}})` — the same exhaustive pass, giving one \
validated value per item in `result['rows']`. Aggregate those in Python.
- A judgement AND a literal from the same item in one pass: make the schema \
an object, `{{'type': 'object', 'properties': {{'label': {{'type': 'string', \
'enum': [...]}}, 'when': {{'type': 'string'}}}}, 'required': ['label', \
'when'], 'additionalProperties': False}}` — each row's `value` then holds \
both, so the item's text never needs parsing again.
- A single slice that needs reading: `llm_query(instruction, source)`. Never \
pass the whole context as `source`; select or partition first.
- A part that is itself large, or needs several search-and-reason steps: \
delegate it with `rlm_query(question, part)`, or `rlm_map(question, parts)` \
for several parts. Each delegate gets its own Python session over its part.

When done, `submit(value)` ends the run. For an exact textual form, build \
`final_text` from the value in Python and call \
`submit(value, final_text=final_text)`: `value` is the computed answer and \
`final_text` is the exact user text. Delivery has no generation-length limit; \
use variables, not retyping. Any value can be submitted, \
including an empty list or zero. \
<answer></answer> is written by you and is cut off when your turn runs out, so \
use it only for a short answer."""


def python_tool(name: str = TOOL_NAME) -> dict[str, Any]:
    """The schema, verbatim in shape from the checkpoint's documented toolset."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": (
                "Execute Python code in a persistent session and return its output. "
                "The session holds the full data as `context` and keeps variables "
                "between calls."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute."}
                },
                "required": ["code"],
            },
        },
    }


def system_prompt(tool_name: str = TOOL_NAME) -> str:
    """The system prompt with the tool's real name substituted in. Built rather than
    stored so the name in the prose can never drift from the name in the schema — the
    model would then be told to call something that does not exist.
    """
    return SYSTEM_PROMPT.format(tool_name=tool_name)


# The counteroffer for the one wrong type that is a reasonable thing to have
# reached for. A model holding parsed records naturally passes the records, and
# saying only "got dict" tells it what it did without telling it what to do —
# in a repository whose rule is that a refusal without a counteroffer changes
# nothing. Query 7 got the bare version twice and died on the error guard.
#
# It names no field and no corpus. Which text to judge belongs to the question;
# that the records line back up by index is a fact about the operation, already
# stated in the note a completed sweep returns.
_RECORDS_HINT = (
    ". If these are records you parsed, pass the text to be judged out of them "
    "— one string per item — and keep the records; the row for each item "
    "carries its position, so they line back up by index"
)


def as_text(value: Any, *, what: str) -> str:
    """Text in — or an unambiguous composite of texts, joined in order.

    A list of strings has exactly one sensible meaning: a group. Everything
    else non-str is refused by name, because the alternative — observed on
    four separate boundaries — is `str()` quietly handing a Python repr to a
    reader, a segmenter or a child, which then works on quote marks and
    literal \n escapes while every downstream number silently rots. Seven
    recursion children each received 34,000 characters on one line that way.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError(f"{what} is an empty list — pass at least one text")
        if all(isinstance(item, str) for item in value):
            return "\n\n".join(value)
        bad = next(i for i, item in enumerate(value) if not isinstance(item, str))
        raise ValueError(
            f"{what}[{bad}] is {type(value[bad]).__name__}; every element must "
            "be str" + _RECORDS_HINT * isinstance(value[bad], dict)
        )
    raise ValueError(f"{what} must be text or a list of texts, got "
                     f"{type(value).__name__}"
                     + _RECORDS_HINT * isinstance(value, dict))


# --- termination -----------------------------------------------------------
_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.S | re.I)


def answer_tag(text: str) -> str | None:
    """`<answer>` is the tag this model family was trained to close on.

    The checkpoint README is explicit that a loop waiting only for the tag will
    "spin past a perfectly good reply", which is why the caller must also treat
    a turn with no tool call as terminal.
    """
    if not text:
        return None
    match = _ANSWER.search(text)
    return match.group(1).strip() if match else None


# --- tool calls ------------------------------------------------------------
@dataclass(frozen=True)
class ToolCall:
    """One parsed call: the tool's name, the code it carries, and the raw block.

    The raw block is kept and excluded from equality on purpose. Comparison is
    about what was asked for; the raw form is what a reader needs when the
    server's parser and this one disagree about what the model actually wrote.
    """
    name: str
    code: str
    raw: dict[str, Any] = field(default_factory=dict, compare=False)


@dataclass(frozen=True)
class ParsedTurn:
    """What the model did this turn, with nothing dropped on the floor.

    `unknown` exists because silence here is indistinguishable from a decision.
    This checkpoint was trained with `google_search` and `read_page` alongside
    the interpreter, so it may well reach for them; if such a call were simply
    filtered out, the loop would see "no tool call", treat the turn as an
    answer, and score a protocol failure as a deliberate direct reply. Every
    unrecognised call is surfaced and answered with an explicit
    `unknown_tool` observation.
    """

    content: str
    calls: list[ToolCall]
    unknown: list[dict[str, Any]]

    @property
    def called_a_tool(self) -> bool:
        """True if the model tried to use *any* tool, recognised or not."""
        return bool(self.calls or self.unknown)


def normalize(message: dict[str, Any], *, tool_name: str = TOOL_NAME) -> ParsedTurn:
    """Pull tool calls out of an OpenAI-shaped message.

    `mlx_lm.server` parses the model's `<tool_call><function=...>` XML with its
    own `qwen3_coder` parser and hands back structured `tool_calls`, so there is
    no regex here on purpose — a second parser would be a second thing to get
    wrong. `arguments` arrives as a JSON string.
    """
    calls: list[ToolCall] = []
    unknown: list[dict[str, Any]] = []
    for entry in message.get("tool_calls") or []:
        function = entry.get("function") or {}
        name = function.get("name")
        if name != tool_name:
            unknown.append({"name": name, "raw": entry})
            continue
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {"code": arguments}
        calls.append(ToolCall(name=name, code=(arguments or {}).get("code", ""), raw=entry))
    return ParsedTurn(content=message.get("content") or "", calls=calls, unknown=unknown)


# The functions bound inside the REPL session. Kept here because the
# unknown-tool reply needs them: a model that calls one of these as an external
# tool is *almost right*, and the reply has to say so.
REPL_FUNCTIONS = (
    "semantic_search", "semantic_map", "retry_failed", "llm_query",
    "llm_query_batched",
    "rlm_query", "rlm_map",
    "search_context", "partition_context", "read_context",
    "save_artifact", "load_artifact",
    # Conditionally bound only in the experimental presentation window, after
    # an invalid candidate and a frozen question-derived specification exist.
    "check_presentation", "render_presentation",
    # A model that calls `submit` as a tool gets the same correction as one
    # calling `semantic_map` as a tool: it is a Python function, here is how to
    # invoke it. The delivery channel being a function is the whole point, so
    # it must be reachable through the same mistake-recovery as the rest.
    "submit",
)


def unknown_tool_observation(name: str | None, *, tool_name: str = TOOL_NAME) -> dict[str, Any]:
    """Answer an unrecognised tool with the same menu shape as a duplicate.

    Two cases, and conflating them cost an episode. In the directed-map run the
    model called `semantic_search` as an external tool — following its
    instructions — and the old reply said "There is no tool named
    'semantic_search' in this session". The model drew the natural conclusion,
    "the semantic_search tool is not available", and fell back to regex. The
    function existed the whole time, one indirection away. A name that matches
    a bound REPL function now gets told it was almost right and shown the
    invocation, instead of being told the thing does not exist.
    """
    if name in REPL_FUNCTIONS:
        return {
            "ok": False,
            "error": "not_a_tool_but_a_function",
            "message": (
                f"{name} DOES exist, but it is a Python function inside your "
                f"session, not a separate tool. Call {tool_name} with code that "
                f"invokes it, for example:\n"
                f"    result = {name}(...)\n"
                f"    print(result)"
            ),
            "available_tools": [tool_name],
            "next_actions": [
                f"call {tool_name} with Python code that calls {name}(...)",
                "the data is already in `context`; you do not need to fetch anything",
            ],
        }
    return {
        "ok": False,
        "error": "unknown_tool",
        "message": (
            f"There is no tool named {name!r} in this session. "
            f"The only tool available is {tool_name}, which runs Python."
        ),
        "available_tools": [tool_name],
        "next_actions": [
            f"call {tool_name} with code that does the same work",
            "the data is already in `context`; you do not need to fetch anything",
            "answer directly if you already have what you need",
        ],
    }


# --- duplicate detection ---------------------------------------------------
def _without_comments(code: str) -> str:
    """Strip comments and blank lines without touching `#` inside strings.

    `probe_11`'s repeats were byte-identical, so exact matching was enough for
    what had been seen. Then a smoke episode re-ran a failing block whose only
    difference from the first attempt was the comment above it — same code, same
    TypeError, and the duplicate check let it through. Two blocks that differ
    only in commentary do the same thing, and running the second one teaches the
    model nothing it did not already know.
    """
    try:
        import io
        import tokenize

        kept: list[tuple[int, str]] = []
        for token in tokenize.generate_tokens(io.StringIO(code).readline):
            if token.type in (tokenize.COMMENT, tokenize.NL):
                continue
            kept.append((token.start[0], token.string))
        lines: dict[int, list[str]] = {}
        for row, text in kept:
            lines.setdefault(row, []).append(text)
        return "\n".join(
            " ".join(parts).strip() for _, parts in sorted(lines.items())
            if " ".join(parts).strip()
        )
    except (tokenize.TokenError, IndentationError, SyntaxError, ValueError):
        # Unparseable code is still code the model sent; fall back rather than
        # silently declaring two different broken blocks identical.
        return code


def code_key(code: str) -> str:
    """Identity of a block: what it does, not how it is annotated."""
    stripped = _without_comments(code.strip())
    normalized = "\n".join(line.rstrip() for line in stripped.splitlines())
    return hashlib.sha256(normalized.encode()).hexdigest()


@dataclass
class CallLedger:
    """Remembers executed blocks, so a repeat is recognised, told and counted.

    It used to refuse them, and the refusal said, in words, "its result is
    unchanged". In a session with state that is not something the harness can
    know: `results.append(3)` between two identical `print(len(results))` makes
    the second one right and different. The menu it carried said it a second
    time — "reuse the previous result, it is unchanged" — so the model was told
    a false thing twice in one observation, which is this project's recurring
    defect exactly: the harness affirming more than it knows.

    It was also the worst-measured message on record: 90 firings, 19 next-turn
    recoveries, 21%. So the claim goes and the behaviour founded on it goes
    with it. What stays is the part that is true and observational — this is
    the same code as turn N — and the consecutive-repeat stop, which asserts
    only that three identical blocks running is not progress.
    """

    seen: dict[str, dict[str, Any]] = field(default_factory=dict)
    duplicates: int = 0

    def record(self, code: str, turn: int) -> None:
        """Remember a call that ran, keyed by its code with comments stripped. Keyed
        that way because a repeat differing only in commentary is still a repeat, and
        the model produced exactly that after being refused once.

        The turn is all that is kept. It used to store the rendered result too,
        so the refusal could replay it; with the refusal gone nothing read that
        field, and the loop was rendering every observation on every turn to
        put a string in it that no one would look at.
        """
        self.seen[code_key(code)] = {"turn": turn}

    def duplicate_of(self, code: str) -> dict[str, Any] | None:
        """The earlier run of this same code, or None."""
        return self.seen.get(code_key(code))

    def note_repeat(self, previous: dict[str, Any]) -> dict[str, Any]:
        """Fields to merge into the real observation of a block that ran before.

        The block ran, so the result below the note is this turn's, not that
        turn's. Naming the earlier turn is what lets the model compare the two
        and see for itself whether anything moved — which is strictly more
        information than being told nothing moved and shown last turn's output.
        """
        self.duplicates += 1
        return {
            "repeated_from_turn": previous["turn"],
            "repeat_note": (
                f"This is the same code as turn {previous['turn']}. It ran "
                "again; the result below is this turn's. If it is the same, "
                "repeating it will not move the answer."
            ),
        }
