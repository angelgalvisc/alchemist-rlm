"""The persistent Python session, in its own process.

Isolation here is *operational*, not a security boundary: it keeps a runaway
loop or a MemoryError from taking the harness with it, and lets the parent kill
a hung block. The plan says so explicitly, and this file will not pretend
otherwise — hostile input needs a real sandbox, which is out of the first
skeleton.

Two behaviours exist because of measured failures:

**The observation does not depend on the model remembering to `print`.** Every
execution returns the value of a trailing expression and a compact view of what
changed in the namespace. `probe_11` re-ran the same conditional print three
times, each returning nothing, with no way to tell "no matches" from "did not
run" — an empty stdout is genuinely ambiguous and the fix is to say more than
stdout.

**`llm_query` is a call back to the parent.** It does not open a socket from in
here. The parent owns the budget, the trace and the concurrency limit, and a
sub-LM call that the parent never saw is a sub-LM call that cannot be scored.
"""

from __future__ import annotations

import ast
import builtins
import copy
import hashlib
import io
import json
import os
import random
import re
import sys
import time
import traceback
from types import ModuleType
from contextlib import redirect_stderr, redirect_stdout
from itertools import islice
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from alchemist_rlm.repl import rpc          # noqa: E402
from alchemist_rlm.inferred_presentation import (  # noqa: E402
    check_presentation,
    render_presentation,
)
from alchemist_rlm.output_contract import canonical_answer_value  # noqa: E402

random.seed(int(os.environ.get("RLM_WORKER_RANDOM_SEED", "0")))
if hasattr(time, "tzset"):
    time.tzset()

STDOUT_LIMIT = 20_000                        # the RLM paper's own truncation
REPR_LIMIT = 400
STATE_LIMIT = 12
# Above this, a structured value crosses the channel as its own JSON text
# rather than as a structure. Not a ceiling on the answer: text of any size has
# always travelled whole, and this is the size past which the parent stops
# wanting a live object and starts wanting the bytes.
STRUCTURED_LIMIT = 20_000


class _Unset:
    """Absence, distinct from every value a model may legitimately deliver.

    `None`, `0`, `False`, `""` and `[]` are all answers. A sentinel object is
    the only way to say "no answer offered" without excluding one of them, and
    excluding one of them is precisely the mistake the previous contract made.
    """

    def __repr__(self) -> str:                                # pragma: no cover
        return "<unset>"


_UNSET = _Unset()


class SubmitRefused(Exception):
    """A delivery the session will not accept, raised into the model's code.

    An exception rather than a returned status because `submit(x)` that quietly
    does nothing is the worst outcome available: the model believes it has
    answered and the episode ends with nothing. Raising also fails the block,
    which under the transactional rule means no delivery — the safe direction.
    """


_PRESENTATION_SOURCE_NAMES = frozenset({
    "PRESENTATION_VALUE", "PRESENTATION_TEXT", "PRESENTATION_CONTRACT",
    "PRESENTATION_DRAFT", "PRESENTATION_RENDERED", "PRESENTATION_SPEC",
    "check_presentation", "render_presentation",
})
_PRESENTATION_FORBIDDEN_CALLS = frozenset({
    "llm_query", "llm_query_batched", "rlm_query", "rlm_map",
    "semantic_map", "semantic_search", "retry_failed", "read_context",
    "search_context", "partition_context", "save_artifact", "load_artifact",
})


def _presentation_submit_calls(tree: ast.AST) -> list[ast.Call]:
    """Direct submit calls in a presentation block."""
    return [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "submit"
    ]


def _commits_presentation_draft(tree: ast.AST) -> bool:
    """Whether the block confirms exactly the harness-held stdout draft."""
    calls = _presentation_submit_calls(tree)
    if len(calls) != 1:
        return False
    call = calls[0]
    values = list(call.args) + [
        keyword.value for keyword in call.keywords
        if keyword.arg in {"final_text", "result", "candidate"}
    ]
    return (
        len(values) == 1
        and isinstance(values[0], ast.Name)
        and values[0].id == "PRESENTATION_DRAFT"
        and all(keyword.arg in {"final_text", "result", "candidate"}
                for keyword in call.keywords)
    )


def _presentation_preflight(tree: ast.AST, *, commit_required: bool,
                            draft_ready: bool,
                            source_name: str | None) -> dict[str, Any] | None:
    """Keep terminal presentation local without forbidding useful drafts."""
    assigned_sources = sorted({
        node.id for node in ast.walk(tree)
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, (ast.Store, ast.Del))
        and node.id in _PRESENTATION_SOURCE_NAMES
    })
    if assigned_sources:
        return {
            "type": "PresentationSourceImmutable",
            "message": (
                f"{', '.join(assigned_sources)} are harness-held presentation "
                "sources; build a new variable instead"
            ),
        }
    forbidden_calls = sorted({
        node.func.id for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _PRESENTATION_FORBIDDEN_CALLS
    })
    reads_context = any(
        isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id in {"context", "question"}
        for node in ast.walk(tree)
    )
    if forbidden_calls or reads_context:
        detail = ", ".join(forbidden_calls) or "context"
        return {
            "type": "PresentationIsolationRequired",
            "message": (
                f"presentation repair cannot use {detail}; transform the "
                "already committed persistent value locally"
            ),
        }
    if draft_ready and not _commits_presentation_draft(tree):
        return {
            "type": "PresentationDraftCommitRequired",
            "message": (
                "the validated stdout draft is already bound; confirm it with "
                "submit(PRESENTATION_DRAFT)"
            ),
        }
    if commit_required and not _presentation_submit_calls(tree):
        return {
            "type": "PresentationCommitRequired",
            "message": (
                "this is a commit-only presentation turn; submit the complete "
                "string variable now"
            ),
        }
    if not _presentation_submit_calls(tree):
        data_names = {
            "PRESENTATION_VALUE", "PRESENTATION_TEXT", "PRESENTATION_DRAFT",
            "PRESENTATION_RENDERED",
        }
        if source_name is not None:
            data_names.add(source_name)
        uses_source = any(
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id in data_names
            for node in ast.walk(tree)
        )
        if not uses_source:
            return {
                "type": "PresentationSourceRequired",
                "message": (
                    "a clean presentation build must use PRESENTATION_VALUE, "
                    "PRESENTATION_TEXT, PRESENTATION_DRAFT, or the directly "
                    "submitted persistent source variable; these are direct "
                    "Python globals, not modules or files, so reference the "
                    "name directly in the block"
                ),
            }
    return None


def _transportable(value: Any) -> str | None:
    """Why this value cannot cross to the parent, or None if it can.

    The session runs in its own process, so an answer has to survive the
    channel. A generator, a file handle or a live object is not an answer that
    can be delivered; saying so at the call is far better than at the end of
    the episode, when there are no turns left to fix it.

    The test is `_value_frame` itself rather than a second rule that ought to
    agree with it. Two rules were written and they did not agree: this function
    accepted a `set` because `json.dumps(value, default=str)` stringifies one,
    while `_value_frame` rejects anything that is not a list, tuple or dict. So
    `submit(a_set)` was accepted, committed, and reported `delivered: True`,
    and what reached the parent was `value: None` — the model told it had
    answered, with nothing arriving. That is the exact outcome `SubmitRefused`
    exists to prevent, and it was reachable because the permissive rule stood
    guard over the strict one. A gate that does not run the thing it guards is
    a guess about it.
    """
    frame = _value_frame(value)
    if frame.get("unserialisable"):
        return (f"{frame.get('describe', type(value).__name__)} cannot cross out "
                "of this session. Convert it to a list, dict, string or number "
                "first")
    return None


def _value_frame(value: Any) -> dict[str, Any]:
    """One value, ready to cross the channel, never substituted for.

    The channel used to describe values it found inconvenient: a structure
    whose JSON ran past 20,000 characters came back as the string "list, 3227
    items", and the engine recorded that sentence as the episode's answer. A
    model that had swept the context and built the pairs correctly had its work
    replaced by a description of its work. Strings of any size already
    travelled whole, so the cliff was never a transport limit.

    Structures under the limit travel as structures, because the parent
    introspects small ones. Past it, what travels is the value's own JSON
    *text* — the same bytes a string of that size always carried. Only a value
    that cannot be serialised at all degrades, and the frame says so instead of
    pretending the description is the value.

    `json.dumps(default=str)` would happily turn a generator into
    "<generator object <genexpr> at 0x…>", so the top-level type decides:
    text, a number, or a list/dict of them. `default` still handles a datetime
    *inside* a structure, which is a lossless enough rendering of something the
    model really did compute.
    """
    frame: dict[str, Any] = {"is_str": isinstance(value, str)}
    if isinstance(value, (str, int, float, bool, type(None))):
        frame["value"] = value
    elif not isinstance(value, (list, tuple, dict)):
        frame.update(value=None, unserialisable=True, describe=_describe(value))
    else:
        try:
            encoded = json.dumps(value, default=str)
        except (TypeError, ValueError):
            frame.update(value=None, unserialisable=True, describe=_describe(value))
        else:
            if len(encoded) <= STRUCTURED_LIMIT:
                frame["value"] = value
            else:
                frame.update(value=None, rendered=encoded,
                             chars=len(encoded), describe=_describe(value))
    return frame


def _truncate(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    half = limit // 2
    return f"{text[:half]}\n...[{len(text) - limit} chars omitted]...\n{text[-half:]}", True


def _failing_scope(exc: BaseException, code: str) -> tuple[dict[str, Any], str]:
    """Where the model's code raised: its variables, and the line itself.

    A counteroffer may only name objects that took part in the failure, and
    this is the whole of what took part — the scope at the point it broke, and
    the one line of the model's own source that broke. Nothing else is
    evidence, and searching anything else is what produced the reply that cost
    query 14 its episode.

    The line is the strongest of the two. `user_data[26503]` names the dict
    outright, which is better than inferring an owner from a session that may
    hold a dozen of them.
    """
    scope: dict[str, Any] = {}
    line = ""
    tb = exc.__traceback__
    lines = code.splitlines()
    while tb is not None:
        if tb.tb_frame.f_code.co_filename == "<rlm>":
            scope = tb.tb_frame.f_locals
            index = tb.tb_lineno - 1
            line = lines[index] if 0 <= index < len(lines) else ""
        tb = tb.tb_next
    return scope, line


def _key_owner(scope: dict[str, Any], line: str, missing: Any,
               skip: set[str] | None = None) -> tuple[str, dict[Any, Any]] | None:
    """The dict this key is missing from, named by the line that raised.

    Candidates are the dicts the *model* bound — the harness's own bindings are
    excluded, because `context_manifest` lacks every key the model ever looks
    up and was enough on its own to make every owner ambiguous — and then the
    failing line decides between them.

    Unique or nothing. Two candidates means the harness does not know which
    dict raised, and a guess between them is the mistake this exists to stop
    making. Silence is a legitimate answer: Python has already said the key is
    absent, and a wrong pointer added to that is worse than none.
    """
    import re as _re

    candidates = []
    for name, value in scope.items():
        if name.startswith("_") or name in (skip or ()) or not isinstance(value, dict):
            continue
        try:
            if missing not in value:
                candidates.append((name, value))
        except TypeError:                       # unhashable key, not our business
            continue
    if not candidates:
        return None
    named = _re.findall(r"\b([A-Za-z_]\w*)\s*\[", line)
    on_the_line = [c for c in candidates if c[0] in named]
    if len(on_the_line) == 1:
        return on_the_line[0]
    return candidates[0] if len(candidates) == 1 else None


def _model_traceback(exc: BaseException) -> str:
    """The traceback with only the model's own frames in it.

    Python's default names every frame it passed through, and most of them are
    ours: the absolute path of `worker.py`, the line inside it, `_map`,
    `_fragments`, `Session`, and the `exec(compile(...))` that ran the block.
    Three separate problems, and the third is the one that cost a day.

    It tells the model nothing. A line number inside the harness is not
    something it can act on, and the frame it *can* act on — its own — is one
    line in the middle of the noise.

    It leaks. Internal names and the absolute path, including the user's home
    directory, reach the model on every exception. That contradicts an
    invariant this project already holds everywhere else: the harness's own
    world stays out of the model's.

    **And it makes the harness irreproducible against its own source.** The
    line number of `execute` inside `worker.py` travels into the model's
    context. Adding thirty-four lines of docstring above it — a change that
    touches nothing the model can call — moved `line 1047` to `line 1081`, and
    query 9 diverged at the turn after its first exception and lost the 0.631
    it had held across four runs. The only textual difference between the two
    observations was that number. Measured over every run on record: 264 of 264
    exception observations carry these frames, across 112 of 166 episodes. Any
    edit to this file, including a comment, could reroute two thirds of them.

    So the frames are filtered to `<rlm>`, which is the name the model's code is
    compiled under. The type, the message and the model's own line survive,
    because those are the parts it can do something about.
    """
    frames = [f for f in traceback.extract_tb(exc.__traceback__)
              if f.filename == "<rlm>"]
    tail = traceback.format_exception_only(type(exc), exc)
    if not frames:
        # Raised before any of the model's code ran, or entirely inside the
        # harness. There is no frame of the model's to show, and inventing one
        # would be worse than showing none.
        return "".join(tail)
    return "".join(["Traceback (most recent call last):\n"]
                   + traceback.format_list(frames) + tail)


def _host_error(kind: str | None, message: str) -> BaseException:
    """Rebuild a host-side exception on this side of the channel.

    Resolved against `builtins` rather than a hand-kept list, so the set of
    types that survive the crossing is "the ones Python defines" and cannot
    drift away from what a model would write in an `except` clause. Anything
    else — the harness's own `SchemaError`, say — keeps its name in the message
    and arrives as RuntimeError, which is what everything used to do.
    """
    cls = getattr(builtins, kind, None) if kind else None
    if isinstance(cls, type) and issubclass(cls, Exception):
        return cls(message)
    return RuntimeError(f"{kind}: {message}" if kind else message)


def _assigned_names(tree: ast.AST) -> set[str]:
    """Every plain name this code would bind if it ran to the end.

    Read off the syntax, so it costs nothing and cannot be wrong about intent:
    an assignment target, a loop variable, a `with ... as`, a comprehension
    target, an import alias, a def or a class. Attribute and subscript targets
    are left out because `d['k'] = 1` binds no new name.

    Used only to tell a later NameError apart from a typo. Over-collecting
    would make the note fire on names the block never meant to define, so the
    walk stays on `ast.Name` in a store context and the handful of statements
    that bind without one.
    """
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update((alias.asname or alias.name).split(".")[0]
                         for alias in node.names)
    return names


def _describe(value: Any) -> str:
    """A short, honest summary. Length first: for a RLM the size of a variable
    is usually the actionable fact, and `repr` of a 200,000-char string is not."""
    try:
        if isinstance(value, str):
            return f"str, {len(value)} chars"
        if isinstance(value, (list, tuple, set, dict)):
            return f"{type(value).__name__}, {len(value)} items"
        # A module's repr is its absolute path on this machine. `import re` was
        # putting the user's home directory, the Python build and the CPU
        # architecture into the model's context — 148 times across 130 of 166
        # episodes, more widely than the traceback frames were. The same three
        # problems as those: nothing the model can act on, a leak of the host,
        # and a string that changes when the interpreter does, so a trajectory
        # that imports anything hangs off a path nobody would think to call an
        # input.
        if isinstance(value, ModuleType):
            return f"module {value.__name__}"
        text = repr(value)
        # Default object/function reprs carry process-specific addresses. They
        # are not actionable and made byte-identical runs expose different
        # observations solely because ASLR chose another address.
        text = re.sub(r"0x[0-9a-fA-F]+", "0xADDR", text)
        return text if len(text) <= REPR_LIMIT else f"{text[:REPR_LIMIT]}..."
    except Exception:                                              # noqa: BLE001
        return f"<{type(value).__name__}, repr failed>"


def _regex_empty_alternative(tree: ast.AST) -> str | None:
    """Return a literal regex containing an unescaped ``||``, if any.

    Python accepts ``a||b``: the middle branch is empty and the expression can
    therefore match without consuming the delimiter the author usually meant.
    That is legal syntax, so ``re`` emits no warning.  We report only constant
    patterns passed directly to the standard ``re`` entry points, and never
    alter execution or guess a replacement.
    """
    entry_points = {
        "compile", "findall", "finditer", "fullmatch", "match", "search",
        "split", "sub", "subn",
    }
    literals = {
        node.targets[0].id: node.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    }
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        function = node.func
        if not (isinstance(function, ast.Attribute)
                and isinstance(function.value, ast.Name)
                and function.value.id == "re"
                and function.attr in entry_points):
            continue
        argument = node.args[0]
        pattern = (argument.value
                   if isinstance(argument, ast.Constant)
                   and isinstance(argument.value, str)
                   else literals.get(argument.id)
                   if isinstance(argument, ast.Name)
                   else None)
        if not isinstance(pattern, str):
            continue
        escaped = False
        in_class = False
        for index, char in enumerate(pattern):
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == "[":
                in_class = True
                continue
            if char == "]" and in_class:
                in_class = False
                continue
            if (char == "|" and not in_class
                    and index + 1 < len(pattern)
                    and pattern[index + 1] == "|"):
                return pattern
    return None


def _map_note(result: dict[str, Any]) -> str:
    """What to do next, conditioned on what actually happened.

    The complete wording is unchanged from the run that worked (t11: 787/787,
    aggregated in Python, answered through `Final`). The other two exist
    because recommending that same move beside a `failed_items` list is what
    lost ten users in t14.
    """
    valid, total = result["valid_items"], result["total_items"]
    if result["status"] == "complete":
        return ("The returned envelope's `rows` list is complete; do not "
                "reload `rows_ref`. Each row is "
                "{'item','value','source'} — 'source' is the text that item "
                "was judged on, so no lookup back into `context` is needed, "
                "and 'item' is its position in what was judged, so anything "
                "you already hold per item lines up by index rather than by "
                "parsing 'source' again. "
                "Aggregate in Python and deliver with submit(result).")
    if result["status"] == "partial":
        retry = ("The explicit retry is exhausted; do not rerun the whole "
                 "sweep automatically. " if result.get("retry_exhausted") else
                 "Call retry_failed(result) once to resend only those items. ")
        return (f"PARTIAL: {total - valid} of {total} items have no validated "
                "value after the internal pass; they are in `failed` and are "
                f"absent from `rows` ({valid} validated rows). {retry}Do "
                "not aggregate as if coverage were complete: any total you "
                f"compute covers {valid} of {total} items, and an answer built "
                "from it must say so.")
    if result["status"] == "unsent":
        return _unsent_note(total)
    retry = ("The explicit retry is exhausted; do not rerun the whole sweep "
             "automatically." if result.get("retry_exhausted") else
             "Call retry_failed(result) once to resend the registered failed "
             "items; do not reconstruct the request from this envelope.")
    return ("FAILED: no item received a validated value; `rows` is empty. "
            + retry)


def _unsent_note(total: int) -> str:
    """No request was issued, so nothing here is evidence about the caller.

    The note this replaces said "the instruction or schema may not fit this
    data" for every empty sweep, including the ones where no sub-call was ever
    made. Three runs took that advice and spent their remaining turns rewriting
    an instruction that was never the problem; one of them wrote the comment
    "Maybe the issue is that the instruction is not clear enough" while its
    previous sweep of the same items had returned 787 of 787.

    So it names the actual cause and offers the only two moves that exist at
    that point. Neither is "try a different instruction".
    """
    return (f"NOT SENT: the run has no budget left, so none of the {total} "
            "items was sent and no sub-model saw them. This says nothing about "
            "your instruction or schema — do not rewrite them. Answer from "
            "what you already have, or deliver the earlier result if one is "
            "still bound.")


def _search_note(result: dict[str, Any]) -> str:
    """The boolean case of `_map_note`, under this operation's own names."""
    valid, total = result["valid_items"], result["total_items"]
    if result["status"] == "complete":
        return ("Every unit got a yes/no decision. The envelope's `rows` holds "
                "one {'item','value','source','start','end'} per unit — "
                "'source' is the unit's own text. Count in "
                "Python: submit(result['positive_count']).")
    if result["status"] == "partial":
        retry = ("The explicit retry is exhausted; do not rerun the whole "
                 "sweep automatically. " if result.get("retry_exhausted") else
                 "Call retry_failed(result) once to resend only those units. ")
        return (f"PARTIAL: {total - valid} of {total} units have no decision "
                f"after the internal pass; they are in `failed`. {retry}"
                "`positive_count` counts only the decided units — do not "
                "report it as a count over everything, and say what it covers "
                "in any answer built from it.")
    if result["status"] == "unsent":
        return _unsent_note(total)
    retry = ("The explicit retry is exhausted; do not rerun the whole sweep "
             "automatically." if result.get("retry_exhausted") else
             "Call retry_failed(result) once; do not rebuild or broaden the "
             "request.")
    return ("FAILED: no unit received a decision; `rows` is empty. " + retry)


class Session:
    """The namespace the model programs against.

    The split between local and host functions is not stylistic. Partitioning
    and literal search are pure computation over a string that is *already here*
    — routing them through the parent would push the whole context back across
    the channel for no gain. Anything that spends inference, budget or disk goes
    to the parent, because those are the things that must be metered and traced,
    and a subcall the parent never saw is a subcall that cannot be scored.
    """

    def __init__(self, out: Any) -> None:
        self.out = out
        self.pending = 0
        self.store: Any = None
        # The harness's own record of what ran, held OUTSIDE the namespace so
        # that `semantic_result = {...}` typed by the model does not become
        # the episode's account of its coverage. The engine reads it through
        # its own channel, never through `peek`.
        #
        # Separated from the ordinary namespace — NOT resistant to adversarial
        # code. The REPL runs Python without a sandbox, and the bound methods
        # in the namespace reach their session:
        # `semantic_map.__self__.audit["operations"].append(...)` was
        # reproduced against this code and succeeded. It defends against the
        # accidental collision that was measured, not against a model that
        # sets out to forge its own record. An account that has to survive
        # that has to be rebuilt in the parent from the trace, whose events
        # this process never writes.
        self.audit: dict[str, Any] = {
            "operations": [], "sweeps": [], "semantic_cache_hits": [],
            "presentation_checks": [], "presentation_renders": []}
        # Exact, complete semantic sweeps are idempotent within one session.
        # The cache lives outside the model namespace so assigning over
        # `semantic_rows` cannot corrupt it. Partial sweeps are never cached:
        # retrying one may legitimately recover missing values.
        self._semantic_cache: dict[str, dict[str, Any]] = {}
        # Private request state behind every model-visible ``sweep_id``.  A
        # retry is reconstructed from this record, never from an envelope the
        # model can freely mutate.
        self._sweep_registry: dict[str, dict[str, Any]] = {}
        # Successful calls across the host boundary are committed work even if
        # later Python in the same block raises.  The loop uses this monotonic
        # counter to distinguish "the operation failed" from "the operation
        # completed and the model mishandled its return value".
        self._successful_host_calls = 0
        # What the last sweep produced, held by the harness so its own wrappers
        # never have to read the model's namespace to find it.
        self._last_rows: list[dict[str, Any]] = []
        self._last_sweep: dict[str, Any] = {}
        # name -> the exact object the harness bound under it, so an
        # overwrite is detected by identity rather than by guessing a type.
        self._bound_tables: dict[str, Any] = {}
        self.question: str | None = None
        # Delivery. Content and presentation commit together on the first
        # submit, then content is immutable. A later presentation window can
        # accept text only and never clears `_submitted`.
        self._submitted: Any = _UNSET
        self._pending: Any = _UNSET
        self._submitted_final_text: Any = _UNSET
        self._pending_final_text: Any = _UNSET
        self._presentation_open = False
        self._presentation_commit_required = False
        self._presentation_draft_ready = False
        self._presentation_source_name: str | None = None
        self._inferred_presentation_spec: dict[str, Any] | None = None
        self._presentation_candidate: Any = _UNSET
        self._pending_presentation: Any = _UNSET
        self._short_batch: tuple[int, int] | None = None
        self._submit_calls = 0
        self._presentation_windows = 0
        # Names a block was going to assign and did not, because it raised
        # first. Python's NameError cannot distinguish "you never wrote this"
        # from "the block that would have written it died halfway", and those
        # call for opposite next moves. The session knows which, so it says so.
        self._aborted_names: set[str] = set()
        self.namespace: dict[str, Any] = {"__name__": "__rlm__"}
        self.namespace["submit"] = self._submit
        # Every name the harness put here itself. A counteroffer may only
        # name objects the model bound, so the harness's own bindings are
        # not candidates: `context_manifest` is a dict and lacks every key
        # the model ever looks up, which was enough to make the owner
        # ambiguous and silence a reply that had something true to say.
        self._harness_names: set[str] = set()
        # The harness's own handles on the host functions, held apart from the
        # copies it puts in the namespace for the model to call.
        #
        # Reading them back out of the namespace was a hole in the central
        # claim, not a style problem. Reproduced: a session that ran
        # `llm_query_batched = lambda jobs: [...]` and then `semantic_search()`
        # got coverage 1.0 and a certificate reading "complete" over labels the
        # model wrote itself, with ZERO sub-model calls made. The certificate
        # says every unit "was sent and answered"; nothing was sent. The same
        # shape applied to `save_artifact`: a replacement returned
        # `artifact://not_a_real_file`, which was recorded as `rows_ref`
        # alongside a real digest of rows no file holds.
        #
        # Same doctrine as the audit: what the harness needs to do its job, and
        # to attest to it afterwards, never travels through the model's world.
        self._host: dict[str, Any] = {}
        for name in ("llm_query", "llm_query_batched", "rlm_query", "rlm_map",
                     "save_artifact", "load_artifact"):
            self._host[name] = self._host_fn(name)
            self.namespace[name] = self._host[name]
        # The model's `llm_query_batched` is wrapped; the harness's own handle
        # in `_host` is not. `semantic.run` reports its own partials and does
        # not need telling.
        self.namespace["llm_query_batched"] = self._batched
        self.namespace["partition_context"] = self._partition
        self.namespace["read_context"] = self._read
        self.namespace["search_context"] = self._search
        self.namespace["semantic_search"] = self._semantic
        self.namespace["semantic_map"] = self._map
        self.namespace["retry_failed"] = self._retry_failed

    def _batched(self, jobs: Any) -> list[Any]:
        """`llm_query_batched`, with a short return saying that it is short.

        The batch keeps what it paid for when the budget stops it midway — that
        is deliberate and tested, because discarding replies already bought is
        worse than returning them. What was missing is that the shortfall was
        silent. A caller who asked for 791 and received 600 got a list of 600
        strings and nothing else; the count is the only signal, and it has to be
        noticed to mean anything.

        Query 16 did not notice. It passed 791 jobs against a 600 sub-call
        allowance, spent all of it, carried on, and two turns later the
        `semantic_map` it had written correctly came back `unsent` over 0 of
        791 — the episode's whole budget gone before the right operation ran.

        So the shortfall is stated, with the numbers and the operation that
        sizes its own fragments. Nothing is refused and nothing is discarded;
        the harness stops being the only party that knows.
        """
        asked = len(jobs) if hasattr(jobs, "__len__") else None
        replies = self._host["llm_query_batched"](jobs)
        if asked is not None and len(replies) < asked:
            self._short_batch = (asked, len(replies))
        return replies

    def _submit(self, value: Any = _UNSET, *, final_text: Any = _UNSET,
                result: Any = _UNSET, candidate: Any = _UNSET) -> None:
        """Deliver the answer. The only way an episode ends with one.

        This replaces assigning a variable named `Final`, and the reason is a
        measured one. `Final` was both a scratch variable and the delivery
        channel, so the harness had to infer intent from the value: query 17
        computed `Total pairs: 0` from a bad parse, wrote `Final = pairs` on
        turn 8 of 14, and the episode ended there with an empty answer. The
        repair attempted was a rule about *shape* — empty containers wait,
        scalars conclude — which sorted by emptiness rather than by intent,
        left a wrong non-empty list ending the run exactly as before, and made
        the empty value losable through a path that never read `Final`. It was
        withdrawn the same day.

        `submit(value)` removes the inference instead of improving it. Saying
        "I am done" is now an act, not a shape to be interpreted, so every
        value is deliverable on equal terms: `submit([])` means none qualify,
        `submit(0)` means zero, `submit(None)` means the model chose None. An
        ordinary variable can stay a draft for the whole run.

        One call per block, and the block must finish. See `execute`.
        """
        if self._presentation_open:
            aliases = [
                (name, alias_value)
                for name, alias_value in (
                    ("result", result), ("candidate", candidate))
                if alias_value is not _UNSET
            ]
            if aliases:
                if (len(aliases) != 1 or value is not _UNSET
                        or final_text is not _UNSET):
                    raise SubmitRefused(
                        "presentation result= and candidate= are text aliases; "
                        "use exactly one and do not combine it with a positional "
                        "value, final_text=, or the other alias"
                    )
                presentation_text = aliases[0][1]
            elif value is not _UNSET and final_text is not _UNSET:
                try:
                    reaffirmed = canonical_answer_value(value)
                except TypeError as error:
                    raise SubmitRefused(
                        f"the reaffirmed presentation value is invalid: {error}"
                    ) from None
                if self._submitted is _UNSET or reaffirmed != self._submitted:
                    raise SubmitRefused(
                        "presentation cannot replace the committed answer value; "
                        "the positional value must be exactly the already "
                        "committed value"
                    )
                presentation_text = final_text
            else:
                presentation_text = (
                    final_text if final_text is not _UNSET else value)
            if presentation_text is _UNSET:
                raise SubmitRefused(
                    "this presentation window needs text: "
                    "submit(corrected_text) or submit(final_text=corrected_text)")
            if not isinstance(presentation_text, str):
                if self._inferred_presentation_spec is not None:
                    try:
                        # This submit is an explicit model decision made only
                        # after the computed value is frozen. Treat a structured
                        # candidate as a request to serialize it with the public,
                        # predeclared grammar; the parent still performs both
                        # structural and content-binding validation before any
                        # promotion.
                        presentation_text = self._render_presentation(
                            presentation_text)
                    except Exception as error:
                        raise SubmitRefused(
                            "the structured presentation candidate could not be "
                            f"serialized by the frozen specification: {error}"
                        ) from None
                else:
                    draft_hint = (
                        "; PRESENTATION_DRAFT is the rejected str candidate: "
                        "transform it and submit a new str, do not submit "
                        "PRESENTATION_VALUE itself"
                        if "PRESENTATION_DRAFT" in self.namespace else ""
                    )
                    raise SubmitRefused(
                        "presentation text must be str, got "
                        f"{type(presentation_text).__name__}{draft_hint}")
            self._submit_calls += 1
            if self._pending_presentation is not _UNSET:
                raise SubmitRefused(
                    "submit() was already called in this block. A block that "
                    "delivers twice is ambiguous, so neither presentation is accepted.")
            self._pending_presentation = presentation_text
            return

        if result is not _UNSET or candidate is not _UNSET:
            raise SubmitRefused(
                "result= and candidate= are available only after the answer "
                "value is committed; the first submit must pass the computed "
                "answer positionally"
            )
        if value is _UNSET:
            raise SubmitRefused(
                "submit() needs the answer as its argument (the computed "
                "answer is its first argument): "
                "submit(value), optionally with final_text='...'.")
        if self._submitted is not _UNSET:
            raise SubmitRefused(
                "this session has already delivered and committed its computed answer; "
                "only the harness can open a text-only presentation window")
        if final_text is not _UNSET and not isinstance(final_text, str):
            raise SubmitRefused(
                f"final_text must be str, got {type(final_text).__name__}")
        self._submit_calls += 1
        if self._pending is not _UNSET:
            raise SubmitRefused(
                "submit() was already called in this block. A block that "
                "delivers twice is ambiguous, so this one delivers nothing. "
                "Call submit() exactly once, with the final value.")
        try:
            normalized = canonical_answer_value(value)
        except TypeError as error:
            raise SubmitRefused(f"submit() did not accept this value: {error}") from None
        self._pending = normalized
        self._pending_final_text = final_text

    def open_presentation(self, initial_text: str,
                          specification: dict[str, Any],
                          draft: str | None = None, *,
                          commit_required: bool = False,
                          draft_ready: bool = False,
                          source_name: str | None = None,
                          inferred_spec: dict[str, Any] | None = None,
                          ) -> None:
        """Open one text-only window and expose immutable-source copies.

        Python cannot make arbitrary nested objects truly read-only, so the
        model receives a deep copy.  The committed record stays private and is
        later compared with the candidate by the adapter's binding.
        """
        if self._presentation_open:
            raise SubmitRefused("a presentation window is already open")
        self._presentation_open = True
        self._presentation_commit_required = bool(commit_required)
        self._presentation_draft_ready = bool(draft_ready)
        self._presentation_source_name = source_name
        self._inferred_presentation_spec = copy.deepcopy(inferred_spec)
        self._presentation_candidate = _UNSET
        self._pending_presentation = _UNSET
        self._presentation_windows += 1
        if self._presentation_windows == 1:
            self.namespace.pop("PRESENTATION_RENDERED", None)
        self.namespace["PRESENTATION_VALUE"] = copy.deepcopy(self._submitted)
        self.namespace["PRESENTATION_TEXT"] = initial_text
        self.namespace["PRESENTATION_CONTRACT"] = copy.deepcopy(specification)
        if draft is None:
            self.namespace.pop("PRESENTATION_DRAFT", None)
        else:
            self.namespace["PRESENTATION_DRAFT"] = draft
        if inferred_spec is None:
            self.namespace.pop("PRESENTATION_SPEC", None)
            self.namespace.pop("check_presentation", None)
            self.namespace.pop("render_presentation", None)
        else:
            self.namespace["PRESENTATION_SPEC"] = copy.deepcopy(inferred_spec)
            self.namespace["check_presentation"] = self._check_presentation
            self.namespace["render_presentation"] = self._render_presentation
        self._harness_names.update(_PRESENTATION_SOURCE_NAMES)

    def _check_presentation(self, text: Any) -> dict[str, Any]:
        """Run the frozen question-derived linter without changing candidate text."""
        if self._inferred_presentation_spec is None:
            raise RuntimeError("no inferred presentation specification is active")
        report = check_presentation(text, self._inferred_presentation_spec)
        encoded = text if isinstance(text, str) else repr(text)
        self.audit["presentation_checks"].append({
            "input_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
            "input_chars": len(encoded),
            "valid": bool(report.get("valid")),
            "issue_codes": [
                str(issue.get("code")) for issue in report.get("issues") or []
            ],
        })
        return copy.deepcopy(report)

    def _render_presentation(self, value: Any) -> str:
        """Render from the frozen DSL without committing or promoting text."""
        if self._inferred_presentation_spec is None:
            raise RuntimeError("no inferred presentation specification is active")
        record: dict[str, Any] = {
            "input_type": type(value).__name__,
            "spec_sha256": hashlib.sha256(json.dumps(
                self._inferred_presentation_spec, sort_keys=True,
                separators=(",", ":")).encode()).hexdigest(),
        }
        try:
            text = render_presentation(value, self._inferred_presentation_spec)
        except Exception as error:
            record.update(ok=False, error_type=type(error).__name__,
                          error=str(error)[:500])
            self.audit["presentation_renders"].append(record)
            raise
        record.update(
            ok=True,
            output_chars=len(text),
            output_sha256=hashlib.sha256(text.encode()).hexdigest(),
        )
        # The model explicitly requested this serialization. Keep its complete
        # result across the bounded presentation turns so a later commit-only
        # turn can submit it without reconstructing or printing a long string.
        # This is only a source binding: promotion still requires submit() and
        # the independent structural/content validation in the parent.
        self.namespace["PRESENTATION_RENDERED"] = text
        self.audit["presentation_renders"].append(record)
        return text

    def void_submission(self) -> None:
        """Discard a value the parent proved could not resolve or transport."""
        self._submitted = _UNSET
        self._submitted_final_text = _UNSET

    def _bind_table(self, name: str, value: Any) -> None:
        """Bind one of the sweep's names and remember what was bound there."""
        self.namespace[name] = value
        self._bound_tables[name] = value

    # --- local context operations -----------------------------------------
    def bind_context(self, text: str, **band: Any) -> dict[str, Any]:
        """Load the context into the session and return its manifest. The text becomes
        an ordinary Python variable here and is never sent back across the channel —
        that is what keeps it out of the model's window.
        """
        from alchemist_rlm.context.store import ContextStore

        self.store = ContextStore(text=text, **{k: v for k, v in band.items() if v})
        self.namespace["context"] = text
        self.namespace["context_manifest"] = self.store.manifest()
        self._harness_names.add("context_manifest")
        return self.store.manifest()

    def _need_store(self) -> Any:
        if self.store is None:
            raise RuntimeError("no context is loaded in this session")
        return self.store

    def _refuse_presentation_operation(self, name: str) -> None:
        """Block host/context work even when a persistent alias calls it."""
        if self._presentation_open:
            raise SubmitRefused(
                f"{name} is unavailable during presentation repair; transform "
                "the already committed persistent value locally"
            )

    def _partition(self, text: Any = None, *,
                   target_chars: int | None = None) -> list[str]:
        """Whole structural units packed to a target size, covering everything.

        `goal` and `mode` are gone: both were accepted and silently ignored,
        which promised behaviour that never existed. An unexpected-keyword
        TypeError now tells the truth at the call site.
        """
        self._refuse_presentation_operation("partition_context")
        from alchemist_rlm.context.segmenter import TARGET_CHARS, segment
        from alchemist_rlm.protocol import as_text

        source = (self._need_store().text if text is None
                  else as_text(text, what="`text`"))
        target = target_chars or TARGET_CHARS
        return [seg.text(source) for seg in segment(source, target_chars=target,
                                                    max_chars=max(target * 3, target))]

    def _read(self, ref_or_start: Any, end: Any = None) -> str:
        """read_context(ref) or read_context(start, end) -> str

        A slice of the context. `ref` is a segment name like 's0007'; the refs
        run from s0000 to the last one named in the opening description. With
        two numbers it is a character range instead.
        """
        self._refuse_presentation_operation("read_context")
        return self._need_store().read(ref_or_start, end)

    def _search(self, pattern: str, **kwargs: Any) -> dict[str, Any]:
        """search_context(pattern, regex=False, ignore_case=True) -> dict

        Exact matching over the whole context. Costs no inference, so it is
        worth trying before anything that does.

        Returns a **dict**, not a list. The keys that matter:
            ['hits']     the list of hits. Each is a dict with 'ref', 'line',
                         'start' and 'excerpt'.
            ['matches']  how many matches there were, as a number
            ['returned'] how many of them are in ['hits'], which is fewer when
                         a search matches more than the cap
            ['refs']     the segment refs the matches fell in

        `len(result)` is the number of keys in that dict and not the number of
        matches. This docstring exists because the one-line summary in the
        opening description said "matches in ['matches']", which reads as
        either "the matches are there" or "the count is there". It is the
        count, two episodes read it the other way, and
        `len(result['matches'])` raised "object of type 'int' has no len()"
        four times between them.

        Every key and field named above is checked against a real call by
        `test_the_search_docstring_names_keys_that_exist`. Writing this by hand
        is how the ambiguity got in; the test is what stops the next hand from
        adding a key that is not there. The first draft of this paragraph
        claimed four keys and named 'segment' and 'end' on a hit. There are
        nine keys and neither field exists.
        """
        self._refuse_presentation_operation("search_context")
        from alchemist_rlm.context.search import literal_search

        return literal_search(self._need_store(), pattern, **kwargs)

    # --- the typed leaf operation ------------------------------------------
    def _fragments(self, items: Any, schema: Any = None,
                   ) -> tuple[list[Any], dict[str, Any], list[Any], list[str]]:
        """The items to judge, the scope they came from, the fragments to send,
        and the text each item was judged on. Two sources, and the difference is
        recorded rather than blurred.

        `items=None` means the context's own structural units: the harness cut
        them, so each fragment carries the context spans it was built from and
        coverage is coverage *of the context*.

        A supplied list is text the model chose. It may be filtered, reordered
        or invented, so it earns coverage of itself and nothing more — without
        that line, a model handing over ten strings it liked would come back
        holding `coverage_complete` over a context it never swept.
        """
        from alchemist_rlm.context.segmenter import units as units_of
        from alchemist_rlm.protocol import as_text
        from alchemist_rlm.semantic import Fragment, check_schema, items_per_fragment
        from alchemist_rlm.tracing import digest

        # How many reply lines fit the sub-call's token budget, from the schema
        # alone. A record of several fields is several times an enum label, and
        # the tail of a reply that overruns comes back as *missing items*, which
        # spends the retry. `None` only reaches here from callers that build no
        # fragments; the enum default keeps them at the measured working point.
        cap = items_per_fragment(check_schema(
            schema if schema is not None else {"type": "boolean"}))

        if items is None:
            store = self._need_store()
            text = store.text
            spans = units_of(text)
            fragments = []
            for seg in store.segments():
                mine = [(gid, s, e) for gid, (s, e) in enumerate(spans)
                        if seg.start <= s < seg.end]
                if not mine:
                    continue
                # A segment holding more items than one reply can carry is split
                # on the cap. The pieces keep the segment's ref with a suffix so
                # provenance still names where they came from.
                for part, start in enumerate(range(0, len(mine), cap)):
                    chunk = mine[start:start + cap]
                    ref = seg.ref if len(mine) <= cap else f"{seg.ref}.{part}"
                    fragments.append(Fragment(
                        ref=ref,
                        ids=[gid for gid, _, _ in chunk],
                        source="\n\n".join(f"[item {gid}]\n{text[s:e].strip()}"
                                           for gid, s, e in chunk),
                        provenance=[[s, e] for _, s, e in chunk],
                        item_sources={gid: f"[item {gid}]\n{text[s:e].strip()}"
                                      for gid, s, e in chunk},
                    ))
            scope = {"kind": "context", "digest": digest(text), "total": len(spans)}
            # Exactly the text each sub-model was shown, item by item. Measured
            # need: two frozen tasks spent every remaining turn trying to get
            # from an item number back to the line it came from — via `start`,
            # via line indices, via the 787-vs-795 offset — and neither
            # arrived. The row that carries its own text has no join to get
            # wrong.
            return spans, scope, fragments, [text[s:e].strip() for s, e in spans]

        if isinstance(items, (str, bytes)):
            raise ValueError(
                "items must be a collection of texts, not a single string. "
                "Iterating a string would judge one CHARACTER per item. Pass a "
                "list — semantic_map(instruction, schema, [a, b]) — or omit "
                "items to work over the context's own units."
            )
        texts = [as_text(item, what=f"items[{i}]") for i, item in enumerate(items)]
        if not texts:
            raise ValueError("items is empty; pass at least one text")

        from alchemist_rlm.context.segmenter import TARGET_CHARS

        # Two bounds, because a fragment can be too big in two directions. The
        # character bound is about the *request*: how much source text one
        # sub-call reads. `cap` is about the *reply*: how many lines fit in the
        # sub-call's 1,024-token budget before the tail is truncated, and a
        # truncated reply reads as missing items and burns the retry. Only the
        # second varies with the schema — a three-field record is several times
        # an enum label — so it is derived from the schema rather than tuned.
        fragments, group, size = [], [], 0
        for index, body in enumerate(texts):
            rendered = f"[item {index}]\n{body.strip()}"
            if group and (size + len(rendered) > TARGET_CHARS or len(group) >= cap):
                fragments.append(Fragment(
                    ref=f"i{fragments.__len__():04d}",
                    ids=[i for i, _ in group],
                    source="\n\n".join(r for _, r in group),
                    item_sources=dict(group),
                ))
                group, size = [], 0
            group.append((index, rendered))
            size += len(rendered)
        if group:
            fragments.append(Fragment(
                ref=f"i{len(fragments):04d}",
                ids=[i for i, _ in group],
                source="\n\n".join(r for _, r in group),
                item_sources=dict(group),
            ))
        scope = {"kind": "provided_items", "digest": digest("\n".join(texts)),
                 "total": len(texts)}
        return None, scope, fragments, [body.strip() for body in texts]

    @staticmethod
    def _copy_json(value: Any) -> Any:
        """Detach a model-visible value from the runtime's private record."""
        return json.loads(json.dumps(value, default=str))

    @staticmethod
    def _sweep_identity(*, operation: str, instruction: str, schema: Any,
                        sources: list[str], scope: dict[str, Any]) -> str:
        """Content identity of the request, with explicit item boundaries."""
        payload = {
            "operation": operation,
            "instruction": instruction,
            "schema": schema,
            "sources": [{"item": i, "source": source}
                        for i, source in enumerate(sources)],
            "scope": scope,
        }
        return hashlib.sha256(json.dumps(
            payload, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False).encode()).hexdigest()

    def _envelope(self, record: dict[str, Any], *, cache_hit: bool,
                  retry_exhausted: bool) -> dict[str, Any]:
        """Derive the complete public result from private sweep state."""
        from alchemist_rlm import certificate as certificate_mod

        values = record["values"]
        sources = record["sources"]
        spans = record["spans"]
        scope = record["scope"]
        total = scope["total"]
        rows: list[dict[str, Any]] = []
        for item in sorted(values):
            row: dict[str, Any] = {
                "item": item, "value": values[item], "source": sources[item]}
            if spans is not None:
                row["start"], row["end"] = spans[item]
            rows.append(row)

        failed_ids = [item for item in range(total) if item not in values]
        complete = total > 0 and not failed_ids
        nothing_sent = total > 0 and not record["presented"]
        status = ("complete" if complete
                  else "unsent" if nothing_sent
                  else "partial" if values else "failed")
        context_complete = complete if spans is not None else None
        failed = []
        for item in failed_ids:
            entry: dict[str, Any] = {"item": item, "source": sources[item]}
            if spans is not None:
                entry["start"], entry["end"] = spans[item]
            failed.append(entry)

        result: dict[str, Any] = {
            # One shape for map and search. ``operation`` and all flat fields
            # remain during migration; new code can use the grouped fields.
            "kind": record["operation"],
            "operation": record["operation"],
            "status": status,
            "rows": rows,
            "coverage": {
                "presented": len(record["presented"]),
                "returned": len(record["returned"]),
                "valid": len(values),
                "total": total,
                "complete": complete,
                "context_complete": context_complete,
            },
            "failed": failed,
            "scope": scope,
            "sweep_id": record["sweep_id"],
            "schema": record["schema"],
            "total_items": total,
            "presented_items": len(record["presented"]),
            "returned_items": len(record["returned"]),
            "unsent_items": len(record["unsent"]),
            "valid_items": len(values),
            "coverage_complete": complete,
            "context_coverage_complete": context_complete,
            "failed_items": failed_ids,
            "parse_errors": list(record["parse_errors"]),
            "cache_hit": cache_hit,
            "retry_exhausted": retry_exhausted,
        }

        if spans is not None:
            cert = certificate_mod.from_run(
                context=self.store.text,
                spans=spans,
                result=result,
                covered_spans=[(row["start"], row["end"]) for row in rows],
            )
            result["certificate"] = cert.to_dict()

        save = self._host.get("save_artifact")
        if callable(save) and rows:
            canonical = json.dumps(rows, sort_keys=True, separators=(",", ":"))
            result["rows_digest"] = hashlib.sha256(canonical.encode()).hexdigest()
            try:
                result["rows_ref"] = save(
                    f"semantic_rows_{result['rows_digest'][:16]}", rows)
            except Exception:                                  # noqa: BLE001
                result.pop("rows_digest", None)

        if record["operation"] == "semantic_search":
            positive = [row["item"] for row in rows if row["value"]]
            result.update(
                ok=True,
                mode="semantic",
                goal=record["instruction"],
                examined_items=len(values),
                positive_ids=positive,
                positive_count=len(positive),
                stored_in="returned envelope",
            )
        result["note"] = (_search_note(result)
                          if record["operation"] == "semantic_search"
                          else _map_note(result))
        return result

    def _publish_envelope(self, result: dict[str, Any], *,
                          audit_operation: str | None = None) -> None:
        """Bind compatibility names and write a detached host audit copy."""
        rows = result["rows"]
        self._last_rows, self._last_sweep = rows, result
        self._bind_table("semantic_rows", rows)
        self._bind_table("semantic_result", result)
        if result["operation"] == "semantic_search":
            self._bind_table("search_results", [
                {"item": row["item"], "decision": row["value"],
                 "source": row["source"], "start": row["start"],
                 "end": row["end"]}
                for row in rows
            ])
        if audit_operation is not None:
            self.audit["operations"].append(audit_operation)
            self.audit["sweeps"].append(self._copy_json(result))

    def _retry_failed(self, result: Any) -> dict[str, Any]:
        """Spend at most one targeted pass on a registered sweep's failures."""
        self._refuse_presentation_operation("retry_failed")
        from alchemist_rlm import semantic

        if not isinstance(result, dict):
            raise TypeError("retry_failed(result) needs a semantic envelope dict")
        sweep_id = result.get("sweep_id")
        if not isinstance(sweep_id, str) or sweep_id not in self._sweep_registry:
            raise ValueError(
                "retry_failed refused an unknown sweep_id; pass an envelope "
                "returned by semantic_map or semantic_search in this session")
        record = self._sweep_registry[sweep_id]
        if record["retry_used"]:
            returned = self._copy_json(record["result"])
            returned["retry_exhausted"] = True
            self._publish_envelope(returned)
            return returned

        # The attempt is consumed even if the host rejects it after dispatch:
        # retrying an uncertain transport forever would violate the same bound.
        record["retry_used"] = True
        unresolved = [item for item in range(record["scope"]["total"])
                      if item not in record["values"]]
        fragments = []
        for fragment in record["fragments"]:
            wanted = [item for item in fragment.ids if item in unresolved]
            if wanted:
                fragments.append(fragment.narrow(wanted))
        if fragments:
            outcome = semantic.run(
                fragments, record["instruction"], record["schema"],
                self._host["llm_query_batched"], retry=False)
            for item, value in outcome["values"].items():
                record["values"].setdefault(item, value)
            record["presented"].update(outcome["presented"])
            record["returned"].update(outcome["returned"])
            record["unsent"].update(outcome["unsent"])
            record["unsent"].difference_update(record["presented"])
            record["parse_errors"] = list(outcome["parse_errors"])

        merged = self._envelope(
            record, cache_hit=False, retry_exhausted=True)
        record["result"] = self._copy_json(merged)
        self._publish_envelope(merged, audit_operation="retry_failed")
        if merged["status"] in {"complete", "partial"}:
            self._semantic_cache[record["cache_key"]] = {
                "result": self._copy_json(merged),
                "rows": self._copy_json(merged["rows"]),
            }
        return merged

    # A note that contradicts the counters beside it is an instruction to
    # ignore them, and that is measured, not conjectured: the directed t14 run
    # printed `failed_items: [702..741]` and `coverage_complete: False` in the
    # same observation as an unconditional "Aggregate in Python ... and put the
    # result in `Final`". The model obeyed the note, aggregated 755 decisions
    # as if they were 795, and lost ten users without ever being told to care.
    # So the note now depends on the status, and a partial sweep is never
    # advised to aggregate as if it were complete.
    def _map(self, instruction: Any = _UNSET, schema: Any = _UNSET,
             items: Any = None,
             *, _operation: str = "semantic_map") -> dict[str, Any]:
        """One validated value per item, and an honest account of the coverage.

        The general operation; `semantic_search` is its boolean case. What the
        values *mean* is entirely in `instruction` — the harness only checks
        that every item was sent, came back, and came back conforming.

        `_operation` is the name recorded in the result and in the session's
        audit — the record an episode's consumer reads to know which
        operations actually ran, as opposed to which were requested.
        """
        self._refuse_presentation_operation(_operation)
        from alchemist_rlm import semantic

        # Both parameters have a sentinel default so a call that omits one is
        # *entered* and can be answered. Required positionals meant Python
        # raised first, and what the model read was
        # `Session._map() missing 1 required positional argument: 'schema'` —
        # a name it has never been given, about an argument it had in fact
        # passed, just in the wrong slot. Query 15 got that twice and died on
        # the error guard with no sub-model call made. The swap check below is
        # exactly the help it needed and it was one arity error out of reach.
        if instruction is _UNSET:
            raise semantic.SchemaError(
                "semantic_map needs the instruction first: "
                "semantic_map(instruction, schema, items=None)")
        if schema is _UNSET:
            if isinstance(instruction, dict) and "type" in instruction:
                raise semantic.SchemaError(
                    "the only positional argument is a schema, so the "
                    "instruction is missing: semantic_map(instruction, schema, "
                    "items=None) — the instruction is what to judge each item "
                    "against")
            raise semantic.SchemaError(
                "semantic_map needs a schema as its second argument: "
                "semantic_map(instruction, schema, items=None), with schema "
                "{'type': 'string', 'enum': [...]} or {'type': 'boolean'}")
        # The order is checked before anything runs, and a swap is named as a
        # swap. Measured: the model passed its label list where the schema goes
        # and got "schema must be a dict, got list" three turns running — true,
        # and no help at all in working out that the arguments were the wrong
        # way round.
        if isinstance(instruction, dict) and "type" in instruction:
            raise semantic.SchemaError(
                "the first argument is a schema, so the arguments look swapped"
            )
        if isinstance(instruction, (list, tuple)):
            raise semantic.SchemaError(
                "the first argument is a list, so it is probably the items or "
                "the labels; the instruction is what to judge each item against"
            )
        if not isinstance(instruction, str) or not instruction.strip():
            raise semantic.SchemaError(
                f"instruction must be non-empty text, got "
                f"{type(instruction).__name__}"
            )
        spans, scope, fragments, sources = self._fragments(items, schema)
        # Validate before consulting the cache: a malformed schema must never
        # borrow the result of a superficially similar valid call.
        semantic.check_schema(schema)
        cache_key = self._sweep_identity(
            operation=_operation, instruction=instruction, schema=schema,
            sources=sources, scope=scope)
        cached = self._semantic_cache.get(cache_key)
        if cached is not None:
            result = self._copy_json(cached["result"])
            result["cache_hit"] = True
            self._publish_envelope(result)
            self.audit["semantic_cache_hits"].append({
                "operation": _operation,
                "scope": result.get("scope"),
                "rows_digest": result.get("rows_digest"),
            })
            return result
        outcome = semantic.run(fragments, instruction, schema,
                               self._host["llm_query_batched"])
        prior = self._sweep_registry.get(cache_key)
        retry_used = bool(prior and prior.get("retry_used"))
        record = {
            "sweep_id": cache_key,
            "cache_key": cache_key,
            "operation": _operation,
            "instruction": instruction,
            "schema": schema,
            "scope": scope,
            "sources": list(sources),
            "spans": spans,
            "fragments": fragments,
            "values": dict(outcome["values"]),
            "presented": set(outcome["presented"]),
            "returned": set(outcome["returned"]),
            "unsent": set(outcome["unsent"]),
            "parse_errors": list(outcome["parse_errors"]),
            "retry_used": retry_used,
        }
        result = self._envelope(
            record, cache_hit=False, retry_exhausted=retry_used)
        record["result"] = self._copy_json(result)
        self._sweep_registry[cache_key] = record
        self._publish_envelope(result, audit_operation=_operation)
        if result["status"] in {"complete", "partial"}:
            self._semantic_cache[cache_key] = {
                "result": self._copy_json(result),
                "rows": self._copy_json(result["rows"]),
            }
        return result

    def _semantic(self, goal: str | None = None) -> dict[str, Any]:
        """The boolean case of `semantic_map`, over the context's own units.

        Kept because it works and is the name the model already reaches for:
        1,600 of 1,600 items decided, coverage 1.0, zero parse errors. What is
        gone is the semantic rule it used to carry — whether a negated or
        averted mention counted was decided inside the harness, which made a
        general-looking operation wrong for any question about negated or
        averted things. The goal is now the whole of the instruction.

        With no goal the node's own question is used verbatim. That default
        exists because a rephrased goal loses part of the question at every
        hop: the neutral frame survived to each child intact and then most
        children dropped its rider when rewriting it, 143 at the root against
        123 through the tree.
        """
        self._refuse_presentation_operation("semantic_search")
        if goal is None:
            if not self.question:
                raise ValueError(
                    "semantic_search() with no goal needs the session's "
                    "question, and none is bound; pass a goal explicitly"
                )
            goal = self.question

        return self._map(
            goal, {"type": "boolean"}, _operation="semantic_search")

    # --- calling back into the parent -------------------------------------
    def _lazy(self, value: Any) -> bool:
        """A generator or a one-shot iterator: something that must not be
        materialised just to cross the channel. Lists, tuples, strings and
        dicts are already in memory, so sending them costs nothing new."""
        if isinstance(value, (str, bytes, list, tuple, dict, set)):
            return False
        return hasattr(value, "__next__") or (
            hasattr(value, "__iter__") and not hasattr(value, "__len__")
        )

    def _host_fn(self, name: str):
        def call(*args: Any, **kwargs: Any) -> Any:
            self._refuse_presentation_operation(name)
            self.pending += 1
            call_id = self.pending
            # Lazy arguments stay here. The parent receives a placeholder and
            # pulls items in fixed chunks, which is the whole point: a generator
            # over a million pairs never becomes a million objects.
            streams: dict[str, Any] = {}

            def mark(value: Any, key: str) -> Any:
                if self._lazy(value):
                    streams[key] = iter(value)
                    return {"__lazy__": key}
                return value

            sent_args = [mark(value, f"a{i}") for i, value in enumerate(args)]
            sent_kwargs = {k: mark(v, f"k:{k}") for k, v in kwargs.items()}
            rpc.send(self.out, {
                "op": "host", "id": call_id, "fn": name,
                "args": sent_args, "kwargs": sent_kwargs,
            })
            while True:
                answer = rpc.recv(sys.stdin)
                op = answer.get("op")
                if op == "pull":
                    key = answer.get("key")
                    want = int(answer.get("count") or 1)
                    stream = streams.get(key)
                    items: list[Any] = []
                    failure: str | None = None
                    if stream is None:
                        failure = f"no lazy argument {key!r}"
                    else:
                        try:
                            items = list(islice(stream, want))
                        except Exception as exc:                   # noqa: BLE001
                            failure = f"{type(exc).__name__}: {exc}"
                    rpc.send(self.out, {
                        "op": "items", "id": call_id, "key": key,
                        "items": items, "exhausted": len(items) < want,
                        "error": failure,
                    })
                    continue
                if op != "reply":
                    raise RuntimeError(f"expected a reply to {name}, got {op!r}")
                if not answer.get("ok"):
                    # Surfaced as a normal Python exception so the model's own
                    # try/except works and the traceback lands in the
                    # observation — which needs the original type, not just its
                    # name inside a string. Everything used to arrive as
                    # RuntimeError, so the sentence above was not true of the
                    # code under it: `except ValueError:` caught nothing, and
                    # the observation reported `RuntimeError: ValueError: parts
                    # must be a collection of texts`, naming the wrong
                    # exception and repeating the right one.
                    raise _host_error(answer.get("error_type"),
                                      answer.get("error") or f"{name} failed")
                self._successful_host_calls += 1
                return answer.get("value")

        call.__name__ = name
        return call

    # --- execution ---------------------------------------------------------
    def execute(self, code: str) -> dict[str, Any]:
        """Run one block and describe what happened, without ever raising.

        Errors in the model's code are the model's to see and recover from, so they
        come back as data. The observation does not depend on the model remembering
        to print: the value of a trailing expression and a summary of what changed
        in the namespace come back too, because an empty stdout cannot distinguish
        'no matches' from 'never ran'.
        """
        before = dict(self.namespace)
        host_before = self._successful_host_calls
        sweeps_before = len(self.audit["sweeps"])
        cache_before = len(self.audit["semantic_cache_hits"])
        presentation_was_open = self._presentation_open
        self._pending = _UNSET                   # each block offers afresh
        self._pending_final_text = _UNSET
        self._pending_presentation = _UNSET
        self._short_batch = None
        stdout, stderr = io.StringIO(), io.StringIO()
        value_repr: str | None = None
        error: dict[str, Any] | None = None
        counteroffer: list[str] | None = None
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            if presentation_was_open:
                self._presentation_open = False
                self._presentation_commit_required = False
                self._presentation_draft_ready = False
                self._presentation_source_name = None
            return {
                "ok": False,
                "stdout": "", "stderr": "",
                "error": {"type": "SyntaxError", "message": str(exc),
                          "traceback": "".join(traceback.format_exception_only(
                              type(exc), exc))},
                "defined": [], "changed": {}, "value": None, "truncated": False,
            }

        presentation_preflight = (
            _presentation_preflight(
                tree,
                commit_required=self._presentation_commit_required,
                draft_ready=self._presentation_draft_ready,
                source_name=self._presentation_source_name,
            ) if self._presentation_open else None
        )
        if presentation_preflight is not None:
            self._presentation_open = False
            self._presentation_commit_required = False
            self._presentation_draft_ready = False
            self._presentation_source_name = None
            return {
                "ok": False,
                "stdout": "", "stderr": "",
                "error": {**presentation_preflight, "traceback": ""},
                "value": None,
                "defined": [],
                "changed": {},
                "truncated": False,
                "delivered": self._submitted is not _UNSET,
                "presentation_candidate": False,
            }

        regex_with_empty_branch = _regex_empty_alternative(tree)
        trailing = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            trailing = ast.Expression(tree.body.pop().value)

        try:
            with redirect_stdout(stdout), redirect_stderr(stderr):
                if tree.body:
                    exec(compile(tree, "<rlm>", "exec"), self.namespace)     # noqa: S102
                if trailing is not None:
                    result = eval(compile(trailing, "<rlm>", "eval"), self.namespace)  # noqa: S307
                    if result is not None:
                        value_repr = _describe(result)
                        self.namespace["_"] = result
        except rpc.ChannelClosed:
            raise
        except BaseException as exc:                                   # noqa: BLE001
            error = {
                "type": type(exc).__name__,
                "message": str(exc)[:2000],
                "traceback": _model_traceback(exc)[-4000:],
            }
            # `from semantic_map import semantic_map` cost t20 three turns: a
            # bare ModuleNotFoundError, a duplicate refusal for retrying it,
            # and a dir() to see what existed. The function was bound the whole
            # time. Same doctrine as the unknown-tool reply: a name that is
            # almost right gets told so, with the working invocation shown.
            # On the observation, not inside `error`: that is where the
            # renderer reads counteroffers from.
            if isinstance(exc, ImportError):
                bound = getattr(exc, "name", None)
                requested = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom):
                        requested.extend(alias.name for alias in node.names
                                         if alias.name in self.namespace)
                if requested:
                    shown = requested[:6]
                    counteroffer = [
                        "the names this import requested are already defined in "
                        "this session: " + ", ".join(shown),
                        "call them directly; do not investigate Python modules",
                    ]
                    counteroffer.extend(
                        (f"call {name}(...) directly"
                         if callable(self.namespace[name])
                         else f"use {name} directly")
                        for name in shown
                    )
                elif bound and bound in self.namespace:
                    # Callable or not. The first version only answered for
                    # functions, and t16 opened with `import context` — the
                    # 78,000-character variable it needed was bound the whole
                    # time, and the reply it got was "No module named
                    # 'context'". A name that is present is present.
                    value = self.namespace[bound]
                    use = (f"call it directly: result = {bound}(...)"
                           if callable(value)
                           else f"use it directly: it is already a "
                                f"{type(value).__name__} in this session")
                    counteroffer = [
                        f"{bound} is not a module: it is already defined in "
                        f"this session",
                        use,
                        "nothing already listed in your session needs an import",
                    ]
            # A dict cannot be sliced, and the message Python gives for it —
            # "unhashable type: 'slice'" — says nothing about what to do. The
            # Summary variables are dicts, and `result[:1000]` is what a model
            # reaches for to peek at one: four episodes hit this
            # (t14, t20, t12, t16), one of them losing four turns to it. Plain
            # Python guidance, no knowledge of any task.
            elif isinstance(exc, TypeError) and "unhashable type: 'slice'" in str(exc):
                counteroffer = [
                    "a dict cannot be sliced with [:n] — that is what this "
                    "error means",
                    "to see part of one: list(d.items())[:n], or d.keys()",
                    "to read one field: d['name']",
                ]
            # The mirror of the one above: an ordinary list treated as a
            # mapping. Semantic operations now return an envelope dict, so no
            # sweep-specific ownership is inferred here.
            elif (isinstance(exc, AttributeError)
                  and "'list' object has no attribute" in str(exc)) or (
                      isinstance(exc, TypeError)
                      and "list indices must be integers" in str(exc)):
                counteroffer = [
                    "this value is a list; the attribute or key you used "
                    "belongs to a dict — that is what this error means",
                    "to see part of a list: rows[:5]; to read one: rows[0]",
                ]
            # A name that a previous block would have bound, in a block that
            # raised before binding it. Python cannot say that; the session
            # can, because it knows which names each failed block was going to
            # assign. t17 spent turns 12 and 13 on `inst_idx is not defined`,
            # a name whose assignment was below the line that raised two turns
            # earlier — retrying it as though the earlier block had half
            # worked. It had not: a block that raises leaves nothing after the
            # raise, and that is the fact worth handing back.
            # A view object indexed as if it were a list. `d.items()[0]` and
            # `enumerate(xs)[0]` are the two on record, six occurrences, and
            # the message names the internal type — `dict_items`, `enumerate` —
            # which is not a thing the model wrote and not a thing it can look
            # up. Plain Python, no knowledge of any task, and the same shape as
            # the dict-slice reply above.
            elif isinstance(exc, TypeError) and "not subscriptable" in str(exc):
                kind = str(exc).split("'")[1] if "'" in str(exc) else "this value"
                counteroffer = [
                    f"a {kind} cannot be indexed with [] — that is what this "
                    "error means",
                    f"to index it, build a list first: list(...)[0]",
                    "to walk it instead: for x in ...:",
                ]
            # `len()` of a number. Seven occurrences over four episodes, and
            # every one of them is a count being treated as the thing counted —
            # the exact reading `search_context`'s annotation now guards
            # against, arriving at the moment it goes wrong rather than only in
            # the announcement the model read many turns earlier.
            elif isinstance(exc, TypeError) and "has no len()" in str(exc):
                counteroffer = [
                    "this value is a single number, not a collection — that is "
                    "what this error means",
                    "if you expected a count, you already have it: use it "
                    "directly instead of measuring it",
                    "if you expected the items, they are under a different "
                    "key or name",
                ]
            # A key that is not there, when the session is holding rows. The
            # keys are read off the rows themselves, so this states a fact
            # about data the harness produced rather than a guess about the
            # model's code — and it is phrased conditionally because the dict
            # that raised may be one of the model's own.
            # A key that is not there. What is worth saying depends entirely on
            # *which* dict raised, and the first version of this did not ask:
            # it fired on any KeyError as long as the session held sweep rows,
            # and then said two things about `semantic_rows`.
            #
            # Query 14 paid for that. It had 325 correct pairs built and was
            # stuck on `KeyError: 26503` against `user_data`, its own dict,
            # whose keys are strings — `'26503'` against `26503`, and nothing
            # else wrong. Three turns running it was told "the rows in
            # semantic_rows have exactly these keys", followed the advice on
            # turn 14 into re-parsing a sweep result that had no bearing on the
            # failure, and ran out of turns holding the answer.
            #
            # A counteroffer aimed at the wrong object is worse than silence,
            # because the model acts on it. So each branch here is checked
            # against the objects that are actually present before it speaks,
            # and when nothing can be checked, nothing is added — Python's own
            # message already says the key is missing.
            # A key that is not there, answered only from the scope that
            # raised. What is worth saying depends on *which* dict failed, and
            # the version this replaces never asked: it fired on any KeyError
            # as long as the session held sweep rows, then said two things
            # about `semantic_rows`.
            #
            # Query 14 paid for it. It had 325 correct pairs built and printed,
            # and was stuck on `KeyError: 26503` against `user_data`, its own
            # dict, whose keys are strings — `'26503'` against `26503` and
            # nothing else wrong. Three turns running it was told about the
            # sweep rows, followed that on turn 14 into re-parsing a result
            # with no bearing on the failure, and ran out of turns holding the
            # answer.
            #
            # Two rules come out of that and both are enforced here rather than
            # remembered. Only objects that took part in the failure may be
            # named, which is why the search is over `_failing_locals` and not
            # the session. And silence is a legitimate answer: when the owner
            # is ambiguous, nothing is added, because Python has already said
            # the key is missing and a wrong pointer is worse than none.
            elif isinstance(exc, KeyError) and exc.args:
                missing = exc.args[0]
                scope, line = _failing_scope(exc, code)
                owner = _key_owner(scope, line, missing, self._harness_names)
                if owner is not None:
                    name, mapping = owner
                    counteroffer = [f"{name} does not have {missing!r}"]
                    # The same key under the other type is the commonest way an
                    # agent gets stuck in a REPL, and it is checkable rather
                    # than guessed at.
                    other: Any = None
                    if isinstance(missing, str) and missing.isdigit():
                        other = int(missing)
                    elif isinstance(missing, int) and not isinstance(missing, bool):
                        other = str(missing)
                    if other is not None and other in mapping:
                        counteroffer = [
                            f"{name} does have {other!r}, but not {missing!r}: "
                            f"its keys are {type(other).__name__} and you "
                            f"looked up a {type(missing).__name__}",
                            f"convert at the lookup: "
                            f"{name}[{type(other).__name__}(k)]",
                        ]
                    else:
                        shown = [repr(k) for k in list(mapping)[:6]]
                        if shown:
                            counteroffer.append(
                                f"the keys it does have: {', '.join(shown)}"
                                + (", …" if len(mapping) > 6 else ""))
            elif isinstance(exc, NameError) and getattr(exc, "name", None) \
                    in self._aborted_names:
                missing = exc.name                             # type: ignore[attr-defined]
                counteroffer = [
                    f"{missing} was going to be assigned by an earlier block "
                    "that raised before reaching that line, so it was never "
                    "bound",
                    "nothing after the point where a block raises has run — "
                    "re-run the work that defines it, do not assume any of it "
                    "survived",
                ]

        out_text, out_cut = _truncate(stdout.getvalue(), STDOUT_LIMIT)
        err_text, _ = _truncate(stderr.getvalue(), 4000)
        defined = [
            name for name in self.namespace
            if not name.startswith("__") and (name not in before or before[name] is not self.namespace[name])
        ]
        # Recorded after the counteroffer above has been decided, so a block
        # that raises NameError on a name it also assigns further down does not
        # end up explaining itself to itself. Anything this block did manage to
        # bind leaves the set: it exists now, and a NameError on it later would
        # be an ordinary typo, which is a different thing to say.
        if error is not None:
            self._aborted_names |= _assigned_names(tree)
        self._aborted_names -= set(defined)
        changed = {name: _describe(self.namespace[name]) for name in sorted(defined)[:STATE_LIMIT]}
        # Delivery commits here and only here. A block that called submit() and
        # then raised has NOT delivered: the exception says the block did not
        # finish computing, and half-finished work is exactly what must never
        # be published. So the offer is discarded and the model keeps its turn.
        #
        # The rule in one line: a block delivers if and only if it called
        # submit() exactly once and completed without raising. Everything else
        # — no call, two calls, a refused value, an exception anywhere — leaves
        # the session undelivered, with no shape to interpret and nothing to
        # guess.
        if error is None and self._pending is not _UNSET:
            self._submitted = self._pending
            self._submitted_final_text = self._pending_final_text
        if error is None and self._pending_presentation is not _UNSET:
            self._presentation_candidate = self._pending_presentation
        if presentation_was_open:
            self._presentation_open = False
            self._presentation_commit_required = False
            self._presentation_draft_ready = False
            self._presentation_source_name = None
        self._pending = _UNSET
        self._pending_final_text = _UNSET
        self._pending_presentation = _UNSET
        observation = {
            "ok": error is None,
            "stdout": out_text,
            "stderr": err_text,
            "error": error,
            "value": value_repr,
            "defined": sorted(defined),
            "changed": changed,
            "truncated": out_cut,
            "delivered": self._submitted is not _UNSET,
            "presentation_candidate": self._presentation_candidate is not _UNSET,
        }
        # A batch the budget cut short returns a shorter list and says nothing
        # else. The count is the only signal there is, and it has to be noticed
        # to mean anything — query 16 asked for 791, received 600, did not
        # notice, and spent the episode's whole sub-call allowance before the
        # operation it had written correctly could run.
        if self._short_batch:
            asked, got = self._short_batch
            observation["next_actions"] = [
                f"llm_query_batched returned {got:,} replies for {asked:,} jobs: "
                f"the sub-call budget stopped it, so {asked - got:,} were never "
                "sent and the replies you have cover only the first "
                f"{got:,} jobs",
                "semantic_map(instruction, schema, items=...) asks one question "
                "per item and packs many items into each sub-call, so it covers "
                "the same items for a fraction of the budget",
            ]
        # A host operation can finish and commit hundreds of validated rows
        # before later code in this same block raises (t16: 794/795).  ``ok``
        # describes the whole Python block; ``progress`` describes durable
        # bounded work.  The controller must not confuse them.
        observation["progress"] = (
            self._successful_host_calls > host_before
            or len(self.audit["sweeps"]) > sweeps_before
            or len(self.audit["semantic_cache_hits"]) > cache_before
        )
        if (len(self.audit["sweeps"]) > sweeps_before
                or len(self.audit["semantic_cache_hits"]) > cache_before):
            sweep = self._last_sweep
            note = (_search_note(sweep) if sweep.get("operation") == "semantic_search"
                    else _map_note(sweep))
            observation["operation_result"] = {
                key: sweep.get(key) for key in (
                    "kind", "operation", "status", "coverage", "sweep_id",
                    "retry_exhausted", "valid_items", "total_items",
                    "coverage_complete", "context_coverage_complete",
                    "failed", "failed_items", "rows_ref",
                )
            }
            observation["operation_result"]["stored_in"] = (
                "the returned envelope; rows are under ['rows']")
            observation["operation_result"]["note"] = note
        if counteroffer:
            observation["next_actions"] = counteroffer
        if regex_with_empty_branch is not None:
            observation.setdefault("next_actions", []).extend([
                "this block used a regex containing unescaped `||`; in Python "
                "regex syntax that creates an empty alternative and can match "
                "without consuming the intended text",
                "if the two pipes are literal data, escape them as `\\|\\|` "
                "or build that fragment with re.escape()",
            ])
        # Sweep bindings remain protected from later explicit assignments.
        # `semantic_map` now returns the envelope, eliminating its natural
        # table/summary collision, but legacy aliases can still be overwritten
        # `semantic_result`, `search_results`, or the table in later code.
        #
        # All three bound names, checked by identity. A first version watched
        # `semantic_rows` alone, which left `semantic_result = semantic_map(...)`
        # — every bit as natural a thing to write — silent in exactly the way
        # this exists to prevent.
        lost = [name for name, held in self._bound_tables.items()
                if self.namespace.get(name) is not held]
        if lost:
            observation.setdefault("next_actions", []).extend([
                f"{', '.join('`' + n + '`' for n in lost)} no longer "
                f"hold{'s' if len(lost) == 1 else ''} what the sweep bound — "
                "this block assigned something else over "
                f"{'it' if len(lost) == 1 else 'them'}",
                "use the envelope returned by semantic_map or semantic_search; "
                "its validated table is under ['rows']",
            ])
        return observation


def main() -> None:
    # Keep the RPC channel out of reach of anything the model's code writes to
    # real stdout: duplicate fd 1 for our own use, then point fd 1 at stderr.
    """Serve the session: read frames, run what is asked, reply until shutdown."""
    out = os.fdopen(os.dup(1), "w", encoding="utf-8")
    os.dup2(2, 1)
    session = Session(out)
    rpc.send(out, {"op": "ready", "pid": os.getpid()})
    while True:
        try:
            message = rpc.recv(sys.stdin)
        except rpc.ChannelClosed:
            return
        op = message.get("op")
        if op == "shutdown":
            return
        if op == "exec":
            result = session.execute(message.get("code") or "")
            rpc.send(out, {"op": "result", **result})
        elif op == "inject":
            for name, value in (message.get("values") or {}).items():
                session.namespace[name] = value
            rpc.send(out, {"op": "injected", "names": sorted(message.get("values") or {})})
        elif op == "bind_context":
            try:
                session.question = message.get("question") or None
                manifest = session.bind_context(message.get("text") or "",
                                                **(message.get("band") or {}))
                rpc.send(out, {"op": "bound", "manifest": manifest})
            except Exception as exc:                               # noqa: BLE001
                rpc.send(out, {"op": "bound", "error": f"{type(exc).__name__}: {exc}"})
        elif op == "audit":
            # The harness's own record, never the namespace's: `peek` reads
            # what the model can write, this reads what only the session's
            # operations wrote. JSON serialisation on send is also a copy.
            rpc.send(out, {"op": "audit", "audit": session.audit})
        elif op == "get":
            name = message.get("name")
            present = name in session.namespace
            value = session.namespace.get(name)
            # The channel never substitutes a description for a value.
            #
            # It used to: a structure whose JSON ran past 20,000 characters came
            # back as the string "list, 3227 items", and the engine accepted
            # that sentence as the episode's answer. A model that had done
            # everything right — swept the context, built the pairs in Python,
            # assigned them to `Final` — had its work replaced by a description
            # of its work, silently. Strings of any size already travelled
            # whole, so the cliff was not a transport limit either.
            #
            # Structures under the limit still travel as structures, because
            # the engine introspects small ones (a child's `semantic_result`).
            # Past it, what travels is the value's own JSON *text*, which is
            # the same bytes a string of that size has always carried. Only a
            # value that cannot be serialised at all degrades, and the frame
            # says so rather than pretending the description is the value.
            # A name holds an answer only if it holds *data*. `json.dumps` with
            # `default=str` will happily turn a generator into the string
            # "<generator object <genexpr> at 0x10643a8e0>", and that repr then
            # travels as if it were the value — the same substitution as the
            # size cliff, with no size needed to trigger it. So the top-level
            # type decides: text, a number, or a list/dict of them. `default`
            # still handles a datetime *inside* a structure, which is a lossless
            # enough rendering of something the model really did compute.
            frame = _value_frame(value)
            frame.update(op="value", present=present)
            rpc.send(out, frame)
        elif op == "submission":
            # The delivery channel, read out of the session's own state rather
            # than out of the namespace. `submit` is bound in the namespace for
            # the model to call, but what it recorded lives on the Session, so
            # rebinding the name cannot forge a delivery — the same doctrine
            # that keeps `llm_query_batched` and the audit out of the model's
            # world after a replacement function earned a "complete"
            # certificate with zero sub-model calls behind it.
            delivered = session._submitted is not _UNSET
            frame = {"op": "submission", "delivered": delivered}
            if delivered:
                frame.update(_value_frame(session._submitted))
                frame["op"] = "submission"
                provided = session._submitted_final_text is not _UNSET
                frame["final_text_provided"] = provided
                frame["final_text"] = (session._submitted_final_text
                                       if provided else None)
            rpc.send(out, frame)
        elif op == "open_presentation":
            try:
                session.open_presentation(
                    str(message.get("initial_text") or ""),
                    message.get("specification") or {},
                    message.get("draft"),
                    commit_required=bool(message.get("commit_required")),
                    draft_ready=bool(message.get("draft_ready")),
                    source_name=message.get("source_name"),
                    inferred_spec=message.get("inferred_spec"),
                )
                rpc.send(out, {"op": "presentation_opened",
                               "windows": session._presentation_windows})
            except SubmitRefused as error:
                rpc.send(out, {"op": "presentation_opened", "error": str(error)})
        elif op == "presentation":
            present = session._presentation_candidate is not _UNSET
            rpc.send(out, {
                "op": "presentation",
                "present": present,
                "text": session._presentation_candidate if present else None,
            })
        elif op == "void_submission":
            session.void_submission()
            rpc.send(out, {"op": "submission_voided"})
        else:
            rpc.send(out, {"op": "error", "message": f"unknown op {op!r}"})


if __name__ == "__main__":
    main()
