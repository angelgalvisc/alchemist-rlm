"""One executed tool call, recorded by the loop.

Its own module because it is the one thing the runtime needs from what used to
be the tasks module, and `tasks` is evaluation: suites, scoring predicates,
what counts as a task passing. The loop imported `Step` from there, so the
runtime could not be read, tested or shipped without the benchmark coming with
it — a dependency that ran the wrong way and existed only because a dataclass
had been put in the nearest file.

Nothing else in `native_loop`, `engine`, `worker`, `calls` or `protocol`
reaches into the evaluation side. Moving this one name is the whole of that
separation, which is smaller than it looked.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Step:
    """One executed tool call, as the runner records it.

    `defined` is the set of names bound in the session *after* this call, which
    is what makes cross-call variable use checkable rather than assumed.

    Frozen because this is a historical record: it asserts what the model ran.
    A mutable one could be edited after the fact, and a trace or a scorer would
    then be reading an action that never happened. It *was* frozen in `tasks`,
    and the move to this module dropped that in a commit whose message called
    the move neutral. Nothing mutates a `Step` today, so nothing broke — but a
    guarantee the type already had went out silently, which is why the test
    below it exists rather than only the keyword.
    """

    code: str
    defined: frozenset[str] = frozenset()
    refused: bool = False
