"""Limits that are shared down the recursion, not re-granted at each level.

The failure this prevents was measured, not imagined. `probe_11` ran 25 turns of
which 23 were regex, ten of them re-runs of searches that had already returned
nothing; the grouped OOLONG run spent 919.9 seconds and produced no answer at
all. Neither was stopped by anything except a wall-clock kill from outside.

The design point is that a child RLM must not get a *fresh* budget. If it did,
depth 2 with a fan-out of eight would licence nine times the spend of the root,
and the "limit" would be a limit on one node rather than on the run. So the
allowances live per node and the *spending* lives in one `Ledger` shared by the
whole tree.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field, replace
from typing import Any


@dataclass
class Ledger:
    """What the whole tree has spent so far. One instance per run."""

    started: float = field(default_factory=time.monotonic)
    turns: int = 0
    output_tokens: int = 0
    subcalls: int = 0
    nodes: int = 1                      # root, plus every child RLM created
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def elapsed(self) -> float:
        """Wall-clock seconds since this ledger started, on the monotonic clock so a
        system time change cannot move a limit.
        """
        return time.monotonic() - self.started

    def add(self, **amounts: int) -> None:
        """Add to any counter by name, under the lock. Batched subcalls settle from
        several threads at once, so this is the only way the totals are touched.
        """
        with self._lock:
            for name, amount in amounts.items():
                setattr(self, name, getattr(self, name) + amount)

    def reserve_subcall(self, limit: int) -> bool:
        """Claim one subcall against `limit`, or refuse. Atomic.

        Checking a limit and then spending against it are two operations, and
        between them any number of concurrent workers pass the same check.
        Measured against a backend with latency, a limit of 10 produced 12 calls
        at four in flight and 17 at eight — the overshoot grows with the
        concurrency, so the guarantee was not one. The claim is made here,
        before the request goes out. Tokens are still added afterwards, because
        nobody can know what a turn costs until it comes back.
        """
        with self._lock:
            if self.subcalls >= limit:
                return False
            self.subcalls += 1
            return True

    def release_subcall(self) -> None:
        """Give back a claim whose request never reached the wire.

        The batched path reserves in submission order, before a job goes to the
        pool, so the budget is spent deterministically rather than by whichever
        worker thread wakes first. A job that then fails its own validation —
        an empty source, one past the size limit — was never issued, and
        charging the run for work it refused to do would make the counter wrong
        in the other direction.
        """
        with self._lock:
            self.subcalls = max(0, self.subcalls - 1)

    def snapshot(self) -> dict[str, Any]:
        """The totals as plain data, for the episode record and the trace. A snapshot,
        not a view: it is written into results that must stay true after the run
        ends.
        """
        return {
            "turns": self.turns,
            "output_tokens": self.output_tokens,
            "subcalls": self.subcalls,
            "nodes": self.nodes,
            "elapsed_s": round(self.elapsed(), 2),
        }


@dataclass
class Budget:
    """Allowances for one node in the tree, spent against a shared `Ledger`."""

    max_turns: int = 12
    max_seconds: float = 900.0
    max_output_tokens: int = 120_000
    max_subcalls: int = 600
    max_nodes: int = 8
    max_depth: int = 2
    # Concurrency is a limit on *materialisation*, not only on execution. The
    # memory wall in this repository came from `_handle_batched` creating one
    # asyncio task per prompt up front while a semaphore bounded only how many
    # ran; 3,182 tiny prompts became 3,182 live objects and the machine swapped.
    max_in_flight: int = 4
    # Disabled by default. Different, progressively corrective errors are not
    # proof of a stall; max_turns remains the hard bound. Callers that need the
    # legacy policy may opt into a positive integer explicitly.
    max_consecutive_errors: int | None = None
    # Truncated generations are counted apart from execution errors: one is a
    # wrong action, the other an incomplete one, and a model that alternates
    # between truncating and succeeding is making progress rather than stalling.
    # They are not free — a model that only ever runs past `max_tokens` is stuck
    # — so they carry their own limit, and any block that runs clears both.
    max_consecutive_truncations: int = 3
    # A refused duplicate is a third failure kind, and it was spending the
    # first one's budget. The note above already recorded the symptom — "one
    # episode died having made no execution error at all: truncation, a refused
    # duplicate, truncation" — but the split it prompted covered truncation
    # only. The v2 sweep of twenty queries showed what was left: the three runs
    # killed by `consecutive_errors` were exactly the three with two or more
    # duplicate refusals, each with zero sub-calls and half its turns unspent,
    # and one of them exhausted the error budget on duplicates without a single
    # code error. Repeating a call is a real stall and keeps a limit of its
    # own; it no longer borrows the one meant for broken code.
    max_consecutive_duplicates: int = 3
    # Node-local turn allowance. A child gets fewer turns than the root even
    # though both draw on the same ledger.
    turns_here: int = 0
    depth: int = 0
    ledger: Ledger = field(default_factory=Ledger)

    def fresh(self) -> "Budget":
        """The same allowances for a new root episode, with no spent state.

        ``RLMEngine`` is a reusable agent tool.  Keeping the template budget's
        ledger across calls made the second call inherit the first call's
        turns, clock and subcalls, even though the engine contract says an
        episode owns its budget.  Children still use :meth:`child` and share
        one ledger inside an episode; only a new public ``complete`` call gets
        a new one.
        """
        return replace(self, turns_here=0, depth=0, ledger=Ledger())

    def child(self) -> "Budget":
        """A child RLM: one level deeper, half the turns, the *same* ledger."""
        self.ledger.add(nodes=1)
        return replace(
            self,
            depth=self.depth + 1,
            max_turns=max(2, self.max_turns // 2),
            turns_here=0,
            ledger=self.ledger,
        )

    def spend_turn(self) -> None:
        """Charge one turn to this node and to the whole tree. Both are needed:
        `turns_here` bounds the conversation at this node, the ledger bounds the run.
        """
        self.turns_here += 1
        self.ledger.add(turns=1)

    def exhausted(self) -> str | None:
        """The reason to stop, or None. Checked before every model call."""
        if self.turns_here >= self.max_turns:
            return "max_turns"
        if self.ledger.elapsed() >= self.max_seconds:
            return "max_seconds"
        if self.ledger.output_tokens >= self.max_output_tokens:
            return "max_output_tokens"
        if self.ledger.subcalls >= self.max_subcalls:
            return "max_subcalls"
        if self.ledger.nodes > self.max_nodes:
            return "max_nodes"
        return None

    def subcall_exhausted(self) -> str | None:
        """The reason no more sub-model work may start inside this turn.

        ``max_turns`` limits root/child conversation turns. Once one of those
        turns has been admitted, Python executed in it may still call a bounded
        operation: refusing its first subcall because ``spend_turn`` reached
        the root ceiling makes the final permitted tool call a false offer.
        Global wall-clock, token, subcall and node bounds still apply.
        """
        if self.ledger.elapsed() >= self.max_seconds:
            return "max_seconds"
        if self.ledger.output_tokens >= self.max_output_tokens:
            return "max_output_tokens"
        if self.ledger.subcalls >= self.max_subcalls:
            return "max_subcalls"
        if self.ledger.nodes > self.max_nodes:
            return "max_nodes"
        return None

    def may_recurse(self) -> str | None:
        """Why recursion is refused here, or None if it is allowed. Returning the reason
        rather than a bool is what lets the refusal message name it, and a refusal
        that does not say why changes nothing about what the model does next.
        """
        if self.depth >= self.max_depth:
            return "max_depth"
        if self.ledger.nodes >= self.max_nodes:
            return "max_nodes"
        return None

    def to_dict(self) -> dict[str, Any]:
        """The limits as declared, for the run manifest. What a run was allowed to spend
        belongs in its record beside what it did spend.
        """
        return {
            "max_turns": self.max_turns,
            "max_seconds": self.max_seconds,
            "max_output_tokens": self.max_output_tokens,
            "max_subcalls": self.max_subcalls,
            "max_nodes": self.max_nodes,
            "max_depth": self.max_depth,
            "max_in_flight": self.max_in_flight,
            "max_consecutive_errors": self.max_consecutive_errors,
            "max_consecutive_truncations": self.max_consecutive_truncations,
            # Added late and left out of here, which made this method's own
            # sentence untrue for the limit that mattered most: four of twenty
            # episodes ended on `consecutive_duplicates` and no result file said
            # what the limit had been. A stop reason naming a bound the record
            # does not declare cannot be argued with.
            "max_consecutive_duplicates": self.max_consecutive_duplicates,
            "depth": self.depth,
        }


class BudgetExceeded(RuntimeError):
    """Raised inside the REPL so the model sees the limit as an observation."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason
