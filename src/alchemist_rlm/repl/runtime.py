"""Parent side of the REPL: spawn, execute, serve callbacks, enforce the clock.

The block timeout is a hard kill of the subprocess, and that is honest about its
cost: the session's variables die with it. The alternative — letting a block run
forever — is what produced a 919.9-second run that returned no answer at all, so
the trade is made deliberately and the model is *told* it happened rather than
being handed a mysteriously empty namespace.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Callable

from alchemist_rlm.repl import rpc

WORKER = Path(__file__).with_name("worker.py")
DEFAULT_BLOCK_TIMEOUT = 300.0
PULL_CHUNK = 16


class _RemoteIterator:
    """An iterator whose items are still in the worker process.

    Single-pass by construction: the underlying generator cannot be rewound, so
    a second iteration would silently yield nothing. It raises instead.
    """

    def __init__(self, runtime: "ReplRuntime", call_id: Any, key: str,
                 chunk: int = PULL_CHUNK):
        self._runtime = runtime
        self._call_id = call_id
        self._key = key
        self._chunk = chunk
        self._buffer: list[Any] = []
        self._exhausted = False
        self._started = False
        self.pulled = 0

    def __iter__(self) -> "_RemoteIterator":
        if self._started:
            raise RuntimeError(
                f"the lazy argument {self._key!r} is a one-pass stream from the "
                "REPL and has already been consumed; build it again in Python "
                "if you need a second pass"
            )
        self._started = True
        return self

    def __next__(self) -> Any:
        while not self._buffer:
            if self._exhausted:
                raise StopIteration
            items, done = self._runtime._pull(self._call_id, self._key, self._chunk)
            self._buffer.extend(items)
            self.pulled += len(items)
            self._exhausted = done
        return self._buffer.pop(0)


class ReplError(RuntimeError):
    """The session itself failed, as opposed to the code inside it."""


class ReplRuntime:
    """A persistent Python session with host callbacks."""

    def __init__(
        self,
        *,
        python: str | None = None,
        block_timeout: float = DEFAULT_BLOCK_TIMEOUT,
        handlers: dict[str, Callable[..., Any]] | None = None,
        trace: Any = None,
    ):
        self.trace = trace
        self.python = python or sys.executable
        self.block_timeout = block_timeout
        self.handlers: dict[str, Callable[..., Any]] = dict(handlers or {})
        self.restarts = 0
        self.host_calls = 0
        self.pulls = 0
        self.pulled_items = 0
        self._process: subprocess.Popen[str] | None = None
        self._start()

    # --- lifecycle ---------------------------------------------------------
    def _start(self) -> None:
        # A working directory of its own, per session. Without it the child
        # inherits the parent's, which is the checkout, and model-written code
        # writes there: `pairs.txt` and `submission.txt` appeared in the repo
        # root during a run and `pairs.txt` was committed into 30fbe42 as if it
        # were source.
        #
        # Three things follow from that, and only the third is obvious. It
        # dirties `git status` mid-run, and the manifest reads the tree to
        # decide provenance — a file the model writes can change what a run
        # records about itself, and a root-level file matches no OUTPUT_PREFIX
        # so it counts as uncommitted *code* and stops the next run. It is
        # memory between episodes: a file that survives can be read by a later
        # one, which would be cross-task contamination invisible in the trace.
        # And the README says the REPL is not a security boundary, which is
        # true and declared — but "not a boundary" was never meant to include
        # "writes into your checkout", and that was written down nowhere.
        #
        # This changes nothing about what model code may do. Only where.
        self._workdir = tempfile.mkdtemp(prefix="rlm-session-")
        worker_env = dict(os.environ)
        worker_env.update({
            "PYTHONHASHSEED": "0",
            "RLM_WORKER_RANDOM_SEED": "0",
            "TZ": "UTC",
            "LC_ALL": "C",
            "LANG": "C",
        })
        self._process = subprocess.Popen(
            [self.python, str(WORKER)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, encoding="utf-8", bufsize=1,
            cwd=self._workdir,
            env=worker_env,
        )
        hello = rpc.recv(self._process.stdout)
        if hello.get("op") != "ready":
            raise ReplError(f"worker did not come up: {hello}")

    def close(self) -> None:
        """Shut the session down, killing it if it will not go quietly."""
        process = self._process
        self._process = None
        if process is None or process.poll() is not None:
            return
        try:
            rpc.send(process.stdin, {"op": "shutdown"})
            process.wait(timeout=5)
        except Exception:                                          # noqa: BLE001
            process.kill()
        finally:
            # Whatever the session wrote goes with it. Keeping it would restore
            # exactly the cross-episode memory the private directory removes.
            workdir, self._workdir = getattr(self, "_workdir", None), None
            if workdir:
                shutil.rmtree(workdir, ignore_errors=True)

    def __enter__(self) -> "ReplRuntime":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    @property
    def alive(self) -> bool:
        """Is the worker process still running?"""
        return self._process is not None and self._process.poll() is None

    # --- messaging ---------------------------------------------------------
    def _send(self, message: dict[str, Any]) -> None:
        if not self.alive:
            raise ReplError("REPL session is not running")
        rpc.send(self._process.stdin, message)                     # type: ignore[union-attr]

    def inject(self, **values: Any) -> None:
        """Bind names in the session. `context` arrives this way and therefore
        never appears in the model's message history."""
        self._send({"op": "inject", "values": values})
        rpc.recv(self._process.stdout)                             # type: ignore[union-attr]

    def bind_context(self, text: str, question: str | None = None,
                     **band: Any) -> dict[str, Any]:
        """Load the context *into the session*. It never enters the message
        history, which is the defining property of a RLM rather than a detail.

        The node's question travels too: `semantic_search()` with no goal uses
        it verbatim, so rephrasing becomes optional instead of mandatory."""
        self._context = (text, band, question)
        self._send({"op": "bind_context", "text": text, "band": band,
                    "question": question})
        frame = rpc.recv(self._process.stdout)                     # type: ignore[union-attr]
        if frame.get("error"):
            raise ReplError(str(frame["error"]))
        return frame.get("manifest") or {}

    _context: tuple[str, dict[str, Any], str | None] | None = None

    def peek(self, name: str) -> dict[str, Any]:
        """Read one name out of the session without executing anything."""
        self._send({"op": "get", "name": name})
        return rpc.recv(self._process.stdout)                      # type: ignore[union-attr]

    def submission(self) -> dict[str, Any]:
        """What the session delivered, if it has delivered anything.

        This is how the answer is collected: as a value out of the REPL, never
        through the model's tokens. It is held on the Session rather than in
        the namespace, so a model that rebinds the name `submit` cannot forge a
        delivery — the same doctrine that keeps the audit and the host
        functions out of the model's world.

        `delivered` is a field of its own rather than a property of the value,
        because `None`, `0`, `False` and `[]` are all answers a model may
        legitimately deliver and none of them can double as "nothing yet".
        """
        self._send({"op": "submission"})
        return rpc.recv(self._process.stdout)                      # type: ignore[union-attr]

    def open_presentation(self, initial_text: str,
                          specification: dict[str, Any],
                          draft: str | None = None, *,
                          commit_required: bool = False,
                          draft_ready: bool = False,
                          source_name: str | None = None,
                          inferred_spec: dict[str, Any] | None = None,
                          ) -> dict[str, Any]:
        """Grant one text-only repair window without clearing the answer value."""
        self._send({"op": "open_presentation", "initial_text": initial_text,
                    "specification": specification, "draft": draft,
                    "commit_required": commit_required,
                    "draft_ready": draft_ready,
                    "source_name": source_name,
                    "inferred_spec": inferred_spec})
        return rpc.recv(self._process.stdout)                      # type: ignore[union-attr]

    def presentation(self) -> dict[str, Any]:
        """The candidate text committed in the presentation window, if any."""
        self._send({"op": "presentation"})
        return rpc.recv(self._process.stdout)                      # type: ignore[union-attr]

    def void_submission(self) -> dict[str, Any]:
        """Discard a committed frame the parent proved was not a real value."""
        self._send({"op": "void_submission"})
        return rpc.recv(self._process.stdout)                      # type: ignore[union-attr]

    def peek_audit(self) -> dict[str, Any]:
        """The session's own record of the operations it performed — held by
        the harness, outside the namespace, so an ordinary assignment does not
        rewrite it. `peek` reads the model's world; this reads the session's.

        Separated, not sealed: the REPL is not a sandbox and a bound method's
        `__self__` reaches the session. See the note on `Session.audit`.
        """
        self._send({"op": "audit"})
        return rpc.recv(self._process.stdout)                      # type: ignore[union-attr]

    def execute(self, code: str, *, timeout: float | None = None) -> dict[str, Any]:
        """Run a block and return a structured observation. Never raises for
        errors *in* the code — those are the model's to see and recover from.

        The clock measures the model's code, not the harness's work. A host
        call — `rlm_map` spawning children, a batch of fifty subcalls — can
        legitimately take many minutes, and that time is spent *here*, in the
        parent, while the worker sits idle waiting for the reply. The first
        recursion episode died to exactly this: seven children ran for over
        five minutes inside one `rlm_map` call, the block timer treated the
        whole stretch as runaway code, killed the worker mid-conversation, and
        the reply had no one to go to. So the timer is disarmed while a host
        call is being served and rearmed fresh when execution resumes: the
        limit applies to each stretch of pure Python between host calls.
        """
        limit = self.block_timeout if timeout is None else timeout
        if not self.alive:
            self._restart()
        self._send({"op": "exec", "code": code})

        expired = threading.Event()

        def kill_it() -> None:
            expired.set()
            process = self._process
            if process is not None and process.poll() is None:
                process.kill()

        timer: threading.Timer | None = None

        def arm() -> None:
            nonlocal timer
            timer = threading.Timer(limit, kill_it)
            timer.daemon = True
            timer.start()

        def disarm() -> None:
            if timer is not None:
                timer.cancel()

        arm()
        try:
            return self._pump(pause=disarm, resume=arm)
        except rpc.ChannelClosed:
            if expired.is_set():
                self._restart()
                return {
                    "ok": False,
                    "stdout": "", "stderr": "", "value": None,
                    "defined": [], "changed": {}, "truncated": False,
                    "error": {
                        "type": "BlockTimeout",
                        "message": (
                            f"this block ran longer than {limit:.0f}s and was stopped. "
                            "The Python session was restarted, so variables from "
                            "earlier calls are gone and `context` has been reloaded. "
                            "Work on a smaller slice."
                        ),
                        "traceback": "",
                    },
                }
            self._restart()
            raise ReplError("REPL session died mid-execution") from None
        finally:
            disarm()

    def _pump(self, pause=None, resume=None) -> dict[str, Any]:
        """Read until the result, answering host callbacks along the way.

        `pause`/`resume` bracket the host-call service so the block timer never
        counts time the parent spends working on the model's behalf."""
        while True:
            message = rpc.recv(self._process.stdout)               # type: ignore[union-attr]
            op = message.get("op")
            if op == "result":
                return {key: value for key, value in message.items() if key != "op"}
            if op == "host":
                if pause is not None:
                    pause()
                try:
                    self._serve(message)
                finally:
                    if resume is not None:
                        resume()
                continue
            raise ReplError(f"unexpected frame from worker: {op!r}")

    def _rehydrate(self, value: Any, call_id: Any) -> Any:
        """Turn a `{"__lazy__": key}` placeholder back into a real iterator.

        The items still live in the worker; iterating this pulls them a chunk at
        a time. Nothing here ever holds the whole sequence, which is the
        difference between a batch of 3,182 jobs that runs and one that swaps."""
        if isinstance(value, dict) and set(value) == {"__lazy__"}:
            return _RemoteIterator(self, call_id, value["__lazy__"])
        return value

    def _serve(self, message: dict[str, Any]) -> None:
        name = message.get("fn") or ""
        handler = self.handlers.get(name)
        self.host_calls += 1
        call_id = message.get("id")
        if handler is None:
            self._send({"op": "reply", "id": call_id, "ok": False,
                        "error": f"{name} is not available in this session"})
            return
        args = [self._rehydrate(v, call_id) for v in (message.get("args") or [])]
        kwargs = {k: self._rehydrate(v, call_id)
                  for k, v in (message.get("kwargs") or {}).items()}
        try:
            value = handler(*args, **kwargs)
        except Exception as error:                                 # noqa: BLE001
            # The type travels as its own field, not folded into the message.
            # Folded in, the worker could only re-raise everything as
            # RuntimeError, so `except ValueError:` in the model's own code
            # caught nothing and the observation named the wrong exception —
            # `RuntimeError: ValueError: parts must be a collection of texts`,
            # with the real type demoted to the first word of a string.
            # `args[0]` rather than `str(error)`, because `str(KeyError("no
            # artifact named 'x'"))` is already quoted, and rebuilding a
            # KeyError from that quotes it again: `'"no artifact named \'x\'"'`.
            # The single argument is the message the raiser wrote.
            args = getattr(error, "args", ())
            message = (args[0] if len(args) == 1 and isinstance(args[0], str)
                       else str(error))
            self._send({"op": "reply", "id": call_id, "ok": False,
                        "error": message,
                        "error_type": type(error).__name__})
            return
        self._send({"op": "reply", "id": call_id, "ok": True, "value": value})

    def _pull(self, call_id: Any, key: str, count: int) -> tuple[list[Any], bool]:
        self.pulls += 1
        self._send({"op": "pull", "id": call_id, "key": key, "count": count})
        frame = rpc.recv(self._process.stdout)                     # type: ignore[union-attr]
        if frame.get("op") != "items":
            raise ReplError(f"expected items for {key!r}, got {frame.get('op')!r}")
        if frame.get("error"):
            raise ReplError(str(frame["error"]))
        items = list(frame.get("items") or [])
        self.pulled_items += len(items)
        if self.trace is not None:
            # Evidence that the generator was consumed in chunks rather than
            # materialised: a lazy argument that was never pulled leaves no
            # event, and one that was pulled once for everything leaves one.
            self.trace.event("lazy_pull", key=key, asked=count, got=len(items),
                             exhausted=bool(frame.get("exhausted")),
                             pulls_so_far=self.pulls)
        return items, bool(frame.get("exhausted"))

    # --- recovery ----------------------------------------------------------
    def _restart(self) -> None:
        """After a kill the namespace is gone. The context is restored because a
        session that comes back without it is unusable for the rest of the run;
        the model's own variables are not, and it is told so."""
        self.restarts += 1
        process = self._process
        if process is not None and process.poll() is None:
            process.kill()
        self._start()
        if self._context is not None:
            text, band, question = self._context
            self._send({"op": "bind_context", "text": text, "band": band,
                        "question": question})
            rpc.recv(self._process.stdout)                          # type: ignore[union-attr]
