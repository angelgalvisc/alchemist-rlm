"""What every run must record about itself before it is allowed to count.

A result is only evidence if you can say what produced it. This project has
already lost time twice to the answer being "we don't know":

  - `mlx_lm.server` swaps the resident model when a request names a different
    path, silently and with a 200. Two models answered the same prompt and the
    difference looked like nondeterminism. Hence `requested_model` is recorded
    per request, not per run: the run-level intent is not what the server acted
    on.
  - A benchmark artifact was overwritten by a later debug run because the output
    path was a fixed constant. Hence `git_commit` and the task hash: a result
    file that cannot be tied to the code and the questions that made it is not
    resumable, only suggestive.

Nothing here loads a model or costs an episode.
"""

from __future__ import annotations

import inspect
import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent.parent


# Paths a run writes into. A trace or a result file appearing after an episode
# is the run doing its job, not the code changing underneath it.
OUTPUT_PREFIXES = ("runs/", "configs/suite_", "configs/smoke_record",
                   "configs/oolong_", "logs/")


def runtime_determinism_record(*, request_seed: int | None = None) -> dict[str, Any]:
    """Canonical declaration of runtime inputs shared by formal runners."""
    return {
        "request_seed": request_seed,
        "temperature": 0.0,
        "worker": {
            "PYTHONHASHSEED": "0", "random_seed": 0, "TZ": "UTC",
            "locale": "C", "address_repr": "normalized",
        },
        "tool_call_ids": "canonical_turn_index",
        "visible_requests": "strict_json_sha256",
    }


def interaction_contract_sha256() -> str:
    """One hash over every static controller text the root model can read.

    "Every" has been wrong once and the correction is worth keeping. The notes
    an operation appends to its own result — what to do next, whether coverage
    was complete — are controller text the model reads on the turn that matters
    most, and they were outside this hash. A commit rewrote the complete note
    and the fingerprint did not move, so a run before and after it looked
    identical on the one field whose job is to say they are not.

    They are conditioned on the result, so all three statuses are rendered from
    a fixed result and hashed together: changing any branch changes the hash.
    """
    from alchemist_rlm.adapters.agents import STRATEGY_DIRECTIVES
    from alchemist_rlm.engine import CHILD_MAP_CONTRACT, _context_line
    from alchemist_rlm.native_loop import (
        COMMIT_FIRST, COMMIT_SECOND, MALFORMED_TOOL_RECOVERY, OUTPUT_REPAIR,
        PRESENTATION_RETRY, TRUNCATION_RECOVERY,
    )
    from alchemist_rlm.repl.worker import _map_note, _search_note

    canonical_manifest = {
        "chars": 100_000, "lines": 2_000, "structure": "rows",
        "segments": 25, "segment_chars": {"mean": 4_000},
        "first_refs": ["s0000", "s0001", "s0002"],
    }
    statuses = ("complete", "partial", "failed")
    payload = {
        "context_line": _context_line(canonical_manifest),
        "child_map_contract": CHILD_MAP_CONTRACT,
        "commit_first": COMMIT_FIRST,
        "commit_second": COMMIT_SECOND,
        "output_repair": OUTPUT_REPAIR,
        "presentation_retry": PRESENTATION_RETRY,
        "truncation_recovery": TRUNCATION_RECOVERY,
        "malformed_tool_recovery": MALFORMED_TOOL_RECOVERY,
        "strategy_directives": STRATEGY_DIRECTIVES,
        "map_notes": {status: _map_note(
            {"status": status, "valid_items": 90, "total_items": 100})
            for status in statuses},
        "search_notes": {status: _search_note(
            {"status": status, "valid_items": 90, "total_items": 100})
            for status in statuses},
    }
    return sha256_text(json.dumps(payload, sort_keys=True, ensure_ascii=False))


def observation_contract_sha256() -> str:
    """Fingerprint host code that turns runtime state into model-visible text.

    Static prompts are only half of the interaction. Tracebacks, value
    descriptions, counteroffers and observation rendering are generated at
    runtime, and changing one of them has already redirected a greedy run while
    the old interaction hash stayed fixed. Source is intentionally over-broad:
    a false-positive provenance change is safe; a silent visible change is not.
    """
    from alchemist_rlm import protocol
    from alchemist_rlm.context import search
    from alchemist_rlm import native_loop
    from alchemist_rlm.repl import worker

    surfaces = {
        "native_loop.render": native_loop.render,
        "native_loop._without_the_host": native_loop._without_the_host,
        "protocol.unknown_tool_observation": protocol.unknown_tool_observation,
        "search.literal_search": search.literal_search,
        "worker._describe": worker._describe,
        "worker._model_traceback": worker._model_traceback,
        "worker.Session.execute": worker.Session.execute,
    }
    modules = (native_loop, protocol, search, worker)
    payload = {
        "live_surfaces": {
            name: inspect.getsource(value) for name, value in surfaces.items()
        },
        # Helper functions called by a renderer can change its output without
        # changing the renderer's own source. Whole-module text makes that a
        # conservative provenance change instead of another silent blind spot.
        "module_sources": {
            module.__name__: Path(module.__file__).read_text(encoding="utf-8")
            for module in modules
        },
    }
    return sha256_text(json.dumps(payload, sort_keys=True, ensure_ascii=False))


_VISIBLE_MESSAGE_FIELDS = (
    "role", "content", "name", "tool_calls", "tool_call_id",
)


def visible_transcript_sha256(messages: list[dict[str, Any]]) -> str:
    """Hash exactly the ordered chat payload visible to the model.

    Run ids, timestamps and usage never enter the chat and are ignored. Tool
    call ids are included because they are sent back in tool observations and
    therefore really are part of the next model input.
    """
    visible = [
        {key: message[key] for key in _VISIBLE_MESSAGE_FIELDS if key in message}
        for message in messages
    ]
    payload = json.dumps(
        visible, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
    )
    return sha256_text(payload)


def visible_requests_sha256(requests: list[dict[str, Any]]) -> str:
    """Hash the ordered model-visible requests across every conversation channel.

    A presentation repair deliberately starts from a fresh compact transcript.
    Hashing only the root ``messages`` would therefore omit exactly the request
    that can promote the final text.  Each request record contains its channel
    and strict payload hash; this aggregate preserves their order without
    duplicating large prompts in the episode file.
    """
    compact = [
        {
            "turn": int(request["turn"]),
            "channel": str(request["channel"]),
            "sha256": str(request["sha256"]),
        }
        for request in requests
    ]
    payload = json.dumps(
        compact, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    return sha256_text(payload)


def git_state() -> dict[str, Any]:
    """Commit plus two dirty flags.

    `dirty` is the whole tree. `code_dirty` ignores the paths a run writes into,
    and it is the one the runners gate on: what has to be committed is the code
    that produced the result. Gating on `dirty` made the second episode of any
    suite impossible, because the first episode had just written its own trace —
    the guard refused the repeat that the plan's own infrastructure rule
    requires.
    """
    def run(*args: str, keep_leading_space: bool = False) -> str | None:
        try:
            done = subprocess.run(
                ["git", *args], cwd=REPO, capture_output=True, text=True, timeout=10
            )
            if done.returncode != 0:
                return None
            return (done.stdout.rstrip("\n") if keep_leading_space
                    else done.stdout.strip())
        except Exception:                                          # noqa: BLE001
            return None

    # NOT `run(...)`: that helper strips the whole output, and a porcelain
    # line for an unstaged modification begins with a space — ` M path`. The
    # global strip ate that space from the FIRST line only, leaving `M path`,
    # so `line[3:]` cut one character too many and `runs/x` became `uns/x`.
    # It no longer matched an output prefix, so a run's own trace counted as
    # uncommitted code and the next run refused to start. Measured: it cost a
    # 200K episode that never launched.
    status = run("status", "--porcelain", keep_leading_space=True)
    changed = [line[3:].strip().strip('"')
               for line in (status or "").splitlines() if line.strip()]
    code_changes = [
        path for path in changed
        if not any(path.startswith(prefix) for prefix in OUTPUT_PREFIXES)
    ]
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status) if status is not None else None,
        "code_dirty": bool(code_changes) if status is not None else None,
        "uncommitted_code": sorted(code_changes)[:20],
    }


def mlx_server_argv() -> list[str] | None:
    """The server's *effective* arguments, read from the running process.

    Not what `serve.sh` says it passes — what the process actually has. Those
    diverged once already: the script was edited to `--prompt-cache-size 10`
    while a server started before the edit kept running with the old value.
    """
    try:
        pids = subprocess.run(
            ["pgrep", "-f", "mlx_lm.server --model|serve_patched.py --model"],
            capture_output=True, text=True, timeout=10,
        ).stdout.split()
        if not pids:
            return None
        command = subprocess.run(
            ["ps", "-ww", "-o", "command=", "-p", pids[0]],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        # ``ps`` returns display text, not a recoverable argv. Splitting it
        # fabricated arguments at spaces inside local model paths.
        return [command] if command else None
    except Exception:                                              # noqa: BLE001
        return None


@dataclass
class RunManifest:
    """Everything needed to decide whether two results may be compared."""

    run_id: str
    arm: str                                  # alchemist | agents-bf16 | qwen4b-base
    suite: str                                # e.g. "gate_a"
    fingerprint_sha256: str                   # configs/fingerprint.json
    tasks_sha256: str
    system_prompt_sha256: str
    tool_schema_sha256: str
    tool_name: str
    sampling: dict[str, Any]                  # temperature, max_tokens, enable_thinking...
    budget: dict[str, Any] = field(default_factory=dict)
    # Output assistance is part of the experimental arm. These fields make it
    # impossible to compare a raw generation to one validated, repaired or
    # backend-constrained without the manifest exposing the difference.
    output_mode: str = "raw"
    output_contract: dict[str, Any] | None = None
    output_contract_sha256: str = ""
    terminal_policy: dict[str, Any] = field(default_factory=dict)
    output_backend_constraint: str = "none"
    presentation_spec_source: str = "adapter"
    presentation_linter: str = "none"
    runtime_determinism: dict[str, Any] = field(default_factory=dict)
    isolation_policy: dict[str, Any] = field(default_factory=dict)
    # Two prompts govern a run and the manifest froze only one. Every subcall
    # — every label a sweep produces — reads `SUB_SYSTEM`, and the leaf's own
    # token limit decides whether long replies survive or truncate into
    # "missing items". Changing either would alter every classification in a
    # run without moving `system_prompt_sha256`, which is exactly the silent
    # drift a manifest exists to refuse.
    leaf_prompt_sha256: str = ""
    leaf_max_tokens: int = 0
    # The signature list the context line puts in front of the model every
    # episode. It is a separate message, so `system_prompt_sha256` does not
    # move when it changes — and it changed under exactly that blind spot:
    # registering `restore_rows` altered what every episode reads while the
    # recorded prompt hash stayed 8a059c7b. Two runs whose context lines
    # differ are not comparable, however identical the prompt hash looks.
    bound_names_sha256: str = ""
    # The per-fragment contract, which is the larger part of what every
    # sub-model reads and was frozen nowhere. Changing the line that makes its
    # format authoritative moved every leaf's input while `leaf_prompt_sha256`
    # — which covers SUB_SYSTEM only — stayed put.
    leaf_contract_sha256: str = ""
    # The root context line, forced/recovery turns, strategy directives and
    # child-map contract are model-visible too.  Hashing them together avoids
    # adding one manifest field after every newly discovered blind spot.
    interaction_contract_sha256: str = ""
    # Runtime-generated observations are a second model-visible contract. They
    # are separated from static interaction text so a diff says which surface
    # moved.
    observation_contract_sha256: str = ""
    git: dict[str, Any] = field(default_factory=git_state)
    server_argv: list[str] | None = field(default_factory=mlx_server_argv)
    # Filled in per request by the client. The run says which model it *asked*
    # for; this says which it got, and where that changed.
    requests: int = 0
    model_segments: list[dict[str, Any]] = field(default_factory=list)

    def note_request(self, requested: str, served: str | None) -> None:
        """Record which model was asked for and which one answered.

        Watched per request, not once per run: the server swaps the resident
        model when a request names a different path, silently, with a 200, and
        every result after such a swap would be attributed to the wrong weights.

        Stored run-length encoded. Two parallel lists of one string per request
        cost 94,932 bytes on a three-query run — 610 identical paths — to answer
        one question: did the pair ever change, and where. A segment answers it
        without losing the order or the position of a swap, and a run that never
        swaps is a single entry rather than six hundred.
        """
        self.requests += 1
        got = served or ""
        last = self.model_segments[-1] if self.model_segments else None
        if last and last["requested"] == requested and last["served"] == got:
            last["requests"] += 1
            return
        self.model_segments.append({
            "from_request": self.requests, "requested": requested,
            "served": got, "requests": 1,
        })

    @property
    def model_stayed_put(self) -> bool:
        """False the moment the server answered as a model we did not ask for."""
        return all(segment["requested"] == segment["served"]
                   for segment in self.model_segments if segment["served"])

    def to_dict(self) -> dict[str, Any]:
        """The manifest as plain data, with the checks it exists to support already
        computed — whether the model stayed put, and every distinct model that
        actually served a turn.
        """
        record = asdict(self)
        record["model_stayed_put"] = self.model_stayed_put
        record["distinct_served_models"] = sorted(
            {segment["served"] for segment in self.model_segments if segment["served"]})
        return record

    def write(self, path: Path) -> None:
        """Write the manifest beside its results. Sorted keys so two runs of the same
        configuration produce byte-identical files and a diff means something.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=1, sort_keys=True))


def sha256_text(text: str) -> str:
    """Full hash of a string, for pinning prompts and schemas in a manifest."""
    import hashlib

    return hashlib.sha256(text.encode()).hexdigest()
