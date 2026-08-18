"""The one place that talks to `mlx_lm.server`.

Three things here are deliberate.

**Raw JSON, not the OpenAI SDK object.** The plan's snippet reads
`getattr(msg, "reasoning", None)` off an SDK message. That works, but it depends
on the SDK keeping a field it does not model. Since the whole point of this
module is that a field was being silently dropped — `server.py:1345` emits
`choice["message"]["reasoning"]` and `rlm/clients/openai.py:118` reads only
`content` — the fix should not itself rely on a client's tolerance for unknown
keys. The response dict is returned intact and both fields are read from it.

**The served model is recorded on every request, never once per run.** The
server swaps the resident model when a request names a different path, silently,
with a 200. Two models answering the same prompt looked like nondeterminism for
an afternoon. `manifest.note_request` is called per call, not per episode.

**Errors become observations, not exceptions, where the model can act on them.**
A dead server is an infrastructure failure and must raise; a refused request or
a truncated generation is evidence about the run and is returned.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import httpx

DEFAULT_BASE_URL = "http://127.0.0.1:8081/v1"


class ServerUnavailable(RuntimeError):
    """The server is genuinely gone. This is `infrastructure_invalid`."""


class GenerationRejected(RuntimeError):
    """The request died but the server is alive and still serving.

    This is not infrastructure, and calling it that would hide a real result.
    The observed case: the model emits `<tool_call>...</tool_call>` with no
    `<function=...>` inside, `mlx_lm/tool_parsers/qwen3_coder.py:114` raises
    `ValueError("No function provided.")`, the exception escapes `do_POST` into
    `socketserver`, and the connection is closed with no response. The client
    sees a transport error; the server is untouched and answers the next
    request normally. Under greedy decoding the model produces the same
    malformed call again, so it is a reproducible property of this checkpoint
    on this server, and it belongs in the denominator.
    """


@dataclass
class Reply:
    """One assistant turn, with nothing from the wire discarded."""

    content: str
    reasoning: str | None
    tool_calls: list[dict[str, Any]]
    finish_reason: str | None
    served_model: str | None
    usage: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def output_tokens(self) -> int:
        """What this turn cost, or 0 when the server sent no usage block. Read rather
        than estimated: the budget is spent in these units.
        """
        return int(self.usage.get("completion_tokens") or 0)

    @property
    def truncated(self) -> bool:
        """`length` means the answer was cut off, which is not the same as a
        model choosing to stop and must not be scored as one."""
        return self.finish_reason == "length"


class Chat(Protocol):
    """What the harness needs from a backend, and no more.

    The engine, the scheduler and every recursion child take a `Chat`. Keeping
    it this narrow is what lets `ScriptedClient` stand in for a served model in
    the deterministic tests, so the whole loop is testable without inference.
    """

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = ...,
        max_tokens: int = ...,
    ) -> Reply:
        """One assistant turn. `tools` is passed through untouched: whether the
        schema ever reached the server is the founding question of this
        project, and a backend that quietly drops it would hide the answer."""
        ...


@dataclass
class MLXClient:
    """OpenAI-compatible chat against a local `mlx_lm.server`."""

    model: str
    base_url: str = DEFAULT_BASE_URL
    temperature: float = 0.0          # greedy: the suite is one attempt per config
    seed: int | None = None           # formal runners declare explicit int or null
    enable_thinking: bool = False     # controlled variable; see the plan
    timeout: float = 600.0
    manifest: Any = None              # RunManifest, optional
    trace: Any = None
    server_log: str | None = None     # tailed when a generation is rejected

    last_request: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self._client = httpx.Client(timeout=self.timeout)

    def payload(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None,
        max_tokens: int,
    ) -> dict[str, Any]:
        """The request body, built in one place so the manifest can hash exactly what
        was sent. Sampling is not defaulted here: whatever the manifest declares is
        what goes on the wire.
        """
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
        }
        if self.seed is not None:
            body["seed"] = self.seed
        if tools:
            body["tools"] = tools
        return body

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
    ) -> Reply:
        """Ask the server for one turn and return everything it said.

        Nothing from the wire is discarded. `reasoning` in particular arrives in its
        own field on `mlx_lm.server`, and the official client reads only `content` —
        a harness that never asks for it can never see why the model chose what it
        chose.
        """
        body = self.payload(messages, tools=tools, max_tokens=max_tokens)
        # Kept so a rejected generation is reproducible. The server discards the
        # raw text when its parser raises, so the request is the only artifact
        # left on this side, and without it the V1 t08 failure could be shown to
        # repeat but not shown to anyone.
        self.last_request = body
        try:
            response = self._client.post(f"{self.base_url}/chat/completions", json=body)
        except httpx.HTTPError as error:                       # noqa: PERF203
            # Ask the server whether it is still there before deciding what kind
            # of failure this was. Assuming a transport error means the server
            # died gave a failed episode a free pass out of the denominator.
            raise self._classify(error) from error
        if response.status_code >= 500:
            raise ServerUnavailable(f"HTTP {response.status_code}: {response.text[:400]}")
        if response.status_code >= 400:
            # 4xx is our bug — a schema the server rejects, a model path it does
            # not have. Surfacing it as a distinct type keeps it out of the
            # episode denominator: the plan says a preflight failure is not an
            # episode at all.
            raise ServerUnavailable(f"HTTP {response.status_code}: {response.text[:400]}")
        try:
            data = response.json()
        except json.JSONDecodeError as error:
            raise ServerUnavailable(f"non-JSON response: {response.text[:200]}") from error
        return self._reply(data)

    def healthy(self, timeout: float = 5.0) -> bool:
        """Is the server answering at all? This is the question that separates
        infrastructure failure from a rejected generation, and an episode misfiled as
        the wrong one of those wastes a run.
        """
        try:
            probe = httpx.get(f"{self.base_url}/models", timeout=timeout)
            return probe.status_code == 200
        except httpx.HTTPError:
            return False

    def server_traceback(self, lines: int = 60) -> str:
        """The server's own account of why it dropped us.

        The raw generation never reaches the client — `mlx_lm` parses it, raises,
        and `socketserver` closes the socket — so the only record of *what* was
        malformed is in the server's log. Capturing it turns "it disconnects
        again" into an auditable diagnosis.
        """
        if not self.server_log:
            return ""
        try:
            tail = Path(self.server_log).read_text(errors="replace").splitlines()[-lines:]
        except OSError:
            return ""
        return "\n".join(tail)

    def _classify(self, error: Exception) -> RuntimeError:
        detail = f"{type(error).__name__}: {error}"
        if self.healthy():
            rejection = GenerationRejected(
                f"{detail} — the server is still serving, so this is a failed "
                "generation, not an infrastructure failure."
            )
            rejection.request = self.last_request
            rejection.server_traceback = self.server_traceback()
            return rejection
        return ServerUnavailable(detail)

    def _reply(self, data: dict[str, Any]) -> Reply:
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        served = data.get("model")
        if self.manifest is not None:
            self.manifest.note_request(self.model, served)
        return Reply(
            content=message.get("content") or "",
            reasoning=message.get("reasoning"),
            tool_calls=list(message.get("tool_calls") or []),
            finish_reason=choice.get("finish_reason"),
            served_model=served,
            usage=data.get("usage") or {},
            raw=data,
        )

    def close(self) -> None:
        """Release the connection pool. Long suites open one client and keep it."""
        self._client.close()


@dataclass
class ScriptedClient:
    """A backend for the deterministic tests: no server, no model, no episode.

    Every contract in the loop — tool dispatch, duplicate refusal, unknown
    tools, budget stops, termination — is testable against this. That is what
    makes it defensible to build the whole skeleton before spending the two
    smoke episodes.
    """

    replies: list[Reply | dict[str, Any]]
    model: str = "scripted"
    manifest: Any = None
    calls: list[dict[str, Any]] = field(default_factory=list)

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
    ) -> Reply:
        """Return the next scripted reply, or fail loudly.

        Running out of replies raises rather than returning something empty: a test
        whose loop asked for more turns than it scripted has stopped testing what it
        meant to, and silence there produced a passing test of nothing.
        """
        self.calls.append({
            "messages": list(messages), "tools": tools,
            "max_tokens": max_tokens,
        })
        if not self.replies:
            raise AssertionError(
                f"scripted client ran out of replies after {len(self.calls)} calls; "
                "the loop asked for more turns than the test scripted"
            )
        nxt = self.replies.pop(0)
        if isinstance(nxt, Reply):
            reply = nxt
        else:
            reply = tool_reply(**nxt) if "code" in nxt else text_reply(**nxt)
        if self.manifest is not None:
            self.manifest.note_request(self.model, reply.served_model or self.model)
        return reply


def text_reply(content: str = "", *, reasoning: str | None = None,
               finish_reason: str = "stop", model: str = "scripted") -> Reply:
    """A scripted turn with no tool call — the shape that ends an episode."""
    return Reply(
        content=content, reasoning=reasoning, tool_calls=[],
        finish_reason=finish_reason, served_model=model, usage={"completion_tokens": 8},
    )


def tool_reply(code: str, *, name: str = "PythonInterpreter", content: str = "",
               reasoning: str | None = None, model: str = "scripted") -> Reply:
    """Shaped exactly like what `mlx_lm`'s `qwen3_coder` parser hands back:
    `arguments` is a JSON *string*, not a dict."""
    return Reply(
        content=content,
        reasoning=reasoning,
        tool_calls=[{
            "id": "call_0",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps({"code": code})},
        }],
        finish_reason="tool_calls",
        served_model=model,
        usage={"completion_tokens": 32},
    )
