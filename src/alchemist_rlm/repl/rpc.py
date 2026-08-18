"""Newline-delimited JSON framing between the harness and the REPL worker.

The channel is bidirectional and reentrant: while the parent waits for the
result of an `exec`, the child may ask the parent to perform an `llm_query`.
That is the whole reason for a message loop rather than a simple request /
response — the sub-LM calls originate *inside* the code the model wrote, and
they have to be metered and traced by the parent, not by the sandbox.
"""

from __future__ import annotations

import json
from typing import Any, IO


class ChannelClosed(RuntimeError):
    """The peer went away mid-conversation. For the worker that means the
    subprocess died, which is an infrastructure failure, not a result."""


def send(stream: IO[str], message: dict[str, Any]) -> None:
    """Write one framed message. Flushed immediately: the peer is blocked reading, so a
    buffered frame is a deadlock.
    """
    stream.write(json.dumps(message, ensure_ascii=False, default=str) + "\n")
    stream.flush()


def recv(stream: IO[str]) -> dict[str, Any]:
    """Read one framed message, or raise when the peer is gone. An empty read means the
    channel closed, which is a different failure from a malformed frame and must not
    be silently read as one.
    """
    line = stream.readline()
    if not line:
        raise ChannelClosed("peer closed the channel")
    try:
        return json.loads(line)
    except json.JSONDecodeError as error:
        raise ChannelClosed(f"malformed frame: {line[:200]!r}") from error
