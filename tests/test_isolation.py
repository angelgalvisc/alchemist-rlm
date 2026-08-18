"""Deterministic runtime inputs and explicit episode isolation."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from alchemist_rlm.isolation import MLXPromptCacheIsolation
from alchemist_rlm.mlx_client import MLXClient, ServerUnavailable
from alchemist_rlm.repl.runtime import ReplRuntime


@dataclass
class _Response:
    body: dict
    status_code: int = 200

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError("bad status")

    def json(self):
        return self.body


def test_cache_reset_requires_and_records_the_empty_postcondition(monkeypatch):
    seen = {}

    def post(url, **kwargs):
        seen.update(url=url, kwargs=kwargs)
        return _Response({
            "ok": True, "epoch": 7,
            "before": {"sequences": 4, "bytes": 99},
            "after": {"sequences": 0, "bytes": 0},
        })

    monkeypatch.setattr("alchemist_rlm.isolation.httpx.post", post)
    policy = MLXPromptCacheIsolation("http://127.0.0.1:8081/v1")
    attestation = policy.before_episode(run_id="t07")

    assert seen["url"] == "http://127.0.0.1:8081/admin/prompt-cache/reset"
    assert attestation["verified"] is True
    assert attestation["epoch"] == 7
    assert attestation["before"]["sequences"] == 4


def test_cache_reset_fails_closed_when_server_keeps_state(monkeypatch):
    monkeypatch.setattr(
        "alchemist_rlm.isolation.httpx.post",
        lambda *args, **kwargs: _Response({
            "ok": True, "after": {"sequences": 1, "bytes": 10},
        }),
    )

    with pytest.raises(ServerUnavailable, match="did not establish"):
        MLXPromptCacheIsolation("http://localhost:8081/v1").before_episode(run_id="x")


def test_worker_process_has_canonical_environment_and_repr():
    code = (
        "import os, random\n"
        "class A: pass\n"
        "a = A()\n"
        "print(os.environ['PYTHONHASHSEED'], os.environ['TZ'], "
        "os.environ['LC_ALL'], random.random())\n"
        "a"
    )
    with ReplRuntime() as first, ReplRuntime() as second:
        one = first.execute(code)
        two = second.execute(code)

    assert one["stdout"] == two["stdout"] == "0 UTC C 0.8444218515250481\n"
    assert one["value"] == two["value"]
    assert "0xADDR" in one["value"]


def test_mlx_requests_declare_the_seed():
    client = MLXClient("model", seed=17)
    body = client.payload([{"role": "user", "content": "q"}], tools=None,
                          max_tokens=20)

    assert body["seed"] == 17


def test_mlx_greedy_baseline_omits_a_null_seed_from_the_wire():
    client = MLXClient("model", seed=None)
    body = client.payload([{"role": "user", "content": "q"}], tools=None,
                          max_tokens=20)

    assert "seed" not in body
