"""Repeated stdout is compact in prompts and complete in evidence."""

from __future__ import annotations

import json

from alchemist_rlm.budgets import Budget
from alchemist_rlm.engine import RLMEngine
from alchemist_rlm.mlx_client import ScriptedClient, tool_reply


def _tool_messages(client: ScriptedClient) -> list[str]:
    return [
        message["content"]
        for message in client.calls[-1]["messages"]
        if message.get("role") == "tool"
    ]


def test_identical_stdout_is_shown_once_but_both_blocks_execute(tmp_path):
    client = ScriptedClient([
        tool_reply("counter = globals().get('counter', 0) + 1\nprint('same')"),
        tool_reply("counter = counter + 1\nprint('same')"),
        tool_reply("submit(counter)"),
    ])
    episode = RLMEngine(
        client=client, budget=Budget(max_turns=4), runs_dir=tmp_path,
    ).complete("context", "q", run_id="stdout_repeat")

    first, second = _tool_messages(client)[:2]
    assert "stdout (5 chars):\nsame" in first
    # Both calls mutate ``counter``, so the safety rule keeps both full.
    assert "stdout (5 chars):\nsame" in second
    assert episode.answer_value == 2


def test_quiet_identical_stdout_is_compacted_only_in_the_next_prompt(tmp_path):
    client = ScriptedClient([
        tool_reply("print('same')"),
        tool_reply("print('same')"),
        tool_reply("submit('done')"),
    ])
    episode = RLMEngine(
        client=client, budget=Budget(max_turns=4), runs_dir=tmp_path,
    ).complete("context", "q", run_id="stdout_quiet_repeat")

    first, second = _tool_messages(client)[:2]
    assert "stdout (5 chars):\nsame" in first
    assert "repeated byte-for-byte" in second
    assert "sha256" in second and "stdout (5 chars):\nsame" not in second

    observations = [
        event
        for event in map(json.loads, episode.trace_path.read_text().splitlines())
        if event.get("kind") == "observation"
    ]
    assert [event["observation"]["stdout"] for event in observations[:2]] == [
        "same\n", "same\n",
    ]


def test_different_stdout_is_never_compacted(tmp_path):
    client = ScriptedClient([
        tool_reply("print('first')"),
        tool_reply("print('second')"),
        tool_reply("submit('done')"),
    ])
    RLMEngine(
        client=client, budget=Budget(max_turns=4), runs_dir=tmp_path,
    ).complete("context", "q", run_id="stdout_different")

    first, second = _tool_messages(client)[:2]
    assert "first" in first and "repeated byte-for-byte" not in first
    assert "second" in second and "repeated byte-for-byte" not in second
