"""Backend request IDs cannot make otherwise identical prompts diverge."""

from alchemist_rlm.budgets import Budget
from alchemist_rlm.engine import RLMEngine
from alchemist_rlm.mlx_client import ScriptedClient, tool_reply


def _run(tmp_path, run_id: str, upstream_id: str):
    first = tool_reply("x = 1")
    first.tool_calls[0]["id"] = upstream_id
    client = ScriptedClient([first, tool_reply("submit(x)")])
    episode = RLMEngine(
        client=client, budget=Budget(max_turns=3), runs_dir=tmp_path,
    ).complete("context", "q", run_id=run_id)
    return episode, client


def test_tool_call_ids_are_stable_and_shared_by_call_and_observation(tmp_path):
    first, client_a = _run(tmp_path, "ids_a", "backend-random-a")
    second, client_b = _run(tmp_path, "ids_b", "backend-random-b")

    assert first.visible_transcript_sha256 == second.visible_transcript_sha256
    for client in (client_a, client_b):
        messages = client.calls[1]["messages"]
        assistant = next(message for message in messages if message["role"] == "assistant")
        observation = next(message for message in messages if message["role"] == "tool")
        assert assistant["tool_calls"][0]["id"] == "call_1_0"
        assert observation["tool_call_id"] == "call_1_0"
