"""Execution mistakes use the declared turn budget unless explicitly capped."""

from alchemist_rlm.budgets import Budget
from alchemist_rlm.engine import RLMEngine
from alchemist_rlm.mlx_client import ScriptedClient, tool_reply


def test_progressive_errors_do_not_force_an_early_commit_by_default(tmp_path):
    client = ScriptedClient([
        tool_reply("missing_one"),
        tool_reply("missing_two"),
        tool_reply("missing_three"),
        tool_reply("submit('bounded')"),
    ])
    episode = RLMEngine(
        client=client, budget=Budget(max_turns=3), runs_dir=tmp_path,
    ).complete("context", "q", run_id="errors_use_turn_budget")

    assert episode.stop_reason == "forced_final:max_turns:submitted"
    assert episode.answer == "bounded"
    assert episode.ledger["turns"] == 3


def test_optional_consecutive_error_policy_remains_available(tmp_path):
    client = ScriptedClient([
        tool_reply("missing_one"),
        tool_reply("missing_two"),
        tool_reply("submit('bounded')"),
    ])
    episode = RLMEngine(
        client=client,
        budget=Budget(max_turns=6, max_consecutive_errors=2),
        runs_dir=tmp_path,
    ).complete("context", "q", run_id="explicit_error_guard")

    assert episode.stop_reason == "forced_final:consecutive_errors:submitted"
    assert episode.ledger["turns"] == 2
