"""Formal result consolidation fails closed before producing an aggregate."""

from __future__ import annotations

import json

import pytest

from alchemist_rlm.budgets import Budget
from alchemist_rlm.consolidate import (
    ConsolidationError,
    _official_pair_truths,
    consolidate_pair_results,
    manifest_signature,
    verify_episode_output,
)
from alchemist_rlm.engine import RLMEngine
from alchemist_rlm.mlx_client import ScriptedClient, tool_reply
from alchemist_rlm.oolong_pairs import pair_output_contract


def _manifest():
    contract = pair_output_contract()
    return {
        "run_id": "shard",
        "arm": "alchemist",
        "suite": "oolong_pairs_auto",
        "sampling": {"temperature": 0.0, "seed": 0},
        "output_mode": "validate_repair",
        "output_contract": contract.manifest(),
        "output_contract_sha256": contract.sha256,
        "terminal_policy": {
            "max_commit_attempts": 2,
            "max_partial_recovery_attempts": 1,
            "max_presentation_attempts": 3,
            "max_protocol_retries": 2,
            "max_presentation_build_turns": 2,
            "max_presentation_commit_reserve": 1,
            "max_presentation_tokens": 1024,
            "max_feedback_chars": 4000,
        },
        "runtime_determinism": {
            "request_seed": 0,
            "tool_call_ids": "canonical_turn_index",
            "visible_requests": "strict_json_sha256",
        },
        "isolation_policy": {"name": "mlx_prompt_cache_reset_v1"},
        "output_backend_constraint": "none",
        "git": {"commit": "abc", "code_dirty": False},
        "server_argv": ["serve --prompt-cache-size 10 --prompt-cache-bytes 1GB"],
        "model_segments": [{"requested": "model", "served": "model"}],
        "model_stayed_put": True,
    }


def test_manifest_signature_moves_with_contract_and_budget():
    first = _manifest()
    changed_contract = json.loads(json.dumps(first))
    changed_contract["output_contract_sha256"] = "different"
    changed_budget = json.loads(json.dumps(first))
    changed_budget["budget"] = {"max_turns": 99}

    assert manifest_signature(first) != manifest_signature(changed_contract)
    assert manifest_signature(first) != manifest_signature(changed_budget)


def test_consolidation_rebuilds_truths_from_the_one_official_context():
    truths = _official_pair_truths()

    assert len(truths) == 20
    assert len(truths[12]) == 1505
    assert len(truths[14]) == 1137
    assert len(truths[16]) == 209
    assert len(truths[18]) == 980
    assert len(truths[20]) == 314


def test_consolidation_rejects_a_duplicate_task_before_scoring(tmp_path):
    document = {"manifest": _manifest(), "results": [
        {"task": 1, "execution_status": "completed"},
    ]}
    one = tmp_path / "one.json"
    two = tmp_path / "two.json"
    one.write_text(json.dumps(document))
    two.write_text(json.dumps(document))

    with pytest.raises(ConsolidationError, match="duplicate task 1"):
        consolidate_pair_results([one, two], runs_dir=tmp_path, expected_tasks=(1,))


def test_output_verification_rejects_a_tampered_artifact(tmp_path):
    value = list(range(12_000))
    episode = RLMEngine(
        client=ScriptedClient([tool_reply(f"submit({value!r})")]),
        budget=Budget(max_turns=2), runs_dir=tmp_path,
    ).complete("context", "q", run_id="tamper_artifact")
    stored = json.loads((tmp_path / "tamper_artifact" / "episode.json").read_text())
    ref = stored["answer_value_record"]["ref"].removeprefix("artifact://")
    (tmp_path / "tamper_artifact" / "artifacts" / f"{ref}.txt").write_text("[]")

    with pytest.raises(ConsolidationError, match="digest mismatch"):
        verify_episode_output(stored, tmp_path / "tamper_artifact" / "artifacts")


def test_output_verification_rejects_unvalidated_promotion(tmp_path):
    episode = RLMEngine(
        client=ScriptedClient([
            tool_reply("submit([['1', '2']])"),
            tool_reply("submit(final_text='(1, 2)')"),
        ]),
        budget=Budget(max_turns=3), runs_dir=tmp_path,
        output_mode="validate_repair", output_contract=pair_output_contract(),
    ).complete("context", "q", run_id="bad_promotion")
    stored = json.loads((tmp_path / "bad_promotion" / "episode.json").read_text())
    stored["output_repair"]["candidate_validation"]["valid"] = False

    with pytest.raises(ConsolidationError, match="not validated"):
        verify_episode_output(stored, tmp_path / "bad_promotion" / "artifacts")
