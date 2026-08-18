"""Gate A invariants: one finishing machine and complete provenance.

These tests intentionally describe the migration boundary, not any OOLONG
answer.  They use a scripted backend so no inference is spent.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from alchemist_rlm.budgets import Budget
from alchemist_rlm.engine import RLMEngine
from alchemist_rlm.mlx_client import Reply, ScriptedClient, text_reply, tool_reply
from alchemist_rlm.oolong_pairs import pair_output_contract


def _engine(tmp_path: Path, replies: list[Reply]) -> RLMEngine:
    return RLMEngine(
        client=ScriptedClient(replies),
        budget=Budget(max_turns=1),
        runs_dir=tmp_path,
    )


@pytest.mark.parametrize(
    "first_commit",
    [
        tool_reply("pairs = [(1, 2)]"),
        tool_reply("raise ValueError('not delivered')"),
        text_reply("I still need to deliver it"),
        Reply(
            content="unfinished",
            tool_calls=[],
            reasoning=None,
            finish_reason="length",
            served_model="scripted",
        ),
    ],
    ids=["clean_python", "error", "prose", "truncation"],
)
def test_every_undelivered_first_commit_gets_one_final_opportunity(
    tmp_path: Path, first_commit: Reply
) -> None:
    engine = _engine(
        tmp_path,
        [tool_reply("seed = 1"), first_commit, tool_reply("submit([(1, 2)])")],
    )

    episode = engine.complete("context", "return the pair", run_id="commit_retry")

    assert episode.answer_value == [[1, 2]]
    assert episode.answer_delivered is True
    assert episode.turns == 3


def test_two_undelivered_commit_turns_end_without_promoting_prose_or_stdout(
    tmp_path: Path,
) -> None:
    engine = _engine(
        tmp_path,
        [
            tool_reply("seed = 1"),
            tool_reply("print('debug one')"),
            text_reply("I would continue if another turn existed"),
        ],
    )

    episode = engine.complete("context", "q", run_id="commit_exhausted")

    assert episode.answer is None
    assert episode.answer_delivered is False
    assert episode.turns == 3


def test_a_tag_and_tool_call_cannot_create_two_commit_authorities(tmp_path: Path) -> None:
    hybrid = tool_reply("submit('typed')")
    hybrid.content = "<answer>tagged</answer>"
    engine = _engine(tmp_path, [tool_reply("seed = 1"), hybrid])

    episode = engine.complete("context", "q", run_id="one_commit_authority")

    assert episode.answer == "tagged"
    assert episode.answer_delivered is False
    assert not any(
        error.get("kind") == "final_block_executed"
        for error in episode.protocol_errors
    )


def test_commit_and_output_repair_share_two_auxiliary_turns(tmp_path: Path) -> None:
    client = ScriptedClient(
        [
            tool_reply("pairs = [(1, 2)]"),
            tool_reply("submit(pairs)"),
            tool_reply("submit(final_text='(1, 2)')"),
            tool_reply("submit(final_text='unreachable')"),
        ]
    )
    engine = RLMEngine(
        client=client,
        budget=Budget(max_turns=1),
        runs_dir=tmp_path,
        output_mode="validate_repair",
        output_contract=pair_output_contract(),
    )

    episode = engine.complete("context", "q", run_id="shared_auxiliary_ceiling")

    assert episode.turns == 3
    assert len(client.calls) == 3
    assert episode.answer_value == [[1, 2]]
    assert episode.final_text == "(1, 2)"


def test_normal_delivery_uses_only_one_output_repair(
    tmp_path: Path,
) -> None:
    client = ScriptedClient(
        [
            tool_reply("submit([(1, 2)])"),
            tool_reply("submit(final_text='(1, 2)')"),
            tool_reply("submit(final_text='unreachable')"),
        ]
    )
    engine = RLMEngine(
        client=client,
        budget=Budget(max_turns=4),
        runs_dir=tmp_path,
        output_mode="validate_repair",
        output_contract=pair_output_contract(),
    )

    episode = engine.complete("context", "q", run_id="two_conformance_turns")

    assert episode.turns == 2
    assert episode.answer_value == [[1, 2]]
    assert episode.final_text == "(1, 2)"
    assert len(client.calls) == 2


def test_manifest_exposes_both_model_visible_contract_hashes() -> None:
    from alchemist_rlm.manifest import RunManifest

    names = {field.name for field in dataclasses.fields(RunManifest)}
    assert {"interaction_contract_sha256", "observation_contract_sha256"} <= names


def test_every_runner_records_the_observation_contract() -> None:
    scripts = Path(__file__).resolve().parent.parent / "scripts"
    for runner in ("run_pairs_pilot.py", "smoke.py", "run_suite.py"):
        source = (scripts / runner).read_text()
        assert "observation_contract_sha256=observation_contract_sha256()" in source, runner


def test_output_repair_text_moves_the_interaction_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    from alchemist_rlm import manifest, native_loop

    before = manifest.interaction_contract_sha256()
    monkeypatch.setattr(native_loop, "OUTPUT_REPAIR", native_loop.OUTPUT_REPAIR + " changed")
    assert manifest.interaction_contract_sha256() != before


def test_observation_renderer_moves_its_own_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    from alchemist_rlm import manifest, native_loop

    before = manifest.observation_contract_sha256()
    original = native_loop.render
    monkeypatch.setattr(native_loop, "render", lambda observation: original(observation) + " changed")
    assert manifest.observation_contract_sha256() != before


def test_visible_transcript_hash_is_canonical_and_ignores_invisible_metadata() -> None:
    from alchemist_rlm.manifest import visible_transcript_sha256

    messages = [
        {"role": "system", "content": "rules", "timestamp": 1},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "PythonInterpreter", "arguments": '{"code":"x=1"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "ok", "run_id": "a"},
    ]
    same_visible = [dict(message, timestamp=99, run_id="b") for message in messages]
    changed = [dict(message) for message in messages]
    changed[-1] = dict(changed[-1], content="different")

    assert visible_transcript_sha256(messages) == visible_transcript_sha256(same_visible)
    assert visible_transcript_sha256(messages) != visible_transcript_sha256(changed)
    assert visible_transcript_sha256(messages) != visible_transcript_sha256(list(reversed(messages)))


def test_completed_episode_records_its_visible_transcript_digest(tmp_path: Path) -> None:
    engine = RLMEngine(
        client=ScriptedClient([tool_reply("submit(0)")]),
        budget=Budget(max_turns=2),
        runs_dir=tmp_path,
    )

    episode = engine.complete("context", "q", run_id="transcript_digest")
    stored = json.loads((tmp_path / "transcript_digest" / "episode.json").read_text())
    events = [
        json.loads(line)
        for line in (tmp_path / "transcript_digest" / "trace.jsonl").read_text().splitlines()
    ]

    assert len(episode.visible_transcript_sha256) == 64
    assert stored["visible_transcript_sha256"] == episode.visible_transcript_sha256
    assert events[-1]["visible_transcript_sha256"] == episode.visible_transcript_sha256


def test_pairs_result_records_the_episode_transcript_digest() -> None:
    """The aggregate gate record must retain the episode's audit identity."""
    runner = (Path(__file__).resolve().parent.parent /
              "scripts" / "run_pairs_pilot.py").read_text()
    assert "visible_transcript_sha256=(\n                              episode.visible_transcript_sha256)" in runner
