"""Phase 1: computed values and their presentation are separate authorities."""

from __future__ import annotations

import json

import pytest

from alchemist_rlm.budgets import Budget
from alchemist_rlm.engine import RLMEngine
from alchemist_rlm.mlx_client import ScriptedClient, text_reply, tool_reply
from alchemist_rlm.oolong_pairs import pair_output_contract, score_answer_value
from alchemist_rlm.output_contract import (
    OutputContract,
    TerminalPolicy,
    ValidationResult,
    canonical_answer_value,
    read_answer_value_record,
)


def _engine(tmp_path, replies, *, mode="raw", contract=None, turns=4,
            terminal_policy=None):
    return RLMEngine(
        client=ScriptedClient(replies),
        budget=Budget(max_turns=turns),
        runs_dir=tmp_path,
        output_mode=mode,
        output_contract=contract,
        terminal_policy=terminal_policy or TerminalPolicy(),
    )


def test_first_submit_keeps_value_and_explicit_text_apart(tmp_path):
    engine = _engine(
        tmp_path,
        [tool_reply("submit([('1', '2')], final_text='(1, 2)')")],
    )

    episode = engine.complete("context", "q", run_id="explicit_text")
    stored = json.loads((tmp_path / "explicit_text" / "episode.json").read_text())

    assert episode.answer_value == [["1", "2"]]
    assert episode.initial_final_text == "(1, 2)"
    assert episode.final_text == "(1, 2)"
    assert episode.answer == episode.final_text
    assert episode.presentation_source == "model_final_text"
    assert stored["answer_value_record"]["storage"] == "inline"
    assert stored["answer_value_record"]["value"] == [["1", "2"]]


def test_validated_repair_can_change_only_text(tmp_path):
    engine = _engine(
        tmp_path,
        [
            tool_reply("submit([('1', '2')])"),
            tool_reply("submit(final_text='(1, 2)')"),
        ],
        mode="validate_repair",
        contract=pair_output_contract(),
    )

    episode = engine.complete("context", "q", run_id="repair_text")

    assert episode.answer_value == [["1", "2"]]
    assert episode.initial_final_text == '[["1", "2"]]'
    assert episode.repair_candidate_text == "(1, 2)"
    assert episode.final_text == "(1, 2)"
    assert episode.contract_validation["valid"] is True
    assert episode.output_repair["promoted"] is True
    assert (episode.turns, episode.ledger["turns"]) == (2, 1)


def test_repair_window_rejects_a_new_value(tmp_path):
    engine = _engine(
        tmp_path,
        [
            tool_reply("submit([('1', '2')])"),
            tool_reply("submit([('9', '10')], final_text='(9, 10)')"),
        ],
        mode="validate_repair",
        contract=pair_output_contract(),
        terminal_policy=TerminalPolicy(
            max_presentation_attempts=1, max_protocol_retries=0),
    )

    episode = engine.complete("context", "q", run_id="repair_value_refused")

    assert episode.answer_value == [["1", "2"]]
    assert episode.repair_candidate_text is None
    assert episode.final_text == episode.initial_final_text
    assert episode.contract_validation["valid"] is False
    assert episode.output_repair["promoted"] is False
    assert episode.output_repair["error"]


def test_repair_window_accepts_atomic_reaffirmation_of_frozen_value(tmp_path):
    client = ScriptedClient([
        tool_reply("pairs = [('1', '2')]; submit(pairs)"),
        tool_reply(
            "candidate = '\\n'.join(f'({a}, {b})' for a, b in pairs)\n"
            "submit(pairs, final_text=candidate)"
        ),
    ])
    episode = RLMEngine(
        client=client, budget=Budget(max_turns=1), runs_dir=tmp_path,
        output_mode="validate_repair", output_contract=pair_output_contract(),
    ).complete("context", "q", run_id="atomic_presentation_reaffirmation")

    assert episode.answer_value == [["1", "2"]]
    assert episode.final_text == "(1, 2)"
    assert episode.contract_validation["valid"] is True
    assert episode.output_repair["attempts"][0]["candidate_committed"] is True


def test_invalid_repair_candidate_is_recorded_but_not_promoted(tmp_path):
    engine = _engine(
        tmp_path,
        [
            tool_reply("submit([('1', '2')])"),
            tool_reply("submit(final_text='1, 2')"),
        ],
        mode="validate_repair",
        contract=pair_output_contract(),
        terminal_policy=TerminalPolicy(max_presentation_attempts=1),
    )

    episode = engine.complete("context", "q", run_id="repair_invalid")

    assert episode.repair_candidate_text == "1, 2"
    assert episode.final_text == episode.initial_final_text
    assert episode.contract_validation["valid"] is False
    assert episode.output_repair["candidate_valid"] is False


def test_structurally_valid_repair_cannot_change_committed_content(tmp_path):
    engine = _engine(
        tmp_path,
        [
            tool_reply("submit([('1', '2')])"),
            tool_reply("submit('(9, 10)')"),
        ],
        mode="validate_repair",
        contract=pair_output_contract(),
        terminal_policy=TerminalPolicy(max_presentation_attempts=1),
    )

    episode = engine.complete("context", "q", run_id="binding_refuses_spoof")

    assert episode.repair_candidate_text == "(9, 10)"
    candidate = episode.output_repair["candidate_validation"]
    assert candidate["structural_valid"] is True
    assert candidate["binding"]["valid"] is False
    assert episode.final_text == episode.initial_final_text
    assert episode.answer_value == [["1", "2"]]


def test_invalid_committed_candidate_gets_one_bounded_validation_retry(tmp_path):
    client = ScriptedClient([
        tool_reply("submit([('1', '2'), ('3', '4')])"),
        tool_reply("submit('1, 2')"),
        tool_reply(
            "candidate = '\\n'.join(f'({a}, {b})' "
            "for a, b in PRESENTATION_VALUE)\nsubmit(candidate)"
        ),
    ])
    episode = RLMEngine(
        client=client, budget=Budget(max_turns=1), runs_dir=tmp_path,
        output_mode="validate_repair", output_contract=pair_output_contract(),
    ).complete("context", "q", run_id="validation_retry")

    assert episode.answer_value == [["1", "2"], ["3", "4"]]
    assert episode.final_text == "(1, 2)\n(3, 4)"
    assert episode.contract_validation["valid"] is True
    assert len(episode.output_repair["attempts"]) == 2
    assert episode.output_repair["attempts"][0]["validation_failed"] is True
    retry = client.calls[2]["messages"][-1]["content"]
    assert "contract rejected it" in retry
    assert "complete PRESENTATION_VALUE" in retry
    assert "printing cannot commit" in retry
    assert '"line": "(lower_numeric_id, higher_numeric_id)"' in retry
    assert episode.output_repair["history_compacted_after_attempt"] == 1


def test_progressive_validation_can_fix_shape_then_duplicates(tmp_path):
    """A new failure class revealed by a repair gets one bounded next step."""
    client = ScriptedClient([
        tool_reply("submit([('1', '2'), ('1', '2'), ('3', '4')])"),
        tool_reply("submit('1, 2\\n1, 2\\n3, 4')"),
        tool_reply("submit('(1, 2)\\n(1, 2)\\n(3, 4)')"),
        tool_reply("submit('(1, 2)\\n(3, 4)')"),
    ])
    episode = RLMEngine(
        client=client, budget=Budget(max_turns=1), runs_dir=tmp_path,
        output_mode="validate_repair", output_contract=pair_output_contract(),
    ).complete("context", "q", run_id="progressive_validation")

    assert episode.final_text == "(1, 2)\n(3, 4)"
    assert episode.contract_validation["valid"] is True
    attempts = episode.output_repair["attempts"]
    assert len(attempts) == 3
    assert attempts[0]["candidate_validation"]["issues"][0]["code"] == \
        "pair_parentheses_required"
    assert attempts[1]["candidate_validation"]["issues"][0]["code"] == \
        "duplicate_pair"
    assert attempts[2]["candidate_validation"]["valid"] is True
    assert episode.output_repair["history_compacted_after_attempt"] == 1
    assert episode.output_repair["history_compacted_after_attempts"] == [1, 2]


def test_pair_diagnostics_show_bounded_candidate_lines_not_only_numbers():
    result = pair_output_contract().validate(
        "1, 2\nnot a pair\n(9, 3)\n(4, 5)\n(4, 5)")

    by_code = {issue.code: issue for issue in result.issues}
    parentheses = by_code["pair_parentheses_required"]
    assert parentheses.count == 1
    assert parentheses.examples == ("line 1: '1, 2'",)
    assert "lacks the required literal opening '(' and closing ')'" in \
        parentheses.message
    assert by_code["invalid_pair_line"].examples == ("line 2: 'not a pair'",)
    assert by_code["pair_order"].examples == ("line 3: '(9, 3)'",)
    assert by_code["duplicate_pair"].examples == (
        "line 5: '(4, 5)' repeats line 4",)


def test_bare_pair_diagnostic_does_not_hide_order_or_duplicate_issues():
    result = pair_output_contract().validate("9, 3\n1, 2\n1, 2")

    by_code = {issue.code: issue for issue in result.issues}
    assert by_code["pair_parentheses_required"].count == 3
    assert by_code["pair_order"].examples == ("line 1: '9, 3'",)
    assert by_code["duplicate_pair"].examples == (
        "line 3: '1, 2' repeats line 2",)


def test_repeated_invalid_candidate_stops_without_spending_more_retries(tmp_path):
    client = ScriptedClient([
        tool_reply("submit([('1', '2')])"),
        tool_reply("submit('1, 2')"),
        tool_reply("submit('1, 2')"),
        tool_reply("submit('(1, 2)')"),
    ])
    episode = RLMEngine(
        client=client, budget=Budget(max_turns=1), runs_dir=tmp_path,
        output_mode="validate_repair", output_contract=pair_output_contract(),
    ).complete("context", "q", run_id="repeated_candidate")

    assert episode.contract_validation["valid"] is False
    assert episode.output_repair["error"] == "repeated_invalid_candidate"
    assert len(episode.output_repair["attempts"]) == 2
    assert len(client.calls) == 3


def test_binding_failure_is_rendered_and_can_recover_complete_value(tmp_path):
    client = ScriptedClient([
        tool_reply("submit([('1', '2'), ('3', '4')])"),
        tool_reply("submit('(1, 2)')"),
        tool_reply("submit('(1, 2)\\n(3, 4)')"),
    ])
    episode = RLMEngine(
        client=client, budget=Budget(max_turns=1), runs_dir=tmp_path,
        output_mode="validate_repair", output_contract=pair_output_contract(),
    ).complete("context", "q", run_id="binding_feedback")

    assert episode.contract_validation["valid"] is True
    retry = client.calls[2]["messages"][-1]["content"]
    assert "content binding" in retry
    assert "presentation_missing_content" in retry
    assert "(3, 4)" in retry
    assert episode.output_repair["history_compacted_after_attempt"] == 1


def test_rejected_candidate_is_rebound_exactly_as_presentation_draft(tmp_path):
    client = ScriptedClient([
        tool_reply("submit([('1', '2'), ('3', '4')])"),
        tool_reply("candidate = '1,2\\n3,4'; submit(candidate)"),
        tool_reply(
            "fixed = '\\n'.join(f'({line.replace(chr(44), chr(44) + chr(32))})' "
            "for line in PRESENTATION_DRAFT.splitlines())\nsubmit(fixed)"
        ),
    ])
    episode = RLMEngine(
        client=client, budget=Budget(max_turns=1), runs_dir=tmp_path,
        output_mode="validate_repair", output_contract=pair_output_contract(),
    ).complete("context", "q", run_id="rejected_candidate_draft")

    first = episode.output_repair["attempts"][0]
    assert first["persisted_as_draft"] is True
    assert episode.output_repair["rejected_candidate_draft_sha256"] == \
        first["candidate_sha256"]
    assert episode.final_text == "(1, 2)\n(3, 4)"
    assert episode.contract_validation["valid"] is True
    retry_messages = client.calls[2]["messages"]
    assert "exact rejected model-authored bytes" in retry_messages[-1]["content"]
    assert "PRESENTATION_DRAFT" in retry_messages[0]["content"]


def test_rejected_draft_survives_irrelevant_probe_and_non_text_submit(tmp_path):
    """Replay the protocol mistakes observed after the current t06 candidate."""
    client = ScriptedClient([
        tool_reply("submit([('1', '2'), ('3', '4')])"),
        tool_reply("submit('1,2\\n3,4')"),
        tool_reply("import sys\nprint(sys.version)"),
        tool_reply("submit(PRESENTATION_VALUE)"),
        tool_reply(
            "fixed = '\\n'.join(f'({line.replace(chr(44), chr(44) + chr(32))})' "
            "for line in PRESENTATION_DRAFT.splitlines())\nsubmit(fixed)"
        ),
    ])
    episode = RLMEngine(
        client=client, budget=Budget(max_turns=1), runs_dir=tmp_path,
        output_mode="validate_repair", output_contract=pair_output_contract(),
    ).complete("context", "q", run_id="draft_after_protocol_mistakes")

    attempts = episode.output_repair["attempts"]
    assert attempts[1]["protocol_error"] == "presentation_source_required"
    assert "direct Python globals, not modules or files" in \
        attempts[1]["error"]["message"]
    assert attempts[2]["error"]["type"] == "SubmitRefused"
    assert "PRESENTATION_DRAFT is the rejected str candidate" in \
        attempts[2]["error"]["message"]
    assert attempts[3]["candidate_validation"]["valid"] is True
    assert episode.final_text == "(1, 2)\n(3, 4)"


def test_invalid_debug_stdout_does_not_replace_rejected_candidate_draft(tmp_path):
    client = ScriptedClient([
        tool_reply("submit([('1', '2'), ('3', '4')])"),
        tool_reply("submit('1,2\\n3,4')"),
        tool_reply("print(type(PRESENTATION_DRAFT).__name__)"),
        tool_reply(
            "fixed = '\\n'.join(f'({line.replace(chr(44), chr(44) + chr(32))})' "
            "for line in PRESENTATION_DRAFT.splitlines())\n"
            "submit(result=fixed)"
        ),
    ])
    episode = RLMEngine(
        client=client, budget=Budget(max_turns=1), runs_dir=tmp_path,
        output_mode="validate_repair", output_contract=pair_output_contract(),
    ).complete("context", "q", run_id="retain_candidate_over_debug_stdout")

    second = episode.output_repair["attempts"][1]
    assert second["retained_rejected_candidate_draft"] is True
    assert "draft_persisted" not in second
    assert episode.output_repair["attempts"][2]["candidate_validation"]["valid"] is True
    assert episode.final_text == "(1, 2)\n(3, 4)"


def test_result_keyword_is_text_alias_only_in_presentation_window(tmp_path):
    episode = _engine(
        tmp_path,
        [tool_reply("submit([('1', '2')])"),
         tool_reply("candidate = '(1, 2)'; submit(result=candidate)")],
        mode="validate_repair", contract=pair_output_contract(),
    ).complete("context", "q", run_id="presentation_result_alias")

    assert episode.answer_value == [["1", "2"]]
    assert episode.final_text == "(1, 2)"
    assert episode.contract_validation["binding"]["valid"] is True


def test_candidate_keyword_is_text_alias_only_in_presentation_window(tmp_path):
    episode = _engine(
        tmp_path,
        [tool_reply("submit([('1', '2')])"),
         tool_reply("candidate = '(1, 2)'; submit(candidate=candidate)")],
        mode="validate_repair", contract=pair_output_contract(),
    ).complete("context", "q", run_id="presentation_candidate_alias")

    assert episode.answer_value == [["1", "2"]]
    assert episode.final_text == "(1, 2)"
    assert episode.contract_validation["binding"]["valid"] is True


@pytest.mark.parametrize("call", [
    "submit('(1, 2)', candidate='(1, 2)')",
    "submit(final_text='(1, 2)', candidate='(1, 2)')",
    "submit(result='(1, 2)', candidate='(1, 2)')",
])
def test_candidate_keyword_refuses_ambiguous_presentation_calls(tmp_path, call):
    episode = _engine(
        tmp_path,
        [tool_reply("submit([('1', '2')])"), tool_reply(call)],
        mode="validate_repair", contract=pair_output_contract(),
        terminal_policy=TerminalPolicy(
            max_presentation_attempts=1, max_protocol_retries=0),
    ).complete("context", "q", run_id="ambiguous_candidate_alias")

    attempt = episode.output_repair["attempts"][0]
    assert attempt["error"]["type"] == "SubmitRefused"
    assert "use exactly one" in attempt["error"]["message"]
    assert episode.final_text == episode.initial_final_text


def test_candidate_keyword_still_requires_text(tmp_path):
    episode = _engine(
        tmp_path,
        [tool_reply("submit([('1', '2')])"),
         tool_reply("submit(candidate=PRESENTATION_VALUE)")],
        mode="validate_repair", contract=pair_output_contract(),
        terminal_policy=TerminalPolicy(
            max_presentation_attempts=1, max_protocol_retries=0),
    ).complete("context", "q", run_id="candidate_alias_requires_text")

    attempt = episode.output_repair["attempts"][0]
    assert attempt["error"]["type"] == "SubmitRefused"
    assert "presentation text must be str" in attempt["error"]["message"]
    assert episode.final_text == episode.initial_final_text


def test_rejected_draft_can_progress_from_shape_failure_to_binding_failure(
    tmp_path,
):
    client = ScriptedClient([
        tool_reply("submit([('1', '2'), ('3', '4')])"),
        tool_reply("submit('1,2')"),
        tool_reply(
            "fixed = f'({PRESENTATION_DRAFT.replace(chr(44), chr(44) + chr(32))})'\n"
            "submit(fixed)"
        ),
        tool_reply(
            "complete = '\\n'.join(f'({a}, {b})' for a, b in "
            "PRESENTATION_VALUE)\nsubmit(complete)"
        ),
    ])
    episode = RLMEngine(
        client=client, budget=Budget(max_turns=1), runs_dir=tmp_path,
        output_mode="validate_repair", output_contract=pair_output_contract(),
    ).complete("context", "q", run_id="draft_shape_then_binding")

    attempts = episode.output_repair["attempts"]
    assert attempts[0]["candidate_validation"]["structural_valid"] is False
    assert attempts[1]["candidate_validation"]["structural_valid"] is True
    assert attempts[1]["candidate_validation"]["binding"]["valid"] is False
    assert attempts[1]["persisted_as_draft"] is True
    assert attempts[2]["candidate_validation"]["valid"] is True
    assert episode.final_text == "(1, 2)\n(3, 4)"


def test_binding_can_prove_a_text_value_was_only_reformatted(tmp_path):
    episode = _engine(
        tmp_path,
        [tool_reply("submit('1, 2\\n3, 9')"),
         tool_reply("submit('(1, 2)\\n(3, 9)')")],
        mode="validate_repair", contract=pair_output_contract(),
    ).complete("context", "q", run_id="text_value_binding")

    assert episode.answer_value == "1, 2\n3, 9"
    assert episode.final_text == "(1, 2)\n(3, 9)"
    assert episode.contract_validation["binding"]["valid"] is True
    assert episode.output_repair["promoted"] is True


def test_t14_terminal_state_replay_repairs_462_unparenthesized_lines(tmp_path):
    """Replay the observed terminal shape without repeating semantic work."""
    client = ScriptedClient([
        tool_reply(
            "pairs = [(10000 + i, 20000 + i) for i in range(462)]\n"
            "pairs_str = '\\n'.join(f'{a},{b}' for a, b in pairs)\n"
            "submit(pairs_str)"
        ),
        tool_reply(
            "fixed = '\\n'.join("
            "f'({line.replace(chr(44), chr(44) + chr(32))})' "
            "for line in PRESENTATION_VALUE.splitlines())\n"
            "submit(fixed)"
        ),
    ])
    episode = RLMEngine(
        client=client, budget=Budget(max_turns=1), runs_dir=tmp_path,
        output_mode="validate_repair", output_contract=pair_output_contract(),
    ).complete("context", "return pairs in the requested textual form",
               run_id="t14_terminal_replay")

    assert isinstance(episode.answer_value, str)
    assert len(episode.answer_value.splitlines()) == 462
    assert episode.initial_final_text.splitlines()[0] == "10000,20000"
    assert episode.final_text.splitlines()[0] == "(10000, 20000)"
    assert len(episode.final_text.splitlines()) == 462
    assert episode.contract_validation["valid"] is True
    assert episode.output_repair["promoted"] is True
    assert [item["channel"] for item in episode.visible_request_sha256s] == [
        "reasoning", "presentation"]


def test_t06_terminal_state_replay_repairs_5050_typed_pairs(tmp_path):
    """5050 pairs exercise the long typed-value path used by the t06 canary."""
    client = ScriptedClient([
        tool_reply(
            "users = list(range(10000, 10101))\n"
            "pairs = [(users[i], users[j]) for i in range(len(users)) "
            "for j in range(i + 1, len(users))]\n"
            "submit(pairs)"
        ),
        tool_reply(
            "fixed = '\\n'.join(f'({a}, {b})' "
            "for a, b in PRESENTATION_VALUE)\n"
            "submit(fixed)"
        ),
    ])
    episode = RLMEngine(
        client=client, budget=Budget(max_turns=1), runs_dir=tmp_path,
        output_mode="validate_repair", output_contract=pair_output_contract(),
    ).complete("context", "return pairs in the requested textual form",
               run_id="t06_terminal_replay")

    assert len(episode.answer_value) == 5_050
    assert len(episode.final_text.splitlines()) == 5_050
    assert episode.contract_validation["valid"] is True
    assert episode.contract_validation["binding"]["valid"] is True
    assert episode.output_repair["promoted"] is True


def test_t06_long_rejected_candidate_is_corrected_from_persistent_draft(
    tmp_path,
):
    """Replay the current 6,670-line failure without reserializing its value."""
    client = ScriptedClient([
        tool_reply(
            "pairs = [(10000 + i, 30000 + i) for i in range(6670)]\n"
            "submit(pairs)"
        ),
        tool_reply(
            "pairs_str = '\\n'.join(f'{a},{b}' for a, b in pairs)\n"
            "submit(pairs_str)"
        ),
        tool_reply(
            "fixed = '\\n'.join(f'({line.replace(chr(44), chr(44) + chr(32))})' "
            "for line in PRESENTATION_DRAFT.splitlines())\nsubmit(fixed)"
        ),
    ])
    episode = RLMEngine(
        client=client, budget=Budget(max_turns=1), runs_dir=tmp_path,
        output_mode="validate_repair", output_contract=pair_output_contract(),
    ).complete("context", "q", run_id="t06_rejected_draft_replay")

    attempts = episode.output_repair["attempts"]
    assert len(episode.answer_value) == 6_670
    assert len(episode.final_text.splitlines()) == 6_670
    assert attempts[0]["persisted_as_draft"] is True
    assert attempts[1]["candidate_validation"]["valid"] is True
    assert episode.contract_validation["binding"]["valid"] is True
    assert [call["max_tokens"] for call in client.calls] == [4096, 4096, 1024]


def test_t06_replay_can_truncate_then_inspect_then_submit_persistent_value(tmp_path):
    """The measured t06 path gets one clean inspection after a truncation."""
    from alchemist_rlm.mlx_client import Reply

    cut = Reply(
        content="",
        tool_calls=tool_reply("candidate = 'unfinished'").tool_calls,
        reasoning=None,
        finish_reason="length",
        served_model="scripted",
    )
    client = ScriptedClient([
        tool_reply(
            "users = list(range(10000, 10101))\n"
            "pairs = [(users[i], users[j]) for i in range(len(users)) "
            "for j in range(i + 1, len(users))]\n"
            "submit(pairs)"
        ),
        cut,
        tool_reply(
            "print(type(PRESENTATION_VALUE).__name__, len(PRESENTATION_VALUE))"
        ),
        tool_reply(
            "final_text = '\\n'.join(f'({a}, {b})' for a, b in "
            "PRESENTATION_VALUE) + '\\n'\nsubmit(final_text)"
        ),
    ])
    episode = RLMEngine(
        client=client, budget=Budget(max_turns=1), runs_dir=tmp_path,
        output_mode="validate_repair", output_contract=pair_output_contract(),
    ).complete("context", "q", run_id="t06_truncate_inspect_submit")

    assert len(episode.final_text.splitlines()) == 5_050
    assert episode.contract_validation["valid"] is True
    attempts = episode.output_repair["attempts"]
    assert attempts[0]["protocol_error"] == "truncated"
    assert attempts[1]["clean_progress"] is True
    assert attempts[2]["candidate_committed"] is True
    assert episode.output_repair["clean_build_turns"] == 1


def test_t11_replay_can_fix_shape_then_missing_binding_then_complete(tmp_path):
    """Each newly exposed failure class receives one bounded correction."""
    client = ScriptedClient([
        tool_reply("submit([('1', '2'), ('3', '4'), ('5', '6')])"),
        tool_reply("submit('1, 2\\n3, 4\\n5, 6')"),
        tool_reply("submit('(1, 2)\\n(3, 4)')"),
        tool_reply("submit('(1, 2)\\n(3, 4)\\n(5, 6)')"),
    ])
    episode = RLMEngine(
        client=client, budget=Budget(max_turns=1), runs_dir=tmp_path,
        output_mode="validate_repair", output_contract=pair_output_contract(),
    ).complete("context", "q", run_id="t11_progressive_binding")

    attempts = episode.output_repair["attempts"]
    assert attempts[0]["candidate_validation"]["issues"][0]["code"] == \
        "pair_parentheses_required"
    assert attempts[1]["candidate_validation"]["structural_valid"] is True
    assert attempts[1]["candidate_validation"]["binding"]["issues"][0]["code"] == \
        "presentation_missing_content"
    assert attempts[2]["candidate_validation"]["valid"] is True
    assert episode.final_text == "(1, 2)\n(3, 4)\n(5, 6)"


def test_commit_prompts_expose_atomic_value_and_text_delivery():
    from alchemist_rlm.native_loop import COMMIT_FIRST, COMMIT_SECOND, \
        TRUNCATION_RECOVERY

    for prompt in (COMMIT_FIRST, COMMIT_SECOND):
        assert "submit(value, final_text=final_text)" in prompt
        assert "existing" in prompt
        assert "inline" in prompt
    assert "session variable" in TRUNCATION_RECOVERY
    assert "short submit call" in TRUNCATION_RECOVERY


def test_last_turn_partial_sweep_gets_one_targeted_terminal_retry(tmp_path):
    from alchemist_rlm.native_loop import NativeLoop, Submission

    delivered = {"value": None}

    def observation(*, status, retry_exhausted):
        return {
            "ok": True, "stdout": "", "stderr": "", "error": None,
            "defined": [], "changed": {}, "value": None, "truncated": False,
            "progress": True, "delivered": delivered["value"] is not None,
            "presentation_candidate": False,
            "operation_result": {
                "operation": "semantic_map", "status": status,
                "valid_items": 7 if status == "partial" else 8,
                "total_items": 8, "retry_exhausted": retry_exhausted,
            },
        }

    def execute(code):
        if code == "start_partial()":
            return observation(status="partial", retry_exhausted=False)
        if "retry_failed(semantic_result)" in code:
            assert "semantic_map(" not in code
            return observation(status="complete", retry_exhausted=True)
        assert code == "submit(8)"
        delivered["value"] = 8
        return observation(status="complete", retry_exhausted=True)

    client = ScriptedClient([
        tool_reply("start_partial()"),
        tool_reply("result = semantic_map('again', {'type': 'boolean'})"),
        tool_reply("result = retry_failed(semantic_result)"),
        tool_reply("submit(8)"),
    ])
    result = NativeLoop(
        client=client, execute=execute, budget=Budget(max_turns=1),
        read_submission=lambda: Submission(
            delivered=delivered["value"] is not None,
            value=delivered["value"],
        ),
    ).run("q")

    assert result.answer_value == 8
    commit = client.calls[1]["messages"][-1]["content"]
    assert "7 of 8" in commit
    assert "retry_failed(semantic_result) exactly once" in commit
    assert "do not call semantic_map" in commit
    refusal = client.calls[2]["messages"][-2]["content"]
    assert "TerminalSweepRefused" in refusal
    assert "retry_failed(semantic_result)" in refusal


def test_truncated_retyping_reaches_commit_with_persistent_value(tmp_path):
    """A cut attempt to inline a large answer must leave the earlier value
    reachable and put an explicit no-retyping instruction on the commit turn.

    This is the terminal transition observed in the real t14 canary: `pairs`
    existed, but the model tried to serialize the collection inside a new tool
    argument until the generation limit cut it off.
    """
    from alchemist_rlm.mlx_client import Reply

    cut = Reply(
        content="",
        tool_calls=tool_reply("pairs = [(1, 2)] * 10000").tool_calls,
        reasoning=None,
        finish_reason="length",
        served_model="scripted",
    )
    client = ScriptedClient([
        tool_reply("pairs = [(1, 2), (3, 4)]"),
        cut,
        tool_reply("submit(pairs)"),
    ])
    episode = RLMEngine(
        client=client, budget=Budget(max_turns=2), runs_dir=tmp_path,
    ).complete("context", "return every pair", run_id="t14_commit_transition")

    assert episode.answer_value == [[1, 2], [3, 4]]
    assert episode.stop_reason == "forced_final:max_turns:submitted"
    commit_message = client.calls[2]["messages"][-1]["content"]
    assert "reference its existing variable by name" in commit_message
    assert "do not spell out its items" in commit_message


def test_protocol_retry_recovers_print_without_promoting_stdout(tmp_path):
    client = ScriptedClient([
        tool_reply("submit([('1', '2')])"),
        tool_reply(
            "for a, b in PRESENTATION_VALUE:\n    print(f'({a}, {b})')"
        ),
        tool_reply("submit(PRESENTATION_DRAFT)"),
    ])
    episode = RLMEngine(
        client=client, budget=Budget(max_turns=2), runs_dir=tmp_path,
        output_mode="validate_repair", output_contract=pair_output_contract(),
    ).complete("context", "q", run_id="presentation_protocol_retry")

    assert episode.final_text == "(1, 2)\n"
    first = episode.output_repair["attempts"][0]
    assert first["source"] == "stdout_draft"
    assert first["draft_validation"]["valid"] is True
    assert first["candidate_committed"] is False
    assert episode.output_repair["attempts"][1]["candidate_committed"] is True
    assert episode.ledger["turns"] == 1
    assert [item["channel"] for item in episode.visible_request_sha256s] == [
        "reasoning", "presentation", "presentation"]
    retry_message = client.calls[2]["messages"][-1]["content"]
    assert "submit(PRESENTATION_DRAFT)" in retry_message


def test_t20_terminal_replay_preserves_valid_stdout_until_explicit_commit(tmp_path):
    client = ScriptedClient([
        tool_reply(
            "sorted_pairs = [(10000 + i, 20000 + i) for i in range(231)]\n"
            "submit(sorted_pairs)"
        ),
        tool_reply(
            "for pair in sorted_pairs:\n"
            "    print(f'({pair[0]}, {pair[1]})')"
        ),
        tool_reply("submit(PRESENTATION_DRAFT)"),
    ])
    episode = RLMEngine(
        client=client, budget=Budget(max_turns=1), runs_dir=tmp_path,
        output_mode="validate_repair", output_contract=pair_output_contract(),
    ).complete("context", "q", run_id="t20_stdout_commit_replay")

    assert len(episode.final_text.splitlines()) == 231
    assert episode.contract_validation["valid"] is True
    assert episode.output_repair["source_variable"] == "sorted_pairs"
    first = episode.output_repair["attempts"][0]
    assert first["source"] == "stdout_draft"
    assert first["candidate_committed"] is False
    assert first["draft_validation"]["valid"] is True
    assert episode.output_repair["attempts"][1]["source"] == "submit"
    first_prompt = client.calls[1]["messages"][-1]["content"]
    # Preserve the historically proven first repair request byte-for-byte.
    # The factual source name remains recorded and usable inside the REPL; it
    # is not injected into the initial model prompt.
    assert "persistent variable `sorted_pairs`" not in first_prompt
    assert episode.output_repair["source_variable"] == "sorted_pairs"


def test_t07_terminal_replay_recovers_after_wrong_type_and_unparenthesized_text(
    tmp_path,
):
    """Replay the real t07 terminal sequence with its original answer size."""
    client = ScriptedClient([
        tool_reply(
            "valid_pairs = [(10000 + i, 20000 + i) for i in range(5671)]\n"
            "submit(valid_pairs)"
        ),
        tool_reply(
            "final_text = '\\n'.join("
            "f'{pair[0]},{pair[1]}' for pair in valid_pairs)\n"
            "submit(valid_pairs)"
        ),
        tool_reply("submit(final_text)"),
        tool_reply(
            "corrected = '\\n'.join("
            "f'({pair[0]}, {pair[1]})' for pair in valid_pairs)\n"
            "submit(corrected)"
        ),
    ])
    episode = RLMEngine(
        client=client, budget=Budget(max_turns=1), runs_dir=tmp_path,
        output_mode="validate_repair", output_contract=pair_output_contract(),
    ).complete("context", "q", run_id="t07_historical_terminal_replay")

    attempts = episode.output_repair["attempts"]
    assert episode.output_repair["source_variable"] == "valid_pairs"
    assert attempts[0]["protocol_error"] == "invalid_signature_or_block"
    assert attempts[0]["error"]["type"] == "SubmitRefused"
    assert attempts[1]["candidate_validation"]["structural_valid"] is False
    examples = attempts[1]["candidate_validation"]["issues"][0]["examples"]
    assert examples[:2] == [
        "line 1: '10000,20000'",
        "line 2: '10001,20001'",
    ]
    assert attempts[2]["candidate_validation"]["valid"] is True
    assert episode.contract_validation["valid"] is True
    assert episode.contract_validation["binding"]["valid"] is True
    assert len(episode.final_text.splitlines()) == 5671
    assert [call["max_tokens"] for call in client.calls] == [
        4096, 4096, 1024, 1024,
    ]


def test_two_clean_turns_allow_inspection_then_variable_then_submit(tmp_path):
    client = ScriptedClient([
        tool_reply("pairs = [('1', '2'), ('3', '4')]; submit(pairs)"),
        tool_reply(
            "print(type(PRESENTATION_VALUE).__name__, len(PRESENTATION_VALUE))"
        ),
        tool_reply(
            "output = '\\n'.join(f'({a}, {b})' for a, b in PRESENTATION_VALUE)"
        ),
        tool_reply("submit(output)"),
    ])
    episode = RLMEngine(
        client=client, budget=Budget(max_turns=1), runs_dir=tmp_path,
        output_mode="validate_repair", output_contract=pair_output_contract(),
    ).complete("context", "q", run_id="inspect_build_submit")

    assert episode.final_text == "(1, 2)\n(3, 4)"
    assert episode.contract_validation["valid"] is True
    assert episode.output_repair["clean_build_turns"] == 2
    assert episode.output_repair["attempts"][0]["draft_validation"]["valid"] is False
    assert episode.output_repair["attempts"][1]["defined"] == ["output"]
    assert episode.output_repair["attempts"][2]["candidate_committed"] is True


def test_last_clean_build_earns_one_commit_only_reserve(tmp_path):
    client = ScriptedClient([
        tool_reply("submit([['1', '2']])"),
        tool_reply(
            "candidate = '\\n'.join(f'({a}, {b})' for a, b in "
            "PRESENTATION_VALUE)"
        ),
        tool_reply("submit(candidate)"),
    ])
    episode = RLMEngine(
        client=client, budget=Budget(max_turns=1), runs_dir=tmp_path,
        output_mode="validate_repair", output_contract=pair_output_contract(),
        terminal_policy=TerminalPolicy(
            max_presentation_attempts=1,
            max_protocol_retries=0,
            max_presentation_commit_reserve=1,
        ),
    ).complete("context", "q", run_id="last_build_commit_reserve")

    attempts = episode.output_repair["attempts"]
    assert episode.final_text == "(1, 2)"
    assert episode.contract_validation["valid"] is True
    assert episode.output_repair["commit_reserve_earned_after_attempt"] == 1
    assert episode.output_repair["commit_reserve_used"] is True
    assert attempts[0]["clean_progress"] is True
    assert attempts[1]["reserved_commit_turn"] is True
    assert attempts[1]["candidate_committed"] is True
    reserved_messages = client.calls[2]["messages"]
    assert [message["role"] for message in reserved_messages] == [
        "system", "user", "user", "user"]
    assert "reserved commit-only turn" in reserved_messages[-1]["content"]
    assert all("candidate = '\\n'.join" not in message.get("content", "")
               for message in reserved_messages)
    assert episode.output_repair[
        "history_compacted_for_commit_reserve_after_attempt"] == 1


def test_reserved_commit_turn_refuses_more_inspection_and_does_not_extend(tmp_path):
    client = ScriptedClient([
        tool_reply("submit([['1', '2']])"),
        tool_reply(
            "candidate = '\\n'.join(f'({a}, {b})' for a, b in "
            "PRESENTATION_VALUE)"
        ),
        tool_reply("print(candidate)"),
        tool_reply("submit(candidate)"),
    ])
    episode = RLMEngine(
        client=client, budget=Budget(max_turns=1), runs_dir=tmp_path,
        output_mode="validate_repair", output_contract=pair_output_contract(),
        terminal_policy=TerminalPolicy(
            max_presentation_attempts=1,
            max_protocol_retries=0,
            max_presentation_commit_reserve=1,
        ),
    ).complete("context", "q", run_id="commit_reserve_is_bounded")

    attempts = episode.output_repair["attempts"]
    assert len(client.calls) == 3
    assert len(attempts) == 2
    assert attempts[1]["reserved_commit_turn"] is True
    assert attempts[1]["protocol_error"] == "presentation_commit_required"
    assert attempts[1]["error"]["type"] == "PresentationCommitRequired"
    assert episode.final_text == episode.initial_final_text


def test_final_protocol_error_does_not_earn_commit_reserve(tmp_path):
    client = ScriptedClient([
        tool_reply("submit([['1', '2']])"),
        tool_reply("raise TypeError('bad block')"),
        tool_reply("submit('(1, 2)')"),
    ])
    episode = RLMEngine(
        client=client, budget=Budget(max_turns=1), runs_dir=tmp_path,
        output_mode="validate_repair", output_contract=pair_output_contract(),
        terminal_policy=TerminalPolicy(
            max_presentation_attempts=1,
            max_protocol_retries=0,
            max_presentation_commit_reserve=1,
        ),
    ).complete("context", "q", run_id="error_has_no_commit_reserve")

    assert len(client.calls) == 2
    assert len(episode.output_repair["attempts"]) == 1
    assert "commit_reserve_used" not in episode.output_repair
    assert "commit_reserve_earned_after_attempt" not in episode.output_repair


def test_stdout_draft_with_changed_content_is_not_promoted(tmp_path):
    client = ScriptedClient([
        tool_reply("pairs = [('1', '2')]; submit(pairs)"),
        tool_reply(
            "print(f'({int(PRESENTATION_VALUE[0][0]) + 8}, 10)')"
        ),
        tool_reply(
            "candidate = '\\n'.join(f'({a}, {b})' for a, b in "
            "PRESENTATION_VALUE)\nsubmit(candidate)"
        ),
    ])
    episode = RLMEngine(
        client=client, budget=Budget(max_turns=1), runs_dir=tmp_path,
        output_mode="validate_repair", output_contract=pair_output_contract(),
    ).complete("context", "q", run_id="stdout_binding_refusal")

    first = episode.output_repair["attempts"][0]
    assert first["draft_validation"]["structural_valid"] is True
    assert first["draft_validation"]["binding"]["valid"] is False
    assert first["candidate_committed"] is False
    assert episode.final_text == "(1, 2)"


def test_repeated_invalid_stdout_draft_stops_without_looping(tmp_path):
    client = ScriptedClient([
        tool_reply("submit([('1', '2')])"),
        tool_reply("print(str(PRESENTATION_VALUE))"),
        tool_reply("print(str(PRESENTATION_VALUE))"),
        tool_reply("submit('(1, 2)')"),
    ])
    episode = RLMEngine(
        client=client, budget=Budget(max_turns=1), runs_dir=tmp_path,
        output_mode="validate_repair", output_contract=pair_output_contract(),
    ).complete("context", "q", run_id="repeated_stdout_draft")

    assert episode.output_repair["error"] == "repeated_invalid_draft"
    assert len(episode.output_repair["attempts"]) == 2
    assert len(client.calls) == 3
    assert episode.final_text == episode.initial_final_text


def test_repeated_unhashable_block_gets_generic_python_recovery(tmp_path):
    failing_block = (
        "seen = set()\n"
        "for item in PRESENTATION_VALUE:\n"
        "    seen.add(item)"
    )
    client = ScriptedClient([
        tool_reply("pairs = [['1', '2']]; submit(pairs)"),
        tool_reply(failing_block),
        tool_reply(failing_block),
        tool_reply(
            "keys = [tuple(item) for item in PRESENTATION_VALUE]\n"
            "candidate = '\\n'.join(f'({a}, {b})' for a, b in keys)\n"
            "submit(candidate)"
        ),
    ])
    episode = RLMEngine(
        client=client, budget=Budget(max_turns=1), runs_dir=tmp_path,
        output_mode="validate_repair", output_contract=pair_output_contract(),
    ).complete("context", "q", run_id="unhashable_python_recovery")

    attempts = episode.output_repair["attempts"]
    assert episode.final_text == "(1, 2)"
    assert episode.contract_validation["valid"] is True
    assert attempts[0]["error"]["type"] == "TypeError"
    assert any("hashable" in action and "tuple(value)" in action
               for action in attempts[0]["next_actions"])
    assert attempts[1]["repeated_failed_block"] is True
    assert attempts[1]["previous_failed_attempt"] == 1
    first_retry = "\n".join(
        message.get("content", "") for message in client.calls[2]["messages"])
    second_retry = "\n".join(
        message.get("content", "") for message in client.calls[3]["messages"])
    assert "hashable" in first_retry and "tuple(value)" in first_retry
    assert "exact block already produced the same error" in second_retry


def test_third_clean_noncommit_turn_is_refused_but_one_submit_retry_remains(tmp_path):
    client = ScriptedClient([
        tool_reply("pairs = [('1', '2')]; submit(pairs)"),
        tool_reply("x = len(pairs)"),
        tool_reply("candidate = '\\n'.join(f'({a}, {b})' for a, b in pairs)"),
        tool_reply("z = len(pairs)"),
        tool_reply("submit(candidate)"),
    ])
    episode = RLMEngine(
        client=client, budget=Budget(max_turns=1), runs_dir=tmp_path,
        output_mode="validate_repair", output_contract=pair_output_contract(),
    ).complete("context", "q", run_id="bounded_clean_builds")

    assert episode.final_text == "(1, 2)"
    assert episode.output_repair["clean_build_turns"] == 2
    assert episode.output_repair["attempts"][2]["protocol_error"] == \
        "presentation_commit_required"
    assert episode.output_repair["attempts"][3]["candidate_committed"] is True


def test_truncated_presentation_retry_keeps_repl_but_compacts_root_history(tmp_path):
    from alchemist_rlm.mlx_client import Reply

    cut = Reply(
        content="",
        tool_calls=tool_reply("candidate = 'unfinished'").tool_calls,
        reasoning=None,
        finish_reason="length",
        served_model="scripted",
    )
    client = ScriptedClient([
        tool_reply("pairs = [('1', '2')]; submit(pairs)"),
        cut,
        tool_reply(
            "candidate = '\\n'.join(f'({a}, {b})' "
            "for a, b in PRESENTATION_VALUE)\nsubmit(candidate)"
        ),
    ])
    episode = RLMEngine(
        client=client, budget=Budget(max_turns=1), runs_dir=tmp_path,
        output_mode="validate_repair", output_contract=pair_output_contract(),
    ).complete("context", "q", run_id="compact_truncated_presentation")

    assert episode.final_text == "(1, 2)"
    assert episode.contract_validation["valid"] is True
    assert episode.output_repair["history_compacted_after_attempt"] == 1
    retry_messages = client.calls[2]["messages"]
    assert [message["role"] for message in retry_messages] == [
        "system", "user", "user"]
    assert "PRESENTATION_VALUE" in retry_messages[1]["content"]
    assert "PRESENTATION-ONLY TERMINAL STATE" in retry_messages[1]["content"]
    assert "Current text:" not in retry_messages[1]["content"]
    assert "presentation-only terminal state" in \
        retry_messages[0]["content"].lower()
    assert client.calls[0]["max_tokens"] == 4096
    assert client.calls[1]["max_tokens"] == 4096
    assert client.calls[2]["max_tokens"] == 1024
    assert "truncated" in retry_messages[2]["content"]
    assert all("pairs = [('1', '2')]" not in message.get("content", "")
               for message in retry_messages)


def test_protocol_retry_names_a_code_fence_that_never_called_the_tool(tmp_path):
    client = ScriptedClient([
        tool_reply("submit([('1', '2')])"),
        text_reply("```python\nsubmit('(1, 2)')\n```"),
        tool_reply("submit('(1, 2)')"),
    ])
    episode = RLMEngine(
        client=client, budget=Budget(max_turns=1), runs_dir=tmp_path,
        output_mode="validate_repair", output_contract=pair_output_contract(),
    ).complete("context", "q", run_id="fenced_protocol_retry")

    assert episode.final_text == "(1, 2)"
    assert episode.output_repair["attempts"][0]["protocol_error"] == \
        "code_fence_without_tool"
    retry = client.calls[2]["messages"][-1]["content"]
    assert "code_fence_without_tool" in retry
    assert "Invoke PythonInterpreter through the provided tool interface" in retry


def test_two_clean_build_turns_remain_bounded_and_compact_stdout(tmp_path):
    client = ScriptedClient([
        tool_reply("submit([('1', '2')])"),
        tool_reply("print(str(PRESENTATION_VALUE) + 'x' * 2_000)"),
        tool_reply("print(str(PRESENTATION_VALUE) + 'y' * 2_000)"),
        tool_reply("submit('(1, 2)')"),
    ])
    episode = RLMEngine(
        client=client, budget=Budget(max_turns=1), runs_dir=tmp_path,
        output_mode="validate_repair", output_contract=pair_output_contract(),
    ).complete("context", "q", run_id="two_protocol_retries")

    assert episode.final_text == "(1, 2)"
    assert len(episode.output_repair["attempts"]) == 3
    third_prompt = client.calls[3]["messages"]
    assert third_prompt[0]["role"] == "system"
    assert episode.output_repair["clean_build_turns"] == 2
    assert all("x" * 1_100 not in message.get("content", "")
               and "y" * 1_100 not in message.get("content", "")
               for message in third_prompt)
    assert "history_compacted_after_attempts" not in episode.output_repair
    assert "Two clean local construction turns" in third_prompt[-1]["content"]


def test_repair_keeps_root_history_and_trained_tool_channel(tmp_path):
    client = ScriptedClient([
        tool_reply("submit([('1', '2')])"),
        tool_reply("submit('(1, 2)')"),
    ])
    RLMEngine(
        client=client, budget=Budget(max_turns=1), runs_dir=tmp_path,
        output_mode="validate_repair", output_contract=pair_output_contract(),
    ).complete("context", "q", run_id="presentation_system_anchor")

    messages = client.calls[1]["messages"]
    assert [message["role"] for message in messages] == [
        "system", "user", "assistant", "tool", "user"]
    system = messages[0]["content"]
    assert "persistent Python session" in system
    assert "PRESENTATION-ONLY TERMINAL STATE" not in system
    assert messages[1]["content"].endswith("\n\nq")
    assert messages[2]["tool_calls"]
    repair = messages[-1]["content"]
    assert "Continue in this same conversation" in repair
    assert "committed and cannot be changed" in repair
    assert "submit(result=" not in repair
    assert "list[1] of list[2] of str[1 chars]" in repair
    assert client.calls[1]["max_tokens"] == 4096
    assert "exactly one textual presentation" in repair
    assert "source variable" not in repair


def test_repair_feedback_and_history_are_bounded(tmp_path):
    errors = tuple(f"row {index} has an invalid shape" for index in range(10_000))

    def validate(text):
        return (ValidationResult(True) if text == "ok"
                else ValidationResult(False, errors))

    contract = OutputContract("bounded", "1", {"shape": "ok"}, validate)
    client = ScriptedClient([text_reply("bad"), tool_reply("submit('ok')")])
    episode = RLMEngine(
        client=client, budget=Budget(max_turns=2), runs_dir=tmp_path,
        output_mode="validate_repair", output_contract=contract,
    ).complete("context", "q", run_id="bounded_feedback")

    repair_messages = client.calls[1]["messages"]
    assert [message["role"] for message in repair_messages] == [
        "system", "user", "assistant", "user"]
    feedback = repair_messages[-1]["content"]
    assert len(feedback) < 6_000
    assert "row 0 has an invalid shape" in feedback
    assert "row 9999 has an invalid shape" not in feedback
    assert "This message intentionally contains no excerpt" in \
        feedback
    assert "Current-text preview" not in feedback
    assert "no typed value; use PRESENTATION_TEXT" in feedback
    assert episode.final_text == "ok"


def test_typed_validation_requires_a_declared_content_binding(tmp_path):
    contract = OutputContract(
        "yes", "1", {"enum": ["yes"]},
        lambda text: ValidationResult(text == "yes", () if text == "yes" else ("yes",)),
    )
    episode = _engine(
        tmp_path, [tool_reply("submit(True, final_text='yes')")],
        mode="validate_only", contract=contract,
    ).complete("context", "q", run_id="missing_binding")

    assert episode.contract_validation["structural_valid"] is True
    assert episode.contract_validation["binding"]["valid"] is False
    assert episode.contract_validation["valid"] is False


def test_valid_initial_text_needs_no_auxiliary_turn(tmp_path):
    client = ScriptedClient([
        tool_reply("submit([('1', '2')], final_text='(1, 2)')"),
        tool_reply("submit(final_text='unreached')"),
    ])
    engine = RLMEngine(
        client=client,
        budget=Budget(max_turns=4),
        runs_dir=tmp_path,
        output_mode="validate_repair",
        output_contract=pair_output_contract(),
    )

    episode = engine.complete("context", "q", run_id="already_valid")

    assert episode.final_text == "(1, 2)"
    assert episode.output_repair is None
    assert len(client.calls) == 1


def test_terminal_text_can_be_repaired_without_inventing_a_value(tmp_path):
    engine = _engine(
        tmp_path,
        [text_reply("1, 2"), tool_reply("submit(final_text='(1, 2)')")],
        mode="validate_repair",
        contract=pair_output_contract(),
    )

    episode = engine.complete("context", "q", run_id="text_repair")

    assert episode.answer_delivered is False
    assert episode.answer_value_record is None
    assert episode.initial_final_text == "1, 2"
    assert episode.final_text == "(1, 2)"


def test_pair_contract_is_structural_and_never_receives_gold():
    contract = pair_output_contract()

    assert contract.validate("").valid is True
    assert contract.validate("(1, 2)\n(3, 9)").valid is True
    assert contract.validate("(2, 1)").valid is False
    assert contract.validate("(1, 2)\n(1, 2)").valid is False
    assert contract.validate("answer: (1, 2)").valid is False
    assert len(contract.sha256) == 64


def test_typed_pair_scorer_is_independent_of_presentation():
    truth = {("1", "2"), ("3", "9")}

    assert score_answer_value([["2", "1"], [3, 9]], truth)["f1"] == 1.0
    assert score_answer_value([], truth)["predicted"] == 0
    assert score_answer_value(["(1, 2)"], truth) is None


def test_answer_value_contract_rejects_silent_stringification():
    assert canonical_answer_value((1, {"x": [2]})) == [1, {"x": [2]}]

    with pytest.raises(TypeError, match="transportable"):
        canonical_answer_value({"when": object()})


def test_large_answer_value_uses_a_verified_artifact(tmp_path):
    value = list(range(12_000))
    engine = _engine(tmp_path, [tool_reply(f"submit({value!r})")])

    episode = engine.complete("context", "q", run_id="large_value")
    record = episode.answer_value_record

    assert record["storage"] == "artifact"
    assert record["sha256"]
    assert record["ref"].startswith("artifact://answer-value-")
    assert not isinstance(episode.answer_value, str)
    assert read_answer_value_record(record, tmp_path / "large_value" / "artifacts") == value


def test_large_string_answer_is_stored_as_canonical_json(tmp_path):
    value = "(1, 2)\n" * 5_000
    engine = _engine(tmp_path, [tool_reply(f"submit({value!r})")])

    episode = engine.complete("context", "q", run_id="large_string")
    record = episode.answer_value_record
    artifact_root = tmp_path / "large_string" / "artifacts"
    artifact = artifact_root / f"{record['ref'].removeprefix('artifact://')}.txt"

    assert json.loads(artifact.read_text()) == value
    assert read_answer_value_record(record, artifact_root) == value


def test_answer_value_record_refuses_tampered_inline_data(tmp_path):
    engine = _engine(tmp_path, [tool_reply("submit([1, 2])")])
    episode = engine.complete("context", "q", run_id="tampered")
    record = dict(episode.answer_value_record)
    record["value"] = [1, 3]

    with pytest.raises(ValueError, match="digest mismatch"):
        read_answer_value_record(record, tmp_path / "tampered" / "artifacts")


@pytest.mark.parametrize("value", [None, False, 0, "", [], {}])
def test_every_empty_json_value_is_a_real_delivery(tmp_path, value):
    engine = _engine(tmp_path, [tool_reply(f"submit({value!r})")])
    episode = engine.complete("context", "q", run_id=f"empty_{type(value).__name__}")

    assert episode.answer_delivered is True
    assert episode.answer_value == value
    assert episode.answer_value_record is not None


def test_validate_only_records_failure_without_feedback(tmp_path):
    client = ScriptedClient([
        tool_reply("submit([['1', '2']])"),
        tool_reply("submit(final_text='unreached')"),
    ])
    episode = RLMEngine(
        client=client, budget=Budget(max_turns=4), runs_dir=tmp_path,
        output_mode="validate_only", output_contract=pair_output_contract(),
    ).complete("context", "q", run_id="validate_only")

    assert episode.contract_validation["valid"] is False
    assert episode.output_repair is None
    assert len(client.calls) == 1


def test_presentation_budget_remains_after_two_commit_attempts(tmp_path):
    client = ScriptedClient([
        tool_reply("x = 1"),
        tool_reply("y = 2"),
        tool_reply("submit([['1', '2']])"),
        tool_reply("submit(final_text='unreached')"),
    ])
    episode = RLMEngine(
        client=client, budget=Budget(max_turns=1), runs_dir=tmp_path,
        output_mode="validate_repair", output_contract=pair_output_contract(),
        terminal_policy=TerminalPolicy(max_presentation_attempts=1),
    ).complete("context", "q", run_id="commit_two_no_repair")

    assert episode.stop_reason == "forced_final:max_turns:submitted"
    assert episode.contract_validation["valid"] is False
    assert episode.output_repair["attempted"] is True
    assert episode.repair_candidate_text == "unreached"
    assert len(client.calls) == 4


def test_constrained_mode_is_labelled_assisted_and_never_repairs(tmp_path):
    client = ScriptedClient([
        tool_reply("submit([['1', '2']], final_text='(1, 2)')"),
        tool_reply("submit(final_text='unreached')"),
    ])
    episode = RLMEngine(
        client=client, budget=Budget(max_turns=4), runs_dir=tmp_path,
        output_mode="constrained", output_contract=pair_output_contract(),
    ).complete("context", "q", run_id="constrained")

    assert episode.presentation_source == "assisted_initial"
    assert episode.output_repair is None
    assert len(client.calls) == 1


def test_output_modes_require_an_explicit_compatible_contract(tmp_path):
    with pytest.raises(ValueError, match="requires an OutputContract"):
        _engine(tmp_path, [], mode="validate_only").complete("context", "q")
    with pytest.raises(ValueError, match="cannot carry"):
        _engine(tmp_path, [], mode="raw", contract=pair_output_contract()).complete(
            "context", "q")


@pytest.mark.parametrize(
    ("name", "validator", "good", "bad"),
    [
        ("json", lambda text: ValidationResult(text == '{\"ok\":true}',
                                                () if text == '{\"ok\":true}' else ("JSON object required",)),
         '{"ok":true}', "ok"),
        ("choice", lambda text: ValidationResult(text in {"yes", "no"},
                                                  () if text in {"yes", "no"} else ("closed choice required",)),
         "yes", "maybe"),
        ("csv", lambda text: ValidationResult(text.count(",") == 1,
                                               () if text.count(",") == 1 else ("two CSV columns required",)),
         "a,b", "a"),
    ],
)
def test_output_contract_is_generic_across_unrelated_formats(name, validator, good, bad):
    contract = OutputContract(name, "1", {"example": good}, validator)

    assert contract.validate(good).valid is True
    assert contract.validate(bad).valid is False
