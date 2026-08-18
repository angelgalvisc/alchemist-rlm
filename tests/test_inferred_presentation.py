"""Question-only format inference and its deterministic local linter."""

from __future__ import annotations

import pytest

from alchemist_rlm.budgets import Budget
from alchemist_rlm.engine import RLMEngine
from alchemist_rlm.inferred_presentation import (
    check_presentation,
    infer_presentation_spec,
    normalize_presentation_spec,
    parse_presentation_spec,
    render_presentation,
)
from alchemist_rlm.mlx_client import Reply, ScriptedClient, tool_reply
from alchemist_rlm.oolong_pairs import pair_output_contract


PAIR_SPEC = {
    "version": 1,
    "kind": "records",
    "record_separator": "newline",
    "prefix": "(",
    "suffix": ")",
    "field_separator": ", ",
    "fields": ["integer", "integer"],
    "ordering": "numeric_ascending",
    "duplicates": False,
    "allow_empty": True,
    "additional_text": False,
}


def test_question_only_inference_freezes_a_bounded_spec():
    client = ScriptedClient([Reply(
        content="", reasoning=None, finish_reason="tool_calls",
        served_model="scripted", usage={"completion_tokens": 20},
        tool_calls=[{
            "id": "format_0", "type": "function",
            "function": {
                "name": "declare_records",
                "arguments": __import__("json").dumps({
                    key: value for key, value in PAIR_SPEC.items()
                    if key not in {"version", "kind"}
                }),
            },
        }],
    )])
    question = "Return one pair per line in the format (user_id_1, user_id_2)."

    record = infer_presentation_spec(client, question)

    assert record["status"] == "ok"
    assert record["spec"] == PAIR_SPEC
    sent = client.calls[0]["messages"]
    assert question in sent[1]["content"]
    assert "context" not in sent[1]["content"].lower()
    assert client.calls[0]["tools"][0]["function"]["name"] == \
        "declare_records"
    assert record["inference_channel"] == "tool_call"
    assert record["declaration_tool"] == "declare_records"
    assert record["declaration_arguments"]["fields"] == ["integer", "integer"]
    assert len(record["declaration_arguments_sha256"]) == 64


def test_invalid_declaration_preserves_raw_tool_evidence():
    arguments = {
        **{key: value for key, value in PAIR_SPEC.items()
           if key not in {"version", "kind"}},
        "fields": ["string", "string", "string"],
    }
    client = ScriptedClient([Reply(
        content="", reasoning=None, finish_reason="tool_calls",
        served_model="scripted", usage={"completion_tokens": 20},
        tool_calls=[{
            "id": "format_bad", "type": "function",
            "function": {
                "name": "declare_records",
                "arguments": __import__("json").dumps(arguments),
            },
        }],
    )])

    record = infer_presentation_spec(client, "Return records.")

    assert record["status"] == "invalid"
    assert record["error"] == "numeric_ascending requires exactly two fields"
    assert record["declaration_tool"] == "declare_records"
    assert record["declaration_arguments"] == arguments
    assert "string" in record["declaration_arguments_text"]


def test_spec_parser_accepts_wrapping_but_rejects_an_unbounded_language():
    parsed = parse_presentation_spec(
        "```json\n" + __import__("json").dumps(PAIR_SPEC) + "\n```")
    assert parsed == PAIR_SPEC
    with pytest.raises(ValueError, match="kind is not supported"):
        normalize_presentation_spec({
            "version": 1, "kind": "python", "code": "return True",
        })


def test_record_linter_reports_shape_order_and_duplicates_without_rewriting():
    candidate = "9, 3\n1, 2\n1, 2"

    report = check_presentation(candidate, PAIR_SPEC)

    assert report["valid"] is False
    by_code = {issue["code"]: issue for issue in report["issues"]}
    assert by_code["record_delimiters"]["count"] == 3
    assert by_code["record_order"]["examples"] == ["line 1: '9, 3'"]
    assert by_code["duplicate_record"]["examples"] == [
        "line 3: '1, 2' repeats line 2"]
    assert candidate == "9, 3\n1, 2\n1, 2"


def test_numeric_order_is_an_independent_constraint_on_string_fields():
    spec = {**PAIR_SPEC, "field_separator": ",",
            "fields": ["string", "string"]}

    assert normalize_presentation_spec(spec) == spec
    assert check_presentation("(2,10)", spec)["valid"] is True

    nonnumeric = check_presentation("(left,right)", spec)
    assert nonnumeric["valid"] is False
    assert {issue["code"] for issue in nonnumeric["issues"]} == {
        "numeric_order_type"}

    descending = check_presentation("(10,2)", spec)
    assert descending["valid"] is False
    assert {issue["code"] for issue in descending["issues"]} == {"record_order"}


def test_generic_renderer_uses_only_the_frozen_records_spec():
    value = [["9", "3"], ["1", "2"], ["1", "2"]]

    rendered = render_presentation(value, PAIR_SPEC)

    assert rendered == "(3, 9)\n(1, 2)\n"
    assert value == [["9", "3"], ["1", "2"], ["1", "2"]]
    assert check_presentation(rendered, PAIR_SPEC)["valid"] is True


def test_generic_renderer_accepts_only_unambiguous_textual_records():
    compact_spec = {**PAIR_SPEC, "field_separator": ","}

    assert render_presentation(
        ["9, 3", "(1,2)", "1,2"], compact_spec
    ) == "(3,9)\n(1,2)\n"

    with pytest.raises(ValueError, match="cannot be split unambiguously"):
        render_presentation(["1,2,3"], compact_spec)
    with pytest.raises(ValueError, match="incomplete frozen delimiters"):
        render_presentation(["(1,2"], compact_spec)


def test_generic_renderer_rejects_content_it_cannot_serialize_safely():
    with pytest.raises(ValueError, match="strict numeric ordering"):
        render_presentation([["7", "7"]], PAIR_SPEC)
    string_spec = {
        **PAIR_SPEC, "fields": ["string", "string"], "ordering": "none",
    }
    with pytest.raises(ValueError, match="unambiguously"):
        render_presentation([["1, 2", "3"]], string_spec)
    with pytest.raises(TypeError, match="primitive"):
        render_presentation([[object(), "3"]], PAIR_SPEC)


def test_generic_renderer_supports_the_other_declared_families():
    assert render_presentation(
        {"answer": [1, 2]}, {"version": 1, "kind": "json", "root": "object"}
    ) == '{"answer":[1,2]}'
    assert render_presentation(
        42, {"version": 1, "kind": "scalar", "value_type": "integer",
             "additional_text": False}
    ) == "42"
    assert render_presentation(
        "explanation", {"version": 1, "kind": "free_text"}
    ) == "explanation"
    with pytest.raises(ValueError, match="json_root"):
        render_presentation([], {"version": 1, "kind": "json", "root": "object"})


def test_model_invoked_renderer_still_requires_validation_and_submit(tmp_path):
    client = ScriptedClient([
        tool_reply(
            "pairs = [['9', '3'], ['1', '2'], ['1', '2']]\nsubmit(pairs)"
        ),
        tool_reply(
            "candidate = render_presentation(PRESENTATION_VALUE)\n"
            "assert check_presentation(candidate)['valid']\n"
            "submit(candidate)"
        ),
    ])
    episode = RLMEngine(
        client=client,
        budget=Budget(max_turns=1),
        runs_dir=tmp_path,
        output_mode="validate_repair",
        output_contract=pair_output_contract(),
        inferred_presentation_spec=PAIR_SPEC,
    ).complete("private context", "public question", run_id="generic_renderer")

    assert episode.answer_value == [["9", "3"], ["1", "2"], ["1", "2"]]
    assert episode.final_text == "(3, 9)\n(1, 2)\n"
    assert episode.contract_validation["valid"] is True
    assert episode.output_repair["promoted"] is True
    assert len(episode.presentation_renders) == 1
    assert episode.presentation_renders[0]["ok"] is True
    assert len(episode.presentation_checks) == 1
    assert "submit(render_presentation(PRESENTATION_VALUE))" in \
        client.calls[1]["messages"][-1]["content"]


def test_model_can_submit_textual_records_through_one_generic_transaction(tmp_path):
    compact_spec = {**PAIR_SPEC, "field_separator": ","}
    client = ScriptedClient([
        tool_reply("rows = ['9, 3', '1,2']; submit(rows)"),
        tool_reply("submit(render_presentation(PRESENTATION_VALUE))"),
    ])

    episode = RLMEngine(
        client=client,
        budget=Budget(max_turns=1),
        runs_dir=tmp_path,
        output_mode="validate_repair",
        output_contract=pair_output_contract(),
        inferred_presentation_spec=compact_spec,
    ).complete("private context", "public question", run_id="text_record_renderer")

    assert episode.final_text == "(3,9)\n(1,2)\n"
    assert episode.contract_validation["valid"] is True
    assert episode.output_repair["promoted"] is True
    assert len(episode.presentation_renders) == 1
    assert "submit(render_presentation(PRESENTATION_VALUE))" not in \
        client.calls[1]["messages"][-1]["content"]
    assert "check_presentation" in client.calls[1]["messages"][-1]["content"]


def test_explicit_structured_submit_is_rendered_then_bound_and_validated(tmp_path):
    compact_spec = {**PAIR_SPEC, "field_separator": ","}
    client = ScriptedClient([
        tool_reply("rows = ['9, 3', '1,2']; submit(rows)"),
        tool_reply("submit(PRESENTATION_VALUE)"),
    ])

    episode = RLMEngine(
        client=client,
        budget=Budget(max_turns=1),
        runs_dir=tmp_path,
        output_mode="validate_repair",
        output_contract=pair_output_contract(),
        inferred_presentation_spec=compact_spec,
    ).complete("private context", "public question", run_id="structured_submit")

    assert episode.answer_value == ["9, 3", "1,2"]
    assert episode.final_text == "(3,9)\n(1,2)\n"
    assert episode.contract_validation["valid"] is True
    assert episode.output_repair["promoted"] is True
    assert len(episode.presentation_renders) == 1


def test_inferred_linter_appears_only_after_a_rejected_candidate(tmp_path):
    client = ScriptedClient([
        tool_reply("pairs = [('1', '2')]; submit(pairs)"),
        tool_reply("submit('1, 2')"),
        tool_reply(
            "report = check_presentation(PRESENTATION_DRAFT)\n"
            "candidate = f'({PRESENTATION_DRAFT})'\n"
            "assert check_presentation(candidate)['valid']\n"
            "submit(candidate)"
        ),
    ])
    episode = RLMEngine(
        client=client,
        budget=Budget(max_turns=1),
        runs_dir=tmp_path,
        output_mode="validate_repair",
        output_contract=pair_output_contract(),
        inferred_presentation_spec=PAIR_SPEC,
    ).complete("private context", "public question", run_id="inferred_linter")

    assert episode.final_text == "(1, 2)"
    assert episode.contract_validation["valid"] is True
    assert episode.output_repair["inferred_presentation_spec"] == PAIR_SPEC
    # The checker is available as soon as the original delivery has factually
    # failed, while the task-solving trajectory itself remains unchanged.
    assert "check_presentation" in client.calls[1]["messages"][-1]["content"]
    assert "check_presentation" in client.calls[2]["messages"][-1]["content"]


def test_inferred_linter_is_callable_during_the_first_repair(tmp_path):
    client = ScriptedClient([
        tool_reply("pairs = [('1', '2')]; submit(pairs)"),
        tool_reply(
            "candidate = '\\n'.join(f'({a}, {b})' for a, b in pairs)\n"
            "report = check_presentation(candidate)\n"
            "assert report['valid']\n"
            "submit(candidate)"
        ),
    ])
    episode = RLMEngine(
        client=client,
        budget=Budget(max_turns=1),
        runs_dir=tmp_path,
        output_mode="validate_repair",
        output_contract=pair_output_contract(),
        inferred_presentation_spec=PAIR_SPEC,
    ).complete("private context", "public question", run_id="first_repair_linter")

    assert episode.final_text == "(1, 2)"
    assert episode.contract_validation["valid"] is True
    assert len(episode.presentation_checks) == 1
    assert episode.presentation_checks[0]["valid"] is True
