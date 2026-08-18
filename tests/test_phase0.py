"""Phase 0 tests: everything that can be checked without loading a model.

The plan budgets 38 inference episodes for the whole decision, so anything a
fake backend can catch must be caught here — a wasted episode is a wasted
minute of the 90-minute budget and, worse, an ambiguous result.

Run: ./.venv/bin/python -m pytest tests/test_phase0.py -q
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from alchemist_rlm import fingerprint as fp  # noqa: E402
from alchemist_rlm import protocol, tasks  # noqa: E402


# --- protocol: parsing what the server actually returns --------------------
def _server_message(code: str, name: str = protocol.TOOL_NAME) -> dict:
    """The shape mlx_lm.server produces: arguments as a JSON *string*."""
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_0",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps({"code": code})},
            }
        ],
    }


def test_normalize_extracts_code():
    turn = protocol.normalize(_server_message("print(len(context))"))
    assert len(turn.calls) == 1
    assert turn.calls[0].code == "print(len(context))"
    assert turn.calls[0].name == protocol.TOOL_NAME
    assert turn.unknown == []


def test_unknown_tool_is_surfaced_never_dropped():
    """The Alchemist was trained with google_search and read_page too.

    If such a call vanished, the loop would see "no tool call", treat the turn
    as a final answer, and record a protocol failure as a deliberate reply.
    """
    turn = protocol.normalize(_server_message("x", name="google_search"))
    assert turn.calls == []
    assert len(turn.unknown) == 1 and turn.unknown[0]["name"] == "google_search"
    assert turn.called_a_tool is True

    observation = protocol.unknown_tool_observation("google_search")
    assert observation["error"] == "unknown_tool"
    assert observation["next_actions"]


def test_turn_with_no_tool_call_is_distinguishable():
    turn = protocol.normalize({"content": "Paris", "tool_calls": []})
    assert turn.called_a_tool is False and turn.content == "Paris"


def test_normalize_survives_unparseable_arguments():
    """A truncated generation can yield arguments that are not valid JSON.

    Losing the turn to a JSONDecodeError would be scored as a protocol failure
    when it is really a truncation, so the raw string is kept as code.
    """
    message = {
        "tool_calls": [
            {"function": {"name": protocol.TOOL_NAME, "arguments": "print(1"}}
        ]
    }
    turn = protocol.normalize(message)
    assert len(turn.calls) == 1 and turn.calls[0].code == "print(1"


def test_alias_tool_has_identical_schema():
    """Gate B's name ablation is only valid if nothing else differs."""
    a = protocol.python_tool(protocol.TOOL_NAME)
    b = protocol.python_tool(protocol.ALIAS_TOOL_NAME)
    a["function"]["name"] = b["function"]["name"] = "_"
    assert a == b


# --- termination -----------------------------------------------------------
def test_answer_tag_takes_the_last_pair():
    assert protocol.answer_tag("noise <answer>Paris</answer>") == "Paris"
    assert protocol.answer_tag("<ANSWER>\n 42 \n</ANSWER>") == "42"
    assert protocol.answer_tag("no tags here") is None
    assert protocol.answer_tag("") is None


# --- duplicate ledger ------------------------------------------------------
def test_identical_block_is_recognised_and_told_but_still_runs():
    """The ledger recognises a repeat and names the turn it first ran on. It no
    longer refuses it: the refusal asserted "its result is unchanged", which a
    session with state does not entitle the harness to say."""
    ledger = protocol.CallLedger()
    code = "print([r for r in rows if 'x' in r])"
    ledger.record(code, turn=3)

    previous = ledger.duplicate_of(code + "\n")          # trailing newline only
    assert previous is not None, "whitespace-only change must still count as duplicate"

    note = ledger.note_repeat(previous)
    assert note["repeated_from_turn"] == 3
    assert "same code as turn 3" in note["repeat_note"]
    assert ledger.duplicates == 1
    # The note merges into a real observation, so it carries no verdict of its
    # own — no `ok`, no `error` — and it never claims the result is the same.
    assert "ok" not in note and "error" not in note
    assert not any("unchanged" in str(v) for v in note.values())


def test_genuinely_different_block_is_not_refused():
    ledger = protocol.CallLedger()
    ledger.record("print(a)", turn=1)
    assert ledger.duplicate_of("print(b)") is None


# --- frozen tasks ----------------------------------------------------------
def test_gate_a_shape():
    assert len(tasks.GATE_A) == 6
    assert len(tasks.PYTHON_TASKS) == 5
    assert len({t.id for t in tasks.GATE_A}) == 6


def test_truths_are_consistent_with_the_context_the_model_sees():
    """Guards against the truth drifting from the literal after an edit."""
    rows = [line.split(" | ") for line in tasks.LEDGER.splitlines()]
    assert sum(int(r[2]) for r in rows if r[3] == "paid") == tasks.GATE_A[2].truth
    assert sum(1 for r in rows if r[1] == "logistics") == tasks.GATE_A[1].truth
    assert len(rows) * 7 == tasks.GATE_A[3].truth
    void = [r for r in rows if r[3] == "void"]
    assert max(void, key=lambda r: int(r[2]))[0] == tasks.GATE_A[4].truth


def test_scorers_reject_near_misses():
    """A loose scorer already turned a 1/2 into a 2/2 in this repository."""
    arithmetic = tasks.GATE_A[0]
    assert arithmetic.scores("48408847")
    assert arithmetic.scores("The answer is 48,408,847.")
    assert not arithmetic.scores("48408848")
    assert not arithmetic.scores("484088470")     # substring of a longer number
    assert not arithmetic.scores("")

    paris = tasks.GATE_A[5]
    assert paris.scores("Paris") and paris.scores("it is paris")
    assert not paris.scores("Lyon")


def test_tasks_hash_is_actually_sensitive():
    """Stability alone proves nothing — a constant is stable too."""
    import dataclasses, hashlib, json

    baseline = [t.to_dict() for t in tasks.GATE_A]
    assert hashlib.sha256(json.dumps(baseline, sort_keys=True).encode()).hexdigest() \
        == tasks.TASKS_SHA256

    edited = list(tasks.GATE_A)
    edited[0] = dataclasses.replace(edited[0], question=edited[0].question + " Now.")
    changed = hashlib.sha256(
        json.dumps([t.to_dict() for t in edited], sort_keys=True).encode()
    ).hexdigest()
    assert changed != tasks.TASKS_SHA256, "editing a question must change the hash"


# --- fingerprints ----------------------------------------------------------
def test_fingerprint_detects_a_changed_template(tmp_path):
    """The failure this whole module exists to catch.

    The Alchemist's chat template differs from its upstream original only in
    identity strings — invisible to size, name and mtime, caught only by a hash.
    """
    model = tmp_path / "m"
    model.mkdir()
    (model / "config.json").write_text('{"model_type": "qwen3_5"}')
    (model / "chat_template.jinja").write_text("You are Intern-A1.")
    (model / "model.safetensors").write_bytes(b"weights")

    before = fp.model_fingerprint(model)
    (model / "chat_template.jinja").write_text("You are Agent-A1 (The Alchemist).")
    after = fp.model_fingerprint(model)

    assert before["metadata_sha256"] != after["metadata_sha256"]
    assert before["weights"] == after["weights"], "weights did not change"


def test_fingerprint_detects_same_size_different_weights(tmp_path):
    """Same size, same headers, different checkpoint — this actually happened."""
    model = tmp_path / "m"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (model / "model.safetensors").write_bytes(b"A" * 1024)
    before = fp.model_fingerprint(model)
    (model / "model.safetensors").write_bytes(b"B" * 1024)
    after = fp.model_fingerprint(model)

    assert before["weights"]["model.safetensors"]["bytes"] == \
           after["weights"]["model.safetensors"]["bytes"]
    assert before["weights"] != after["weights"]


def test_matches_names_what_changed(tmp_path):
    model = tmp_path / "m"
    model.mkdir()
    (model / "config.json").write_text("{}")
    a = {"models": {"x": fp.model_fingerprint(model, hash_weights=False)}}
    (model / "config.json").write_text('{"changed": true}')
    b = {"models": {"x": fp.model_fingerprint(model, hash_weights=False)}}

    same, differences = fp.matches(a, b)
    assert not same
    assert any("config.json" in d for d in differences), differences


def test_frozen_fingerprint_still_matches_this_machine():
    """The resume rule, enforced. Fails loudly when the environment drifts."""
    if os.environ.get("RLM_VALIDATE_FROZEN_ENV") != "1":
        pytest.skip("set RLM_VALIDATE_FROZEN_ENV=1 to validate the recorded machine")
    frozen_path = Path(__file__).resolve().parent.parent / "configs" / "fingerprint.json"
    if not frozen_path.exists():
        return
    frozen = json.loads(frozen_path.read_text())
    current = fp.environment_fingerprint()
    # The public snapshot redacts the machine-local interpreter path. Package
    # versions remain strict; the spelling of the local path is not evidence.
    server = frozen["environment"]["server_venv"]
    if server.get("python_path") == "${RLM_SERVER_PYTHON}":
        server["python_path"] = current["server_venv"]["python_path"]
    same, differences = fp.matches(
        {"environment": frozen["environment"]}, {"environment": current}
    )
    assert same, (
        "environment drifted from the frozen fingerprint; prior results are not "
        f"resumable: {differences}"
    )


# --- Phase 0.1: scorers that judge the trajectory, not just the reply -------
def test_a5_rejects_an_id_that_merely_contains_the_answer():
    """`"008" in "1008"` is True. A substring scorer passes a wrong row."""
    a5 = tasks.GATE_A[4]
    assert a5.scores("008") and a5.scores("8") and a5.scores("row 008")
    assert not a5.scores("1008"), "must not accept a longer id containing the answer"
    assert not a5.scores("009")


def test_a4_requires_the_session_to_have_persisted():
    """A correct 84 from a single call is a failed persistence test."""
    a4 = tasks.GATE_A[3]
    one_shot = [tasks.Step(code="print(len(context.splitlines()) * 7)")]
    assert a4.scores("84"), "the number itself is right"
    assert not a4.passed("84", one_shot), "but nothing persisted between calls"

    two_step = [
        tasks.Step(code="n = len(context.splitlines())", defined=frozenset({"n"})),
        tasks.Step(code="print(n * 7)", defined=frozenset({"n"})),
    ]
    assert a4.passed("84", two_step)


def test_a4_is_not_fooled_by_two_unrelated_calls():
    a4 = tasks.GATE_A[3]
    unrelated = [
        tasks.Step(code="x = 1", defined=frozenset({"x"})),
        tasks.Step(code="print(len(context.splitlines()) * 7)", defined=frozenset({"x"})),
    ]
    assert not a4.passed("84", unrelated)


def test_a5_requires_a_changed_action_after_the_refusal():
    a5 = tasks.GATE_A[4]
    assert a5.inject == "refuse_first_call_as_duplicate", "the runner must create the condition"

    repeated = [
        tasks.Step(code="print(rows)", refused=True),
        tasks.Step(code="print(rows)"),
    ]
    assert not a5.passed("008", repeated), "re-emitting the same block is the probe_08 failure"

    recovered = [
        tasks.Step(code="print(rows)", refused=True),
        tasks.Step(code="print(max(void, key=amount))"),
    ]
    assert a5.passed("008", recovered)


def test_only_the_two_process_tasks_carry_a_process_scorer():
    with_process = {t.id for t in tasks.GATE_A if t.scores_process}
    assert with_process == {"a4_persistence", "a5_duplicate_recovery"}


# --- Phase 0.1: run manifest -----------------------------------------------
def test_manifest_flags_a_model_swap():
    """The incident this field exists for: a 200 from a model we did not ask for."""
    from alchemist_rlm.manifest import RunManifest

    m = RunManifest(
        run_id="t", arm="alchemist", suite="gate_a", fingerprint_sha256="f",
        tasks_sha256=tasks.TASKS_SHA256, system_prompt_sha256="s",
        tool_schema_sha256="c", tool_name=protocol.TOOL_NAME,
        sampling={"temperature": 0.0, "enable_thinking": False},
    )
    m.note_request("agent-a1-alchemist-4bit", "agent-a1-alchemist-4bit")
    assert m.model_stayed_put

    m.note_request("agent-a1-alchemist-4bit", "Qwen3.5-9B-MLX-4bit")
    assert not m.model_stayed_put
    assert len(m.to_dict()["distinct_served_models"]) == 2


def test_manifest_records_git_and_is_serialisable():
    from alchemist_rlm.manifest import RunManifest

    m = RunManifest(
        run_id="t", arm="alchemist", suite="gate_a", fingerprint_sha256="f",
        tasks_sha256="t", system_prompt_sha256="s", tool_schema_sha256="c",
        tool_name=protocol.TOOL_NAME, sampling={},
    )
    record = m.to_dict()
    # `code_dirty` ignores the paths a run writes into; `dirty` is the whole
    # tree. The runners gate on the first, and both are recorded so a result can
    # be read either way afterwards.
    assert set(record["git"]) == {"commit", "branch", "dirty", "code_dirty",
                                  "uncommitted_code"}
    json.dumps(record)          # must survive serialisation
