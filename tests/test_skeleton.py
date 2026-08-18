"""Contracts of the walking skeleton, all with a fake backend.

Not one test here loads a model or spends an episode. That is the arrangement
the plan is built on: the twenty inference episodes are for questions only a
real model can answer, and everything mechanical — does a duplicate get
refused, does a generator stay lazy, does a child share its parent's budget —
is answered here in under a second.

Each test names the observed failure it guards against where there is one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from alchemist_rlm import protocol
from alchemist_rlm.artifacts import ArtifactStore
from alchemist_rlm.budgets import Budget, Ledger
from alchemist_rlm.calls.recursive import RecursionRefused, RecursiveCaller, signature
from alchemist_rlm.calls.scheduler import SubcallScheduler
from alchemist_rlm.context.search import literal_search
from alchemist_rlm.context.segmenter import covers, detect_structure, segment
from alchemist_rlm.context.store import ContextStore
from alchemist_rlm.engine import RLMEngine
from alchemist_rlm.mlx_client import ScriptedClient, text_reply, tool_reply
from alchemist_rlm.native_loop import NativeLoop, render
from alchemist_rlm.repl.runtime import ReplRuntime
from alchemist_rlm.tracing import Trace


# --- REPL -------------------------------------------------------------------
@pytest.fixture
def repl():
    with ReplRuntime(block_timeout=30.0) as runtime:
        yield runtime


def test_variables_survive_between_calls(repl):
    repl.execute("n = 6 * 7")
    result = repl.execute("print(n + 1)")
    assert result["ok"] and result["stdout"].strip() == "43"


def test_an_observation_says_more_than_stdout(repl):
    """`probe_11` re-ran the same conditional print three times because an empty
    stdout cannot distinguish "no matches" from "did not run"."""
    result = repl.execute("hits = [x for x in range(5) if x > 99]\nlen(hits)")
    assert result["stdout"] == ""
    assert result["value"] == "0"                    # the trailing expression
    assert "hits" in result["defined"]
    assert "list, 0 items" in result["changed"]["hits"]


def test_an_exception_is_an_observation_not_a_crash(repl):
    result = repl.execute("1 / 0")
    assert result["ok"] is False
    assert result["error"]["type"] == "ZeroDivisionError"
    assert "ZeroDivisionError" in render(result)


def test_presentation_only_window_allows_clean_local_inspection(repl):
    assert repl.execute("submit([['1', '2']])")["delivered"] is True
    repl.open_presentation(
        '[["1", "2"]]', {"line": "(lower_numeric_id, higher_numeric_id)"}
    )

    inspected = repl.execute("print(PRESENTATION_VALUE)")

    assert inspected["ok"] is True
    assert inspected["stdout"] == "[['1', '2']]\n"
    assert repl.presentation()["present"] is False


def test_clean_presentation_turn_must_use_a_persistent_answer_source(repl):
    assert repl.execute("submit([['1', '2']])")["delivered"] is True
    repl.open_presentation(
        '[["1", "2"]]', {"line": "(lower_numeric_id, higher_numeric_id)"}
    )

    refused = repl.execute("import sys\nprint(sys.version)")

    assert refused["ok"] is False
    assert refused["error"]["type"] == "PresentationSourceRequired"


def test_presentation_linter_is_read_only_audited_and_local(repl):
    spec = {
        "version": 1, "kind": "records", "record_separator": "newline",
        "prefix": "(", "suffix": ")", "field_separator": ", ",
        "fields": ["integer", "integer"], "ordering": "numeric_ascending",
        "duplicates": False, "allow_empty": True, "additional_text": False,
    }
    assert repl.execute("submit([['1', '2']])")["delivered"] is True
    repl.open_presentation(
        '[["1", "2"]]', {"line": "(lower_numeric_id, higher_numeric_id)"},
        "1, 2", inferred_spec=spec,
    )

    checked = repl.execute(
        "report = check_presentation(PRESENTATION_DRAFT)\nprint(report)"
    )

    assert checked["ok"] is True
    assert "record_delimiters" in checked["stdout"]
    assert repl.presentation()["present"] is False
    audit = repl.peek_audit()["audit"]["presentation_checks"]
    assert audit == [{
        "input_sha256": __import__("hashlib").sha256(b"1, 2").hexdigest(),
        "input_chars": 4,
        "valid": False,
        "issue_codes": ["record_delimiters"],
    }]

    repl.open_presentation(
        '[["1", "2"]]', {"line": "(lower_numeric_id, higher_numeric_id)"},
        "1, 2", inferred_spec=spec,
    )
    refused = repl.execute("check_presentation = lambda text: {'valid': True}")
    assert refused["ok"] is False
    assert refused["error"]["type"] == "PresentationSourceImmutable"


def test_presentation_renderer_is_local_audited_and_never_submits(repl):
    spec = {
        "version": 1, "kind": "records", "record_separator": "newline",
        "prefix": "(", "suffix": ")", "field_separator": ", ",
        "fields": ["integer", "integer"], "ordering": "numeric_ascending",
        "duplicates": False, "allow_empty": True, "additional_text": False,
    }
    assert repl.execute("submit([['9', '3']])")["delivered"] is True
    repl.open_presentation(
        '[["9", "3"]]', {"line": "(lower_numeric_id, higher_numeric_id)"},
        inferred_spec=spec,
    )

    rendered = repl.execute(
        "candidate = render_presentation(PRESENTATION_VALUE)\nprint(candidate)"
    )

    assert rendered["ok"] is True
    assert rendered["stdout"] == "(3, 9)\n\n"
    assert repl.presentation()["present"] is False
    audit = repl.peek_audit()["audit"]["presentation_renders"]
    assert len(audit) == 1 and audit[0]["ok"] is True

    repl.open_presentation(
        '[["9", "3"]]', {"line": "(lower_numeric_id, higher_numeric_id)"},
        inferred_spec=spec,
    )
    refused = repl.execute("render_presentation = lambda value: 'forged'")
    assert refused["ok"] is False
    assert refused["error"]["type"] == "PresentationSourceImmutable"


def test_preserved_renderer_result_is_immutable_and_survives_next_window(repl):
    spec = {
        "version": 1, "kind": "records", "record_separator": "newline",
        "prefix": "(", "suffix": ")", "field_separator": ", ",
        "fields": ["integer", "integer"], "ordering": "numeric_ascending",
        "duplicates": False, "allow_empty": True, "additional_text": False,
    }
    assert repl.execute("submit([['9', '3']])")["delivered"] is True
    repl.open_presentation("", {}, inferred_spec=spec)
    assert repl.execute(
        "candidate = render_presentation(PRESENTATION_VALUE)"
    )["ok"] is True

    repl.open_presentation("", {}, inferred_spec=spec, commit_required=True)
    refused = repl.execute("PRESENTATION_RENDERED = 'forged'")
    assert refused["ok"] is False
    assert refused["error"]["type"] == "PresentationSourceImmutable"

    repl.open_presentation("", {}, inferred_spec=spec, commit_required=True)
    accepted = repl.execute("submit(PRESENTATION_RENDERED)")
    assert accepted["ok"] is True
    assert accepted["presentation_candidate"] is True


def test_explicit_structured_presentation_submit_uses_only_frozen_renderer(repl):
    spec = {
        "version": 1, "kind": "records", "record_separator": "newline",
        "prefix": "(", "suffix": ")", "field_separator": ",",
        "fields": ["integer", "integer"], "ordering": "numeric_ascending",
        "duplicates": False, "allow_empty": True, "additional_text": False,
    }
    assert repl.execute("submit(['9, 3', '1,2'])")["delivered"] is True
    repl.open_presentation("", {}, inferred_spec=spec)

    accepted = repl.execute("submit(PRESENTATION_VALUE)")

    assert accepted["ok"] is True
    assert accepted["presentation_candidate"] is True
    assert repl.presentation()["text"] == "(3,9)\n(1,2)\n"
    audit = repl.peek_audit()["audit"]["presentation_renders"]
    assert len(audit) == 1 and audit[0]["ok"] is True


def test_structured_presentation_submit_without_frozen_spec_is_still_refused(repl):
    assert repl.execute("submit(['9, 3'])")["delivered"] is True
    repl.open_presentation("", {})

    refused = repl.execute("submit(PRESENTATION_VALUE)")

    assert refused["ok"] is False
    assert refused["error"]["type"] == "SubmitRefused"
    assert "presentation text must be str" in refused["error"]["message"]


def test_presentation_window_refuses_semantic_recomputation(repl):
    assert repl.execute("submit([['1', '2']])")["delivered"] is True
    repl.open_presentation(
        '[["1", "2"]]', {"line": "(lower_numeric_id, higher_numeric_id)"}
    )

    refused = repl.execute(
        "semantic_map('classify', {'type': 'string'}, items=['x'])"
    )

    assert refused["ok"] is False
    assert refused["error"]["type"] == "PresentationIsolationRequired"


def test_presentation_window_refuses_persistent_semantic_alias(repl):
    assert repl.execute(
        "mapper = semantic_map\nsubmit([['1', '2']])"
    )["delivered"] is True
    repl.open_presentation(
        '[["1", "2"]]', {"line": "(lower_numeric_id, higher_numeric_id)"}
    )

    refused = repl.execute(
        "source = PRESENTATION_VALUE\n"
        "mapper('classify', {'type': 'string'}, items=['x'])"
    )

    assert refused["ok"] is False
    assert refused["error"]["type"] == "SubmitRefused"


def test_valid_presentation_draft_requires_exact_explicit_commit(repl):
    assert repl.execute("submit([['1', '2']])")["delivered"] is True
    repl.open_presentation(
        '[["1", "2"]]', {"line": "(lower_numeric_id, higher_numeric_id)"},
        "(1, 2)\n", commit_required=True, draft_ready=True,
    )

    refused = repl.execute("submit('(1, 2)')")

    assert refused["ok"] is False
    assert refused["error"]["type"] == "PresentationDraftCommitRequired"


def test_candidate_alias_can_confirm_exact_valid_presentation_draft(repl):
    assert repl.execute("submit([['1', '2']])")["delivered"] is True
    repl.open_presentation(
        '[["1", "2"]]', {"line": "(lower_numeric_id, higher_numeric_id)"},
        "(1, 2)\n", commit_required=True, draft_ready=True,
    )

    accepted = repl.execute("submit(candidate=PRESENTATION_DRAFT)")

    assert accepted["ok"] is True
    assert accepted["presentation_candidate"] is True


def test_candidate_alias_is_refused_before_value_commit(repl):
    refused = repl.execute("submit(candidate='not a computed value')")

    assert refused["ok"] is False
    assert refused["error"]["type"] == "SubmitRefused"
    assert "only after the answer value is committed" in \
        refused["error"]["message"]


def test_a_hung_block_is_killed_and_the_context_comes_back(repl):
    repl.bind_context("the context survives a restart")
    repl.execute("keeper = 1")
    result = repl.execute("while True: pass", timeout=1.0)
    assert result["error"]["type"] == "BlockTimeout"
    assert repl.restarts == 1
    assert repl.peek("keeper")["present"] is False           # honestly gone
    assert repl.peek("context")["value"] == "the context survives a restart"


def test_writing_to_real_stdout_cannot_corrupt_the_channel(repl):
    """The worker duplicates fd 1 for RPC and points fd 1 at stderr, so a
    subprocess or C library that writes to the real stdout cannot desync us."""
    repl.execute("import os; os.write(1, b'raw bytes on fd 1\\n')")
    assert repl.execute("2 + 2")["value"] == "4"


# --- the lazy argument protocol ---------------------------------------------
def test_a_generator_never_materialises_to_cross_the_channel(repl):
    """The memory wall: `_handle_batched` built one task per prompt up front,
    so 3,182 prompts became 3,182 live objects before the first one ran."""
    seen: list[int] = []

    def consume(jobs):
        for job in jobs:
            seen.append(job)
            if len(seen) == 5:
                break
        return len(seen)

    repl.handlers["llm_query_batched"] = consume
    result = repl.execute(
        "n = llm_query_batched(i for i in range(1_000_000))\nprint(n)"
    )
    assert result["ok"], result["error"]
    assert result["stdout"].strip() == "5"
    # Bounded pulls, not a million items: one chunk of 16 covered five reads.
    assert len(seen) == 5


def test_a_list_argument_is_passed_through_untouched(repl):
    repl.handlers["llm_query_batched"] = lambda jobs: list(jobs)
    result = repl.execute("llm_query_batched(['a', 'b'])")
    assert result["value"] == "list, 2 items"


def test_a_lazy_stream_refuses_a_second_pass(repl):
    def twice(jobs):
        first = list(jobs)
        return len(first) + len(list(jobs))

    repl.handlers["llm_query_batched"] = twice
    result = repl.execute("llm_query_batched(i for i in range(3))")
    assert result["ok"] is False
    assert "already been consumed" in result["error"]["message"]


# --- segmentation -----------------------------------------------------------
def test_segments_tile_the_text_exactly():
    """Coverage is what makes atomic attribution possible: if a segment holding
    the evidence was never handed to a subcall, the failure is retrieval."""
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 60 for i in range(40))
    segments = segment(text, target_chars=1200)
    assert covers(segments, text)
    assert "".join(s.text(text) for s in segments) == text


def test_an_oversized_unit_is_split_rather_than_dropped():
    text = "x" * 30_000
    segments = segment(text, target_chars=4_000, max_chars=8_000)
    assert covers(segments, text)
    assert max(s.chars for s in segments) <= 8_000


def test_structure_is_detected_not_assumed():
    rows = "\n".join(f"{i:03d} | dept | {i * 10} | paid" for i in range(30))
    assert detect_structure(rows) == "rows"
    assert detect_structure("# Title\n\ntext\n\n## Next\n\nmore") == "sections"
    assert detect_structure("def f():\n    return 1\n\ndef g():\n    return 2") == "code"


def test_a_bad_band_is_refused_loudly():
    with pytest.raises(ValueError, match="min <= target <= max"):
        segment("abc", target_chars=100, min_chars=200)


# --- search -----------------------------------------------------------------
def test_zero_matches_is_a_statement_not_an_empty_string():
    store = ContextStore(text="alpha beta gamma " * 100)
    found = literal_search(store, "Aurelio Vance")
    assert found["matches"] == 0
    assert found["searched_chars"] == store.chars
    assert any("semantic" in action for action in found["next_actions"])


def test_hits_carry_a_segment_ref_so_they_can_be_reopened():
    store = ContextStore(text="\n".join(f"line {i}" for i in range(500)))
    found = literal_search(store, "line 431")
    assert found["matches"] == 1
    assert found["hits"][0]["ref"].startswith("s")
    assert store.read(found["hits"][0]["ref"])



def test_a_child_spends_its_parents_ledger():
    """A fresh budget per node would let depth 2 with fan-out 8 licence nine
    times the run it was supposed to bound."""
    parent = Budget(max_turns=8, ledger=Ledger())
    child = parent.child()
    assert child.ledger is parent.ledger
    assert child.max_turns == 4 and child.depth == 1
    child.spend_turn()
    assert parent.ledger.turns == 1


def test_depth_and_node_count_both_stop_recursion():
    budget = Budget(max_depth=2, max_nodes=3)
    assert budget.may_recurse() is None
    assert budget.child().child().may_recurse() == "max_depth"
    deep = Budget(max_depth=9, max_nodes=2)
    deep.child()
    assert deep.may_recurse() == "max_nodes"


# --- recursion --------------------------------------------------------------
def test_recursion_refuses_a_cycle_by_content():
    text = "the same bytes"
    mark = signature("q", text)
    caller = RecursiveCaller(spawn=lambda **_: "never", budget=Budget(), lineage=(mark,))
    with pytest.raises(RecursionRefused, match="already being answered"):
        caller("q", text)


def test_recursion_requires_an_explicit_context():
    caller = RecursiveCaller(spawn=lambda **_: "x", budget=Budget())
    with pytest.raises(ValueError, match="explicit `context`"):
        caller("what is in there?")


# --- subcalls ---------------------------------------------------------------
def test_a_subcall_without_a_source_is_refused():
    """Every fabricated answer in this project came from a model answering
    about text it did not have in front of it."""
    scheduler = SubcallScheduler(client=ScriptedClient([]), budget=Budget())
    with pytest.raises(ValueError, match="non-empty `source`"):
        scheduler.query("count the rows")


def test_batched_results_stay_in_order(tmp_path):
    client = ScriptedClient([text_reply(f"answer {i}") for i in range(6)])
    scheduler = SubcallScheduler(client=client, budget=Budget(max_in_flight=3))
    jobs = ({"instruction": "read", "source": f"chunk {i}"} for i in range(6))
    assert scheduler.query_batched(jobs) == [f"answer {i}" for i in range(6)]


def test_the_trace_keeps_the_source_so_the_atom_can_be_judged(tmp_path):
    """The plan scores atomic failure from the same episode: if the subcall
    received the right chunk and still answered wrong, the atom failed."""
    trace = Trace(tmp_path / "t.jsonl", run_id="t")
    scheduler = SubcallScheduler(
        client=ScriptedClient([text_reply("Casimir dolan")]), budget=Budget(), trace=trace,
    )
    scheduler.query("who signed it?", "Entry 122: signed by Aurelio Vance.", source_ref="s0004")
    trace.finish(reason="test", answer=None, ledger={})
    record = next(e for e in Trace.read(tmp_path / "t.jsonl") if e["kind"] == "subcall")
    assert record["source_ref"] == "s0004"
    assert "Aurelio Vance" in record["source"]["preview"]
    assert record["response"] == "Casimir dolan"


# --- the loop ---------------------------------------------------------------
def loop_with(replies, **kwargs):
    executed: list[str] = []

    def execute(code):
        executed.append(code)
        return {"ok": True, "stdout": f"ran {len(executed)}", "defined": ["x"],
                "changed": {"x": "int"}, "value": None, "stderr": "", "truncated": False}

    loop = NativeLoop(client=ScriptedClient(list(replies)), execute=execute,
                      budget=kwargs.pop("budget", Budget()), **kwargs)
    return loop, executed


def test_the_loop_sends_the_tool_schema():
    """The whole finding: the earlier evaluation never sent `tools`, so the
    tool-calling branch of the template never activated."""
    loop, _ = loop_with([text_reply("<answer>Paris</answer>")])
    loop.run("capital of France?")
    sent = loop.client.calls[0]
    assert sent["tools"][0]["function"]["name"] == "PythonInterpreter"
    assert "code" in sent["tools"][0]["function"]["parameters"]["properties"]


def test_answer_tag_closes_the_episode():
    loop, executed = loop_with([tool_reply("print(1)"), text_reply("<answer>48415447</answer>")])
    result = loop.run("7919 * 6113?")
    assert result.answer == "48415447" and result.stop_reason == "answer_tag"
    assert executed == ["print(1)"]


def test_a_turn_with_no_tool_call_is_terminal():
    """The checkpoint's README warns a loop waiting only for the tag will
    "spin past a perfectly good reply"."""
    loop, _ = loop_with([text_reply("Paris.")])
    result = loop.run("capital of France?")
    assert result.stop_reason == "no_tool_call" and result.answer == "Paris."


def test_a_truncated_turn_is_not_mistaken_for_an_answer():
    loop, _ = loop_with([
        text_reply("I will start by count", finish_reason="length"),
        text_reply("<answer>12</answer>"),
    ])
    result = loop.run("how many rows?")
    assert result.answer == "12"
    assert result.protocol_errors[0]["kind"] == "truncated_generation"


def test_a_repeated_block_runs_again_and_is_told_that_it_repeated():
    """A repeated block runs and is told it repeated.

    It used to be refused, and the refusal said "its result is unchanged". The
    session has state, so the harness cannot know that: `xs.append(3)` between
    two identical `print(len(xs))` makes the second one right and different.
    Executing costs one call and reports something true — here is what it does
    *now* — instead of asserting something the harness cannot check."""
    loop, executed = loop_with([
        tool_reply("print(len(context))"),
        tool_reply("print(len(context))"),
        text_reply("<answer>done</answer>"),
    ])
    result = loop.run("how long?")
    assert executed == ["print(len(context))"] * 2              # ran both times
    assert result.duplicates_observed == 1
    shown = json.loads(json.dumps(loop.client.calls[2]["messages"][-1]))
    assert "same code as turn 1" in shown["content"]
    # And the two claims the old refusal made are gone from what the model reads.
    assert "unchanged" not in shown["content"]
    assert "was not run again" not in shown["content"]


def test_a_repeat_that_differs_only_in_a_comment_is_still_a_duplicate():
    """A smoke episode re-ran a failing block whose only difference was the
    comment above it: same code, same TypeError, and the check let it through."""
    assert protocol.code_key("n = 1\nFinal(n)") == protocol.code_key(
        "# Store the number of rows\nn = 1\nFinal(n)  # answer"
    )
    assert protocol.code_key("n = 1") != protocol.code_key("n = 2")
    # A `#` inside a string is not a comment.
    assert protocol.code_key("x = '# not a comment'") != protocol.code_key("x = ''")


def test_a_refusal_replays_the_failure_not_a_clean_stdout():
    """The block that raised had a clean stdout right up to the exception."""
    def execute(code):
        return {"ok": False, "stdout": "Number of rows (n): 12\n", "stderr": "",
                "value": None, "defined": [], "changed": {}, "truncated": False,
                "error": {"type": "TypeError",
                          "message": "'NoneType' object is not callable",
                          "traceback": ""}}

    loop = NativeLoop(client=ScriptedClient([
        tool_reply("n = 12\nFinal(n)"),
        tool_reply("# retry\nn = 12\nFinal(n)"),
        text_reply("<answer>12</answer>"),
    ]), execute=execute, budget=Budget())
    result = loop.run("how many?")
    assert result.duplicates_observed == 1
    replay = loop.client.calls[2]["messages"][-1]["content"]
    assert "TypeError" in replay and "not callable" in replay


def test_an_unknown_tool_is_answered_never_dropped():
    """This checkpoint was trained with `google_search` alongside the
    interpreter; a filtered-out call would look like a deliberate direct reply."""
    loop, _ = loop_with([
        tool_reply("q", name="google_search"),
        text_reply("<answer>ok</answer>"),
    ])
    result = loop.run("who?")
    assert result.protocol_errors[0]["kind"] == "unknown_tool"
    observation = loop.client.calls[1]["messages"][-1]["content"]
    assert "no tool named 'google_search'" in observation


def test_only_the_first_call_of_a_turn_runs():
    two = tool_reply("print(1)")
    two.tool_calls.append({"id": "call_1", "type": "function",
                           "function": {"name": "PythonInterpreter",
                                        "arguments": json.dumps({"code": "print(2)"})}})
    loop, executed = loop_with([two, text_reply("<answer>ok</answer>")])
    result = loop.run("q")
    assert executed == ["print(1)"]
    assert result.protocol_errors[0]["kind"] == "multiple_calls"


def test_the_injected_duplicate_makes_the_recovery_task_testable():
    """Waiting for the model to repeat itself by chance would make `a5`
    untestable on the runs where it simply does not.

    The key still says "refuse" because it is written into a frozen V1 suite
    whose hash is pinned; what it injects is now the repeat *condition*, and
    the block runs like any other."""
    loop, executed = loop_with(
        [tool_reply("first()"), tool_reply("second()"), text_reply("<answer>008</answer>")],
        inject="refuse_first_call_as_duplicate",
    )
    result = loop.run("largest void row?")
    assert executed == ["first()", "second()"]          # marked, not withheld
    assert result.duplicates_observed == 1
    assert [s.refused for s in result.steps] == [False, False]


def test_the_budget_forces_one_final_turn_instead_of_stopping_empty():
    """A run that ends with no answer teaches nothing; 919.9 seconds of that is
    already on record."""
    loop, _ = loop_with(
        [tool_reply("print(1)"), tool_reply("print(2)"), text_reply("<answer>partial</answer>")],
        budget=Budget(max_turns=2),
    )
    result = loop.run("q")
    assert result.answer == "partial"
    assert result.stop_reason == "forced_final:max_turns"
    # Tools ARE offered on the forced turn, and this line used to assert the
    # opposite. `FORCE_FINAL` asks the model to deliver with `submit(...)` —
    # a tool call — while the request withheld the tools, so the turn demanded
    # the one interface it had just removed. A model that cannot call anything
    # writes prose instead, and seven of ten episodes whose stop_reason carries
    # `truncated` are exactly that shape: clean turns throughout, then a single
    # truncated final full of narration.
    assert loop.client.calls[-1]["tools"] is not None


def test_a_submitted_value_ends_the_episode():
    loop, _ = loop_with([tool_reply("submit('from python')")])
    loop.read_submission = lambda: (True, "from python")
    result = loop.run("q")
    assert result.stop_reason == "submitted" and result.answer == "from python"


# --- artifacts --------------------------------------------------------------
def test_final_may_be_an_artifact_larger_than_the_window(tmp_path):
    store = ArtifactStore(tmp_path)
    ref = store.save("report", "x" * 50_000)
    assert ref == "artifact://report"
    assert len(store.resolve(ref)) == 50_000


def test_loading_a_structured_artifact_restores_its_python_type(tmp_path):
    store = ArtifactStore(tmp_path)
    rows = [{"item": 0, "value": "entity"}]
    ref = store.save("rows", rows)

    assert isinstance(store.load(ref), str)       # disk remains plain, auditable JSON
    assert store.load_value(ref) == rows          # live REPL gets the saved value
    assert store.manifest()[0]["kind"] == "json"


def test_the_repl_loads_a_structured_artifact_as_a_value(tmp_path):
    client = ScriptedClient([
        tool_reply("ref = save_artifact('rows', [{'item': 7}])\n"
                   "loaded = load_artifact(ref)\n"
                   "submit(f\"{type(loaded).__name__}:{loaded[0]['item']}\")"),
    ])
    episode = RLMEngine(client=client, runs_dir=tmp_path).complete(
        "", "round trip rows", run_id="ep_typed_artifact")

    assert episode.answer == "list:7"
    assert episode.artifacts[0]["kind"] == "json"


def test_semantic_cache_hits_survive_in_the_episode_record(tmp_path):
    client = ScriptedClient([
        tool_reply(
            "a = semantic_map('classify', {'type': 'string', 'enum': ['no']})\n"
            "b = semantic_map('classify', {'enum': ['no'], 'type': 'string'})\n"
            "submit(f\"{b['cache_hit']}:{len(b['rows'])}\")"),
        text_reply("item 0: no\nitem 1: no\nitem 2: no"),
    ])
    episode = RLMEngine(client=client, runs_dir=tmp_path).complete(
        "a\nb\nc", "classify", run_id="ep_semantic_cache")

    assert episode.answer == "True:3"
    assert episode.ledger["subcalls"] == 1
    assert len(episode.sweeps) == 1
    assert len(episode.semantic_cache_hits) == 1
    written = json.loads((tmp_path / "ep_semantic_cache" / "episode.json").read_text())
    assert len(written["semantic_cache_hits"]) == 1


def test_an_artifact_name_cannot_escape_the_run_directory(tmp_path):
    store = ArtifactStore(tmp_path)
    with pytest.raises(ValueError, match="path separator"):
        store.save("../../etc/passwd", "nope")


# --- end to end -------------------------------------------------------------
def test_a_whole_episode_runs_against_a_scripted_model(tmp_path):
    """Loop, REPL, context, subcall, artifact and trace, wired together, with
    no model and no episode spent."""
    ledger = "\n".join(f"{i:03d} | dept | {i * 100} | paid" for i in range(1, 13))
    client = ScriptedClient([
        tool_reply("rows = context.splitlines()\nprint(len(rows))"),
        tool_reply("verdict = llm_query('is this a ledger? yes or no', read_context('s0000'))\n"
                   "submit(f'{len(rows)} rows, {verdict}')"),
        text_reply("yes"),          # consumed by the subcall inside turn 2's code
    ])
    engine = RLMEngine(client=client, budget=Budget(max_turns=4), runs_dir=tmp_path)
    episode = engine.complete(ledger, "How many rows are there?", run_id="ep_test")

    assert episode.stop_reason == "submitted"
    assert episode.answer == "12 rows, yes"
    assert episode.context_manifest["structure"] == "rows"
    assert episode.ledger["subcalls"] == 1

    events = Trace.read(episode.trace_path)
    kinds = [e["kind"] for e in events]
    assert kinds[0] == "run_start" and kinds[-1] == "run_end"
    assert "subcall" in kinds and "tool_call" in kinds and "model_turn" in kinds
    # The defining property: the context never enters the *root's* history. It
    # does reach subcalls, which is the point — those are given an explicit
    # slice as `source`, and the trace records which one.
    root_calls = [c for c in client.calls
                  if c["messages"][0]["content"].startswith("You answer questions")]
    assert len(root_calls) == 2
    for call in root_calls:
        for message in call["messages"]:
            assert ledger not in (message.get("content") or "")
    subcall = next(e for e in events if e["kind"] == "subcall")
    assert subcall["source"]["chars"] == len(ledger)


def test_the_context_line_tells_the_model_what_it_cannot_see(tmp_path):
    client = ScriptedClient([text_reply("<answer>ok</answer>")])
    engine = RLMEngine(client=client, runs_dir=tmp_path)
    engine.complete("a" * 5000, "anything?", run_id="ep_line")
    opening = client.calls[0]["messages"][1]["content"]
    assert "5,000 characters" in opening
    assert "NOT in this conversation" in opening
    assert "read_context(ref_or_start, end=None)" in opening
    # Naming a helper is not the same as saying where it lives: smoke_2 spent
    # four of seven turns on `from context import read_context` before finding
    # the names with dir().
    assert "not a module" in opening and "ALREADY DEFINED" in opening
    assert "do not import" in opening


def test_a_child_rlm_gets_its_own_repl_and_the_same_ledger(tmp_path):
    """The line between a RLM and a retrieval agent: `rlm_query` reaches a root
    that can partition its own slice again, not just an LM."""
    client = ScriptedClient([
        # root turn 1: delegate the first half to a child
        tool_reply("half = '\\n'.join(context.splitlines()[:20])\n"
                   "submit(rlm_query('how many lines?', half))"),
        # the child's own turn 1, in its own REPL
        tool_reply("submit(str(len(context.splitlines())))"),
    ])
    engine = RLMEngine(client=client, budget=Budget(max_turns=4, max_depth=2),
                       runs_dir=tmp_path)
    text = "\n".join(f"line {i}" for i in range(40))
    episode = engine.complete(text, "count them", run_id="ep_rec")

    assert episode.answer == "20"
    assert episode.recursions[0]["depth"] == 1
    assert episode.recursions[0]["question"] == "how many lines?"
    assert episode.ledger["nodes"] == 2
    assert episode.ledger["turns"] == 2          # one ledger, both nodes
    events = [e["kind"] for e in Trace.read(episode.trace_path)]
    assert "rlm_query" in events and "rlm_query_return" in events


def test_recursion_is_refused_at_the_depth_limit(tmp_path):
    client = ScriptedClient([
        tool_reply("try:\n    rlm_query('again', context)\nexcept Exception as e:\n"
                   "    submit(f'refused: {e}')"),
    ])
    engine = RLMEngine(client=client, budget=Budget(max_depth=0), runs_dir=tmp_path)
    episode = engine.complete("some text", "q", run_id="ep_depth")
    assert episode.answer.startswith("refused:")
    assert "max_depth" in episode.answer
    assert episode.recursions == []


def test_an_answer_too_large_to_write_is_returned_as_an_artifact(tmp_path):
    client = ScriptedClient([
        tool_reply("body = 'x' * 40000\nsubmit(save_artifact('report', body))"),
    ])
    engine = RLMEngine(client=client, runs_dir=tmp_path)
    episode = engine.complete("", "write the report", run_id="ep_art")
    assert len(episode.answer) == 40_000        # resolved, not the ref string
    assert episode.artifacts[0]["name"] == "report"
    assert (tmp_path / "ep_art" / "artifacts" / "report.txt").exists()


def test_reasoning_is_captured_not_dropped(tmp_path):
    """`server.py:1345` emits `reasoning` in its own field and the official
    client reads only `content`, so it never reached the logger."""
    client = ScriptedClient([text_reply("<answer>4</answer>", reasoning="two plus two")])
    engine = RLMEngine(client=client, runs_dir=tmp_path)
    episode = engine.complete("", "2+2?", run_id="ep_reason")
    turn = next(e for e in Trace.read(episode.trace_path) if e["kind"] == "model_turn")
    assert turn["reasoning"] == "two plus two"


# --- telemetry and scoring honesty -----------------------------------------
def test_the_episode_and_its_ledger_agree_on_the_subcall_count(tmp_path):
    """These disagreed: the episode reported 0 while its own ledger and trace
    both showed 1, because the field was built empty and never filled."""
    client = ScriptedClient([
        tool_reply("submit(llm_query('what is this?', context))"),
        text_reply("a ledger"),
    ])
    row = "001 | logistics | 100 | paid"
    engine = RLMEngine(client=client, runs_dir=tmp_path)
    episode = engine.complete(row, "what is it?", run_id="ep_tel")
    record = episode.to_dict()
    assert record["subcalls"] == episode.ledger["subcalls"] == 1
    assert record["subcall_detail"][0]["source_chars"] == len(row)
    traced = [e for e in Trace.read(episode.trace_path) if e["kind"] == "subcall"]
    assert len(traced) == record["subcalls"]


def test_the_smoke_verdict_reads_the_checks_it_declares():
    """An earlier verdict passed on "a tool was called and an answer exists",
    so smoke_2 was reported as passing while its record said
    reused_a_variable: false. A check the verdict does not read is decoration."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "smoke", Path(__file__).resolve().parent.parent / "scripts" / "smoke.py")
    smoke = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smoke)

    requires = smoke.SMOKE[1]["requires"]
    assert requires["reused_a_variable"] is True
    result = {"id": "smoke_2", "used_the_tool": True, "integrated_observation": True,
              "answer_correct": True, "reused_a_variable": False,
              "made_a_sourced_subcall": True}
    result["unmet"] = smoke.unmet(result, requires)
    assert result["unmet"] == ["reused_a_variable"]
    assert smoke.verdict([result]).startswith("PARTIAL")

    result.update(reused_a_variable=True, unmet=[])
    assert smoke.verdict([result]).startswith("GO (electrical)")


def test_the_block_clock_stops_while_the_parent_works(repl):
    """The first recursion episode died to this: seven children ran for minutes
    inside one rlm_map host call, the block timer counted that as runaway model
    code, and killed the worker mid-conversation. The clock now measures each
    stretch of pure Python between host calls, never the harness's own work."""
    import time

    def slow_handler(*_a, **_k):
        time.sleep(1.6)                     # far beyond the 1s block limit
        return "took a while"

    repl.handlers["llm_query"] = slow_handler
    result = repl.execute("r = llm_query('q', 'src')\nprint(r)", timeout=1.0)
    assert result["ok"], result.get("error")
    assert "took a while" in result["stdout"]
    assert repl.restarts == 0               # nobody was killed

    # And genuinely hung model code still dies on schedule.
    result = repl.execute("while True: pass", timeout=1.0)
    assert result["error"]["type"] == "BlockTimeout"
    assert repl.restarts == 1
