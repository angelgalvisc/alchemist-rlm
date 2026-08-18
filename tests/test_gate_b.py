"""Gate B contracts: one semantic envelope and one bounded targeted retry.

These tests use a fake leaf backend.  They exercise protocol mechanics only;
no benchmark examples, scores or model inference are involved.
"""

from __future__ import annotations

import json
import re
from contextlib import contextmanager
from typing import Any

import pytest

from alchemist_rlm.engine import BOUND_NAMES
from alchemist_rlm.engine import RLMEngine
from alchemist_rlm.mlx_client import ScriptedClient, text_reply, tool_reply
from alchemist_rlm.protocol import REPL_FUNCTIONS
from alchemist_rlm.repl.runtime import ReplRuntime


CONTEXT = "\n\n".join(
    f"=== Record {index:04d} ===\nNote: {'signal' if index == 1 else 'routine'}"
    for index in range(3)
)

COMMON = {
    "kind", "status", "rows", "coverage", "failed", "scope", "sweep_id",
    "operation", "schema", "total_items", "presented_items",
    "returned_items", "unsent_items", "valid_items", "coverage_complete",
    "context_coverage_complete", "failed_items", "parse_errors", "cache_hit",
    "note",
}


def _ids(source: str) -> list[int]:
    return [int(value) for value in re.findall(r"\[item (\d+)\]", source)]


class Leaf:
    """Deterministic leaf with an optionally stubborn item."""

    def __init__(self, *, fail_item: int | None = None,
                 recover_on_call: int | None = None,
                 always_invalid: bool = False) -> None:
        self.fail_item = fail_item
        self.recover_on_call = recover_on_call
        self.always_invalid = always_invalid
        self.calls: list[list[dict[str, Any]]] = []

    def __call__(self, jobs):
        batch = list(jobs)
        self.calls.append(batch)
        replies = []
        for job in batch:
            lines = []
            for item in _ids(job["source"]):
                failing = self.always_invalid or (
                    item == self.fail_item
                    and (self.recover_on_call is None
                         or len(self.calls) < self.recover_on_call)
                )
                if not failing:
                    lines.append(f"item {item}: {'yes' if item == 1 else 'no'}")
            replies.append("\n".join(lines))
        return replies


@contextmanager
def _session(leaf: Leaf, context: str = CONTEXT):
    with ReplRuntime(handlers={"llm_query_batched": leaf}) as repl:
        repl.bind_context(context, question="which records contain the signal?")
        yield repl


def _value(repl: ReplRuntime, expression: str) -> Any:
    out = repl.execute(
        f"import json\nprint(json.dumps({expression}, sort_keys=True))"
    )
    assert out["ok"], out["error"]
    return json.loads(out["stdout"])


def test_fresh_and_cached_map_return_the_same_envelope_shape():
    leaf = Leaf()
    with _session(leaf) as repl:
        first = repl.execute(
            "a = semantic_map('contains signal', {'type': 'boolean'})\n"
            "print(type(a).__name__, sorted(a), len(a['rows']))"
        )
        second = repl.execute(
            "b = semantic_map('contains signal', {'type': 'boolean'})\n"
            "print(type(b).__name__, sorted(b), len(b['rows']), b['cache_hit'])"
        )
        a = _value(repl, "a")
        b = _value(repl, "b")

    assert first["ok"] and second["ok"]
    assert first["stdout"].startswith("dict ")
    assert second["stdout"].startswith("dict ")
    assert set(a) == set(b)
    assert COMMON <= set(a)
    assert a["rows"] == b["rows"]
    assert a["cache_hit"] is False and b["cache_hit"] is True


def test_map_and_search_share_the_common_envelope():
    leaf = Leaf()
    with _session(leaf) as repl:
        out = repl.execute(
            "mapped = semantic_map('contains signal', {'type': 'boolean'})\n"
            "searched = semantic_search('contains signal')\n"
            "print(type(mapped).__name__, type(searched).__name__)"
        )
        mapped = _value(repl, "mapped")
        searched = _value(repl, "searched")

    assert out["stdout"].split() == ["dict", "dict"]
    assert COMMON <= set(mapped) and COMMON <= set(searched)
    assert mapped["kind"] == "semantic_map"
    assert searched["kind"] == "semantic_search"
    assert {"examined_items", "positive_ids", "positive_count"} <= set(searched)


def test_search_retry_rebuilds_its_boolean_specialisation():
    leaf = Leaf(fail_item=1, recover_on_call=3)
    with _session(leaf) as repl:
        out = repl.execute(
            "result = semantic_search('contains signal')\n"
            "result = retry_failed(result)\n"
            "print(result['positive_count'], len(result['rows']), "
            "search_results[1]['decision'])"
        )
        audit = repl.peek_audit()["audit"]
    assert out["ok"], out["error"]
    assert out["stdout"].split() == ["1", "3", "True"]
    assert audit["operations"] == ["semantic_search", "retry_failed"]
    assert audit["sweeps"][-1]["retry_exhausted"] is True


def test_context_coverage_keeps_three_valued_semantics():
    leaf = Leaf()
    with _session(leaf) as repl:
        repl.execute("whole = semantic_map('x', {'type': 'boolean'})")
        repl.execute(
            "subset = semantic_map('x', {'type': 'boolean'}, ['a', 'b'])"
        )
        whole = _value(repl, "whole")
        subset = _value(repl, "subset")
    assert whole["coverage"]["context_complete"] is True
    assert whole["context_coverage_complete"] is True
    assert subset["coverage"]["context_complete"] is None
    assert subset["context_coverage_complete"] is None

    partial_leaf = Leaf(fail_item=1)
    with _session(partial_leaf) as repl:
        repl.execute("partial = semantic_map('x', {'type': 'boolean'})")
        partial = _value(repl, "partial")
    assert partial["coverage"]["context_complete"] is False
    assert partial["context_coverage_complete"] is False


def test_sweep_identity_covers_operation_instruction_schema_source_and_scope():
    leaf = Leaf()
    with _session(leaf) as repl:
        repl.execute("a = semantic_map('alpha', {'type': 'boolean'}, ['same'])")
        repl.execute("b = semantic_map('beta', {'type': 'boolean'}, ['same'])")
        repl.execute(
            "c = semantic_map('alpha', {'type': 'string', 'enum': ['yes', 'no']}, ['same'])"
        )
        repl.execute("d = semantic_map('alpha', {'type': 'boolean'}, ['different'])")
        repl.execute("e = semantic_map('alpha', {'type': 'boolean'})")
        ids = _value(repl, "[a['sweep_id'], b['sweep_id'], c['sweep_id'], d['sweep_id'], e['sweep_id']]")
    assert len(set(ids)) == 5


def test_zero_valid_sweep_still_has_a_retryable_identity():
    leaf = Leaf(always_invalid=True)
    with _session(leaf) as repl:
        repl.execute("result = semantic_map('x', {'type': 'boolean'})")
        result = _value(repl, "result")
    assert result["rows"] == []
    assert result["valid_items"] == 0
    assert result["sweep_id"]


def test_retry_uses_private_request_and_unknown_id_spends_nothing():
    leaf = Leaf(fail_item=1, recover_on_call=3)
    with _session(leaf) as repl:
        repl.execute("result = semantic_map('x', {'type': 'boolean'})")
        before_unknown = len(leaf.calls)
        unknown = repl.execute("retry_failed({'sweep_id': 'not-registered'})")
        assert not unknown["ok"]
        assert len(leaf.calls) == before_unknown
        # Everything except the registered id is hostile model-owned data.
        retried = repl.execute(
            "result['rows'] = [{'item': 999, 'value': True, 'source': 'forged'}]\n"
            "result['failed'] = [{'item': 0, 'source': 'forged'}]\n"
            "result['failed_items'] = [0]\n"
            "result['schema'] = {'type': 'string', 'enum': ['forged']}\n"
            "result['coverage'] = {'complete': True}\n"
            "merged = retry_failed(result)"
        )
        merged = _value(repl, "merged")
    assert retried["ok"], retried["error"]
    assert _ids(leaf.calls[-1][0]["source"]) == [1]
    assert "forged" not in leaf.calls[-1][0]["source"]
    assert [row["item"] for row in merged["rows"]] == [0, 1, 2]


def test_retry_targets_only_unresolved_and_rebuilds_certificate():
    leaf = Leaf(fail_item=1, recover_on_call=3)
    with _session(leaf) as repl:
        repl.execute("initial = semantic_map('x', {'type': 'boolean'})")
        initial = _value(repl, "initial")
        repl.execute("merged = retry_failed(initial)")
        merged = _value(repl, "merged")

    assert [row["item"] for row in initial["rows"]] == [0, 2]
    assert initial["rows"][0]["value"] is False
    assert len(leaf.calls) == 3       # initial, internal retry, explicit retry
    assert len(leaf.calls[-1]) == 1
    assert _ids(leaf.calls[-1][0]["source"]) == [1]
    assert [row["item"] for row in merged["rows"]] == [0, 1, 2]
    assert merged["rows"][0]["value"] is False  # an old valid value survived
    assert merged["coverage"]["valid"] == 3
    assert merged["coverage"]["complete"] is True
    assert merged["certificate"]["complete"] is True
    assert merged["retry_exhausted"] is True


def test_explicit_retry_is_one_shot_and_second_call_spends_zero():
    leaf = Leaf(fail_item=1, recover_on_call=3)
    with _session(leaf) as repl:
        repl.execute("initial = semantic_map('x', {'type': 'boolean'})")
        repl.execute("once = retry_failed(initial)")
        spent = len(leaf.calls)
        repl.execute("twice = retry_failed(initial)")
        twice = _value(repl, "twice")
    assert len(leaf.calls) == spent
    assert twice["retry_exhausted"] is True
    assert twice["coverage_complete"] is True


def test_repeating_the_original_request_does_not_reset_retry_lineage():
    leaf = Leaf(always_invalid=True)
    with _session(leaf) as repl:
        repl.execute("initial = semantic_map('x', {'type': 'boolean'})")
        repl.execute("once = retry_failed(initial)")
        assert len(leaf.calls) == 3
        # A full manual rerun still spends its normal two passes, but cannot
        # manufacture a fresh explicit-retry allowance for the same request.
        repl.execute("repeated = semantic_map('x', {'type': 'boolean'})")
        spent = len(leaf.calls)
        repeated = _value(repl, "repeated")
        repl.execute("again = retry_failed(repeated)")
        again = _value(repl, "again")
    assert spent == 5
    assert len(leaf.calls) == spent
    assert repeated["retry_exhausted"] is True
    assert again["retry_exhausted"] is True
    assert "retry is exhausted" in again["note"]


def test_public_contract_uses_the_envelope_and_retires_restore_rows():
    leaf = Leaf()
    with _session(leaf) as repl:
        out = repl.execute(
            "result = semantic_map('x', {'type': 'boolean'})\n"
            "print(type(result).__name__, result is semantic_result, "
            "result['rows'] is semantic_rows, 'restore_rows' in globals())"
        )
    assert out["ok"], out["error"]
    assert out["stdout"].split() == ["dict", "True", "True", "False"]
    assert "retry_failed" in REPL_FUNCTIONS
    assert "restore_rows" not in REPL_FUNCTIONS
    assert any(line.startswith("retry_failed") for line in BOUND_NAMES)
    semantic_line = next(line for line in BOUND_NAMES
                         if line.startswith("semantic_map"))
    assert "returns a dict" in semantic_line and "result['rows']" in semantic_line


def test_engine_record_keeps_envelope_metadata_and_artifacts_rows(tmp_path):
    client = ScriptedClient([
        tool_reply(
            "result = semantic_map('x', {'type': 'boolean'})\n"
            "submit(result['coverage']['valid'])"
        ),
        text_reply("item 0: no\nitem 1: yes\nitem 2: no"),
    ])
    episode = RLMEngine(client=client, runs_dir=tmp_path).complete(
        CONTEXT, "which records contain the signal?", run_id="gate_b_episode")

    assert episode.answer_value == 3
    sweep = episode.sweeps[-1]
    assert sweep["kind"] == "semantic_map"
    assert sweep["coverage"]["valid"] == 3
    assert sweep["failed"] == []
    assert sweep["sweep_id"]
    assert sweep["retry_exhausted"] is False
    assert episode.semantic_result["sweep_id"] == sweep["sweep_id"]
    assert "rows" not in episode.semantic_result
    assert episode.semantic_result["rows_ref"].startswith("artifact://")
