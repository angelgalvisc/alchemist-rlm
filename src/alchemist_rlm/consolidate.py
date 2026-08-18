"""Offline consolidation for OOLONG-Pairs shards; never spends inference."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from alchemist_rlm import oolong_pairs
from alchemist_rlm.output_contract import (
    TerminalPolicy,
    read_answer_value_record,
    read_text_record,
)
from alchemist_rlm.manifest import visible_requests_sha256


class ConsolidationError(ValueError):
    """The supplied shards cannot support one comparable measurement."""


_CONFIG_FIELDS = (
    "arm", "suite", "fingerprint_sha256", "tasks_sha256",
    "system_prompt_sha256", "tool_schema_sha256", "tool_name", "sampling",
    "budget", "leaf_prompt_sha256", "leaf_max_tokens", "bound_names_sha256",
    "leaf_contract_sha256", "interaction_contract_sha256",
    "observation_contract_sha256", "output_mode", "output_contract",
    "output_contract_sha256", "terminal_policy", "output_backend_constraint",
    "presentation_spec_source", "presentation_linter",
    "runtime_determinism", "isolation_policy",
)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def manifest_signature(manifest: dict[str, Any]) -> str:
    """Comparable configuration, excluding counters and shard identity."""
    payload = {field: manifest.get(field) for field in _CONFIG_FIELDS}
    payload["git_commit"] = (manifest.get("git") or {}).get("commit")
    payload["server_argv"] = manifest.get("server_argv")
    segments = manifest.get("model_segments") or []
    payload["models"] = sorted({
        (segment.get("requested"), segment.get("served"))
        for segment in segments
    })
    return hashlib.sha256(_canonical(payload).encode()).hexdigest()


def _score_all(text: str, truth: set[oolong_pairs.Pair]) -> dict[str, Any]:
    return {
        "paper": oolong_pairs.f1(oolong_pairs.parse_answer(text), truth),
        "repl": oolong_pairs.f1(oolong_pairs.parse_answer_repl(text), truth),
        "loose": oolong_pairs.f1(oolong_pairs.parse_answer_loose(text), truth),
    }


def _official_pair_truths() -> dict[int, set[oolong_pairs.Pair]]:
    """Rebuild every truth from the one context bound by OOLONG-Pairs 32K."""
    frozen = oolong_pairs.load()
    sample = json.loads((oolong_pairs.REPO / "oolong" / "sample.json").read_text())[
        "sample"
    ]["32768"]
    problems = oolong_pairs.check_official_binding(sample, frozen)
    if problems:
        raise ConsolidationError(
            "local OOLONG-Pairs binding is invalid: " + "; ".join(problems))
    item = oolong_pairs.official_context(sample, frozen)
    return {
        int(task["task"]): oolong_pairs.gold(
            item["context_window_text_with_labels"], task["spec"])
        for task in frozen["tasks"]
    }


def _require_equal(label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise ConsolidationError(
            f"{label} mismatch: got {_canonical(actual)}, expected {_canonical(expected)}"
        )


def verify_episode_output(episode: dict[str, Any], artifacts: Path) -> dict[str, Any]:
    """Verify output references, immutable repair transitions and digests."""
    value = None
    value_record = episode.get("answer_value_record")
    if value_record is not None:
        try:
            value = read_answer_value_record(value_record, artifacts)
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            raise ConsolidationError(f"invalid answer_value_record: {error}") from error

    texts: dict[str, str | None] = {}
    for name in ("initial_final_text", "repair_candidate_text", "final_text"):
        record = episode.get(f"{name}_record")
        if record is None:
            texts[name] = None
            if episode.get(name) is not None:
                raise ConsolidationError(f"{name} has text but no verified record")
            continue
        try:
            texts[name] = read_text_record(record, artifacts)
        except (OSError, ValueError) as error:
            raise ConsolidationError(f"invalid {name}_record: {error}") from error
        _require_equal(name, episode.get(name), texts[name])

    repair = episode.get("output_repair")
    if isinstance(repair, dict) and repair.get("promoted"):
        if texts["repair_candidate_text"] is None:
            raise ConsolidationError("a promoted repair has no candidate")
        if not (repair.get("candidate_validation") or {}).get("valid"):
            raise ConsolidationError("a promoted repair was not validated")
        if texts["final_text"] != texts["repair_candidate_text"]:
            raise ConsolidationError("promoted candidate is not final_text")
    elif texts["final_text"] != texts["initial_final_text"]:
        raise ConsolidationError("final_text changed without a promoted repair")
    return {"answer_value": value, **texts}


def _verify_rows_artifacts(result: dict[str, Any], run_dir: Path) -> None:
    for sweep in result.get("sweeps") or []:
        ref, digest = sweep.get("rows_ref"), sweep.get("rows_digest")
        if not ref and not digest:
            continue
        if not isinstance(ref, str) or not ref.startswith("artifact://") or not digest:
            raise ConsolidationError("semantic rows artifact record is incomplete")
        name = ref.removeprefix("artifact://")
        try:
            rows = json.loads((run_dir / "artifacts" / f"{name}.txt").read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ConsolidationError(f"invalid semantic rows artifact {ref}: {error}") from error
        actual = hashlib.sha256(_canonical(rows).encode()).hexdigest()
        if actual != digest:
            raise ConsolidationError(f"semantic rows digest mismatch for {ref}")


def consolidate_pair_results(
    paths: Iterable[Path | str], *, runs_dir: Path | str,
    expected_tasks: Iterable[int] = range(1, 21),
) -> dict[str, Any]:
    """Validate shards and return one result per task under one configuration."""
    documents = [json.loads(Path(path).read_text()) for path in paths]
    if not documents:
        raise ConsolidationError("no result shards were supplied")
    manifests = [document.get("manifest") or {} for document in documents]
    signatures = {manifest_signature(manifest) for manifest in manifests}
    if len(signatures) != 1:
        raise ConsolidationError("result shards have different manifests")
    first_manifest = manifests[0]
    if (first_manifest.get("git") or {}).get("code_dirty") is not False:
        raise ConsolidationError("formal consolidation requires a clean code commit")
    if first_manifest.get("output_mode") != "validate_repair":
        raise ConsolidationError("formal gate accepts only output_mode=validate_repair")
    _require_equal("terminal_policy", first_manifest.get("terminal_policy"),
                   TerminalPolicy().to_dict())
    isolation_policy = first_manifest.get("isolation_policy") or {}
    if isolation_policy.get("name") != "mlx_prompt_cache_reset_v1":
        raise ConsolidationError("formal gate requires per-episode MLX cache reset")
    determinism = first_manifest.get("runtime_determinism") or {}
    if ("request_seed" not in determinism
            or determinism.get("tool_call_ids") != "canonical_turn_index"
            or determinism.get("visible_requests") != "strict_json_sha256"):
        raise ConsolidationError("formal gate lacks the frozen determinism policy")
    _require_equal("declared request seed",
                   (first_manifest.get("sampling") or {}).get("seed"),
                   determinism.get("request_seed"))
    contract = oolong_pairs.pair_output_contract()
    _require_equal("output_contract_sha256",
                   first_manifest.get("output_contract_sha256"), contract.sha256)
    _require_equal("output_contract", first_manifest.get("output_contract"),
                   contract.manifest())
    command = " ".join(first_manifest.get("server_argv") or [])
    for required in ("--prompt-cache-size 10", "--prompt-cache-bytes 1GB"):
        if required not in command:
            raise ConsolidationError(f"effective server argv lacks {required}")
    for manifest in manifests:
        if manifest.get("model_stayed_put") is not True:
            raise ConsolidationError("the served model changed within a shard")

    expected = set(expected_tasks)
    by_task: dict[int, tuple[dict[str, Any], dict[str, Any]]] = {}
    for document, manifest in zip(documents, manifests):
        for result in document.get("results") or []:
            task = int(result.get("task"))
            if task in by_task:
                raise ConsolidationError(f"duplicate task {task}")
            by_task[task] = (result, manifest)
    if set(by_task) != expected:
        raise ConsolidationError(
            f"task set mismatch: got {sorted(by_task)}, expected {sorted(expected)}"
        )

    truths = _official_pair_truths()
    verified: list[dict[str, Any]] = []
    runs_root = Path(runs_dir)
    for task in sorted(expected):
        result, manifest = by_task[task]
        if result.get("execution_status") != "completed":
            raise ConsolidationError(f"task {task} is not a completed episode")
        run_id = result.get("episode_run_id") or f"{manifest['run_id']}_t{task:02d}"
        run_dir = runs_root / run_id
        try:
            episode = json.loads((run_dir / "episode.json").read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ConsolidationError(f"task {task} has no valid episode.json: {error}") from error
        restored = verify_episode_output(episode, run_dir / "artifacts")
        attestation = episode.get("isolation_attestation") or {}
        after = attestation.get("after") or {}
        if (attestation.get("verified") is not True
                or attestation.get("policy") != isolation_policy.get("name")
                or after.get("sequences") != 0
                or after.get("bytes") != 0):
            raise ConsolidationError(
                f"task {task} has no verified empty-cache attestation")
        _require_equal(f"task {task} isolation_attestation",
                       result.get("isolation_attestation"), attestation)
        request_hashes = episode.get("visible_request_sha256s") or []
        _require_equal(f"task {task} visible_transcript_sha256",
                       episode.get("visible_transcript_sha256"),
                       visible_requests_sha256(request_hashes))
        _require_equal(f"task {task} visible request records",
                       result.get("visible_request_sha256s"), request_hashes)
        _verify_rows_artifacts(result, run_dir)
        truth = truths[task]
        raw = _score_all(restored["initial_final_text"] or "", truth)
        final = _score_all(restored["final_text"] or "", truth)
        _require_equal(f"task {task} scores_raw", result.get("scores_raw"), raw)
        _require_equal(f"task {task} paper_raw", result.get("paper_raw"), raw["paper"])
        _require_equal(f"task {task} scores_final", result.get("scores_final"), final)
        text = restored["final_text"] or ""
        structural = contract.validate(text)
        validation = structural.to_dict()
        validation["structural_valid"] = structural.valid
        if episode.get("answer_delivered"):
            if structural.valid:
                validation["binding"] = contract.binding.validate(
                    restored["answer_value"], text).to_dict()
            else:
                validation["binding"] = {
                    "valid": False,
                    "errors": ["structural validation failed before equivalence"],
                    "skipped": True,
                }
        else:
            validation["binding"] = None
        validation["valid"] = structural.valid and (
            not episode.get("answer_delivered")
            or bool(validation["binding"] and validation["binding"].get("valid"))
        )
        _require_equal(f"task {task} contract_validation",
                       result.get("contract_validation"), validation)
        typed = oolong_pairs.score_answer_value(restored["answer_value"], truth)
        content = result.get("content")
        if typed is None:
            _require_equal(f"task {task} content", content, None)
        elif not isinstance(content, dict) or content.get("metrics") != typed:
            raise ConsolidationError(f"task {task} typed content score mismatch")
        verified.append(result)
    return {
        "manifest_signature": next(iter(signatures)),
        "configuration": {field: first_manifest.get(field) for field in _CONFIG_FIELDS},
        "tasks": sorted(expected),
        "results": verified,
    }
