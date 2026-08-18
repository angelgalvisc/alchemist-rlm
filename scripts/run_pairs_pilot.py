"""Three OOLONG-Pairs queries, flat arm, run against a frozen harness.

The question this answers is not "how good is the model" but "can this
measurement produce a signal at all". A 4B against a benchmark where GPT-5
scores 58% and Qwen3-Coder-480B scores 23.1% may well score nothing, and two
arms of nothing compare nothing.

Asymmetric queries on purpose. Their degenerate floor — the F1 of answering
every possible pair without reading anything — is 0.004-0.11, against 0.46-0.56
for the symmetric ones. A score on the symmetric half would be almost
impossible to read; here it is not.

**The reading is three-way, and the run records what each branch needs.**
A high context coverage with a high content F1 and a failed strict score is a
delivery problem: the classification worked and the answer came back in the
wrong shape. A `provided_items` scope, or Python errors, is an operation
problem: the root model never ran the sweep over the context. An artifact that
does not match its digest, a coverage figure with no certificate behind it, or
units that vanish between sweep and table is a harness bug, whatever the score
says — which is why `verify_rows_artifact` recomputes the digest from the bytes
on disk rather than trusting the number the run wrote down.

Flat arm: `max_depth=0`, so `rlm_query` refuses. That is a budget setting, not
a change to the harness, and the recursive arm is the same command with
`max_depth=2`.

    ./.venv/bin/python scripts/run_pairs_pilot.py
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from alchemist_rlm import fingerprint, oolong_pairs, protocol          # noqa: E402
from alchemist_rlm.budgets import Budget                               # noqa: E402
from alchemist_rlm.calls.scheduler import SUB_SYSTEM
from alchemist_rlm.semantic import contract_fingerprint          # noqa: E402
from alchemist_rlm.engine import BOUND_NAMES, RLMEngine                             # noqa: E402
from alchemist_rlm.isolation import MLXPromptCacheIsolation               # noqa: E402
from alchemist_rlm.inferred_presentation import infer_presentation_spec  # noqa: E402
from alchemist_rlm.manifest import (                                  # noqa: E402
    RunManifest, interaction_contract_sha256, observation_contract_sha256,
    runtime_determinism_record, sha256_text,
)
from alchemist_rlm.mlx_client import (                                 # noqa: E402
    GenerationRejected, MLXClient, ServerUnavailable,
)
from alchemist_rlm.output_contract import (                            # noqa: E402
    OUTPUT_MODES, TerminalPolicy, read_answer_value_record,
)

ALCHEMIST = os.environ.get("ALCHEMIST_MODEL")
# The last two intact tasks, run together as the final cold measurement.
# Everything else has been used to diagnose or confirm. 15 carries the most
# conditions of any query - three on its `a` side - at a floor of 0.080; 17 has
# the highest floor in the whole set at 0.161, which is the hardest thing here
# to clear, and 2,330 gold pairs. After this there is no cold ammunition left.
PILOT_TASKS = (15, 17)
MAX_DEPTH = 0                       # the flat arm

# `--directed` appends the frozen `classify` directive to the question. It names
# the operation and the schema shape and stops: what the categories are and what
# to do with the result stay in the question, or the run would be measuring the
# directive. Without it the question is untouched, and only *that* run can say
# anything about the model selecting the operation itself.
DIRECTED_HELP = "--directed appends the frozen `classify` strategy directive"


def verify_rows_artifact(run_dir: Path, sweep: dict) -> dict:
    """Re-derive a sweep's digest from the bytes on disk.

    A recorded digest that nobody recomputes is a claim, not a check. This
    parses the artifact the run wrote, re-canonicalises it exactly as the
    worker did — sorted keys, no spaces — and compares. It is also what
    distinguishes a harness bug from a model failure in the reading below: an
    artifact that is missing, unparseable or does not match its digest is the
    harness's fault whatever the score says.
    """
    ref, digest = sweep.get("rows_ref"), sweep.get("rows_digest")
    if not ref or not digest:
        return {"checked": False, "why": "the sweep wrote no artifact"}
    # `ArtifactStore` is rooted at run_dir/artifacts, and writes `<name>.txt`.
    path = run_dir / "artifacts" / f"{str(ref).split('//', 1)[-1]}.txt"
    if not path.exists():
        return {"checked": True, "valid": False, "why": f"no file {path.name}"}
    try:
        rows = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        return {"checked": True, "valid": False, "why": f"unparseable: {error}"}
    recomputed = hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"checked": True, "valid": recomputed == str(digest),
            "rows": len(rows) if isinstance(rows, list) else None,
            "recomputed": recomputed[:16], "recorded": str(digest)[:16]}


def sweep_record(sweep: dict, run_dir: Path) -> dict:
    """One sweep's coverage facts, flattened for the results file."""
    certificate = sweep.get("certificate") or {}
    return {
        "operation": sweep.get("operation"),
        "status": sweep.get("status"),
        "scope": sweep.get("scope"),
        "valid_items": sweep.get("valid_items"),
        "total_items": sweep.get("total_items"),
        "context_coverage_complete": sweep.get("context_coverage_complete"),
        "failed_items": len(sweep.get("failed_items") or []),
        "rows_ref": sweep.get("rows_ref"),
        "rows_digest": sweep.get("rows_digest"),
        "artifact_check": verify_rows_artifact(run_dir, sweep),
        "certificate_complete": certificate.get("complete"),
        "certificate_failed_units": certificate.get("failed_units"),
        "certificate_gaps": len(certificate.get("gaps") or []),
    }


def score_all(answer: str, truth: set) -> dict:
    """One answer under all three parsers, never collapsed into one number.

    `paper` is the format the query names and nothing else, `repl` adds the
    shape a delivered Python value renders to, `loose` adds a line that is two
    ids and a separator. They answer three different questions — was it
    returned as asked, was it computed and delivered as an object, was it
    computed at all — and the project's rule is that no one of them may stand
    in for another. Grouping them here is what stops a caller scoring two of
    the three and quoting whichever it has.
    """
    return {
        "paper": oolong_pairs.f1(oolong_pairs.parse_answer(answer), truth),
        "repl": oolong_pairs.f1(oolong_pairs.parse_answer_repl(answer), truth),
        "loose": oolong_pairs.f1(oolong_pairs.parse_answer_loose(answer), truth),
    }


def task_status(*, stop_reason: str, answer: str, strict: dict, loose: dict,
                floor_f1: float) -> str:
    """A task verdict, kept separate from whether inference returned normally.

    "failed" means nothing was computed, and it is read off the answer rather
    than off the stop reason. The line used to single out `consecutive_errors`,
    the only stall counter when it was written; there are three now and it was
    never widened, so the same delivery would be called failed or scored on its
    merits depending on which guard happened to trip.

    Not consulting the stop reason at all was the first attempt, and it was too
    blunt: an episode that died on repeated code errors having emitted the word
    "thinking" is not an attempt that scored badly, and "at_or_below_floor"
    says it was. `predicted` keeps that distinction and takes it from the same
    place the score comes from — no pair parsed under any parser is nothing
    computed, whatever ended the run.

    Checked before changing it: of nine episodes on record ending on
    `consecutive_errors`, every one also scored 0.000 under both parsers, so
    the old clause never mislabelled anything. It was latent, not live, and
    this note exists so the next reader does not credit the change with a
    rescue.
    """
    if not answer or not loose.get("predicted"):
        return "failed"
    if strict["f1"] > floor_f1:
        return "above_floor"
    if strict["f1"] <= floor_f1 and loose["f1"] > floor_f1:
        return "invalid_format"
    return "at_or_below_floor"


def main() -> int:
    """Run the registered pilot queries and report each against its floor."""
    from alchemist_rlm.adapters.agents import STRATEGY_DIRECTIVES

    parser = argparse.ArgumentParser()
    parser.add_argument("--directed", action="store_true", help=DIRECTED_HELP)
    parser.add_argument("--tasks", default=",".join(map(str, PILOT_TASKS)),
                        help="comma-separated frozen task numbers")
    parser.add_argument("--max-depth", type=int, default=MAX_DEPTH)
    parser.add_argument(
        "--output-mode", default="validate_repair",
        choices=sorted(OUTPUT_MODES - {"constrained"}),
        help="raw, validate_only, or one objective text-only repair after delivery",
    )
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument(
        "--infer-presentation-spec", action="store_true",
        help=("experimental: infer a frozen format spec from the public question "
              "and expose a local linter only after an invalid candidate"),
    )
    parser.add_argument("--model", default=ALCHEMIST, required=ALCHEMIST is None)
    parser.add_argument(
        "--seed", default="none",
        help="backend sampling seed as an integer, or 'none' for greedy baseline",
    )
    parser.add_argument("--output", default=None,
                        help="result path; default is immutable and run-addressed")
    args = parser.parse_args()
    try:
        request_seed = (None if args.seed.lower() == "none" else int(args.seed))
    except ValueError:
        parser.error("--seed must be an integer or 'none'")
    selected = tuple(int(value.strip()) for value in args.tasks.split(",")
                     if value.strip())
    directed = args.directed
    directive = STRATEGY_DIRECTIVES["classify"] if directed else ""
    problems = oolong_pairs.check_specs()
    if problems:
        print("the gold specs disagree with their queries:", *problems, sep="\n  ")
        return 4
    frozen = oolong_pairs.load()
    items = json.loads((REPO / "oolong" / "sample.json").read_text())["sample"]["32768"]
    binding_problems = oolong_pairs.check_official_binding(items, frozen)
    if binding_problems:
        print("the local benchmark does not match official OOLONG-Pairs 32K:",
              *binding_problems, sep="\n  ")
        return 4
    item = oolong_pairs.official_context(items, frozen)

    output_contract = (None if args.output_mode == "raw"
                       else oolong_pairs.pair_output_contract())
    terminal_policy = TerminalPolicy()
    run_budget = Budget(max_turns=14, max_seconds=900, max_in_flight=2,
                        max_depth=args.max_depth)
    manifest = RunManifest(
        run_id=f"pairs_pilot_{int(time.time())}",
        arm="alchemist",
        suite=("oolong_pairs_directed" if directed else "oolong_pairs_auto")
              + ("_inferred_format" if args.infer_presentation_spec else ""),
        fingerprint_sha256="", tasks_sha256=frozen["sha256"],
        system_prompt_sha256=sha256_text(protocol.system_prompt()),
        tool_schema_sha256=sha256_text(json.dumps(protocol.python_tool(), sort_keys=True)),
        tool_name=protocol.TOOL_NAME,
        # Read off the engine, not retyped beside it. This said 4096 while the
        # root generated at 8192 — a manifest describing a run that did not
        # happen, in the one record whose whole job is to be trusted later.
        sampling={"temperature": 0.0, "seed": request_seed,
                  "max_tokens": RLMEngine.max_tokens,
                  "enable_thinking": False},
        budget=run_budget.to_dict(),
        output_mode=args.output_mode,
        output_contract=(output_contract.manifest() if output_contract else None),
        output_contract_sha256=(output_contract.sha256 if output_contract else ""),
        terminal_policy=terminal_policy.to_dict(),
        output_backend_constraint="none",
        presentation_spec_source=("model_question_only_v1"
                                  if args.infer_presentation_spec else "adapter"),
        presentation_linter=("check_presentation_v1_after_invalid_candidate"
                             if args.infer_presentation_spec else "none"),
        runtime_determinism=runtime_determinism_record(
            request_seed=request_seed),
        isolation_policy=MLXPromptCacheIsolation(
            "http://127.0.0.1:8081/v1").manifest(),
        leaf_prompt_sha256=sha256_text(SUB_SYSTEM),
        leaf_max_tokens=RLMEngine.sub_max_tokens,
        bound_names_sha256=sha256_text("\n".join(BOUND_NAMES)),
        leaf_contract_sha256=sha256_text(contract_fingerprint()),
        interaction_contract_sha256=interaction_contract_sha256(),
        observation_contract_sha256=observation_contract_sha256(),
    )
    if manifest.server_argv is not None:
        effective_server = " ".join(manifest.server_argv)
        missing_flags = [
            flag for flag in ("--prompt-cache-size 10", "--prompt-cache-bytes 1GB")
            if flag not in effective_server
        ]
        if missing_flags:
            print("effective server argv lacks required cache limits:",
                  ", ".join(missing_flags))
            return 2
    if manifest.git.get("code_dirty"):
        if not args.allow_dirty:
            print("uncommitted code:",
                  ", ".join(manifest.git.get("uncommitted_code") or []))
            return 3
        print("WARNING: running from an uncommitted tree. The recorded commit "
              "does not contain the code that will run; these numbers are not "
              "citable.\n", flush=True)
    fp = json.loads((REPO / "configs" / "fingerprint.json").read_text())
    manifest.fingerprint_sha256 = fp["sha256"]
    live = fingerprint.model_fingerprint(Path(args.model), hash_weights=False)
    ok, differences = fingerprint.matches(live["metadata_sha256"],
                                          fp["models"]["alchemist"]["metadata_sha256"])
    if not ok:
        print("model metadata drifted:", differences)
        return 2

    client = MLXClient(model=args.model, manifest=manifest, timeout=600,
                       seed=request_seed,
                       server_log=str(REPO / "logs" / "suite_server.log"))
    isolation = MLXPromptCacheIsolation(client.base_url)
    by_id = {t["task"]: t for t in frozen["tasks"]}
    results = []
    infrastructure_failed = False
    print(f"{manifest.suite} | max_depth={args.max_depth} | tasks={list(selected)} "
          f"| strategy={'classify (directed)' if directed else 'auto'}\n")

    for number in selected:
        if number not in by_id:
            print(f"unknown frozen task: {number}")
            return 4
        task = by_id[number]
        truth = oolong_pairs.gold(item["context_window_text_with_labels"], task["spec"])
        floor = oolong_pairs.f1(
            oolong_pairs.every_pair(item["context_window_text_with_labels"]), truth)
        record = {"task": number, "id": item["id"], "kind": task["spec"]["kind"],
                  "gold_pairs": len(truth), "floor_f1": floor["f1"]}
        started = time.monotonic()
        try:
            inferred_record = None
            inferred_spec = None
            if args.infer_presentation_spec:
                format_isolation = isolation.before_episode(
                    run_id=f"{manifest.run_id}_t{number:02d}_format_inference")
                inferred_record = infer_presentation_spec(client, task["query"])
                inferred_record["isolation_attestation"] = format_isolation
                inferred_spec = (inferred_record.get("spec")
                                 if inferred_record.get("status") == "ok" else None)
                record["presentation_spec_inference"] = inferred_record
            episode = RLMEngine(
                client=client, manifest=manifest, block_timeout=300,
                budget=run_budget,
                output_mode=args.output_mode,
                output_contract=output_contract,
                inferred_presentation_spec=inferred_spec,
                terminal_policy=terminal_policy,
                episode_isolation=isolation,
            ).complete(item["context_window_text"],
                       f"{task['query']}\n\n{directive}" if directed else task["query"],
                       run_id=f"{manifest.run_id}_t{number:02d}")
            initial_text = episode.initial_final_text or ""
            final_text = episode.final_text or ""
            scores_raw = score_all(initial_text, truth)
            scores_final = (scores_raw if final_text == initial_text
                            else score_all(final_text, truth))
            scored, loose = scores_raw["paper"], scores_raw["loose"]
            typed_content = None
            if episode.answer_value_record is not None:
                verified_value = read_answer_value_record(
                    episode.answer_value_record,
                    REPO / "runs" / f"{manifest.run_id}_t{number:02d}" / "artifacts",
                )
                typed_metrics = oolong_pairs.score_answer_value(verified_value, truth)
                if typed_metrics is not None:
                    scorer_source = "\n".join((
                        inspect.getsource(oolong_pairs.pairs_from_answer_value),
                        inspect.getsource(oolong_pairs.f1),
                    ))
                    typed_content = {
                        "input": "answer_value",
                        "scorer_sha256": sha256_text(scorer_source),
                        "metrics": typed_metrics,
                    }
            contract_validated = (
                {"valid": True, "scores_final": scores_final}
                if (episode.contract_validation or {}).get("valid") is True
                else None
            )
            record.update(
                episode_run_id=f"{manifest.run_id}_t{number:02d}",
                scores_raw=scores_raw,
                paper_raw=scores_raw["paper"],
                content=typed_content,
                contract_validation=episode.contract_validation,
                scores_final=scores_final,
                contract_validated=contract_validated,
                output_repair=episode.output_repair,
                presentation_checks=episode.presentation_checks,
                presentation_renders=episode.presentation_renders,
            )
            # Named apart so it can never be quoted as the score: what was
            # computed, against whether it came back in the shape the query
            # asked for. An answer can be right and returned wrong.
            record["content_f1_ignoring_format"] = loose
            record["scores"] = scores_raw
            run_dir = REPO / "runs" / f"{manifest.run_id}_t{number:02d}"
            sweeps = [sweep_record(s, run_dir) for s in episode.sweeps]
            # The context sweep is the one a coverage claim can rest on, and
            # the last of them is the one the episode's own summary reports.
            over_context = [s for s in sweeps if s["scope"] == "context"]
            grounding = over_context[-1] if over_context else (
                sweeps[-1] if sweeps else {})
            record.update(execution_status="completed",
                          task_status=task_status(
                              stop_reason=episode.stop_reason, answer=initial_text,
                              strict=scored, loose=loose, floor_f1=floor["f1"]),
                          operations=episode.operations,
                          semantic_cache_hits=len(episode.semantic_cache_hits),
                          sweeps=sweeps,
                          semantic_status=grounding.get("status"),
                          semantic_scope=grounding.get("scope"),
                          context_coverage_complete=grounding.get(
                              "context_coverage_complete"),
                          artifact_valid=(grounding.get("artifact_check") or {}
                                          ).get("valid"),
                          stop_reason=episode.stop_reason,
                          turns=episode.turns, subcalls=episode.ledger.get("subcalls", 0),
                          visible_transcript_sha256=(
                              episode.visible_transcript_sha256),
                          visible_request_sha256s=episode.visible_request_sha256s,
                          isolation_attestation=episode.isolation_attestation,
                          answer_chars=len(final_text),
                          answer_head=final_text[:200], **scored)
        except ServerUnavailable as error:
            record.update(execution_status="infrastructure_invalid",
                          task_status="not_run", error=str(error)[:200])
            infrastructure_failed = True
        except GenerationRejected as error:
            record.update(execution_status="server_rejected_generation",
                          task_status="not_run", error=str(error)[:200])
            infrastructure_failed = True
        record["seconds"] = round(time.monotonic() - started, 1)
        results.append(record)
        # Three parsers named as themselves. This line used to read "F1", which
        # is the number a reader compares to the paper — and it was the widened
        # parser, accepting `[a, b]` where the query says `(a, b)`. The label
        # claimed a conformance the regex did not have, so all three are
        # printed and none of them is called the score.
        print(f"  task {number:>2}  paper {record.get('f1', 0):.3f}"
              f"  repl {(record.get('scores') or {}).get('repl', {}).get('f1', 0):.3f}"
              f"  loose {record.get('content_f1_ignoring_format', {}).get('f1', 0):.3f}"
              f"  (floor {record['floor_f1']:.3f})"
              f"  pred {record.get('predicted', 0):,}/{len(truth):,}"
              f"  {record.get('task_status', record['execution_status'])}"
              f"/{record.get('stop_reason', '-')}  {record['seconds']:.0f}s"
              f"  final-paper {(record.get('scores_final') or {}).get('paper', {}).get('f1', 0):.3f}",
              flush=True)
        if record.get("sweeps"):
            for sweep in record["sweeps"]:
                check = sweep["artifact_check"]
                print(f"           {sweep['operation']} {sweep['scope']} "
                      f"{sweep['valid_items']}/{sweep['total_items']} "
                      f"ctx_complete={sweep['context_coverage_complete']} "
                      f"cert={sweep['certificate_complete']} "
                      f"artifact={'ok' if check.get('valid') else check.get('why') or 'MISMATCH'}",
                      flush=True)
        if infrastructure_failed:
            print("           stopping: infrastructure failure invalidates later tasks",
                  flush=True)
            break

    live_results = [r for r in results if r.get("execution_status") == "completed"]
    beat = [r for r in live_results if r["f1"] > r["floor_f1"]]
    beat_final = [
        r for r in live_results
        if ((r.get("scores_final") or {}).get("paper") or {}).get("f1", 0)
        > r["floor_f1"]
    ]
    # A boolean buried in the manifest was not enough. `--allow-dirty` made the
    # provenance guard optional, and its first use produced two result files
    # carrying the SAME recorded commit, `code_dirty: True`, and opposite
    # scores — 0.000 and 0.808 — with nothing at the top of either saying the
    # recorded commit does not contain the code that ran. This says it where a
    # reader cannot miss it, and names the files that were uncommitted.
    provenance: dict[str, Any] | None = None
    if manifest.git.get("code_dirty"):
        provenance = {
            "reproducible_from_commit": False,
            "warning": ("RUN FROM AN UNCOMMITTED TREE. The recorded commit does "
                        "NOT contain the code that produced this result, and "
                        "another run may record the same commit with different "
                        "code. Do not cite these numbers."),
            "uncommitted_code": manifest.git.get("uncommitted_code") or [],
        }

    out = {"manifest": manifest.to_dict(), "queries_sha256": frozen["sha256"],
           "max_depth": args.max_depth, "pilot_tasks": list(selected),
           "directed": directed,
           **({"provenance": provenance} if provenance else {}),
           "output_mode": args.output_mode,
           "infer_presentation_spec": args.infer_presentation_spec,
           "output_contract": (output_contract.manifest()
                               if output_contract else None),
           "claim_scope": (
               "directed runs test the runtime after naming the operation; "
               "task_status and the paper-strict score decide task success"
               if directed else
               "automatic runs test whether the model selects and uses an operation; "
               "task_status and the paper-strict score decide task success"),
           "reference_points": {
               # Which column these may be set beside, stated in the file that
               # carries them. They are paper-strict numbers, so only `paper`
               # compares — `repl` and `loose` are this project's own widenings
               # and putting either next to 0.2311 would be comparing a parser
               # to a model. The floor belongs in the comparison too: the same
               # queries answered without reading anything average 0.196.
               "comparable_to": "scores.paper, and only with floor_f1 beside it",
               "gpt5_rlm": 0.580, "qwen3_coder_480b_rlm": 0.2311,
               "gpt5_base": 0.0004, "qwen3_coder_480b_base": 0.0006},
           "results": results}
    output = (Path(args.output) if args.output else
              REPO / "configs" / f"{manifest.suite}_{manifest.run_id}.json")
    if not output.is_absolute():
        output = REPO / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=1, ensure_ascii=False, default=str))
    print(f"\n  raw paper above its own floor: {len(beat)}/{len(live_results)}")
    print(f"  final paper above its own floor: {len(beat_final)}/{len(live_results)}")
    print(f"  written: {output.relative_to(REPO) if output.is_relative_to(REPO) else output}")
    return 5 if infrastructure_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
