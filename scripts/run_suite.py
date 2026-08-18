"""The decision suite: ten Alchemist tasks, eight controls, two ablations.

Run once, from a clean tree. The plan's rules are enforced here rather than
remembered:

  - a dirty working tree is refused, because the commit recorded against the
    result would not describe the code that produced it;
  - a fingerprint mismatch is caught before the model is called, so a preflight
    failure never becomes an episode;
  - a demonstrable infrastructure failure is marked `infrastructure_invalid`,
    keeps its trace, and stays out of the denominator;
  - a timeout with a healthy server counts as a failed result, not as noise;
  - nothing is re-run after a failure to rescue the same episode id.

    ./.venv/bin/python scripts/run_suite.py --arm alchemist
    ./.venv/bin/python scripts/run_suite.py --arm agents-bf16 --controls
    ./.venv/bin/python scripts/run_suite.py --ablation
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from alchemist_rlm import fingerprint, protocol                       # noqa: E402
from alchemist_rlm.budgets import Budget                              # noqa: E402
from alchemist_rlm.calls.scheduler import SUB_SYSTEM
from alchemist_rlm.semantic import contract_fingerprint          # noqa: E402
from alchemist_rlm.engine import BOUND_NAMES, RLMEngine                            # noqa: E402
from alchemist_rlm.isolation import MLXPromptCacheIsolation              # noqa: E402
from alchemist_rlm.manifest import (                                 # noqa: E402
    RunManifest, interaction_contract_sha256, observation_contract_sha256,
    runtime_determinism_record, sha256_text,
)
from alchemist_rlm.mlx_client import (                                # noqa: E402
    GenerationRejected, MLXClient, ServerUnavailable,
)
from alchemist_rlm.suite import (                                     # noqa: E402
    ABLATION_IDS, CONTROL_IDS, SUITE_SHA256, TASKS, TASKS_BY_ID, trajectory,
)
from alchemist_rlm.adapters.agents import STRATEGY_DIRECTIVES            # noqa: E402
from alchemist_rlm.output_contract import TerminalPolicy                 # noqa: E402
from alchemist_rlm.suite_v2 import (                                  # noqa: E402
    SUITE_V2_SHA256, TASKS_V2_BY_ID, lexical_tell,
)

# One registry so a run can mix frozen V1 tasks with V2 ones. Which suite a
# task came from is recorded per result, because a V1 task re-run under the
# current sub-model prompt is a new measurement and must not be compared with
# the V1 record that used the old one.
ALL_TASKS = {**TASKS_BY_ID, **TASKS_V2_BY_ID}
from alchemist_rlm.tracing import Trace                               # noqa: E402

MODELS = {
    "alchemist": os.environ.get("ALCHEMIST_MODEL"),
    "agents-bf16": os.environ.get("AGENTS_BF16_MODEL"),
    "qwen4b-base": os.environ.get("QWEN4B_BF16_MODEL"),
}


def run_one(task, engine, manifest, run_id: str, strategy: str = "auto") -> dict:
    """Run one task and score it on both layers.

    A server that stops answering and a server that refuses a generation are
    different outcomes and are recorded as such: the first invalidates the
    episode, the second is a result about the model. Deciding that by probing
    `/v1/models` is what keeps a parser crash from being filed as infrastructure.
    """
    started = time.monotonic()
    directive = STRATEGY_DIRECTIVES.get(strategy, "")
    question = f"{task.question}\n\n{directive}" if directive else task.question
    try:
        episode = engine.complete(task.context, question,
                                  run_id=run_id, inject=task.inject)
    except ServerUnavailable as error:
        # Demonstrably external: the server is gone. Keep the record, leave it
        # out of the denominator, repeat later under the same id.
        return {"id": task.id, "status": "infrastructure_invalid",
                "error": str(error), "seconds": round(time.monotonic() - started, 1)}
    except GenerationRejected as error:
        # The server is alive; the generation was not usable. This is a result
        # about the system and it counts, exactly as a timeout on a healthy
        # server does. It is not retried as if it were noise.
        return {"id": task.id, "status": "server_rejected_generation",
                "error": str(error), "result_correct": False, "passed": False,
                # The raw generation is gone -- the server parsed it, raised and
                # closed the socket -- so the request and the server's own
                # traceback are the only artifacts that make it reproducible.
                "request": getattr(error, "request", None),
                "server_traceback": getattr(error, "server_traceback", ""),
                "seconds": round(time.monotonic() - started, 1)}

    events = Trace.read(episode.trace_path)
    facts = trajectory(episode, events, task.context)
    correct = bool(task.scores_result(episode.answer or ""))
    unmet = [name for name, wanted in task.requires.items() if facts.get(name) != wanted]

    record = {
        "id": task.id,
        "strategy": strategy,
        "status": "ok",
        "seconds": round(time.monotonic() - started, 1),
        "answer": episode.answer,
        "truth": str(task.truth),
        "result_correct": correct,
        "trajectory": facts,
        "requires": task.requires,
        "unmet": unmet,
        # Both layers, never collapsed. A correct answer with an unmet
        # trajectory requirement is a correct number and a failed capability
        # test, and the plan is explicit that the two are kept apart.
        "passed": correct and not unmet,
        "stop_reason": episode.stop_reason,
        "ledger": episode.ledger,
        "protocol_errors": episode.protocol_errors,
        "trace": str(episode.trace_path),
    }
    if episode.stop_reason.startswith("forced_final:max_seconds"):
        record["status"] = "runtime_timeout"
    if task.attribute is not None:
        record["attribution"] = task.attribute(events, task.context, correct)
    if task.id.startswith("t08v2"):
        # Names the lexical failure mode instead of reporting a bare wrong
        # number: 488 is what a keyword sweep returns and 146 is the truth.
        record["attribution"] = lexical_tell(episode.answer or "")
    return record


def main() -> int:
    """Parse the arguments, refuse a dirty tree, and run the selected suite."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", default="alchemist", choices=sorted(MODELS))
    parser.add_argument("--controls", action="store_true",
                        help="run only the four discriminating control tasks")
    parser.add_argument("--ablation", action="store_true",
                        help="run the two name-ablation tasks with run_python")
    parser.add_argument("--only", default=None, help="a single task id")
    parser.add_argument("--tasks", default=None,
                        help="comma-separated task ids, V1 or V2")
    parser.add_argument("--label", default=None, help="name for the result file")
    parser.add_argument("--strategy", default="auto",
                        choices=["auto", "map", "recursive"],
                        help="append the adapter's frozen strategy directive to "
                             "each question. Measures directive-following, a "
                             "different question from spontaneous selection.")
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--max-seconds", type=float, default=900)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()

    tool_name = protocol.ALIAS_TOOL_NAME if args.ablation else protocol.TOOL_NAME
    if args.tasks:
        ids = [i.strip() for i in args.tasks.split(",") if i.strip()]
        missing = [i for i in ids if i not in ALL_TASKS]
        if missing:
            print(f"unknown task ids: {missing}")
            return 2
        tasks = [ALL_TASKS[i] for i in ids]
        suite_name = args.label or "mixed"
    elif args.only:
        tasks = [ALL_TASKS[args.only]]
        suite_name = f"single-{args.only}"
    elif args.ablation:
        tasks = [TASKS_BY_ID[i] for i in ABLATION_IDS]
        suite_name = "ablation_run_python"
    elif args.controls:
        tasks = [TASKS_BY_ID[i] for i in CONTROL_IDS]
        suite_name = "controls"
    else:
        tasks = list(TASKS)
        suite_name = "alchemist_ten"

    model = MODELS[args.arm]
    if not model:
        parser.error(
            f"model path for {args.arm!r} is not configured; set the matching "
            "*_MODEL environment variable"
        )
    manifest = RunManifest(
        run_id=f"{suite_name}_{args.arm}_{int(time.time())}",
        arm=args.arm, suite=suite_name,
        fingerprint_sha256="",
        tasks_sha256=f"v1:{SUITE_SHA256[:16]}|v2:{SUITE_V2_SHA256[:16]}",
        system_prompt_sha256=sha256_text(protocol.system_prompt(tool_name)),
        tool_schema_sha256=sha256_text(json.dumps(protocol.python_tool(tool_name),
                                                  sort_keys=True)),
        tool_name=tool_name,
        sampling={"temperature": 0.0, "seed": None, "max_tokens": 4096,
                  "enable_thinking": False},
        terminal_policy=TerminalPolicy().to_dict(),
        runtime_determinism=runtime_determinism_record(),
        isolation_policy=MLXPromptCacheIsolation(
            "http://127.0.0.1:8081/v1").manifest(),
        leaf_prompt_sha256=sha256_text(SUB_SYSTEM),
        leaf_max_tokens=RLMEngine.sub_max_tokens,
        bound_names_sha256=sha256_text("\n".join(BOUND_NAMES)),
        leaf_contract_sha256=sha256_text(contract_fingerprint()),
        interaction_contract_sha256=interaction_contract_sha256(),
        observation_contract_sha256=observation_contract_sha256(),
    )
    if manifest.git.get("code_dirty") and not args.allow_dirty:
        print("uncommitted code: " + ", ".join(manifest.git.get("uncommitted_code") or []))
        print("commit first, or pass --allow-dirty to record this as construction "
              "evidence rather than a decision episode.")
        return 3
    formal = not manifest.git.get("code_dirty")

    fp = json.loads((REPO / "configs" / "fingerprint.json").read_text())
    manifest.fingerprint_sha256 = fp["sha256"]
    live = fingerprint.model_fingerprint(Path(model), hash_weights=False)
    ok, differences = fingerprint.matches(live["metadata_sha256"],
                                          fp["models"][args.arm]["metadata_sha256"])
    if not ok:
        print(f"model metadata changed since the fingerprint was frozen: {differences}")
        return 2

    client = MLXClient(model=model, manifest=manifest, timeout=args.max_seconds + 60,
                       seed=None,
                       server_log=str(REPO / "logs" / "suite_server.log"))
    isolation = MLXPromptCacheIsolation(client.base_url)
    results = []
    print(f"{suite_name} | arm={args.arm} | tool={tool_name} | tasks={len(tasks)}\n")
    for task in tasks:
        engine = RLMEngine(
            client=client, tool_name=tool_name, manifest=manifest, block_timeout=300,
            budget=Budget(max_turns=args.max_turns, max_seconds=args.max_seconds,
                          max_in_flight=2, max_depth=2),
            episode_isolation=isolation,
        )
        record = run_one(task, engine, manifest, f"{manifest.run_id}_{task.id}",
                         strategy=args.strategy)
        results.append(record)
        mark = "PASS" if record.get("passed") else record.get("status", "FAIL").upper()
        print(f"  {task.id:28s} {mark:22s} {record.get('seconds', 0):6.1f}s  "
              f"answer={str(record.get('answer'))[:40]!r}")
        if record.get("unmet"):
            print(f"      unmet: {record['unmet']}")
        if record.get("attribution"):
            print(f"      attribution: {record['attribution'].get('verdict')}")

    live_results = [r for r in results if r["status"] != "infrastructure_invalid"]
    rejected = [r for r in results if r["status"] == "server_rejected_generation"]
    passed = [r for r in live_results if r.get("passed")]
    correct = [r for r in live_results if r.get("result_correct")]
    out = REPO / "configs" / f"suite_{suite_name}_{args.arm}.json"
    out.write_text(json.dumps({
        "manifest": manifest.to_dict(),
        "formal": formal,
        "model_stayed_put": manifest.model_stayed_put,
        "counts_as_decision_episodes": formal and manifest.model_stayed_put,
        "suite_sha256": SUITE_SHA256,
        "totals": {
            "episodes": len(live_results),
            "infrastructure_invalid": len(results) - len(live_results),
            "server_rejected_generation": len(rejected),
            "result_correct": len(correct),
            "passed_both_layers": len(passed),
        },
        "results": results,
    }, indent=1, ensure_ascii=False, default=str))

    print(f"\n  result correct:      {len(correct)}/{len(live_results)}")
    print(f"  passed both layers:  {len(passed)}/{len(live_results)}")
    print(f"  formal: {formal} | model stayed put: {manifest.model_stayed_put}")
    print(f"  written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
