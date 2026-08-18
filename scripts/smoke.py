"""The electrical check: two episodes, and the only early stop in the plan.

These do not ask whether the Alchemist can plan. They ask whether the wiring
carries current — whether it selects `PythonInterpreter` at all, integrates the
observation that comes back, keeps a variable between calls, and delegates a
slice with a real source. That is the hypothesis this whole harness rests on:
the evaluation does not send `tools`, so the tool-calling branch of the
template never activated and the model was judged on an interface it was not
trained for.

If both fail because the tool is never chosen or never integrated, the plan says
to stop and decide between `pivot_to_training` and `stop`. If they pass, they
count as two of the ten integrated tasks and are not re-run to inflate the
sample.

    ./.venv/bin/python scripts/smoke.py [--tool-name run_python] [--model ...]
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

from alchemist_rlm import fingerprint, protocol                      # noqa: E402
from alchemist_rlm.budgets import Budget                             # noqa: E402
from alchemist_rlm.calls.scheduler import SUB_SYSTEM
from alchemist_rlm.semantic import contract_fingerprint          # noqa: E402
from alchemist_rlm.engine import BOUND_NAMES, RLMEngine                           # noqa: E402
from alchemist_rlm.isolation import MLXPromptCacheIsolation             # noqa: E402
from alchemist_rlm.manifest import (                                # noqa: E402
    RunManifest, interaction_contract_sha256, observation_contract_sha256,
    runtime_determinism_record, sha256_text,
)
from alchemist_rlm.mlx_client import MLXClient, ServerUnavailable    # noqa: E402
from alchemist_rlm.output_contract import TerminalPolicy             # noqa: E402
from alchemist_rlm.tasks import TASKS_SHA256                         # noqa: E402
from alchemist_rlm.tracing import Trace                              # noqa: E402

ALCHEMIST = os.environ.get("ALCHEMIST_MODEL")

LEDGER = "\n".join(
    f"{i:03d} | {dept} | {amount} | {status}"
    for i, (dept, amount, status) in enumerate(
        [("logistics", 1420, "paid"), ("kitchen", 780, "pending"),
         ("logistics", 310, "paid"), ("front", 95, "void"),
         ("kitchen", 2050, "paid"), ("logistics", 640, "pending"),
         ("front", 1180, "paid"), ("kitchen", 415, "void"),
         ("logistics", 2270, "paid"), ("front", 530, "pending"),
         ("kitchen", 1325, "paid"), ("logistics", 860, "paid")],
        start=1,
    )
)

# `requires` is the pass condition, not a comment. The initial fixture listed the
# checks in prose and then scored the verdict on "a tool was called and some
# answer exists"; smoke_2 was reported as passing while its own record said
# `reused_a_variable: false`. A check that the verdict does not read is decoration.
SMOKE = [
    {
        "id": "smoke_1_exact_calculation",
        "question": ("What is 7919 multiplied by 6113? Compute it exactly, then "
                     "give the number."),
        "context": "",
        "truth": str(7919 * 6113),
        "requires": {
            "used_the_tool": True,
            "integrated_observation": True,
            "answer_correct": True,
        },
    },
    {
        "id": "smoke_2_persistence_and_subcall",
        "question": (
            "The data is a table of rows 'id | department | amount | status'. "
            "First store the number of rows in a variable called n. Then, in a "
            "later step, use llm_query to ask a language model what the third "
            "column means, passing read_context('s0000') as the source. "
            "Finally answer with n."
        ),
        "context": LEDGER,
        "truth": "12",
        "requires": {
            "used_the_tool": True,
            "integrated_observation": True,
            "answer_correct": True,
            # The task exists to test these two. A correct `12` reached in one
            # call, with no `n` and no delegation, is a correct number and a
            # failed persistence-and-subcall test.
            "reused_a_variable": True,
            "made_a_sourced_subcall": True,
        },
    },
]


def evaluate(episode, truth: str) -> dict:
    """Result and trajectory, scored separately — a correct number reached by
    reading the table by eye is not a demonstration of anything."""
    events = Trace.read(episode.trace_path)
    tool_calls = [e for e in events if e["kind"] == "tool_call"]
    subcalls = [e for e in events if e["kind"] == "subcall"]
    executed = [s for s in episode.steps if not getattr(s, "refused", False)]

    bound: set[str] = set()
    reused = False
    for index, step in enumerate(executed):
        if index and (bound & set(_names(step.code))):
            reused = True
        bound |= set(step.defined)

    answer = episode.answer or ""
    sourced = sum(1 for s in subcalls if (s["source"]["chars"] or 0) > 0)
    return {
        "answer_correct": truth in answer.replace(",", ""),
        "used_the_tool": bool(tool_calls),
        "tool_calls": len(tool_calls),
        "integrated_observation": len(tool_calls) > 0 and episode.answer is not None,
        "reused_a_variable": reused,
        "subcalls": len(subcalls),
        "subcalls_with_a_source": sourced,
        "made_a_sourced_subcall": sourced > 0,
        "stop_reason": episode.stop_reason,
        "duplicates_observed": episode.duplicates_observed,
        "protocol_errors": episode.protocol_errors,
    }


def _names(code: str) -> set[str]:
    import re

    return set(re.findall(r"\b[A-Za-z_]\w*\b", code or ""))


def main() -> int:
    """Run the two smoke episodes that prove the loop works end to end."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=ALCHEMIST, required=ALCHEMIST is None)
    parser.add_argument("--arm", default="alchemist")
    parser.add_argument("--tool-name", default=protocol.TOOL_NAME)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--only", default=None, help="run a single smoke id")
    parser.add_argument(
        "--allow-dirty", action="store_true",
        help="run from an uncommitted tree. The result is marked construction "
             "evidence and must not be counted as a decision episode.",
    )
    args = parser.parse_args()

    fp = json.loads((REPO / "configs" / "fingerprint.json").read_text())
    manifest = RunManifest(
        run_id=f"smoke_{int(time.time())}",
        arm=args.arm,
        suite="smoke",
        fingerprint_sha256=fp["sha256"],
        tasks_sha256=TASKS_SHA256,
        system_prompt_sha256=sha256_text(protocol.system_prompt(args.tool_name)),
        tool_schema_sha256=sha256_text(json.dumps(protocol.python_tool(args.tool_name),
                                                  sort_keys=True)),
        tool_name=args.tool_name,
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

    # A dirty tree means the recorded commit does not describe the code that
    # ran, so the episode cannot be reproduced from it. Recording `dirty: true`
    # and continuing anyway is what let a run be filed against a commit whose
    # working tree had already moved on; the manifest noticed and nothing
    # stopped it. Now it stops.
    if manifest.git.get("code_dirty") and not args.allow_dirty:
        print("uncommitted code: " + ", ".join(manifest.git.get("uncommitted_code") or []))
        print("commit first, or pass --allow-dirty to record this as construction "
              "evidence rather than a decision episode.")
        return 3
    formal = not manifest.git.get("code_dirty")

    # Checked before the model is called, so a mismatch never creates an
    # episode: the plan is explicit that a preflight failure is not a result
    # about the system. Metadata only — the weight hashes are in the frozen
    # record and re-reading gigabytes to start a two-episode check is not worth
    # it, but a changed chat template silently invalidates everything and costs
    # milliseconds to catch.
    live = fingerprint.model_fingerprint(Path(args.model), hash_weights=False)
    stored = fp["models"].get(args.arm)
    if stored is None:
        print(f"no fingerprint recorded for arm {args.arm!r}; refusing to run")
        return 2
    ok, differences = fingerprint.matches(
        live["metadata_sha256"], stored["metadata_sha256"]
    )
    if not ok:
        print(f"model metadata changed since the fingerprint was frozen: {differences}")
        return 2

    client = MLXClient(model=args.model, manifest=manifest, timeout=900, seed=None)
    isolation = MLXPromptCacheIsolation(client.base_url)
    results = []
    for task in SMOKE:
        if args.only and task["id"] != args.only:
            continue
        engine = RLMEngine(
            client=client,
            budget=Budget(max_turns=args.max_turns, max_seconds=900, max_in_flight=2),
            tool_name=args.tool_name,
            manifest=manifest,
            block_timeout=180,
            episode_isolation=isolation,
        )
        print(f"\n=== {task['id']} ===", flush=True)
        started = time.monotonic()
        try:
            episode = engine.complete(task["context"], task["question"],
                                      run_id=f"{manifest.run_id}_{task['id']}")
        except ServerUnavailable as error:
            print(f"  infrastructure_invalid: {error}")
            results.append({"id": task["id"], "status": "infrastructure_invalid",
                            "error": str(error)})
            continue
        scored = evaluate(episode, task["truth"])
        scored.update(id=task["id"], seconds=round(time.monotonic() - started, 1),
                      answer=episode.answer, trace=str(episode.trace_path),
                      requires=task["requires"])
        scored["unmet"] = unmet(scored, task["requires"])
        scored["passed"] = not scored["unmet"]
        results.append(scored)
        for key, value in scored.items():
            if key not in ("id", "trace", "requires"):
                print(f"  {key}: {value}")

    out = REPO / "configs" / "smoke_record.json"
    out.write_text(json.dumps({
        "manifest": manifest.to_dict(),
        "formal": formal,
        "counts_as_decision_episodes": formal and manifest.model_stayed_put,
        "model_stayed_put": manifest.model_stayed_put,
        "results": results,
        "verdict": verdict(results),
    }, indent=1, ensure_ascii=False, default=str))
    print(f"\nverdict: {verdict(results)}")
    print(f"formal (clean tree): {formal} | model stayed put: {manifest.model_stayed_put}")
    print(f"written: {out}")
    return 0


def unmet(result: dict, requires: dict) -> list[str]:
    """Which declared requirements a run failed to meet. Read from the task's own
    `requires` rather than a fixed list: a verdict that checked fewer facts than it
    declared reported a passing run whose record said otherwise.
    """
    return [name for name, wanted in requires.items() if result.get(name) != wanted]


def verdict(results: list[dict]) -> str:
    """What was demonstrated, stated no more strongly than the checks support."""
    live = [r for r in results if r.get("status") != "infrastructure_invalid"]
    if not live:
        return "NO RESULT — every attempt was infrastructure_invalid"
    if not any(r.get("used_the_tool") for r in live):
        return "STOP — the tool was never selected; decide pivot_to_training or stop"

    missing = {r["id"]: r.get("unmet", []) for r in live if r.get("unmet")}
    electrical = all(r.get("used_the_tool") and r.get("integrated_observation")
                     for r in live)
    if not missing:
        return ("GO (electrical) — every declared check met. The protocol carries "
                "current. This says nothing yet about planning over a large context.")
    if electrical:
        detail = "; ".join(f"{task}: {', '.join(names)}" for task, names in missing.items())
        return (f"PARTIAL — the protocol carries current, but declared checks failed "
                f"[{detail}]. Not a GO for the capability those checks stood for.")
    return f"PARTIAL — read the traces before deciding; unmet: {missing}"


if __name__ == "__main__":
    raise SystemExit(main())
