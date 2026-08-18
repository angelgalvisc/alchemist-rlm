"""Run frozen OOLONG-Pairs tasks through an unmodified upstream RLM checkout.

This is a harness comparison, not an upstream fork.  The official checkout is
loaded from ``--upstream-checkout`` and must be clean at the pinned commit.  It
is never written.  Model output is scored afterward with the same local,
paper-strict scorer used for Alchemist-RLM Harness runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import requests

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from alchemist_rlm import fingerprint, oolong_pairs  # noqa: E402

PINNED_UPSTREAM_COMMIT = "caf0bffa1acec17c062559433b4cd4ed92eee3d6"
BASE_URL = "http://127.0.0.1:8081/v1"


def git(checkout: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(checkout), *args], text=True
    ).strip()


def require_clean_pinned_upstream(checkout: Path) -> dict[str, Any]:
    if not (checkout / "rlm" / "core" / "rlm.py").is_file():
        raise ValueError(f"not an upstream RLM checkout: {checkout}")
    commit = git(checkout, "rev-parse", "HEAD")
    dirty = git(checkout, "status", "--porcelain")
    if commit != PINNED_UPSTREAM_COMMIT:
        raise ValueError(
            f"upstream commit is {commit}, expected {PINNED_UPSTREAM_COMMIT}"
        )
    if dirty:
        raise ValueError("upstream RLM checkout is modified; refusing comparison")
    return {
        "repository": "https://github.com/alexzhang13/rlm",
        "commit": commit,
        "dirty": False,
    }


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def reset_prompt_cache(run_id: str) -> dict[str, Any]:
    response = requests.post(
        "http://127.0.0.1:8081/admin/prompt-cache/reset",
        json={"run_id": run_id, "policy": "mlx_prompt_cache_reset_v1"},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    after = payload.get("after") or {}
    sequences = after.get("sequences")
    byte_count = after.get("bytes")
    verified = payload.get("ok") is True and sequences == 0 and byte_count == 0
    if not verified:
        raise RuntimeError(f"prompt cache reset did not verify zero state: {payload}")
    return {
        "run_id": run_id,
        "policy": "mlx_prompt_cache_reset_v1",
        "epoch": payload.get("epoch"),
        "before": payload.get("before"),
        "after": after,
        "verified": True,
    }


def score_all(answer: str, truth: set[tuple[str, str]]) -> dict[str, Any]:
    return {
        "paper": oolong_pairs.f1(oolong_pairs.parse_answer(answer), truth),
        "repl": oolong_pairs.f1(oolong_pairs.parse_answer_repl(answer), truth),
        "loose": oolong_pairs.f1(oolong_pairs.parse_answer_loose(answer), truth),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-checkout", type=Path, required=True)
    parser.add_argument("--tasks", default="18")
    parser.add_argument(
        "--model",
        default=os.environ.get("ALCHEMIST_MODEL"),
        help="local Alchemist checkpoint (or set ALCHEMIST_MODEL)",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-iterations", type=int, default=16)
    parser.add_argument("--max-seconds", type=float, default=900.0)
    args = parser.parse_args()

    if not args.model:
        parser.error("--model is required unless ALCHEMIST_MODEL is set")

    upstream = args.upstream_checkout.resolve()
    upstream_record = require_clean_pinned_upstream(upstream)
    # The checkout wins over the installed rlms package.  No file in upstream
    # is patched or imported through a copied compatibility layer.
    sys.path.insert(0, str(upstream))
    from rlm import RLM  # type: ignore[import-not-found]  # noqa: E402
    from rlm.logger import RLMLogger  # type: ignore[import-not-found]  # noqa: E402
    from rlm.utils.prompts import (  # type: ignore[import-not-found]  # noqa: E402
        ORCHESTRATOR_ADDENDUM,
        RLM_SYSTEM_PROMPT,
    )

    selected = [int(value.strip()) for value in args.tasks.split(",") if value.strip()]
    frozen = oolong_pairs.load()
    by_number = {task["task"]: task for task in frozen["tasks"]}
    items = json.loads((REPO / "oolong" / "sample.json").read_text())["sample"]["32768"]
    binding_problems = oolong_pairs.check_official_binding(items, frozen)
    if binding_problems:
        raise RuntimeError(
            "local tasks do not match official OOLONG-Pairs 32K: "
            + "; ".join(binding_problems)
        )
    item = oolong_pairs.official_context(items, frozen)

    fp = json.loads((REPO / "configs" / "fingerprint.json").read_text())
    live = fingerprint.model_fingerprint(Path(args.model), hash_weights=False)
    ok, differences = fingerprint.matches(
        live["metadata_sha256"], fp["models"]["alchemist"]["metadata_sha256"]
    )
    if not ok:
        raise RuntimeError(f"Alchemist model metadata drifted: {differences}")

    run_id = f"upstream_rlm_pairs_{int(time.time())}"
    output = args.output or REPO / "configs" / f"{run_id}.json"
    if not output.is_absolute():
        output = REPO / output
    trace_root = output.parent / f"{run_id}_traces"
    trace_root.mkdir(parents=True, exist_ok=False)

    manifest = {
        "run_id": run_id,
        "comparison": "Upstream RLM vs. Alchemist-RLM Harness",
        "arm": "upstream-rlm",
        "upstream": upstream_record,
        "alchemist_rlm_commit": git(REPO, "rev-parse", "HEAD"),
        "alchemist_rlm_code_dirty": bool(git(REPO, "status", "--porcelain")),
        "model": str(Path(args.model).resolve()),
        "model_fingerprint_sha256": fp["sha256"],
        "queries_sha256": frozen["sha256"],
        "official_binding": frozen["official_binding"],
        "context_item_id": item["id"],
        "sampling": {
            "temperature": 0.0,
            "max_tokens": 4096,
            "enable_thinking": False,
            "transport": {
                "chat_template_kwargs": {"enable_thinking": False},
            },
        },
        "budget_mapping": {
            "max_iterations": args.max_iterations,
            "max_seconds": args.max_seconds,
            "max_depth": 1,
            "max_concurrent_subcalls": 2,
            "note": (
                "Upstream receives 16 ordinary iterations versus Alchemist-RLM "
                "Harness's 14 root turns plus at most 2 presentation turns."
            ),
        },
        "prompt": {
            "source": "unmodified upstream defaults",
            "system_sha256": sha256_text(RLM_SYSTEM_PROMPT),
            "orchestrator_addendum_sha256": sha256_text(ORCHESTRATOR_ADDENDUM),
        },
        "scoring": "external paper-strict OOLONG-Pairs scorer",
        "output_repair": "none",
        "typed_semantic_operations": "none",
    }
    results: list[dict[str, Any]] = []

    for number in selected:
        task = by_number[number]
        truth = oolong_pairs.gold(item["context_window_text_with_labels"], task["spec"])
        floor = oolong_pairs.f1(
            oolong_pairs.every_pair(item["context_window_text_with_labels"]), truth
        )
        episode_id = f"{run_id}_t{number:02d}"
        isolation = reset_prompt_cache(episode_id)
        logger = RLMLogger(log_dir=str(trace_root), file_name=episode_id)
        runtime = RLM(
            backend="openai",
            backend_kwargs={
                "api_key": "local-no-key",
                "base_url": BASE_URL,
                "model_name": str(Path(args.model).resolve()),
                "timeout": 600.0,
            },
            environment="local",
            max_depth=1,
            max_iterations=args.max_iterations,
            max_timeout=args.max_seconds,
            max_concurrent_subcalls=2,
            sampling_args={
                "temperature": 0.0,
                "max_tokens": 4096,
                "extra_body": {
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            },
            sub_sampling_args={
                "temperature": 0.0,
                "max_tokens": 1024,
                "extra_body": {
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            },
            logger=logger,
            verbose=False,
        )
        started = time.monotonic()
        completion = runtime.completion(
            item["context_window_text"], root_prompt=task["query"]
        )
        elapsed = time.monotonic() - started
        answer = completion.response or ""
        scores = score_all(answer, truth)
        trajectory = completion.metadata or {}
        iterations = trajectory.get("iterations") or []
        used_answer_dict = any(
            iteration.get("final_answer") is not None for iteration in iterations
        )
        result = {
            "task": number,
            "id": item["id"],
            "kind": task["spec"]["kind"],
            "gold_pairs": len(truth),
            "floor_f1": floor["f1"],
            "scores": scores,
            "paper": scores["paper"],
            "answer_chars": len(answer),
            "answer_head": answer[:200],
            "seconds": round(elapsed, 1),
            "iterations_logged": len(iterations),
            "submitted_via_answer_dict": used_answer_dict,
            "usage": completion.usage_summary.to_dict(),
            "isolation_attestation": isolation,
            "trace": logger.log_file_path,
            "trajectory_sha256": sha256_text(
                json.dumps(trajectory, sort_keys=True, default=str)
            ),
        }
        results.append(result)
        print(
            f"task {number:02d} paper={scores['paper']['f1']:.4f} "
            f"loose={scores['loose']['f1']:.4f} floor={floor['f1']:.4f} "
            f"pred={scores['paper']['predicted']}/{len(truth)} "
            f"iterations={len(iterations)} answer_dict={used_answer_dict} "
            f"seconds={elapsed:.1f}",
            flush=True,
        )

    payload = {"manifest": manifest, "tasks": selected, "results": results}
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    print(f"written: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
