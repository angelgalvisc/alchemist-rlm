"""Does a sub-model obey the contract when the caller's instruction fights it?

Twenty seconds, one sweep, no root model. This exists because the question was
being asked the expensive way: a fifteen-turn episode over a 78,000-character
corpus, three minutes per attempt, to observe something that is decided in the
first sub-call. Two runs of that answered "the leaf followed the caller"; a
probe answers it directly, and the episodes go back to what they are for —
end-to-end behaviour, not debugging a leaf.

The hostile instruction is the one an episode actually produced, kept because a
failure worth fixing is worth reproducing exactly:

    ... Output in JSON format: {"label": "category"}

Against that, every sub-model answered a single JSON object for a forty-one
item fragment and the sweep validated 0 of 795.

    ./.venv/bin/python scripts/probe_leaf_contract.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from alchemist_rlm import protocol                                    # noqa: E402
from alchemist_rlm.budgets import Budget                              # noqa: E402
from alchemist_rlm.calls.scheduler import SubcallScheduler            # noqa: E402
from alchemist_rlm.manifest import (                                 # noqa: E402
    RunManifest, interaction_contract_sha256, observation_contract_sha256,
    sha256_text,
)
from alchemist_rlm.mlx_client import MLXClient                        # noqa: E402
from alchemist_rlm.repl.runtime import ReplRuntime                    # noqa: E402
from alchemist_rlm.semantic import contract_fingerprint               # noqa: E402

ALCHEMIST = os.environ.get("ALCHEMIST_MODEL")

# Small enough to be one fragment, shaped like the corpus that failed.
CORPUS = "\n\n".join(
    f"Date: Apr {i + 1}, 2022 || User: {10000 + i} || Instance: {q}"
    for i, q in enumerate([
        "What does NATO stand for ?", "Who wrote the letter ?",
        "How many people attended ?", "Where is the river ?",
        "What is a democracy ?", "Which company filed it ?",
        "When did it close ?", "What does GDP mean ?",
    ])
)

LABELS = ["location", "human being", "entity", "abbreviation",
          "numeric value", "description and abstract concept"]

# The caller's instruction, verbatim from the episode that failed.
HOSTILE = ('Label each question with one of the following categories: '
           + ", ".join(LABELS)
           + '. Output in JSON format: {"label": "category"}')
NEUTRAL = "Label each question with its category."


def probe(name: str, instruction: str, model: str) -> bool:
    """One sweep. True if the contract held."""
    manifest = RunManifest(
        run_id=f"probe_{name}", arm="alchemist", suite="leaf_contract_probe",
        fingerprint_sha256="", tasks_sha256="",
        system_prompt_sha256=sha256_text(protocol.system_prompt()),
        tool_schema_sha256="", tool_name=protocol.TOOL_NAME,
        sampling={"temperature": 0.0, "max_tokens": 4096},
        leaf_contract_sha256=sha256_text(contract_fingerprint()),
        interaction_contract_sha256=interaction_contract_sha256(),
        observation_contract_sha256=observation_contract_sha256(),
    )
    client = MLXClient(model=model, manifest=manifest, timeout=300)
    budget = Budget(max_subcalls=8, max_seconds=300)
    scheduler = SubcallScheduler(client=client, budget=budget, max_tokens=1024)

    with ReplRuntime(handlers={"llm_query_batched": scheduler.query_batched}) as repl:
        repl.bind_context(CORPUS)
        repl.inject(INSTRUCTION=instruction, LABELS=LABELS)
        out = repl.execute(
            "r = semantic_map(INSTRUCTION, {'type': 'string', 'enum': LABELS})\n"
            "print(r['status'], r['valid_items'], r['total_items'])\n"
            "print('parse_errors:', r['parse_errors'])",
            timeout=300,
        )

    if not out or not out["ok"]:
        print(f"  {name}: EXECUTION FAILED — {out and out['error']}")
        return False
    status, valid, total = out["stdout"].split()[:3]
    held = status == "complete"
    print(f"  {name}: {status} {valid}/{total}  ->  "
          f"{'contract held' if held else 'CONTRACT LOST'}")
    if not held:
        print(f"    {out['stdout'].splitlines()[-1]}")
    return held


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=ALCHEMIST, required=ALCHEMIST is None)
    args = parser.parse_args()
    print("leaf contract probe | temperature 0 | one fragment, eight items\n")
    neutral = probe("neutral ", NEUTRAL, args.model)
    hostile = probe("hostile ", HOSTILE, args.model)
    print()
    if neutral and hostile:
        print("  the contract holds against an instruction that fights it")
        return 0
    if neutral and not hostile:
        print("  the caller's instruction still overrides the contract")
        return 1
    print("  the neutral case failed: something other than the conflict is wrong")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
