# Alchemist-RLM Harness vs. Upstream RLM

## Scope

This experiment compares two harnesses, not two models. Both arms use the same
local Agents-A1 (The Alchemist) 4-bit checkpoint, greedy decoding with thinking
disabled, the official OOLONG-Pairs 32K context (`context_window_id = 0`), the
verbatim public questions and the same external paper-strict scorer.

- **Upstream RLM:** unmodified `alexzhang13/rlm` at commit
  `caf0bffa1acec17c062559433b4cd4ed92eee3d6`.
- **Alchemist-RLM Harness:** the runtime in this repository.

Upstream receives 16 ordinary iterations. Alchemist-RLM Harness receives 14
root turns and, only after a semantic answer has been submitted, at most two
presentation turns. Both use a 900-second episode limit and at most two
in-flight model operations. The MLX prompt cache is reset and attested before
every task.

## Results

| task | floor | Upstream RLM F1 | Alchemist-RLM F1 | comparison |
|---:|---:|---:|---:|---|
| 1 | 0.4629 | 0.0000 | 0.0000 | retained Harness episode |
| 2 | 0.3356 | 0.0000 | 0.5690 | retained Harness episode |
| 3 | 0.5642 | 0.0000 | 0.6774 | retained Harness episode |
| 4 | 0.2126 | 0.0000 | 0.1734 | retained Harness episode |
| 5 | 0.1711 | 0.0000 | 0.5493 | retained Harness episode |
| 6 | 0.5024 | 0.0000 | 0.7417 | retained Harness episode |
| 7 | 0.2983 | 0.0000 | 0.8647 | retained Harness episode |
| 8 | 0.4686 | 0.0000 | 0.0000 | retained Harness episode |
| 9 | 0.1289 | 0.0000 | 0.0000 | retained Harness episode |
| 10 | 0.1249 | 0.0000 | 0.5429 | retained Harness episode |
| 11 | 0.0506 | 0.0000 | 0.1760 | retained Harness episode |
| 12 | 0.1072 | 0.0000 | 0.0000 | retained Harness episode |
| 13 | 0.1085 | 0.0000 | 0.0000 | retained Harness episode |
| 14 | 0.0821 | 0.0000 | 0.7257 | retained Harness episode |
| 15 | 0.0804 | 0.0000 | 0.4587 | retained Harness episode |
| 16 | 0.0156 | 0.0000 | 0.0000 | retained Harness episode |
| 17 | 0.1613 | 0.0000 | 0.0000 | retained Harness episode |
| 18 | 0.0712 | 0.0000 | 0.6922 | fresh paired run |
| 19 | 0.0045 | 0.0000 | 0.0952 | retained Harness episode |
| 20 | 0.0234 | 0.0000 | 0.0000 | retained Harness episode |
| **macro** | — | **0.0000** | **0.3133** | 20 tasks |

The Alchemist-RLM scores are the task-addressed episodes listed in
[`oolong_pairs_32k.json`](oolong_pairs_32k.json). Task 18 was rerun on the
comparison branch and matched its retained score exactly. Machine-readable
Upstream scores, runtimes and trajectory hashes are in
[`upstream_rlm_20.json`](upstream_rlm_20.json). The five original manifests and
twenty JSONL traces are retained in the checksummed
[`evidence/upstream_rlm_20.tar.gz`](evidence/upstream_rlm_20.tar.gz).

## What the traces show

In 19 of the 20 tasks, Upstream RLM never invoked a sub-model and primarily
spent the iteration budget inspecting context or drafting incomplete local
classification code. In many tasks it printed successive context slices. On
task 11 it parsed all 787 rows and made one batched classification call, but
accepted 3,436 returned labels for 787 inputs. It then repeated inspection of
that malformed result until the iteration limit. Task 17 repeatedly generated
an unfinished local classifier, consuming 742.7 seconds. No Upstream task
submitted parseable pairs under the paper, REPL or loose scorer.

The fresh Alchemist-RLM task-18 trace records validated semantic-map artifacts
covering 6/6 probe items and 787/787 full items. Its initial text was not valid
paper syntax, while its frozen typed value scored 0.6922 under the loose parser.
The bounded presentation phase rendered and validated that same value without
changing its membership; the final paper-strict score was 0.6922.

These traces support three concrete differences for this model and sample:

1. typed semantic operations prevent a long-context task from degenerating
   into sequential inspection;
2. schema and cardinality checks reject malformed batched classifications;
3. semantic computation and textual presentation are repaired separately, so
   a correct typed artifact is not lost solely to output syntax.

The experiment does not isolate the individual causal contribution of each
mechanism. Its conclusion is limited to this model, quantization, endpoint and
frozen twenty-task protocol.

## Reproduction

Clone Upstream RLM separately at the pinned commit and leave that checkout
clean. With the Alchemist-compatible OpenAI endpoint running on port 8081:

```bash
python scripts/run_upstream_rlm_pairs.py \
  --upstream-checkout /absolute/path/to/rlm \
  --model /absolute/path/to/agent-a1-alchemist-4bit \
  --tasks 2,7,11,18,20
```

The runner refuses a different commit, a dirty upstream checkout, a dirty
Alchemist-RLM code tree, a mismatched benchmark binding or a failed cache-reset
attestation. It writes task-level trace hashes and external scores to an
immutable run-addressed JSON file. Alchemist-RLM Harness tasks use:

```bash
python scripts/run_pairs_pilot.py \
  --infer-presentation-spec \
  --tasks 2,7,11,18,20 \
  --model /absolute/path/to/agent-a1-alchemist-4bit
```
