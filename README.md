# Alchemist-RLM Harness

**Alchemist-RLM Harness enables small local models to solve structured reasoning
tasks over data larger than the root model's context window.** On the complete
OOLONG-Pairs 32K evaluation,
[`Agent-A1 Alchemist 4-bit`](https://huggingface.co/angelgalvisc/agent-a1-alchemist-4bit)—a
4-billion-parameter model that runs locally—reaches **31.331% macro F1**, versus
**0.000%** for the same checkpoint with unmodified Upstream RLM.

The runtime keeps the source inside a persistent Python REPL, where the model
can use local code, typed exhaustive semantic operations, bounded sub-model
calls and optional recursive children. It separates semantic computation from
output presentation and returns an auditable execution record.

**Alchemist-RLM Harness** names the runtime in this repository. **Upstream RLM**
refers exclusively to the unmodified official implementation from
[`alexzhang13/rlm`](https://github.com/alexzhang13/rlm), pinned to an exact
commit when used as an experimental baseline. Alchemist-RLM Harness is not a
fork of Upstream RLM: comparisons run the same model and frozen tasks through
two separate runtimes and score both outputs externally.

The project is designed for small local models and emphasizes three properties:

- explicit delivery of typed answers;
- measured coverage rather than self-reported coverage;
- a strict separation between semantic computation and output presentation.

> [!WARNING]
> The REPL executes model-written Python. Its subprocess isolation contains
> failures such as timeouts, but it is not a security sandbox. Do not use it on
> inputs or machines where arbitrary code execution would be unacceptable.

## OOLONG-Pairs 32K evaluation

The repository evaluates the twenty OOLONG-Pairs questions from Appendix D.1
of the [Recursive Language Models paper](https://arxiv.org/abs/2512.24601).
OOLONG-Pairs asks a model to infer one of six semantic answer types for each of
787 user-associated questions, aggregate those labels by user, and return every
user pair satisfying a different set of constraints in each task. It tests
dense long-context understanding, exact aggregation and large structured
outputs rather than retrieval of a single hidden fact.

The exact benchmark source is
[`mit-oasys/oolong-pairs/data/oolong-pairs-32768.json`](https://huggingface.co/datasets/mit-oasys/oolong-pairs/blob/main/data/oolong-pairs-32768.json).

All twenty public questions are stored verbatim in
[`oolong/pairs_queries.json`](oolong/pairs_queries.json). Before a run starts,
the runner verifies both context hashes, the ordered question hash, and the
count and SHA-256 digest of every gold pair set. Gold
sets are derived locally from the source labels and are never exposed to the
model.

### Models and checkpoints

The complete twenty-task evaluation and the controlled harness comparison use
[`angelgalvisc/agent-a1-alchemist-4bit`](https://huggingface.co/angelgalvisc/agent-a1-alchemist-4bit),
the author's MLX 4-bit build of
[`InternScience/Agents-A1-4B`](https://huggingface.co/InternScience/Agents-A1-4B).
The recorded model fingerprint is
`f4c86d1228bdebd3a82562a642c3c9949aea346700c5ceaee1087e8650338c5c`.

### Protocol

The reported evaluation used:

- Agents-A1 (The Alchemist), 4-bit, served locally with MLX;
- temperature `0.0`, no sampling seed, thinking disabled;
- 14 root turns, 900 seconds, depth `0`, at most two in-flight operations;
- the untouched public question (`directed=false`, equivalent to the automatic
  operation-selection arm);
- official pair-format validation and bounded presentation repair after a
  submitted answer;
- presentation grammar inferred from the public question only;
- prompt-cache isolation checked before every episode;
- official pair F1, requiring one numeric ordered pair `(a, b)` per line.

Here, **depth 0 means that no child RLM episode was opened**. It does not mean
that the root model worked without delegation: 15 of the 20 retained episodes
used bounded semantic or leaf-model calls, for 986 subcalls in total. Every
published episode records an empty `recursions` list. Recursive child episodes
are implemented and exercised by the deterministic test suite, but were not
invoked or credited in the twenty-task score below.

The degenerate floor is the F1 obtained by returning every possible pair. It is
reported beside each score because it varies substantially by task.

### Reference points from the RLM paper

The following table places the Alchemist-RLM result beside the OOLONG-Pairs
results reported in Table 1 and Figure 3(a) of the
[official RLM paper](https://arxiv.org/html/2512.24601). All rows report
OOLONG-Pairs 32K macro F1, but only the Alchemist rows were executed in this
repository. The paper rows use different models and execution environments, so
they are reference points rather than controlled model comparisons.

| source | model and runtime | parameters | recursion depth | macro F1 |
|---|---|---:|---:|---:|
| [this repository](evals/oolong_pairs_32k.json) | **[`Alchemist 4-bit`](https://huggingface.co/angelgalvisc/agent-a1-alchemist-4bit) + Alchemist-RLM Harness** | **4B** | **0** | **31.331%** |
| [this repository](evals/upstream_rlm_20.json) | [`Alchemist 4-bit`](https://huggingface.co/angelgalvisc/agent-a1-alchemist-4bit) + Upstream RLM | 4B | 0 observed | 0.000% |
| [RLM paper, Figure 3(a)](https://arxiv.org/html/2512.24601#S4.F3) | [`Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B), direct model | 8B | — | 0.07% |
| [RLM paper, Figure 3(a)](https://arxiv.org/html/2512.24601#S4.F3) | Qwen3-8B + RLM, without post-training | 8B | 1 | 4.26% |
| [RLM paper, Figure 3(a)](https://arxiv.org/html/2512.24601#S4.F3) | **[`RLM-Qwen3-8B`](https://huggingface.co/mit-oasys/rlm-qwen3-8b-v0.1), post-trained** | **8B** | **1** | **5.17%** |
| [RLM paper, Table 1](https://arxiv.org/html/2512.24601#S4.T1) | [`Qwen3-Coder-480B-A35B-Instruct`](https://huggingface.co/Qwen/Qwen3-Coder-480B-A35B-Instruct) | 480B total / 35B active | 0 | 17.3% |
| [RLM paper, Table 1](https://arxiv.org/html/2512.24601#S4.T1) | Qwen3-Coder-480B-A35B-Instruct | 480B total / 35B active | 1 | 23.1% |
| [RLM paper, Table 1](https://arxiv.org/html/2512.24601#S4.T1) | GPT-5 | not disclosed | 0 | 43.9% |
| [RLM paper, Table 1](https://arxiv.org/html/2512.24601#S4.T1) | GPT-5 | not disclosed | 1 | 58.0% |
| [RLM paper, Table 1](https://arxiv.org/html/2512.24601#S4.T1) | GPT-5 | not disclosed | 2 | 65.5% |
| [RLM paper, Table 1](https://arxiv.org/html/2512.24601#S4.T1) | GPT-5 | not disclosed | 3 | 76.0% |

Under these reference conditions, Alchemist-RLM Harness places the local 4-bit
model above the paper's base, scaffolded and post-trained Qwen3-8B results and
above its Qwen3-Coder-480B results at depths 0 and 1, while remaining below
every reported GPT-5 configuration. That ordering is descriptive: it does not
isolate model quality from harness, prompting, serving stack or other protocol
differences. The paper's separately post-trained Qwen3-4B is evaluated on
MRCRv2 rather than OOLONG-Pairs and is therefore not included in this table.

### Alchemist-RLM Harness results

| task | official F1 | floor | task | official F1 | floor |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.0000 | 0.4629 | 11 | 0.1760 | 0.0506 |
| 2 | 0.5690 | 0.3356 | 12 | 0.0000 | 0.1072 |
| 3 | 0.6774 | 0.5642 | 13 | 0.0000 | 0.1085 |
| 4 | 0.1734 | 0.2126 | 14 | 0.7257 | 0.0821 |
| 5 | 0.5493 | 0.1711 | 15 | 0.4587 | 0.0804 |
| 6 | 0.7417 | 0.5024 | 16 | 0.0000 | 0.0156 |
| 7 | 0.8647 | 0.2983 | 17 | 0.0000 | 0.1613 |
| 8 | 0.0000 | 0.4686 | 18 | 0.6922 | 0.0712 |
| 9 | 0.0000 | 0.1289 | 19 | 0.0952 | 0.0045 |
| 10 | 0.5429 | 0.1249 | 20 | 0.0000 | 0.0234 |

The macro-average official F1 is **31.331%** (`6.2662 / 20`, calculated from
the unrounded task metrics). Eleven tasks score above their degenerate floor,
one produces valid paper-format output below its floor, and eight score zero.

Episode termination was retained rather than filtered or retried. Eight
episodes submitted within the root loop. Eight reached the 14-turn limit and
submitted during bounded finalization. Four ended without a delivered answer:
one after consecutive truncations, and one each at the `max_seconds`,
`max_turns` and `max_subcalls` limits. All twenty remain in the denominator.

The `protocol_errors` field is a diagnostic event stream, not a count of
internal harness failures. Across these traces it contains 39 events: 16 final
block executions (12 successful), 13 truncation events, seven bounded-work
notices and three degenerate-repetition notices. These records preserve model
and controller behavior that affected an episode instead of silently hiding it.

### Controlled same-model harness comparison

To isolate the practical effect of the runtime, the same Alchemist checkpoint
was evaluated on the same twenty questions twice: once with an unmodified
checkout of Upstream RLM and once with Alchemist-RLM Harness. Both arms use the
same frozen context, greedy no-thinking decoding, external scorer and
cache-isolation policy. The upstream repository is supplied externally, pinned
to an exact commit and required to be clean; this repository never patches it.

| runtime | tasks | macro F1 | tasks above floor |
|---|---:|---:|---:|
| Upstream RLM | 20 | 0.0000 | 0 |
| **Alchemist-RLM Harness** | **20** | **0.31331** | **11** |

The complete twenty-task comparison found that Upstream RLM did not submit any
scoreable pairs on this model. Alchemist-RLM Harness scored above the task floor
on eleven tasks. A fresh paired run on task 18 reproduced its retained score:
`0.0000` for Upstream RLM and `0.6922` for Alchemist-RLM Harness.

The traces make the observed advantage concrete. Upstream usually spent its
iteration budget inspecting context or producing incomplete classification
code. Alchemist-RLM Harness instead supplied exhaustive typed semantic
operations, cardinality validation and a separate presentation phase. Those
mechanisms let the same model produce scoreable answers without exposing gold
labels or benchmark-specific solution logic. The result demonstrates a clear
advantage for this model under this frozen protocol; it is not evidence of
universal superiority across models or workloads.

The protocol, task-level table and trace analysis are recorded in
[`evals/upstream_rlm_comparison.md`](evals/upstream_rlm_comparison.md). The
comparison runner is [`scripts/run_upstream_rlm_pairs.py`](scripts/run_upstream_rlm_pairs.py),
and the original manifests and twenty JSONL traces are retained in a
[checksummed evidence archive](evals/evidence/README.md).

The 20 tasks were evaluated as independent episodes executed in batches, as is
standard for a task-level benchmark. Every episode used the same model,
benchmark source, root budget and scorer, and retains its own manifest and
trace. Exact task-level provenance is recorded in
[`evals/oolong_pairs_32k.json`](evals/oolong_pairs_32k.json).

## Hardware and local throughput

All Alchemist results reported in this repository were produced locally on one
consumer laptop, without a discrete GPU or remote model API:

| component | specification |
|---|---|
| computer | MacBook Pro (`Mac17,7`) |
| chip | Apple M5 Max |
| CPU | 18 cores |
| integrated GPU | 32 cores, Metal 4 |
| unified memory | 36 GB LPDDR5 |
| operating system | macOS 26.5.2 (`25F84`) |
| model | [`Alchemist 4-bit`](https://huggingface.co/angelgalvisc/agent-a1-alchemist-4bit), 4B parameters, approximately 2.4 GB on disk |
| serving stack | MLX, one decode worker and one prompt worker |

On this machine, a five-request warm throughput check produced **127.46 output
tokens/s** on average. Each request used a short prompt, greedy decoding,
thinking disabled, a reset prompt cache, concurrency `1`, and generated exactly
512 output tokens. The excluded warm-up generated 256 tokens. Across the five
measured requests, 2,560 tokens were generated in 20.085 seconds; individual
rates ranged from 126.90 to 127.86 tokens/s. The machine-readable record is
[`evals/alchemist_local_throughput.json`](evals/alchemist_local_throughput.json).

This is end-to-end generation throughput for short requests, not the effective
throughput of a complete RLM episode. OOLONG-Pairs additionally includes prompt
processing, Python execution, semantic batches, validation and orchestration,
so task latency should be read from the individual run manifests.

### Hardware used by the RLM paper

The [RLM paper](https://arxiv.org/html/2512.24601) discloses hardware for its
small-scale post-training experiment, but not for the hosted models used in its
main OOLONG-Pairs evaluation:

| paper activity | disclosed compute environment |
|---|---|
| OOLONG-Pairs with GPT-5 | OpenAI provider; physical CPU/GPU and memory not reported |
| OOLONG-Pairs with Qwen3-Coder-480B-A35B | costs based on Fireworks; physical CPU/GPU and memory not reported |
| post-training RLM-Qwen3-8B | **48 H100 GPU-hours**, using `prime-rl`; GPU count and node configuration not reported |
| MRCRv2 reinforcement-learning experiment | Prime Intellect Lab; accelerator model and count not reported |

The 48 H100-hours belong to training RLM-Qwen3-8B; they are not identified as
the hardware used to generate the GPT-5 or Qwen3-Coder OOLONG-Pairs scores in
the comparison table. The authors also state that their model calls were
blocking and sequential and warn that runtime depends on the machine, API
latency and call asynchrony.

Consequently, the paper scores can be compared as benchmark reference points,
but a hardware-efficiency or tokens-per-second comparison against this Mac
cannot be made from the published information.

## Scope and limitations

- Results currently cover one local 4-bit model.
- `strategy="auto"` is experimental; directed strategies are more reliable.
- The evaluation uses the RLM's persistent REPL and bounded batched sub-model
  calls. Recursive child episodes are implemented but were not invoked in this
  depth-0 benchmark arm.
- A complete coverage certificate is not a correctness certificate.
- The runtime executes arbitrary model-written Python and is not a security
  sandbox.
- Benchmark scores depend on the recorded model, prompts, runtime, budgets and
  output protocol; comparisons should retain those provenance fields.

## Install

Python 3.11 is the tested runtime. The recorded evaluation used Python 3.11.15.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest
```

The deterministic suite currently contains 474 tests and does not require a
model server. One machine-fingerprint check is skipped by default because it
validates the exact environment used for the recorded experiments; set
`RLM_VALIDATE_FROZEN_ENV=1` on that machine to include it.

## Use

The public adapter exposes one function:

```python
from alchemist_rlm import RLMEngine, analyze_large_context
from alchemist_rlm.mlx_client import MLXClient

client = MLXClient(model="/path/to/model")
engine = RLMEngine(client=client)

result = analyze_large_context(
    context,
    question,
    strategy="classify",
    engine=engine,
)

result["answer"]          # final textual presentation
result["answer_value"]    # typed value committed inside the REPL
result["answer_typed"]    # distinguishes no delivery from submit(None)
result["answer_valid"]    # deliverable under the requested strategy
result["coverage"]        # coverage of the sweep supporting the verdict
result["certificate"]     # structural and span evidence, not correctness
result["trace"]           # complete execution trace
```

The available strategies are:

| strategy | intended use |
|---|---|
| `auto` | Let the root model choose an operation. Experimental. |
| `map` | Exhaustive semantic decision over every segment. |
| `classify` | One typed label per item followed by local aggregation. |
| `recursive` | Delegate large parts that require their own multi-step analysis. |

The client talks to an OpenAI-compatible HTTP endpoint. The model server is
intentionally not a package dependency. For the local MLX setup used in this
repository:

```bash
export ALCHEMIST_MODEL=/absolute/path/to/your/model
export RLM_SERVER_PYTHON=/absolute/path/to/python
./serve.sh alchemist
python scripts/probe_leaf_contract.py
python scripts/run_pairs_pilot.py --infer-presentation-spec --tasks 1,2
```

`RLM_SERVER_PYTHON` may be set to the interpreter running the server so its
environment can be fingerprinted in run manifests.

The controlled Upstream RLM comparison has additional dependencies. Install
them with `python -m pip install -e ".[dev,comparison]"`; the `comparison`
extra pins the evaluated `rlms` and OpenAI client versions.

## Execution model

```text
caller
  │
  ▼
adapter ── selects a strategy and reports answer validity
  │
  ▼
engine ─── owns the episode, budget, artifacts and trace
  │
  ▼
native loop ── root-model conversation and explicit state transitions
  │
  ▼
persistent REPL
  ├── local Python over `context`
  ├── semantic_search / semantic_map
  ├── bounded llm_query / llm_query_batched
  └── optional rlm_query / rlm_map children
```

### Persistent context and bounded work

The full source is bound as the Python variable `context`; it is not inserted
into the root conversation. Variables survive between turns. Exact filtering,
joining and aggregation happen locally, while semantic decisions are delegated
through bounded calls that share the episode's budget.

`semantic_map(instruction, schema)` returns one schema-validated value per item.
For operations over the original context, the runtime records which units were
sent, which returned valid values, which failed, and whether the resulting spans
establish complete coverage.

### Typed delivery

The root model commits an answer explicitly with:

```python
submit(value)
```

or, when it has separately constructed the requested text:

```python
submit(value, final_text=final_text)
```

A block commits only when it calls `submit` exactly once and finishes without
raising. Values such as `[]`, `0` and `None` are valid deliveries; no Python
shape is treated as an implicit answer.

### Presentation validation and repair

Adapters may declare a deterministic output contract. The contract validates
public syntax and may verify that the text represents the model's committed
value, but it never receives benchmark gold and cannot select a better answer.

In `validate_repair` mode, an invalid presentation opens a bounded,
presentation-only phase:

1. the typed answer is frozen;
2. the model receives structural diagnostics;
3. persistent read-only values such as `PRESENTATION_VALUE` remain available;
4. `check_presentation(text)` validates a candidate locally;
5. `render_presentation(PRESENTATION_VALUE)` may render compatible primitive
   structures from a grammar inferred only from the public question;
6. a candidate is promoted only after an explicit model `submit` and successful
   structural and content-binding validation.

The repair phase cannot inspect gold answers, rerun semantic operations or
change the committed semantic value.

### Evidence and isolation

Each episode records:

- the effective code, model and server fingerprints;
- budgets, sampling settings and prompt hashes;
- every model-visible request hash;
- typed-answer and presentation digests;
- semantic sweep coverage and artifact digests;
- termination reason, protocol errors and resource use;
- prompt-cache reset attestations when the MLX isolation policy is active.

A coverage certificate establishes that declared units were processed and
returned the required shape. It does not establish that the model's semantic
judgements or final aggregation are correct.

## Repository layout

```text
src/alchemist_rlm/
  adapters/                 public agent-facing API
  calls/                    bounded and recursive model calls
  context/                  storage, segmentation and literal search
  repl/                     isolated persistent Python session
  engine.py                 episode, budget, artifact and trace ownership
  native_loop.py            root conversation and terminal state machine
  semantic.py               typed exhaustive semantic operations
  output_contract.py        validation, binding and terminal policy
  inferred_presentation.py  question-only presentation grammar
  certificate.py            coverage evidence
  manifest.py               reproducibility metadata

scripts/                     evaluation runners, audits, contract probe and MLX server
oolong/                      frozen benchmark context and OOLONG-Pairs questions
configs/                     retained evaluation configurations and test fixtures
evals/                       result indexes, harness comparison and evidence archive
runs/                        the twenty retained OOLONG-Pairs task episodes
tests/                       deterministic test suite and fixtures
docs/                        publication and redaction documentation
.github/                     continuous-integration workflow
serve.sh                     local MLX model-server launcher
setup.sh                     reproducible Python 3.11 environment setup
pyproject.toml               package metadata, dependencies and test configuration
PUBLICATION_MANIFEST.json    sanitized snapshot and provenance identifiers
```

## Reproducibility rules

A result intended for comparison should come from a committed tree and retain
its run manifest, trace, output contract, task-set digest, model fingerprint and
cache-isolation record. `src/alchemist_rlm/consolidate.py` fails closed when a
formal aggregate mixes incompatible manifests, code states or task bindings.

The public snapshot replaces only machine-local path prefixes with portable
tokens. The exact substitutions and evidence checksums are documented in
[`docs/publication-redactions.md`](docs/publication-redactions.md) and
[`PUBLICATION_MANIFEST.json`](PUBLICATION_MANIFEST.json).

## License and attribution

The project is released under the [MIT License](LICENSE). Citation metadata is
provided in [`CITATION.cff`](CITATION.cff), and redistributed benchmark
material is documented in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

The RLM method and OOLONG-Pairs benchmark are due to Zhang, Kraska and Khattab
(MIT OASYS): [Recursive Language Models](https://arxiv.org/abs/2512.24601) and
the official [`rlm` repository](https://github.com/alexzhang13/rlm).
The official OOLONG-Pairs dataset is also MIT-licensed.
