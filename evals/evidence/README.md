# Upstream RLM evaluation evidence

[`upstream_rlm_20.tar.gz`](upstream_rlm_20.tar.gz) contains the five retained
run manifests and twenty JSONL traces used to assemble
[`../upstream_rlm_20.json`](../upstream_rlm_20.json). Prompts, responses and
metrics are retained as recorded. Machine-local path prefixes were replaced by
the portable tokens documented in
[`../../docs/publication-redactions.md`](../../docs/publication-redactions.md).

Archive SHA-256:

```text
ccfbb45d048a38f6a862c9f5dfbcb11e01da7a827f2861b0418f253f57a265d4
```

Verify and inspect it with:

```bash
shasum -a 256 evals/evidence/upstream_rlm_20.tar.gz
tar -tzf evals/evidence/upstream_rlm_20.tar.gz
repo_root="$PWD"
evidence_dir="$(mktemp -d)"
tar -xzf evals/evidence/upstream_rlm_20.tar.gz -C "$evidence_dir"
(cd "$evidence_dir" && shasum -a 256 -c \
  "$repo_root/evals/evidence/upstream_rlm_20_files.sha256")
```

The task-level `trajectory_sha256` values in the compact index are hashes of
the structured completion metadata produced by Upstream RLM. The archive hash
above authenticates the manifests and logger traces as published. The adjacent
`upstream_rlm_20_files.sha256` records every sanitized file digest.
