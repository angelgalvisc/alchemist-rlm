# Publication redactions

This public repository is a single-root snapshot of the evaluated code and
evidence. The private research history is intentionally not part of the public
Git history.

[`../PUBLICATION_MANIFEST.json`](../PUBLICATION_MANIFEST.json) records the
source snapshot and Git tree identifiers from before redaction. In particular,
the `src/` tree identifier lets the published commit demonstrate that no
harness implementation file changed during sanitization.

Before publication, machine-local path prefixes were replaced mechanically:

| original meaning | public token |
|---|---|
| evaluated Alchemist checkpoint | `${ALCHEMIST_MODEL}` |
| source and control checkpoints | `${AGENTS_BF16_MODEL}`, `${QWEN4B_BF16_MODEL}`, `${QWEN9B_MODEL}` |
| model-server interpreter | `${RLM_SERVER_PYTHON}` |
| repository checkout | `${REPO_ROOT}` |
| official Upstream RLM checkout | `${UPSTREAM_RLM_CHECKOUT}` |
| user home directory | `${HOME}` |
| temporary execution directory | `${TMPDIR}` |

No prompts, model responses, scores, semantic values, questions or source data
were changed. Recorded pre-redaction fingerprints remain provenance identifiers
for the original evaluation. The public evidence archive has its own SHA-256,
and [`../evals/evidence/upstream_rlm_20_files.sha256`](../evals/evidence/upstream_rlm_20_files.sha256)
lists the digest of every sanitized manifest and trace it contains.

These substitutions make paths portable without claiming that another local
directory layout reproduces the original machine. Exact model, benchmark,
dependency and code fingerprints remain recorded independently of path names.
The opt-in environment test normalizes only the redacted interpreter path
before comparison; package versions and platform metadata remain strict.
