"""Freeze the evidence: what ran, on which model, with which code.

Phase 0 of the plan. This exists because of a specific incident, not as
ceremony: `mlx_lm.server` silently swaps the resident model when a request
names a different path, and for several hours that produced "nondeterminism"
that was really two models answering the same prompt. Result files record their
`model_path` now, but nothing recorded the *server's* environment, the chat
template, or the weights themselves.

Two things this must catch:

  1. A different model behind the same name. The Alchemist and
     `artifacts/q4-language-core` have identical size and headers and different
     sha256 — size and name do not identify a checkpoint.
  2. A changed chat template. The Alchemist's template diverges from the
     upstream original only in identity strings, which is invisible to every
     check except a hash.

There are TWO environments here and both matter: the harness runs in
`rlm_test/.venv` (rlms, openai) while the MLX server runs from the codec
project's venv (mlx, mlx-lm). A version drift in either changes behaviour, so
both are fingerprinted.

Resume rule: a result file whose `fingerprint_sha256` differs from the current
environment is not resumable. Re-run it or archive it; never merge.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent.parent


def _server_python() -> Path:
    """The interpreter that runs the model server, which is often NOT this
    package's own.

    It was a hardcoded absolute path into a sibling project — the one line in
    `src/` that made the package unusable on another machine, and it pointed
    outside the repository at that. Three sources, most explicit first:

      1. ``RLM_SERVER_PYTHON``, for anyone running the server anywhere.
      2. The interpreter this checkout already declared, read from its own
         frozen fingerprint. This is what keeps a machine's runs comparable
         across the change, and it is not self-confirming: the frozen record
         stores the *path*, while the drift check compares the mlx, mlx-lm and
         transformers versions found inside it.
      3. The running interpreter — the honest default for a fresh clone that
         has declared nothing, so such a run records that it fingerprinted its
         own environment rather than silently claiming somebody else's.
    """
    declared = os.environ.get("RLM_SERVER_PYTHON")
    if declared:
        return Path(declared)
    frozen = REPO / "configs" / "fingerprint.json"
    try:
        path = json.loads(frozen.read_text())["environment"]["server_venv"][
            "python_path"]
        if path:
            return Path(path)
    except Exception:                                          # noqa: BLE001
        pass
    return Path(sys.executable)


SERVER_PYTHON = _server_python()

# Files that determine how a checkpoint behaves. `chat_template.jinja` is in
# this list because a template change alters every prompt without altering a
# single weight.
MODEL_METADATA_FILES = (
    "config.json",
    "chat_template.jinja",
    "tokenizer_config.json",
    "generation_config.json",
    "preprocessor_config.json",
)

_CHUNK = 1 << 20


def sha256_file(path: Path) -> str:
    """Hash a file in chunks, so weight files do not have to fit in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _versions(python: Path, packages: tuple[str, ...]) -> dict[str, str]:
    """Package versions as reported by the interpreter that will actually run.

    Importing them here would report *our* versions, which for the server-side
    packages is exactly the wrong answer.
    """
    script = (
        "import sys, json, importlib.metadata as md\n"
        f"pkgs = {list(packages)!r}\n"
        "out = {'python': sys.version.split()[0]}\n"
        "for p in pkgs:\n"
        "    try: out[p] = md.version(p)\n"
        "    except Exception: out[p] = None\n"
        "print(json.dumps(out))\n"
    )
    try:
        done = subprocess.run(
            [str(python), "-c", script], capture_output=True, text=True, timeout=60
        )
        return json.loads(done.stdout) if done.returncode == 0 else {"error": done.stderr[:200]}
    except Exception as exc:                                       # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


def model_fingerprint(model_dir: Path, *, hash_weights: bool = True) -> dict[str, Any]:
    """Identify a checkpoint by content, never by name or size.

    `hash_weights=False` skips the multi-second full read of the safetensors —
    useful in tests, wrong for a frozen record.
    """
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        raise FileNotFoundError(f"no such model directory: {model_dir}")

    metadata: dict[str, str] = {}
    for name in MODEL_METADATA_FILES:
        candidate = model_dir / name
        if candidate.exists():
            metadata[name] = sha256_file(candidate)

    weights: dict[str, Any] = {}
    for shard in sorted(model_dir.glob("*.safetensors")):
        entry: dict[str, Any] = {"bytes": shard.stat().st_size}
        if hash_weights:
            entry["sha256"] = sha256_file(shard)
        weights[shard.name] = entry

    quantization = None
    config_path = model_dir / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())
        quantization = config.get("quantization")
        model_type = config.get("model_type")
    else:
        model_type = None

    return {
        "path": str(model_dir),
        "model_type": model_type,
        "quantization": quantization,
        "metadata_sha256": metadata,
        "weights": weights,
    }


def environment_fingerprint() -> dict[str, Any]:
    """What the harness was running on. A result that cannot say which versions produced
    it cannot be reproduced, and MLX moves fast enough for that to matter.
    """
    return {
        "platform": platform.platform(),
        "harness_venv": _versions(
            Path(sys.executable), ("rlms", "openai", "datasets")
        ),
        "server_venv": {
            "python_path": str(SERVER_PYTHON),
            **_versions(SERVER_PYTHON, ("mlx", "mlx-lm", "mlx-metal", "numpy", "transformers")),
        },
    }


def build(model_dirs: dict[str, Path], *, hash_weights: bool = True) -> dict[str, Any]:
    """The full record. `sha256` over its own canonical JSON is the resume key."""
    record: dict[str, Any] = {
        "schema_version": 1,
        "environment": environment_fingerprint(),
        "models": {
            name: model_fingerprint(path, hash_weights=hash_weights)
            for name, path in model_dirs.items()
        },
    }
    canonical = json.dumps(record, sort_keys=True, separators=(",", ":"))
    record["sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    return record


def matches(record: dict[str, Any], other: dict[str, Any]) -> tuple[bool, list[str]]:
    """Compare two fingerprints and say *what* differs, not just whether.

    "Fingerprint mismatch" sends you re-reading two 200-line JSON files; naming
    the changed keys is the difference between a two-minute check and an hour.
    """
    differences: list[str] = []

    def walk(a: Any, b: Any, path: str) -> None:
        if isinstance(a, dict) and isinstance(b, dict):
            for key in sorted(set(a) | set(b)):
                if key == "sha256" and path == "":
                    continue
                if key not in a:
                    differences.append(f"{path}/{key}: added")
                elif key not in b:
                    differences.append(f"{path}/{key}: removed")
                else:
                    walk(a[key], b[key], f"{path}/{key}")
        elif a != b:
            differences.append(f"{path}: {a!r} -> {b!r}")

    walk(record, other, "")
    return not differences, differences
