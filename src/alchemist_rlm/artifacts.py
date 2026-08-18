"""Outputs that may be larger than the window the model writes them from.

A RLM must be able to produce an answer it cannot itself hold, so a submitted
value may be either the answer or an artifact reference to it; the engine
resolves the reference at the end. Artifacts also give the model somewhere to
put intermediate results that would otherwise have to survive as stdout, which
is truncated.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REF = re.compile(r"^artifact://([A-Za-z0-9_.\-]+)$")
_SAFE_NAME = re.compile(r"^[A-Za-z0-9_.\-]{1,120}$")


@dataclass(frozen=True)
class Artifact:
    """One saved value: its name, its reference, and proof of what it held."""
    name: str
    ref: str
    chars: int
    sha256: str
    kind: str
    path: Path


class ArtifactStore:
    """Files under one run directory. Names are validated, never joined blindly."""

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.saved: dict[str, Artifact] = {}

    def _path(self, name: str) -> Path:
        if not _SAFE_NAME.match(name or ""):
            raise ValueError(
                f"artifact name {name!r} must be 1-120 chars of letters, digits, "
                "'_', '.' or '-' and cannot contain a path separator"
            )
        return self.root / f"{name}.txt"

    def save(self, name: str, value: Any) -> str:
        """Write a value to disk and return the reference that stands for it.

        The point is that the reference is small. A model that must carry a large
        intermediate result forward has to spend its window on it; a model that can
        park it spends a dozen characters instead.
        """
        kind = "text" if isinstance(value, str) else "json"
        text = value if kind == "text" else json.dumps(
            value, ensure_ascii=False, default=str)
        path = self._path(name)
        path.write_text(text, encoding="utf-8")
        artifact = Artifact(
            name=name,
            ref=f"artifact://{name}",
            chars=len(text),
            sha256=hashlib.sha256(text.encode()).hexdigest(),
            kind=kind,
            path=path,
        )
        self.saved[name] = artifact
        return artifact.ref

    def load(self, name_or_ref: str) -> str:
        """Read a saved value back, by name or by reference. A missing artifact raises
        rather than returning empty, because an empty string here would flow onward
        as if it were the data.
        """
        name = self.parse_ref(name_or_ref) or name_or_ref
        path = self._path(name)
        if not path.exists():
            raise KeyError(
                f"no artifact named {name!r}; saved so far: {sorted(self.saved) or 'none'}"
            )
        return path.read_text(encoding="utf-8")

    def load_value(self, name_or_ref: str) -> Any:
        """Restore the Python value saved during this run.

        The file stays plain text so it is inspectable and independently
        auditable.  Inside the live session, however, assigning a saved list
        back to a variable should produce a list, not its JSON spelling.  That
        mismatch made all three frozen agents reload rows they already had;
        one overwrote the already-bound ``semantic_rows`` list with a long
        string and spent subsequent turns recovering it.

        An artifact opened by a fresh ``ArtifactStore`` has no trusted type
        metadata and therefore remains text; only values saved by this store
        instance are restored structurally.
        """
        name = self.parse_ref(name_or_ref) or name_or_ref
        text = self.load(name_or_ref)
        artifact = self.saved.get(name)
        return json.loads(text) if artifact and artifact.kind == "json" else text

    @staticmethod
    def parse_ref(value: str) -> str | None:
        """The name inside an `artifact://name` reference, or None if this is not one.
        Used to accept either form wherever a value is expected.
        """
        match = REF.match((value or "").strip())
        return match.group(1) if match else None

    def resolve(self, value: Any) -> Any:
        """Turn a submitted artifact reference into its contents.

        A reference that resolves to nothing is not the reference. This used to
        swallow the KeyError and hand back `"artifact://missing"`, which then
        travelled as the episode's answer — the exact outcome `load` raises to
        prevent, defeated one function away by its only caller. Scored, it reads
        as a delivered answer of twenty characters; read by a person, it is a
        model that named a file it never wrote.

        The raise reaches the engine, where an unresolvable delivery is an
        episode that did not deliver.
        """
        if isinstance(value, str) and self.parse_ref(value):
            return self.load(value)
        return value

    def manifest(self) -> list[dict[str, Any]]:
        """What this store holds, as plain data for the episode record. Sizes and hashes
        only — an artifact exists precisely because its value did not belong in the
        record.
        """
        return [
            {"name": a.name, "ref": a.ref, "chars": a.chars,
             "sha256": a.sha256, "kind": a.kind}
            for a in self.saved.values()
        ]
