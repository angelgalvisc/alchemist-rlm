"""Presentation contracts that never judge the computed answer.

The runtime owns transport and state transitions.  A task adapter may own a
deterministic validator for public output syntax.  The validator receives text
only: it cannot see a typed answer, a gold answer, or a score.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

OUTPUT_MODES = frozenset({"raw", "validate_only", "validate_repair", "constrained"})


@dataclass(frozen=True)
class ValidationIssue:
    """One bounded, machine-identifiable presentation defect.

    Validators may inspect thousands of rows.  The count and a few examples
    preserve that evidence without making the model read the same sentence
    thousands of times.
    """

    code: str
    message: str
    count: int = 1
    examples: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "count": self.count,
            "examples": list(self.examples),
        }


@dataclass(frozen=True)
class ValidationResult:
    """The complete authority a structural validator has."""

    valid: bool
    errors: tuple[str, ...] = ()
    issues: tuple[ValidationIssue, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"valid": self.valid, "errors": list(self.errors)}
        if self.issues:
            result["issues"] = [issue.to_dict() for issue in self.issues]
        return result

    def feedback(self, *, max_issues: int = 8, max_examples: int = 3,
                 max_chars: int = 4_000) -> str:
        """A bounded rendering for a model; :meth:`to_dict` stays complete."""
        return validation_feedback(
            self.to_dict(), max_issues=max_issues, max_examples=max_examples,
            max_chars=max_chars)


def validation_feedback(record: Mapping[str, Any], *, max_issues: int = 8,
                        max_examples: int = 3, max_chars: int = 4_000) -> str:
    """Render complete validation telemetry into bounded model feedback."""

    issues = [
        ValidationIssue(
            str(issue.get("code", "invalid_output")),
            str(issue.get("message", "presentation is invalid")),
            int(issue.get("count", 1)),
            tuple(map(str, issue.get("examples") or ())),
        )
        for issue in (record.get("issues") or ())
        if isinstance(issue, Mapping)
    ]
    if not issues:
        # Backward-compatible validators still return strings.  Exact
        # duplicates can be grouped safely; unlike regex-based rewriting,
        # this never guesses which part of a message is a location.
        counts = Counter(map(str, record.get("errors") or ()))
        issues = [
            ValidationIssue(f"legacy_{index}", message, count)
            for index, (message, count) in enumerate(counts.items(), 1)
        ]
    lines: list[str] = []
    for issue in issues[:max_issues]:
        suffix = f" ({issue.count} occurrences)" if issue.count != 1 else ""
        examples = issue.examples[:max_examples]
        if examples:
            suffix += "; examples: " + ", ".join(examples)
        lines.append(f"- [{issue.code}] {issue.message}{suffix}")
    omitted = max(0, len(issues) - max_issues)
    if omitted:
        lines.append(f"- {omitted} additional issue classes omitted")
    text = "\n".join(lines)
    if len(text) > max_chars:
        marker = f"\n- diagnostics truncated to {max_chars} characters"
        text = text[:max(0, max_chars - len(marker))] + marker
    return text


@dataclass(frozen=True)
class PresentationBinding:
    """Prove that presentation text represents the committed answer value.

    This authority may compare the model's own committed value with public
    output syntax.  It never receives gold data and therefore cannot decide
    whether the answer is correct.
    """

    name: str
    version: str
    specification: dict[str, Any]
    equivalent: Callable[[Any, str], ValidationResult]

    def validate(self, value: Any, text: str) -> ValidationResult:
        result = self.equivalent(canonical_answer_value(value), text)
        if not isinstance(result, ValidationResult):
            raise TypeError("a presentation binding must return ValidationResult")
        if result.valid and (result.errors or result.issues):
            raise ValueError("a valid presentation binding cannot carry diagnostics")
        if not result.valid and not (result.errors or result.issues):
            raise ValueError("an invalid presentation binding must explain why")
        return result

    @property
    def sha256(self) -> str:
        try:
            source = inspect.getsource(self.equivalent)
        except (OSError, TypeError):
            source = f"{self.equivalent.__module__}.{self.equivalent.__qualname__}"
        canonical = json.dumps({
            "name": self.name,
            "version": self.version,
            "specification": self.specification,
            "equivalent_source": source,
        }, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    def manifest(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "specification": self.specification,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class TerminalPolicy:
    """Bound content commitment separately from presentation protocol."""

    max_commit_attempts: int = 2
    max_partial_recovery_attempts: int = 1
    max_presentation_attempts: int = 3
    max_protocol_retries: int = 2
    max_presentation_build_turns: int = 2
    max_presentation_commit_reserve: int = 1
    max_presentation_tokens: int = 1_024
    max_feedback_chars: int = 4_000

    def __post_init__(self) -> None:
        for name in ("max_commit_attempts", "max_partial_recovery_attempts",
                     "max_presentation_attempts",
                     "max_protocol_retries", "max_presentation_build_turns",
                     "max_presentation_commit_reserve",
                     "max_presentation_tokens",
                     "max_feedback_chars"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.max_partial_recovery_attempts > 1:
            raise ValueError("max_partial_recovery_attempts cannot exceed one")
        if self.max_presentation_commit_reserve > 1:
            raise ValueError(
                "max_presentation_commit_reserve cannot exceed one")

    def to_dict(self) -> dict[str, int]:
        return {
            "max_commit_attempts": self.max_commit_attempts,
            "max_partial_recovery_attempts": self.max_partial_recovery_attempts,
            "max_presentation_attempts": self.max_presentation_attempts,
            "max_protocol_retries": self.max_protocol_retries,
            "max_presentation_build_turns": self.max_presentation_build_turns,
            "max_presentation_commit_reserve":
                self.max_presentation_commit_reserve,
            "max_presentation_tokens": self.max_presentation_tokens,
            "max_feedback_chars": self.max_feedback_chars,
        }


@dataclass(frozen=True)
class OutputContract:
    """A predeclared, deterministic contract over presentation text only."""

    name: str
    version: str
    specification: dict[str, Any]
    validator: Callable[[str], ValidationResult]
    binding: PresentationBinding | None = None

    def validate(self, text: str) -> ValidationResult:
        if not isinstance(text, str):
            raise TypeError("OutputContract.validate accepts text only")
        result = self.validator(text)
        if not isinstance(result, ValidationResult):
            raise TypeError("an output validator must return ValidationResult")
        if result.valid and (result.errors or result.issues):
            raise ValueError("a valid ValidationResult cannot carry diagnostics")
        if not result.valid and not (result.errors or result.issues):
            raise ValueError("an invalid ValidationResult must explain why")
        return result

    @property
    def sha256(self) -> str:
        try:
            source = inspect.getsource(self.validator)
        except (OSError, TypeError):
            source = f"{self.validator.__module__}.{self.validator.__qualname__}"
        payload = {
            "name": self.name,
            "version": self.version,
            "specification": self.specification,
            "validator_source": source,
            "binding": self.binding.manifest() if self.binding else None,
        }
        canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False,
                               separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    def manifest(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "specification": self.specification,
            "binding": self.binding.manifest() if self.binding else None,
            "sha256": self.sha256,
        }


def validate_output_mode(mode: str, contract: OutputContract | None) -> None:
    if mode not in OUTPUT_MODES:
        raise ValueError(f"output_mode must be one of {sorted(OUTPUT_MODES)}, got {mode!r}")
    if mode == "raw" and contract is not None:
        raise ValueError("output_mode='raw' cannot carry an OutputContract")
    if mode != "raw" and contract is None:
        raise ValueError(f"output_mode={mode!r} requires an OutputContract")


def canonical_answer_value(value: Any, *, path: str = "answer_value") -> Any:
    """Normalize one answer to the JSON data model without stringifying it.

    Tuples become arrays because that is what the process boundary can preserve.
    Unsupported nested values are refused at their exact path; ``default=str``
    is deliberately absent.
    """

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError(f"{path} must be transportable JSON data, got non-finite float")
        return value
    if isinstance(value, (list, tuple)):
        return [canonical_answer_value(item, path=f"{path}[{index}]")
                for index, item in enumerate(value)]
    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"{path} must be transportable JSON data; dict key {key!r} is not str"
                )
            normalized[key] = canonical_answer_value(item, path=f"{path}.{key}")
        return normalized
    raise TypeError(
        f"{path} must be transportable JSON data, got {type(value).__name__}"
    )


def canonical_json(value: Any) -> str:
    normalized = canonical_answer_value(value)
    return json.dumps(normalized, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False)


def full_sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def answer_value_record(value: Any, artifacts: Any, *, inline_limit: int = 20_000) -> dict[str, Any]:
    """Persist a canonical answer without ever disguising JSON as plain text."""
    normalized = canonical_answer_value(value)
    encoded = canonical_json(normalized)
    sha256 = full_sha256(encoded)
    json_type = (
        "null" if normalized is None else
        "boolean" if isinstance(normalized, bool) else
        "integer" if isinstance(normalized, int) else
        "number" if isinstance(normalized, float) else
        "string" if isinstance(normalized, str) else
        "array" if isinstance(normalized, list) else
        "object"
    )
    base = {"json_type": json_type, "chars": len(encoded), "sha256": sha256}
    if len(encoded) <= inline_limit:
        return {**base, "storage": "inline", "value": normalized}
    name = f"answer-value-{sha256[:16]}"
    # Always persist the canonical JSON bytes. ArtifactStore deliberately saves
    # strings as raw text, so passing a large string value here would otherwise
    # produce non-JSON bytes while the record claimed a canonical JSON digest.
    ref = artifacts.save(name, encoded)
    return {**base, "storage": "artifact", "ref": ref}


def read_answer_value_record(record: dict[str, Any], artifact_root: Path | str) -> Any:
    """Recover and verify the canonical value used by an offline scorer."""
    storage = record.get("storage")
    if storage == "inline":
        value = canonical_answer_value(record.get("value"))
    elif storage == "artifact":
        ref = record.get("ref")
        if not isinstance(ref, str) or not ref.startswith("artifact://"):
            raise ValueError("answer value artifact record has no valid ref")
        name = ref.removeprefix("artifact://")
        if not name or "/" in name or "\\" in name:
            raise ValueError("answer value artifact ref is not a safe name")
        path = Path(artifact_root) / f"{name}.txt"
        raw = path.read_text(encoding="utf-8")
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError:
            # Read-only compatibility for episodes written by the first
            # implementation, which stored large string values as raw bytes.
            # The canonical digest below still decides whether those bytes are
            # the recorded value; non-string records never take this path.
            if record.get("json_type") != "string":
                raise
            decoded = raw
        value = canonical_answer_value(decoded)
    else:
        raise ValueError(f"unknown answer value storage {storage!r}")
    encoded = canonical_json(value)
    if full_sha256(encoded) != record.get("sha256"):
        raise ValueError("answer value record digest mismatch")
    if len(encoded) != record.get("chars"):
        raise ValueError("answer value record length mismatch")
    return value


def text_record(text: str | None, artifacts: Any, *, label: str,
                inline_limit: int = 20_000) -> dict[str, Any] | None:
    if text is None:
        return None
    sha256 = full_sha256(text)
    base = {"chars": len(text), "sha256": sha256}
    if len(text) <= inline_limit:
        return {**base, "storage": "inline", "text": text}
    name = f"{label}-{sha256[:16]}"
    ref = artifacts.save(name, text)
    return {**base, "storage": "artifact", "ref": ref}


def read_text_record(record: dict[str, Any], artifact_root: Path | str) -> str:
    """Recover and verify one initial, candidate or final presentation."""
    storage = record.get("storage")
    if storage == "inline":
        text = record.get("text")
        if not isinstance(text, str):
            raise ValueError("inline text record has no text")
    elif storage == "artifact":
        ref = record.get("ref")
        if not isinstance(ref, str) or not ref.startswith("artifact://"):
            raise ValueError("text artifact record has no valid ref")
        name = ref.removeprefix("artifact://")
        if not name or "/" in name or "\\" in name:
            raise ValueError("text artifact ref is not a safe name")
        text = (Path(artifact_root) / f"{name}.txt").read_text(encoding="utf-8")
    else:
        raise ValueError(f"unknown text record storage {storage!r}")
    if len(text) != record.get("chars"):
        raise ValueError("text record length mismatch")
    if full_sha256(text) != record.get("sha256"):
        raise ValueError("text record digest mismatch")
    return text
