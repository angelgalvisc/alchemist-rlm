"""Model-declared presentation grammar inferred from the public question only.

This is an experimental assistance surface, not an answer scorer.  The
inference call never receives context, a candidate answer, gold data or scores.
It produces a small declarative grammar that can be frozen before the task is
solved and used by a local, deterministic linter later.
"""

from __future__ import annotations

import hashlib
import json
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping


SPEC_SYSTEM = (
    "Extract only the explicitly requested output presentation from the user "
    "question. Do not solve the task and do not infer answer content. Call exactly "
    "one declaration tool and return no prose. Read the final "
    "output sentence especially carefully: words such as format, separated, "
    "per line, JSON, table, or fields are explicit presentation requirements. "
    "Use free_text only when no machine-checkable presentation is requested."
)

SPEC_USER = """Declare the output presentation requested by this question.
Choose records for repeated rows or items with explicit syntax, fields, or
separators. Copy literal delimiters and separators from the question exactly;
spaces inside a shown separator are literal characters and must be preserved.
Describe only the shape of the answer, never which answers are correct. Call
exactly one declaration tool.

QUESTION:
{question}"""

_RECORD_PROPERTIES = {
    "record_separator": {"type": "string", "enum": ["newline"]},
    "prefix": {"type": "string", "maxLength": 8},
    "suffix": {"type": "string", "maxLength": 8},
    "field_separator": {"type": "string", "minLength": 1, "maxLength": 8},
    "fields": {
        "type": "array", "minItems": 1, "maxItems": 8,
        "items": {"type": "string", "enum": ["integer", "number", "string"]},
        "description": (
            "Surface field types. Use integer for digit-only identifiers, especially "
            "when the question says lower/higher or numeric ordering"
        ),
    },
    "ordering": {"type": "string", "enum": ["none", "numeric_ascending"]},
    "duplicates": {"type": "boolean"},
    "allow_empty": {"type": "boolean"},
    "additional_text": {"type": "boolean"},
}

FORMAT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "declare_records",
            "description": (
                "Choose for repeated rows or items when the question specifies their "
                "per-record syntax, fields, or separator. Copy literal delimiters"
            ),
            "parameters": {
                "type": "object", "properties": _RECORD_PROPERTIES,
                "required": list(_RECORD_PROPERTIES), "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "declare_json",
            "description": "Choose only when JSON output is explicitly requested",
            "parameters": {
                "type": "object",
                "properties": {"root": {"type": "string", "enum": [
                    "object", "array", "scalar"]}},
                "required": ["root"], "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "declare_scalar",
            "description": (
                "Choose only when exactly one typed value and no prose is requested"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "value_type": {"type": "string", "enum": [
                        "integer", "number", "string"]},
                    "additional_text": {"type": "boolean"},
                },
                "required": ["value_type", "additional_text"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "declare_free_text",
            "description": (
                "Choose only when the question gives no machine-checkable output "
                "syntax, fields, container, or separators"
            ),
            "parameters": {
                "type": "object", "properties": {},
                "additionalProperties": False,
            },
        },
    },
]

_TOOL_KINDS = {
    "declare_records": "records",
    "declare_json": "json",
    "declare_scalar": "scalar",
    "declare_free_text": "free_text",
}

_FIELD_TYPES = frozenset({"integer", "number", "string"})


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _first_json_object(text: str) -> Mapping[str, Any]:
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping):
            return value
    raise ValueError("the format inference reply contained no JSON object")


def normalize_presentation_spec(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and canonicalise the bounded presentation DSL."""
    if str(value.get("version")) != "1":
        raise ValueError("presentation spec version must be 1")
    kind = value.get("kind")
    if kind == "free_text":
        return {"version": 1, "kind": "free_text"}
    if kind == "json":
        root = value.get("root")
        if root not in {"object", "array", "scalar"}:
            raise ValueError("JSON root must be object, array, or scalar")
        return {"version": 1, "kind": "json", "root": root}
    if kind == "scalar":
        value_type = value.get("value_type")
        if value_type not in _FIELD_TYPES:
            raise ValueError("scalar value_type must be integer, number, or string")
        if value.get("additional_text") is not False:
            raise ValueError("scalar additional_text must be false")
        return {
            "version": 1, "kind": "scalar", "value_type": value_type,
            "additional_text": False,
        }
    if kind != "records":
        raise ValueError("presentation spec kind is not supported")
    if value.get("record_separator") != "newline":
        raise ValueError("records currently require newline separation")
    prefix, suffix = value.get("prefix"), value.get("suffix")
    field_separator = value.get("field_separator")
    if not all(isinstance(item, str) and len(item) <= 8
               for item in (prefix, suffix, field_separator)):
        raise ValueError("record delimiters must be strings of at most 8 characters")
    if field_separator == "":
        raise ValueError("field_separator cannot be empty")
    fields = value.get("fields")
    if (not isinstance(fields, list) or not 1 <= len(fields) <= 8
            or any(field not in _FIELD_TYPES for field in fields)):
        raise ValueError("records require 1-8 supported field types")
    ordering = value.get("ordering", "none")
    if ordering not in {"none", "numeric_ascending"}:
        raise ValueError("record ordering is not supported")
    if ordering == "numeric_ascending" and len(fields) != 2:
        raise ValueError("numeric_ascending requires exactly two fields")
    for boolean in ("duplicates", "allow_empty", "additional_text"):
        if not isinstance(value.get(boolean), bool):
            raise ValueError(f"records {boolean} must be boolean")
    if value["additional_text"]:
        raise ValueError("records with additional text are not machine-checkable")
    return {
        "version": 1,
        "kind": "records",
        "record_separator": "newline",
        "prefix": prefix,
        "suffix": suffix,
        "field_separator": field_separator,
        "fields": list(fields),
        "ordering": ordering,
        "duplicates": bool(value["duplicates"]),
        "allow_empty": bool(value["allow_empty"]),
        "additional_text": False,
    }


def parse_presentation_spec(text: str) -> dict[str, Any]:
    return normalize_presentation_spec(_first_json_object(text))


def infer_presentation_spec(client: Any, question: str, *, max_tokens: int = 768,
                            ) -> dict[str, Any]:
    """Ask the same backend for a grammar without exposing task content."""
    prompt = SPEC_USER.replace("{question}", question)
    reply = client.complete([
        {"role": "system", "content": SPEC_SYSTEM},
        {"role": "user", "content": prompt},
    ], tools=FORMAT_TOOLS, max_tokens=max_tokens)
    record: dict[str, Any] = {
        "mode": "model_inferred_question_only_v1",
        "system_sha256": _sha256(SPEC_SYSTEM),
        "request_sha256": _sha256(prompt),
        "question_sha256": _sha256(question),
        "reply_sha256": _sha256(reply.content),
        "reply_chars": len(reply.content),
        "reply_text": reply.content,
        "served_model": reply.served_model,
        "tool_sha256": _sha256(json.dumps(
            FORMAT_TOOLS, sort_keys=True, separators=(",", ":"))),
    }
    try:
        calls = [call for call in reply.tool_calls
                 if (call.get("function") or {}).get("name") in _TOOL_KINDS]
        if len(calls) == 1:
            function = calls[0].get("function") or {}
            arguments = function.get("arguments") or "{}"
            arguments_text = (arguments if isinstance(arguments, str) else
                              json.dumps(arguments, sort_keys=True,
                                         separators=(",", ":")))
            record.update({
                "declaration_tool": function["name"],
                "declaration_arguments_text": arguments_text,
                "declaration_arguments_sha256": _sha256(arguments_text),
            })
            value = json.loads(arguments) if isinstance(arguments, str) else arguments
            if not isinstance(value, Mapping):
                raise TypeError("format declaration arguments must be an object")
            record["declaration_arguments"] = dict(value)
            value = {**value, "version": 1, "kind": _TOOL_KINDS[function["name"]]}
            spec = normalize_presentation_spec(value)
            record["inference_channel"] = "tool_call"
        elif not calls:
            spec = parse_presentation_spec(reply.content)
            record["inference_channel"] = "text_fallback"
        else:
            raise ValueError("format inference returned multiple declarations")
    except (ValueError, json.JSONDecodeError, TypeError) as error:
        record.update(status="invalid", error=str(error), spec=None)
    else:
        canonical = json.dumps(spec, sort_keys=True, separators=(",", ":"))
        record.update(status="ok", spec=spec, spec_sha256=_sha256(canonical))
    return record


def _value_matches(value: str, expected: str) -> bool:
    if expected == "string":
        return True
    try:
        int(value)
    except ValueError:
        if expected == "integer":
            return False
    else:
        return True
    try:
        float(value)
    except ValueError:
        return False
    return expected == "number"


def check_presentation(text: Any, spec: Mapping[str, Any]) -> dict[str, Any]:
    """Lint candidate text against a frozen inferred spec without rewriting it."""
    if not isinstance(text, str):
        return {"valid": False, "issues": [{
            "code": "text_required", "message": "candidate must be str", "count": 1,
        }]}
    normalized = normalize_presentation_spec(spec)
    kind = normalized["kind"]
    if kind == "free_text":
        return {"valid": True, "issues": []}
    if kind == "json":
        try:
            value = json.loads(text)
        except json.JSONDecodeError as error:
            return {"valid": False, "issues": [{
                "code": "invalid_json", "message": str(error), "count": 1,
            }]}
        root = ("object" if isinstance(value, dict) else
                "array" if isinstance(value, list) else "scalar")
        issues = ([] if root == normalized["root"] else [{
            "code": "json_root", "message": f"expected {normalized['root']} root",
            "count": 1,
        }])
        return {"valid": not issues, "issues": issues}
    if kind == "scalar":
        valid = "\n" not in text and _value_matches(text, normalized["value_type"])
        return {"valid": valid, "issues": [] if valid else [{
            "code": "invalid_scalar",
            "message": f"expected only one {normalized['value_type']} value",
            "count": 1,
        }]}

    if text == "":
        valid = normalized["allow_empty"]
        return {"valid": valid, "issues": [] if valid else [{
            "code": "records_required", "message": "at least one record is required",
            "count": 1,
        }]}
    body = text[:-1] if text.endswith("\n") else text
    lines = body.split("\n")
    buckets: dict[str, list[str]] = {
        "record_delimiters": [], "field_count": [], "field_type": [],
        "numeric_order_type": [], "record_order": [], "duplicate_record": [],
    }
    seen: dict[tuple[str, ...], int] = {}
    prefix, suffix = normalized["prefix"], normalized["suffix"]
    separator = normalized["field_separator"]
    for number, line in enumerate(lines, 1):
        excerpt = line if len(line) <= 120 else f"{line[:119]}…"
        example = f"line {number}: {excerpt!r}"
        delimited = line.startswith(prefix) and line.endswith(suffix)
        if not delimited:
            buckets["record_delimiters"].append(example)
        inner = line[len(prefix):] if prefix and line.startswith(prefix) else line
        if suffix and inner.endswith(suffix):
            inner = inner[:-len(suffix)]
        values = inner.split(separator)
        if len(values) != len(normalized["fields"]):
            buckets["field_count"].append(example)
            continue
        clean = tuple(value.strip() for value in values)
        if any(not _value_matches(value, expected)
               for value, expected in zip(clean, normalized["fields"])):
            buckets["field_type"].append(example)
            continue
        if normalized["ordering"] == "numeric_ascending":
            try:
                numeric = (Decimal(clean[0]), Decimal(clean[1]))
            except InvalidOperation:
                buckets["numeric_order_type"].append(example)
            else:
                if not all(value.is_finite() for value in numeric):
                    buckets["numeric_order_type"].append(example)
                elif numeric[0] >= numeric[1]:
                    buckets["record_order"].append(example)
        if not normalized["duplicates"] and clean in seen:
            buckets["duplicate_record"].append(
                f"{example} repeats line {seen[clean]}")
        else:
            seen[clean] = number
    messages = {
        "record_delimiters": (
            f"each cited line must start with {prefix!r} and end with {suffix!r}"),
        "field_count": (
            f"each record must contain {len(normalized['fields'])} fields separated "
            f"by the literal {separator!r}"),
        "field_type": "record fields do not match the declared types",
        "numeric_order_type": (
            "numeric ordering requires two finite numeric field values"),
        "record_order": "numeric fields must be in strictly ascending order",
        "duplicate_record": "records must not repeat an earlier record",
    }
    issues = [
        {"code": code, "message": messages[code], "count": len(examples),
         "examples": examples[:3]}
        for code, examples in buckets.items() if examples
    ]
    return {"valid": not issues, "issues": issues}


def render_presentation(value: Any, spec: Mapping[str, Any]) -> str:
    """Serialize a value with the frozen question-derived presentation DSL.

    This is deliberately narrower than a general pretty-printer. It accepts
    only primitive scalar fields and bounded container shapes, never submits
    its result, and validates its own output through the same public linter.
    """
    normalized = normalize_presentation_spec(spec)
    kind = normalized["kind"]
    if kind == "free_text":
        if not isinstance(value, str):
            raise TypeError("free_text presentation requires a str value")
        rendered = value
    elif kind == "json":
        rendered = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    elif kind == "scalar":
        if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
            raise TypeError("scalar presentation requires a primitive scalar value")
        rendered = str(value)
    else:
        if not isinstance(value, (list, tuple)):
            raise TypeError("records presentation requires a list or tuple of records")
        prefix = normalized["prefix"]
        suffix = normalized["suffix"]
        separator = normalized["field_separator"]
        seen: set[tuple[str, ...]] = set()
        lines: list[str] = []
        for number, record in enumerate(value, 1):
            if isinstance(record, str):
                # A model may freeze records after an intermediate formatting
                # step (for example, ``"12, 34"``) rather than as two-element
                # lists. Parse that representation only through the literal
                # frozen grammar and only when it has exactly the declared
                # number of fields. No heuristic delimiter or content recovery
                # is permitted here.
                inner = record
                has_prefix = bool(prefix) and inner.startswith(prefix)
                has_suffix = bool(suffix) and inner.endswith(suffix)
                if has_prefix != has_suffix:
                    raise ValueError(
                        f"record {number} has incomplete frozen delimiters")
                if has_prefix and has_suffix:
                    inner = inner[len(prefix):len(inner) - len(suffix)]
                parsed = tuple(part.strip() for part in inner.split(separator))
                if len(parsed) != len(normalized["fields"]):
                    raise ValueError(
                        f"record {number} cannot be split unambiguously into "
                        f"{len(normalized['fields'])} fields")
                record = parsed
            elif not isinstance(record, (list, tuple)):
                raise TypeError(
                    f"record {number} must be text, a list, or a tuple")
            if len(record) != len(normalized["fields"]):
                raise ValueError(
                    f"record {number} must contain {len(normalized['fields'])} fields")
            clean: list[str] = []
            for field, expected in zip(record, normalized["fields"]):
                if isinstance(field, bool) or not isinstance(
                        field, (str, int, float, Decimal)):
                    raise TypeError(
                        f"record {number} fields must be primitive scalars")
                text = str(field)
                if not _value_matches(text, expected):
                    raise ValueError(
                        f"record {number} field {text!r} is not {expected}")
                if ("\n" in text or separator in text
                        or (prefix and text.startswith(prefix))
                        or (suffix and text.endswith(suffix))):
                    raise ValueError(
                        f"record {number} field cannot be represented unambiguously")
                clean.append(text)
            if normalized["ordering"] == "numeric_ascending":
                numeric = (Decimal(clean[0]), Decimal(clean[1]))
                if not all(item.is_finite() for item in numeric):
                    raise ValueError(f"record {number} numeric fields must be finite")
                if numeric[0] == numeric[1]:
                    raise ValueError(
                        f"record {number} cannot satisfy strict numeric ordering")
                if numeric[0] > numeric[1]:
                    clean.reverse()
            key = tuple(clean)
            if not normalized["duplicates"] and key in seen:
                continue
            seen.add(key)
            lines.append(f"{prefix}{separator.join(clean)}{suffix}")
        if not lines and not normalized["allow_empty"]:
            raise ValueError("the frozen presentation requires at least one record")
        rendered = "\n".join(lines)
        if lines:
            rendered += "\n"

    report = check_presentation(rendered, normalized)
    if not report["valid"]:
        codes = ", ".join(
            str(issue.get("code")) for issue in report.get("issues") or [])
        raise ValueError(f"rendered presentation is invalid: {codes or 'unknown issue'}")
    return rendered
