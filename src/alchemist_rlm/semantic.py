"""`semantic_map`: one typed value per item, and coverage the model cannot claim.

    item  →  bounded semantic inference  →  validated value

This is the leaf operation of the RLM. Everything above it — partitioning,
recursion, aggregation — is the model's Python. What happens here is the part a
model cannot be trusted to report on itself: whether every item was actually
sent, whether every item came back, and whether what came back means anything.

**Why it replaced a binary search.** The operation this grew out of was
`semantic_search`, which returned yes or no per unit and carried a semantic rule
inside it: *"negated, hypothetical, averted or no-effect mentions do not count.
Paraphrases and implied statements do count."* That rule was right for the
incident corpus it was written against and wrong for "find every event that was
averted", "list the negated claims", "classify these into six categories". A
general operation must not decide what counts. So the mechanism lives here, the
meaning lives in the caller's `instruction`, and the shape lives in `schema`.

**Why the types stop where they do.** The engine is type-independent apart from
`validate_value`, so a further type is a branch on one function rather than a
rewrite, and this repository ships what a measured need asked for. `boolean`
and `enum` came first. `object` came third, for a failure class the twenty
benchmark queries named: a sweep returns one judged value per item beside the
item's own source prose, so a question wanting a judgement *and* a literal
attribute left the caller parsing that prose — and the five queries shaped that
way cleared their floor once between them, against three of five for the rest.

`object` was held back for a real reason and the reason was answered rather
than waived. The contract is one line per item against a 1,024-token sub-call
budget — already raised once from 512, because a truncated reply reads as
*missing items* and burns the retry — and a record is several times the width
of a label. So the item count per fragment is no longer a constant: it is
derived from the schema's own widest conforming line, and a wider schema asks
for fewer items. `array` remains unimplemented, and unneeded.

**Why `null` is opt-in.** A typed "no result" is right; without one, models
invent values. It is also a typed escape hatch, and this project measured what
one of those costs: an instruction ending "if the excerpt does not contain the
answer, reply exactly NONE" made 28 of 28 subcalls reply NONE, including the
three that held the evidence. So it exists, and a schema has to ask for it.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Iterable

# `item N: value` — the value is whatever follows, parsed against the schema.
# The line framing is kept from the operation this replaces because it is the
# part that demonstrably works with a 4B: a whole-reply JSON array does not.
LINE = re.compile(r"^\s*item\s*(\d+)\s*[:\-]\s*(.*?)\s*$", re.MULTILINE)

TRUE_WORDS = ("yes", "true", "si", "sí")
FALSE_WORDS = ("no", "false")
NULL_WORDS = ("null", "none", "n/a")

IMPLEMENTED = ("boolean", "enum", "object")

# An object is a few decisions per item, not a record format. Past a handful of
# fields the fragment shrinks until the sweep costs more sub-calls than joining
# the fields saved, which is the trade this exists to win.
MAX_OBJECT_FIELDS = 5
# The sub-call reply budget is 1,024 tokens; at roughly four characters a token
# that is the room a fragment's whole reply has. `RESERVED` is the preamble a
# model tends to write before the first line.
SUBCALL_REPLY_CHARS = 4 * 1024
RESERVED_REPLY_CHARS = 400
LINE_PREFIX_CHARS = len("item 000: ")
# A copied field is an identifier, a date, a short name — not a sentence.
# The cap is what keeps one runaway field from eating the whole reply
# budget and truncating the items after it.
MAX_TEXT_FIELD_CHARS = 60
# How many permitted values the contract writes out in full. Past this the rest
# are named on one line: a fifty-label set would otherwise bury the source text
# under its own instructions.
SHOWN_VALUES = 12


SHAPE = ("semantic_map(instruction, schema, items=None) where schema is "
         "{'type': 'string', 'enum': [...]}, {'type': 'boolean'}, or an "
         "object of those: {'type': 'object', 'properties': "
         "{'<name>': {'type': 'string', 'enum': [...]}, "
         "'<name>': {'type': 'string'}}, 'required': [...], "
         "'additionalProperties': False}")


class SchemaError(ValueError):
    """The schema is malformed, and the message says what a good one looks like.

    Every refusal in this harness carries a counteroffer, because `probe_08`
    measured that a bare refusal changes nothing: the model was told a repeated
    block was refused and simply emitted it again. This exception was written
    without one and cost what the doctrine predicts — on the automatic
    evaluation the model produced three consecutive `schema must be a dict, got
    list` errors, passing its label list in the schema position, before finding
    the order by trial.
    """

    def __init__(self, problem: str):
        super().__init__(f"{problem}. Call it as: {SHAPE}")


def check_schema(schema: Any) -> dict[str, Any]:
    """Normalise a JSON Schema fragment, or refuse it by name.

    JSON Schema syntax from the start, not a private dialect: a contract is
    expensive to migrate once results have been published against it.

        {"type": "boolean"}
        {"type": "string", "enum": [...]}
        {"type": ["string", "null"], "enum": [...]}

    Nullability is the intersection of the two constraints, because that is
    what JSON Schema says. A value satisfies the schema by satisfying `type`
    *and* `enum`, so:

        {"type": ["string", "null"], "enum": ["a", "b"]}   -> "a", "b"
        {"type": "string",           "enum": ["a", None]}  -> "a"
        {"type": ["string", "null"], "enum": ["a", None]}  -> "a", null

    All three are valid schemas; only the third admits null. Two earlier
    versions of this function got that wrong in opposite directions — the first
    accepted null whenever `type` allowed it, ignoring the enum; the second
    rejected the mixed forms as inconsistent, which they are not. Both were
    deviations from the syntax adopted for portability, and deviating in
    silence defeats the reason for adopting it.
    """
    if not isinstance(schema, dict):
        raise SchemaError(f"schema must be a dict, got {type(schema).__name__}")
    if "type" not in schema and ("instruction" in schema or "items" in schema):
        raise SchemaError(
            "this looks like the arguments passed as one dict rather than "
            "positionally"
        )
    declared = schema.get("type")
    if declared is None:
        raise SchemaError("schema needs a `type`, e.g. {'type': 'boolean'}")

    types = [declared] if isinstance(declared, str) else list(declared)
    if not all(isinstance(t, str) for t in types):
        raise SchemaError("`type` must be a string or a list of strings")
    nullable = "null" in types
    rest = [t for t in types if t != "null"]
    if len(rest) != 1:
        raise SchemaError(
            "`type` must name exactly one type, optionally with 'null': "
            f"got {declared!r}"
        )
    base = rest[0]

    values = schema.get("enum")
    if values is not None:
        if base != "string":
            raise SchemaError(f"`enum` goes with type 'string', not {base!r}")
        if not isinstance(values, list) or not values:
            raise SchemaError("`enum` must be a non-empty list")
        listed_null = any(v is None for v in values)
        named = [v for v in values if v is not None]
        if not named:
            raise SchemaError("`enum` needs at least one non-null value")
        if not all(isinstance(v, str) and v.strip() for v in named):
            raise SchemaError("every `enum` value must be a non-empty string")
        lowered = [v.strip().lower() for v in named]
        if len(set(lowered)) != len(lowered):
            raise SchemaError("`enum` values must differ once case is ignored")
        # Effective nullability is the intersection, not an agreement to check.
        # `type` and `enum` are both assertions and a value satisfies the schema
        # only by satisfying both, so null is admitted exactly when both admit
        # it. A previous version rejected the mixed forms as "inconsistent";
        # they are ordinary valid schemas that happen to exclude null, and
        # refusing them was a second deviation from the standard on top of the
        # first.
        return {"base": "enum", "values": named,
                "nullable": nullable and listed_null}

    if base == "boolean":
        return {"base": "boolean", "values": None, "nullable": nullable}
    if base == "object":
        return _object_schema(schema, nullable=nullable)
    if base == "string":
        raise NotImplementedError(
            "schema type 'string' without `enum` is not implemented yet; "
            "free-form extraction has not been measured on this runtime"
        )
    if base in ("number", "integer", "array"):
        raise NotImplementedError(
            f"schema type {base!r} is not implemented yet; it needs a measured "
            "fragment size and sub-call token budget before it can be trusted"
        )
    raise SchemaError(f"unknown schema type {base!r}")


def _object_schema(schema: dict[str, Any], *, nullable: bool) -> dict[str, Any]:
    """A small record per item: a few scalar fields, all of them required.

    This exists for a measured failure class, not for generality. A sweep
    returns one *judged* value per item beside the item's own `source` text, so
    a question needing a judgement AND a literal attribute — a label, and the
    date sitting in the same line — forced the caller to parse the attribute
    back out of that prose. Across the twenty benchmark queries, the five that
    needed a second attribute cleared their floor one time in five, against
    three in five for the rest, and the traces name the cause every time: one
    spent nine of its fifteen turns fighting a regex and never made a sub-call,
    another wrote `split("User:")[1].strip()` and turned its ids into whole
    lines. Asking the leaf for both fields at once removes the parse.

    Deliberately narrow. Properties must be types this engine already validates
    per item — enum or boolean — so an object is a fixed set of decisions the
    contract can check one by one, never a nested structure. Every property is
    required, because a field the model may omit cannot be told from one it
    forgot, and the retry path needs that difference to ask again.
    """
    properties = schema.get("properties")
    if not isinstance(properties, dict) or not properties:
        raise SchemaError(
            "an object schema needs a non-empty `properties`, e.g. "
            "{'type': 'object', 'properties': {'label': {'type': 'string', "
            "'enum': [...]}}, 'required': ['label'], "
            "'additionalProperties': False}")
    if len(properties) > MAX_OBJECT_FIELDS:
        raise SchemaError(
            f"an object may declare at most {MAX_OBJECT_FIELDS} properties; got "
            f"{len(properties)}. Each field shrinks the fragment, and past this "
            "the sweep costs more sub-calls than the joined fields save")
    if schema.get("additionalProperties") not in (False, None):
        raise SchemaError("`additionalProperties` must be False for an object "
                          "schema: a field nobody named cannot be validated")
    required = schema.get("required")
    if required is not None and sorted(required) != sorted(properties):
        missing = sorted(set(properties) - set(required or []))
        raise SchemaError(
            "every property must appear in `required`: a field the model may "
            "omit cannot be told from one it forgot, and the retry needs that "
            f"difference. Not required: {missing}")
    fields: dict[str, Any] = {}
    for name, sub in properties.items():
        if not isinstance(name, str) or not name.strip():
            raise SchemaError(f"property names must be non-empty strings, got {name!r}")
        if _is_free_text(sub):
            fields[name] = {"base": "text", "values": None,
                            "nullable": "null" in _types_of(sub)}
            continue
        try:
            inner = check_schema(sub)
        except NotImplementedError as exc:
            raise SchemaError(
                f"property {name!r}: {exc}. An object property must be an enum, "
                "a boolean, or a plain string copied from the source") from exc
        if inner["base"] == "object":
            raise SchemaError(f"property {name!r} is itself an object; nesting "
                              "is not supported")
        fields[name] = inner
    if all(f["base"] == "text" for f in fields.values()):
        raise SchemaError(
            "an object of only free-text fields is not a judgement; at least "
            "one property must be an enum or a boolean. Copying text out of a "
            "source is what plain Python is for, and it does not need a "
            "sub-model")
    return {"base": "object", "values": None, "nullable": nullable,
            "fields": fields}


def _types_of(schema: Any) -> list[str]:
    declared = schema.get("type") if isinstance(schema, dict) else None
    return [declared] if isinstance(declared, str) else list(declared or [])


def _is_free_text(schema: Any) -> bool:
    """A `{"type": "string"}` property with no `enum` — text copied, not judged.

    Permitted *inside* an object and nowhere else, and the asymmetry is the
    point. The measured failure class is a question needing a judgement AND a
    literal attribute from the same item: the caller could get the judgement,
    and then had to recover the attribute by parsing the item's source prose,
    which is where five of five date-bearing queries came apart. Asking for
    both in one reply removes that parse.

    A free-text field is validated for *shape* only — non-empty, one line,
    under a cap — because there is no declared value set to check it against.
    That is a weaker guarantee than an enum's and the certificate must not
    imply otherwise, so an object schema is refused unless at least one field
    is a real judgement. Alone, these fields would be a sub-model call doing
    work `str.split` does for free.
    """
    if not isinstance(schema, dict) or schema.get("enum") is not None:
        return False
    rest = [t for t in _types_of(schema) if t != "null"]
    return rest == ["string"]


def describe_schema(norm: dict[str, Any]) -> str:
    """How the contract asks for a value. The wording never states what any
    value *means* — that is the caller's instruction, and the whole reason this
    operation exists."""
    if norm["base"] == "boolean":
        answer = "yes or no"
    elif norm["base"] == "text":
        answer = (f"the text itself, copied from the item, "
                  f"one line, at most {MAX_TEXT_FIELD_CHARS} characters")
    elif norm["base"] == "object":
        # Shown as the JSON object it must be, with every field's own
        # permitted values inline. Same doctrine as the enum contract: an
        # earlier version that *named* the shape once got the shape back
        # eighteen times out of nineteen with the prefix dropped, so the
        # format is written out rather than described.
        parts = [f'"{name}": <{describe_schema(sub)}>'
                 for name, sub in norm["fields"].items()]
        answer = "a JSON object on one line, {" + ", ".join(parts) + "}"
    else:
        answer = "exactly one of: " + ", ".join(norm["values"])
    if norm["nullable"]:
        answer += "; or null if it does not apply"
    return answer


def items_per_fragment(norm: dict[str, Any]) -> int:
    """How many items one fragment may ask for, from the schema alone.

    Output size, not input size, is what bounds this. A fragment's reply is one
    line per item against a sub-call budget of 1,024 tokens — a limit already
    raised once from 512 because a truncated reply reads as *missing items* and
    burns the retry. An enum label is a few tokens; a three-field JSON object
    is several times that, so the same item count would run a long fragment
    past the budget and lose its tail.

    Deterministic, and derived rather than tuned: the estimate is the contract's
    own rendering of one line, which is the longest a conforming reply can be
    when every field takes its longest declared value.

    There was a flat ceiling of 42 here and it was a mistake. 42 came from a
    docstring describing the *observed average* items per fragment, and the
    segmenter's own character bound produces segments of 39 to 44 — so a
    ceiling set at the average necessarily chopped the top half, splitting
    segments by one or two items and buying a whole extra sub-call each time.
    Measured on query 2: 19 fragments became 25, while the token budget alone
    allowed 88. The input side is already bounded by `TARGET_CHARS`; what was
    missing was an output bound, and that is what this is. A guard against a
    degenerate schema would have to be set where it cannot bite in the measured
    regime, and there is no evidence for where that is.
    """
    per_line = LINE_PREFIX_CHARS + _widest_value(norm)
    return max(1, (SUBCALL_REPLY_CHARS - RESERVED_REPLY_CHARS) // per_line)


def _widest_value(norm: dict[str, Any]) -> int:
    """Characters in the widest conforming value, without the line prefix.

    The prefix belongs to the line, not to each field: counting it per field
    made a two-field object look 20 characters wider than it is. Wrong in the
    safe direction, but wrong.
    """
    if norm["base"] == "boolean":
        body = 3                                          # "yes"
    elif norm["base"] == "text":
        body = MAX_TEXT_FIELD_CHARS
    elif norm["base"] == "object":
        body = 2 + sum(len(name) + 6 + _widest_value(sub)  # "name": value,
                       for name, sub in norm["fields"].items())
    else:
        body = max(len(v) for v in norm["values"])
    return max(body, 4) if norm["nullable"] else body     # "null"


def _bare(raw: str) -> str:
    """Strip decoration a model puts around a value. Syntactic only: quotes,
    markdown emphasis, a trailing full stop. Never a guess at what was meant."""
    text = (raw or "").strip()
    for _ in range(3):
        stripped = text.strip().strip("`").strip("*").strip('"').strip("'").strip()
        stripped = stripped[:-1].strip() if stripped.endswith(".") else stripped
        if stripped == text:
            break
        text = stripped
    return text


def validate_value(raw: str, norm: dict[str, Any]) -> tuple[Any, str | None]:
    """The value this reply carries, or a problem naming what is wrong.

    The one type-aware function in the engine, and the one place where
    tolerance is defined. It is syntactic: casing, quotes, emphasis and a
    trailing stop are normalised away. It is never semantic — an undeclared
    label does not become the nearest declared one, because a harness that
    guesses is a harness whose coverage figure means nothing.
    """
    text = _bare(raw)
    if not text:
        return None, "no value"
    low = text.lower()

    if low in NULL_WORDS:
        if norm["nullable"]:
            return None, None
        return None, f"null is not allowed by this schema (got {text!r})"

    if norm["base"] == "boolean":
        if low in TRUE_WORDS:
            return True, None
        if low in FALSE_WORDS:
            return False, None
        return None, f"{text!r} is not yes or no"

    if norm["base"] == "object":
        return _validate_object(text, norm)

    if norm["base"] == "text":
        # Shape only: there is no declared value set to check against, so this
        # is a length and a line count, never a judgement about the content.
        if "\n" in text:
            return None, "must be a single line"
        if len(text) > MAX_TEXT_FIELD_CHARS:
            return None, (f"{len(text)} characters is past the "
                          f"{MAX_TEXT_FIELD_CHARS} allowed for a copied field")
        return text, None

    for value in norm["values"]:
        if low == value.strip().lower():
            return value, None          # canonicalised to the declared spelling
    # Quoted and capped. This string is echoed back into the retry prompt, so
    # it is the one place where a sub-model's own output re-enters an
    # instruction — the same "text we did not write, treated as instruction"
    # shape as the criterion above, one layer down. `repr` keeps it on one
    # line and visibly a quotation; the cap keeps a runaway reply from
    # becoming the bulk of the retry.
    return None, f"{_bare(text)[:80]!r} is not one of the declared values"


def _object_example(norm: dict[str, Any]) -> str:
    """The record's shape as one literal JSON line, with placeholder values.

    Placeholders name the field's kind rather than picking a value: writing a
    real label here would put one of six in front of the model and nothing
    else, which is the bias the enum contract avoids by writing all of them.
    The permitted values follow on their own lines.
    """
    parts = []
    for name, sub in norm["fields"].items():
        if sub["base"] == "boolean":
            shown = "yes"
        elif sub["base"] == "text":
            shown = "<the text>"
        else:
            shown = "<one label>"
        parts.append(f'"{name}": "{shown}"')
    return "{" + ", ".join(parts) + "}"


def _validate_object(text: str, norm: dict[str, Any]) -> tuple[Any, str | None]:
    """One item's record: strict JSON, every declared field, nothing else.

    Each field goes through `validate_value` under its own schema, so an object
    is exactly as tolerant as the scalars it is made of — casing and quoting
    normalised, meaning never guessed — and no more. The problems are reported
    together rather than one at a time, because the retry re-sends the item
    once and a message naming a single field would earn a reply missing a
    different one.
    """
    if not (text.startswith("{") and text.endswith("}")):
        return None, f"{text[:80]!r} is not a JSON object on one line"
    try:
        raw = json.loads(text)
    except ValueError as exc:
        return None, f"not valid JSON ({exc}); write one object on one line"
    if not isinstance(raw, dict):
        return None, f"expected a JSON object, got {type(raw).__name__}"

    problems: list[str] = []
    extra = sorted(set(raw) - set(norm["fields"]))
    if extra:
        problems.append(f"unexpected field(s) {extra}")
    value: dict[str, Any] = {}
    for name, sub in norm["fields"].items():
        if name not in raw:
            problems.append(f"missing field {name!r}")
            continue
        # `json.loads` has already given us the typed value, so a bool arrives
        # as a bool and a string as a string; `validate_value` reads text, and
        # rendering back through `str` is what lets one validator serve both
        # the line format and the JSON one.
        found = raw[name]
        rendered = "null" if found is None else (
            found if isinstance(found, str) else str(found))
        checked, problem = validate_value(rendered, sub)
        if problem:
            problems.append(f"{name}: {problem}")
        else:
            value[name] = checked
    if problems:
        return None, "; ".join(problems)
    return value, None


class Fragment:
    """One request: which item ids it carries, and the text they were built
    from. `ref` is only a name for the record; ids are the contract."""

    def __init__(self, ref: str, ids: list[int], source: str,
                 provenance: list[list[int]] | None = None,
                 item_sources: dict[int, str] | None = None):
        self.ref = ref
        self.ids = ids
        self.source = source
        self.provenance = provenance
        self.item_sources = item_sources

    def narrow(self, ids: list[int]) -> "Fragment":
        """Build a retry whose source contains only the unresolved items."""
        if not self.item_sources or not all(i in self.item_sources for i in ids):
            return Fragment(self.ref, ids, self.source, self.provenance,
                            self.item_sources)
        source = "\n\n".join(self.item_sources[i] for i in ids)
        provenance = None
        if self.provenance is not None and len(self.provenance) == len(self.ids):
            by_id = dict(zip(self.ids, self.provenance))
            provenance = [by_id[i] for i in ids]
        return Fragment(self.ref, ids, source, provenance,
                        {i: self.item_sources[i] for i in ids})

    def contract(self, instruction: str, norm: dict[str, Any],
                 only: list[int] | None = None) -> str:
        """The instruction, then the format, shown rather than described.

        Measured: an enum contract that named the shape once, as
        `item N: <exactly one of: a, b, c>`, got bare labels back — one per
        line, in order, with the `item N:` prefix dropped, from eighteen of
        nineteen fragments. The boolean contract this grew out of had *shown*
        its format instead, one full line per allowed value, and swept 1,600 of
        1,600. So every permitted value is written out with the prefix on it.

        Showing them all also keeps it unbiased: a couple of examples would put
        two of six labels in front of the model and nothing else.

        `only` narrows the request to the named ids without changing the text:
        the retry path asks again for the items that failed, not for the whole
        fragment, so the items that already validated are never put back at
        risk.
        """
        if norm["base"] == "object":
            # A record cannot enumerate its permitted lines — the product of
            # its fields' value sets is not a list to write out. So the shape
            # is shown once, filled with each field's own permitted values, and
            # the enum inside it still writes those out in full. The
            # instruction-vs-format collision this whole method exists to avoid
            # is the reason the shape is *shown* rather than described.
            lines = [f"item N: {_object_example(norm)}"]
            for name, sub in norm["fields"].items():
                lines.append(f'  "{name}" is {describe_schema(sub)}')
        else:
            allowed = (("yes", "no") if norm["base"] == "boolean"
                       else tuple(norm["values"]))
            shown = list(allowed[:SHOWN_VALUES])
            lines = [f"item N: {value}" for value in shown]
            if len(allowed) > len(shown):
                lines.append(
                    f"...and the same for: {', '.join(allowed[SHOWN_VALUES:])}")
        if norm["nullable"]:
            lines.append("item N: null")
        if only is None:
            asked = (
                f"Answer with exactly one line per item, from item {self.ids[0]} to "
                f"item {self.ids[-1]}, in order, and nothing else."
            )
        else:
            listed = ", ".join(str(i) for i in only)
            asked = (
                f"Answer ONLY for these items, in this order: {listed}. "
                "One line per listed item and nothing else."
            )
        # The caller's instruction is DELIMITED, not concatenated. It used to
        # lead the message as a bare imperative, with this operation's format
        # underneath it, and nothing decided which one a sub-model should obey.
        # A measured episode put "Output in JSON format: {'label': ...}" into
        # its instruction: every sub-model followed the caller, answered a
        # single JSON object for a forty-one item fragment, and not one line
        # parsed — 795 items failed, all twenty fragments landed in
        # `parse_errors`, forty sub-calls paid for nothing. Adding a line that
        # said "ignore any format named above" changed nothing, which is the
        # useful part of the result: the fix for a conflict of imperatives is
        # not a third imperative, it is to stop sending two.
        #
        # `<source>` has always been delimited for the same reason and has
        # never had this problem. So the criterion is quoted the same way, and
        # the only instructions left in the message are this module's.
        return (
            "Judge each item below against this criterion:\n"
            "<criterion>\n"
            f"{instruction}\n"
            "</criterion>\n\n"
            "The criterion says WHAT to judge. It never sets the output "
            "format; the format is fixed here.\n"
            "The text below is split into numbered items. Judge each item on "
            "its own text.\n"
            f"{asked} Every line begins "
            "with its own item number:\n"
            + "\n".join(lines)
        )

    def correction(self, instruction: str, norm: dict[str, Any],
                   problems: list[str], only: list[int] | None = None) -> str:
        # Generic format feedback only: which contract rule broke, never a hint
        # about the meaning of any item.
        return (
            "Your previous answer for this exact text was invalid: "
            f"{'; '.join(problems)}.\n"
            "Answer again and follow the format exactly.\n\n"
            + self.contract(instruction, norm, only=only)
        )


def contract_fingerprint() -> str:
    """The wording every sub-model reads, rendered canonically so it can be
    hashed into a run manifest.

    `SUB_SYSTEM` is not the whole of the leaf's input: this contract is the
    larger part of it, and it was not frozen anywhere. The line that makes the
    format authoritative changed what every sub-model reads while
    `leaf_prompt_sha256` stayed put — the third time a text the model actually
    reads has moved without a recorded hash moving with it.
    """
    norm = check_schema({"type": "string", "enum": ["A", "B"]})
    canonical = Fragment("f", [0, 1], "[item 0]\nx\n\n[item 1]\ny")
    return canonical.contract("<instruction>", norm)


def read_fragment(reply: str, fragment: Fragment, norm: dict[str, Any],
                  known: set[int] | None = None,
                  ) -> tuple[dict[int, Any], set[int], list[str]]:
    """Parse one reply. Returns the valid values, the ids that were answered at
    all, and the problems found in it.

    An id answered twice with different values is a contradiction, and a dict
    that quietly kept the last write would let one win in silence.

    `known` is the set of ids that legitimately belong to the text even when
    this round did not request them. A targeted retry asks for three items out
    of forty; a sub-model that answers all forty anyway is not inventing ids,
    and flagging the thirty-seven as foreign would fail the retry for
    over-answering. They are skipped: not an error, not returned, and never a
    write over a value that already validated.
    """
    wanted = set(fragment.ids)
    known = wanted if known is None else known
    values: dict[int, Any] = {}
    raw_seen: dict[int, str] = {}
    returned: set[int] = set()
    problems: list[str] = []

    for match in LINE.finditer(reply or ""):
        item = int(match.group(1))
        body = match.group(2)
        if item not in wanted:
            if item not in known:
                problems.append(f"id {item} does not belong to this text")
            continue
        returned.add(item)
        if item in raw_seen:
            problems.append(
                f"item {item} answered twice with conflicting values"
                if _bare(raw_seen[item]).lower() != _bare(body).lower()
                else f"item {item} repeated"
            )
            values.pop(item, None)
            continue
        raw_seen[item] = body
        value, problem = validate_value(body, norm)
        if problem:
            problems.append(f"item {item}: {problem}")
        else:
            values[item] = value

    missing = [i for i in fragment.ids if i not in returned]
    if missing:
        shown = ", ".join(str(i) for i in missing[:10])
        more = "" if len(missing) <= 10 else f" and {len(missing) - 10} more"
        problems.append(f"missing items: {shown}{more}")
    return values, returned, sorted(set(problems))


def run(
    fragments: list[Fragment],
    instruction: str,
    schema: Any,
    dispatch: Callable[[Iterable[dict[str, Any]]], list[Any]],
    *,
    retry: bool = True,
) -> dict[str, Any]:
    """Send every fragment, validate, salvage, retry only the failed items.

    `dispatch` takes job dicts and returns replies in order — the batched
    sub-model path, injected so this module can be tested without a REPL or a
    server.

    A valid decision is kept the moment it validates, whatever else its
    fragment did. The earlier version discarded the whole fragment on any
    problem, and the directed t14 run measured what that costs: one label out
    of forty came back as `abstract concept` instead of `description and
    abstract concept`, the retry repeated it, and 39 valid decisions — ten
    users' worth of data — went down with it. So the retry now re-sends the
    same text but asks only for the ids that are still unresolved: missing,
    invalid, or contradictory. Items that validated are never asked again and
    never overwritten; the same run also measured that a retry with the
    problems named recovers most fragments (two of three), so that stays.

    The three counters are sets of distinct ids, so a retried fragment never
    counts its items twice:

        presented   ids that went out in a request that was actually issued
        returned    ids that came back in a parseable line, valid or not
        valid       ids with exactly one non-contradictory conforming value

    `parse_errors` names the fragments with ids still unresolved after their
    retry — the ones whose absence `failed_items` will report downstream.
    """
    norm = check_schema(schema)
    values: dict[int, Any] = {}
    presented: set[int] = set()
    returned: set[int] = set()
    unsent: set[int] = set()
    parse_errors: list[str] = []

    def send(batch: list[Fragment], instructions: list[str]) -> list[Any]:
        # A generator, not a list: this is the path the system prompt
        # recommends, and materialising its jobs made `consumed_lazily`
        # unsatisfiable through the harness's own route.
        jobs = ({"instruction": text, "source": fragment.source,
                 "source_ref": fragment.ref, "provenance": fragment.provenance}
                for fragment, text in zip(batch, instructions))
        replies = list(dispatch(jobs))
        # A short reply list must never silently drop the tail fragments: zip
        # would skip them past both validation and retry.
        short = len(batch) - len(replies)
        while len(replies) < len(batch):
            replies.append(None)
        # Presented is counted from what came back, not from what was planned.
        # Marking the batch up front made it a count of intentions — a budget
        # exhausted midway left later fragments unsent and reported them as
        # delivered — and marking at pull time was no better, since a job can
        # still be refused between being pulled and reaching the wire. A reply
        # slot is the one signal on this side of the channel that a request was
        # actually issued.
        for fragment, reply in zip(batch, replies):
            if reply is not None:
                presented.update(fragment.ids)
        if short > 0:
            unsent.update(i for fragment in batch[-short:] for i in fragment.ids)
        return replies

    failed: list[tuple[Fragment, list[int], list[str]]] = []
    for fragment, reply in zip(
            fragments, send(fragments, [f.contract(instruction, norm) for f in fragments])):
        if reply is None:
            failed.append((fragment, list(fragment.ids),
                           ["no reply was returned for this fragment"]))
            continue
        got, answered, problems = read_fragment(str(reply), fragment, norm)
        returned.update(answered)
        values.update(got)          # salvage: what validated is settled
        unresolved = [i for i in fragment.ids if i not in values]
        if unresolved:
            failed.append((fragment, unresolved, problems))

    if failed and retry:
        # The retry carries the same text but requests only the unresolved ids
        # — unless that is all of them, where the plain contract says the same
        # thing in fewer tokens. The retry Fragment keeps the original ref: it
        # is the same request being finished, not a new one.
        retried = [f.narrow(ids) for f, ids, _ in failed]
        instructions = [
            f.correction(instruction, norm, p,
                         only=ids if len(ids) < len(f.ids) else None)
            for f, ids, p in failed
        ]
        replies = send(retried, instructions)
        for (fragment, ids, _), retry, reply in zip(failed, retried, replies):
            got, answered, _ = read_fragment(str(reply or ""), retry, norm,
                                             known=set(fragment.ids))
            returned.update(answered)
            for item, value in got.items():
                # By construction the retry only wants ids without a value, but
                # the merge is stated anyway: a first-round decision is never
                # replaced by a second-round one.
                values.setdefault(item, value)
            if any(i not in values for i in ids):
                parse_errors.append(fragment.ref)
    elif failed:
        # The explicit runtime retry is itself one pass.  It still has to name
        # fragments left unresolved; disabling the internal retry must not
        # erase the failure account.
        parse_errors.extend(fragment.ref for fragment, ids, _ in failed
                            if any(i not in values for i in ids))

    return {
        "values": values,
        "presented": presented,
        "returned": returned,
        "unsent": unsent - presented,
        "parse_errors": parse_errors,
        "schema": norm,
    }
