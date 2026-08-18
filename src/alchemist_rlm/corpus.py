"""A deterministic corpus whose truths are computed, never written down.

Every answer in this file is derived from the same literal the model will see.
That rule exists because a scorer in this repository once inflated a result from
1/2 to 2/2 by matching a substring, and a hand-typed expected answer is the same
class of error one step earlier: a number nobody can re-derive.

The corpus is built to make three different failures distinguishable:

  - **Lexical decoys.** Records 12, 45 and 197 contain "responsible", "missing"
    and "shipment" literally and are not the answer to the semantic needle. A
    model that greps its way to an answer lands on them. `probe_11` spent 23 of
    25 turns grepping for a name that was never written literally, so the needle
    here is stated only in paraphrase.
  - **A real second hop.** The crate's crew is in one record and that crew's
    lead is in a roster far away. Neither passage answers the question alone.
  - **A near-miss name.** "Perpetua Oyelaran" is a roster entry for the wrong
    week. It is here on purpose: a StateLine run once answered with exactly that
    name, taken from a passage it had seen. A fabrication drawn from context is
    the failure worth catching, and it needs a plausible wrong answer present.
"""

from __future__ import annotations

import hashlib
from typing import Any

HANDLERS = (
    "Ismael Trigo", "Renata Ocampo", "Basil Ngata", "Sofia Berenguer",
    "Idris Halloway", "Nuria Castellanos", "Emeka Adeyemi", "Lucia Ferrante",
    "Osvaldo Prieto", "Margit Halvorsen", "Rashid Benali", "Clara Weisz",
    "Tomas Villalba", "Ingrid Solheim", "Feliciano Ruiz", "Aiko Tanabe",
    "Diego Maldonado", "Petra Kowalczyk", "Samuel Okonkwo", "Beatriz Alarcon",
)
DEPOTS = ("Valparaiso", "Antofagasta", "Coquimbo", "Iquique")
CREWS = ("day", "night", "relief")

# Phrasings that all mean "weather stopped us", none sharing a keyword with the
# others. A regex cannot count these; a sub-LM reading each part can.
WEATHER = (
    "Fog closed the pass and loading stopped until noon.",
    "A squall came through and the yard was shut for two hours.",
    "Heavy rain flooded the lower bay overnight.",
    "Ice on the ramp made the forklifts unusable before dawn.",
    "Wind gusts forced the cranes down for the afternoon.",
)
PLAIN = (
    "Routine transfer, no exceptions noted.",
    "Paperwork countersigned at the gate.",
    "Pallet count verified against the manifest.",
    "Seal intact on arrival and on departure.",
    "Handover completed within the scheduled window.",
    "Weighbridge reading matched the declared load.",
)

RECORDS = 240
WEATHER_EVERY = 7
WEATHER_OFFSET = 3

# --- the planted facts, each stated only once ------------------------------
NEEDLE_INDEX = 88
NEEDLE_PERSON = "Aurelio Vance"
NEEDLE_NOTE = (
    f"{NEEDLE_PERSON} signed the waiver after crate 47 never reached the depot."
)

HOP_INDEX = 61
HOP_CRATE = "crate 112"
# A different crate from the needle's on purpose. Sharing one would let a model
# that found either passage answer both questions from the same read, and the
# two tasks are meant to test different things.
HOP_NOTE = "Crate 112 moved out with the night crew on 12 March."
HOP_ANSWER = "Teodora Bassi"

DECOYS = {
    12: "The responsible party for routine checks is named in the annex.",
    45: "A shipment manifest was missing one signature and was re-issued.",
    197: "Responsibility for the missing pallet seals was assigned to the yard office.",
}

ROSTER = """\
=== Crew roster ===
Week of 3-9 March    | day crew lead: Ismael Trigo    | night crew lead: Gerardo Pinto
Week of 10-16 March  | day crew lead: Renata Ocampo   | night crew lead: Teodora Bassi
Week of 17-23 March  | day crew lead: Basil Ngata     | night crew lead: Perpetua Oyelaran
Week of 24-30 March  | day crew lead: Lucia Ferrante  | night crew lead: Gerardo Pinto"""


def _note(index: int) -> str:
    if index == NEEDLE_INDEX:
        return NEEDLE_NOTE
    if index == HOP_INDEX:
        return HOP_NOTE
    if index in DECOYS:
        return DECOYS[index]
    if index % WEATHER_EVERY == WEATHER_OFFSET:
        return WEATHER[index % len(WEATHER)]
    return PLAIN[index % len(PLAIN)]


def _record(index: int) -> str:
    return (
        f"=== Record {index:03d} ===\n"
        f"Depot: {DEPOTS[index % len(DEPOTS)]}\n"
        f"Crew: {CREWS[index % len(CREWS)]}\n"
        f"Handler: {HANDLERS[index % len(HANDLERS)]}\n"
        f"Note: {_note(index)}"
    )


CORPUS = "\n\n".join(_record(i) for i in range(RECORDS)) + "\n\n" + ROSTER


# --- truths, derived from the literal above --------------------------------
def _weather_indices() -> list[int]:
    """A record counts as weather only if nothing else overwrote its note."""
    return [
        i for i in range(RECORDS)
        if i % WEATHER_EVERY == WEATHER_OFFSET
        and i not in DECOYS and i not in (NEEDLE_INDEX, HOP_INDEX)
    ]


TRUTHS: dict[str, Any] = {
    "records": RECORDS,
    "valparaiso_count": sum(1 for i in range(RECORDS) if DEPOTS[i % len(DEPOTS)] == "Valparaiso"),
    "night_crew_count": sum(1 for i in range(RECORDS) if CREWS[i % len(CREWS)] == "night"),
    "weather_count": len(_weather_indices()),
    "weather_indices": _weather_indices(),
    "needle_person": NEEDLE_PERSON,
    "needle_record": f"Record {NEEDLE_INDEX:03d}",
    "hop_answer": HOP_ANSWER,
    "hop_record": f"Record {HOP_INDEX:03d}",
    "chars": len(CORPUS),
    "sha256": hashlib.sha256(CORPUS.encode()).hexdigest(),
}

# Invariants the corpus must satisfy for the tasks built on it to mean anything.
# Checked at import so a later edit cannot quietly break a task's premise.
assert CORPUS.count(NEEDLE_PERSON) == 1, "the needle person must appear exactly once"
assert CORPUS.count(HOP_ANSWER) == 1, "the hop answer must appear exactly once"
assert "accountable" not in CORPUS.lower(), "the needle question's key word must be absent"
assert CORPUS.lower().count("responsib") >= 2, "lexical decoys must exist"
assert CORPUS.lower().count("crate 47") == 1, "the needle's crate is stated once"
assert CORPUS.lower().count("crate 112") == 1, "the hop's crate is stated once"
assert "Teodora Bassi" not in CORPUS[:CORPUS.index(ROSTER)], "the hop's second leg is only in the roster"


def needle_is_in(text: str) -> bool:
    """Did this slice actually contain the evidence?

    This is the hinge of the plan's atomic attribution: a subcall that never
    received the needle failed at retrieval, and one that received it and
    answered wrong failed at the atom. The two get different verdicts.
    """
    return NEEDLE_NOTE in (text or "")


def hop_pieces_in(text: str) -> set[str]:
    """Which legs of the multi-hop this text contains. Scoring a multi-hop answer as
    right or wrong hides where it broke; naming the legs separates 'never found the
    first passage' from 'found both and joined them wrong'.
    """
    pieces = set()
    if HOP_NOTE in (text or ""):
        pieces.add("crew")
    if "night crew lead: Teodora Bassi" in (text or ""):
        pieces.add("lead")
    return pieces
