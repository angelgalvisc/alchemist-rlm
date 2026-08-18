"""A large corpus where the criterion is semantic, not lexical.

V1's batching task handed the model its own keywords — the question listed
"fog, rain, wind, ice or a squall" — and the model did exactly what the wording
invited: eleven turns of literal search. Even a correct 33 would not have
demonstrated exhaustive semantic reading, so the task could not test what it
existed to test.

Two changes fix that, and both are properties of the corpus rather than of the
prompt:

**The criterion crosses vocabularies.** Work is stopped by weather, by a power
cut, by a strike, by a broken forklift. No single word covers the class, so
there is no keyword to search for.

**The vocabulary crosses the criterion.** Records that mention fog, rain, wind
and ice while work carried on normally are here in numbers. Grepping the
weather words *overcounts*, and the size of the overcount is known by
construction, so a lexical answer is not merely wrong — it is identifiable as
lexical from the number alone.

At 1,600 records this is about 200,000 characters: forty times V1's corpus and
past the point where any part of it fits in the model's turn.
"""

from __future__ import annotations

import hashlib
from typing import Any

from alchemist_rlm.corpus import CREWS, DEPOTS, HANDLERS

# Work actually stopped or was delayed. Mixed causes on purpose.
STOPPAGES = (
    "Fog closed the pass and loading stopped until noon.",
    "The substation failed and the yard ran on batteries for three hours.",
    "A walkout by the night shift halted transfers after 22:00.",
    "Ice on the ramp made the forklifts unusable before dawn.",
    "The main forklift threw a hydraulic line and the bay stood idle.",
    "Heavy rain flooded the lower bay overnight.",
    "A customs hold stopped every outbound crate for the afternoon.",
    "Wind gusts forced the cranes down for the afternoon.",
)

# The lexical traps: same weather words, work carried on. A grep for fog, rain,
# wind or ice hits every one of these and every one is a wrong answer.
CARRIED_ON = (
    "The forecast warned of fog but loading finished ahead of schedule.",
    "Rain held off long enough for the transfer to complete on time.",
    "Wind was noted in the log and had no effect on the schedule.",
    "Ice was cleared from the ramp before the first shift and work began normally.",
    "A squall was forecast, did not arrive, and the yard ran a full day.",
    "Despite the rain warning the crew completed every movement as planned.",
)

PLAIN = (
    "Routine transfer, no exceptions noted.",
    "Paperwork countersigned at the gate.",
    "Pallet count verified against the manifest.",
    "Seal intact on arrival and on departure.",
    "Handover completed within the scheduled window.",
    "Weighbridge reading matched the declared load.",
    "Driver signed for the load without comment.",
)

RECORDS = 1600
STOP_EVERY = 11        # a stoppage every eleventh record
TRAP_EVERY = 7         # a weather-word non-stoppage every seventh


def _note(index: int) -> str:
    if index % STOP_EVERY == 4:
        return STOPPAGES[index % len(STOPPAGES)]
    if index % TRAP_EVERY == 2:
        return CARRIED_ON[index % len(CARRIED_ON)]
    return PLAIN[index % len(PLAIN)]


def _record(index: int) -> str:
    return (
        f"=== Record {index:04d} ===\n"
        f"Depot: {DEPOTS[index % len(DEPOTS)]}\n"
        f"Crew: {CREWS[index % len(CREWS)]}\n"
        f"Handler: {HANDLERS[index % len(HANDLERS)]}\n"
        f"Note: {_note(index)}"
    )


CORPUS_V2 = "\n\n".join(_record(i) for i in range(RECORDS))

_STOPPED = [i for i in range(RECORDS) if _note(i) in STOPPAGES]
_TRAPS = [i for i in range(RECORDS) if _note(i) in CARRIED_ON]

WEATHER_WORDS = ("fog", "rain", "wind", "ice", "squall")


def _lexical_hits() -> int:
    """How many records a keyword search would return. Deliberately computed:
    it is the number a lexical answer produces, and naming it in advance makes
    that failure mode legible in the result rather than merely wrong."""
    return sum(
        1 for i in range(RECORDS)
        if any(word in _note(i).lower() for word in WEATHER_WORDS)
    )


TRUTHS_V2: dict[str, Any] = {
    "records": RECORDS,
    "chars": len(CORPUS_V2),
    "stoppages": len(_STOPPED),
    "lexical_traps": len(_TRAPS),
    "keyword_search_would_return": _lexical_hits(),
    "night_crew_count": sum(1 for i in range(RECORDS) if CREWS[i % len(CREWS)] == "night"),
    "sha256": hashlib.sha256(CORPUS_V2.encode()).hexdigest(),
}

# The premises the task rests on, checked at import so an edit cannot break them
# quietly. If a keyword search ever returned the right answer, the task would be
# testing search again and nobody would notice from the pass rate.
assert TRUTHS_V2["chars"] > 150_000, "V2 must be a genuinely large context"
assert TRUTHS_V2["keyword_search_would_return"] != TRUTHS_V2["stoppages"], (
    "a keyword search must not land on the truth"
)
assert TRUTHS_V2["lexical_traps"] > TRUTHS_V2["stoppages"] * 0.5, (
    "there must be enough traps that a lexical answer is clearly wrong"
)
assert any(
    not any(word in note.lower() for word in WEATHER_WORDS) for note in STOPPAGES
), "some stoppages must have no weather vocabulary at all"
