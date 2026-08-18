"""`RLMEngine.complete(context, question)` — the whole harness behind one call.

This is the surface an agent sees, and the point of the project: a component you
can hand a context larger than the window and get an answer plus an auditable
trace, without the caller knowing about REPLs, segments or subcalls.

An episode owns its REPL, its artifacts and its trace, and shares one budget
ledger with every child it spawns.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from alchemist_rlm import protocol
from alchemist_rlm.artifacts import ArtifactStore
from alchemist_rlm.budgets import Budget
from alchemist_rlm.calls.recursive import RecursiveCaller
from alchemist_rlm.calls.scheduler import SubcallScheduler
from alchemist_rlm.manifest import visible_requests_sha256
from alchemist_rlm.native_loop import NativeLoop, Submission
from alchemist_rlm.output_contract import (
    OutputContract,
    TerminalPolicy,
    answer_value_record,
    text_record,
    validate_output_mode,
)
from alchemist_rlm.repl.runtime import ReplRuntime
from alchemist_rlm.tracing import Trace

REPO = Path(__file__).resolve().parent.parent.parent
DEFAULT_RUNS = REPO / "runs"

CHILD_MAP_CONTRACT = (
    "Your part is one slice of a larger whole. Examine your entire part "
    "completely; do not answer from a partial read. Choose the bounded operation "
    "that matches the question: semantic_search() for a yes/no decision per "
    "unit, semantic_map(...) for a fixed typed label per unit, or ordinary "
    "Python and sourced subcalls for other analyses. Deliver your part's "
    "answer with submit(value)."
)


@dataclass
class Episode:
    """Everything one node did, in a form that outlives the run.

    An episode is the unit of evidence in this project, so it carries the answer
    and the reasons to doubt it side by side: how it terminated, what it spent,
    what it delegated, which protocol errors it hit, and where its trace lives.
    Nothing here is derived from the answer.
    """
    run_id: str
    answer: str | None
    stop_reason: str
    turns: int
    seconds: float
    ledger: dict[str, Any]
    trace_path: Path
    # The answer before it was rendered, and whether one was delivered at all.
    # `answer_value is None` cannot distinguish "no delivery" from "delivered
    # None", so `answer_delivered` says which. Kept out of `to_dict` on
    # purpose: the episode record is JSON, and a value that survives being
    # written to disk is a different, weaker thing than the object itself.
    answer_value: Any = None
    answer_delivered: bool = False
    answer_value_record: dict[str, Any] | None = None
    initial_final_text: str | None = None
    repair_candidate_text: str | None = None
    final_text: str | None = None
    initial_final_text_record: dict[str, Any] | None = None
    repair_candidate_text_record: dict[str, Any] | None = None
    final_text_record: dict[str, Any] | None = None
    presentation_source: str | None = None
    output_mode: str = "raw"
    contract_validation: dict[str, Any] | None = None
    output_repair: dict[str, Any] | None = None
    steps: list[Any] = field(default_factory=list)
    protocol_errors: list[dict[str, Any]] = field(default_factory=list)
    duplicates_observed: int = 0
    subcalls: list[dict[str, Any]] = field(default_factory=list)
    recursions: list[dict[str, Any]] = field(default_factory=list)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    context_manifest: dict[str, Any] = field(default_factory=dict)
    batching: dict[str, Any] = field(default_factory=dict)
    # The node's own validated sweep summary, if it ran one. This is what lets
    # a parent report a child's coverage honestly: handing a child its whole
    # part proves nothing about what the child examined.
    semantic_result: dict[str, Any] | None = None
    # The bounded operations this node performed — grouped, not interleaved:
    # the session's semantic operations in their order, then this node's
    # delegations in theirs. The two are recorded on different sides of the
    # RPC channel, so a single global order is not established and is not
    # claimed. What was *requested* is the caller's fact; this is what ran.
    operations: list[str] = field(default_factory=list)
    # One compact record per sweep, in order, each carrying its own
    # certificate. `semantic_result` keeps the last sweep in full; this keeps
    # every sweep's coverage facts, because the adapter's satisfaction verdict
    # must come from one sweep that carries both the required operation and
    # the required completeness — and its certificate must be that sweep's.
    sweeps: list[dict[str, Any]] = field(default_factory=list)
    # Exact completed sweeps reused inside this node. They spend a root turn
    # but no leaf inference, and are named so a low subcall count is auditable
    # rather than looking like missing work.
    semantic_cache_hits: list[dict[str, Any]] = field(default_factory=list)
    presentation_checks: list[dict[str, Any]] = field(default_factory=list)
    presentation_renders: list[dict[str, Any]] = field(default_factory=list)
    # Exact digest of the ordered chat payload the root model saw. It is an
    # audit field, not a configuration field: it is only knowable after the
    # trajectory has happened.
    visible_transcript_sha256: str = ""
    # One digest per actual request, including the fresh presentation channel.
    # Keeping the compact ordered list makes omissions and channel switches
    # auditable without embedding every prompt twice.
    visible_request_sha256s: list[dict[str, Any]] = field(default_factory=list)
    isolation_attestation: dict[str, Any] | None = None

    @property
    def certificate(self) -> dict[str, Any] | None:
        """The last sweep's certificate, derived rather than stored.

        Held as its own field it became a second authority: the verdict could
        rest on one sweep while the certificate came from another. It lives in
        the sweep records now, and this reads the one that matches
        `semantic_result`. A consumer that needs the certificate behind a
        particular verdict takes it from that verdict's sweep.
        """
        for sweep in reversed(self.sweeps):
            if isinstance(sweep.get("certificate"), dict):
                return sweep["certificate"]
        return None

    def to_dict(self) -> dict[str, Any]:
        """The episode as plain data, with one authority per number.

        The subcall count comes from the ledger and nowhere else. These disagreed
        once — the episode reported zero while its own ledger and trace both showed
        one, because the field was built empty and never filled — and an agent
        reading contradictory telemetry cannot tell which half to trust.
        """
        return {
            "run_id": self.run_id,
            "answer": self.answer,
            "answer_delivered": self.answer_delivered,
            "answer_value_record": self.answer_value_record,
            "initial_final_text": self.initial_final_text,
            "repair_candidate_text": self.repair_candidate_text,
            "final_text": self.final_text,
            "initial_final_text_record": self.initial_final_text_record,
            "repair_candidate_text_record": self.repair_candidate_text_record,
            "final_text_record": self.final_text_record,
            "presentation_source": self.presentation_source,
            "output_mode": self.output_mode,
            "contract_validation": self.contract_validation,
            "output_repair": self.output_repair,
            "stop_reason": self.stop_reason,
            "turns": self.turns,
            "seconds": round(self.seconds, 2),
            "ledger": self.ledger,
            "tool_calls": len([s for s in self.steps if not getattr(s, "refused", False)]),
            "duplicates_observed": self.duplicates_observed,
            "protocol_errors": self.protocol_errors,
            "subcalls": self.ledger.get("subcalls", 0),
            "subcalls_here": len(self.subcalls),
            "subcall_detail": self.subcalls,
            "recursions": self.recursions,
            "batching": self.batching,
            "semantic_result": self.semantic_result,
            "sweeps": self.sweeps,
            "semantic_cache_hits": self.semantic_cache_hits,
            "presentation_checks": self.presentation_checks,
            "presentation_renders": self.presentation_renders,
            "operations": self.operations,
            "certificate": self.certificate,
            "artifacts": self.artifacts,
            "context": self.context_manifest,
            "trace": str(self.trace_path),
            "visible_transcript_sha256": self.visible_transcript_sha256,
            "visible_request_sha256s": self.visible_request_sha256s,
            "isolation_attestation": self.isolation_attestation,
        }


@dataclass
class RLMEngine:
    """The whole harness behind one call.

    This is the surface an agent sees, and the point of the project: hand it a
    context larger than the window and get an answer plus an auditable trace,
    without the caller knowing about REPLs, segments, batches or depth.
    """
    client: Any                                   # Chat, or a factory taking no args
    budget: Budget = field(default_factory=Budget)
    tool_name: str = protocol.TOOL_NAME
    runs_dir: Path = DEFAULT_RUNS
    block_timeout: float = 300.0
    python: str | None = None
    manifest: Any = None                          # RunManifest
    # Per-item decision replies run one line per unit: ~35 units per segment
    # needs headroom that 512 does not give, and a truncated reply reads as
    # "missing items" and burns the retry.
    sub_max_tokens: int = 1024
    # The root turn's ceiling. Raised to 8,192 and put back, and the round trip
    # is worth more than either number.
    #
    # The raise rested on ten episodes whose stop_reason contained `truncated`,
    # read as one failure. They are two. Three truncate repeatedly and die on
    # the consecutive guard — `....TT.TT`, `....TTTT`, `........TTTT`. The other
    # seven share a different signature entirely: fourteen clean turns and one
    # truncated turn at the end, `..............T`. Those did not run out of
    # room; they ran out of *turns*, and what got cut was the forced final
    # trying to write the answer as prose.
    #
    # Raising the ceiling could only ever help the first three, and the
    # measurement says it did not help the rest at all. Query 2 of the run at
    # 8,192 spent all 8,192 tokens inside `<answer>` narrating what it was about
    # to do, made no tool call, and delivered nothing — the same failure as at
    # 4,096, at twice the price and twice the wall clock.
    #
    # And the headroom was never the constraint. Across 62 root turns the median
    # completion was 381 tokens, the 90th percentile 2,341, and exactly one turn
    # reached 4,096. A ceiling the 98th percentile does not touch is not what is
    # ending these runs; a forced final that writes prose instead of calling
    # `submit` is.
    #
    # The leaf stays at 1,024 deliberately: `items_per_fragment` sizes a
    # fragment from SUBCALL_REPLY_CHARS = 4 * 1024, so raising one without the
    # other would make the sizing lie about the budget it sizes for.
    max_tokens: int = 4096
    output_mode: str = "raw"
    output_contract: OutputContract | None = None
    # Experimental: inferred from the public question before the episode and
    # exposed only after the model has committed an invalid presentation.
    inferred_presentation_spec: dict[str, Any] | None = None
    terminal_policy: TerminalPolicy = field(default_factory=TerminalPolicy)
    episode_isolation: Any = None

    def complete(
        self,
        context: str,
        question: str,
        *,
        run_id: str | None = None,
        inject: str | None = None,
        band: dict[str, Any] | None = None,
    ) -> Episode:
        """Answer `question` about `context`, and record how.

        The context is bound into the REPL as a variable and never enters the
        message history — that is the defining property of a RLM rather than an
        optimisation. `band` overrides the segmenter's sizes; `inject` adds names to
        the session; `run_id` names the directory the trace and artifacts land in.
        """
        validate_output_mode(self.output_mode, self.output_contract)
        run_id = run_id or f"ep_{uuid.uuid4().hex[:12]}"
        run_dir = self.runs_dir / run_id
        isolation_attestation = (
            self.episode_isolation.before_episode(run_id=run_id)
            if self.episode_isolation is not None else None
        )
        # ``self.budget`` is a configuration template.  A public engine may be
        # held by an agent and called repeatedly; spent state belongs only to
        # the episode currently running.
        episode_budget = self.budget.fresh()
        trace = Trace(
            run_dir / "trace.jsonl",
            run_id=run_id,
            meta={
                "tool_name": self.tool_name,
                "budget": episode_budget.to_dict(),
                "question": question,
                "context_chars": len(context or ""),
                "manifest": self.manifest.to_dict() if self.manifest is not None else None,
                "output": {
                    "mode": self.output_mode,
                    "contract": (self.output_contract.manifest()
                                 if self.output_contract is not None else None),
                    "terminal_policy": self.terminal_policy.to_dict(),
                },
                "isolation_attestation": isolation_attestation,
            },
        )
        started = time.monotonic()
        artifacts = ArtifactStore(run_dir / "artifacts")
        try:
            episode = self._run_node(
                context=context, question=question, budget=episode_budget,
                lineage=(), trace=trace, artifacts=artifacts, run_id=run_id,
                inject=inject, band=band or {}, depth=0,
            )
        except BaseException as error:
            # Close an adjudicable trace even when the caller must handle the
            # exception.  Previously the append handle and the run both ended
            # mid-air, with no terminal event saying why.
            trace.finish(reason=f"exception:{type(error).__name__}", answer=None,
                         ledger=episode_budget.ledger.snapshot())
            raise
        episode.seconds = time.monotonic() - started
        episode.isolation_attestation = isolation_attestation
        episode.trace_path = run_dir / "trace.jsonl"
        before_output_artifacts = set(artifacts.saved)
        if episode.answer_delivered:
            episode.answer_value_record = answer_value_record(
                episode.answer_value, artifacts)
        episode.initial_final_text_record = text_record(
            episode.initial_final_text, artifacts, label="initial-final-text")
        episode.repair_candidate_text_record = text_record(
            episode.repair_candidate_text, artifacts, label="repair-candidate-text")
        episode.final_text_record = text_record(
            episode.final_text, artifacts, label="final-text")
        for name in sorted(set(artifacts.saved) - before_output_artifacts):
            saved = artifacts.saved[name]
            trace.artifact(name=name, ref=saved.ref, chars=saved.chars,
                           sha256=saved.sha256)
        episode.artifacts = artifacts.manifest()
        trace.event(
            "output",
            output_mode=episode.output_mode,
            answer_delivered=episode.answer_delivered,
            answer_value_record=episode.answer_value_record,
            initial_final_text_record=episode.initial_final_text_record,
            repair_candidate_text_record=episode.repair_candidate_text_record,
            final_text_record=episode.final_text_record,
            contract_validation=episode.contract_validation,
            output_repair=episode.output_repair,
            presentation_checks=episode.presentation_checks,
            presentation_renders=episode.presentation_renders,
        )
        trace.finish(
            reason=episode.stop_reason,
            answer=episode.answer,
            ledger=episode_budget.ledger.snapshot(),
            visible_transcript_sha256=episode.visible_transcript_sha256,
        )
        (run_dir / "episode.json").write_text(
            json.dumps(episode.to_dict(), indent=1, ensure_ascii=False, default=str)
        )
        return episode

    # --- one node of the recursion ----------------------------------------
    def _run_node(
        self,
        *,
        context: str,
        question: str,
        budget: Budget,
        lineage: tuple[str, ...],
        trace: Trace,
        artifacts: ArtifactStore,
        run_id: str,
        inject: str | None,
        band: dict[str, Any],
        depth: int,
    ) -> Episode:
        scheduler = SubcallScheduler(
            client=self.client, budget=budget, trace=trace, depth=depth,
            max_tokens=self.sub_max_tokens, context=context or "",
        )

        def spawn(*, question: str, context: str, budget: Budget,
                  lineage: tuple[str, ...], **_: Any) -> "Episode":
            return self._run_node(
                context=context, question=question, budget=budget, lineage=lineage,
                trace=trace, artifacts=artifacts, run_id=run_id, inject=None,
                band=band, depth=depth + 1,
            )

        recurse = RecursiveCaller(
            spawn=spawn, budget=budget, lineage=lineage, trace=trace, depth=depth,
            parent_context=context or "",
        )

        def save_artifact(name: str, value: Any) -> str:
            ref = artifacts.save(name, value)
            saved = artifacts.saved[name]
            trace.artifact(name=name, ref=ref, chars=saved.chars, sha256=saved.sha256)
            return ref

        def rlm_map(question: str, parts: Any = None) -> list[dict[str, Any]]:
            """One child per part, structured results, parts cut on real edges.

            Exists because writing the partition-delegate-aggregate loop by
            hand proved to be too much friction for a 4B. The model still
            decides *whether* and *what* to delegate; this makes the recursion
            operable as a single primitive. Sequential on purpose — one model
            is resident on the MLX server — and every child spends the same
            shared ledger, so fan-out is bounded by the budget.

            Three lessons from the first real recursion run are folded in:

            - Default parts are grouped on segment boundaries, never raw
              character cuts: that run's children began with 'd 0320 ===' and
              'work began normally.' because their parent sliced mid-record.
            - Children answered in four shapes — '7', '18', prose, a list —
              and nothing could reduce them. Each result is now a dict with
              `part`, `status`, `answer`, `stop_reason`, `coverage_complete`
              and the child's own validated `semantic_result`, so the root
              can filter on status and reduce over checked numbers in Python
              without ever parsing a child's prose.
            - Each child's question carries the operational consequence of
              map: examine the whole part, no partial reads. That is not a
              domain heuristic; it is what choosing map means.
            """
            text = context or ""
            if isinstance(parts, (str, bytes)):
                # Observed verbatim: rlm_map("Split the context...", context).
                # Python happily iterates a string character by character, so
                # the harness spawned children whose entire context was "=" —
                # each answered sensibly about its absurd input — until the
                # node budget drowned. An iterable that silently means
                # something catastrophic is not a contract; it is a trap.
                raise ValueError(
                    "parts must be a collection of texts, not a single string. "
                    "Iterating a string would create one child per CHARACTER. "
                    "Either pass a list — rlm_map(question, [part1, part2]) — "
                    "or omit parts entirely and the harness will split the "
                    "context on segment boundaries for you: rlm_map(question)"
                )
            slots = max(1, budget.max_nodes - budget.ledger.nodes)
            if parts is None:
                if not text:
                    raise ValueError("rlm_map with no parts needs a loaded context")
                from alchemist_rlm.context.segmenter import grouped_parts

                parts = grouped_parts(text, slots,
                                      **{k: v for k, v in band.items() if v})
            elif hasattr(parts, "__len__") and len(parts) > slots:
                # Preflight, before any child is spawned: burning every slot
                # and then discovering 43 fragments were never visited is the
                # worst possible way to learn the budget. The list is NOT
                # regrouped automatically — if the model separated documents,
                # sections or categories on purpose, silently merging them would
                # change the meaning of its plan. The refusal states the
                # numbers and both honest ways out. (An unsized generator
                # cannot be preflighted without consuming it; the per-element
                # loop still stops it at the budget.)
                raise ValueError(
                    f"You supplied {len(parts)} parts, but only {slots} child "
                    f"slots remain. Combine them into at most {slots} parts, "
                    "or omit `parts` and rlm_map will split the context to "
                    "fit the available slots: rlm_map(question)"
                )
            asked = (
                f"{question}\n\n"
                + CHILD_MAP_CONTRACT
            )
            results: list[dict[str, Any]] = []
            for index, part in enumerate(parts):
                # Element validation happens here, while consuming, so a
                # generator is never materialised just to be checked. A wrong
                # element ends the map with a named index — finished children
                # keep their results — instead of being str()-coerced into a
                # child whose context is "{'a': 1}".
                if isinstance(part, (list, tuple)) and part and all(
                        isinstance(x, str) for x in part):
                    # A grouped element is unambiguous — it is exactly what the
                    # preflight asks for when there are too many parts. The
                    # model grouped naturally once, our validation rejected the
                    # lists, and it satisfied us with str(): seven children
                    # each got 34,000 chars of repr on a single line.
                    part = "\n\n".join(part)
                if not isinstance(part, str):
                    results.append({
                        "part": f"p{index:03d}", "status": "error",
                        "error": f"parts[{index}] is {type(part).__name__}; "
                                 "expected str or a list of str",
                    })
                    break
                if not part.strip():
                    results.append({
                        "part": f"p{index:03d}", "status": "error",
                        "error": f"parts[{index}] is empty; every part must be "
                                 "non-empty text",
                    })
                    break
                part_text = part
                record: dict[str, Any] = {"part": f"p{index:03d}",
                                          "chars": len(part_text)}
                try:
                    child = recurse.invoke(asked, part_text)
                except Exception as error:                     # noqa: BLE001
                    record["status"] = "error"
                    record["error"] = str(error)
                    results.append(record)
                    break                       # budget refusals hit every later part too
                answer = (child.answer or "").strip()
                clean = child.stop_reason in ("answer_tag", "submitted",
                                              "no_tool_call")
                sweep = child.semantic_result or {}
                record["answer"] = answer
                # The child's answer as it delivered it, not as it renders. A
                # parent reducing over children was already told never to parse
                # their prose; this is what it reduces over instead when the
                # child submitted a structure. `None` when the child ended in
                # prose or not at all — `record["status"]` says which.
                record["value"] = child.answer_value
                record["stop_reason"] = child.stop_reason
                # Honest, from the child's own validated sweep — or None,
                # meaning "not established", never a guess.
                record["coverage_complete"] = sweep.get("coverage_complete")
                # The number to reduce over, already checked item by item by the
                # per-item contract, so the parent never parses a child's prose.
                #
                # `positive_count` exists only for the boolean operation. The
                # child contract used to demand that one; it now offers
                # `semantic_map` as well, and a child taking that offer comes
                # back with no `positive_count` at all — which this read as
                # "nothing established" and marked the part unusable. The
                # child's operation was widened without widening its reader.
                # So the sweep's own words are carried instead: what it ran,
                # and how many items it validated.
                record["count"] = sweep.get("positive_count")
                # Which operation the child ran, and how much it validated. A
                # typed sweep has no single number to sum — its result is a
                # table, reachable through `rows_ref` — so these are what the
                # parent can read about it without parsing prose.
                record["operation"] = sweep.get("operation")
                record["valid_items"] = sweep.get("valid_items")
                record["rows_ref"] = sweep.get("rows_ref")
                # How the child's conversation ended.
                record["status"] = ("ok" if answer and clean
                                    else "empty" if not answer else "budget")
                # Whether the record can be summed — which is NOT the same
                # question, and conflating them was a real contradiction: five
                # of seven children in the recursion run were labelled `budget`
                # while carrying a complete, validated sweep. Running out of
                # turns to write prose says nothing about a sweep that finished
                # and was checked. So usability is decided by the evidence:
                # a validated count over a complete sweep, or a clean finish.
                #
                # The evidence half used to require `positive_count`, which
                # only the boolean operation produces. The child contract now
                # offers `semantic_map` too, and a child that took the offer,
                # swept its whole part and then ran out of turns to write prose
                # was marked unusable — while a `semantic_search` child in the
                # identical state was usable. A complete validated sweep is
                # evidence whichever operation produced it.
                validated = (record["count"] is not None
                             or bool(sweep.get("valid_items")))
                record["usable"] = bool(
                    (validated and record["coverage_complete"])
                    or record["status"] == "ok"
                )
                # The per-item ids do NOT travel with the record. They are the
                # child's own unit numbers, local to its part, and nothing said
                # so: the root saw seven overlapping integer lists, concluded
                # its parts overlapped — they did not, not by a single record —
                # and spent two full generations and both its truncations
                # chasing a duplication that never existed. They stay reachable,
                # labelled with the frame they belong to.
                if sweep.get("positive_ids"):
                    record["detail_ref"] = save_artifact(
                        f"{record['part']}_sweep",
                        {"part": record["part"],
                         "ids_are_local_to_this_part": True,
                         "examined_items": sweep.get("examined_items"),
                         "total_items": sweep.get("total_items"),
                         "positive_ids": sweep.get("positive_ids")},
                    )
                results.append(record)
            return results

        # Which bounded operations this node actually performed — as opposed to
        # which were requested of it, a distinction the adapter needs to state.
        # Delegation is recorded here where it happens; the session's own
        # semantic operations are read out of the session's audit afterwards.
        operations: list[str] = []

        def rlm_query(question: str, part: Any = None, **kwargs: Any) -> str:
            answer = recurse(question, part, **kwargs)
            operations.append("rlm_query")      # after the guards: it ran
            return answer

        def rlm_map_recorded(question: str, parts: Any = None) -> list[dict[str, Any]]:
            results = rlm_map(question, parts)
            operations.append("rlm_map")
            return results

        handlers = {
            "llm_query": scheduler.query,
            "llm_query_batched": scheduler.query_batched,
            "rlm_query": rlm_query,
            "rlm_map": rlm_map_recorded,
            "save_artifact": save_artifact,
            "load_artifact": artifacts.load_value,
        }

        with ReplRuntime(python=self.python, block_timeout=self.block_timeout,
                         handlers=handlers, trace=trace) as repl:
            context_manifest = repl.bind_context(context or "",
                                                 question=question, **band)
            trace.event("context_bound", depth=depth, manifest=context_manifest)

            def _unwrap(found: dict[str, Any]) -> Any | None:
                """One transported frame as a Python value, never a description.

                A value too large to cross inline arrives as canonical JSON and
                is decoded back into the same JSON data model. It never becomes
                presentation text merely because it was large.
                """
                if found.get("unserialisable"):
                    return None
                rendered = found.get("rendered")
                if rendered:
                    return json.loads(rendered)
                value = found.get("value")
                return artifacts.resolve(value) if isinstance(value, str) else value

            def read_submission() -> Submission:
                """Whether the session delivered, and what.

                Two values because one cannot carry both. `submit(None)`,
                `submit(0)` and `submit([])` are answers, so no value can stand
                in for "nothing was delivered" — the conflation the previous
                contract made when it read a variable and had to guess from its
                shape whether the model meant it.
                """
                found = repl.submission()
                if not found.get("delivered"):
                    return Submission()
                try:
                    return Submission(
                        delivered=True,
                        value=_unwrap(found),
                        final_text_provided=bool(found.get("final_text_provided")),
                        final_text=found.get("final_text"),
                    )
                except KeyError:
                    # A reference to an artifact that was never written. It used
                    # to resolve to itself, so `artifact://missing` became the
                    # episode's answer: twenty characters that score as a
                    # delivery and read, to a person, as a model naming a file
                    # it never saved.
                    #
                    # Reporting it as undelivered is not enough on its own, and
                    # the first attempt here proved it: the session still held
                    # the reference as its committed answer, so `submit` refused
                    # every later call and the episode could never deliver
                    # again. Saying "nothing arrived" while the delivery channel
                    # stays closed is a worse failure than the one being fixed.
                    #
                    # So the delivery is voided rather than merely disbelieved.
                    # Voiding leaves the model exactly where it was before it submitted:
                    # free to write the artifact, or to submit the value itself.
                    repl.void_submission()
                    return Submission()

            def read_presentation() -> str | None:
                found = repl.presentation()
                return found.get("text") if found.get("present") else None

            loop = NativeLoop(
                client=self.client,
                execute=lambda code: self._execute(repl, code, scheduler, turn_of=trace),
                budget=budget, trace=trace, tool_name=self.tool_name,
                max_tokens=self.max_tokens,
                read_submission=read_submission,
                # Root only. A sub-episode's answer is consumed by its parent's
                # code as a value, not read by anyone against a format, so the
                # presentation repair there would be a model call spent on a
                # question that does not apply. Decided here because here is
                # where depth is known; the loop just honours `None`.
                open_presentation=(repl.open_presentation
                                   if self.output_contract is not None and depth == 0
                                   else None),
                read_presentation=(read_presentation
                                   if self.output_contract is not None and depth == 0
                                   else None),
                output_mode=(self.output_mode if depth == 0 else "raw"),
                output_contract=(self.output_contract if depth == 0 else None),
                inferred_presentation_spec=(self.inferred_presentation_spec
                                            if depth == 0 else None),
                terminal_policy=self.terminal_policy,
                inject=inject, depth=depth,
            )
            described = _context_line(context_manifest)
            result = loop.run(question, context_line=described)
            # From the audit channel, never from `peek`: `semantic_result` and
            # any list in the namespace are the model's to reassign, and the
            # episode's sworn account of its coverage cannot be. The audit is
            # written only by the session's own operations.
            audit = (repl.peek_audit() or {}).get("audit") or {}
            sweeps_full = [s for s in audit.get("sweeps") or []
                           if isinstance(s, dict)]
            session_ops = [str(op) for op in audit.get("operations") or []]
            semantic_cache_hits = [hit for hit in
                                   audit.get("semantic_cache_hits") or []
                                   if isinstance(hit, dict)]
            presentation_checks = [check for check in
                                   audit.get("presentation_checks") or []
                                   if isinstance(check, dict)]
            presentation_renders = [render for render in
                                    audit.get("presentation_renders") or []
                                    if isinstance(render, dict)]
            # One compact record per sweep, so a consumer can ask which
            # operation established what — the last sweep alone was measured
            # to let a later trivial sweep launder or clobber an earlier one.
            #
            # The certificate travels *inside* its own sweep's record, because
            # a certificate belongs to one sweep and to no other. Held apart on
            # the episode it was the last sweep's, which is not necessarily the
            # sweep a verdict rests on: a complete context classification
            # followed by a small provided-items check reported the verdict
            # from the first and a certificate of None from the second.
            sweeps = [{
                "kind": s.get("kind"),
                "operation": s.get("operation"), "status": s.get("status"),
                "scope": (s.get("scope") or {}).get("kind"),
                "coverage": s.get("coverage"),
                "valid_items": s.get("valid_items"),
                "total_items": s.get("total_items"),
                "coverage_complete": s.get("coverage_complete"),
                "context_coverage_complete": s.get("context_coverage_complete"),
                "failed": s.get("failed"),
                "failed_items": s.get("failed_items"),
                "sweep_id": s.get("sweep_id"),
                "retry_exhausted": s.get("retry_exhausted"),
                "rows_ref": s.get("rows_ref"), "rows_digest": s.get("rows_digest"),
                "certificate": (s.get("certificate")
                                if isinstance(s.get("certificate"), dict) else None),
            } for s in sweeps_full]
            # `semantic_result` stays the last sweep's metadata. The rows are
            # omitted here because their content-addressed artifact is already
            # referenced by rows_ref; embedding them again would make every
            # episode file scale with the context.
            own_semantic = None
            if sweeps_full:
                own_semantic = {k: v for k, v in sweeps_full[-1].items()
                                if k not in ("certificate", "rows")}

        return Episode(
            run_id=run_id,
            answer=result.answer,
            answer_value=result.answer_value,
            answer_delivered=result.answer_delivered,
            initial_final_text=result.initial_final_text,
            repair_candidate_text=result.repair_candidate_text,
            final_text=result.final_text,
            presentation_source=result.presentation_source,
            output_mode=result.output_mode,
            contract_validation=result.contract_validation,
            output_repair=result.output_repair,
            stop_reason=result.stop_reason,
            turns=result.turns,
            seconds=0.0,
            ledger=budget.ledger.snapshot(),
            trace_path=self.runs_dir / run_id / "trace.jsonl",
            steps=result.steps,
            protocol_errors=result.protocol_errors,
            duplicates_observed=result.duplicates_observed,
            subcalls=scheduler.records,
            recursions=recurse.calls,
            semantic_result=own_semantic,
            sweeps=sweeps,
            semantic_cache_hits=semantic_cache_hits,
            presentation_checks=presentation_checks,
            presentation_renders=presentation_renders,
            operations=session_ops + operations,
            visible_request_sha256s=result.visible_request_sha256s,
            visible_transcript_sha256=visible_requests_sha256(
                result.visible_request_sha256s),
            # Every counter says its scope. `_here` is this node; `_tree` is
            # this node plus everything below it. They used to be mixed without
            # saying so — `subcalls` came from the shared ledger and counted the
            # whole tree, `batches` came from this node's scheduler alone — and
            # the recursion run that delegated all its work reported "50
            # subcalls, 0 batches, 0 sequential calls" in one record. The bare
            # names are kept, and are the tree, because that is what a reader
            # asking "did this episode batch?" means.
            batching={
                "batches_here": scheduler.batches,
                "batches_tree": scheduler.batches + recurse.subtree["batches"],
                "batches": scheduler.batches + recurse.subtree["batches"],
                "peak_in_flight_here": scheduler.peak_in_flight,
                "peak_in_flight_tree": max(scheduler.peak_in_flight,
                                           recurse.subtree["peak_in_flight"]),
                "peak_in_flight": max(scheduler.peak_in_flight,
                                      recurse.subtree["peak_in_flight"]),
                "sequential_calls_here": scheduler.sequential_calls,
                "sequential_calls_tree": (scheduler.sequential_calls
                                          + recurse.subtree["sequential_calls"]),
                "sequential_calls": (scheduler.sequential_calls
                                     + recurse.subtree["sequential_calls"]),
                "lazy_pulls_here": repl.pulls,
                "lazy_pulls_tree": repl.pulls + recurse.subtree["lazy_pulls"],
                "lazy_pulls": repl.pulls + recurse.subtree["lazy_pulls"],
                "lazy_items_pulled": repl.pulled_items,
                "subcalls_below": recurse.subtree["subcalls"],
                "provenance_rejected": scheduler.provenance_rejected,
            },
            context_manifest=context_manifest,
        )

    @staticmethod
    def _execute(repl: ReplRuntime, code: str, scheduler: SubcallScheduler,
                 turn_of: Trace) -> dict[str, Any]:
        return repl.execute(code)


BOUND_NAMES = (
    "read_context(ref_or_start, end=None)",
    # The return shape is announced for the same reason `semantic_map`'s is.
    # This one returns a dict, and a model that treats it as a sequence gets two
    # wrong answers in a row: `len(...)` is the number of keys — nine — which
    # reads exactly like a count of matches, and slicing it raises
    # `TypeError: unhashable type: 'slice'`, which names neither this function
    # nor the type it returned. Query 18 spent four turns there and then looped
    # on the same block until the duplicate guard ended the episode.
    #
    # The first wording of this line read "matches in ['matches'], the hits
    # themselves in ['hits']", which is grammatical English for two different
    # claims: `['matches']` holds the matches, or `['matches']` is how many
    # matches there are. It is the second, and two episodes read the first —
    # `len(matches['matches'])` raising "object of type 'int' has no len()",
    # t14 on turns 5 and 6 and t19 on turns 3 and 7. t14 had just worked out
    # that the labels are absent from its copy, concluded search_context was
    # broken, and abandoned the approach. Four turns, from an annotation
    # written to save them. Each key is now named by what it is.
    "search_context(pattern, regex=False)   # returns a dict: ['hits'] is the "
    "list of hits, ['matches'] is how many there are (a number)",
    "partition_context(text=None, target_chars=None)",
    "semantic_search(goal=None)     # no goal: uses your question, verbatim",
    "semantic_map(instruction, schema, items=None)   # returns a dict; "
    "validated values are in result['rows']; schema={'type': 'string', "
    "'enum': [...]}, {'type': 'boolean'}, or an object of those for several "
    "fields per item",
    "retry_failed(result)          # retry result['failed'] once",
    "llm_query(instruction, source)",
    "llm_query_batched(jobs)",
    "rlm_query(question, context)",
    # Neutral on purpose. This line once ended with "NEVER a single string",
    # and the very next episode the model — greedy decoding, identical
    # question, identical everything else — dropped every form of delegation
    # it had used the run before and did twelve turns of keyword regex. A
    # warning bolted to a function's name may read as "dangerous function"
    # to a 4B. The runtime guard already teaches the string rule at the exact
    # moment of misuse, which is where correction belongs; the signature only
    # needs to say what the argument is.
    "rlm_map(question, parts=None)   # omit parts to split the context "
    "automatically, or pass a list of strings",
    "save_artifact(name, value)",
    "load_artifact(name)",
    # The line this replaces read "Final = <your answer>  # a variable to
    # assign, NOT a function to call" — a clarification that existed because
    # models kept calling `Final(...)`. They were reaching for the right shape
    # and the contract was the wrong one: delivering is an act, and a function
    # is what an act looks like in Python. It is a function now.
    "submit(value)                # deliver the answer; this ends the run",
)


def _ref_span(manifest: dict[str, Any]) -> str:
    """The segment refs as a span, first and last, never a truncated prefix.

    It used to render `s0000, s0001, s0002, ...` beside a segment count, and a
    model that reads three single-digit refs and "19 segments" builds the tenth
    one by generalising the pattern it was shown. Query 3 wrote
    `f's000{i}' for i in range(19)`, which is right nine times and produces
    `s00010` on the tenth. The refusal names the real range — *"no segment
    's00010'; refs run s0000..s0018"* — but the model had already committed to
    the shape, resent the same block three times, and the episode ended on the
    error guard twenty-two seconds in, having swept nothing.

    Showing the last ref costs one token and makes the padding unambiguous. The
    harness was teaching a width that is wrong past nine.
    """
    refs = manifest.get("first_refs") or []
    if not refs:
        return "none"
    last = manifest.get("last_ref")
    if last and last != refs[0]:
        return f"{refs[0]}..{last}"
    return ", ".join(refs[:3]) + (", ..." if len(refs) > 3 else "")


def _context_line(manifest: dict[str, Any]) -> str:
    """What the model is told about a context it cannot see.

    Numbers, structure and the names that reach it — enough to plan with, and
    nothing that would let it pretend it has read the text.

    The "already defined, do not import" sentence is not decoration. In the
    first smoke episode the model read an earlier wording as a list of modules
    and spent four of its seven turns on `from context import read_context`,
    `import context` and `import globals` before finding the names with
    `dir()`. Naming a helper is not the same as saying where it lives.
    """
    names = "\n".join(f"  {name}" for name in BOUND_NAMES)
    if not manifest or not manifest.get("chars"):
        return (
            "There is no external context loaded for this question.\n"
            "These names are already defined in your Python session (do not "
            f"import anything):\n{names}"
        )
    return (
        f"`context` is a str of {manifest['chars']:,} characters "
        f"({manifest.get('lines', 0):,} lines), structured as "
        f"{manifest.get('structure')}, pre-segmented into "
        f"{manifest.get('segments')} segments of about "
        f"{manifest.get('segment_chars', {}).get('mean', 0):,} characters "
        f"(refs {_ref_span(manifest)}). "
        "It is NOT in this conversation.\n\n"
        "`context` is a plain str variable, not a module. It and these helpers "
        "are ALREADY DEFINED in your Python session — use them directly and do "
        f"not import anything:\n{names}\n\n"
        "A good first call is `print(context_manifest)` or "
        "`print(context[:500])`."
    )
