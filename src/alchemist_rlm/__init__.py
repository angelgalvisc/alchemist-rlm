"""A Recursive Language Model runtime with coverage you can check.

The context lives as a Python variable in a persistent REPL and never enters
the model's window. The model writes code against it; bounded sub-model calls
do the semantic work; and what comes back is accounted for item by item, so
"I read all of it" is derived from the run rather than taken from the model's
word for it.

The one function most callers need:

    from alchemist_rlm import analyze_large_context

    result = analyze_large_context(context, question, strategy="classify")
    result["answer"]            # what it concluded
    result["answer_valid"]      # whether that is deliverable under the strategy
    result["coverage"]          # what the sweep behind it established
    result["certificate"]       # spans and validation, never correctness

`answer_valid` is the field to branch on. It fails closed: an answer built over
an incomplete sweep, or delivered by a generation that was cut off, comes back
labelled rather than suppressed.

The REPL executes model-written Python in a subprocess. That isolation is
operational — it contains a runaway loop and lets the parent kill a hung block
— and is **not a security boundary**. Do not point this at input you would not
run as code.
"""

from alchemist_rlm.adapters.agents import (
    STRATEGY_DIRECTIVES,
    TOOL_SCHEMA,
    analyze_large_context,
)
from alchemist_rlm.budgets import Budget
from alchemist_rlm.certificate import Certificate
from alchemist_rlm.engine import Episode, RLMEngine
from alchemist_rlm.output_contract import (
    OutputContract,
    PresentationBinding,
    TerminalPolicy,
    ValidationIssue,
    ValidationResult,
)
from alchemist_rlm.isolation import MLXPromptCacheIsolation

__version__ = "0.1.0"

__all__ = [
    "Budget",
    "Certificate",
    "Episode",
    "RLMEngine",
    "OutputContract",
    "PresentationBinding",
    "TerminalPolicy",
    "ValidationIssue",
    "MLXPromptCacheIsolation",
    "ValidationResult",
    "STRATEGY_DIRECTIVES",
    "TOOL_SCHEMA",
    "analyze_large_context",
    "__version__",
]
