#!/usr/bin/env python3
"""Consolidate formal OOLONG-Pairs shards without calling a model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from alchemist_rlm.consolidate import (  # noqa: E402
    ConsolidationError, consolidate_pair_results,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        combined = consolidate_pair_results(args.results, runs_dir=REPO / "runs")
    except ConsolidationError as error:
        print(f"consolidation refused: {error}", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(combined, indent=1, ensure_ascii=False))
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
