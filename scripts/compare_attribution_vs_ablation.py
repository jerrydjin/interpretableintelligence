#!/usr/bin/env python
"""Compare attribution scores with measured ablations, next to a magnitude baseline.

Reads a run folder that contains ``features/attribution.parquet`` and ablation
effects in ``scoreboard.json`` and/or ``features/clip_effects.parquet``. Prints
Spearman correlation and top-K overlap for the attribution objective, then the
same numbers for activation magnitude from ``features/feature_acts.parquet``.
Attribution has to beat that baseline to be worth its cost. The attribution
file is a shortlist: the ablation column is the confirmation.

    python scripts/compare_attribution_vs_ablation.py runs/demo --k 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from physint.attribute import compare_run, format_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Run folder with attribution and ablation files.")
    parser.add_argument("--k", type=int, default=None, help="Top-K for the overlap. Defaults to 10.")
    args = parser.parse_args(argv)
    try:
        result = compare_run(args.run_dir, k=args.k)
    except (OSError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(format_report(result))
    primary = result["spearman_feature_heldout"]
    if primary is None:
        primary = result["spearman_feature_all"]
    if primary is None:
        primary = result["spearman_clip"]
    if primary is None:
        print("error: no paired attribution and ablation scores", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
