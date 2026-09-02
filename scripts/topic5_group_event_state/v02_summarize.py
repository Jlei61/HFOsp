#!/usr/bin/env python3
"""Cohort tables + the load-bearing figure for Agent A (H1/H2a).

Writes the small machine-readable outputs and the figure into the repository
results root; the large per-subject payloads stay under ``/data``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_group_event_state.v02.aggregate import (  # noqa: E402
    PRIMARY_ENDPOINTS,
    denominator_table,
    gain_cells,
    load_results,
    seed_noise_floor,
    seed_spread_table,
)
from src.topic5_group_event_state.v02.figures import (  # noqa: E402
    plot_future_block_figure,
    plot_mark_block_figure,
    plot_memory_truncation_figure,
)
from src.topic5_group_event_state.v02.registry import atomic_write_json  # noqa: E402

REPO_OUT = Path(
    "/home/honglab/leijiaxin/HFOsp/results/epi_prssm/group_event_state/v0_2/h1_h2a"
)

# Which arm names in the per-subject payloads belong to which plotted curve.
DEFAULT_ARMS = {
    "P_local": "B+S(P_local_seed",
    "P_slow": "B+S(P_slow_seed",
    # The control that decides the reading: its state is a function of the time
    # since the last event and nothing else.  Leaving it off the figure would let
    # a reader conclude "nothing helps anywhere" without seeing that a model
    # carrying nothing does at least as well as the ones that carry history.
    "memoryless": "B+S(P_memoryless_seed",
    # Only the load-bearing producer's shifted state; "B+shift" alone would
    # silently pool the two producers' nulls into one grey curve.
    "shift": "(S(P_slow_seed",
}


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--future-root", type=Path, required=True)
    parser.add_argument("--prefix-root", type=Path, default=None)
    parser.add_argument("--out-root", type=Path, default=REPO_OUT)
    parser.add_argument("--tag", default="main")
    parser.add_argument("--diagnostics-root", type=Path, default=None)
    parser.add_argument("--arms", nargs="*", default=None,
                        help="curve=pattern pairs, e.g. P_slow=B+S(P_slow_seed")
    args = parser.parse_args()

    results = load_results(args.future_root)
    if not results:
        raise SystemExit(f"no per-subject results under {args.future_root}")
    prefix = None
    if args.prefix_root and (Path(args.prefix_root) / "per_subject").exists():
        prefix = [json.loads(p.read_text())
                  for p in sorted((Path(args.prefix_root) / "per_subject").glob("*.json"))]

    arms = DEFAULT_ARMS
    if args.arms:
        arms = dict(a.split("=", 1) for a in args.arms)

    out = Path(args.out_root) / args.tag
    (out / "figures").mkdir(parents=True, exist_ok=True)

    cells = gain_cells(results, endpoints=PRIMARY_ENDPOINTS)
    atomic_write_json(out / "cohort_gain_cells.json",
                      {"cells": [c.as_dict() for c in cells]})
    _write_csv(out / "cohort_gain_summary.csv", [
        {k: v for k, v in c.as_dict().items() if k != "per_subject"} for c in cells
    ])
    _write_csv(out / "denominators.csv", denominator_table(results))

    spread = seed_spread_table(results, producers=["P_local", "P_slow"])
    _write_csv(out / "seed_spread.csv", spread)
    atomic_write_json(out / "seed_noise_floor.json", seed_noise_floor(spread))

    payload = plot_future_block_figure(
        results, prefix,
        out / "figures" / "future_block_state_gain.png",
        out / "figures" / "future_block_state_gain.pdf",
        arms=arms,
    )
    atomic_write_json(out / "figures" / "future_block_state_gain_metadata.json", {
        "future_root": str(args.future_root),
        "prefix_root": str(args.prefix_root) if args.prefix_root else None,
        "arms": arms,
        "n_subjects": len(results),
        "subjects": sorted(r["subject"] for r in results),
        "panels": payload["panels"],
    })
    block_payload = plot_mark_block_figure(
        results,
        out / "figures" / "which_part_of_the_block.png",
        out / "figures" / "which_part_of_the_block.pdf",
        arms={k: v for k, v in arms.items() if k != "shift"},
    )
    atomic_write_json(out / "figures" / "which_part_of_the_block_metadata.json",
                      {"arms": arms, "blocks": block_payload})

    if args.diagnostics_root and (Path(args.diagnostics_root) / "per_subject").exists():
        diag = load_results(args.diagnostics_root)
        trunc = plot_memory_truncation_figure(
            diag, results,
            out / "figures" / "memory_truncation_diagnostic.png",
            out / "figures" / "memory_truncation_diagnostic.pdf",
        )
        atomic_write_json(
            out / "figures" / "memory_truncation_diagnostic_metadata.json",
            {"diagnostics_root": str(args.diagnostics_root), "ladder": trunc},
        )

    print(json.dumps({
        "n_subjects": len(results),
        "n_cells": len(cells),
        "figure": str(out / "figures" / "future_block_state_gain.png"),
    }, indent=2))


if __name__ == "__main__":
    main()
