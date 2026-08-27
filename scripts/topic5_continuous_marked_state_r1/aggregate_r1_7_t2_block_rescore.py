#!/usr/bin/env python3
"""Aggregate the independent-block T2 re-scoring, patient-first.

Mirrors the R1.7A H1 uncertainty treatment: align blocks across seeds by
(segment, block), take the median across seeds per block, then bootstrap over
blocks.  A T2 effect is only credible if it survives this view, because the
frozen next-event average pools rows whose exponential exposure histories
overlap by construction.

Exploratory instrument development; does not restate the frozen T2 verdict.
"""
from __future__ import annotations

import argparse
import collections
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from src.topic5_continuous_marked_state_r1 import contract


CONTROLS = ("no_edge", "state_matched_placebo", "current_event_only")


def bootstrap(values: np.ndarray, *, draws: int = 2000, seed: int = 1701) -> dict:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return {"estimate": float(values.mean()) if len(values) else None,
                "ci95": [None, None], "n_blocks": int(len(values))}
    rng = np.random.default_rng(seed)
    sampled = [float(np.mean(values[rng.integers(0, len(values), len(values))]))
               for _ in range(draws)]
    lo, hi = np.quantile(sampled, [.025, .975]).tolist()
    return {"estimate": float(values.mean()), "median": float(np.median(values)),
            "ci95": [lo, hi], "n_blocks": int(len(values)),
            "fraction_favourable": float(np.mean(values < 0)),
            "excludes_zero_favourable": bool(hi < 0),
            "excludes_zero_unfavourable": bool(lo > 0),
            "draws": draws, "seed": seed}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=contract.RESULT_ROOT / "r1_7a")
    args = parser.parse_args()
    source_dir = args.root / "t2_r2_block_rescore"
    cells = collections.defaultdict(list)
    for path in sorted(source_dir.glob("*/*/result.json")):
        value = json.loads(path.read_text())
        cells[(value["subject"], value["source"])].append(value)
    rows = []
    for (subject, source), values in sorted(cells.items()):
        row = {"subject": subject, "source": source, "n_seeds": len(values),
               "n_blocks": values[0]["n_blocks"]}
        for control in CONTROLS:
            per_seed = np.asarray([
                v["independent_block_contrasts"][f"real_minus_{control}"]["per_block"]
                for v in values
            ], dtype=np.float64)
            if per_seed.shape[1] != row["n_blocks"]:
                raise ValueError(f"{subject}/{source}: block count disagrees across seeds")
            row[f"real_minus_{control}"] = bootstrap(np.median(per_seed, axis=0))
            row[f"real_minus_{control}_next_event_average"] = float(np.median([
                v["next_event_average_from_frozen_cell"][f"real_minus_{control}"]
                for v in values
            ]))
        # frozen verdict for side-by-side comparison, not restated as our own
        row["frozen_primary_increment_seeds"] = int(sum(
            v["primary_next_event_increment"] for v in values))
        row["block_level_support"] = bool(all(
            row[f"real_minus_{c}"]["excludes_zero_favourable"] for c in CONTROLS))
        rows.append(row)
    summary = {
        "status": "COMPLETE",
        "revision": "r1_7_t2_independent_block_rescore_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "patient_source": rows,
        "block_level_support": [
            {"subject": r["subject"], "source": r["source"]}
            for r in rows if r["block_level_support"]
        ],
        "uncertainty": "per-block seed median, then block bootstrap (2000 draws)",
        "refitted": False,
        "exploratory_instrument_development_not_preregistered": True,
        "formal_test_partition_opened": False, "sealed_opened": False,
    }
    report = args.root / "reports"; report.mkdir(parents=True, exist_ok=True)
    contract.atomic_json(report / "t2_block_rescore_summary.json", summary)
    print(f"{'patient':18s} {'source':13s} {'blk':>4s} {'avg':>10s} {'blockmean':>10s} "
          f"{'CI95':>26s} {'frac<0':>6s} {'support'}")
    for r in rows:
        b = r["real_minus_no_edge"]
        ci = b["ci95"]
        cis = f"[{ci[0]:+.5f},{ci[1]:+.5f}]" if ci[0] is not None else "n/a"
        print(f"{r['subject']:18s} {r['source']:13s} {r['n_blocks']:4d} "
              f"{r['real_minus_no_edge_next_event_average']:+10.5f} {b['estimate']:+10.5f} "
              f"{cis:>26s} {b.get('fraction_favourable', float('nan')):6.2f} "
              f"{r['block_level_support']}")
    print("\nblock-level support:", summary["block_level_support"] or "none")


if __name__ == "__main__":
    main()
