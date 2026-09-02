#!/usr/bin/env python3
"""A no-state reference: how well does the patient's own average rate do?

"M1 beats M0" is only interesting if M0 itself knows something.  This fits a
negative binomial to the **TRAIN** block counts alone -- no state, no background,
no clock, just "this patient averages so many events per half hour, with this much
spread" -- and scores the same held-out disjoint blocks with it.

Every arm's score is then reportable as a gain over a reference a reader can hold
in their head.  Fitted by moments on TRAIN only, so it cannot see the blocks it
is scored on.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import gammaln

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_group_event_state_h3.io import write_json_atomic  # noqa: E402
from src.topic5_group_event_state_h3.support import (  # noqa: E402
    MAIN_HORIZONS_MINUTES,
    POSTICTAL_EXCLUSION_SECONDS,
    build_coverage_segments,
    cut_intervals_at_seizures,
    load_block_time_ranges,
    load_seizures,
    segment_anchor_grid,
    segment_bounds,
    select_disjoint_anchors,
    split_by_physical_time,
)

V0_1 = Path("/home/honglab/leijiaxin/HFOsp/results/epi_prssm/group_event_state/v0_1")
DATASET = Path("/data/hfosp_group_event_state_v0_1/dataset")
FEATURES = Path("/data/hfosp_group_event_state_v0_2/agent_c/features")
OUT = ROOT / "results/epi_prssm/group_event_state/v0_2/h3/machine/climatology_reference.json"
MIN_BLOCKS = 6


def nb_logpmf(k: np.ndarray, mu: float, phi: float) -> np.ndarray:
    k = np.asarray(k, dtype=np.float64)
    return (
        gammaln(k + phi) - gammaln(phi) - gammaln(k + 1.0)
        + phi * (np.log(phi) - np.log(phi + mu))
        + k * (np.log(mu) - np.log(phi + mu))
    )


def fit_nb_by_moments(counts: np.ndarray) -> tuple[float, float]:
    """Mean and dispersion from TRAIN counts; falls back to near-Poisson."""

    mu = float(np.mean(counts)) if counts.size else 1e-3
    var = float(np.var(counts)) if counts.size else mu
    mu = max(mu, 1e-3)
    phi = mu**2 / max(var - mu, 1e-6) if var > mu else 1e6
    return mu, float(np.clip(phi, 1e-3, 1e6))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizons", nargs="*", type=int, default=list(MAIN_HORIZONS_MINUTES))
    parser.add_argument("--subjects", nargs="*", default=None)
    args = parser.parse_args()

    subjects = args.subjects or sorted(p.name for p in DATASET.iterdir() if (p / "index.json").exists())
    rows = []
    for subject in subjects:
        segments = build_coverage_segments(load_block_time_ranges(V0_1 / "block_inventory.csv", subject))
        cut = cut_intervals_at_seizures(
            segments, load_seizures(DATASET / subject / "index.json"),
            postictal_exclusion_s=POSTICTAL_EXCLUSION_SECONDS,
        )
        intervals = split_by_physical_time(cut)
        with np.load(FEATURES / f"{subject}.npz") as feat:
            t = feat["t_abs"].astype(np.float64)

        bounds = segment_bounds(intervals)
        row = {"subject": subject, "horizons": {}}
        for horizon in args.horizons:
            span = float(horizon) * 60.0
            train_counts, test_counts = [], []
            for segment_id in sorted(bounds):
                lo, hi = bounds[segment_id]
                grid = segment_anchor_grid(lo, hi)
                members = [i for i in intervals if i.segment_id == segment_id]
                # TRAIN uses every valid anchor (it is a fit, not a denominator);
                # the held-out side uses only the pre-registered disjoint blocks.
                for anchor, split, _seg in select_disjoint_anchors(
                    grid, members, horizon, disjoint_exposure=False
                ):
                    n = int(((t >= anchor) & (t < anchor + span)).sum())
                    (test_counts if split == "development_test" else
                     train_counts if split == "train" else []).append(n)
            if len(train_counts) < 3 or len(test_counts) < MIN_BLOCKS:
                row["horizons"][str(horizon)] = {
                    "status": "insufficient_blocks",
                    "n_train_blocks": len(train_counts),
                    "n_test_blocks": len(test_counts),
                }
                continue
            mu, phi = fit_nb_by_moments(np.asarray(train_counts, dtype=np.float64))
            scores = nb_logpmf(np.asarray(test_counts), mu, phi)
            row["horizons"][str(horizon)] = {
                "status": "ok",
                "n_train_blocks": len(train_counts),
                "n_test_blocks": len(test_counts),
                "train_mean_count": mu,
                "train_dispersion_phi": phi,
                "mean_count_logscore": float(np.mean(scores)),
                "median_test_count": float(np.median(test_counts)),
            }
        rows.append(row)
        summary = " ".join(
            f"{h}m:{row['horizons'][str(h)].get('mean_count_logscore', float('nan')):.3f}"
            for h in args.horizons
        )
        print(f"{subject:26s} {summary}", flush=True)

    write_json_atomic(
        {
            "reference": "negative binomial fitted by moments on TRAIN block counts only",
            "note": "no state, no background, no clock; the floor every arm must beat",
            "min_test_blocks": MIN_BLOCKS,
            "horizons_minutes": args.horizons,
            "subjects": rows,
        },
        OUT,
    )
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
