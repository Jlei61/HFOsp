#!/usr/bin/env python3
"""Model-free directional persistence of interictal rank-set sequences.

No RNN, no fitting.  Take each event's rank-set centroids, form the step
vectors between consecutive rank sets, and measure how aligned each step is
with the one before it.

Drift along a *fixed* axis is the wrong statistic here: the models under test
say the direction is chosen by the event's own early movement, so the first
step is unbiased by construction and any fixed-axis drift averages to zero.
What a directional transport motif predicts is *persistence* -- consecutive
steps pointing the same way -- and that is axis-free.

The reference is not zero.  A contact cannot be recruited twice, so leaving a
region anti-correlates consecutive steps all by itself; on synthetic cells with
no direction at all the mean alignment is about -0.11.  The null therefore
shuffles the rank order inside each event, preserving which contacts took part
and how many rank sets there were, and destroying only the order.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

N_ANGLES = 24
N_PERMUTATIONS = 200


def step_vectors(ranks: np.ndarray, xy: np.ndarray) -> list[np.ndarray]:
    """Centroid-to-centroid step vectors of every event with at least three ranks."""
    out = []
    for row in ranks:
        present = row[row >= 0]
        if present.size < 3:
            continue
        length = int(present.max()) + 1
        centroids = [xy[row == step].mean(axis=0) for step in range(length)
                     if np.any(row == step)]
        if len(centroids) < 3:
            continue
        steps = np.diff(np.asarray(centroids), axis=0)
        norm = np.linalg.norm(steps, axis=1)
        steps = steps[norm > 1e-9]
        if steps.shape[0] >= 2:
            out.append(steps)
    return out


def persistence(steps: list[np.ndarray]) -> dict:
    values, first_alignment = [], []
    for block in steps:
        norm = np.linalg.norm(block, axis=1)
        cosine = (block[1:] * block[:-1]).sum(1) / (norm[1:] * norm[:-1])
        values.extend(cosine.tolist())
        later = block[1:]
        later_norm = np.linalg.norm(later, axis=1)
        first_alignment.extend(((later @ block[0]) / (later_norm * norm[0])).tolist())
    values = np.asarray(values)
    first_alignment = np.asarray(first_alignment)
    if values.size == 0:
        return {"n_pairs": 0}
    return {"n_pairs": int(values.size),
            "mean_consecutive_cosine": float(values.mean()),
            "p_same_half_plane": float(np.mean(values > 0)),
            "n_first_pairs": int(first_alignment.size),
            "mean_first_step_cosine": float(first_alignment.mean())
            if first_alignment.size else float("nan")}


def order_shuffle(ranks: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Keep participation and event length; destroy only the order."""
    out = ranks.copy()
    for row in out:
        labels = row[row >= 0]
        if labels.size > 1:
            row[row >= 0] = rng.permutation(labels)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", default="GEOMETRY_ONLY_PCA2")
    parser.add_argument("--out-root", type=Path,
                        default=ROOT / "results/topic5_dynamical_motif_rnn_v0_1")
    parser.add_argument("--suffix", default="")
    parser.add_argument("--permutations", type=int, default=N_PERMUTATIONS)
    args = parser.parse_args()

    started = time.time()
    cache = args.out_root / "frame_cache" / args.frame
    rows = []
    for directory in sorted(p for p in cache.iterdir() if p.is_dir()):
        plane = np.load(directory / "plane.npz", allow_pickle=False)
        events = np.load(directory / "events.npz", allow_pickle=True)
        provenance = json.loads((directory / "provenance.json").read_text())
        xy = np.asarray(plane["contacts_xy_mm"], dtype=float)
        ranks = np.asarray(events["ranks"])
        split = np.asarray(events["split"])
        observed = ranks[split >= 0] if (split >= 0).any() else ranks
        measured = persistence(step_vectors(observed, xy))
        if measured.get("n_pairs", 0) == 0:
            continue
        rng = np.random.default_rng(20260816)
        null = np.asarray([
            persistence(step_vectors(order_shuffle(observed, rng), xy))
            .get("mean_consecutive_cosine", np.nan)
            for _ in range(int(args.permutations))])
        null = null[np.isfinite(null)]
        excess = measured["mean_consecutive_cosine"] - (float(null.mean()) if null.size else np.nan)
        rows.append({
            "frame": args.frame, "subject": provenance["subject"],
            "n_contacts": provenance["n_contacts"],
            "geometry_class": provenance.get("geometry_class"),
            "n_events": int(observed.shape[0]),
            **measured,
            "null_mean_consecutive_cosine": float(null.mean()) if null.size else float("nan"),
            "null_sd": float(null.std(ddof=1)) if null.size > 1 else float("nan"),
            "persistence_excess": float(excess),
            "z_vs_order_shuffle": float(excess / null.std(ddof=1))
            if null.size > 1 and null.std(ddof=1) > 0 else float("nan"),
            "p_value_vs_order_shuffle": float(
                (np.sum(null >= measured["mean_consecutive_cosine"]) + 1) / (null.size + 1))
            if null.size else float("nan"),
        })
        print(f"[persistence] {provenance['subject']}: cos={measured['mean_consecutive_cosine']:+.4f} "
              f"null={rows[-1]['null_mean_consecutive_cosine']:+.4f} "
              f"excess={excess:+.4f} z={rows[-1]['z_vs_order_shuffle']:+.2f}", flush=True)

    table = pd.DataFrame(rows)
    suffix = args.suffix
    table.to_csv(args.out_root / f"MODEL_FREE_PERSISTENCE{suffix}.csv", index=False)
    summary = {"contract": "topic5_dynamical_motif_persistence_v0_1",
               "frame": args.frame, "n_subjects": int(len(table)),
               "n_permutations": int(args.permutations),
               "statistic": "mean cosine between consecutive rank-set step vectors",
               "null": "rank order shuffled inside each event; participation and length kept",
               "seconds": time.time() - started}
    if not table.empty:
        summary.update({
            "median_observed": float(table.mean_consecutive_cosine.median()),
            "median_null": float(table.null_mean_consecutive_cosine.median()),
            "median_excess": float(table.persistence_excess.median()),
            "n_positive_excess": int((table.persistence_excess > 0).sum()),
            "n_p_below_0_05": int((table.p_value_vs_order_shuffle < 0.05).sum()),
            "median_z": float(table.z_vs_order_shuffle.median()),
        })
    (args.out_root / f"MODEL_FREE_PERSISTENCE_SUMMARY{suffix}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
