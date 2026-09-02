#!/usr/bin/env python3
"""G6: cheap exploratory sidecars on what the fitted models leave behind.

Three questions, none of which trains a new model:

1. Is the residual between the real contact field and the generated one low
   rank?  A low-rank residual would say a small number of extra spatial modes
   would go a long way; a full-rank one says the miss is diffuse.
2. Does the within-event timing proxy ``event_lag_raw`` carry distance
   information the rank order does not?  The models use ordinal rank steps, so
   any distance-lag structure is by construction outside them.
3. Do the answers repeat across recording blocks inside a patient?

These only decide whether a later spec is worth writing.  Nothing here is a
claim about white matter or conduction delay.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_topic5_dynamical_motif_unseen_v0_1 import load_unit_model, write_json  # noqa: E402
from src.topic5_dynamical_motif_data_v0_1 import load_frame_unit  # noqa: E402
from src.topic5_dynamical_motif_rnn_v0_1 import build_motif_event_tensors  # noqa: E402
from src.topic5_dynamical_motif_rollout_v0_1 import stochastic_rollout  # noqa: E402


def effective_rank(matrix: np.ndarray) -> tuple[float, list[float]]:
    singular = np.linalg.svd(np.asarray(matrix, float), compute_uv=False)
    total = float((singular ** 2).sum())
    if total <= 0:
        return float("nan"), []
    share = (singular ** 2) / total
    entropy = -float(np.sum(share * np.log(np.clip(share, 1e-12, None))))
    return float(np.exp(entropy)), share[:5].tolist()


def distance_lag(unit, indices: np.ndarray) -> dict:
    """Does the timing proxy add distance information beyond the rank step?"""
    xy = np.asarray(unit.contacts_xy_mm, float)
    lag = np.asarray(unit.event_lag_raw, float)
    ranks = np.asarray(unit.ranks)
    distances, lags, steps = [], [], []
    for index in indices:
        row, times = ranks[index], lag[index]
        present = np.flatnonzero(row >= 0)
        if present.size < 2 or not np.isfinite(times[present]).all():
            continue
        order = present[np.argsort(row[present])]
        # All ordered pairs inside the event, not just consecutive ones: with
        # consecutive pairs the rank step is 1 for every pair, so "controlling
        # for rank step" would control for a constant and the partial
        # correlation would equal the raw one by construction.
        for position, a in enumerate(order[:-1]):
            for b in order[position + 1:]:
                distances.append(float(np.linalg.norm(xy[a] - xy[b])))
                lags.append(float(times[b] - times[a]))
                steps.append(float(row[b] - row[a]))
    if len(distances) < 50:
        return {"estimable": False, "n_pairs": len(distances)}
    distances = np.asarray(distances)
    lags = np.asarray(lags)
    steps = np.asarray(steps)
    design = np.column_stack([np.ones_like(steps), steps])
    residual_lag = lags - design @ np.linalg.lstsq(design, lags, rcond=None)[0]
    residual_distance = distances - design @ np.linalg.lstsq(design, distances, rcond=None)[0]
    denominator = residual_lag.std() * residual_distance.std()
    partial = (float(np.mean(residual_lag * residual_distance) / denominator)
               if denominator > 0 else float("nan"))
    return {
        "estimable": True, "n_pairs": int(distances.size),
        "raw_correlation": float(np.corrcoef(distances, lags)[0, 1]),
        "partial_correlation_given_rank_step": partial,
        "median_lag_s": float(np.median(lags)),
        "rank_step_range": [float(steps.min()), float(steps.max())],
        "median_distance_mm": float(np.median(distances)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", default="GEOMETRY_ONLY_PCA2")
    parser.add_argument("--tag", default="formal")
    parser.add_argument("--model", default="DM0_ISOTROPIC")
    parser.add_argument("--seed-index", type=int, default=0)
    parser.add_argument("--out-root", type=Path,
                        default=ROOT / "results/topic5_dynamical_motif_rnn_v0_1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--draws", type=int, default=8)
    parser.add_argument("--max-events", type=int, default=3000)
    args = parser.parse_args()

    started = time.time()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    base = args.out_root / args.tag / args.frame
    rows = []
    for subject in sorted(p.name for p in base.iterdir() if p.is_dir()):
        unit_dir = base / subject / args.model / f"seed{args.seed_index}"
        if not (unit_dir / "checkpoint.pt").exists():
            continue
        unit = load_frame_unit(args.out_root, args.frame, subject)
        tensors = build_motif_event_tensors(unit.ranks, unit.contacts_xy_mm)
        unseen = unit.indices(-1)
        if unseen.size > args.max_events:
            unseen = unseen[np.linspace(0, unseen.size - 1, args.max_events).astype(int)]
        model, head, contract, _ = load_unit_model(unit, unit_dir, device)
        starts = tensors["x"][:, 0][torch.as_tensor(unseen)]
        generated = []
        for draw in range(int(args.draws)):
            result = stochastic_rollout(model, head, contract, starts, unit.contacts_xy_mm,
                                        device, mode="FULL_STOP",
                                        rng_label=f"{subject}|g6|{draw}")
            within = np.arange(result["sequence"].shape[1])[None, :] <= result["n_emitted"][:, None]
            generated.append(((result["sequence"] * within[..., None]).sum(1) > 0).astype(float))
        generated = np.mean(generated, axis=0)
        observed = (unit.ranks[unseen] >= 0).astype(float)
        residual = observed - generated
        residual = residual - residual.mean(axis=0, keepdims=True)
        rank_observed, share_observed = effective_rank(observed - observed.mean(0, keepdims=True))
        rank_residual, share_residual = effective_rank(residual)
        timing = distance_lag(unit, unseen)
        rows.append({
            "frame": args.frame, "subject": subject, "model": args.model,
            "n_contacts": unit.n_contacts, "n_events": int(unseen.size),
            "effective_rank_observed_field": rank_observed,
            "effective_rank_residual_field": rank_residual,
            "residual_rank_fraction": rank_residual / max(unit.n_contacts, 1),
            "residual_variance_share": float(np.var(residual) / max(np.var(observed), 1e-12)),
            "leading_residual_share": share_residual[0] if share_residual else None,
            **{f"timing_{k}": v for k, v in timing.items()},
        })
        print(f"[g6] {subject}: residual effective rank {rank_residual:.2f} of "
              f"{unit.n_contacts} contacts; timing partial r = "
              f"{timing.get('partial_correlation_given_rank_step')}", flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(args.out_root / "G6_RESIDUAL_SIDECAR_PER_PATIENT.csv", index=False)
    summary = {"contract": "topic5_dynamical_motif_g6_sidecar_v0_1",
               "frame": args.frame, "model": args.model, "n_subjects": int(len(table)),
               "note": "exploratory only; decides whether a later spec is worth writing",
               "seconds": time.time() - started}
    if not table.empty:
        summary.update({
            "median_residual_effective_rank": float(table.effective_rank_residual_field.median()),
            "median_residual_rank_fraction": float(table.residual_rank_fraction.median()),
            "median_residual_variance_share": float(table.residual_variance_share.median()),
            "n_timing_estimable": int(table.timing_estimable.sum())
            if "timing_estimable" in table else 0,
            "median_timing_raw_correlation": float(table.timing_raw_correlation.median())
            if "timing_raw_correlation" in table else None,
            "median_timing_partial_correlation": float(
                table.timing_partial_correlation_given_rank_step.median())
            if "timing_partial_correlation_given_rank_step" in table else None,
        })
    write_json(args.out_root / "G6_RESIDUAL_SIDECAR_SUMMARY.json", summary)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
