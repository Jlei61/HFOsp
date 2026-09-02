#!/usr/bin/env python3
"""C2: can the M0/M1/M2 comparison see an edge that is really there?

Runs the identical training protocol on synthetic recordings whose feedback edge
is known by construction, and reports whether the contrasts recover it.  A pass
calibrates the instrument at this support and effect size.  It is not a gate on
the human analysis and it is not evidence about H3.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_group_event_state_h3.io import write_json_atomic  # noqa: E402
from src.topic5_group_event_state_h3.models import ARM_NAMES, H3Config, build_model  # noqa: E402
from src.topic5_group_event_state_h3.support import (  # noqa: E402
    MAIN_HORIZONS_MINUTES,
    build_coverage_segments,
    split_by_physical_time,
)
from src.topic5_group_event_state_h3.synthetic import TRUTHS, generate  # noqa: E402
from src.topic5_group_event_state_h3.timeline import build_timelines  # noqa: E402
from src.topic5_group_event_state_h3.train import (  # noqa: E402
    TrainConfig, prepare_subject, run_epoch, train_arm, validation_objective,
)


def endpoint_scores(collected, horizons) -> dict[str, float]:
    """Held-out count and mark log scores, kept apart.

    Pooling them halves a count-only truth: a generator whose feedback edge acts
    on the rate cannot move the mark endpoint at all, so a pooled objective
    reports roughly half the effect and adds the mark endpoint's noise on top.
    That is exactly the dilution that made the first calibration pass look blind.
    """

    out = {}
    for name, key in (("count", "count"), ("mark", "mark")):
        total, n = 0.0, 0
        for horizon in horizons:
            values = collected.get(f"{key}_{horizon}")
            has = collected.get(f"has_{horizon}")
            if values is None or values.size == 0:
                continue
            selected = values[has] if key == "mark" and has is not None else values
            if selected.size == 0:
                continue
            total -= float(np.sum(selected))
            n += int(selected.size)
        out[name] = total / n if n else float("nan")
    return out

OUT_ROOT = ROOT / "results/epi_prssm/group_event_state/v0_2/h3"


def score_one(patient, arm, seed, horizons, device, train_cfg):
    intervals = split_by_physical_time(build_coverage_segments(patient.block_ranges))
    timelines = build_timelines(
        intervals, patient.features.t_abs,
        (patient.background_time, patient.background_features),
    )
    tensors = prepare_subject(patient.features, timelines, intervals, horizons, device)
    model = build_model(
        arm, H3Config(horizons_minutes=tuple(horizons)), tensors.n_drive_features,
        patient.features.count_features.shape[1], patient.features.mark_features.shape[1],
        seed, mean_event_rate_hz=tensors.train_event_rate_hz,
    ).to(device)
    fit = train_arm(model, tensors, horizons, device, train_cfg, seed=seed)
    model.eval()
    with torch.no_grad():
        _loss, collected = run_epoch(
            model, tensors, horizons, device, train_cfg,
            train_split="train", collect_splits=("development_test",),
        )
    held_out = collected.get("development_test", {})
    return (
        validation_objective(held_out, horizons),
        endpoint_scores(held_out, horizons),
        fit,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truths", nargs="*", default=list(TRUTHS))
    parser.add_argument(
        "--strengths", nargs="*", type=float, default=[0.6],
        help="dose ladder for the feedback edge; the report is a detection "
             "threshold in units of the latent's own variance, not a pass/fail",
    )
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hours", type=float, default=120.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--horizons", nargs="*", type=int, default=list(MAIN_HORIZONS_MINUTES))
    parser.add_argument("--tag", default="synthetic")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train_cfg = TrainConfig(lr=args.lr, max_epochs=args.max_epochs, max_train_seconds=1200.0)
    records = []
    ladder = [(t, st) for t in args.truths
              for st in (args.strengths if t in ("count_feedback", "mark_feedback") else [0.0])]
    for truth, strength in ladder:
        for seed in args.seeds:
            patient = generate(
                truth, hours=args.hours, seed=seed,
                feedback_strength=strength if strength else 0.6,
            )
            row = {
                "truth": truth,
                "strength": strength,
                "feedback_variance_fraction": patient.params["feedback_variance_fraction"],
                "seed": seed,
                "n_events": patient.params["n_events"],
                "arms": {},
            }
            for arm in ARM_NAMES:
                started = time.time()
                score, endpoints, fit = score_one(
                    patient, arm, seed, args.horizons, device, train_cfg
                )
                row["arms"][arm] = {
                    "held_out_objective": float(score),
                    "held_out_count": float(endpoints["count"]),
                    "held_out_mark": float(endpoints["mark"]),
                    "selected_epoch": fit["selected_epoch"],
                    "n_epochs_run": fit["n_epochs_run"],
                    "seconds": round(time.time() - started, 1),
                }
            # Lower score is better, so a positive gain is an improvement.  Count
            # and mark are reported separately, because the two synthetic truths
            # act on different endpoints and a pooled number hides which moved.
            for key, suffix in (("held_out_objective", ""), ("held_out_count", "_count"),
                                ("held_out_mark", "_mark")):
                m0 = row["arms"]["M0_no_feedback"][key]
                m1 = row["arms"]["M1_count_rate_feedback"][key]
                m2 = row["arms"]["M2_mark_specific_feedback"][key]
                row[f"gain_M1_over_M0{suffix}"] = float(m0 - m1)
                row[f"gain_M2_over_M1{suffix}"] = float(m1 - m2)
            m0 = row["arms"]["M0_no_feedback"]["held_out_objective"]
            m1 = row["arms"]["M1_count_rate_feedback"]["held_out_objective"]
            m2 = row["arms"]["M2_mark_specific_feedback"]["held_out_objective"]
            records.append(row)
            print(f"{truth:16s} str={strength:5.1f} varfrac={row['feedback_variance_fraction']:.4f} "
                  f"seed{seed} n_ev={row['n_events']:7d} "
                  f"COUNT gain(M1-M0)={row['gain_M1_over_M0_count']:+.4f} "
                  f"gain(M2-M1)={row['gain_M2_over_M1_count']:+.4f} | "
                  f"MARK gain(M1-M0)={row['gain_M1_over_M0_mark']:+.4f} "
                  f"gain(M2-M1)={row['gain_M2_over_M1_mark']:+.4f}", flush=True)

    summary = {}
    for truth, strength in ladder:
        rows = [r for r in records if r["truth"] == truth and r["strength"] == strength]
        if not rows:
            continue
        summary[f"{truth}|strength={strength}"] = {
            "n_seeds": len(rows),
            "feedback_variance_fraction": float(
                np.median([r["feedback_variance_fraction"] for r in rows])
            ),
            **{
                f"median_{key}": float(np.median([r[key] for r in rows]))
                for key in ("gain_M1_over_M0_count", "gain_M2_over_M1_count",
                            "gain_M1_over_M0_mark", "gain_M2_over_M1_mark")
            },
            "n_seeds_M1_better_count": int(sum(r["gain_M1_over_M0_count"] > 0 for r in rows)),
            "n_seeds_M2_better_mark": int(sum(r["gain_M2_over_M1_mark"] > 0 for r in rows)),
        }
    write_json_atomic(
        {"tier": "implementation_conformance_only_not_evidence_about_H3",
         "config": {"hours": args.hours, "lr": args.lr, "max_epochs": args.max_epochs,
                    "horizons": args.horizons, "seeds": args.seeds,
                    "strengths": args.strengths},
         "summary": summary, "runs": records},
        OUT_ROOT / "machine" / f"synthetic_recovery_{args.tag}.json",
    )
    print("\n" + "\n".join(f"{k:16s} {v}" for k, v in summary.items()))


if __name__ == "__main__":
    main()
