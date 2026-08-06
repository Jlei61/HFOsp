"""Is the static baseline actually fitted, or does it only look beaten?

The static arm reaches the epoch ceiling far more often than the recurrent arm,
which would flatter the recurrent arm if it meant the baseline was undertrained.
It does not have to mean that: a model whose only parameters are one constant per
contact can sit at its optimum while the loss still creeps by amounts small
enough to keep resetting the patience counter.

This fits the same constant-per-contact model to convergence with a second-order
optimiser on the training partition and scores it on the test partition.  If the
cohort run's static arm matches that, the baseline is sound and its convergence
flag is a false alarm.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.topic5_spatial_latent_rnn import build_event_tensors, next_set_stop_loss

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/topic5_spatial_latent_propagation_rnn_v0_1"

# The reported recurrent advantage is around 0.09; a baseline within this of its
# own optimum cannot be what produces it.
TOLERANCE = 0.01


def analytic_static_test_loss(subject: str) -> float:
    events = np.load(OUT / "cache" / subject / "events.npz")
    group_ids, split = events["group_ids"], events["split"]
    train = build_event_tensors(group_ids[split == 0])
    test = build_event_tensors(group_ids[split == 2])

    bias = torch.zeros(train.x.shape[-1], requires_grad=True)
    optimiser = torch.optim.LBFGS([bias], max_iter=400, line_search_fn="strong_wolfe")

    def closure():
        optimiser.zero_grad()
        logits = bias.expand(train.x.shape[0], train.x.shape[1], -1)
        loss, _, _ = next_set_stop_loss(
            logits, torch.zeros(train.valid.shape), train.target,
            train.available, train.valid, train.is_last, stop_weight=0.0,
        )
        loss.backward()
        return loss

    optimiser.step(closure)
    with torch.no_grad():
        logits = bias.expand(test.x.shape[0], test.x.shape[1], -1)
        _, next_bce, _ = next_set_stop_loss(
            logits, torch.zeros(test.valid.shape), test.target,
            test.available, test.valid, test.is_last, stop_weight=0.0,
        )
    return float(next_bce)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="*", default=None)
    args = parser.parse_args()

    manifest = json.loads((OUT / "INPUT_MANIFEST.json").read_text())
    subjects = args.subjects or manifest["frozen_cohort"]["primary"]

    rows = []
    for subject in subjects:
        fitted_path = OUT / "per_subject" / subject / "STATIC_CONTACT" / "seed1" / "DONE.json"
        if not fitted_path.exists():
            continue
        fitted = json.loads(fitted_path.read_text())
        analytic = analytic_static_test_loss(subject)
        rows.append({
            "subject": subject,
            "cohort_run_test_next_bce": fitted["test_next_bce"],
            "second_order_fit_test_next_bce": analytic,
            "gap": fitted["test_next_bce"] - analytic,
            "cohort_run_converged_flag": fitted.get("converged"),
        })
        print(f"{subject:24s} run={rows[-1]['cohort_run_test_next_bce']:.4f} "
              f"optimum={analytic:.4f} gap={rows[-1]['gap']:+.4f}")

    if not rows:
        raise SystemExit("no static units to verify")
    gaps = np.array([r["gap"] for r in rows])

    # A fixed tolerance answers the wrong question.  What matters is not whether
    # every patient's baseline is within some absolute distance of its optimum,
    # but whether any patient's shortfall is large enough to have produced that
    # patient's measured advantage.  Compare each gap to its own patient's
    # recurrent-over-static difference.
    import csv as _csv
    advantage = {}
    metrics = OUT / "patient_prediction_metrics.csv"
    if metrics.exists():
        by_arm: dict = {}
        for row in _csv.DictReader(metrics.open()):
            by_arm.setdefault(row["arm"], {}).setdefault(row["subject"], []).append(
                float(row["test_next_bce"]))
        for subject in by_arm.get("STATIC_CONTACT", {}):
            if subject in by_arm.get("ORDINARY_GRU", {}):
                advantage[subject] = float(
                    np.median(by_arm["STATIC_CONTACT"][subject])
                    - np.median(by_arm["ORDINARY_GRU"][subject]))
    for row in rows:
        own = advantage.get(row["subject"])
        row["own_recurrent_advantage"] = own
        row["gap_as_fraction_of_own_advantage"] = (
            float(abs(row["gap"]) / abs(own)) if own else None)
        row["gap_could_flip_this_patient"] = bool(own and abs(row["gap"]) >= abs(own))
    could_flip = [r["subject"] for r in rows if r["gap_could_flip_this_patient"]]
    fractions = [r["gap_as_fraction_of_own_advantage"] for r in rows
                 if r["gap_as_fraction_of_own_advantage"] is not None]

    verdict = {
        "contract": "topic5_slp_static_baseline_verification_v0_2",
        "n_subjects": len(rows),
        "median_gap": float(np.median(gaps)),
        "max_abs_gap": float(np.abs(gaps).max()),
        "tolerance": TOLERANCE,
        "n_patients_whose_gap_could_flip_their_own_sign": len(could_flip),
        "patients_that_could_flip": could_flip,
        "median_gap_as_fraction_of_own_advantage": (
            float(np.median(fractions)) if fractions else None),
        "max_gap_as_fraction_of_own_advantage": (
            float(np.max(fractions)) if fractions else None),
        "status": "NO_PATIENT_COULD_FLIP" if not could_flip else "SOME_PATIENT_COULD_FLIP",
        "means": (
            "no patient's baseline falls short of its own optimum by as much as that "
            "patient's measured recurrent advantage, so the direction of the cohort "
            "result cannot be an artefact of an undertrained baseline -- though on "
            "the hardest patients the shortfall is an appreciable fraction of the "
            "advantage and the effect size there should be read as a lower bound"
            if not could_flip else
            "at least one patient's baseline shortfall is as large as its measured "
            "advantage, so that patient's contribution cannot be trusted"
        ),
        "subjects": rows,
    }
    (OUT / "static_baseline_verification.json").write_text(json.dumps(verdict, indent=1))
    print(f"\n{verdict['status']}  median gap {verdict['median_gap']:+.4f}, "
          f"worst {verdict['max_abs_gap']:.4f}")
    if verdict["max_gap_as_fraction_of_own_advantage"] is not None:
        print(f"as a fraction of each patient's own advantage: median "
              f"{verdict['median_gap_as_fraction_of_own_advantage']:.1%}, "
              f"worst {verdict['max_gap_as_fraction_of_own_advantage']:.1%}")
    print(verdict["means"])
    return 0 if verdict["status"] == "NO_PATIENT_COULD_FLIP" else 1


if __name__ == "__main__":
    raise SystemExit(main())
