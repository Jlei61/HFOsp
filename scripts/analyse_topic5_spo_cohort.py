"""Nested comparisons, parameter reliability and operator ablations.

Three questions, kept apart because they license different statements:

1. which spatial component actually improves held-out prediction -- answered by
   consecutive differences in the nested family, patient by patient;
2. whether a patient's fitted parameters are more like themselves refitted than
   like another patient's -- answered only for the layers the recovery gate
   certified, and with the patient as the unit;
3. what breaks when a component is switched off -- answered by editing the
   fitted operator, not by retraining, so nothing else moves.

The patient is the unit everywhere. Seeds are pooled inside a patient.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.topic5_spatial_latent_rnn import build_event_tensors  # noqa: E402
from src.topic5_spatial_propagation_operator import (  # noqa: E402
    OperatorConfig, SPOModel,
)
from scripts.train_topic5_spo_unit import evaluate, load, partition  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/topic5_spatial_propagation_operator_v0_2"

# Only a nested pair isolates a component. Written out, the parameter sets are
#   STATIC              {}
#   FIELD_NULL          {gamma_a, beta, xi, gamma_r, eta}
#   ISOTROPIC_DIFFUSION {gamma_a,                   eta, D}
#   ANISOTROPIC_DRIFT   {gamma_a,                   eta, D_par, D_perp, v}
#   ANISOTROPIC_RECOVERY{gamma_a, beta, xi, gamma_r, eta, D_par, D_perp, v}
# FIELD_NULL and ISOTROPIC_DIFFUSION do not nest -- one has recovery and no
# transport, the other transport and no recovery -- so that pair cannot be read
# as what diffusion buys. It is kept, because it is a real question, but under a
# name that says what it actually compares.
#
# The clean test of spatial transport is the full operator against FIELD_NULL:
# both carry recovery, and the only difference is the three transport numbers.
LADDER = [
    ("field_over_static", "FIELD_NULL", "STATIC"),
    ("transport_over_no_transport", "ANISOTROPIC_RECOVERY", "FIELD_NULL"),
    ("drift_over_isotropic", "ANISOTROPIC_DRIFT", "ISOTROPIC_DIFFUSION"),
    ("recovery_over_drift", "ANISOTROPIC_RECOVERY", "ANISOTROPIC_DRIFT"),
    ("full_over_static", "ANISOTROPIC_RECOVERY", "STATIC"),
    ("transport_no_recovery_over_recovery_no_transport",
     "ISOTROPIC_DIFFUSION", "FIELD_NULL"),
]
NOT_NESTED = {"transport_no_recovery_over_recovery_no_transport"}
FULL = "ANISOTROPIC_RECOVERY"
METRIC = "test_next_bce"

ABLATIONS = {
    "no_drift": {"v": 0.0},
    "reversed_drift": {"v": "flip"},
    "isotropic": {"tie_diffusion": True},
    "no_recovery": {"beta": 0.0},
    "no_transport": {"v": 0.0, "D": 0.0},
}


def collect(root: Path) -> list[dict]:
    rows = []
    if not root.exists():
        return rows
    for subject_dir in sorted(root.iterdir()):
        for variant_dir in sorted(subject_dir.iterdir()):
            for seed_dir in sorted(variant_dir.glob("seed*")):
                done = seed_dir / "DONE.json"
                if done.exists():
                    rows.append(json.loads(done.read_text()))
    return rows


def paired(a: dict, b: dict, subjects: list[str]) -> dict:
    """Positive means the first model wins; the metric is a loss."""
    common, delta = [], []
    for s in subjects:
        if s not in a or s not in b:
            continue
        seeds = sorted(set(a[s]) & set(b[s]))
        if not seeds:
            continue
        common.append(s)
        delta.append(float(np.median([b[s][k] for k in seeds])
                           - np.median([a[s][k] for k in seeds])))
    if len(common) < 3:
        return {"status": "INSUFFICIENT", "n": len(common)}
    delta = np.array(delta)
    rng = np.random.default_rng(20260806)
    boot = np.array([np.median(rng.choice(delta, len(delta), replace=True))
                     for _ in range(4000)])
    return {
        "status": "COMPLETE", "n": len(common),
        "median_delta": float(np.median(delta)),
        "bootstrap_95ci": [float(np.percentile(boot, 2.5)),
                           float(np.percentile(boot, 97.5))],
        "n_positive": int((delta > 0).sum()),
        "wilcoxon_two_sided_p": float(stats.wilcoxon(delta).pvalue),
        "per_patient_delta": dict(zip(common, map(float, delta))),
    }


# --- 2. parameter reliability -------------------------------------------
PARAMETER_KEYS = ("v", "D_parallel", "D_perp", "gamma_a", "beta", "gamma_r")


def theta(estimates: dict) -> np.ndarray:
    return np.array([
        estimates["v"],
        np.log(max(estimates["D_parallel"], 1e-9)),
        np.log(max(estimates["D_perp"], 1e-9)),
        estimates["gamma_a"],
        estimates["beta"],
        estimates["gamma_r"],
    ], float)


def reliability(rows: list[dict]) -> dict:
    by_subject: dict[str, dict[int, np.ndarray]] = {}
    for r in rows:
        if r["variant"] != FULL:
            continue
        by_subject.setdefault(r["subject"], {})[r["seed"]] = theta(r["parameters"])
    paired_subjects = {s: v for s, v in by_subject.items() if len(v) >= 2}
    if len(paired_subjects) < 3:
        return {"status": "INSUFFICIENT_SEEDS", "n_with_two_seeds": len(paired_subjects)}

    stacked = np.stack([v[sorted(v)[0]] for v in by_subject.values()])
    scale = stacked.std(axis=0)
    scale[scale == 0] = 1.0

    def similarity(x: np.ndarray, y: np.ndarray) -> float:
        return float(-np.linalg.norm((x - y) / scale))

    rows_out = []
    for subject, seeds in sorted(paired_subjects.items()):
        first, second = sorted(seeds)[:2]
        within = similarity(seeds[first], seeds[second])
        others = [similarity(seeds[first], v[sorted(v)[1]])
                  for q, v in paired_subjects.items() if q != subject]
        rows_out.append({"subject": subject, "within": within,
                         "between_median": float(np.median(others)),
                         "delta": within - float(np.median(others))})
    deltas = np.array([r["delta"] for r in rows_out])
    rng = np.random.default_rng(20260806)
    boot = np.array([np.median(rng.choice(deltas, len(deltas), replace=True))
                     for _ in range(4000)])
    return {
        "status": "COMPLETE", "n_patients": len(rows_out),
        "median_delta": float(np.median(deltas)),
        "bootstrap_95ci": [float(np.percentile(boot, 2.5)),
                           float(np.percentile(boot, 97.5))],
        "n_positive": int((deltas > 0).sum()),
        "wilcoxon_exact_p": float(stats.wilcoxon(deltas, mode="exact").pvalue),
        "geometry_confound": geometry_confound(by_subject),
        "caveat": ("two fits of one patient share that patient's electrode "
                   "geometry as well as their events, and the coefficients are "
                   "expressed in grid cells, so a patient-specific fit is not by "
                   "itself a patient-specific mechanism; see geometry_confound"),
        "per_patient": rows_out,
    }


def geometry_confound(by_subject: dict) -> dict:
    """How much of the between-patient spread is patient-level nuisance?

    The reliability test asks whether a patient's operator resembles itself more
    than someone else's. Two fits of one patient share their geometry, and every
    coefficient is in grid cells, so geometry alone would produce that result
    with no mechanism behind it -- the same trap that made v0.1 report a
    patient-specific propagation order which turned out to be node position.

    This does not remove the confound. It measures it, so the closeout cannot
    quietly claim more than the design supports.
    """
    descriptors, thetas = [], []
    for subject, seeds in sorted(by_subject.items()):
        cache = OUT / "cache" / subject
        if not (cache / "grid.npz").exists():
            continue
        grid = np.load(cache / "grid.npz")
        ranks = np.load(cache / "events.npz")["group_ids"]
        pitch = float(np.median(np.diff(np.unique(np.round(grid["centres"][:, 0], 6)))))
        descriptors.append([
            pitch, float(grid["sigma_mm"][0]), float(ranks.shape[1]),
            float(np.median([r[r >= 0].max() + 1 for r in ranks if (r >= 0).any()])),
        ])
        thetas.append(by_subject[subject][sorted(seeds)[0]])
    if len(descriptors) < 6:
        return {"status": "INSUFFICIENT", "n": len(descriptors)}
    D, T = np.array(descriptors), np.stack(thetas)
    names = ("grid_pitch_mm", "read_kernel_sigma_mm", "n_contacts",
             "median_event_length")
    worst = {}
    for j, parameter in enumerate(PARAMETER_KEYS):
        best = max(
            ((abs(float(stats.spearmanr(D[:, i], T[:, j]).statistic)), names[i])
             for i in range(D.shape[1])), key=lambda t: t[0])
        worst[parameter] = {"strongest_geometry_correlation": best[0],
                            "with": best[1]}
    strongest = max(v["strongest_geometry_correlation"] for v in worst.values())
    return {
        "status": "COMPLETE", "n_patients": len(descriptors),
        "per_parameter": worst,
        "strongest_absolute_spearman": float(strongest),
        "descriptors": list(names),
        "reading": ("a fitted coefficient that tracks a patient-level descriptor "
                    "this closely is not evidence of a patient-specific mechanism"
                    if strongest >= 0.6 else
                    "no single coefficient is strongly predicted by these "
                    "descriptors, which weakens but does not remove the confound"),
    }


# --- 3. operator ablations ----------------------------------------------
def ablate(rows: list[dict], subjects: list[str]) -> list[dict]:
    out = []
    for subject in subjects:
        unit = OUT / "per_subject" / subject / FULL / "seed1"
        if not (unit / "DONE.json").exists():
            continue
        grid, H, events = load(subject)
        shape = tuple(int(v) for v in grid["shape"])
        config = OperatorConfig(
            variant=FULL, n_contacts=H.shape[0], grid_shape=shape,
            microsteps=int(json.loads((unit / "config.json").read_text())["microsteps"]),
            seed=1, observation_operator=H, grid_mask=grid["mask"],
        )
        model = SPOModel(config)
        model.load_state_dict(torch.load(unit / "checkpoint.pt", weights_only=True))
        model.eval()
        parts = partition(events, None)
        baseline = evaluate(model, parts["test"])["next_bce"]
        for name, edit in ABLATIONS.items():
            state = {k: v.clone() for k, v in model.state_dict().items()}
            with torch.no_grad():
                if edit.get("v") == "flip":
                    model.operator.v.mul_(-1.0)
                elif "v" in edit:
                    model.operator.v.fill_(0.0)
                if edit.get("D") == 0.0:
                    model.operator.raw_D_parallel.fill_(-30.0)
                    model.operator.raw_D_perp.fill_(-30.0)
                if edit.get("tie_diffusion"):
                    model.operator.raw_D_perp.copy_(model.operator.raw_D_parallel)
                if edit.get("beta") == 0.0:
                    model.operator.raw_beta.fill_(-30.0)
                after = evaluate(model, parts["test"])["next_bce"]
            model.load_state_dict(state)
            out.append({"subject": subject, "ablation": name,
                        "baseline_next_bce": baseline, "ablated_next_bce": after,
                        "delta_next_bce": after - baseline})
        print(f"  ablated {subject}", flush=True)
    return out


# --- 4. leave-contact-out ------------------------------------------------
HELDOUT_FLOOR = "STATIC"


def leave_contact_out(subjects: list[str]) -> dict:
    """Raw held-out performance, per arm. Never a relative degradation.

    v0.1 reported how far each arm fell from its OWN retained score and read the
    smaller fall as the better generaliser. That systematically favours whichever
    arm was already worse everywhere -- a low baseline has less room to drop. The
    absolute score on the withheld contacts is the comparison; the retained score
    is context and nothing more.

    STATIC is the floor. With the withheld contact's bias neutralised it assigns
    every withheld contact the same number, so it knows nothing about them at
    all. An arm that beats it is using where the contact sits.
    """
    rows = collect(OUT / "leave_contact_out")
    if not rows:
        return {"status": "NOT_RUN"}

    def table(key):
        out: dict[str, dict[str, dict[int, float]]] = {}
        for r in rows:
            if key in r:
                out.setdefault(r["variant"], {}).setdefault(
                    r["subject"], {})[r["seed"]] = r[key]
        return out

    heldout, retained = table("heldout_next_bce"), table("retained_next_bce")
    heldout_top1 = table("heldout_top1")

    # The arms are compared patient by patient, which only means anything if they
    # withheld the same contacts. They are drawn from the run seed alone, so they
    # should agree -- checked against the data rather than the code.
    withheld: dict[str, set] = {}
    for r in rows:
        if "holdout_contacts" in r:
            withheld.setdefault(r["subject"], set()).add(tuple(r["holdout_contacts"]))
    disagreeing = sorted(s for s, v in withheld.items() if len(v) > 1)

    report = {
        "status": "COMPLETE",
        "question": ("does the model predict a contact it never trained on, from "
                     "where that contact sits"),
        "condition": ("strong: the withheld contact is absent from the training "
                      "ranks, absent from the training loss, absent from the test "
                      "input, and carries the average retained contact's bias -- "
                      "so the only thing left that is specific to it is position"),
        "comparison_rule": ("absolute held-out score, patient-paired against the "
                            "floor arm; relative degradation is not reported and "
                            "must not be substituted"),
        "floor_arm": HELDOUT_FLOOR,
        "arms_withheld_the_same_contacts": not disagreeing,
        "patients_where_arms_disagree": disagreeing,
        "n_units": len(rows),
        "absolute": {}, "over_floor": {},
    }
    for variant in sorted(heldout):
        per_patient = {s: float(np.median(list(v.values())))
                       for s, v in heldout[variant].items()}
        report["absolute"][variant] = {
            "n_patients": len(per_patient),
            "median_heldout_next_bce": float(np.median(list(per_patient.values()))),
            "median_heldout_top1": float(np.median([
                np.median(list(v.values())) for v in heldout_top1[variant].values()]))
            if variant in heldout_top1 else None,
            "median_retained_next_bce": float(np.median([
                np.median(list(v.values())) for v in retained[variant].values()]))
            if variant in retained else None,
            "per_patient_heldout_next_bce": per_patient,
        }
    if HELDOUT_FLOOR in heldout:
        for variant in sorted(heldout):
            if variant != HELDOUT_FLOOR:
                report["over_floor"][variant] = paired(
                    heldout[variant], heldout[HELDOUT_FLOOR], subjects)
    else:
        report["over_floor"] = {"status": "FLOOR_ARM_MISSING"}
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-ablations", action="store_true")
    args = parser.parse_args()

    subjects = json.loads(
        (OUT / "INPUT_MANIFEST.json").read_text())["frozen_cohort"]["primary"]
    rows = collect(OUT / "per_subject")
    if not rows:
        raise SystemExit("no completed units")

    with (OUT / "patient_prediction_metrics.csv").open("w", newline="") as fh:
        fields = ["subject", "variant", "seed", "test_next_bce", "test_contact_nll",
                  "test_top1", "test_stop_bce", "epochs_run", "converged",
                  "hit_epoch_ceiling_while_improving", "n_parameters"]
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    table: dict[str, dict[str, dict[int, float]]] = {}
    for r in rows:
        table.setdefault(r["variant"], {}).setdefault(r["subject"], {})[r["seed"]] = \
            r[METRIC]

    summary = {
        "contract": "topic5_spo_cohort_v0_2",
        "metric": METRIC,
        "metric_direction": "lower is better; a positive delta means the first model wins",
        "unit_of_analysis": "patient; seeds pooled within patient",
        "geometry_status": json.loads(
            (OUT / "INPUT_MANIFEST.json").read_text())["geometry_status"],
        "n_units": len(rows),
        "variants_present": sorted(table),
        "ladder": {},
        "convergence": {
            v: {"converged": sum(1 for r in rows
                                 if r["variant"] == v and r["converged"]),
                "hit_ceiling": sum(1 for r in rows if r["variant"] == v
                                   and r["hit_epoch_ceiling_while_improving"])}
            for v in sorted(table)
        },
    }
    for name, a, b in LADDER:
        if a in table and b in table:
            summary["ladder"][name] = paired(table[a], table[b], subjects)

    summary["parameter_reliability"] = reliability(rows)

    at_bound = [r for r in rows if r["variant"] == FULL
                and r["parameters"].get("anisotropy_is_bounded_estimate")]
    summary["parameters_at_stability_bound"] = {
        "n_units": len(at_bound),
        "subjects": sorted({r["subject"] for r in at_bound}),
        "note": ("a diffusion coefficient sitting on the explicit-scheme bound is "
                 "censored; the anisotropy built from it is a bound, not an estimate"),
    }

    summary["leave_contact_out"] = leave_contact_out(subjects)
    (OUT / "cohort_statistics.json").write_text(json.dumps(summary, indent=1))

    with (OUT / "parameter_estimates.csv").open("w", newline="") as fh:
        keys = sorted({k for r in rows if r["variant"] == FULL for k in r["parameters"]})
        writer = csv.writer(fh)
        writer.writerow(["subject", "seed", *keys])
        for r in rows:
            if r["variant"] == FULL:
                writer.writerow([r["subject"], r["seed"],
                                 *[r["parameters"].get(k) for k in keys]])

    if not args.skip_ablations:
        ablations = ablate(rows, subjects)
        if ablations:
            with (OUT / "operator_ablation.csv").open("w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(ablations[0]))
                writer.writeheader()
                writer.writerows(ablations)
            by_name: dict[str, list[float]] = {}
            for a in ablations:
                by_name.setdefault(a["ablation"], []).append(a["delta_next_bce"])
            summary["ablations"] = {
                name: {"n": len(v), "median_delta_next_bce": float(np.median(v)),
                       "n_worse": int(sum(1 for x in v if x > 0)),
                       "wilcoxon_two_sided_p": float(stats.wilcoxon(v).pvalue)
                       if len(v) >= 3 else None}
                for name, v in sorted(by_name.items())
            }
            (OUT / "cohort_statistics.json").write_text(json.dumps(summary, indent=1))

    print(f"units {len(rows)}   variants {summary['variants_present']}")
    for name, entry in summary["ladder"].items():
        if entry.get("status") == "COMPLETE":
            print(f"  {name:24s} n={entry['n']:2d} median={entry['median_delta']:+.4f} "
                  f"pos={entry['n_positive']}/{entry['n']} "
                  f"p={entry['wilcoxon_two_sided_p']:.3g}")
    rel = summary["parameter_reliability"]
    if rel.get("status") == "COMPLETE":
        print(f"  parameter reliability     n={rel['n_patients']} "
              f"median={rel['median_delta']:+.4f} pos={rel['n_positive']} "
              f"p={rel['wilcoxon_exact_p']:.3g}")
    for name, entry in summary.get("ablations", {}).items():
        print(f"  ablation {name:16s} median={entry['median_delta_next_bce']:+.4f} "
              f"worse in {entry['n_worse']}/{entry['n']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
