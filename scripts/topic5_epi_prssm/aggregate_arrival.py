#!/usr/bin/env python3
"""Aggregate the arrival channel: does anything slow move the rate, and does it
have to be updated by the discharges themselves?"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from _common import (  # noqa: E402
    OUTPUT_ROOT, atomic_write_csv, atomic_write_json, code_revision, package_hash,
)
from src.topic5_epi_prssm.stats import holm, paired_effect  # noqa: E402

OUT = OUTPUT_ROOT / "arrival_channel"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", default="all34")
    parser.add_argument("--package", default="")
    parser.add_argument("--markov-renewal", action="store_true",
                        help="require the Markov-renewal fits; without it "
                             "the pre-Markov fits are the target family")
    args = parser.parse_args()

    # Fail closed on everything that made the graph-null contrasts wrong, because the
    # same directory already holds both the pre-Markov and the Markov fits of the same
    # seeds: pooling them would compare two model families as if they were arms.
    # Do NOT default to the current package: source edits during a run mean a job can
    # stamp a hash that is newer than the code it actually executed.  Default instead
    # to the modal package among the runs of the requested family, and record it.
    import collections as _c
    seen_packages = _c.Counter()
    for path in sorted((OUT / "runs").glob("*.json")):
        record = json.loads(path.read_text())
        if record.get("cohort") != args.cohort or not record.get("per_patient"):
            continue
        if bool(record.get("markov_renewal", False)) != bool(args.markov_renewal):
            continue
        seen_packages[(record.get("package_hash") or "?")[:12]] += 1
    target_package = args.package or (seen_packages.most_common(1)[0][0]
                                      if seen_packages else "")

    import collections
    candidates, dropped, seen = [], collections.Counter(), {}
    for path in sorted((OUT / "runs").glob("*.json")):
        record = json.loads(path.read_text())
        if record.get("cohort") != args.cohort or not record.get("per_patient"):
            continue
        pkg = (record.get("package_hash") or "?")[:12]
        if pkg != target_package:
            dropped[f"package:{pkg}"] += 1
            continue
        if bool(record.get("markov_renewal", False)) != bool(args.markov_renewal):
            dropped["markov_renewal_mismatch"] += 1
            continue
        key = (record["arm"], record["seed"])
        if key in seen:
            dropped["duplicate_arm_seed"] += 1
            continue
        seen[key] = record["job_id"]
        candidates.append(record)
    runs = candidates

    # a ladder is only a ladder if every rung was fitted over the same seeds
    seeds_by_arm = collections.defaultdict(set)
    for record in runs:
        seeds_by_arm[record["arm"]].add(record["seed"])
    seed_sets = {arm: sorted(v) for arm, v in seeds_by_arm.items()}
    consistent = len({tuple(v) for v in seed_sets.values()}) <= 1
    provenance = {
        "package_pinned_to": target_package,
        "markov_renewal_required": bool(args.markov_renewal),
        "n_runs_kept": len(runs),
        "dropped": dict(dropped),
        "seed_sets_by_arm": seed_sets,
        "seed_sets_consistent": consistent,
    }
    if not consistent:
        atomic_write_json(OUT / "ARRIVAL_EVIDENCE_CARD.json", {
            "contract": "topic5_epi_prssm_v0_2_arrival_card",
            "status": "REFUSED_INCONSISTENT_SEED_SETS",
            "why": "the arms were not fitted over the same seeds, so a rung-to-rung "
                   "difference would partly be a difference in which seeds were "
                   "available; returning empty rather than a contaminated contrast",
            "run_provenance": provenance})
        print(json.dumps(provenance, indent=1))
        return
    if not runs:
        atomic_write_json(OUT / "ARRIVAL_EVIDENCE_CARD.json", {"status": "NO_COMPLETED_RUN"})
        print("no completed arrival run")
        return

    rows = []
    for record in runs:
        for subject, values in record["per_patient"].items():
            gof = values.pop("gof", None) or {}
            rows.append({"arm": record["arm"], "seed": record["seed"], "subject": subject,
                         "gof_mean": gof.get("mean", values.get("rescaled_mean")),
                         "gof_sd": gof.get("sd", values.get("rescaled_sd")),
                         "gof_ks": gof.get("ks_statistic"),
                         "gof_acf_lag1": gof.get("acf_lag1"),
                         "gof_acf_lag2": gof.get("acf_lag2"),
                         "gof_acf_lag5": gof.get("acf_lag5"),
                         "gof_qq": gof.get("qq_max_abs_deviation"),
                         **values})
    frame = pd.DataFrame(rows)
    atomic_write_csv(OUT / "arrival_per_patient.csv", frame)

    per = (frame.groupby(["arm", "subject"]).arrival_nll_per_event.median()
           .reset_index())
    by_arm = {arm: dict(zip(g.subject, g.arrival_nll_per_event))
              for arm, g in per.groupby("arm")}

    contrasts, family = {}, {}
    # Each step is one question, and the ladder is ordered so that a step only means
    # what it says if the step below it held.  The v0.1 arms bundled two hypotheses
    # into one contrast and could identify neither; these do not.
    for better, worse, question in (
        ("t0_exogenous_clock", "renewal_only",
         "does anything slow move the rate at all, beyond a fixed renewal shape "
         "and a per-patient offset?"),
        ("t1_observer", "t0_exogenous_clock",
         "does knowing the past discharges help -- are they observations of the state?"),
        ("t2_physical", "t1_observer",
         "do the discharges have to push the state, over and above informing us about "
         "it?  The observer arm updates on the unpredicted part only, so a perfectly "
         "predicted discharge moves it not at all; the physical arm still pushes."),
        ("t2_physical", "renewal_only",
         "combined: a discharge-driven state versus no state at all"),
    ):
        if better not in by_arm or worse not in by_arm:
            continue
        shared = sorted(set(by_arm[better]) & set(by_arm[worse]))
        if len(shared) < 5:
            continue
        effect = paired_effect({s: by_arm[better][s] for s in shared},
                               {s: by_arm[worse][s] for s in shared},
                               label=f"{better}-vs-{worse}")
        contrasts[f"{better}-vs-{worse}"] = {"question": question, **effect.as_dict()}
        family[f"{better}-vs-{worse}"] = effect.sign_test_p

    order = ["t0_exogenous_clock-vs-renewal_only", "t1_observer-vs-t0_exogenous_clock",
             "t2_physical-vs-t1_observer"]
    intact = True
    for name in order:
        entry = contrasts.get(name)
        if entry is None:
            intact = False
            continue
        entry["chain_intact"] = intact
        if not (entry["median_delta"] < 0 and entry.get("ci_high", 1) < 0):
            intact = False
        entry["creditable"] = bool(entry.get("chain_intact") and
                                   entry["median_delta"] < 0 and entry.get("ci_high", 1) < 0)

    # Recompute the goodness-of-fit verdict here from the stored numbers rather than
    # trusting the `status` a run wrote: an earlier version of that function returned
    # "OK" for any sample of 50 or more regardless of the diagnostics, and runs written
    # by either version sit side by side.
    def verdict(row) -> tuple[str, list[str]]:
        checks = {
            "mean_within_20pct": 0.8 <= row.get("gof_mean", float("nan")) <= 1.2,
            "sd_within_20pct": 0.8 <= row.get("gof_sd", float("nan")) <= 1.2,
            "no_residual_serial_structure": max(
                abs(row.get(f"gof_acf_lag{k}", 0.0)) for k in (1, 2, 5)) < 0.05,
            "qq_within_tolerance": row.get("gof_qq", float("nan")) < 0.5,
        }
        failed = [k for k, ok in checks.items() if not ok]
        return ("OK" if not failed else "MISSPECIFIED"), failed

    fit = {}
    for arm, group in frame.groupby("arm"):
        clean = group[np.isfinite(group.rescaled_mean.astype(float))]
        fit[arm] = {
            "rescaled_mean_median": float(clean.rescaled_mean.median()) if len(clean) else None,
            "rescaled_sd_median": float(clean.rescaled_sd.median()) if len(clean) else None,
            # count patients, not patient-by-seed rows: the numerator used to run over
            # every run and the denominator over subjects, so it read "79/34"
            "n_patients_within_20pct_of_unit_mean": int(
                clean.groupby("subject").rescaled_mean.median().between(0.8, 1.2).sum())
                if len(clean) else 0,
            "n_patients_with_sd_within_20pct_of_one": int(
                clean.groupby("subject").rescaled_sd.median().between(0.8, 1.2).sum())
                if len(clean) else 0,
            "n_patients": int(clean.subject.nunique()),
        }

    for arm, block in fit.items():
        mean_ok = (block["rescaled_mean_median"] is not None
                   and 0.8 <= block["rescaled_mean_median"] <= 1.2)
        sd_ok = (block["rescaled_sd_median"] is not None
                 and 0.8 <= block["rescaled_sd_median"] <= 1.2)
        # patient-level, not a cohort median: a cohort median can sit inside the band
        # while most individual patients sit outside it
        verdicts = [verdict(r)[0] for _, r in frame[frame.arm == arm].iterrows()]
        n_ok = sum(1 for v in verdicts if v == "OK")
        block["n_patient_runs_passing_all_checks"] = n_ok
        block["n_patient_runs"] = len(verdicts)
        block["goodness_of_fit_gate"] = (
            "PASS" if (mean_ok and sd_ok and n_ok >= 0.5 * max(len(verdicts), 1))
            else "FAIL")
        block["gate_note"] = (
            "the frozen rule requires the rescaled residuals to have BOTH mean and sd "
            "near 1; when they do not, an arm-versus-arm contrast compares two "
            "misspecified models and must not be read as an answer")
    any_fail = any(b["goodness_of_fit_gate"] == "FAIL" for b in fit.values())

    taus = {r["arm"]: r.get("time_constants_seconds") for r in runs}
    card = {
        "contract": "topic5_epi_prssm_v0_2_arrival_card",
        "question": "v0.1 never asked the likelihood to explain when discharges arrive, "
                    "so the state was free to track event index and did. This channel "
                    "supplies the missing term.",
        "status": "EXPLORATORY_DEVELOPMENT",
        "primary_endpoint": "arrival_nll_per_event on held-out intervals",
        "contrasts": contrasts,
        "holm_corrected": holm(family),
        "run_provenance": provenance,
        "time_rescaling_fit": fit,
        "contrasts_admissible": not any_fail,
        "admissibility_note":
            "contrasts are reported for diagnosis regardless, but when "
            "contrasts_admissible is false none of them may enter a scientific claim",
        "fitted_time_constants_seconds": taus,
        "ladder_rule": "a rung is creditable only when every rung below it held; an arm "
                       "that beats its own reference while a lower rung failed is "
                       "reported but not credited",
        "reading": "a negative median means the richer arm predicted the arrival times "
                   "better. rescaled_mean and rescaled_sd both near 1 mean the intensity "
                   "is correctly specified; far from 1 means the fit is wrong and the "
                   "contrasts are comparisons between two wrong models.",
        "denominators": {"n_runs": len(runs), "arms": sorted({r["arm"] for r in runs}),
                         "seeds": sorted({r["seed"] for r in runs}),
                         "n_patients": int(per.subject.nunique())},
        "claim_boundary": [
            "an arrival-time gain is not evidence that discharges cause seizures",
            "the rate state and the spatial state are separate here; a gain on one is "
            "not a gain on the other",
            "development-partition result; no untouched-test claim is made here",
        ],
        "code_revision": code_revision(), "package_hash": package_hash(),
    }
    atomic_write_json(OUT / "ARRIVAL_EVIDENCE_CARD.json", card)
    print(json.dumps({k: {kk: v[kk] for kk in ("median_delta", "n_favourable",
                                               "n_patients", "sign_test_p")}
                      for k, v in contrasts.items()}, indent=1))


if __name__ == "__main__":
    main()
