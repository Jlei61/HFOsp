#!/usr/bin/env python3
"""Aggregate Goal 1 into patient-first tables and the H1 evidence card.

Every number here is recomputed from the per-run JSON artefacts, never copied
from a log.  Seeds are aggregated inside a patient first; the patient is the unit
of cohort inference.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.topic5_epi_prssm.contracts import (  # noqa: E402
    FROZEN, OUTPUT_ROOT, atomic_write_csv, atomic_write_json, code_revision, package_hash,
)
from src.topic5_epi_prssm.stats import aggregate_seeds, holm, paired_effect, stratify  # noqa: E402

OUT = OUTPUT_ROOT / "generator_ladder"

#: the ladder steps whose increments are the H1 question, in order
LADDER = ["static", "frozen_state", "ct_ewma_g0", "g1_graph_clds", "g2_graph_gru_ode",
          "g3_resource"]
#: The first rung is measured against the capacity-matched frozen-state arm, not
#: against the bare fixed repertoire: the no-state synthetic showed that most of
#: a "state" gain over `static` is the adapter's own per-node parameters.
LADDER_STEPS = [("ct_ewma_g0", "frozen_state_node"), ("g1_graph_clds", "ct_ewma_g0"),
                ("g2_graph_gru_ode", "g1_graph_clds"), ("g3_resource", "g2_graph_gru_ode")]
EXTRA_CONTRASTS = [("g1_graph_clds_order_weighted", "nuisance_timing_baseline_order_weighted"),
                   ("g1_graph_clds_order_weighted", "g1_graph_clds"),
                   ("g3_resource_on_g1", "g1_graph_clds"),
                   ("g3_resource", "g1_graph_clds"),
                   ("g3_resource_on_g1", "nuisance_timing_baseline"),
                   ("ct_ewma_g0", "nuisance_timing_baseline"),
                   ("g1_graph_clds", "nuisance_timing_baseline"),
                   ("g2_graph_gru_ode", "nuisance_timing_baseline"),
                   ("g3_resource", "nuisance_timing_baseline"),
                   ("nuisance_timing_baseline", "frozen_state_node"),
                   ("g2_graph_gru_ode_long_window", "g2_graph_gru_ode"),
                   ("ct_ewma_g0_long_window", "ct_ewma_g0"),
                   ("frozen_state", "static"),
                   ("frozen_state_node", "frozen_state"),
                   ("ct_ewma_g0", "frozen_state"),
                   ("ct_ewma_g0", "static"),
                   ("unconstrained_gru", "ct_ewma_g0"),
                   ("event_index_ewma", "ct_ewma_g0"),
                   ("g3_flexible_resource_control", "g3_resource"),
                   ("g2_compressed_state", "g2_graph_gru_ode")]
PRIMARY_ENDPOINT = "event_nll"
ENDPOINTS = ["event_nll", "order_nll", "selection_nll", "stop_nll", "participation_nll"]


def load_runs(cohort: str) -> list[dict]:
    runs = []
    for path in sorted((OUT / "runs").glob("*.json")):
        record = json.loads(path.read_text())
        if record.get("cohort") == cohort:
            runs.append(record)
    return runs


def filtered_by_arm(runs, endpoint: str) -> dict[str, dict[str, float]]:
    per_arm: dict[str, list[dict[str, float]]] = {}
    for run in runs:
        if run.get("evaluation") is None:
            continue
        values = {s: v[endpoint] for s, v in run["evaluation"]["filtered"].items()}
        per_arm.setdefault(run["arm"], []).append(values)
    return {arm: aggregate_seeds(v) for arm, v in per_arm.items()}


def open_loop_by_arm(runs, key: str = "open_loop_event_nll") -> dict[str, dict[int, dict[str, float]]]:
    out: dict[str, dict[int, list[dict[str, float]]]] = {}
    for run in runs:
        if run.get("evaluation") is None or not run["evaluation"].get(key):
            continue
        for subject, horizons in run["evaluation"][key].items():
            for horizon, value in horizons.items():
                out.setdefault(run["arm"], {}).setdefault(int(horizon), []).append({subject: value})
    return {arm: {h: aggregate_seeds(v) for h, v in per_h.items()} for arm, per_h in out.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", default="all34")
    args = parser.parse_args()
    runs = load_runs(args.cohort)
    if not runs:
        raise SystemExit(f"no generator-ladder runs for cohort {args.cohort}")

    dataset = {}
    for run in runs:
        dataset.update(run.get("dataset", {}))

    # ---- model_runs.csv -------------------------------------------------
    rows = []
    for run in runs:
        report = run["train_report"]
        row = {"arm": run["arm"], "family": run["family"], "seed": run["seed"],
               "cohort": run["cohort"], "status": report["status"],
               "failure_reason": report.get("failure_reason"),
               "epochs_run": report["epochs_run"], "best_epoch": report["best_epoch"],
               "best_validation": report["best_validation"],
               "wall_seconds": report["wall_seconds"],
               "correction_energy": report["peak_correction_energy"],
               "resource_floor_fraction": report["resource_floor_fraction"],
               "stability_margin": report["stability_margin"],
               "n_parameters": report["diagnostics"].get("n_parameters"),
               "uses_graph_messages": report["diagnostics"].get("uses_graph_messages"),
               "job_id": run["job_id"], "package_hash": run["package_hash"]}
        row.update({f"diag_{k}": v for k, v in (run.get("state_diagnostics") or {}).items()
                    if not isinstance(v, (list, dict))})
        rows.append(row)
    atomic_write_csv(OUT / "model_runs.csv", pd.DataFrame(rows))

    # ---- patient effects along the ladder -------------------------------
    filtered = {e: filtered_by_arm(runs, e) for e in ENDPOINTS}
    effects, effect_rows = {}, []
    families = {}
    for endpoint in ENDPOINTS:
        by_arm = filtered[endpoint]
        for better, worse in LADDER_STEPS + EXTRA_CONTRASTS:
            if better not in by_arm or worse not in by_arm:
                continue
            label = f"{endpoint}::{better}-vs-{worse}"
            effect = paired_effect(by_arm[better], by_arm[worse], label=label)
            effects[label] = effect
            row = {"endpoint": endpoint, "contrast": f"{better} - {worse}",
                   "n_patients": effect.n_patients, "median_delta": effect.median_delta,
                   "ci_low": effect.ci_low, "ci_high": effect.ci_high,
                   "n_favourable": effect.n_favourable, "sign_test_p": effect.sign_test_p,
                   "wilcoxon_p": effect.wilcoxon_p}
            row.update({f"stratum_{k}": json.dumps(v) for k, v in
                        stratify(effect, dataset).items()})
            effect_rows.append(row)
            if endpoint == PRIMARY_ENDPOINT and (better, worse) in LADDER_STEPS:
                families[label] = effect.sign_test_p
    atomic_write_csv(OUT / "patient_effects.csv", pd.DataFrame(effect_rows))
    holm_corrected = holm(families)

    # ---- per-patient filtered table -------------------------------------
    per_patient_rows = []
    for endpoint, by_arm in filtered.items():
        for arm, values in by_arm.items():
            for subject, value in values.items():
                per_patient_rows.append({"endpoint": endpoint, "arm": arm,
                                         "subject": subject, "dataset": dataset.get(subject),
                                         "value": value})
    atomic_write_csv(OUT / "patient_filtered_scores.csv", pd.DataFrame(per_patient_rows))

    # ---- open loop -------------------------------------------------------
    open_rows = []
    for key, tag in (("open_loop_event_nll", "event_nll"), ("open_loop_order_nll", "order_nll")):
        by_arm = open_loop_by_arm(runs, key)
        reference = by_arm.get("static", {})
        for arm, horizons in by_arm.items():
            for horizon, values in sorted(horizons.items()):
                base = reference.get(horizon, {})
                effect = paired_effect(values, base, label=f"{tag}::{arm}-vs-static::H{horizon}") \
                    if base else None
                for subject, value in values.items():
                    open_rows.append({"endpoint": tag, "arm": arm, "horizon": horizon,
                                      "subject": subject, "dataset": dataset.get(subject),
                                      "value": value,
                                      "delta_vs_static": (value - base[subject]) if subject in base else np.nan})
                if effect is not None:
                    open_rows.append({"endpoint": tag, "arm": arm, "horizon": horizon,
                                      "subject": "__cohort__", "dataset": "__cohort__",
                                      "value": float(np.median(list(values.values()))),
                                      "delta_vs_static": effect.median_delta,
                                      "ci_low": effect.ci_low, "ci_high": effect.ci_high,
                                      "n_favourable": effect.n_favourable,
                                      "n_patients": effect.n_patients,
                                      "sign_test_p": effect.sign_test_p})
    atomic_write_csv(OUT / "open_loop_horizon.csv", pd.DataFrame(open_rows))

    # ---- state reset and delta-t shuffle ---------------------------------
    reset_rows, shuffle_rows = [], []
    for run in runs:
        evaluation = run.get("evaluation") or {}
        for subject, horizons in (evaluation.get("state_reset") or {}).items():
            for horizon, value in horizons.items():
                reset_rows.append({"arm": run["arm"], "seed": run["seed"], "subject": subject,
                                   "dataset": dataset.get(subject),
                                   "horizon": int(horizon), "reset_penalty_nll": value})
        for subject, values in (evaluation.get("delta_t_shuffle") or {}).items():
            base = evaluation["filtered"].get(subject, {})
            for endpoint, value in values.items():
                shuffle_rows.append({"arm": run["arm"], "seed": run["seed"], "subject": subject,
                                     "dataset": dataset.get(subject), "endpoint": endpoint,
                                     "shuffled": value, "intact": base.get(endpoint),
                                     "shuffle_penalty": value - base.get(endpoint, np.nan)})
    atomic_write_csv(OUT / "state_reset.csv", pd.DataFrame(reset_rows))
    atomic_write_csv(OUT / "delta_t_shuffle.csv", pd.DataFrame(shuffle_rows))

    # ---- evidence card ----------------------------------------------------
    card = _evidence_card(runs, filtered, effects, holm_corrected, open_loop_by_arm(runs),
                          pd.DataFrame(reset_rows), pd.DataFrame(shuffle_rows), dataset)
    atomic_write_json(OUT / "GENERATOR_EVIDENCE_CARD.json", card)
    print(json.dumps({k: card[k] for k in ("verdict", "supported_layer", "denominators")}, indent=2))


def _evidence_card(runs, filtered, effects, holm_corrected, open_loop, reset, shuffle,
                   dataset) -> dict:
    primary = filtered[PRIMARY_ENDPOINT]
    # The ladder is nested: a rung may only be credited if every rung below it held.
    # Crediting "G3 beats G2" after "G2 lost to G1" would report a layer the data
    # never supported.
    ladder_supported = "none"
    branch_notes = []
    chain_intact = True
    for better, worse in LADDER_STEPS:
        label = f"{PRIMARY_ENDPOINT}::{better}-vs-{worse}"
        effect = effects.get(label)
        if effect is None:
            branch_notes.append(f"{better} vs {worse}: arm missing")
            chain_intact = False
            continue
        beats = (effect.median_delta < 0 and effect.ci_high < 0)
        note = (f"{better} vs {worse}: median {effect.median_delta:+.4f} nats/event, "
                f"95% CI [{effect.ci_low:+.4f}, {effect.ci_high:+.4f}], "
                f"{effect.n_favourable}/{effect.n_patients} patients favourable, "
                f"sign-test p={effect.sign_test_p:.3g}")
        if beats and chain_intact:
            ladder_supported = better
            note += " -> supported"
        elif beats and not chain_intact:
            note += (" -> beats its own reference, but a rung below it did not hold, "
                     "so this layer is not credited")
        else:
            note += " -> not supported"
            chain_intact = False
        branch_notes.append(note)
    allowed = {
        "none": "no layer beats the capacity-matched frozen state",
        "ct_ewma_g0": "leaky history state / observer tracking",
        "g1_graph_clds": "structured graph recurrent slow state",
        "g2_graph_gru_ode": "nonlinear graph recurrent dynamics add an increment",
        "g3_resource": "bounded resource anchor adds predictive value",
        "g3_resource_on_g1": "bounded resource anchor on the best stable recurrent family "
                             "adds predictive value",
    }
    open_loop_summary = {}
    references = {"frozen_state_node": open_loop.get("frozen_state_node", {}),
                  "nuisance_timing_baseline": open_loop.get("nuisance_timing_baseline", {}),
                  "static": open_loop.get("static", {})}
    for reference_name, reference in references.items():
        for arm, horizons in open_loop.items():
            if arm == reference_name:
                continue
            for horizon, values in sorted(horizons.items()):
                base = reference.get(horizon, {})
                if not base:
                    continue
                effect = paired_effect(values, base,
                                       label=f"{arm}-vs-{reference_name}-H{horizon}")
                open_loop_summary[f"{arm}::vs_{reference_name}::H{horizon}"] = {
                    "median_delta": effect.median_delta,
                    "ci": [effect.ci_low, effect.ci_high],
                    "n_favourable": effect.n_favourable, "n_patients": effect.n_patients,
                    "sign_test_p": effect.sign_test_p}
    reset_summary = {}
    if len(reset):
        for horizon, group in reset.groupby("horizon"):
            per_patient = group.groupby(["arm", "subject"]).reset_penalty_nll.median()
            reset_summary[int(horizon)] = {
                "median_penalty": float(per_patient.median()),
                "n": int(len(per_patient))}
    shuffle_summary = {}
    if len(shuffle):
        for endpoint, group in shuffle[shuffle.endpoint == PRIMARY_ENDPOINT].groupby("arm"):
            per_patient = group.groupby("subject").shuffle_penalty.median()
            shuffle_summary[endpoint] = {"median_penalty": float(per_patient.median()),
                                         "n_patients": int(len(per_patient))}
    statuses = {}
    for run in runs:
        statuses[run["train_report"]["status"]] = statuses.get(run["train_report"]["status"], 0) + 1
    timing_gate = {}
    for arm in ("ct_ewma_g0", "g1_graph_clds", "g2_graph_gru_ode", "g3_resource",
                "g3_resource_on_g1", "g1_graph_clds_order_weighted"):
        entry = {}
        for endpoint in ("event_nll", "order_nll", "stop_nll", "participation_nll"):
            effect = effects.get(f"{endpoint}::{arm}-vs-nuisance_timing_baseline")
            if effect is None:
                continue
            entry[endpoint] = {"median_delta": effect.median_delta,
                               "ci": [effect.ci_low, effect.ci_high],
                               "n_favourable": effect.n_favourable,
                               "n_patients": effect.n_patients,
                               "sign_test_p": effect.sign_test_p,
                               "beats_observable_timing": bool(
                                   effect.median_delta < 0 and effect.ci_high < 0)}
        if entry:
            timing_gate[arm] = entry
    return {
        "contract": "topic5_epi_prssm_v0_1_generator_evidence_card",
        "observable_timing_gate": timing_gate,
        "endpoint_split_warning": (
            "the ladder was trained on the joint event likelihood (next contact among all "
            "remaining, plus STOP, plus participation).  The masked recruitment-order "
            "likelihood is therefore an unoptimised read-out of the same scores, and an arm "
            "that gains on selection can lose on pure ordering.  Whether the state explains "
            "the ordering when the ordering is a training target is answered by the "
            "order-weighted arms, not by the main ladder."),
        "observable_timing_gate_meaning": (
            "Topic 2 establishes that the event rate itself drifts slowly and that its "
            "autocorrelation is still positive at eight hours; a latent-state claim must beat a "
            "readout conditioned on causal multi-scale rate, interval, coverage and time of day, "
            "not merely beat a static repertoire"),
        "hypothesis": "H1: is there a slow state that still predicts future IED repertoire "
                      "after the observer correction is switched off?",
        "status": "EXPLORATORY_DEVELOPMENT",
        "primary_endpoint": PRIMARY_ENDPOINT,
        "supported_layer": ladder_supported,
        "verdict": allowed[ladder_supported],
        "ladder_notes": branch_notes,
        "holm_corrected_primary_family": holm_corrected,
        "open_loop_contrasts": open_loop_summary,
        "open_loop_reference_note": (
            "reported against three references: the capacity-matched frozen state, the "
            "observable-timing baseline, and the bare fixed repertoire"),
        "state_reset_penalty_by_horizon": reset_summary,
        "delta_t_shuffle_penalty": shuffle_summary,
        "denominators": {
            "n_runs": len(runs), "run_status_counts": statuses,
            "n_patients": len(primary.get("static", {})) or len(next(iter(primary.values()), {})),
            "n_epilepsiae": sum(1 for v in dataset.values() if v == "epilepsiae"),
            "n_yuquan": sum(1 for v in dataset.values() if v == "yuquan"),
            "arms": sorted(primary),
        },
        "claim_boundary": [
            "development-partition result; no untouched-test claim is made here",
            "an improvement in held-out prediction is not evidence of a causal mechanism",
            "G0 is a leaky baseline and is never described as a graph recurrent generator",
        ],
        "code_revision": code_revision(), "package_hash": package_hash(),
    }


if __name__ == "__main__":
    main()
