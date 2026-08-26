#!/usr/bin/env python3
"""Aggregate Goal 4 into the H3a and H3b evidence cards.

H3a is reported on its own.  H3b is only written when H3a and the frozen H2b
endpoints both exist and point the same way; it is never allowed to gate H1 or H2.
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

OUT = OUTPUT_ROOT / "exposure_mechanism"
SEIZURE = OUTPUT_ROOT / "seizure_link"
#: H3a's primary outcomes must include at least one that is not a synonym for load
NON_LOAD_ENDPOINTS = ["order_nll"]
ENDPOINTS = ["event_nll", "order_nll", "selection_nll", "stop_nll", "participation_nll"]


def load_runs(cohort: str) -> list[dict]:
    out = []
    for path in sorted((OUT / "runs").glob("*.json")):
        record = json.loads(path.read_text())
        if record.get("cohort") == cohort and record.get("evaluation"):
            out.append(record)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", default="all34")
    args = parser.parse_args()
    runs = load_runs(args.cohort)
    if not runs:
        raise SystemExit("no completed exposure runs")
    dataset = {}
    for run in runs:
        dataset.update(run.get("dataset", {}))

    filtered = {}
    for endpoint in ENDPOINTS:
        per_arm: dict[str, list[dict[str, float]]] = {}
        for run in runs:
            per_arm.setdefault(run["arm"], []).append(
                {s: v[endpoint] for s, v in run["evaluation"]["filtered"].items()})
        filtered[endpoint] = {arm: aggregate_seeds(v) for arm, v in per_arm.items()}

    ladder_rows = []
    for run in runs:
        diagnostics = run.get("resource_diagnostics", {})
        ladder_rows.append({
            "arm": run["arm"], "resource_arm": run["resource_arm"], "seed": run["seed"],
            "status": run["train_report"]["status"],
            "best_validation": run["train_report"]["best_validation"],
            "tau_r_seconds": diagnostics.get("tau_r_seconds"),
            "tau_x_seconds": diagnostics.get("tau_x_seconds"),
            "exposure_kind": diagnostics.get("exposure_kind"),
            "gamma_q": diagnostics.get("gamma_q"), "gamma_L": diagnostics.get("gamma_L"),
            "gamma_x": diagnostics.get("gamma_x"),
            "resource_boundary_occupancy": diagnostics.get("resource_boundary_occupancy"),
            "resource_collapsed": diagnostics.get("resource_collapsed"),
            "resource_static": diagnostics.get("resource_static"),
            "wall_seconds": run["train_report"]["wall_seconds"], "job_id": run["job_id"]})
    atomic_write_csv(OUT / "resource_ladder.csv", pd.DataFrame(ladder_rows))

    base_arm = _base_arm(filtered["event_nll"])
    contrasts, family, rows = {}, {}, []
    for endpoint in ENDPOINTS:
        by_arm = filtered[endpoint]
        for arm in sorted(by_arm):
            if arm == base_arm:
                continue
            effect = paired_effect(by_arm[arm], by_arm[base_arm],
                                   label=f"{endpoint}::{arm}-vs-{base_arm}")
            contrasts[f"{endpoint}::{arm}"] = effect
            row = {"endpoint": endpoint, "arm": arm, "reference": base_arm,
                   "n_patients": effect.n_patients, "median_delta": effect.median_delta,
                   "ci_low": effect.ci_low, "ci_high": effect.ci_high,
                   "n_favourable": effect.n_favourable, "sign_test_p": effect.sign_test_p,
                   "wilcoxon_p": effect.wilcoxon_p}
            row.update({f"stratum_{k}": json.dumps(v) for k, v in stratify(effect, dataset).items()})
            rows.append(row)
            if endpoint in NON_LOAD_ENDPOINTS and arm.startswith("t2_"):
                family[f"{endpoint}::{arm}"] = effect.sign_test_p
    atomic_write_csv(OUT / "t1_t2_patient_effects.csv", pd.DataFrame(rows))

    curve_rows = []
    for endpoint in ENDPOINTS:
        for arm, values in filtered[endpoint].items():
            if not arm.startswith("t2_r3_"):
                continue
            kind = "event_count" if "events" in arm else "clock"
            scale = float(arm.split("clock")[-1] if kind == "clock" else arm.split("events")[-1])
            effect = contrasts.get(f"{endpoint}::{arm}")
            curve_rows.append({
                "endpoint": endpoint, "arm": arm, "kernel": kind, "scale": scale,
                "median_delta_vs_base": effect.median_delta if effect else np.nan,
                "ci_low": effect.ci_low if effect else np.nan,
                "ci_high": effect.ci_high if effect else np.nan,
                "n_favourable": effect.n_favourable if effect else np.nan,
                "n_patients": effect.n_patients if effect else np.nan,
                "sign_test_p": effect.sign_test_p if effect else np.nan})
    atomic_write_csv(OUT / "exposure_timescale_curve.csv", pd.DataFrame(curve_rows))

    innovation = None
    innovation_path = OUT / "innovation_controls_summary.json"
    if innovation_path.exists():
        innovation = json.loads(innovation_path.read_text())

    tau_freeze = None
    freeze_path = OUTPUT_ROOT / "manifests/RESOURCE_TAU_FREEZE.json"
    if freeze_path.exists():
        tau_freeze = json.loads(freeze_path.read_text())

    h3a = {
        "contract": "topic5_epi_prssm_v0_1_h3a_evidence_card",
        "hypothesis": "H3a: does IED exposure update the interictal functional state?",
        "status": "EXPLORATORY_DEVELOPMENT",
        "reference_arm": base_arm,
        "tau_r_freeze": tau_freeze,
        "predictive_leg": {k: v.as_dict() for k, v in contrasts.items()
                           if k.split("::")[0] in NON_LOAD_ENDPOINTS},
        "predictive_leg_all_endpoints": {k: {"median_delta": v.median_delta,
                                             "ci": [v.ci_low, v.ci_high],
                                             "n_favourable": v.n_favourable,
                                             "n_patients": v.n_patients,
                                             "sign_test_p": v.sign_test_p}
                                         for k, v in contrasts.items()},
        "innovation_leg": innovation["by_tau"] if innovation else
                          {"status": "NOT_RUN", "reason": "innovation controls have not been run"},
        "holm_corrected_primary_family": holm(family),
        "resource_health": _resource_health(pd.DataFrame(ladder_rows)),
        "denominators": {"n_runs": len(runs), "arms": sorted({r["arm"] for r in runs}),
                         "n_patients": len(dataset),
                         "n_epilepsiae": sum(1 for v in dataset.values() if v == "epilepsiae"),
                         "n_yuquan": sum(1 for v in dataset.values() if v == "yuquan")},
        "claim_boundary": [
            "H3a's primary outcome is the masked recruitment-order likelihood, which is "
            "invariant to how many contacts participated, so a gain is not a restatement of load",
            "raw-load gain without the innovation leg may only be called a history-dependent "
            "predictor",
            "participation and extent are secondary outcomes only",
        ],
        "code_revision": code_revision(), "package_hash": package_hash(),
    }
    atomic_write_json(OUT / "H3A_EVIDENCE_CARD.json", h3a)
    atomic_write_json(OUT / "H3B_EVIDENCE_CARD.json", _h3b(h3a))
    print(json.dumps({"base_arm": base_arm,
                      "n_arms": len(filtered["event_nll"]),
                      "resource_health": h3a["resource_health"]}, indent=2)[:900])


def _base_arm(by_arm: dict) -> str:
    for candidate in ("t1_r1_free_tau", "t1_r1_tau1800", "t1_r0"):
        if candidate in by_arm:
            return candidate
    return sorted(by_arm)[0]


def _resource_health(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {}
    # The resource falls from 1, so it has two degenerate ends and they mean opposite
    # things.  Older runs recorded only the floor, so the ceiling case is recovered
    # here from the stored quantiles rather than left silently uncounted.
    never = frame.get("resource_never_consumed")
    if never is None and "resource_q01" in frame:
        never = frame["resource_q01"] > 0.99
    return {
        "n_collapsed_runs": int(frame["resource_collapsed"].fillna(False).sum()),
        "n_static_runs": int(frame["resource_static"].fillna(False).sum()),
        "n_never_consumed_runs": int(never.fillna(False).sum()) if never is not None else None,
        "median_floor_occupancy": float(frame["resource_boundary_occupancy"].median())
            if frame["resource_boundary_occupancy"].notna().any() else None,
        "note": "a resource that collapsed to its floor, sat at its ceiling, or never moved "
                "carries no resource information; such an arm's comparison must be read as "
                "'this pathway was not used', not as evidence about a resource. The floor and "
                "the ceiling are opposite failures and are counted separately.",
    }


def _h3b(h3a: dict) -> dict:
    cards = sorted(SEIZURE.glob("runs/*.json"))
    if not cards:
        return {
            "contract": "topic5_epi_prssm_v0_1_h3b_evidence_card",
            "hypothesis": "H3b: is exposure-related updating consistent with participation in "
                          "the interictal-to-ictal transition?",
            "status": "NOT_EVALUABLE",
            "reason": "H3b may only be read after H2b has been frozen and run; no seizure-link "
                      "result exists yet",
            "note": "H3b is never a gate on H1, H2a, H2b or H3a",
        }
    seizure = [json.loads(p.read_text()) for p in cards]
    return {
        "contract": "topic5_epi_prssm_v0_1_h3b_evidence_card",
        "hypothesis": "H3b: is exposure-related updating consistent with participation in "
                      "the interictal-to-ictal transition?",
        "status": "READ_ONLY_COMBINATION",
        "requires": "H3a supported AND H2b supported AND the two point the same way",
        "h3a_evidence": _evidence_vector(h3a),
        "h2b_layers_available": [s.get("layer") for s in seizure],
        "h2b_primary": [{"layer": s.get("layer"),
                         "primary_window": s.get("primary_validation_window", {})}
                        for s in seizure],
        "verdict": "H3b is only asserted when both legs are supported and agree in direction; "
                   "otherwise it is reported as not asserted, which is not a negative for "
                   "H1, H2a, H2b or H3a",
        "code_revision": code_revision(), "package_hash": package_hash(),
    }


def _evidence_vector(h3a: dict) -> dict:
    """H3a as a vector of independently checkable conditions, never one boolean.

    A single `any arm has CI < 0` flag ignored resource health, the innovation
    layer, directionality and multiplicity, so a boundary-collapsed arm with an
    incidental gain could have triggered the downstream H3b combination.
    """
    predictive = h3a.get("predictive_leg", {}) or {}
    wins = {k: e for k, e in predictive.items()
            if e.get("median_delta") is not None and e["median_delta"] < 0
            and e.get("ci_high", 1) < 0}
    holm = h3a.get("holm_corrected_primary_family", {}) or {}
    holm_survivors = {k: v for k, v in holm.items() if isinstance(v, (int, float)) and v < 0.05}
    health = h3a.get("resource_health", {}) or {}
    innovation = h3a.get("innovation_leg") or {}
    innovation_ran = isinstance(innovation, dict) and "status" not in innovation

    directional = None
    if innovation_ran:
        directional = all(
            (block.get("real_minus_time_reversal") or {}).get("median_delta", -1) > 0
            for block in innovation.values() if isinstance(block, dict))

    n_runs = max(int(health.get("n_collapsed_runs", 0)) + int(health.get("n_static_runs", 0)), 0)
    return {
        "predictive_increment": {"n_arms_with_ci_below_zero": len(wins),
                                 "arms": sorted(wins)},
        "multiplicity": {"n_surviving_holm": len(holm_survivors),
                         "arms": sorted(holm_survivors)},
        "resource_health": {"n_collapsed_runs": health.get("n_collapsed_runs"),
                            "n_static_runs": health.get("n_static_runs"),
                            "n_never_consumed_runs": health.get("n_never_consumed_runs"),
                            "note": "a collapsed or static resource carries no resource "
                                    "information, so a gain from such an arm is not "
                                    "evidence about depletion"},
        "innovation_layer_ran": innovation_ran,
        "directionality_passed": directional,
        "asserted": bool(wins) and bool(holm_survivors) and innovation_ran
                    and directional is True and not health.get("n_collapsed_runs"),
        "why_not_asserted": None if (bool(wins) and bool(holm_survivors) and innovation_ran
                                     and directional is True
                                     and not health.get("n_collapsed_runs"))
                            else "H3a is asserted only when a predictive increment survives "
                                 "multiplicity, the innovation layer has run, the forward "
                                 "direction beats time reversal, and no contributing arm "
                                 "had a collapsed resource",
    }


if __name__ == "__main__":
    main()
