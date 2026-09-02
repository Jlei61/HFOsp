#!/usr/bin/env python3
"""Phase F5: the compact model-unseen confirmation.

This is the only code in the stage permitted to read ``split == -1``.  It runs
after every model, rank, data fraction, basis, regulariser and null family is
frozen, it retrains nothing, and it scores exactly the pre-locked combination:

    r = 4, 100% split-0 training events, strong unordered baseline
    autonomous shared operator
    patient aligned vs that patient's median angle-rotated null
    free low-rank vs the unordered baseline
    direct-horizon read-out as the predictive ceiling

Everything else — other ranks, the 25/50% curves, the weak baseline, the
locality-rewired family, the transition-form sensitivities, the time proxy and
the full transplant family — stays on the development test.  Every access is
written to an audit log so a wider access is visible after the fact.

It is a model-unseen confirmation, never a prospective validation.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic5_strict_history_data_v0_2 import load_sample_set  # noqa: E402
from src.topic5_strict_history_motif_v0_2 import (  # noqa: E402
    MotifConfig,
    OrderedMotif,
    checkpoint_objective,
    evaluate,
    primary_field_kind,
    tensors_from_samples,
)
from src.topic5_structural_identifiability_v0_2 import load_basis_bundle  # noqa: E402

RESULT_ROOT = ROOT / "results/topic5_capacity_constrained_history_motif_v0_2"
FRAME_ROOT = ROOT / "results/topic5_dynamical_motif_rnn_v0_1/frame_cache/GEOMETRY_ONLY_PCA2"
LOCKED = {"block": "CORE1", "rank": 4, "data_fraction": 100, "basis_fraction": 100,
          "baseline_level": "U_FULL_SET", "f_form": "FULL", "prefix_len": 3, "time_head": False}
ALLOWED_STRUCTURES = ("H1_PATIENT_ALIGNED", "H1_ANGLE_ROTATED_AXIS", "H1_FREE_LOW_RANK")
ALLOWED_FAMILIES = ("AUTONOMOUS_SHARED_OPERATOR", "DIRECT_HORIZON_UPPER_BOUND")


def matches_lock(unit: dict) -> bool:
    return (all(unit[key] == value for key, value in LOCKED.items())
            and unit["structure"] in ALLOWED_STRUCTURES
            and unit["family"] in ALLOWED_FAMILIES)


def score_patient(payload: dict) -> dict:
    torch.set_num_threads(int(os.environ.get("TOPIC5_TORCH_THREADS", "2")))
    patient, units = payload["patient"], payload["units"]
    samples = load_sample_set(RESULT_ROOT / "sample_cache" / "prefix3" / f"{patient}.npz")
    rows = np.flatnonzero(samples.split == -1)
    if rows.size == 0:
        return {"patient": patient, "skipped": "no model-unseen events"}
    batch = tensors_from_samples(samples, rows)
    contact_xy = torch.as_tensor(
        np.asarray(np.load(FRAME_ROOT / patient / "plane.npz")["contacts_xy_mm"], dtype=np.float32))
    bases, index = load_basis_bundle(RESULT_ROOT / "basis" / "per_patient" / f"{patient}.npz")
    baseline_payload = np.load(
        RESULT_ROOT / "baseline" / "U_FULL_SET" / "prefix3" / patient / "logits.npz",
        allow_pickle=False)
    baseline = {name: torch.as_tensor(baseline_payload[name][rows], dtype=torch.float32)
                for name in ("contact", "cardinality", "suffix")}

    record: dict = {"patient": patient, "n_model_unseen_events": int(rows.size), "arms": {}}
    for family in ALLOWED_FAMILIES:
        record["arms"][family] = {}
        for structure in ALLOWED_STRUCTURES:
            scores = []
            for unit in units:
                if unit["structure"] != structure or unit["family"] != family:
                    continue
                path = RESULT_ROOT / unit["output_dir"] / "checkpoint.pt"
                if not path.exists():
                    continue
                free = structure == "H1_FREE_LOW_RANK"
                model = OrderedMotif(MotifConfig(
                    structure=structure, family=family, rank=4, n_contacts=samples.n_contacts,
                    n_horizons=batch.n_horizons, max_cardinality=samples.max_cardinality,
                    free_basis=free), None if free else bases[unit["basis_key"]])
                state = torch.load(path, weights_only=True)["state_dict"]
                model.load_state_dict({key[2:]: value for key, value in state.items()
                                       if key.startswith("0.")})
                model.eval()
                result = evaluate(model, baseline, batch, contact_xy)
                scores.append({"null_id": unit["null_id"], "seed": int(unit["seed"]),
                               "objective": checkpoint_objective(result, family),
                               "suffix_balanced_bce": result.scalars[
                                   f"{primary_field_kind(family)}_balanced_bce"],
                               "total_nll_h1": result.per_horizon["total_nll"][0]})
            if scores:
                record["arms"][family][structure] = {
                    "n_units": len(scores),
                    "median_objective": float(np.median([s["objective"] for s in scores])),
                    "median_suffix_balanced_bce": float(
                        np.median([s["suffix_balanced_bce"] for s in scores])),
                    "units": scores,
                }
    baseline_result = evaluate(None, baseline, batch, contact_xy)
    record["unordered_baseline"] = {
        family: checkpoint_objective(baseline_result, family) for family in ALLOWED_FAMILIES}
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--confirm", action="store_true",
                        help="required: acknowledges that every choice is already frozen")
    arguments = parser.parse_args()
    if not arguments.confirm:
        raise SystemExit("refusing to touch the model-unseen split without --confirm")

    manifest = pd.read_csv(RESULT_ROOT / "MASTER_UNIT_MANIFEST.csv")
    eligible = manifest[manifest["eligible"]].to_dict("records")
    selected = [unit for unit in eligible if matches_lock(unit)]
    rejected = len(eligible) - len(selected)
    payloads: dict[str, list[dict]] = {}
    for unit in selected:
        payloads.setdefault(unit["patient"], []).append(unit)
    jobs = [{"patient": patient, "units": units} for patient, units in sorted(payloads.items())]
    with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
        records = list(pool.map(score_patient, jobs))

    eligibility = pd.read_csv(RESULT_ROOT / "basis" / "BASIS_ELIGIBILITY.csv").set_index("patient")
    rows = []
    for record in records:
        if "skipped" in record:
            continue
        auto = record["arms"].get("AUTONOMOUS_SHARED_OPERATOR", {})
        direct = record["arms"].get("DIRECT_HORIZON_UPPER_BOUND", {})
        angle_ok = bool(eligibility.loc[record["patient"], "angle_null_eligible"])
        aligned = auto.get("H1_PATIENT_ALIGNED", {}).get("median_objective")
        null = auto.get("H1_ANGLE_ROTATED_AXIS", {}).get("median_objective")
        free = auto.get("H1_FREE_LOW_RANK", {}).get("median_objective")
        base = record["unordered_baseline"]["AUTONOMOUS_SHARED_OPERATOR"]
        rows.append({
            "patient": record["patient"],
            "n_model_unseen_events": record["n_model_unseen_events"],
            "angle_null_eligible": angle_ok,
            "autonomous_aligned": aligned,
            "autonomous_angle_null_median": null,
            "structure_effect": (null - aligned) if (angle_ok and aligned is not None
                                                     and null is not None) else np.nan,
            "autonomous_free": free,
            "unordered_baseline": base,
            "free_minus_baseline": (base - free) if free is not None else np.nan,
            "direct_aligned_ceiling": direct.get("H1_PATIENT_ALIGNED", {}).get("median_objective"),
            "direct_free_ceiling": direct.get("H1_FREE_LOW_RANK", {}).get("median_objective"),
        })
    table = pd.DataFrame(rows)
    table.to_csv(RESULT_ROOT / "PER_PATIENT_MODEL_UNSEEN_CONFIRMATION.csv", index=False)

    def summarise(values: np.ndarray) -> dict:
        """Median with a patient bootstrap interval.

        A median plus a positive/negative split does not say whether the sign is
        determined by the cohort or by two or three patients, so the interval is
        reported next to it.  This is the confirmation tier, so the interval is
        descriptive: it is never used to declare a new result.
        """
        values = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
        if values.size == 0:
            return {"n": 0}
        summary = {"n": int(values.size), "median": float(np.median(values)),
                   "n_positive": int((values > 0).sum()),
                   "n_negative": int((values < 0).sum())}
        if values.size >= 3:
            rng = np.random.default_rng(20260819)
            draws = np.median(
                rng.choice(values, size=(10000, values.size), replace=True), axis=1)
            low, high = np.percentile(draws, [2.5, 97.5])
            summary["median_ci95"] = [float(low), float(high)]
            summary["crosses_zero"] = bool(low < 0.0 < high)
        return summary

    log = {
        "contract": "topic5_capacity_constrained_history_motif_v0_2_model_unseen_confirmation",
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "tier_name": "model-unseen confirmation (never a prospective validation)",
        "locked_combination": {**LOCKED, "structures": list(ALLOWED_STRUCTURES),
                               "families": list(ALLOWED_FAMILIES)},
        "units_accessed": len(selected),
        "units_refused_because_outside_the_lock": rejected,
        "patients_scored": int(len(table)),
        "angle_comparison_denominator": int(table["angle_null_eligible"].sum()) if len(table) else 0,
        "structure_effect_autonomous": summarise(table["structure_effect"].to_numpy())
        if len(table) else {"n": 0},
        "free_minus_unordered_baseline": summarise(table["free_minus_baseline"].to_numpy())
        if len(table) else {"n": 0},
        "note": "patients without a two-dimensional geometry keep the free-vs-baseline "
                "confirmation but leave the angle-null denominator",
        "per_patient": records,
    }
    (RESULT_ROOT / "SPLIT_MINUS_ONE_ACCESS_LOG.json").write_text(json.dumps(log, indent=2) + "\n")
    print(f"model-unseen confirmation: {len(table)} patients, "
          f"{len(selected)} units inside the lock, {rejected} refused")
    print(f"  structure effect : {log['structure_effect_autonomous']}")
    print(f"  free vs baseline : {log['free_minus_unordered_baseline']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
