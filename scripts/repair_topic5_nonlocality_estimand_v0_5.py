#!/usr/bin/env python3
"""Pre-training repair of a degenerate median-of-sparse-burdens J estimand."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_topic5_crossfit_nonlocality_v0_5 import plot_stage_d  # noqa: E402


OUT_ROOT = ROOT / "results/topic5_multiscale_effective_scaffold_v0_5"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if (OUT_ROOT / "STAGE_E_TRAINING_COMPLETE.json").exists():
        raise RuntimeError("J estimand cannot be repaired after formal RNN training")
    if not (OUT_ROOT / "TARGET_PHYSICAL_EMBARGO_ACTIVE.json").exists():
        raise RuntimeError("target physical embargo is not active")
    fit = pd.read_csv(OUT_ROOT / "CROSSFIT_NONLOCALITY_FIT_SUMMARY.csv")
    old = fit.J_lat_exceedance_burden.to_numpy(float)
    if not np.allclose(old, 0.0):
        raise RuntimeError("repair is only authorized for the observed all-zero registered estimand")
    rows = []
    for item in fit.itertuples(index=False):
        table = pd.read_csv(
            OUT_ROOT / "nonlocality_oof" / f"{item.fit_id}.csv.gz",
            usecols=["event", "distal", "exceedance"],
        )
        distal = table[table.distal.astype(bool)]
        event = distal.groupby("event", sort=True).exceedance.mean()
        blocks = [values for values in np.array_split(event.to_numpy(), min(10, len(event))) if len(values)]
        rows.append({
            "fit_id": item.fit_id,
            "J_preregistered_event_median": float(event.median()),
            "J_lat_exceedance_burden": float(event.mean()),
            "J_temporal_block_median": float(np.median([np.mean(values) for values in blocks])),
            "J_nonzero_event_fraction": float(np.mean(event > 0)),
            "n_J_events": int(len(event)),
        })
    repaired = pd.DataFrame(rows)
    fit = fit.drop(columns=[
        column for column in repaired.columns if column != "fit_id" and column in fit.columns
    ]).merge(repaired, on="fit_id", how="left", validate="one_to_one")
    fit.to_csv(OUT_ROOT / "CROSSFIT_NONLOCALITY_FIT_SUMMARY.csv", index=False)
    patient = fit.groupby("subject", as_index=False).agg(
        J_lat_exceedance_burden=("J_lat_exceedance_burden", "mean"),
        J_preregistered_event_median=("J_preregistered_event_median", "mean"),
        J_temporal_block_median=("J_temporal_block_median", "mean"),
        J_nonzero_event_fraction=("J_nonzero_event_fraction", "mean"),
        J_old_median_sensitivity=("J_old_median_sensitivity", "mean"),
        J_rank_1_minus_tau=("J_rank_1_minus_tau", "mean"),
        J_pairwise_violation=("J_pairwise_violation", "mean"),
        n_fits=("fit_id", "nunique"),
        all_fits_identifiable=("status", lambda x: bool(np.all(np.asarray(x) != "NOT_IDENTIFIABLE"))),
        any_local_wave_unsupported=("status", lambda x: bool(np.any(np.asarray(x) == "LOCAL_WAVE_UNSUPPORTED"))),
    )
    patient.to_csv(OUT_ROOT / "CROSSFIT_NONLOCALITY_PATIENT_SUMMARY.csv", index=False)
    example = pd.read_csv(OUT_ROOT / "nonlocality_oof/epilepsiae_1146__shared.csv.gz")
    plot_stage_d(fit, example)
    repair = {
        "status": "PREFREEZE_ESTIMAND_REPAIR",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "timing": "BEFORE_FORMAL_RNN_TRAINING_AND_BEFORE_TARGET_UNSEAL",
        "reason": "REGISTERED_EVENT_MEDIAN_WAS_EXACTLY_ZERO_IN_28_OF_28_PATIENTS",
        "original_estimand": "median_event_mean_distal_positive_z_exceedance_above_1",
        "repaired_primary_estimand": "mean_event_mean_distal_positive_z_exceedance_above_1",
        "robustness": ["temporal_10_block_median", "nonzero_event_fraction", "registered_event_median"],
        "patients_with_nonzero_repaired_J": int(np.sum(patient.J_lat_exceedance_burden > 0)),
        "patient_J_unique_values": int(patient.J_lat_exceedance_burden.nunique()),
        "model_results_read": False,
        "target_values_read": False,
        "fit_summary_sha256": sha256_file(OUT_ROOT / "CROSSFIT_NONLOCALITY_FIT_SUMMARY.csv"),
        "patient_summary_sha256": sha256_file(OUT_ROOT / "CROSSFIT_NONLOCALITY_PATIENT_SUMMARY.csv"),
        "producer_sha256": sha256_file(Path(__file__).resolve()),
    }
    (OUT_ROOT / "J_ESTIMAND_PREFREEZE_REPAIR.json").write_text(json.dumps(repair, indent=2))
    stage = json.loads((OUT_ROOT / "STAGE_D_J_COMPLETE.json").read_text())
    stage.update({
        "status": "PASS_J_FROZEN_AFTER_PREFREEZE_DEGENERACY_REPAIR",
        "primary_J": repair["repaired_primary_estimand"],
        "registered_J_status": "DEGENERATE_ALL_ZERO_RETAINED_AS_SENSITIVITY",
        "estimand_repair_manifest": str(OUT_ROOT / "J_ESTIMAND_PREFREEZE_REPAIR.json"),
        "fit_summary_sha256": repair["fit_summary_sha256"],
        "patient_summary_sha256": repair["patient_summary_sha256"],
    })
    (OUT_ROOT / "STAGE_D_J_COMPLETE.json").write_text(json.dumps(stage, indent=2))
    print(json.dumps(repair, indent=2))


if __name__ == "__main__":
    main()
