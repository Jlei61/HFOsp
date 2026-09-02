#!/usr/bin/env python3
"""Patient-first rank-1/2/3 cumulative shared-subspace sensitivity summary."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic5_latent_landscape_v0_2 import atomic_write_csv, atomic_write_json, sha256_file  # noqa: E402
from src.topic5_shared_functional_necessity_v0_1 import LESION_DOSES, REAL_ARMS, dose_auc, holm_adjust  # noqa: E402
from scripts.run_topic5_latent_pass1_v0_2 import OUT, PARENT  # noqa: E402
from scripts.run_topic5_shared_functional_necessity_v0_1 import (  # noqa: E402
    FUTURE_TAU,
    NECESSITY,
    SUBSPACE,
    SUBSPACE_REVISION,
    direction_dir,
    subspace_dir,
)
from scripts.summarize_topic5_shared_functional_necessity_v0_1 import (  # noqa: E402
    bootstrap_median_ci,
    greater_p,
    positive_counts,
    state_weighted_effect,
)


REVISION = "PATIENT_FIRST_RANK123_PAIRWISE_COMMON_SUPPORT_R1_TARGET_FREE_PHASE_CENTER"
METRICS = ("SHARED", "SHARED_MINUS_C_SUFFIX", "SHARED_MINUS_PCA")


def extract_cell(row: pd.Series) -> tuple[list[dict[str, object]], dict[str, object]]:
    path = subspace_dir(row) / "subspace_response.npz"
    with np.load(path, allow_pickle=False) as source:
        z = {name: np.asarray(source[name]) for name in source.files}
    family = {str(name): index for index, name in enumerate(z["family_names"].tolist())}
    delta = z["delta_nll"].astype(float)
    valid = z["valid"].astype(bool)
    phase = z["phase_target"].astype(float)
    delayed = np.asarray(FUTURE_TAU, int)
    selectors = {"ALL": np.ones(len(phase), bool)}
    for phase_target in np.sort(np.unique(phase)):
        selectors[f"{phase_target:.2f}"] = np.isclose(phase, phase_target)
    comparisons = {
        "SHARED": ("SHARED", None),
        "SHARED_MINUS_C_SUFFIX": ("SHARED", "C_SUFFIX"),
        "SHARED_MINUS_PCA": ("SHARED", "PCA"),
    }
    rows = []
    for rank_index, rank in enumerate(z["ranks"].astype(int)):
        for dose_index, dose in enumerate(z["doses"].astype(float)):
            for phase_name, selector in selectors.items():
                for metric, (left_name, right_name) in comparisons.items():
                    left_index = family[left_name]
                    right_index = family[right_name] if right_name is not None else None
                    left_values = np.take(delta[:, left_index, rank_index, dose_index, :], delayed, axis=-1)
                    left_flags = np.take(valid[:, left_index, rank_index, dose_index, :], delayed, axis=-1)
                    right_values = None if right_index is None else np.take(
                        delta[:, right_index, rank_index, dose_index, :], delayed, axis=-1
                    )
                    right_flags = None if right_index is None else np.take(
                        valid[:, right_index, rank_index, dose_index, :], delayed, axis=-1
                    )
                    effect, n_states, n_decisions = state_weighted_effect(
                        left_values, left_flags, right_values, right_flags, selector
                    )
                    rows.append({
                        "patient": str(row.patient), "fit_id": str(row.fit_id),
                        "public_arm": str(row.public_arm), "seed": int(row.seed),
                        "phase": phase_name, "rank": int(rank), "dose": float(dose),
                        "metric": metric, "effect_nll_per_decision": effect,
                        "n_reference_states": n_states, "n_delayed_decisions": n_decisions,
                    })
    with np.load(PARENT / "cache" / str(row.fit_id) / "events.npz", allow_pickle=False) as source:
        split = np.asarray(source["split"])
    done = json.loads((subspace_dir(row) / "DONE.json").read_text())
    direction_path = direction_dir(str(row.fit_id), str(row.public_arm)) / "direction_contract.npz"
    audit = {
        "fit_id": str(row.fit_id), "public_arm": str(row.public_arm), "seed": int(row.seed),
        "valid_values_finite": bool(np.isfinite(delta[valid]).all()),
        "max_displacement_norm_error": float(np.nanmax(np.abs(
            z["actual_displacement_norm"] - z["displacement_norm"][:, None]
        ))),
        "direction_hash_matches": done["direction_contract_sha256"] == sha256_file(direction_path),
        "heldout_reference_split_is_test": bool(np.all(split[z["event_index"].astype(int)] == 2)),
        "all_ranks_have_shared_support": bool(all(
            any(item["phase"] == "ALL" and item["metric"] == "SHARED"
                and item["rank"] == rank and item["n_delayed_decisions"] > 0 for item in rows)
            for rank in (1, 2, 3)
        )),
    }
    return rows, audit


def aggregate(cell: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    keys = ["phase", "rank", "dose", "metric"]
    fit_arm = cell.groupby(["patient", "fit_id", "public_arm", *keys], as_index=False).agg(
        effect_nll_per_decision=("effect_nll_per_decision", "mean"),
        n_seeds=("seed", "nunique"), n_reference_states=("n_reference_states", "sum"),
        n_delayed_decisions=("n_delayed_decisions", "sum"),
    )
    fit = fit_arm.groupby(["patient", "fit_id", *keys], as_index=False).agg(
        effect_nll_per_decision=("effect_nll_per_decision", "mean"),
        n_arms=("public_arm", "nunique"), n_reference_states=("n_reference_states", "sum"),
        n_delayed_decisions=("n_delayed_decisions", "sum"),
    )
    patient = fit.groupby(["patient", *keys], as_index=False).agg(
        effect_nll_per_decision=("effect_nll_per_decision", "mean"),
        n_fits=("fit_id", "nunique"), n_reference_states=("n_reference_states", "sum"),
        n_delayed_decisions=("n_delayed_decisions", "sum"),
    )
    return fit_arm, fit, patient


def auc_table(patient: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in patient.groupby(["patient", "phase", "rank", "metric"], sort=True):
        values = group.set_index("dose")["effect_nll_per_decision"]
        effects = np.asarray([values.get(float(dose), np.nan) for dose in LESION_DOSES], float)
        rows.append({
            "patient": keys[0], "phase": keys[1], "rank": int(keys[2]), "metric": keys[3],
            "dose_auc_nll": dose_auc(LESION_DOSES, effects),
        })
    return pd.DataFrame(rows)


def inference(auc: pd.DataFrame) -> pd.DataFrame:
    rows = []
    use = auc[auc.phase.eq("ALL")]
    for rank in (1, 2, 3):
        for metric in METRICS:
            values = use[(use["rank"] == rank) & use.metric.eq(metric)]["dose_auc_nll"].to_numpy(float)
            values = values[np.isfinite(values)]
            low, high = bootstrap_median_ci(values, seed=6100 + rank * 10 + METRICS.index(metric))
            positive, negative, zero = positive_counts(values)
            rows.append({
                "rank": rank, "metric": metric, "n_patients": int(len(values)),
                "median_dose_auc_nll": float(np.median(values)) if len(values) else float("nan"),
                "ci95_low": low, "ci95_high": high,
                "positive": positive, "negative": negative, "zero": zero,
                "p_greater": greater_p(values),
            })
    frame = pd.DataFrame(rows)
    sensitivity = frame[frame["rank"].isin([2, 3])].index
    frame["p_holm_rank23_sensitivity"] = np.nan
    frame.loc[sensitivity, "p_holm_rank23_sensitivity"] = holm_adjust(frame.loc[sensitivity, "p_greater"])
    return frame


def main() -> None:
    manifest = pd.read_csv(OUT / "CHECKPOINT_MANIFEST.csv")
    real = manifest[manifest.public_arm.isin(REAL_ARMS)].copy()
    rows, audits = [], []
    for _, row in real.iterrows():
        cell_rows, audit = extract_cell(row)
        rows.extend(cell_rows); audits.append(audit)
    cell = pd.DataFrame(rows); audit_frame = pd.DataFrame(audits)
    fit_arm, fit, patient = aggregate(cell)
    auc = auc_table(patient); stats = inference(auc)
    atomic_write_csv(NECESSITY / "SUBSPACE_CELL_PAIR_DOSE_EFFECTS.csv", cell)
    atomic_write_csv(NECESSITY / "SUBSPACE_FIT_ARM_PAIR_DOSE_EFFECTS.csv", fit_arm)
    atomic_write_csv(NECESSITY / "SUBSPACE_FIT_PAIR_DOSE_EFFECTS.csv", fit)
    atomic_write_csv(NECESSITY / "SUBSPACE_PATIENT_PAIR_DOSE_EFFECTS.csv", patient)
    atomic_write_csv(NECESSITY / "SUBSPACE_PATIENT_AUC_EFFECTS.csv", auc)
    atomic_write_csv(NECESSITY / "SUBSPACE_INFERENCE.csv", stats)
    atomic_write_csv(NECESSITY / "SUBSPACE_CELL_AUDIT.csv", audit_frame)
    execution = json.loads((SUBSPACE / "SUBSPACE_EXECUTION_STATUS.json").read_text())
    cell_summary = pd.read_csv(SUBSPACE / "SUBSPACE_CELL_SUMMARY.csv")
    checks = {
        "subspace_504_pass": execution.get("status") == "PASS" and execution.get("completed_cells") == 504,
        "subspace_target_free_revision": (
            execution.get("revision") == SUBSPACE_REVISION
            and bool((cell_summary.revision == SUBSPACE_REVISION).all())
        ),
        "target_free_center_contract": (
            execution.get("state_center_definition") == "TRAIN_FITTED_PHASE_CURVE_GAMMA"
            and execution.get("heldout_future_field_used_in_state_center") is False
            and execution.get("heldout_future_field_used_in_support_gate") is False
            and execution.get("heldout_outcome_keys_dropped_before_lesion") is True
        ),
        "valid_nll_values_finite": bool(audit_frame.valid_values_finite.all()),
        "control_displacement_norm_exact": bool(audit_frame.max_displacement_norm_error.max() <= 1e-6),
        "direction_hashes_match": bool(audit_frame.direction_hash_matches.all()),
        "heldout_references_all_test": bool(audit_frame.heldout_reference_split_is_test.all()),
        "all_cells_all_ranks_have_support": bool(audit_frame.all_ranks_have_shared_support.all()),
        "model_hashes_504_unchanged": int(cell_summary.model_hash_unchanged.sum()) == 504,
        "decoder_hashes_504_unchanged": int(cell_summary.decoder_hash_unchanged.sum()) == 504,
        "patient_denominator_28": int(auc[auc.phase.eq("ALL")].patient.nunique()) == 28,
    }
    audit = {
        "contract": "topic5_shared_subspace_sensitivity_audit_v0_2",
        "created_utc": datetime.now(timezone.utc).isoformat(), "revision": REVISION,
        "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
        "max_control_displacement_norm_error": float(audit_frame.max_displacement_norm_error.max()),
        "execution_status_sha256": sha256_file(SUBSPACE / "SUBSPACE_EXECUTION_STATUS.json"),
    }
    atomic_write_json(NECESSITY / "SUBSPACE_FINAL_AUDIT.json", audit)
    rank23 = stats[stats["rank"].isin([2, 3])]
    selective = []
    for rank in (2, 3):
        part = rank23[rank23["rank"] == rank].set_index("metric")
        selective.append(bool(
            (part.loc[list(METRICS), "median_dose_auc_nll"] > 0).all()
            and (part.loc[list(METRICS), "p_holm_rank23_sensitivity"] < 0.05).all()
        ))
    if any(selective):
        verdict = "EXPLORATORY_HIGHER_RANK_SELECTIVE_DAMAGE"
    elif any(
        (rank23[(rank23["rank"] == rank) & rank23.metric.eq("SHARED")]["median_dose_auc_nll"] > 0).all()
        and (rank23[(rank23["rank"] == rank) & rank23.metric.eq("SHARED")]["p_holm_rank23_sensitivity"] < 0.05).all()
        for rank in (2, 3)
    ):
        verdict = "GENERIC_HIGHER_RANK_DAMAGE_ONLY"
    else:
        verdict = "RANK123_SENSITIVITY_UNSUPPORTED"
    summary = {
        "contract": "topic5_shared_subspace_sensitivity_summary_v0_2",
        "created_utc": datetime.now(timezone.utc).isoformat(), "revision": REVISION,
        "status": "PASS" if audit["status"] == "PASS" else "AUDIT_FAILED",
        "verdict": verdict,
        "does_not_override_rank1_primary": True,
        "inference": [
            {key: (None if isinstance(value, float) and not np.isfinite(value) else value)
             for key, value in record.items()}
            for record in stats.to_dict(orient="records")
        ],
    }
    atomic_write_json(NECESSITY / "SUBSPACE_SENSITIVITY_SUMMARY.json", summary)
    print(json.dumps({"audit": audit, "summary": summary}, indent=2))
    if audit["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
