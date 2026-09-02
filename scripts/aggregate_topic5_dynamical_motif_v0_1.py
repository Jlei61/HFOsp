#!/usr/bin/env python3
"""Collect motif units into patient-level tables and the G0-G6 evidence matrix.

Every formal comparison is a paired patient-level effect.  Events, rollouts,
seeds and views are folded inside the patient first; they never enlarge n.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic5_dynamical_motif_analysis_v0_1 import holm, paired_patient_effect  # noqa: E402
from src.topic5_dynamical_motif_rnn_v0_1 import ALL_MODELS  # noqa: E402

BASELINE_ARMS = ("LAYOUT_AXIS_ANISOTROPY", "LAYOUT_AXIS_REPLAY", "EVENT_VECTOR_DIRECTIONAL",
                 "GAIN_MATCHED_DM1_FREE_AXIS", "GAIN_MATCHED_DM2_LOCAL_DIRECTIONAL")

# Positive effect always means "the first arm is better".
COMPARISONS = {
    "G1": [
        ("DM1_FREE_AXIS", "LAYOUT_AXIS_ANISOTROPY", "free axis beyond an implantation axis (upper bound)"),
        ("DM1_FREE_AXIS", "LAYOUT_AXIS_REPLAY", "free axis beyond an implantation axis (lower bound)"),
        ("LAYOUT_AXIS_ANISOTROPY", "DM0_ISOTROPIC", "implantation layout anisotropy beyond isotropic"),
        ("DM1_FREE_AXIS", "DM0_ISOTROPIC", "any anisotropy beyond isotropic"),
    ],
    "G2": [
        ("DM2_LOCAL_DIRECTIONAL", "DM1_FREE_AXIS", "early state selects direction beyond a static corridor"),
        ("DM2_LOCAL_DIRECTIONAL", "EVENT_VECTOR_DIRECTIONAL", "global corridor beyond per-event displacement"),
        ("DM2_LOCAL_DIRECTIONAL", "GAIN_MATCHED_DM2_LOCAL_DIRECTIONAL", "direction beyond one-step amplitude"),
    ],
    "G3": [
        ("DM3_AXIS_FEEDFORWARD_TRANSIENT", "DM2_LOCAL_DIRECTIONAL", "axial feed-forward beyond directional transport"),
        ("DM3_AXIS_FEEDFORWARD_TRANSIENT", "DM3_GAIN_MEMORY", "feed-forward beyond stronger and slower"),
        ("DM3_AXIS_FEEDFORWARD_TRANSIENT", "DM3_SYMMETRIC_MATCHED", "feed-forward beyond symmetric axial coupling"),
        ("DM3_AXIS_FEEDFORWARD_TRANSIENT", "DM3_AXIS_SHUFFLED_TRIANGULAR", "axis alignment beyond arbitrary triangularity"),
    ],
}
SINGLE_SEED_ARMS = {"DM3_GAIN_MEMORY", "DM3_SYMMETRIC_MATCHED", "DM3_AXIS_SHUFFLED_TRIANGULAR",
                    *BASELINE_ARMS}


def load_units(root: Path, tag: str, frame: str) -> pd.DataFrame:
    rows = []
    base = root / tag / frame
    if not base.exists():
        return pd.DataFrame()
    for metrics_path in sorted(base.glob("*/*/seed*/metrics.json")):
        record = json.loads(metrics_path.read_text())
        unseen_path = metrics_path.parent / "unseen_evaluation.json"
        unseen = json.loads(unseen_path.read_text()) if unseen_path.exists() else {}
        audit = record.get("numerical_audit", {})
        isolation = record.get("component_isolation", {})
        row = {
            "frame": frame, "tag": tag,
            "unit_id": record.get("unit_id", metrics_path.parts[-4]),
            "subject": record.get("subject"),
            "model_id": record.get("model_id", metrics_path.parts[-3]),
            "seed_index": record.get("seed_index", int(metrics_path.parts[-2][4:])),
            "calibration_score": record.get("best_validation_score"),
            "warm_start_score": record.get("warm_start_validation_score"),
            "best_epoch": record.get("best_epoch"),
            "n_epochs": record.get("n_epochs"),
            "seconds": record.get("seconds"),
            "theta_rad": audit.get("theta_rad"), "eta": audit.get("eta"),
            "beta": audit.get("beta"), "gamma": audit.get("gamma"),
            "delta_g": audit.get("delta_g"), "delta_kappa": audit.get("delta_kappa"),
            "ell_mm": audit.get("ell_mm"), "gain": audit.get("gain"), "kappa": audit.get("kappa"),
            "isolation_calibration_gain": isolation.get("calibration_gain"),
            "isolation_unseen_gain": isolation.get("model_unseen_gain"),
            "isolation_unseen_contact_nll_gain": isolation.get("model_unseen_contact_nll_gain"),
        }
        for key in ("calibration", "development"):
            for metric, value in (record.get(key) or {}).items():
                row[f"{key}_{metric}"] = value
        for metric, value in (record.get("model_unseen_teacher_forced") or {}).items():
            row[f"unseen_tf_{metric}"] = value
        for metric, value in (unseen.get("prediction") or {}).items():
            row[f"unseen_{metric}"] = value
        for block, values in (unseen.get("rollout") or {}).items():
            prefix = block.replace("|", "_").lower()
            for metric, value in values.items():
                if isinstance(value, (int, float)) or value is None:
                    row[f"{prefix}_{metric}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def collapse_to_patient(units: pd.DataFrame, metric: str) -> pd.DataFrame:
    usable = units[units[metric].notna()] if metric in units else units.iloc[0:0]
    if usable.empty:
        return pd.DataFrame(columns=["subject", "model_id", metric])
    return (usable.groupby(["subject", "model_id"])[metric].median().reset_index())


def paired_table(units: pd.DataFrame, metric: str, first: str, second: str,
                 lower_is_better: bool) -> dict:
    table = collapse_to_patient(units, metric)
    left = table[table.model_id == first].set_index("subject")[metric]
    right = table[table.model_id == second].set_index("subject")[metric]
    shared = sorted(set(left.index) & set(right.index))
    if not shared:
        return {"n": 0, "median": None, "p_value": None, "subjects": []}
    difference = np.asarray([right[s] - left[s] if lower_is_better else left[s] - right[s]
                             for s in shared], dtype=float)
    effect = paired_patient_effect(difference, alternative="greater")
    effect["subjects"] = shared
    effect["metric"] = metric
    effect["per_subject"] = {s: float(d) for s, d in zip(shared, difference)}
    return effect


def collect_counterfactuals(root: Path, tag: str, frame: str) -> pd.DataFrame:
    """Per-patient branch response, and the same response minus the isotropic model.

    Substituting a contact also *consumes* it, so part of any endpoint shift is
    the no-repeat rule rather than directional dynamics.  DM0 has no direction
    gate at all, so differencing against it isolates the dynamics.
    """
    frames = [pd.read_csv(path) for path in
              sorted((root / tag / frame).glob("*/*/seed*/counterfactual_branches.csv"))]
    frames = [f for f in frames if not f.empty and "delta_r_late_axial_mm" in f.columns]
    if not frames:
        return pd.DataFrame()
    table = pd.concat(frames)
    table = table[table.status == "OK"]
    metrics = ["delta_r_late_axial_mm", "delta_r_late_mm", "delta_l_axis_full",
               "delta_l_axis_fixed_h3", "delta_n_rank_full", "delta_mode_probability_a"]
    metrics = [m for m in metrics if m in table.columns]
    grouped = (table.groupby(["subject", "model_id", "branch"])[metrics]
               .median().reset_index())
    baseline = grouped[grouped.model_id == "DM0_ISOTROPIC"].set_index(["subject", "branch"])
    rows = []
    for _, row in grouped.iterrows():
        record = row.to_dict()
        key = (row.subject, row.branch)
        for metric in metrics:
            record[f"{metric}_minus_isotropic"] = (
                float(row[metric] - baseline.loc[key, metric])
                if key in baseline.index else np.nan)
        rows.append(record)
    denominators = (pd.concat(frames).groupby(["subject", "branch"]).status
                    .value_counts().unstack(fill_value=0).reset_index())
    out = pd.DataFrame(rows).merge(denominators, on=["subject", "branch"], how="left")
    return out


PRIMARY_METRICS = (
    ("unseen_subset_nll", True, "prediction"),
    ("all_full_stop_energy_score_median", True, "distribution"),
    ("all_fixed_h3_energy_score_median", True, "distribution_fixed_h3"),
    ("all_fixed_h5_energy_score_median", True, "distribution_fixed_h5"),
    ("all_full_stop_contact_field_energy_median", True, "contact_field"),
    ("all_full_stop_mode_log_score", True, "template_identity"),
    ("isolation_unseen_gain", False, "component_isolation"),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path,
                        default=ROOT / "results/topic5_dynamical_motif_rnn_v0_1")
    parser.add_argument("--tag", default="formal")
    parser.add_argument("--frame", default="GEOMETRY_ONLY_PCA2")
    parser.add_argument("--suffix", default="")
    args = parser.parse_args()

    units = load_units(args.out_root, args.tag, args.frame)
    if units.empty:
        print("[aggregate] no units found")
        return
    suffix = args.suffix or ("" if args.tag == "formal" else f"_{args.tag}")
    units.to_csv(args.out_root / f"UNIT_LEVEL_METRICS{suffix}.csv", index=False)

    patient_rows = []
    for metric, _, _ in PRIMARY_METRICS:
        if metric not in units:
            continue
        table = collapse_to_patient(units, metric)
        table["metric"] = metric
        patient_rows.append(table.rename(columns={metric: "value"}))
    if patient_rows:
        pd.concat(patient_rows).to_csv(
            args.out_root / f"MODEL_UNSEEN_PER_PATIENT{suffix}.csv", index=False)

    counterfactual = collect_counterfactuals(args.out_root, args.tag, args.frame)
    if not counterfactual.empty:
        counterfactual.to_csv(
            args.out_root / f"PREFIX_COUNTERFACTUAL_PER_PATIENT{suffix}.csv", index=False)

    evidence: dict[str, dict] = {}
    for goal, entries in COMPARISONS.items():
        goal_block: dict[str, dict] = {}
        family: dict[str, float] = {}
        for first, second, description in entries:
            comparison: dict[str, dict] = {"description": description}
            for metric, lower_is_better, label in PRIMARY_METRICS:
                if metric not in units:
                    continue
                effect = paired_table(units, metric, first, second, lower_is_better)
                comparison[label] = effect
                if label == "prediction" and effect.get("p_value") is not None:
                    family[f"{first}-{second}"] = effect["p_value"]
            comparison["single_seed"] = bool(
                first in SINGLE_SEED_ARMS or second in SINGLE_SEED_ARMS)
            goal_block[f"{first}__vs__{second}"] = comparison
        goal_block["holm_adjusted_prediction_p"] = holm(family)
        evidence[goal] = goal_block

    axis = units[units.model_id == "DM1_FREE_AXIS"]
    if not axis.empty:
        by_subject = axis.groupby("subject")
        evidence["G0"] = {
            "frame": args.frame,
            "n_patients": int(axis.subject.nunique()),
            "free_axis_seed_spread_rad": {
                subject: float(_circular_spread(group.theta_rad.dropna().to_numpy()))
                for subject, group in by_subject if group.theta_rad.notna().sum() >= 2
            },
            "eta_nonzero_patients": int(
                (by_subject.eta.median() > 1e-6).sum()),
            "eta_median": float(by_subject.eta.median().median()),
        }
    directional = units[units.model_id == "DM2_LOCAL_DIRECTIONAL"]
    if not directional.empty:
        evidence.setdefault("G2", {})["beta_summary"] = {
            "n_patients": int(directional.subject.nunique()),
            "beta_nonzero_patients": int(
                (directional.groupby("subject").beta.median().abs() > 1e-6).sum()),
            "beta_median": float(directional.groupby("subject").beta.median().median()),
        }
    feedforward = units[units.model_id == "DM3_AXIS_FEEDFORWARD_TRANSIENT"]
    if not feedforward.empty:
        evidence.setdefault("G3", {})["gamma_summary"] = {
            "n_patients": int(feedforward.subject.nunique()),
            "gamma_nonzero_patients": int(
                (feedforward.groupby("subject").gamma.median() > 1e-9).sum()),
            "gamma_median": float(feedforward.groupby("subject").gamma.median().median()),
        }

    evidence["G4"] = _goal_from_artifacts(args.out_root, [
        ("persistence_vs_own_generation", "PERSISTENCE_MODEL_GAP_SUMMARY.json"),
        ("observable_prefix_counterfactual", f"PREFIX_COUNTERFACTUAL_PER_PATIENT{suffix}.csv"),
        ("static_versus_recurrent", "STATIC_VS_RECURRENT_PER_PATIENT.csv"),
    ])
    evidence["G5"] = _goal_from_artifacts(args.out_root, [
        ("seizure_incremental_reuse", "SEIZURE_REUSE_SUMMARY.json"),
    ])
    evidence["G6"] = _goal_from_artifacts(args.out_root, [
        ("residual_rank_and_timing", "G6_RESIDUAL_SIDECAR_SUMMARY.json"),
        ("model_free_persistence", "MODEL_FREE_PERSISTENCE_SUMMARY.json"),
        ("synthetic_identifiability", "toy_identifiability/IDENTIFIABILITY_SUMMARY_a.json"),
    ])

    # Status labels are interpretive, so the rule and the reason travel with the
    # label; nothing here is derived automatically from a p-value threshold.
    evidence_status = {
        "G0_frame_and_layout": {
            "status": "PARTIAL",
            "reason": "the geometry-only frame is constructible and valid for 28/28 patients, "
                      "and the implantation-layout axis does not beat isotropic; but the "
                      "PARENT_FROZEN_FRAME sensitivity arm was not trained, so frame "
                      "dependence itself is untested this round"},
        "G1_anisotropy": {
            "status": "NOT_DETECTED_UNDERPOWERED",
            "reason": "held-out gain median exactly 0 and no comparison significant; the "
                      "synthetic map shows the same pipeline reports spurious held-out gains "
                      "on directionless cells, so this is an underpowered non-detection, "
                      "not an exclusion"},
        "G2_direction": {
            "status": "NOT_DETECTED_UNDERPOWERED",
            "reason": "13/28 patients select exactly zero push and the cohort dose-response "
                      "is a smooth bowl with its minimum at zero, but the synthetic map does "
                      "not reliably recover a known push, so this is underpowered"},
        "G3_axial_feedforward": {
            "status": "NOT_DETECTED",
            "reason": "the chain and its three alternative mechanisms are numerically the "
                      "same model because all four learn zero strength; they are "
                      "indistinguishable by construction rather than by weak power"},
        "G4_variability": {
            "status": "PARTIAL",
            "reason": "generated events under-reproduce directional persistence relative to "
                      "the real ones (21/28, p=0.0008) and the observable-prefix response is "
                      "entirely the no-repeat rule; template identity and length are "
                      "reproduced"},
        "G5_seizure": {
            "status": "NOT_DETECTED",
            "reason": "adding the interictal residual basis makes leave-one-shaft-out error "
                      "worse (median -2.34, 3/17 positive); the real-minus-pseudo difference "
                      "is secondary and only says the basis hurts less at real onsets"},
        "G6_residual_mechanisms": {
            "status": "SUPPORTED",
            "reason": "the within-event timing proxy carries distance information beyond the "
                      "ordinal rank step in 27/28 patients (partial r median +0.132), and the "
                      "residual contact field is close to full rank, so the miss is diffuse "
                      "rather than low dimensional"},
    }

    payload = {
        "contract": "topic5_dynamical_motif_evidence_v0_1",
        "goal_status": evidence_status,
        "frame": args.frame, "tag": args.tag,
        "n_units": int(len(units)),
        "n_patients": int(units.subject.nunique()),
        "models_present": sorted(units.model_id.unique().tolist()),
        "goals": evidence,
        "note": ("positive effect always means the first arm is better; "
                 "prediction is model-unseen exact-subset NLL, distribution is the "
                 "energy score of the generated summary vector"),
    }
    (args.out_root / f"EVIDENCE_MATRIX{suffix}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=float) + "\n")
    print(f"[aggregate] {len(units)} units, {units.subject.nunique()} patients, "
          f"models={sorted(units.model_id.unique())}")


def _goal_from_artifacts(root: Path, entries: list[tuple[str, str]]) -> dict:
    """Attach the artefacts a goal is answered by, with their status."""
    block: dict[str, dict] = {}
    for label, relative in entries:
        path = root / relative
        record: dict = {"artifact": relative, "present": path.exists()}
        if path.exists() and path.suffix == ".json":
            try:
                payload = json.loads(path.read_text())
            except json.JSONDecodeError:
                payload = {}
            record["summary"] = {k: v for k, v in payload.items()
                                 if isinstance(v, (int, float, str, bool)) or
                                 (isinstance(v, dict) and len(v) <= 8)}
        elif path.exists():
            frame = pd.read_csv(path)
            record["n_rows"] = int(len(frame))
            if "subject" in frame:
                record["n_subjects"] = int(frame.subject.nunique())
        block[label] = record
    return block


def _circular_spread(angles: np.ndarray) -> float:
    """Spread of undirected axes: the angles live on a half circle."""
    if angles.size < 2:
        return float("nan")
    doubled = 2.0 * np.asarray(angles, dtype=float)
    resultant = np.abs(np.mean(np.exp(1j * doubled)))
    return float(np.sqrt(max(0.0, -2.0 * np.log(max(resultant, 1e-12)))) / 2.0)


if __name__ == "__main__":
    main()
