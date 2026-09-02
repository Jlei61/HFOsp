#!/usr/bin/env python3
"""Seizure reuse for Topic 5.2: does the early ictal field carry an IED component?

S1 asks a deliberately weak but clean question.  Take the early seizure field,
regress out everything a static picture already explains -- a constant, the
patient's own mean interictal participation, and shaft/geometry covariates --
and ask whether a two-dimensional basis built from *interictal motif rollouts*
explains any of what is left, out of sample.  Folds are spatial (leave one shaft
out), so the basis cannot win by memorising contacts.

S2 asks whether early coefficients predict the late field.  It runs only when
the time-resolved band is estimable.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_topic5_dynamical_motif_unseen_v0_1 import load_unit_model, write_json  # noqa: E402
from src.topic5_dynamical_motif_analysis_v0_1 import paired_patient_effect  # noqa: E402
from src.topic5_dynamical_motif_data_v0_1 import load_frame_unit  # noqa: E402
from src.topic5_dynamical_motif_rnn_v0_1 import build_motif_event_tensors  # noqa: E402
from src.topic5_dynamical_motif_rollout_v0_1 import stochastic_rollout  # noqa: E402

TRACE_ROOT = ROOT / "results/topic5_dynamical_motif_rnn_v0_1/seizure_trace_cache"
PSEUDO_OFFSETS = tuple(float(v) for v in range(-120, -19, 10))
ONSET_WINDOW = (0.0, 10.0)
LATE_WINDOW = (10.0, 20.0)
BASIS_DIMENSION = 2


def field_at(trace: np.ndarray, relative: np.ndarray, start: float,
             window: tuple[float, float] = ONSET_WINDOW) -> np.ndarray:
    selected = np.where((relative >= start + window[0]) & (relative <= start + window[1]))[0]
    if selected.size == 0:
        return np.full(trace.shape[0], np.nan)
    return np.nanmean(trace[:, selected], axis=1)


RIDGE = 1e-3


def static_design(unit) -> tuple[np.ndarray, list[str]]:
    """``Z`` = constant, mean interictal participation and shaft/geometry terms.

    These patients have 6-52 contacts, so the design has to be budgeted: a
    saturated ``Z`` leaves leave-one-shaft-out with fewer training contacts than
    columns and the cross-validated error becomes meaningless (it was returning
    exact zeros and values of +-17 on a unit-variance field).  Columns are added
    in priority order while at least three contacts per column remain.
    """
    train = unit.indices(0)
    participation = (unit.ranks[train] >= 0).mean(axis=0)
    start_removed = participation - (unit.ranks[train] == 0).mean(axis=0)
    budget = max(2, unit.n_contacts // 3)
    columns = [np.ones(unit.n_contacts), start_removed]
    names = ["constant", "start_removed_participation"]
    optional = [("frame_x", unit.contacts_xy_mm[:, 0]),
                ("frame_y", unit.contacts_xy_mm[:, 1])]
    for shaft in sorted(set(unit.shafts))[:-1]:
        optional.append((f"shaft_{shaft}",
                         np.asarray([1.0 if s == shaft else 0.0 for s in unit.shafts])))
    for name, column in optional:
        if len(columns) >= budget:
            break
        columns.append(column)
        names.append(name)
    design = np.column_stack(columns)
    scale = design.std(axis=0)
    scale[scale < 1e-9] = 1.0
    design = design / scale
    return design, names


def ridge_fit(design: np.ndarray, target: np.ndarray) -> np.ndarray:
    gram = design.T @ design + RIDGE * np.eye(design.shape[1])
    return np.linalg.solve(gram, design.T @ target)


def residualise(fields: np.ndarray, design: np.ndarray) -> np.ndarray:
    return fields - (design @ ridge_fit(design, fields.T)).T


def spatial_folds(unit) -> tuple[list[np.ndarray], str]:
    shafts = np.asarray(unit.shafts)
    unique = sorted(set(shafts.tolist()))
    if len(unique) >= 3:
        return [np.flatnonzero(shafts == shaft) for shaft in unique], "leave_one_shaft_out"
    return [np.asarray([index]) for index in range(unit.n_contacts)], "leave_one_contact_out"


def cross_validated_error(target: np.ndarray, design: np.ndarray,
                          folds: list[np.ndarray]) -> float:
    """Mean squared out-of-fold error of a least-squares fit."""
    error, count = 0.0, 0
    for fold in folds:
        mask = np.ones(len(target), dtype=bool)
        mask[fold] = False
        if mask.sum() < design.shape[1] + 2:
            continue
        prediction = design[fold] @ ridge_fit(design[mask], target[mask])
        error += float(np.sum((target[fold] - prediction) ** 2))
        count += int(fold.size)
    return error / max(1, count)


def build_ied_basis(model, head, contract, unit, tensors, device, gate_rule,
                    n_events: int = 512, draws: int = 8) -> dict:
    """Target-free basis from interictal rollouts, after removing the static field."""
    train = unit.indices(0)
    if train.size == 0:
        return {"available": False}
    step = max(1, train.size // n_events)
    chosen = train[::step][:n_events]
    starts = tensors["x"][:, 0][torch.as_tensor(chosen)]
    repeated = starts.repeat_interleave(draws, dim=0)
    result = stochastic_rollout(
        model, head, contract, repeated, unit.contacts_xy_mm, device,
        mode="FULL_STOP", gate_rule=gate_rule,
        rng_label=f"{unit.unit_id}|ied_basis")
    sequence = result["sequence"]
    within = np.arange(sequence.shape[1])[None, :] <= result["n_emitted"][:, None]
    fields = ((sequence * within[..., None]).sum(axis=1) > 0).astype(float)
    design, names = static_design(unit)
    residual = residualise(fields, design)
    residual = residual - residual.mean(axis=0, keepdims=True)
    _, singular, right = np.linalg.svd(residual, full_matrices=False)
    basis = right[:BASIS_DIMENSION].T
    # Deterministic sign: the largest absolute loading is made positive.
    for column in range(basis.shape[1]):
        dominant = int(np.argmax(np.abs(basis[:, column])))
        if basis[dominant, column] < 0:
            basis[:, column] *= -1
    return {
        "available": True,
        "basis": basis,
        "design": design,
        "design_names": names,
        "n_rollout_fields": int(fields.shape[0]),
        "explained_variance_ratio": (singular[:BASIS_DIMENSION] ** 2
                                     / max(float((singular ** 2).sum()), 1e-12)).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", default="GEOMETRY_ONLY_PCA2")
    parser.add_argument("--model", default="DM2_LOCAL_DIRECTIONAL")
    parser.add_argument("--seed-index", type=int, default=0)
    parser.add_argument("--tag", default="formal")
    parser.add_argument("--out-root", type=Path,
                        default=ROOT / "results/topic5_dynamical_motif_rnn_v0_1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gate-rule", default="M2-2RANK")
    parser.add_argument("--parity-only", action="store_true")
    args = parser.parse_args()

    started = time.time()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    event_rows, patient_rows, skipped = [], [], []
    for trace_path in sorted(TRACE_ROOT.glob("*.json")):
        if trace_path.name == "SEIZURE_TRACE_BUILD_REPORT.json":
            continue
        meta = json.loads(trace_path.read_text())
        if not isinstance(meta, dict) or "seizures_cached" not in meta:
            continue
        subject = f"{meta['dataset']}_{meta['subject']}"
        unit_dir = (args.out_root / args.tag / args.frame / subject / args.model
                    / f"seed{args.seed_index}")
        if not (unit_dir / "checkpoint.pt").exists():
            skipped.append({"subject": subject, "reason": "no_interictal_checkpoint"})
            continue
        unit = load_frame_unit(args.out_root, args.frame, subject)
        model, head, contract, _ = load_unit_model(unit, unit_dir, device)
        tensors = build_motif_event_tensors(unit.ranks, unit.contacts_xy_mm,
                                            gate_rule=args.gate_rule)
        basis = build_ied_basis(model, head, contract, unit, tensors, device, args.gate_rule)
        if not basis["available"]:
            skipped.append({"subject": subject, "reason": "no_basis"})
            continue
        folds, fold_rule = spatial_folds(unit)
        design = basis["design"]
        augmented = np.column_stack([design, basis["basis"]])

        data = np.load(trace_path.with_suffix(".npz"), allow_pickle=True)
        channels = [str(v) for v in data["channels"]]
        try:
            columns = [channels.index(name) for name in unit.contact_names]
        except ValueError:
            skipped.append({"subject": subject, "reason": "contact_join_failed"})
            continue
        verified = set(meta.get("parity_verified_seizures", []))
        deltas_real, deltas_pseudo = [], []
        for index in meta["seizures_cached"]:
            if args.parity_only and index not in verified:
                continue
            trace = np.asarray(data[f"bb150_zt__{index}"], float)[columns]
            relative = np.asarray(data[f"bb150_relt__{index}"], float)
            for label, offset in [("real_onset", 0.0)] + [
                    ("pseudo_onset", value) for value in PSEUDO_OFFSETS]:
                target = field_at(trace, relative, offset)
                if not np.isfinite(target).all():
                    continue
                target = (target - target.mean()) / max(float(target.std()), 1e-9)
                error_static = cross_validated_error(target, design, folds)
                error_augmented = cross_validated_error(target, augmented, folds)
                projection = basis["basis"].T @ np.clip(target, 0, None)
                positive = np.clip(target, 0, None)
                event_rows.append({
                    "subject": subject, "seizure_index": int(index), "kind": label,
                    "offset_s": offset, "parity_verified": bool(index in verified),
                    "fold_rule": fold_rule, "n_folds": len(folds),
                    "cv_error_static": error_static,
                    "cv_error_static_plus_ied": error_augmented,
                    "delta_error": error_static - error_augmented,
                    "A_energy": float(np.sum(projection ** 2)),
                    "Q_fraction": float(np.sum(projection ** 2)
                                        / max(float(np.sum(positive ** 2)), 1e-12)),
                })
                (deltas_real if label == "real_onset" else deltas_pseudo).append(
                    error_static - error_augmented)
        if not deltas_real:
            skipped.append({"subject": subject, "reason": "no_usable_seizure"})
            continue
        patient_rows.append({
            "subject": subject, "n_seizures": len(deltas_real),
            "n_pseudo": len(deltas_pseudo),
            "fold_rule": fold_rule, "n_folds": len(folds),
            "n_contacts": unit.n_contacts,
            "n_design_columns": int(design.shape[1]),
            "basis_explained_variance": basis["explained_variance_ratio"][0],
            "delta_error_real_median": float(np.median(deltas_real)),
            "delta_error_pseudo_median": float(np.median(deltas_pseudo))
            if deltas_pseudo else float("nan"),
            "real_minus_pseudo": float(np.median(deltas_real) - np.median(deltas_pseudo))
            if deltas_pseudo else float("nan"),
            "all_parity_verified": bool(
                set(meta["seizures_cached"]) <= set(meta.get("parity_verified_seizures", []))),
        })
        print(f"[seizure] {subject}: real dE={patient_rows[-1]['delta_error_real_median']:+.4f} "
              f"pseudo dE={patient_rows[-1]['delta_error_pseudo_median']:+.4f}", flush=True)

    suffix = "_parity_only" if args.parity_only else ""
    events = pd.DataFrame(event_rows)
    patients = pd.DataFrame(patient_rows)
    events.to_csv(args.out_root / f"SEIZURE_INCREMENTAL_REUSE_PER_EVENT{suffix}.csv", index=False)
    patients.to_csv(args.out_root / f"SEIZURE_INCREMENTAL_REUSE_PER_PATIENT{suffix}.csv", index=False)
    summary = {
        "contract": "topic5_dynamical_motif_seizure_reuse_v0_1",
        "frame": args.frame, "model": args.model, "seed_index": args.seed_index,
        "parity_only": bool(args.parity_only),
        "n_patients": int(len(patients)),
        "n_seizure_events": int((events.kind == "real_onset").sum()) if not events.empty else 0,
        "n_pseudo_events": int((events.kind == "pseudo_onset").sum()) if not events.empty else 0,
        "pseudo_offsets_s": list(PSEUDO_OFFSETS),
        "basis_dimension": BASIS_DIMENSION,
        "skipped": skipped,
        "S1_real_delta_error": paired_patient_effect(
            patients.delta_error_real_median.to_numpy(), "greater") if not patients.empty else {},
        "S1_real_minus_pseudo": paired_patient_effect(
            patients.real_minus_pseudo.to_numpy(), "greater") if not patients.empty else {},
        "S2_status": "NOT_TESTED_IN_THIS_PASS",
        "seconds": time.time() - started,
    }
    write_json(args.out_root / f"SEIZURE_REUSE_SUMMARY{suffix}.json", summary)
    print(json.dumps({k: summary[k] for k in
                      ("n_patients", "n_seizure_events", "n_pseudo_events")}), flush=True)


if __name__ == "__main__":
    main()
