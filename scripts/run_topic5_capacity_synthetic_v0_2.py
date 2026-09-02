#!/usr/bin/env python3
"""Phase E: synthetic identifiability surface (S0 correctness, S1 power, S2 misspecification).

The student here is the *same* pipeline the real data goes through: the same
prefix/horizon construction, the same two unordered baselines, the same exact
subset law, the same frozen aligned basis estimated from split-0 events only,
and the same angle-rotated nulls.  Only the data comes from a teacher whose
ordered structure is known.

The surface calibrates how a real negative should be read.  It never decides
which real patients are analysed and never gates a real experiment.
"""
from __future__ import annotations

# One worker must not also fan out inside BLAS: these processes are run many at a
# time on a shared machine, and the default OpenMP thread count is the core count,
# which produced a load average of ~860 on an 80-core host before this was set.
import os as _os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os.environ.setdefault(_var, _os.environ.get("TOPIC5_TORCH_THREADS", "1"))

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic5_capacity_synthetic_v0_2 import TeacherSpec, synthesise  # noqa: E402
from src.topic5_strict_history_data_v0_2 import build_sample_set, load_seeg_patient  # noqa: E402
from src.topic5_strict_history_motif_v0_2 import (  # noqa: E402
    MotifConfig,
    OrderedMotif,
    TrainConfig,
    UnorderedBaseline,
    checkpoint_objective,
    combine_logits,
    evaluate,
    fit,
    perturb_prefix_order,
    primary_field_kind,
    tensors_from_samples,
    training_loss,
    unordered_features,
)
from src.topic5_structural_identifiability_v0_2 import (  # noqa: E402
    ANGLE_GRID_RAD,
    ANGLE_SUBSET_2,
    aligned_dictionary,
    estimate_axis_2d,
    isotropic_kernel,
    local_graph,
    local_kernel_sigma,
    orthonormal_truncation,
    rotate_axis,
)

RESULT_ROOT = ROOT / "results/topic5_capacity_constrained_history_motif_v0_2"
SYNTH_ROOT = RESULT_ROOT / "synthetic"
FRAME_ROOT = ROOT / "results/topic5_dynamical_motif_rnn_v0_1/frame_cache/GEOMETRY_ONLY_PCA2"
AUTO = "AUTONOMOUS_SHARED_OPERATOR"
DIRECT = "DIRECT_HORIZON_UPPER_BOUND"
BAG = "ORDERLESS_BAG"
TRAIN = TrainConfig(max_epochs=250, patience=30, batch_size=2048, min_updates_per_epoch=16)


def student_bases(patient, samples, rank: int, oracle_axis: np.ndarray | None = None
                  ) -> dict[str, np.ndarray]:
    """Exactly the real recipe: split-0-only axis, frozen kernel, matched rotations.

    ``oracle_axis`` replaces the estimated axis with the teacher's true one.  It is
    never available on real data; it exists so that "the machinery cannot detect
    an aligned structure" can be told apart from "the frozen axis estimator did
    not find the axis", which the S0 cells showed are different failures.
    """
    coords, coords_2d, shafts = patient.coords_3d_mm, patient.contacts_xy_mm, patient.shafts
    sigma = local_kernel_sigma(coords)
    kernel = isotropic_kernel(coords, sigma, local_graph(coords))
    train = np.flatnonzero(samples.split == 0)
    start = np.asarray(samples.start_set[train], dtype=float)
    centroid = (start @ coords_2d) / np.maximum(start.sum(axis=1, keepdims=True), 1.0)
    estimated, _ = estimate_axis_2d(
        np.asarray(samples.late_field_centroid[train], dtype=float) - centroid)
    axis = np.asarray(oracle_axis, dtype=float) if oracle_axis is not None else estimated
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    limit = max(1, min(rank, patient.n_contacts - len(set(shafts))))
    out = {"__axis__": axis, "__estimated_axis__": estimated, "__rank__": limit}
    out["aligned"] = orthonormal_truncation(
        aligned_dictionary(kernel, coords, coords_2d, axis, shafts), limit)[0]
    for index in ANGLE_SUBSET_2:
        out[f"angle{index}"] = orthonormal_truncation(
            aligned_dictionary(kernel, coords, coords_2d,
                               rotate_axis(axis, ANGLE_GRID_RAD[index]), shafts), limit)[0]
    return out


def train_baseline(batch, samples, level: str, rows):
    features = unordered_features(batch, level)
    module = UnorderedBaseline(level, samples.n_contacts, features.shape[1], batch.n_horizons,
                               samples.max_cardinality, rank=4)
    train_batch, valid_batch = batch.index(rows["train"]), batch.index(rows["valid"])
    ftrain, fvalid = features[rows["train"]], features[rows["valid"]]
    forward = lambda piece, index: training_loss(combine_logits(module(ftrain[index]), None),
                                                 piece, "full_suffix")
    objective = lambda _m: checkpoint_objective(
        evaluate(None, module(fvalid), valid_batch, rows["xy"]), None)
    fit(module, forward, train_batch, valid_batch, objective, replace(TRAIN, seed=11))
    with torch.no_grad():
        return {key: value for key, value in module(features).items()}


def train_arm(batch, samples, baseline, rows, basis, family, seed):
    free = basis is None
    config = MotifConfig("H1_PATIENT_ALIGNED" if not free else "H1_FREE_LOW_RANK", family,
                         rows["rank"], samples.n_contacts, batch.n_horizons,
                         samples.max_cardinality, free_basis=free)
    torch.manual_seed(seed + 1000)
    model = OrderedMotif(config, basis)
    kind = primary_field_kind(family)
    train_batch, valid_batch = batch.index(rows["train"]), batch.index(rows["valid"])
    btrain = {k: v[rows["train"]] for k, v in baseline.items()}
    bvalid = {k: v[rows["valid"]] for k, v in baseline.items()}
    forward = lambda piece, index: training_loss(
        combine_logits({k: v[index] for k, v in btrain.items()}, model(piece)), piece, kind)
    objective = lambda _m: checkpoint_objective(
        evaluate(model, bvalid, valid_batch, rows["xy"]), family)
    fit(model, forward, train_batch, valid_batch, objective, replace(TRAIN, seed=seed))
    test_batch = batch.index(rows["test"])
    btest = {k: v[rows["test"]] for k, v in baseline.items()}
    intact = evaluate(model, btest, test_batch, rows["xy"])
    score = checkpoint_objective(intact, family)
    ablated = checkpoint_objective(
        evaluate(model, btest, test_batch, rows["xy"], ordered_path=False), family)
    permuted = checkpoint_objective(
        evaluate(model, btest, perturb_prefix_order(test_batch, "swap_middle"), rows["xy"]), family)
    return {"objective": score, "ordered_path_ablation_cost": ablated - score,
            "prefix_order_cost": permuted - score}


def run_cell(payload: dict) -> dict:
    torch.set_num_threads(int(os.environ.get("TOPIC5_TORCH_THREADS", "2")))
    spec = TeacherSpec(**payload["spec"])
    block = payload["block"]
    try:
        patient, truth = synthesise(spec)
        samples = build_sample_set(patient, prefix_len=3)
        if samples.n_samples < 200 or (samples.split == 0).sum() < 100:
            return {"block": block, "spec": spec.key(), "skipped": "too few eligible events",
                    **truth}
        observed = np.flatnonzero(samples.split >= 0)
        batch = tensors_from_samples(samples, observed)
        split = samples.split[observed]
        rows = {
            "train": torch.as_tensor(np.flatnonzero(split == 0)),
            "valid": torch.as_tensor(np.flatnonzero(split == 1)),
            "test": torch.as_tensor(np.flatnonzero(split == 2)),
            "xy": torch.as_tensor(patient.contacts_xy_mm, dtype=torch.float32),
        }
        bases = student_bases(patient, samples, spec.rank,
                              np.asarray(truth["axis"]) if payload.get("oracle_axis") else None)
        rows["rank"] = bases["__rank__"]
        record = {"block": block, "spec": spec.key(), "montage": spec.montage,
                  "n_observed_contacts": patient.n_contacts, "n_samples": samples.n_samples,
                  "student_rank": bases["__rank__"], **truth}
        record["oracle_axis"] = bool(payload.get("oracle_axis"))
        estimated = bases["__estimated_axis__"]
        record["axis_angle_error_rad"] = float(np.arccos(np.clip(
            abs(np.dot(estimated / np.linalg.norm(estimated),
                       np.asarray(truth["axis"]) / np.linalg.norm(truth["axis"]))), 0.0, 1.0)))

        for level in payload["levels"]:
            baseline = train_baseline(batch, samples, level, rows)
            baseline_score = checkpoint_objective(
                evaluate(None, {k: v[rows["test"]] for k, v in baseline.items()},
                         batch.index(rows["test"]), rows["xy"]), AUTO)
            record[f"{level}_baseline_objective"] = baseline_score
            for family in payload["families"]:
                aligned = train_arm(batch, samples, baseline, rows, bases["aligned"], family, 0)
                nulls = [train_arm(batch, samples, baseline, rows, bases[f"angle{index}"], family, 0)
                         for index in ANGLE_SUBSET_2]
                null_median = float(np.median([entry["objective"] for entry in nulls]))
                tag = f"{level}_{'auto' if family == AUTO else 'direct'}"
                record[f"{tag}_aligned_objective"] = aligned["objective"]
                record[f"{tag}_angle_null_median"] = null_median
                record[f"{tag}_structure_effect"] = null_median - aligned["objective"]
                record[f"{tag}_aligned_beats_null"] = bool(aligned["objective"] < null_median)
                record[f"{tag}_prefix_order_cost"] = aligned["prefix_order_cost"]
                record[f"{tag}_ordered_path_ablation_cost"] = aligned["ordered_path_ablation_cost"]
            if payload["with_free"]:
                free = train_arm(batch, samples, baseline, rows, None, AUTO, 0)
                record[f"{level}_free_objective"] = free["objective"]
                record[f"{level}_free_minus_baseline"] = baseline_score - free["objective"]
            if payload["with_bag"]:
                bag = train_arm(batch, samples, baseline, rows, bases["aligned"], BAG, 0)
                record[f"{level}_bag_objective"] = bag["objective"]
                record[f"{level}_ordered_minus_bag"] = (
                    bag["objective"] - record[f"{level}_direct_aligned_objective"]
                    if f"{level}_direct_aligned_objective" in record else float("nan"))
        if {"U_MINIMAL", "U_FULL_SET"} <= set(payload["levels"]):
            record["bypass_interaction"] = (record["U_MINIMAL_auto_structure_effect"]
                                            - record["U_FULL_SET_auto_structure_effect"])
        return record
    except Exception:
        return {"block": block, "spec": spec.key(), "error": traceback.format_exc()}


def s0_cells() -> list[dict]:
    """Canonical correctness cells — implementation only, never a science gate."""
    base = {"montage": "medium_many_2d_near", "n_events": 3000}
    cells = []
    for label, override in (
        ("effect_zero", {"effect": 0.0, "bypass": 1.0}),
        ("aligned_strong", {"effect": 2.5, "bypass": 0.5}),
        ("aligned_strong_high_bypass", {"effect": 2.5, "bypass": 3.0}),
        ("bypass_only", {"effect": 0.0, "bypass": 3.0}),
    ):
        for seed in (0, 1, 2):
            for oracle in (False, True):
                cells.append({
                    "block": f"S0_{label}" + ("_oracle_axis" if oracle else ""),
                    "spec": {**base, **override, "seed": seed}, "oracle_axis": oracle,
                    "levels": ["U_FULL_SET", "U_MINIMAL"], "families": [AUTO, DIRECT],
                    "with_free": True, "with_bag": True})
    return cells


def s1_cells() -> list[dict]:
    from src.topic5_capacity_synthetic_v0_2 import MONTAGE_LIBRARY
    cells = []
    for montage in MONTAGE_LIBRARY:
        for effect in (0.0, 1.2, 2.5):
            for bypass in (0.5, 3.0):
                for noise in (0.7, 1.4):
                    cells.append({
                        "block": "S1_power_oracle_axis",
                        "spec": {"montage": montage, "effect": effect, "bypass": bypass,
                                 "noise": noise, "n_events": 3000, "seed": 0},
                        "levels": ["U_FULL_SET"], "families": [AUTO],
                        "with_free": False, "with_bag": False, "oracle_axis": True})
                    cells.append({
                        "block": "S1_power",
                        "spec": {"montage": montage, "effect": effect, "bypass": bypass,
                                 "noise": noise, "n_events": 3000, "seed": 0},
                        "levels": ["U_FULL_SET", "U_MINIMAL"], "families": [AUTO, DIRECT],
                        "with_free": True, "with_bag": False, "oracle_axis": False})
    return cells


def s2_cells(n_lhs: int) -> list[dict]:
    rng = np.random.default_rng(20260817)
    cells = []
    # Latin hypercube over the misspecification knobs
    grid = (np.argsort(rng.random((n_lhs, 6)), axis=0) + rng.random((n_lhs, 6))) / n_lhs
    for row in range(n_lhs):
        unobserved, extra, jitter, bypass, noise, source = grid[row]
        cells.append({
            "block": "S2_misspecification",
            "spec": {"montage": "large_many_2d_near", "effect": 1.8,
                     "bypass": float(0.5 + 3.0 * bypass), "noise": float(0.6 + 1.2 * noise),
                     "extra_state": int(round(3 * extra)),
                     "direction_jitter_rad": float(0.6 * jitter),
                     "unobserved_fraction": float(0.5 * unobserved),
                     "mask_kind": ("random", "shaft_like", "source_avoiding")[int(3 * source) % 3],
                     "n_events": 3000, "seed": row},
            "levels": ["U_FULL_SET"], "families": [AUTO], "with_free": True,
            "with_bag": False, "oracle_axis": False})
    # mask-kind comparison at frozen teacher settings
    for kind in ("random", "shaft_like", "source_avoiding"):
        for fraction in (0.15, 0.30, 0.45):
            cells.append({
                "block": "S2_mask_kind",
                "spec": {"montage": "large_many_2d_near", "effect": 1.8, "bypass": 1.5,
                         "noise": 1.0, "unobserved_fraction": fraction, "mask_kind": kind,
                         "n_events": 3000, "seed": 7},
                "levels": ["U_FULL_SET"], "families": [AUTO], "with_free": True,
                "with_bag": False, "oracle_axis": False})
    # the 28 recorded implantation layouts under one frozen canonical teacher
    for path in sorted(FRAME_ROOT.iterdir()):
        if not path.is_dir():
            continue
        patient = load_seeg_patient(FRAME_ROOT, path.name)
        cells.append({
            "block": "S2_patient_montage",
            "spec": {"montage": path.name, "effect": 1.8, "bypass": 1.5, "noise": 1.0,
                     "n_events": 3000, "seed": 3,
                     "montage_override": {"coords": patient.coords_3d_mm.tolist(),
                                          "shafts": list(patient.shafts), "source": "near"}},
            "levels": ["U_FULL_SET"], "families": [AUTO], "with_free": True,
            "with_bag": False, "oracle_axis": False})
    return cells


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--blocks", default="S0,S1,S2")
    parser.add_argument("--lhs", type=int, default=32)
    arguments = parser.parse_args()
    SYNTH_ROOT.mkdir(parents=True, exist_ok=True)

    cells: list[dict] = []
    wanted = set(arguments.blocks.split(","))
    if "S0" in wanted:
        cells += s0_cells()
    if "S1" in wanted:
        cells += s1_cells()
    if "S2" in wanted:
        cells += s2_cells(arguments.lhs)
    print(f"synthetic cells: {len(cells)}", flush=True)

    started = time.time()
    with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
        records = []
        for index, record in enumerate(pool.map(run_cell, cells), start=1):
            records.append(record)
            if index % 25 == 0:
                print(f"  [{index}/{len(cells)}] elapsed={time.time() - started:.0f}s", flush=True)
    table = pd.DataFrame(records)
    table.to_csv(SYNTH_ROOT / "SYNTHETIC_CELLS.csv", index=False)

    payload = {name: table[name].to_numpy() for name in table.columns
               if table[name].dtype.kind in "fbi"}
    payload["spec"] = table["spec"].astype(str).to_numpy()
    payload["block"] = table["block"].astype(str).to_numpy()
    np.savez_compressed(SYNTH_ROOT / "SYNTHETIC_IDENTIFIABILITY_SURFACE.npz", **payload)
    np.savez_compressed(RESULT_ROOT / "SYNTHETIC_IDENTIFIABILITY_SURFACE.npz", **payload)

    def rate(frame: pd.DataFrame, column: str) -> float | None:
        values = frame[column].dropna() if column in frame else pd.Series(dtype=float)
        return float(values.mean()) if len(values) else None

    summary = {
        "contract": "topic5_capacity_constrained_history_motif_v0_2_synthetic",
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "n_cells": int(len(table)),
        "n_failed": int(table["error"].notna().sum()) if "error" in table else 0,
        "n_skipped": int(table["skipped"].notna().sum()) if "skipped" in table else 0,
        "role": "calibrates how a real negative should be read; never a gate",
        "S0_correctness": {},
        "S1_power": {},
        "S2_misspecification": {},
    }
    ok = table[table.get("error").isna()] if "error" in table else table
    for label in sorted({b for b in ok["block"] if b.startswith("S0")}):
        frame = ok[ok["block"] == label]
        summary["S0_correctness"][label] = {
            "n": int(len(frame)),
            "P_aligned_beats_angle_null_autonomous": rate(frame, "U_FULL_SET_auto_aligned_beats_null"),
            "P_aligned_beats_angle_null_direct": rate(frame, "U_FULL_SET_direct_aligned_beats_null"),
            "median_structure_effect_autonomous": float(
                frame["U_FULL_SET_auto_structure_effect"].median())
            if "U_FULL_SET_auto_structure_effect" in frame else None,
            "median_prefix_order_cost": float(frame["U_FULL_SET_auto_prefix_order_cost"].median())
            if "U_FULL_SET_auto_prefix_order_cost" in frame else None,
            "median_ordered_path_ablation_cost": float(
                frame["U_FULL_SET_auto_ordered_path_ablation_cost"].median())
            if "U_FULL_SET_auto_ordered_path_ablation_cost" in frame else None,
            "median_bypass_interaction": float(frame["bypass_interaction"].median())
            if "bypass_interaction" in frame else None,
        }
    for name in ("S1_power", "S1_power_oracle_axis"):
        power = ok[ok["block"] == name]
        if not len(power):
            continue
        for (effect, bypass), frame in power.groupby(["effect", "bypass"]):
            summary["S1_power"][f"{name}|effect{effect}|bypass{bypass}"] = {
                "n": int(len(frame)),
                "P_aligned_beats_angle_null_autonomous": rate(frame, "U_FULL_SET_auto_aligned_beats_null"),
                "P_aligned_beats_angle_null_direct": rate(frame, "U_FULL_SET_direct_aligned_beats_null"),
                "median_axis_angle_error_rad": float(frame["axis_angle_error_rad"].median()),
                "median_cloud_aspect_2d": float(frame["cloud_aspect_2d"].median())
                if "cloud_aspect_2d" in frame else None,
            }
    for label in ("S2_misspecification", "S2_mask_kind", "S2_patient_montage"):
        frame = ok[ok["block"] == label]
        if not len(frame):
            continue
        summary["S2_misspecification"][label] = {
            "n": int(len(frame)),
            "P_aligned_beats_angle_null": rate(frame, "U_FULL_SET_auto_aligned_beats_null"),
            "median_structure_effect": float(frame["U_FULL_SET_auto_structure_effect"].median()),
            "median_axis_angle_error_rad": float(frame["axis_angle_error_rad"].median()),
            "median_observed_fraction": float(frame["observed_fraction"].median()),
        }
    (SYNTH_ROOT / "SYNTHETIC_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2)[:2500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
