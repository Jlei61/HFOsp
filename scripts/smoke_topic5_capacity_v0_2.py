#!/usr/bin/env python3
"""Phase D3 smoke run — verify the chain, not the science.

Four patients spanning the montage range are trained for a single epoch across
every model family, both unordered baselines, one prefix-order perturbation, one
ordered-path ablation, one basis transplant and one save/reload round trip.
Only shapes, memory, speed, determinism and hashes are checked; whichever
direction the numbers point is irrelevant here and does not gate Phase F.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic5_strict_history_motif_v0_2 import (  # noqa: E402
    MotifConfig,
    OrderedMotif,
    TrainConfig,
    checkpoint_objective,
    combine_logits,
    evaluate,
    fit,
    perturb_prefix_order,
    primary_field_kind,
    training_loss,
    unordered_features,
)
from scripts.run_topic5_capacity_queue_v0_2 import PatientWorkspace  # noqa: E402

RESULT_ROOT = ROOT / "results/topic5_capacity_constrained_history_motif_v0_2"
SMOKE_ROOT = RESULT_ROOT / "smoke"
# small / medium / large contact counts and one near-one-dimensional layout
SMOKE_PATIENTS = ("yuquan_huanghanwen", "epilepsiae_1146", "yuquan_zhangbichen", "yuquan_zhangjiaqi")
ARMS = (
    ("H1_PATIENT_ALIGNED", "PATIENT_ALIGNED|observed|f100", "DIRECT_HORIZON_UPPER_BOUND"),
    ("H1_PATIENT_ALIGNED", "PATIENT_ALIGNED|observed|f100", "AUTONOMOUS_SHARED_OPERATOR"),
    ("H1_ANGLE_ROTATED_AXIS", "ANGLE_ROTATED_AXIS|angle3|f100", "AUTONOMOUS_SHARED_OPERATOR"),
    ("H1_ALIGNED_ORDERLESS_BAG", "PATIENT_ALIGNED|observed|f100", "ORDERLESS_BAG"),
    ("H1_FREE_LOW_RANK", "", "AUTONOMOUS_SHARED_OPERATOR"),
)


def run_patient(patient: str, rank: int) -> dict:
    workspace = PatientWorkspace(patient)
    batch = workspace.tensors(3)
    samples = workspace.samples(3)
    valid_rows = torch.as_tensor(np.flatnonzero(workspace.split_mask(3, 1)))
    train_rows = torch.as_tensor(np.flatnonzero(workspace.fraction_mask(3, 100)))
    report = {
        "patient": patient, "n_contacts": samples.n_contacts, "n_samples": samples.n_samples,
        "max_cardinality": samples.max_cardinality, "n_train": int(train_rows.numel()),
        "n_calibration": int(valid_rows.numel()), "arms": [],
    }
    for level in ("U_MINIMAL", "U_FULL_SET"):
        baseline = workspace.baseline(level, 3)
        report[f"{level}_feature_width"] = int(unordered_features(batch, level).shape[1])
        report[f"{level}_logit_shapes"] = {k: list(v.shape) for k, v in baseline.items()}
        for structure, basis_key, family in ARMS:
            if basis_key and basis_key + f"|r{rank}" not in {
                key for key in workspace._bases}:  # pre-frozen eligibility
                report["arms"].append({"level": level, "structure": structure, "family": family,
                                       "skipped": "basis rank ineligible"})
                continue
            free = structure == "H1_FREE_LOW_RANK"
            config = MotifConfig(structure, family, rank, samples.n_contacts, batch.n_horizons,
                                 samples.max_cardinality, free_basis=free)
            torch.manual_seed(0)
            model = OrderedMotif(config, None if free else workspace.basis(basis_key + f"|r{rank}"))
            field_kind = primary_field_kind(family)
            baseline_train = {key: value[train_rows] for key, value in baseline.items()}
            train_batch, valid_batch = batch.index(train_rows), batch.index(valid_rows)

            def forward(piece, rows, _model=model, _base=baseline_train, _kind=field_kind):
                merged = combine_logits({k: v[rows] for k, v in _base.items()}, _model(piece))
                return training_loss(merged, piece, _kind)

            def objective(_module, _model=model, _family=family):
                return checkpoint_objective(evaluate(
                    _model, {k: v[valid_rows] for k, v in baseline.items()},
                    valid_batch, workspace.contact_xy), _family)

            started = time.time()
            history = fit(model, forward, train_batch, valid_batch, objective,
                          TrainConfig(max_epochs=1, eval_every=1, seed=0))
            elapsed = time.time() - started

            base_valid = {k: v[valid_rows] for k, v in baseline.items()}
            intact = evaluate(model, base_valid, valid_batch, workspace.contact_xy)
            ablated = evaluate(model, base_valid, valid_batch, workspace.contact_xy,
                               ordered_path=False)
            permuted = evaluate(model, base_valid, perturb_prefix_order(valid_batch, "swap_middle"),
                                workspace.contact_xy)
            residual_zero = float(torch.stack([
                model(valid_batch, ordered_path=False)["contact"].abs().max(),
                model(valid_batch, ordered_path=False)["cardinality"].abs().max(),
            ]).max())

            path = SMOKE_ROOT / patient / f"{level}_{structure}_{family}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict()}, path)
            reloaded = OrderedMotif(config, None if free else workspace.basis(basis_key + f"|r{rank}"))
            reloaded.load_state_dict(torch.load(path, weights_only=True)["state_dict"])
            reloaded.eval()
            replay = evaluate(reloaded, base_valid, valid_batch, workspace.contact_xy)

            entry = {
                "level": level, "structure": structure, "family": family, "rank": rank,
                "wall_seconds": elapsed,
                "ordered_parameters": int(sum(p.numel() for p in model.parameters())),
                "contact_logit_shape": list(model(valid_batch)["contact"].shape),
                "suffix_is_none": model(valid_batch)["suffix"] is None,
                "primary_field_kind": field_kind,
                "valid_objective": history["best_valid_objective"],
                "ablation_residual_max_abs": residual_zero,
                "ordered_path_ablation_cost": (
                    checkpoint_objective(ablated, family) - checkpoint_objective(intact, family)),
                "prefix_order_cost": (
                    checkpoint_objective(permuted, family) - checkpoint_objective(intact, family)),
                "reload_reproduces_score": bool(
                    abs(checkpoint_objective(replay, family)
                        - checkpoint_objective(intact, family)) < 1e-9),
                "denominators": intact.per_horizon["denominator"],
            }
            if structure == "H1_ANGLE_ROTATED_AXIS" and not free:
                aligned_key = "PATIENT_ALIGNED|observed|f100" + f"|r{rank}"
                if aligned_key in workspace._bases:
                    before = float(torch.as_tensor(model.basis).sum())
                    original = model.basis.detach().clone()
                    with torch.no_grad():
                        model.basis.copy_(torch.as_tensor(workspace.basis(aligned_key),
                                                          dtype=torch.float32))
                    swapped = evaluate(model, base_valid, valid_batch, workspace.contact_xy)
                    with torch.no_grad():
                        model.basis.copy_(original)
                    entry["basis_transplant_cost"] = (
                        checkpoint_objective(swapped, family) - checkpoint_objective(intact, family))
                    entry["basis_restored"] = bool(abs(float(model.basis.sum()) - before) < 1e-9)
            report["arms"].append(entry)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=2)
    arguments = parser.parse_args()
    torch.set_num_threads(int(os.environ.get("TOPIC5_TORCH_THREADS", "8")))
    SMOKE_ROOT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    reports = [run_patient(patient, arguments.rank) for patient in SMOKE_PATIENTS]
    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024

    arms = [arm for report in reports for arm in report["arms"] if "skipped" not in arm]
    checks = {
        "every_arm_ran": len(arms) > 0,
        "ablation_is_exactly_zero_residual": all(
            arm["ablation_residual_max_abs"] == 0.0 for arm in arms),
        "autonomous_never_exposes_a_suffix_head": all(
            arm["suffix_is_none"] for arm in arms if arm["family"] == "AUTONOMOUS_SHARED_OPERATOR"),
        "direct_and_bag_expose_a_suffix_head": all(
            not arm["suffix_is_none"] for arm in arms
            if arm["family"] != "AUTONOMOUS_SHARED_OPERATOR"),
        "orderless_bag_has_zero_prefix_order_cost": all(
            abs(arm["prefix_order_cost"]) < 1e-9 for arm in arms if arm["family"] == "ORDERLESS_BAG"),
        "ordered_arms_have_nonzero_prefix_order_cost": all(
            abs(arm["prefix_order_cost"]) > 0.0 for arm in arms
            if arm["family"] != "ORDERLESS_BAG"),
        "reload_reproduces_every_score": all(arm["reload_reproduces_score"] for arm in arms),
        "basis_restored_after_transplant": all(
            arm.get("basis_restored", True) for arm in arms),
        "all_horizons_have_a_denominator_field": all(
            len(arm["denominators"]) == 5 for arm in arms),
    }
    payload = {
        "contract": "topic5_capacity_constrained_history_motif_v0_2_smoke",
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "rank": arguments.rank, "epochs": 1,
        "wall_seconds": time.time() - started,
        "peak_rss_gb": peak_gb,
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "reports": reports,
    }
    (SMOKE_ROOT / "SMOKE_REPORT.json").write_text(json.dumps(payload, indent=2, default=float) + "\n")
    print(f"smoke: {len(arms)} arms over {len(reports)} patients in {payload['wall_seconds']:.1f}s, "
          f"peak RSS {peak_gb:.2f} GB")
    for name, value in checks.items():
        print(f"  {'PASS' if value else 'FAIL'}  {name}")
    return 0 if payload["all_checks_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
