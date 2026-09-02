#!/usr/bin/env python3
"""Phase F4: what a frozen model actually uses at test time.

Three separate questions, deliberately named apart:

``PREFIX_ORDER_COST``
    reorder the middle of the observed prefix while holding the start set, the
    cumulative set, the prefix length and the cardinality fixed.  Both unordered
    baselines are provably unchanged, so any cost is attributable to the ordered
    path.
``ORDERED_PATH_ABLATION_COST``
    set the low-dimensional state to zero, leaving the unordered baseline, the
    availability mask, the checkpoint and the temperature untouched.
``BASIS_TRANSPLANT_COST``
    swap the frozen spatial dictionary between the aligned model and the
    patient-median angle-rotated null with no retraining and no recalibration.
    This is a subspace-specificity probe; it is never a runtime lesion and never
    an online-necessity claim.

STOP is fitted here, after the spatial models are frozen, and is reported on its
own.  A STOP result may not be used to rescue a spatial result.
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
import hashlib
import json
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic5_strict_history_motif_v0_2 import (  # noqa: E402
    MotifConfig,
    OrderedMotif,
    checkpoint_objective,
    combine_logits,
    evaluate,
    perturb_prefix_order,
)
from scripts.run_topic5_capacity_queue_v0_2 import PatientWorkspace  # noqa: E402

RESULT_ROOT = ROOT / "results/topic5_capacity_constrained_history_motif_v0_2"
STOP_EPOCHS = 300
STOP_LR = 0.05


def load_model(workspace: PatientWorkspace, unit: dict) -> OrderedMotif:
    free = unit["structure"] == "H1_FREE_LOW_RANK"
    config = MotifConfig(
        structure=unit["structure"], family=unit["family"], rank=int(unit["rank"]),
        n_contacts=workspace.samples(int(unit["prefix_len"])).n_contacts,
        n_horizons=len(workspace.tensors(int(unit["prefix_len"])).target_valid[0]),
        max_cardinality=workspace.samples(int(unit["prefix_len"])).max_cardinality,
        f_form=unit["f_form"], free_basis=free,
    )
    model = OrderedMotif(config, None if free else workspace.basis(unit["basis_key"]))
    payload = torch.load(RESULT_ROOT / unit["output_dir"] / "checkpoint.pt", weights_only=True)
    prefix = "0."
    state = {key[len(prefix):]: value for key, value in payload["state_dict"].items()
             if key.startswith(prefix)}
    model.load_state_dict(state)
    model.eval()
    return model


def scalar_summary(result) -> dict:
    kind = result.scalars["primary_field_kind"]
    return {
        "total_nll_h1": result.per_horizon["total_nll"][0],
        "total_nll_h2": result.per_horizon["total_nll"][1],
        "total_nll_h3": result.per_horizon["total_nll"][2],
        "total_nll_h4": result.per_horizon["total_nll"][3],
        "total_nll_h5": result.per_horizon["total_nll"][4],
        "suffix_balanced_bce": result.scalars[f"{kind}_balanced_bce"],
        "suffix_balanced_brier": result.scalars[f"{kind}_balanced_brier"],
        "endpoint_distance_mm": result.scalars[f"{kind}_endpoint_distance_mm"],
        "primary_field_kind": kind,
    }


def fit_stop_head(model: OrderedMotif, workspace: PatientWorkspace, unit: dict) -> dict:
    """``p(STOP_{t+h} | z_t, t, |S_t|)`` on the frozen spatial state."""
    prefix_len = int(unit["prefix_len"])
    batch = workspace.tensors(prefix_len)
    train = torch.as_tensor(np.flatnonzero(workspace.fraction_mask(prefix_len, int(unit["data_fraction"]))))
    valid = torch.as_tensor(np.flatnonzero(workspace.split_mask(prefix_len, 1)))
    test = torch.as_tensor(np.flatnonzero(workspace.split_mask(prefix_len, 2)))
    with torch.no_grad():
        state = model.prefix_state(batch)
    features = torch.cat([
        state,
        torch.full((batch.n_samples, 1), float(prefix_len)),
        batch.cumulative_set.sum(dim=1, keepdim=True) / float(batch.n_contacts),
    ], dim=1)
    target = (~batch.target_valid).float()
    head = torch.nn.Linear(features.shape[1], batch.n_horizons)
    optimiser = torch.optim.Adam(head.parameters(), lr=STOP_LR)
    best, best_state = float("inf"), None
    for _ in range(STOP_EPOCHS):
        optimiser.zero_grad(set_to_none=True)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            head(features[train]), target[train])
        loss.backward()
        optimiser.step()
        with torch.no_grad():
            score = float(torch.nn.functional.binary_cross_entropy_with_logits(
                head(features[valid]), target[valid]))
        if score < best - 1e-9:
            best, best_state = score, {k: v.clone() for k, v in head.state_dict().items()}
    if best_state is not None:
        head.load_state_dict(best_state)
    with torch.no_grad():
        logits = head(features[test])
        probability = torch.sigmoid(logits)
        truth = target[test]
        per_horizon = []
        for horizon in range(batch.n_horizons):
            positives = float(truth[:, horizon].sum())
            per_horizon.append({
                "bce": float(torch.nn.functional.binary_cross_entropy_with_logits(
                    logits[:, horizon], truth[:, horizon])),
                "brier": float(((probability[:, horizon] - truth[:, horizon]) ** 2).mean()),
                "positive_rate": positives / max(1.0, float(truth.shape[0])),
                "denominator": int(truth.shape[0]),
            })
    return {"validation_bce": best, "per_horizon": per_horizon,
            "state_dict": {k: v.detach().clone() for k, v in head.state_dict().items()},
            "note": "fitted after the spatial checkpoint was frozen; never enters L_space"}


def _stop_under_basis(model: OrderedMotif, head: torch.nn.Module, workspace: PatientWorkspace,
                      unit: dict, basis: np.ndarray) -> float:
    prefix_len = int(unit["prefix_len"])
    batch = workspace.tensors(prefix_len)
    test = torch.as_tensor(np.flatnonzero(workspace.split_mask(prefix_len, 2)))
    piece = batch.index(test)
    original = model.basis.detach().clone()
    with torch.no_grad():
        model.basis.copy_(torch.as_tensor(basis, dtype=torch.float32))
        state = model.prefix_state(piece)
        features = torch.cat([
            state,
            torch.full((piece.n_samples, 1), float(prefix_len)),
            piece.cumulative_set.sum(dim=1, keepdim=True) / float(piece.n_contacts),
        ], dim=1)
        score = float(torch.nn.functional.binary_cross_entropy_with_logits(
            head(features), (~piece.target_valid).float()))
        model.basis.copy_(original)
    return score


def process_patient(payload: dict) -> dict:
    torch.set_num_threads(int(os.environ.get("TOPIC5_TORCH_THREADS", "2")))
    patient, units = payload["patient"], payload["units"]
    workspace = PatientWorkspace(patient)
    rows, errors = [], []
    for unit in units:
        directory = RESULT_ROOT / unit["output_dir"]
        if not (directory / "checkpoint.pt").exists():
            continue
        try:
            model = load_model(workspace, unit)
            prefix_len = int(unit["prefix_len"])
            batch = workspace.tensors(prefix_len)
            baseline = workspace.baseline(unit["baseline_level"], prefix_len)
            test = torch.as_tensor(np.flatnonzero(workspace.split_mask(prefix_len, 2)))
            piece = batch.index(test)
            base = {key: value[test] for key, value in baseline.items()}

            intact = evaluate(model, base, piece, workspace.contact_xy)
            ablated = evaluate(model, base, piece, workspace.contact_xy, ordered_path=False)
            permuted_piece = perturb_prefix_order(piece, "swap_middle")
            permuted = evaluate(model, base, permuted_piece, workspace.contact_xy)

            summary = {"intact": scalar_summary(intact), "ablated": scalar_summary(ablated),
                       "permuted": scalar_summary(permuted)}
            record = {
                "patient": patient, "unit_id": unit["unit_id"], "block": unit["block"],
                "structure": unit["structure"], "null_id": unit["null_id"],
                "family": unit["family"], "baseline_level": unit["baseline_level"],
                "rank": int(unit["rank"]), "seed": int(unit["seed"]),
                "checkpoint_sha256_before": hashlib.sha256(
                    (directory / "checkpoint.pt").read_bytes()).hexdigest(),
            }
            for key in ("total_nll_h1", "total_nll_h2", "total_nll_h3",
                        "suffix_balanced_bce", "endpoint_distance_mm"):
                record[f"intact_{key}"] = summary["intact"][key]
                record[f"prefix_order_cost_{key}"] = summary["permuted"][key] - summary["intact"][key]
                record[f"ordered_path_ablation_cost_{key}"] = (
                    summary["ablated"][key] - summary["intact"][key]
                )
            stop = fit_stop_head(model, workspace, unit)
            record["stop_validation_bce"] = stop["validation_bce"]
            for horizon, entry in enumerate(stop["per_horizon"][:3], start=1):
                record[f"stop_bce_h{horizon}"] = entry["bce"]
                record[f"stop_positive_rate_h{horizon}"] = entry["positive_rate"]
            record["checkpoint_sha256_after"] = hashlib.sha256(
                (directory / "checkpoint.pt").read_bytes()).hexdigest()
            record["checkpoint_unchanged"] = (
                record["checkpoint_sha256_before"] == record["checkpoint_sha256_after"]
            )
            (directory / "use_phase.json").write_text(json.dumps(
                {"summary": summary,
                 "stop": {k: v for k, v in stop.items() if k != "state_dict"}},
                indent=2, default=float) + "\n")
            rows.append(record)
        except Exception:
            errors.append({"unit_id": unit["unit_id"], "error": traceback.format_exc()})
    return {"patient": patient, "rows": rows, "errors": errors}


def transplant_for_patient(payload: dict) -> dict:
    """2x2 aligned/null train-test basis transplant with no parameter update."""
    torch.set_num_threads(int(os.environ.get("TOPIC5_TORCH_THREADS", "2")))
    patient, aligned_unit, null_unit = payload["patient"], payload["aligned"], payload["null"]
    workspace = PatientWorkspace(patient)
    prefix_len = int(aligned_unit["prefix_len"])
    batch = workspace.tensors(prefix_len)
    baseline = workspace.baseline(aligned_unit["baseline_level"], prefix_len)
    test = torch.as_tensor(np.flatnonzero(workspace.split_mask(prefix_len, 2)))
    piece = batch.index(test)
    base = {key: value[test] for key, value in baseline.items()}

    models, digests = {}, {}
    for label, unit in (("A", aligned_unit), ("N", null_unit)):
        models[label] = load_model(workspace, unit)
        digests[label] = hashlib.sha256(
            (RESULT_ROOT / unit["output_dir"] / "checkpoint.pt").read_bytes()).hexdigest()
    bases = {"A": workspace.basis(aligned_unit["basis_key"]),
             "N": workspace.basis(null_unit["basis_key"])}

    scores, logit_norms = {}, {}
    for train_label in ("A", "N"):
        for test_label in ("A", "N"):
            model = models[train_label]
            original = model.basis.detach().clone()
            with torch.no_grad():
                model.basis.copy_(torch.as_tensor(bases[test_label], dtype=torch.float32))
                residual = model(piece)
                logit_norms[f"{train_label}{test_label}"] = float(
                    residual["contact"].norm(dim=2).mean())
            result = evaluate(model, base, piece, workspace.contact_xy)
            scores[f"{train_label}{test_label}"] = scalar_summary(result)
            with torch.no_grad():
                model.basis.copy_(original)

    stop_intact = fit_stop_head(models["A"], workspace, aligned_unit)
    stop_head = torch.nn.Linear(int(aligned_unit["rank"]) + 2, batch.n_horizons)
    stop_head.load_state_dict(stop_intact["state_dict"])
    stop_change = _stop_under_basis(models["A"], stop_head, workspace, aligned_unit,
                                    bases["N"]) - _stop_under_basis(
        models["A"], stop_head, workspace, aligned_unit, bases["A"])

    def value(cell: str, key: str = "suffix_balanced_bce") -> float:
        return scores[cell][key]

    after = {label: hashlib.sha256(
        (RESULT_ROOT / unit["output_dir"] / "checkpoint.pt").read_bytes()).hexdigest()
        for label, unit in (("A", aligned_unit), ("N", null_unit))}
    row = {
        "patient": patient,
        "aligned_unit_id": aligned_unit["unit_id"],
        "null_unit_id": null_unit["unit_id"],
        "null_id": null_unit["null_id"],
        "delta_test_given_A": value("AN") - value("AA"),
        "delta_test_given_N": value("NN") - value("NA"),
        "delta_train_given_A": value("NA") - value("AA"),
        "delta_train_given_N": value("NN") - value("AN"),
        "transplant_interaction": (value("AN") - value("AA")) - (value("NN") - value("NA")),
        "immediate_logit_norm_change": logit_norms["AN"] - logit_norms["AA"],
        "future_field_change": value("AN", "endpoint_distance_mm") - value("AA", "endpoint_distance_mm"),
        "stop_change": stop_change,
        "optimizer_called_on_spatial_model": False,
        "stop_head_refit_note": "STOP head fitted once on the intact model, then evaluated "
                                "under the transplanted basis; no spatial parameter changed",
        "checkpoints_unchanged": bool(digests == after),
        **{f"cell_{cell}_suffix_balanced_bce": value(cell) for cell in ("AA", "AN", "NA", "NN")},
        **{f"cell_{cell}_total_nll_h1": value(cell, "total_nll_h1") for cell in ("AA", "AN", "NA", "NN")},
    }
    return {"patient": patient, "row": row}


def median_angle_null(manifest: pd.DataFrame, patient: str, family: str, level: str) -> dict | None:
    """The angle null whose held-out score is the patient's median angle null."""
    candidates = manifest[
        (manifest["patient"] == patient) & (manifest["structure"] == "H1_ANGLE_ROTATED_AXIS")
        & (manifest["family"] == family) & (manifest["baseline_level"] == level)
        & (manifest["block"] == "CORE1") & manifest["eligible"]
    ]
    scored = []
    for unit in candidates.to_dict("records"):
        path = RESULT_ROOT / unit["output_dir"] / "metrics.json"
        if not path.exists():
            continue
        metrics = json.loads(path.read_text())
        kind = metrics["metrics"]["calibration"]["scalars"]["primary_field_kind"]
        scored.append((metrics["metrics"]["calibration"]["scalars"][f"{kind}_balanced_bce"], unit))
    if not scored:
        return None
    scored.sort(key=lambda item: item[0])
    return scored[len(scored) // 2][1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--blocks", default="CORE1,CORE2")
    parser.add_argument("--patients", default="")
    arguments = parser.parse_args()

    manifest = pd.read_csv(RESULT_ROOT / "MASTER_UNIT_MANIFEST.csv")
    manifest = manifest[manifest["eligible"]]
    selected = manifest[manifest["block"].isin(arguments.blocks.split(","))]
    if arguments.patients:
        selected = selected[selected["patient"].isin(arguments.patients.split(","))]

    payloads = [{"patient": patient, "units": group.to_dict("records")}
                for patient, group in selected.groupby("patient")]
    with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
        outcomes = list(pool.map(process_patient, payloads))
    rows = [row for outcome in outcomes for row in outcome["rows"]]
    errors = [error for outcome in outcomes for error in outcome["errors"]]
    pd.DataFrame(rows).to_csv(RESULT_ROOT / "PER_PATIENT_ORDER_AND_PATH_ABLATION.csv", index=False)

    transplant_jobs = []
    for patient in sorted(selected["patient"].unique()):
        aligned = manifest[
            (manifest["patient"] == patient) & (manifest["structure"] == "H1_PATIENT_ALIGNED")
            & (manifest["family"] == "AUTONOMOUS_SHARED_OPERATOR") & (manifest["block"] == "CORE1")
            & (manifest["seed"] == 0)
        ]
        null = median_angle_null(manifest, patient, "AUTONOMOUS_SHARED_OPERATOR", "U_FULL_SET")
        if aligned.empty or null is None:
            continue
        if not (RESULT_ROOT / aligned.iloc[0]["output_dir"] / "checkpoint.pt").exists():
            continue
        transplant_jobs.append({"patient": patient, "aligned": aligned.iloc[0].to_dict(), "null": null})
    with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
        transplants = list(pool.map(transplant_for_patient, transplant_jobs))
    pd.DataFrame([entry["row"] for entry in transplants]).to_csv(
        RESULT_ROOT / "PER_PATIENT_BASIS_TRANSPLANT.csv", index=False)

    audit = {
        "contract": "topic5_capacity_constrained_history_motif_v0_2_use_phase",
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "n_units_scored": len(rows),
        "n_units_failed": len(errors),
        "all_checkpoints_unchanged": bool(all(row["checkpoint_unchanged"] for row in rows)),
        "transplant_patients": len(transplants),
        "transplant_checkpoints_unchanged": bool(
            all(entry["row"]["checkpoints_unchanged"] for entry in transplants)),
        "naming": {
            "seeg": "BASIS_TRANSPLANT_COST — subspace specificity, never a runtime lesion",
            "ecog": "RUNTIME_GRAPH_SWAP — reported separately and never pooled with SEEG",
        },
        "errors": errors[:20],
    }
    (RESULT_ROOT / "USE_PHASE_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(f"use-phase units scored: {len(rows)}  failed: {len(errors)}")
    print(f"transplant pairs: {len(transplants)}")
    print(f"checkpoints unchanged: {audit['all_checkpoints_unchanged']} / "
          f"{audit['transplant_checkpoints_unchanged']}")
    for error in errors[:3]:
        print(error["error"].splitlines()[-1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
