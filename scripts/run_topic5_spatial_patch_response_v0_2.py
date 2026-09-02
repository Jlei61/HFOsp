#!/usr/bin/env python3
"""Run exact-N0 Gaussian tissue-patch central differences on frozen references."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic5_latent_landscape_v0_2 import (  # noqa: E402
    atomic_write_csv, atomic_write_json, load_frozen_cell, parameter_state_sha256, sha256_file,
)
from src.topic5_latent_perturbation_v0_2 import DOSES  # noqa: E402
from src.topic5_latent_response_v0_2 import project_centered_contact_response  # noqa: E402
from scripts.freeze_topic5_cross_patient_geometry_mappings_v0_2 import SPATIAL  # noqa: E402
from scripts.freeze_topic5_latent_reference_states_v0_2 import reference_dir  # noqa: E402
from scripts.freeze_topic5_spatial_patch_contract_v0_2 import (  # noqa: E402
    PATCH, PATCH_FREEZE_REVISION, patch_dir,
)
from scripts.run_topic5_axis_perturbations_v0_2 import HORIZON, open_loop_pair  # noqa: E402
from scripts.run_topic5_latent_pass1_v0_2 import OUT, PARENT  # noqa: E402


RESPONSE = SPATIAL / "patch_response"
OPERATOR = SPATIAL / "patch_operator"
PATCH_RESPONSE_REVISION = "PATCH_RESPONSE_R0_CENTRAL_OPEN_LOOP_PREMASK"
# The R0 stage kept only the two scalar projections of each patch response.  R1 keeps
# the full future-contact response vector so a tissue-node -> future-contact operator
# can be built without any fitted hidden axis.  Both revisions share the same frozen
# reference states, patch contract, doses and support gate, so R1 must reproduce the
# R0 projections exactly; that identity is checked per cell.
PATCH_OPERATOR_REVISION = "PATCH_OPERATOR_R1_FULL_FUTURE_CONTACT_RESPONSE"
BATCH_PAIRS = 4096
AXIS_NAMES = ("PROGRESS", "FIELD")


def response_dir(row: pd.Series) -> Path:
    return RESPONSE / "per_cell" / str(row.fit_id) / str(row.public_arm) / f"seed{int(row.seed)}"


def operator_dir(row: pd.Series) -> Path:
    return OPERATOR / "per_cell" / str(row.fit_id) / str(row.public_arm) / f"seed{int(row.seed)}"


def write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream: np.savez_compressed(stream, **arrays)
    temporary.replace(path)


@torch.no_grad()
def run_cell(
    row: pd.Series, device: torch.device, *, collect_operator: bool = False
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    started = time.perf_counter()
    model, decoder, _, _ = load_frozen_cell(PARENT, row, device)
    model_hash, decoder_hash = parameter_state_sha256(model), parameter_state_sha256(decoder)
    with np.load(reference_dir(row) / "reference_contract.npz", allow_pickle=False) as source:
        q = {name: np.asarray(source[name]) for name in source.files}
    with np.load(patch_dir(row) / "patch_contract.npz", allow_pickle=False) as source:
        patch = {name: np.asarray(source[name]) for name in source.files}
    cache = PARENT / "cache" / str(row.fit_id)
    with np.load(cache / "events.npz", allow_pickle=False) as source: ranks = np.asarray(source["ranks"])
    h = q["hidden"].astype(float); recruited = q["recruited"].astype(np.uint8)
    step = q["step"].astype(int); event = q["reference_event_index"].astype(int)
    phase_target = q["phase_target"].astype(float); phase_values = np.sort(np.unique(phase_target))
    phase_index = np.searchsorted(phase_values, phase_target)
    directions = patch["patch_directions"].astype(float); local_sd = patch["patch_local_sd"].astype(float)
    support = patch["support_checks"].astype(bool)
    output_progress = q["contact_progress_axis"].astype(float); output_field = q["contact_future_field_axis"].astype(float)
    n_phase, n_center, n_dose = len(phase_values), len(directions), len(DOSES)
    sum_scores = np.zeros((n_phase, n_center, n_dose, HORIZON + 1, 2), np.float64)
    sum_norm = np.zeros((n_phase, n_center, n_dose, HORIZON + 1), np.float64)
    positive = np.zeros_like(sum_scores, dtype=np.int32)
    negative = np.zeros_like(sum_scores, dtype=np.int32)
    counts = np.zeros((n_phase, n_center, n_dose, HORIZON + 1), np.int32)
    eligible_pairs = np.zeros((n_phase, n_center, n_dose), np.int32)
    n_contacts = recruited.shape[1]
    # Split halves are keyed by reference-state index parity so the split is
    # deterministic, response-blind, and identical across arms and seeds.
    sum_contact = np.zeros((n_phase, n_center, n_dose, HORIZON + 1, n_contacts), np.float64) if collect_operator else None
    half_contact = np.zeros((2, n_phase, n_center, n_dose, HORIZON + 1, n_contacts), np.float64) if collect_operator else None
    half_counts = np.zeros((2, n_phase, n_center, n_dose, HORIZON + 1), np.int32) if collect_operator else None
    for dose_index, dose in enumerate(DOSES):
        eligible = support[:, :, dose_index].all(axis=(2, 3))
        eligible &= np.isfinite(local_sd) & (local_sd > 1e-8)
        reference_indices, center_indices = np.where(eligible)
        for start in range(0, len(reference_indices), BATCH_PAIRS):
            take = slice(start, min(start + BATCH_PAIRS, len(reference_indices)))
            ref = reference_indices[take]; center = center_indices[take]
            magnitude = float(dose) * local_sd[ref, center]
            delta = directions[center] * magnitude[:, None]
            result = open_loop_pair(
                model, h[ref] - delta, h[ref] + delta, recruited[ref], step[ref], event[ref], ranks,
                2.0 * magnitude, output_progress, output_field, device,
            )
            phases = phase_index[ref]
            for local in range(len(ref)):
                p, c = phases[local], center[local]
                valid = result["valid"][local].astype(bool)
                sum_scores[p, c, dose_index, valid] += result["scores"][local, valid]
                sum_norm[p, c, dose_index, valid] += result["logit_response_norm"][local, valid]
                positive[p, c, dose_index, valid] += (result["scores"][local, valid] > 0)
                negative[p, c, dose_index, valid] += (result["scores"][local, valid] < 0)
                counts[p, c, dose_index, valid] += 1
                eligible_pairs[p, c, dose_index] += 1
                if collect_operator:
                    half = int(ref[local]) % 2
                    sum_contact[p, c, dose_index, valid] += result["contact_response"][local, valid]
                    half_contact[half, p, c, dose_index, valid] += result["contact_response"][local, valid]
                    half_counts[half, p, c, dose_index, valid] += 1
    mean_scores = np.divide(
        sum_scores, counts[..., None], out=np.full_like(sum_scores, np.nan), where=counts[..., None] > 0,
    )
    mean_norm = np.divide(sum_norm, counts, out=np.full_like(sum_norm, np.nan), where=counts > 0)
    sign_fraction = np.divide(
        positive, positive + negative, out=np.full_like(sum_scores, np.nan), where=(positive + negative) > 0,
    )
    arrays = {
        "axis_names": np.asarray(AXIS_NAMES), "phase_targets": phase_values.astype(np.float32),
        "node_xy_mm": patch["node_xy_mm"].astype(np.float32), "doses": DOSES.astype(np.float32),
        "mean_scores": mean_scores.astype(np.float32), "mean_logit_response_norm": mean_norm.astype(np.float32),
        "positive_sign_fraction": sign_fraction.astype(np.float32), "valid_counts": counts,
        "eligible_pairs": eligible_pairs,
    }
    projection_error = None
    if collect_operator:
        mean_contact = np.divide(
            sum_contact, counts[..., None], out=np.full_like(sum_contact, np.nan), where=counts[..., None] > 0,
        )
        half_mean = np.divide(
            half_contact, half_counts[..., None],
            out=np.full_like(half_contact, np.nan), where=half_counts[..., None] > 0,
        )
        # The projection is linear, so projecting the mean response must reproduce the
        # frozen R0 mean_scores exactly; this is the faithfulness check for the re-run.
        reprojected = project_centered_contact_response(mean_contact, output_progress, output_field)
        both = np.isfinite(reprojected) & np.isfinite(mean_scores)
        projection_error = float(np.max(np.abs(reprojected[both] - mean_scores[both]))) if both.any() else float("nan")
        arrays = {
            **{key: arrays[key] for key in ("axis_names", "phase_targets", "node_xy_mm", "doses")},
            "mean_contact_operator": mean_contact.astype(np.float32),
            "half_contact_operator": half_mean.astype(np.float32),
            "valid_counts": counts, "half_valid_counts": half_counts,
            "contact_progress_axis": output_progress.astype(np.float32),
            "contact_future_field_axis": output_field.astype(np.float32),
        }
    metrics = {
        "contract": "topic5_spatial_patch_response_cell_v0_2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "patch_freeze_revision": PATCH_FREEZE_REVISION, "patch_response_revision": PATCH_RESPONSE_REVISION,
        "status": "PASS", "patient": str(row.patient), "fit_id": str(row.fit_id),
        "public_arm": str(row.public_arm), "seed": int(row.seed),
        "n_reference_states": len(h), "n_patch_centers": n_center,
        "eligible_state_center_dose_pairs": int(eligible_pairs.sum()),
        "finite_state_center_dose_tau": int(counts.sum()),
        "model_hash_unchanged": model_hash == parameter_state_sha256(model),
        "decoder_hash_unchanged": decoder_hash == parameter_state_sha256(decoder),
        "reference_contract_sha256": sha256_file(reference_dir(row) / "reference_contract.npz"),
        "patch_contract_sha256": sha256_file(patch_dir(row) / "patch_contract.npz"),
        "n0_policy": "BOTH_SIGNS_PASS_NODE_KNN_MANIFOLD_NO_CLIP_NO_RESCALE",
        "target_values_read": False, "elapsed_seconds": time.perf_counter() - started,
    }
    if collect_operator:
        metrics["patch_operator_revision"] = PATCH_OPERATOR_REVISION
        metrics["max_abs_projection_error_vs_R0"] = projection_error
        metrics["n_contacts"] = int(n_contacts)
    return arrays, metrics


def write_cell(
    row: pd.Series, arrays: dict[str, np.ndarray], metrics: dict[str, object], *, operator: bool = False
) -> None:
    target = operator_dir(row) if operator else response_dir(row)
    payload = "patch_operator.npz" if operator else "patch_response.npz"
    target.mkdir(parents=True, exist_ok=True)
    write_npz(target / payload, arrays); atomic_write_json(target / "metrics.json", metrics)
    atomic_write_json(target / "DONE.json", {
        "ok": True, "patch_response_revision": PATCH_RESPONSE_REVISION,
        **({"patch_operator_revision": PATCH_OPERATOR_REVISION} if operator else {}),
        "response_sha256": sha256_file(target / payload),
        "metrics_sha256": sha256_file(target / "metrics.json"), "target_values_read": False,
    })


def aggregate(manifest: pd.DataFrame, *, operator: bool = False) -> dict[str, object]:
    root = OPERATOR if operator else RESPONSE
    rows, missing = [], []
    for item in manifest.itertuples(index=False):
        row = pd.Series(item._asdict()); target = operator_dir(row) if operator else response_dir(row)
        if not (target / "DONE.json").is_file(): missing.append(f"{item.fit_id}/{item.public_arm}/seed{item.seed}")
        else: rows.append(json.loads((target / "metrics.json").read_text()))
    if rows:
        atomic_write_csv(
            root / ("PATCH_OPERATOR_CELL_SUMMARY.csv" if operator else "PATCH_RESPONSE_CELL_SUMMARY.csv"),
            pd.DataFrame(rows),
        )
    payload = {
        "contract": "topic5_spatial_patch_operator_execution_v0_2" if operator
        else "topic5_spatial_patch_response_execution_v0_2",
        "created_utc": datetime.now(timezone.utc).isoformat(), "patch_response_revision": PATCH_RESPONSE_REVISION,
        "status": "PASS" if len(rows) == 630 and not missing else "INCOMPLETE",
        "scheduled_cells": 630, "completed_cells": len(rows),
        "eligible_state_center_dose_pairs": int(sum(row["eligible_state_center_dose_pairs"] for row in rows)),
        "finite_state_center_dose_tau": int(sum(row["finite_state_center_dose_tau"] for row in rows)),
        "missing_count": len(missing), "missing_first20": missing[:20], "target_values_read": False,
    }
    if operator:
        errors = [row["max_abs_projection_error_vs_R0"] for row in rows if row.get("max_abs_projection_error_vs_R0") is not None]
        payload["patch_operator_revision"] = PATCH_OPERATOR_REVISION
        payload["max_abs_projection_error_vs_R0"] = float(max(errors)) if errors else None
        payload["projection_identity_note"] = (
            "Projecting the R1 mean contact operator onto the frozen train-only contact axes must "
            "reproduce the R0 mean_scores; this is the re-run faithfulness check."
        )
    atomic_write_json(
        root / ("PATCH_OPERATOR_STATUS.json" if operator else "PATCH_RESPONSE_STATUS.json"), payload
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--device", default="cuda")
    parser.add_argument("--cell-key"); parser.add_argument("--limit", type=int); parser.add_argument("--force", action="store_true")
    parser.add_argument("--operator", action="store_true", help="also keep the full future-contact response operator")
    args = parser.parse_args()
    freeze = json.loads((PATCH / "PATCH_FREEZE_STATUS.json").read_text())
    if not args.limit and not args.cell_key and freeze.get("status") != "PASS": raise RuntimeError("full patch freeze must pass")
    manifest_all = pd.read_csv(OUT / "CHECKPOINT_MANIFEST.csv"); manifest = manifest_all.copy()
    if args.cell_key:
        fit, arm, seed_text = args.cell_key.split("/"); manifest = manifest[
            manifest.fit_id.eq(fit) & manifest.public_arm.eq(arm) & manifest.seed.eq(int(seed_text.removeprefix("seed")))
        ]
    elif args.limit is not None: manifest = manifest.iloc[:args.limit]
    device = torch.device(args.device); failures = []
    for position, (_, row) in enumerate(manifest.iterrows(), start=1):
        target = operator_dir(row) if args.operator else response_dir(row)
        if (target / "DONE.json").is_file() and not args.force: continue
        try:
            arrays, metrics = run_cell(row, device, collect_operator=args.operator)
            write_cell(row, arrays, metrics, operator=args.operator)
            print(f"done {position}/{len(manifest)} {row.fit_id}/{row.public_arm}/seed{row.seed} pairs={metrics['eligible_state_center_dose_pairs']} {metrics['elapsed_seconds']:.2f}s", flush=True)
        except Exception as error:
            failures.append({"fit_id": row.fit_id, "public_arm": row.public_arm, "seed": int(row.seed), "error_type": type(error).__name__, "error": str(error)})
            atomic_write_json(target / "FAILURE.json", failures[-1]); print(f"FAIL {row.fit_id}/{row.public_arm}/seed{row.seed}: {error}", flush=True)
    status = aggregate(manifest_all, operator=args.operator); print(json.dumps({"run_failures": failures, "aggregate": status}, indent=2))
    if failures: raise SystemExit(1)


if __name__ == "__main__": main()
