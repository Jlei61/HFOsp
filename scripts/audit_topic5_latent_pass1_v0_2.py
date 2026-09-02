#!/usr/bin/env python3
"""Audit all Topic 5.2 Pass 1 cell artifacts before scientific aggregation."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic5_latent_landscape_v0_2 import atomic_write_json, sha256_file  # noqa: E402
from scripts.run_topic5_latent_pass1_v0_2 import (  # noqa: E402
    ANALYSIS_REVISION,
    OUT,
    SYSTEM,
    cell_dir,
)


FINITE_METRICS = (
    "r2_O", "r2_P", "r2_PF", "r2_PF_null", "delta_PF_minus_P",
    "delta_PF_minus_PF_null", "residual_delta_PF_minus_P", "output_r2_P",
    "output_r2_PF",
)


def main() -> None:
    manifest = pd.read_csv(OUT / "CHECKPOINT_MANIFEST.csv")
    sample_hash = sha256_file(OUT / "PASS1_EVENT_SAMPLE_MANIFEST.csv")
    axis_hash = sha256_file(OUT / "MODE_AXIS_ELIGIBILITY.csv")
    failures: list[dict[str, object]] = []
    counters = {
        "done_missing": 0,
        "done_hash_mismatch": 0,
        "revision_mismatch": 0,
        "target_or_parameter_failure": 0,
        "metric_nonfinite": 0,
        "pca_fraction_out_of_range": 0,
        "array_contract_failure": 0,
        "emergence_nonfinite": 0,
    }
    tier_counts: dict[str, int] = {}
    collinear_cells = 0
    negative_emergence_r2 = 0
    for item in manifest.itertuples(index=False):
        row = pd.Series(item._asdict())
        target = cell_dir(row)
        label = f"{item.fit_id}/{item.public_arm}/seed{item.seed}"
        try:
            done_path = target / "DONE.json"
            if not done_path.is_file():
                counters["done_missing"] += 1
                raise RuntimeError("DONE missing")
            done = json.loads(done_path.read_text())
            expected = {
                "metrics_sha256": sha256_file(target / "metrics.json"),
                "arrays_sha256": sha256_file(target / "geometry_arrays.npz"),
                "emergence_sha256": sha256_file(target / "emergence.csv"),
            }
            if any(done.get(key) != value for key, value in expected.items()):
                counters["done_hash_mismatch"] += 1
                raise RuntimeError("DONE content hash mismatch")
            metrics = json.loads((target / "metrics.json").read_text())
            if metrics.get("analysis_revision") != ANALYSIS_REVISION:
                counters["revision_mismatch"] += 1
                raise RuntimeError("analysis revision mismatch")
            if (
                metrics.get("target_values_read") is not False
                or metrics.get("model_hash_unchanged") is not True
                or metrics.get("decoder_hash_unchanged") is not True
                or metrics.get("sample_manifest_sha256") != sample_hash
                or metrics.get("mode_axis_manifest_sha256") != axis_hash
            ):
                counters["target_or_parameter_failure"] += 1
                raise RuntimeError("target/hash/frozen-parameter contract failed")
            geometry = metrics["heldout_geometry"]
            if not all(np.isfinite(float(geometry[name])) for name in FINITE_METRICS):
                counters["metric_nonfinite"] += 1
                raise RuntimeError("heldout geometry contains nonfinite values")
            fraction = float(metrics["pca"]["variance_fraction_top8"])
            if not (0.0 <= fraction <= 1.0 + 1e-6):
                counters["pca_fraction_out_of_range"] += 1
                raise RuntimeError(f"PCA fraction out of range: {fraction}")
            with np.load(target / "geometry_arrays.npz", allow_pickle=False) as arrays:
                required = {
                    "robust_center", "robust_scale", "pca_components", "phase_grid",
                    "gamma_raw", "progress_axes_raw", "field_axes_raw",
                    "local_residual_eigenvalues", "local_residual_components",
                    "local_residual_diagonal", "contact_future_field_axis",
                }
                if not required.issubset(arrays.files):
                    raise RuntimeError("geometry arrays missing required fields")
                if not (
                    np.isfinite(arrays["robust_center"]).all()
                    and np.isfinite(arrays["robust_scale"]).all()
                    and (arrays["robust_scale"] > 0).all()
                    and np.isfinite(arrays["gamma_raw"]).all()
                    and np.isfinite(arrays["contact_future_field_axis"]).all()
                ):
                    raise RuntimeError("core geometry arrays are nonfinite")
            emergence = pd.read_csv(target / "emergence.csv")
            emergence_columns = ["r2_h", "r2_o", "r2_oh", "incremental_r2_oh_minus_o"]
            if len(emergence) != 5 or not np.isfinite(emergence[emergence_columns]).all().all():
                counters["emergence_nonfinite"] += 1
                raise RuntimeError("emergence curve invalid")
            negative_emergence_r2 += int((emergence[["r2_h", "r2_oh"]] < 0).sum().sum())
            tier = str(metrics["field_axis_tier"])
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
            collinear_cells += int(metrics["axis"]["field_collinear_grid_points"] > 0)
        except Exception as error:
            if "geometry arrays" in str(error) or "core geometry" in str(error):
                counters["array_contract_failure"] += 1
            failures.append({"cell": label, "error": f"{type(error).__name__}: {error}"})
    complete = len(manifest) == 630 and not failures
    payload = {
        "contract": "topic5_latent_pass1_audit_v0_2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_revision": ANALYSIS_REVISION,
        "status": "PASS" if complete else "FAIL",
        "audited_cells": int(len(manifest)),
        "expected_cells": 630,
        "failure_counters": counters,
        "failures": failures[:50],
        "field_axis_tier_counts": tier_counts,
        "cells_with_field_collinearity": collinear_cells,
        "negative_heldout_emergence_r2_values_retained": negative_emergence_r2,
        "negative_r2_policy": "RETAINED_NOT_CLIPPED",
        "target_values_read": False,
    }
    atomic_write_json(SYSTEM / "PASS1_AUDIT.json", payload)
    print(json.dumps(payload, indent=2))
    if not complete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
