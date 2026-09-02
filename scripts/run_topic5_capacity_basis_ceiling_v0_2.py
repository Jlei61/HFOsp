#!/usr/bin/env python3
"""Phase C0: can the candidate bases represent the held-out residual at all?

This runs before the ordered models are interpreted and never selects anything.
For each frozen basis it asks the purely representational question: if the
coefficients were allowed to be re-optimised for every single held-out field,
how much of the residual left by the unordered baseline could that basis span?

Because the coefficients are free per field, the answer is a ceiling, not a
deployable predictor.  Reading it:

* aligned ceiling no better than the nulls  -> the basis itself has no
  representational advantage;
* ceiling good but trained model flat       -> the problem is the state input,
  the shared dynamics or the optimisation;
* ceiling and orderless bag both good with no ordered gain -> the evidence is a
  suffix dictionary, not an ordered motif.
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
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic5_strict_history_data_v0_2 import PRIMARY_PREFIX_LEN, load_sample_set  # noqa: E402
from src.topic5_strict_history_motif_v0_2 import (  # noqa: E402
    BASELINE_LEVELS,
    autonomous_suffix_field,
    tensors_from_samples,
)
from src.topic5_structural_identifiability_v0_2 import (  # noqa: E402
    effective_rank,
    load_basis_bundle,
    masked_projection_residual,
    orthonormal_truncation,
    principal_angles,
)

RESULT_ROOT = ROOT / "results/topic5_capacity_constrained_history_motif_v0_2"
CEILING_RANK = 4
CEILING_KINDS = ("GEOMETRY_LAYOUT", "SHAFT_GRADIENT", "PATIENT_ALIGNED",
                 "ANGLE_ROTATED_AXIS", "IDENTITY_PERMUTED")


def residual_fields(samples, batch, logits, kind: str) -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        if kind == "suffix5":
            predicted = autonomous_suffix_field(logits["contact"], logits["cardinality"], batch)
            truth = batch.suffix5_field
        else:
            predicted = torch.sigmoid(logits["suffix"])
            truth = batch.full_suffix_field
    return (truth - predicted).numpy(), batch.suffix_eval_mask.numpy()


def process(patient: str) -> list[dict]:
    torch.set_num_threads(int(os.environ.get("TOPIC5_TORCH_THREADS", "2")))
    samples = load_sample_set(
        RESULT_ROOT / "sample_cache" / f"prefix{PRIMARY_PREFIX_LEN}" / f"{patient}.npz")
    bases, index = load_basis_bundle(RESULT_ROOT / "basis" / "per_patient" / f"{patient}.npz")
    available = {entry["key"] for entry in index}
    rows: list[dict] = []
    for level in BASELINE_LEVELS:
        path = (RESULT_ROOT / "baseline" / level / f"prefix{PRIMARY_PREFIX_LEN}" / patient / "logits.npz")
        if not path.exists():
            continue
        payload = np.load(path, allow_pickle=False)
        residuals: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]] = {}
        for split_value in (0, 2):
            rows_index = np.flatnonzero(samples.split == split_value)
            if rows_index.size == 0:
                continue
            batch = tensors_from_samples(samples, rows_index)
            logits = {name: torch.as_tensor(payload[name][rows_index], dtype=torch.float32)
                      for name in ("contact", "cardinality", "suffix")}
            for field_kind in ("suffix5", "full_suffix"):
                residuals[(split_value, field_kind)] = residual_fields(
                    samples, batch, logits, field_kind)

        aligned_basis = bases.get(f"PATIENT_ALIGNED|observed|f100|r{CEILING_RANK}")
        for field_kind in ("suffix5", "full_suffix"):
            if (2, field_kind) not in residuals:
                continue
            residual, mask = residuals[(2, field_kind)]
            candidates: dict[str, np.ndarray] = {}
            for kind in CEILING_KINDS:
                for null_id in sorted({entry["null_id"] for entry in index
                                       if entry["kind"] == kind and entry["rank"] == CEILING_RANK}):
                    key = f"{kind}|{null_id}|r{CEILING_RANK}"
                    if key in available:
                        candidates[f"{kind}|{null_id}"] = bases[key]
            if (0, field_kind) in residuals:
                train_residual, train_mask = residuals[(0, field_kind)]
                masked = train_residual * train_mask
                if masked.shape[0] >= CEILING_RANK:
                    candidates["TRAIN_ONLY_FREE_PCA|observed|f100"] = orthonormal_truncation(
                        masked.T, CEILING_RANK)[0]
            candidate_counts = mask.sum(axis=1)
            available_median = (float(np.median(candidate_counts[mask.any(axis=1)]))
                                if mask.any() else 0.0)
            for label, basis in candidates.items():
                residual_energy, energy = masked_projection_residual(residual, mask, basis)
                angles = (principal_angles(aligned_basis, basis)
                          if aligned_basis is not None and basis.shape[1] == aligned_basis.shape[1]
                          else [])
                rows.append({
                    "patient": patient, "baseline_level": level, "field_kind": field_kind,
                    "split": "development_test", "rank": CEILING_RANK,
                    "basis": label.split("|")[0], "null_id": label.split("|")[1],
                    "relative_projection_error": residual_energy / max(energy, 1e-12),
                    "available_contacts_median": available_median,
                    # With only a handful of candidate contacts left after the prefix, a
                    # rank-4 basis spans essentially everything and the ceiling is
                    # degenerate rather than informative; the flag keeps those patients
                    # visible instead of letting them pull the cohort median.
                    "ceiling_informative": bool(available_median >= 2 * CEILING_RANK),
                    "residual_energy": energy,
                    "residual_effective_rank": effective_rank(residual * mask),
                    "n_fields": int(mask.any(axis=1).sum()),
                    "subspace_overlap_with_aligned": (
                        float(np.mean(np.cos(angles) ** 2)) if angles else float("nan")),
                })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=12)
    arguments = parser.parse_args()
    patients = sorted(
        path.stem for path in (RESULT_ROOT / "basis" / "per_patient").glob("*.npz"))
    with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
        rows = [row for chunk in pool.map(process, patients) for row in chunk]
    table = pd.DataFrame(rows)
    table.to_csv(RESULT_ROOT / "basis" / "BASIS_CEILING_PER_PATIENT.csv", index=False)
    table.to_csv(RESULT_ROOT / "PER_PATIENT_BASIS_CEILING.csv", index=False)

    focus = table[(table["field_kind"] == "suffix5") & (table["baseline_level"] == "U_FULL_SET")]
    print(f"patients: {len(patients)}   rows: {len(table)}")
    for label, frame in (("all patients", focus),
                         ("informative only (candidates >= 2 x rank)",
                          focus[focus["ceiling_informative"]])):
        print(f"\nsuffix-5 residual, strong unordered baseline, rank 4 — {label} "
              f"({frame['patient'].nunique()} patients; lower = spans more):")
        print(frame.groupby(["basis"])["relative_projection_error"]
              .agg(["count", "median"]).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
