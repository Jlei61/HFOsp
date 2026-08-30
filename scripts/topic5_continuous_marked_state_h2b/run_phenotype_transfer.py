#!/usr/bin/env python3
"""Run frozen-target H2b phenotype-transfer probes from seizure-level CSV."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
# Load cuda_env's C++ runtime before pandas extensions; see run_risk_probe.py.
import torch as _torch  # noqa: F401
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.topic5_continuous_marked_state_h2b.contract import (  # noqa: E402
    H2B_REVISION, H2B_V0_2_REVISION, RESULT_ROOT, RunBoundary,
    assert_safe_output_path, atomic_csv, atomic_json,
    sha256_file, utc_now,
)
from src.topic5_continuous_marked_state_h2b.phenotype_transfer import (  # noqa: E402
    make_synthetic_phenotype_table, run_phenotype_table, target_table_hash,
)


def _records(frame: pd.DataFrame) -> list[dict]:
    return frame.replace({np.nan: None}).to_dict(orient="records")


def _read(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={
        "patient_id": str, "seizure_id": str, "split": str,
        "evaluation_tier": str, "target_name": str, "target_kind": str,
    })
    if "target_frozen" in frame:
        values = frame["target_frozen"]
        if not pd.api.types.is_bool_dtype(values):
            lowered = values.astype(str).str.strip().str.lower()
            if not set(lowered.unique()).issubset({"true", "false"}):
                raise ValueError("target_frozen must be strict true/false")
            frame["target_frozen"] = lowered.map({"true": True, "false": False})
    return frame


def run(
    *,
    input_path: Path | None,
    output_dir: Path,
    regularization_grid: tuple[float, ...],
    synthetic_only: bool,
    overwrite: bool,
    h2b_revision: str = H2B_REVISION,
) -> dict:
    output_dir = assert_safe_output_path(output_dir)
    targets = {
        "per_seed": output_dir / "per_seed_phenotype_metrics.csv",
        "patient": output_dir / "patient_median_phenotype_metrics.csv",
        "synthetic": output_dir / "positive_synthetic.json",
        "audit": output_dir / "phenotype_transfer_machine_audit.json",
    }
    existing = [str(path) for path in targets.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"refusing to overwrite phenotype outputs: {existing}")
    if synthetic_only:
        frame = make_synthetic_phenotype_table()
        provenance = {"kind": "positive_synthetic", "input_path": None,
                      "input_sha256": None}
    else:
        if input_path is None:
            raise ValueError("--input is required unless --synthetic-only is used")
        frame = _read(input_path)
        provenance = {
            "kind": "external_pre_frozen_target_table",
            "input_path": str(input_path.resolve()),
            "input_sha256": sha256_file(input_path),
        }
    result = run_phenotype_table(frame, regularization_grid=regularization_grid)
    synthetic = run_phenotype_table(
        make_synthetic_phenotype_table(random_seed=4816),
        regularization_grid=regularization_grid,
    )
    synthetic_effects = {
        row["target_name"]: float(row["state_minus_observation_loss"])
        for row in synthetic.patient_medians.to_dict(orient="records")
    }
    synthetic_payload = {
        "status": "PASS" if synthetic_effects and all(
            value < 0 for value in synthetic_effects.values()
        ) else "FAIL",
        "target_reclustered": False,
        "effects_state_minus_observation": synthetic_effects,
        "expected_sign": "negative",
        "random_seed": 4816,
    }
    source = REPO_ROOT / "src/topic5_continuous_marked_state_h2b/phenotype_transfer.py"
    script = Path(__file__).resolve()
    audit = {
        "status": result.audit["status"],
        "created_utc": utc_now(),
        "boundary": asdict(RunBoundary(revision=str(h2b_revision))),
        "input": provenance,
        "target_reclustered": False,
        "target_table_hash": (
            target_table_hash(frame) if result.audit["status"] !=
            "NOT_ESTIMABLE_MISSING_TARGET_COLUMNS" else None
        ),
        "target_frozen_before_probe": result.audit.get(
            "target_frozen_before_probe", False
        ),
        "train_select_test_seizure_ids": result.audit.get("split_seizure_ids", {}),
        "identical_seizure_target_rows_across_arms": result.audit.get(
            "identical_seizure_target_rows_across_arms", False
        ),
        "regularization_selected_only_on_train_select": result.audit.get(
            "regularization_selected_only_on_train_select", False
        ),
        "seed_is_patient_replicate": False,
        "seed_aggregation": "median_within_patient_before_cohort_inference",
        "effect_definition": (
            "held-out state-minus-observation loss; negative favours transfer"
        ),
        "probe_audit": result.audit,
        "positive_synthetic": synthetic_payload,
        "source_sha256": {
            str(source.relative_to(REPO_ROOT)): sha256_file(source),
            str(script.relative_to(REPO_ROOT)): sha256_file(script),
        },
        "outputs": {name: str(path) for name, path in targets.items()},
    }
    atomic_csv(targets["per_seed"], _records(result.per_seed))
    atomic_csv(targets["patient"], _records(result.patient_medians))
    atomic_json(targets["synthetic"], synthetic_payload)
    atomic_json(targets["audit"], audit)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument(
        "--output-dir", type=Path,
        default=RESULT_ROOT / "fits/phenotype_transfer_instrument",
    )
    parser.add_argument(
        "--regularization-grid", type=float, nargs="+",
        default=(0.01, 0.1, 1.0, 10.0),
    )
    parser.add_argument("--synthetic-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--h2b-revision", default=H2B_REVISION,
        choices=(H2B_REVISION, H2B_V0_2_REVISION),
    )
    args = parser.parse_args()
    audit = run(
        input_path=args.input,
        output_dir=args.output_dir,
        regularization_grid=tuple(args.regularization_grid),
        synthetic_only=args.synthetic_only,
        overwrite=args.overwrite,
        h2b_revision=args.h2b_revision,
    )
    print(json.dumps({
        "status": audit["status"], "output": str(args.output_dir),
        "synthetic": audit["positive_synthetic"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
