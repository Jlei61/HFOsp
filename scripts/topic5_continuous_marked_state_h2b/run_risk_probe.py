#!/usr/bin/env python3
"""Run the frozen-state H2b conditional risk-set probe from a CSV table."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
# cuda_env otherwise lets pandas bind the system libstdc++ first; importing torch
# loads the environment's compatible C++ runtime before pandas/sklearn extensions.
import torch as _torch  # noqa: F401
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.topic5_continuous_marked_state_h2b.contract import (  # noqa: E402
    PROBE_ARMS,
    RESULT_ROOT,
    RunBoundary,
    assert_safe_output_path,
    atomic_csv,
    atomic_json,
    sha256_file,
    utc_now,
)
from src.topic5_continuous_marked_state_h2b.risk_probe import (  # noqa: E402
    make_positive_synthetic_risk_table,
    risk_set_hash,
    run_probe_table,
    time_label_permutation_audit,
)


BOOLEAN_COLUMNS = (
    "is_case",
    "horizon_seizure_free",
    "in_ictal_or_postictal",
    "observation_available",
    "wrong_time_donor_valid",
    "wrong_time_same_segment",
    "wrong_time_exclusion_clear",
)


def _boolean_series(values: pd.Series, name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.astype(bool)
    if pd.api.types.is_numeric_dtype(values):
        unique = set(values.dropna().astype(int).tolist())
        if unique.issubset({0, 1}):
            return values.astype(int).astype(bool)
    lowered = values.astype(str).str.strip().str.lower()
    if set(lowered.unique()).issubset({"true", "false"}):
        return lowered.map({"true": True, "false": False}).astype(bool)
    raise ValueError(f"column {name!r} cannot be parsed as strict boolean")


def _read_risk_table(path: Path) -> pd.DataFrame:
    identity_columns = {
        "patient_id": str,
        "seizure_id": str,
        "risk_set_id": str,
        "anchor_id": str,
        "split": str,
        "evaluation_tier": str,
        "segment_id": str,
        "observation_signature": str,
    }
    frame = pd.read_csv(path, dtype=identity_columns)
    for column in BOOLEAN_COLUMNS:
        if column in frame:
            frame[column] = _boolean_series(frame[column], column)
    for column in ("anchor_time", "seizure_onset", "segment_start", "segment_end"):
        if column in frame:
            frame[column] = frame[column].astype(np.float64)
    return frame


def _records(frame: pd.DataFrame) -> list[dict]:
    return frame.replace({np.nan: None}).to_dict(orient="records")


def _lead_curve(patient_medians: pd.DataFrame) -> pd.DataFrame:
    metric = "state_minus_observation_conditional_log_loss"
    if patient_medians.empty or metric not in patient_medians:
        return pd.DataFrame(columns=[
            "evaluation_tier", "lead_minutes", "n_patients", "patient_median_effect",
        ])
    rows = []
    for (tier, lead), group in patient_medians.groupby(
        ["evaluation_tier", "lead_minutes"], sort=True,
    ):
        values = group[metric].dropna().to_numpy(dtype=float)
        rows.append({
            "evaluation_tier": str(tier),
            "lead_minutes": int(lead),
            "n_patients": int(len(values)),
            "patient_median_effect": float(np.median(values)) if len(values) else None,
            "effect_definition": (
                "B_state_minus_B_observation_conditional_log_loss; negative favours state"
            ),
        })
    return pd.DataFrame(rows)


def run(
    *,
    risk_table_path: Path | None,
    output_dir: Path,
    ridge_grid: tuple[float, ...],
    n_permutations: int,
    random_seed: int,
    synthetic_only: bool,
    overwrite: bool,
    arms: tuple[str, ...] = PROBE_ARMS,
) -> dict:
    output_dir = assert_safe_output_path(output_dir)
    targets = {
        "per_seed": output_dir / "per_seed_probe_metrics.csv",
        "patient": output_dir / "patient_median_probe_metrics.csv",
        "lead": output_dir / "lead_curve.csv",
        "permutation": output_dir / "time_label_permutation.json",
        "synthetic": output_dir / "positive_synthetic.json",
        "audit": output_dir / "risk_probe_machine_audit.json",
    }
    if not overwrite and any(path.exists() for path in targets.values()):
        existing = [str(path) for path in targets.values() if path.exists()]
        raise FileExistsError(f"refusing to overwrite existing probe outputs: {existing}")

    if synthetic_only:
        frame = make_positive_synthetic_risk_table(random_seed=random_seed)
        input_provenance = {
            "kind": "positive_synthetic",
            "generator_seed": int(random_seed),
            "risk_table_path": None,
            "risk_table_sha256": None,
        }
    else:
        if risk_table_path is None:
            raise ValueError("--risk-table is required unless --synthetic-only is used")
        frame = _read_risk_table(risk_table_path)
        input_provenance = {
            "kind": "external_frozen_state_risk_table",
            "risk_table_path": str(risk_table_path.resolve()),
            "risk_table_sha256": sha256_file(risk_table_path),
        }

    fitted = run_probe_table(frame, ridge_grid=ridge_grid, arms=arms)
    permutation = time_label_permutation_audit(
        frame,
        n_permutations=int(n_permutations),
        ridge_grid=ridge_grid,
        random_seed=int(random_seed) + 1,
    )
    synthetic_frame = make_positive_synthetic_risk_table(random_seed=int(random_seed) + 2)
    synthetic_run = run_probe_table(synthetic_frame, ridge_grid=ridge_grid)
    synthetic_effect = float(
        synthetic_run.patient_medians.loc[
            synthetic_run.patient_medians["lead_minutes"] == 30,
            "state_minus_observation_conditional_log_loss",
        ].median()
    )
    synthetic_payload = {
        "status": "PASS" if synthetic_effect < 0 else "FAIL",
        "generator": "seizure choice sampled from softmax(frozen_like_persistent_state)",
        "random_seed": int(random_seed) + 2,
        "n_seizures": int(synthetic_frame["seizure_id"].nunique()),
        "state_minus_observation_conditional_log_loss": synthetic_effect,
        "expected_sign": "negative",
        "risk_set_hash": risk_set_hash(synthetic_frame),
    }
    lead_curve = _lead_curve(fitted.patient_medians)
    source_path = REPO_ROOT / "src/topic5_continuous_marked_state_h2b/risk_probe.py"
    script_path = Path(__file__).resolve()
    machine_audit = {
        "status": "COMPLETE",
        "created_utc": utc_now(),
        "boundary": asdict(RunBoundary()),
        "input": input_provenance,
        "risk_set_hash": risk_set_hash(frame),
        "identical_risk_sets_across_arms": True,
        "identical_risk_set_hash_across_arms": risk_set_hash(frame),
        "arms": list(arms),
        "train_select_test_seizure_ids": fitted.audit["train_select_test_seizure_ids"],
        "lead_to_split_consistency": fitted.audit["lead_to_split_consistency"],
        "regularization_selected_only_on_train_select": True,
        "wrong_time_donors_same_patient_segment_and_exclusion_clear": (
            True if "wrong_time" in arms else None
        ),
        "wrong_time_confounders_adjusted_in_all_direct_comparisons": (
            True if "wrong_time" in arms else None
        ),
        "regularization_scope": (
            "primary: TRAIN fit, SELECT choose, TRAIN+SELECT refit; "
            "LOSO: nested selection excludes held-out seizure"
        ),
        "seed_is_patient_replicate": False,
        "seed_aggregation": "median_within_patient_before_cohort_inference",
        "main_effect_definition": (
            "held-out 30-min conditional log loss B_state-B_observation; "
            "negative means frozen persistent state adds information"
        ),
        "probe_audit": fitted.audit,
        "positive_synthetic": synthetic_payload,
        "time_label_permutation": {
            key: value for key, value in permutation.items() if key != "null_values"
        },
        "source_sha256": {
            str(source_path.relative_to(REPO_ROOT)): sha256_file(source_path),
            str(script_path.relative_to(REPO_ROOT)): sha256_file(script_path),
        },
        "outputs": {key: str(value) for key, value in targets.items()},
    }
    atomic_csv(targets["per_seed"], _records(fitted.per_seed))
    atomic_csv(targets["patient"], _records(fitted.patient_medians))
    atomic_csv(targets["lead"], _records(lead_curve))
    atomic_json(targets["permutation"], permutation)
    atomic_json(targets["synthetic"], synthetic_payload)
    atomic_json(targets["audit"], machine_audit)
    return machine_audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--risk-table", type=Path)
    parser.add_argument(
        "--output-dir", type=Path,
        default=RESULT_ROOT / "fits/risk_probe_instrument",
    )
    parser.add_argument(
        "--ridge-grid", type=float, nargs="+", default=(0.01, 0.1, 1.0, 10.0),
    )
    parser.add_argument("--n-permutations", type=int, default=100)
    parser.add_argument("--random-seed", type=int, default=9917)
    parser.add_argument("--synthetic-only", action="store_true")
    parser.add_argument(
        "--arms", nargs="+", choices=PROBE_ARMS, default=list(PROBE_ARMS),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    payload = run(
        risk_table_path=args.risk_table,
        output_dir=args.output_dir,
        ridge_grid=tuple(args.ridge_grid),
        n_permutations=args.n_permutations,
        random_seed=args.random_seed,
        synthetic_only=args.synthetic_only,
        overwrite=args.overwrite,
        arms=tuple(args.arms),
    )
    print(json.dumps({
        "status": payload["status"],
        "risk_set_hash": payload["risk_set_hash"],
        "output": str(args.output_dir),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
