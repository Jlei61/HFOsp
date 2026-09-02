#!/usr/bin/env python3
"""Freeze target-free analysis tables before early-ictal authorization.

This sidecar does not compute an endpoint and never imports a target reader.
It waits for every Stage-F producer to finish, verifies the expected target-free
denominators, and records the exact bytes later consumed by figures, claim
adjudication and the closeout report.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results/topic5_multiscale_effective_scaffold_v0_5"
REQUIRED_MARKERS = (
    "MECHANISM_ANALYSIS_COMPLETE.json",
    "MODE_FLOW_ATTENUATION_COMPLETE.json",
    "ATTENUATED_FIELDS_FROZEN.json",
    "GAIN_ADJUSTED_SENSITIVITY_COMPLETE.json",
)
TABLES = (
    "mechanism/MECHANISM_PER_FIT_SEED.csv",
    "mechanism/MECHANISM_PER_PATIENT.csv",
    "mechanism/MODE_FLOW_ATTENUATION_PER_DRAW.csv",
    "mechanism/MODE_FLOW_ATTENUATION_PER_PATIENT.csv",
    "mechanism/MODE_FLOW_ATTENUATION_SUMMARY.json",
    "ATTENUATION_PER_DRAW.csv",
    "ATTENUATION_PER_PATIENT_DOSE.csv",
    "ATTENUATION_PER_PATIENT_AUC.csv",
    "GAIN_ADJUSTED_PER_FIT_SEED.csv",
    "GAIN_ADJUSTED_PER_PATIENT.csv",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def require_false_target_flag(frame: pd.DataFrame, relative: str) -> None:
    if "target_values_read" in frame and not frame.target_values_read.eq(False).all():
        raise RuntimeError(f"target flag is not uniformly false: {relative}")


def assess_gain_matching(gain: pd.DataFrame, tolerance: float = 0.01) -> dict:
    """Assess the complete 126-pair validation-only gain-match contract."""
    gain_pairs = gain.pivot(
        index=["subject", "fit_id", "scope", "seed"],
        columns="arm", values="validation_G3_matched",
    )
    if gain_pairs.shape[1] == 2:
        denominator = gain_pairs.abs().max(axis=1).clip(lower=1e-12)
        relative_error = (
            gain_pairs.iloc[:, 0] - gain_pairs.iloc[:, 1]
        ).abs() / denominator
    else:
        relative_error = pd.Series(dtype=float)
    finite = bool(
        np.isfinite(gain.validation_G3_intact.to_numpy(float)).all()
        and np.isfinite(gain.validation_G3_matched.to_numpy(float)).all()
        and np.isfinite(relative_error.to_numpy(float)).all()
    )
    scale_valid = bool(
        gain.recurrent_scale.between(0.0, 1.0, inclusive="right").all()
    )
    maximum = float(relative_error.max()) if len(relative_error) else float("inf")
    return {
        "pass": bool(
            len(gain_pairs) == 126 and gain_pairs.shape[1] == 2
            and finite and scale_valid and maximum <= float(tolerance)
        ),
        "pairs": int(len(gain_pairs)),
        "maximum_relative_error": maximum,
        "tolerance": float(tolerance),
        "scale_valid": scale_valid,
    }


def assess_attenuation_coverage(attenuation: pd.DataFrame) -> dict:
    """Assess all 504 unit-targets at each of the four frozen doses."""
    dose_groups = attenuation[[
        "subject", "fit_id", "target", "alpha", "seed",
    ]].drop_duplicates()
    unit_targets = attenuation[[
        "subject", "fit_id", "target", "seed",
    ]].drop_duplicates()
    passed = bool(
        attenuation.target_values_read.eq(False).all()
        and attenuation.subject.nunique() == 28
        and attenuation.fit_id.nunique() == 42
        and attenuation.target.nunique() == 4
        and attenuation.alpha.nunique() == 4
        and attenuation.seed.nunique() == 3
        and len(unit_targets) == 504
        and len(dose_groups) == 504 * 4
    )
    return {
        "pass": passed,
        "unit_targets": int(len(unit_targets)),
        "unit_target_dose_groups": int(len(dose_groups)),
    }


def assess_attenuation_draw_semantics(attenuation: pd.DataFrame) -> dict:
    """Validate arm-specific draws while allowing explicit rollout collapse."""
    failures: list[str] = []
    rollout_undefined = 0
    valid_rows = attenuation[attenuation.draw.astype(int) >= 0].copy()
    required_finite = (
        "contact_nll", "local_nll", "distal_nll",
        "local_damage", "distal_damage", "distal_selectivity",
    )
    for column in required_finite:
        if not np.isfinite(valid_rows[column].to_numpy(float)).all():
            failures.append(f"NONFINITE_VALID_ROW:{column}")
    undefined = ~np.isfinite(valid_rows.rollout_spearman.to_numpy(float))
    rollout_undefined = int(undefined.sum())
    if undefined.any() and not valid_rows.loc[undefined, "rollout_spearman_n"].eq(0).all():
        failures.append("UNDEFINED_ROLLOUT_WITH_NONZERO_DENOMINATOR")
    group_columns = ["subject", "fit_id", "target", "seed"]
    for keys, group in attenuation.groupby(group_columns, sort=False):
        label = "|".join(map(str, keys))
        doses = sorted(map(float, group.alpha.unique()))
        if doses != [0.25, 0.5, 0.75, 1.0]:
            failures.append(f"DOSE_SET:{label}")
            continue
        valid_draws = group.n_valid_matched_draws.astype(int).unique()
        if len(valid_draws) != 1:
            failures.append(f"VALID_DRAW_COUNT_DRIFT:{label}")
            continue
        n_valid = int(valid_draws[0])
        expected_eligible = n_valid >= 200 if keys[2] == "L3_MATCHED_LOCAL" else True
        if not group.inferential_eligible.astype(bool).eq(expected_eligible).all():
            failures.append(f"ELIGIBILITY_MISMATCH:{label}")
        if keys[2] != "L3_MATCHED_LOCAL":
            if len(group) != 4 or set(group.draw.astype(int)) != {0}:
                failures.append(f"ADDED_EDGE_DRAW_CONTRACT:{label}")
            if group.target_mask_sha256.nunique(dropna=True) != 1:
                failures.append(f"ADDED_EDGE_MASK_DRIFT:{label}")
            continue
        expected_draws = min(16, n_valid)
        if expected_draws == 0:
            if len(group) != 4 or set(group.draw.astype(int)) != {-1}:
                failures.append(f"EMPTY_MATCHED_LOCAL_PLACEHOLDER:{label}")
            continue
        expected_ids = set(range(expected_draws))
        if len(group) != 4 * expected_draws:
            failures.append(f"MATCHED_LOCAL_ROW_COUNT:{label}")
        for _alpha, dose in group.groupby("alpha"):
            if set(dose.draw.astype(int)) != expected_ids:
                failures.append(f"MATCHED_LOCAL_DRAW_SET:{label}:{_alpha}")
        mask_by_draw = group.groupby("draw").target_mask_sha256.nunique(dropna=True)
        if not mask_by_draw.eq(1).all() or group.target_mask_sha256.nunique(dropna=True) != expected_draws:
            failures.append(f"MATCHED_LOCAL_MASK_CONTRACT:{label}")
    return {
        "pass": not failures,
        "rows": int(len(attenuation)),
        "valid_rows": int(len(valid_rows)),
        "rollout_undefined_rows": rollout_undefined,
        "rollout_undefined_fraction": (
            float(rollout_undefined / len(valid_rows)) if len(valid_rows) else 0.0
        ),
        "interpretation": "UNDEFINED_ONLY_WHEN_POST_SEED_ROLLOUT_DENOMINATOR_IS_ZERO",
        "failures": failures,
    }


def validate_tables(out: Path) -> dict[str, dict]:
    evidence: dict[str, dict] = {}
    for relative in TABLES:
        path = out / relative
        if not path.exists():
            raise FileNotFoundError(path)
        row: dict = {
            "path": relative,
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        if path.suffix == ".csv":
            frame = pd.read_csv(path)
            require_false_target_flag(frame, relative)
            row["rows"] = len(frame)
            row["columns"] = list(map(str, frame.columns))
        evidence[relative] = row

    expected = {
        "mechanism/MECHANISM_PER_FIT_SEED.csv": 378,
        "mechanism/MECHANISM_PER_PATIENT.csv": 84,
        "GAIN_ADJUSTED_PER_FIT_SEED.csv": 252,
        "GAIN_ADJUSTED_PER_PATIENT.csv": 56,
        "ATTENUATION_PER_PATIENT_DOSE.csv": 448,
        "ATTENUATION_PER_PATIENT_AUC.csv": 112,
    }
    for relative, rows in expected.items():
        if evidence[relative].get("rows") != rows:
            raise RuntimeError(
                f"unexpected pre-unseal denominator: {relative}: "
                f"{evidence[relative].get('rows')} != {rows}"
            )
    gain = pd.read_csv(out / "GAIN_ADJUSTED_PER_FIT_SEED.csv")
    gain_assessment = assess_gain_matching(gain)
    if not gain_assessment["pass"]:
        raise RuntimeError(
            "validation-only gain matching did not meet the frozen 1% tolerance"
        )
    evidence["GAIN_ADJUSTED_PER_FIT_SEED.csv"]["validation_gain_matching"] = (
        gain_assessment
    )

    attenuation = pd.read_csv(out / "ATTENUATION_PER_DRAW.csv")
    attenuation_assessment = assess_attenuation_coverage(attenuation)
    if not attenuation_assessment["pass"]:
        raise RuntimeError("attenuation draw table does not cover all frozen unit-target doses")
    evidence["ATTENUATION_PER_DRAW.csv"]["coverage"] = attenuation_assessment
    draw_semantics = assess_attenuation_draw_semantics(attenuation)
    if not draw_semantics["pass"]:
        raise RuntimeError("attenuation draw semantics failed")
    evidence["ATTENUATION_PER_DRAW.csv"]["draw_semantics"] = draw_semantics
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--poll-seconds", type=int, default=2)
    parser.add_argument("--timeout-hours", type=float, default=36.0)
    args = parser.parse_args()
    out = args.out_root.resolve()
    destination = out / "PREUNSEAL_ANALYSIS_METRIC_MANIFEST.json"
    if destination.exists():
        return
    begin = time.monotonic()
    while not all((out / marker).exists() for marker in REQUIRED_MARKERS):
        if (out / "TARGET_UNSEAL_AUTHORIZATION.json").exists():
            raise RuntimeError("target authorization preceded analysis-metric freeze")
        if (out / "STAGE_F_TARGET_FREE_FAILED.json").exists():
            raise RuntimeError("Stage F failed before analysis-metric freeze")
        if time.monotonic() - begin > args.timeout_hours * 3600:
            raise TimeoutError("timed out waiting for Stage-F analysis tables")
        time.sleep(max(1, int(args.poll_seconds)))
    if (out / "TARGET_UNSEAL_AUTHORIZATION.json").exists():
        raise RuntimeError("target authorization preceded analysis-metric freeze")
    evidence = validate_tables(out)
    payload = {
        "contract": "topic5_preunseal_analysis_metric_manifest_v0_5",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_TARGET_FREE",
        "target_values_read": False,
        "files": evidence,
    }
    write_json(destination, payload)
    write_json(out / "PREUNSEAL_ANALYSIS_METRIC_FREEZE_COMPLETE.json", {
        "status": "PASS_TARGET_FREE",
        "created_utc": payload["created_utc"],
        "manifest": str(destination),
        "manifest_sha256": sha256_file(destination),
        "target_values_read": False,
    })


if __name__ == "__main__":
    main()
