#!/usr/bin/env python3
"""Run A3--A5 exploratory H2b hazard analyses for one frozen cell."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.topic5_continuous_marked_state_h2b.contract import (  # noqa: E402
    CANONICAL_V0_2_RESULT_ROOT,
    CANONICAL_V0_3_RESULT_ROOT,
    atomic_json,
    sha256_file,
    utc_now,
)
from src.topic5_continuous_marked_state_h2b.v03_contract import (  # noqa: E402
    assert_frozen_contract_matches,
)
from src.topic5_continuous_marked_state_h2b.v03_exploration_policy import (  # noqa: E402
    assert_frozen_exploration_policy_matches,
)
from src.topic5_continuous_marked_state_h2b.v03_hazard import (  # noqa: E402
    build_hazard_design,
    lagged_persistent_state,
    prequential_nested_hazard,
)


PRODUCER = Path(__file__).resolve()
HAZARD_MODULE = REPO / "src/topic5_continuous_marked_state_h2b/v03_hazard.py"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _onsets(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    times, segments = [], []
    total = 0
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            total += 1
            try:
                segment = int(row["segment_id"])
            except (TypeError, ValueError):
                continue
            times.append(float(row["onset_time"]))
            segments.append(segment)
    order = np.argsort(np.asarray(times, dtype=np.float64), kind="stable")
    return (
        np.asarray(times, dtype=np.float64)[order],
        np.asarray(segments, dtype=np.int64)[order], total,
    )


def _a1_row(root: Path, subject: str) -> dict:
    summary = _json(root / "qualification/instrument_smoke_summary.json")
    selected = [row for row in summary["patient_rows"]
                if str(row["subject"]) == str(subject)]
    return selected[0] if len(selected) == 1 else {
        "subject": subject, "exploration_stratum": "A1_NOT_AVAILABLE",
        "state_qualified": False,
    }


def _tau(root: Path, subject: str, seed: int) -> tuple[float | None, str]:
    path = root / "instrument/by_cell" / subject / f"seed_{seed}/instrument_manifest.json"
    if not path.is_file():
        return None, "A1_CELL_UNAVAILABLE"
    q4 = _json(path)["diagnostics"]["Q4_time_constant"]
    empirical = q4.get("empirical_decoder_tau_minutes")
    if empirical is not None and np.isfinite(float(empirical)):
        return float(empirical), "empirical_decoder_tau_minutes"
    analytic = q4.get("analytic_generator_slowest_mode_minutes")
    if analytic is not None and np.isfinite(float(analytic)):
        return float(analytic), "analytic_generator_tau_fallback"
    return None, "TAU_NOT_IDENTIFIABLE"


def _wrong_time_override(data: MappingLike, source_index: np.ndarray,
                         memoryless: np.ndarray) -> tuple[np.ndarray, int]:
    state = np.asarray(data["wrong_time_donor_state"], dtype=np.float64)[source_index]
    valid = np.asarray(data["wrong_time_valid"], dtype=bool)[source_index]
    result = np.array(memoryless, copy=True)
    n_valid = 0
    for row in range(len(source_index)):
        donor = np.flatnonzero(valid[row])
        if len(donor):
            result[row] = state[row, donor[0]]
            n_valid += 1
    return result, n_valid


class MappingLike:
    """Typing shim for numpy NpzFile without importing private numpy types."""

    def __getitem__(self, key: str):  # pragma: no cover - protocol only
        raise NotImplementedError


def run(subject: str, seed: int, *, v02_root: Path, result_root: Path,
        state_cache_root: Path | None = None, output_subdir: str = "hazard",
        allow_diagnostic_exploration: bool = False) -> dict:
    assert_frozen_contract_matches(_json(result_root / "analysis_contract.json"))
    assert_frozen_exploration_policy_matches(_json(result_root / "exploration_policy.json"))
    a1 = _a1_row(result_root, subject)
    if (a1.get("state_qualified") is not True
            and not allow_diagnostic_exploration):
        raise ValueError(
            f"{subject}: A3--A5 not released because A1 state qualification failed"
        )
    final_assay_path = result_root / "assay/type1_power_summary.json"
    diagnostic_assay_path = result_root / "assay/type1_power_summary_smoke.json"
    assay_path = (
        diagnostic_assay_path if allow_diagnostic_exploration
        else final_assay_path
    )
    if not assay_path.is_file():
        raise ValueError("the selected A2 assay receipt is missing")
    assay = _json(assay_path)
    if not allow_diagnostic_exploration and (
        assay.get("status") != "PASS_FINAL_ASSAY_ACCEPTANCE"
        or assay.get("claim_bearing_route_released") is not True
    ):
        raise ValueError("A3--A5 not released because A2 assay did not pass")
    selected_k = int(assay["selected_initial_k"])
    cache_base = state_cache_root or (v02_root / "state_cache")
    cache_root = cache_base / subject / f"seed_{seed}"
    cache_path = cache_root / "states.npz"
    state_manifest_path = cache_root / "states.manifest.json"
    state_manifest = _json(state_manifest_path)
    if state_manifest.get("cache_sha256") != sha256_file(cache_path):
        raise ValueError(f"{subject}/seed_{seed}: state cache SHA256 drift")
    if state_manifest.get("all_parameters_frozen") is not True:
        raise ValueError("hazard input is not a frozen state cache")
    full_recorded_grid = (
        state_manifest.get("full_recorded_five_minute_grid") is True
    )
    if not full_recorded_grid and not allow_diagnostic_exploration:
        raise ValueError(
            "A3 requires a full recorded-coverage state grid; v0.2 risk-set "
            "query caches are not admissible"
        )
    seizure_path = v02_root / "risk_sets" / subject / "seizures.csv"
    onset_time, onset_segment, n_inventory_seizures = _onsets(seizure_path)
    with np.load(cache_path, allow_pickle=False) as data:
        design = build_hazard_design(
            time_epoch=data["anchor_time_epoch"],
            segment=data["coverage_segment_index"],
            history=data["deterministic_history"],
            current_observation=data["current_explicit_summary"],
            persistent_state=data["persistent_state"],
            memoryless_state=data["memoryless_observation_code"],
            observation_available=data["observation_available"],
            onset_time=onset_time, onset_segment=onset_segment,
        )
        wrong_time, n_wrong = _wrong_time_override(
            data, design.source_index, design.memoryless_state,
        )
    by_k = {
        str(k): prequential_nested_hazard(design, initial_k=k)
        for k in (2, 3, 4, 5)
    }
    secondary = {
        str(horizon): prequential_nested_hazard(
            design, initial_k=selected_k, horizon_minutes=float(horizon),
        )
        for horizon in (5, 15, 60)
    }
    wrong_time_result = prequential_nested_hazard(
        design, initial_k=selected_k, persistent_override=wrong_time,
    )
    tau, tau_source = _tau(result_root, subject, seed)
    lag_rows = []
    if tau is not None:
        for multiplier in (0.5, 1.0, 2.0, 4.0):
            lag_minutes = float(tau) * multiplier
            lagged, valid = lagged_persistent_state(design, lag_minutes)
            lagged[~valid] = design.memoryless_state[~valid]
            result = prequential_nested_hazard(
                design, initial_k=selected_k, persistent_override=lagged,
            )
            lag_rows.append({
                "tau_multiplier": multiplier, "lag_minutes": lag_minutes,
                "n_valid_past_donors": int(np.sum(valid)),
                "valid_past_donor_fraction": float(np.mean(valid)),
                "invalid_donor_policy": "replace_with_same_anchor_memoryless_code",
                "result": result,
            })
    output = result_root / output_subdir / "by_cell" / subject / f"seed_{seed}"
    payload = {
        "status": "COMPLETE_EXPLORATORY",
        "revision": "h2b_v0_3_hazard_cell_v2",
        "created_utc": utc_now(), "subject": subject, "seed": int(seed),
        "selected_initial_k": selected_k,
        "primary_30min_by_initial_k": by_k,
        "primary_selected_k": by_k[str(selected_k)],
        "secondary_horizon_refits": secondary,
        "secondary_horizon_caveat": (
            "separate development refits, not horizons derived from one fitted hazard"
        ),
        "matched_wrong_time": {
            "n_rows_with_donor": n_wrong,
            "donor_fraction": float(n_wrong / len(design.time_epoch)),
            "invalid_donor_policy": "replace_with_same_anchor_memoryless_code",
            "result": wrong_time_result,
        },
        "tau_lag_response": {
            "tau_minutes": tau, "tau_source": tau_source,
            "rows": lag_rows,
        },
        "A1_patient_stratum": a1,
        "A2_assay_status": assay["status"],
        "A2_transfer_assay_sensitive": bool(
            assay.get("status") == "PASS_FINAL_ASSAY_ACCEPTANCE"
            and assay.get("claim_bearing_route_released") is True
        ),
        "analysis_scope": (
            "full_recorded_development_grid_exploratory" if full_recorded_grid else
            "seizure_support_conditioned_control_grid_exploratory"
        ),
        "n_grid_rows": len(design.time_epoch),
        "n_mapped_seizures": len(onset_time),
        "n_inventory_seizures": n_inventory_seizures,
        "source": {
            "state_cache": str(cache_path),
            "state_cache_sha256": sha256_file(cache_path),
            "state_manifest": str(state_manifest_path),
            "state_manifest_sha256": sha256_file(state_manifest_path),
            "seizure_table": str(seizure_path),
            "seizure_table_sha256": sha256_file(seizure_path),
            "producer_sha256": sha256_file(PRODUCER),
            "hazard_module_sha256": sha256_file(HAZARD_MODULE),
            "assay_summary_sha256": sha256_file(assay_path),
        },
        "real_seizure_outcome_read": True,
        "development_only": True,
        "formal_test_partition_opened": False, "sealed_opened": False,
        "h3_or_t2_run": False,
        "claim_status": (
            "CLAIM_ROUTE_RELEASED_DEVELOPMENT_ONLY"
            if not allow_diagnostic_exploration else
            (
                "EXPLORATORY_A1_EMPTY_ASSAY_NOT_SENSITIVE_FULL_GRID"
                if full_recorded_grid else
                "EXPLORATORY_A1_EMPTY_ASSAY_NOT_SENSITIVE_SUPPORT_CONDITIONED"
            )
        ),
        "omp_num_threads": int(os.environ.get("OMP_NUM_THREADS", "1")),
    }
    atomic_json(output / "result.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--v0-2-root", type=Path, default=CANONICAL_V0_2_RESULT_ROOT)
    parser.add_argument("--result-root", type=Path, default=CANONICAL_V0_3_RESULT_ROOT)
    parser.add_argument("--state-cache-root", type=Path, default=None)
    parser.add_argument("--output-subdir", default="hazard")
    parser.add_argument("--allow-diagnostic-exploration", action="store_true")
    parser.add_argument("--allow-support-conditioned-exploration", action="store_true")
    args = parser.parse_args()
    payload = run(
        str(args.subject), int(args.seed),
        v02_root=args.v0_2_root.resolve(), result_root=args.result_root.resolve(),
        state_cache_root=(
            args.state_cache_root.resolve() if args.state_cache_root is not None else None
        ),
        output_subdir=str(args.output_subdir),
        allow_diagnostic_exploration=bool(
            args.allow_diagnostic_exploration
            or args.allow_support_conditioned_exploration
        ),
    )
    print(payload["status"], payload["subject"], payload["seed"])


if __name__ == "__main__":
    main()
