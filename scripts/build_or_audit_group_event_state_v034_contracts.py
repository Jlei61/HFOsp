#!/usr/bin/env python3
"""Build the v0.3.4 anchor/baseline eligibility audit from metadata only.

This script never opens a model result, a target value or the sealed partition.
It compares the existing v0.3.3 structural eligibility artifact with a fresh
calculation from the real recorded coverage segments, then records which raw
interictal measurement families physically exist for each requested sentinel.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_group_event_state.v02.timeline import build_carry_segments, sessions_from_inventory  # noqa: E402
from src.topic5_group_event_state.v032_eval.partition import EVAL_PHASES, eval_partition  # noqa: E402
from src.topic5_group_event_state.v034_contracts.anchors import (  # noqa: E402
    build_fixed_time_anchors,
    independent_window_count,
)
from src.topic5_group_event_state.v034_contracts.eligibility import (  # noqa: E402
    audit_array_capabilities,
    endpoint_rows,
)

DEFAULT_SUBJECTS = ("epilepsiae_1146", "epilepsiae_583", "epilepsiae_548", "epilepsiae_922")
HORIZONS = (300, 1800, 7200, 21600)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        tmp = Path(handle.name)
    os.replace(tmp, path)


def _inventory_rows(path: Path, subject: str) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle) if row["subject"] == subject]
    if not rows:
        raise FileNotFoundError(f"{subject}: no session inventory rows in {path}")
    return rows


def _normalise_subject(name: str, available: set[str]) -> str:
    if name in available:
        return name
    matches = [subject for subject in available if subject.endswith(f"_{name}")]
    if len(matches) != 1:
        raise ValueError(f"subject {name!r} does not resolve uniquely: {matches}")
    return matches[0]


def _requirement(old: dict, key: str) -> int | None:
    raw = old.get("requirements", {}).get(key, {}).get("required_blocks")
    return None if raw is None else int(raw)


def _target_records(path: Path | None) -> dict[str, dict]:
    if path is None:
        return {}
    payload = json.loads(path.read_text())
    if isinstance(payload.get("subjects"), dict):
        return {str(k): dict(v) for k, v in payload["subjects"].items()}
    rows = payload.get("records")
    if isinstance(rows, list) and all(isinstance(row, dict) and "subject" in row for row in rows):
        return {str(row["subject"]): dict(row) for row in rows}
    raise ValueError("H2b target registry must contain subjects{} or records[{subject,...}]")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/topic5_group_event_state_v032_eval.json")
    parser.add_argument(
        "--prior-eligibility",
        type=Path,
        default=Path("/data/hfosp_group_event_state_v0_3_3/shared/eligibility/eligibility_by_endpoint_horizon.json"),
    )
    parser.add_argument("--subjects", nargs="*", default=list(DEFAULT_SUBJECTS))
    parser.add_argument("--h2b-target-registry", type=Path, default=None)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/data/hfosp_group_event_state_v0_3_4/shared/contracts/fixed_anchor_baseline_eligibility_audit.json"),
    )
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    prior = json.loads(args.prior_eligibility.read_text())
    if bool(prior.get("sealed_partition_opened")):
        raise ValueError("prior eligibility artifact claims the sealed partition was opened")
    available_subjects = set(prior.get("support_by_subject", {}))
    subjects = [_normalise_subject(name, available_subjects) for name in args.subjects]
    dataset_root = Path(config["dataset_root"])
    inventory_path = Path(config["session_inventory"])
    count_requirement = _requirement(prior, "count_profile|1800")
    grammar_requirement = _requirement(prior, "grammar|1800")
    h2b_records = _target_records(args.h2b_target_registry)

    records: list[dict] = []
    all_crosschecks: list[bool] = []
    for subject in subjects:
        subject_dir = dataset_root / subject
        index_path = subject_dir / "index.json"
        index = json.loads(index_path.read_text())
        sessions = sessions_from_inventory(_inventory_rows(inventory_path, subject))
        segments = build_carry_segments(
            sessions,
            index.get("seizures", []),
            postictal_exclusion_seconds=float(config["timeline"]["postictal_exclusion_seconds"]),
            min_segment_seconds=float(config["timeline"]["min_segment_seconds"]),
        )
        partition = eval_partition(segments, config["partition"]["boundary_fractions"])
        prior_support = prior["support_by_subject"][subject]

        blocks: dict[int, dict[str, int]] = {}
        anchor_counts: dict[int, dict[str, int]] = {}
        crosscheck: dict[str, dict[str, int | bool]] = {}
        for horizon in HORIZONS:
            blocks[horizon] = {
                phase: independent_window_count(segments, partition, phase=phase, horizon_seconds=horizon)
                for phase in EVAL_PHASES
            }
            anchors = build_fixed_time_anchors(
                segments,
                partition,
                horizons_seconds=(horizon,),
                grid_seconds=float(config["timeline"]["anchor_grid_seconds"]),
                warmup_seconds=float(config["timeline"]["min_warmup_seconds"]),
                embargo_seconds=float(horizon),
            )
            anchor_counts[horizon] = {
                phase: sum(row.phase == phase for row in anchors) for phase in EVAL_PHASES
            }
            prior_blocks = prior_support.get("blocks", {}).get(str(horizon))
            if prior_blocks is not None:
                exact = all(int(prior_blocks[phase]) == blocks[horizon][phase] for phase in EVAL_PHASES)
                crosscheck[str(horizon)] = {
                    "exact": exact,
                    "prior_dev_test": int(prior_blocks["dev_test"]),
                    "recomputed_dev_test": blocks[horizon]["dev_test"],
                }
                all_crosschecks.append(exact)

        capabilities = audit_array_capabilities(
            subject_dir, index, h2b_target_record=h2b_records.get(subject)
        )
        rows = endpoint_rows(
            subject=subject,
            blocks_by_horizon=blocks,
            prior_support=prior_support,
            capabilities=capabilities,
            count_requirement_30m=count_requirement,
            grammar_requirement_30m=grammar_requirement,
        )
        records.append({
            "subject": subject,
            "dataset": index.get("dataset"),
            "index_sha256": _sha256(index_path),
            "n_coverage_segments": len(segments),
            "recorded_seconds_by_phase": partition.recorded_seconds,
            "independent_blocks_by_horizon": {str(k): v for k, v in blocks.items()},
            "fixed_anchor_counts_by_horizon": {str(k): v for k, v in anchor_counts.items()},
            "prior_eligibility_crosscheck": crosscheck,
            "capabilities": capabilities,
            "endpoint_rows": rows,
        })

    payload = {
        "format": "group_event_state_v0_3_4_fixed_anchor_baseline_eligibility_audit",
        "generated": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "subjects": subjects,
        "sources": {
            "config": str(args.config),
            "config_sha256": _sha256(args.config),
            "prior_eligibility": str(args.prior_eligibility),
            "prior_eligibility_sha256": _sha256(args.prior_eligibility),
            "session_inventory": str(inventory_path),
            "session_inventory_sha256": _sha256(inventory_path),
            "h2b_target_registry": str(args.h2b_target_registry) if args.h2b_target_registry else None,
            "h2b_target_registry_sha256": _sha256(args.h2b_target_registry) if args.h2b_target_registry else None,
        },
        "audit_scope": "coverage, split, array metadata and seizure-count support only; no model result or target value read",
        "development_results_opened": False,
        "sealed_partition_opened": False,
        "horizon_policy": {
            "core_seconds": [300, 1800],
            "exploratory_seconds": [7200, 21600],
            "reason": "120 min and 6 h have sparse independent blocks and no calibrated endpoint-specific power contract",
        },
        "level_control_registry": {
            "train_mean_adapter": {
                "causal": True,
                "fit": "TRAIN_only",
                "role": "deployable_static_patient_calibration",
            },
            "rolling_prefix_level": {
                "causal": True,
                "availability_rule": "a block may update the level only after its target_stop",
                "role": "causal_slow_level_candidate",
            },
            "selection_period_mean": {
                "causal": False,
                "input_only": True,
                "uses_future_labels": False,
                "role": "noncausal_input_oracle_diagnostic_only",
            },
        },
        "baseline_contract": {
            "name": "B_multiscale",
            "causal": True,
            "tau_seconds": [300, 1800, 7200, 10800, 21600, 43200],
            "event_rule": "event_time < anchor_time",
            "resets_at": "coverage segment",
            "normalization": "TRAIN_only_then_frozen",
            "future_seizure_information": "forbidden",
        },
        "requirements_reused_without_recalibration": {
            "future_event_count_1800": count_requirement,
            "conditional_spatial_grammar_1800": grammar_requirement,
        },
        "prior_geometry_crosscheck_all_exact": bool(all(all_crosschecks)),
        "records": records,
    }
    _atomic_json(args.out, payload)
    print(json.dumps({
        "status": "complete",
        "out": str(args.out),
        "subjects": len(records),
        "prior_geometry_crosscheck_all_exact": payload["prior_geometry_crosscheck_all_exact"],
        "sealed_partition_opened": False,
    }, indent=2))


if __name__ == "__main__":
    main()
