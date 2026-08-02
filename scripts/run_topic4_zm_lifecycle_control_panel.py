#!/usr/bin/env python3
"""Build/run finite E-threshold-uplift calibration and dose panels."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import run_topic4_zm_lifecycle_sprint_batch as B  # noqa: E402


OUT = ROOT / "results/topic4_sef_hfo/zm_fast_lifecycle_development/lifecycle_sprint"
CALIBRATION_UPLIFTS_MV = (0.25, 0.5, 1.0, 2.0, 4.0)
DOSE_MULTIPLIERS = (0.5, 1.0, 1.5)
DOSE_DURATIONS_MS = (50.0, 200.0)
CONTROL_CLOCK_VERSION = "relative_to_pre_entry_checkpoint_v2"


def _now():
    return datetime.now(timezone.utc).isoformat()


def _candidate_config(row):
    keep = {
        key: row[key] for key in (
            "arm", "tau_D_ms", "d_star", "strength_scale", "tau_aI_ms", "f_aI",
            "g_M", "tau_M_ms", "g_Z",
        ) if key in row
    }
    if keep.get("arm") not in {"i2e", "combined"}:
        raise ValueError("control candidate requires an i2e or combined fast arm")
    for key in ("g_M", "tau_M_ms", "g_Z"):
        if key not in keep:
            raise ValueError(f"control candidate is missing {key}")
    onset = row.get("onset_ms")
    if onset is None:
        raise ValueError("control candidate must have a measured uncontrolled onset")
    if row.get("offset_ms") is not None or row.get("duration_right_censored") is False:
        raise ValueError("control candidate must be persistent in its paired uncontrolled run")
    t0 = float(row.get("control_t0_ms", float(onset) + 1500.0))
    if t0 < float(onset) + 1500.0:
        raise ValueError("control time must be at least 1500 ms after uncontrolled onset")
    uncontrolled = {
        "uncontrolled_source_candidate_id": row.get("config_id", row.get("stem")),
        "uncontrolled_onset_ms": float(onset),
        "uncontrolled_offset_ms": row.get("offset_ms"),
        "uncontrolled_duration_right_censored": bool(
            row.get("duration_right_censored", row.get("offset_ms") is None)
        ),
        "uncontrolled_summary_path": row.get("summary_path"),
        "control_timing_rule": row.get("control_timing_rule", "onset_plus_1500ms_fallback"),
        "uncontrolled_core_mean_hz_at_control": row.get("uncontrolled_core_mean_hz_at_control"),
    }
    return keep, t0, uncontrolled


def _selected(payload):
    rows = list(payload.get("selected", payload.get("rows", [])))
    if not 1 <= len(rows) <= 4:
        raise ValueError("control panel requires one to four selected persistent candidates")
    return rows


def build_calibration_manifest(selection, *, T_ms=20000.0):
    prepared = []
    for position, source in enumerate(_selected(selection)):
        rank = int(source.get("selection_rank", position))
        base, t0, uncontrolled = _candidate_config(source)
        source_id = source.get("config_id", source.get("stem", f"selected_{rank}"))
        prepared.append((rank, base, t0, source_id, uncontrolled))
    rows = []
    for uplift in CALIBRATION_UPLIFTS_MV:
        for rank, base, t0, source_id, uncontrolled in prepared:
            row = {
                "family": "control_u_ref_calibration",
                "selection_rank": rank,
                "source_candidate_id": source_id,
                **base,
                **uncontrolled,
                "control_target": "all_E",
                "control_clock": CONTROL_CLOCK_VERSION,
                "control_t0_ms": t0,
                "control_duration_ms": 50.0,
                "control_uplift_mV": uplift,
                "T_ms": float(T_ms), "burn_ms": 1000.0,
            }
            row["config_id"] = B._cfg_id(row)
            rows.append(row)
    return {
        "schema": "topic4_zm_lifecycle_control_calibration_v1_2026-08-02",
        "created_at_utc": _now(), "seed": 1, "paired_noise": True,
        "selection_source": selection.get("selection_source"),
        "n_configs": len(rows), "rows": rows,
    }


def build_dose_manifest(selection, calibration, *, T_ms=20000.0):
    decision_rows = calibration.get("calibration_decisions", calibration.get("rows", []))
    refs = {int(row["selection_rank"]): row for row in decision_rows}
    prepared = []
    for position, source in enumerate(_selected(selection)):
        rank = int(source.get("selection_rank", position))
        if rank not in refs or refs[rank].get("u_ref_mV") is None:
            raise ValueError(f"selection rank {rank} has no calibrated u_ref")
        base, t0, uncontrolled = _candidate_config(source)
        source_id = source.get("config_id", source.get("stem", f"selected_{rank}"))
        u_ref = float(refs[rank]["u_ref_mV"])
        prepared.append((rank, base, t0, source_id, u_ref, uncontrolled))
    rows = []
    for multiplier in DOSE_MULTIPLIERS:
        for duration in DOSE_DURATIONS_MS:
            for rank, base, t0, source_id, u_ref, uncontrolled in prepared:
                row = {
                    "family": "finite_control_dose",
                    "selection_rank": rank,
                    "source_candidate_id": source_id,
                    **base,
                    **uncontrolled,
                    "u_ref_mV": u_ref,
                    "dose_multiplier": multiplier,
                    "control_target": "all_E",
                    "control_clock": CONTROL_CLOCK_VERSION,
                    "control_t0_ms": t0,
                    "control_duration_ms": duration,
                    "control_uplift_mV": multiplier * u_ref,
                    "T_ms": float(T_ms), "burn_ms": 1000.0,
                }
                row["config_id"] = B._cfg_id(row)
                rows.append(row)
    return {
        "schema": "topic4_zm_lifecycle_control_dose_v1_2026-08-02",
        "created_at_utc": _now(), "seed": 1, "paired_noise": True,
        "selection_source": selection.get("selection_source"),
        "calibration_source": calibration.get("source_path"),
        "n_configs": len(rows), "rows": rows,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("calibration", "dose"))
    ap.add_argument("--selection-json", type=Path, required=True)
    ap.add_argument("--calibration-json", type=Path)
    ap.add_argument("--T-ms", type=float, default=20000.0)
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--min-mem-gb", type=float, default=90.0)
    ap.add_argument("--poll-s", type=float, default=30.0)
    ap.add_argument("--manifest-only", action="store_true")
    args = ap.parse_args()
    selection = json.loads(args.selection_json.read_text())
    if args.mode == "calibration":
        manifest = build_calibration_manifest(selection, T_ms=args.T_ms)
        stem = "control_calibration"
    else:
        if args.calibration_json is None:
            raise SystemExit("dose mode requires --calibration-json")
        calibration = json.loads(args.calibration_json.read_text())
        manifest = build_dose_manifest(selection, calibration, T_ms=args.T_ms)
        stem = "control_dose"
    manifest_path = OUT / f"{stem}_manifest.json"
    B._atomic_json(manifest_path, manifest)
    if args.manifest_only:
        print(manifest_path); return
    print(B.run_manifest(
        manifest_path, OUT / f"{stem}_run_ledger.json",
        args.max_workers, args.min_mem_gb, args.poll_s,
    ))


if __name__ == "__main__":
    main()
