#!/usr/bin/env python3
"""Merge fast, M, and control receipts into one auditable lifecycle ledger."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/topic4_sef_hfo/zm_fast_lifecycle_development/lifecycle_sprint"
STATUS_PRIORITY = {
    "not_run_after_adaptive_stop": 0,
    "unresolved_running_without_artifact": 1,
    "worker_failed": 2,
    "adaptively_cancelled": 3,
    "scientific_early_stop": 4,
    "success": 5,
}


def _read(path):
    return json.loads(path.read_text()) if path.is_file() else None


def _stage(row):
    family = row.get("family", "")
    if family == "m_response_panel":
        return "M_response"
    if family == "control_u_ref_calibration":
        return "control_calibration"
    if family == "finite_control_dose":
        return "control_dose"
    return "fast_phase_map"


def merge_adjudicated_ledgers(named_ledgers):
    grouped = {}
    for source_path, payload in named_ledgers:
        for config_id, row in payload.get("rows", {}).items():
            candidate = dict(row)
            candidate["config_id"] = config_id
            candidate["stage"] = _stage(candidate)
            candidate["source_ledger_path"] = str(source_path)
            grouped.setdefault(config_id, []).append(candidate)
    rows = {}
    for config_id, candidates in grouped.items():
        chosen = max(
            candidates,
            key=lambda row: (
                STATUS_PRIORITY.get(row.get("adjudicated_status"), -1),
                row.get("terminal_time_utc") or row.get("start_time_utc") or "",
            ),
        )
        chosen = dict(chosen)
        chosen["source_ledger_paths"] = sorted({row["source_ledger_path"] for row in candidates})
        chosen.pop("source_ledger_path", None)
        rows[config_id] = chosen
    return rows


def _analysis_indexes():
    by_config = {}
    by_summary = {}
    phase = _read(OUT / "batch1_phase_map.json") or {}
    for row in phase.get("rows", []):
        by_summary[row.get("summary_path")] = {
            "phenotype": row.get("phenotype"),
            "onset_ms": row.get("episode", {}).get("onset_ms"),
            "offset_ms": row.get("episode", {}).get("offset_ms"),
            "returning_event_candidate": row.get("recovery", {}).get("single_event_candidate", False),
            "returning_distribution_recovered": row.get("recovery", {}).get("distribution_recovered", False),
            "median_energy_gain_db": row.get("intensity", {}).get("median_gain_db_across_contacts"),
            "energy_occupancy_6db": row.get("intensity", {}).get("occupancy_above_6db"),
            "spatial_effective_rank": row.get("within_episode_spatial", {}).get("spatial_effective_rank"),
            "common_mode_pc1_fraction": row.get("within_episode_spatial", {}).get("common_mode_pc1_fraction"),
        }
    for filename in ("m_response_surface.json", "control_calibration_analysis.json", "control_dose_analysis.json"):
        payload = _read(OUT / filename) or {}
        for row in payload.get("rows", []):
            if row.get("status") != "complete" or row.get("config_id") is None:
                continue
            by_config[row["config_id"]] = {
                key: row.get(key) for key in (
                    "phenotype", "onset_ms", "offset_ms", "episode_duration_ms",
                    "causal_exit_candidate", "causal_control_exit_candidate",
                    "returning_event_candidate", "returning_distribution_recovered",
                    "median_energy_gain_db", "energy_occupancy_6db",
                    "spatial_effective_rank", "common_mode_pc1_fraction",
                    "exit_latency_from_control_ms", "rapid_reentry_count",
                ) if key in row
            }
    return by_config, by_summary


def annotate_rows(rows):
    by_config, by_summary = _analysis_indexes()
    for config_id, row in rows.items():
        readout = by_config.get(config_id)
        if readout is None:
            readout = by_summary.get(row.get("artifact_path"))
        row["scientific_readout"] = readout
    return rows


def build_payload(named_ledgers):
    rows = annotate_rows(merge_adjudicated_ledgers(named_ledgers))
    counts = {}
    stages = {}
    for row in rows.values():
        status = row.get("adjudicated_status", "unknown")
        counts[status] = counts.get(status, 0) + 1
        stage = row["stage"]
        stages[stage] = stages.get(stage, 0) + 1
    return {
        "schema": "topic4_zm_lifecycle_unified_ledger_v1_2026-08-02",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "semantic_scope": "seed1 development sprint; success receipt is not ictal lifecycle acceptance",
        "source_ledgers": [str(path) for path, _ in named_ledgers],
        "status_counts": counts,
        "stage_counts": stages,
        "n_unique_configs": len(rows),
        "rows": rows,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=OUT / "unified_run_ledger.json")
    args = ap.parse_args()
    paths = sorted(OUT.glob("*adjudicated_ledger.json"))
    if not paths:
        raise SystemExit("no adjudicated ledgers found")
    named = [(path.relative_to(ROOT), json.loads(path.read_text())) for path in paths]
    payload = build_payload(named)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({
        "n_unique_configs": payload["n_unique_configs"],
        "status_counts": payload["status_counts"],
        "stage_counts": payload["stage_counts"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
