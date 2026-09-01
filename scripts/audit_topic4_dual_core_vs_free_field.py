#!/usr/bin/env python3
"""Engineering and claim-contract audit for the Node-field comparison."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_topic4_rev10_sa_shaft_aware_target import _atomic_json  # noqa: E402


DEFAULT_CONFIG = ROOT / "config/topic4_dual_core_vs_free_field.json"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    root = ROOT / config["output_root"]
    manifest_path = root / "candidate_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    source_root = ROOT / (
        "results/topic4_sef_hfo/data_driven_local_connectivity_rev11_nlc/"
        "frozen_substrate_confirmation"
    )
    rows, failures = [], []
    for seed in config["search"]["confirmation_network_seeds"]:
        manual_stem = root / "workers" / f"hand_dual_core_budget_matched_seed_{seed}"
        free_stem = source_root / "workers" / f"node_baseline_seed_{seed}"
        manual_json = manual_stem.with_suffix(".json")
        manual_npz = manual_stem.with_suffix(".npz")
        free_json = free_stem.with_suffix(".json")
        free_npz = free_stem.with_suffix(".npz")
        manual = json.loads(manual_json.read_text())
        free = json.loads(free_json.read_text())
        checks = {
            "manual_array_hash": manual["arrays"]["sha256"] == _sha256(manual_npz),
            "free_array_hash": free["arrays"]["sha256"] == _sha256(free_npz),
            "manual_complete": manual["status"] == "REV10R_EDGE_FLOW_WORKER_COMPLETE",
            "same_seed": manual["seed"] == free["seed"] == int(seed),
            "same_duration": manual["simulation"]["duration_ms"] == free["simulation"]["duration_ms"] == 20000.0,
            "same_detector": manual["simulation"]["common_detector_threshold"] == free["simulation"]["common_detector_threshold"],
            "same_network_size": manual["network"]["n_E"] == free["network"]["n_E"] and manual["network"]["n_I"] == free["network"]["n_I"],
            "same_signed_depth": manual["node_field"]["node_hashes"]["d_vector_sha256"] == free["node_field"]["node_hashes"]["d_vector_sha256"],
            "manual_budget": abs(manual["node_field"]["sum_h"] - 1129.0) <= 1e-9,
            "free_budget": abs(free["node_field"]["sum_h"] - 1129.0) <= 1e-9,
            "manual_edge_exact_noop": manual["edge_audit"].get("exact_noop") is True,
            "free_edge_exact_noop": free["edge_audit"].get("exact_noop") is True,
            "manual_no_runaway": manual["run"]["runaway_early_stop_ms"] is None,
            "free_no_runaway": free["run"]["runaway_early_stop_ms"] is None,
            "manual_zm_off": manual["mz_slow_state"]["mode"] == "off",
            "free_zm_off": free["mz_slow_state"]["mode"] == "off",
        }
        with np.load(manual_npz, allow_pickle=False) as left, np.load(
                free_npz, allow_pickle=False) as right:
            checks.update({
                "same_positions": np.array_equal(left["positions_E"], right["positions_E"]),
                "same_contacts": np.array_equal(left["contact_names"], right["contact_names"]),
                "same_contact_geometry": np.array_equal(left["contact_xy_mm"], right["contact_xy_mm"]),
                "manual_binary_h": np.all(np.isin(left["h"], [0.0, 1.0])),
            })
        failed = [key for key, value in checks.items() if not value]
        failures.extend(f"seed {seed}: {key}" for key in failed)
        rows.append({
            "seed": int(seed), "checks": checks, "failed": failed,
            "manual_n_events": manual["run"]["n_common_detector_events"],
            "free_n_events": free["run"]["n_common_detector_events"],
            "manual_json_sha256": _sha256(manual_json),
            "free_json_sha256": _sha256(free_json),
        })
    summary_path = root / "comparison_summary.json"
    summary = json.loads(summary_path.read_text())
    figures = root / "figures"
    required_figures = [
        figures / f"dual_core_vs_free_field_{stem}.{suffix}"
        for stem in ("explanatory_power", "kmeans")
        for suffix in ("png", "pdf")
    ]
    global_checks = {
        "manifest_status": manifest["status"] == "REV11NLC_DUAL_CORE_COMPARISON_LIBRARY_FROZEN",
        "manifest_config_hash": manifest["config"]["sha256"] == _sha256(config_path),
        "all_12_seed_pairs_present": len(rows) == 12,
        "summary_complete": summary["status"] == "TOPIC4_DUAL_CORE_VS_FREE_FIELD_COMPLETE",
        "primary_target_is_heldout": summary["primary_target"].startswith("patient heldout"),
        "figures_complete": all(path.exists() and path.stat().st_size > 0 for path in required_figures),
        "figure_readme": (figures / "README.md").exists(),
    }
    failures.extend(key for key, value in global_checks.items() if not value)
    payload = {
        "status": (
            "TOPIC4_DUAL_CORE_VS_FREE_FIELD_AUDIT_PASS"
            if not failures else "TOPIC4_DUAL_CORE_VS_FREE_FIELD_AUDIT_FAIL"
        ),
        "global_checks": global_checks,
        "per_seed": rows,
        "failures": failures,
        "artifacts": {
            "manifest": {"path": str(manifest_path), "sha256": _sha256(manifest_path)},
            "summary": {"path": str(summary_path), "sha256": _sha256(summary_path)},
            "figures": [str(path) for path in required_figures],
        },
        "claim_boundary": config["claim_boundary"],
    }
    _atomic_json(root / "comparison_audit.json", payload)
    print(json.dumps({
        "status": payload["status"], "failures": failures,
        "seed_pairs": len(rows),
    }, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
