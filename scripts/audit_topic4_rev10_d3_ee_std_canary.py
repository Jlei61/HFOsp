"""Adjudicate source-specific E->E STD against mean-matched global STD."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config/topic4_rev10_d3_dynamic_ee_std_canary.json"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def adjudicate(summary):
    rows = summary["candidate_rows"]
    off = next(row for row in rows if row["candidate_id"] == "edge_noop")
    local = {
        (float(row["ee_std_u"]), float(row["ee_std_tau_ms"])): row
        for row in rows if row["ee_std_mode"] == "local"
    }
    global_ = {
        (float(row["ee_std_u"]), float(row["ee_std_tau_ms"])): row
        for row in rows if row["ee_std_mode"] == "global"
    }
    if set(local) != set(global_) or len(local) != 4:
        raise RuntimeError("dynamic E->E STD grid is incomplete")
    comparisons, passed = [], []
    for key in sorted(local):
        row_local, row_global = local[key], global_[key]
        local_specific = bool(
            row_local["n_runaway_networks"] == 0
            and row_local["networks_with_both_clean_modes"] >= 2
            and row_local["networks_with_clean_B"] >= 2
            and row_local["networks_with_both_clean_modes"]
            > row_global["networks_with_both_clean_modes"]
            and row_local["networks_with_both_clean_modes"]
            > off["networks_with_both_clean_modes"]
        )
        comparison = {
            "u": key[0],
            "tau_ms": key[1],
            "local_candidate_id": row_local["candidate_id"],
            "global_candidate_id": row_global["candidate_id"],
            "local_networks_with_A": row_local["networks_with_clean_A"],
            "global_networks_with_A": row_global["networks_with_clean_A"],
            "off_networks_with_A": off["networks_with_clean_A"],
            "local_networks_with_B": row_local["networks_with_clean_B"],
            "local_networks_with_both": row_local["networks_with_both_clean_modes"],
            "global_networks_with_both": row_global["networks_with_both_clean_modes"],
            "off_networks_with_both": off["networks_with_both_clean_modes"],
            "local_score": row_local["selection_score_equal_network"],
            "global_score": row_global["selection_score_equal_network"],
            "off_score": off["selection_score_equal_network"],
            "local_mean_minimum_availability": row_local[
                "mean_network_minimum_std_availability"
            ],
            "global_mean_minimum_availability": row_global[
                "mean_network_minimum_std_availability"
            ],
            "local_specific_route_access": local_specific,
        }
        comparisons.append(comparison)
        if local_specific:
            passed.append(comparison)
    passed.sort(key=lambda row: row["local_score"])
    return {
        "status": (
            "REV10D3_SOURCE_SPECIFIC_DYNAMIC_EDGE_ACCESS_OBSERVED"
            if passed else
            "REV10D3_SOURCE_SPECIFIC_DYNAMIC_EDGE_ACCESS_NOT_OBSERVED"
        ),
        "selected_local_candidate_id": (
            passed[0]["local_candidate_id"] if passed else None
        ),
        "matched_global_candidate_id": (
            passed[0]["global_candidate_id"] if passed else None
        ),
        "comparisons": comparisons,
        "off_baseline": off,
        "claim_boundary": (
            "three-network development canary; returned-only; local versus "
            "mean-matched global STD; not patient-blind or an ictal lifecycle test"
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    root = ROOT / config["output_root"]
    summary_path = root / "canary_summary_returned_only.json"
    summary = json.loads(summary_path.read_text())
    if summary.get("status") != "REV10D3_RETURNED_ONLY_CANARY_COMPLETE":
        raise RuntimeError("D3 summary incomplete")
    payload = adjudicate(summary)
    payload["inputs"] = {
        "config": {
            "path": str(config_path.relative_to(ROOT)),
            "sha256": _sha256(config_path),
        },
        "summary": {
            "path": str(summary_path),
            "sha256": _sha256(summary_path),
        },
    }
    output = root / "canary_verdict.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=output.parent, suffix=".tmp")
    os.close(fd)
    try:
        Path(temporary).write_text(json.dumps(payload, indent=2, sort_keys=True))
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(json.dumps({
        "status": payload["status"],
        "selected_local_candidate_id": payload["selected_local_candidate_id"],
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
