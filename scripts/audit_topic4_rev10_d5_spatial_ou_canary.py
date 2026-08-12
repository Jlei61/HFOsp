"""Adjudicate local spatial OU against exact-marginal permutation controls."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config/topic4_rev10_d5_spatial_ou_accessibility_canary.json"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def adjudicate(summary):
    rows = summary["candidate_rows"]
    off = next(row for row in rows if row["candidate_id"] == "edge_noop")
    local = {
        (float(row["spatial_ou_sigma_rate_per_ms"]), float(row["spatial_ou_ell_mm"])): row
        for row in rows if row["spatial_ou_mode"] == "local"
    }
    permuted = {
        (float(row["spatial_ou_sigma_rate_per_ms"]), float(row["spatial_ou_ell_mm"])): row
        for row in rows if row["spatial_ou_mode"] == "permuted"
    }
    if set(local) != set(permuted) or len(local) != 4:
        raise RuntimeError("D5 local/permuted grid is incomplete")
    comparisons, local_specific, nonlocal_access = [], [], []
    for key in sorted(local):
        row_local, row_permuted = local[key], permuted[key]
        local_access = bool(
            row_local["n_runaway_networks"] == 0
            and row_local["networks_with_both_clean_modes"] >= 2
        )
        permutation_access = bool(
            row_permuted["n_runaway_networks"] == 0
            and row_permuted["networks_with_both_clean_modes"] >= 2
        )
        locality_specific = bool(
            local_access
            and row_local["networks_with_both_clean_modes"]
            > row_permuted["networks_with_both_clean_modes"]
            and row_local["networks_with_both_clean_modes"]
            > off["networks_with_both_clean_modes"]
        )
        comparison = {
            "sigma_rate_per_ms": key[0], "ell_mm": key[1],
            "local_candidate_id": row_local["candidate_id"],
            "permuted_candidate_id": row_permuted["candidate_id"],
            "local_networks_with_A": row_local["networks_with_clean_A"],
            "local_networks_with_B": row_local["networks_with_clean_B"],
            "local_networks_with_both": row_local["networks_with_both_clean_modes"],
            "permuted_networks_with_both": row_permuted[
                "networks_with_both_clean_modes"
            ],
            "off_networks_with_both": off["networks_with_both_clean_modes"],
            "local_score": row_local["selection_score_equal_network"],
            "permuted_score": row_permuted["selection_score_equal_network"],
            "off_score": off["selection_score_equal_network"],
            "local_ood_fraction": row_local["mean_network_ood_fraction"],
            "permuted_ood_fraction": row_permuted["mean_network_ood_fraction"],
            "local_clip_fraction": row_local[
                "max_network_spatial_ou_clip_fraction"
            ],
            "local_access": local_access,
            "permutation_access": permutation_access,
            "locality_specific_access": locality_specific,
        }
        comparisons.append(comparison)
        if locality_specific:
            local_specific.append(comparison)
        elif local_access and permutation_access:
            nonlocal_access.append(comparison)
    local_specific.sort(key=lambda row: row["local_score"])
    nonlocal_access.sort(key=lambda row: row["local_score"])
    if local_specific:
        status = "REV10D5_SPATIAL_LOCALITY_ACCESS_OBSERVED"
        selected = local_specific[0]
    elif nonlocal_access:
        status = "REV10D5_NONLOCAL_MARGINAL_ACCESS_OBSERVED"
        selected = nonlocal_access[0]
    else:
        status = "REV10D5_CONTINUOUS_SPATIAL_FLUCTUATION_ACCESS_NOT_OBSERVED"
        selected = None
    return {
        "status": status,
        "selected_local_candidate_id": (
            None if selected is None else selected["local_candidate_id"]
        ),
        "matched_permuted_candidate_id": (
            None if selected is None else selected["permuted_candidate_id"]
        ),
        "comparisons": comparisons,
        "off_baseline": off,
        "claim_boundary": (
            "three-network development canary; continuous observation-invariant "
            "afferent-rate field; returned-only; not patient blind or Fig4 confirmation"
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
    if summary.get("status") != "REV10D5_RETURNED_ONLY_CANARY_COMPLETE":
        raise RuntimeError("D5 summary incomplete")
    payload = adjudicate(summary)
    payload["inputs"] = {
        "config": {"path": str(config_path.relative_to(ROOT)), "sha256": _sha256(config_path)},
        "summary": {"path": str(summary_path), "sha256": _sha256(summary_path)},
    }
    output = root / "canary_verdict.json"
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
        "matched_permuted_candidate_id": payload["matched_permuted_candidate_id"],
    }, indent=2))


if __name__ == "__main__":
    main()
