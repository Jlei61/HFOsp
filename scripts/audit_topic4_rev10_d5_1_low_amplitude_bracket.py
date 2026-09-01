"""Freeze the lowest D5.1 amplitude that opens both routes per network."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config/topic4_rev10_d5_1_spatial_ou_low_amplitude.json"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _ratio(value, baseline):
    return None if baseline == 0 else float(value / baseline)


def _activity(row, off):
    keys = (
        "mean_network_detected_events_descriptive",
        "mean_network_returned_events_scored",
        "mean_network_returned_fraction",
        "mean_network_fraction_time_above_detector",
        "max_network_fraction_time_above_detector",
        "mean_network_peak_active_fraction",
    )
    values = {key: float(row[key]) for key in keys}
    values["fraction_time_above_detector_delta_from_off"] = float(
        row["mean_network_fraction_time_above_detector"]
        - off["mean_network_fraction_time_above_detector"]
    )
    values["fraction_time_above_detector_ratio_to_off"] = _ratio(
        row["mean_network_fraction_time_above_detector"],
        off["mean_network_fraction_time_above_detector"],
    )
    values["detected_events_ratio_to_off"] = _ratio(
        row["mean_network_detected_events_descriptive"],
        off["mean_network_detected_events_descriptive"],
    )
    return values


def adjudicate(summary):
    rows = summary["candidate_rows"]
    off = next(row for row in rows if row["candidate_id"] == "edge_noop")
    local = {
        float(row["spatial_ou_sigma_rate_per_ms"]): row
        for row in rows if row["spatial_ou_mode"] == "local"
    }
    permuted = {
        float(row["spatial_ou_sigma_rate_per_ms"]): row
        for row in rows if row["spatial_ou_mode"] == "permuted"
    }
    if set(local) != set(permuted) or sorted(local) != [0.1, 0.2, 0.35]:
        raise RuntimeError("D5.1 local/permuted amplitude bracket is incomplete")

    comparisons = []
    for sigma in sorted(local):
        row_local, row_permuted = local[sigma], permuted[sigma]
        local_access = bool(
            row_local["n_runaway_networks"] == 0
            and row_local["networks_with_both_clean_modes"] >= 2
        )
        permutation_access = bool(
            row_permuted["n_runaway_networks"] == 0
            and row_permuted["networks_with_both_clean_modes"] >= 2
        )
        comparisons.append({
            "sigma_rate_per_ms": sigma,
            "local_candidate_id": row_local["candidate_id"],
            "permuted_candidate_id": row_permuted["candidate_id"],
            "local_networks_with_A": row_local["networks_with_clean_A"],
            "local_networks_with_B": row_local["networks_with_clean_B"],
            "local_networks_with_both": row_local[
                "networks_with_both_clean_modes"
            ],
            "permuted_networks_with_both": row_permuted[
                "networks_with_both_clean_modes"
            ],
            "local_score": row_local["selection_score_equal_network"],
            "permuted_score": row_permuted["selection_score_equal_network"],
            "local_ood_fraction": row_local["mean_network_ood_fraction"],
            "permuted_ood_fraction": row_permuted[
                "mean_network_ood_fraction"
            ],
            "local_clip_fraction": row_local[
                "max_network_spatial_ou_clip_fraction"
            ],
            "permuted_clip_fraction": row_permuted[
                "max_network_spatial_ou_clip_fraction"
            ],
            "local_access": local_access,
            "permutation_access": permutation_access,
            "local_activity": _activity(row_local, off),
            "permuted_activity": _activity(row_permuted, off),
        })

    eligible = [row for row in comparisons if row["local_access"]]
    selected = eligible[0] if eligible else None
    status = (
        "REV10D5_1_LOWEST_ACCESSIBLE_AMPLITUDE_FROZEN"
        if selected is not None
        else "REV10D5_1_LOW_AMPLITUDE_ACCESS_NOT_OBSERVED"
    )
    return {
        "status": status,
        "selection_rule": (
            "smallest scanned sigma with no local runaway and same-network "
            "clean A+B support in at least 2/3 fit networks"
        ),
        "selected_sigma_rate_per_ms": (
            None if selected is None else selected["sigma_rate_per_ms"]
        ),
        "selected_local_candidate_id": (
            None if selected is None else selected["local_candidate_id"]
        ),
        "matched_permuted_candidate_id": (
            None if selected is None else selected["permuted_candidate_id"]
        ),
        "selected_marginal_access_also_sufficient": (
            None if selected is None else selected["permutation_access"]
        ),
        "comparisons": comparisons,
        "off_baseline": off,
        "off_activity": _activity(off, off),
        "claim_boundary": (
            "same D5 fit networks; amplitude bracket only; activity burden is "
            "reported but was not retrofitted as a gate; not fresh-network or "
            "Fig4 confirmation"
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
    if summary.get("status") != "REV10D5_1_RETURNED_ONLY_BRACKET_COMPLETE":
        raise RuntimeError("D5.1 bracket summary incomplete")
    payload = adjudicate(summary)
    payload["inputs"] = {
        "config": {
            "path": str(config_path.relative_to(ROOT)),
            "sha256": _sha256(config_path),
        },
        "summary": {
            "path": str(summary_path.relative_to(ROOT)),
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
        "selected_sigma_rate_per_ms": payload[
            "selected_sigma_rate_per_ms"
        ],
        "selected_local_candidate_id": payload[
            "selected_local_candidate_id"
        ],
        "matched_permuted_candidate_id": payload[
            "matched_permuted_candidate_id"
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
