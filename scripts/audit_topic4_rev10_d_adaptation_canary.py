"""Adjudicate local route memory against global and exact-off controls."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config/topic4_rev10_d_local_adaptation_canary.json"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".json.tmp")
    os.close(fd)
    try:
        Path(temporary).write_text(json.dumps(payload, indent=2, sort_keys=True))
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _key(row):
    return (float(row["adaptation_tau_ms"]),
            float(row["adaptation_increment_mV"]))


def adjudicate(summary):
    rows = summary["candidate_rows"]
    off = next(row for row in rows if row["candidate_id"] == "edge_noop")
    local = {_key(row): row for row in rows if row["adaptation_mode"] == "local"}
    global_ = {_key(row): row for row in rows if row["adaptation_mode"] == "global"}
    if set(local) != set(global_) or len(local) != 9:
        raise RuntimeError("local/global adaptation grid is incomplete")
    comparisons, informative = [], []
    for key in sorted(local):
        local_row, global_row = local[key], global_[key]
        local_both = int(local_row["networks_with_both_clean_modes"])
        global_both = int(global_row["networks_with_both_clean_modes"])
        off_both = int(off["networks_with_both_clean_modes"])
        local_a = int(local_row["networks_with_clean_A"])
        global_a = int(global_row["networks_with_clean_A"])
        local_b = int(local_row["networks_with_clean_B"])
        local_specific = bool(
            local_row["n_runaway_networks"] == 0
            and local_both >= 2
            and local_b >= 2
            and local_both > global_both
            and local_both > off_both
            and local_a >= global_a
        )
        record = {
            "tau_ms": key[0], "increment_mV": key[1],
            "local_candidate_id": local_row["candidate_id"],
            "global_candidate_id": global_row["candidate_id"],
            "off_candidate_id": off["candidate_id"],
            "local_networks_with_A": local_a,
            "global_networks_with_A": global_a,
            "off_networks_with_A": int(off["networks_with_clean_A"]),
            "local_networks_with_B": local_b,
            "local_networks_with_both": local_both,
            "global_networks_with_both": global_both,
            "off_networks_with_both": off_both,
            "local_score": float(local_row["selection_score_equal_network"]),
            "global_score": float(global_row["selection_score_equal_network"]),
            "off_score": float(off["selection_score_equal_network"]),
            "local_specific_route_access": local_specific,
        }
        comparisons.append(record)
        if local_specific:
            informative.append(record)
    informative.sort(key=lambda row: (row["local_score"], row["tau_ms"],
                                      row["increment_mV"]))
    strongest = sorted(
        comparisons,
        key=lambda row: (-row["local_networks_with_both"],
                         -row["local_networks_with_A"], row["local_score"]),
    )[0]
    passed = bool(informative)
    return {
        "status": (
            "REV10D_LOCAL_ADAPTATION_ROUTE_ACCESS_OBSERVED"
            if passed else "REV10D_LOCAL_ADAPTATION_ROUTE_ACCESS_NOT_OBSERVED"
        ),
        "selected_local_candidate_id": (
            informative[0]["local_candidate_id"] if passed else None
        ),
        "matched_global_candidate_id": (
            informative[0]["global_candidate_id"] if passed else None
        ),
        "n_informative_local_candidates": len(informative),
        "informative_candidates": informative,
        "strongest_local_diagnostic": strongest,
        "comparisons": comparisons,
        "off_baseline": off,
        "claim": (
            "At least one local E-neuron adaptation arm produced same-network A/B "
            "support beyond both its global brake and exact-off controls."
            if passed else
            "The frozen local-adaptation grid did not produce same-network A/B "
            "support beyond both global-brake and exact-off controls."
        ),
        "claim_boundary": (
            "three-network development canary; returned-only shaft-aware support; "
            "not patient-blind and not an ictal lifecycle test"
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
    if summary.get("status") != "REV10D_RETURNED_ONLY_CANARY_COMPLETE":
        raise RuntimeError("rev10-D canary summary is incomplete")
    payload = adjudicate(summary)
    payload["inputs"] = {
        "config": {"path": str(config_path.relative_to(ROOT)),
                   "sha256": _sha256(config_path)},
        "summary": {"path": str(summary_path), "sha256": _sha256(summary_path)},
    }
    output = root / "canary_verdict.json"
    _atomic_json(output, payload)
    print(json.dumps({
        "status": payload["status"],
        "selected_local_candidate_id": payload["selected_local_candidate_id"],
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
