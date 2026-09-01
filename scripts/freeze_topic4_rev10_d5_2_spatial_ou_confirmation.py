"""Freeze off/local/permuted D5.2 arms before fresh confirmation networks."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.getcwd())
from scripts.run_topic4_rev9l_forced_source_worker import (  # noqa: E402
    _runtime_provenance,
    _sha256,
)
from src.topic4_core_field_runner import atomic_write_json  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ROLE = "development_only_translation_invariant_spatial_ou_confirmation"


def build_manifest(config_path, expected_commit):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    if config.get("scientific_role") != EXPECTED_ROLE:
        raise RuntimeError("D5.2 confirmation scientific role changed")
    for record in config["inputs"].values():
        if _sha256(ROOT / record["path"]) != record["sha256"]:
            raise RuntimeError(f"input hash changed: {record['path']}")

    source_path = ROOT / config["inputs"]["bracket_candidate_manifest"]["path"]
    summary_path = ROOT / config["inputs"]["bracket_summary"]["path"]
    verdict_path = ROOT / config["inputs"]["bracket_verdict"]["path"]
    source = json.loads(source_path.read_text())
    summary = json.loads(summary_path.read_text())
    verdict = json.loads(verdict_path.read_text())
    if source.get("status") != "REV10D5_1_SPATIAL_OU_LOW_AMPLITUDE_LIBRARY_FROZEN":
        raise RuntimeError("D5.1 source manifest is invalid")
    if summary.get("status") != "REV10D5_1_RETURNED_ONLY_BRACKET_COMPLETE":
        raise RuntimeError("D5.1 summary is incomplete")
    if verdict.get("status") != "REV10D5_1_LOWEST_ACCESSIBLE_AMPLITUDE_FROZEN":
        raise RuntimeError("D5.1 did not freeze an accessible amplitude")
    if summary["manifest"]["sha256"] != _sha256(source_path):
        raise RuntimeError("D5.1 summary and manifest differ")

    by_id = {
        row["candidate_id"]: row
        for row in source["candidate_set"]["candidates"]
    }
    selected = verdict["selected_local_candidate_id"]
    permuted = verdict["matched_permuted_candidate_id"]
    frozen_ids = ["edge_noop", selected, permuted]
    if len(set(frozen_ids)) != 3 or any(value not in by_id for value in frozen_ids):
        raise RuntimeError("D5.1 did not nominate one local/permuted pair")
    if by_id[selected]["spatial_ou"]["mode"] != "local":
        raise RuntimeError("selected D5.2 candidate is not local")
    if by_id[permuted]["spatial_ou"]["mode"] != "permuted":
        raise RuntimeError("matched D5.2 candidate is not permuted")
    if (by_id[selected]["spatial_ou"]["sigma_rate_per_ms"]
            != by_id[permuted]["spatial_ou"]["sigma_rate_per_ms"]):
        raise RuntimeError("D5.2 local/permuted amplitudes differ")

    commit = subprocess.check_output(
        ["git", "rev-parse", expected_commit], cwd=ROOT, text=True,
    ).strip()
    provenance = _runtime_provenance(commit)
    config_dirty = bool(subprocess.check_output(
        ["git", "status", "--porcelain", "--", str(config_path.relative_to(ROOT))],
        cwd=ROOT, text=True,
    ).strip())
    provenance["config_dirty"] = config_dirty
    if (config_dirty or provenance["runtime_modules_dirty"]
            or not provenance["runtime_modules_match_expected_commit"]):
        raise RuntimeError("D5.2 freezer runtime is not frozen")

    selection_row = next(
        row for row in summary["candidate_rows"]
        if row["candidate_id"] == selected
    )
    return {
        "status": "REV10D5_2_SPATIAL_OU_CONFIRMATION_LIBRARY_FROZEN",
        "scientific_role": EXPECTED_ROLE,
        "candidate_set": {
            "candidates": [by_id[value] for value in frozen_ids],
            "n_exact_off": 1,
            "n_local": 1,
            "n_matched_permuted": 1,
        },
        "selection_freeze": {
            "rule": verdict["selection_rule"],
            "selected_nonzero_candidate_id": selected,
            "matched_permuted_candidate_id": permuted,
            "selected_sigma_rate_per_ms": verdict[
                "selected_sigma_rate_per_ms"
            ],
            "selection_row": selection_row,
            "source_bracket_verdict": {
                "path": str(verdict_path.relative_to(ROOT)),
                "sha256": _sha256(verdict_path),
            },
            "confirmation_networks_were_read": False,
            "claim_role": "fresh-network development confirmation for Fig.4",
        },
        "direction_classifier": source["direction_classifier"],
        "fixed_contract": {
            "node_anchor": config["node_anchor"],
            "confirmation_network_seeds": config["search"][
                "confirmation_network_seeds"
            ],
            "acceptance": config["search"]["acceptance"],
            "static_edge_coefficients": "all exact zero",
            "beta": "closed",
        },
        "inputs": config["inputs"],
        "config": {
            "path": str(config_path.relative_to(ROOT)),
            "sha256": _sha256(config_path),
        },
        "provenance": provenance,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--out")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    output = Path(
        args.out or ROOT / config["output_root"] / "candidate_manifest.json"
    )
    payload = build_manifest(args.config, args.expected_commit)
    atomic_write_json(payload, output)
    print(json.dumps({
        "status": payload["status"],
        "selected": payload["selection_freeze"][
            "selected_nonzero_candidate_id"
        ],
        "matched_permuted": payload["selection_freeze"][
            "matched_permuted_candidate_id"
        ],
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
