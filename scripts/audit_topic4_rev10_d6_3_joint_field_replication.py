"""Adjudicate the fresh-network D6.3 joint-field replication."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_topic4_rev10_d6_1_natural_kmeans_closeout import (  # noqa: E402
    paired_contrast,
)
from scripts.rescore_topic4_rev10_d6_natural_kmeans import (  # noqa: E402
    _candidate_metrics,
    _jsonable,
)
from scripts.run_topic4_rev9l_forced_source_worker import _sha256  # noqa: E402


DEFAULT_CONFIG = ROOT / "config/topic4_rev10_d6_3_joint_field_replication.json"


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        Path(temporary).write_text(json.dumps(
            _jsonable(payload), indent=2, sort_keys=True, allow_nan=False,
        ))
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def density_support(row):
    return int(sum(
        seed_row["natural_kmeans"].get(
            "heldout_gmm_k2_minus_k1_loglik_per_event", -np.inf,
        ) is not None
        and seed_row["natural_kmeans"].get(
            "heldout_gmm_k2_minus_k1_loglik_per_event", -np.inf,
        ) > 0
        for seed_row in row["natural_kmeans_by_network"].values()
    ))


def replication_pass(contrast, *, density_count, n_runaway):
    natural = contrast.get("natural_alignment_delta")
    crossfit = contrast.get("crossfit_margin_delta")
    return bool(
        natural is not None and crossfit is not None
        and natural["network_bootstrap_q05"] > 0
        and crossfit["network_bootstrap_q05"] > 0
        and int(density_count) >= 8 and int(n_runaway) == 0
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    root = ROOT / config["output_root"]
    manifest_path = root / "candidate_manifest.json"
    summary_path = root / "confirmation_summary_returned_only.json"
    manifest = json.loads(manifest_path.read_text())
    summary = json.loads(summary_path.read_text())
    if manifest.get("status") != "REV10D6_3_JOINT_FIELD_REPLICATION_PAIR_FROZEN":
        raise RuntimeError("D6.3 manifest is not frozen")
    if summary.get("status") != "REV10D6_3_RETURNED_ONLY_REPLICATION_COMPLETE":
        raise RuntimeError("D6.3 aggregate is incomplete")
    contract = json.loads((ROOT / config["inputs"]["contact_contract"]["path"]).read_text())
    aggregate = {row["candidate_id"]: row for row in summary["candidate_rows"]}
    metrics = {}
    for candidate in manifest["candidate_set"]["candidates"]:
        row = _candidate_metrics(
            config_path, root, candidate, contract,
            config["search"]["kmeans_replication"],
        )
        row["n_runaway_networks"] = aggregate[candidate["candidate_id"]][
            "n_runaway_networks"
        ]
        metrics[candidate["candidate_id"]] = row
    baseline_id = manifest["selection_freeze"]["baseline_candidate_id"]
    candidate_id = manifest["selection_freeze"]["primary_candidate_id"]
    baseline, candidate = metrics[baseline_id], metrics[candidate_id]
    seeds = config["search"]["confirmation_network_seeds"]
    contrast = paired_contrast(candidate, baseline, seeds)
    candidate_density = density_support(candidate)
    baseline_density = density_support(baseline)
    passed = replication_pass(
        contrast, density_count=candidate_density,
        n_runaway=candidate["n_runaway_networks"],
    )
    status = (
        "REV10D6_3_JOINT_CONTINUOUS_FIELD_REPLICATION_PASS"
        if passed else "REV10D6_3_JOINT_CONTINUOUS_FIELD_NOT_REPLICATED"
    )
    payload = {
        "status": status,
        "replication_candidate_id": candidate_id,
        "baseline_candidate_id": baseline_id,
        "diagnostic_display_candidate_id": candidate_id,
        "replication_pass": passed,
        "paired_candidate_minus_warm_baseline": contrast,
        "candidate_k2_over_k1_positive_networks": candidate_density,
        "baseline_k2_over_k1_positive_networks": baseline_density,
        "candidate_metrics": candidate,
        "baseline_metrics": baseline,
        "replication_rule": config["search"]["kmeans_replication"][
            "replication_rule"
        ],
        "D6_2_networks_excluded_from_replication_gate": True,
        "recruitment_is_reported_not_a_hard_gate": True,
        "network_seed_is_the_independent_unit": True,
        "complete_patient_distribution_acceptance": "NOT_TESTED_BY_D6_3",
        "fig4_acceptance": "DIAGNOSTIC_ONLY",
        "claim_boundary": (
            "twelve fresh development networks replicate one frozen continuous-field "
            "candidate relative to warm baseline; no patient-blind generalization, "
            "complete interictal-distribution, core, edge, beta, optimizer, slow-variable, "
            "or ictal-lifecycle claim"
        ),
        "inputs": {
            "config": {"path": str(config_path.relative_to(ROOT)), "sha256": _sha256(config_path)},
            "manifest": {"path": str(manifest_path.relative_to(ROOT)), "sha256": _sha256(manifest_path)},
            "summary": {"path": str(summary_path.relative_to(ROOT)), "sha256": _sha256(summary_path)},
            "analysis_code": {"path": str(Path(__file__).resolve().relative_to(ROOT)), "sha256": _sha256(Path(__file__).resolve())},
        },
        "analysis_git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        ).strip(),
    }
    output = root / "confirmation_verdict.json"
    _atomic_json(output, payload)
    print(json.dumps({
        "status": status,
        "replication_pass": passed,
        "candidate_k2_over_k1_positive_networks": candidate_density,
        "diagnostic_display_candidate_id": candidate_id,
    }, indent=2))


if __name__ == "__main__":
    main()
