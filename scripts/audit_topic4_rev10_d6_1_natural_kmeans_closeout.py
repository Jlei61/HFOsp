"""Adjudicate D6.1 with paired network-level natural KMeans metrics."""
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

from scripts.rescore_topic4_rev10_d6_natural_kmeans import (  # noqa: E402
    _candidate_metrics,
    _jsonable,
)
from scripts.run_topic4_rev9l_forced_source_worker import _sha256  # noqa: E402
from src.topic4_d6_natural_kmeans import network_bootstrap  # noqa: E402


DEFAULT_CONFIG = ROOT / "config/topic4_rev10_d6_1_natural_kmeans_closeout.json"


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


def _seed_values(row, seed):
    seed_row = row["natural_kmeans_by_network"][str(seed)]
    natural = seed_row["natural_kmeans"]
    recruitment = [
        seed_row["recruitment"][mode]["absolute_error_fraction_of_15"]
        for mode in ("A", "B")
    ]
    return {
        "natural_balanced_alignment": (
            natural.get("direction_balanced_alignment")
            if natural.get("status") == "OK" else None
        ),
        "crossfit_patient_margin": seed_row[
            "crossfit_patient_readout"
        ]["signed_margin"],
        "recruitment_worst_mode_error": (
            max(value for value in recruitment if value is not None)
            if any(value is not None for value in recruitment) else None
        ),
    }


def paired_contrast(candidate, baseline, seeds):
    by_seed, deltas = {}, {"natural": [], "crossfit": [], "recruitment": []}
    for seed in seeds:
        current = _seed_values(candidate, seed)
        warm = _seed_values(baseline, seed)
        row = {"candidate": current, "warm_baseline": warm}
        if current["natural_balanced_alignment"] is not None and warm[
                "natural_balanced_alignment"] is not None:
            row["natural_alignment_delta"] = (
                current["natural_balanced_alignment"]
                - warm["natural_balanced_alignment"]
            )
            deltas["natural"].append(row["natural_alignment_delta"])
        else:
            row["natural_alignment_delta"] = None
        if current["crossfit_patient_margin"] is not None and warm[
                "crossfit_patient_margin"] is not None:
            row["crossfit_margin_delta"] = (
                current["crossfit_patient_margin"]
                - warm["crossfit_patient_margin"]
            )
            deltas["crossfit"].append(row["crossfit_margin_delta"])
        else:
            row["crossfit_margin_delta"] = None
        if current["recruitment_worst_mode_error"] is not None and warm[
                "recruitment_worst_mode_error"] is not None:
            row["recruitment_error_improvement"] = (
                warm["recruitment_worst_mode_error"]
                - current["recruitment_worst_mode_error"]
            )
            deltas["recruitment"].append(row["recruitment_error_improvement"])
        else:
            row["recruitment_error_improvement"] = None
        by_seed[str(seed)] = row
    return {
        "by_seed": by_seed,
        "natural_alignment_delta": network_bootstrap(deltas["natural"]),
        "crossfit_margin_delta": network_bootstrap(deltas["crossfit"]),
        "recruitment_error_improvement": network_bootstrap(deltas["recruitment"]),
        "networks_with_positive_natural_delta": int(sum(
            value > 0 for value in deltas["natural"]
        )),
        "networks_with_positive_crossfit_delta": int(sum(
            value > 0 for value in deltas["crossfit"]
        )),
        "networks_with_positive_recruitment_improvement": int(sum(
            value > 0 for value in deltas["recruitment"]
        )),
        "n_natural_paired_networks": len(deltas["natural"]),
    }


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
    if manifest.get("status") != "REV10D6_1_NATURAL_KMEANS_CLOSEOUT_LIBRARY_FROZEN":
        raise RuntimeError("D6.1 manifest is not frozen")
    if summary.get("status") != "REV10D6_1_RETURNED_ONLY_FRESH_CLOSEOUT_COMPLETE":
        raise RuntimeError("D6.1 aggregate is incomplete")
    contract_path = ROOT / config["inputs"]["contact_contract"]["path"]
    contract = json.loads(contract_path.read_text())
    aggregate = {row["candidate_id"]: row for row in summary["candidate_rows"]}
    rows = []
    for candidate in manifest["candidate_set"]["candidates"]:
        row = _candidate_metrics(
            config_path, root, candidate, contract,
            config["search"]["kmeans_selection"],
        )
        row["selection_roles"] = candidate["fresh_closeout_selection_roles"]
        row["n_runaway_networks"] = aggregate[candidate["candidate_id"]][
            "n_runaway_networks"
        ]
        rows.append(row)
    by_id = {row["candidate_id"]: row for row in rows}
    baseline = by_id["edge_noop"]
    seeds = config["search"]["confirmation_network_seeds"]
    contrasts = {
        candidate_id: paired_contrast(row, baseline, seeds)
        for candidate_id, row in by_id.items() if candidate_id != "edge_noop"
    }
    replicated = [
        candidate_id for candidate_id, contrast in contrasts.items()
        if contrast["n_natural_paired_networks"] >= 4
        and contrast["networks_with_positive_natural_delta"] >= 4
        and by_id[candidate_id]["crossfit_margin_equal_network"] is not None
        and by_id[candidate_id]["crossfit_margin_equal_network"][
            "equal_network_mean"
        ] > 0
        and by_id[candidate_id]["n_runaway_networks"] == 0
    ]
    ranked = sorted(
        [row for row in rows if row["candidate_id"] != "edge_noop"],
        key=lambda row: (
            -(
                row["natural_balanced_alignment_equal_network"] or {}
            ).get("equal_network_mean", -1.0),
            -(
                row["crossfit_margin_equal_network"] or {}
            ).get("equal_network_mean", -1.0),
            row["recruitment_worst_mode_error"]
            if row["recruitment_worst_mode_error"] is not None else np.inf,
            row["candidate_id"],
        ),
    )
    descriptive_best = ranked[0]["candidate_id"] if ranked else None
    primary = manifest["selection_freeze"]["primary_candidate_id"]
    density_support = {
        candidate_id: int(sum(
            seed_row["natural_kmeans"].get(
                "heldout_gmm_k2_minus_k1_loglik_per_event", -np.inf,
            ) is not None
            and seed_row["natural_kmeans"].get(
                "heldout_gmm_k2_minus_k1_loglik_per_event", -np.inf,
            ) > 0
            for seed_row in row["natural_kmeans_by_network"].values()
        ))
        for candidate_id, row in by_id.items()
    }
    patient_aligned = [
        candidate_id for candidate_id, contrast in contrasts.items()
        if contrast["natural_alignment_delta"] is not None
        and contrast["natural_alignment_delta"]["network_bootstrap_q05"] > 0
        and contrast["crossfit_margin_delta"] is not None
        and contrast["crossfit_margin_delta"]["network_bootstrap_q05"] > 0
        and density_support[candidate_id] >= 4
    ]
    status = (
        "REV10D6_1_PATIENT_ALIGNED_NATURAL_REPERTOIRE_CONFIRMED"
        if patient_aligned
        else "REV10D6_1_ORTHOGONAL_PARTIAL_SENSITIVITY_REPERTOIRE_UNRESOLVED"
    )
    payload = {
        "status": status,
        "exploratory_four_of_six_signal_candidate_ids": replicated,
        "patient_aligned_natural_repertoire_candidate_ids": patient_aligned,
        "prefrozen_primary_candidate_id": primary,
        "prefrozen_primary_replication_status": (
            "EXPLORATORY_SIGNAL_RULE_PASS"
            if primary in replicated else "NOT_REPLICATED"
        ),
        "k2_over_k1_positive_network_counts": density_support,
        "minimum_density_support_networks": 4,
        "patient_aligned_natural_repertoire_status": (
            "CONFIRMED" if patient_aligned else "UNRESOLVED"
        ),
        "descriptive_fresh_best_candidate_id": descriptive_best,
        "candidate_rows": rows,
        "paired_candidate_minus_warm_baseline": contrasts,
        "canary_subset_seeds": config["search"]["canary_network_seeds"],
        "extension_subset_seeds": config["search"]["extension_network_seeds"],
        "primary_analysis_uses_natural_event_proportions": True,
        "balanced_mode_purity_is_secondary_only": True,
        "patient_readout_is_contact_split_cross_fitted": True,
        "network_seed_is_the_independent_unit": True,
        "complete_patient_distribution_acceptance": "NOT_TESTED_BY_D6_KMEANS_CLOSEOUT",
        "fig4_acceptance": "DIAGNOSTIC_ONLY",
        "claim_boundary": (
            "fresh development networks test local continuous-field KMeans signal "
            "only; no patient-blind generalization, complete interictal-distribution, "
            "causal core, edge, beta, or ictal-lifecycle claim"
        ),
        "inputs": {
            "config": {"path": str(config_path.relative_to(ROOT)), "sha256": _sha256(config_path)},
            "manifest": {"path": str(manifest_path.relative_to(ROOT)), "sha256": _sha256(manifest_path)},
            "summary": {"path": str(summary_path.relative_to(ROOT)), "sha256": _sha256(summary_path)},
            "analysis_code": {
                "path": str(Path(__file__).resolve().relative_to(ROOT)),
                "sha256": _sha256(Path(__file__).resolve()),
            },
        },
        "analysis_git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        ).strip(),
    }
    output = root / "confirmation_verdict.json"
    _atomic_json(output, payload)
    print(json.dumps({
        "status": status,
        "exploratory_four_of_six_signal_candidate_ids": replicated,
        "patient_aligned_natural_repertoire_candidate_ids": patient_aligned,
        "prefrozen_primary_replication_status": payload[
            "prefrozen_primary_replication_status"
        ],
        "prefrozen_primary_candidate_id": payload["prefrozen_primary_candidate_id"],
        "descriptive_fresh_best_candidate_id": descriptive_best,
    }, indent=2))


if __name__ == "__main__":
    main()
