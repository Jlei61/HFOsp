"""Adjudicate the D6.2 continuous two-direction response surface."""
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


DEFAULT_CONFIG = ROOT / "config/topic4_rev10_d6_2_joint_continuous_field_surface.json"


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


def _density_support(row):
    return int(sum(
        seed_row["natural_kmeans"].get(
            "heldout_gmm_k2_minus_k1_loglik_per_event", -np.inf,
        ) is not None
        and seed_row["natural_kmeans"].get(
            "heldout_gmm_k2_minus_k1_loglik_per_event", -np.inf,
        ) > 0
        for seed_row in row["natural_kmeans_by_network"].values()
    ))


def _joint_signal(contrast, density_support, n_runaway):
    natural = contrast.get("natural_alignment_delta")
    crossfit = contrast.get("crossfit_margin_delta")
    return bool(
        natural is not None and crossfit is not None
        and natural["network_bootstrap_q05"] > 0
        and crossfit["network_bootstrap_q05"] > 0
        and int(density_support) >= 4 and int(n_runaway) == 0
    )


def _pareto(rows):
    """Mark descriptive response-space nondominance; no scalar objective."""
    output = {}
    for candidate_id, row in rows.items():
        value = np.asarray([
            row["natural_mean_delta"], row["crossfit_mean_delta"],
            row["recruitment_mean_improvement"], row["density_support_fraction"],
        ], float)
        output[candidate_id] = not any(
            np.all(other >= value) and np.any(other > value)
            for other_id, other in (
                (other_id, np.asarray([
                    other_row["natural_mean_delta"],
                    other_row["crossfit_mean_delta"],
                    other_row["recruitment_mean_improvement"],
                    other_row["density_support_fraction"],
                ], float))
                for other_id, other_row in rows.items()
                if other_id != candidate_id
            )
        )
    return output


def _diagnostic_display_candidate(rows):
    """Choose a plot candidate only; this is not a scientific winner rule."""
    eligible = [
        row for row in rows
        if row["n_runaway_networks"] == 0
        and row["paired_contrast"]["natural_alignment_delta"] is not None
        and row["paired_contrast"]["crossfit_margin_delta"] is not None
        and row["recruitment_worst_mode_error"] is not None
    ]
    supported = [row for row in eligible if row["k2_over_k1_positive_networks"] >= 4]
    pool = supported or eligible
    if not pool:
        return None
    return max(pool, key=lambda row: (
        min(
            row["paired_contrast"]["natural_alignment_delta"][
                "network_bootstrap_q05"
            ],
            row["paired_contrast"]["crossfit_margin_delta"][
                "network_bootstrap_q05"
            ],
        ),
        -row["recruitment_worst_mode_error"],
        row["candidate_id"],
    ))["candidate_id"]


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
    if manifest.get("status") != "REV10D6_2_JOINT_CONTINUOUS_FIELD_SURFACE_FROZEN":
        raise RuntimeError("D6.2 manifest is not frozen")
    if summary.get("status") != "REV10D6_2_RETURNED_ONLY_JOINT_SURFACE_COMPLETE":
        raise RuntimeError("D6.2 aggregate is incomplete")
    contract = json.loads((ROOT / config["inputs"]["contact_contract"]["path"]).read_text())
    aggregate = {row["candidate_id"]: row for row in summary["candidate_rows"]}
    metrics = []
    for candidate in manifest["candidate_set"]["candidates"]:
        row = _candidate_metrics(
            config_path, root, candidate, contract,
            config["search"]["kmeans_selection"],
        )
        row["d6_2_latent_coordinates"] = candidate["d6_2_latent_coordinates"]
        row["d6_2_role"] = candidate["d6_2_role"]
        row["n_runaway_networks"] = aggregate[candidate["candidate_id"]][
            "n_runaway_networks"
        ]
        metrics.append(row)
    by_id = {row["candidate_id"]: row for row in metrics}
    baseline = by_id["edge_noop"]
    seeds = config["search"]["confirmation_network_seeds"]
    response_rows, response_space = [], {}
    for candidate in manifest["candidate_set"]["candidates"]:
        candidate_id = candidate["candidate_id"]
        if candidate_id == "edge_noop":
            continue
        row = by_id[candidate_id]
        contrast = paired_contrast(row, baseline, seeds)
        density = _density_support(row)
        natural = contrast["natural_alignment_delta"]
        crossfit = contrast["crossfit_margin_delta"]
        recruitment = contrast["recruitment_error_improvement"]
        response_space[candidate_id] = {
            "natural_mean_delta": (
                natural["equal_network_mean"] if natural is not None else -np.inf
            ),
            "crossfit_mean_delta": (
                crossfit["equal_network_mean"] if crossfit is not None else -np.inf
            ),
            "recruitment_mean_improvement": (
                recruitment["equal_network_mean"]
                if recruitment is not None else -np.inf
            ),
            "density_support_fraction": density / len(seeds),
        }
        response_rows.append({
            "candidate_id": candidate_id,
            "d6_2_latent_coordinates": candidate["d6_2_latent_coordinates"],
            "d6_2_role": candidate["d6_2_role"],
            "paired_contrast": contrast,
            "k2_over_k1_positive_networks": density,
            "n_runaway_networks": row["n_runaway_networks"],
            "recruitment_worst_mode_error": row["recruitment_worst_mode_error"],
            "joint_signal": _joint_signal(
                contrast, density, row["n_runaway_networks"],
            ),
        })
    nondominated = _pareto(response_space)
    for row in response_rows:
        row["response_space_pareto_nondominated"] = nondominated[
            row["candidate_id"]
        ]
    joint = [row["candidate_id"] for row in response_rows if row["joint_signal"]]
    display = _diagnostic_display_candidate(response_rows)
    status = (
        "REV10D6_2_JOINT_CONTINUOUS_FIELD_SIGNAL_OBSERVED"
        if joint else "REV10D6_2_JOINT_CONTINUOUS_FIELD_SIGNAL_NOT_OBSERVED"
    )
    payload = {
        "status": status,
        "joint_signal_candidate_ids": joint,
        "diagnostic_display_candidate_id": display,
        "candidate_rows": metrics,
        "response_surface_rows": response_rows,
        "baseline_candidate_id": "edge_noop",
        "baseline_metrics": baseline,
        "joint_signal_definition": config["search"]["kmeans_selection"][
            "joint_signal_rule"
        ],
        "diagnostic_display_rule": config["search"]["kmeans_selection"][
            "diagnostic_display_rule"
        ],
        "response_surface_is_exploratory": True,
        "no_scalar_winner_objective": True,
        "recruitment_is_reported_not_a_hard_gate": True,
        "network_seed_is_the_independent_unit": True,
        "complete_patient_distribution_acceptance": "NOT_TESTED_BY_D6_2",
        "fig4_acceptance": "DIAGNOSTIC_ONLY",
        "claim_boundary": (
            "fresh development networks test a fixed two-direction local subspace "
            "of one continuous field; no patient-blind generalization, complete "
            "interictal-distribution, causal core, edge, beta, optimizer, or "
            "ictal-lifecycle claim"
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
        "joint_signal_candidate_ids": joint,
        "diagnostic_display_candidate_id": display,
    }, indent=2))


if __name__ == "__main__":
    main()
