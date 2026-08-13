"""Exploratory paired audit for learned h with and without transferred Z/M."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_topic4_rev10_d5_2_spatial_ou_confirmation import (  # noqa: E402
    _patient_matched_benchmark,
)
from scripts.paper_figures.plot_fig4_spatial_edge_flow_validation import (  # noqa: E402
    _canonical_rank_kmeans,
    _column_stats,
    _load_bundle,
    _patient_profiles,
    _similarity,
    normalize_event_ranks,
)
from scripts.run_topic4_rev9l_forced_source_worker import _sha256  # noqa: E402
from src.topic4_core_field_runner import atomic_write_json  # noqa: E402


DEFAULT_CONFIG = ROOT / "config/topic4_rev10_zm1_data_driven_h_zm.json"


def _patient_geometry(bundle):
    clean_index = np.flatnonzero(bundle["clean"])
    labels = np.asarray(bundle["labels"][clean_index], int)
    ranks = normalize_event_ranks(bundle["ranks"][clean_index])
    model = np.asarray([
        _column_stats(ranks[labels == mode])[0] for mode in (0, 1)
    ])
    return _similarity(model, _patient_profiles(bundle)[0])


def _slow_state(root, candidate_id, seeds):
    rows = []
    for seed in seeds:
        path = root / "workers" / f"{candidate_id}_seed_{seed}.json"
        payload = json.loads(path.read_text())
        rows.append({
            "seed": int(seed),
            "runaway_early_stop_ms": payload["run"]["runaway_early_stop_ms"],
            "n_detected_events": payload["run"]["n_common_detector_events"],
            "n_returned_events": payload["run"]["n_returned_events"],
            **payload.get("mz_slow_state", {}),
        })
    numeric = {}
    for key in (
        "final_z_mean", "minimum_z", "final_m_mean", "maximum_m",
        "peak_mean_adaptation_current", "mean_fraction_above_z_threshold",
    ):
        values = [row[key] for row in rows if row.get(key) is not None]
        numeric[key] = {
            "mean_across_networks": float(np.mean(values)) if values else None,
            "minimum_across_networks": float(np.min(values)) if values else None,
            "maximum_across_networks": float(np.max(values)) if values else None,
        }
    return {
        "by_network": rows,
        "equal_network_summary": numeric,
        "n_runaway_networks": int(sum(
            row["runaway_early_stop_ms"] is not None for row in rows
        )),
    }


def _arm_audit(config_path, root, candidate_id, row):
    bundle = _load_bundle(
        config_path, root, candidate_id, allow_exploratory_candidate=True,
    )
    matrix = _patient_geometry(bundle)
    output = {
        "candidate_id": candidate_id,
        "activity": row,
        "supervised_direction_vs_patient_spearman": matrix.tolist(),
        "slow_state": _slow_state(
            root, candidate_id, bundle["config"]["search"][
                "confirmation_network_seeds"
            ],
        ),
    }
    try:
        canonical = _canonical_rank_kmeans(bundle)
        selected = canonical["clean_global_index"]
        direction = np.asarray(canonical["direction"], int)
        output["canonical_fig4c_kmeans"] = {
            "status": "EVALUABLE",
            "n_events": int(len(selected)),
            "cluster_counts": canonical["cluster_counts"].tolist(),
            "direction_contingency": canonical[
                "direction_contingency"
            ].tolist(),
            "direction_purity": float(canonical["direction_purity"]),
            "within_cluster_tau_mean": float(
                canonical["within_cluster_tau_mean"]
            ),
            "kmeans_stability_ami_median": float(
                canonical["stability_ami_median"]
            ),
            "silhouette_median": float(canonical["silhouette_median"]),
            "patient_matched_direction_purity": _patient_matched_benchmark(
                bundle, direction, draws=256, seed=20260813,
            ),
        }
    except (RuntimeError, ValueError) as error:
        output["canonical_fig4c_kmeans"] = {
            "status": "NOT_EVALUABLE", "reason": str(error),
        }
    return output


def _delta(active, control, key):
    active_value = active["activity"].get(key)
    control_value = control["activity"].get(key)
    if active_value is None or control_value is None:
        return None
    return float(active_value - control_value)


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
    if manifest.get("status") != "REV10ZM1_H_PLUS_ZM_LIBRARY_FROZEN":
        raise RuntimeError("ZM1 manifest is invalid")
    if summary.get("status") != "REV10R_RETURNED_ONLY_CONFIRMATION_COMPLETE":
        raise RuntimeError("ZM1 equal-network aggregation is incomplete")
    rows = {row["candidate_id"]: row for row in summary["candidate_rows"]}
    control = _arm_audit(
        config_path, root, "h_spou_slow_off", rows["h_spou_slow_off"],
    )
    active = _arm_audit(
        config_path, root, "h_spou_zm_transfer", rows["h_spou_zm_transfer"],
    )
    payload = {
        "status": "REV10ZM1_EXPLORATORY_H_PLUS_ZM_TRANSFER_COMPLETE",
        "scientific_role": config["scientific_role"],
        "control": control,
        "z_plus_m": active,
        "paired_equal_network_deltas_zm_minus_slow_off": {
            key: _delta(active, control, key) for key in (
                "mean_network_detected_events_descriptive",
                "mean_network_returned_events_scored",
                "mean_network_fraction_time_above_detector",
                "mean_network_peak_active_fraction",
                "mean_network_shape_A", "mean_network_shape_B",
                "mean_network_ood_fraction",
                "networks_with_both_clean_modes",
            )
        },
        "interpretation": {
            "z_m_were_dynamically_engaged": (
                active["slow_state"]["equal_network_summary"][
                    "mean_fraction_above_z_threshold"
                ]["mean_across_networks"] not in (None, 0.0)
                and active["slow_state"]["equal_network_summary"][
                    "peak_mean_adaptation_current"
                ]["maximum_across_networks"] not in (None, 0.0)
            ),
            "exploratory_only": True,
            "no_new_blocker_or_gate": True,
            "claim_boundary": config["claim_boundary"],
        },
        "inputs": {
            "config": {"path": str(config_path.relative_to(ROOT)),
                       "sha256": _sha256(config_path)},
            "manifest": {"path": str(manifest_path.relative_to(ROOT)),
                         "sha256": _sha256(manifest_path)},
            "summary": {"path": str(summary_path.relative_to(ROOT)),
                        "sha256": _sha256(summary_path)},
        },
    }
    output = root / "zm_transfer_audit.json"
    atomic_write_json(payload, output)
    print(json.dumps({
        "status": payload["status"],
        "z_m_were_dynamically_engaged": payload["interpretation"][
            "z_m_were_dynamically_engaged"
        ],
        "paired_deltas": payload[
            "paired_equal_network_deltas_zm_minus_slow_off"
        ],
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
