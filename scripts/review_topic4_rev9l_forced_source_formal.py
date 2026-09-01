"""Write the scientific L1 interpretation from frozen formal-fit artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.getcwd())
from src.topic4_core_field_runner import atomic_write_json  # noqa: E402
from src.topic4_forced_source_capacity import (  # noqa: E402
    source_mode_correlation_summary,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = "config/topic4_rev9l_forced_source_formal.json"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _provenance(expected_commit):
    paths = set()
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if not filename:
            continue
        path = Path(filename).resolve()
        if path.suffix != ".py":
            continue
        try:
            paths.add(str(path.relative_to(ROOT)))
        except ValueError:
            continue
    paths.add(str(Path(__file__).resolve().relative_to(ROOT)))
    paths = sorted(paths)
    expected = subprocess.check_output(
        ["git", "rev-parse", expected_commit], cwd=ROOT, text=True).strip()
    current = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--", *paths], cwd=ROOT,
        text=True).strip()
    hashes = {path: _sha256(ROOT / path) for path in paths}
    expected_hashes = {
        path: hashlib.sha256(subprocess.check_output(
            ["git", "show", f"{expected}:{path}"], cwd=ROOT)).hexdigest()
        for path in paths
    }
    match = all(hashes[path] == expected_hashes[path] for path in paths)
    if dirty or not match:
        raise RuntimeError("L1 review modules differ from the review commit")
    return {
        "git_commit_at_review": current,
        "expected_git_commit": expected,
        "runtime_modules_dirty": False,
        "runtime_modules_match_expected_commit": True,
        "runtime_module_sha256": hashes,
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
    }


def _mode_descriptor_delta(summary, arm, mode, key):
    baseline = summary["arms"]["Null"]["primary_mode_descriptors"]["modes"][mode][key]
    value = summary["arms"][arm]["primary_mode_descriptors"]["modes"][mode][key]
    return None if baseline is None or value is None else float(value - baseline)


def _injection_frame_sensitivity(config, prototypes):
    root = Path(config["output_root"]) / "formal_fit" / "workers"
    output = {}
    for arm in config["arms"]:
        slug = arm.lower().replace("+", "_")
        sources, primary, inclusive = [], [], []
        for seed in config["network_seeds"]["fit"]:
            with np.load(root / f"{slug}_seed{seed}.npz", allow_pickle=False) as loaded:
                sources.extend(np.asarray(loaded["source_ids"]).astype(str).tolist())
                primary.extend(np.asarray(loaded["rank_curves"], float))
                inclusive.extend(np.asarray(
                    loaded["inclusive_packet_frame_rank_curves"], float))
        excluded = source_mode_correlation_summary(
            primary, sources, prototypes,
            source_order=config["packet"]["formal_sources"])
        included = source_mode_correlation_summary(
            inclusive, sources, prototypes,
            source_order=config["packet"]["formal_sources"])
        output[arm] = {}
        for source, target in (
                (config["primary_mapping"]["mode_A_source"], 0),
                (config["primary_mapping"]["mode_B_source"], 1)):
            left = np.asarray(excluded["sources"][source][
                "per_network_correlation_to_A_B"], float)
            right = np.asarray(included["sources"][source][
                "per_network_correlation_to_A_B"], float)
            difference = np.abs(left - right)
            output[arm][source] = {
                "target_mode": "A" if target == 0 else "B",
                "excluded_target_median": float(np.nanmedian(left[:, target])),
                "inclusive_target_median": float(np.nanmedian(right[:, target])),
                "max_abs_change_across_A_B_and_networks": float(
                    np.nanmax(difference, initial=0.0)),
            }
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text())
    for name, record in config["inputs"].items():
        if _sha256(record["path"]) != record["sha256"]:
            raise RuntimeError(f"L1 review input hash changed: {name}")
    root = Path(config["output_root"])
    summary_path = root / "formal_fit" / "forced_source_capacity_summary.json"
    summary = json.loads(summary_path.read_text())
    if summary["status"] != "REV9L_L1_FORCED_FIT_COMPLETE":
        raise RuntimeError("L1 formal summary is incomplete")
    if summary["provenance"]["expected_git_commit"] != \
            "73b916e1fdd7d4db911638fdff10065acdfc9822":
        raise RuntimeError("L1 worker/aggregator commit changed")
    with np.load(config["inputs"]["patient_training_target"]["path"],
                 allow_pickle=False) as loaded:
        prototypes = np.asarray(loaded["patient_train_mode_prototypes"], float)

    lower_better = (
        "recruitment_mean_absolute_error",
        "precedence_mean_absolute_error",
        "mean_rank_profile_absolute_error",
        "event_distribution_sliced_wasserstein",
    )
    edge_a_deltas = {
        key: _mode_descriptor_delta(summary, "Edge", "A", key)
        for key in lower_better
    }
    edge_a_deltas["curve_prototype_spearman"] = _mode_descriptor_delta(
        summary, "Edge", "A", "curve_prototype_spearman")
    edge_improves_a = (
        edge_a_deltas["curve_prototype_spearman"] > 0.0
        or any(edge_a_deltas[key] < 0.0 for key in lower_better)
    )
    source_specificity = {}
    for arm in config["arms"]:
        source_rows = summary["arms"][arm]["source_mode_correlation"]["sources"]
        source_specificity[arm] = {
            source: {
                "n_usable": int(source_rows[source]["n_usable"]),
                "n_total": int(source_rows[source]["n_total"]),
            }
            for source in config["packet"]["formal_sources"]
        }

    node_edge_a = summary["arms"]["Node+Edge"]["primary_source_margins"][
        config["primary_mapping"]["mode_A_source"]]["per_network"]
    node_edge_b = summary["arms"]["Node+Edge"]["primary_source_margins"][
        config["primary_mapping"]["mode_B_source"]]["per_network"]
    unstable = [
        int(seed) for seed, a, b in zip(
            config["network_seeds"]["fit"], node_edge_a, node_edge_b)
        if a is None or b is None or a <= 0.0 or b <= 0.0
    ]
    sensitivity = _injection_frame_sensitivity(config, prototypes)
    paired = summary["paired_factorial_by_source"]
    review = {
        "status": "L1_SOURCE_LOCATION_SPECIFIC_BASELINE_SCAFFOLD_DOMINATED",
        "scientific_role": "post-aggregation interpretation; no new simulation",
        "safe_claim": (
            "forcing the two learned component locations selects two directional "
            "responses on the frozen scaffold, but scalar node/edge modulation does "
            "not reproduce the weak patient mode A profile"
        ),
        "source_location_specificity": source_specificity,
        "absolute_mode_descriptors_by_arm": {
            arm: summary["arms"][arm]["primary_mode_descriptors"]["modes"]
            for arm in config["arms"]
        },
        "scalar_edge_vs_null_mode_A_deltas": edge_a_deltas,
        "scalar_edge_improves_any_mode_A_descriptor": bool(edge_improves_a),
        "scalar_edge_factorial_effects": {
            source: {
                metric: paired[source][metric]["delta_edge"]
                for metric in (
                    "intended_minus_cross_spearman",
                    "downstream_positive_spike_mass", "r90_mm")
            }
            for source in ("component_1", "component_2")
        },
        "node_edge_unstable_fit_seeds": unstable,
        "packet_frame_sensitivity": sensitivity,
        "interpretation": {
            "ignition": (
                "forced source locations can trigger returned propagation; "
                "spontaneous ignition and occupancy remain untested"
            ),
            "node": (
                "mixed small mode-A descriptor changes, with no prototype improvement"
            ),
            "scalar_edge": (
                "conditional gain/localization modifier: more downstream spikes and "
                "slightly smaller r90, without improved mode-A pattern"
            ),
            "node_edge": (
                "not shared-stable across all six fit networks"
            ),
        },
        "next_recommendation": "RUN_L2_COMPONENT_PAIR_EDGE_ORACLE",
        "beta_status": "DO_NOT_OPEN_UNTIL_A_RADIAL_SCALE_DEFECT_IS_ISOLATED",
        "patient_heldout_scores_computed": False,
        "inputs": {
            "formal_summary": {
                "path": str(summary_path), "sha256": _sha256(summary_path)},
            "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        },
        "provenance": _provenance(args.expected_commit),
    }
    output_path = root / "formal_fit" / "l1_scientific_review.json"
    atomic_write_json(review, output_path)

    decision_path = root.parent / "decision.json"
    decision = json.loads(decision_path.read_text())
    decision["status"] = "L1_COMPLETE_L2_COMPONENT_PAIR_ORACLE_REQUIRED"
    decision["ignition"] = {
        "status": "FORCED_SOURCE_LOCATION_CAN_TRIGGER_RETURNED_PROPAGATION",
        "spontaneous_ignition_tested": False,
        "interpretation": review["interpretation"]["ignition"],
    }
    decision["propagation_family"] = {
        "status": "BASELINE_SCAFFOLD_SOURCE_LOCATION_DOMINATES_SCALAR_NODE_EDGE",
        "scalar_edge_pattern_capacity": "MODE_A_IMPROVEMENT_NOT_OBSERVED",
        "node_edge_shared_stability": (
            "UNSTABLE_ON_FIT_SEED_" + "_".join(map(str, unstable))
            if unstable else "NO_DIRECTIONAL_FAILURE_IN_SIX_FIT_NETWORKS"),
        "formal_acceptance_gate_applied": False,
        "next_recommendation": review["next_recommendation"],
        "review_path": str(output_path),
    }
    decision["network_realization"] = {
        "status": "FIT_NETWORK_SENSITIVITY_OBSERVED_NOT_YET_ORACLE_QUANTIFIED",
        "unstable_node_edge_fit_seeds": unstable,
        "next_task": "L3 after L2 candidate exists",
    }
    decision["patient_heldout_scores_computed"] = False
    decision["l1_scientific_review_provenance"] = review["provenance"]
    atomic_write_json(decision, decision_path)
    print(json.dumps({
        "status": review["status"],
        "scalar_edge_improves_any_mode_A_descriptor": edge_improves_a,
        "node_edge_unstable_fit_seeds": unstable,
        "next_recommendation": review["next_recommendation"],
        "patient_heldout_scores_computed": False,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
