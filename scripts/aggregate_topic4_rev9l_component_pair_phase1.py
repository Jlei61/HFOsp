"""Aggregate L2 component-pair finite-difference fit-network responses."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.getcwd())
from scripts.run_topic4_core_field_stage3_profile_round1 import axial_map  # noqa: E402
from scripts.run_topic4_rev9l_objective_replay import (  # noqa: E402
    _load_reference,
    _patient_training_arrays,
)
from src.topic4_core_field_profile import fit_profile_modes  # noqa: E402
from src.topic4_core_field_runner import atomic_write_json  # noqa: E402
from src.topic4_forced_source_capacity import (  # noqa: E402
    source_mode_correlation_summary,
)
from src.topic4_mode_learnability import (  # noqa: E402
    mode_conditioned_descriptor_replay,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = "config/topic4_rev9l_component_pair_edge.json"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _slug(value):
    return str(value).lower().replace("+", "_")


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
        raise RuntimeError("L2 aggregator differs from the launcher commit")
    return {
        "git_commit_at_aggregation": current,
        "expected_git_commit": expected,
        "runtime_modules_dirty": False,
        "runtime_modules_match_expected_commit": True,
        "runtime_module_sha256": hashes,
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "systemd_unit": os.environ.get("REV9L_SYSTEMD_UNIT"),
    }


def _summary(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"median": None, "q05": None, "q95": None, "n": 0}
    return {
        "median": float(np.median(values)),
        "q05": float(np.quantile(values, 0.05)),
        "q95": float(np.quantile(values, 0.95)),
        "n": int(len(values)),
    }


def _load_workers(config, config_path, expected_commit):
    root = Path(config["output_root"]) / "phase1"
    config_sha = _sha256(config_path)
    expected = subprocess.check_output(
        ["git", "rev-parse", expected_commit], cwd=ROOT, text=True).strip()
    output, inputs = {}, []
    contact_names = None
    for candidate in config["component_pair_family"]["phase1_candidates"]:
        candidate_id = candidate["candidate_id"]
        rows, arrays, diagnostics = [], [], []
        for seed in config["network_seeds"]["fit"]:
            stem = root / "workers" / f"{candidate_id}_seed{seed}"
            json_path, npz_path = stem.with_suffix(".json"), stem.with_suffix(".npz")
            payload = json.loads(json_path.read_text())
            if (payload["status"] != "REV9L_FORCED_SOURCE_WORKER_COMPLETE"
                    or payload["scientific_role"] != config["scientific_role"]
                    or payload["edge_family"] != "component_pair_residual_target_normalized"
                    or payload["candidate_id"] != candidate_id
                    or not np.allclose(payload["component_pair_gamma"], candidate["gamma"])
                    or int(payload["seed"]) != int(seed)):
                raise RuntimeError(f"L2 worker identity mismatch: {json_path}")
            provenance = payload["provenance"]
            if (payload["config"]["sha256"] != config_sha
                    or provenance.get("expected_git_commit") != expected
                    or provenance.get("runtime_modules_dirty")
                    or not provenance.get("runtime_modules_match_expected_commit")
                    or payload["arrays"]["sha256"] != _sha256(npz_path)):
                raise RuntimeError(f"L2 worker provenance mismatch: {json_path}")
            with np.load(npz_path, allow_pickle=False) as loaded:
                names = np.asarray(loaded["contact_names"]).astype(str)
                if contact_names is None:
                    contact_names = names
                elif not np.array_equal(contact_names, names):
                    raise RuntimeError("L2 contact order changed")
                record = {key: np.asarray(loaded[key]) for key in (
                    "source_ids", "rank_curves", "contact_ranks",
                    "assigned_ood", "assigned_distance_to_A_B",
                )}
            for index, row in enumerate(payload["runs"]):
                if str(record["source_ids"][index]) != row["source_id"]:
                    raise RuntimeError("L2 JSON/NPZ source order changed")
                rows.append({"seed": int(seed), **row})
                arrays.append({key: value[index] for key, value in record.items()
                               if key != "source_ids"})
            diagnostics.append(payload["network"]["edge_diagnostics"])
            inputs.append({
                "candidate_id": candidate_id, "seed": int(seed),
                "json": {"path": str(json_path), "sha256": _sha256(json_path)},
                "npz": {"path": str(npz_path), "sha256": _sha256(npz_path)},
            })
        output[candidate_id] = {
            "gamma": candidate["gamma"], "rows": rows, "arrays": arrays,
            "edge_diagnostics": diagnostics,
        }
    return output, contact_names, inputs


def _patient(config, contact_names):
    reference = _load_reference(config["inputs"]["profile_reference"]["path"])
    patient = _patient_training_arrays(reference, list(contact_names))
    modes = fit_profile_modes(patient["curves"], reference)
    if modes.get("status") != "ok":
        raise RuntimeError("patient training modes are no longer reconstructable")
    with np.load(config["inputs"]["patient_training_target"]["path"],
                 allow_pickle=False) as loaded:
        prototypes = np.asarray(loaded["patient_train_mode_prototypes"], float)
        counts = np.asarray(loaded["patient_train_mode_counts"], int)
    if (not np.allclose(modes["prototypes"], prototypes, atol=1e-7)
            or not np.array_equal(modes["cluster_counts"], counts)):
        raise RuntimeError("patient training target changed")
    return reference, patient, np.asarray(modes["labels"], int), prototypes


def _candidate_summary(data, config, reference, patient, patient_labels,
                       prototypes):
    curves = np.asarray([row["rank_curves"] for row in data["arrays"]], float)
    ranks = np.asarray([row["contact_ranks"] for row in data["arrays"]], float)
    sources = np.asarray([row["source_id"] for row in data["rows"]])
    source_order = config["packet"]["formal_sources"]
    correlations = source_mode_correlation_summary(
        curves, sources, prototypes, source_order=source_order)
    mapping = {
        config["primary_mapping"]["mode_A_source"]: 0,
        config["primary_mapping"]["mode_B_source"]: 1,
    }
    use, labels = [], []
    for index, source in enumerate(sources):
        if source in mapping and np.isfinite(curves[index]).all():
            use.append(index)
            labels.append(mapping[source])
    descriptors = None
    if set(labels) == {0, 1}:
        descriptors = mode_conditioned_descriptor_replay(
            curves[use], ranks[use], labels,
            patient["curves"], patient["ranks"], patient_labels, reference)
    geometry = {}
    for source in source_order:
        selected = [row for row in data["rows"] if row["source_id"] == source]
        geometry[source] = {
            "downstream_positive_spike_mass": _summary([
                row["paired_geometry"]["downstream_positive_spike_mass"]
                for row in selected]),
            "r90_mm": _summary([
                row["paired_geometry"]["r90_mm"] for row in selected]),
            "curve_usable_fraction": float(np.mean([
                row["paired_excess_readout"]["curve_usable"] for row in selected])),
            "ood_fraction": float(np.mean([
                row["paired_excess_readout"]["ood"] for row in selected])),
        }
    structural = {
        "edge_ratio_min": _summary([
            row["edge_ratio"]["min"] for row in data["edge_diagnostics"]]),
        "edge_ratio_p99": _summary([
            row["edge_ratio"]["p99"] for row in data["edge_diagnostics"]]),
        "edge_ratio_max": _summary([
            row["edge_ratio"]["max"] for row in data["edge_diagnostics"]]),
        "max_abs_incoming_E_error": max(
            row["max_abs_incoming_E_error"] for row in data["edge_diagnostics"]),
        "all_topology_unchanged": all(
            row["topology_unchanged"] for row in data["edge_diagnostics"]),
        "all_e_to_i_unchanged": all(
            row["e_to_i_unchanged"] for row in data["edge_diagnostics"]),
        "all_gaba_unchanged": all(
            row["gaba_unchanged"] for row in data["edge_diagnostics"]),
    }
    return {
        "gamma": data["gamma"],
        "source_mode_correlation": correlations,
        "mode_descriptors": descriptors,
        "geometry": geometry,
        "structural": structural,
        "n_pretrigger_mismatch": int(sum(
            not row["pretrigger_spikes_bit_identical"] for row in data["rows"])),
        "n_runaway": int(sum(
            row["runaway_early_stop_ms"] is not None for row in data["rows"])),
    }


def _metric_vector(candidate):
    names = [
        f"{mode}_{metric}" for mode in ("A", "B") for metric in (
            "prototype_spearman", "recruitment_mae", "precedence_mae",
            "rank_profile_mae", "event_sliced_wasserstein")
    ]
    if candidate["mode_descriptors"] is None:
        return {name: float("nan") for name in names}
    modes = candidate["mode_descriptors"]["modes"]
    output = {}
    for mode in ("A", "B"):
        row = modes[mode]
        output[f"{mode}_prototype_spearman"] = row["curve_prototype_spearman"]
        output[f"{mode}_recruitment_mae"] = row[
            "recruitment_mean_absolute_error"]
        output[f"{mode}_precedence_mae"] = row["precedence_mean_absolute_error"]
        output[f"{mode}_rank_profile_mae"] = row[
            "mean_rank_profile_absolute_error"]
        output[f"{mode}_event_sliced_wasserstein"] = row[
            "event_distribution_sliced_wasserstein"]
    return output


def _finite_differences(candidates, config):
    epsilon = float(config["component_pair_family"]["epsilon"])
    pair_ids = {
        "gamma_c1_from_c1": ("g_c1_c1_p", "g_c1_c1_m"),
        "gamma_c1_from_c2": ("g_c1_c2_p", "g_c1_c2_m"),
        "gamma_c2_from_c1": ("g_c2_c1_p", "g_c2_c1_m"),
        "gamma_c2_from_c2": ("g_c2_c2_p", "g_c2_c2_m"),
        "gamma_bg_from_c1": ("g_bg_c1_p", "g_bg_c1_m"),
        "gamma_bg_from_c2": ("g_bg_c2_p", "g_bg_c2_m"),
    }
    metrics = {candidate_id: _metric_vector(row)
               for candidate_id, row in candidates.items()}
    derivatives = {}
    for parameter, (positive, negative) in pair_ids.items():
        derivatives[parameter] = {
            key: float((metrics[positive][key] - metrics[negative][key])
                       / (2.0 * epsilon))
            for key in metrics[positive]
        }
    return {"epsilon": epsilon, "candidate_metrics": metrics,
            "central_derivatives": derivatives}


def _plot(candidates, finite, config, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    parameters = config["component_pair_family"]["gamma_order"]
    metrics = list(next(iter(finite["central_derivatives"].values())))
    matrix = np.asarray([
        [finite["central_derivatives"][parameter][metric] for parameter in parameters]
        for metric in metrics
    ])
    scale = np.nanmax(np.abs(matrix), initial=1.0)
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.8), constrained_layout=True)
    image = axes[0].imshow(matrix, cmap="RdBu_r", vmin=-scale, vmax=scale,
                           aspect="auto")
    axes[0].set_xticks(range(len(parameters)), [value.replace("_", "\n")
                                                for value in parameters])
    axes[0].set_yticks(range(len(metrics)), [value.replace("_", " ")
                                             for value in metrics], fontsize=7)
    axes[0].set_title("A  Local response Jacobian", loc="left", weight="bold")
    fig.colorbar(image, ax=axes[0], shrink=0.75, label="central derivative")

    for index, (candidate_id, row) in enumerate(candidates.items()):
        metric = finite["candidate_metrics"][candidate_id]
        axes[1].scatter(metric["A_prototype_spearman"],
                        metric["B_prototype_spearman"], s=42,
                        label=candidate_id if index < 5 else None)
        axes[1].annotate(candidate_id, (metric["A_prototype_spearman"],
                                        metric["B_prototype_spearman"]),
                         xytext=(3, 2), textcoords="offset points", fontsize=6)
    axes[1].set_xlabel("patient A prototype Spearman")
    axes[1].set_ylabel("patient B prototype Spearman")
    axes[1].set_title("B  Candidate mode profiles", loc="left", weight="bold")

    ids = list(candidates)
    p99 = [candidates[value]["structural"]["edge_ratio_p99"]["median"]
           for value in ids]
    maximum = [candidates[value]["structural"]["edge_ratio_max"]["median"]
               for value in ids]
    axes[2].plot(range(len(ids)), p99, marker="o", label="edge ratio p99")
    axes[2].plot(range(len(ids)), maximum, marker="s", label="edge ratio max")
    axes[2].set_xticks(range(len(ids)), ids, rotation=55, ha="right", fontsize=7)
    axes[2].set_ylabel("new / old E-to-E weight")
    axes[2].set_title("C  Structural perturbation", loc="left", weight="bold")
    axes[2].legend(frameon=False, fontsize=7)
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"rev9l_l2_component_pair_phase1.{suffix}", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text())
    for name, record in config["inputs"].items():
        if _sha256(record["path"]) != record["sha256"]:
            raise RuntimeError(f"L2 input hash changed: {name}")
    if (config["interpretation"].get("no_formal_acceptance_gate") is not True
            or config["interpretation"].get("beta_opened") is not False):
        raise RuntimeError("L2 exploratory or beta boundary changed")
    workers, contact_names, worker_inputs = _load_workers(
        config, config_path, args.expected_commit)
    reference, patient, patient_labels, prototypes = _patient(
        config, contact_names)
    candidates = {
        candidate_id: _candidate_summary(
            data, config, reference, patient, patient_labels, prototypes)
        for candidate_id, data in workers.items()
    }
    finite = _finite_differences(candidates, config)
    all_structural = all(
        row["structural"]["all_topology_unchanged"]
        and row["structural"]["all_e_to_i_unchanged"]
        and row["structural"]["all_gaba_unchanged"]
        for row in candidates.values())
    summary = {
        "status": "REV9L_L2_COMPONENT_PAIR_PHASE1_COMPLETE",
        "scientific_role": config["scientific_role"],
        "claim_boundary": (
            "finite-difference fit-network response audit; no candidate selected, "
            "no selection/confirmation networks, no patient held-out"
        ),
        "candidates": candidates,
        "finite_difference": finite,
        "all_structural_contracts_preserved": bool(all_structural),
        "next_task": "run the frozen 64-point six-dimensional Sobol exploration",
        "patient_heldout_scores_computed": False,
        "worker_inputs": worker_inputs,
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "provenance": _provenance(args.expected_commit),
    }
    output_root = Path(config["output_root"]) / "phase1"
    output_path = output_root / "phase1_summary.json"
    atomic_write_json(summary, output_path)
    figures = output_root / "figures"
    _plot(candidates, finite, config, figures)
    (figures / "README.md").write_text(
        "### rev9l_l2_component_pair_phase1.png\n"
        "左图是六个 component-pair residual 参数对 A/B 五类 mode descriptor 的中心有限差分；中图直接显示 13 个候选的患者训练集 A/B prototype correlation；右图报告对应 edge ratio 扰动。所有结果来自相同六个 fit networks 和 paired forced sources。\n\n"
        "**关注点**：是否存在能连续改善 mode A 且不损害 mode B 的参数方向，以及这种方向是否只靠过大的 edge 权重畸变产生。\n"
    )
    decision_path = Path(config["output_root"]).parent / "decision.json"
    decision = json.loads(decision_path.read_text())
    decision["status"] = "L2_PHASE1_COMPLETE_RESPONSE_CANDIDATES_REQUIRED"
    decision["propagation_family"]["component_pair_phase1"] = {
        "status": summary["status"],
        "all_structural_contracts_preserved": bool(all_structural),
        "summary_path": str(output_path),
        "next_task": summary["next_task"],
    }
    decision["patient_heldout_scores_computed"] = False
    decision["l2_phase1_provenance"] = summary["provenance"]
    atomic_write_json(decision, decision_path)
    print(json.dumps({
        "status": summary["status"],
        "n_candidates": len(candidates),
        "all_structural_contracts_preserved": all_structural,
        "patient_heldout_scores_computed": False,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
