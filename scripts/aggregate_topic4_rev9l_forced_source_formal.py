"""Aggregate rev9-L L1 forced-source fit-network assays."""
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
DEFAULT_CONFIG = "config/topic4_rev9l_forced_source_formal.json"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _runtime_provenance(expected_commit):
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
    current = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    expected = subprocess.check_output(
        ["git", "rev-parse", expected_commit], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--", *paths], cwd=ROOT,
        text=True).strip()
    hashes = {path: _sha256(ROOT / path) for path in paths}
    expected_hashes = {}
    for path in paths:
        content = subprocess.check_output(
            ["git", "show", f"{expected}:{path}"], cwd=ROOT)
        expected_hashes[path] = hashlib.sha256(content).hexdigest()
    match = all(hashes[path] == expected_hashes[path] for path in paths)
    if dirty or not match:
        raise RuntimeError("formal aggregator differs from its launcher commit")
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


def _slug(arm):
    return arm.lower().replace("+", "_")


def _validate_inputs(config):
    for name, record in config["inputs"].items():
        if _sha256(record["path"]) != record["sha256"]:
            raise RuntimeError(f"formal input hash changed: {name}")
    canary = json.loads(Path(
        config["inputs"]["packet_canary_summary"]["path"]).read_text())
    selected = canary["selection"]["selected"]["packet_fraction_of_E"]
    frozen = config["packet"]["frozen_fraction_of_E"]
    if canary["status"] != "PACKET_FRACTION_FROZEN" or not np.isclose(
            selected, frozen):
        raise RuntimeError("formal packet fraction differs from canary lock")


def _load_workers(config, expected_commit):
    formal_dir = Path(config["output_root"]) / "formal_fit"
    config_sha = _sha256(config["_config_path"])
    expected_commit = subprocess.check_output(
        ["git", "rev-parse", expected_commit], cwd=ROOT, text=True).strip()
    expected_sources = config["packet"]["formal_sources"]
    expected_fraction = float(config["packet"]["frozen_fraction_of_E"])
    arms = {}
    worker_inputs = []
    contact_names = None
    envelope_dt = None
    for arm in config["arms"]:
        rows, arrays = [], []
        for seed in config["network_seeds"]["fit"]:
            stem = formal_dir / "workers" / f"{_slug(arm)}_seed{int(seed)}"
            json_path, npz_path = stem.with_suffix(".json"), stem.with_suffix(".npz")
            payload = json.loads(json_path.read_text())
            if (payload["status"] != "REV9L_FORCED_SOURCE_WORKER_COMPLETE"
                    or payload["arm"] != arm or int(payload["seed"]) != int(seed)):
                raise RuntimeError(f"formal worker identity mismatch: {json_path}")
            if payload["config"]["sha256"] != config_sha:
                raise RuntimeError(f"formal worker config hash mismatch: {json_path}")
            if payload["arrays"]["sha256"] != _sha256(npz_path):
                raise RuntimeError(f"formal worker array hash mismatch: {npz_path}")
            if payload["sources"] != expected_sources or len(
                    payload["packet_fractions_of_E"]) != 1 or not np.isclose(
                        payload["packet_fractions_of_E"][0], expected_fraction):
                raise RuntimeError(f"formal source/packet contract changed: {json_path}")
            provenance = payload["provenance"]
            if (provenance.get("expected_git_commit") != expected_commit
                    or provenance.get("runtime_modules_dirty")
                    or not provenance.get("runtime_modules_match_expected_commit")):
                raise RuntimeError(f"formal worker provenance failed: {json_path}")
            with np.load(npz_path, allow_pickle=False) as loaded:
                current_names = np.asarray(loaded["contact_names"]).astype(str)
                current_dt = float(loaded["envelope_dt_ms"])
                if contact_names is None:
                    contact_names = current_names
                    envelope_dt = current_dt
                elif (not np.array_equal(contact_names, current_names)
                      or not np.isclose(envelope_dt, current_dt)):
                    raise RuntimeError("formal contact/envelope contract changed")
                record = {key: np.asarray(loaded[key]) for key in (
                    "source_ids", "rank_curves", "contact_ranks",
                    "assigned_mode", "assigned_ood", "assigned_distance_to_A_B",
                    "excess_contact_envelope",
                )}
            if len(record["source_ids"]) != len(payload["runs"]):
                raise RuntimeError("formal JSON/NPZ row count differs")
            for index, row in enumerate(payload["runs"]):
                if str(record["source_ids"][index]) != row["source_id"]:
                    raise RuntimeError("formal JSON/NPZ source order differs")
                rows.append({"seed": int(seed), **row})
                arrays.append({key: value[index] for key, value in record.items()
                               if key != "source_ids"})
            worker_inputs.append({
                "arm": arm, "seed": int(seed),
                "json": {"path": str(json_path), "sha256": _sha256(json_path)},
                "npz": {"path": str(npz_path), "sha256": _sha256(npz_path)},
            })
        arms[arm] = {"rows": rows, "arrays": arrays}
    return arms, contact_names, float(envelope_dt), worker_inputs


def _patient_target(config, contact_names):
    reference = _load_reference(config["inputs"]["profile_reference"]["path"])
    patient = _patient_training_arrays(reference, list(contact_names))
    patient_modes = fit_profile_modes(patient["curves"], reference)
    if patient_modes.get("status") != "ok":
        raise RuntimeError("patient training curves no longer define two modes")
    with np.load(config["inputs"]["patient_training_target"]["path"],
                 allow_pickle=False) as loaded:
        prototypes = np.asarray(loaded["patient_train_mode_prototypes"], float)
        counts = np.asarray(loaded["patient_train_mode_counts"], int)
    if (not np.allclose(patient_modes["prototypes"], prototypes, atol=1e-7)
            or not np.array_equal(patient_modes["cluster_counts"], counts)):
        raise RuntimeError("patient training target changed")
    return reference, patient, np.asarray(patient_modes["labels"], int), prototypes


def _arm_summary(arm_data, config, reference, patient, patient_labels,
                 prototypes):
    rows, arrays = arm_data["rows"], arm_data["arrays"]
    curves = np.asarray([record["rank_curves"] for record in arrays], float)
    ranks = np.asarray([record["contact_ranks"] for record in arrays], float)
    source_ids = np.asarray([row["source_id"] for row in rows])
    source_order = config["packet"]["formal_sources"]
    correlations = source_mode_correlation_summary(
        curves, source_ids, prototypes, source_order=source_order)

    primary_source_to_mode = {
        config["primary_mapping"]["mode_A_source"]: 0,
        config["primary_mapping"]["mode_B_source"]: 1,
    }
    primary_index, labels = [], []
    for index, source in enumerate(source_ids):
        if source in primary_source_to_mode and np.isfinite(curves[index]).all():
            primary_index.append(index)
            labels.append(primary_source_to_mode[source])
    descriptor = None
    if len(set(labels)) == 2:
        descriptor = mode_conditioned_descriptor_replay(
            curves[primary_index], ranks[primary_index], labels,
            patient["curves"], patient["ranks"], patient_labels, reference)
        descriptor["mode_proportion_note"] = (
            "forced design is source-balanced; mode-proportion JS is not interpreted"
        )

    margins = {}
    for source, intended in ((config["primary_mapping"]["mode_A_source"], 0),
                             (config["primary_mapping"]["mode_B_source"], 1)):
        values = np.asarray(correlations["sources"][source][
            "per_network_correlation_to_A_B"], float)
        margin = values[:, intended] - values[:, 1 - intended]
        margins[source] = {
            "intended_patient_mode": "A" if intended == 0 else "B",
            "intended_minus_cross_correlation": _summary(margin),
            "per_network": [None if not np.isfinite(value) else float(value)
                            for value in margin],
        }

    geometry = {}
    for source in source_order:
        selected = [row for row in rows if row["source_id"] == source]
        geometry[source] = {
            "downstream_positive_spike_mass": _summary([
                row["paired_geometry"]["downstream_positive_spike_mass"]
                for row in selected]),
            "downstream_positive_neurons": _summary([
                row["paired_geometry"]["downstream_positive_neurons"]
                for row in selected]),
            "r50_mm": _summary([
                row["paired_geometry"]["r50_mm"] for row in selected]),
            "r90_mm": _summary([
                row["paired_geometry"]["r90_mm"] for row in selected]),
            "triggered_return_fraction": float(np.mean([
                row["forced_triggered_event"] is not None for row in selected])),
            "curve_usable_fraction": float(np.mean([
                row["paired_excess_readout"]["curve_usable"] for row in selected])),
            "ood_fraction": float(np.mean([
                row["paired_excess_readout"]["ood"] for row in selected])),
        }
    directional = all(
        margins[source]["intended_minus_cross_correlation"]["median"] is not None
        and margins[source]["intended_minus_cross_correlation"]["median"] > 0.0
        for source in margins)
    return {
        "source_mode_correlation": correlations,
        "primary_source_margins": margins,
        "primary_directional_alignment_observed": bool(directional),
        "primary_mode_descriptors": descriptor,
        "source_geometry": geometry,
        "n_pretrigger_mismatch": int(sum(
            not row["pretrigger_spikes_bit_identical"] for row in rows)),
        "n_runaway": int(sum(row["runaway_early_stop_ms"] is not None
                             for row in rows)),
    }


def _plot_capacity(arm_data, summaries, config, prototypes, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    arms = config["arms"]
    sources = config["packet"]["formal_sources"]
    fig, axes = plt.subplots(2, 4, figsize=(13.2, 7.1), constrained_layout=True)
    image_handle = None
    for column, arm in enumerate(arms):
        matrix = np.asarray(summaries[arm]["source_mode_correlation"][
            "median_correlation_matrix"], float)
        ax = axes[0, column]
        image_handle = ax.imshow(matrix, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
        for row in range(len(sources)):
            for mode in range(2):
                value = matrix[row, mode]
                ax.text(mode, row, "NA" if not np.isfinite(value) else f"{value:.2f}",
                        ha="center", va="center", fontsize=7,
                        color="white" if np.isfinite(value) and abs(value) > 0.55
                        else "black")
        ax.set_xticks([0, 1], ["patient A", "patient B"])
        if column == 0:
            ax.set_yticks(range(len(sources)),
                          [source.replace("_", " ") for source in sources])
        else:
            ax.set_yticks([])
        ax.set_title(f"{chr(65 + column)}  {arm}", loc="left", weight="bold")

        bottom = axes[1, column]
        rows = arm_data[arm]["rows"]
        arrays = arm_data[arm]["arrays"]
        for source, target, color, marker in (
                (config["primary_mapping"]["mode_A_source"], 0, "#d1495b", "o"),
                (config["primary_mapping"]["mode_B_source"], 1, "#277da1", "s")):
            values = []
            for row, record in zip(rows, arrays):
                if row["source_id"] != source:
                    continue
                result = source_mode_correlation_summary(
                    np.asarray(record["rank_curves"])[None, :], [source],
                    prototypes, source_order=[source])
                corr = result["sources"][source][
                    "per_network_correlation_to_A_B"][0]
                values.append(np.nan if corr[target] is None else
                              float(corr[target]) - float(corr[1 - target]))
            x = np.arange(len(values)) + (-0.08 if target == 0 else 0.08)
            bottom.scatter(x, values, color=color, marker=marker, s=34,
                           label=f"{source.replace('_', ' ')} to {'A' if target == 0 else 'B'}")
            bottom.plot(x, values, color=color, alpha=0.35, lw=0.8)
        bottom.axhline(0.0, color="0.35", ls=":", lw=1)
        bottom.set_xticks(range(len(config["network_seeds"]["fit"])),
                          config["network_seeds"]["fit"], rotation=45)
        bottom.set_ylim(-2.05, 2.05)
        bottom.set_xlabel("network seed")
        if column == 0:
            bottom.set_ylabel("intended minus cross Spearman")
            bottom.legend(frameon=False, fontsize=7)
    fig.colorbar(image_handle, ax=axes[0, :], shrink=0.72,
                 label="median prototype Spearman")
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"rev9l_l1_source_mode_capacity.{suffix}", dpi=300)
    plt.close(fig)


def _plot_direct_readout(arm_data, config, envelope_dt, output_dir):
    arms = config["arms"]
    primary = ((config["primary_mapping"]["mode_A_source"], "A"),
               (config["primary_mapping"]["mode_B_source"], "B"))
    traces = {}
    maximum = 0.0
    for arm in arms:
        for source, _ in primary:
            selected = [record["excess_contact_envelope"]
                        for row, record in zip(arm_data[arm]["rows"],
                                               arm_data[arm]["arrays"])
                        if row["source_id"] == source]
            value = np.nanmedian(np.asarray(selected, float), axis=0)
            traces[(arm, source)] = value
            maximum = max(maximum, float(np.nanmax(value, initial=0.0)))
    scale = maximum if maximum > 0.0 else 1.0
    offset = 1.15 * scale
    fig, axes = plt.subplots(2, 4, figsize=(13.2, 6.6), sharex=True,
                             constrained_layout=True)
    for row_index, (source, mode) in enumerate(primary):
        for column, arm in enumerate(arms):
            ax = axes[row_index, column]
            value = traces[(arm, source)]
            time_ms = np.arange(value.shape[1]) * envelope_dt
            use = (time_ms >= 90.0) & (time_ms <= 260.0)
            for contact in range(value.shape[0]):
                ax.plot(time_ms[use], value[contact, use] + contact * offset,
                        color="#2f4858", lw=0.75)
            ax.axvline(config["simulation"]["forced_spike_ms"], color="#d1495b",
                       ls="--", lw=0.9)
            ax.axvspan(*config["readout"]["primary_window_ms"], color="#59a14f",
                       alpha=0.07, lw=0)
            ax.set_yticks([])
            ax.set_title(
                f"{chr(65 + row_index * 4 + column)}  {arm}: {source.replace('_', ' ')} to {mode}",
                loc="left", weight="bold", fontsize=9)
            if row_index == 1:
                ax.set_xlabel("time (ms)")
            if column == 0:
                ax.set_ylabel("contacts (offset)")
            if row_index == 0 and column == 3:
                x0 = 238.0
                ax.plot([x0, x0], [0.0, scale], color="black", lw=1.5)
                ax.text(x0 - 2.0, 0.5 * scale, f"{scale:.2g}", ha="right",
                        va="center", fontsize=7)
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"rev9l_l1_direct_electrode_readout.{suffix}", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text())
    config["_config_path"] = str(config_path)
    if config["interpretation"].get("no_formal_acceptance_gate") is not True:
        raise RuntimeError("L1 was changed from exploratory to gate-driven")
    _validate_inputs(config)
    arm_data, contact_names, envelope_dt, worker_inputs = _load_workers(
        config, args.expected_commit)
    reference, patient, patient_labels, prototypes = _patient_target(
        config, contact_names)
    summaries = {
        arm: _arm_summary(arm_data[arm], config, reference, patient,
                          patient_labels, prototypes)
        for arm in config["arms"]
    }
    descriptive_status = {
        arm: ("FORCED_PRIMARY_DIRECTIONAL_ALIGNMENT_OBSERVED"
              if summaries[arm]["primary_directional_alignment_observed"]
              else "FORCED_PRIMARY_DIRECTIONAL_ALIGNMENT_NOT_OBSERVED")
        for arm in config["arms"]
    }
    node_positive = summaries["Node"]["primary_directional_alignment_observed"]
    edge_positive = summaries["Edge"]["primary_directional_alignment_observed"]
    recommendation = (
        "RUN_L2_COMPONENT_PAIR_EDGE_ORACLE" if not edge_positive
        else "SCALAR_EDGE_FORCED_CAPACITY_OBSERVED_REVIEW_BEFORE_L2"
    )
    provenance = _runtime_provenance(args.expected_commit)
    output_root = Path(config["output_root"])
    formal_dir = output_root / "formal_fit"
    summary = {
        "status": "REV9L_L1_FORCED_FIT_COMPLETE",
        "scientific_role": config["scientific_role"],
        "claim_boundary": (
            "forced propagation capacity on six fit networks; not spontaneous, "
            "not patient held-out, not confirmation-network evidence"
        ),
        "packet_fraction_of_E": config["packet"]["frozen_fraction_of_E"],
        "descriptive_arm_status": descriptive_status,
        "arms": summaries,
        "next_recommendation": recommendation,
        "patient_heldout_scores_computed": False,
        "worker_inputs": worker_inputs,
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "provenance": provenance,
    }
    atomic_write_json(summary, formal_dir / "forced_source_capacity_summary.json")
    figures = formal_dir / "figures"
    _plot_capacity(arm_data, summaries, config, prototypes, figures)
    _plot_direct_readout(arm_data, config, envelope_dt, figures)
    (figures / "README.md").write_text(
        "### rev9l_l1_source_mode_capacity.png\n"
        "上排按实验臂展示六个冻结 source 对患者训练集 A/B prototype 的中位 Spearman；下排逐网络展示两个预设 primary source 的 intended-minus-cross margin。source identity 在仿真前冻结，不运行 KMeans，也不按结果重新配对。\n\n"
        "**关注点**：component 2 是否稳定更像 A、component 1 是否稳定更像 B，以及这种方向性在 Node、Edge 与 Node+Edge 间如何改变。\n\n"
        "### rev9l_l1_direct_electrode_readout.png\n"
        "展示 component 2 到 A、component 1 到 B 的 paired forced-minus-sham contact envelope；每条线是一个 contact，按固定幅度间隔错开。红虚线为 100 ms deterministic packet，浅绿色区为冻结的 100-250 ms primary readout。\n\n"
        "**关注点**：响应是否在刺激后形成有时序结构的跨 contact 传播，而不是只有注入瞬间或单点放电。\n"
    )

    decision_path = output_root.parent / "decision.json"
    decision = json.loads(decision_path.read_text())
    decision["status"] = "L1_FORCED_FIT_COMPLETE_NEXT_REVIEW_REQUIRED"
    decision["ignition"] = {
        "status": ("FORCED_CAPACITY_OBSERVED_SPONTANEOUS_GAP_REMAINS"
                   if node_positive else "FORCED_NODE_CAPACITY_NOT_OBSERVED"),
        "interpretation": (
            "forced initiation removes source occupancy from the readout; "
            "spontaneous ignition remains untested in rev9-L"
        ),
    }
    decision["propagation_family"] = {
        "status_by_arm": descriptive_status,
        "formal_acceptance_gate_applied": False,
        "next_recommendation": recommendation,
        "summary_path": str(formal_dir / "forced_source_capacity_summary.json"),
    }
    decision["patient_heldout_scores_computed"] = False
    decision["l1_provenance"] = provenance
    atomic_write_json(decision, decision_path)
    print(json.dumps({
        "status": summary["status"],
        "descriptive_arm_status": descriptive_status,
        "next_recommendation": recommendation,
        "n_worker_inputs": len(worker_inputs),
        "patient_heldout_scores_computed": False,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
