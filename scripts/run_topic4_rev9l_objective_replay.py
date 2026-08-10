"""Run the zero-simulation rev9-L objective and target-reliability audit."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.getcwd())
from scripts.calibrate_topic4_core_field_stage3_joint_observable import (  # noqa: E402
    HELD_OUT_FRAC,
    SPLIT_SEED,
)
from scripts.run_topic4_core_field_stage3_profile_round1 import (  # noqa: E402
    PATIENT,
    axial_map,
)
from src.interictal_propagation import load_subject_propagation_events  # noqa: E402
from src.topic4_core_field_profile import (  # noqa: E402
    fit_profile_modes,
    normalized_rank_curve,
    profile_grid,
    split_by_block,
    transform_rank_curves,
)
from src.topic4_core_field_runner import atomic_write_json, provenance  # noqa: E402
from src.topic4_mode_learnability import (  # noqa: E402
    block_mode_reliability,
    candidate_replay_rows,
    centered_smooth_worst,
    correlation_loss,
    dominates,
    mode_conditioned_descriptor_replay,
    pareto_front_indices,
    spearman_association,
)


DEFAULT_CONFIG = "config/topic4_rev9l_mode_learnability.json"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_reference(path):
    with np.load(path, allow_pickle=False) as loaded:
        return {key: np.asarray(loaded[key]) for key in (
            "grid", "center", "components", "score_center", "score_scale",
            "reference_z", "directions",
        )}


def _atomic_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "history_index", "generation", "theta_sha256", "old_joint_loss",
        "global_distance", "mode_a_correlation", "mode_b_correlation",
        "mode_a_loss", "mode_b_loss", "weak_mode_loss", "support_eligible",
        "cluster_counts", "n_usable", "n_detected",
    ]
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".csv")
    os.close(fd)
    try:
        with open(temporary, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({
                    key: (json.dumps(row.get(key)) if isinstance(row.get(key), list)
                          else row.get(key))
                    for key in fields
                })
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _patient_training_arrays(reference, canonical_contacts):
    """Build curves only for frozen training blocks; never score held-out."""
    data = load_subject_propagation_events(PATIENT)
    names = np.asarray([str(value) for value in data["channel_names"]])
    name_to_index = {name: index for index, name in enumerate(names)}
    if set(canonical_contacts) != set(names):
        raise RuntimeError("patient/model contact sets differ")
    reorder = np.asarray([name_to_index[name] for name in canonical_contacts], int)

    block_ids = np.asarray(data["block_ids"])
    train_all, heldout_all = split_by_block(block_ids, HELD_OUT_FRAC, SPLIT_SEED)
    train_blocks = set(block_ids[train_all].tolist())
    heldout_blocks = set(block_ids[heldout_all].tolist())
    if train_blocks & heldout_blocks:
        raise RuntimeError("patient recording-block split leaked")

    axial = axial_map()
    grid = profile_grid(axial)
    if not np.array_equal(grid, np.asarray(reference["grid"])):
        raise RuntimeError("patient curve grid differs from frozen reference")

    curves, ranks, used_blocks, event_indices = [], [], [], []
    raw_ranks = np.asarray(data["ranks"], float)
    raw_bools = np.asarray(data["bools"], bool)
    for event_index in train_all:
        participating = np.flatnonzero(raw_bools[:, event_index])
        rank_dict = {
            names[index]: float(raw_ranks[index, event_index])
            for index in participating
        }
        curve = normalized_rank_curve(rank_dict, axial, grid=grid)
        if curve is None:
            continue
        row = np.full(len(names), np.nan)
        row[participating] = raw_ranks[participating, event_index]
        curves.append(curve)
        ranks.append(row[reorder])
        used_blocks.append(block_ids[event_index])
        event_indices.append(int(event_index))
    return {
        "curves": np.asarray(curves, float),
        "ranks": np.asarray(ranks, float),
        "block_ids": np.asarray(used_blocks),
        "event_indices": np.asarray(event_indices, int),
        "n_excluded_heldout_events": int(len(heldout_all)),
        "n_excluded_heldout_blocks": int(len(heldout_blocks)),
    }


def _selection_rows(selection, tau):
    rows = []
    for rank, candidate in enumerate(selection.get("ranked_candidates", [])):
        metrics = candidate.get("selection_metrics") or {}
        mode = metrics.get("mode") or {}
        matched = mode.get("matched_correlations")
        losses = None if not matched else [correlation_loss(value) for value in matched]
        rows.append({
            "selection_rank": int(rank),
            "candidate_id": candidate.get("candidate_id"),
            "theta_sha256": candidate.get("theta_sha256"),
            "selected": candidate.get("candidate_id") == selection.get("selected_candidate_id"),
            "source_role": candidate.get("source_role"),
            "old_joint_loss": metrics.get("joint_loss"),
            "global_distance": metrics.get("distance"),
            "mode_a_correlation": None if losses is None else float(matched[0]),
            "mode_b_correlation": None if losses is None else float(matched[1]),
            "mode_a_loss": None if losses is None else float(losses[0]),
            "mode_b_loss": None if losses is None else float(losses[1]),
            "weak_mode_loss": None if losses is None else centered_smooth_worst(
                losses, tau=tau),
            "support_eligible": bool(mode.get("support_eligible", False)),
            "cluster_counts": mode.get("cluster_counts"),
        })
    return rows


def _associations(rows):
    eligible = [row for row in rows if row["support_eligible"]
                and row["old_joint_loss"] is not None]
    old = [row["old_joint_loss"] for row in eligible]
    return {
        key: spearman_association(old, [row[key] for row in eligible])
        for key in ("mode_a_loss", "mode_b_loss", "weak_mode_loss")
    }


def _reorder_ranks(ranks, source_names, target_names):
    source_names = [str(value) for value in source_names]
    lookup = {name: index for index, name in enumerate(source_names)}
    if set(source_names) != set(target_names):
        raise RuntimeError("descriptor contact sets differ")
    return np.asarray(ranks, float)[:, [lookup[name] for name in target_names]]


def _full_descriptor_replay(config, reference, patient, patient_labels,
                            canonical_contacts):
    inputs = config["inputs"]
    output = {
        "availability_boundary": (
            "full descriptors require retained per-event curves and ranks; "
            "they are unavailable for the 48 fit candidates"
        ),
        "datasets": {},
    }
    with np.load(inputs["final_event_profiles"], allow_pickle=False) as final:
        final_ranks = np.asarray(final["model_rank_matrix"], float).T
        final_ranks = _reorder_ranks(
            final_ranks, final["contact_names"], canonical_contacts)
        output["datasets"]["rev8_1_final_confirmation"] = (
            mode_conditioned_descriptor_replay(
                final["model_curves"], final_ranks, final["model_labels"],
                patient["curves"], patient["ranks"], patient_labels, reference))

    with np.load(inputs["rev9_factorial_arrays"], allow_pickle=False) as factorial:
        for arm, slug in (("Null", "null"), ("Node", "node"),
                          ("Edge", "edge"), ("Node+Edge", "node_edge")):
            ranks = _reorder_ranks(
                factorial[f"{slug}_ranks"], factorial[f"{slug}_contact_names"],
                canonical_contacts)
            output["datasets"][f"rev9_{slug}"] = (
                mode_conditioned_descriptor_replay(
                    factorial[f"{slug}_curves"], ranks,
                    factorial[f"{slug}_frozen_labels"],
                    patient["curves"], patient["ranks"], patient_labels,
                    reference))
            output["datasets"][f"rev9_{slug}"]["arm"] = arm
            output["datasets"][f"rev9_{slug}"]["assignment"] = (
                "frozen classifier; OOD remains reported in rev9 factorial summary")
    return output


def _plot_objective(rows, selected_index, front, associations, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.8), constrained_layout=True)
    eligible = np.asarray([row["support_eligible"] for row in rows], bool)
    da = np.asarray([row["mode_a_loss"] for row in rows], float)
    db = np.asarray([row["mode_b_loss"] for row in rows], float)
    old = np.asarray([np.nan if row["old_joint_loss"] is None else row["old_joint_loss"]
                      for row in rows], float)
    generation = np.asarray([row["generation"] for row in rows], float)

    ax = axes[0]
    ax.scatter(da[~eligible], db[~eligible], marker="x", color="0.7", label="support不足")
    points = ax.scatter(da[eligible], db[eligible], c=generation[eligible], cmap="viridis",
                        s=34, label="support合格")
    if front:
        order = sorted(front, key=lambda index: da[index])
        ax.plot(da[order], db[order], color="#d1495b", lw=1.2, label="Pareto front")
    ax.scatter(da[selected_index], db[selected_index], marker="*", s=170,
               color="#c23b33", edgecolor="white", linewidth=0.8, label="冻结候选")
    ax.set_xlabel("mode A loss")
    ax.set_ylabel("mode B loss")
    ax.set_title("A  历史候选的双模式前沿", loc="left", weight="bold")
    ax.legend(frameon=False, fontsize=8)
    fig.colorbar(points, ax=ax, label="generation")

    for ax, key, title in zip(
            axes[1:], ("mode_a_loss", "weak_mode_loss"),
            ("B  旧目标几乎不推动 mode A", "C  旧目标主要跟随综合损失")):
        values = np.asarray([row[key] for row in rows], float)
        ax.scatter(old[eligible], values[eligible], c=generation[eligible], cmap="viridis", s=34)
        result = associations[key]
        ax.text(0.04, 0.96, f"Spearman rho={result['rho']:.2f}\np={result['pvalue']:.3g}, n={result['n']}",
                transform=ax.transAxes, ha="left", va="top", fontsize=9)
        ax.set_xlabel("old joint loss")
        ax.set_ylabel(key.replace("_", " "))
        ax.set_title(title, loc="left", weight="bold")
    for suffix in ("png", "pdf"):
        fig.savefig(out_dir / f"rev9l_l0_objective_replay.{suffix}", dpi=300)
    plt.close(fig)


def _plot_reliability(reliability, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.7), constrained_layout=True)
    colors = {"A": "#d1495b", "B": "#277da1"}
    for x, mode in enumerate(("A", "B")):
        rows = reliability["modes"][mode]["block_rows"]
        values = [row["block_to_complement_spearman"] for row in rows]
        axes[0].scatter(np.full(len(values), x), values, color=colors[mode], s=24, alpha=0.75)
        axes[0].plot([x - 0.18, x + 0.18], [np.median(values)] * 2,
                     color="black", lw=1.5)
    axes[0].set_xticks([0, 1], ["mode A", "mode B"])
    axes[0].set_ylabel("block vs complement Spearman")
    axes[0].set_title("A  患者训练模式的分块稳定性", loc="left", weight="bold")

    props = reliability["mode_proportion_by_block"]
    axes[1].scatter(np.arange(len(props)), [row["mode_a_fraction"] for row in props],
                    color="#6a4c93", s=20)
    axes[1].axhline(np.mean([row["mode_a_fraction"] for row in props]),
                    color="0.25", ls="--", lw=1)
    axes[1].set_xlabel("training recording block")
    axes[1].set_ylabel("mode A fraction")
    axes[1].set_title("B  模式占比随记录块变化", loc="left", weight="bold")

    for x, mode in enumerate(("A", "B")):
        rows = reliability["modes"][mode]["block_rows"]
        axes[2].scatter(
            np.full(len(rows), x - 0.08),
            [row["within_block_dispersion"] for row in rows],
            color=colors[mode], marker="o", s=20, alpha=0.65)
        axes[2].scatter(
            np.full(len(rows), x + 0.08),
            [row["between_block_dispersion"] for row in rows],
            color=colors[mode], marker="^", s=20, alpha=0.65)
    axes[2].set_xticks([0, 1], ["mode A", "mode B"])
    axes[2].set_ylabel("distance in frozen embedding")
    axes[2].set_title("C  块内与块间离散", loc="left", weight="bold")
    axes[2].scatter([], [], color="0.4", marker="o", label="within block")
    axes[2].scatter([], [], color="0.4", marker="^", label="between block")
    axes[2].legend(frameon=False, fontsize=8)
    for suffix in ("png", "pdf"):
        fig.savefig(out_dir / f"rev9l_l0_patient_mode_reliability.{suffix}", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text())
    if config["scientific_role"] != "development_oracle_audit_no_patient_blind_claim":
        raise RuntimeError("rev9-L scientific role changed")
    if config["l0"].get("heldout_scores_permitted") is not False:
        raise RuntimeError("L0 must prohibit patient held-out scoring")
    output_root = Path(config["result_root"])
    replay_dir = output_root / "objective_replay"
    figures_dir = replay_dir / "figures"
    replay_dir.mkdir(parents=True, exist_ok=True)

    input_manifest = {
        key: {"path": path, "sha256": _sha256(path)}
        for key, path in config["inputs"].items()
    }
    input_manifest["config"] = {"path": str(config_path), "sha256": _sha256(config_path)}
    tau = float(config["l0"]["weak_mode_temperature"])
    checkpoint = json.loads(Path(config["inputs"]["fit_checkpoint"]).read_text())
    selection = json.loads(Path(config["inputs"]["selection"]).read_text())
    rows = candidate_replay_rows(checkpoint, tau=tau)
    selection_rows = _selection_rows(selection, tau)
    associations = _associations(rows)

    selected_hash = selection["selected_theta_sha256"]
    selected_matches = [index for index, row in enumerate(rows)
                        if row["theta_sha256"] == selected_hash]
    if len(selected_matches) != 1:
        raise RuntimeError("frozen selected theta does not map uniquely to fit history")
    selected_index = selected_matches[0]
    front = pareto_front_indices(rows, require_support=True)
    fit_dominators = [index for index, row in enumerate(rows)
                      if row["support_eligible"]
                      and dominates(row, rows[selected_index])]
    selected_selection = next(row for row in selection_rows if row["selected"])
    selection_dominators = [
        row["candidate_id"] for row in selection_rows
        if row["support_eligible"] and dominates(row, selected_selection)
    ]

    with np.load(config["inputs"]["final_event_profiles"], allow_pickle=False) as final:
        canonical_contacts = [str(value) for value in final["contact_names"]]
    reference = _load_reference(config["inputs"]["profile_reference"])
    patient = _patient_training_arrays(reference, canonical_contacts)
    patient_modes = fit_profile_modes(patient["curves"], reference)
    if patient_modes.get("status") != "ok":
        raise RuntimeError("patient training curves no longer define two modes")
    with np.load(config["inputs"]["patient_training_target"], allow_pickle=False) as target:
        frozen_prototypes = np.asarray(target["patient_train_mode_prototypes"], float)
        frozen_counts = np.asarray(target["patient_train_mode_counts"], int)
    if not np.allclose(patient_modes["prototypes"], frozen_prototypes, atol=1e-7):
        raise RuntimeError("reconstructed patient training prototypes changed")
    if not np.array_equal(patient_modes["cluster_counts"], frozen_counts):
        raise RuntimeError("reconstructed patient training mode counts changed")
    patient_labels = np.asarray(patient_modes["labels"], int)
    reliability = block_mode_reliability(
        patient["curves"], patient["block_ids"], patient_labels,
        embedded=transform_rank_curves(patient["curves"], reference),
        min_events_per_block_mode=config["l0"]["patient_min_events_per_block_mode"],
        bootstrap_seed=config["l0"]["bootstrap_seed"],
        bootstrap_repeats=config["l0"]["bootstrap_repeats"],
    )
    reliability["split"] = {
        "unit": "recording_block", "seed": SPLIT_SEED,
        "heldout_fraction": HELD_OUT_FRAC,
        "n_excluded_heldout_events": patient["n_excluded_heldout_events"],
        "n_excluded_heldout_blocks": patient["n_excluded_heldout_blocks"],
        "heldout_scores_computed": False,
        "heldout_prototypes_computed": False,
    }
    descriptors = _full_descriptor_replay(
        config, reference, patient, patient_labels, canonical_contacts)

    objective_status = (
        "OLD_OBJECTIVE_OR_SELECTION_MISS" if selection_dominators else
        "OBJECTIVE_DOES_NOT_PROTECT_MODE_A" if (
            associations["mode_a_loss"]["rho"] is not None
            and abs(associations["mode_a_loss"]["rho"]) < 0.3) else
        "OBJECTIVE_MODE_A_ASSOCIATION_PRESENT"
    )
    decision = {
        "status": "L0_COMPLETE_L1_REQUIRED",
        "scientific_role": config["scientific_role"],
        "target_objective": {
            "status": objective_status,
            "old_objective_associations": associations,
            "fit_library_dominators_of_selected": fit_dominators,
            "selection_evaluated_dominators_of_selected": selection_dominators,
            "interpretation": (
                "fit-only candidates cannot establish a selection miss; mode A "
                "must be explicitly protected in the next objective"
            ),
        },
        "patient_training_target": {
            "status": "TRAINING_BLOCK_RELIABILITY_MEASURED",
            "mode_a": reliability["modes"]["A"]["block_to_complement_spearman"],
            "mode_b": reliability["modes"]["B"]["block_to_complement_spearman"],
            "heldout_scores_computed": False,
        },
        "ignition": {"status": "not_yet_tested", "next_task": "L1 forced initiation"},
        "propagation_family": {"status": "not_yet_tested", "next_task": "L1"},
        "network_realization": {"status": "not_yet_tested", "next_task": "L3"},
        "optimizer": {"status": "not_yet_tested", "next_task": "L3 after oracle"},
        "identifiability": {"status": "not_yet_tested"},
        "patient_heldout_scores_computed": False,
        "provenance": provenance(),
    }

    _atomic_csv(replay_dir / "candidate_metric_table.csv", rows)
    atomic_write_json({"rows": rows}, replay_dir / "candidate_metric_table.json")
    atomic_write_json(reliability, replay_dir / "patient_mode_reliability.json")
    atomic_write_json({"associations": associations}, replay_dir / "old_vs_new_objective.json")
    atomic_write_json({
        "fit_history_indices": front,
        "rows": [rows[index] for index in front],
    }, replay_dir / "pareto_front.json")
    atomic_write_json({
        "selected_fit_history_index": selected_index,
        "fit_library_dominators": fit_dominators,
        "selection_rows": selection_rows,
        "selection_evaluated_dominators": selection_dominators,
    }, replay_dir / "counterfactual_selection.json")
    atomic_write_json(descriptors, replay_dir / "full_descriptor_replay.json")
    atomic_write_json(input_manifest, replay_dir / "input_manifest.json")
    atomic_write_json(decision, output_root / "decision.json")

    _plot_objective(rows, selected_index, front, associations, figures_dir)
    _plot_reliability(reliability, figures_dir)
    (figures_dir / "README.md").write_text(
        "### rev9l_l0_objective_replay.png\n"
        "展示 48 个 rev8.1 候选在 mode A/B loss 空间的位置、支持度和 Pareto 前沿，并把旧 joint loss 分别与 mode A 和 weakest-mode loss 对照。"
        "该图只使用 checkpoint 真正保留的指标；没有逐事件数组的候选不补算 recruitment 或 precedence。\n\n"
        "**关注点**：旧目标是否实际推动最弱的 mode A，以及冻结候选是否被已评估候选 Pareto 支配。\n\n"
        "### rev9l_l0_patient_mode_reliability.png\n"
        "只使用患者 training recording blocks，逐块比较 mode prototype 与其余训练块的同 mode prototype，并展示 mode 占比与块内/块间离散。"
        "held-out recording 没有计算分数或 prototype。\n\n"
        "**关注点**：mode A 是否比 mode B 更异质，以及这一差异是否足以影响下一轮 weakest-mode objective 的解释。\n"
    )
    print(json.dumps({
        "status": decision["status"],
        "target_objective": objective_status,
        "n_fit_candidates": len(rows),
        "n_fit_support_eligible": sum(row["support_eligible"] for row in rows),
        "selection_dominators": selection_dominators,
        "patient_training_events": reliability["n_events"],
        "patient_training_blocks": reliability["n_blocks"],
    }, indent=2))


if __name__ == "__main__":
    main()
