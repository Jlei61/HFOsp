"""Calibrate the Stage 3 joint event-profile observable without new simulation.

The candidate keeps each event's normalized rank profile on a fixed axial grid,
fits an unlabeled embedding on patient training recordings only, and compares
joint clouds with sliced Wasserstein distance. Direction labels and the final
template-opposition gate are not inputs; opposition is computed afterward only
as a falsification diagnostic.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

sys.path.insert(0, os.getcwd())
from scripts.run_topic4_core_field_stage3_profile_round1 import PATIENT, axial_map
from src.interictal_propagation import load_subject_propagation_events
from src.topic4_core_field_profile import (
    OBJECTIVE_N_EVENTS,
    fit_rank_curve_reference,
    normalized_rank_curve,
    profile_grid,
    rank_curve_reference_summary,
    rank_curve_table,
    sliced_rank_curve_distance,
    split_by_block,
    transform_rank_curves,
)
from src.topic4_core_field_runner import _placement, atomic_write_json, provenance


ROOT = "results/topic4_sef_hfo/data_driven_core_field_stage3"
RUN = "results/topic4_sef_hfo/field_swap_subject_snn"
STAGE2 = "results/topic4_sef_hfo/data_driven_core_field"
CELLS = f"{ROOT}/cells/sigma1.2"
SPLIT_SEED, HELD_OUT_FRAC = 20260808, 0.3
CLUSTER_SEED = 20260809
DISTANCE_BOOTSTRAP_SEED = 20260810
N_DISTANCE_BOOTSTRAP = 500
OPPOSITION_MIN_CLUSTER_EVENTS = 10


def _event_curves(events, axial, grid):
    return rank_curve_table(events, axial, grid=grid)


def _patient_curves(axial, grid):
    data = load_subject_propagation_events(PATIENT)
    names = [str(x) for x in data["channel_names"]]
    curves, keep = [], []
    for event_index in range(data["ranks"].shape[1]):
        use = np.flatnonzero(data["bools"][:, event_index])
        ranks = {names[i]: float(data["ranks"][i, event_index]) for i in use}
        curve = normalized_rank_curve(ranks, axial, grid=grid)
        if curve is not None:
            curves.append(curve)
            keep.append(event_index)
    return np.asarray(curves, float), np.asarray(data["block_ids"])[keep]


def _model_curves(paths, axial, grid):
    events = []
    for path in paths:
        events.extend(json.load(open(path)).get("events", []))
    return _event_curves(events, axial, grid)


def _prototype_diagnostic(curves, reference):
    z = transform_rank_curves(curves, reference)
    if len(z) < 8:
        return dict(status="insufficient", n_events=int(len(z)))
    labels = KMeans(n_clusters=2, n_init=50, random_state=CLUSTER_SEED).fit_predict(z)
    prototypes = np.asarray([curves[labels == k].mean(axis=0) for k in (0, 1)])
    # Deterministic display order only; clustering and the distance are unlabeled.
    x = np.arange(prototypes.shape[1], dtype=float)
    order = np.argsort([np.corrcoef(x, p)[0, 1] for p in prototypes])
    prototypes = prototypes[order]
    counts = np.bincount(labels, minlength=2)[order]
    corr = float(np.corrcoef(prototypes[0], prototypes[1])[0, 1])
    return dict(status="ok", n_events=int(len(z)), prototype_correlation=corr,
                cluster_counts=counts.astype(int).tolist(),
                min_cluster_count=int(counts.min()),
                minority_fraction=float(counts.min() / counts.sum()),
                opposition_support_eligible=bool(
                    counts.min() >= OPPOSITION_MIN_CLUSTER_EVENTS),
                prototypes=prototypes)


def _matched_distance(curves, reference, n_events, seed,
                      n_bootstrap=N_DISTANCE_BOOTSTRAP):
    """Finite-sample distance distribution at one common event count."""
    x = np.asarray(curves, float)
    if len(x) < int(n_events):
        raise ValueError(f"need {n_events} curves, got {len(x)}")
    rng = np.random.default_rng(int(seed))
    draws = 1 if len(x) == int(n_events) else int(n_bootstrap)
    values = np.asarray([
        sliced_rank_curve_distance(
            x if len(x) == int(n_events) else
            x[rng.choice(len(x), size=int(n_events), replace=False)],
            reference)
        for _ in range(draws)
    ])
    return dict(n_events=int(n_events), n_bootstrap=int(draws), seed=int(seed),
                median=float(np.median(values)),
                p05=float(np.quantile(values, 0.05)),
                p95=float(np.quantile(values, 0.95)), values=values)


def _nearest_centroid_sensitivity(reference, axial, grid):
    cfg = json.load(open(f"{STAGE2}/config/stage_config.json"))
    sweep = json.load(open(f"{ROOT}/config/sweep_config.json"))
    reg = _placement(cfg)
    center = np.asarray(reg["center"], float)
    axis = np.asarray(reg["axis_unit_vec"], float)
    axial_truth = np.asarray([
        (np.asarray(xy, float) - center) @ axis for xy in sweep["grid"]["centers"]
    ])
    cuts = np.quantile(axial_truth, [1.0 / 3.0, 2.0 / 3.0])
    classes = np.where(axial_truth < cuts[0], 0,
                       np.where(axial_truth > cuts[1], 2, 1))

    runs = []
    for cell_index in range(len(axial_truth)):
        for path in sorted(glob.glob(f"{CELLS}/c{cell_index:03d}_s*.json")):
            record = json.load(open(path))
            curves = _event_curves(record.get("events", []), axial, grid)
            if not len(curves):
                continue
            seed = int(record.get("seed", os.path.basename(path).split("_s")[-1][:-5]))
            mean_z = transform_rank_curves(curves, reference).mean(axis=0)
            runs.append(dict(cell=cell_index, seed=seed, group=int(classes[cell_index]),
                             mean_z=mean_z, n_events=int(len(curves))))

    def score(key, chance):
        correct = eligible = 0
        for held_seed in sorted({row["seed"] for row in runs}):
            train = [row for row in runs if row["seed"] != held_seed]
            test = [row for row in runs if row["seed"] == held_seed]
            keys = sorted({row[key] for row in train})
            centroids = {
                value: np.mean([row["mean_z"] for row in train if row[key] == value], axis=0)
                for value in keys
            }
            for row in test:
                if row[key] not in centroids:
                    continue
                pred = min(keys, key=lambda value: np.linalg.norm(
                    row["mean_z"] - centroids[value]))
                correct += int(pred == row[key])
                eligible += 1
        return dict(accuracy=float(correct / eligible), correct=int(correct),
                    eligible=int(eligible), chance=float(chance))

    return dict(
        statistical_unit="one source-position x network-seed run",
        split="leave one network seed out",
        n_usable_runs=int(len(runs)),
        n_events=int(sum(row["n_events"] for row in runs)),
        axial_tertile_cuts_mm=cuts.tolist(),
        left_middle_right=score("group", 1.0 / 3.0),
        exact_49_cell=score("cell", 1.0 / 49.0),
    )


def _atomic_npz(path, **arrays):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".npz")
    os.close(fd)
    try:
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _plot(summary, diagnostics, grid, out_dir):
    colors = dict(patient_heldout="#333333", hand_placed_two_cores="#277da1",
                  stage2_filament="#43aa8b", stage3_flexible="#c23b33")
    labels = dict(patient_heldout="patient held-out", hand_placed_two_cores="hand-placed cores",
                  stage2_filament="Stage 2 filament", stage3_flexible="Stage 3 flexible")
    order = list(colors)
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 7.6), constrained_layout=True)

    ax = axes[0, 0]
    for row, key in enumerate(order):
        diag = diagnostics[key]
        if diag["status"] != "ok":
            continue
        offset = 2.7 * (len(order) - 1 - row)
        ax.plot(grid, diag["prototypes"][0] + offset, color=colors[key], lw=1.8)
        ax.plot(grid, diag["prototypes"][1] + offset, color=colors[key], lw=1.8,
                ls="--")
        ax.text(grid[0], offset + 0.45, labels[key], color=colors[key], fontsize=9,
                ha="left", va="bottom")
    ax.set_title("A  Unsupervised event-profile prototypes", loc="left", weight="bold")
    ax.set_xlabel("position along the frozen axis (mm)")
    ax.set_yticks([])
    ax.spines[["left", "right", "top"]].set_visible(False)

    ax = axes[0, 1]
    values = [summary["arms"][key]["fixed_count_distance"]["median"]
              for key in order]
    low = [value - summary["arms"][key]["fixed_count_distance"]["p05"]
           for value, key in zip(values, order)]
    high = [summary["arms"][key]["fixed_count_distance"]["p95"] - value
            for value, key in zip(values, order)]
    ax.bar(np.arange(len(order)), values, color=[colors[key] for key in order], width=0.68)
    ax.errorbar(np.arange(len(order)), values, yerr=[low, high], fmt="none",
                color="0.2", capsize=3, lw=1)
    ax.set_xticks(np.arange(len(order)), [labels[key] for key in order], rotation=18,
                  ha="right")
    ax.set_ylabel("joint profile distance (lower is closer)")
    ax.set_title(f"B  Sample-size matched at n={summary['calibration_event_count']}",
                 loc="left", weight="bold")
    ax.spines[["right", "top"]].set_visible(False)

    ax = axes[1, 0]
    corr = [summary["arms"][key]["prototype_correlation"] for key in order]
    ax.barh(np.arange(len(order)), corr, color=[colors[key] for key in order], height=0.62)
    ax.axvline(0.0, color="0.25", lw=0.9)
    ax.set_yticks(np.arange(len(order)), [labels[key] for key in order])
    ax.set_xlim(-1.0, 1.0)
    ax.set_xlabel("correlation between post-hoc K=2 prototypes")
    ax.set_title("C  Opposition is a diagnostic, not the objective", loc="left",
                 weight="bold")
    ax.spines[["right", "top"]].set_visible(False)

    ax = axes[1, 1]
    sens = summary["sensitivity"]
    acc = [sens["left_middle_right"]["accuracy"], sens["exact_49_cell"]["accuracy"]]
    chance = [sens["left_middle_right"]["chance"], sens["exact_49_cell"]["chance"]]
    xx = np.arange(2)
    ax.bar(xx, acc, color=["#577590", "#f9c74f"], width=0.62)
    ax.scatter(xx, chance, color="black", marker="_", s=500, linewidths=2,
               label="chance")
    ax.set_xticks(xx, ["left / middle / right", "exact 49-cell"])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("leave-one-network-seed-out accuracy")
    ax.set_title("D  Known-source sensitivity", loc="left", weight="bold")
    ax.legend(frameon=False, loc="upper right")
    ax.spines[["right", "top"]].set_visible(False)

    stem = os.path.join(out_dir, "figures", "stage3_joint_observable_calibration")
    os.makedirs(os.path.dirname(stem), exist_ok=True)
    fig.savefig(stem + ".png", dpi=220, facecolor="white")
    fig.savefig(stem + ".pdf", facecolor="white")
    plt.close(fig)

    readme = """# Joint observable 图说明

### stage3_joint_observable_calibration

这张图用已有仿真和患者事件标定下一轮目标函数，不包含新仿真。A 展示无监督得到的两条事件剖面原型；B 把四组都配平到 18 个事件后计算不使用方向标签的联合距离，误差线为固定事件数 bootstrap 的 5–95% 区间；C 只在拟合后检查两簇是否相反，不进入目标；D 用已知源位置的 Leg A 扫描检查观测量是否真的携带空间信息。

**关注点**：Stage 3 的一维边缘量虽然较近，但完整事件剖面的联合距离最远，且两条后验原型为正相关；新观测量自然暴露了这一失败，而不是把模板相关门硬塞进目标函数。
"""
    with open(os.path.join(out_dir, "figures", "README.md"), "w") as fh:
        fh.write(readme)
    return stem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{ROOT}/joint_observable")
    args = ap.parse_args()
    axial = axial_map()
    grid = profile_grid(axial)
    patient, blocks = _patient_curves(axial, grid)
    train_index, test_index = split_by_block(blocks, HELD_OUT_FRAC, SPLIT_SEED)
    reference = fit_rank_curve_reference(patient[train_index])

    arm_paths = {
        "hand_placed_two_cores": sorted(glob.glob(
            f"{RUN}/readout_epilepsiae_1146_paired_tsrc_highn_s*_20260721.json")),
        "stage2_filament": sorted(glob.glob(
            f"{RUN}/readout_epilepsiae_1146_learned_core_field_pool_s*.json")),
        "stage3_flexible": [
            f"{RUN}/readout_epilepsiae_1146_stage3_flexible_field_s5.json"],
    }
    curves = {"patient_heldout": patient[test_index]}
    curves.update({key: _model_curves(paths, axial, grid)
                   for key, paths in arm_paths.items()})
    diagnostics = {key: _prototype_diagnostic(value, reference)
                   for key, value in curves.items()}

    calibration_n = int(min(len(value) for value in curves.values()))
    arms = {}
    for arm_index, (key, value) in enumerate(curves.items()):
        diag = diagnostics[key]
        matched = _matched_distance(
            value, reference, calibration_n,
            DISTANCE_BOOTSTRAP_SEED + arm_index)
        matched.pop("values")
        arms[key] = dict(
            n_events=int(len(value)),
            all_events_distance_descriptive=sliced_rank_curve_distance(value, reference),
            fixed_count_distance=matched,
            prototype_correlation=diag.get("prototype_correlation"),
            cluster_counts=diag.get("cluster_counts"),
            min_cluster_count=diag.get("min_cluster_count"),
            minority_fraction=diag.get("minority_fraction"),
            opposition_support_eligible=diag.get(
                "opposition_support_eligible"),
        )
    optimization_floor = _matched_distance(
        curves["patient_heldout"], reference, OBJECTIVE_N_EVENTS,
        DISTANCE_BOOTSTRAP_SEED + len(arms))
    optimization_floor.pop("values")
    sensitivity = _nearest_centroid_sensitivity(reference, axial, grid)
    gates = dict(
        patient_heldout_is_closest_at_matched_n=bool(
            arms["patient_heldout"]["fixed_count_distance"]["median"]
            < min(arms[key]["fixed_count_distance"]["median"]
                  for key in arm_paths)),
        stage3_farther_than_both_opposing_controls_at_matched_n=bool(
            arms["stage3_flexible"]["fixed_count_distance"]["median"]
            > max(arms["hand_placed_two_cores"]["fixed_count_distance"]["median"],
                  arms["stage2_filament"]["fixed_count_distance"]["median"])),
        stage3_positive_controls_negative=bool(
            arms["stage3_flexible"]["prototype_correlation"] > 0
            and arms["patient_heldout"]["prototype_correlation"] < 0
            and arms["hand_placed_two_cores"]["prototype_correlation"] < 0
            and arms["stage2_filament"]["prototype_correlation"] < 0),
        source_position_sensitivity=bool(
            sensitivity["left_middle_right"]["accuracy"] >= 0.60
            and sensitivity["exact_49_cell"]["accuracy"]
            > 5.0 * sensitivity["exact_49_cell"]["chance"]),
    )

    os.makedirs(args.out, exist_ok=True)
    ref_path = os.path.join(args.out, "rank_curve_reference.npz")
    _atomic_npz(ref_path, grid=grid, center=reference["center"],
                components=reference["components"],
                score_center=reference["score_center"],
                score_scale=reference["score_scale"],
                reference_index=reference["reference_index"],
                reference_z=reference["reference_z"],
                directions=reference["directions"],
                explained_variance_ratio=reference["explained_variance_ratio"])
    with open(ref_path, "rb") as fh:
        reference_sha256 = hashlib.sha256(fh.read()).hexdigest()

    summary = dict(
        status=("CANDIDATE_PASSES_EXISTING_ARTIFACT_CALIBRATION"
                if all(gates.values()) else "CANDIDATE_FAILS_CALIBRATION"),
        scientific_role=("candidate optimization observable; final bidirectional-template "
                         "gate remains held out and is not part of the distance"),
        patient_split=dict(unit="recording block", frac=HELD_OUT_FRAC,
                           seed=SPLIT_SEED, n_train=int(len(train_index)),
                           n_heldout=int(len(test_index))),
        reference=rank_curve_reference_summary(reference),
        reference_npz=ref_path,
        reference_sha256=reference_sha256,
        calibration_event_count=calibration_n,
        optimization_event_count=OBJECTIVE_N_EVENTS,
        optimization_patient_floor=optimization_floor,
        arms=arms,
        sensitivity=sensitivity,
        calibration_gates=gates,
        limitations=[
            "Stage 3 calibration uses one existing network and only 18 usable events",
            "fixed-count event bootstrap does not model network-seed clustering",
            "passing calibration licenses a bounded optimizer run, not a mechanism claim",
            "the embedding is subject-specific and must be rebuilt inside each patient training split",
        ],
        inputs=dict(patient=PATIENT, model_paths=arm_paths),
        provenance=provenance(),
    )
    atomic_write_json(summary, os.path.join(args.out, "calibration_summary.json"))
    stem = _plot(summary, diagnostics, grid, args.out)
    print(summary["status"])
    for key, row in arms.items():
        fixed = row["fixed_count_distance"]
        print(f"{key:24s} n={row['n_events']:5d}  matched distance="
              f"{fixed['median']:.3f} [{fixed['p05']:.3f}, {fixed['p95']:.3f}]  "
              f"prototype r={row['prototype_correlation']:+.3f}")
    print(f"3-class accuracy {sensitivity['left_middle_right']['accuracy']:.3f}; "
          f"49-cell accuracy {sensitivity['exact_49_cell']['accuracy']:.3f}")
    print(f"wrote {stem}.png / .pdf and {args.out}/calibration_summary.json")


if __name__ == "__main__":
    main()
