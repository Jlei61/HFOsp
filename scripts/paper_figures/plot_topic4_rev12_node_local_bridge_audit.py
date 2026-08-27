#!/usr/bin/env python3
"""Plot the static-Node local bridge against the frozen natural-KMeans gate."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
RELATIVE_ROOT = Path(
    "results/topic4_sef_hfo/data_driven_node_dualmode_rev12/"
    "node_stage_al_dual_node_local_bridge"
)
ENDPOINTS = (
    "soft_objective", "mode_0", "mode_1",
    "mode_0_direction", "mode_1_direction",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _align_similarity_matrix(matrix: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Choose the KMeans row labelling that protects the weakest match."""
    candidates = []
    for permutation in ([0, 1], [1, 0]):
        aligned = matrix[permutation, :]
        diagonal = np.diag(aligned)
        cross = np.asarray([aligned[0, 1], aligned[1, 0]])
        score = (float(np.min(diagonal) - np.max(cross)),
                 float(np.mean(diagonal) - np.mean(cross)))
        candidates.append((score, aligned, permutation))
    _, aligned, permutation = max(candidates, key=lambda item: item[0])
    return aligned, list(permutation)


def _point(metadata: dict) -> dict:
    natural = metadata["natural_kmeans"]
    matrix, permutation = _align_similarity_matrix(
        np.asarray(natural["similarity_matrix"], float)
    )
    cross = np.asarray([matrix[0, 1], matrix[1, 0]])
    return {
        "k2_minus_k1": float(
            natural["heldout_gmm_k2_minus_k1_loglik_per_event"]
        ),
        "weakest_patient_diagonal": float(min(matrix[0, 0], matrix[1, 1])),
        "strongest_patient_cross": float(max(cross)),
        "patient_sign_structure": bool(
            np.min(np.diag(matrix)) > 0.0 and np.max(cross) < 0.0
        ),
        "cluster_row_permutation": permutation,
        "direction_balanced_alignment": float(
            natural["direction_balanced_alignment"]
        ),
        "similarity_matrix": matrix.tolist(),
        "n_events": int(natural["n_events"]),
    }


def build_audit(root: Path, result: dict, manual_metadata: dict,
                stage_ak_metadata: dict, *,
                manual_weakest_utility: float | None = None,
                stage_ak_weakest_utility: float | None = None) -> dict:
    result_rows = {row["candidate_id"]: row for row in result["candidates"]}
    rows = []
    for candidate_id in sorted(result_rows):
        path = root / "figures" / candidate_id / "metadata.json"
        metadata = json.loads(path.read_text())
        utilities = result_rows[candidate_id]["relative_to_historical_anchor"]
        point = _point(metadata)
        rows.append({
            "candidate_id": candidate_id,
            "role": "local_bridge",
            **point,
            "weakest_primary_mean_utility": float(min(
                utilities[endpoint]["mean_utility"] for endpoint in ENDPOINTS
            )),
            "all_primary_means_positive": bool(all(
                utilities[endpoint]["mean_utility"] > 0.0
                for endpoint in ENDPOINTS
            )),
            "metadata": {"path": str(path), "sha256": _sha256(path)},
        })
    rows.extend([
        {
            "candidate_id": "manual smooth capacity",
            "role": "nonselectable_capacity_control",
            **_point(manual_metadata),
            "weakest_primary_mean_utility": manual_weakest_utility,
            "all_primary_means_positive": None,
        },
        {
            "candidate_id": "Stage-AK cross",
            "role": "previous_cross_mapping",
            **_point(stage_ak_metadata),
            "weakest_primary_mean_utility": stage_ak_weakest_utility,
            "all_primary_means_positive": None,
        },
    ])
    hidden_passes = [
        row["candidate_id"] for row in rows
        if row["role"] == "local_bridge"
        and row["k2_minus_k1"] > 0.0
        and row["patient_sign_structure"]
    ]
    return {
        "schema_id": "topic4_rev12_nd_node_local_bridge_kmeans_audit_v1",
        "status": (
            "LOCAL_BRIDGE_HIDDEN_KMEANS_PASS_FOUND" if hidden_passes else
            "NO_LOCAL_BRIDGE_CANDIDATE_JOINTLY_SUPPORTS_K2_AND_BOTH_PATIENT_MODES"
        ),
        "hidden_pass_candidate_ids": hidden_passes,
        "rows": rows,
        "gate_semantics": {
            "x_positive": "held-out GMM favors K=2 over K=1",
            "y_positive": "both model clusters correlate positively with their patient mode",
            "patient_sign_structure": "positive matched diagonals and negative crossed cells after the better of the two KMeans row labellings",
            "upper_right": "necessary visual acceptance region; patient sign structure and full endpoints remain required",
        },
    }


def render(audit: dict, output: Path) -> None:
    local = [row for row in audit["rows"] if row["role"] == "local_bridge"]
    x = np.asarray([row["k2_minus_k1"] for row in local])
    y = np.asarray([row["weakest_patient_diagonal"] for row in local])
    color = np.asarray([row["weakest_primary_mean_utility"] for row in local])
    reference_color = [
        float(row["weakest_primary_mean_utility"])
        for row in audit["rows"]
        if row["role"] != "local_bridge"
        and row["weakest_primary_mean_utility"] is not None
    ]
    all_color = np.concatenate([color, np.asarray(reference_color, float)])
    limit = max(abs(float(np.min(all_color))), abs(float(np.max(all_color))), 0.05)

    x_low = min(-110.0, float(np.min(x)) - 5.0)
    x_high = 25.0
    with plt.rc_context({
        "font.size": 8.0, "axes.titlesize": 9.0, "axes.labelsize": 8.0,
        "xtick.labelsize": 7.0, "ytick.labelsize": 7.0,
    }):
        fig, ax = plt.subplots(figsize=(5.4, 3.8), facecolor="white")
    ax.add_patch(Rectangle(
        (0.0, 0.0), x_high, 0.55, facecolor="#E9F3EA",
        edgecolor="none", zorder=0,
    ))
    ax.axvline(0.0, color="#777777", lw=0.8, ls="--")
    ax.axhline(0.0, color="#777777", lw=0.8, ls="--")
    scatter = ax.scatter(
        x, y, c=color, cmap="coolwarm_r", vmin=-limit, vmax=limit,
        s=43, edgecolor="white", linewidth=0.6, zorder=3,
    )
    references = {
        "nonselectable_capacity_control": ("*", "#222222", 105),
        "previous_cross_mapping": ("s", "#777777", 48),
    }
    for row in audit["rows"]:
        if row["role"] not in references:
            continue
        marker, _, size = references[row["role"]]
        utility = row["weakest_primary_mean_utility"]
        ax.scatter(
            row["k2_minus_k1"], row["weakest_patient_diagonal"],
            marker=marker, s=size, c=[utility], cmap="coolwarm_r",
            vmin=-limit, vmax=limit, edgecolor="white",
            linewidth=0.6, zorder=4,
        )
        ax.annotate(
            row["candidate_id"],
            (row["k2_minus_k1"], row["weakest_patient_diagonal"]),
            xytext=(-5, 5), textcoords="offset points", fontsize=6.5,
            ha="right",
        )
    labels = {
        "stage_al_m100_d020": "K=2, wrong modes",
        "stage_al_m100_d050": "best fit",
        "stage_al_m125_d020": "balanced means",
    }
    for row in local:
        if row["candidate_id"] in labels:
            ax.annotate(
                labels[row["candidate_id"]],
                (row["k2_minus_k1"], row["weakest_patient_diagonal"]),
                xytext=(5, -9), textcoords="offset points", fontsize=6.5,
            )
    ax.set(
        xlabel="held-out K=2 evidence\n(K2 - K1 log-likelihood/event)",
        ylabel="weakest model-patient diagonal (Spearman rho)",
        xlim=(x_low, x_high),
        ylim=(-0.82, 0.55),
    )
    ax.set_title(
        "Local Node bridge misses one of the two mode criteria",
        weight="bold", fontsize=9,
    )
    ax.xaxis.label.set_size(8)
    ax.yaxis.label.set_size(8)
    ax.tick_params(labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.05, pad=0.03)
    colorbar.set_label("weakest primary utility", fontsize=7)
    colorbar.ax.tick_params(labelsize=6)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf", ".svg"):
        fig.savefig(output.with_suffix(suffix), dpi=300, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    args = parser.parse_args()
    artifact_root = args.artifact_root.resolve()
    root = artifact_root / RELATIVE_ROOT
    result_path = root / "analysis" / "dual_node_local_bridge_result.json"
    result = json.loads(result_path.read_text())
    manual_path = (
        artifact_root / "results/topic4_sef_hfo/data_driven_node_dualmode_rev12/"
        "node_stage_ad_manual_capacity_replication/figures/"
        "stage_t_manual_smooth_capacity/metadata.json"
    )
    stage_ak_path = (
        artifact_root / "results/topic4_sef_hfo/data_driven_node_dualmode_rev12/"
        "node_stage_ak_dual_node_channel_canary/figures/"
        "stage_ak_mean_g10_p_disp_g08_m/metadata.json"
    )
    manual_audit_path = (
        artifact_root / "results/topic4_sef_hfo/data_driven_node_dualmode_rev12/"
        "node_stage_ad_manual_capacity_replication/analysis/paired_capacity_audit.json"
    )
    stage_ak_result_path = (
        artifact_root / "results/topic4_sef_hfo/data_driven_node_dualmode_rev12/"
        "node_stage_ak_dual_node_channel_canary/analysis/dual_node_channel_result.json"
    )
    manual_audit = json.loads(manual_audit_path.read_text())
    manual_endpoints = manual_audit["paired_comparisons"]["stage_z_g04_m"]["endpoints"]
    manual_weakest = min(
        float(manual_endpoints[key]["mean"])
        for key in ("soft_objective", "mode_0", "mode_1", "causal_direction")
    )
    stage_ak_result = json.loads(stage_ak_result_path.read_text())
    stage_ak_row = next(
        row for row in stage_ak_result["candidates"]
        if row["candidate_id"] == "stage_ak_mean_g10_p_disp_g08_m"
    )
    stage_ak_weakest = min(
        float(stage_ak_row["relative_to_anchor_coupled"][key]["mean_utility"])
        for key in ENDPOINTS
    )
    audit = build_audit(
        root, result, json.loads(manual_path.read_text()),
        json.loads(stage_ak_path.read_text()),
        manual_weakest_utility=manual_weakest,
        stage_ak_weakest_utility=stage_ak_weakest,
    )
    audit["inputs"] = {
        "result": {"path": str(result_path), "sha256": _sha256(result_path)},
        "manual_control": {"path": str(manual_path), "sha256": _sha256(manual_path)},
        "stage_ak_cross": {"path": str(stage_ak_path), "sha256": _sha256(stage_ak_path)},
        "manual_capacity_audit": {
            "path": str(manual_audit_path), "sha256": _sha256(manual_audit_path),
        },
        "stage_ak_result": {
            "path": str(stage_ak_result_path), "sha256": _sha256(stage_ak_result_path),
        },
    }
    audit_path = root / "analysis" / "local_bridge_kmeans_tradeoff.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    stem = root / "figures" / "local_bridge_kmeans_tradeoff"
    render(audit, stem)
    readme = root / "figures" / "README.md"
    readme.write_text("""### local_bridge_kmeans_tradeoff.png

Stage-AL 全部连续 Node 候选的双重验收。横轴检查事件分布是否真的支持 K=2，纵轴检查两个模型簇中较弱者是否仍与对应患者模式正相关；星号是不可选择的手工平滑容量对照，方块是上一轮交叉映射。

**关注点**：局部桥候选没有进入右上区域，说明不存在被软目标漏选的患者对齐双模解；手工容量对照证明同一 SNN 本身仍有表达容量。
""")
    print(json.dumps({
        "status": audit["status"],
        "hidden_pass_candidate_ids": audit["hidden_pass_candidate_ids"],
        "figure": str(stem.with_suffix('.png')),
    }, indent=2))


if __name__ == "__main__":
    main()
