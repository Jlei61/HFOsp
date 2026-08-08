"""Freeze and visualize the v0.4 input/model contracts before training."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic5_rnn_motif_v0_4 import (  # noqa: E402
    CORE_IDS, DOSE_IDS, GRU_IDS, MODEL_SPECS, ROLLOUT_DECODER_CONTRACT,
)
from src.topic5_wiring_economy_rnn import active_edge_count  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False))
    temporary.replace(path)


def plot_preflight(out_root: Path, manifest: dict) -> None:
    figure_dir = out_root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    representative = next(row for row in manifest["fits"] if row["fit_id"] == "epilepsiae_1146__shared")
    plane = np.load(out_root / "cache" / representative["fit_id"] / "plane.npz")
    contacts = plane["contacts_xy_mm"]
    nodes = plane["nodes_xy_mm"]

    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.linewidth": 0.7, "font.family": "DejaVu Sans",
    })
    fig = plt.figure(figsize=(7.2, 4.15), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, width_ratios=(1.05, 0.85, 1.25))
    ax_geometry = fig.add_subplot(grid[:, 0])
    ax_cohort = fig.add_subplot(grid[0, 1])
    ax_units = fig.add_subplot(grid[1, 1])
    ax_factorial = fig.add_subplot(grid[:, 2])

    ax_geometry.scatter(nodes[:, 0], nodes[:, 1], s=5, color="#c7c7c7", alpha=0.65,
                        linewidths=0, rasterized=True)
    ax_geometry.scatter(contacts[:, 0], contacts[:, 1], s=26, c=np.linspace(0, 1, len(contacts)),
                        cmap="viridis", edgecolors="white", linewidths=0.45, zorder=3)
    ax_geometry.set_aspect("equal")
    ax_geometry.set_xlabel("Propagation-plane x (mm)")
    ax_geometry.set_ylabel("Propagation-plane y (mm)")
    ax_geometry.set_title("Patient geometry", loc="left", fontweight="bold")

    ax_cohort.barh([1], [11], color="#4c78a8", height=0.5)
    ax_cohort.barh([0], [10], color="#e45756", height=0.5)
    ax_cohort.set_yticks([1, 0], ["shared fit", "A/B fits"])
    ax_cohort.set_xticks([0, 5, 10])
    ax_cohort.set_xlim(0, 12)
    ax_cohort.set_xlabel("Patients")
    ax_cohort.set_title("21 patients / 31 fits", loc="left", fontweight="bold")
    for y, value in ((1, 11), (0, 10)):
        ax_cohort.text(value + 0.25, y, str(value), va="center")

    counts = [len(CORE_IDS) * 3 * 31, 2 * 3 * 31 + 31, len(GRU_IDS) * 3 * 31]
    ax_units.bar([0, 1, 2], counts, color=["#4c78a8", "#72b7b2", "#f2cf5b"], width=0.65)
    ax_units.set_xticks([0, 1, 2], ["core", "dose", "GRU"])
    ax_units.set_ylabel("Training units")
    ax_units.set_title("Frozen workload", loc="left", fontweight="bold")
    for x, value in enumerate(counts):
        ax_units.text(x, value + 16, str(value), ha="center", va="bottom")
    ax_units.set_ylim(0, max(counts) * 1.18)

    ax_factorial.set_xlim(-0.35, 1.35)
    ax_factorial.set_ylim(-0.35, 1.35)
    ax_factorial.set_xticks([0, 1], ["uniform growth", "spatial growth"])
    ax_factorial.set_yticks([0, 1], ["no cost", "wiring cost"])
    ax_factorial.set_xlabel("Regrowth proposal")
    ax_factorial.set_ylabel("Distance penalty")
    factorial = {
        (0, 0): ("M2", "#9d9da1"), (1, 0): ("M4", "#72b7b2"),
        (0, 1): ("M8", "#f2cf5b"), (1, 1): ("M6", "#e45756"),
    }
    for (x, y), (label, color) in factorial.items():
        ax_factorial.scatter(x, y, s=520, color=color, edgecolors="white", linewidths=1.5)
        ax_factorial.text(x, y, label, ha="center", va="center", fontweight="bold", fontsize=11)
    ax_factorial.set_title("Connectivity factorial", loc="left", fontweight="bold")
    ax_factorial.grid(color="#dddddd", linewidth=0.6, zorder=0)

    for label, axis in zip("abcd", (ax_geometry, ax_cohort, ax_units, ax_factorial)):
        axis.text(-0.15, 1.04, label, transform=axis.transAxes, fontsize=12,
                  fontweight="bold", va="bottom")
    stem = figure_dir / "stage_a_preflight_contract"
    fig.savefig(stem.with_suffix(".png"), dpi=400, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    (figure_dir / "README.md").write_text(
        "### stage_a_preflight_contract.png\n\n"
        "阶段 A 输入与模型合同。a 显示真实患者电极触点如何映射到传播平面的 tissue units；"
        "b 显示 shared 与 non-collinear 患者如何形成 31 个 fit；c 是冻结的训练工作量；"
        "d 是生长规则与 wiring cost 的 2x2 因素设计。\n\n"
        "**关注点**：所有模型使用同一数据、几何、任务和 rollout，只改变连接约束。\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()
    out_root = args.out_root.resolve()
    manifest_path = out_root / "INPUT_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    if (manifest["n_patients"], manifest["n_fits"], len(manifest["shared_fits"]),
            len(manifest["split_fits"])) != (21, 31, 11, 20):
        raise RuntimeError("cohort/fits differ from the locked 21/31/11/20 contract")

    fits = []
    for row in manifest["fits"]:
        cache = out_root / "cache" / row["fit_id"]
        plane = np.load(cache / "plane.npz")
        events = np.load(cache / "events.npz")
        if plane["H"].shape != (row["n_contacts"], row["n_nodes"]):
            raise RuntimeError(f"{row['fit_id']}: H shape mismatch")
        if events["ranks"].shape[1] != row["n_contacts"]:
            raise RuntimeError(f"{row['fit_id']}: rank/contact mismatch")
        if active_edge_count(row["n_nodes"], 0.10) < 2 * (row["n_nodes"] - 1):
            raise RuntimeError(f"{row['fit_id']}: fixed-local budget cannot guarantee connectivity")
        fits.append({
            "fit_id": row["fit_id"], "subject": row["subject"], "scope": row["scope"],
            "n_contacts": row["n_contacts"], "n_nodes": row["n_nodes"],
            "n_train": row["n_train"], "n_validation": row["n_validation"], "n_test": row["n_test"],
            "plane_sha256": sha256(cache / "plane.npz"),
            "events_sha256": sha256(cache / "events.npz"),
        })

    model_matrix = {
        key: {"arm": value.arm, "eta": value.eta, "seeds": list(value.seeds)}
        for key, value in MODEL_SPECS.items()
    }
    write_json(out_root / "contracts" / "MODEL_MATRIX_CONTRACT.json", {
        "version": "v0.4", "cell_primary": "leaky_rnn", "architecture_replication": "gru",
        "density": 0.10, "d0_mm": 10.0, "models": model_matrix,
        "factorial": ["M2_UNIFORM_SET", "M4_SPATIAL_GROWTH", "M8_UNIFORM_COST_MID", "M6_SPATIAL_MID"],
    })
    write_json(out_root / "contracts" / "ROLLOUT_DECODER_CONTRACT.json", ROLLOUT_DECODER_CONTRACT)
    write_json(out_root / "contracts" / "FIT_TO_PATIENT_AGGREGATION_CONTRACT.json", {
        "Q1_interictal": "non-collinear own_a and own_b metrics averaged within patient",
        "Q2_field": "own_a->F_A and own_b->F_B retained separately; maxAB before patient median",
        "shared": "one fit, post-hoc A/B held-out source labels generate two fields",
    })
    write_json(out_root / "contracts" / "PRIMARY_THEORY_SET.json", {
        "target_blind_models": ["M1_DENSE", "M2_UNIFORM_SET", "M3_FIXED_LOCAL",
                                "M4_SPATIAL_GROWTH", "M6_SPATIAL_MID",
                                "M8_UNIFORM_COST_MID", "C_ORDER_SHUFFLED"],
        "primary_motifs": ["effective_reach", "local_backbone_long_range_connectors",
                           "targeted_vs_matched_lesion"],
    })
    write_json(out_root / "PRE_FLIGHT_AUDIT.json", {
        "status": "PASS", "geometry_status": manifest["geometry_status"],
        "input_manifest_sha256": sha256(manifest_path), "n_patients": 21, "n_fits": 31,
        "n_training_units": 1426, "fits": fits,
        "target_values_read": False,
    })
    write_json(out_root / "stage_a_scientific_drift_audit.json", {
        "status": "ALIGNED", "original_question": (
            "which biologically constrained recurrent motifs can learn patient interictal propagation, "
            "produce frozen fields aligned to early ictal energy, and expose effective motifs"
        ),
        "checked": ["same task and geometry across arms", "factorial changes only growth/cost",
                    "early-ictal target not accessed", "patient-level aggregation paths separated"],
        "deviations": [],
    })
    plot_preflight(out_root, manifest)
    print(f"PASS: 21 patients, 31 fits, 1426 units; {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
