"""Aggregate the rev10-SA non-component continuous-field capacity canary."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.getcwd())
from scripts.aggregate_topic4_rev10_sa_dual_shaft_capacity import (  # noqa: E402
    _atomic_csv,
    _metric,
)
from scripts.build_topic4_rev10_sa_shaft_aware_target import _atomic_json  # noqa: E402
from scripts.rescore_topic4_rev10_sa_historical_artifacts import (  # noqa: E402
    load_scoring_contract,
    score_mode_conditioned_events,
)
from scripts.run_topic4_rev9l_forced_source_worker import (  # noqa: E402
    _load_json_input,
    _runtime_provenance,
    _sha256,
)
from src.topic4_continuous_field import (  # noqa: E402
    continuous_surface,
    distance_to_segments,
)
from src.topic4_core_field_stage3 import params_to_q  # noqa: E402
from src.topic4_shaft_aware import contract_groups, contract_pairs  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config/topic4_rev10_sa_continuous_field_canary.json"


def _worker_complete(payload, npz_path, config_sha, commit, worker_status):
    provenance = payload.get("provenance", {})
    return bool(
        payload.get("status") == worker_status
        and payload.get("config", {}).get("sha256") == config_sha
        and provenance.get("expected_git_commit") == commit
        and provenance.get("runtime_modules_match_expected_commit") is True
        and not provenance.get("runtime_modules_dirty")
        and payload.get("arrays", {}).get("sha256") == _sha256(npz_path)
    )


def _relative_field(candidate, positions, L=20.0):
    if candidate["field_type"] == "gaussian_k3_benchmark":
        q = params_to_q(candidate["theta"], positions, K=3, L=L)
        latent = np.log(np.maximum(q, 1e-12))
    elif candidate["field_type"] == "continuous_bspline":
        latent = continuous_surface(
            candidate["coefficients"], positions,
            n_basis=int(candidate["n_basis"]), degree=int(candidate["degree"]), L=L,
        )
    elif candidate["field_type"] == "continuous_corridor":
        distance = distance_to_segments(positions, candidate["segments"])
        latent = -0.5 * (distance / float(candidate["width_mm"])) ** 2
    else:
        raise ValueError(f"unknown continuous field type: {candidate['field_type']}")
    return np.exp(np.clip(latent - np.max(latent), -30.0, 0.0))


def _plot(summary, manifest, output_root, *, support_control=False):
    contacts = np.asarray(summary["contact_xy_mm"], float)
    shafts = np.asarray(summary["shaft_ids"]).astype(str)
    candidates = {row["candidate_id"]: row
                  for row in manifest["candidate_set"]["candidates"]}
    rows = summary["candidate_rows"]
    row_by_id = {row["candidate_id"]: row for row in rows}
    if support_control:
        disconnected = [row for row in rows
                        if row["representation_role"] == "continuous_disconnected_support"]
        connected = [row for row in rows
                     if row["representation_role"] == "continuous_connected_support"]
        narrow_connected = min(connected, key=lambda row: row["width_mm"])
        broad_connected = max(connected, key=lambda row: row["width_mm"])
        matched_disconnected = min(
            disconnected,
            key=lambda row: abs(row["width_mm"] - narrow_connected["width_mm"]),
        )
        map_ids = [matched_disconnected["candidate_id"],
                   narrow_connected["candidate_id"], broad_connected["candidate_id"]]
        map_titles = [
            f"disconnected support, width {matched_disconnected['width_mm']:.2f} mm",
            f"connected support, width {narrow_connected['width_mm']:.2f} mm",
            f"connected support, width {broad_connected['width_mm']:.2f} mm",
        ]
        palette = {
            "continuous_disconnected_support": "#4E79A7",
            "continuous_connected_support": "#E15759",
        }
        labels = {
            "continuous_disconnected_support": "two shaft paths",
            "continuous_connected_support": "shaft paths + bridge",
        }
    else:
        primary = [row for row in rows if row["representation_role"] == "matched_dof"]
        sensitivity = [row for row in rows if row["representation_role"] == "resolution"]
        primary_best = min(primary, key=lambda row: row["descriptive_loss"])
        sensitivity_best = min(sensitivity, key=lambda row: row["descriptive_loss"])
        map_ids = ["frozen_K3_benchmark", primary_best["candidate_id"],
                   sensitivity_best["candidate_id"]]
        map_titles = ["historical K=3 benchmark", "4x4 continuous matched-DoF",
                      "6x6 continuous sensitivity"]
        palette = {"K3_benchmark": "#7F7F7F", "matched_dof": "#4E79A7",
                   "resolution": "#59A14F"}
        labels = {"K3_benchmark": "historical K=3", "matched_dof": "4x4 continuous",
                  "resolution": "6x6 sensitivity"}
    axis = np.linspace(0.0, 20.0, 120)
    xx, yy = np.meshgrid(axis, axis)
    grid = np.column_stack([xx.ravel(), yy.ravel()])

    fig = plt.figure(figsize=(16.2, 8.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.95])
    for column, (candidate_id, title) in enumerate(zip(map_ids, map_titles)):
        ax = fig.add_subplot(gs[0, column])
        relative = _relative_field(candidates[candidate_id], grid).reshape(xx.shape)
        image = ax.contourf(
            xx, yy, relative, levels=np.linspace(0, 1, 16), cmap="magma",
        )
        for shaft, color in (("ICL", "#198E99"), ("SCL", "#E9822B")):
            selected = shafts == shaft
            ax.plot(contacts[selected, 0], contacts[selected, 1], "o-",
                    color=color, markersize=4, linewidth=1.2, label=shaft)
        ax.set_aspect("equal")
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 20)
        ax.set_title(f"{chr(65 + column)}  {title}", loc="left", weight="bold")
        if support_control:
            path = row_by_id[candidate_id]["field_near_path"]
            bridge_text = (
                f" | bridge h={path['bridge_mean_h']:.2f}"
                if "bridge_mean_h" in path else ""
            )
            ax.text(
                0.02, 0.02, f"actual path h={path['mean_h']:.2f}{bridge_text}",
                transform=ax.transAxes, fontsize=7.5, color="white",
                ha="left", va="bottom",
                bbox={"facecolor": "black", "alpha": 0.55, "pad": 2,
                      "edgecolor": "none"},
            )
        ax.set_xlabel("sheet x (mm)")
        if column == 0:
            ax.set_ylabel("sheet y (mm)")
            ax.legend(frameon=False, fontsize=8)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03, label="relative field")

    ax = fig.add_subplot(gs[1, 0])
    seen = set()
    for row in rows:
        role = row["representation_role"]
        label = None if role in seen else labels[role]
        seen.add(role)
        ax.scatter(
            row["mode_A_ICL_precedence_excess"],
            row["worst_mode_SCL_recruitment_excess"],
            s=35 + 90 * row["forced_both_mode_SCL_network_fraction"],
            color=palette[role], alpha=0.8, edgecolor="white", linewidth=0.5,
            label=label,
        )
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.9)
    ax.set_xlabel("mode A ICL-ICL precedence excess")
    ax.set_ylabel("worst-mode SCL recruitment excess")
    ax.set_title("D  Shaft-aware capacity plane", loc="left", weight="bold")
    ax.legend(frameon=False, fontsize=8)

    shown = sorted(rows, key=lambda row: row["descriptive_loss"])[:10]
    ax = fig.add_subplot(gs[1, 1])
    y = np.arange(len(shown))
    width = 0.24
    ax.barh(y - width, [row["forced_ICL_A_to_SCL_mean_recruited_fraction"]
                        for row in shown], height=width, color="#E15759",
            label="ICL-A to SCL")
    ax.barh(y, [row["forced_ICL_B_to_SCL_mean_recruited_fraction"]
                for row in shown], height=width, color="#F28E2B",
            label="ICL-B to SCL")
    ax.barh(y + width, [row["forced_SCL_to_ICL_mean_recruited_fraction"]
                        for row in shown], height=width, color="#59A14F",
            label="SCL to ICL")
    ax.set_yticks(y, [row["candidate_id"] for row in shown], fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("mean recruited-contact fraction")
    ax.set_title("E  Directed forced response", loc="left", weight="bold")
    ax.legend(frameon=False, fontsize=7)

    ax = fig.add_subplot(gs[1, 2])
    x = np.arange(len(shown))
    ax.bar(x - 0.18, [row["spontaneous_multishaft_fraction"] for row in shown],
           width=0.36, color="#E9822B", label="multishaft events")
    ax.bar(x + 0.18, [row["spontaneous_returned_fraction"] for row in shown],
           width=0.36, color="#76B7B2", label="returned events")
    ax.set_xticks(x, [str(index + 1) for index in range(len(shown))])
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("candidate rank in panel E")
    ax.set_ylabel("spontaneous event fraction")
    ax.set_title("F  Short spontaneous confirmation", loc="left", weight="bold")
    ax.legend(frameon=False, fontsize=8)
    for index, row in enumerate(shown):
        if row["spontaneous_event_count"]:
            ax.text(index + 0.18, min(0.96, row["spontaneous_returned_fraction"]),
                    f"n={row['spontaneous_event_count']}", rotation=90,
                    ha="center", va="top", fontsize=6, color="#2F6F6A")

    phase = "SA6G continuous support control" if support_control else "SA6F continuous field canary"
    fig.suptitle(f"{phase} | {summary['status']}", fontsize=14, weight="bold")
    figure_dir = Path(output_root) / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    stem = figure_dir / (
        "rev10_sa_continuous_support_capacity" if support_control
        else "rev10_sa_continuous_field_capacity"
    )
    fig.savefig(stem.with_suffix(".png"), dpi=240, facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    readme = (
        """### rev10_sa_continuous_support_capacity

这张图比较两条不相连的连续杆支撑与加入最短跨杆桥后的连续场。线段只定义几何容量正控，既不是 Gaussian component，也不代表生物学 core；A-C 展示连续场，D-F 展示 shaft-aware 误差、定向 forced response 和短 spontaneous 返回。

**关注点**：该图只判断固定质量预算下的连续场是否具备跨杆可达容量，不是患者场拟合或 blind generalization。
""" if support_control else
        """### rev10_sa_continuous_field_capacity

这张图比较历史 K=3 Gaussian 基准、4x4 matched-DoF 连续 B-spline 场和 6x6 分辨率敏感性。控制系数只是连续场的数值自由度，不代表 core；A-C 展示场本身，D-F 展示 shaft-aware 误差、双向 forced response 和短 spontaneous 返回。

**关注点**：这是 patient-training-only exploratory canary，不是 blind generalization；连续场没有预设 component、峰数或每杆 core 数量。
"""
    )
    (figure_dir / "README.md").write_text(readme, encoding="utf-8")
    return stem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--worker-commit")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    support_control = "sa6g_continuous_support" in config
    if support_control:
        assay = config["sa6g_continuous_support"]
        output_subdir = "continuous_support_capacity"
        worker_status = "SA6G_CONTINUOUS_SUPPORT_WORKER_COMPLETE"
        summary_name = "continuous_support_capacity_summary.json"
        csv_name = "continuous_support_candidate_summary.csv"
    else:
        assay = config["sa6f_continuous_field"]
        output_subdir = "continuous_field_capacity"
        worker_status = "SA6F_CONTINUOUS_FIELD_WORKER_COMPLETE"
        summary_name = "continuous_field_capacity_summary.json"
        csv_name = "continuous_field_candidate_summary.csv"
    config_sha = _sha256(config_path)
    commit = subprocess.check_output(
        ["git", "rev-parse", args.expected_commit], cwd=ROOT, text=True,
    ).strip()
    worker_commit = subprocess.check_output(
        ["git", "rev-parse", args.worker_commit or args.expected_commit],
        cwd=ROOT, text=True,
    ).strip()
    output_root = ROOT / config["output_root"] / output_subdir
    manifest = json.loads((output_root / "candidate_manifest.json").read_text())
    if manifest["config"]["sha256"] != config_sha:
        raise RuntimeError("continuous-field manifest uses another config")

    inputs = config["inputs"]
    contract = _load_json_input(inputs["contact_contract"])
    contact_names, embedding, targets, floors = load_scoring_contract(
        inputs["shaft_aware_target_npz"]["path"],
        inputs["shaft_aware_floors"]["path"], "FULL_TIMING",
        fixed_events_per_mode=3,
    )
    groups, pairs = contract_groups(contract), contract_pairs(contract)
    contract_names = np.asarray([
        row["contact_name"] for row in contract["contacts"]
    ]).astype(str)
    if not np.array_equal(contact_names, contract_names):
        raise RuntimeError("scoring and SA0 contact order differ")
    scoring_config = json.loads(
        (ROOT / "config/topic4_rev10_sa_shaft_aware.json").read_text()
    )
    seeds = [int(value) for value in assay["network_seeds"]]
    worker_dir = output_root / "workers"
    candidate_rows, worker_inputs = [], []
    for candidate in manifest["candidate_set"]["candidates"]:
        forced, spontaneous, metadata, field_neighborhood, field_path = [], [], [], [], []
        for seed in seeds:
            stem = worker_dir / f"{candidate['candidate_id']}_seed_{seed}"
            json_path, npz_path = stem.with_suffix(".json"), stem.with_suffix(".npz")
            payload = json.loads(json_path.read_text())
            if not _worker_complete(
                    payload, npz_path, config_sha, worker_commit, worker_status):
                raise RuntimeError(f"incomplete or stale continuous-field worker: {stem}")
            if payload["candidate"]["field_sha256"] != candidate["field_sha256"]:
                raise RuntimeError(f"candidate field hash changed in {stem}")
            with np.load(npz_path, allow_pickle=False) as loaded:
                if not np.array_equal(loaded["contact_names"].astype(str), contact_names):
                    raise RuntimeError(f"contact order changed in {stem}")
                forced.append(np.asarray(loaded["forced_onsets"], float))
                spontaneous.append(np.asarray(loaded["spontaneous_onsets"], float))
                positions = np.asarray(loaded["positions_E"], float)
                h = np.asarray(loaded["h"], float)
                delta_vtheta = np.asarray(loaded["delta_vtheta"], float)
                contact_xy = np.asarray([
                    row["sheet_xy_mm"] for row in contract["contacts"]
                ], float)
                shaft_ids = np.asarray([
                    row["shaft_id"] for row in contract["contacts"]
                ])
                shaft_row = {}
                for shaft in ("ICL", "SCL"):
                    near = np.any([
                        np.linalg.norm(positions - contact, axis=1) <= 1.0
                        for contact in contact_xy[shaft_ids == shaft]
                    ], axis=0)
                    shaft_row[shaft] = {
                        "mean_h": float(np.mean(h[near])),
                        "median_h": float(np.median(h[near])),
                        "fraction_h_ge_0p5": float(np.mean(h[near] >= 0.5)),
                        "mean_delta_vtheta_mV": float(np.mean(delta_vtheta[near])),
                    }
                field_neighborhood.append(shaft_row)
                if support_control:
                    path_radius = float(assay["field_strength_audit"]["path_radius_mm"])
                    segments = np.asarray(candidate["segments"], float)
                    path_near = distance_to_segments(positions, segments) <= path_radius
                    path_row = {
                        "path_radius_mm": path_radius,
                        "mean_h": float(np.mean(h[path_near])),
                        "fraction_h_ge_0p5": float(np.mean(h[path_near] >= 0.5)),
                    }
                    if candidate.get("bridge_segment") is not None:
                        bridge = np.asarray(candidate["bridge_segment"], float)[None, :, :]
                        bridge_near = distance_to_segments(positions, bridge) <= path_radius
                        path_row["bridge_mean_h"] = float(np.mean(h[bridge_near]))
                        path_row["bridge_fraction_h_ge_0p5"] = float(
                            np.mean(h[bridge_near] >= 0.5)
                        )
                    field_path.append(path_row)
            metadata.append(payload)
            worker_inputs.append({
                "candidate_id": candidate["candidate_id"], "seed": seed,
                "json": str(json_path.relative_to(ROOT)),
                "json_sha256": _sha256(json_path), "npz_sha256": _sha256(npz_path),
            })

        forced = np.asarray(forced, float)
        values = np.concatenate([forced[:, 0, :], forced[:, 1, :]], axis=0)
        labels = np.asarray([0] * len(seeds) + [1] * len(seeds), int)
        score = score_mode_conditioned_events(
            values, labels, groups=groups, pairs=pairs, embedding=embedding,
            targets=targets, floors=floors, config=scoring_config,
            fixed_events_per_mode=3,
        )
        if score["status"] != "OK":
            raise RuntimeError(f"three-event score failed for {candidate['candidate_id']}")
        icl, scl = groups["ICL"], groups["SCL"]
        both_mode = np.asarray([
            np.isfinite(row[0, scl]).any() and np.isfinite(row[1, scl]).any()
            for row in forced
        ])
        spontaneous_all = (np.concatenate(spontaneous, axis=0)
                           if sum(len(row) for row in spontaneous)
                           else np.empty((0, len(contact_names))))
        spontaneous_multishaft = np.asarray([
            np.isfinite(row[icl]).any() and np.isfinite(row[scl]).any()
            for row in spontaneous_all
        ])
        source_rows = {
            source: [run for payload in metadata for run in payload["runs"]
                     if run["source_id"] == source]
            for source in ("ICL_mode_A", "ICL_mode_B", "SCL")
        }
        n_spontaneous = sum(row["spontaneous"]["n_common_detector_events"]
                            for row in metadata)
        n_returned = sum(row["spontaneous"]["n_returned_events"]
                         for row in metadata)
        n_runaway = sum(
            row["spontaneous"]["runaway_early_stop_ms"] is not None
            for row in metadata
        ) + sum(
            run["runaway_early_stop_ms"] is not None
            for row in metadata for run in row["runs"]
        )
        mode_a_scl = _metric(score, 0, "floor_excess", "recruitment.SCL")
        mode_b_scl = _metric(score, 1, "floor_excess", "recruitment.SCL")
        mode_a_icl = _metric(score, 0, "floor_excess", "precedence.ICL-ICL")
        mode_b_icl = _metric(score, 1, "floor_excess", "precedence.ICL-ICL")
        field_summary = {
            shaft: {
                key: float(np.mean([row[shaft][key] for row in field_neighborhood]))
                for key in field_neighborhood[0][shaft]
            } for shaft in ("ICL", "SCL")
        }
        representation_role = (
            candidate["role"] if support_control else (
                "K3_benchmark" if candidate["field_type"] == "gaussian_k3_benchmark"
                else ("matched_dof" if int(candidate["n_basis"]) == 4 else "resolution")
            )
        )
        row = {
            "candidate_id": candidate["candidate_id"],
            "field_sha256": candidate["field_sha256"],
            "representation_role": representation_role,
            "target_id": candidate.get("target_id", "historical_K3"),
            "n_basis": candidate.get("n_basis"),
            "roughness": candidate.get("roughness"),
            "contrast": candidate.get("contrast"),
            "support_id": candidate.get("support_id"),
            "width_mm": candidate.get("width_mm"),
            "mode_A_SCL_recruitment_excess": mode_a_scl,
            "mode_B_SCL_recruitment_excess": mode_b_scl,
            "worst_mode_SCL_recruitment_excess": max(mode_a_scl, mode_b_scl),
            "mode_A_ICL_precedence_excess": mode_a_icl,
            "mode_B_ICL_precedence_excess": mode_b_icl,
            "mode_A_cross_precedence_excess": _metric(
                score, 0, "floor_excess", "precedence.ICL-SCL"
            ),
            "mode_B_cross_precedence_excess": _metric(
                score, 1, "floor_excess", "precedence.ICL-SCL"
            ),
            "mode_B_ICL_profile_excess": _metric(
                score, 1, "floor_excess", "profile.ICL"
            ),
            "forced_both_mode_SCL_network_fraction": float(both_mode.mean()),
            "forced_ICL_A_to_SCL_mean_recruited_fraction": float(np.mean([
                run["SCL_recruited_contact_fraction"]
                for run in source_rows["ICL_mode_A"]
            ])),
            "forced_ICL_B_to_SCL_mean_recruited_fraction": float(np.mean([
                run["SCL_recruited_contact_fraction"]
                for run in source_rows["ICL_mode_B"]
            ])),
            "forced_SCL_to_ICL_mean_recruited_fraction": float(np.mean([
                run["ICL_recruited_contact_fraction"]
                for run in source_rows["SCL"]
            ])),
            "field_near_ICL": field_summary["ICL"],
            "field_near_SCL": field_summary["SCL"],
            "field_near_path": ({
                key: float(np.mean([value[key] for value in field_path if key in value]))
                for key in sorted({key for value in field_path for key in value})
                if key != "path_radius_mm"
            } if field_path else None),
            "spontaneous_event_count": int(n_spontaneous),
            "spontaneous_multishaft_fraction": (
                float(spontaneous_multishaft.mean()) if len(spontaneous_multishaft)
                else 0.0
            ),
            "spontaneous_returned_fraction": (
                float(n_returned / n_spontaneous) if n_spontaneous else 0.0
            ),
            "n_runaway": int(n_runaway),
            "descriptive_loss": float(
                max(mode_a_scl, mode_b_scl) + mode_a_icl + 0.25 * mode_b_icl
            ),
            "score": score,
        }
        candidate_rows.append(row)

    eligible = [row for row in candidate_rows if row["n_runaway"] == 0]
    if not eligible:
        raise RuntimeError("every continuous-field candidate entered runaway")
    least_loss = min(eligible, key=lambda row: row["descriptive_loss"])
    cross_support = [row for row in eligible
                     if row["forced_both_mode_SCL_network_fraction"] >= 2 / 3]
    if support_control:
        status = (
            "NO_K_CONTINUOUS_CONNECTED_FIELD_CROSS_SHAFT_OBSERVED_EXPLORATORY"
            if cross_support else
            "NO_K_CONTINUOUS_CONNECTED_FIELD_FAILS_CROSS_SHAFT_AT_FIXED_PACKET_AND_BUDGET"
        )
    else:
        status = (
            "CONTINUOUS_FIELD_CROSS_SHAFT_SUPPORT_OBSERVED_EXPLORATORY"
            if cross_support else
            "CONTINUOUS_FIELD_INITIALIZATION_CANARY_NO_CROSS_SHAFT_SUPPORT"
        )
    summary = {
        "status": status,
        "scientific_role": (
            "development-only no-K continuous support capacity positive control; "
            "not a patient-fit field or blind generalization"
            if support_control else
            "development-only non-component continuous-field initialization "
            "canary; no formal optimizer or patient blind generalization"
        ),
        "least_loss_descriptive_candidate": least_loss["candidate_id"],
        "n_cross_shaft_support_candidates": len(cross_support),
        "cross_shaft_support_candidate_ids": [row["candidate_id"] for row in cross_support],
        "candidate_rows": candidate_rows,
        "contact_names": contact_names.tolist(),
        "shaft_ids": [row["shaft_id"] for row in contract["contacts"]],
        "contact_xy_mm": [row["sheet_xy_mm"] for row in contract["contacts"]],
        "interpretation_boundary": {
            "spline_coefficients_are_cores": False,
            "support_segments_are_cores": False,
            "component_or_peak_count_is_fixed": False,
            "formal_shaft_aware_SNN_optimization": "not run",
            "K3_role": "historical benchmark only",
            "beta": "closed",
            "edge": "closed",
            "patient_blind": "not available; development only",
        },
        "inputs": worker_inputs,
        "worker_commit": worker_commit,
        "config": {"path": str(config_path.relative_to(ROOT)), "sha256": config_sha},
        "provenance": _runtime_provenance(args.expected_commit),
    }
    output_json = output_root / summary_name
    _atomic_json(output_json, summary)
    _atomic_csv(
        output_root / csv_name,
        [{key: value for key, value in row.items() if key != "score"}
         for row in candidate_rows],
    )
    stem = _plot(summary, manifest, output_root, support_control=support_control)
    print(json.dumps({
        "status": status,
        "least_loss_descriptive_candidate": least_loss["candidate_id"],
        "n_cross_shaft_support_candidates": len(cross_support),
        "summary": str(output_json),
        "figure": str(stem.with_suffix(".png")),
    }, indent=2))


if __name__ == "__main__":
    main()
