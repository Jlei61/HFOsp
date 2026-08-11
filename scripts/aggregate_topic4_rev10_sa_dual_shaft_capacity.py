"""Aggregate and plot the exploratory SA6 dual-shaft field canary."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.getcwd())
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
from src.topic4_core_field_stage3 import params_to_q  # noqa: E402
from src.topic4_shaft_aware import contract_groups, contract_pairs  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config/topic4_rev10_sa_dual_shaft_canary.json"


def _atomic_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".csv.tmp")
    os.close(handle)
    try:
        fieldnames = list(rows[0]) if rows else []
        with open(temporary, "w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _worker_complete(payload, npz_path, config_sha, commit):
    provenance = payload.get("provenance", {})
    return bool(
        payload.get("status") == "SA6_DUAL_SHAFT_WORKER_COMPLETE"
        and payload.get("config", {}).get("sha256") == config_sha
        and provenance.get("expected_git_commit") == commit
        and provenance.get("runtime_modules_match_expected_commit") is True
        and not provenance.get("runtime_modules_dirty")
        and payload.get("arrays", {}).get("sha256") == _sha256(npz_path)
    )


def _metric(score, mode, family, component):
    return float(score["modes"][str(mode)][family][component])


def _plot(summary, manifest, config, output_root):
    contacts = np.asarray(summary["contact_xy_mm"], float)
    shafts = np.asarray(summary["shaft_ids"]).astype(str)
    candidates = {row["candidate_id"]: row
                  for row in manifest["candidate_set"]["candidates"]}
    rows = summary["candidate_rows"]
    best_id = summary["least_loss_descriptive_candidate"]
    map_ids = ["frozen", "component3_scl_relocation", best_id]
    map_ids = list(dict.fromkeys(map_ids))
    axis = np.linspace(0.0, 20.0, 100)
    xx, yy = np.meshgrid(axis, axis)
    grid = np.column_stack([xx.ravel(), yy.ravel()])

    fig = plt.figure(figsize=(16.2, 8.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.95])
    for column, candidate_id in enumerate(map_ids[:3]):
        ax = fig.add_subplot(gs[0, column])
        q = params_to_q(candidates[candidate_id]["theta"], grid, K=3, L=20.0)
        q = (q / q.max()).reshape(xx.shape)
        image = ax.contourf(xx, yy, q, levels=np.linspace(0, 1, 16), cmap="magma")
        for shaft, color in (("ICL", "#37A6A6"), ("SCL", "#F28E2B")):
            selected = shafts == shaft
            ax.plot(contacts[selected, 0], contacts[selected, 1], "o-",
                    color=color, markersize=4, linewidth=1.2, label=shaft)
        ax.set_aspect("equal")
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 20)
        ax.set_title(candidate_id.replace("_", " "), loc="left", weight="bold")
        ax.set_xlabel("sheet x (mm)")
        if column == 0:
            ax.set_ylabel("sheet y (mm)")
            ax.legend(frameon=False, fontsize=8)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03, label="relative q")

    ax = fig.add_subplot(gs[1, 0])
    role_colors = {
        "frozen_baseline": "#7F7F7F",
        "matched_scl_relocation": "#E15759",
        "matched_offshaft_control": "#B07AA1",
        "scl_mass_width_grid": "#4E79A7",
    }
    seen_roles = set()
    for row in rows:
        label = None if row["role"] in seen_roles else row["role"].replace("_", " ")
        seen_roles.add(row["role"])
        ax.scatter(row["mode_A_ICL_precedence_excess"],
                   row["worst_mode_SCL_recruitment_excess"],
                   s=35 + 90 * row["forced_both_mode_SCL_network_fraction"],
                   color=role_colors[row["role"]], alpha=0.8,
                   edgecolor="white", linewidth=0.5, label=label)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.9)
    ax.set_xlabel("mode A ICL-ICL precedence excess")
    ax.set_ylabel("worst-mode SCL recruitment excess")
    ax.set_title("D  Capacity plane", loc="left", weight="bold")
    ax.legend(frameon=False, fontsize=7, loc="upper right")

    ax = fig.add_subplot(gs[1, 1])
    ordered = sorted(rows, key=lambda row: row["worst_mode_SCL_recruitment_excess"])
    shown = ordered[:10]
    y = np.arange(len(shown))
    ax.barh(y, [row["forced_both_mode_SCL_network_fraction"] for row in shown],
            color="#59A14F")
    ax.set_yticks(y, [row["candidate_id"] for row in shown], fontsize=7)
    ax.invert_yaxis()
    ax.axvline(2 / 3, color="black", linestyle="--", linewidth=0.9)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("networks recruiting SCL from both ICL sources")
    ax.set_title("E  Paired-network support", loc="left", weight="bold")
    if not any(row["forced_both_mode_SCL_network_fraction"] > 0.0 for row in shown):
        ax.text(0.5, 0.5, "all tested candidates: 0/3 networks",
                transform=ax.transAxes, ha="center", va="center", fontsize=10)

    ax = fig.add_subplot(gs[1, 2])
    x = np.arange(len(shown))
    ax.bar(x - 0.18, [row["spontaneous_multishaft_fraction"] for row in shown],
           width=0.36, color="#F28E2B", label="multishaft events")
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

    fig.suptitle(
        f"SA6 fixed-budget dual-shaft field canary | {summary['status']}",
        fontsize=14, weight="bold",
    )
    figure_dir = Path(output_root) / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    stem = figure_dir / "rev10_sa_dual_shaft_capacity"
    fig.savefig(stem.with_suffix(".png"), dpi=240, facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    (figure_dir / "README.md").write_text(
        """### rev10_sa_dual_shaft_capacity

这张图比较固定 K=3、固定总 field mass 的 21 个 Node-only 候选。A-C 显示冻结场、component-3 SCL relocation 和探索性最优候选；D 同时展示 mode A 的 ICL 内 precedence 误差与最弱模式 SCL recruitment 误差；E/F 给出逐网络 forced support 和短 spontaneous 事件。

**关注点**：这是 development-only capacity canary；虚线 1.0 是 patient-floor excess 参考，不是 patient blind generalization，气泡大小表示三张配对网络中两种 ICL source 都招募 SCL 的比例。
""",
        encoding="utf-8",
    )
    return stem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--worker-commit")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    config_sha = _sha256(config_path)
    commit = subprocess.check_output(
        ["git", "rev-parse", args.expected_commit], cwd=ROOT, text=True,
    ).strip()
    worker_commit = subprocess.check_output(
        ["git", "rev-parse", args.worker_commit or args.expected_commit],
        cwd=ROOT, text=True,
    ).strip()
    output_root = ROOT / config["output_root"] / "dual_shaft_capacity"
    manifest_path = output_root / "candidate_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["config"]["sha256"] != config_sha:
        raise RuntimeError("candidate manifest uses another config")
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

    seeds = [int(value) for value in config["sa6_dual_shaft_field"]["network_seeds"]]
    worker_dir = output_root / "workers"
    candidate_rows = []
    worker_inputs = []
    for candidate in manifest["candidate_set"]["candidates"]:
        forced = []
        spontaneous = []
        metadata = []
        field_neighborhood = []
        for seed in seeds:
            stem = worker_dir / f"{candidate['candidate_id']}_seed_{seed}"
            json_path, npz_path = stem.with_suffix(".json"), stem.with_suffix(".npz")
            payload = json.loads(json_path.read_text())
            if not _worker_complete(payload, npz_path, config_sha, worker_commit):
                raise RuntimeError(f"incomplete or stale SA6 worker: {stem}")
            if payload["candidate"]["theta_sha256"] != candidate["theta_sha256"]:
                raise RuntimeError(f"candidate hash changed in {stem}")
            with np.load(npz_path, allow_pickle=False) as loaded:
                if not np.array_equal(loaded["contact_names"].astype(str), contact_names):
                    raise RuntimeError(f"contact order changed in {stem}")
                forced.append(np.asarray(loaded["forced_onsets"], float))
                spontaneous.append(np.asarray(loaded["spontaneous_onsets"], float))
                positions = np.asarray(loaded["positions_E"], float)
                h = np.asarray(loaded["h"], float)
                delta_vtheta = np.asarray(loaded["delta_vtheta"], float)
                shaft_row = {}
                contact_xy = np.asarray([
                    row["sheet_xy_mm"] for row in contract["contacts"]
                ], float)
                shaft_ids = np.asarray([
                    row["shaft_id"] for row in contract["contacts"]
                ])
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
                        "fraction_threshold_lowered": float(np.mean(
                            delta_vtheta[near] < 0.0
                        )),
                    }
                field_neighborhood.append(shaft_row)
            metadata.append(payload)
            worker_inputs.append({
                "candidate_id": candidate["candidate_id"], "seed": seed,
                "json": str(json_path.relative_to(ROOT)),
                "json_sha256": _sha256(json_path),
                "npz_sha256": _sha256(npz_path),
            })
        forced = np.asarray(forced, float)
        mode_values = np.concatenate([forced[:, 0, :], forced[:, 1, :]], axis=0)
        labels = np.asarray([0] * len(seeds) + [1] * len(seeds), int)
        score = score_mode_conditioned_events(
            mode_values, labels, groups=groups, pairs=pairs,
            embedding=embedding, targets=targets, floors=floors,
            config=json.loads((ROOT / "config/topic4_rev10_sa_shaft_aware.json").read_text()),
            fixed_events_per_mode=3,
        )
        if score["status"] != "OK":
            raise RuntimeError(f"three-event score failed for {candidate['candidate_id']}")
        scl = groups["SCL"]
        icl = groups["ICL"]
        both_mode_by_seed = np.asarray([
            np.isfinite(row[0, scl]).any() and np.isfinite(row[1, scl]).any()
            for row in forced
        ])
        scl_source_cross = np.asarray([
            np.isfinite(row[2, scl]).any() and np.isfinite(row[2, icl]).any()
            for row in forced
        ])
        spontaneous_all = (np.concatenate(spontaneous, axis=0)
                           if sum(len(row) for row in spontaneous)
                           else np.empty((0, len(contact_names))))
        spontaneous_multishaft = np.asarray([
            np.isfinite(row[icl]).any() and np.isfinite(row[scl]).any()
            for row in spontaneous_all
        ])
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
        source_rows = {
            source: [run for payload in metadata for run in payload["runs"]
                     if run["source_id"] == source]
            for source in ("ICL_mode_A", "ICL_mode_B", "SCL")
        }
        field_summary = {
            shaft: {
                key: float(np.mean([row[shaft][key] for row in field_neighborhood]))
                for key in field_neighborhood[0][shaft]
            } for shaft in ("ICL", "SCL")
        }
        row = {
            "candidate_id": candidate["candidate_id"],
            "role": candidate["role"],
            "theta_sha256": candidate["theta_sha256"],
            "mode_A_SCL_recruitment_excess": mode_a_scl,
            "mode_B_SCL_recruitment_excess": mode_b_scl,
            "worst_mode_SCL_recruitment_excess": max(mode_a_scl, mode_b_scl),
            "mode_A_ICL_precedence_excess": _metric(
                score, 0, "floor_excess", "precedence.ICL-ICL"
            ),
            "mode_B_ICL_precedence_excess": _metric(
                score, 1, "floor_excess", "precedence.ICL-ICL"
            ),
            "mode_A_cross_precedence_excess": _metric(
                score, 0, "floor_excess", "precedence.ICL-SCL"
            ),
            "mode_B_cross_precedence_excess": _metric(
                score, 1, "floor_excess", "precedence.ICL-SCL"
            ),
            "mode_B_ICL_profile_excess": _metric(
                score, 1, "floor_excess", "profile.ICL"
            ),
            "forced_both_mode_SCL_network_fraction": float(both_mode_by_seed.mean()),
            "forced_SCL_source_cross_network_fraction": float(scl_source_cross.mean()),
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
            "spontaneous_event_count": int(n_spontaneous),
            "spontaneous_multishaft_fraction": (
                float(spontaneous_multishaft.mean()) if len(spontaneous_multishaft)
                else 0.0
            ),
            "spontaneous_returned_fraction": (
                float(n_returned / n_spontaneous) if n_spontaneous else 0.0
            ),
            "n_runaway": int(n_runaway),
            "capacity_reference_met": bool(
                max(mode_a_scl, mode_b_scl) <= 1.0
                and both_mode_by_seed.sum() >= 2
                and n_runaway == 0
            ),
            "score": score,
        }
        candidate_rows.append(row)

    eligible = [row for row in candidate_rows if row["n_runaway"] == 0]
    if not eligible:
        raise RuntimeError("every SA6 candidate entered runaway")
    selected = min(eligible, key=lambda row: (
        not row["capacity_reference_met"],
        row["worst_mode_SCL_recruitment_excess"],
        row["mode_A_ICL_precedence_excess"],
        row["mode_B_ICL_precedence_excess"],
    ))
    positive = [row for row in candidate_rows if row["capacity_reference_met"]]
    status = ("DUAL_SHAFT_CAPACITY_POSITIVE_EXPLORATORY" if positive
              else "DUAL_SHAFT_FIELD_CAPACITY_NOT_FOUND_IN_TESTED_GRID_CANARY")
    lookup = {row["candidate_id"]: row for row in candidate_rows}
    relocation_specificity = bool(
        lookup["component3_scl_relocation"]["capacity_reference_met"]
        and not lookup["component3_offshaft_control"]["capacity_reference_met"]
    )
    summary = {
        "status": status,
        "scientific_role": (
            "development-only fixed-budget K=3 Node field capacity; no patient blind "
            "generalization and no edge result"
        ),
        "least_loss_descriptive_candidate": selected["candidate_id"],
        "selection_warning": (
            "least-loss is descriptive only; no candidate met the exploratory "
            "capacity reference" if not positive else
            "one or more candidates met the exploratory capacity reference"
        ),
        "n_capacity_reference_candidates": len(positive),
        "capacity_reference_candidate_ids": [row["candidate_id"] for row in positive],
        "matched_relocation_position_specificity": relocation_specificity,
        "strongest_SCL_field_candidate": max(
            candidate_rows, key=lambda row: row["field_near_SCL"]["mean_h"]
        )["candidate_id"],
        "candidate_rows": candidate_rows,
        "contact_names": contact_names.tolist(),
        "shaft_ids": [row["shaft_id"] for row in contract["contacts"]],
        "contact_xy_mm": [row["sheet_xy_mm"] for row in contract["contacts"]],
        "decision_boundary": {
            "can_enter_SA7_low_dimensional_refit": bool(positive),
            "beta": "closed",
            "edge": "closed until a dual-shaft field is frozen",
            "patient_blind": "not available; development only",
        },
        "inputs": worker_inputs,
        "worker_commit": worker_commit,
        "config": {"path": str(config_path.relative_to(ROOT)), "sha256": config_sha},
        "provenance": _runtime_provenance(args.expected_commit),
    }
    output_json = output_root / "dual_shaft_capacity_summary.json"
    _atomic_json(output_json, summary)
    csv_rows = [{key: value for key, value in row.items() if key != "score"}
                for row in candidate_rows]
    _atomic_csv(output_root / "dual_shaft_candidate_summary.csv", csv_rows)
    stem = _plot(summary, manifest, config, output_root)
    print(json.dumps({
        "status": status,
        "selected": selected["candidate_id"],
        "n_capacity_reference_candidates": len(positive),
        "matched_relocation_position_specificity": relocation_specificity,
        "summary": str(output_json),
        "figure": str(stem.with_suffix(".png")),
    }, indent=2))


if __name__ == "__main__":
    main()
