#!/usr/bin/env python3
"""Build the target-free LBSS v0.2 run contract and geometry audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.topic5_lbss_rnn_v0_2 import (
    build_pool_contract,
    source_balanced_sample,
    strong_component_audit,
)


DEFAULT_SOURCE = Path(
    "/home/honglab/leijiaxin/HFOsp/.worktrees/"
    "topic5-rnn-motif-cross-state-v0-4/results/"
    "topic5_rnn_motif_cross_state_benchmark_v0_4"
)
DEFAULT_OUT = Path("results/topic5_lbss_rnn_v0_2")
EARLY_PRIMARY = {
    "epilepsiae_1084", "epilepsiae_1150", "epilepsiae_253",
    "epilepsiae_384", "epilepsiae_442", "epilepsiae_548",
    "epilepsiae_590", "epilepsiae_620", "epilepsiae_922",
    "epilepsiae_958",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def durable_link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256(source) != sha256(destination):
            raise RuntimeError(f"stale cache copy: {destination}")
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def git_text(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def plot_audit(table: pd.DataFrame, representative: dict, out: Path) -> None:
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    xy = representative["xy"]
    local = representative["local"].astype(bool)
    extra = representative["extra_sample"].astype(bool)
    lr = representative["lr_sample"].astype(bool)

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.35), gridspec_kw={"width_ratios": [1.05, 1]})
    ax = axes[0]
    for mask, color, alpha, width in [
        (local, "#9aa1a8", 0.24, 0.42),
        (extra, "#3c78a8", 0.62, 0.72),
        (lr, "#cf4b3e", 0.62, 0.72),
    ]:
        for target, source in np.argwhere(mask):
            ax.plot(xy[[source, target], 0], xy[[source, target], 1], color=color,
                    alpha=alpha, lw=width, zorder=1)
    ax.scatter(xy[:, 0], xy[:, 1], s=12, color="#252525", linewidth=0, zorder=3)
    ax.set_aspect("equal")
    ax.set_xlabel("Propagation axis (mm)")
    ax.set_ylabel("Transverse axis (mm)")
    ax.set_title(representative["fit_id"].split("__")[0].replace("epilepsiae_", "E"))
    ax.spines[["top", "right"]].set_visible(False)

    patient = table.groupby("subject", sort=False).agg(
        local_edges=("local_edges", "mean"),
        extra_pool=("extra_pool_edges", "mean"),
        lr_pool=("nonlocal_pool_edges", "mean"),
        added=("k_added", "mean"),
    )
    order = np.argsort(patient["local_edges"].to_numpy())
    x = np.arange(len(patient))
    ax = axes[1]
    ax.scatter(x, patient["extra_pool"].to_numpy()[order], s=23, color="#3c78a8", label="Extra-local pool")
    ax.scatter(x, patient["lr_pool"].to_numpy()[order], s=23, color="#cf4b3e", label="Nonlocal pool")
    ax.scatter(x, patient["added"].to_numpy()[order], s=18, color="#222222", label="Active additions")
    ax.set_yscale("log")
    ax.set_xlabel("Patients")
    ax.set_ylabel("Directed edges")
    ax.set_xticks([])
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")

    for label, axis in zip("AB", axes):
        axis.text(-0.16, 1.05, label, transform=axis.transAxes, fontsize=12,
                  fontweight="bold", va="top")
    fig.tight_layout(w_pad=2.0)
    for suffix in ("png", "pdf"):
        fig.savefig(figures / f"stage_a_geometry_and_candidate_pools.{suffix}", dpi=600,
                    bbox_inches="tight")
    plt.close(fig)
    (figures / "README.md").write_text(
        "### stage_a_geometry_and_candidate_pools.png\n\n"
        "A 展示预先固定的代表患者组织平面：灰色为双向可达的局部骨架，蓝色与红色分别为额外局部和非局部候选池。"
        "B 汇总所有患者的候选池规模与正式模型实际增加的边数，说明三条加边 arm 具有足够而清楚分离的搜索空间。\n\n"
        "**关注点**：局部骨架覆盖完整组织平面，且非局部候选池不是由发作期结果定义。\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--representative-fit", default="epilepsiae_1084__shared")
    args = parser.parse_args()
    source = args.source_root.resolve()
    out = args.out_root.resolve()
    out.mkdir(parents=True, exist_ok=True)

    manifest_path = source / "INPUT_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["n_fits"] != 31 or manifest["n_patients"] != 21:
        raise RuntimeError("unexpected frozen v0.4 cohort revision")

    rows: list[dict] = []
    cache_rows: list[dict] = []
    representative: dict | None = None
    invalid_subjects: set[str] = set()
    invalid_primary: set[str] = set()
    for fit in manifest["fits"]:
        fit_id, subject = fit["fit_id"], fit["subject"]
        source_fit = source / "cache" / fit_id
        target_fit = out / "cache" / fit_id
        for name in ("plane.npz", "events.npz", "provenance.json"):
            durable_link(source_fit / name, target_fit / name)
            cache_rows.append({
                "fit_id": fit_id,
                "file": name,
                "source": str((source_fit / name).resolve()),
                "local": str((target_fit / name).resolve()),
                "sha256": sha256(target_fit / name),
            })
        plane = np.load(target_fit / "plane.npz", allow_pickle=False)
        contract = build_pool_contract(plane["D_mm"])
        supported = np.asarray(plane["H"].sum(axis=0) > 1e-8)
        audit = strong_component_audit(contract.local_mask, supported=supported)
        valid = (
            audit["all_supported_nodes_one_strong_component"]
            and audit["contact_supported_pairwise_reachability"] == 1.0
            and audit["minimum_in_degree"] >= 1
            and audit["minimum_out_degree"] >= 1
            and int(contract.extra_local_pool.sum()) >= contract.k_added
            and int(contract.nonlocal_pool.sum()) >= contract.k_added
        )
        if not valid:
            invalid_subjects.add(subject)
            if subject in EARLY_PRIMARY:
                invalid_primary.add(subject)
        distance = plane["D_mm"]
        local_lengths = distance[contract.local_mask.astype(bool)]
        extra_lengths = distance[contract.extra_local_pool.astype(bool)]
        lr_lengths = distance[contract.nonlocal_pool.astype(bool)]
        rows.append({
            "fit_id": fit_id,
            "subject": subject,
            "scope": fit["scope"],
            "n_nodes": int(distance.shape[0]),
            "k_neighbors": contract.k_neighbors,
            "target_local_edges": contract.target_local_edges,
            "local_edges": int(contract.local_mask.sum()),
            "k_added": contract.k_added,
            "r_local_mm": contract.r_local_mm,
            "extra_pool_edges": int(contract.extra_local_pool.sum()),
            "nonlocal_pool_edges": int(contract.nonlocal_pool.sum()),
            "local_length_median_mm": float(np.median(local_lengths)),
            "extra_length_median_mm": float(np.median(extra_lengths)) if extra_lengths.size else np.nan,
            "nonlocal_length_median_mm": float(np.median(lr_lengths)) if lr_lengths.size else np.nan,
            "min_in_degree": audit["minimum_in_degree"],
            "min_out_degree": audit["minimum_out_degree"],
            "n_strong_components": audit["n_strong_components"],
            "valid": bool(valid),
        })
        if fit_id == args.representative_fit:
            representative = {
                "fit_id": fit_id,
                "xy": plane["nodes_xy_mm"].copy(),
                "local": contract.local_mask.copy(),
                "extra_sample": source_balanced_sample(
                    contract.extra_local_pool, contract.k_added, seed=41
                ),
                "lr_sample": source_balanced_sample(
                    contract.nonlocal_pool, contract.k_added, seed=43
                ),
            }

    table = pd.DataFrame(rows)
    table.to_csv(out / "CANDIDATE_POOL_AUDIT.csv", index=False)
    write_json(out / "CANDIDATE_POOL_AUDIT.json", {
        "contract": "topic5_lbss_candidate_pool_audit_v0_2",
        "n_patients": int(table.subject.nunique()),
        "n_fits": len(table),
        "n_valid_fits": int(table.valid.sum()),
        "invalid_subjects": sorted(invalid_subjects),
        "invalid_early_ictal_primary_subjects": sorted(invalid_primary),
        "pause_rule": "invalid_subjects > 2 OR any primary early-ictal subject invalid",
        "rows": rows,
    })
    write_json(out / "INPUT_CACHE_MANIFEST.json", {
        "contract": "topic5_lbss_input_cache_manifest_v0_2",
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": sha256(manifest_path),
        "target_values_read": False,
        "files": cache_rows,
    })

    spec = Path("docs/superpowers/specs/2026-08-10-topic5-local-backbone-selective-shortcut-rnn-design.md")
    plan = Path("docs/superpowers/plans/2026-08-10-topic5-local-backbone-selective-shortcut-rnn.md")
    snapshot_files = [Path(__file__), Path("src/topic5_lbss_rnn_v0_2.py"), spec, plan]
    snapshot = out / "run_snapshot"
    snapshot.mkdir(exist_ok=True)
    code_hashes = {}
    for file in snapshot_files:
        destination = snapshot / file.name
        shutil.copy2(file, destination)
        code_hashes[file.name] = sha256(destination)
    contract = {
        "contract": "topic5_lbss_rnn_v0_2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_text(["rev-parse", "HEAD"]),
        "git_branch": git_text(["branch", "--show-current"]),
        "geometry_status": "RETROSPECTIVE_TEST_INFORMED",
        "edge_time_status": "ORDINAL_NO_PHYSICAL_DELAY",
        "target_access_count": 0,
        "target_values_read": False,
        "cohort": {"n_patients": 21, "n_fits": 31, "early_ictal_primary_n": 10},
        "arms": [
            "L0_LOCAL_ONLY", "L1_LOCAL_PLUS_LEARNED_EXTRA_LOCAL",
            "L2_LOCAL_PLUS_RANDOM_LR", "L3_LOCAL_PLUS_LEARNED_LR",
            "C_L3_ORDER_SHUFFLED",
        ],
        "local_backbone": "symmetrized kNN candidate mask with independent directed weights",
        "added_fraction": 0.10,
        "nonlocal_cutoff": "2 * median local-edge length within patient fit",
        "checkpoint_rule": "best checkpoint eligible only at/after mask freeze",
        "order_shuffle": "rank 1 fixed; rank 2..T deranged when at least two later ranks exist",
        "source_code_hashes": code_hashes,
        "input_manifest_sha256": sha256(out / "INPUT_CACHE_MANIFEST.json"),
        "candidate_pool_audit_sha256": sha256(out / "CANDIDATE_POOL_AUDIT.csv"),
    }
    write_json(out / "RUN_CONTRACT.json", contract)
    if representative is None:
        raise RuntimeError("representative fit missing")
    plot_audit(table, representative, out)

    blocked = len(invalid_subjects) > 2 or bool(invalid_primary)
    marker = out / ("PREFLIGHT_BLOCKED.json" if blocked else "PREFLIGHT_COMPLETE.json")
    other = out / ("PREFLIGHT_COMPLETE.json" if blocked else "PREFLIGHT_BLOCKED.json")
    other.unlink(missing_ok=True)
    write_json(marker, {
        "status": "BLOCKED" if blocked else "PASS",
        "n_valid_fits": int(table.valid.sum()),
        "n_fits": len(table),
        "invalid_subjects": sorted(invalid_subjects),
        "invalid_early_ictal_primary_subjects": sorted(invalid_primary),
        "target_values_read": False,
    })
    if blocked:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
