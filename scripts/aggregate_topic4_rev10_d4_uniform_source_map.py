"""Aggregate the D4 uniform forced-source map by network-independent source."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config/topic4_rev10_d4_uniform_forced_source_map.json"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        Path(temporary).write_text(json.dumps(payload, indent=2, sort_keys=True))
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_csv(path, rows):
    keys = sorted({key for row in rows for key in row})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        with open(temporary, "w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=keys, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def adjudicate(source_rows, minimum_networks):
    eligible = [
        row for row in source_rows
        if int(row["clean_A_networks"]) >= int(minimum_networks)
    ]
    eligible.sort(key=lambda row: (
        -int(row["clean_A_networks"]),
        -int(row["clean_B_networks"]),
        row["source_id"],
    ))
    any_a = int(sum(row["clean_A_networks"] for row in source_rows))
    return {
        "status": (
            "REV10D4_UNIFORM_FORCED_MODE_A_ROUTE_CAPACITY_OBSERVED"
            if eligible else
            "REV10D4_UNIFORM_FORCED_MODE_A_ROUTE_CAPACITY_NOT_OBSERVED"
        ),
        "selected_source_id": eligible[0]["source_id"] if eligible else None,
        "n_shared_A_sources": int(len(eligible)),
        "total_clean_A_source_network_responses": any_a,
        "single_network_A_only": bool(not eligible and any_a > 0),
    }


def _plot(rows, coordinates, figures):
    coordinates = np.asarray(coordinates, float)
    lookup = {(float(row["x_mm"]), float(row["y_mm"])): row for row in rows}
    h = np.asarray([
        [lookup[(x, y)]["mean_source_h"] for x in coordinates]
        for y in coordinates
    ])
    a = np.asarray([
        [lookup[(x, y)]["clean_A_networks"] for x in coordinates]
        for y in coordinates
    ])
    b = np.asarray([
        [lookup[(x, y)]["clean_B_networks"] for x in coordinates]
        for y in coordinates
    ])
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.5), constrained_layout=True)
    extent = (0.0, 20.0, 0.0, 20.0)
    images = [
        axes[0].imshow(h, origin="lower", extent=extent, cmap="viridis",
                       interpolation="bilinear", aspect="equal"),
        axes[1].imshow(a, origin="lower", extent=extent,
                       cmap=LinearSegmentedColormap.from_list(
                           "clean_a", ["#ffffff", "#d95f5f"]),
                       vmin=0, vmax=3, interpolation="nearest", aspect="equal"),
        axes[2].imshow(b, origin="lower", extent=extent,
                       cmap=LinearSegmentedColormap.from_list(
                           "clean_b", ["#ffffff", "#2f78a8"]),
                       vmin=0, vmax=3, interpolation="nearest", aspect="equal"),
    ]
    titles = (
        "Frozen Node field at forced sources",
        "Clean forced mode A support",
        "Clean forced mode B support",
    )
    for axis, title in zip(axes, titles):
        axis.set_title(title, fontsize=10, weight="bold")
        axis.set_xlabel("sheet x (mm)")
        axis.set_ylabel("sheet y (mm)")
        axis.set_xticks([0, 5, 10, 15, 20])
        axis.set_yticks([0, 5, 10, 15, 20])
    for axis, matrix in zip(axes[1:], (a, b)):
        for iy, y in enumerate(coordinates):
            for ix, x in enumerate(coordinates):
                axis.text(x, y, str(int(matrix[iy, ix])), ha="center",
                          va="center", fontsize=8, color="#171717")
    fig.colorbar(images[0], ax=axes[0], fraction=0.046, pad=0.03, label="mean h")
    fig.colorbar(images[1], ax=axes[1], fraction=0.046, pad=0.03,
                 ticks=[0, 1, 2, 3], label="networks (of 3)")
    fig.colorbar(images[2], ax=axes[2], fraction=0.046, pad=0.03,
                 ticks=[0, 1, 2, 3], label="networks (of 3)")
    fig.suptitle(
        "Uniform-source causal map: route capacity independent of source placement",
        fontsize=12, weight="bold",
    )
    figures.mkdir(parents=True, exist_ok=True)
    png = figures / "rev10_d4_uniform_forced_source_route_map.png"
    pdf = figures / "rev10_d4_uniform_forced_source_route_map.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png, pdf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    root = ROOT / config["output_root"]
    manifest_path = root / "source_grid_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    config_sha, manifest_sha = _sha256(config_path), _sha256(manifest_path)
    if manifest["config"]["sha256"] != config_sha:
        raise RuntimeError("D4 manifest uses another config")
    expected_commit = subprocess.check_output(
        ["git", "rev-parse", args.expected_commit], cwd=ROOT, text=True,
    ).strip()

    source_ids = [row["source_id"] for row in manifest["source_grid"]["sources"]]
    by_source = {source_id: [] for source_id in source_ids}
    worker_inputs = []
    for seed in map(int, config["network_seeds"]):
        stem = root / "workers" / f"uniform_source_seed_{seed}"
        json_path, npz_path = stem.with_suffix(".json"), stem.with_suffix(".npz")
        payload = json.loads(json_path.read_text())
        provenance = payload.get("provenance", {})
        if not (
            payload.get("status") == "REV10D4_UNIFORM_SOURCE_WORKER_COMPLETE"
            and payload.get("seed") == seed
            and payload.get("config", {}).get("sha256") == config_sha
            and payload.get("manifest", {}).get("sha256") == manifest_sha
            and payload.get("arrays", {}).get("sha256") == _sha256(npz_path)
            and provenance.get("expected_git_commit") == expected_commit
            and provenance.get("runtime_modules_match_expected_commit") is True
            and not provenance.get("runtime_modules_dirty")
        ):
            raise RuntimeError(f"stale D4 worker: {stem}")
        rows = payload["source_rows"]
        if [row["source_id"] for row in rows] != source_ids:
            raise RuntimeError(f"D4 source order changed: seed {seed}")
        for row in rows:
            by_source[row["source_id"]].append(row)
        worker_inputs.append({
            "seed": seed,
            "json": str(json_path), "json_sha256": _sha256(json_path),
            "npz": str(npz_path), "npz_sha256": _sha256(npz_path),
        })

    rows = []
    for source in manifest["source_grid"]["sources"]:
        records = by_source[source["source_id"]]
        clean_a = sum(
            row["clean_route_response"] and row["assigned_mode"] == "A"
            for row in records
        )
        clean_b = sum(
            row["clean_route_response"] and row["assigned_mode"] == "B"
            for row in records
        )
        rows.append({
            "source_id": source["source_id"],
            "x_mm": float(source["xy_mm"][0]),
            "y_mm": float(source["xy_mm"][1]),
            "clean_A_networks": int(clean_a),
            "clean_B_networks": int(clean_b),
            "joint_networks": int(sum(row["joint_shaft"] for row in records)),
            "returned_networks": int(sum(
                row["triggered_event"] is not None
                and row["triggered_event"]["returned"] for row in records
            )),
            "in_distribution_networks": int(sum(not row["ood"] for row in records)),
            "mean_source_h": float(np.mean([row["mean_source_h"] for row in records])),
            "mean_downstream_positive_spike_mass": float(np.mean([
                row["downstream_positive_spike_mass"] for row in records
            ])),
            "all_pretrigger_bit_identical": bool(all(
                row["pretrigger_spikes_bit_identical"] for row in records
            )),
            "n_runaway": int(sum(
                row["runaway_early_stop_ms"] is not None for row in records
            )),
        })
    decision = adjudicate(
        rows, config["decision"]["minimum_networks_same_source_for_A"],
    )
    figures = root / "figures"
    png, pdf = _plot(rows, config["source_grid"]["coordinates_mm"], figures)
    readme = figures / "README.md"
    readme.write_text(
        "### rev10_d4_uniform_forced_source_route_map.png\n\n"
        "左图是在均匀强制源网格上读取的冻结 Node 场强度；中、右图分别给出同一源位置在 3 张网络中产生 returned、双杆且位于患者支持域内的 mode A/B 次数。源位置按均匀网格预先冻结，不由电极、患者标签或场峰选择。\n\n"
        "**关注点**：mode A 是否在同一网格点达到至少 2/3 网络，以及 mode B 是否作为正对照保留。\n"
    )
    payload = {
        **decision,
        "scientific_role": config["scientific_role"],
        "safe_claim": (
            "uniform forced-source development diagnostic; source placement is "
            "observation-invariant, but A/B support uses the patient-training classifier"
        ),
        "source_rows": rows,
        "network_seeds": list(map(int, config["network_seeds"])),
        "figures": {
            "png": str(png), "png_sha256": _sha256(png),
            "pdf": str(pdf), "pdf_sha256": _sha256(pdf),
            "readme": str(readme), "readme_sha256": _sha256(readme),
        },
        "worker_inputs": worker_inputs,
        "config": {"path": str(config_path.relative_to(ROOT)), "sha256": config_sha},
        "manifest": {"path": str(manifest_path), "sha256": manifest_sha},
        "claim_boundary": (
            "forced initiation is not spontaneous activity or patient-blind confirmation"
        ),
    }
    _atomic_csv(root / "uniform_source_summary.csv", rows)
    _atomic_json(root / "uniform_source_verdict.json", payload)
    print(json.dumps({
        "status": payload["status"],
        "selected_source_id": payload["selected_source_id"],
        "n_shared_A_sources": payload["n_shared_A_sources"],
        "total_clean_A_source_network_responses": payload[
            "total_clean_A_source_network_responses"
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
