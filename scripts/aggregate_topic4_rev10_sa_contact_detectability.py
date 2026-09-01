"""Aggregate paired-network SA5 contact detectability workers."""
from __future__ import annotations

import argparse
import csv
import hashlib
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
from src.topic4_core_field_runner import atomic_write_json  # noqa: E402
from src.topic4_rev10_sa_canary import (  # noqa: E402
    classify_contact_detectability,
    paired_shaft_ratio,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config/topic4_rev10_sa_dual_shaft_canary.json"
LFP_KEY = "peak_lfp_excess_per_packet_cell"
NEURAL_KEY = "local_positive_spike_excess_per_cell"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _summary(values):
    values = np.asarray(values, float)
    return {
        "median": float(np.median(values)),
        "q05": float(np.quantile(values, 0.05)),
        "q95": float(np.quantile(values, 0.95)),
        "n": int(len(values)),
    }


def _bootstrap_shaft_ratio(network_rows, value_key, *, repeats=20000, seed=20260811):
    per_seed = [paired_shaft_ratio(rows, value_key) for rows in network_rows]
    icl = np.asarray([row["ICL_median"] for row in per_seed], float)
    scl = np.asarray([row["SCL_median"] for row in per_seed], float)
    point = float(np.median(scl) / max(np.median(icl), 1e-12))
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(repeats), float)
    for index in range(int(repeats)):
        selected = rng.integers(0, len(per_seed), size=len(per_seed))
        draws[index] = np.median(scl[selected]) / max(
            np.median(icl[selected]), 1e-12,
        )
    return {
        "equal_network_weighted_SCL_over_ICL": point,
        "bootstrap_q05": float(np.quantile(draws, 0.05)),
        "bootstrap_q95": float(np.quantile(draws, 0.95)),
        "per_seed": per_seed,
        "ICL_network_medians": _summary(icl),
        "SCL_network_medians": _summary(scl),
        "bootstrap_repeats": int(repeats),
    }


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _plot(rows, result, output_root):
    names = []
    contacts = []
    for name in dict.fromkeys(row["contact_name"] for row in rows):
        selected = [row for row in rows if row["contact_name"] == name]
        names.append(name)
        contacts.append({
            "name": name,
            "shaft": selected[0]["shaft_id"],
            "xy": np.asarray(selected[0]["sheet_xy_mm"], float),
            "lfp": float(np.median([row[LFP_KEY] for row in selected])),
            "neural": float(np.median([row[NEURAL_KEY] for row in selected])),
        })
    fig, axes = plt.subplots(1, 4, figsize=(15.4, 3.9))
    xy = np.asarray([row["xy"] for row in contacts])
    gain = np.asarray([row["lfp"] for row in contacts])
    scatter = axes[0].scatter(
        xy[:, 0], xy[:, 1], c=gain, cmap="viridis", s=65,
        edgecolor="white", linewidth=0.6,
    )
    for row in contacts:
        axes[0].text(row["xy"][0] + 0.2, row["xy"][1] + 0.15,
                     row["name"], fontsize=6)
    axes[0].set_aspect("equal")
    axes[0].set_xlabel("sheet x (mm)")
    axes[0].set_ylabel("sheet y (mm)")
    axes[0].set_title("A  Contact-current gain", loc="left", weight="bold")
    fig.colorbar(scatter, ax=axes[0], fraction=0.05, label="peak excess / packet cell")

    colors = {"ICL": "#E76F51", "SCL": "#277DA1"}
    for shaft in ("ICL", "SCL"):
        selected = [row for row in contacts if row["shaft"] == shaft]
        axes[1].scatter(
            [row["name"] for row in selected], [row["lfp"] for row in selected],
            color=colors[shaft], s=32, label=shaft,
        )
    axes[1].tick_params(axis="x", labelrotation=65, labelsize=7)
    axes[1].set_ylabel("median current gain")
    axes[1].set_title("B  Fixed identity", loc="left", weight="bold")
    axes[1].legend(frameon=False)

    for shaft in ("ICL", "SCL"):
        selected = [row for row in contacts if row["shaft"] == shaft]
        axes[2].scatter(
            [row["neural"] for row in selected], [row["lfp"] for row in selected],
            color=colors[shaft], s=38, label=shaft,
        )
    axes[2].set_xlabel("local neural excess / cell")
    axes[2].set_ylabel("current gain / packet cell")
    axes[2].set_title("C  Neural vs observation", loc="left", weight="bold")

    seed_order = result["network_seeds"]
    x = np.arange(len(seed_order))
    for key, color, label in (("lfp", "#6A4C93", "current"),
                              ("neural", "#2A9D8F", "neural")):
        value = [row["SCL_over_ICL"] for row in result["shaft_ratios"][key]["per_seed"]]
        axes[3].plot(x, value, marker="o", color=color, label=label)
    reference = result["exploratory_ratio_reference"]
    axes[3].axhline(reference, color="#555555", linestyle="--", linewidth=1)
    axes[3].set_xticks(x, seed_order, rotation=45, fontsize=7)
    axes[3].set_ylabel("SCL / ICL paired ratio")
    axes[3].set_title("D  Network-level ratios", loc="left", weight="bold")
    axes[3].legend(frameon=False)
    for axis in axes:
        axis.spines[["right", "top"]].set_visible(False)
    fig.suptitle(
        f"SA5 contact detectability | {result['status']}",
        fontsize=11, weight="bold",
    )
    fig.tight_layout()
    figure_dir = output_root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    stem = figure_dir / "rev10_sa_contact_detectability"
    fig.savefig(stem.with_suffix(".png"), dpi=240, facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    readme = figure_dir / "README.md"
    text = readme.read_text() if readme.exists() else "# rev10-SA 图说明\n"
    if "### rev10_sa_contact_detectability" not in text:
        text += """

### rev10_sa_contact_detectability

这张图在 uniform-threshold Null substrate 上，对 15 个 contact 注入同半径、同神经元数的 exact spike packet。A/B 显示 current-based virtual-contact gain；C 将局部神经响应和电极响应分开；D 以 network seed 为独立单位比较 SCL/ICL 配对比值。

**关注点**：若 SCL 神经响应与 ICL 相近而 current readout 明显偏低，才支持 observation failure；虚线 0.5 只作探索性参照，不是新的硬 gate。
"""
        readme.write_text(text)
    return stem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    assay = config["sa5_contact_detectability"]
    output_root = ROOT / config["output_root"] / "contact_detectability"
    config_sha = _sha256(config_path)
    expected_commit = subprocess.check_output(
        ["git", "rev-parse", args.expected_commit], cwd=ROOT, text=True,
    ).strip()
    network_rows, flat_rows = [], []
    input_records = []
    for seed in assay["network_seeds"]:
        path = output_root / "workers" / f"seed_{seed}.json"
        arrays = output_root / "workers" / f"seed_{seed}.npz"
        payload = json.loads(path.read_text())
        provenance = payload["provenance"]
        if payload["status"] != "SA5_CONTACT_DETECTABILITY_WORKER_COMPLETE":
            raise RuntimeError(f"SA5 seed {seed} is incomplete")
        if payload["config"]["sha256"] != config_sha:
            raise RuntimeError(f"SA5 seed {seed} config hash mismatch")
        if (provenance["expected_git_commit"] != expected_commit
                or provenance["runtime_modules_match_expected_commit"] is not True
                or provenance["runtime_modules_dirty"]):
            raise RuntimeError(f"SA5 seed {seed} provenance failed")
        if payload["arrays"]["sha256"] != _sha256(arrays):
            raise RuntimeError(f"SA5 seed {seed} array hash mismatch")
        network_rows.append(payload["contacts"])
        for row in payload["contacts"]:
            flat_rows.append({"seed": int(seed), **row})
        input_records.append({
            "seed": int(seed), "json": str(path.relative_to(ROOT)),
            "json_sha256": _sha256(path), "npz_sha256": _sha256(arrays),
        })

    lfp = _bootstrap_shaft_ratio(network_rows, LFP_KEY, seed=20260811)
    neural = _bootstrap_shaft_ratio(network_rows, NEURAL_KEY, seed=20260812)
    reference = float(assay["exploratory_ratio_reference"]["scl_over_icl_lower"])
    status = classify_contact_detectability(
        lfp["equal_network_weighted_SCL_over_ICL"],
        neural["equal_network_weighted_SCL_over_ICL"],
        reference_ratio=reference,
    )
    result = {
        "status": status,
        "scientific_role": (
            "development-only observation-versus-local-network audit; no patient "
            "held-out score and no field-capacity result"
        ),
        "network_seeds": [int(seed) for seed in assay["network_seeds"]],
        "n_networks": len(network_rows),
        "n_contacts_per_network": len(network_rows[0]),
        "packet_contract": {
            "same_radius_within_network": True,
            "same_count_within_network": True,
            "radius_mm": float(assay["packet_radius_mm"]),
            "common_counts_by_seed": [
                int(rows[0]["packet_n_E"]) for rows in network_rows
            ],
        },
        "shaft_ratios": {"lfp": lfp, "neural": neural},
        "exploratory_ratio_reference": reference,
        "interpretation": {
            "SCL_READOUT_NOT_PRIMARY_LIMIT": (
                "SCL neural and current response both remain at least half of ICL"
            ),
            "VIRTUAL_CONTACT_OBSERVATION_FAIL": (
                "SCL neural response is retained but current readout is selectively weak"
            ),
            "SCL_LOCAL_NETWORK_RESPONSE_LIMIT": (
                "equal forced packets fail to recruit comparable local SCL neural response"
            ),
            "ratio_role": "exploratory mechanism branch, not formal acceptance gate",
        },
        "positive_detector_margin_fraction": {
            shaft: float(np.mean([
                row["absolute_detector_margin"] > 0.0 for row in flat_rows
                if row["shaft_id"] == shaft
            ])) for shaft in ("ICL", "SCL")
        },
        "inputs": input_records,
        "config": {"path": str(config_path.relative_to(ROOT)), "sha256": config_sha},
        "provenance": {
            "git_commit": expected_commit,
            "producer_sha256": _sha256(__file__),
            "runtime_file_dirty": bool(subprocess.check_output(
                ["git", "status", "--porcelain", "--",
                 str(Path(__file__).resolve().relative_to(ROOT))], cwd=ROOT, text=True,
            ).strip()),
        },
    }
    _write_csv(output_root / "contact_detectability_per_contact.csv", flat_rows)
    summary_path = output_root / "contact_detectability_summary.json"
    atomic_write_json(result, summary_path)
    stem = _plot(flat_rows, result, output_root)
    print(json.dumps({
        "status": status,
        "lfp_SCL_over_ICL": lfp["equal_network_weighted_SCL_over_ICL"],
        "neural_SCL_over_ICL": neural["equal_network_weighted_SCL_over_ICL"],
        "summary": str(summary_path), "figure": str(stem),
    }, indent=2))


if __name__ == "__main__":
    main()
