"""Aggregate D4.1 fresh-network packet-dose confirmation."""
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

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config/topic4_rev10_d4_1_packet_dose_confirmation.json"


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
    path = Path(path)
    keys = sorted({key for row in rows for key in row})
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


def adjudicate(dose_rows, *, minimum_networks, source_ids):
    eligible = [
        row for row in dose_rows
        if all(
            int(row[f"clean_{source_id}_networks"]) >= int(minimum_networks)
            for source_id in source_ids
        ) and int(row["n_runaway"]) == 0
    ]
    selected = min(eligible, key=lambda row: row["packet_fraction_of_E"], default=None)
    maximum = max(dose_rows, key=lambda row: row["packet_fraction_of_E"])
    return {
        "status": (
            "REV10D4_1_FRESH_NETWORK_FORCED_AB_ROUTE_CONFIRMED"
            if selected is not None else
            "REV10D4_1_FRESH_NETWORK_FORCED_AB_ROUTE_NOT_CONFIRMED"
        ),
        "selected_packet_fraction_of_E": (
            None if selected is None else float(selected["packet_fraction_of_E"])
        ),
        "selected_packet_n_E": (
            None if selected is None else int(selected["packet_n_E"])
        ),
        "maximum_dose_passed": bool(selected is not None or all(
            int(maximum[f"clean_{source_id}_networks"]) >= int(minimum_networks)
            for source_id in source_ids
        )),
    }


def _plot(rows, source_ids, figures):
    fractions = np.asarray([row["packet_fraction_of_E"] for row in rows], float)
    packet_n = np.asarray([row["packet_n_E"] for row in rows], int)
    colors = {source_ids[0]: "#c64b45", source_ids[1]: "#2f78a8"}
    labels = {source_ids[0]: "A source (18, 6)", source_ids[1]: "B source (2, 14)"}
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.3), constrained_layout=True)
    for source_id in source_ids:
        axes[0].plot(
            packet_n, [row[f"clean_{source_id}_networks"] for row in rows],
            marker="o", lw=2, color=colors[source_id], label=labels[source_id],
        )
        axes[1].plot(
            packet_n, [row[f"median_ood_{source_id}"] for row in rows],
            marker="o", lw=2, color=colors[source_id], label=labels[source_id],
        )
    axes[0].axhline(5, color="#555555", ls="--", lw=1, label="5/6 gate")
    axes[0].set_ylim(-0.2, 6.3)
    axes[0].set_ylabel("clean expected-mode networks (of 6)")
    axes[0].set_title("Fresh-network route access")
    axes[1].set_ylabel("median frozen-classifier OOD distance")
    axes[1].set_title("Patient-support distance")
    for axis in axes:
        axis.set_xlabel("synchronized E neurons in packet")
        axis.set_xticks(packet_n)
        axis.grid(axis="y", color="#dddddd", lw=0.7)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle(
        "Forced A/B route dose confirmation on six fresh networks",
        fontsize=11, weight="bold",
    )
    figures.mkdir(parents=True, exist_ok=True)
    png = figures / "rev10_d4_1_fresh_network_packet_dose.png"
    pdf = figures / "rev10_d4_1_fresh_network_packet_dose.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png, pdf


def _median(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    return None if not len(values) else float(np.median(values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    root = ROOT / config["output_root"]
    manifest_path = root / "packet_dose_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    config_sha, manifest_sha = _sha256(config_path), _sha256(manifest_path)
    if manifest["config"]["sha256"] != config_sha:
        raise RuntimeError("D4.1 manifest uses another config")
    expected_commit = subprocess.check_output(
        ["git", "rev-parse", args.expected_commit], cwd=ROOT, text=True,
    ).strip()

    all_rows, worker_inputs = [], []
    for seed in map(int, config["network_seeds"]):
        stem = root / "workers" / f"packet_dose_seed_{seed}"
        json_path, npz_path = stem.with_suffix(".json"), stem.with_suffix(".npz")
        payload = json.loads(json_path.read_text())
        provenance = payload.get("provenance", {})
        if not (
            payload.get("status") == "REV10D4_1_PACKET_DOSE_WORKER_COMPLETE"
            and payload.get("seed") == seed
            and payload.get("config", {}).get("sha256") == config_sha
            and payload.get("manifest", {}).get("sha256") == manifest_sha
            and payload.get("arrays", {}).get("sha256") == _sha256(npz_path)
            and provenance.get("expected_git_commit") == expected_commit
            and provenance.get("runtime_modules_match_expected_commit") is True
            and not provenance.get("runtime_modules_dirty")
        ):
            raise RuntimeError(f"stale D4.1 worker: {stem}")
        all_rows.extend({**row, "seed": seed} for row in payload["response_rows"])
        worker_inputs.append({
            "seed": seed,
            "json": str(json_path), "json_sha256": _sha256(json_path),
            "npz": str(npz_path), "npz_sha256": _sha256(npz_path),
        })

    source_ids = [row["source_id"] for row in config["sources"]]
    dose_rows = []
    for fraction in map(float, config["packet_fractions_of_E"]):
        selected = [
            row for row in all_rows
            if np.isclose(row["packet_fraction_of_E"], fraction)
        ]
        summary = {
            "packet_fraction_of_E": fraction,
            "packet_n_E": int(selected[0]["packet_n_E"]),
            "n_runaway": int(sum(row["runaway_early_stop_ms"] is not None for row in selected)),
            "all_pretrigger_bit_identical": bool(all(
                row["pretrigger_spikes_bit_identical"] for row in selected
            )),
        }
        for source_id in source_ids:
            source_rows = [row for row in selected if row["source_id"] == source_id]
            summary[f"clean_{source_id}_networks"] = int(sum(
                row["clean_expected_response"] for row in source_rows
            ))
            summary[f"returned_{source_id}_networks"] = int(sum(
                row["triggered_event"] is not None
                and row["triggered_event"]["returned"] for row in source_rows
            ))
            summary[f"joint_{source_id}_networks"] = int(sum(
                row["joint_shaft"] for row in source_rows
            ))
            summary[f"in_distribution_{source_id}_networks"] = int(sum(
                not row["ood"] for row in source_rows
            ))
            summary[f"expected_mode_{source_id}_networks"] = int(sum(
                row["expected_mode_match"] for row in source_rows
            ))
            summary[f"median_ood_{source_id}"] = _median([
                row["ood_distance"] for row in source_rows
            ])
            summary[f"median_downstream_mass_{source_id}"] = _median([
                row["downstream_positive_spike_mass"] for row in source_rows
            ])
        dose_rows.append(summary)

    decision = adjudicate(
        dose_rows,
        minimum_networks=config["decision"]["minimum_networks_per_source_at_same_dose"],
        source_ids=source_ids,
    )
    figures = root / "figures"
    png, pdf = _plot(dose_rows, source_ids, figures)
    readme = figures / "README.md"
    readme.write_text(
        "### rev10_d4_1_fresh_network_packet_dose.png\n\n"
        "左图比较冻结 A/B 源在 6 张全新网络中的干净预期方向响应数，虚线是 5/6 判据；右图给出同一响应相对冻结患者训练支持域的中位 OOD 距离。横轴是同步注入的 E 神经元数，所有剂量与 sham 使用相同随机数。\n\n"
        "**关注点**：A、B 是否在同一最小剂量同时达到 5/6；本图是强制路由剂量诊断，不是 Fig.4 自发模式验收。\n"
    )
    payload = {
        **decision,
        "scientific_role": config["scientific_role"],
        "dose_rows": dose_rows,
        "network_seeds": list(map(int, config["network_seeds"])),
        "source_contract": config["sources"],
        "figures": {
            "png": str(png), "png_sha256": _sha256(png),
            "pdf": str(pdf), "pdf_sha256": _sha256(pdf),
            "readme": str(readme), "readme_sha256": _sha256(readme),
        },
        "worker_inputs": worker_inputs,
        "config": {"path": str(config_path.relative_to(ROOT)), "sha256": config_sha},
        "manifest": {"path": str(manifest_path), "sha256": manifest_sha},
        "safe_claim": (
            "fresh-network forced route capacity only; source A was outcome-selected "
            "on development networks and the intervention is not spontaneous activity"
        ),
        "claim_boundary": "not Fig4, not patient blind, not a recovered core",
    }
    _atomic_csv(root / "packet_dose_summary.csv", dose_rows)
    _atomic_csv(root / "packet_dose_per_network.csv", all_rows)
    _atomic_json(root / "packet_dose_verdict.json", payload)
    print(json.dumps({
        "status": payload["status"],
        "selected_packet_fraction_of_E": payload["selected_packet_fraction_of_E"],
        "selected_packet_n_E": payload["selected_packet_n_E"],
    }, indent=2))


if __name__ == "__main__":
    main()
