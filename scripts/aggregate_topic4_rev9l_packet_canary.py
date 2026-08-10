"""Aggregate rev9-L forced-packet canary workers and freeze one packet size."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.getcwd())
from src.topic4_core_field_runner import atomic_write_json  # noqa: E402
from src.topic4_forced_source_capacity import select_packet_fraction  # noqa: E402


DEFAULT_CONFIG = "config/topic4_rev9l_forced_source.json"
ROOT = Path(__file__).resolve().parents[1]


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _runtime_provenance():
    paths = set()
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if not filename:
            continue
        path = Path(filename).resolve()
        if path.suffix != ".py":
            continue
        try:
            paths.add(str(path.relative_to(ROOT)))
        except ValueError:
            continue
    paths.add(str(Path(__file__).resolve().relative_to(ROOT)))
    paths = sorted(paths)
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--", *paths],
        cwd=ROOT, text=True).strip()
    return {
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "runtime_modules_dirty": bool(dirty),
        "runtime_module_sha256": {path: _sha256(ROOT / path) for path in paths},
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "systemd_unit": os.environ.get("REV9L_SYSTEMD_UNIT"),
    }


def _plot(rows, selection, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = sorted({row["source_id"] for row in rows})
    fractions = sorted({float(row["packet_fraction_of_E"]) for row in rows})
    colors = {source: color for source, color in zip(
        sources, ("#d1495b", "#277da1", "#59a14f", "#8c564b"))}
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.7), constrained_layout=True)

    summary_lookup = {
        row["packet_fraction_of_E"]: row for row in selection["fractions"]}
    for source in sources:
        coverage = [summary_lookup[fraction]["eligible_networks_by_source"][source]
                    for fraction in fractions]
        axes[0].plot(np.asarray(fractions) * 100.0, coverage, marker="o",
                     color=colors[source], label=source.replace("_", " "))
    axes[0].axvline(
        selection["selected"]["packet_fraction_of_E"] * 100.0,
        color="0.2", ls="--", lw=1)
    axes[0].set_xlabel("forced E neurons (%)")
    axes[0].set_ylabel("eligible networks (of 3)")
    axes[0].set_title("A  Packet readability", loc="left", weight="bold")
    axes[0].legend(frameon=False, fontsize=8)

    for source in sources:
        selected = [row for row in rows if row["source_id"] == source]
        axes[1].scatter(
            np.asarray([row["packet_fraction_of_E"] for row in selected]) * 100.0,
            [row["paired_geometry"]["downstream_positive_spike_mass"]
             for row in selected], color=colors[source], s=28, alpha=0.7)
    axes[1].set_xlabel("forced E neurons (%)")
    axes[1].set_ylabel("paired downstream excess spikes")
    axes[1].set_title("B  Downstream response", loc="left", weight="bold")

    for source in sources:
        selected = [row for row in rows if row["source_id"] == source]
        axes[2].scatter(
            np.asarray([row["packet_fraction_of_E"] for row in selected]) * 100.0,
            [row["paired_excess_readout"]["n_part"] for row in selected],
            color=colors[source], s=28, alpha=0.7)
    axes[2].axhline(7, color="0.3", ls=":", lw=1, label="rank-curve floor")
    axes[2].set_xlabel("forced E neurons (%)")
    axes[2].set_ylabel("participating contacts")
    axes[2].set_title("C  Contact readout support", loc="left", weight="bold")
    axes[2].legend(frameon=False, fontsize=8)
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"rev9l_forced_packet_canary.{suffix}", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--out")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text())
    output_root = Path(config["output_root"])
    canary_dir = output_root / "canary"
    output = Path(args.out or canary_dir / "packet_canary_summary.json")
    config_sha = _sha256(config_path)
    sources = config["packet"]["canary_sources"]
    fractions = config["packet"]["canary_fractions_of_E"]
    worker_inputs, rows = [], []
    for seed in config["network_seeds"]["canary"]:
        stem = canary_dir / "workers" / f"node_seed{int(seed)}"
        json_path, npz_path = stem.with_suffix(".json"), stem.with_suffix(".npz")
        payload = json.loads(json_path.read_text())
        if (payload["status"] != "REV9L_FORCED_SOURCE_WORKER_COMPLETE"
                or payload["arm"] != "Node" or int(payload["seed"]) != int(seed)):
            raise RuntimeError(f"canary worker identity mismatch: {json_path}")
        if payload["config"]["sha256"] != config_sha:
            raise RuntimeError(f"canary worker config hash mismatch: {json_path}")
        if payload["arrays"]["sha256"] != _sha256(npz_path):
            raise RuntimeError(f"canary worker arrays hash mismatch: {npz_path}")
        if payload["sources"] != sources or not np.allclose(
                payload["packet_fractions_of_E"], fractions):
            raise RuntimeError(f"canary source/fraction contract changed: {json_path}")
        worker_inputs.append({
            "seed": int(seed),
            "json": {"path": str(json_path), "sha256": _sha256(json_path)},
            "npz": {"path": str(npz_path), "sha256": _sha256(npz_path)},
        })
        for row in payload["runs"]:
            rows.append({"seed": int(seed), **row})

    selection = select_packet_fraction(
        rows, source_ids=sources, min_networks_per_source=2)
    summary = {
        "status": selection["status"],
        "scientific_role": (
            "instrument packet-size selection on Node canary only; not patient capacity evidence"
        ),
        "selection_rule": (
            "smallest fraction with usable paired-excess rank curve and any positive "
            "downstream response in at least 2/3 networks for each primary source, "
            "with no runaway; sparse fallback maximizes minimum source coverage"
        ),
        "selection": selection,
        "n_runs": len(rows),
        "n_pretrigger_mismatch": int(sum(
            not row["pretrigger_spikes_bit_identical"] for row in rows)),
        "n_runaway": int(sum(row["runaway_early_stop_ms"] is not None for row in rows)),
        "rows": rows,
        "worker_inputs": worker_inputs,
        "config": {"path": str(config_path), "sha256": config_sha},
        "provenance": _runtime_provenance(),
    }
    atomic_write_json(summary, output)
    figures = canary_dir / "figures"
    _plot(rows, selection, figures)
    (figures / "README.md").write_text(
        "### rev9l_forced_packet_canary.png\n"
        "在三个新 network seeds 上比较 0.5%、1% 和 2% E-neuron deterministic spike packet。"
        "左图按两个 primary source 报告可读网络数，中图是 paired downstream excess，右图是 contact rank readout 支持度。\n\n"
        "**关注点**：选择最小且跨两个 source 均可读的 packet；该选择只冻结 instrument，不是患者模式 capacity 结论。\n"
    )
    print(json.dumps({
        "status": summary["status"],
        "selected_packet_fraction_of_E": selection["selected"][
            "packet_fraction_of_E"],
        "selection": selection["fractions"],
        "n_pretrigger_mismatch": summary["n_pretrigger_mismatch"],
        "n_runaway": summary["n_runaway"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
