#!/usr/bin/env python3
"""Plot and adjudicate the dual-core bounded-to-tonic-runaway boundary."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_topic4_dual_core_spatial_z_bifurcation import atomic_json  # noqa: E402


DEFAULT_ROOT = Path(
    "/data/hfosp_topic4_fig45_artifacts/fig5/"
    "data_driven_dual_core_spatial_z")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _scan_records(prefix: Path, gate: dict) -> list[dict]:
    payload = json.loads(prefix.with_suffix(".json").read_text())
    with np.load(prefix.with_suffix(".npz"), allow_pickle=False) as archive:
        time_ms = np.asarray(archive["time_ms"], float)
        traces = np.asarray(archive["low_initial__mean_e_trace_hz"], float)
    tail_ms = float(gate["tail_duration_ms"])
    keep = time_ms >= time_ms[-1] + (time_ms[1] - time_ms[0]) - tail_ms
    result = []
    for record in payload["records"]:
        if record["initial_branch"] != "low_initial":
            continue
        row, column = int(record["row"]), int(record["column"])
        trace = traces[row, column]
        tail = trace[keep]
        first, second = np.array_split(tail, 2)
        mean = float(np.mean(tail))
        drift = float(np.mean(second) - np.mean(first))
        regions = np.asarray(record["regional_tail_rate_hz"], float)
        runaway = bool(
            mean >= float(gate["minimum_population_mean_rate_hz"])
            and np.min(regions) >= float(gate["minimum_each_regional_rate_hz"])
            and abs(drift) <= float(
                gate["maximum_absolute_half_tail_drift_hz"]))
        result.append({
            "s": float(record["s"]),
            "mean_e_rate_hz": mean,
            "half_tail_drift_hz": drift,
            "regional_e_rate_hz": regions.tolist(),
            "runaway": runaway,
            "trace": trace,
            "time_ms": time_ms,
            "source_json": str(prefix.with_suffix(".json")),
            "source_npz": str(prefix.with_suffix(".npz")),
        })
    return result


def _classify(records: list[dict], gate: dict, bounded_gate: dict) -> None:
    for record in records:
        bounded = bool(
            record["mean_e_rate_hz"] <= float(
                bounded_gate["maximum_population_mean_rate_hz"])
            and abs(record["half_tail_drift_hz"]) <= float(
                gate["maximum_absolute_half_tail_drift_hz"]))
        record["state"] = (
            "runaway" if record["runaway"]
            else "bounded" if bounded else "transitioning")


def _readme(path: Path, *, lower: float, upper: float, fold: float) -> None:
    path.write_text(
        "### dualcore-bounded-to-tonic-runaway-boundary.png\n\n"
        "这是一张诊断图，不是正式 Fig. 5。A 展示带动态 M 的 frozen-spatial-Z "
        f"平衡支 saddle-node（s={fold:.6f}）；B–D 展示原生延迟系统在同一低态"
        "初值下的 10 s 非线性结果；C 的红线还完整承接蓝线末端的 synapses、delay histories 与 M。"
        "蓝色表示 bounded spatial attractor，红色表示"
        "满足预先冻结的近饱和 tonic-runaway 门槛，灰色表示有限时长内仍在过渡。\n\n"
        f"**关注点**：原生延迟 stable/runaway 分界目前被夹在 s={lower:.4f} 与 "
        f"s={upper:.4f} 之间；它与较早的平衡支 fold 不重合，因此不能把 fold "
        "单独写成 delay-system runaway bifurcation。\n\n"
        "### dualcore-tonic-runaway-hysteresis-discovery.png\n\n"
        "连续保留 rates、synaptic currents、全部 delay histories 与 M 的上/下扫。该图使用"
        "守恒连接权重的 5× delay-bin coarsening，只用于判断是否存在滞后，不用于给出正式临界值。\n\n"
        "**关注点**：上扫和下扫走不同分支，说明 stable/runaway 不是由一个与初值无关的 s 阈值分开；"
        "正式相图应画 basin/hysteresis，而不是唯一分岔线。\n",
        encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fold-prefix", type=Path,
        default=DEFAULT_ROOT / "runaway_boundary/localized_to_global_fold_m1")
    parser.add_argument(
        "--scan-prefix", type=Path, action="append",
        default=None, help="10-s low-initial scan prefix; may be repeated")
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "config/topic4_dual_core_runaway_boundary_v1.json")
    parser.add_argument(
        "--hysteresis-json", type=Path,
        default=DEFAULT_ROOT / "runaway_boundary/continuous_state_hysteresis.json")
    parser.add_argument(
        "--edge-prefix", type=Path,
        default=DEFAULT_ROOT / "runaway_boundary/native_edge_tracking_0p428_to_0p429")
    parser.add_argument(
        "--out-dir", type=Path,
        default=DEFAULT_ROOT / "runaway_boundary/figures")
    args = parser.parse_args()
    scan_prefixes = args.scan_prefix or [
        DEFAULT_ROOT / "zm_regime_map/localized_fold_long10s_m1",
        DEFAULT_ROOT / "zm_regime_map/runaway_plateau_boundary_s0p42_0p44_m1_10s",
        DEFAULT_ROOT / "zm_regime_map/runaway_plateau_boundary_zoom_s0p426_0p429_m1_10s",
    ]
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    gate = config["predeclared_tonic_runaway_gate"]
    bounded_gate = config["bounded_gate"]
    records = []
    for prefix in scan_prefixes:
        records.extend(_scan_records(prefix.resolve(), gate))
    # Repeated s values must be numerically identical; retain the longest trace.
    unique = {}
    for record in records:
        key = round(record["s"], 12)
        if key not in unique or record["time_ms"].size > unique[key]["time_ms"].size:
            unique[key] = record
    records = sorted(unique.values(), key=lambda item: item["s"])
    _classify(records, gate, bounded_gate)
    bounded = [record for record in records if record["state"] == "bounded"]
    runaway = [record for record in records if record["state"] == "runaway"]
    if not bounded or not runaway:
        raise RuntimeError("10-s scans do not bracket bounded and runaway states")
    lower = max(record["s"] for record in bounded)
    upper = min(record["s"] for record in runaway if record["s"] > lower)

    fold_prefix = args.fold_prefix.resolve()
    fold_payload = json.loads(fold_prefix.with_suffix(".json").read_text())
    fold = fold_payload["localized_to_global_candidate_fold"]
    fold_s = float(fold["s"])
    with np.load(fold_prefix.with_suffix(".npz"), allow_pickle=False) as archive:
        branch_s = np.asarray(archive["branch__s"], float)
        branch_surround = np.asarray(archive["branch__surround_hz"], float)
        tangent = np.asarray(archive["branch__tangent_s"], float)
    turn = int(np.flatnonzero(tangent[:-1] * tangent[1:] <= 0.0)[0])

    colors = {"bounded": "#355C7D", "transitioning": "#8A8A8A",
              "runaway": "#C53B32"}
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    fig = plt.figure(figsize=(10.4, 6.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=(1.0, 1.05))
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    ax_a.plot(branch_s[:turn + 1], branch_surround[:turn + 1],
              color="#355C7D", lw=2.0, label="localized equilibrium locus")
    ax_a.plot(branch_s[turn + 1:], branch_surround[turn + 1:],
              color="#777777", lw=1.7, ls="--", label="saddle locus")
    ax_a.scatter([fold_s], [fold["regional_e_rate_hz"]["surround"]],
                 s=42, color="#C53B32", zorder=4, label="equilibrium fold")
    ax_a.set(xlabel="Core disinhibition  s = 1 - Zcore",
             ylabel="Surround E rate (Hz)", title="Equilibrium organizer")
    ax_a.legend(frameon=False, fontsize=8, loc="best")
    ax_a.text(0.02, 0.04, "Both loci are delay-unstable (~37 Hz)",
              transform=ax_a.transAxes, fontsize=8, color="#555555")

    s = np.asarray([record["s"] for record in records])
    mean = np.asarray([record["mean_e_rate_hz"] for record in records])
    ax_b.axhspan(0, float(bounded_gate["maximum_population_mean_rate_hz"]),
                 color="#355C7D", alpha=0.06)
    ax_b.axhspan(float(gate["minimum_population_mean_rate_hz"]), 500,
                 color="#C53B32", alpha=0.06)
    ax_b.plot(s, mean, color="#777777", lw=1.0, zorder=1)
    for state in colors:
        choose = np.asarray([record["state"] == state for record in records])
        ax_b.scatter(s[choose], mean[choose], s=34, color=colors[state],
                     label=state, zorder=3)
    ax_b.axvline(fold_s, color="#6B4C9A", ls=":", lw=1.3)
    ax_b.axvspan(lower, upper, color="#C53B32", alpha=0.13,
                 label="runaway boundary bracket")
    ax_b.axhline(float(gate["minimum_population_mean_rate_hz"]),
                 color="#C53B32", ls="--", lw=1.0)
    ax_b.set(xlabel="Core disinhibition  s = 1 - Zcore",
             ylabel="Terminal population E rate (Hz)",
             title="Native-delay 10-s state boundary", ylim=(0, 500))
    ax_b.legend(frameon=False, fontsize=7.5, ncol=2, loc="upper left")

    edge_prefix = args.edge_prefix.resolve()
    edge_payload = json.loads(
        edge_prefix.with_suffix(".json").read_text(encoding="utf-8"))
    with np.load(edge_prefix.with_suffix(".npz"), allow_pickle=False) as edge:
        edge_time = np.asarray(edge["time_ms"], float)
        edge_traces = (
            np.asarray(edge["source_trace_hz"], float),
            np.asarray(edge["target_trace_hz"], float),
        )
    for record, trace, state in zip(
            edge_payload["records"], edge_traces, ("bounded", "runaway")):
        ax_c.plot(edge_time / 1000.0, trace,
                  color=colors[state], lw=1.0,
                  label=f"s={record['s']:.4f}  {state}")
    ax_c.axhline(float(gate["minimum_population_mean_rate_hz"]),
                 color="#C53B32", ls="--", lw=1.0)
    ax_c.set(xlabel="Time (s)", ylabel="Population E rate (Hz)",
             title="Boundary-side trajectories", ylim=(0, 500))
    ax_c.legend(frameon=False, fontsize=8)

    regional = np.asarray([record["regional_e_rate_hz"] for record in records])
    for index, (label, color) in enumerate((
            ("core A", "#E67E43"), ("core B", "#D95F8D"),
            ("surround", "#2AA7B8"))):
        ax_d.plot(s, regional[:, index], marker="o", ms=3.5, lw=1.4,
                  color=color, label=label)
    ax_d.axhline(float(gate["minimum_each_regional_rate_hz"]),
                 color="#555555", ls="--", lw=1.0)
    ax_d.axvspan(lower, upper, color="#C53B32", alpha=0.13)
    ax_d.set(xlabel="Core disinhibition  s = 1 - Zcore",
             ylabel="Terminal regional E rate (Hz)",
             title="Spatial recruitment", ylim=(0, 500))
    ax_d.legend(frameon=False, fontsize=8)

    for label, axis in zip("ABCD", (ax_a, ax_b, ax_c, ax_d)):
        axis.text(-0.13, 1.05, label, transform=axis.transAxes,
                  fontsize=14, fontweight="bold", va="top")
    fig.suptitle("Dual-core spatial Z/M: bounded-to-tonic-runaway boundary",
                 fontsize=13, fontweight="bold")

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / "dualcore-bounded-to-tonic-runaway-boundary"
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(stem.with_suffix("." + suffix), dpi=300,
                    bbox_inches="tight")
    plt.close(fig)

    hysteresis_path = args.hysteresis_json.resolve()
    hysteresis = json.loads(hysteresis_path.read_text(encoding="utf-8"))
    fig_h, ax_h = plt.subplots(figsize=(5.5, 3.7), constrained_layout=True)
    for direction, color, marker in (
            ("up", "#D97732", ">"), ("down", "#355C7D", "<")):
        subset = [record for record in hysteresis["records"]
                  if record["direction"] == direction]
        hx = np.asarray([record["s"] for record in subset])
        hy = np.asarray([record["population_tail_mean_hz"] for record in subset])
        ax_h.plot(hx, hy, color=color, marker=marker, ms=4, lw=1.5,
                  label=f"{direction}-sweep")
    ax_h.axhline(float(gate["minimum_population_mean_rate_hz"]),
                 color="#C53B32", ls="--", lw=1.0)
    ax_h.axvspan(lower, upper, color="#C53B32", alpha=0.13,
                 label="fixed-preparation bracket")
    ax_h.set(xlabel="Core disinhibition  s = 1 - Zcore",
             ylabel="Segment-tail population E rate (Hz)",
             title="Continuous-state hysteresis (discovery)", ylim=(0, 500))
    ax_h.legend(frameon=False, fontsize=8)
    hstem = out_dir / "dualcore-tonic-runaway-hysteresis-discovery"
    for suffix in ("png", "pdf", "svg"):
        fig_h.savefig(hstem.with_suffix("." + suffix), dpi=300,
                      bbox_inches="tight")
    plt.close(fig_h)
    _readme(out_dir / "README.md", lower=lower, upper=upper, fold=fold_s)

    serializable = [{key: value for key, value in record.items()
                     if key not in ("trace", "time_ms")}
                    for record in records]
    metadata = {
        "status": "NATIVE_DELAY_RUNAWAY_BOUNDARY_BRACKETED",
        "substrate": config["substrate"],
        "equilibrium_fold_s": fold_s,
        "native_delay_runaway_boundary_s_bracket": [lower, upper],
        "fold_to_boundary_separation_s": [lower - fold_s, upper - fold_s],
        "tonic_runaway_gate": gate,
        "bounded_gate": bounded_gate,
        "records": serializable,
        "interpretation": (
            "The equilibrium saddle-node is an earlier organizer, not the "
            "native-delay runaway bifurcation. Native carried-state tracking "
            "confirms an upper bounded-attractor escape bracket at 0.428-0.429; "
            "its local type is unresolved between a limit-cycle fold and a "
            "boundary crisis."),
        "sources": {
            "fold_json": {"path": str(fold_prefix.with_suffix('.json')),
                          "sha256": _sha256(fold_prefix.with_suffix('.json'))},
            "fold_npz": {"path": str(fold_prefix.with_suffix('.npz')),
                         "sha256": _sha256(fold_prefix.with_suffix('.npz'))},
            "scans": [{
                "json": {"path": str(prefix.resolve().with_suffix('.json')),
                         "sha256": _sha256(prefix.resolve().with_suffix('.json'))},
                "npz": {"path": str(prefix.resolve().with_suffix('.npz')),
                        "sha256": _sha256(prefix.resolve().with_suffix('.npz'))},
            } for prefix in scan_prefixes],
            "config": {"path": str(config_path), "sha256": _sha256(config_path)},
            "hysteresis": {"path": str(hysteresis_path),
                           "sha256": _sha256(hysteresis_path)},
            "edge_json": {"path": str(edge_prefix.with_suffix('.json')),
                          "sha256": _sha256(edge_prefix.with_suffix('.json'))},
            "edge_npz": {"path": str(edge_prefix.with_suffix('.npz')),
                         "sha256": _sha256(edge_prefix.with_suffix('.npz'))},
        },
        "figure": {suffix: str(stem.with_suffix('.' + suffix))
                   for suffix in ("png", "pdf", "svg")},
        "hysteresis_figure": {
            suffix: str(hstem.with_suffix('.' + suffix))
            for suffix in ("png", "pdf", "svg")},
    }
    atomic_json(metadata, stem.with_suffix(".metadata.json"))
    print(json.dumps({
        "status": metadata["status"],
        "fold_s": fold_s,
        "runaway_boundary_s_bracket": [lower, upper],
        "figure": str(stem.with_suffix('.png')),
    }, indent=2))


if __name__ == "__main__":
    main()
