#!/usr/bin/env python3
"""Weighted coarse E->I->E audit for the frozen LC6A graph family."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts", ROOT / "src/snn_engine"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_m4_phaseplane as PP  # noqa: E402
from src.topic4_fcxr_lc6_surround import (  # noqa: E402
    EToIGraph, extract_i_to_e, graph_sha256,
)
from src.topic4_fcxr_lc6_twohop import (  # noqa: E402
    coarse_two_hop_operator,
    sample_two_hop_latencies,
    spatial_bins,
    summarize_two_hop_operator,
)


OUT = ROOT / "results/topic4_sef_hfo/fcxr_lc6a_patient_axis_surround"
GRAPH_IDS = ("C0", "C1", "Q1", "Q2", "Q3")
COLORS = ("#555555", "#999999", "#4C78A8", "#F2A541", "#C44E52")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _load_graph(path):
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        graph = EToIGraph(
            np.asarray(z["sources"], np.int32),
            np.asarray(z["weights"]),
            np.asarray(z["delay_steps"], np.int32),
        )
        expected = str(z["graph_sha256"][0])
        metadata = json.loads(str(z["metadata_json"][0]))
    actual = graph_sha256(graph)
    if actual != expected or metadata.get("graph_sha256") != expected:
        raise RuntimeError(f"graph artifact hash mismatch: {path}")
    return graph, metadata


MICROSTATE_Q_TOLERANCE = 0.05


def _plot(audits, figure_dir):
    """Four independent graph-only questions; the ladder itself is one panel, not two."""

    figure_dir.mkdir(parents=True, exist_ok=True)
    ids = list(GRAPH_IDS)
    marginal = [audits[key]["construction_q_marginal"] for key in ids]
    operators = [audits[key]["operator"] for key in ids]
    twohop = [row["q_parallel_two_hop"] for row in operators]
    fig, axes = plt.subplots(1, 4, figsize=(15.0, 3.9), constrained_layout=True)

    # a. Did the construction coordinate actually land on the functional two-hop loop?
    ax = axes[0]
    anchor = marginal[ids.index("C0")]
    ax.axvspan(
        anchor - MICROSTATE_Q_TOLERANCE, anchor + MICROSTATE_Q_TOLERANCE,
        color="#BBBBBB", alpha=.3, lw=0,
    )
    span = [min(marginal + twohop) - .05, max(marginal + twohop) + .05]
    ax.plot(span, span, color="#888888", lw=.9, ls="--", zorder=1)
    for index, key in enumerate(ids):
        ax.scatter(marginal[index], twohop[index], s=70, color=COLORS[index], zorder=3)
        # C0/C1/Q1 sit on top of each other; stagger their labels off the cluster.
        offset = ((-25, -11), (-25, 9), (10, -12), (10, -4), (10, -4))[index]
        ax.annotate(
            key, (marginal[index], twohop[index]), textcoords="offset points",
            xytext=offset, fontsize=9, color=COLORS[index],
        )
    ax.set_xlabel(r"construction $q_{\parallel}^{marginal}$")
    ax.set_ylabel(r"actual $q_{\parallel}^{2hop}$")
    ax.set_title("a  Construction coordinate tracks the\n     real loop; C0/C1/Q1 are one rung", fontsize=9.5)
    ax.text(
        .04, .9, f"grey: registered ±{MICROSTATE_Q_TOLERANCE:g}\nsame-$q$ tolerance",
        transform=ax.transAxes, fontsize=7.5, color="#444444", va="top",
    )

    # b. Did the widening stay axial, or did the perpendicular kernel drift with it?
    ax = axes[1]
    x = np.arange(len(ids), dtype=float)
    ax.bar(
        x - .19, [row["sigma_parallel_mm"] for row in operators], width=.36,
        color=COLORS, edgecolor="white", linewidth=.7,
    )
    ax.bar(
        x + .19, [row["sigma_perpendicular_mm"] for row in operators], width=.36,
        color=COLORS, edgecolor="white", linewidth=.7, alpha=.4, hatch="///",
    )
    ax.set_xticks(x, ids)
    ax.set_ylabel("two-hop σ (mm)")
    ax.set_title("b  Axial σ grows, perpendicular σ\n     does not: the change is axial", fontsize=9.5)
    ax.text(.04, .92, "solid: axial   hatched: perpendicular", transform=ax.transAxes, fontsize=7.5, color="#444444")

    # c. Where did the fixed inhibitory mass move?
    ax = axes[2]
    total = [row["total_inhibitory_magnitude"] for row in operators]
    ax.bar(
        x - .19, [row["center_mass"] / t for row, t in zip(operators, total)], width=.36,
        color=COLORS, edgecolor="white", linewidth=.7,
    )
    ax.bar(
        x + .19, [row["surround_mass"] / t for row, t in zip(operators, total)], width=.36,
        color=COLORS, edgecolor="white", linewidth=.7, alpha=.4, hatch="///",
    )
    ax.set_xticks(x, ids)
    ax.set_ylabel("fraction of total inhibitory mass")
    ax.set_title("c  Surround gains what the centre\n     loses (fixed 800 in-degree)", fontsize=9.5)
    ax.text(.04, .92, "solid: centre   hatched: surround", transform=ax.transAxes, fontsize=7.5, color="#444444")

    # d. What conduction delay did the wider reach cost?
    ax = axes[3]
    ax.bar(
        x - .19, [audits[key]["latency"]["median_ms"] for key in ids], width=.36,
        color=COLORS, edgecolor="white", linewidth=.7,
    )
    ax.bar(
        x + .19, [audits[key]["latency"]["q95_ms"] for key in ids], width=.36,
        color=COLORS, edgecolor="white", linewidth=.7, alpha=.4, hatch="///",
    )
    ax.set_xticks(x, ids)
    ax.set_ylabel("two-hop latency (ms)")
    ax.set_title("d  Wider reach also costs\n     disynaptic delay", fontsize=9.5)
    ax.text(.04, .92, "solid: median   hatched: q95", transform=ax.transAxes, fontsize=7.5, color="#444444")

    for ax in axes:
        ax.grid(axis="y", alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("LC6A graph geometry: realized reach, axial specificity, mass reallocation, delay cost", fontsize=12)
    png = figure_dir / "lc6a_graph_and_twohop.png"
    pdf = figure_dir / "lc6a_graph_and_twohop.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    (figure_dir / "README.md").write_text(
        "### lc6a_graph_and_twohop.png\n\n"
        "只审计连接，不含任何自然发作轨迹。a 把构图坐标与实际两跳算子放在同一张散点上（对角线=两者一致），"
        "并画出注册的同 q 容差带——C0/C1/Q1 三张图落在同一带里，实际只有 Q2、Q3 两个真 reach 档；"
        "b 检查加宽是否只发生在轴向；c 显示在 800 条 E→I 输入总量不变的前提下，抑制质量从中心搬到了周边；"
        "d 给出加宽同时付出的两跳传导延迟。\n\n"
        "**关注点**：承重的是 a 的纵轴（实际两跳宽度）与 c 的搬运方向；b 是 confound 检查，"
        "d 说明 reach 不是免费的。graph-only readout 不能单独证明 bounded carrier。\n\n"
        "### lc6a_graph_and_twohop.pdf\n\n与 PNG 相同的矢量版本。\n\n"
        "**关注点**：所有量均为 graph-only readout，不能单独证明 bounded carrier。\n"
    )
    return {"png": str(png), "pdf": str(pdf)}


def replot_only():
    """Re-render the frozen two-hop audit figure without recomputing the operator."""

    payload = json.loads((OUT / "two_hop_kernel_audit.json").read_text())
    if payload.get("status") != "COMPLETE":
        raise RuntimeError("frozen two-hop audit is not complete")
    return _plot(payload["audits"], OUT / "figures")


def run(*, n_bins_axis=24, n_paths=20000, audit_seed=662000):
    done = OUT / "DONE_LC6A_GRAPH_FAMILY.json"
    graph_audit_path = OUT / "graph_audit.json"
    if not done.is_file() or not graph_audit_path.is_file():
        raise RuntimeError("graph family must complete before the two-hop audit")
    graph_audit = json.loads(graph_audit_path.read_text())
    S = PP.build_substrate(1)
    i2e = extract_i_to_e(S["net"], S["NE"], S["NI"])
    bins = spatial_bins(S["posE"], sheet_size_mm=S["L"], n_bins_axis=n_bins_axis)
    ee_width = graph_audit["frozen_reference_widths"]["e_to_e"]
    audits, operators = {}, {}
    for index, condition in enumerate(GRAPH_IDS):
        graph_path = OUT / f"graphs/{condition}.npz"
        graph, metadata = _load_graph(graph_path)
        operator = coarse_two_hop_operator(
            graph, i2e, bins, n_e=S["NE"], n_i=S["NI"],
        )
        operator_summary = summarize_two_hop_operator(
            operator, bins, S["axis_unit"],
            ee_sigma_parallel_mm=ee_width["sigma_parallel_mm"],
            ee_sigma_perpendicular_mm=ee_width["sigma_perpendicular_mm"],
            edge_margin_mm=1.0,
        )
        latency = sample_two_hop_latencies(
            graph, i2e, n_e=S["NE"], n_i=S["NI"], engine_dt_ms=S["p"].dt,
            n_paths=n_paths, audit_seed=audit_seed + index,
        )
        audits[condition] = {
            "graph_sha256": graph_sha256(graph),
            "construction_q_marginal": metadata["construction_q"],
            "operator": operator_summary,
            "latency": latency,
        }
        operators[f"{condition}_magnitude"] = operator.toarray().astype(np.float32)
        operators[f"{condition}_signed"] = (-operator).toarray().astype(np.float32)
    operator_path = OUT / "two_hop_operators.npz"
    tmp = operator_path.with_name(operator_path.name + ".tmp.npz")
    np.savez_compressed(
        tmp,
        n_bins_axis=np.asarray([n_bins_axis], np.int32),
        centers=bins.centers.astype(np.float32),
        **operators,
    )
    os.replace(tmp, operator_path)
    figures = _plot(audits, OUT / "figures")
    payload = {
        "status": "COMPLETE",
        "stage": "LC6A_TWO_HOP_GRAPH_AUDIT",
        "graph_audit": str(graph_audit_path),
        "graph_audit_sha256": _sha(graph_audit_path),
        "n_bins_axis": int(n_bins_axis),
        "n_latency_paths_per_graph": int(n_paths),
        "audit_seed": int(audit_seed),
        "model_internal_state_used": False,
        "trajectory_outcome_used": False,
        "audits": audits,
        "operator_artifact": str(operator_path),
        "operator_artifact_sha256": _sha(operator_path),
        "figures": figures,
        "source_sha256": {
            "module": _sha(ROOT / "src/topic4_fcxr_lc6_twohop.py"),
            "runner": _sha(Path(__file__)),
        },
    }
    _write_json(OUT / "two_hop_kernel_audit.json", payload)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-bins-axis", type=int, default=24)
    parser.add_argument("--n-paths", type=int, default=20000)
    parser.add_argument("--audit-seed", type=int, default=662000)
    parser.add_argument("--confirm-run", action="store_true")
    parser.add_argument(
        "--replot-only", action="store_true",
        help="re-render the figure from the frozen audit JSON; recomputes nothing",
    )
    args = parser.parse_args()
    if not args.confirm_run:
        raise SystemExit("two-hop graph audit requires --confirm-run")
    if args.replot_only:
        print(json.dumps(replot_only(), indent=2, sort_keys=True))
        return
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / ".two_hop.lock").open("w") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit("LC6A two-hop audit is already running") from exc
        running = OUT / "RUNNING_LC6A_TWO_HOP_AUDIT.json"
        failed = OUT / "FAILED_LC6A_TWO_HOP_AUDIT.json"
        done = OUT / "DONE_LC6A_TWO_HOP_AUDIT.json"
        _write_json(running, {"status": "RUNNING", "pid": os.getpid()})
        try:
            payload = run(
                n_bins_axis=args.n_bins_axis, n_paths=args.n_paths,
                audit_seed=args.audit_seed,
            )
            _write_json(done, {
                "status": "DONE", "audit": str(OUT / "two_hop_kernel_audit.json"),
            })
            failed.unlink(missing_ok=True)
            print(json.dumps(payload, indent=2, sort_keys=True))
        except BaseException as exc:
            _write_json(failed, {
                "status": "FAILED", "error": f"{type(exc).__name__}: {exc}",
            })
            raise
        finally:
            running.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
