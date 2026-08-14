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


def _plot(audits, figure_dir):
    figure_dir.mkdir(parents=True, exist_ok=True)
    ids = list(GRAPH_IDS)
    marginal = [audits[key]["construction_q_marginal"] for key in ids]
    twohop = [audits[key]["operator"]["q_parallel_two_hop"] for key in ids]
    surround = [audits[key]["operator"]["surround_center_ratio"] for key in ids]
    latency = [audits[key]["latency"]["q95_ms"] for key in ids]
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.4), constrained_layout=True)
    for ax, values, title, ylabel in zip(
        axes,
        (marginal, twohop, surround, latency),
        ("a  Construction coordinate", "b  Actual two-hop width",
         "c  Surround / center mass", "d  Two-hop latency q95"),
        (r"$q_{\parallel}^{marginal}$", r"$q_{\parallel}^{2hop}$", "mass ratio", "ms"),
    ):
        ax.bar(ids, values, color=COLORS, edgecolor="white", linewidth=.7)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=.18)
    fig.suptitle("LC6A graph geometry: nominal reach versus functional E→I→E loop", fontsize=12)
    png = figure_dir / "lc6a_graph_and_twohop.png"
    pdf = figure_dir / "lc6a_graph_and_twohop.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    (figure_dir / "README.md").write_text(
        "### lc6a_graph_and_twohop.png\n\n"
        "这张图只审计连接，不包含任何自然发作轨迹。a 是用于构图的 E→I 与 I→E 协方差合成坐标；b 是实际加权两跳 E→I→E 算子相对冻结 E→E 轴宽的结果；c 比较患者轴远端抑制质量与中心抑制质量；d 检查扩大 reach 同时带来的物理 delay。\n\n"
        "**关注点**：真正承重的是 b、c，而不是输入 sampler 的宽度；即使 a 达标，若 b/c 不变，也不能声称功能性 inhibitory surround 已建立。\n\n"
        "### lc6a_graph_and_twohop.pdf\n\n与 PNG 相同的矢量版本。\n\n"
        "**关注点**：所有量均为 graph-only readout，不能单独证明 bounded carrier。\n"
    )
    return {"png": str(png), "pdf": str(pdf)}


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
    args = parser.parse_args()
    if not args.confirm_run:
        raise SystemExit("two-hop graph audit requires --confirm-run")
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
