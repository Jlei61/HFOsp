#!/usr/bin/env python3
"""Plot the numerical-convergence audit for the dual-core oscillation map."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np


DEFAULT_ROOT = Path(
    "/data/hfosp_topic4_fig45_artifacts/fig5/"
    "data_driven_dual_core_spatial_z/oscillatory_phase_map")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(handle)
    try:
        Path(temporary).write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _edges(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, float)
    middle = 0.5 * (values[1:] + values[:-1])
    return np.r_[values[0] - (middle[0] - values[0]), middle,
                 values[-1] + (values[-1] - middle[-1])]


def _panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(-0.16, 1.08, label, transform=axis.transAxes, fontsize=15,
              fontweight="bold", ha="left", va="top")


def _load_native(paths: list[Path]) -> list[dict]:
    records = []
    for path in paths:
        records.extend(
            row for row in json.loads(path.read_text())["records"]
            if row["initial_branch"] == "high_initial")
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    discovery_json = root / "refined_s_tau_dt0p2.json"
    discovery_npz = root / "refined_s_tau_dt0p2.npz"
    native_paths = [
        root / "native_validation_s_tau.json",
        root / "native_candidate_s0p28_t8.json",
    ]
    native_npz = root / "native_candidate_s0p28_t8.npz"
    native_full_json = root / "native_refined_s_tau_dt0p1.json"
    native_full_npz = root / "native_refined_s_tau_dt0p1.npz"
    discovery = json.loads(discovery_json.read_text())
    grid = np.load(discovery_npz)
    native_records = _load_native(native_paths)
    native_grid = np.load(native_npz)
    s = np.asarray(grid["core_disinhibition_s"], float)
    tau = np.asarray(grid["tau_d_GABA_ms"], float)
    code = np.asarray(grid["high_initial__state_code"], int)
    depth = np.asarray(grid["high_initial__modulation_depth"], float)
    mean = np.asarray(grid["high_initial__mean_rate_hz"], float)
    frequency = np.asarray(grid["high_initial__dominant_hz"], float)
    full_native_available = native_full_json.exists() and native_full_npz.exists()
    if full_native_available:
        native_full = json.loads(native_full_json.read_text())
        native_full_grid = np.load(native_full_npz)
        map_grid = native_full_grid
        map_dt = float(native_full["integration"]["discovery_dt_ms"])
        native_paths = [native_full_json]
        native_records = _load_native(native_paths)
    else:
        map_grid = grid
        map_dt = float(discovery["integration"]["discovery_dt_ms"])
    map_code = np.asarray(map_grid["high_initial__state_code"], int)
    map_depth = np.asarray(map_grid["high_initial__modulation_depth"], float)
    map_mean = np.asarray(map_grid["high_initial__mean_rate_hz"], float)
    map_frequency = np.asarray(map_grid["high_initial__dominant_hz"], float)

    fig = plt.figure(figsize=(11.4, 7.0), constrained_layout=False)
    layout = fig.add_gridspec(2, 2, left=0.07, right=0.96, bottom=0.09,
                              top=0.91, wspace=0.30, hspace=0.38)
    ax_a = fig.add_subplot(layout[0, 0])
    ax_b = fig.add_subplot(layout[0, 1])
    ax_c = fig.add_subplot(layout[1, 0])
    ax_d = fig.add_subplot(layout[1, 1])
    fig.suptitle(
        "Dual-core oscillatory-state search: state map and numerical convergence",
        fontsize=13, fontweight="bold", y=0.965)

    colors = ["#E7E9ED", "#7BA7C7", "#D56A5B", "#7C4D9B"]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    mesh = ax_a.pcolormesh(_edges(s), _edges(tau), map_code.T, cmap=cmap,
                           norm=norm, shading="flat")
    labels = ("low", "localized", "tonic", "oscillatory")
    colorbar = fig.colorbar(mesh, ax=ax_a, ticks=[0, 1, 2, 3], fraction=0.048,
                            pad=0.03)
    colorbar.ax.set_yticklabels(labels, fontsize=7)
    ax_a.set_title(f"{map_dt:.1f}-ms nonlinear high-basin outcome", fontsize=10,
                   fontweight="bold")
    ax_a.set_xlabel("core disinhibition  $s$  ($Z_{A,B}=1-s$)")
    ax_a.set_ylabel(r"GABA decay  $\tau_{\mathrm{GABA}}$ (ms)")
    _panel_label(ax_a, "A")

    image = ax_b.pcolormesh(_edges(s), _edges(tau), map_depth.T, cmap="magma",
                            vmin=0.10, vmax=0.40, shading="flat")
    global_gate = map_mean >= 120.0
    for row, ss in enumerate(s):
        for column, tt in enumerate(tau):
            marker = "*" if map_code[row, column] == 3 else (
                "o" if global_gate[row, column] else "x")
            kwargs = ({"facecolor": "white", "edgecolor": "black"}
                      if marker != "x" else {"color": "black"})
            ax_b.scatter(ss, tt, marker=marker, s=52 if marker == "*" else 25,
                         linewidth=0.8, **kwargs)
    fig.colorbar(image, ax=ax_b, fraction=0.048, pad=0.03,
                 label="cycle modulation depth")
    ax_b.set_title("Clause map (× localized; ○ global; ★ all gates)", fontsize=10,
                   fontweight="bold")
    ax_b.set_xlabel("core disinhibition  $s$")
    ax_b.set_ylabel(r"GABA decay  $\tau_{\mathrm{GABA}}$ (ms)")
    _panel_label(ax_b, "B")

    trace_discovery = np.asarray(
        grid["high_initial__mean_e_trace_hz"][4, 3], float)
    time_discovery = np.asarray(grid["time_ms"], float)
    if full_native_available:
        trace_native = np.asarray(
            native_full_grid["high_initial__mean_e_trace_hz"][4, 3], float)
        time_native = np.asarray(native_full_grid["time_ms"], float)
    else:
        trace_native = np.asarray(
            native_grid["high_initial__mean_e_trace_hz"][0, 0], float)
        time_native = np.asarray(native_grid["time_ms"], float)
    keep_discovery = time_discovery >= time_discovery[-1] - 999.8
    keep_native = time_native >= time_native[-1] - 999.9
    ax_c.plot(time_discovery[keep_discovery] - time_discovery[keep_discovery][0],
              trace_discovery[keep_discovery], color="#7C4D9B", lw=1.0,
              label="dt=0.2 ms: pass (depth 0.223)")
    ax_c.plot(time_native[keep_native] - time_native[keep_native][0],
              trace_native[keep_native], color="#D56A5B", lw=0.9,
              label="dt=0.1 ms: tonic (depth 0.172)")
    ax_c.set_xlim(0, 1000)
    ax_c.set_xlabel("last 1 s of trajectory (ms)")
    ax_c.set_ylabel("population E rate (Hz)")
    ax_c.set_title(r"Same point: $s=0.28$, $\tau_{\mathrm{GABA}}=8$ ms",
                   fontsize=10, fontweight="bold")
    ax_c.legend(frameon=False, fontsize=8, loc="upper right")
    _panel_label(ax_c, "C")

    if full_native_available:
        depth_inflation = depth - map_depth
        limit = max(0.05, float(np.nanmax(np.abs(depth_inflation))))
        difference = ax_d.pcolormesh(
            _edges(s), _edges(tau), depth_inflation.T, cmap="coolwarm",
            vmin=-limit, vmax=limit, shading="flat")
        ax_d.scatter(0.28, 8.0, marker="x", s=60, color="black", lw=1.2)
        ax_d.set_xlabel("core disinhibition  $s$")
        ax_d.set_ylabel(r"GABA decay  $\tau_{\mathrm{GABA}}$ (ms)")
        ax_d.set_title("Step-size bias: depth(0.2 ms) − depth(0.1 ms)",
                       fontsize=10, fontweight="bold")
        fig.colorbar(difference, ax=ax_d, fraction=0.048, pad=0.03,
                     label="modulation-depth difference")
    else:
        native_s = np.asarray([row["s"] for row in native_records], float)
        native_tau = np.asarray(
            [row["tau_d_GABA_ms"] for row in native_records], float)
        native_depth = np.asarray(
            [row["whole_tail"]["modulation_depth"] for row in native_records], float)
        native_freq = np.asarray(
            [row["whole_tail"]["dominant_hz"] for row in native_records], float)
        sizes = 45 + 150 * np.clip((native_depth - 0.10) / 0.12, 0, 1)
        points = ax_d.scatter(native_s, native_tau, c=native_depth, s=sizes,
                              cmap="magma", vmin=0.10, vmax=0.22,
                              edgecolor="black", lw=0.65)
        for ss, tt, dd, ff in zip(
                native_s, native_tau, native_depth, native_freq):
            ax_d.text(ss + 0.002, tt + 0.10, f"{ff:.0f} Hz\n{dd:.3f}",
                      fontsize=7, ha="left", va="bottom")
        ax_d.axvline(0.337591, color="#245B9B", lw=1.0, ls="--",
                     label="low fixed-point fold")
        ax_d.set_xlim(0.245, 0.35)
        ax_d.set_ylim(3.2, 9.3)
        ax_d.set_xlabel("core disinhibition  $s$")
        ax_d.set_ylabel(r"GABA decay  $\tau_{\mathrm{GABA}}$ (ms)")
        ax_d.set_title("0.1-ms validation: all tested high states are tonic",
                       fontsize=10, fontweight="bold")
        ax_d.legend(frameon=False, fontsize=8, loc="upper right")
        fig.colorbar(points, ax=ax_d, fraction=0.048, pad=0.03,
                     label="cycle modulation depth")
    _panel_label(ax_d, "D")

    for axis in (ax_a, ax_b, ax_c, ax_d):
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=8)

    out_dir = root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / "dualcore-oscillation-phase-convergence-audit"
    outputs = {}
    for suffix, kwargs in (("png", {"dpi": 300}), ("pdf", {}), ("svg", {})):
        path = stem.with_suffix("." + suffix)
        fig.savefig(path, bbox_inches="tight", facecolor="white", **kwargs)
        outputs[suffix] = {"path": str(path), "sha256": _sha256(path)}
    plt.close(fig)
    metadata = {
        "status": ("NATIVE_GRID_OSCILLATORY_PHASE_NOT_ESTABLISHED"
                   if full_native_available
                   else "OSCILLATORY_PHASE_NOT_NATIVE_DT_ESTABLISHED"),
        "discovery_dt_ms": discovery["integration"]["discovery_dt_ms"],
        "native_dt_ms": 0.1,
        "discovery_pass_point": {
            "s": 0.28, "tau_d_GABA_ms": 8.0,
            "modulation_depth": float(depth[4, 3]),
        },
        "native_same_point": {
            "state": "tonic_recruited", "modulation_depth": 0.17182256151139705,
        },
        "outputs": outputs,
        "sources": [
            {"path": str(path), "sha256": _sha256(path)}
            for path in ([discovery_json, discovery_npz, *native_paths]
                         + ([] if full_native_available else [native_npz]))
        ],
        "claim_boundary": (
            "This is a numerical convergence audit. It rejects the apparent "
            "0.2-ms oscillatory phase point as not reproduced at native 0.1-ms "
            "integration; it is not a paper-ready Figure 5 phase diagram."),
    }
    _atomic_json(stem.with_suffix(".metadata.json"), metadata)
    print(json.dumps(metadata["outputs"], indent=2))


if __name__ == "__main__":
    main()
