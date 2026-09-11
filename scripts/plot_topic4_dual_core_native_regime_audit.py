#!/usr/bin/env python3
"""Plot the native-step narrow-regime and basin audit for dual-core spatial Z."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic4_dual_core_oscillation_phase import population_cycle_modulation  # noqa: E402


DEFAULT_ROOT = Path(
    "/data/hfosp_topic4_fig45_artifacts/fig5/"
    "data_driven_dual_core_spatial_z/oscillatory_phase_map")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(descriptor)
    try:
        Path(temporary).write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _edges(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, float)
    middle = 0.5 * (values[:-1] + values[1:])
    return np.r_[values[0] - (middle[0] - values[0]), middle,
                 values[-1] + (values[-1] - middle[-1])]


def _label(axis: plt.Axes, label: str) -> None:
    axis.text(-0.15, 1.08, label, transform=axis.transAxes, fontsize=15,
              fontweight="bold", va="top")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    paths = {
        "native": root / "native_refined_s_tau_dt0p1",
        "zoom": root / "native_zoom_s0p27_t9_dt0p1",
        "high": root / "native_long_s0p27_t9_5s",
        "low": root / "native_long_low_s0p27_t9_5s",
    }
    for stem in paths.values():
        if not stem.with_suffix(".json").exists() or not stem.with_suffix(".npz").exists():
            raise RuntimeError(f"required audit artifact is missing: {stem}")
    native = np.load(paths["native"].with_suffix(".npz"))
    zoom = np.load(paths["zoom"].with_suffix(".npz"))
    high = np.load(paths["high"].with_suffix(".npz"))
    low = np.load(paths["low"].with_suffix(".npz"))
    native_meta = json.loads(paths["native"].with_suffix(".json").read_text())
    zoom_meta = json.loads(paths["zoom"].with_suffix(".json").read_text())
    high_meta = json.loads(paths["high"].with_suffix(".json").read_text())
    low_meta = json.loads(paths["low"].with_suffix(".json").read_text())

    s = np.asarray(native["core_disinhibition_s"], float)
    tau = np.asarray(native["tau_d_GABA_ms"], float)
    code = np.asarray(native["high_initial__state_code"], int)
    zs = np.asarray(zoom["core_disinhibition_s"], float)
    zt = np.asarray(zoom["tau_d_GABA_ms"], float)
    zcode = np.asarray(zoom["high_initial__state_code"], int)
    zdepth = np.asarray(zoom["high_initial__modulation_depth"], float)

    fig = plt.figure(figsize=(11.5, 7.1))
    grid = fig.add_gridspec(2, 2, left=0.075, right=0.96, bottom=0.09,
                            top=0.90, wspace=0.30, hspace=0.40)
    ax_a, ax_b = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])
    ax_c, ax_d = fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])
    fig.suptitle(
        "Native-step dual-core regime audit: a narrow oscillatory ridge",
        fontsize=13, fontweight="bold", y=0.965)

    colors = ["#E7E9ED", "#78A6C8", "#D96B5C", "#76519B"]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    state = ax_a.pcolormesh(_edges(s), _edges(tau), code.T,
                            cmap=cmap, norm=norm, shading="flat")
    bar = fig.colorbar(state, ax=ax_a, ticks=[0, 1, 2, 3], fraction=0.05)
    bar.ax.set_yticklabels(["low", "localized", "tonic", "oscillatory"],
                           fontsize=7)
    ax_a.set_title("0.1-ms high-basin outcome, 2 s", fontweight="bold", fontsize=10)
    ax_a.set_xlabel(r"core disinhibition $s$ ($Z_{A,B}=1-s$)")
    ax_a.set_ylabel(r"GABA decay $\tau_{\mathrm{GABA}}$ (ms)")
    _label(ax_a, "A")

    depth = ax_b.pcolormesh(_edges(zs), _edges(zt), zdepth.T, cmap="magma",
                            vmin=0.12, vmax=0.27, shading="flat")
    for row, ss in enumerate(zs):
        for column, tt in enumerate(zt):
            marker = "*" if zcode[row, column] == 3 else (
                "o" if zcode[row, column] == 2 else "x")
            kwargs = ({"facecolor": "white", "edgecolor": "black"}
                      if marker != "x" else {"color": "black"})
            ax_b.scatter(ss, tt, marker=marker,
                         s=55 if marker == "*" else 25, lw=0.8, **kwargs)
    fig.colorbar(depth, ax=ax_b, fraction=0.05, label="cycle modulation depth")
    ax_b.set_title("Native zoom (× localized; ○ tonic; ★ all gates)",
                   fontweight="bold", fontsize=10)
    ax_b.set_xlabel("core disinhibition $s$")
    ax_b.set_ylabel(r"GABA decay $\tau_{\mathrm{GABA}}$ (ms)")
    _label(ax_b, "B")

    high_time = np.asarray(high["time_ms"], float) / 1000.0
    high_trace = np.asarray(high["high_initial__mean_e_trace_hz"][0, 0], float)
    ax_c.plot(high_time, high_trace, color="#76519B", lw=0.7)
    ax_c.axvspan(1.0, 5.0, color="#76519B", alpha=0.06, lw=0)
    ax_c.set_xlim(0, 5)
    ax_c.set_xlabel("time (s)")
    ax_c.set_ylabel("population E rate (Hz)")
    ax_c.set_title(r"5-s persistence at $s=0.27$, $\tau_{\mathrm{GABA}}=9$ ms",
                   fontweight="bold", fontsize=10)
    _label(ax_c, "C")

    low_time = np.asarray(low["time_ms"], float) / 1000.0
    low_trace = np.asarray(low["low_initial__mean_e_trace_hz"][0, 0], float)
    ax_d.plot(high_time, np.maximum(high_trace, 1e-3), color="#76519B", lw=0.8,
              label="high initial → oscillatory recruited")
    ax_d.plot(low_time, np.maximum(low_trace, 1e-3), color="#245B9B", lw=0.9,
              label="low initial → low")
    ax_d.set_yscale("log")
    ax_d.set_xlim(0, 5)
    ax_d.set_xlabel("time (s)")
    ax_d.set_ylabel("population E rate (Hz, log)")
    ax_d.set_title("Same parameters, different basins", fontweight="bold", fontsize=10)
    ax_d.legend(frameon=False, fontsize=7, loc="center right")
    _label(ax_d, "D")

    for axis in (ax_a, ax_b, ax_c, ax_d):
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=8)
    out = root / "figures"
    out.mkdir(parents=True, exist_ok=True)
    stem = out / "dualcore-native-regime-and-basin-audit"
    outputs = {}
    for suffix, kwargs in (("png", {"dpi": 300}), ("pdf", {}), ("svg", {})):
        path = stem.with_suffix("." + suffix)
        fig.savefig(path, bbox_inches="tight", facecolor="white", **kwargs)
        outputs[suffix] = {"path": str(path), "sha256": _sha256(path)}
    plt.close(fig)

    one_second_metrics = []
    for lo in range(1000, 5000, 1000):
        selected = ((high_time * 1000.0 >= lo)
                    & (high_time * 1000.0 < lo + 1000))
        measured = population_cycle_modulation(high_trace[selected], dt_ms=0.1)
        one_second_metrics.append({
            "window_ms": [lo, lo + 1000],
            "mean_rate_hz": measured["mean_rate_hz"],
            "dominant_hz": measured["dominant_hz"],
            "modulation_depth": measured["modulation_depth"],
        })
    metadata = {
        "status": "NATIVE_NARROW_OSCILLATORY_RIDGE_WITH_BASIN_COEXISTENCE",
        "native_grid_counts": native_meta["counts"],
        "native_zoom_counts": zoom_meta["counts"],
        "high_initial_long_state": high_meta["records"][0]["state"],
        "low_initial_long_state": low_meta["records"][0]["state"],
        "high_initial_one_second_metrics": one_second_metrics,
        "outputs": outputs,
        "sources": [
            {"path": str(stem_path.with_suffix(suffix)),
             "sha256": _sha256(stem_path.with_suffix(suffix))}
            for stem_path in paths.values() for suffix in (".json", ".npz")
        ],
        "claim_boundary": (
            "The native-step deterministic 2-mm coarse system has a sustained "
            "oscillatory-recruited high-basin candidate on a narrow recruitment "
            "ridge and a coexisting low basin. Two of 15 zoom cells pass; this "
            "is not yet a robust finite-area multi-topology phase or full-SNN result."),
    }
    _atomic(stem.with_suffix(".metadata.json"),
            json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n")
    readme = out / "README.md"
    existing = readme.read_text(encoding="utf-8") if readme.exists() else ""
    marker = "### dualcore-native-regime-and-basin-audit.png"
    if marker not in existing:
        existing += (
            "\n" + marker + "\n\n"
            "这张图把原生 0.1-ms high-basin 网格、候选周围的细网格、5-s 持续轨迹和同参数 low/high 初值对照放在一起。通过点沿 localized 到 tonic 的窄 ridge 分布；同一参数下低初值保持低态，高初值进入持续 30-Hz recruited rhythm。\n\n"
            "**关注点**：这是 deterministic 2-mm coarse system 的窄振荡窗与 basin 共存证据，不是有限面积、多 topology 或完整 SNN 的正式 phase diagram。\n")
        _atomic(readme, existing.lstrip())
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
