#!/usr/bin/env python3
"""Run the cheap-first empirical axial-mode audit on existing lifecycle traces."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic4_zm_empirical_modes import fit_axial_dmd  # noqa: E402


SPRINT = ROOT / "results/topic4_sef_hfo/zm_fast_lifecycle_development/lifecycle_sprint"
OUT = ROOT / "results/topic4_sef_hfo/zm_mode_lifecycle"
ANCHOR_ID = "16b8129aeae9"


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _compact(fit: dict) -> dict:
    return {key: value for key, value in fit.items() if key not in {"operator", "mean_field", "modes"}}


def main() -> None:
    phase_path = SPRINT / "batch1_phase_map.json"
    phase = json.loads(phase_path.read_text())
    rows = []
    anchor_fit = None
    anchor_field = None
    anchor_meta = None
    inputs = [{"path": str(phase_path.relative_to(ROOT)), "sha256": _sha(phase_path)}]
    for row in phase["rows"]:
        trace_path = ROOT / row["trace_path"]
        with np.load(trace_path, allow_pickle=False) as arrays:
            field = np.asarray(arrays["coarse_kymo_axial"], float)
        onset = float(row["episode"]["onset_ms"] or 0.0)
        start = max(1000.0, onset + 1000.0)
        fit = fit_axial_dmd(field, dt_ms=25.0, start_ms=start)
        compact = {
            "config_id": row["config_id"],
            "stem": row["stem"],
            "phenotype": row["phenotype"],
            "median_vseeg_gain_db": row["intensity"]["median_gain_db_across_contacts"],
            "trajectory_pc1_fraction": row["within_episode_spatial"]["common_mode_pc1_fraction"],
            "trace_path": row["trace_path"],
            "trace_sha256": _sha(trace_path),
            "dmd": _compact(fit),
        }
        rows.append(compact)
        inputs.append({"path": row["trace_path"], "sha256": compact["trace_sha256"]})
        if row["config_id"] == ANCHOR_ID:
            anchor_fit, anchor_field, anchor_meta = fit, field, compact
    if anchor_fit is None:
        raise RuntimeError(f"locked anchor {ANCHOR_ID} is absent")

    long_path = SPRINT / "seed1/i2e__tauD439__d0.8227__s1__T45s__gM10__tauM2000/traces.npz"
    with np.load(long_path, allow_pickle=False) as arrays:
        long_field = np.asarray(arrays["coarse_kymo_axial"], float)
    long_windows = {
        "dense_before_density_offset": fit_axial_dmd(long_field, dt_ms=25.0, start_ms=1500.0, end_ms=5950.0),
        "persistent_burst_tail": fit_axial_dmd(long_field, dt_ms=25.0, start_ms=6450.0, end_ms=45000.0),
    }
    inputs.append({"path": str(long_path.relative_to(ROOT)), "sha256": _sha(long_path)})

    mode = anchor_fit["pathological_mode_candidate"]
    phase_like = bool(
        mode["frequency_hz"] >= 0.5
        and mode["uniform_overlap"] < 0.70
        and mode["phase_gradient_r2"] is not None
        and mode["phase_gradient_r2"] >= 0.50
        and abs(mode["phase_gradient_rad_per_bin"] or 0.0) >= 0.05
    )
    summary = {
        "schema": "topic4_zm_empirical_axial_mode_audit_v1_2026-08-02",
        "semantic_scope": "existing_seed1_24bin_E_axial_traces_exploratory_not_full_SNN_jacobian",
        "anchor_selection": {
            "rule": "review-locked: relaxation burst with vSEEG gain about 26 dB and minimum observed PC1 fraction",
            "config_id": ANCHOR_ID,
            "stem": anchor_meta["stem"],
        },
        "n_fast_trajectories": len(rows),
        "rows": rows,
        "anchor_dmd": _compact(anchor_fit),
        "anchor_has_phase_staggered_empirical_mode": phase_like,
        "long45_windows": {name: _compact(value) for name, value in long_windows.items()},
        "inputs": inputs,
        "claim_boundary": "DMD of saved E-only axial bins; routes the next mechanism but does not establish a Hopf eigenmode",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "empirical_axial_mode_summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    ax = axes[0, 0]
    colors = {"relaxation_burst_train": "#2878b5", "spreading_plateau": "#d95f45"}
    for item in rows:
        ax.scatter(item["median_vseeg_gain_db"], item["trajectory_pc1_fraction"],
                   color=colors[item["phenotype"]], s=55, alpha=0.85)
    ax.scatter(anchor_meta["median_vseeg_gain_db"], anchor_meta["trajectory_pc1_fraction"],
               marker="*", s=260, color="#15803d", edgecolor="black", label="locked F*")
    ax.set(xlabel="median virtual-SEEG gain (dB)", ylabel="trajectory PC1 fraction",
           title="reviewed fast-state map")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    im = ax.imshow(anchor_field, aspect="auto", origin="lower", cmap="magma",
                   extent=[0, anchor_field.shape[1] * 0.025, 0, anchor_field.shape[0]])
    ax.set(xlabel="time after pre-entry checkpoint (s)", ylabel="pathological-axis bin",
           title="locked relaxation-burst anchor")
    fig.colorbar(im, ax=ax, label="spikes / coarse bin")

    path_mode = anchor_fit["pathological_mode_candidate"]
    v = np.asarray(path_mode["right_real"]) + 1j * np.asarray(path_mode["right_imag"])
    ax = axes[1, 0]
    is_complex = path_mode["phase_gradient_rad_per_bin"] is not None
    if is_complex:
        ax.plot(np.abs(v), color="#1f77b4", marker="o", label="|right mode|")
        phase_ax = ax.twinx()
        phase_ax.plot(np.unwrap(np.angle(v)), color="#d62728", marker=".", label="phase")
        phase_ax.set_ylabel("unwrapped phase (rad)", color="#d62728")
        ylabel = "mode amplitude"
        mode_kind = "complex axial mode"
    else:
        ax.axhline(0.0, color="0.75", lw=0.8)
        ax.plot(np.real(v), color="#1f77b4", marker="o", label="signed right mode")
        ylabel = "signed mode weight"
        mode_kind = "real fixed spatial mode"
    ax.set(xlabel="pathological-axis bin", ylabel=ylabel,
           title=f"{mode_kind}: {path_mode['frequency_hz']:.1f} Hz, U={path_mode['uniform_overlap']:.2f}")

    ax = axes[1, 1]
    eig = anchor_fit["modes"]
    mu = np.asarray([complex(row["mu_real"], row["mu_imag"]) for row in eig])
    theta = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(theta), np.sin(theta), "--", color="0.6", lw=1)
    ax.scatter(mu.real, mu.imag, c=[row["uniform_overlap"] for row in eig], cmap="viridis", s=45)
    chosen = complex(path_mode["mu_real"], path_mode["mu_imag"])
    ax.scatter([chosen.real], [chosen.imag], marker="*", s=240, color="#15803d", edgecolor="black")
    ax.axhline(0, color="0.8", lw=0.8); ax.axvline(0, color="0.8", lw=0.8)
    ax.set_aspect("equal", adjustable="box")
    ax.set(xlabel="Re(mu)", ylabel="Im(mu)", title="empirical one-step DMD spectrum")

    fig_dir = OUT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "empirical_axial_mode_audit.png", dpi=180)
    plt.close(fig)
    (fig_dir / "README.md").write_text(
        "### empirical_axial_mode_audit.png\n\n"
        "从已有 16 条快抑制轨迹中锁定高能量且 PC1 最低的 relaxation-burst anchor，"
        "并对其 24 个病理轴 bin 拟合经验 DMD 传播算子。星号只表示下一阶段的机制输入，"
        "不是已经发现 Hopf 或 ictal eigenmode。\n\n"
        "**关注点**：当前被选中的方向是 0 Hz 实模态；左下角因此画有符号的空间权重而不制造 0/π 相位梯度。"
        "右下角若没有接近单位圆的非实模态，就只能解释为固定空间图样的振幅开关。\n"
    )
    print(json.dumps({
        "anchor": ANCHOR_ID,
        "phase_staggered": phase_like,
        "frequency_hz": mode["frequency_hz"],
        "uniform_overlap": mode["uniform_overlap"],
        "phase_gradient": mode["phase_gradient_rad_per_bin"],
        "phase_r2": mode["phase_gradient_r2"],
        "heldout_error": anchor_fit["heldout_relative_error"],
        "out": str(OUT / "empirical_axial_mode_summary.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
