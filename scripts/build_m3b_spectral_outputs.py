#!/usr/bin/env python
"""Generate the M3B-R2 spectral phase-map artifacts + figures (TDD-15).

Runs the frozen spectral pipeline (dispersion -> phase map -> controls -> rate-field spot check ->
readout projection), writes the declared JSON/CSV artifacts and the key figures, and records the
M3A overlay audit (refused — M3A artifacts absent). The SNN spot-check summary is written from one
real tiny pilot.

Honest verdict: the leading-eigenvalue map is stable->global->runaway, NOT axial-leading. See STATUS.md.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import src.topic4_m3b_spectral_phase as spm
from src.sef_hfo_m3_interface import audit_m3a_interface

OUT = Path("results/topic4_sef_hfo/m3b_spectral_phase_map")
FIG = OUT / "figures"


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    grid = spm.Grid(n=8, L=5.0)
    kernels = spm.build_kernels(grid, ell_perp=0.6)
    core = spm.make_core_mask(grid, kind="single", radius=0.9)
    gE, gI = spm.homogeneous_gains(1.0, 1.5)

    # 1. homogeneous dispersion -----------------------------------------------------------------
    disp = spm.homogeneous_dispersion(gE, gI, w_ee_mult=1.5, nk=41, kmax=3.0)
    (OUT / "homogeneous_dispersion.json").write_text(
        json.dumps(spm.dispersion_to_json(disp), indent=1), encoding="utf-8")

    # 2. phase map ------------------------------------------------------------------------------
    x_vals = [0.0, 0.4, 0.8, 1.2]                       # core excitability (mu_core)
    y_vals = [1.0, 0.97, 0.94, 0.9]                     # global disinhibition (q_global, lower=more)
    points = spm.build_phase_map(grid, kernels, core, x_values=x_vals, y_values=y_vals, w_ee_mult=1.3)
    rows = spm.phase_map_to_rows(points)
    with (OUT / "mode_metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(spm.MODE_METRICS_COLUMNS))
        w.writeheader()
        w.writerows(rows)
    grid_meta = {
        "axes_built_from_slow_to_rate_mapping_id": None,    # M3A mapping absent (overlay refused)
        "axis_space": "raw_knob_unit",                      # frozen map uses raw knobs (no M3A norm)
        "x_axis": "mu_core (core excitability, mV)",
        "y_axis": "q_global (GABA efficacy; lower = disinhibition)",
        "x_values": x_vals, "y_values": y_vals,
        "grid_n": grid.n, "grid_L": grid.L, "w_ee_mult": 1.3,
        "unresolved_fraction": spm.unresolved_fraction(points),
        "mode_class_counts": {c: sum(1 for p in points if p.mode_class == c)
                              for c in sorted({p.mode_class for p in points})},
        "points": rows,
    }
    (OUT / "finite_jacobian_grid.json").write_text(json.dumps(grid_meta, indent=1), encoding="utf-8")

    # 3. controls -------------------------------------------------------------------------------
    controls = spm.run_controls(grid, mu_core=0.8, w_ee_mult=1.3)
    (OUT / "control_summary.json").write_text(json.dumps(controls, indent=1), encoding="utf-8")

    # 4. rate-field spot check ------------------------------------------------------------------
    g24 = spm.Grid(n=24, L=10.0)
    k24_ar2 = spm.build_kernels(g24, ell_perp=0.7)
    k24_ar1 = spm.build_kernels(g24, ar=1.0, ell_perp=0.7)
    c24 = spm.make_core_mask(g24, kind="single", radius=1.2)
    inh0 = spm.build_inhibition_field(g24, c24)
    e0 = spm.build_excitability_field(g24, c24, mu_core=0.0)
    ar2 = spm.simulate_ratefield_response(g24, k24_ar2, e0, inh0, w_ee_mult=1.0, stim_amp=8.0)
    ar1 = spm.simulate_ratefield_response(g24, k24_ar1, e0, inh0, w_ee_mult=1.0, stim_amp=8.0)
    e_hi = spm.build_excitability_field(g24, c24, mu_core=0.7)
    runaway = spm.simulate_ratefield_response(g24, k24_ar2, e_hi, inh0, w_ee_mult=1.4, stim_amp=8.0)
    rf_summary = {
        "axial_ar2": {"response_axis_score": ar2.response_axis_score, "max_active": ar2.max_active,
                      "returned": ar2.returned},
        "isotropic_ar1": {"response_axis_score": ar1.response_axis_score, "max_active": ar1.max_active,
                          "returned": ar1.returned},
        "runaway": {"max_active": runaway.max_active, "returned": runaway.returned},
        "note": "AR2 kick response is axial (resp_axis>AR1); runaway does not return. Sharp "
                "resting->runaway transition (substrate bottleneck), little self-limited window.",
    }
    (OUT / "ratefield_spotcheck_summary.json").write_text(json.dumps(rf_summary, indent=1),
                                                          encoding="utf-8")

    # 5. SNN spot check (one real tiny pilot) ---------------------------------------------------
    try:
        snn_rec = spm.run_snn_spotcheck(mu_core=0.3, q_global=1.0, w_ee_mult=1.0)
        snn_summary = {"pilots": [snn_rec],
                       "note": "single tiny frozen-state pilot; full grid deferred. Substrate gives "
                               "global recruitment (R4a) — confirms the bottleneck."}
    except Exception as exc:                                       # keep the build robust
        snn_summary = {"pilots": [], "note": f"SNN pilot skipped: {exc!r}"}
    (OUT / "snn_spotcheck_summary.json").write_text(json.dumps(snn_summary, indent=1), encoding="utf-8")

    # 6. mode -> readout projection -------------------------------------------------------------
    # project the most core-localized resolved phase-map point's leading mode
    op = spm.solve_operating_point(grid, kernels,
                                   spm.build_excitability_field(grid, core, mu_core=0.8),
                                   spm.build_inhibition_field(grid, core), ratio=1.0, w_ee_mult=1.3)
    res = spm.rate_eigenpairs(spm.build_jacobian_dense(grid, kernels, op), grid, n_modes=2)
    eE = np.abs(spm.mode_e_field(res.right[:, 0], grid))
    rec = spm.project_mode_to_record(eE, grid, model_id="m3b_leading_mode")
    readout = {
        "model_record_n_channels": rec.get("n_channels"),
        "model_record_scalars": rec.get("scalars"),
        "geometry_null_beaten": None,        # requires the real cohort; not run here
        "verdict": spm.readout_bridge_verdict(float("nan"), geometry_null_beaten=False),
        "note": "mode projected through the Round-1 virtual-SEEG readout; cohort placement + geometry "
                "null require the real cohort (placement-only until then).",
    }
    (OUT / "mode_readout_projection.json").write_text(json.dumps(readout, indent=1), encoding="utf-8")

    # 7. M3A overlay audit (refused — M3A artifacts absent) -------------------------------------
    audit = audit_m3a_interface(mapping=None, ranges=None, trajectory_rows=None, summary=None,
                                axes_meta=None)
    (OUT / "m3a_interface_audit.json").write_text(json.dumps(audit, indent=1), encoding="utf-8")

    # ---- figures ------------------------------------------------------------------------------
    _fig_dispersion(disp)
    _fig_phase_map(points, x_vals, y_vals)
    _fig_controls(controls)
    _fig_readout(rec)
    _write_readme()
    print(f"wrote artifacts + figures to {OUT}")


def _fig_dispersion(disp):
    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.plot(disp["k"], disp["lambda_re"], "-o", ms=3, color="#1f77b4")
    ax.axhline(0, color="0.6", lw=0.8, ls="--")
    ax.set_xlabel("wavenumber k along E→E axis (1/mm)")
    ax.set_ylabel("leading rate-branch growth  Re λ")
    ax.set_title(f"Homogeneous dispersion (regime: {disp['regime']}; k*={disp['k_star']:.2f})")
    fig.tight_layout()
    fig.savefig(FIG / "homogeneous_dispersion.png", dpi=130)
    plt.close(fig)


def _fig_phase_map(points, x_vals, y_vals):
    classes = ["stable", "local", "axial", "mixed", "global", "runaway", "unresolved"]
    cidx = {c: i for i, c in enumerate(classes)}
    grid_cls = np.array([cidx[p.mode_class] for p in points]).reshape(len(y_vals), len(x_vals))
    fig, ax = plt.subplots(figsize=(5.2, 4))
    # y_vals run high->low (1.0 -> 0.9); row 0 = first y_val must sit at the TOP -> origin="upper"
    im = ax.imshow(grid_cls, origin="upper", aspect="auto", cmap="viridis",
                   vmin=0, vmax=len(classes) - 1,
                   extent=[min(x_vals), max(x_vals), min(y_vals), max(y_vals)])
    ax.set_xlabel("core excitability  μ_core (mV)")
    ax.set_ylabel("global GABA efficacy  q_global  (lower = disinhibition)")
    ax.set_title("Leading-mode class phase map")
    cb = fig.colorbar(im, ax=ax, ticks=range(len(classes)))
    cb.ax.set_yticklabels(classes)
    fig.tight_layout()
    fig.savefig(FIG / "phase_map_mode_class.png", dpi=130)
    plt.close(fig)


def _fig_controls(controls):
    labels = ["core", "no_core"]
    vals = [controls["core"]["core_localization"], controls["no_core"]["core_localization"]]
    labels += ["AR2", "AR1"]
    vals += [controls["ar2_anisotropic"]["dispersion_anisotropy"],
             controls["ar1_isotropic"]["dispersion_anisotropy"]]
    labels += ["contig", "shuffled"]
    vals += [controls["shuffled_core"]["contiguous_core_localization"],
             controls["shuffled_core"]["core_localization_mean"]]
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    colors = ["#2ca02c", "#aaaaaa", "#2ca02c", "#aaaaaa", "#2ca02c", "#aaaaaa"]
    ax.bar(range(len(vals)), vals, color=colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20)
    ax.set_ylabel("control observable")
    ax.set_title("Core/scaffold specificity controls\n(green = with structure, grey = ablation)")
    fig.tight_layout()
    fig.savefig(FIG / "controls_summary.png", dpi=130)
    plt.close(fig)


def _fig_readout(rec):
    chans = rec.get("channels", [])
    if not chans:
        return
    xs = [c["x_norm"] for c in chans]
    ys = [c["y_norm"] for c in chans]
    rk = [c["typical_rank"] for c in chans]
    fig, ax = plt.subplots(figsize=(5, 3.6))
    sc = ax.scatter(xs, ys, c=rk, cmap="viridis", s=60, edgecolor="k", lw=0.4)
    ax.axhline(0, color="0.7", lw=0.6)
    ax.set_xlabel("along-axis (normalized)")
    ax.set_ylabel("transverse (normalized)")
    ax.set_title("Leading mode → virtual-SEEG readout")
    fig.colorbar(sc, ax=ax, label="typical rank (early→late)")
    fig.tight_layout()
    fig.savefig(FIG / "mode_readout_projection.png", dpi=130)
    plt.close(fig)


def _write_readme():
    (FIG / "README.md").write_text(
        """# M3B-R2 谱相图 figures 说明

### homogeneous_dispersion.png
均质衬底沿 E→E 轴的色散曲线：横轴是空间花样的波数 k，纵轴是该花样的增长率 Re λ。
关键观察是增长率在 **k=0（整片同步）最大**，没有出现"某个有限波长的行波最先长大"的
Brunel/Turing 式有限-k 失稳——这是均质衬底"想全场一起点火"的体现。
**关注点**：曲线在 k=0 处最高、随 k 单调下降，没有有限-k 峰。

### phase_map_mode_class.png
核兴奋度（横，μ_core）× 全局去抑制（纵，q_global 越小越去抑制）二维相图，每格颜色=最先
长大的那个花样的类别。整张图基本只有 stable / global / runaway 三类，**没有 axial 档**——
最先失稳的永远是整片同步花样，不是沿轴行波。
**关注点**：有没有出现 axial（轴向）格子；本图诚实结果是没有，过渡是 stable→global→runaway。

### controls_summary.png
对照消融：绿柱=带结构（核/各向异性/连续核），灰柱=去掉结构（无核/各向同性/打散核）。
核把模式能量聚到核里（绿>灰）、各向异性核才有 45° 方向偏好（灰≈0）、连续核比打散核聚得更紧。
**关注点**：每对里绿柱是否明显高于灰柱（证明结构是核/骨架特异的，不是任意扰动都有）。

### mode_readout_projection.png
把谱里那个最核局域的本征模式，经虚拟 SEEG 触点读回来：点=触点，位置=沿轴/横向归一化坐标，
颜色=典型先后排名。这一步只验证"模型模式能干净地走完 Round-1 读出管线"，跟真实队列的对齐
+ 几何零模型检验需要真实数据，未做，所以读出结论停在 placement-only。
**关注点**：点是否沿一条轴排开、颜色（排名）沿轴是否有梯度。

### N/A（本轮未生成，原因如下）
- `example_modes.png` — 需要挑代表性 local/axial/mixed/global 本征模式做四联图；因为没有 axial-leading 模式，留待方向确定后生成。
- `phase_map_gap_gain.png` — spectral-gap + finite-time-gain 等值线图，待主相图方向确定后补。
- `non_normal_gain_controllability.png` — 非正规瞬态增益 × 核可控性图（轴向信号真正所在层），待按 §5 重做读法后生成。
- `snn_spotcheck_grid.png` — 需要成组 SNN 抽查（每点 ~3s × 多点）；本轮只跑了 1 个 pilot（见 snn_spotcheck_summary.json）。
- `slow_trajectory_overlay.png` — M3A 轨迹叠加；M3A 五个交接产物全不存在、overlay 判定为 refused，故 N/A。
""", encoding="utf-8")


if __name__ == "__main__":
    main()
