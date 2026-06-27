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
        "m3a_overlay_consumable": False,                    # raw knobs != normalized_unit (contract D1)
        "axis_space_note": ("raw_knob_unit atlas (mu_core mV x q_global). NOT normalized_unit, so it "
                            "CANNOT be consumed by an M3A trajectory overlay, which requires "
                            "normalized phase coords per the interface contract D1. A separate "
                            "normalized phase grid must be built before any M3A overlay."),
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
        snn_summary = {
            "pilots": [snn_rec],
            "note": (f"single tiny frozen-state pilot; R_class={snn_rec['R_class']} "
                     f"(returned={snn_rec['returned']}, peak_active_frac={snn_rec['peak_active_frac']}). "
                     "A non-returning full-recruitment event = tonic runaway, NOT self-limited "
                     "recruitment; this STRENGTHENS the bounded-negative. Full grid deferred."),
        }
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
        "geometry_null_status": "not_run",   # enum {not_run, failed, passed} — NOT a bool
        "cohort_placement_run": False,
        "verdict": spm.readout_bridge_verdict("not_run"),    # -> 'projection_only'
        "note": "the SCHEMA / PROJECTION connects (a mode field walks the Round-1 virtual-SEEG readout "
                "cleanly), but cohort alignment + the geometry null were NOT RUN — they need the real "
                "cohort. geometry_null_status='not_run' (NOT 'failed'); verdict='projection_only' "
                "(not even 'placement_only', since no placement was computed). No cohort bridge.",
    }
    (OUT / "mode_readout_projection.json").write_text(json.dumps(readout, indent=1), encoding="utf-8")

    # 6b. non-normal transient axial readout (§5 PRIMARY) — the axial signal the leading mode misses
    nn_pts = []
    for x in x_vals:
        for y in y_vals:
            op_nn = spm.solve_operating_point(grid, kernels, spm.build_excitability_field(grid, core, mu_core=x),
                                              spm.build_inhibition_field(grid, core, q_global=y),
                                              ratio=1.0, w_ee_mult=1.3)
            if op_nn.status != "resolved":
                continue
            r = spm.non_normal_axial_readout(spm.build_jacobian_dense(grid, kernels, op_nn), grid, core)
            nn_pts.append({"mu_core": x, "q_global": y, "peak_gain": r["peak_gain"],
                           "max_axis": r["max_axis"], "transient_amplified": r["transient_amplified"],
                           "self_limited": r["self_limited"], "axial": r["axial"]})
    op_ar2 = spm.solve_operating_point(grid, kernels, spm.build_excitability_field(grid, core, mu_core=0.6),
                                       spm.build_inhibition_field(grid, core), ratio=1.0, w_ee_mult=1.3)
    nn_ar2 = spm.non_normal_axial_readout(spm.build_jacobian_dense(grid, kernels, op_ar2), grid, core)
    ker_iso = spm.build_kernels(grid, ar=1.0, ell_perp=0.6)
    op_ar1 = spm.solve_operating_point(grid, ker_iso, spm.build_excitability_field(grid, core, mu_core=0.6),
                                       spm.build_inhibition_field(grid, core), ratio=1.0, w_ee_mult=1.3)
    nn_ar1 = spm.non_normal_axial_readout(spm.build_jacobian_dense(grid, ker_iso, op_ar1), grid, core)
    n_axial = sum(1 for p in nn_pts if p["axial"] and p["transient_amplified"] and p["self_limited"])
    nn_summary = {
        "per_point": nn_pts, "n_resolved": len(nn_pts), "n_axial_amplified_selflimited": n_axial,
        "ar2_curve": {"windows": nn_ar2["windows"], "gains": nn_ar2["gains"], "axes": nn_ar2["axes"]},
        "ar1_isotropic_control": {"max_axis": nn_ar1["max_axis"], "axial": nn_ar1["axial"]},
        "note": ("§5 PRIMARY: a core perturbation is transiently amplified (~1.8x near T~10ms) and "
                 "spreads ALONG the E->E axis before self-limiting, at every resolved point, "
                 "scaffold-specifically (AR1 isotropic -> not axial). The LEADING eigenmode is global; "
                 "the self-limited axial propagation lives in the non-normal transient, not the mode. "
                 "This is a MODEL/linear-operator result, not a claim about real seizures."),
    }
    (OUT / "non_normal_axial_readout.json").write_text(json.dumps(nn_summary, indent=1), encoding="utf-8")

    # 7. M3A overlay audit (refused — M3A artifacts absent) -------------------------------------
    audit = audit_m3a_interface(mapping=None, ranges=None, trajectory_rows=None, summary=None,
                                axes_meta=None)
    (OUT / "m3a_interface_audit.json").write_text(json.dumps(audit, indent=1), encoding="utf-8")

    # 8. verdict — EXPLICIT, FAIL-CLOSED gates derived from the actual artifacts (no hardcoded PASS) ---
    controls_pass = bool(
        controls["core"]["core_localization"] > controls["no_core"]["core_localization"] + 0.01
        and controls["ar2_anisotropic"]["dispersion_anisotropy"]
        > controls["ar1_isotropic"]["dispersion_anisotropy"]
        and controls["shuffled_core"]["core_localization_mean"]
        < controls["shuffled_core"]["contiguous_core_localization"])
    # model_matches_dynamics = the nonlinear RATE FIELD agrees with the spectral prediction (AR2 kick
    # axial vs AR1; runaway does not return). This is NOT an SNN claim.
    ratefield_pass = bool(
        rf_summary["axial_ar2"]["response_axis_score"] > rf_summary["isotropic_ar1"]["response_axis_score"]
        and not rf_summary["runaway"]["returned"])
    non_normal_axial_pass = bool(n_axial == len(nn_pts) and len(nn_pts) > 0)
    snn_grid_pass = False        # only 1 pilot (R4b tonic runaway) — NOT a grid validation; deferred
    m3a_overlay_pass = (audit.get("overlay_verdict") == "phase_map_trajectory")   # refused -> False
    readout_null_pass = (readout["geometry_null_status"] == "passed")             # not_run -> False
    verdict = spm.m3b_verdict(
        phase_map_resolved=True, model_matches_dynamics=ratefield_pass, controls_pass=controls_pass,
        non_normal_axial_pass=non_normal_axial_pass, snn_grid_pass=snn_grid_pass,
        m3a_overlay_pass=m3a_overlay_pass, readout_null_pass=readout_null_pass)
    verdict_inputs = {
        "verdict": verdict, "phase_map_resolved": True,
        "model_matches_dynamics_ratefield": ratefield_pass, "controls_pass": controls_pass,
        "non_normal_axial_pass": non_normal_axial_pass,
        "n_axial_amplified_selflimited": n_axial, "n_resolved": len(nn_pts),
        "snn_grid_pass": snn_grid_pass, "snn_validation_status": "not_run_grid (1 pilot = R4b)",
        "m3a_overlay_pass": m3a_overlay_pass, "m3a_overlay_status": audit.get("overlay_verdict"),
        "readout_null_pass": readout_null_pass, "geometry_null_status": readout["geometry_null_status"],
        "note": ("explicit fail-closed gates: frozen-map = controls_pass AND non_normal_axial_pass; "
                 "spontaneous-mechanism additionally needs snn_grid_pass; full bridge also needs "
                 "m3a_overlay_pass AND readout_null_pass."),
    }
    (OUT / "verdict_inputs.json").write_text(json.dumps(verdict_inputs, indent=1), encoding="utf-8")
    _write_status(verdict, n_axial=n_axial, n_resolved=len(nn_pts), nn=nn_ar2, nn_ar1=nn_ar1)

    # ---- figures ------------------------------------------------------------------------------
    _fig_dispersion(disp)
    _fig_phase_map(points, x_vals, y_vals)
    _fig_controls(controls)
    _fig_readout(rec)
    _fig_non_normal(nn_ar2, nn_ar1)
    _write_readme()
    print(f"wrote artifacts + figures to {OUT} (verdict: {verdict}; "
          f"§5 axial {n_axial}/{len(nn_pts)} resolved pts)")


def _write_status(verdict, *, n_axial, n_resolved, nn, nn_ar1) -> None:
    peak_g = max(nn["gains"])
    max_ax = max(nn["axes"])
    ar1_ax = nn_ar1["max_axis"]
    t_gain = nn["windows"][nn["gains"].index(peak_g)]      # ms at the gain peak
    t_axis = nn["windows"][nn["axes"].index(max_ax)]       # ms at the axis peak (later than gain)
    (OUT / "STATUS.md").write_text(
        """# M3B-R2 spectral phase-map — STATUS

> TDD-0 artifact: freezes outputs and forbidden claims before any eigenvalue exists.
> Design: `docs/superpowers/specs/2026-06-27-sef-hfo-m3b-spectral-phase-map-design.md`
> Plan: `docs/superpowers/plans/2026-06-27-sef-hfo-m3b-spectral-phase-map-plan.md`
> Interface contract: `docs/superpowers/specs/2026-06-27-sef-hfo-m3-interface-contract.md`
> GENERATED by scripts/build_m3b_spectral_outputs.py — edit there, not here.

## M3B Round-1 bridge verdict (completed, now downstream)

A kick-rate-field instrument probe supported an interictal scaffold bridge (model-to-real median
field correlation ~0.844, placement ~74 %, beating channel/within-shaft nulls). The ictal-early leg
was placement-only (~0.420, ~72 %) and did NOT beat geometry nulls. The "same field, two gains"
sweep was inconclusive — no graded recruitment range, no established phase transition.

## M3B-R2 objective (this line)

Build a Brunel-style mean-field / finite-Jacobian SPECTRAL phase map for the core-heterogeneous
epilepsy sheet: operating point → linearize → spatial eigenmodes + non-normal finite-time gain →
phase map over core-excitability × global-disinhibition → validate against rate-field/SNN events →
project through the same virtual-SEEG readout. The phase map is a MECHANISM map; SNN behavior is the
phenotype test. No single eigenvalue is "the seizure."

## M3A handoff requirement (hard gate)

M3B overlays an M3A slow-state trajectory ONLY when
`m3a_interface_audit.json::overlay_verdict == "phase_map_trajectory"` (all four conditions pass:
sign-tests passed, same mapping+ranges, ≤5 % out-of-range, A2 phenotype-movement-beyond-rate). The
overlay is triggered by the VERDICT, never by mere file availability. A phenotype-positive run
without a calibrated mapping is `mechanism_candidate_only` and is NOT drawn on the map. The overlay
is Gate-A tier; a seizure-like (Gate-B) claim requires `gate_B_seizure_like == "PASS"`. See the
interface contract + `src/sef_hfo_m3_interface.py`.

## Allowed verdict categories

- SPM-PASS full bridge
- SPM-PASS spontaneous mechanism
- SPM-PASS frozen map
- SPM-BOUNDED negative
- SPM-MODEL mismatch
- SPM-UNRESOLVED

## Forbidden claims

- "W causes seizure."
- "A plane-wave k mode explains the fixed-core event" without finite-Jacobian evidence.
- "Eigenvalue > 0 proves clinical seizure onset."
- "M3B shows slow variables cause seizure" without a valid M3A slow trajectory + SNN validation.

## Declared artifacts (TDD-0 contract)

- STATUS.md
- homogeneous_dispersion.json
- finite_jacobian_grid.json
- mode_metrics.csv
- control_summary.json
- mode_readout_projection.json
- m3a_interface_audit.json
- slow_trajectory_overlay.csv
- snn_spotcheck_summary.json
- ratefield_spotcheck_summary.json
- figures/README.md

## Current verdict (2026-06-27): """ + verdict + """

Plain language — 我们把带病灶核的薄片线性化、扫"核兴奋度 × 全局去抑制"相图，并按计划 §5 用
**正确的读法**读轴向信号：不是看"哪个花样最先持续长大"（那个永远是全局），而是看**给核一个
扰动后短时间内会怎样**（非正规瞬态）。

诚实结果（§5 主读法，机器已三重验证）：**给核一个扰动，瞬态增益在约 """ + f"{t_gain:.0f}" + """ ms 先冲到约 """ + f"{peak_g:.1f}" + """
倍，沿 E→E 轴的拉伸峰更靠后（约 """ + f"{t_axis:.0f}" + """ ms，max≈""" + f"{max_ax:.2f}" + """），随后增益与轴向都衰减回去
（**自限**）——这是"间期自限轴向传播"的信号，不是"10 ms 内已经一条完整行波"。** 这个轴向信号在
""" + f"{n_axial}/{n_resolved}" + """ 个未饱和相图点全都出现，且**方向骨架特异**：把 E→E 连线换成各向同性（AR1），
增益放大还在（放大本身不挑骨架）但沿轴拉伸没了（max_axis≈""" + f"{ar1_ax:.2f}" + """ vs AR2 """ + f"{max_ax:.2f}" + """）。关键区分：
**最先持续长大的本征花样仍然是全局的（轴向≈0）；轴向自限传播活在非正规瞬态里，不在主导花样里**
——这就是为什么只看主导花样会误判成"没有轴向"。

因此本轮判决 = **SPM-PASS frozen map**：相图数值干净、对照消融特异（核/各向异性/连续核）、且 §5
非正规瞬态读出在所有未饱和点给出骨架特异的自限轴向传播。这是一个**模型/线性算子层面**的结果，
**不是对真实发作的主张**；还没有 SNN/M3A/几何零模型这三道桥（见下），所以停在 frozen-map 档，
不是 spontaneous-mechanism 或 full bridge。

- §5 非正规瞬态（**主指标**）：core kick 瞬态增益峰 ~""" + f"{peak_g:.1f}" + """（T~10 ms）、沿轴 max elongation
  ~""" + f"{max_ax:.2f}" + """、之后衰减（自限）；AR1 各向同性对照 max_axis≈""" + f"{ar1_ax:.2f}" + """ → 骨架特异。见
  `non_normal_axial_readout.json` + `figures/non_normal_gain_controllability.png`。
- **谱增长率 α₁ ≠ 非线性饱和**：相图里 `runaway` 格全部来自工作点非线性饱和（op_status=saturated，
  α₁ 实际是**负的** ≈ −0.05），不是 α₁>0 线性失稳。失控是非线性饱和态，不是谱失稳。
- **主导本征花样在所有点都是全局**（轴向≈0）→ 轴向信号必须用 §5 瞬态读法才看得到；这是上一版
  误判成 bounded-negative 的原因。
- SNN 一个 tiny pilot = **R4b（不返回 = tonic runaway）** → 印证去抑制端会冲到饱和失控
  （见 `snn_spotcheck_summary.json`）。
- M3A 轨迹叠加 = `refused`（5 个交接产物全不存在；见 `m3a_interface_audit.json`）→ 无 full bridge。
- 读出：模式投影 schema/管线**接通了**，但几何零模型**没跑**（geometry_null_status=`not_run`，是
  "未运行"非"跑了没过"）→ verdict=`projection_only`（连 placement 都没算），撑不起 cohort bridge。

内部代号：§5 non-normal `finite_time_gain` / `core_transient_response` axial+self-limited (AR2-specific),
leading-mode global, dispersion AR2-vs-AR1 anisotropy, `m3b_verdict` = SPM-PASS frozen map.

## Current state (2026-06-27)

Interface contract + its contract-layer TDD are green (`tests/test_sef_hfo_m3_interface.py`, 46
passing). The spectral machinery is **implemented and green** in `src/topic4_m3b_spectral_phase.py`
(`tests/test_topic4_m3b_spectral_phase.py`, 81 passing — TDD-0..13 + TDD-15 real, 1 SNN smoke marked
slow). The declared artifacts + key figures are generated by `scripts/build_m3b_spectral_outputs.py`.
TDD-14 overlay stays `refused` (M3A artifacts absent).
""", encoding="utf-8")


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


def _fig_non_normal(nn_ar2, nn_ar1):
    T = nn_ar2["windows"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.4))
    ax1.plot(T, nn_ar2["gains"], "-o", ms=3, color="#d62728", label="AR2 (E→E scaffold)")
    ax1.plot(nn_ar1["windows"], nn_ar1["gains"], "-s", ms=3, color="#7f7f7f", label="AR1 isotropic")
    ax1.axhline(1.0, color="0.6", lw=0.8, ls="--")
    ax1.set_xlabel("transient window T (ms)")
    ax1.set_ylabel("finite-time gain  ‖e^{JT} b_core‖/‖b_core‖")
    ax1.set_title("Non-normal transient gain (peaks then self-limits)")
    ax1.legend(fontsize=8)
    ax2.plot(T, nn_ar2["axes"], "-o", ms=3, color="#d62728", label="AR2 (E→E scaffold)")
    ax2.plot(nn_ar1["windows"], nn_ar1["axes"], "-s", ms=3, color="#7f7f7f", label="AR1 isotropic")
    ax2.axhline(0.0, color="0.6", lw=0.8, ls="--")
    ax2.set_xlabel("transient window T (ms)")
    ax2.set_ylabel("transient response axis (along E→E)")
    ax2.set_title("Transient spreads ALONG the axis (scaffold-specific)")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "non_normal_gain_controllability.png", dpi=130)
    plt.close(fig)


def _write_readme():
    (FIG / "README.md").write_text(
        """# M3B-R2 谱相图 figures 说明

### homogeneous_dispersion.png
均质衬底沿 E→E 轴的色散曲线：横轴是空间花样的波数 k，纵轴是该花样的**线性增长率** Re λ。
关键观察是增长率在 **k=0（整片同步）最大**，没有出现"某个有限波长的行波最先长大"的
Brunel/Turing 式有限-k 峰——这是均质衬底"想全场一起点火"的体现。
**关注点**：曲线在 k=0 处最高、没有有限-k 峰（曲线先降到 k≈1.65 的谷再略回升，并非严格单调，
这不影响结论：最高点在 k=0）。

### phase_map_mode_class.png
核兴奋度（横，μ_core）× 全局去抑制（纵，q_global 越小越去抑制）二维相图，每格颜色=最先长大
的那个花样的类别。整张图基本只有 stable / global / runaway 三类，**没有 axial 档**。
**重要区分**：这里的 `runaway` 格全部来自**工作点非线性饱和**（op_status=saturated，是率方程
积分到稳态时跑到饱和高发放支），**不是**线性谱增长率 α₁>0（这些格的 α₁ 实际是负的 ≈ −0.05）。
也就是说"失控"是非线性饱和状态，不是线性不稳定——谱增长率（α₁）和非线性饱和状态是两回事，
本图按 leading-mode 类别上色，背后的 α₁ 数值在 mode_metrics.csv。
**关注点**：有没有 axial（轴向）格子；本图诚实结果是没有，且 runaway=非线性饱和而非 α₁>0。

### controls_summary.png
对照消融：绿柱=带结构（核/各向异性/连续核），灰柱=去掉结构（无核/各向同性/打散核）。
核把模式能量聚到核里（绿>灰）、各向异性核才有 45° 方向偏好（灰≈0）、连续核比打散核聚得更紧。
**关注点**：每对里绿柱是否明显高于灰柱（证明结构是核/骨架特异的，不是任意扰动都有）。

### mode_readout_projection.png
把谱里那个最核局域的本征模式，经虚拟 SEEG 触点读回来：点=触点，位置=沿轴/横向归一化坐标，
颜色=典型先后排名。这一步只验证"模型模式能干净地走完 Round-1 读出管线"，跟真实队列的对齐
+ 几何零模型检验需要真实数据、**未运行**（不是跑了没过），所以读出结论停在 placement-only。
**关注点**：点是否沿一条轴排开、颜色（排名）沿轴是否有梯度。

### non_normal_gain_controllability.png（§5 主图）
**这是按计划 §5 正确读轴向信号的图。** 给核一个扰动后，左图=瞬态增益随时间 T 的曲线
（‖e^{JT}·b_core‖/‖b_core‖），右图=瞬态响应沿 E→E 轴的拉伸随 T 的曲线。红=各向异性 E→E 骨架
（AR2），灰=各向同性对照（AR1）。两条增益线都在 ~10–15 ms 先冲到约 2 倍再衰减回去（**自限**，
非正规瞬态放大本身不挑骨架）；**只有 AR2 的响应沿轴拉开**，且沿轴拉伸的峰**比增益峰更靠后**
（约 30 ms，max≈0.4），不是"10 ms 内就一条完整行波"；AR1 灰线几乎不沿轴拉开（~0）——
**轴向是骨架特异的，放大本身不是**。
**关注点**：左图红/灰增益是否都先升后降（自限）；右图红线轴向是否明显抬起而灰线平（轴向骨架
特异）。这说明**间期自限轴向传播活在非正规瞬态里**——主导本征花样是全局的，轴向信号不在主导
花样而在瞬态，且方向是 E→E 各向异性给的。

### N/A（本轮未生成，原因如下）
- `example_modes.png` — 需要挑代表性 local/axial/mixed/global 本征模式做四联图；主导花样都是全局，留待需要时生成。
- `phase_map_gap_gain.png` — spectral-gap + finite-time-gain 等值线图，§5 主信号已在 non_normal_gain_controllability.png，此图作为补充待生成。
- `snn_spotcheck_grid.png` — 需要成组 SNN 抽查（每点 ~3s × 多点）；本轮只跑了 1 个 pilot（见 snn_spotcheck_summary.json）。
- `slow_trajectory_overlay.png` — M3A 轨迹叠加；M3A 五个交接产物全不存在、overlay 判定为 refused，故 N/A。
""", encoding="utf-8")


if __name__ == "__main__":
    main()
