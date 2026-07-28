#!/usr/bin/env python
"""Diagnostic figures for the Z/M minimal-carrier branch decision (plan Task 14).

These are DIAGNOSTIC panels, not a paper-ready lifecycle figure: recovery is not established, so the
Topic-4 four-column paper standard deliberately does not apply here (figure_style_guide.md, "M3A-v2
diagnostic variant" clause). Only phases that actually ran are plotted; anything the stop rule cut
is drawn as an explicit "not run" state, never as a blank pass.

  python scripts/plot_topic4_zm_branch_decision.py
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPTS)
for _p in (_ROOT, os.path.join(_ROOT, "src", "snn_engine")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

OUT = os.path.join(_ROOT, "results", "topic4_sef_hfo", "zm_branch_decision")
FIG = os.path.join(OUT, "figures")

C_SIM, C_OBS = "#3B6FB6", "#B0B7C3"
ARM_COLOR = {"freeze_all": "#D1495B", "freeze_zm": "#EDAE49", "freeze_zsg": "#00798C",
             "freeze_z": "#7B5AA6", "dynamic_replay": "#2E2E2E", "dynamic_z_only": "#8A8A8A"}
KLASS_SHORT = {"stable_carrier": "ST", "metastable_carrier": "MS", "transient_carrier_like": "TR",
               "hfo_like_relaxation_train": "HF", "runaway": "RA", "saturated_plateau": "PL",
               "probabilistically_indeterminate": "??", "no_evidence": "--"}


def _load(path, default=None):
    return json.load(open(path)) if os.path.exists(path) else default


def _style(ax, title=None, xlabel=None, ylabel=None):
    ax.spines[["top", "right"]].set_visible(False)
    if title:
        ax.set_title(title, fontsize=10, loc="left", fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)


# ============================================================ Fig 1: state + exact-resume parity
def fig_phase0():
    inv = _load(os.path.join(OUT, "phase0", "state_inventory.json"))
    if not inv:
        return None
    rows = inv["rows"]
    cats = {}
    for r in rows:
        c = cats.setdefault(r["category"], [0, 0])
        c[0 if r["role"] == "simulator" else 1] += 1
    order = sorted(cats, key=lambda k: -(cats[k][0] + cats[k][1]))

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), gridspec_kw=dict(width_ratios=[1, 1.35]))
    ax = axes[0]
    y = np.arange(len(order))
    sim = [cats[k][0] for k in order]
    obs = [cats[k][1] for k in order]
    ax.barh(y, sim, color=C_SIM, label="simulator state (in snapshot)")
    ax.barh(y, obs, left=sim, color=C_OBS, label="observer only")
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=8)
    ax.invert_yaxis()
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    _style(ax, "dynamic-state inventory", xlabel="number of states")

    # live split-resume parity demonstration on the small fixture
    ax = axes[1]
    try:
        from params import Params
        from connectivity import place_neurons, build_connectivity
        from kick_probe import simulate_kick
        from slow_field import SpatialSlowField, SpatialSlowFieldConfig
        import src.topic4_zm_checkpoint as CK

        p = Params(L=1.0, density=400.0, T=220.0, dt=0.1, seed=1, nu_ext_ratio=1.0)
        rng = np.random.default_rng(1)
        pos, labels, NE, NI = place_neurons(p, rng)
        net = build_connectivity(p, pos, labels, NE, NI, rng, verbose=False)
        N = NE + NI
        vth = np.full(N, 18.0)
        vth[:5] = 16.0
        core = np.linalg.norm(pos[:NE] - np.array([p.L / 2, p.L / 2]), axis=1) <= 0.3

        def mk():
            return SpatialSlowField(N, 18.0, pos[:NE], pos[NE:], p.L, core_mask_E=core,
                                    cfg=SpatialSlowFieldConfig(
                                        use_qI=False, use_gK=False, use_z=True, use_m=True,
                                        tau_z=200.0, I_th_EI=0.6, tau_adp=200.0, eta_m=0.5,
                                        use_SG=True, alpha_G=16.0, r50_psi=0.05, n_grid=16))

        def run(T, ckpt=None):
            net["rng"] = np.random.default_rng(1)
            import dataclasses
            return simulate_kick(dataclasses.replace(p, T=T), net, 5.0,
                                 kick_center=np.array([p.L / 2, p.L / 2]), r_kick=0.3, t_kick=50.0,
                                 V_th_per_neuron=vth, slow=mk(), verbose=False, zm_ckpt=ckpt)

        tf = 1200
        ck = CK.ZMCheckpoint(snapshot_steps=[tf])
        full = run(220.0, ck)
        cont = run((full["E_spk_bool"].shape[0] - tf) * 0.1,
                   CK.ZMCheckpoint(initial_state=ck.snapshots[tf]))
        t_full = np.arange(full["rate_E"].size) * 0.1
        ax.plot(t_full, full["rate_E"], color="#2E2E2E", lw=1.1, label="continuous run")
        ax.plot(t_full[tf:], cont["rate_E"], color="#D1495B", lw=1.1, ls="--",
                label="checkpoint -> restore -> continue")
        d = np.abs(full["rate_E"][tf:] - cont["rate_E"]).max()
        ax.axvline(tf * 0.1, color="#00798C", lw=1.0)
        ax.text(tf * 0.1 + 2, ax.get_ylim()[1] * 0.92, f"fork\nmax |diff| = {d:.0f}", fontsize=8,
                color="#00798C", va="top")
        ax.legend(fontsize=8, frameon=False, loc="upper left")
        _style(ax, "exact split-resume parity", xlabel="time (ms)", ylabel="E rate (Hz)")
    except Exception as e:                                    # pragma: no cover - figure only
        ax.text(0.5, 0.5, f"parity demo unavailable:\n{e}", ha="center", va="center", fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    path = os.path.join(FIG, "phase0_state_and_resume_parity.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ============================================================ Fig 2: anchors
def fig_anchors():
    paths = sorted(glob.glob(os.path.join(OUT, "anchors", "seed*", "anchor.json")))
    if not paths:
        return None
    fig, axes = plt.subplots(len(paths), 3, figsize=(15, 3.0 * len(paths)), squeeze=False)
    for i, p in enumerate(paths):
        man = json.load(open(p))
        tr = np.load(os.path.join(os.path.dirname(p), "anchor_traces.npz"))
        bin_ms = man["bin_ms"]
        t = np.arange(len(tr["r_core"])) * bin_ms / 1000.0
        esc = man["selection"]["eligibility"]["escalation_ms"]
        seed = man["seed"]
        snaps = [(s["bin_name"], s["fast_phase"], s["t_ms"] / 1000.0)
                 for s in man.get("captured_states", [])]

        ax = axes[i][0]
        floor = 0.05
        ax.plot(t, np.maximum(tr["r_core"], floor), color="#D1495B", lw=0.7, label="core")
        ax.plot(t, np.maximum(tr["r_surround"], floor), color="#00798C", lw=0.7, label="surround")
        if esc:
            ax.axvline(esc / 1000.0, color="#7B5AA6", lw=1.4, label="escalation")
        for _, _, ts in snaps:
            ax.axvline(ts, color="#888", lw=0.5, alpha=0.7)
        ax.set_yscale("log")
        ax.set_ylim(bottom=floor)
        ax.legend(fontsize=7, frameon=False, loc="lower right", ncol=3)
        _style(ax, f"seed {seed}: source rate (25 ms bins)", xlabel="time (s)", ylabel="rate (Hz)")

        ax = axes[i][1]
        ax.plot(t, tr["slow_z_core"], color="#D1495B", lw=1.1, label="z core")
        ax.plot(t, tr["slow_z_surround"], color="#EDAE49", lw=1.1, label="z surround")
        ax2 = ax.twinx()
        ax2.plot(t, tr["slow_S_G"], color="#00798C", lw=0.9, label="S_G")
        mmax = max(1e-9, float(np.max(tr["slow_m_core"])))
        ax2.plot(t, tr["slow_m_core"] / mmax, color="#7B5AA6", lw=0.8, ls=":",
                 label=f"m core / {mmax:.0f}")
        ax2.set_ylabel("S_G  /  normalized m", fontsize=8)
        ax2.set_ylim(0, 1.05)
        ax2.tick_params(labelsize=8)
        ax2.spines[["top"]].set_visible(False)
        if esc:
            ax.axvline(esc / 1000.0, color="#7B5AA6", lw=1.4)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=7, frameon=False, loc="upper right", ncol=2)
        ax.set_ylim(0.3, 1.12)
        _style(ax, f"seed {seed}: slow coordinates", xlabel="time (s)",
               ylabel="z (inhibitory efficacy)")

        ax = axes[i][2]
        locks = man.get("locks") or {}
        d = np.asarray(locks.get("d_rest_anchor", []), float)
        if d.size:
            ax.plot(np.arange(d.size) * bin_ms / 1000.0, d, color="#2E2E2E", lw=0.7)
            ax.axhline(locks["d_rest_thresh"], color="#D1495B", lw=1.0, ls="--",
                       label=f"rest threshold {locks['d_rest_thresh']:.2f}")
            ax.legend(fontsize=7, frameon=False, loc="upper left")
        if esc:
            ax.axvline(esc / 1000.0, color="#7B5AA6", lw=1.2)
        _style(ax, f"seed {seed}: distance from interictal rest", xlabel="time (s)",
               ylabel="d_rest (SD units)")
    fig.tight_layout()
    path = os.path.join(FIG, "anchor_trajectory_bins.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ============================================================ Fig 3: carrier matrix
def fig_carrier_matrix():
    v = _load(os.path.join(OUT, "branch_verdict.json"))
    if not v or not v.get("cells"):
        return None
    cells = v["cells"]
    arms = [a for a in ARM_COLOR if any(c["arm"] == a for c in cells)]
    cols = sorted({(c["seed"], c["bin_name"], c["fast_phase"]) for c in cells})
    M = np.full((len(arms), len(cols)), np.nan)
    lab = np.full((len(arms), len(cols)), "", dtype=object)
    for c in cells:
        i, j = arms.index(c["arm"]), cols.index((c["seed"], c["bin_name"], c["fast_phase"]))
        M[i, j] = c["p_median"] if c["p_median"] is not None else np.nan
        lab[i, j] = KLASS_SHORT.get(c["klass"], "?")

    rows_raw = []
    for p in sorted(glob.glob(os.path.join(OUT, "forks", "seed*", "fork_matrix.json"))):
        rows_raw.extend(json.load(open(p))["rows"])

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 4.6), gridspec_kw=dict(width_ratios=[1.5, 1]))
    ax = axes[0]
    im = ax.imshow(M, cmap="RdYlBu_r", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([f"s{s}\n{b.replace('bounded_', '')}\n{p}" for s, b, p in cols], fontsize=7)
    ax.set_yticks(range(len(arms)))
    ax.set_yticklabels(arms, fontsize=8)
    for i in range(len(arms)):
        for j in range(len(cols)):
            if lab[i, j]:
                ax.text(j, i, lab[i, j], ha="center", va="center", fontsize=8, color="#111")
    cb = fig.colorbar(im, ax=ax, fraction=0.035)
    cb.set_label("posterior median P_carrier(8 s)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    _style(ax, "minimal subsystem x slow state: carrier probability")
    coverage = v.get("coverage", {})
    n_done = int(coverage.get("n_cells_planned_run", coverage.get("n_cells_run", 0)))
    n_plan = int(coverage.get("n_cells_planned", 0))
    n_extra = int(coverage.get("n_cells_extra", 0))
    coverage_note = (
        f"discovery coverage {n_done}/{n_plan} planned + {n_extra} extra"
        if n_plan else "coverage unavailable"
    )
    ax.set_title("minimal subsystem x slow state: carrier probability\n"
                 f"{coverage_note} · blank = not computed · ST/MS require v1.1 stationarity",
                 fontsize=9, loc="left", fontweight="bold")

    ax = axes[1]
    for arm in arms:
        sel = [r for r in rows_raw if r["arm"] == arm]
        if not sel:
            continue
        ax.scatter([r["rest_returns"] for r in sel], [r["lifetime_ms"] for r in sel],
                   s=42, alpha=0.85, color=ARM_COLOR[arm], label=arm, edgecolor="white", lw=0.6)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_yscale("symlog", linthresh=10)
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    _style(ax, "how each arm fails", xlabel="returns to the interictal basin in 8 s",
           ylabel="carrier lifetime (ms)")
    fig.tight_layout()
    path = os.path.join(FIG, "carrier_subsystem_matrix.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ============================================================ Fig 4: representative dynamics
def fig_dynamics():
    files = sorted(glob.glob(os.path.join(OUT, "forks", "seed*", "traces", "*.npz")))
    if not files:
        return None
    by_arm = {}
    for f in files:
        # Trace names are ``bin__phase__arm__replicate.npz``.  Do not infer the
        # arm from a fixed token that actually points at the fast phase.
        tokens = os.path.basename(f).removesuffix(".npz").split("__")
        arm = next((a for a in ARM_COLOR if a in tokens), None)
        if arm is not None:
            by_arm.setdefault(arm, f)
    arms = [a for a in ARM_COLOR if a in by_arm]
    if not arms:
        return None
    fig, axes = plt.subplots(len(arms), 1, figsize=(11, 2.1 * len(arms)), squeeze=False, sharex=True)
    for i, arm in enumerate(arms):
        z = np.load(by_arm[arm])
        bm = float(z["bin_ms"])
        t = np.arange(len(z["r_all"])) * bm / 1000.0
        b0 = int(round(float(z["burn_in_ms"]) / bm))
        r_post = np.asarray(z["r_all"], float)[b0:]
        mean_r = float(np.mean(r_post)) if r_post.size else float("nan")
        cv_r = (float(np.std(r_post) / mean_r)
                if r_post.size and mean_r > 0 else float("nan"))
        # Descriptive morphology only; this does not alter the preregistered
        # carrier verdict.  It prevents a stationary high-rate branch from
        # being visually narrated as an ictal oscillation.
        morphology = "tonic-like" if np.isfinite(cv_r) and cv_r < 0.05 else "modulated"
        ax = axes[i][0]
        ax.plot(t, z["r_all"], color=ARM_COLOR[arm], lw=0.9, label="population rate (Hz)")
        ax2 = ax.twinx()
        ax2.plot(t, z["d_rest"], color="#2E2E2E", lw=0.7, alpha=0.75, label="d_rest")
        ax2.axhline(float(z["d_rest_thresh"]), color="#D1495B", lw=0.9, ls="--")
        ax2.set_ylabel("d_rest", fontsize=8)
        ax2.tick_params(labelsize=8)
        ax2.spines[["top"]].set_visible(False)
        ax.axvspan(0, float(z["burn_in_ms"]) / 1000.0, color="#DDD", alpha=0.6)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        if i == 0:
            ax.legend(h1 + h2, l1 + l2, fontsize=7, frameon=False, loc="upper right")
        _style(
            ax, f"{arm} — {morphology}, mean={mean_r:.1f} Hz, CV={cv_r:.3f}",
            ylabel="rate (Hz)",
        )
    axes[-1][0].set_xlabel("time after fork (s)", fontsize=9)
    fig.tight_layout()
    path = os.path.join(FIG, "carrier_continuation_dynamics.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ============================================================ Fig 4b: native confirmation morphology
def fig_confirmation_morphology():
    """Show spatial/readout morphology of completed native confirmation runs.

    A continuation can satisfy the source-space stationarity gate while being a
    nearly uniform tonic fixed point. The rate, axial kymograph, and contact
    readout therefore need to be shown together before the word ``ictal`` is
    used.
    """
    audit = _load(os.path.join(OUT, "confirmations", "carrier_morphology.json"), {})
    audit_by_key = {
        (r["tier"], int(r["seed"]), r["row_key"]): r
        for r in audit.get("rows", [])
    }
    manifests = []
    for tier in ("dt2", "long"):
        for p in sorted(glob.glob(os.path.join(
                OUT, "confirmations", tier, "seed*", "fork_matrix.json"))):
            man = _load(p, {})
            for row in man.get("rows", []):
                trace = os.path.join(
                    os.path.dirname(p), "traces",
                    f"{row['bin_name']}__{row['fast_phase']}__"
                    f"{row['arm']}__{row['replicate']}.npz",
                )
                if os.path.exists(trace):
                    manifests.append((tier, row, trace))
    if not manifests:
        return None

    fig, axes = plt.subplots(
        len(manifests), 3, figsize=(14.5, 3.0 * len(manifests)), squeeze=False
    )
    for i, (tier, row, trace) in enumerate(manifests):
        z = np.load(trace)
        morphology = audit_by_key.get((tier, int(row["seed"]), row["key"]), {})
        bin_ms = float(z["bin_ms"])
        t_rate = np.arange(z["r_all"].size) * bin_ms / 1000.0
        burn_s = float(z["burn_in_ms"]) / 1000.0
        coarse_label = morphology.get(
            "coarse_rate_label",
            str(row.get("morphology_label", "unclassified")).replace(
                "tonic_like_fixed", "tonic_at_25ms"
            ),
        )
        readout_label = morphology.get("readout_temporal_class", "readout audit pending")
        title = (
            f"seed {row['seed']} · {tier} · {row.get('T_cont_ms', 0) / 1000:g}s · "
            f"{coarse_label} / {readout_label}\n"
            f"mean={row.get('r_all_mean_hz', float('nan')):.1f} Hz, "
            f"CV={row.get('r_all_cv', float('nan')):.3f}, "
            f"extent={row.get('spatial_extent_fraction', float('nan')):.2f}, "
            f"stationary={bool(row.get('stationarity_ok', False))}"
        )

        ax = axes[i, 0]
        ax.plot(t_rate, z["r_all"], color=ARM_COLOR["freeze_all"], lw=0.8)
        ax.axvspan(0, burn_s, color="#DDD", alpha=0.6)
        _style(ax, title, xlabel="time after fork (s)", ylabel="population rate (Hz)")

        ax = axes[i, 1]
        if "kymo_axial" in z:
            kymo = np.asarray(z["kymo_axial"], float)
            vmax = float(np.nanpercentile(kymo, 99)) if np.any(np.isfinite(kymo)) else 1.0
            ax.imshow(
                kymo, origin="lower", aspect="auto", interpolation="nearest",
                extent=[0, kymo.shape[1] * bin_ms / 1000.0, 0, kymo.shape[0]],
                cmap="magma", vmin=0, vmax=max(vmax, 1e-9),
            )
            ktcv = morphology.get("kymograph_temporal_cv_median", float("nan"))
            _style(ax, f"axial activity kymograph · temporal CV={ktcv:.3f}",
                   xlabel="time after fork (s)",
                   ylabel="axial spatial bin")
        else:
            ax.text(0.5, 0.5, "kymograph unavailable", ha="center", va="center")
            ax.axis("off")

        ax = axes[i, 2]
        if "lfp" in z:
            lfp = np.asarray(z["lfp"], float)
            if lfp.ndim == 1:
                lfp = lfp[:, None]
            # Canonical producer stores time x contact. Fail visibly rather
            # than silently treating a short time axis as contacts.
            if lfp.shape[0] < lfp.shape[1]:
                lfp = lfp.T
            med = np.nanmedian(lfp, axis=0, keepdims=True)
            scale = 1.4826 * np.nanmedian(np.abs(lfp - med), axis=0, keepdims=True)
            scale = np.where(scale > 1e-12, scale, 1.0)
            lfp_z = np.clip((lfp - med) / scale, -6, 6).T
            fs = float(z["lfp_fs"])
            ax.imshow(
                lfp_z, origin="lower", aspect="auto", interpolation="nearest",
                extent=[0, lfp.shape[0] / fs, 0, lfp.shape[1]],
                cmap="RdBu_r", vmin=-6, vmax=6,
            )
            f0 = morphology.get("dominant_frequency_median_hz", float("nan"))
            agree = morphology.get("dominant_frequency_agreement", float("nan"))
            plv = morphology.get("phase_coherence", float("nan"))
            _style(ax, f"multi-contact readout ({fs:g} Hz) · "
                       f"f0={f0:.1f} Hz, agree={agree:.2f}, PLV={plv:.2f}",
                   xlabel="time after fork (s)", ylabel="contact index")
        else:
            ax.text(0.5, 0.5, "readout unavailable", ha="center", va="center")
            ax.axis("off")

    fig.tight_layout()
    path = os.path.join(FIG, "native_confirmation_spatiotemporal_morphology.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ============================================================ Fig 4c: fine E/I source rhythm
def fig_source_rhythm():
    manifests = sorted(glob.glob(os.path.join(
        OUT, "source_rhythm", "*", "seed*", "source_rhythm.json"
    )))
    if not manifests:
        return None
    fig, axes = plt.subplots(
        len(manifests), 3, figsize=(13.5, 3.0 * len(manifests)), squeeze=False
    )
    for i, p in enumerate(manifests):
        man = _load(p, {})
        metrics = man.get("source_rhythm") or {}
        z = np.load(os.path.join(_ROOT, man["fields_path"]))
        bin_ms = float(z["bin_ms"])
        burn = int(round(float(z["burn_in_ms"]) / bin_ms))
        E = np.asarray(z["E_rate_grid"], float)[burn:]
        I = np.asarray(z["I_rate_grid"], float)[burn:]
        gE = np.asarray(z["global_E_rate_hz"], float)[burn:]
        gI = np.asarray(z["global_I_rate_hz"], float)[burn:]
        t = np.arange(gE.size) * bin_ms / 1000.0
        show = t <= min(0.75, t[-1])

        ax = axes[i, 0]
        ax.plot(t[show], gE[show], color="#D1495B", lw=0.8, label="global E")
        ax.plot(t[show], gI[show], color="#00798C", lw=0.8, label="global I")
        ax.legend(fontsize=7, frameon=False, ncol=2)
        _style(
            ax,
            f"seed {man['seed']} · {man['resolution']} · "
            f"{metrics.get('source_temporal_class', 'unclassified')}\n"
            f"f0={metrics.get('dominant_frequency_median_hz', float('nan')):.1f} Hz, "
            f"local agreement={metrics.get('local_frequency_agreement', float('nan')):.2f}",
            xlabel="time after burn-in (s)", ylabel="rate (Hz)",
        )

        # Complex Fourier coefficient at the locally dominant frequency.
        f0 = float(metrics.get("dominant_frequency_median_hz", np.nan))
        x = E - np.mean(E, axis=0, keepdims=True)
        freq = np.fft.rfftfreq(x.shape[0], d=bin_ms * 1e-3)
        idx = int(np.argmin(np.abs(freq - f0))) if np.isfinite(f0) else 0
        coef = np.fft.rfft(x, axis=0)[idx]
        amplitude = np.abs(coef) / max(1, x.shape[0])
        phase = np.angle(coef)

        ax = axes[i, 1]
        im = ax.imshow(amplitude, origin="lower", cmap="magma", interpolation="nearest")
        fig.colorbar(im, ax=ax, fraction=0.046, label="rate amplitude")
        _style(ax, f"local amplitude at {freq[idx]:.1f} Hz",
               xlabel="grid x", ylabel="grid y")

        ax = axes[i, 2]
        im = ax.imshow(phase, origin="lower", cmap="twilight", vmin=-np.pi, vmax=np.pi,
                       interpolation="nearest")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, ticks=[-np.pi, 0, np.pi])
        cb.ax.set_yticklabels(["-π", "0", "π"])
        _style(
            ax,
            f"local phase · PLV={metrics.get('local_phase_locking', float('nan')):.2f}",
            xlabel="grid x", ylabel="grid y",
        )
    fig.tight_layout()
    path = os.path.join(FIG, "fine_source_rhythm_and_phase_map.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ============================================================ Fig 6: standardized slow-coordinate rank
def fig_effective_rank():
    path_json = os.path.join(OUT, "effective_rank", "effective_rank_summary.json")
    summary = _load(path_json)
    if not summary:
        return None
    if "static_bootstrap" not in summary or "impulse_bootstrap" not in summary:
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.7))
        coverage = summary.get("coverage", [])
        seeds = [str(row.get("seed", "?")) for row in coverage]
        valid = [int(row.get("n_valid_rows", 0)) for row in coverage]
        invalid = [
            sum((row.get("invalid_by_coordinate") or {}).values())
            for row in coverage
        ]

        ax = axes[0]
        x = np.arange(len(seeds))
        ax.bar(x, valid, color="#3B6FB6", label="valid response")
        ax.bar(x, invalid, bottom=valid, color="#D1495B",
               label="physical central pair unavailable")
        ax.set_xticks(x)
        ax.set_xticklabels([f"seed {seed}" for seed in seeds])
        ax.legend(fontsize=8, frameon=False)
        _style(ax, "registered probe coverage",
               xlabel="seed", ylabel="completed probe rows")

        ax = axes[1]
        coordinates = sorted({
            coordinate
            for row in coverage
            for coordinate in (
                set((row.get("valid_by_coordinate") or {}))
                | set((row.get("invalid_by_coordinate") or {}))
            )
        })
        if coverage and coordinates:
            matrix = np.asarray([
                [
                    int((row.get("invalid_by_coordinate") or {}).get(
                        coordinate, 0
                    ))
                    for coordinate in coordinates
                ]
                for row in coverage
            ], float)
            im = ax.imshow(matrix, cmap="Reds", vmin=0, aspect="auto")
            ax.set_xticks(range(len(coordinates)))
            ax.set_xticklabels([c.upper() if c != "sg" else "S_G"
                                for c in coordinates])
            ax.set_yticks(range(len(seeds)))
            ax.set_yticklabels([f"seed {seed}" for seed in seeds])
            fig.colorbar(im, ax=ax, fraction=0.046,
                         label="invalid central-pair rows")
            _style(ax, "physical-boundary failures",
                   xlabel="trajectory coordinate", ylabel="seed")
        else:
            ax.axis("off")
            ax.text(0.5, 0.5, "No completed probe coverage",
                    ha="center", va="center", transform=ax.transAxes)

        ax = axes[2]
        ax.axis("off")
        verdict = summary.get("verdict", "no_evidence")
        text = (
            f"{verdict}\n\n"
            "A symmetric finite difference cannot cross a physical slow-state "
            "boundary. The missing M column is therefore unavailable evidence, "
            "not evidence for rank 1 or rank 2.\n\n"
            f"Complete analyzable seeds: {summary.get('n_seeds', 0)}"
        )
        ax.text(0.02, 0.98, text, ha="left", va="top", wrap=True,
                transform=ax.transAxes, fontsize=9)
        _style(ax, "fail-closed interpretation")
        fig.tight_layout()
        path = os.path.join(FIG, "slow_coordinate_effective_rank.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path
    static = summary["static_bootstrap"]
    impulse = summary["impulse_bootstrap"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.7))

    ax = axes[0]
    for label, result, color in (
        ("static", static, "#D1495B"),
        ("impulse", impulse, "#00798C"),
    ):
        s = np.asarray(result["point"]["singular_values"], float)
        s = s / max(s[0], 1e-12)
        ax.plot(np.arange(1, s.size + 1), s, marker="o", color=color, label=label)
    ax.axhline(0.2, color="#888", ls="--", lw=0.8)
    ax.set_yscale("log")
    ax.legend(fontsize=8, frameon=False)
    _style(ax, f"standardized rank · {summary['verdict']}",
           xlabel="singular direction", ylabel="singular value / s1")

    ax = axes[1]
    states = list(summary["per_state"])
    x = np.arange(len(states))
    st = [summary["per_state"][s]["static_point"]["s2_over_s1"] for s in states]
    it = [summary["per_state"][s]["impulse_point"]["s2_over_s1"] for s in states]
    ax.plot(x, st, marker="o", color="#D1495B", label="static")
    ax.plot(x, it, marker="o", color="#00798C", label="impulse")
    ax.axhline(0.2, color="#888", ls="--", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("bounded_", "").replace("__peak", "") for s in states])
    ax.legend(fontsize=8, frameon=False)
    _style(ax, "rank across visited carrier states", ylabel="s2 / s1")

    ax = axes[2]
    mats = []
    for state in states:
        mats.extend(np.asarray(summary["per_state"][state]["static_seed_matrices"], float))
    mean_matrix = np.mean(mats, axis=0)
    vmax = max(1e-9, float(np.max(np.abs(mean_matrix))))
    im = ax.imshow(mean_matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_yticks(range(4))
    ax.set_yticklabels(["rate", "active area", "spatial entropy", "vSEEG energy"])
    ax.set_xticks(range(3))
    ax.set_xticklabels(["Z", "M", "S_G"])
    fig.colorbar(im, ax=ax, fraction=0.046, label="standardized sensitivity")
    _style(ax, "mean static response matrix",
           xlabel="trajectory-field coordinate", ylabel="observable")
    fig.tight_layout()
    path = os.path.join(FIG, "slow_coordinate_effective_rank.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ============================================================ Fig 7: projected modal/operator audit
def fig_modal_operator():
    summary_path = os.path.join(
        OUT, "modal_operator", "modal_operator_summary.json"
    )
    summary = _load(summary_path)
    rows = (summary or {}).get("rows", [])
    if not rows:
        return None
    rows = sorted(rows, key=lambda row: int(row["seed"]))
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 7.0))
    seeds = [f"seed {row['seed']}" for row in rows]
    x = np.arange(len(rows))

    ax = axes[0, 0]
    ax.bar(x - 0.17, [row["spectral_radius"] for row in rows], 0.34,
           color="#D1495B", label="spectral radius")
    ax.bar(x + 0.17, [row["finite_time_gain"] for row in rows], 0.34,
           color="#00798C", label="finite-time gain")
    ax.axhline(1.0, color="#777", ls="--", lw=0.8)
    ax.set_xticks(x, seeds)
    ax.legend(fontsize=7, frameon=False)
    _style(ax, f"projected operator · {summary['status']}", ylabel="gain")

    ax = axes[0, 1]
    errors = [row["heldout_median_relative_error"] for row in rows]
    ax.bar(x, errors, color="#EDAE49")
    ax.axhline(0.20, color="#D1495B", ls="--", lw=0.9,
               label="locked max error")
    ax.set_xticks(x, seeds)
    ax.legend(fontsize=7, frameon=False)
    _style(ax, "held-out composite prediction", ylabel="relative error")

    ax = axes[0, 2]
    width = 0.26
    for offset, key, label, color in (
        (-width, "right_mode_angle_to_E_pathology_axis_deg", "right→E axis", "#D1495B"),
        (0.0, "right_mode_angle_to_axial_EI_subspace_deg", "right→axial E/I", "#00798C"),
        (width, "optimal_input_angle_to_axial_EI_subspace_deg", "optimal→axial E/I", "#7B5AA6"),
    ):
        ax.bar(x + offset, [row[key] for row in rows], width,
               color=color, label=label)
    ax.set_xticks(x, seeds)
    ax.set_ylim(0, 90)
    ax.legend(fontsize=6.5, frameon=False)
    _style(ax, "mode alignment", ylabel="acute angle (deg)")

    heat_axes = [axes[1, 0], axes[1, 1]]
    for panel, row in zip(heat_axes, rows):
        manifest_path = os.path.join(
            OUT, "modal_operator", f"seed{row['seed']}", "modal_probes.json"
        )
        manifest = _load(manifest_path)
        operator = None
        if manifest and manifest.get("operator_path"):
            horizon = float(row["horizon_ms"])
            energy = float(row["total_energy_mv2"])
            key = f"h{horizon:g}_E{energy:g}".replace(".", "p")
            with np.load(os.path.join(_ROOT, manifest["operator_path"]),
                         allow_pickle=False) as arrays:
                operator = np.asarray(arrays[f"{key}_operator"], float)
        if operator is None:
            panel.axis("off")
            continue
        vmax = max(1e-12, float(np.max(np.abs(operator))))
        image = panel.imshow(
            operator, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal"
        )
        panel.set_xlabel("input voltage mode", fontsize=8)
        panel.set_ylabel("future voltage mode", fontsize=8)
        panel.set_title(
            f"seed {row['seed']} · {row['operator_tool']}", fontsize=9, loc="left"
        )
        panel.tick_params(labelsize=6)
        fig.colorbar(image, ax=panel, fraction=0.046)

    ax = axes[1, 2]
    axial = [row["axial_column_gain"] for row in rows]
    transverse = [row["transverse_column_gain"] for row in rows]
    ax.bar(x - 0.17, axial, 0.34, color="#D1495B", label="axial")
    ax.bar(x + 0.17, transverse, 0.34, color="#3B6FB6", label="transverse")
    ax.set_xticks(x, seeds)
    ax.legend(fontsize=7, frameon=False)
    _style(ax, "pathology-axis response contrast", ylabel="column gain")
    fig.tight_layout()
    path = os.path.join(FIG, "trajectory_conditioned_modal_operator.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ============================================================ Fig 8: conditional Z-entry boundary
def fig_entry_boundary():
    summary = _load(os.path.join(
        OUT, "boundaries", "entry", "entry_boundary_summary.json"
    ))
    if not summary or not summary.get("n_rows"):
        return None
    boundary = summary["boundary"]
    curve = boundary.get("curve", [])
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))

    ax = axes[0]
    if curve:
        q = np.asarray([row["q"] for row in curve], float)
        median = np.asarray([row["posterior_median"] for row in curve], float)
        ci = np.asarray([row["posterior_ci"] for row in curve], float)
        ax.plot(q, median, marker="o", color="#D1495B")
        ax.fill_between(q, ci[:, 0], ci[:, 1], color="#D1495B", alpha=0.18)
    ax.axhline(0.5, color="#777", ls="--", lw=0.8)
    if boundary.get("q_half") is not None:
        ax.axvline(boundary["q_half"], color="#00798C", lw=1.2)
        qci = boundary.get("q_half_ci")
        if qci is not None:
            ax.axvspan(qci[0], qci[1], color="#00798C", alpha=0.14)
    ax.set_ylim(0, 1)
    _style(ax, f"P_enter · {summary['verdict']}",
           xlabel="actual-field Z interpolation λ", ylabel="posterior P(carrier)")

    ax = axes[1]
    manifests = [
        _load(path) for path in sorted(glob.glob(os.path.join(
            OUT, "boundaries", "entry", "seed*", "entry_probes.json"
        )))
    ]
    colors = {1: "#D1495B", 3: "#00798C", 4: "#7B5AA6"}
    for manifest in manifests:
        if not manifest:
            continue
        seed = int(manifest["seed"])
        rows = manifest.get("rows", [])
        levels = sorted({float(row["lambda"]) for row in rows})
        probability = [
            np.mean([
                bool(row["entered_carrier"]) for row in rows
                if np.isclose(row["lambda"], level)
            ])
            for level in levels
        ]
        ax.plot(levels, probability, marker="o", color=colors.get(seed, "#555"),
                label=f"seed {seed}")
    ax.axhline(0.5, color="#777", ls="--", lw=0.8)
    ax.set_ylim(-0.03, 1.03)
    ax.legend(fontsize=8, frameon=False)
    _style(ax, "replicate outcomes by seed",
           xlabel="conditional Z coordinate λ", ylabel="empirical carrier fraction")
    fig.tight_layout()
    path = os.path.join(FIG, "conditional_z_entry_boundary.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ============================================================ Fig 9: existing-coordinate offset
def fig_offset_boundary():
    summary = _load(os.path.join(
        OUT, "boundaries", "offset", "offset_boundary_summary.json"
    ))
    families = (summary or {}).get("family_results", {})
    if not families:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.0))
    colors = {"M_alone": "#D1495B", "M_SG": "#00798C",
              "M_Z_recovery": "#7B5AA6"}

    ax = axes[0]
    for family, result in families.items():
        curve = result["boundary"].get("curve", [])
        if not curve:
            continue
        q = np.asarray([row["q"] for row in curve], float)
        p = np.asarray([row["posterior_median"] for row in curve], float)
        ax.plot(q, p, marker="o", color=colors[family], label=family)
        q_half = result["boundary"].get("q_half")
        if q_half is not None:
            ax.axvline(q_half, color=colors[family], lw=0.8, alpha=0.7)
    ax.axhline(0.5, color="#777", ls="--", lw=0.8)
    ax.axvline(1.0, color="#999", ls=":", lw=0.8, label="actual range end")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, frameon=False)
    _style(ax, f"P_remain · {summary['verdict']}",
           xlabel="existing-coordinate λ", ylabel="posterior P(remain carrier)")

    ax = axes[1]
    names = list(families)
    values = [
        families[name].get("low_basin_persistence_fraction") or 0.0
        for name in names
    ]
    ax.bar(np.arange(len(names)), values,
           color=[colors[name] for name in names])
    ax.set_xticks(np.arange(len(names)),
                  [name.replace("_", "\n") for name in names])
    ax.set_ylim(0, 1)
    _style(ax, "matched-low basin coexistence",
           ylabel="fraction returning/staying in low basin")

    ax = axes[2]
    dynamic = summary.get("dynamic_ZM", {})
    posterior = dynamic.get("posterior_offset_reached") or {}
    median = float(posterior.get("median", 0.0))
    lo, hi = float(posterior.get("lo", 0.0)), float(posterior.get("hi", 0.0))
    ax.errorbar([0], [median], yerr=[[median - lo], [hi - median]],
                fmt="o", color="#00798C", capsize=4)
    ax.axhline(0.8, color="#D1495B", ls="--", lw=0.8)
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(0, 1)
    ax.set_xticks([0], ["dynamic Z+M\nS_G frozen"])
    _style(ax, "actual ODE realization",
           ylabel="Jeffreys P(offset to rest basin)")
    fig.tight_layout()
    path = os.path.join(FIG, "existing_slow_coordinate_offset.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ============================================================ Fig 5: readout impostor separation
def fig_impostors():
    import src.topic4_zm_empirical_carrier as EC
    FS, DUR = 2000.0, 6.0
    classes = {"broadband\ncarrier": EC.synth_broadband_carrier,
               "sharp harmonic\npulse train": EC.synth_pulse_train,
               "stationary global\noscillator": EC.synth_global_oscillator}
    metrics = ("harmonic_comb", "spectral_entropy", "phase_coherence", "inst_freq_drift_hz")
    titles = ("harmonic comb\n(pulse train -> high)", "spectral entropy\n(broadband -> high)",
              "phase coherence\n(global rhythm -> high)", "inst. frequency drift\n(stationary -> low)")
    vals = {k: [] for k in metrics}
    for name, fn in classes.items():
        b = [EC.metric_battery(fn(FS, DUR, seed=s), FS) for s in range(4)]
        for k in metrics:
            vals[k].append([x[k] for x in b])
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.4))
    colors = ["#00798C", "#D1495B", "#EDAE49"]
    for j, (k, ttl) in enumerate(zip(metrics, titles)):
        ax = axes[j]
        for i, name in enumerate(classes):
            v = np.asarray(vals[k][i], float)
            ax.bar(i, np.nanmean(v), color=colors[i], width=0.62)
            ax.scatter([i] * v.size, v, s=14, color="#222", zorder=3)
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(list(classes), fontsize=7)
        _style(ax, ttl)
    fig.tight_layout()
    path = os.path.join(FIG, "readout_impostor_discrimination.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ============================================================ Fig 6: neighbourhood
def fig_neighbourhood():
    paths = sorted(glob.glob(os.path.join(OUT, "neighbourhood", "seed*", "neighbourhood.json")))
    if not paths:
        return None
    fig, axes = plt.subplots(1, len(paths), figsize=(6.2 * len(paths), 4.2), squeeze=False)
    for i, p in enumerate(paths):
        m = json.load(open(p))
        ax = axes[0][i]
        fams = {}
        for r in m["rows"]:
            fams.setdefault((r["family"], r["label"]), []).append(r)
        keys = sorted(fams)
        y = np.arange(len(keys))
        surv = [np.mean([bool(x["survived"]) for x in fams[k]]) for k in keys]
        life = [np.median([x["lifetime_ms"] for x in fams[k]]) for k in keys]
        # This is an exploratory scaffold until the manifest's fail-closed audit is complete.
        # A raw survivor is not a formal local-carrier result.
        ax.barh(y, life, color=["#B0B7C3" if s == 0 else "#7B5AA6" for s in surv])
        ax.set_yticks(y)
        ax.set_yticklabels([f"{a}\n{b}" for a, b in keys], fontsize=7)
        ax.invert_yaxis()
        ax.set_xscale("symlog", linthresh=10)
        _style(ax, f"seed {m['seed']}: local neighbourhood probes",
               xlabel="median carrier lifetime (ms)")
        ax.legend(handles=[Patch(color="#B0B7C3", label="no raw replica survived 8 s"),
                           Patch(color="#7B5AA6", label="raw survivor; not formal evidence")],
                  fontsize=7, frameon=False, loc="lower right")
        ax.text(0.01, -0.18, "exploratory only: formal neighbourhood audit incomplete",
                transform=ax.transAxes, fontsize=7.5, color="#7B5AA6")
    fig.tight_layout()
    path = os.path.join(FIG, "neighbourhood_local_audit.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ============================================================ Fig 7: phase completion
def _phase_status_rows():
    """Return phase labels from scientific verdicts, not merely artifact counts."""
    v = _load(os.path.join(OUT, "branch_verdict.json"), {})
    coverage = v.get("coverage", {})
    n_cells_run = int(coverage.get("n_cells_planned_run", coverage.get("n_cells_run", 0)))
    n_cells_planned = int(coverage.get("n_cells_planned", 0))
    source_layer = (v.get("layers") or {}).get("source_space_carrier")
    if source_layer in {
        "carrier_window", "provisional_carrier_window", "source_space_carrier"
    }:
        fork_status = "carrier window complete"
    else:
        fork_status = ("complete" if n_cells_planned and n_cells_run >= n_cells_planned
                       else "in progress" if n_cells_run else "not started")
    n_neighbourhood = int(v.get("n_neighbourhood_rows", 0))
    n_source_rhythm = len(glob.glob(os.path.join(
        OUT, "source_rhythm", "*", "seed*", "source_rhythm.json"
    )))
    source_rhythm_summary = _load(os.path.join(
        OUT, "source_rhythm", "source_rhythm_summary.json"
    ), {})
    rank_summary = _load(os.path.join(
        OUT, "effective_rank", "effective_rank_summary.json"
    ), {})
    n_rank_seed = len(glob.glob(os.path.join(
        OUT, "effective_rank", "seed*", "rank_probes.json"
    )))
    modal_summary = _load(os.path.join(
        OUT, "modal_operator", "modal_operator_summary.json"
    ), {})
    n_modal_seed = int(modal_summary.get("n_complete_seeds", 0))
    entry_summary = _load(os.path.join(
        OUT, "boundaries", "entry", "entry_boundary_summary.json"
    ), {})
    n_entry_seed = int(entry_summary.get("n_complete_seeds", 0))
    offset_summary = _load(os.path.join(
        OUT, "boundaries", "offset", "offset_boundary_summary.json"
    ), {})
    n_offset_seed = int(offset_summary.get("n_complete_seeds", 0))
    entry_verdict = entry_summary.get("verdict")
    if entry_verdict == "conditional_Z_entry_boundary_crossed":
        entry_status = "boundary crossed"
    elif entry_verdict == "conditional_Z_entry_boundary_unresolved":
        entry_status = "unresolved"
    elif entry_verdict == "no_evidence":
        entry_status = "no evidence"
    else:
        entry_status = (
            "in progress" if n_entry_seed else "conditional / not authorized"
        )
    offset_verdict = offset_summary.get("verdict")
    if offset_verdict in {
        "M_sufficient_and_reached",
        "M_SG_joint_offset_reached",
        "M_Z_recovery_offset_reached",
    }:
        offset_status = "offset reached"
    elif offset_verdict in {
        "M_boundary_near_but_unreached",
        "M_Z_recovery_boundary_exists_but_unreached",
    }:
        offset_status = "exists / unreached"
    elif offset_verdict in {"M_shapes_but_no_offset_surface", "no_evidence"}:
        offset_status = "no evidence: offset surface"
    else:
        offset_status = (
            "in progress" if n_offset_seed else "conditional / not authorized"
        )
    if source_rhythm_summary.get("status") == "class_disagreement":
        source_rhythm_status = "complete: class disagreement"
        modal_status = "skipped: source-class disagreement"
    else:
        source_rhythm_status = (
            "complete" if n_source_rhythm >= 2 else
            "in progress" if n_source_rhythm else "conditional / not authorized"
        )
        modal_status = (
            "complete" if n_modal_seed >= 2 else
            "no evidence" if modal_summary.get("status") == "insufficient_seeds" else
            "in progress" if n_modal_seed else "conditional / not authorized"
        )
    neighbourhood = v.get("neighbourhood") or {}
    neighbourhood_status = (
        "complete" if neighbourhood.get("evidence_complete") is True
        else "exploratory partial" if n_neighbourhood
        else "not started"
    )
    reference_status = (
        "complete" if v.get("reference_artifacts") == "locked"
        else "blocked: returning-event windows" if v.get("reference_artifacts") == "blocked"
        else "not started"
    )
    confirmation_status = (v.get("confirmation") or {}).get("status")
    confirmation_label = {
        "passed": "complete",
        "pending": "in progress",
        "failed": "failed",
    }.get(confirmation_status, "not started")
    phases = [
        ("0A state inventory", "complete" if v.get("state_inventory", {}).get("status") == "ok"
         else "not started"),
        ("0B exact resume + noise", "complete" if v.get("gates", {}).get("passed")
         else "not started"),
        ("0C synthetic readout sanity", "complete"),
        ("0C real reference lock", reference_status),
        ("0D vertical slice", "complete" if os.path.exists(os.path.join(
            OUT, "smoke", "vertical_slice", "vertical_slice_seed1.json")) else "not started"),
        ("1A anchors", "complete" if len(v.get("eligible_seeds", [])) >= 3 else "in progress"),
        ("1B minimal-subsystem forks", fork_status),
        ("1B 20 s + native dt/2 confirm", confirmation_label),
        ("1.5 carrier-type source audit", source_rhythm_status),
        ("1C neighbourhood audit", neighbourhood_status),
        ("1.5A functional rank",
         "no evidence: central-pair boundary"
         if rank_summary.get("verdict") == "no_evidence_incomplete_central_pairs"
         else "complete" if rank_summary else
         "in progress" if n_rank_seed else "conditional / not authorized"),
        ("1.5B modal / gain", modal_status),
        ("2A Z-entry boundary", entry_status),
        ("2B offset boundary", offset_status),
        ("3 exit-driver comparison", "conditional / not authorized")]
    return phases, v


def fig_phase_status(made):
    phases, v = _phase_status_rows()
    colors = {
        "complete": "#00798C",
        "discovery complete": "#00798C",
        "carrier window complete": "#00798C",
        "complete: class disagreement": "#7B5AA6",
        "boundary crossed": "#00798C",
        "offset reached": "#00798C",
        "exists / unreached": "#7B5AA6",
        "in progress": "#EDAE49",
        "exploratory partial": "#7B5AA6",
        "blocked: returning-event windows": "#D1495B",
        "failed": "#D1495B",
        "unresolved": "#7B5AA6",
        "no evidence": "#B0B7C3",
        "no evidence: central-pair boundary": "#B0B7C3",
        "no evidence: offset surface": "#B0B7C3",
        "skipped: source-class disagreement": "#B0B7C3",
        "not started": "#E3E5E8",
        "conditional / not authorized": "#E3E5E8",
    }
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    y = np.arange(len(phases))
    ax.barh(y, [1] * len(phases), color=[colors[status] for _, status in phases])
    for i, (name, status) in enumerate(phases):
        dark = status in {
            "complete", "discovery complete", "carrier window complete",
            "complete: class disagreement", "boundary crossed", "offset reached",
            "exists / unreached", "exploratory partial",
            "blocked: returning-event windows", "failed", "unresolved"
        }
        ax.text(0.02, i, name, va="center", fontsize=8.5,
                color="white" if dark else "#444",
                fontweight="bold" if status in {
                    "complete", "discovery complete", "carrier window complete",
                    "boundary crossed", "offset reached"
                } else "normal")
        ax.text(0.98, i, status, va="center", ha="right",
                fontsize=7.5, color="white" if dark else "#666")
    ax.set_yticks([])
    ax.set_xticks([])
    ax.invert_yaxis()
    ax.spines[:].set_visible(False)
    ax.set_title(f"phase completion — verdict: {v.get('verdict', 'not adjudicated')}",
                 fontsize=10, loc="left", fontweight="bold")
    fig.tight_layout()
    path = os.path.join(FIG, "phase_completion_status.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


FIGURE_NOTES = {
    "phase0_state_and_resume_parity.png": (
        "展示 canonical Z/M 状态清单、checkpoint round-trip 与 exact-resume 门。"
        "这是工程/状态语义门，不是 carrier 阳性证据。",
        "状态字段完整且 split/resume 与连续运行一致。"
    ),
    "anchor_trajectory_bins.png": (
        "展示三个 primary seed 的 Z/M+S_G 自然轨迹，以及被选中的 slow-state bin 和"
        " fast-phase snapshot。纵轴使用对数率，便于同时看到间期小事件和 bounded 段。",
        "snapshot 必须来自自然轨迹；比较 early/mid/late 与 trough/rising/peak。"
    ),
    "carrier_subsystem_matrix.png": (
        "展示已完成 fork cell 的 Jeffreys posterior carrier probability 与失败方式。"
        "空白是未计算，?? 是 posterior/replica 尚未闭合；不能读成阴性。",
        "只有同一 arm 在相邻 slow bins、两个 fast phases、至少两个 seeds 的兼容阳性"
        "才能形成 carrier window。"
    ),
    "carrier_continuation_dynamics.png": (
        "展示代表性 continuation 的全场率与到间期 basin 的距离。灰区是 burn-in，"
        "虚线是 rest-distance threshold。",
        "区分持续高活动支、返回间期、relaxation train 与 runaway；平坦高率支本身"
        "不等于已经复现 ictal oscillation。"
    ),
    "native_confirmation_spatiotemporal_morphology.png": (
        "把原生 dt/2 与 20 秒 confirmation 的 population rate、轴向时空图和多触点"
        "虚拟电极读数并列。只有实际完成并保存完整 trace 的 confirmation 才会出现。",
        "即使 source-space stationarity 通过，也要检查空间范围、传播/相位结构和触点"
        "能量；全场 tonic fixed point 不能表述成有界 ictal oscillation。"
    ),
    "fine_source_rhythm_and_phase_map.png": (
        "展示 confirmation 通过后，从同一自然 snapshot 得到的 2 ms E/I source-rate field。"
        "左图是全局 E/I，中央和右侧分别是主频局部振幅与相位。",
        "判断高频成分是全局同步、局部相位错开还是不规则活动；该图只负责 operator 路由，"
        "不等于 observation-space 或 ictal lifecycle 验收。"
    ),
    "slow_coordinate_effective_rank.png": (
        "展示沿真实 early-to-late Z、M、S_G 场方向做配对中心差分后，标准化响应矩阵的"
        "奇异值谱、各慢状态的 s2/s1 以及平均静态灵敏度。",
        "判断这些既有慢变量在 carrier 附近是否只是局部共线；rank-1 不能外推成整个"
        "慢流形一维。"
    ),
    "trajectory_conditioned_modal_operator.png": (
        "展示按 fine-source carrier 类型选择的 E/I 膜电位有限时传播算子、held-out "
        "组合扰动误差、轴向模式夹角和轴向/横向增益。输入与输出保持在同一膜电位坐标，"
        "避免把 voltage-to-rate susceptibility 误写成 eigen/Floquet 算子。",
        "先看 held-out 与子空间残差是否过门；只有预测有效时，谱半径、有限时增益和"
        "病理轴夹角才有动力学解释。"
    ),
    "conditional_z_entry_boundary.png": (
        "展示从 matched pre-entry fast state 出发，沿真实 pre-entry→carrier Z 场方向"
        "插值时的 P_enter；M 和完整 S_G family 固定在 onset-adjacent 值。阴影为"
        "Jeffreys/Bootstrap 不确定度。",
        "只有 P=0.5 被实际采样点包围、bootstrap 稳定且真实 λ=0→1 方向穿越，才能写"
        "conditional Z-entry boundary；它不是 Z 的全局充分性。"
    ),
    "existing_slow_coordinate_offset.png": (
        "比较 M、M+S_G 和 M+Z-recovery 三条真实慢场方向上的 P_remain=0.5 "
        "边界，同时显示 matched-low basin 是否仍存在，以及动态 Z+M、固定 S_G 时"
        "是否真正退入 rest basin。",
        "offset 只表示离开 carrier；即使回到 rest basin，也不能自动写成原有间期事件"
        "恢复，更不能单独构成完整 lifecycle。"
    ),
    "readout_impostor_discrimination.png": (
        "展示合成 broadband carrier、尖锐谐波 pulse train 与全局固定振荡器在"
        " readout 指标上的可分性。这是 observation gate 的 synthetic sanity check。",
        "真实 observation claim 仍需被锁定的 returning-event/early-ictal 参考窗。"
    ),
    "neighbourhood_local_audit.png": (
        "展示 slow-field PCA、trajectory interpolation 与 pathology-axis 邻域探针。"
        "raw survivor 仅是探索性信号，除非 manifest 的 formal audit 完整。",
        "三种表示必须一致，才能在 Branch T 与 Branch F 之间作决定。"
    ),
    "phase_completion_status.png": (
        "展示 Rev3.1 各阶段完成度、阻断项和当前 fail-closed verdict。",
        "区分 complete、in progress、blocked 与 conditional/not authorized。"
    ),
}


def write_figure_readme(paths):
    """Write notes only for figures that were actually produced in this invocation."""
    lines = [
        "# Z/M minimal-carrier branch-decision 图说明",
        "",
        "本目录是动力学诊断图，不是 paper-ready lifecycle figure。当前阶段未运行或被阻断的"
        "证据不得由空白面板代替。",
        "",
    ]
    for path in paths:
        name = os.path.basename(path)
        if name not in FIGURE_NOTES:
            continue
        body, focus = FIGURE_NOTES[name]
        lines.extend([f"### {name}", "", body, "", f"**关注点**：{focus}", ""])
    with open(os.path.join(FIG, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def main():
    os.makedirs(FIG, exist_ok=True)
    made = []
    failures = []
    for fn in (fig_phase0, fig_anchors, fig_carrier_matrix, fig_dynamics,
               fig_confirmation_morphology, fig_source_rhythm, fig_impostors,
               fig_neighbourhood, fig_effective_rank, fig_modal_operator,
               fig_entry_boundary, fig_offset_boundary):
        try:
            p = fn()
        except Exception as e:                                # pragma: no cover - figure only
            print(f"[plot] {fn.__name__} failed: {type(e).__name__}: {e}")
            failures.append((fn.__name__, type(e).__name__, str(e)))
            p = None
        if p:
            made.append(p)
            print(f"[plot] {os.path.relpath(p, _ROOT)}")
        else:
            print(f"[plot] {fn.__name__}: no data yet (skipped)")
    p = fig_phase_status(made)
    made.append(p)
    print(f"[plot] {os.path.relpath(p, _ROOT)}")
    write_figure_readme(made)
    if failures:
        raise RuntimeError(f"figure generation failed: {failures}")
    return made


if __name__ == "__main__":
    main()
