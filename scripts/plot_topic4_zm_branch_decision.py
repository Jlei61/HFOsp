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
        bin_ms = float(z["bin_ms"])
        t_rate = np.arange(z["r_all"].size) * bin_ms / 1000.0
        burn_s = float(z["burn_in_ms"]) / 1000.0
        title = (
            f"seed {row['seed']} · {tier} · {row.get('T_cont_ms', 0) / 1000:g}s · "
            f"{row.get('morphology_label', 'unclassified')}\n"
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
            _style(ax, "axial activity kymograph", xlabel="time after fork (s)",
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
            _style(ax, f"multi-contact virtual readout ({fs:g} Hz)",
                   xlabel="time after fork (s)", ylabel="contact index")
        else:
            ax.text(0.5, 0.5, "readout unavailable", ha="center", va="center")
            ax.axis("off")

    fig.tight_layout()
    path = os.path.join(FIG, "native_confirmation_spatiotemporal_morphology.png")
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
def fig_phase_status(made):
    v = _load(os.path.join(OUT, "branch_verdict.json"), {})
    coverage = v.get("coverage", {})
    n_cells_run = int(coverage.get("n_cells_planned_run", coverage.get("n_cells_run", 0)))
    n_cells_planned = int(coverage.get("n_cells_planned", 0))
    source_layer = (v.get("layers") or {}).get("source_space_carrier")
    if source_layer in {"provisional_carrier_window", "source_space_carrier"}:
        fork_status = "discovery complete"
    else:
        fork_status = ("complete" if n_cells_planned and n_cells_run >= n_cells_planned
                       else "in progress" if n_cells_run else "not started")
    n_neighbourhood = int(v.get("n_neighbourhood_rows", 0))
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
        ("1C neighbourhood audit", neighbourhood_status),
        ("1.5A functional rank", "conditional / not authorized"),
        ("1.5B modal / gain", "conditional / not authorized"),
        ("2A Z-entry boundary", "conditional / not authorized"),
        ("2B offset boundary", "conditional / not authorized"),
        ("3 exit-driver comparison", "conditional / not authorized")]
    colors = {
        "complete": "#00798C",
        "discovery complete": "#00798C",
        "in progress": "#EDAE49",
        "exploratory partial": "#7B5AA6",
        "blocked: returning-event windows": "#D1495B",
        "failed": "#D1495B",
        "not started": "#E3E5E8",
        "conditional / not authorized": "#E3E5E8",
    }
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    y = np.arange(len(phases))
    ax.barh(y, [1] * len(phases), color=[colors[status] for _, status in phases])
    for i, (name, status) in enumerate(phases):
        dark = status in {
            "complete", "discovery complete", "exploratory partial",
            "blocked: returning-event windows", "failed"
        }
        ax.text(0.02, i, name, va="center", fontsize=8.5,
                color="white" if dark else "#444",
                fontweight="bold" if status in {"complete", "discovery complete"} else "normal")
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
               fig_confirmation_morphology, fig_impostors, fig_neighbourhood):
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
