"""M4 dynamic q_I -> virtual SEEG LFP readout GIF (E1146 real-electrode geometry).

Two-row comparison GIF in the style of `fig_m3a_v2_2_qI_stim_site_compare` (same 3-column
per-arm layout: permissivity | 2D SNN activity | virtual-SEEG readout):

    row 0  alpha_G=16  divisive shared inhibitory pool S_G ON  -> BOUNDED broad sustained
                                                                  (non-runaway) state
    row 1  alpha_G=0   no pool                                 -> whole-field runaway

Same E1146 subject substrate (`run_m4_phaseplane.build_substrate`, seed=1, twoend_equal, L=20,
N=40000): two small low-V_th axial cores at the registered source/sink centroids fire
SPONTANEOUSLY (KICK_BOOST=0, no kick), the inhibitory-resource field q_I depletes DYNAMICALLY
(k_q=0.10, not frozen), and the two arms differ ONLY in the M4 global divisive shared pool S_G
(alpha_G=16 vs 0). The M4 dynamic runs recorded NO LFP, so this runner RE-RUNS the two arms WITH an
`LFPRecorder` placed on the patient's REAL registered contacts (S["reg"]["montage_sheet"], SCL/ICL
shafts) -> the virtual SEEG readout is what a clinician would see on the patient's own electrodes.

Sim and render are separated: `--run` runs the two sims, caches the light-weight render inputs to
`<figdir>/m4_seeg_readout_cache.npz`, then renders; `--render-only` re-renders from the cache with NO
re-sim (fast layout iteration for the render->eyeball->fix loop over a ~30 min sim).

SCIENTIFIC STATUS (held; honoured in titles/README):
  - The alpha_G=16 bounded state is a NON-RUNAWAY BOUNDED ATTRACTOR CANDIDATE, not a full seizure
    cycle: the pool BOUNDS the q_I-depletion runaway, it does NOT terminate it (q_I stays pinned in
    the cores; discharge self-locks).
  - The bounded state is BROAD (full-width band, ~60% of the sheet, sheet-MEAN q_I above the 0.05
    floor), NOT a localized ictal core.
  - Runaway / tonic saturation is NEVER an ictal-like event.
  - Visual diagnostic: single seed (=1), one trajectory per arm; not a statistical sweep. (Cohort
    status: aG16 bound holds seed 1/3/4, delayed-runaway seed2; non-monotonic in alpha_G. seed=1 is
    a bounded seed.)

Nothing runs on import; the sim + render is gated behind __main__ / --run / --render-only.

Run:
    python scripts/paper_figures/plot_fig_m4_seeg_readout_gif.py --run              # T=8000, ~30 min
    python scripts/paper_figures/plot_fig_m4_seeg_readout_gif.py --render-only      # re-render from cache
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
ENG = ROOT / "src" / "snn_engine"
for _p in (str(ROOT), str(ROOT / "scripts"), str(ENG)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_m4_phaseplane as PP  # noqa: E402  (E1146 build_substrate, R_KICK, CORE_R)
import plot_fig_m3a_v2_2_hG_runaway_transition_gif as H  # noqa: E402  (contact drawing helpers)
from kick_probe import simulate_kick  # noqa: E402
from lfp import LFPRecorder  # noqa: E402
from slow_field import SpatialSlowField, SpatialSlowFieldConfig, firing_rate_field  # noqa: E402

FIG_NAME = "fig_m4_dynamic_qi"
STEM = "m4_seeg_readout"
OUT_DIR = ROOT / "results" / "paper-ready-figure" / FIG_NAME / "figures"
CACHE = OUT_DIR / f"{STEM}_cache.npz"

# ---- dynamic q_I + pool params (IDENTICAL to run_m4_dynamic_qi.py's confirmed bounded arm) ----
K_Q = 0.10
TAU_Q, TAU_A, SIGMA_Q, Q_MIN = 5000.0, 20.0, 1.5, 0.05
R50_PSI, N_PSI, P_POOL, TAU_MU, TAU_S, S_MAX = 0.4, 2.0, 3.0, 30.0, 80.0, 1.0
ALPHA_G_BOUNDED = 16.0
DT = 0.1
RUNAWAY_HZ, RUNAWAY_DUR_MS = 120.0, 100.0
# The no-pool runaway arm blows up early (~386 ms for kq0.10, per the M4 sweep) and then sits in a
# near-constant saturated plateau; sim it just past onset + a solid plateau, then HOLD (see _run_arm).
RUNAWAY_T_SIM = 1000.0

N_GRID = 32               # slow-field lattice + activity field resolution
ACTIVITY_WINDOW_MS = 10.0
TRACE_OFF = 1.5           # vertical spacing between stacked SEEG contact traces
SG_COL = "#5b2a86"        # S_G pool output (purple, distinct from the greens)
FOCUS_COL = "crimson"


# ===========================================================================
# Slow-field subclass that snapshots q_I frames (simulate_kick keeps only the
# final field + the mean trace; the permissivity column needs per-frame fields).
# Non-invasive: this object is constructed HERE, step() calls super() then records.
# ===========================================================================
class _RecordingSlowField(SpatialSlowField):
    def set_frames(self, frame_steps) -> None:
        self._frame_set = {int(x) for x in frame_steps}
        self._sc = 0
        self.q_frames: list[np.ndarray] = []
        self.trace_qI_min: list[float] = []

    def step(self, spk, labels, dt):
        super().step(spk, labels, dt)              # advances q_I / S_G, appends trace_qI_mean / trace_SG
        self.trace_qI_min.append(float(self.q_I.min()))
        if self._sc in self._frame_set:
            self.q_frames.append(self.q_I.copy())
        self._sc += 1


def _make_cfg(use_SG: bool, alpha_G: float) -> SpatialSlowFieldConfig:
    return SpatialSlowFieldConfig(
        use_qI=True, use_gK=False, k_q=K_Q, k_K=0.0, sigma_q=SIGMA_Q, sigma_K=0.5,
        q_min=Q_MIN, q_init=1.0, tau_q=TAU_Q, tau_a=TAU_A,
        use_SG=use_SG, alpha_G=alpha_G, beta_SG=0.0, r0_psi=0.0, r50_psi=R50_PSI,
        n_psi=N_PSI, p_pool=P_POOL, tau_mu=TAU_MU, tau_S=TAU_S, S_max=S_MAX)


# ===========================================================================
# Runaway detector (shared 120 Hz / 100 ms / 80%-rule; same as run_m4_dynamic_qi.py)
# ===========================================================================
def _smooth(rate, dt, win_ms=20.0):
    n = max(1, int(round(win_ms / dt)))
    return np.convolve(np.asarray(rate, float), np.ones(n) / n, mode="same")


def _first_sustained(rate, dt, thr=RUNAWAY_HZ, dur=RUNAWAY_DUR_MS):
    above = np.asarray(rate) >= thr
    n = max(1, int(round(dur / dt)))
    if above.size < n:
        return None
    c = np.convolve(above.astype(float), np.ones(n), mode="valid")
    idx = np.flatnonzero(c >= 0.80 * n)
    return None if idx.size == 0 else round(float(idx[0] * dt), 1)


# ===========================================================================
# Per-arm sim + light-weight extraction (frees the ~2.5 GB E_spk_bool right after).
# t_sim<T_full: sim only to t_sim, then HOLD the end state to the full display length.
# The runaway arm blows up at ~386 ms and then sits in a near-constant DC saturated
# plateau (|LFP| median ~= p98); simulating its long saturated tail costs ~200 ms/step
# (all neurons firing every step) for no new information, so we sim just past onset and
# hold the flat fixed point -- faithful for that arm, and it cuts the runaway sim ~3x.
# The bounded arm (t_sim=None) sims the full window (its band keeps growing, must be real).
# ===========================================================================
def _run_arm(S, rec, use_SG, alpha_G, frame_steps, t_sim=None):
    T_full = float(S["p"].T)
    nsteps_full = int(round(T_full / DT))
    nsteps_sim = nsteps_full if t_sim is None else int(round(float(t_sim) / DT))
    sim_frames = frame_steps[frame_steps < nsteps_sim]           # frames actually simulated (sorted first k)
    n_hold = len(frame_steps) - len(sim_frames)                  # frames past t_sim -> hold the last one
    slow = _RecordingSlowField(S["N"], 18.0, S["posE"], S["posI"], S["L"], cfg=_make_cfg(use_SG, alpha_G))
    slow.set_frames(sim_frames)
    S["net"]["rng"] = np.random.default_rng(S["seed"])
    S["p"].T = nsteps_sim * DT                                   # sim only to t_sim ...
    t0 = time.time()
    res = simulate_kick(S["p"], S["net"], 0.0, slow=slow, kick_center=list(S["src_xy"]),
                        r_kick=PP.R_KICK, t_kick=1e9, V_th_per_neuron=S["vth"], lfp_recorder=rec)
    S["p"].T = T_full                                            # ... then restore (render reads S["p"].T)
    rate = np.asarray(res["rate_E"], float)                      # already per-neuron Hz (kick_probe:363)
    runaway = _first_sustained(_smooth(rate, DT), DT)
    act, vals = [], []                                           # 2D activity field per simulated frame
    for step in sim_frames:
        lo = max(0, int(step) - int(round(ACTIVITY_WINDOW_MS / DT)))
        fired = res["E_spk_bool"][lo: int(step) + 1].any(axis=0)
        A = firing_rate_field(fired, S["posE"], S["L"], N_GRID, sigma=0.5)
        act.append(A)
        if np.any(A > 0):
            vals.append(A[A > 0])
    activity_vmax = max(1.0, float(np.percentile(np.concatenate(vals), 98))) if vals else 1.0
    q_frames = list(slow.q_frames) + [slow.q_frames[-1]] * n_hold        # hold last q_I field past t_sim
    act = act + [act[-1]] * n_hold                                       # hold last activity field

    def _pad(a):                                                # edge-hold a per-step trace to full length
        a = np.asarray(a, float)
        return a if a.shape[0] >= nsteps_full else np.pad(a, (0, nsteps_full - a.shape[0]), mode="edge")
    abslfp = np.abs(np.asarray(res["lfp_trace"], float).T)              # (n_contacts, nsteps_sim)
    if abslfp.shape[1] < nsteps_full:                                   # edge-hold each contact past t_sim
        abslfp = np.pad(abslfp, ((0, 0), (0, nsteps_full - abslfp.shape[1])), mode="edge")
    out = dict(
        label=("bounded_aG16" if use_SG else "runaway_aG0"), use_SG=bool(use_SG), alpha_G=float(alpha_G),
        runaway_ms=runaway, max_rate_hz=round(float(_smooth(rate, DT).max()), 1),
        q_mean_final=round(float(slow.q_I.mean()), 4), q_min_final=round(float(slow.q_I.min()), 4),
        S_G_max=round(float(max(slow.trace_SG)) if slow.trace_SG else 0.0, 4),
        t_sim_ms=(None if t_sim is None else round(nsteps_sim * DT, 1)),
        times=np.arange(nsteps_full) * DT,
        q_frames=np.asarray(q_frames, float), act_fields=np.asarray(act, float),
        activity_vmax=activity_vmax, abslfp=abslfp.copy(),
        qI_mean=_pad(slow.trace_qI_mean), qI_min=_pad(slow.trace_qI_min),
        SG=(_pad(slow.trace_SG) if slow.trace_SG else np.zeros(0, float)),
        wall_s=round(time.time() - t0, 1),
    )
    del res  # free E_spk_bool (~2.5 GB) + lfp_trace before the next arm
    return out


# ===========================================================================
# SHARED-ABSOLUTE SEEG normalisation: ONE (base, scale) for BOTH arms, anchored at
# the bounded arm's quiet median (base) and the combined 98th percentile (scale).
# The real |LFP| distributions justify this: bounded is mostly low (median ~17) with
# intermittent deflections that REACH ~94% of the saturated runaway level, while
# runaway is a near-constant high plateau (median ~= p98 ~= 2600). On a shared
# absolute scale the readout then shows, on the SAME axis:
#   bounded  -> quiet baseline + a sustained bounded rhythmic train on the ICL
#               contacts over the active band; distal SCL contacts stay flat-low
#   runaway  -> every contact jumps to a saturated HIGH plateau
# (Per-arm self-normalisation would hide this cross-arm amplitude/saturation contrast.)
# Computed ONCE over both arms; stored on each arm as arm["zlfp"].
# ===========================================================================
def _zlfp_shared_abs(arms):
    cat = np.concatenate([a["abslfp"] for a in arms], axis=1)
    quiet = [a["abslfp"] for a in arms if a["use_SG"]]                 # bounded arm = quiet baseline ref
    base = float(np.median(quiet[0])) if quiet else float(np.median(cat))
    scale = max(float(np.percentile(cat, 98)) - base, 1e-9)
    for a in arms:
        a["zlfp"] = (a["abslfp"] - base) / scale


# ===========================================================================
# Cache (sim outputs -> npz) so rendering can iterate without re-simming.
# ===========================================================================
_ARR_KEYS = ("times", "q_frames", "act_fields", "abslfp", "qI_mean", "qI_min", "SG")
_SCALAR_KEYS = ("label", "use_SG", "alpha_G", "runaway_ms", "max_rate_hz", "q_mean_final",
                "q_min_final", "S_G_max", "activity_vmax", "t_sim_ms", "wall_s")


def _save_cache(path, S, arms, frame_steps):
    contacts = np.asarray(S["reg"]["montage_sheet"].contacts, float)
    names = list(S["reg"]["montage_sheet"].names)
    blob = dict(seed=int(S["seed"]), L=float(S["L"]), N=int(S["N"]), NE=int(S["NE"]), T=float(S["p"].T),
                src_xy=[float(x) for x in S["src_xy"]], snk_xy=[float(x) for x in S["snk_xy"]],
                names=names, frame_steps=[int(x) for x in frame_steps],
                arms=[{k: a[k] for k in _SCALAR_KEYS} for a in arms])
    arrs = {"contacts": contacts, "meta_json": np.array(json.dumps(blob))}
    for a in arms:
        for k in _ARR_KEYS:
            arrs[f"{a['label']}__{k}"] = a[k]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrs)


def _load_cache(path):
    z = np.load(path, allow_pickle=True)
    blob = json.loads(str(z["meta_json"]))
    contacts = np.asarray(z["contacts"], float)
    S = {"L": blob["L"], "seed": blob["seed"], "N": blob["N"], "NE": blob["NE"],
         "src_xy": np.asarray(blob["src_xy"], float), "snk_xy": np.asarray(blob["snk_xy"], float),
         "p": type("P", (), {"T": blob["T"]})()}
    arms = []
    for sa in blob["arms"]:
        a = dict(sa)
        for k in _ARR_KEYS:
            a[k] = np.asarray(z[f"{sa['label']}__{k}"])
        a["SG"] = a["SG"] if a["SG"].size else None
        arms.append(a)
    return S, arms, contacts, blob["names"], np.asarray(blob["frame_steps"], int)


# ===========================================================================
# Drawing
# ===========================================================================
def _draw_spatial_overlays(ax, S, contacts, names, L):
    ax.plot([S["src_xy"][0], S["snk_xy"][0]], [S["src_xy"][1], S["snk_xy"][1]],     # E->E anisotropy axis
            color="white", lw=1.2, alpha=0.9, zorder=5)
    for xy, lab in ((S["src_xy"], "src"), (S["snk_xy"], "snk")):
        ax.add_patch(plt.Circle(xy, PP.CORE_R, fill=False, ec=FOCUS_COL, lw=1.1, ls="--", zorder=7))
        ax.text(xy[0], xy[1] + PP.CORE_R + 0.35, lab, fontsize=7.5, color=FOCUS_COL, fontweight="bold",
                ha="center", va="bottom", path_effects=[pe.withStroke(linewidth=1.8, foreground="white")])
    H._draw_contacts(ax, contacts, names)
    H._style_spatial(ax, L)


def _draw_arm(fig, row_spec, S, arm, contacts, names, qi, tm_cursor, *, row_title):
    L = float(S["L"])
    T = float(S["p"].T)
    shafts = sorted({H._shaft(n) for n in names})
    times = arm["times"]
    runaway = arm["runaway_ms"]
    rg = row_spec.subgridspec(1, 3, width_ratios=[1.0, 1.0, 2.15], wspace=0.20)

    # --- col 0: permissivity (1 - q_I) ---
    ax0 = fig.add_subplot(rg[0, 0])
    im0 = ax0.imshow(1.0 - arm["q_frames"][qi], origin="lower", extent=[0, L, 0, L],
                     cmap="plasma", vmin=0.0, vmax=1.0)
    _draw_spatial_overlays(ax0, S, contacts, names, L)
    ax0.set_title(f"{row_title} — permissivity (1-$q_I$)", fontsize=8.4, fontweight="bold", pad=3)
    fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.02).ax.tick_params(labelsize=6)

    # --- col 1: real-time 2D SNN E activity (per-arm vmax = spatial EXTENT, not magnitude) ---
    ax1 = fig.add_subplot(rg[0, 1])
    im1 = ax1.imshow(arm["act_fields"][qi], origin="lower", extent=[0, L, 0, L],
                     cmap="viridis", vmin=0.0, vmax=arm["activity_vmax"])
    _draw_spatial_overlays(ax1, S, contacts, names, L)
    ax1.set_title("2D SNN activity", fontsize=8.4, fontweight="bold", pad=3)
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.02).ax.tick_params(labelsize=6)

    # --- col 2 split: top q_I(mean/min)+S_G | bottom continuous virtual-SEEG readout ---
    sub = rg[0, 2].subgridspec(2, 1, height_ratios=[1.0, 2.4], hspace=0.08)
    axg = fig.add_subplot(sub[0, 0])
    axg.plot(times, arm["qI_mean"], color=H.QI_MEAN_COL, lw=1.6, zorder=4, label="mean $q_I$")
    axg.plot(times, arm["qI_min"], color=H.QI_MIN_COL, lw=1.1, ls="--", zorder=3, label="min $q_I$")
    axg.axhline(Q_MIN, color="0.6", lw=0.8, ls=":", zorder=1, label="$q_{\\min}$ floor")
    if arm["SG"] is not None and len(arm["SG"]):
        axg.plot(times, arm["SG"], color=SG_COL, lw=1.5, zorder=5, label="pool $S_G$")
    axg.axvline(tm_cursor, color="black", lw=1.1, alpha=0.9, zorder=7)
    if runaway is not None:
        axg.axvline(runaway, color="crimson", lw=1.0, ls="--", alpha=0.9, zorder=6)
    axg.set_xlim(0.0, T); axg.set_ylim(-0.03, 1.05)
    axg.tick_params(axis="x", labelbottom=False, length=2.0)
    axg.tick_params(axis="y", labelsize=6.2, length=2.0)
    axg.set_ylabel("$q_I,\\ S_G$", fontsize=7.4)
    axg.spines["top"].set_visible(False); axg.spines["right"].set_visible(False)
    if runaway is None:
        verdict, vcol = f"bounded — no runaway (peak {arm['max_rate_hz']:.0f} Hz)", "#2e7d32"
    else:
        verdict, vcol = f"runaway {runaway:.0f} ms (peak {arm['max_rate_hz']:.0f} Hz)", "crimson"
    axg.set_title(verdict, fontsize=8.2, fontweight="bold", pad=2, color=vcol)
    axg.legend(frameon=False, fontsize=6.0, loc="upper right", ncol=2, handlelength=1.2, columnspacing=0.7)

    # bottom: stacked continuous virtual-SEEG readout, coloured by shaft (shared absolute scale)
    ax2 = fig.add_subplot(sub[1, 0], sharex=axg)
    zlfp = arm["zlfp"]
    trace_y = np.arange(len(names)) * TRACE_OFF
    if runaway is not None:
        ax2.axvspan(runaway, T, color="crimson", alpha=0.06, lw=0, zorder=0)
    for i, nm in enumerate(names):
        ax2.plot(times, np.clip(zlfp[i], -TRACE_OFF * 0.72, TRACE_OFF * 0.72) + trace_y[i],
                 color=H._shaft_color(nm, shafts), lw=0.6, alpha=0.9, zorder=3)
    ax2.axvline(tm_cursor, color="black", lw=1.1, alpha=0.9, zorder=7)
    if runaway is not None:
        ax2.axvline(runaway, color="crimson", lw=1.0, ls="--", alpha=0.9, zorder=6)
    ax2.set_xlim(0.0, T)
    ax2.set_ylim(-TRACE_OFF, trace_y[-1] + TRACE_OFF)
    ax2.set_yticks(trace_y); ax2.set_yticklabels(names, fontsize=6.0)
    for tick, nm in zip(ax2.get_yticklabels(), names):
        tick.set_color(H._shaft_color(nm, shafts))
    ax2.tick_params(axis="x", labelsize=6.8, length=2.5); ax2.tick_params(axis="y", length=2.0)
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
    ax2.set_ylabel("contact (virtual SEEG)", fontsize=7.4); ax2.set_xlabel("time (ms)", fontsize=7.4)
    handles = [Patch(facecolor=H._shaft_color(f"{sh}0", shafts), edgecolor="none", label=f"{sh} shaft")
               for sh in shafts]
    if runaway is not None:
        handles.append(Patch(facecolor="crimson", alpha=0.2, edgecolor="none", label="post-runaway"))
    ax2.legend(handles=handles, frameon=False, fontsize=6.2, loc="upper right",
               bbox_to_anchor=(1.0, 1.10), ncol=len(handles), handlelength=1.3, columnspacing=0.7)


def _render(S, arms, contacts, names, frame_steps, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    _zlfp_shared_abs(arms)                          # shared absolute readout scale (both arms comparable)
    gif = out_dir / f"{STEM}.gif"; png = out_dir / f"{STEM}_final.png"; pdf = out_dir / f"{STEM}_final.pdf"
    T = float(S["p"].T)
    main = ("M4 dynamic $q_I$ depletion $\\rightarrow$ virtual SEEG readout: divisive shared pool $S_G$ "
            "bounds the runaway into a broad sustained state (E1146 geometry) | t={t:.0f} ms")
    foot = ("Bounded $\\alpha_G$=16 = NON-runaway bounded attractor candidate (BROAD band, not a localized "
            "ictal core), NOT a full seizure cycle; the pool BOUNDS but does not terminate. "
            "Runaway / tonic saturation is NOT an ictal-like event. Visual diagnostic, seed=1, one "
            "trajectory per arm. Readout on a shared absolute scale: bounded = sustained bounded rhythm on "
            "the ICL contacts over the band + quiet distal SCL; runaway = all contacts jump to a saturated plateau.")
    frames = []
    for qi, step in enumerate(frame_steps):
        last = qi == len(frame_steps) - 1
        tm_cursor = T if last else float(arms[0]["times"][int(step)])
        fig = plt.figure(figsize=(14.0, 4.9 * len(arms)), facecolor="white")
        outer = fig.add_gridspec(len(arms), 1, left=0.06, right=0.985, bottom=0.075, top=0.9, hspace=0.34)
        for ri, arm in enumerate(arms):
            title = ("$\\alpha_G$=16 pool ON" if arm["use_SG"] else "$\\alpha_G$=0 no pool")
            _draw_arm(fig, outer[ri], S, arm, contacts, names, qi, tm_cursor, row_title=title)
        fig.suptitle(main.format(t=tm_cursor), fontsize=11.5, fontweight="bold", y=0.975)
        fig.text(0.50, 0.02, foot, fontsize=7.4, ha="center", color="0.2")
        if last:
            fig.savefig(png, dpi=170, bbox_inches="tight", facecolor="white")
            fig.savefig(pdf, bbox_inches="tight", facecolor="white")
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
        plt.close(fig)
    frames.extend([frames[-1]] * 8)
    imageio.mimsave(gif, frames, duration=0.11, loop=0)
    return gif, png, pdf


def _write_metadata(S, arms, names, out_dir: Path, gif, png, pdf, frame_steps, wall_s):
    def arm_meta(a):
        return {k: a[k] for k in ("alpha_G", "use_SG", "runaway_ms", "max_rate_hz", "q_mean_final",
                                  "q_min_final", "S_G_max", "t_sim_ms", "wall_s")}
    meta = {
        "figure": FIG_NAME, "stem": STEM,
        "status": ("visual diagnostic; virtual SEEG LFP readout of the M4 dynamic-q_I bounded (alpha_G=16) "
                   "vs runaway (alpha_G=0) arms on the E1146 REAL registered contacts; seed=1, one "
                   "trajectory/arm; bounded = NON-runaway bounded attractor candidate (broad, not localized, "
                   "pool bounds not terminates); runaway/tonic is NOT ictal-like; NOT a statistical sweep"),
        "substrate": {"subject": PP.SUBJECT, "montage": PP.MONTAGE, "L": float(S["L"]), "N": int(S["N"]),
                      "NE": int(S["NE"]), "src_xy": [round(float(x), 3) for x in S["src_xy"]],
                      "snk_xy": [round(float(x), 3) for x in S["snk_xy"]], "seed": int(S["seed"]),
                      "placement": "template_source_twoend_equal (register_to_sheet montage_sheet contacts)"},
        "config": {"k_q": K_Q, "tau_q": TAU_Q, "tau_a": TAU_A, "sigma_q": SIGMA_Q, "q_min": Q_MIN,
                   "pool": {"alpha_G_bounded": ALPHA_G_BOUNDED, "r50_psi": R50_PSI, "n_psi": N_PSI,
                            "p_pool": P_POOL, "tau_mu": TAU_MU, "tau_S": TAU_S, "S_max": S_MAX},
                   "T": float(S["p"].T), "dt": DT, "runaway_criterion_hz": RUNAWAY_HZ,
                   "runaway_criterion_dur_ms": RUNAWAY_DUR_MS, "n_frames": int(len(frame_steps))},
        "contacts": {"n": len(names), "names": list(names),
                     "shafts": sorted({H._shaft(n) for n in names}),
                     "source": "S['reg']['montage_sheet'] (real E1146 geometry, plane-fit; NOT the synthetic "
                               "A/B montage that H._contacts falls back to for a layout-less substrate)"},
        "arms": {a["label"]: arm_meta(a) for a in arms},
        "readout_normalisation": "SHARED ABSOLUTE across both arms: base=median(bounded |LFP|) (quiet baseline), "
                                 "scale=(p98(combined)-base), display-clipped to +-0.72*TRACE_OFF. Justified by "
                                 "the real |LFP| (bounded median ~17 with events reaching ~94% of the runaway "
                                 "level; runaway a near-constant high plateau ~2600): band-overlying ICL show a "
                                 "sustained bounded rhythm, distal SCL stay low, runaway saturates all contacts",
        "activity_vmax": "per-arm (2D activity column shows spatial EXTENT; amplitude carried by the readout)",
        "runaway_arm_sim": ("no-pool arm blows up ~386 ms then sits in a near-constant saturated plateau; it is "
                            "SIMULATED to t_sim_ms and then HELD (edge) to the display T (the flat DC fixed point "
                            "is faithful and skips the ~200 ms/step saturated tail). Bounded arm is simulated fully."),
        "outputs": {"gif": str(gif.relative_to(ROOT)), "final_png": str(png.relative_to(ROOT)),
                    "final_pdf": str(pdf.relative_to(ROOT)), "cache": str(CACHE.relative_to(ROOT))},
        "wall_s": round(wall_s, 1),
    }
    (out_dir / f"{STEM}_metadata.json").write_text(json.dumps(meta, indent=2))
    return meta


def main():
    ap = argparse.ArgumentParser(description="M4 dynamic q_I virtual SEEG readout GIF")
    ap.add_argument("--run", action="store_true", help="run the two SNN sims (WITH LFPRecorder), cache, render")
    ap.add_argument("--render-only", action="store_true", help="render from the cache npz (no re-sim)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--T", type=float, default=3000.0,
                    help="ms display window (bounded band settles ~3-4s; runaway arm is held past --runaway-t-sim)")
    ap.add_argument("--runaway-t-sim", type=float, default=RUNAWAY_T_SIM,
                    help="ms actually simulated for the no-pool runaway arm before holding its saturated plateau")
    ap.add_argument("--target-frames", type=int, default=64)
    a = ap.parse_args()
    if not (a.run or a.render_only):
        print("REFUSED: pass --run (runs 2 SNN sims + renders) or --render-only (re-render from cache).")
        return 0
    os.chdir(ROOT)
    t0 = time.time()

    if a.render_only:
        S, arms, contacts, names, frame_steps = _load_cache(CACHE)
        print(f"loaded cache {CACHE.name}: T={S['p'].T} arms={[x['label'] for x in arms]} "
              f"contacts={len(names)} n_frames={len(frame_steps)}", flush=True)
    else:
        S = PP.build_substrate(a.seed)
        S["p"].T = float(a.T)
        mont = S["reg"]["montage_sheet"]
        contacts = np.asarray(mont.contacts, float)
        names = list(mont.names)
        rec = LFPRecorder(S["p"], S["net"]["pos"], S["net"]["labels"], sites=contacts)
        nsteps = int(round(a.T / DT))
        gif_dt_ms = max(20.0, round(a.T / a.target_frames))
        frame_steps = np.unique(np.clip((np.arange(0.0, a.T + 1e-9, gif_dt_ms) / DT).round().astype(int),
                                        0, nsteps - 1))
        frame_steps = np.unique(np.concatenate([frame_steps, [nsteps - 1]]))
        print(f"substrate E1146 {PP.MONTAGE} L={S['L']} N={S['N']} src={S['src_xy'].round(1)} "
              f"snk={S['snk_xy'].round(1)} | contacts={len(names)} shafts={sorted({H._shaft(n) for n in names})} "
              f"| T={a.T} nsteps={nsteps} n_frames={len(frame_steps)}", flush=True)
        arms = []
        for use_SG, aG in ((True, ALPHA_G_BOUNDED), (False, 0.0)):     # bounded first (full T), then runaway
            t_sim = None if use_SG else float(a.runaway_t_sim)         # runaway: sim to t_sim then hold plateau
            arm = _run_arm(S, rec, use_SG, aG, frame_steps, t_sim=t_sim)
            print(f"  {arm['label']:>12} aG={aG:>4}: runaway={arm['runaway_ms']} max_rate={arm['max_rate_hz']}Hz "
                  f"q_final=[{arm['q_min_final']},{arm['q_mean_final']}] S_G_max={arm['S_G_max']} "
                  f"t_sim={arm['t_sim_ms']} lfp_p98={np.percentile(arm['abslfp'], 98):.3g} ({arm['wall_s']}s)", flush=True)
            arms.append(arm)
        _save_cache(CACHE, S, arms, frame_steps)
        print(f"cached {CACHE.relative_to(ROOT)}", flush=True)

    gif, png, pdf = _render(S, arms, contacts, names, frame_steps, OUT_DIR)
    meta = _write_metadata(S, arms, names, OUT_DIR, gif, png, pdf, frame_steps, time.time() - t0)
    print(f"wrote {gif}")
    print(f"wrote {png}")
    print(json.dumps(meta["arms"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
