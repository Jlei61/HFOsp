"""Continuous M3A-v2.2 q_I build-up -> runaway GIF, WITH the global recovery h_G.

Companion to `plot_fig_m3a_v2_1_qigk_runaway_transition_gif.py`. Same continuous
single trajectory (same substrate / seed / multi-pulse drive / q_I carrier) so it
is apples-to-apples with the v2.1 build-up-to-runaway GIF, but here the M3A-v2.2
global inhibitory recovery scalar h_G(t) is turned ON (`use_hG=True`). The extra
panel shows h_G(t) and its smooth-AND globality trigger time-aligned above the
virtual-SEEG readout, so one can see whether/when the global brake engages
relative to the runaway onset, and what it does to the readout.

SELF-CONTAINED on purpose: it vendors the integration loop (copied from
`kick_probe.simulate_kick`, extended with a multi-pulse drive schedule) and only
imports git-tracked engine modules under `src/`. It does NOT import the (untracked)
v2.1 plotting script.

SCIENTIFIC STATUS (held): visual diagnostic, ONE continuous trajectory, h_G ON.
This is NOT a recovery/closed-loop claim, NOT a statistical sweep, and tonic /
multi-burst saturation is never an ictal-like event. If the readout goes quiet
after h_G rises, that is a global brake clamping the whole sheet, which is NOT the
same as a controlled, spatially-graded return to baseline -- the figure only lets
you see which of those it is, it does not assert recovery.

Across an eta_G ladder from 0 to 80 (i.e. up to >10x the 7 mV reset->threshold span) the
runaway is structurally unaffected (onset 771 ms, ~471 Hz at end, every value): a subtractive
global brake cannot reverse a saturated recurrent-excitation avalanche. The globality sensor
itself works -- h_G ~ 0 through the local axial events, ~0.94 only on the runaway.

Run (probe sensor scale first, then render the figure with defaults):
    python scripts/paper_figures/plot_fig_m3a_v2_2_hG_runaway_transition_gif.py --probe
    python scripts/paper_figures/plot_fig_m3a_v2_2_hG_runaway_transition_gif.py
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.patches import Ellipse, Patch, Polygon

ROOT = Path(__file__).resolve().parents[2]
ENG = ROOT / "src" / "snn_engine"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ENG))

import run_m3a_v2_step2_qI as S2  # noqa: E402
from connectivity import place_neurons  # noqa: E402
from connectivity_rot import build_connectivity_rot  # noqa: E402
from kick_probe import _flatten_by_source  # noqa: E402
from lfp import LFPRecorder  # noqa: E402
from params import Params, compute_nu_theta  # noqa: E402
from slow_field import SpatialSlowField, SpatialSlowFieldConfig, firing_rate_field  # noqa: E402
from src.sef_hfo_axial_intervention import intervention_vth_at_time  # noqa: E402
from src.sef_hfo_heterogeneity import sample_core_field  # noqa: E402
from src.topic4_m3a_v2_2_sensors import chi_G  # noqa: E402
from src.topic4_m3a_v2_phenotype import region_masks  # noqa: E402

FIG_NAME = "fig_m3a_v2_2_hG_runaway_transition"
STAGE5_FIGDATA = (
    ROOT / "results" / "topic4_sef_hfo" / "observation_layer"
    / "snn_cm_spontaneous" / "per_event" / "rep_s3_brakeoff.npz"
)
SUBJECT1146_FIGDATA = (
    ROOT / "results" / "topic4_sef_hfo" / "field_swap_subject_snn"
    / "figdata_epilepsiae_1146_twoend_equal_tsrc_s3.npz"
)

PULSE_A = "#f4b266"
PULSE_B = "#78a6d8"
AXIS_COL = "#a65f00"
HG_COL = "#9b1d64"        # h_G global recovery scalar (magenta-ish)
CHI_COL = "#c98aae"       # trigger chi_G (lighter)
QI_MEAN_COL = "#1f7a5a"   # mean q_I inhibitory resource (green)
QI_MIN_COL = "#7fc4a6"    # min q_I (lighter green)
GK_COL = "#b8860b"        # axial g_K fatigue field (dark goldenrod)
SHAFT_COLS = ["#e8743b", "#1f9e9e", "#7b5cb8", "#3b7a3b"]
TRACE_OFF = 1.35


@dataclass
class ProtocolConfig:
    substrate: str = "primary"
    seed: int = 1
    T: float = 1600.0
    pulse_start: float = 130.0
    pulse_interval: float = 135.0
    n_pulses: int = 9
    pulse_duration: float = 18.0
    kick_boost: float = 3.0
    r_kick: float = 0.30
    # ---- q_I carrier (identical to the v2.1 runaway GIF) ----
    q_min: float = 0.05
    k_q: float = 0.18
    sigma_q: float = 1.5
    tau_q: float = 5000.0
    tau_a: float = 20.0
    # ---- g_K fatigue field (off by default; the q_I figure turns it on to VISUALIZE the axial
    # fatigue). Coupling is OFF here (eta_K=0): g_K still builds from local E rate (the true fatigue
    # accumulation is shown) but does not feed back, so the approved q_I -> runaway trajectory is
    # preserved. NOTE: coupled at nominal eta_K=1 (gK_max=1), g_K builds EARLY during the small
    # events and suppresses the cores before ignition -> it PREVENTS the runaway (max ~24 Hz,
    # q_I barely depletes). That is the 'limit' role; it is a different figure, not this one. ----
    use_gK: bool = False
    k_K: float = 1.0
    sigma_K: float = 0.5
    eta_K: float = 0.0
    gK_max: float = 1.0
    tau_K: float = 5000.0
    # ---- h_G(t) global inhibitory recovery (M3A-v2.2 §B6), ON here ----
    # Defaults reproduce the rendered figure with no CLI args. eta_G=6.0 -> up to ~5.6 mV
    # subtractive E-membrane brake at h_G~0.94, ~80% of the reset(11)->threshold(18)=7 mV span
    # ("strong physiological" global inhibition). The M50/B50/Pi50 thresholds are the geometric
    # midpoints between the --probe local-event ceiling and the runaway floor (M:0.031/0.373,
    # B:0.508/0.592, Pi:0.300/0.997) so chi_G ~ 2e-4 through local events, ~0.52 during runaway.
    use_hG: bool = True
    eta_G: float = 6.0        # E-membrane coupling strength (strong physiological)
    k_G: float = 0.05         # build-rate knob (chi_G-gated); 0 -> no build
    tau_G: float = 600.0      # ms, h_G decay
    hG_max: float = 1.0
    M50: float = 0.11         # Hill half-trigger (probe geometric midpoint)
    B50: float = 0.55
    Pi50: float = 0.55
    # ---- two heterogeneous low-threshold cores ----
    core_mean: float = 16.5
    core_std: float = 1.0
    core_radius: float = 1.0
    # ---- Stage-4 spontaneous big-focus (single core, no kick; layout="stage4_patch") ----
    drive: float = 0.6         # nu_ext_ratio for the stage4_patch build (spontaneous background drive)
    L: float = 20.0            # sheet size for stage4_patch (subject1146/stage5 set L internally)
    gif_dt_ms: float = 20.0
    activity_window_ms: float = 10.0
    # ---- layout / top-panel / footer (presentation) ----
    layout: str = "stage5"     # "stage5" (default montage) | "subject1146" (E1146 geometry)
    top: str = "hG"            # "hG" (h_G global recovery) | "qI" (inhibitory resource trace)
    footer: bool = True        # bottom diagnostic text line
    fig_name: str = FIG_NAME


def _source_xy(S: dict, source: str) -> np.ndarray:
    if "layout" in S and "foci" in S["layout"]:
        foci = S["layout"]["foci"]
        if S["layout"].get("kind") == "stage4_patch" or len(foci) == 1:
            return np.asarray(foci[0], float)             # single focus: any source -> the one core
        return np.asarray(foci[0 if source == "tempA" else 1], float)
    sign = -1.0 if source == "tempA" else 1.0
    return np.asarray(S["center"], float) + sign * 0.6 * (float(S["L"]) / 2.0) * np.asarray(S["axis_unit"], float)


def _two_core_vth(S: dict, cfg: ProtocolConfig) -> np.ndarray:
    is_E = np.zeros(S["N"], bool)
    is_E[: S["NE"]] = True
    vth = np.full(S["N"], 18.0, float)
    core_radius = float(S.get("layout", {}).get("core_r", cfg.core_radius))
    for source, off in (("tempA", 7), ("tempB", 8)):
        cf = sample_core_field(
            S["net"]["pos"], is_E, _source_xy(S, source), core_radius,
            np.random.default_rng(int(cfg.seed) + off),
            core_mean=cfg.core_mean, core_std=cfg.core_std, base_mean=18.0,
        )
        core = cf["core_mask"]
        vth[core] = cf["vth"][core]
    return vth


def _contacts(S: dict):
    if "layout" in S and "contacts" in S["layout"]:
        return np.asarray(S["layout"]["contacts"], float), list(S["layout"]["names"])
    z = np.load(STAGE5_FIGDATA, allow_pickle=True)
    ref_L = float(z["L"])
    scale = float(S["L"]) / ref_L
    contacts = np.asarray(z["contacts"], float) * scale
    names = [str(x) for x in z["names"]]
    return contacts, names


def _subject1146_layout(target_L: float) -> dict:
    """E1146 foci + virtual-SEEG contacts from the subject-SNN figdata, scaled to target_L."""
    fd = np.load(SUBJECT1146_FIGDATA, allow_pickle=True)
    ref_L = float(fd["L"])
    scale = float(target_L) / ref_L
    foci = np.asarray(fd["foci"], float) * scale
    contacts = np.asarray(fd["contacts"], float) * scale
    names = [str(x) for x in fd["names"]]
    axis = foci[1] - foci[0]
    axis = axis / max(float(np.linalg.norm(axis)), 1e-9)
    return {
        "kind": "subject1146", "label": "E1146 geometry",
        "source": str(SUBJECT1146_FIGDATA.relative_to(ROOT)),
        "reference_L": ref_L, "scale": scale,
        "contacts": contacts, "names": names, "foci": foci,
        "core_r": float(fd["core_r"]) * scale,
        "axis_unit": axis, "theta_rad": float(np.arctan2(axis[1], axis[0])),
    }


def _out_dir(fig_name: str) -> Path:
    return ROOT / "results" / "paper-ready-figure" / fig_name / "figures"


def _pulse_schedule(cfg: ProtocolConfig):
    out = []
    for k in range(cfg.n_pulses):
        src = "tempA" if k % 2 == 0 else "tempB"
        t0 = cfg.pulse_start + k * cfg.pulse_interval
        out.append({"source": src, "t0": float(t0), "t1": float(t0 + cfg.pulse_duration)})
    return out


def _build_slow(S: dict, cfg: ProtocolConfig):
    scfg = SpatialSlowFieldConfig(
        use_qI=True, use_gK=cfg.use_gK, k_q=cfg.k_q, k_K=(cfg.k_K if cfg.use_gK else 0.0),
        sigma_q=cfg.sigma_q, sigma_K=cfg.sigma_K, eta_K=cfg.eta_K, gK_max=cfg.gK_max, tau_K=cfg.tau_K,
        q_min=cfg.q_min, q_init=1.0, tau_q=cfg.tau_q, tau_a=cfg.tau_a,
        use_hG=cfg.use_hG, eta_G=cfg.eta_G, k_G=cfg.k_G, tau_G=cfg.tau_G,
        hG_max=cfg.hG_max, M50=cfg.M50, B50=cfg.B50, Pi50=cfg.Pi50,
    )
    return SpatialSlowField(S["N"], 18.0, S["posE"], S["posI"], S["L"], cfg=scfg)


def _build(cfg: ProtocolConfig):
    if cfg.layout == "subject1146":
        return _build_subject1146(cfg)
    if cfg.layout == "stage4_patch":
        return _build_stage4_patch(cfg)
    S = S2.build(S2.SUBSTRATES[cfg.substrate], cfg.seed, T=cfg.T)
    S["masks"] = region_masks(S["L"], S2.N_GRID, S["center"], S["axis_unit"], S2.CORRIDOR_HW)
    return S


def _build_stage4_patch(cfg: "ProtocolConfig"):
    """ONE large isotropic excitable disk at the sheet centre (Stage-4 extended_patch),
    spontaneous (no kick). Built directly via sample_core_field (the runner's build_lesion_vth
    extended_patch path passes elongation/axis_unit, which the current sample_core_field signature
    does not accept -- a pre-existing runner drift). Substrate params are the CANONICAL Stage-4
    spontaneous runner values (source of truth: run_sef_hfo_snn_cm_spontaneous_readout.py:520-525 ->
    Params(g=3.6, ...), CLI defaults AR=2.0 / theta=45deg / density=100 / drive=0.6), NOT
    S2.SUBSTRATES (g=8.0/AR=4.0 = the v2 kick-driven substrate, a DIFFERENT regime)."""
    L = float(cfg.L)
    theta_rad = np.deg2rad(45.0)                      # canonical Stage-4 theta (runner CLI default)
    axis_unit = np.array([np.cos(theta_rad), np.sin(theta_rad)])
    center = np.array([L / 2.0, L / 2.0])
    p = Params(g=3.6, L=L, density=100.0, T=cfg.T, dt=0.1, nu_ext_ratio=cfg.drive, seed=cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    pos, labels, NE, NI = place_neurons(p, rng)
    net = build_connectivity_rot(p, pos, labels, NE, NI, rng, theta_EE=theta_rad, AR=2.0, verbose=False)
    pos = net["pos"]
    is_E = np.zeros(NE + NI, bool); is_E[:NE] = True
    cf = sample_core_field(pos, is_E, center, cfg.core_radius, np.random.default_rng(cfg.seed + 7),
                           core_mean=cfg.core_mean, core_std=cfg.core_std, base_mean=18.0)
    layout = {"kind": "stage4_patch", "label": "Stage-4 big focus", "foci": [center.tolist()],
              "core_r": float(cfg.core_radius), "axis_unit": axis_unit.tolist(), "L": L}
    S = dict(p=p, net=net, NE=NE, NI=NI, posE=pos[:NE], posI=pos[NE:], N=NE + NI, labels=labels,
             axis_unit=axis_unit, center=center, L=L, layout=layout,
             core_mask=cf["core_mask"], patch_vth=cf["vth"])
    S["masks"] = region_masks(L, S2.N_GRID, center, axis_unit, S2.CORRIDOR_HW)
    return S


def _build_subject1146(cfg: ProtocolConfig):
    """Same anisotropic E/I sheet but rotated/placed on the E1146 two-focus geometry (vendored from
    the v2.1 subject1146 runner; imports only tracked engine modules)."""
    L = 10.0
    layout = _subject1146_layout(L)
    sub = S2.SUBSTRATES[cfg.substrate]
    theta = float(layout["theta_rad"])
    axis_unit = np.asarray(layout["axis_unit"], float)
    center = np.asarray(layout["foci"], float).mean(axis=0)
    p = Params(g=sub["g"], L=L, density=100.0, T=cfg.T, dt=0.1, nu_ext_ratio=sub["nu"],
               seed=cfg.seed, w_EE=0.1575, l_EE=0.380, C_EE=800, l_EI=sub["l_EI"], C_EI=sub["C_EI"])
    rng = np.random.default_rng(cfg.seed)
    pos, labels, NE, NI = place_neurons(p, rng)
    net = build_connectivity_rot(p, pos, labels, NE, NI, rng, theta_EE=theta, AR=sub["AR"], verbose=False)
    pos = net["pos"]
    S = dict(p=p, net=net, NE=NE, NI=NI, posE=pos[:NE], posI=pos[NE:], N=NE + NI,
             labels=labels, axis_unit=axis_unit, center=center, L=L, layout=layout)
    S["masks"] = region_masks(L, S2.N_GRID, center, axis_unit, S2.CORRIDOR_HW)
    return S


def _simulate_continuous(S: dict, cfg: ProtocolConfig, *, record_gif: bool,
                         stim_target=None, stim_on=None, stim_off=None, clamp_level: float = 1e6,
                         vth=None, abort_on_runaway: bool = False, abort_check_every: int = 25):
    """Continuous q_I build-up loop. Optional `stim_target` (full-network bool mask) clamps its
    E cells' V_th to `clamp_level` during [stim_on, stim_off) via the parity-tested
    `intervention_vth_at_time` -- the ONLY behavioural addition. With stim_target=None (or
    stim_on=None) the threshold helper returns base_vth unchanged and NO rng draw is added, so the
    no-stim path stays byte-identical to the original loop (the stim-vs-no-stim arms therefore share
    a bit-identical trajectory until stim_on).

    `vth` (full-N float) overrides the default `_two_core_vth` (Stage-4 passes the single-big-core
    threshold field). `abort_on_runaway=True` breaks the loop at the first onset of the SHARED
    runaway criterion (`_smooth_rate` 20 ms + `_first_sustained` 120 Hz / 100 ms, 80%-rule),
    truncates every per-step array, and sets `res["aborted_ms"]` (float; None if never aborted) --
    the abort and post-hoc detectors agree by construction. Both new kwargs default to the original
    behaviour (vth=None -> _two_core_vth; abort off), so existing callers are byte-identical."""
    p = S["p"]
    net = S["net"]
    net["rng"] = np.random.default_rng(int(cfg.seed) + 3101)
    rng = net["rng"]
    NE, NI = net["NE"], net["NI"]
    N = NE + NI
    labels = net["labels"]
    pos = net["pos"]
    ampa = net["ampa_by_delay"]
    gaba = net["gaba_by_delay"]
    M = net["max_delay_steps"] + 1
    dt = p.dt
    nsteps = int(round(p.T / dt))

    decay_sE = np.exp(-dt / p.tau_r_AMPA)
    decay_IE = np.exp(-dt / p.tau_d_AMPA)
    decay_sI = np.exp(-dt / p.tau_r_GABA)
    decay_II = np.exp(-dt / p.tau_d_GABA)
    tau_m = np.where(labels == 0, p.tau_m_E, p.tau_m_I).astype(np.float64)
    decay_V = np.exp(-dt / tau_m)
    ref_steps = np.where(labels == 0, int(round(p.tau_ref_E / dt)), int(round(p.tau_ref_I / dt))).astype(np.int32)
    ext_incr = (tau_m / p.tau_r_AMPA) * np.where(labels == 0, p.J_ext_E, p.J_ext_I)

    ampa_bins = [d for d in range(M) if ampa[d].nnz > 0]
    gaba_bins = [d for d in range(M) if gaba[d].nnz > 0]
    if "ampa_flat" not in net:
        net["ampa_flat"] = _flatten_by_source(ampa, ampa_bins, NE)
        net["gaba_flat"] = _flatten_by_source(gaba, gaba_bins, NI)
    a_indptr, a_dst, a_dly, a_w = net["ampa_flat"]
    g_indptr, g_dst, g_dly, g_w = net["gaba_flat"]

    nu_theta, _, _ = compute_nu_theta(p)
    nu_sig_const = p.nu_ext_ratio * nu_theta
    sigma_n_inv_ms = p.sigma_n * 1e-3
    sigma_xi = sigma_n_inv_ms * np.sqrt(p.tau_n / 2.0)
    ou_a = np.exp(-dt / p.tau_n)
    ou_b = sigma_xi * np.sqrt(1.0 - ou_a * ou_a)
    xi = 0.0

    is_E = labels == 0
    pulses = _pulse_schedule(cfg)
    masks = {}
    for source in sorted({pl["source"] for pl in pulses}):     # only sources that actually fire
        center = _source_xy(S, source)
        masks[source] = is_E & (np.linalg.norm(pos - center, axis=1) <= cfg.r_kick)

    slow = _build_slow(S, cfg)
    vth = _two_core_vth(S, cfg) if vth is None else np.asarray(vth, float)
    contacts, names = _contacts(S)
    rec = LFPRecorder(p, pos, labels, sites=contacts) if record_gif else None

    V = np.full(N, p.V_reset, dtype=np.float64)
    ref = np.zeros(N, dtype=np.int32)
    s_E = np.zeros(N); I_E = np.zeros(N); s_I = np.zeros(N); I_I = np.zeros(N)
    ring_sE = np.zeros((M, N)); ring_sI = np.zeros((M, N))
    rate_E = np.zeros(nsteps)
    E_spk_bool = np.zeros((nsteps, NE), dtype=bool)
    lfp_trace = np.zeros((nsteps, len(contacts))) if rec is not None else None
    frame_steps = np.unique(np.clip((np.arange(0.0, p.T + 1e-9, cfg.gif_dt_ms) / dt).round().astype(int), 0, nsteps - 1))
    frame_set = set(int(x) for x in frame_steps)
    q_frames = []; q_frame_steps = []
    qI_min_trace = np.ones(nsteps)        # per-step min q_I over the sheet (most-depleted spot)
    axis_mask = S["masks"]["axis"]        # axial-corridor lattice mask (n_grid x n_grid)
    gK_axial_trace = np.zeros(nsteps)     # per-step mean g_K over the axial region
    stim_active = np.zeros(nsteps, dtype=bool)   # per-step: is the V_th clamp window open
    aborted_step = None                          # set to t if early-abort fires (shared criterion)

    t_wall = time.time()
    for t in range(nsteps):
        tm = t * dt
        xi = ou_a * xi + ou_b * rng.standard_normal()
        nu_now = max(nu_sig_const + xi, 0.0)

        s_E *= decay_sE; s_I *= decay_sI
        slot = t % M
        s_E += ring_sE[slot]; ring_sE[slot] = 0.0
        s_I += ring_sI[slot]; ring_sI[slot] = 0.0

        nu_vec = np.full(N, nu_now)
        for pulse in pulses:
            if pulse["t0"] <= tm < pulse["t1"]:
                nu_vec[masks[pulse["source"]]] += cfg.kick_boost
        ext = rng.poisson(nu_vec * dt, size=N).astype(np.float64)
        s_E += ext * ext_incr

        I_E = s_E + (I_E - s_E) * decay_IE
        I_I = s_I + (I_I - s_I) * decay_II
        if lfp_trace is not None:
            lfp_trace[t] = rec.sample(I_E, I_I)

        I_net = slow.apply_currents(I_E, I_I, labels)
        Vtmp = I_net + (V - I_net) * decay_V
        ref -= 1
        np.maximum(ref, 0, out=ref)
        free = ref == 0
        V = np.where(free, Vtmp, p.V_reset)
        vth_eff = intervention_vth_at_time(vth, stim_target, is_E, tm, stim_on, stim_off, clamp_level)
        spk = free & (V >= vth_eff)
        if stim_target is not None and stim_on is not None and stim_off is not None:
            stim_active[t] = stim_on <= tm < stim_off
        V[spk] = p.V_reset
        ref[spk] = ref_steps[spk]
        slow.step(spk, labels, dt)

        rate_E[t] = spk[:NE].sum()
        E_spk_bool[t] = spk[:NE]
        qI_min_trace[t] = float(slow.q_I.min())
        gK_axial_trace[t] = float(slow.g_K[axis_mask].mean())
        if record_gif and t in frame_set:
            q_frames.append(slow.q_I.copy())
            q_frame_steps.append(t)

        # Early-abort on the SHARED runaway criterion (rate_E holds spike COUNTS -> convert to Hz).
        if abort_on_runaway and t >= abort_check_every and (t % abort_check_every == 0):
            _rate_hz = rate_E[:t + 1] / NE / dt * 1e3
            if _first_sustained(_smooth_rate(_rate_hz, dt, 20.0), dt, 120.0, 100.0) is not None:
                aborted_step = t
                break

        if spk.any():
            spE = np.where(spk[:NE])[0]; spI = np.where(spk[NE:])[0]
            if spE.size:
                st = a_indptr[spE]; cnt = a_indptr[spE + 1] - st
                tot = int(cnt.sum())
                if tot:
                    idx = np.arange(tot) - np.repeat(np.cumsum(cnt) - cnt, cnt) + np.repeat(st, cnt)
                    np.add.at(ring_sE, ((t + a_dly[idx]) % M, a_dst[idx]), a_w[idx])
            if spI.size:
                st = g_indptr[spI]; cnt = g_indptr[spI + 1] - st
                tot = int(cnt.sum())
                if tot:
                    idx = np.arange(tot) - np.repeat(np.cumsum(cnt) - cnt, cnt) + np.repeat(st, cnt)
                    np.add.at(ring_sI, ((t + g_dly[idx]) % M, g_dst[idx]), g_w[idx])

    if aborted_step is not None:
        # executed steps 0..aborted_step (slow.step ran once per step -> its traces are already
        # length k); truncate every nsteps-preallocated per-step array to match.
        k = aborted_step + 1
        nsteps = k
        rate_E = rate_E[:k]
        E_spk_bool = E_spk_bool[:k]
        if lfp_trace is not None:
            lfp_trace = lfp_trace[:k]
        qI_min_trace = qI_min_trace[:k]
        gK_axial_trace = gK_axial_trace[:k]
        stim_active = stim_active[:k]

    return {
        "times": np.arange(nsteps) * dt,
        "rate_E": rate_E / NE / dt * 1e3,
        "E_spk_bool": E_spk_bool,
        "lfp_trace": lfp_trace,
        "contacts": contacts,
        "names": names,
        "pulses": pulses,
        "q_frames": q_frames,
        "q_frame_steps": q_frame_steps,
        "slow": slow,
        "trace_hG": np.asarray(slow.trace_hG, float),
        "trace_M": np.asarray(slow.trace_M, float),
        "trace_B": np.asarray(slow.trace_B, float),
        "trace_Pi": np.asarray(slow.trace_Pi, float),
        "trace_qI_mean": np.asarray(slow.trace_qI_mean, float),
        "trace_qI_min": qI_min_trace,
        "trace_gK_axial": gK_axial_trace,
        "stim_active": stim_active,
        "stim_window": (None if stim_target is None or stim_on is None else (float(stim_on), float(stim_off))),
        "aborted_ms": (aborted_step * dt) if aborted_step is not None else None,
        "wall_s": time.time() - t_wall,
    }


def _smooth_rate(rate, dt, win_ms=20.0):
    n = max(1, int(round(win_ms / dt)))
    return np.convolve(rate, np.ones(n) / n, mode="same")


def _first_sustained(rate, dt, threshold_hz=120.0, dur_ms=100.0):
    above = np.asarray(rate) >= threshold_hz
    n = max(1, int(round(dur_ms / dt)))
    if above.size < n:
        return None
    c = np.convolve(above.astype(float), np.ones(n), mode="valid")
    idx = np.flatnonzero(c >= 0.80 * n)
    return None if idx.size == 0 else float(idx[0] * dt)


def _chi_trace(res, cfg: ProtocolConfig):
    """Recompute the smooth-AND trigger chi_G(t) from the stored M/B/Pi sensor traces."""
    M = res["trace_M"]; B = res["trace_B"]; Pi = res["trace_Pi"]
    if M.size == 0:
        return np.zeros_like(res["times"])
    return np.array([chi_G(M[i], B[i], Pi[i], cfg.M50, cfg.B50, cfg.Pi50, 4, 4, 4) for i in range(M.size)])


def _activity_metrics(res, S, cfg: ProtocolConfig):
    dt = S["p"].dt
    rate_s = _smooth_rate(res["rate_E"], dt, 20.0)
    runaway_t = _first_sustained(rate_s, dt)
    pulses = res["pulses"]
    pulse_rows = []
    for pulse in pulses:
        lo = int(round(pulse["t0"] / dt))
        hi = int(round(min(pulse["t0"] + 85.0, S["p"].T) / dt))
        peak = float(np.max(rate_s[lo:hi])) if hi > lo else 0.0
        fired = res["E_spk_bool"][lo:hi].any(axis=0) if hi > lo else np.zeros(S["NE"], bool)
        pulse_rows.append({
            "source": pulse["source"], "t0": pulse["t0"], "peak_hz": round(peak, 2),
            "active_frac": round(float(fired.sum() / S["NE"]), 4),
            "before_runaway": bool(runaway_t is None or pulse["t0"] < runaway_t),
        })
    chi = _chi_trace(res, cfg)
    hG = res["trace_hG"]
    out = {
        "runaway_start_ms": runaway_t,
        "max_rate_hz": round(float(np.max(rate_s)), 2),
        "q_mean_final": round(float(res["slow"].q_I.mean()), 4),
        "q_min_final": round(float(res["slow"].q_I.min()), 4),
        "hG_max_reached": round(float(hG.max()), 4) if hG.size else 0.0,
        "hG_final": round(float(hG[-1]), 4) if hG.size else 0.0,
        "chi_max": round(float(chi.max()), 4) if chi.size else 0.0,
        "M_max": round(float(res["trace_M"].max()), 4) if res["trace_M"].size else 0.0,
        "B_max": round(float(res["trace_B"].max()), 4) if res["trace_B"].size else 0.0,
        "Pi_max": round(float(res["trace_Pi"].max()), 4) if res["trace_Pi"].size else 0.0,
        "pulse_rows": pulse_rows,
    }
    return out


def _shaft(name: str) -> str:
    m = re.match(r"[A-Za-z]+", str(name))
    return m.group(0) if m else str(name)


def _shaft_color(name: str, shafts: list[str]) -> str:
    return SHAFT_COLS[shafts.index(_shaft(name)) % len(SHAFT_COLS)]


def _draw_contacts(ax, contacts, names):
    shafts = sorted({_shaft(n) for n in names})
    for sh in shafts:
        color = _shaft_color(sh, shafts)
        marker = "s" if sh in {"B", "SCL"} else "o"
        idx = [i for i, n in enumerate(names) if _shaft(n) == sh]
        if not idx:
            continue
        pts = contacts[idx]
        ax.plot(pts[:, 0], pts[:, 1], color=color, lw=0.9, alpha=0.60, zorder=5)
        ax.scatter(pts[:, 0], pts[:, 1], s=32, marker=marker, fc="white", ec=color, lw=0.9, zorder=6)
        for j in sorted({idx[0], idx[-1]}):
            ax.text(contacts[j, 0], contacts[j, 1], names[j], fontsize=6.5, color=color,
                    fontweight="bold", ha="center", va="center", zorder=8,
                    path_effects=[pe.withStroke(linewidth=1.8, foreground="white")])


def _axis_range_patch(S):
    center = np.asarray(S["center"], float)
    u = np.asarray(S["axis_unit"], float)
    perp = np.array([-u[1], u[0]])
    foci = np.vstack([_source_xy(S, "tempA"), _source_xy(S, "tempB")])
    l_par = 0.380 * np.sqrt(float(S2.SUBSTRATES["primary"]["AR"]))
    l_perp = 0.380 / np.sqrt(float(S2.SUBSTRATES["primary"]["AR"]))
    half_w = max(0.42, 3.0 * l_perp)
    ext = max(0.80, 3.0 * l_par)
    proj = (foci - center) @ u
    a = center + u * (float(proj.min()) - ext)
    b = center + u * (float(proj.max()) + ext)
    return np.vstack([a + half_w * perp, b + half_w * perp, b - half_w * perp, a - half_w * perp])


def _axis_ellipse(S):
    """Anisotropic E->E connectivity footprint as an ellipse (elongated along the long axis).
    Returns (center_xy, width, height, angle_deg) for matplotlib.patches.Ellipse."""
    center = np.asarray(S["center"], float)
    u = np.asarray(S["axis_unit"], float)
    foci = np.vstack([_source_xy(S, "tempA"), _source_xy(S, "tempB")])
    l_par = 0.380 * np.sqrt(float(S2.SUBSTRATES["primary"]["AR"]))
    l_perp = 0.380 / np.sqrt(float(S2.SUBSTRATES["primary"]["AR"]))
    half_w = max(0.42, 3.0 * l_perp)
    ext = max(0.80, 3.0 * l_par)
    proj = (foci - center) @ u
    span = (float(proj.max()) - float(proj.min())) + 2.0 * ext
    ell_center = center + u * ((float(proj.min()) + float(proj.max())) / 2.0)
    angle_deg = float(np.degrees(np.arctan2(u[1], u[0])))
    return ell_center, span, 2.0 * half_w, angle_deg


def _style_spatial(ax, L):
    ax.set_xlim(0, L); ax.set_ylim(0, L); ax.set_aspect("equal")
    ax.set_xlabel("x (mm)", fontsize=7.0); ax.set_ylabel("y (mm)", fontsize=7.0)
    ax.tick_params(axis="both", labelsize=6.5, length=2.0)
    for sp in ax.spines.values():
        sp.set_linewidth(0.8); sp.set_color("0.25")


def _render_gif(S, res, metrics, cfg: ProtocolConfig, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    times = res["times"]
    dt = S["p"].dt
    frame_steps = res["q_frame_steps"]
    q_frames = res["q_frames"]
    contacts = res["contacts"]; names = res["names"]
    shafts = sorted({_shaft(n) for n in names})
    lfp = res["lfp_trace"].T
    base = np.median(lfp, axis=1, keepdims=True)
    scale = np.maximum(np.percentile(lfp, 98, axis=1, keepdims=True) - base, 1e-9)
    zlfp = (lfp - base) / scale
    trace_y = np.arange(len(names)) * TRACE_OFF
    hG = res["trace_hG"]; chi = _chi_trace(res, cfg)
    runaway = metrics["runaway_start_ms"]

    activity_fields = []; activity_vals = []
    for step in frame_steps:
        lo = max(0, step - int(round(cfg.activity_window_ms / dt)))
        fired = res["E_spk_bool"][lo: step + 1].any(axis=0)
        A = firing_rate_field(fired, S["posE"], S["L"], S2.N_GRID, sigma=0.5)
        activity_fields.append(A)
        if np.any(A > 0):
            activity_vals.append(A[A > 0])
    activity_vmax = max(1.0, float(np.percentile(np.concatenate(activity_vals), 98))) if activity_vals else 1.0

    frames = []
    for qi, (step, A) in enumerate(zip(frame_steps, activity_fields)):
        tm = float(times[step])
        tm_cursor = float(S["p"].T) if qi == len(frame_steps) - 1 else tm
        fig = plt.figure(figsize=(13.8, 4.9), facecolor="white")
        gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 2.15],
                              left=0.055, right=0.985, bottom=0.13, top=0.83, wspace=0.14)

        # --- col 0: permissivity (1 - q_I) with the E->E connectivity gradient as an ellipse ---
        ax0 = fig.add_subplot(gs[0, 0])
        perm = 1.0 - q_frames[qi]
        im0 = ax0.imshow(perm, origin="lower", extent=[0, S["L"], 0, S["L"]], cmap="plasma", vmin=0.0, vmax=1.0)
        ell_c, ell_w, ell_h, ell_ang = _axis_ellipse(S)
        ax0.add_patch(Ellipse(ell_c, ell_w, ell_h, angle=ell_ang, fc=PULSE_A, ec=AXIS_COL,
                              lw=1.2, alpha=0.22, zorder=4))
        for source, label in (("tempA", "A"), ("tempB", "B")):
            xy = _source_xy(S, source)
            ax0.add_patch(plt.Circle(xy, cfg.core_radius, fill=False, ec="crimson", lw=1.0, ls="--", zorder=7))
            ax0.text(xy[0], xy[1] + 0.44, label, fontsize=8, color="crimson", fontweight="bold",
                     ha="center", va="bottom", path_effects=[pe.withStroke(linewidth=1.8, foreground="white")])
        _draw_contacts(ax0, contacts, names)
        _style_spatial(ax0, S["L"])
        ax0.set_title("permissivity (1 - $q_I$)", fontsize=9.0, fontweight="bold", pad=4)
        cb0 = fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.02); cb0.ax.tick_params(labelsize=6.5)

        # --- col 1: real-time 2D SNN E activity ---
        ax1 = fig.add_subplot(gs[0, 1])
        im1 = ax1.imshow(A, origin="lower", extent=[0, S["L"], 0, S["L"]], cmap="viridis", vmin=0.0, vmax=activity_vmax)
        for pulse in res["pulses"]:
            if pulse["t0"] <= tm <= pulse["t1"] + 8.0:
                xy = _source_xy(S, pulse["source"])
                ax1.scatter([xy[0]], [xy[1]], marker="*", s=130, c="white", ec="black", lw=0.8, zorder=8)
        p0 = np.asarray(S["center"]) - S["axis_unit"] * 4.6
        p1 = np.asarray(S["center"]) + S["axis_unit"] * 4.6
        ax1.plot([p0[0], p1[0]], [p0[1], p1[1]], color="white", lw=1.2, alpha=0.9, zorder=5)
        _draw_contacts(ax1, contacts, names)
        _style_spatial(ax1, S["L"])
        ax1.set_title("2D SNN activity", fontsize=9.0, fontweight="bold", pad=4)
        cb1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.02); cb1.ax.tick_params(labelsize=6.5)

        # --- col 2 split: top time-trace (h_G OR q_I) | bottom continuous readout ---
        sub = gs[0, 2].subgridspec(2, 1, height_ratios=[1.0, 2.5], hspace=0.08)
        axg = fig.add_subplot(sub[0, 0])
        if cfg.top == "qI":
            axg.plot(times, res["trace_qI_mean"], color=QI_MEAN_COL, lw=1.6, zorder=4, label="mean $q_I$")
            axg.plot(times, res["trace_qI_min"], color=QI_MIN_COL, lw=1.1, ls="--", zorder=3, label="min $q_I$")
            axg.axhline(cfg.q_min, color="0.6", lw=0.8, ls=":", zorder=2, label="$q_{\\min}$ floor")
            if cfg.use_gK:
                axg.plot(times, res["trace_gK_axial"], color=GK_COL, lw=1.5, zorder=5, label="axial $g_K$ (fatigue)")
            axg.set_ylabel("$q_I,\\ g_K$", fontsize=8.0)
            axg.set_title("inhibitory resource $q_I$ depletes, axial $g_K$ fatigue builds $\\rightarrow$ runaway",
                          fontsize=8.0, fontweight="bold", pad=3)
            leg_loc, leg_anchor, leg_ncol = "upper right", (1.0, 1.04), 2
        else:
            axg.plot(times, hG, color=HG_COL, lw=1.5, zorder=4, label="$h_G$ (recovery)")
            axg.plot(times, chi, color=CHI_COL, lw=1.0, ls="--", zorder=3, label="$\\chi_G$ (trigger)")
            axg.set_ylabel("$h_G,\\ \\chi_G$", fontsize=7.6)
            axg.set_title("global inhibitory recovery $h_G(t)$ (ON)", fontsize=8.6, fontweight="bold", pad=3)
            leg_loc, leg_anchor, leg_ncol = "lower left", None, 2
        axg.axvline(tm_cursor, color="black", lw=1.2, alpha=0.9, zorder=7)
        if runaway is not None:
            axg.axvline(runaway, color="crimson", lw=1.0, ls="--", alpha=0.9, zorder=6)
        axg.set_xlim(0.0, float(S["p"].T)); axg.set_ylim(-0.03, 1.05)
        axg.tick_params(axis="x", labelbottom=False, length=2.0)
        axg.tick_params(axis="y", labelsize=6.6, length=2.0)
        axg.spines["top"].set_visible(False); axg.spines["right"].set_visible(False)
        axg.legend(frameon=False, fontsize=6.4, loc=leg_loc, bbox_to_anchor=leg_anchor,
                   ncol=leg_ncol, handlelength=1.3, columnspacing=0.8)

        ax2 = fig.add_subplot(sub[1, 0], sharex=axg)
        for pulse in res["pulses"]:
            if runaway is not None and pulse["t0"] >= runaway:
                continue
            color = PULSE_A if pulse["source"] == "tempA" else PULSE_B
            ax2.axvspan(pulse["t0"], pulse["t1"], color=color, alpha=0.22, lw=0, zorder=0)
        for i, nm in enumerate(names):
            ax2.plot(times, zlfp[i] + trace_y[i], color=_shaft_color(nm, shafts), lw=0.62, alpha=0.88, zorder=3)
        ax2.axvline(tm_cursor, color="black", lw=1.2, alpha=0.90, zorder=7)
        if runaway is not None:
            ax2.axvline(runaway, color="crimson", lw=1.0, ls="--", alpha=0.9, zorder=6)
        ax2.set_xlim(0.0, float(S["p"].T))
        ax2.set_yticks(trace_y); ax2.set_yticklabels(names, fontsize=6.8)
        for tick, nm in zip(ax2.get_yticklabels(), names):
            tick.set_color(_shaft_color(nm, shafts))
        ax2.tick_params(axis="x", labelsize=7.0, length=2.5); ax2.tick_params(axis="y", labelsize=6.8, length=2.0)
        ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
        ax2.set_ylabel("contact", fontsize=7.8); ax2.set_xlabel("time (ms)", fontsize=7.8)
        ax2.legend(handles=[
            Patch(facecolor=PULSE_A, alpha=0.40, edgecolor="none", label="pre-runaway tempA drive"),
            Patch(facecolor=PULSE_B, alpha=0.40, edgecolor="none", label="pre-runaway tempB drive"),
        ], frameon=False, fontsize=6.8, loc="upper right", bbox_to_anchor=(1.0, 1.10), ncol=2,
            handlelength=1.4, columnspacing=0.8)

        geo = S.get("layout", {}).get("label", "Stage5 geometry")
        title = (f"q_I build-up $\\rightarrow$ runaway ({geo}) | t={tm_cursor:.0f} ms" if cfg.top == "qI"
                 else f"q_I build-up -> runaway WITH global recovery h_G ({geo}) | t={tm_cursor:.0f} ms")
        fig.text(0.50, 0.93, title, fontsize=10.0, fontweight="bold", ha="center")
        if cfg.footer:
            foot = (f"runaway_start={runaway} ms | q_mean_final={metrics['q_mean_final']} | "
                    f"q_min_final={metrics['q_min_final']}" if cfg.top == "qI"
                    else f"runaway_start={runaway} ms | h_G_max={metrics['hG_max_reached']} | "
                    f"chi_max={metrics['chi_max']} | eta_G={cfg.eta_G} k_G={cfg.k_G} tau_G={cfg.tau_G} ms")
            fig.text(0.50, 0.025, foot, fontsize=7.3, ha="center", color="0.25")
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
        plt.close(fig)

    base = "qI_runaway_transition" if cfg.top == "qI" else "hG_runaway_transition"
    gif = out_dir / f"{base}.gif"
    frames.extend([frames[-1]] * 8)
    imageio.mimsave(gif, frames, duration=0.11, loop=0)
    png = out_dir / f"{base}_final.png"
    imageio.imwrite(png, frames[-1])
    return gif, png, activity_vmax


def _write_readme(metrics, cfg: ProtocolConfig, out_dir: Path):
    if cfg.top == "qI":
        _write_readme_qI(metrics, cfg, out_dir)
        return
    text = f"""# M3A-v2.2 q_I build-up -> runaway WITH global recovery h_G — GIF

### hG_runaway_transition.gif

这张 GIF 是**连续单轨迹 visual diagnostic**，不是统计 sweep。它和 v2.1 的
`fig_m3a_v2_1_qigk_runaway_transition` 用**同一条轨迹**（同 substrate / seed / 多脉冲驱动 / `q_I` 载体），
唯一区别是把 M3A-v2.2 的全局抑制恢复标量 `h_G(t)` **打开**（`use_hG=True`）。

电极沿用 Fig5/Stage5 的 A0-A5/B0-B5 双 shaft montage，按当前 L 等比例缩放；SEEG readout 是同一条
连续 trace，没有拼接 gap。

**布局**：一行三列——`1-q_I` permissivity map | 实时 2D SNN E 活动 |（上）`h_G(t)` 全局恢复标量
与触发器 `χ_G(t)` ／（下）连续 SEEG readout（共享时间轴，纵向对齐）。

**关注点**：
1. 前几次 tempA/tempB 局部轴向事件时，`χ_G`（globality 触发器）应**接近 0**、`h_G` 基本不升——
   说明全局恢复**没有**误伤小的局部事件（它是全局性传感器：单点热斑 → 参与度低 → 不触发）。
2. 当 runaway 把活动铺成全场时，`χ_G` 抬起、`h_G` 随之上升——看 `h_G` 上升相对 runaway onset 的**时序**。
3. 看 readout：`h_G` 升起后 readout 是**安静下来**还是**继续高幅振荡**。

**这条轨迹里实际看到的**：`h_G` 确实只在 runaway 时升起（局部轴向事件期间 `χ_G≈0`、`h_G≈0`，
全局性传感器没误伤它们），但 runaway **没有被逆转**——`h_G` 升到 0.94 之后 readout 仍持续高幅饱和。
配套 `eta_G` 阶梯（0→80，即把膜耦合一路加到 reset(11)→threshold(18)=7 mV 跨度的 >10 倍）**全程对
runaway 无效**（onset 恒为 771 ms、末段 ~471 Hz）：一个**减法式**的全局刹车在结构上拉不回一个已经
饱和的 recurrent-excitation 雪崩——瓶颈在 recurrent E→E 衬底，不在恢复变量。

**红线（务必照读）**：
- readout 在 `h_G` 升起后变安静 ≠ "受控、空间分级地回到基线"。那可能只是全局抑制把**整片** clamp 死
  （全或无的另一侧），**不是** recovery。这张图只让你**看见**是哪一种，**不**主张 recovery / 闭环成立。
- tonic / 多 burst 饱和**永远不是** ictal-like 事件。
- 不把"`h_G` 升起后变安静"归因为"慢变量解决了发作"。

### hG_runaway_transition_final.png

GIF 末帧静态快照，快速核对 runaway 末态、`h_G` 终值、readout 是否非空。

runaway_start_ms: `{metrics['runaway_start_ms']}`; h_G_max: `{metrics['hG_max_reached']}`;
chi_max: `{metrics['chi_max']}`; eta_G: `{cfg.eta_G}`; k_G: `{cfg.k_G}`; tau_G: `{cfg.tau_G}` ms.
"""
    (out_dir / "README.md").write_text(text)


def _write_readme_qI(metrics, cfg: ProtocolConfig, out_dir: Path):
    geo = ("E1146 真实电极几何（两灶 + 触点来自 `figdata_epilepsiae_1146_twoend_equal_tsrc_s3.npz`，"
           "等比例缩放到 L=10）" if cfg.layout == "subject1146"
           else "Fig5/Stage5 A0-A5/B0-B5 双 shaft montage，按当前 L 等比例缩放")
    text = f"""# M3A-v2.2 q_I build-up -> runaway（{('E1146 几何' if cfg.layout=='subject1146' else 'Stage5')}）— GIF

### qI_runaway_transition.gif

这张 GIF 是**连续单轨迹 visual diagnostic**，不是统计 sweep。和 v2.1 的
`fig_m3a_v2_1_qigk_runaway_transition_epilepsiae_1146` **同一条轨迹**（同 substrate / seed / 多脉冲驱动 /
`q_I` 载体）。电极用{geo}；SEEG readout 是同一条连续 trace，没有拼接 gap。

**布局**：一行三列——左=`1-q_I` permissivity map（E→E 连接梯度画成**椭圆**=各向异性长轴连接足迹）|
中=实时 2D SNN E 活动 |（上）抑制资源 `q_I(t)`（mean + min）+ **轴向区域 `g_K` 疲劳场**（共享时间轴，
legend 在右上角）／（下）连续 SEEG readout。

全局恢复 `h_G` 在本图里**关掉**（`use_hG=False`，已另证它对这套衬底的 runaway 结构性无效）。`g_K` 疲劳场
**打开画出来**，但本图把它的膜耦合**关掉**（`eta_K=0`）——`g_K` 仍按局部 E 率累积（画的是**真实疲劳累积**），
只是不反馈，所以保住了已确认的 `q_I→runaway` 轨迹。这张图看 `q_I`（push/去抑制）耗竭 + 轴向 `g_K`
（limit/疲劳）累积，沿着同一条 runaway transition。

**重要（机制岔路）**：若把 `g_K` **真耦合**到 nominal（`eta_K=1`、`gK_max=1`），它会在**小事件期**就把核压住、
**直接阻止 runaway**（实测 max~24 Hz、`q_I` 几乎不耗竭、无 runaway）——这是 `g_K`=limit 成功限流的角色，
是**另一张图**，不是本张"看 runaway transition"的图。

**关注点**：
1. 先看左图 / 顶图：每次 tempA/tempB 局部沿轴事件后，`q_I`（尤其 **min q_I**=轴向走廊那一点）一步步往
   下掉（去抑制累积），同时**轴向 `g_K` 疲劳**一步步往上爬。
2. 看 readout：前几次是**局部、短暂**的小事件；当 `q_I` 掉到地板附近，事件**铺成全场 runaway**（持续高幅振荡）。
3. runaway onset（红虚线）相对 `q_I` 耗竭 / `g_K` 累积曲线的时序——是"资源耗尽 + 疲劳累积→失控"，
   不是外部把驱动调大。注意 `g_K` 这点疲劳**拦不住** runaway（与 eta_G 阶梯同结论：减法刹车拉不回雪崩）。

**红线**：runaway / tonic 饱和**不是** ictal-like 事件；这张图是"`q_I` 去抑制载体把全或无事件推到失控、
`g_K` 疲劳沿途累积"的可视化诊断，**不**主张任何发作机制 / recovery。

### qI_runaway_transition_final.png

GIF 末帧静态快照，核对 runaway 末态、`q_I` 终值、readout 是否非空。

runaway_start_ms: `{metrics['runaway_start_ms']}`; q_mean_final: `{metrics['q_mean_final']}`;
q_min_final: `{metrics['q_min_final']}`; max_rate_hz: `{metrics['max_rate_hz']}`.
"""
    (out_dir / "README.md").write_text(text)


def run_one(cfg: ProtocolConfig, *, record_gif: bool):
    S = _build(cfg)
    res = _simulate_continuous(S, cfg, record_gif=record_gif)
    metrics = _activity_metrics(res, S, cfg)
    return S, res, metrics


def probe():
    """eta_G=0 (no feedback): h_G builds but does NOT perturb dynamics -> the trajectory
    matches v2.1 while we read the M/B/Pi sensor scale during local events vs runaway."""
    cfg = ProtocolConfig(use_hG=True, eta_G=0.0, k_G=0.05)
    S, res, metrics = run_one(cfg, record_gif=False)
    dt = S["p"].dt
    rate_s = _smooth_rate(res["rate_E"], dt, 20.0)
    runaway = _first_sustained(rate_s, dt)
    M, B, Pi = res["trace_M"], res["trace_B"], res["trace_Pi"]
    t = res["times"]
    if runaway is not None:
        pre = t < (runaway - 20.0)
        post = t >= runaway
    else:
        pre = np.ones_like(t, bool); post = np.zeros_like(t, bool)
    def stat(arr, m):
        a = arr[m]
        return dict(max=round(float(a.max()), 5), p99=round(float(np.percentile(a, 99)), 5),
                    median=round(float(np.median(a)), 5)) if a.size else {}
    report = {
        "runaway_start_ms": runaway,
        "max_rate_hz": metrics["max_rate_hz"],
        "pre_runaway": {"M": stat(M, pre), "B": stat(B, pre), "Pi": stat(Pi, pre)},
        "runaway": {"M": stat(M, post), "B": stat(B, post), "Pi": stat(Pi, post)},
        "note": "eta_G=0 probe: set M50/B50/Pi50 between pre-runaway p99 and runaway median so chi_G "
                "separates local events (low) from runaway (high); set eta_G on the runaway drive scale.",
    }
    print(json.dumps(report, indent=2))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="eta_G=0 sensor-scale probe (prints M/B/Pi pre vs runaway)")
    ap.add_argument("--eta-G", type=float, default=ProtocolConfig.eta_G)
    ap.add_argument("--k-G", type=float, default=ProtocolConfig.k_G)
    ap.add_argument("--tau-G", type=float, default=ProtocolConfig.tau_G)
    ap.add_argument("--M50", type=float, default=ProtocolConfig.M50)
    ap.add_argument("--B50", type=float, default=ProtocolConfig.B50)
    ap.add_argument("--Pi50", type=float, default=ProtocolConfig.Pi50)
    ap.add_argument("--T", type=float, default=ProtocolConfig.T)
    ap.add_argument("--layout", choices=["stage5", "subject1146"], default="stage5",
                    help="electrode/focus geometry: default Stage5 montage or E1146 subject geometry")
    ap.add_argument("--top", choices=["hG", "qI"], default="hG",
                    help="top time-trace panel: h_G global recovery (default) or q_I inhibitory resource")
    ap.add_argument("--no-footer", action="store_true", help="omit the bottom diagnostic text line")
    ap.add_argument("--no-gk", action="store_true", help="for --top qI: do NOT overlay the axial g_K fatigue field")
    ap.add_argument("--fig-name", default=None)
    args = ap.parse_args()
    os.chdir(ROOT)

    if args.probe:
        probe()
        return 0

    # top=qI isolates the q_I carrier (h_G OFF; we proved it inert across the eta_G ladder) and overlays
    # the axial g_K fatigue field. top=hG is the h_G figure with the q_I carrier ON, no g_K.
    use_hG = (args.top == "hG")
    use_gK = (args.top == "qI") and not args.no_gk
    if args.fig_name:
        fig_name = args.fig_name
    else:
        stem = "fig_m3a_v2_2_qI_runaway_transition" if args.top == "qI" else FIG_NAME
        fig_name = f"{stem}_epilepsiae_1146" if args.layout == "subject1146" else stem

    cfg = ProtocolConfig(
        use_hG=use_hG, use_gK=use_gK, eta_G=float(args.eta_G), k_G=float(args.k_G), tau_G=float(args.tau_G),
        M50=float(args.M50), B50=float(args.B50), Pi50=float(args.Pi50), T=float(args.T),
        layout=str(args.layout), top=str(args.top), footer=(not args.no_footer), fig_name=str(fig_name),
    )
    S, res, metrics = run_one(cfg, record_gif=True)
    out_dir = _out_dir(cfg.fig_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    gif, png, activity_vmax = _render_gif(S, res, metrics, cfg, out_dir)
    base = gif.stem
    if cfg.top == "qI":
        status = ("visual diagnostic, ONE continuous trajectory, q_I carrier build-up + axial g_K fatigue "
                  "-> runaway (h_G OFF); NOT a seizure-mechanism/recovery claim, NOT a statistical sweep; "
                  "runaway/tonic is never ictal-like")
        notes = [
            "Same substrate/seed/multi-pulse drive/q_I carrier as the v2.1 runaway GIF; top panel shows q_I(t) "
            "(mean+min) and the axial-region g_K fatigue field; leftmost panel draws the E->E gradient as an ellipse.",
            "h_G is OFF (use_hG=False): h_G was separately shown structurally inert on this substrate (eta_G ladder 0..80).",
            "g_K fatigue is built and VISUALIZED but its membrane coupling is OFF (eta_K=0): g_K accumulates from "
            "local E rate (true fatigue shown) without feeding back, so the approved q_I -> runaway trajectory is preserved.",
            "FORK: coupled at nominal eta_K=1 (gK_max=1), g_K builds early during the small events and suppresses the "
            "cores before ignition -> it PREVENTS the runaway (max ~24 Hz, q_I barely depletes). That is the 'limit' "
            "role and a different figure, not this runaway-transition one.",
            "min q_I (axial corridor) depletes first; mean q_I drops as the runaway spreads sheet-wide; axial g_K rises in step.",
            "The local runner copies the kick_probe integration loop and extends localized drive to a multi-pulse "
            "schedule; engine core untouched.",
        ]
    else:
        status = ("visual diagnostic, ONE continuous trajectory, h_G global recovery ON; "
                  "NOT a recovery/closed-loop claim, NOT a statistical sweep; tonic/multiburst is never ictal-like")
        notes = [
            "Same substrate/seed/multi-pulse drive/q_I carrier as the v2.1 runaway GIF; only use_hG flips to True.",
            "h_G(t) panel shows the global recovery scalar and its smooth-AND globality trigger chi_G(t), "
            "time-aligned above the continuous virtual-SEEG readout.",
            "h_G coupling is E-only (-eta_G*h_G); the M/B/Pi sensor is a globality detector -> it should leave "
            "small local axial events alone and only engage on whole-sheet runaway.",
            "A quiet readout after h_G rises is a global brake clamping the sheet, NOT a controlled return; "
            "this GIF only visualizes which it is and asserts neither recovery nor closed-loop.",
            "eta_G ladder (0,2,4,6,8,12,20,40,80) is structurally inert: runaway onset stays 771 ms and "
            "end-rate ~471 Hz at every value, i.e. up to >10x the 7 mV reset->threshold span. A subtractive "
            "global brake cannot reverse a saturated recurrent-excitation avalanche; the bottleneck is the "
            "recurrent E->E substrate, not the recovery variable (consistent with the M3A-v2.2 NEGATIVE verdict).",
            "The local runner copies the kick_probe integration loop and extends localized drive to a multi-pulse "
            "schedule; engine core untouched (byte-parity preserved with use_hG=False).",
        ]
    meta = {
        "figure": cfg.fig_name,
        "status": status,
        "companion_baseline": "fig_m3a_v2_1_qigk_runaway_transition (same trajectory)",
        "geometry": S.get("layout", {}).get("label", "Stage5 geometry"),
        "config": asdict(cfg),
        "metrics": metrics,
        "outputs": {"gif": str(gif.relative_to(ROOT)), "final_png": str(png.relative_to(ROOT))},
        "colorbars": {"permissivity_vmin": 0.0, "permissivity_vmax": 1.0,
                      "activity_vmin": 0.0, "activity_vmax": activity_vmax},
        "notes": notes,
    }
    (out_dir / f"{base}_metadata.json").write_text(json.dumps(meta, indent=2))
    _write_readme(metrics, cfg, out_dir)
    print(f"wrote {gif}")
    print(f"wrote {png}")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
