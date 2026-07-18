"""Continuous M3A-v2.1 q_I build-up -> runaway GIF diagnostic.

This is a plotting/diagnostic runner, not a new engine feature. The core
integration loop is copied locally from `kick_probe.simulate_kick` and extended
only with a multi-pulse localized drive schedule so one continuous trajectory can
show:

    repeated local axial events -> q_I/permissivity build-up -> runaway

Recovery/termination is intentionally excluded. The GIF is a visual diagnostic;
the metadata records the exact parameter combination and transition checks.
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
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Patch

ROOT = Path(__file__).resolve().parents[2]
ENG = ROOT / "src" / "snn_engine"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ENG))

import run_m3a_v2_step2_qI as S2  # noqa: E402
from connectivity import place_neurons  # noqa: E402
from connectivity_rot import build_connectivity_rot  # noqa: E402
from kick_probe import _flatten_by_source, membrane_step  # noqa: E402
from lfp import LFPRecorder  # noqa: E402
from params import Params, compute_nu_theta  # noqa: E402
from slow_field import SpatialSlowField, SpatialSlowFieldConfig, firing_rate_field  # noqa: E402
from src.sef_hfo_heterogeneity import sample_core_field  # noqa: E402
from src.topic4_m3a_v2_phenotype import make_field_grid_xy, region_masks  # noqa: E402


FIG_NAME = "fig_m3a_v2_1_qigk_runaway_transition"
STAGE5_FIGDATA = (
    ROOT
    / "results"
    / "topic4_sef_hfo"
    / "observation_layer"
    / "snn_cm_spontaneous"
    / "per_event"
    / "rep_s3_brakeoff.npz"
)
SUBJECT1146_FIGDATA = (
    ROOT
    / "results"
    / "topic4_sef_hfo"
    / "field_swap_subject_snn"
    / "figdata_epilepsiae_1146_twoend_equal_tsrc_s3.npz"
)

SHAFT_A = "#e8743b"
SHAFT_B = "#1f9e9e"
PULSE_A = "#f4b266"
PULSE_B = "#78a6d8"
AXIS_COL = "#a65f00"
SHAFT_COLS = ["#e8743b", "#1f9e9e", "#7b5cb8", "#3b7a3b"]
TRACE_OFF = 1.35
SHADE_PAD_MS = 18.0
PULSE_READOUT_MS = 85.0


@dataclass
class ProtocolConfig:
    substrate: str = "primary"
    seed: int = 1
    T: float = 1600.0
    pulse_start: float = 130.0
    pulse_interval: float = 135.0
    n_pulses: int = 9
    pulse_duration: float = 18.0
    pulse_first_source: str = "tempA"
    kick_boost: float = 3.0
    r_kick: float = 0.30
    q_min: float = 0.05
    k_q: float = 0.18
    sigma_q: float = 1.5
    tau_q: float = 5000.0
    tau_a: float = 20.0
    core_mean: float = 16.5
    core_std: float = 1.0
    core_radius: float = 1.0
    core_radius_scale: float = 1.0
    core_transverse_scale: float | None = None
    ee_ar_override: float | None = None
    gif_dt_ms: float = 20.0
    activity_window_ms: float = 10.0
    layout: str = "stage5"
    fig_name: str = FIG_NAME


def _axis(theta_deg: float = 45.0):
    th = np.deg2rad(theta_deg)
    u = np.array([np.cos(th), np.sin(th)])
    p = np.array([-u[1], u[0]])
    return u, p


def _source_xy(S: dict, source: str) -> np.ndarray:
    if "layout" in S and "foci" in S["layout"]:
        return np.asarray(S["layout"]["foci"][0 if source == "tempA" else 1], float)
    sign = -1.0 if source == "tempA" else 1.0
    return np.asarray(S["center"], float) + sign * 0.6 * (float(S["L"]) / 2.0) * np.asarray(S["axis_unit"], float)


def _two_core_vth(S: dict, cfg: ProtocolConfig) -> np.ndarray:
    is_E = np.zeros(S["N"], bool)
    is_E[: S["NE"]] = True
    vth = np.full(S["N"], 18.0, float)
    core_parallel, core_transverse = _effective_core_radii(S, cfg)
    axis = np.asarray(S["axis_unit"], float)
    axis /= max(float(np.linalg.norm(axis)), 1e-12)
    transverse = np.asarray([-axis[1], axis[0]], float)
    for source, off in (("tempA", 7), ("tempB", 8)):
        center = _source_xy(S, source)
        cf = sample_core_field(
            S["net"]["pos"],
            is_E,
            center,
            max(core_parallel, core_transverse),
            np.random.default_rng(int(cfg.seed) + off),
            core_mean=cfg.core_mean,
            core_std=cfg.core_std,
            base_mean=18.0,
        )
        delta = np.asarray(S["net"]["pos"], float) - center[None, :]
        ellipse = (
            (delta @ axis / core_parallel) ** 2
            + (delta @ transverse / core_transverse) ** 2
            <= 1.0
        )
        core = is_E & ellipse
        vth[core] = cf["vth"][core]
    return vth


def _effective_core_radius(S: dict, cfg: ProtocolConfig) -> float:
    """Low-threshold core radius in the current simulation's millimetres."""
    return _effective_core_radii(S, cfg)[0]


def _effective_core_radii(S: dict, cfg: ProtocolConfig) -> tuple[float, float]:
    """Parallel/transverse low-threshold radii in simulation millimetres."""
    base_radius = float(S.get("layout", {}).get("core_r", cfg.core_radius))
    parallel_scale = float(cfg.core_radius_scale)
    transverse_scale = (
        parallel_scale
        if cfg.core_transverse_scale is None
        else float(cfg.core_transverse_scale)
    )
    if not np.isfinite(parallel_scale) or parallel_scale <= 0.0:
        raise ValueError(f"core_radius_scale must be positive, got {parallel_scale}")
    if not np.isfinite(transverse_scale) or transverse_scale <= 0.0:
        raise ValueError(
            f"core_transverse_scale must be positive, got {transverse_scale}"
        )
    return base_radius * parallel_scale, base_radius * transverse_scale


def _contacts(S: dict):
    """Stage5/Fig5 virtual SEEG placement, scaled onto the current sheet.

    The reference Fig5 artifact uses 12 contacts (A0-A5, B0-B5) on an L=20 mm
    sheet with 4 mm pitch. M3A-v2.1 runs on L=10 mm, so the coordinates are
    scaled by current_L / reference_L while preserving shaft geometry and names.
    """
    if "layout" in S and "contacts" in S["layout"]:
        return np.asarray(S["layout"]["contacts"], float), list(S["layout"]["names"])
    z = np.load(STAGE5_FIGDATA, allow_pickle=True)
    ref_L = float(z["L"])
    scale = float(S["L"]) / ref_L
    contacts = np.asarray(z["contacts"], float) * scale
    names = [str(x) for x in z["names"]]
    return contacts, names


def _out_dir(fig_name: str) -> Path:
    return ROOT / "results" / "paper-ready-figure" / fig_name / "figures"


def _jsonable_layout(layout):
    if layout is None:
        return None
    out = {}
    for key, val in layout.items():
        if isinstance(val, np.ndarray):
            out[key] = val.tolist()
        elif isinstance(val, (np.floating, np.integer)):
            out[key] = val.item()
        else:
            out[key] = val
    return out


def _subject1146_layout(target_L: float) -> dict:
    fd = np.load(SUBJECT1146_FIGDATA, allow_pickle=True)
    ref_L = float(fd["L"])
    scale = float(target_L) / ref_L
    foci = np.asarray(fd["foci"], float) * scale
    contacts = np.asarray(fd["contacts"], float) * scale
    names = [str(x) for x in fd["names"]]
    axis = foci[1] - foci[0]
    axis = axis / max(float(np.linalg.norm(axis)), 1e-9)
    theta = float(np.arctan2(axis[1], axis[0]))
    try:
        source = str(SUBJECT1146_FIGDATA.relative_to(ROOT))
    except ValueError:
        # Results-light worktrees may explicitly reuse the accepted geometry
        # from the canonical checkout; retain its absolute provenance.
        source = str(SUBJECT1146_FIGDATA)
    return {
        "kind": "subject1146",
        "label": "E1146 geometry",
        "source": source,
        "reference_L": ref_L,
        "scale": scale,
        "contacts": contacts,
        "names": names,
        "foci": foci,
        "core_r": float(fd["core_r"]) * scale,
        "axis_unit": axis,
        "theta_rad": theta,
        "theta_deg": float(np.degrees(theta)),
    }


def _pulse_schedule(cfg: ProtocolConfig):
    if cfg.pulse_first_source not in {"tempA", "tempB"}:
        raise ValueError(
            f"pulse_first_source must be tempA or tempB, got {cfg.pulse_first_source}"
        )
    second = "tempB" if cfg.pulse_first_source == "tempA" else "tempA"
    out = []
    for k in range(cfg.n_pulses):
        src = cfg.pulse_first_source if k % 2 == 0 else second
        t0 = cfg.pulse_start + k * cfg.pulse_interval
        out.append({"source": src, "t0": float(t0), "t1": float(t0 + cfg.pulse_duration)})
    return out


def _build_slow(S: dict, cfg: ProtocolConfig):
    scfg = SpatialSlowFieldConfig(
        use_qI=True,
        use_gK=False,
        k_q=cfg.k_q,
        k_K=0.0,
        sigma_q=cfg.sigma_q,
        sigma_K=0.5,
        q_min=cfg.q_min,
        q_init=1.0,
        tau_q=cfg.tau_q,
        tau_a=cfg.tau_a,
    )
    return SpatialSlowField(S["N"], 18.0, S["posE"], S["posI"], S["L"], cfg=scfg)


def _simulate_continuous(S: dict, cfg: ProtocolConfig, *, record_gif: bool):
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
    for source in ("tempA", "tempB"):
        center = _source_xy(S, source)
        masks[source] = is_E & (np.linalg.norm(pos - center, axis=1) <= cfg.r_kick)

    slow = _build_slow(S, cfg)
    vth = _two_core_vth(S, cfg)
    contacts, names = _contacts(S)
    rec = LFPRecorder(p, pos, labels, sites=contacts) if record_gif else None

    V = np.full(N, p.V_reset, dtype=np.float64)
    ref = np.zeros(N, dtype=np.int32)
    s_E = np.zeros(N)
    I_E = np.zeros(N)
    s_I = np.zeros(N)
    I_I = np.zeros(N)
    ring_sE = np.zeros((M, N))
    ring_sI = np.zeros((M, N))
    rate_E = np.zeros(nsteps)
    rate_I = np.zeros(nsteps)
    E_spk_bool = np.zeros((nsteps, NE), dtype=bool)
    lfp_trace = np.zeros((nsteps, len(contacts))) if rec is not None else None
    frame_steps = np.unique(np.clip((np.arange(0.0, p.T + 1e-9, cfg.gif_dt_ms) / dt).round().astype(int), 0, nsteps - 1))
    frame_set = set(int(x) for x in frame_steps)
    q_frames = []
    q_frame_steps = []

    t_wall = time.time()
    for t in range(nsteps):
        tm = t * dt
        xi = ou_a * xi + ou_b * rng.standard_normal()
        nu_now = max(nu_sig_const + xi, 0.0)

        s_E *= decay_sE
        s_I *= decay_sI
        slot = t % M
        s_E += ring_sE[slot]
        ring_sE[slot] = 0.0
        s_I += ring_sI[slot]
        ring_sI[slot] = 0.0

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
        spk = free & (V >= vth)
        V[spk] = p.V_reset
        ref[spk] = ref_steps[spk]
        slow.step(spk, labels, dt)

        rate_E[t] = spk[:NE].sum()
        rate_I[t] = spk[NE:].sum()
        E_spk_bool[t] = spk[:NE]
        if record_gif and t in frame_set:
            q_frames.append(slow.q_I.copy())
            q_frame_steps.append(t)

        if spk.any():
            spE = np.where(spk[:NE])[0]
            spI = np.where(spk[NE:])[0]
            if spE.size:
                st = a_indptr[spE]
                cnt = a_indptr[spE + 1] - st
                tot = int(cnt.sum())
                if tot:
                    idx = np.arange(tot) - np.repeat(np.cumsum(cnt) - cnt, cnt) + np.repeat(st, cnt)
                    np.add.at(ring_sE, ((t + a_dly[idx]) % M, a_dst[idx]), a_w[idx])
            if spI.size:
                st = g_indptr[spI]
                cnt = g_indptr[spI + 1] - st
                tot = int(cnt.sum())
                if tot:
                    idx = np.arange(tot) - np.repeat(np.cumsum(cnt) - cnt, cnt) + np.repeat(st, cnt)
                    np.add.at(ring_sI, ((t + g_dly[idx]) % M, g_dst[idx]), g_w[idx])

    return {
        "times": np.arange(nsteps) * dt,
        "rate_E": rate_E / NE / dt * 1e3,
        "rate_I": rate_I / NI / dt * 1e3,
        "E_spk_bool": E_spk_bool,
        "lfp_trace": lfp_trace,
        "lfp_sites": contacts,
        "names": names,
        "contacts": contacts,
        "pulses": pulses,
        "q_frames": q_frames,
        "q_frame_steps": q_frame_steps,
        "slow": slow,
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


def _activity_metrics(res, S, cfg: ProtocolConfig):
    t = res["times"]
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
        A = firing_rate_field(fired, S["posE"], S["L"], S2.N_GRID, sigma=0.5)
        pulse_rows.append({
            "source": pulse["source"],
            "t0": pulse["t0"],
            "peak_hz": round(peak, 2),
            "active_frac": round(float(fired.sum() / S["NE"]), 4),
            "field_sum": round(float(A.sum()), 4),
            "before_runaway": bool(runaway_t is None or pulse["t0"] < runaway_t),
        })
    pre_local = [
        r for r in pulse_rows
        if (runaway_t is None or r["t0"] < runaway_t - 20.0)
        and 10.0 <= r["peak_hz"] <= 120.0
        and r["active_frac"] >= 0.02
    ]
    pre_sources = sorted({r["source"] for r in pre_local})
    early_ok = len(pre_local) >= 3 and set(pre_sources) == {"tempA", "tempB"}
    delayed_runaway = runaway_t is not None and runaway_t > pulses[min(3, len(pulses) - 1)]["t0"]
    return {
        "runaway_start_ms": runaway_t,
        "max_rate_hz": round(float(np.max(rate_s)), 2),
        "early_axis_like_events": bool(early_ok),
        "pre_runaway_local_event_count": int(len(pre_local)),
        "pre_runaway_local_sources": pre_sources,
        "delayed_runaway": bool(delayed_runaway),
        "transition_candidate": bool(early_ok and delayed_runaway),
        "q_mean_final": round(float(res["slow"].q_I.mean()), 4),
        "q_min_final": round(float(res["slow"].q_I.min()), 4),
        "pulse_rows": pulse_rows,
    }


def _build(cfg: ProtocolConfig):
    if cfg.layout == "subject1146":
        return _build_subject1146(cfg)
    S = S2.build(S2.SUBSTRATES[cfg.substrate], cfg.seed, T=cfg.T)
    S["masks"] = region_masks(S["L"], S2.N_GRID, S["center"], S["axis_unit"], S2.CORRIDOR_HW)
    return S


def _build_subject1146(cfg: ProtocolConfig):
    L = 10.0
    layout = _subject1146_layout(L)
    sub = S2.SUBSTRATES[cfg.substrate]
    theta = float(layout["theta_rad"])
    axis_unit = np.asarray(layout["axis_unit"], float)
    center = np.asarray(layout["foci"], float).mean(axis=0)
    p = Params(
        g=sub["g"],
        L=L,
        density=100.0,
        T=cfg.T,
        dt=0.1,
        nu_ext_ratio=sub["nu"],
        seed=cfg.seed,
        w_EE=0.1575,
        l_EE=0.380,
        C_EE=800,
        l_EI=sub["l_EI"],
        C_EI=sub["C_EI"],
    )
    rng = np.random.default_rng(cfg.seed)
    pos, labels, NE, NI = place_neurons(p, rng)
    ee_ar = float(sub["AR"] if cfg.ee_ar_override is None else cfg.ee_ar_override)
    if not np.isfinite(ee_ar) or ee_ar <= 0.0:
        raise ValueError(f"ee_ar_override must be positive, got {ee_ar}")
    net = build_connectivity_rot(
        p, pos, labels, NE, NI, rng, theta_EE=theta, AR=ee_ar, verbose=False
    )
    pos = net["pos"]
    N = NE + NI
    S = dict(
        p=p,
        net=net,
        NE=NE,
        NI=NI,
        posE=pos[:NE],
        posI=pos[NE:],
        N=N,
        labels=labels,
        axis_unit=axis_unit,
        center=center,
        L=L,
        layout=layout,
    )
    S["masks"] = region_masks(L, S2.N_GRID, center, axis_unit, S2.CORRIDOR_HW)
    return S


def _axis_ellipse_patch(S):
    center = np.asarray(S["center"], float)
    u = np.asarray(S["axis_unit"], float)
    foci = np.vstack([_source_xy(S, "tempA"), _source_xy(S, "tempB")])
    l_par = 0.380 * np.sqrt(float(S2.SUBSTRATES["primary"]["AR"]))
    l_perp = 0.380 / np.sqrt(float(S2.SUBSTRATES["primary"]["AR"]))
    span = float(np.ptp((foci - center) @ u))
    width = max(2.4, span + 5.2 * l_par)
    height = max(1.0, 8.0 * l_perp)
    angle = float(np.degrees(np.arctan2(u[1], u[0])))
    return Ellipse(
        xy=tuple(center),
        width=width,
        height=height,
        angle=angle,
        fc=PULSE_A,
        ec=AXIS_COL,
        lw=1.1,
        alpha=0.22,
        zorder=4,
    )


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
            ax.text(
                contacts[j, 0],
                contacts[j, 1],
                names[j],
                fontsize=6.5,
                color=color,
                fontweight="bold",
                ha="center",
                va="center",
                zorder=8,
                path_effects=[pe.withStroke(linewidth=1.8, foreground="white")],
            )


def _style_spatial(ax, L):
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)", fontsize=7.0)
    ax.set_ylabel("y (mm)", fontsize=7.0)
    ax.tick_params(axis="both", labelsize=6.5, length=2.0)
    for sp in ax.spines.values():
        sp.set_linewidth(0.8)
        sp.set_color("0.25")


def _render_gif(S, res, metrics, cfg: ProtocolConfig, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    times = res["times"]
    dt = S["p"].dt
    frame_steps = res["q_frame_steps"]
    q_frames = res["q_frames"]
    contacts = res["contacts"]
    names = res["names"]
    shafts = sorted({_shaft(n) for n in names})
    lfp = np.abs(res["lfp_trace"].T)
    runaway_t = metrics["runaway_start_ms"]
    readout_hi = float(S["p"].T)
    readout_sel = (times >= 0.0) & (times <= readout_hi)
    lfp_readout = lfp[:, readout_sel]
    t_readout = times[readout_sel]
    norm_hi = float(S["p"].T if runaway_t is None else max(80.0, min(float(S["p"].T), float(runaway_t))))
    norm_sel = (times >= 0.0) & (times <= norm_hi)
    lfp_norm = lfp[:, norm_sel]
    base = np.median(lfp_norm, axis=1, keepdims=True)
    pre_scale = np.maximum(np.percentile(lfp_norm, 99, axis=1, keepdims=True) - base, 1e-9)
    full_scale = np.maximum(np.percentile(lfp_readout, 99, axis=1, keepdims=True) - base, 1e-9)
    scale = np.maximum(pre_scale, 0.35 * full_scale)
    zlfp = (lfp - base) / scale
    zlfp_readout = zlfp[:, readout_sel]
    trace_y = np.arange(len(names)) * TRACE_OFF
    pulse_windows = []
    for pulse in res["pulses"]:
        if runaway_t is not None and pulse["t0"] >= runaway_t:
            continue
        w0 = max(float(times[0]), float(pulse["t0"]) - SHADE_PAD_MS)
        w1 = min(readout_hi, float(pulse["t0"]) + PULSE_READOUT_MS + SHADE_PAD_MS)
        if w1 <= w0:
            continue
        pulse_windows.append((w0, w1, pulse))
    peak_pulse_t0s = {
        round(float(r["t0"]), 6)
        for r in metrics["pulse_rows"]
        if r["before_runaway"] and float(r["peak_hz"]) >= 5.0 and float(r["active_frac"]) >= 0.02
    }

    def _plot_peak_order(ax):
        for w0, w1, pulse in pulse_windows:
            color = PULSE_A if pulse["source"] == "tempA" else PULSE_B
            ax.axvspan(w0, w1, color=color, alpha=0.26, lw=0, zorder=0)
            if round(float(pulse["t0"]), 6) not in peak_pulse_t0s:
                continue
            inner0 = max(float(pulse["t0"]), 0.0)
            inner1 = min(readout_hi, float(pulse["t0"]) + PULSE_READOUT_MS)
            m = (t_readout >= inner0) & (t_readout <= inner1)
            if int(m.sum()) < 2:
                continue
            pts = []
            idx = np.flatnonzero(m)
            for i in range(len(names)):
                seg = zlfp_readout[i, m]
                local_peak = float(np.max(seg))
                local_amp = local_peak - float(np.median(seg))
                if local_peak < 0.06 or local_amp < 0.035:
                    continue
                local = idx[int(np.argmax(seg))]
                x = float(t_readout[local])
                y = float(zlfp_readout[i, local] + trace_y[i])
                pts.append((x, y))
                ax.plot(x, y, "o", ms=2.1, mfc="black", mec="white", mew=0.35, zorder=6)
            if len(pts) >= 3:
                px, py = zip(*sorted(pts, key=lambda p: p[1], reverse=True))
                ax.plot(px, py, "-", color="black", lw=0.62, alpha=0.50, zorder=5)

    activity_fields = []
    activity_vals = []
    for step in frame_steps:
        lo = max(0, step - int(round(cfg.activity_window_ms / dt)))
        fired = res["E_spk_bool"][lo: step + 1].any(axis=0)
        A = firing_rate_field(fired, S["posE"], S["L"], S2.N_GRID, sigma=0.5)
        activity_fields.append(A)
        if np.any(A > 0):
            activity_vals.append(A[A > 0])
    if activity_vals:
        activity_vmax = max(1.0, float(np.percentile(np.concatenate(activity_vals), 98)))
    else:
        activity_vmax = 1.0

    frames = []
    gif = out_dir / "qigk_runaway_transition.gif"
    png = out_dir / "qigk_runaway_transition_final.png"
    pdf = out_dir / "qigk_runaway_transition_final.pdf"
    for qi, (step, A) in enumerate(zip(frame_steps, activity_fields)):
        tm = float(times[step])
        tm_cursor = float(S["p"].T) if qi == len(frame_steps) - 1 else tm
        fig = plt.figure(figsize=(13.6, 4.8), facecolor="white")
        gs = fig.add_gridspec(
            1,
            3,
            width_ratios=[1.0, 1.0, 2.15],
            left=0.055,
            right=0.985,
            bottom=0.14,
            top=0.84,
            wspace=0.12,
        )

        ax0 = fig.add_subplot(gs[0, 0])
        perm = 1.0 - q_frames[qi]
        im0 = ax0.imshow(perm, origin="lower", extent=[0, S["L"], 0, S["L"]], cmap="plasma", vmin=0.0, vmax=1.0)
        ax0.add_patch(_axis_ellipse_patch(S))
        for source, color, label in (("tempA", PULSE_A, "A"), ("tempB", PULSE_B, "B")):
            xy = _source_xy(S, source)
            core_parallel, core_transverse = _effective_core_radii(S, cfg)
            core_angle = float(
                np.degrees(np.arctan2(S["axis_unit"][1], S["axis_unit"][0]))
            )
            ax0.add_patch(Ellipse(
                xy=xy,
                width=2.0 * core_parallel,
                height=2.0 * core_transverse,
                angle=core_angle,
                fill=False,
                ec="crimson",
                lw=1.0,
                ls="--",
                zorder=7,
            ))
            ax0.text(xy[0], xy[1] + 0.44, label, fontsize=8, color="crimson", fontweight="bold",
                     ha="center", va="bottom", path_effects=[pe.withStroke(linewidth=1.8, foreground="white")])
        _draw_contacts(ax0, contacts, names)
        _style_spatial(ax0, S["L"])
        ax0.set_title("permissivity", fontsize=9.0, fontweight="bold", pad=4)
        cb0 = fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.02)
        cb0.ax.tick_params(labelsize=6.5)

        ax1 = fig.add_subplot(gs[0, 1])
        im1 = ax1.imshow(A, origin="lower", extent=[0, S["L"], 0, S["L"]],
                         cmap="viridis", vmin=0.0, vmax=activity_vmax)
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
        cb1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.02)
        cb1.ax.tick_params(labelsize=6.5)

        ax2 = fig.add_subplot(gs[0, 2])
        _plot_peak_order(ax2)
        for i, nm in enumerate(names):
            col = _shaft_color(nm, shafts)
            ax2.plot(t_readout, zlfp_readout[i] + trace_y[i], color=col, lw=0.72, alpha=0.90, zorder=3)
        if tm_cursor <= readout_hi:
            ax2.axvline(tm_cursor, color="black", lw=1.2, alpha=0.90, zorder=7)
        if runaway_t is not None and runaway_t <= readout_hi + 1e-9:
            ax2.axvline(runaway_t, color="crimson", lw=1.0, ls="--", alpha=0.9, zorder=6)
        ax2.set_xlim(0.0, readout_hi)
        ax2.set_yticks(trace_y)
        ax2.set_yticklabels(names, fontsize=6.8)
        for tick, nm in zip(ax2.get_yticklabels(), names):
            tick.set_color(_shaft_color(nm, shafts))
        ax2.tick_params(axis="x", labelsize=7.0, length=2.5)
        ax2.tick_params(axis="y", labelsize=6.8, length=2.0)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.set_ylabel("contact", fontsize=7.8)
        ax2.set_xlabel("time (ms)", fontsize=7.8)
        ax2.legend(
            handles=[
                Patch(facecolor=PULSE_A, alpha=0.40, edgecolor="none", label="tempA response"),
                Patch(facecolor=PULSE_B, alpha=0.40, edgecolor="none", label="tempB response"),
                Line2D([0], [0], color="black", lw=0.8, marker="o", ms=2.5, mfc="black", mec="white",
                       label="peak line"),
            ],
            frameon=False,
            fontsize=7.0,
            loc="upper right",
            bbox_to_anchor=(1.0, 1.08),
            ncol=3,
            handlelength=1.4,
            columnspacing=0.8,
        )

        fig.text(0.016, 0.93, "A", fontsize=17, fontweight="bold")
        layout_label = S.get("layout", {}).get("label", "Stage5 geometry")
        fig.text(0.50, 0.93, f"qI build-up to runaway ({layout_label}) | t={tm_cursor:.0f} ms", fontsize=10.0, fontweight="bold", ha="center")
        fig.text(
            0.50,
            0.04,
            f"full readout 0-{readout_hi:.0f} ms | normalized from pre-runaway activity | runaway_start={metrics['runaway_start_ms']} ms",
            fontsize=7.5,
            ha="center",
            color="0.25",
        )
        if qi == len(frame_steps) - 1:
            fig.savefig(png, dpi=180, bbox_inches="tight", facecolor="white")
            fig.savefig(pdf, bbox_inches="tight", facecolor="white")
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        frames.append(frame)
        plt.close(fig)

    frames.extend([frames[-1]] * 8)
    imageio.mimsave(gif, frames, duration=0.11, loop=0)
    return gif, png, pdf, activity_vmax, readout_hi


def _write_readme(metrics, cfg: ProtocolConfig, out_dir: Path):
    if cfg.layout == "subject1146":
        layout_sentence = (
            "电极排布和两个灶的位置来自 `fig_subject_snn_epilepsiae_1146` 的 "
            "`figdata_epilepsiae_1146_twoend_equal_tsrc_s3.npz`，并等比例缩放到当前 M3A L=10 sheet。"
        )
    else:
        layout_sentence = (
            "电极摆放沿用 Fig5/Stage5 的 A0-A5/B0-B5 双 shaft montage，并按当前 L 等比例缩放。"
        )
    text = f"""# M3A-v2.1 qI build-up to runaway GIF

### qigk_runaway_transition.gif

这张 GIF 是连续单轨迹 visual diagnostic，不是新的统计 sweep。{layout_sentence}SEEG readout 是同一条连续 trace，没有拼接 gap。

**布局**：一行三列：`1-q_I` permissivity map | 实时 2D SNN E 活动 | full continuous SEEG readout。

**关注点**：先看前几次 tempA/tempB 局部轴向事件是否仍短暂，再看 `1-q_I` 是否逐步累积，最后看 sustained high-rate runaway 是否在同一条轨迹中出现。

**Readout 规则**：右侧显示完整 0-T trace，红虚线标 `runaway_start`。归一化主要用 pre-runaway activity 估计，同时保留 runaway 后 trace，不再把 readout 截在 onset 前。暖/蓝 shading 只表示 runaway 前的 tempA/tempB pulse response window；黑点表示每个 response window 内各 contact 的局部峰值，黑线按 y 轴从上到下连接，不表示 rank-order 分类。这不是 KMeans 事件分类。

### qigk_runaway_transition_final.png

这是 GIF 最后一帧的静态快照，用来快速检查 runaway 末态和 readout 是否非空。

**关注点**：若中间 2D 活动接近全场，而右侧 readout 同时保留前期小事件和 runaway 后高幅段，则这张图展示的是 build-up-to-runaway，不是 recovery。

### qigk_runaway_transition_final.pdf

同一最后帧的 PDF 版本，便于 paper-ready 视觉审阅。

Transition candidate: `{metrics['transition_candidate']}`; runaway_start_ms: `{metrics['runaway_start_ms']}`.
"""
    (out_dir / "README.md").write_text(text)


def run_one(cfg: ProtocolConfig, *, record_gif: bool):
    S = _build(cfg)
    res = _simulate_continuous(S, cfg, record_gif=record_gif)
    metrics = _activity_metrics(res, S, cfg)
    return S, res, metrics


def screen():
    rows = []
    for k_q in (0.06, 0.10, 0.14, 0.18, 0.24, 0.32):
        for q_min in (0.00, 0.05, 0.10):
            cfg = ProtocolConfig(k_q=k_q, q_min=q_min)
            S, res, metrics = run_one(cfg, record_gif=False)
            row = {"config": asdict(cfg), "metrics": metrics, "wall_s": round(float(res["wall_s"]), 2)}
            rows.append(row)
            print(json.dumps({
                "k_q": k_q,
                "q_min": q_min,
                "transition": metrics["transition_candidate"],
                "runaway": metrics["runaway_start_ms"],
                "max_rate": metrics["max_rate_hz"],
                "q_final": metrics["q_mean_final"],
                "wall_s": row["wall_s"],
            }), flush=True)
            if metrics["transition_candidate"]:
                return row, rows
    return None, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen", action="store_true", help="run a small q_I parameter screen and write JSON only")
    ap.add_argument("--k-q", type=float, default=ProtocolConfig.k_q)
    ap.add_argument("--q-min", type=float, default=ProtocolConfig.q_min)
    ap.add_argument("--kick-boost", type=float, default=ProtocolConfig.kick_boost)
    ap.add_argument("--r-kick", type=float, default=ProtocolConfig.r_kick)
    ap.add_argument("--T", type=float, default=ProtocolConfig.T)
    ap.add_argument("--layout", choices=["stage5", "subject1146"], default="stage5")
    ap.add_argument("--fig-name", default=None)
    args = ap.parse_args()
    os.chdir(ROOT)
    if args.screen:
        best, rows = screen()
        out_dir = _out_dir(FIG_NAME)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "screen_rows.json").write_text(json.dumps({"best": best, "rows": rows}, indent=2))
        print(f"wrote {out_dir / 'screen_rows.json'}")
        return 0 if best is not None else 2

    fig_name = args.fig_name or (
        FIG_NAME if args.layout == "stage5" else f"{FIG_NAME}_epilepsiae_1146"
    )
    cfg = ProtocolConfig(
        k_q=float(args.k_q),
        q_min=float(args.q_min),
        kick_boost=float(args.kick_boost),
        r_kick=float(args.r_kick),
        T=float(args.T),
        layout=str(args.layout),
        fig_name=str(fig_name),
    )
    S, res, metrics = run_one(cfg, record_gif=True)
    out_dir = _out_dir(cfg.fig_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    gif, png, pdf, activity_vmax, readout_hi = _render_gif(S, res, metrics, cfg, out_dir)
    meta = {
        "figure": cfg.fig_name,
        "status": "visual diagnostic, continuous trajectory, no recovery/termination claim",
        "config": asdict(cfg),
        "metrics": metrics,
        "outputs": {
            "gif": str(gif.relative_to(ROOT)),
            "final_png": str(png.relative_to(ROOT)),
            "final_pdf": str(pdf.relative_to(ROOT)),
        },
        "montage": {
            "source": str(S.get("layout", {}).get("source", STAGE5_FIGDATA.relative_to(ROOT))),
            "rule": (
                "E1146 foci and contacts scaled by current_L/reference_L"
                if cfg.layout == "subject1146"
                else "Fig5/Stage5 A0-A5/B0-B5 contacts scaled by current_L/reference_L"
            ),
            "n_contacts": len(res["names"]),
            "names": list(res["names"]),
            "layout": _jsonable_layout(S.get("layout", None)),
        },
        "shading": {
            "method": "scheduled pre-runaway pulse response windows with per-window peak dots and top-to-bottom connector lines",
            "readout_window_ms": [0.0, readout_hi],
            "normalization": "per contact baseline/scale estimated from pre-runaway activity, with full 0-T readout still displayed",
            "post_runaway": "no readout shading after runaway_start_ms; no KMeans/event clustering after runaway",
        },
        "colorbars": {
            "permissivity_vmin": 0.0,
            "permissivity_vmax": 1.0,
            "activity_vmin": 0.0,
            "activity_vmax": activity_vmax,
            "activity_vmax_rule": "global 98th percentile over positive activity pixels across all GIF frames",
        },
        "layout": {
            "columns": "permissivity map | real-time 2D E activity | full continuous virtual-SEEG readout",
        },
        "notes": [
            "The right readout is the same continuous simulated trace from 0 to T; no tempA/tempB segments are spliced.",
            "Readout scaling keeps small pre-runaway responses visible while still showing the post-runaway segment.",
            "The virtual SEEG montage is recorded in metadata and scaled to the current M3A sheet when needed.",
            "Readout shading/peak markers are scheduled-response diagnostics, not KMeans-derived propagation labels.",
            "Peak connector lines are drawn from top to bottom on the y-axis and do not encode temporal rank order.",
            "The local runner copies the kick_probe integration loop and only extends localized drive to a multi-pulse schedule.",
            "h_G/recovery is off by construction; this GIF only visualizes the build-up-to-runaway leg.",
        ],
    }
    (out_dir / "qigk_runaway_transition_metadata.json").write_text(json.dumps(meta, indent=2))
    _write_readme(metrics, cfg, out_dir)
    print(f"wrote {gif}")
    print(f"wrote {png}")
    print(f"wrote {pdf}")
    print(f"wrote {out_dir / 'qigk_runaway_transition_metadata.json'}")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
