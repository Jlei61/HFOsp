"""M3A-v2.1 q_I+g_K gap-sweep visual diagnostics in the Topic-4 SNN style.

This script is intentionally diagnostic, not a new statistical sweep. It
re-runs a small set of representative probes so the q_I/g_K negative screen can
be visually inspected with the same visual grammar as Fig5:

    mechanism | tempA source | tempB source | electrode readout

The quantified q_I/g_K gap sweep used a single-core probe. To avoid mixing
source identity with direction, this diagnostic mirrors the probe at the two
scaffold ends (tempA/tempB) and keeps the source labels explicit. The cases
illustrate the observed regimes: baseline axial, returned axis-only closest
miss, a small-footprint metric-edge suppress case, dynamic g_K suppress, and
off-axis/global runaway. The per-run JSON remains the statistical source of
truth.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib import gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ENG = ROOT / "src" / "snn_engine"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ENG))

import run_m3a_v2_step2_qI as S2  # noqa: E402
from kick_probe import simulate_kick  # noqa: E402
from lfp import LFPRecorder  # noqa: E402
from slow_field import SpatialSlowField, SpatialSlowFieldConfig  # noqa: E402
from src.sef_hfo_heterogeneity import sample_core_field  # noqa: E402
from src.sef_hfo_snn_metrics import self_limit  # noqa: E402
from src.topic4_m3a_v2_phenotype import (  # noqa: E402
    axis_score,
    make_field_grid_xy,
    offaxis_fraction,
    recruitment_area,
)

FIG_NAME = "fig_m3a_v2_1_qigk_gap_dynamics"
OUT = ROOT / "results" / "paper-ready-figure" / FIG_NAME / "figures"

FWD_SHADE = "#f4b266"
REV_SHADE = "#78a6d8"
SHAFT_A = "#e8743b"
SHAFT_B = "#1f9e9e"
AXIS_COL = "#a65f00"
SPATIAL_DOT_SIZE = 8.0
SPATIAL_ALPHA = 0.90
TRACE_OFF = 1.35
SHADE_PAD_MS = 18.0
N_CONTACTS_PER_SHAFT = 7
CONTACT_PITCH = 1.35


CASES = [
    {
        "case_id": "baseline_axial",
        "label": "baseline axial",
        "substrate": "primary",
        "seed": 1,
        "q_axis": None,
        "q_off": None,
        "gK_axis": 0.0,
        "gK_off": 0.0,
        "k_K": 0.0,
        "eta_K": 0.0,
        "template_mode": "none",
        "source_row": "slow off",
        "description": "slow off; self-limited axial recruitment",
    },
    {
        "case_id": "returned_axis_only_clean",
        "label": "returned axis-only",
        "substrate": "primary",
        "seed": 1,
        "q_axis": 0.9,
        "q_off": 0.5,
        "gK_axis": 0.0,
        "gK_off": 0.0,
        "k_K": 0.0,
        "eta_K": 30.0,
        "template_mode": "2region",
        "source_row": "stage2 returned axis-only representative; both mirrored sources return",
        "description": "returned stage-2 clamp cell with off-axis pressure but axis-dominated events",
    },
    {
        "case_id": "metric_edge_small_suppress",
        "label": "metric edge",
        "substrate": "primary",
        "seed": 1,
        "q_axis": 0.9,
        "q_off": 0.5,
        "gK_axis": 0.75,
        "gK_off": 0.0,
        "k_K": 1.0,
        "eta_K": 60.0,
        "template_mode": "core_protected",
        "source_row": "stage2 dynamic gK suppress; low S_axis but small R/F",
        "description": "small-footprint suppress case where axis score can be misleading",
    },
    {
        "case_id": "dynamic_gk_suppress",
        "label": "gK suppress",
        "substrate": "primary",
        "seed": 1,
        "q_axis": 0.9,
        "q_off": 0.5,
        "gK_axis": 1.0,
        "gK_off": 0.0,
        "k_K": 1.0,
        "eta_K": 60.0,
        "template_mode": "core_protected",
        "source_row": "stage2 dynamic gK suppress",
        "description": "dynamic gK reduces the event instead of redirecting it off-axis",
    },
    {
        "case_id": "dynamic_gk_runaway",
        "label": "global runaway",
        "substrate": "primary",
        "seed": 1,
        "q_axis": 0.9,
        "q_off": 0.3,
        "gK_axis": 1.0,
        "gK_off": 0.0,
        "k_K": 1.0,
        "eta_K": 60.0,
        "template_mode": "core_protected",
        "source_row": "stage2 dynamic gK runaway high F_off",
        "description": "off-axis/global recruitment appears only with runaway, not recovery",
    },
]


def _axis(theta_deg: float = 45.0):
    th = np.deg2rad(theta_deg)
    u = np.array([np.cos(th), np.sin(th)])
    p = np.array([-u[1], u[0]])
    return u, p


def _source_xy(S: dict, source: str) -> np.ndarray:
    sign = -1.0 if source == "tempA" else 1.0
    return np.asarray(S["center"], float) + sign * 0.6 * (float(S["L"]) / 2.0) * np.asarray(S["axis_unit"], float)


def _source_vth(S: dict, source: str) -> np.ndarray:
    is_E = np.zeros(S["N"], bool)
    is_E[: S["NE"]] = True
    cf = sample_core_field(
        S["net"]["pos"],
        is_E,
        _source_xy(S, source),
        1.0,
        np.random.default_rng(int(S["p"].seed) + (7 if source == "tempA" else 8)),
        core_mean=16.5,
        core_std=1.0,
        base_mean=18.0,
    )
    return cf["vth"]


def _contacts(S: dict):
    center = np.asarray(S["center"], float)
    u = np.asarray(S["axis_unit"], float)
    p = np.array([-u[1], u[0]])
    half = (N_CONTACTS_PER_SHAFT - 1) / 2.0
    offsets = (np.arange(N_CONTACTS_PER_SHAFT) - half) * CONTACT_PITCH
    a = np.array([center + d * u for d in offsets])
    b = np.array([center + d * p for d in offsets])
    names = [f"A{i}" for i in range(N_CONTACTS_PER_SHAFT)] + [f"B{i}" for i in range(N_CONTACTS_PER_SHAFT)]
    return np.vstack([a, b]), names


def _make_field(S: dict, case: dict):
    if case["q_axis"] is None:
        return None
    cfg = SpatialSlowFieldConfig(
        use_qI=True,
        use_gK=case["k_K"] > 0.0,
        k_q=0.0,
        k_K=float(case["k_K"]),
        sigma_q=1.5,
        sigma_K=0.5,
        q_min=0.25,
        eta_K=float(case["eta_K"]),
        gK_max=1.0,
        tau_q=S2.TAU_Q,
        tau_K=S2.TAU_Q,
        tau_a=S2.TAU_A,
        q_init=1.0,
    )
    field = SpatialSlowField(S["N"], 18.0, S["posE"], S["posI"], S["L"], cfg=cfg)
    am = S["masks"]["axis"]
    om = S["masks"]["offaxis"]
    field.q_I[om] = float(case["q_off"])
    field.q_I[am] = float(case["q_axis"])
    field.g_K[om] = float(case["gK_off"])
    field.g_K[am] = float(case["gK_axis"])
    if case.get("template_mode") == "core_protected":
        gxy = make_field_grid_xy(S["L"], S2.N_GRID)
        cm = np.linalg.norm(gxy - S["core_xy"], axis=-1) <= 1.5
        field.q_I[cm] = float(case["q_off"])
        field.g_K[cm] = 0.0
    return field


def _event_window(rate, dt):
    sl = self_limit(rate, dt, S2.T_KICK)
    t = np.arange(len(rate)) * dt
    rest = float(sl["rest_rate"])
    peak = float(sl["peak"])
    thr = rest + 0.25 * max(peak - rest, 1e-9)
    m = (t >= S2.T_KICK) & (rate > thr)
    if not np.any(m):
        return S2.T_KICK, min(float(t[-1]), S2.T_KICK + 40.0), sl
    idx = np.flatnonzero(m)
    return float(t[idx[0]]), float(t[idx[-1]]), sl


def _run_probe(case: dict, source: str):
    S = S2.build(S2.SUBSTRATES[case["substrate"]], int(case["seed"]))
    S["core_xy"] = _source_xy(S, source)
    S["vth"] = _source_vth(S, source)
    contacts, names = _contacts(S)
    rec = LFPRecorder(S["p"], S["net"]["pos"], S["labels"], sites=contacts)
    field = _make_field(S, case)
    seed_offset = 0 if source == "tempA" else 1000
    S["net"]["rng"] = np.random.default_rng(int(case["seed"]) + seed_offset)
    res = simulate_kick(
        S["p"],
        S["net"],
        KICK_BOOST=S2.KICK,
        slow=field,
        kick_center=S["core_xy"],
        r_kick=0.3,
        t_kick=S2.T_KICK,
        V_th_per_neuron=S["vth"],
        lfp_recorder=rec,
    )
    t_on, t_off, sl = _event_window(res["rate_E"], S["p"].dt)
    onset = _onset_from_window(res["E_spk_bool"], S["p"].dt, t_on, min(t_off, S["p"].T - S["p"].dt))
    metrics = _source_metrics(res, S, onset, sl)
    return {
        "S": S,
        "field": field,
        "res": res,
        "contacts": contacts,
        "names": names,
        "t_on": t_on,
        "t_off": t_off,
        "onset": onset,
        "metrics": metrics,
        "source": source,
        "case": case,
    }


def _onset_from_window(E_spk_bool, dt, t_on, t_off):
    s = int(round(t_on / dt))
    e = max(s + 1, int(round(t_off / dt)))
    seg = np.asarray(E_spk_bool)[s:e]
    fired = seg.any(axis=0)
    onset = np.full(seg.shape[1], np.nan)
    idx = np.flatnonzero(fired)
    if idx.size:
        onset[idx] = np.argmax(seg[:, idx], axis=0).astype(float) * dt + t_on
    return onset


def _source_metrics(res, S, onset, sl):
    posE = S["posE"]
    ever = np.isfinite(onset)
    A = S2.firing_rate_field(ever, posE, S["L"], S2.N_GRID, sigma=0.5)
    gxy = make_field_grid_xy(S["L"], S2.N_GRID)
    A_thr = S2.THETA_A_FRAC * A.max() if A.max() > 0 else 0.0
    s_axis = axis_score(posE, onset, S["axis_unit"])
    return {
        "n_onsets": int(ever.sum()),
        "R_area": round(float(recruitment_area(A, A_thr)), 4),
        "S_axis": None if s_axis != s_axis else round(float(s_axis), 4),
        "F_off": round(float(offaxis_fraction(A, gxy, S["center"], S["axis_unit"], S2.CORRIDOR_HW)), 4),
        "peak_rate": round(float(sl["peak"]), 2),
        "returned": bool(sl["returned"] and sl["tail_complete"]),
        "event_t_on": round(float(sl["peak_t"]), 1),
        "burst_duration_ms": round(float(sl["burst_duration_ms"]), 1),
    }


def _axis_range_patch(S):
    center = np.asarray(S["center"], float)
    u = np.asarray(S["axis_unit"], float)
    p = np.array([-u[1], u[0]])
    foci = np.vstack([_source_xy(S, "tempA"), _source_xy(S, "tempB")])
    l_par = 0.380 * np.sqrt(float(S2.SUBSTRATES["primary"]["AR"]))
    l_perp = 0.380 / np.sqrt(float(S2.SUBSTRATES["primary"]["AR"]))
    half_w = max(0.42, 3.0 * l_perp)
    ext = max(0.80, 3.0 * l_par)
    proj = (foci - center) @ u
    a = center + u * (float(proj.min()) - ext)
    b = center + u * (float(proj.max()) + ext)
    return np.vstack([a + half_w * p, b + half_w * p, b - half_w * p, a - half_w * p])


def _draw_contacts(ax, contacts, names):
    for prefix, color, marker in (("A", SHAFT_A, "o"), ("B", SHAFT_B, "s")):
        idx = [i for i, n in enumerate(names) if n.startswith(prefix)]
        pts = contacts[idx]
        ax.plot(pts[:, 0], pts[:, 1], color=color, lw=1.0, alpha=0.60, zorder=5)
        ax.scatter(pts[:, 0], pts[:, 1], s=38, marker=marker, fc="white", ec=color, lw=1.0, zorder=6)
        for j in (idx[0], idx[-1]):
            ax.text(
                contacts[j, 0],
                contacts[j, 1],
                names[j],
                fontsize=7,
                color=color,
                fontweight="bold",
                ha="center",
                va="center",
                zorder=8,
                path_effects=[pe.withStroke(linewidth=2.0, foreground="white")],
            )


def _style_spatial(ax, L):
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)", fontsize=7.6)
    ax.set_ylabel("y (mm)", fontsize=7.6)
    ax.tick_params(axis="both", labelsize=7.0, length=2.5)
    for sp in ax.spines.values():
        sp.set_linewidth(0.8)
        sp.set_color("0.25")


def _plot_mechanism(ax, runA, runB):
    S = runA["S"]
    pos = S["posE"]
    v = np.minimum(runA["S"]["vth"][: S["NE"]], runB["S"]["vth"][: S["NE"]])
    foci = np.vstack([_source_xy(S, "tempA"), _source_xy(S, "tempB")])
    contacts = runA["contacts"]
    names = runA["names"]
    ax.scatter(
        pos[:, 0],
        pos[:, 1],
        c=np.clip(18.0 - v, 0.0, None),
        s=SPATIAL_DOT_SIZE,
        cmap="plasma",
        vmin=0.0,
        vmax=1.2,
        alpha=SPATIAL_ALPHA,
        linewidths=0,
        rasterized=True,
        zorder=2,
    )
    ax.add_patch(Polygon(_axis_range_patch(S), closed=True, fc=FWD_SHADE, ec=AXIS_COL, lw=1.2, alpha=0.30, zorder=4))
    for i, f in enumerate(foci):
        ax.add_patch(plt.Circle(f, 1.0, fill=False, ec="crimson", lw=1.2, ls="--", zorder=7))
        ax.text(
            f[0],
            f[1] + 0.54,
            "A" if i == 0 else "B",
            fontsize=8.5,
            color="crimson",
            fontweight="bold",
            ha="center",
            va="bottom",
            path_effects=[pe.withStroke(linewidth=2.0, foreground="white")],
        )
    p0 = np.asarray(S["center"]) - S["axis_unit"] * 4.6
    p1 = np.asarray(S["center"]) + S["axis_unit"] * 4.6
    # scaffold axis ORIENTATION only -- NOT a propagation/seizure direction (no arrowhead; review 2026-06-28)
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=AXIS_COL, lw=1.55, alpha=0.9,
            solid_capstyle="round", zorder=8)
    _draw_contacts(ax, contacts, names)
    ax.set_title("mechanism", fontsize=9.5, fontweight="bold", pad=5)
    _style_spatial(ax, S["L"])


def _plot_event(ax, run, title):
    S = run["S"]
    pos = S["posE"]
    onset = run["onset"]
    fin = np.isfinite(onset)
    bg = np.zeros(len(pos), bool)
    bg[::4] = True
    ax.scatter(pos[bg & ~fin, 0], pos[bg & ~fin, 1], s=1.2, c="0.86", alpha=0.35, linewidths=0, rasterized=True, zorder=1)
    if fin.any():
        rel = onset.copy()
        rel[fin] -= np.nanmin(rel[fin])
        vmax = max(1.0, float(np.nanpercentile(rel[fin], 98)))
        ax.scatter(
            pos[fin, 0],
            pos[fin, 1],
            c=rel[fin],
            s=SPATIAL_DOT_SIZE,
            cmap="viridis",
            vmin=0.0,
            vmax=vmax,
            alpha=SPATIAL_ALPHA,
            linewidths=0,
            rasterized=True,
            zorder=2,
        )
    for f in (_source_xy(S, "tempA"), _source_xy(S, "tempB")):
        ax.add_patch(plt.Circle(f, 1.0, fill=False, ec="crimson", lw=1.1, ls="--", zorder=5))
    src = _source_xy(S, run["source"])
    ax.scatter([src[0]], [src[1]], marker="*", s=145, c="black", ec="white", lw=0.8, zorder=7)
    p0 = np.asarray(S["center"]) - S["axis_unit"] * 4.6
    p1 = np.asarray(S["center"]) + S["axis_unit"] * 4.6
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="0.20", lw=1.1, alpha=0.75, zorder=4)
    _draw_contacts(ax, run["contacts"], run["names"])
    ax.set_title(title, fontsize=9.5, fontweight="bold", pad=5)
    _style_spatial(ax, S["L"])


def _active_contacts(runA, runB):
    contacts = runA["contacts"]
    names = runA["names"]
    active = []
    for k in range(len(names)):
        c = contacts[k]
        for run in (runA, runB):
            pos = run["S"]["posE"]
            fired = np.isfinite(run["onset"])
            if fired.any() and np.any(np.linalg.norm(pos[fired] - c, axis=1) <= 1.2):
                active.append(k)
                break
    if not active:
        active = list(range(len(names)))
    proj = contacts[active] @ runA["S"]["axis_unit"]
    is_b = np.array([names[i].startswith("B") for i in active])
    order = np.lexsort((is_b, proj))
    return [active[i] for i in order]


def _plot_readout(ax, runA, runB):
    keep = _active_contacts(runA, runB)
    names = runA["names"]
    traces = []
    spans = []
    shift = 0.0
    gap = 70.0
    for run, color, label in ((runA, FWD_SHADE, "tempA-source probe"), (runB, REV_SHADE, "tempB-source probe")):
        t = run["res"]["times"]
        lfp = np.asarray(run["res"]["lfp_trace"], float).T[keep]
        sel = (t >= 60.0) & (t <= 460.0)
        tt = t[sel] - 60.0 + shift
        sub = lfp[:, sel]
        traces.append((tt, sub, color, label, shift, run))
        spans.append((max(tt[0], run["t_on"] - 60.0 + shift - SHADE_PAD_MS), min(tt[-1], run["t_off"] - 60.0 + shift + SHADE_PAD_MS), color, label))
        shift = float(tt[-1] + gap)

    all_sub = np.concatenate([x[1] for x in traces], axis=1)
    base = np.median(all_sub, axis=1, keepdims=True)
    scale = np.maximum(np.percentile(all_sub, 98, axis=1, keepdims=True) - base, 1e-9)
    y = np.arange(len(keep)) * TRACE_OFF
    for span0, span1, color, _ in spans:
        ax.axvspan(span0, span1, color=color, alpha=0.30, lw=0, zorder=0)
    for tt, sub, _, _, _, _ in traces:
        zt = (sub - base) / scale
        for i, ci in enumerate(keep):
            col = SHAFT_B if names[ci].startswith("B") else SHAFT_A
            ax.plot(tt, zt[i] + y[i], color=col, lw=0.76, alpha=0.92, zorder=3)

    ax.set_xlim(0.0, max(x[0][-1] for x in traces))
    ax.set_yticks(y)
    ax.set_yticklabels([names[i] for i in keep], fontsize=7.0)
    for tick, ci in zip(ax.get_yticklabels(), keep):
        tick.set_color(SHAFT_B if names[ci].startswith("B") else SHAFT_A)
    ax.tick_params(axis="y", length=2.5, labelsize=7.0, color="0.35")
    ax.tick_params(axis="x", length=3.0, labelsize=7.5, color="0.35")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("0.35")
        ax.spines[side].set_linewidth(0.8)
    ax.set_xlabel("time (ms)", fontsize=8.2)
    ax.legend(
        handles=[
            Patch(facecolor=FWD_SHADE, alpha=0.40, edgecolor="none", label="tempA-source probe"),
            Patch(facecolor=REV_SHADE, alpha=0.40, edgecolor="none", label="tempB-source probe"),
            Line2D([0], [0], color=SHAFT_A, lw=0.9, label="A shaft"),
            Line2D([0], [0], color=SHAFT_B, lw=0.9, label="B shaft"),
        ],
        frameon=False,
        fontsize=7.4,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.045),
        borderaxespad=0.0,
        ncol=4,
        handlelength=1.5,
        columnspacing=0.8,
    )


def _compose_case(case: dict):
    runA = _run_probe(case, "tempA")
    runB = _run_probe(case, "tempB")
    fig = plt.figure(figsize=(18.0, 4.45), facecolor="white")
    gs = gridspec.GridSpec(
        1,
        4,
        width_ratios=[1.0, 1.0, 1.0, 2.75],
        left=0.045,
        right=0.992,
        bottom=0.16,
        top=0.82,
        wspace=0.075,
    )
    _plot_mechanism(fig.add_subplot(gs[0, 0]), runA, runB)
    _plot_event(fig.add_subplot(gs[0, 1]), runA, "tempA source")
    _plot_event(fig.add_subplot(gs[0, 2]), runB, "tempB source")
    _plot_readout(fig.add_subplot(gs[0, 3]), runA, runB)
    fig.text(0.012, 0.925, "A", fontsize=19, fontweight="bold")
    fig.text(0.50, 0.925, case["label"], fontsize=10.0, fontweight="bold", ha="center")

    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / f"{case['case_id']}.png"
    pdf = OUT / f"{case['case_id']}.pdf"
    fig.savefig(png, dpi=180, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {
        "case": case,
        "outputs": {"png": str(png.relative_to(ROOT)), "pdf": str(pdf.relative_to(ROOT))},
        "tempA_metrics": runA["metrics"],
        "tempB_metrics": runB["metrics"],
    }


def _write_readme():
    text = """# M3A-v2.1 qI/gK gap dynamics diagnostics

### baseline_axial.png

这张图展示 slow field 关闭时，同一 scaffold 两端作为 source 的代表性轴向 probe。它用于确认视觉基线：事件沿 E->E scaffold 传播，并能回到低活动状态。

**关注点**：tempA/tempB source 是 source identity，不是方向标签；看中间两格的 onset 梯度和右侧同一 montage 的 readout。

### returned_axis_only_clean.png

这张图展示 Stage-2 clamp 中两端 source 都 returned 的 axis-only 代表状态。慢变量状态给了 off-axis permissive pressure，但事件仍沿结构轴走，并没有达到破轴后恢复。

**关注点**：看 `F_offaxis` 稍升是否只是轴向带变宽，而不是真正的 tempA/tempB 旁路事件。

### metric_edge_small_suppress.png

这张图专门展示小范围 metric 边界：部分 run 的 `S_axis` 会变低，但事件面积和 off-axis fraction 都很小。它用于检查“读数塌”是否被误读成真正旁路招募。

**关注点**：若空间图只有稀疏局部点，不能把低 `S_axis` 当成破轴成功。

### dynamic_gk_suppress.png

这张图展示动态 g_K 的 suppress 代表状态。它对应 gap sweep 里“疲劳刹车压小事件，而不是把活动改道到 off-axis”的现象。

**关注点**：看 readout 和空间图是否表现为缩小/局部化，而不是 recovered off-axis recruitment。

### dynamic_gk_runaway.png

这张图展示 off-axis/global 成分最高的一类动态 g_K runaway。它用于目视确认“旁路/全局招募”出现时是否伴随不恢复。

**关注点**：如果空间图接近全场、readout 长时间高活动，这不是成功的可恢复候选。
"""
    (OUT / "README.md").write_text(text)


def main():
    os.chdir(ROOT)
    summaries = []
    for case in CASES:
        print(f"[plot] {case['case_id']}", flush=True)
        summaries.append(_compose_case(case))
    _write_readme()
    meta = {
        "figure": FIG_NAME,
        "status": "visual diagnostic, not a new statistical sweep",
        "source_of_truth": [
            ".worktrees/topic4-m3a-v2-1/results/topic4_m3a_v2_1_qigk_gap_sweep/20260629_003015/per_run.jsonl",
            ".worktrees/topic4-m3a-v2-1/results/topic4_m3a_v2_1_qigk_gap_sweep/20260629_003015/REPORT.md",
        ],
        "notes": [
            "The q_I/g_K quantified screen used a single-core probe; this diagnostic mirrors the probe at two scaffold ends to preserve the four-panel source-identity visual standard.",
            "Representative q_I/g_K states are frozen from Stage-2 clamp rows; this avoids rerunning the full parameter sweep.",
            "Shading labels source identity, not propagation direction.",
            "The mechanism-panel axis marks scaffold ORIENTATION only (non-directional line, no arrowhead); it is not a propagation or seizure direction.",
        ],
        "cases": summaries,
    }
    (OUT / "m3a_v2_1_qigk_gap_dynamics_metadata.json").write_text(json.dumps(meta, indent=2))
    for s in summaries:
        print(f"wrote {ROOT / s['outputs']['png']} and {ROOT / s['outputs']['pdf']}", flush=True)
    print(f"wrote {OUT / 'README.md'}", flush=True)


if __name__ == "__main__":
    main()
