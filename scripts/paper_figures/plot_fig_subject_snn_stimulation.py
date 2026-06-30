"""Subject-layout SNN stimulation illustration for E1146.

This is the subject-specific counterpart of ``plot_stage3_stimulation_ab_figure.py``:
it keeps the E1146 real electrode layout / template-source core placement used by
``fig_subject_snn_epilepsiae_1146.png``, then applies a finite E-only high-threshold
clamp around the four middle ICL contacts.

The script runs a real dynamic-threshold simulation and writes one AB-only figure:
substrate + stimulation contacts | pre-stim propagation event | stim-on local event
| read-out with stim ON/OFF and post-off recovery.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from matplotlib import gridspec

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src/snn_engine"))

from connectivity import place_neurons  # noqa: E402
from connectivity_rot import build_connectivity_rot  # noqa: E402
from lfp import LFPRecorder  # noqa: E402
from params import Params  # noqa: E402
from src.sef_hfo_axial_intervention import simulate_dynamic_vth  # noqa: E402
from src.sef_hfo_events import detect_events  # noqa: E402
from src.sef_hfo_heterogeneity import sample_core_field  # noqa: E402
from src.sef_hfo_snn_adapter import snn_event_envelope  # noqa: E402
from src.sef_hfo_subject_placement import register_to_sheet, template_source_foci  # noqa: E402

import scripts.run_sef_hfo_snn_cm_spontaneous_readout as C  # noqa: E402

DT = C.DT
OFF = 1.25
SHAFT_COLS = ["#e8743b", "#1f9e9e", "#7b5cb8", "#3b7a3b"]
STIM_COL = "#2f80ed"
FWD_SHADE = "#f4b266"
REV_SHADE = "#78a6d8"


def _shaft(name: str) -> str:
    m = re.match(r"[A-Za-z]+", str(name))
    return m.group(0) if m else str(name)


def _shaft_color(name: str, shafts: list[str]) -> str:
    return SHAFT_COLS[shafts.index(_shaft(name)) % len(SHAFT_COLS)]


def _axis(theta_deg: float) -> tuple[np.ndarray, np.ndarray]:
    th = np.deg2rad(theta_deg)
    u = np.array([np.cos(th), np.sin(th)])
    return u, np.array([-u[1], u[0]])


def _style_square(ax, L: float) -> None:
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.8)
        sp.set_color("0.28")


def _middle_stim_indices(names: list[str], contacts: np.ndarray, center: np.ndarray, n: int = 4) -> np.ndarray:
    """Pick the four central contacts on the main ICL shaft, falling back to all contacts."""
    icl = [i for i, nm in enumerate(names) if _shaft(nm) == "ICL"]
    pool = icl or list(range(len(names)))
    ranked = sorted(pool, key=lambda i: float(np.linalg.norm(contacts[i] - center)))
    return np.array(sorted(ranked[:n]), dtype=int)


def _electrode_stim_target(pos: np.ndarray, is_E: np.ndarray, stim_contacts: np.ndarray, radius: float) -> np.ndarray:
    d = np.linalg.norm(pos[:, None, :] - np.asarray(stim_contacts, float)[None, :, :], axis=2)
    return np.asarray(is_E, bool) & (d.min(axis=1) <= radius)


def _build_sim(a: argparse.Namespace) -> dict:
    C.KDIR = int(a.k_dir)
    C.PART_MIN = 2 * int(a.k_dir)

    m_real, core_a, core_b = template_source_foci(a.subject, a.montage, a.k_early)
    reg = register_to_sheet(m_real, core_a, core_b, L=a.L, target_inter_core_mm=a.target_inter_core)
    msheet = reg["montage_sheet"]
    src_xy = np.asarray(reg["source_centroid"], float)
    snk_xy = np.asarray(reg["sink_centroid"], float)
    center = np.asarray(reg["center"], float)
    axis_unit = (snk_xy - src_xy) / np.linalg.norm(snk_xy - src_xy)
    theta_rad = np.deg2rad(reg["theta_deg"])

    p = Params(g=3.6, L=a.L, density=a.density, T=a.T, dt=DT, nu_ext_ratio=a.drive, seed=a.seed)
    rng = np.random.default_rng(a.seed)
    pos, labels, NE, NI = place_neurons(p, rng)
    net = build_connectivity_rot(p, pos, labels, NE, NI, rng, theta_EE=theta_rad, AR=a.AR, verbose=False)
    is_E = np.zeros(NE + NI, dtype=bool)
    is_E[:NE] = True

    def core(xy: np.ndarray, seed: int) -> dict:
        return sample_core_field(
            net["pos"],
            is_E,
            xy,
            a.core_r,
            np.random.default_rng(seed),
            core_mean=a.core_mean,
            core_std=a.core_std,
            base_mean=18.0,
        )

    cf1 = core(src_xy, a.seed + 7)
    cf2 = core(snk_xy, a.seed + 8)
    vth = np.minimum(cf1["vth"], cf2["vth"])

    contacts = np.asarray(msheet.contacts, float)
    names = [str(x) for x in msheet.names]
    stim_idx = _middle_stim_indices(names, contacts, center, n=a.n_stim_contacts)
    stim_contacts = contacts[stim_idx]
    target = _electrode_stim_target(net["pos"], is_E, stim_contacts, a.stim_radius)

    valid = C.valid_mask(msheet, net["pos"][:NE], a.L, p.Rr)
    rec = LFPRecorder(p, net["pos"], net["labels"], sites=contacts)
    net["rng"] = np.random.default_rng(a.seed)
    res = simulate_dynamic_vth(
        p,
        net,
        base_vth=vth,
        target_mask=target,
        is_E=is_E,
        on_ms=a.stim_on_ms,
        off_ms=a.stim_off_ms,
        lfp_recorder=rec,
    )
    return dict(
        p=p,
        net=net,
        NE=NE,
        reg=reg,
        core_a=core_a,
        core_b=core_b,
        msheet=msheet,
        contacts=contacts,
        names=names,
        stim_idx=stim_idx,
        stim_contacts=stim_contacts,
        target=target,
        valid=valid,
        axis_unit=axis_unit,
        foci=np.vstack([src_xy, snk_xy]),
        vth=vth[:NE],
        spk=res["E_spk_bool"],
        lfp=res["lfp_trace"],
        times=res["times"],
        intervention_active=res["intervention_active"],
    )


def _read_events(sim: dict) -> tuple[list[dict], dict]:
    spk = sim["spk"]
    posE = sim["net"]["pos"][:sim["NE"]]
    af, bin_w = C.active_fraction(spk, DT, C.BIN_MS)
    nb0 = int(C.BASELINE_MS[0] / bin_w)
    nb1 = int(C.BASELINE_MS[1] / bin_w)
    floor = float(np.percentile(af[nb0:nb1], 95)) if nb1 > nb0 else float(af.min())
    peak = float(af.max())
    bar = floor + C.CAL_FRAC * (peak - floor)
    raw_events = detect_events(af, bin_w, event_on_frac=bar)
    env_f, fdt, _ = snn_event_envelope(spk, posE, sim["msheet"], DT)
    events = []
    for ev in raw_events:
        rd = C.read_event(env_f, fdt, sim["msheet"], sim["valid"], (ev["t_on"], ev["t_off"]), sim["axis_unit"])
        s = int(ev["t_on"] / bin_w)
        e = int(ev["t_off"] / bin_w)
        peak_t = (s + int(np.argmax(af[s:e]))) * bin_w if e > s else ev["t_on"]
        events.append(
            dict(
                t_on=round(ev["t_on"], 1),
                t_off=round(ev["t_off"], 1),
                event_peak_t=round(peak_t, 1),
                returned=bool(ev["returned"]),
                n_part=rd["n_part"],
                sign=rd["sign"],
                axis_err=rd["axis_err"],
                readability=rd["readability"],
                ranks=rd["ranks"],
            )
        )
    return events, dict(floor=floor, peak=peak, bar=bar, bin_w=bin_w)


def _event_onsets(spk: np.ndarray, t_on: float, t_off: float) -> np.ndarray:
    s = int(round(t_on / DT))
    e = int(round(t_off / DT))
    seg = spk[s:e]
    fired = seg.any(axis=0)
    onset = np.full(seg.shape[1], np.nan)
    idx = np.flatnonzero(fired)
    if idx.size:
        onset[idx] = np.argmax(seg[:, idx], axis=0).astype(float) * DT
    return onset


def _clean_prop_events(events: list[dict], a: argparse.Namespace) -> list[dict]:
    return [
        e
        for e in events
        if e["returned"]
        and e["sign"] in (-1.0, 1.0)
        and e["axis_err"] is not None
        and e["axis_err"] < a.max_axis_err
        and e["n_part"] >= C.PART_MIN
    ]


def _local_stim_events(events: list[dict], a: argparse.Namespace) -> list[dict]:
    return [
        e
        for e in events
        if e["returned"]
        and a.stim_on_ms <= e["t_on"] < a.stim_off_ms
        and 0 < e["n_part"] < C.PART_MIN
    ]


def _active_order(sim: dict, events: list[dict]) -> list[int]:
    names = sim["names"]
    contacts = sim["contacts"]
    u = sim["axis_unit"]
    p = np.array([-u[1], u[0]])
    active = set()
    for e in events:
        active.update(n for n, v in (e.get("ranks") or {}).items() if v is not None)
    keep = [i for i, n in enumerate(names) if n in active] or list(range(len(names)))
    pp = np.array([contacts[i] @ u for i in keep])
    qq = np.array([contacts[i] @ p for i in keep])
    order = np.lexsort((qq, pp))
    return [keep[i] for i in order]


def _draw_contacts(ax, sim: dict, show_stim: bool = True, small: bool = False) -> None:
    names = sim["names"]
    contacts = sim["contacts"]
    shafts = sorted(set(_shaft(n) for n in names))
    stim_set = set(int(i) for i in sim["stim_idx"])
    size = 22 if small else 32
    for sh in shafts:
        idx = [i for i, n in enumerate(names) if _shaft(n) == sh]
        c = contacts[idx]
        col = _shaft_color(sh, shafts)
        ax.plot(c[:, 0], c[:, 1], color=col, lw=0.9, alpha=0.48, zorder=5)
        normal = [i for i in idx if i not in stim_set or not show_stim]
        if normal:
            ax.scatter(
                contacts[normal, 0],
                contacts[normal, 1],
                s=size,
                marker="o",
                fc="white",
                ec=col,
                lw=0.9,
                zorder=6,
            )
    if show_stim:
        sc = sim["stim_contacts"]
        ax.scatter(sc[:, 0], sc[:, 1], s=size * 1.65, marker="s", fc=STIM_COL, ec="white", lw=0.7, zorder=8)


def _plot_substrate(ax, sim: dict, a: argparse.Namespace) -> None:
    posE = sim["net"]["pos"][:sim["NE"]]
    low = np.clip(18.0 - sim["vth"], 0.0, None)
    ax.scatter(posE[:, 0], posE[:, 1], c=low, s=1.1, cmap="plasma", vmin=0, vmax=1.2, rasterized=True)
    for lab, xy in zip(("A", "B"), sim["foci"]):
        ax.add_patch(plt.Circle(xy, a.core_r, fill=False, ec="crimson", lw=1.2, ls="--", zorder=7))
        ax.text(
            xy[0],
            xy[1] + a.core_r * 0.33,
            lab,
            color="crimson",
            fontsize=9,
            fontweight="bold",
            ha="center",
            va="center",
            path_effects=[pe.withStroke(linewidth=2.0, foreground="white")],
            zorder=8,
        )
    _draw_contacts(ax, sim, show_stim=True)
    ax.set_title("substrate + stimulation site", fontsize=9.0, fontweight="bold")
    _style_square(ax, a.L)


def _plot_event_map(ax, sim: dict, event: dict, a: argparse.Namespace, title: str, show_stim: bool) -> None:
    posE = sim["net"]["pos"][:sim["NE"]]
    onset = _event_onsets(sim["spk"], event["t_on"], event["t_off"])
    fired = np.isfinite(onset)
    ax.scatter(posE[:, 0], posE[:, 1], c="0.88", s=0.8, alpha=0.55, rasterized=True)
    if fired.any():
        rel = onset.copy()
        rel[fired] -= np.nanmin(rel[fired])
        vmax = max(1.0, float(np.nanpercentile(rel[fired], 98)))
        ax.scatter(posE[fired, 0], posE[fired, 1], c=rel[fired], s=1.15, cmap="viridis", vmin=0, vmax=vmax, rasterized=True)
    _draw_contacts(ax, sim, show_stim=show_stim, small=True)
    ax.set_title(title, fontsize=9.0, fontweight="bold")
    _style_square(ax, a.L)


def _plot_readout(ax, sim: dict, events: list[dict], a: argparse.Namespace) -> dict:
    visible = [e for e in events if e["returned"] and e["t_on"] <= a.window_ms]
    order = _active_order(sim, visible)
    names = sim["names"]
    shafts = sorted(set(_shaft(n) for n in names))
    lfp = np.abs(np.asarray(sim["lfp"], float))
    t = np.asarray(sim["times"], float)
    sel = (t >= 0.0) & (t <= a.window_ms)
    ts = t[sel]
    sub = lfp[sel][:, order].T
    base = np.median(sub, axis=1, keepdims=True)
    scale = np.maximum(sub.max(axis=1, keepdims=True) - base, 1e-9)
    zt = (sub - base) / scale
    y = np.arange(len(order)) * OFF

    ax.axvspan(a.stim_on_ms, min(a.stim_off_ms, a.window_ms), color=STIM_COL, alpha=0.12, lw=0, zorder=0)
    clean = _clean_prop_events(events, a)
    clean_pre = clean_during = clean_after = 0
    local_during = 0

    for row, ci in enumerate(order):
        ax.plot(ts, zt[row] + y[row], color=_shaft_color(names[ci], shafts), lw=0.78, alpha=0.92, zorder=3)

    for e in visible:
        is_clean = e in clean
        is_local = a.stim_on_ms <= e["t_on"] < a.stim_off_ms and 0 < e["n_part"] < C.PART_MIN
        if is_clean:
            if e["t_on"] < a.stim_on_ms:
                clean_pre += 1
            elif e["t_on"] < a.stim_off_ms:
                clean_during += 1
            else:
                clean_after += 1
        if is_local:
            local_during += 1
        if is_clean:
            shade = FWD_SHADE if e["sign"] > 0 else REV_SHADE
            alpha = 0.27
        elif is_local:
            shade = "#bcd7ff"
            alpha = 0.40
        else:
            shade = "0.86"
            alpha = 0.22
        ax.axvspan(e["t_on"], e["t_off"], color=shade, alpha=alpha, lw=0, zorder=1)
        if not is_clean:
            continue
        pts = []
        ranks = e.get("ranks") or {}
        for row, ci in enumerate(order):
            if ranks.get(names[ci]) is None:
                continue
            m = (ts >= e["t_on"]) & (ts <= e["t_off"])
            if m.sum() < 2:
                continue
            pi = np.flatnonzero(m)[int(np.argmax(zt[row][m]))]
            pts.append((ts[pi], zt[row][pi] + y[row]))
            ax.plot(ts[pi], zt[row][pi] + y[row], "o", ms=2.35, mfc="black", mec="white", mew=0.35, zorder=6)
        if len(pts) >= 2:
            px, py = zip(*sorted(pts))
            ax.plot(px, py, "-", color="black", lw=0.75, alpha=0.72, zorder=5)

    ax.axvline(a.stim_on_ms, color=STIM_COL, lw=1.3)
    ax.text(a.stim_on_ms + 20, y[-1] + 0.82, "stim ON", color=STIM_COL, fontsize=9, fontweight="bold", va="top")
    ax.axvline(a.stim_off_ms, color=STIM_COL, lw=1.15, ls="--")
    ax.text(a.stim_off_ms + 20, y[-1] + 0.82, "stim OFF", color=STIM_COL, fontsize=9, fontweight="bold", va="top")

    ax.set_xlim(0, a.window_ms)
    ax.set_yticks(y)
    ax.set_yticklabels([names[i] for i in order], fontsize=7.1)
    for tick, ci in zip(ax.get_yticklabels(), order):
        tick.set_color(_shaft_color(names[ci], shafts))
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=7.5)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xlabel("time (ms)", fontsize=8.0)
    ax.set_box_aspect(1.0 / 2.75)
    ax.set_title(
        f"read-out | clean propagation pre/stim/off: {clean_pre}/{clean_during}/{clean_after}",
        fontsize=9.35,
        fontweight="bold",
        loc="left",
    )
    return dict(pre=clean_pre, during_stim=clean_during, after_stim_off=clean_after,
                local_during_stim=local_during, n_contacts=len(order))


def _compose(sim: dict, events: list[dict], detector: dict, a: argparse.Namespace) -> tuple[Path, Path, dict]:
    clean = _clean_prop_events(events, a)
    pre = [e for e in clean if e["t_on"] < a.stim_on_ms]
    if not pre:
        raise RuntimeError("No clean pre-stim propagation event was detected.")
    pre_event = max(pre, key=lambda e: (e["n_part"], e["readability"] or 0.0))

    local = _local_stim_events(events, a)
    if local:
        local_event = max(local, key=lambda e: e["n_part"])
    else:
        stim_candidates = [e for e in events if e["returned"] and a.stim_on_ms <= e["t_on"] < a.stim_off_ms]
        if not stim_candidates:
            raise RuntimeError("No returned stim-window event was detected for the local-event panel.")
        local_event = min(stim_candidates, key=lambda e: e["n_part"])

    fig = plt.figure(figsize=(18.0, 4.35), facecolor="white")
    gs = gridspec.GridSpec(
        1,
        4,
        width_ratios=[1.0, 1.0, 1.0, 2.75],
        left=0.045,
        right=0.992,
        bottom=0.18,
        top=0.82,
        wspace=0.11,
    )
    _plot_substrate(fig.add_subplot(gs[0, 0]), sim, a)
    _plot_event_map(fig.add_subplot(gs[0, 1]), sim, pre_event, a, "pre-stim propevent", show_stim=False)
    _plot_event_map(fig.add_subplot(gs[0, 2]), sim, local_event, a, "post-stim local event", show_stim=True)
    readout_stats = _plot_readout(fig.add_subplot(gs[0, 3]), sim, events, a)

    fig.text(0.012, 0.935, "A", fontsize=19, fontweight="bold")
    fig.suptitle(
        f"E1146 subject-layout SNN with four-contact stimulation "
        f"(stim ON {a.stim_on_ms:.0f}-{a.stim_off_ms:.0f} ms)",
        fontsize=13.3,
        fontweight="bold",
        y=0.992,
    )

    outdir = ROOT / f"results/paper-ready-figure/{a.fig_name}/figures"
    outdir.mkdir(parents=True, exist_ok=True)
    png = outdir / f"{a.fig_name}.png"
    pdf = outdir / f"{a.fig_name}.pdf"
    fig.savefig(png, dpi=180, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    summary = dict(
        figure=a.fig_name,
        subject=a.subject,
        montage=a.montage,
        config=dict(
            L=a.L,
            density=a.density,
            drive=a.drive,
            T=a.T,
            seed=a.seed,
            core_mean=a.core_mean,
            core_std=a.core_std,
            core_r=a.core_r,
            k_dir=a.k_dir,
            stim_on_ms=a.stim_on_ms,
            stim_off_ms=a.stim_off_ms,
            stim_radius=a.stim_radius,
            stim_contacts=[sim["names"][i] for i in sim["stim_idx"]],
            target_E_cells=int((sim["target"][:sim["NE"]]).sum()),
            window_ms=a.window_ms,
        ),
        placement="template_source (earliest-3 of each template)",
        source_core=sim["core_a"],
        sink_core=sim["core_b"],
        detector={k: round(float(v), 6) for k, v in detector.items()},
        n_events=len(events),
        readout_clean_propagation=readout_stats,
        selected_pre_event={k: pre_event[k] for k in ("t_on", "t_off", "n_part", "sign", "axis_err")},
        selected_stim_window_event={k: local_event[k] for k in ("t_on", "t_off", "n_part", "sign", "axis_err")},
        notes=[
            "Dynamic-threshold simulation; not plotting-only.",
            "Stimulus is drawn only as four middle ICL contacts, not as a band.",
            "During-stim residual events can remain local; the intended readout is clean propagation pre/stim/off.",
        ],
    )
    meta = outdir / f"{a.fig_name}_metadata.json"
    meta.write_text(json.dumps(summary, indent=2))
    _write_readme(outdir, a, summary)
    return png, meta, summary


def _write_readme(outdir: Path, a: argparse.Namespace, summary: dict) -> None:
    stats = summary["readout_clean_propagation"]
    stim_contacts = ", ".join(summary["config"]["stim_contacts"])
    text = f"""# {a.fig_name}

### {a.fig_name}.png / .pdf

E1146 真实电极布局上的 subject-specific SNN 刺激示意图。左侧为同一 template-source 双核底物和刺激位点；中间两块分别显示刺激前的传播事件和刺激打开后的局部事件；右侧为同一段 readout，蓝色阴影为刺激打开窗口，虚线为刺激关闭。

**关注点**：刺激只放在中轴附近 4 个真实触点 `{stim_contacts}`，不是条带。clean propagation 计数 pre/stim/off = {stats['pre']}/{stats['during_stim']}/{stats['after_stim_off']}；因此这张图只支持“刺激 ON 期间可读传播被压掉/局部化，OFF 后可恢复”的示意，不支持“所有活动被永久消除”。
"""
    (outdir / "README.md").write_text(text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="epilepsiae_1146")
    ap.add_argument("--montage", default="narrow")
    ap.add_argument("--L", type=float, default=20.0)
    ap.add_argument("--density", type=float, default=100.0)
    ap.add_argument("--AR", type=float, default=2.0)
    ap.add_argument("--drive", type=float, default=0.6)
    ap.add_argument("--T", type=float, default=5000.0)
    ap.add_argument("--core-mean", type=float, default=17.5)
    ap.add_argument("--core-std", type=float, default=1.0)
    ap.add_argument("--core-r", type=float, default=2.87)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--k-dir", type=int, default=2)
    ap.add_argument("--k-early", type=int, default=3)
    ap.add_argument("--target-inter-core", type=float, default=None)
    ap.add_argument("--stim-on-ms", type=float, default=1200.0)
    ap.add_argument("--stim-off-ms", type=float, default=2400.0)
    ap.add_argument("--stim-radius", type=float, default=2.0)
    ap.add_argument("--n-stim-contacts", type=int, default=4)
    ap.add_argument("--window-ms", type=float, default=5000.0)
    ap.add_argument("--max-axis-err", type=float, default=25.0)
    ap.add_argument("--fig-name", default="fig_subject_snn_epilepsiae_1146_stimulation")
    a = ap.parse_args()

    C._engine_guard()
    sim = _build_sim(a)
    events, detector = _read_events(sim)
    png, meta, summary = _compose(sim, events, detector, a)
    print(f"wrote {png}")
    print(f"wrote {meta}")
    print(json.dumps({
        "n_events": summary["n_events"],
        "stim_contacts": summary["config"]["stim_contacts"],
        "target_E_cells": summary["config"]["target_E_cells"],
        "readout_clean_propagation": summary["readout_clean_propagation"],
        "selected_pre_event": summary["selected_pre_event"],
        "selected_stim_window_event": summary["selected_stim_window_event"],
    }, indent=2))


if __name__ == "__main__":
    main()
