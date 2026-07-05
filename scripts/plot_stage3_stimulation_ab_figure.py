"""AB-only Stage-3 stimulation illustration on the s3_brakeoff substrate.

This is an intervention figure, not a new formal gate. It reuses the
`core_model_s3_brakeoff` substrate settings by default, then turns on continuous
E-only high-threshold stimulation around the four middle contacts (A2/A3/B2/B3).

The figure is a single-row AB-only composition: substrate + stimulation contacts,
one pre-stim propagation event, one post-stim local event, and one fused read-out
trace. No C/KMeans panel is produced.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

sys.path.insert(0, os.getcwd())
import scripts.run_sef_hfo_snn_cm_spontaneous_readout as C  # noqa: E402
from src.sef_hfo_axial_intervention import simulate_dynamic_vth  # noqa: E402

OUT = Path("results/topic4_sef_hfo/observation_layer/snn_cm_spontaneous")
FIG = OUT / "figures"
DT = C.DT
OFF = 1.35


def _central_stim_contact_indices(m, center, n_per_shaft=2):
    """Middle stimulation contacts: the two closest-to-center contacts on each shaft."""
    names = [str(n) for n in m.names]
    groups = {}
    for i, name in enumerate(names):
        groups.setdefault(name[0], []).append(i)
    chosen = []
    for shaft in sorted(groups):
        idx = groups[shaft]
        idx = sorted(idx, key=lambda j: np.linalg.norm(np.asarray(m.contacts[j]) - center))
        chosen.extend(idx[:n_per_shaft])
    return np.array(sorted(chosen), dtype=int)


def _electrode_stim_target(pos, is_E, stim_contacts, radius):
    """Full-network mask: E cells within radius of any selected stimulation contact."""
    pos = np.asarray(pos, float)
    is_E = np.asarray(is_E, bool)
    d = np.linalg.norm(pos[:, None, :] - np.asarray(stim_contacts, float)[None, :, :], axis=2)
    return is_E & (d.min(axis=1) <= radius)


def _clean_events(events, sign):
    return [
        e for e in events
        if e["returned"]
        and e["sign"] == sign
        and e["axis_err"] is not None and e["axis_err"] < 25
        and e["n_part"] >= C.PART_MIN
    ]


def _event_onsets(spk, t_on, t_off):
    s, e = int(round(t_on / DT)), int(round(t_off / DT))
    seg = spk[s:e]
    fired = seg.any(axis=0)
    onset = np.full(seg.shape[1], np.nan)
    idx = np.flatnonzero(fired)
    if idx.size:
        onset[idx] = np.argmax(seg[:, idx], axis=0).astype(float) * DT
    return onset


def _simulate(a):
    theta_rad = np.deg2rad(a.theta)
    axis_unit = np.array([np.cos(theta_rad), np.sin(theta_rad)])
    p = C.Params(g=3.6, L=a.L, density=a.density, T=a.T, dt=DT, nu_ext_ratio=a.drive, seed=a.seed)
    rng = np.random.default_rng(a.seed)
    pos, labels, NE, NI = C.place_neurons(p, rng)
    center = np.array([a.L / 2.0, a.L / 2.0])
    half = a.L / 2.0
    net = C.build_connectivity_rot(p, pos, labels, NE, NI, rng, theta_EE=theta_rad, AR=a.AR, verbose=False)
    vth, core_mask, foci, core_masks = C.build_lesion_vth(
        net, NE, axis_unit, center, half, "twoend_equal",
        a.core_mean, a.core_std, a.core_r, a.dephase, a.seed,
        sep_frac=a.sep_frac,
    )
    is_E = np.zeros(NE + NI, bool)
    is_E[:NE] = True
    m = C.montage(center, a.theta, 0.0, a.nc)
    stim_idx = _central_stim_contact_indices(m, center)
    stim_contacts = np.asarray(m.contacts)[stim_idx]
    target = _electrode_stim_target(net["pos"], is_E, stim_contacts, a.stim_radius)
    valid = C.valid_mask(m, net["pos"][:NE], a.L, p.Rr)
    rec = C.LFPRecorder(p, net["pos"], net["labels"], sites=m.contacts)
    net["rng"] = np.random.default_rng(a.seed)
    res = simulate_dynamic_vth(
        p, net, base_vth=vth, target_mask=target, is_E=is_E,
        on_ms=a.stim_on_ms, off_ms=a.stim_off_ms, lfp_recorder=rec,
    )
    return dict(
        p=p, net=net, NE=NE, axis_unit=axis_unit, center=center, foci=np.asarray(foci),
        vth=vth, core_mask=core_mask, target=target, m=m, valid=valid,
        stim_contact_indices=stim_idx, stim_contacts=stim_contacts,
        spk=res["E_spk_bool"], lfp=res["lfp_trace"], times=res["times"],
    )


def _read_events(sim):
    spk = sim["spk"]
    m = sim["m"]
    posE = sim["net"]["pos"][:sim["NE"]]
    af, bin_w = C.active_fraction(spk, DT, C.BIN_MS)
    nb0, nb1 = int(C.BASELINE_MS[0] / bin_w), int(C.BASELINE_MS[1] / bin_w)
    floor = float(np.percentile(af[nb0:nb1], 95)) if nb1 > nb0 else float(af.min())
    peak = float(af.max())
    bar = floor + C.CAL_FRAC * (peak - floor)
    raw_events = C.detect_events(af, bin_w, event_on_frac=bar)
    env_f, fdt, _ = C.snn_event_envelope(spk, posE, m, DT)
    out = []
    for ev in raw_events:
        rd = C.read_event(env_f, fdt, m, sim["valid"], (ev["t_on"], ev["t_off"]), sim["axis_unit"])
        out.append(dict(
            t_on=round(ev["t_on"], 1),
            t_off=round(ev["t_off"], 1),
            returned=bool(ev["returned"]),
            n_part=rd["n_part"],
            axis_err=rd["axis_err"],
            sign=rd["sign"],
            ranks=rd["ranks"],
        ))
    return out, dict(floor=floor, peak=peak, bar=bar)


def _active_contacts(names, contacts, axis_unit, events):
    active = set()
    for e in events:
        active.update(nm for nm, v in (e.get("ranks") or {}).items() if v is not None)
    keep = [i for i, n in enumerate(names) if n in active]
    if not keep:
        keep = list(range(len(names)))
    p = np.array([-axis_unit[1], axis_unit[0]])
    pp = np.array([contacts[i] @ axis_unit for i in keep])
    qq = np.array([contacts[i] @ p for i in keep])
    order = np.lexsort((qq, pp))
    return [keep[i] for i in order]


def _plot_substrate(ax, sim, a):
    posE = sim["net"]["pos"][:sim["NE"]]
    v = sim["vth"][:sim["NE"]]
    sc = ax.scatter(posE[:, 0], posE[:, 1], c=np.clip(18.0 - v, 0, None), s=1.0,
                    cmap="magma", vmin=0, vmax=1.2, rasterized=True)
    for f in sim["foci"]:
        ax.add_patch(plt.Circle(f, a.core_r, ec="crimson", fc="none", lw=1.4, ls="--", zorder=7))
    contacts = np.asarray(sim["m"].contacts)
    stim_idx = set(int(i) for i in sim["stim_contact_indices"])
    nonstim = [i for i in range(len(contacts)) if i not in stim_idx]
    ax.scatter(contacts[nonstim, 0], contacts[nonstim, 1], s=16, marker="s",
               color="white", edgecolor="0.25", lw=0.6, zorder=8)
    ax.scatter(sim["stim_contacts"][:, 0], sim["stim_contacts"][:, 1], s=45, marker="s",
               color="#2f80ed", edgecolor="white", lw=0.8, zorder=9)
    ax.set_title("substrate + stimulation site", fontsize=8.8, fontweight="bold")
    ax.set_xlim(0, a.L); ax.set_ylim(0, a.L); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    return sc


def _plot_event_map(ax, sim, event, a, title, show_stim_contacts=True):
    posE = sim["net"]["pos"][:sim["NE"]]
    onset = _event_onsets(sim["spk"], event["t_on"], event["t_off"])
    fired = np.isfinite(onset)
    ax.scatter(posE[:, 0], posE[:, 1], c="0.90", s=0.7, rasterized=True)
    if fired.any():
        ax.scatter(posE[fired, 0], posE[fired, 1], c=onset[fired], s=1.1, cmap="viridis",
                   vmin=0, vmax=max(1.0, np.nanpercentile(onset[fired], 98)), rasterized=True)
    if show_stim_contacts:
        ax.scatter(sim["stim_contacts"][:, 0], sim["stim_contacts"][:, 1], s=35, marker="s",
                   color="#2f80ed", edgecolor="white", lw=0.6, zorder=6)
    ax.set_title(title, fontsize=8.8, fontweight="bold")
    ax.set_xlim(0, a.L); ax.set_ylim(0, a.L); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])


def _plot_train(ax, sim, events, sign, a, title):
    names = [str(x) for x in sim["m"].names]
    contacts = np.asarray(sim["m"].contacts)
    row_events = _clean_events(events, sign)
    combined = _active_contacts(names, contacts, sim["axis_unit"], row_events)
    lfp = np.asarray(sim["lfp"]).T
    t = np.asarray(sim["times"])
    sel = (t >= 0) & (t <= a.window_ms)
    ts = t[sel]
    sub = lfp[combined][:, sel]
    base = np.median(sub, axis=1, keepdims=True)
    scale = np.maximum(sub.max(axis=1, keepdims=True) - base, 1e-9)
    zt = (sub - base) / scale
    y = np.arange(len(combined)) * OFF

    ax.axvspan(a.stim_on_ms, min(a.stim_off_ms, a.window_ms), color="#2f80ed", alpha=0.13, lw=0, zorder=0)
    for i, ci in enumerate(combined):
        col = "#1f9e9e" if names[ci].startswith("B") else "#e8743b"
        ax.plot(ts, zt[i] + y[i], color=col, lw=0.85, alpha=0.95)

    pre = post = 0
    for e in row_events:
        if e["t_on"] < 0 or e["t_on"] > a.window_ms:
            continue
        if e["t_on"] < a.stim_on_ms:
            pre += 1
        else:
            post += 1
        ax.axvspan(e["t_on"], e["t_off"], color="0.82", alpha=0.50, lw=0, zorder=1)
        pts = []
        ranks = e.get("ranks") or {}
        for i, ci in enumerate(combined):
            if ranks.get(names[ci]) is None:
                continue
            m = (ts >= e["t_on"]) & (ts <= e["t_off"])
            if m.sum() < 2:
                continue
            pi = np.flatnonzero(m)[int(np.argmax(zt[i][m]))]
            pts.append((ts[pi], zt[i][pi] + y[i]))
            ax.plot(ts[pi], zt[i][pi] + y[i], "o", ms=2.7, mfc="k", mec="white", mew=0.35, zorder=5)
        if len(pts) >= 2:
            px, py = zip(*sorted(pts))
            ax.plot(px, py, "-", color="k", lw=0.9, alpha=0.72, zorder=4)

    ax.axvline(a.stim_on_ms, color="#2f80ed", lw=1.4)
    ax.text(a.stim_on_ms + 22, y[-1] + 0.85, "stim ON", color="#2f80ed",
            fontsize=9, fontweight="bold", va="top")
    ax.set_xlim(0, a.window_ms)
    ax.set_yticks(y)
    ax.set_yticklabels([names[i] for i in combined], fontsize=7.4)
    for tick, ci in zip(ax.get_yticklabels(), combined):
        tick.set_color("#1f9e9e" if names[ci].startswith("B") else "#e8743b")
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=7.6)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xlabel("time (ms)", fontsize=8)
    ax.set_title(f"{title} | clean events pre/post stim: {pre}/{post}", fontsize=9.1, fontweight="bold", loc="left")
    return dict(pre=pre, post=post, n_clean=len(row_events), n_contacts=len(combined))


def _clean_prop_events(events):
    return [
        e for e in events
        if e["returned"]
        and e["sign"] in (1.0, -1.0)
        and e["axis_err"] is not None and e["axis_err"] < 25
        and e["n_part"] >= C.PART_MIN
    ]


def _local_post_events(events, stim_on_ms, stim_off_ms):
    return [
        e for e in events
        if e["returned"]
        and e["t_on"] >= stim_on_ms
        and e["t_on"] < stim_off_ms
        and 0 < e["n_part"] < C.PART_MIN
    ]


def _plot_readout(ax, sim, events, a):
    names = [str(x) for x in sim["m"].names]
    contacts = np.asarray(sim["m"].contacts)
    visible_events = [e for e in events if e["returned"] and e["t_on"] <= a.window_ms]
    combined = _active_contacts(names, contacts, sim["axis_unit"], visible_events)
    lfp = np.asarray(sim["lfp"]).T
    t = np.asarray(sim["times"])
    sel = (t >= 0) & (t <= a.window_ms)
    ts = t[sel]
    sub = lfp[combined][:, sel]
    base = np.median(sub, axis=1, keepdims=True)
    scale = np.maximum(sub.max(axis=1, keepdims=True) - base, 1e-9)
    zt = (sub - base) / scale
    y = np.arange(len(combined)) * OFF

    stim_off_plot = min(a.stim_off_ms, a.window_ms)
    ax.axvspan(a.stim_on_ms, stim_off_plot, color="#2f80ed", alpha=0.12, lw=0, zorder=0)
    for i, ci in enumerate(combined):
        col = "#1f9e9e" if names[ci].startswith("B") else "#e8743b"
        ax.plot(ts, zt[i] + y[i], color=col, lw=0.85, alpha=0.95)

    clean_props = _clean_prop_events(events)
    clean_pre = clean_during = clean_after = 0
    for e in visible_events:
        is_clean = e in clean_props
        is_local_post = a.stim_on_ms <= e["t_on"] < a.stim_off_ms and 0 < e["n_part"] < C.PART_MIN
        if is_clean:
            if e["t_on"] < a.stim_on_ms:
                clean_pre += 1
            elif e["t_on"] < a.stim_off_ms:
                clean_during += 1
            else:
                clean_after += 1
        color = "#d8d8d8" if e["t_on"] < a.stim_on_ms else ("#bcd7ff" if is_local_post else "#e6e6e6")
        alpha = 0.55 if (is_clean or is_local_post) else 0.28
        ax.axvspan(e["t_on"], e["t_off"], color=color, alpha=alpha, lw=0, zorder=1)
        if not is_clean:
            continue
        pts = []
        ranks = e.get("ranks") or {}
        for i, ci in enumerate(combined):
            if ranks.get(names[ci]) is None:
                continue
            m = (ts >= e["t_on"]) & (ts <= e["t_off"])
            if m.sum() < 2:
                continue
            pi = np.flatnonzero(m)[int(np.argmax(zt[i][m]))]
            pts.append((ts[pi], zt[i][pi] + y[i]))
            ax.plot(ts[pi], zt[i][pi] + y[i], "o", ms=2.6, mfc="k", mec="white", mew=0.35, zorder=5)
        if len(pts) >= 2:
            px, py = zip(*sorted(pts))
            ax.plot(px, py, "-", color="k", lw=0.9, alpha=0.72, zorder=4)

    ax.axvline(a.stim_on_ms, color="#2f80ed", lw=1.4)
    ax.text(a.stim_on_ms + 22, y[-1] + 0.88, "stim ON", color="#2f80ed",
            fontsize=9, fontweight="bold", va="top")
    if np.isfinite(a.stim_off_ms) and a.stim_off_ms <= a.window_ms:
        ax.axvline(a.stim_off_ms, color="#2f80ed", lw=1.2, ls="--")
        ax.text(a.stim_off_ms + 22, y[-1] + 0.88, "stim OFF", color="#2f80ed",
                fontsize=9, fontweight="bold", va="top")
    ax.set_xlim(0, a.window_ms)
    ax.set_yticks(y)
    ax.set_yticklabels([names[i] for i in combined], fontsize=7.4)
    for tick, ci in zip(ax.get_yticklabels(), combined):
        tick.set_color("#1f9e9e" if names[ci].startswith("B") else "#e8743b")
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=7.6)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xlabel("time (ms)", fontsize=8)
    ax.set_box_aspect(1.0 / 2.75)
    ax.set_title(
        f"read-out | clean propagation pre/stim/off: {clean_pre}/{clean_during}/{clean_after}",
        fontsize=9.4, fontweight="bold", loc="left",
    )
    return dict(pre=clean_pre, during_stim=clean_during, after_stim_off=clean_after,
                n_contacts=len(combined))


def _compose(sim, events, detector, a):
    fwd = _clean_events(events, 1.0)
    rev = _clean_events(events, -1.0)
    if not fwd:
        raise RuntimeError(f"Need at least one clean forward event for the pre-stim panel; got {len(fwd)}")
    rep_fwd = next((e for e in fwd if e["t_on"] < a.stim_on_ms), fwd[0])
    local_post = _local_post_events(events, a.stim_on_ms, a.stim_off_ms)
    if not local_post:
        post_candidates = [e for e in events if e["returned"] and e["t_on"] >= a.stim_on_ms]
        if not post_candidates:
            raise RuntimeError("Need at least one returned post-stim event for the local-event panel")
        local_event = max(post_candidates, key=lambda e: e["n_part"])
    else:
        local_event = max(local_post, key=lambda e: e["n_part"])

    fig = plt.figure(figsize=(18.0, 4.35), facecolor="white")
    gs = gridspec.GridSpec(
        1, 4,
        width_ratios=[1.0, 1.0, 1.0, 2.75],
        wspace=0.10,
        left=0.045,
        right=0.992,
        bottom=0.18,
        top=0.82,
    )
    ax = fig.add_subplot(gs[0, 0]); _plot_substrate(ax, sim, a)
    ax = fig.add_subplot(gs[0, 1]); _plot_event_map(
        ax, sim, rep_fwd, a, "pre-stim propevent", show_stim_contacts=False)
    ax = fig.add_subplot(gs[0, 2]); _plot_event_map(
        ax, sim, local_event, a, "post-stim local event", show_stim_contacts=True)
    ax = fig.add_subplot(gs[0, 3])
    stats_readout = _plot_readout(ax, sim, events, a)

    fig.text(0.012, 0.935, "A", fontsize=19, fontweight="bold")
    stim_off_label = "end" if np.isinf(a.stim_off_ms) else f"{a.stim_off_ms:.0f} ms"
    fig.suptitle(
        "Stage-3 two-focus substrate with four-contact E-only stimulation "
        f"(stim ON at {a.stim_on_ms:.0f} ms to {stim_off_label})",
        fontsize=13.5, fontweight="bold", y=0.992,
    )
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / a.out_name
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    summary = dict(
        figure=str(out),
        config=dict(
            L=a.L, density=a.density, theta=a.theta, AR=a.AR, drive=a.drive, T=a.T,
            lesion="twoend_equal", core_mean=a.core_mean, core_std=a.core_std,
            core_r=a.core_r, sep_frac=a.sep_frac, dephase=a.dephase, seed=a.seed,
            stim_on_ms=a.stim_on_ms,
            stim_off_ms=(None if np.isinf(a.stim_off_ms) else a.stim_off_ms),
            stim_radius=a.stim_radius,
            stim_contacts=[str(sim["m"].names[i]) for i in sim["stim_contact_indices"]],
            window_ms=a.window_ms,
        ),
        detector={k: round(float(v), 6) for k, v in detector.items()},
        total_events=len(events),
        readout_clean_propagation=stats_readout,
        clean_forward_pre=sum(1 for e in fwd if e["t_on"] < a.stim_on_ms),
        clean_reverse_pre=sum(1 for e in rev if e["t_on"] < a.stim_on_ms),
        post_local_events=len(local_post),
        selected_pre_event=dict(t_on=rep_fwd["t_on"], t_off=rep_fwd["t_off"],
                                n_part=rep_fwd["n_part"], sign=rep_fwd["sign"]),
        selected_post_local_event=dict(t_on=local_event["t_on"], t_off=local_event["t_off"],
                                       n_part=local_event["n_part"], sign=local_event["sign"]),
    )
    summary_path = out.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2))
    return out, summary_path, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=float, default=20.0)
    ap.add_argument("--density", type=float, default=100.0)
    ap.add_argument("--theta", type=float, default=45.0)
    ap.add_argument("--AR", type=float, default=2.0)
    ap.add_argument("--drive", type=float, default=0.6)
    ap.add_argument("--T", type=float, default=3400.0)
    ap.add_argument("--core-mean", type=float, default=17.5)
    ap.add_argument("--core-std", type=float, default=1.0)
    ap.add_argument("--core-r", type=float, default=1.5)
    ap.add_argument("--sep-frac", type=float, default=0.7)
    ap.add_argument("--dephase", type=float, default=0.3)
    ap.add_argument("--nc", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--stim-on-ms", type=float, default=1200.0)
    ap.add_argument("--stim-off-ms", type=float, default=2400.0)
    ap.add_argument("--stim-radius", type=float, default=2.0,
                    help="mm; E cells within this radius of A2/A3/B2/B3 are clamped after stim ON")
    ap.add_argument("--window-ms", type=float, default=3400.0)
    ap.add_argument("--out-name", default="core_model_s3_brakeoff_stim_ab.png")
    a = ap.parse_args()
    C._engine_guard()
    sim = _simulate(a)
    events, detector = _read_events(sim)
    out, summary_path, summary = _compose(sim, events, detector, a)
    print(f"wrote {out}")
    print(f"wrote {summary_path}")
    print(json.dumps({k: summary[k] for k in (
        "total_events", "readout_clean_propagation", "post_local_events",
        "selected_pre_event", "selected_post_local_event")}, indent=2))


if __name__ == "__main__":
    main()
