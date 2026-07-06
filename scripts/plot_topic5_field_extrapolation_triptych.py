#!/usr/bin/env python3
"""Topic 5 energy-field extrapolation — paper-ready per-subject figure (PREVIEW).

Question: does the interictal propagation pattern found from the NARROW core, EXTRAPOLATED to the
interictally-"hidden" contacts (broad∖narrow), predict their seizure ENERGY better than each hidden
contact's OWN (broad) interictal order?  Yardstick = seizure energy (independent ground truth).
This is exactly the cohort's F_core_only (1-3) vs C1 (2-3); the |r| in the title are the cohort's
per-seizure statistics (self-consistent with energy_field_extrapolation_FINAL).

Three aligned panels (shared plane / axes / positions, compact) — is B or C more like A?
  (A) seizure ENERGY                     (viridis 0=low→1=high)              — the yardstick
  (B) core-field EXTRAPOLATION (1-3)      order predicted at each contact,    — oriented to A
      hidden filled by the CORE field's prediction (= F_core_only predictor)
  (C) each contact's OWN broad order (2-3)                                    — oriented to A
Hidden (broad∖narrow) = crimson squares in every panel. SOZ = black ring (overlay only).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from scripts.plot_contact_plane_static import (_subject_display_frame, _display_points,
                                               _smooth_rank_field_mm, _attach_real_coords)
from scripts.plot_topic5_axis_alignment_fields import _ictal_activation, _rank01
from src.topic5_field_extrapolation import (channel_names_from_pool, predicted_interictal_order,
                                            DEF_NARROW_POOL)

BROAD_DIR = _ROOT / "results/spatial_modulation/propagation_geometry_broad/observation_readout/real_subjects"
COH_DIR = _ROOT / "results/topic5_ictal_recruitment/field_extrapolation/cohort_per_subject"
OUT = _ROOT / "results/topic5_ictal_recruitment/field_extrapolation/figures/triptych_preview"
ACT_KEY = {"broadband": "bb_auc", "hfa": "hfa_auc"}
ACT_LBL = {"broadband": "broadband power 1–45 Hz, 0–10 s", "hfa": "fast activity 60–100 Hz, 0–10 s"}
HID = "crimson"


def _smooth(frame, xs, ys, vals, sup):
    m = np.isfinite(vals)
    _, _, T, _, _ = _smooth_rank_field_mm(xs[m], ys[m], vals[m], sup[m],
                                          frame["xlim"], frame["ylim"], frame["sigma_mm"])
    return T


def _draw(ax, frame, T, xs, ys, cvals, hidden, soz, title, fs=15):
    xlim, ylim = frame["xlim"], frame["ylim"]
    im = ax.imshow(T, origin="lower", extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
                   aspect="equal", cmap="viridis", vmin=0, vmax=1)
    m = np.isfinite(cvals)
    nh = (~hidden) & m
    if nh.any():
        ax.scatter(xs[nh], ys[nh], c=cvals[nh], cmap="viridis", vmin=0, vmax=1, marker="o", s=95,
                   edgecolors=["k" if z else "white" for z in soz[nh]],
                   linewidths=[2.4 if z else 0.9 for z in soz[nh]], zorder=3)
    hv = hidden & m
    if hv.any():
        ax.scatter(xs[hv], ys[hv], c=cvals[hv], cmap="viridis", vmin=0, vmax=1, marker="s", s=108,
                   edgecolors=HID, linewidths=2.6, zorder=4)
    hn = hidden & ~m
    if hn.any():
        ax.scatter(xs[hn], ys[hn], facecolors="none", marker="s", s=108, edgecolors=HID,
                   linewidths=2.6, zorder=4)
    ax.set_title(title, fontsize=fs)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([]); ax.set_yticks([])
    return im


def _rho(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    mk = np.isfinite(a) & np.isfinite(b)
    if mk.sum() < 3 or np.std(a[mk]) < 1e-12 or np.std(b[mk]) < 1e-12:
        return np.nan
    return float(spearmanr(a[mk], b[mk]).correlation)


def plot_subject(ds_sid, activation="broadband"):
    bf = BROAD_DIR / f"{ds_sid}_t_a.json"
    if not bf.exists():
        print(f"  {ds_sid}: no broad geometry"); return None
    broad = json.loads(bf.read_text())
    if not broad.get("channels"):
        return None
    _attach_real_coords([broad])
    frame = _subject_display_frame([broad])
    if frame is None:
        print(f"  {ds_sid}: no display frame"); return None
    xs, ys = _display_points(broad, frame)
    names = [c["name"] for c in broad["channels"]]
    sup = np.array([c["support"] for c in broad["channels"]], float)
    soz = np.array([bool(c.get("is_soz")) for c in broad["channels"]])
    own_a = np.array([c["typical_rank"] for c in broad["channels"]], float)        # 2-3 predictor (t_a)
    narrow_set = set(channel_names_from_pool(ds_sid, str(_ROOT / DEF_NARROW_POOL)))
    hidden = np.array([n not in narrow_set for n in names])

    act = _ictal_activation(ds_sid, ACT_KEY[activation])
    if not act:
        print(f"  {ds_sid}: no ictal cache"); return None
    ict = _rank01([act.get(n, np.nan) for n in names])

    # 1-3 predictor = core field (broad ranks restricted to core) extrapolated to every contact.
    # rank01 (the statistic is a RANK corr; raw predicted_interictal_order values are compressed).
    # A/B two templates (cohort uses A-B-max): pick the template that better tracks energy at hidden.
    def _core_pred(rec):
        p = predicted_interictal_order(rec, [c["name"] for c in rec["channels"]],
                                       loo=False, core_names=narrow_set)
        return _rank01(np.array([p.get(n, np.nan) for n in names], float))
    core_a = _core_pred(broad)
    core_b = own_b = None
    tb_f = BROAD_DIR / f"{ds_sid}_t_b.json"
    if tb_f.exists():
        tb = json.loads(tb_f.read_text())
        if tb.get("channels"):
            core_b = _core_pred(tb)
            ob = {c["name"]: c["typical_rank"] for c in tb["channels"]}
            own_b = np.array([ob.get(n, np.nan) for n in names], float)

    def _pick(cands):
        best, br = None, -1.0
        for c in cands:
            if c is None:
                continue
            r = abs(_rho(c[hidden], ict[hidden]))
            if np.isfinite(r) and r > br:
                best, br = c, r
        return best, (br if br >= 0 else np.nan)
    core, Fd = _pick([core_a, core_b])          # descriptive single-map A/B-max
    own, Cd = _pick([own_a, own_b])

    coreO = (1 - core) if (_rho(core, ict) or 0) < 0 else core
    ownO = (1 - own) if (_rho(own, ict) or 0) < 0 else own
    core_src = ~hidden                          # sources of the core field = core (non-hidden) contacts
    T_A = _smooth(frame, xs, ys, ict, sup)
    T_B = _smooth(frame, xs[core_src], ys[core_src], coreO[core_src], sup[core_src])
    T_C = _smooth(frame, xs, ys, ownO, sup)

    # authoritative cohort statistics (per-seizure LOO A-B-max) — F_core_only (1-3) vs C1 (2-3)
    cj = COH_DIR / f"{ds_sid}__{ACT_KEY[activation]}.json"
    Fc = C1 = None
    if cj.exists():
        d = json.loads(cj.read_text())
        Fc, C1 = d.get("F_core_only"), d.get("C1")

    def _verd(a, b):
        if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
            return "—"
        return "1-3 WINS" if a > b + 0.03 else "2-3 WINS" if b > a + 0.03 else "tie"
    verdict = _verd(Fc, C1)

    # (D) per-seizure |corr(prediction, seizure energy)| at hidden — AUTHORITATIVE, taken directly
    # from the cohort per-subject series (median = F_core_only / C1); self-consistent, not a field.
    s_core = np.array((d.get("series", {}).get("F_core") if cj.exists() else []) or [], float)
    s_own = np.array((d.get("series", {}).get("C1") if cj.exists() else []) or [], float)

    pretty = ds_sid.replace("epilepsiae_", "E").replace("yuquan_", "Y-")
    FS_T, FS_L, FS_TK = 17, 15, 14                       # big fonts (report holds the prose)
    fig = plt.figure(figsize=(23.5, 6.9))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 0.035, 1.05],
                          wspace=0.07, left=0.012, right=0.985, top=0.9, bottom=0.16)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    cax = fig.add_subplot(gs[0, 3])
    axd = fig.add_subplot(gs[0, 4])
    imA = _draw(axes[0], frame, T_A, xs, ys, ict, hidden, soz, "(A) seizure ENERGY  (yardstick)", FS_T)
    _draw(axes[1], frame, T_B, xs, ys, coreO, hidden, soz, "(B) 1-3  core-extrapolation", FS_T)
    _draw(axes[2], frame, T_C, xs, ys, ownO, hidden, soz, "(C) 2-3  own order", FS_T)
    cb = fig.colorbar(imA, cax=cax, ticks=[0, 0.5, 1.0])
    cb.ax.tick_params(labelsize=FS_TK); cb.set_label("rank 0 → 1", fontsize=FS_L)
    axes[0].scatter([], [], marker="s", facecolors="none", edgecolors=HID, linewidths=2.6,
                    label="hidden (broad∖narrow)")
    axes[0].scatter([], [], marker="o", facecolors="0.7", edgecolors="white", label="narrow core")
    axes[0].legend(loc="upper left", fontsize=FS_L - 1, framealpha=0.9)

    fc_c, own_c = "#1f6feb", "#e8710a"
    if s_core.size and s_own.size:
        m = np.isfinite(s_core) & np.isfinite(s_own)
        sc, so = s_core[m], s_own[m]
        rng = np.random.default_rng(0)
        for a, b in zip(sc, so):
            axd.plot([1, 2], [a, b], color="0.78", lw=0.9, zorder=1)
        axd.scatter(1 + rng.uniform(-.06, .06, sc.size), sc, s=46, color=fc_c, zorder=3,
                    edgecolors="white", linewidths=0.5)
        axd.scatter(2 + rng.uniform(-.06, .06, so.size), so, s=46, color=own_c, zorder=3,
                    edgecolors="white", linewidths=0.5)
        bp = axd.boxplot([sc, so], positions=[1, 2], widths=0.5, showfliers=False,
                         medianprops=dict(color="k", lw=2.8), patch_artist=True, zorder=2)
        for patch, c in zip(bp["boxes"], (fc_c, own_c)):
            patch.set(facecolor=c, alpha=0.18)
        axd.set_xticks([1, 2])
        axd.set_xticklabels(["1-3\ncore-extrap", "2-3\nown order"], fontsize=FS_L)
        axd.set_ylabel("|corr(prediction, seizure energy)|  per seizure", fontsize=FS_L)
        axd.set_ylim(0, 1); axd.set_xlim(0.5, 2.5)
        axd.tick_params(axis="y", labelsize=FS_TK); axd.grid(axis="y", alpha=0.3)
    else:
        axd.text(0.5, 0.5, "no per-seizure series", ha="center", va="center", transform=axd.transAxes)
        axd.set_xticks([])
    vt = (f"(D) per-seizure |corr|  (n={s_core.size})\n1-3={Fc:.2f}  vs  2-3={C1:.2f}  →  {verdict}"
          if isinstance(Fc, (int, float)) else "(D) per-seizure")
    axd.set_title(vt, fontsize=FS_T - 2)

    fig.text(0.5, 0.022, f"{pretty} · {activation} · n_hidden={int(hidden.sum())}  ·  "
             "red square = hidden (broad∖narrow)  ·  black ring = SOZ (overlay only)",
             ha="center", fontsize=FS_L - 1, color="0.4")
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{ds_sid}_extrapVown_{activation}.png"
    fig.savefig(out, dpi=135)
    plt.close(fig)
    print(f"  wrote {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="*", default=["epilepsiae_583"])
    ap.add_argument("--activation", choices=list(ACT_KEY), default="broadband")
    args = ap.parse_args()
    for s in args.subjects:
        plot_subject(s, args.activation)


if __name__ == "__main__":
    main()
