"""Paper-ready candidate figures for Topic 4 MZ state-aligned finite-time spatial mode tracking.

Design contract: docs/superpowers/specs/2026-07-21-topic4-mz-m-eigenmode-tracking-design.md §8.
Reads results/topic4_sef_hfo/mz_m_eigenmode_tracking/ and renders:
  Figure A  state-aligned fixed-kick tracking (baseline / approach_75 / settled_plateau maps + trajectory)
  Figure B  finite-time mode tracking (identifiability strip + robust U1 + sigma_hat_1(T) + m-controls)

This is the direct current-based MZ spiking network at the z+m plateau — NOT a rate-field surrogate,
NOT exact eigenmodes. The empirical operator is shown ONLY where the strict low-k audit passes; an
unresolved state is left BLANK (never drawn as gain 0). Fail-closed when a required sidecar is missing
(spec §7 E20). Never touch the z-only Figure 5 directories.

Visual contract (spec §8): width ~7.2in, 300 dpi, editable PDF text (fonttype=42), no suptitle, no
background grid, short left panel letters. baseline = grey #555555, slow-state progression = a green
ramp, settled plateau = dark green accent; NO red/blue (reserved for template A/B). Signed fields -> PuOr.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "results", "topic4_sef_hfo", "mz_m_eigenmode_tracking")
CAND_DIR = os.path.join(ROOT, "results", "paper-ready-figure",
                        "fig5_mz_m_eigenmode_tracking_candidate", "figures")

STATE_ORDER = ["baseline", "approach_25", "approach_50", "approach_75", "settled_plateau"]
STATE_COLOR = {"baseline": "#555555", "approach_25": "#a1d99b", "approach_50": "#74c476",
               "approach_75": "#31a354", "settled_plateau": "#006d2c"}
STATE_SHORT = {"baseline": "base", "approach_25": "a25", "approach_50": "a50",
               "approach_75": "a75", "settled_plateau": "plat"}
MAIN3 = ["baseline", "approach_75", "settled_plateau"]
SIGNED, MAG = "PuOr", "magma"
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 8,
                     "axes.linewidth": 0.6, "figure.dpi": 150})


# ------------------------------------------------------------------ loaders (fail-closed / None if absent)
def load_state(seed, state, out=OUT):
    j = os.path.join(out, "per_seed", f"state_seed{seed}_{state}.json")
    npz = os.path.join(out, "per_seed", f"arrays_seed{seed}_{state}.npz")
    if not os.path.exists(j):
        return None
    d = dict(summary=json.load(open(j)), arr=(dict(np.load(npz, allow_pickle=True)) if os.path.exists(npz) else {}))
    return d


def load_registration(out=OUT):
    p = os.path.join(out, "state_registration.json")
    return json.load(open(p)) if os.path.exists(p) else None


def load_traj(seed, out=OUT):
    p = os.path.join(out, "per_seed", f"traj_seed{seed}.npz")
    return dict(np.load(p)) if os.path.exists(p) else None


def seeds_present(out=OUT):
    ss = set()
    for f in glob.glob(os.path.join(out, "per_seed", "state_seed*_baseline.json")):
        ss.add(int(os.path.basename(f).split("seed")[1].split("_")[0]))
    return sorted(ss)


def require_registration(out=OUT):
    reg = load_registration(out)
    if reg is None:
        raise SystemExit(f"[fig] missing state_registration.json in {out} (run `register` first)")
    return reg


def _letter(ax, s):
    ax.text(-0.02, 1.06, s, transform=ax.transAxes, fontsize=11, fontweight="bold", ha="right", va="bottom")


def _grididx(xy, n):
    return (xy[0] / 5.0 + 0.5) * (n - 1), (xy[1] / 5.0 + 0.5) * (n - 1)


def _signed_field(ax, field, *, vmax, title=None, src=None, snk=None, color="k"):
    im = ax.imshow(np.asarray(field, float).T, origin="lower", cmap=SIGNED, vmin=-vmax, vmax=vmax, aspect="equal")
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=8, pad=2, color=color)
    n = np.asarray(field).shape[0]
    for xy, mk in ((src, "o"), (snk, "s")):
        if xy is not None:
            gx, gy = _grididx(xy, n)
            ax.plot(gx, gy, mk, mfc="none", mec="k", mew=0.9, ms=5)
    return im


def _blank(ax, text="unresolved"):
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center", fontsize=7, color="#999999")
    for s in ax.spines.values():
        s.set_edgecolor("#dddddd")


# =============================================================== Figure A: state-aligned fixed-kick tracking
def figure_a(seed, cfg, out=OUT):
    reg = require_registration(out)
    srec = reg["seeds"].get(str(seed))
    if srec is None:
        raise SystemExit(f"[figA] seed{seed} not registered")
    data = {st: load_state(seed, st, out) for st in MAIN3}
    if data["baseline"] is None:
        raise SystemExit(f"[figA] missing baseline fixed-kick sidecar for seed{seed}")
    centers = [float(c) for c in cfg["local_map_centers_ms"]]
    src = data["baseline"]["summary"].get("src_g"); snk = data["baseline"]["summary"].get("snk_g")
    have = [st for st in MAIN3 if data[st] and data[st]["summary"].get("resolved") and "fk_dmaps" in data[st]["arr"]]
    dmax = max((np.nanmax(np.abs(data[st]["arr"]["fk_dmaps"])) for st in have), default=1.0) or 1.0

    fig = plt.figure(figsize=(7.2, 6.4))
    # top: D/a trajectory with 5 checkpoints
    axT = fig.add_axes([0.09, 0.79, 0.86, 0.16])
    tr = load_traj(seed, out)
    if tr is not None:
        axT.plot(tr["t_ms"] / 1000.0, tr["D_allE"], color="#333333", lw=0.8, zorder=1)
        axTa = axT.twinx()
        axTa.plot(tr["t_ms"] / 1000.0, tr["a_allE"], color="#b39ddb", lw=0.7, alpha=0.9, zorder=1)
        axTa.set_ylabel("a (adapt.)", fontsize=7, color="#7e57c2"); axTa.tick_params(labelsize=6.5)
    for st in STATE_ORDER:
        d = srec["states"].get(st, {})
        if d.get("branch_step") is not None:
            axT.scatter([d["branch_step"] * 0.1 / 1000.0], [d["D"]], s=34, color=STATE_COLOR[st], zorder=3,
                        edgecolor="k", linewidth=0.5, label=STATE_SHORT[st])
    axT.set_xlabel("time (s)", fontsize=7.5); axT.set_ylabel("D = 1 − z̄", fontsize=7.5)
    axT.tick_params(labelsize=6.5); axT.spines[["top"]].set_visible(False)
    axT.legend(fontsize=6, ncol=5, loc="upper left", frameon=False, handletextpad=0.2, columnspacing=0.8)
    axT.set_title(f"z+m plateau slow-state trajectory · seed {seed} (settled={srec.get('settled')})",
                  fontsize=8, pad=3)
    _letter(axT, "a")

    # body: baseline / approach_75 / settled_plateau x 4 time maps
    gsb = GridSpec(3, 5, figure=fig, width_ratios=[1, 1, 1, 1, 0.9], left=0.07, right=0.965,
                   top=0.70, bottom=0.30, hspace=0.16, wspace=0.16)
    im = None
    for ri, st in enumerate(MAIN3):
        d = data[st]
        resolved = bool(d and d["summary"].get("resolved") and "fk_dmaps" in d["arr"])
        for ci, c in enumerate(centers[:4]):
            ax = fig.add_subplot(gsb[ri, ci])
            if resolved:
                im = _signed_field(ax, d["arr"]["fk_dmaps"][ci], vmax=dmax,
                                   title=(f"{int(c)} ms" if ri == 0 else None), src=src, snk=snk)
            else:
                _blank(ax, "unresolved" if ci == 1 else "")
            if ci == 0:
                ax.set_ylabel(st.replace("_", "\n"), color=STATE_COLOR[st], fontsize=8, fontweight="bold")
            if ri == 0 and ci == 0:
                _letter(ax, "b")
    if im is not None:
        cax = fig.add_subplot(gsb[0:3, 4]); cax.set_axis_off()
        fig.colorbar(im, ax=cax, fraction=0.5, pad=0.0).set_label("Δ E-rate (Hz)\nkick − control", fontsize=7)

    # bottom: corridor vs off-axis (all seeds), distal recruitment, arrival slope
    gsc = GridSpec(1, 3, figure=fig, left=0.09, right=0.965, top=0.21, bottom=0.07, wspace=0.5)
    ss = seeds_present(out)
    axC = fig.add_subplot(gsc[0, 0])
    for xi, st in enumerate(MAIN3):
        vals = [load_state(s, st, out)["summary"]["fixed_kick"]["region"].get("axis_corridor")
                for s in ss if load_state(s, st, out) and load_state(s, st, out)["summary"].get("resolved")]
        vals = [v for v in vals if v is not None]
        axC.scatter([xi] * len(vals), vals, color=STATE_COLOR[st], s=20, zorder=3)
        if vals:
            axC.plot([xi - 0.28, xi + 0.28], [np.mean(vals)] * 2, color=STATE_COLOR[st], lw=1.8)
    axC.set_xticks(range(3)); axC.set_xticklabels([STATE_SHORT[s] for s in MAIN3], fontsize=7.5)
    axC.set_xlim(-0.6, 2.6); axC.set_ylabel("axial-corridor\n|Δ E-rate| (Hz)", fontsize=7.5)
    axC.spines[["top", "right"]].set_visible(False); _letter(axC, "c")

    axD = fig.add_subplot(gsc[0, 1])
    for xi, st in enumerate(MAIN3):
        vals = []
        for s in ss:
            ls = load_state(s, st, out)
            if ls and ls["summary"].get("resolved"):
                r = ls["summary"]["fixed_kick"].get("distal_corridor_over_matched_off_axis")
                if r is not None:
                    vals.append(r)
        axD.scatter([xi] * len(vals), vals, color=STATE_COLOR[st], s=20, zorder=3)
        if vals:
            axD.plot([xi - 0.28, xi + 0.28], [np.mean(vals)] * 2, color=STATE_COLOR[st], lw=1.8)
    axD.axhline(1.0, color="k", lw=0.7, ls=":")
    axD.set_xticks(range(3)); axD.set_xticklabels([STATE_SHORT[s] for s in MAIN3], fontsize=7.5)
    axD.set_xlim(-0.6, 2.6); axD.set_ylabel("distal-corridor /\nmatched off-axis", fontsize=7.5)
    axD.spines[["top", "right"]].set_visible(False); _letter(axD, "d")

    axE = fig.add_subplot(gsc[0, 2])
    plotted = False
    for st in MAIN3:
        d = data.get(st)
        if not (d and d["summary"].get("resolved") and "fk_kymo" in d["arr"]):
            continue
        fit = d["summary"]["fixed_kick"].get("arrival_fit", {})
        if not fit.get("eligible"):
            continue
        ky = d["arr"]["fk_kymo"]; dist = d["arr"]["fk_kymo_dist"]; times = d["arr"]["fk_kymo_times"]
        thr = 0.1 * np.nanmax(np.abs(ky))
        arr = np.array([times[np.argmax(np.abs(ky[:, p]) >= thr)] if np.any(np.abs(ky[:, p]) >= thr) else np.nan
                        for p in range(ky.shape[1])])
        ok = np.isfinite(arr)
        axE.scatter(dist[ok], arr[ok], color=STATE_COLOR[st], s=18, zorder=3)
        xs = np.array([dist[ok].min(), dist[ok].max()])
        axE.plot(xs, fit["slope"] * xs + (arr[ok].mean() - fit["slope"] * dist[ok].mean()),
                 color=STATE_COLOR[st], lw=1.3, label=f"{STATE_SHORT[st]} R²={fit['r2']:.2f}")
        plotted = True
    if plotted:
        axE.legend(fontsize=6, frameon=False, loc="upper left")
    else:
        _blank(axE, "no qualified\narrival front")
    axE.set_xlabel("axial dist. (src→sink)", fontsize=7.5); axE.set_ylabel("first-arrival (ms)", fontsize=7.5)
    axE.spines[["top", "right"]].set_visible(False); _letter(axE, "e")

    _save(fig, "figure5_mz_eigenmode_A_fixed_kick_tracking")


# =============================================================== Figure B: finite-time mode tracking
def figure_b(cfg, out=OUT):
    require_registration(out)
    op_path = os.path.join(out, "operator_tracking_summary.json")
    if not os.path.exists(op_path):
        raise SystemExit("[figB] missing operator_tracking_summary.json (run `aggregate` first)")
    op = json.load(open(op_path))
    rows = {(r["seed"], r["state"]): r for r in op["rows"] if r.get("resolved")}
    tol = float(op["tol"])
    ident = [r for r in op["rows"] if r.get("identifiable")]
    Tmid = int(round(cfg["T_windows_ms"][1]))

    fig = plt.figure(figsize=(7.2, 5.6))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.0, 1.0], width_ratios=[1.35, 1, 1],
                  left=0.085, right=0.965, top=0.9, bottom=0.1, hspace=0.5, wspace=0.45)

    # (a) identifiability strip: discrepancy point + split-half whisker vs 15% gate; filled = robust
    axA = fig.add_subplot(gs[0, 0])
    for xi, st in enumerate(STATE_ORDER):
        seeds = sorted({s for (s, ss) in rows if ss == st})
        for k, s in enumerate(seeds):
            r = rows[(s, st)]
            if r.get("discrepancy") is None:
                continue
            x = xi + (k - 1) * 0.16
            lo, hi = sorted([r.get("disc_repeatA", np.nan), r.get("disc_repeatB", np.nan)])
            axA.plot([x, x], [lo, hi], color=STATE_COLOR[st], lw=1.0, zorder=2)
            axA.scatter([x], [r["discrepancy"]], s=24, zorder=3, linewidth=1.0,
                        facecolor=(STATE_COLOR[st] if r.get("identifiable") else "none"), edgecolor=STATE_COLOR[st])
    axA.axhline(tol, color="k", lw=0.9, ls="--")
    axA.text(4.05, tol, "15% gate", fontsize=6.4, va="center", ha="right")
    axA.text(0.02, 0.98, f"robust {len(ident)}/{len(rows)} (filled)\npoint=N16 · whisker=8+8",
             transform=axA.transAxes, fontsize=6.2, va="top")
    axA.set_xticks(range(5)); axA.set_xticklabels([STATE_SHORT[s] for s in STATE_ORDER], fontsize=7)
    axA.set_xlim(-0.5, 4.5)
    axA.set_ylabel("linearity discrepancy", fontsize=7.3)
    axA.spines[["top", "right"]].set_visible(False); _letter(axA, "a")

    # (b) U1 for robustly identifiable (seed,state) — or an honest note if none
    axB = fig.add_subplot(gs[0, 1])
    if ident:
        r = sorted(ident, key=lambda x: x.get("discrepancy", 1))[0]
        npz = os.path.join(out, "per_seed", f"arrays_seed{r['seed']}_{r['state']}.npz")
        u1 = np.load(npz, allow_pickle=True).get(f"corr_u1_T{Tmid}") if os.path.exists(npz) else None
        st0 = load_state(r["seed"], r["state"], out)
        src = st0["summary"].get("src_g") if st0 else None
        snk = st0["summary"].get("snk_g") if st0 else None
        if u1 is not None:
            _signed_field(axB, u1, vmax=(np.nanmax(np.abs(u1)) or 1.0), src=src, snk=snk,
                          title=f"U₁ {STATE_SHORT[r['state']]} s{r['seed']}", color=STATE_COLOR[r["state"]])
            axB.set_xlabel(f"axis={r.get(f'u1_axis_T{Tmid}'):+.2f} · corr={r.get(f'u1_corridor_T{Tmid}', float('nan')):.2f}",
                           fontsize=6.6)
        else:
            _blank(axB)
    else:
        _blank(axB, "no robustly\nidentifiable\noperator")
    _letter(axB, "b")

    # (c) sigma_hat_1(T) for identifiable states (units: Hz / current-fraction)
    axC = fig.add_subplot(gs[0, 2])
    Ts = [float(t) for t in cfg["T_windows_ms"]]
    any_sig = False
    for r in ident:
        ys = [r.get(f"sigma1_T{int(t)}") for t in Ts]
        if all(y is not None for y in ys):
            axC.plot(Ts, ys, "-o", ms=3, color=STATE_COLOR[r["state"]], lw=1.2,
                     label=f"{STATE_SHORT[r['state']]} s{r['seed']}")
            any_sig = True
    if any_sig:
        axC.legend(fontsize=5.6, frameon=False)
    else:
        _blank(axC, "σ̂₁ undefined\n(no identifiable\nstate)")
    axC.set_xlabel("T (ms)", fontsize=7.3); axC.set_ylabel("σ̂₁ (Hz / frac)", fontsize=7.3)
    axC.spines[["top", "right"]].set_visible(False); _letter(axC, "c")

    # (d) adjacent mode overlap / principal angle (only both-identifiable pairs)
    axD = fig.add_subplot(gs[1, 0])
    mt = [r for r in op.get("mode_tracking", []) if r.get("both_identifiable")]
    if mt:
        labels = [r["pair"].replace("->", "→").replace("_", "") for r in mt]
        axD.bar(range(len(mt)), [r["u1_overlap"] for r in mt], color="#31a354", width=0.6)
        axD.set_xticks(range(len(mt))); axD.set_xticklabels(labels, fontsize=6, rotation=30, ha="right")
        axD.set_ylim(0, 1); axD.set_ylabel("|U₁ overlap| (sign-inv.)", fontsize=7.3)
    else:
        _blank(axD, "no adjacent identifiable pair\n(mode trajectory unavailable)")
    axD.spines[["top", "right"]].set_visible(False); _letter(axD, "d")

    # (e) axis alignment across identifiable states
    axE = fig.add_subplot(gs[1, 1])
    got = False
    for xi, st in enumerate(STATE_ORDER):
        vals = [rows[(s, st)].get(f"u1_axis_T{Tmid}") for (s, ss) in rows if ss == st
                and rows[(s, ss)].get("identifiable")]
        vals = [v for v in vals if v is not None]
        if vals:
            axE.scatter([xi] * len(vals), vals, color=STATE_COLOR[st], s=22, zorder=3); got = True
    axE.axhline(0, color="k", lw=0.6, ls=":")
    if not got:
        _blank(axE, "U₁ axis undefined\n(no identifiable state)")
    axE.set_xticks(range(5)); axE.set_xticklabels([STATE_SHORT[s] for s in STATE_ORDER], fontsize=6.6)
    axE.set_xlim(-0.5, 4.5); axE.set_ylim(-1.05, 1.05)
    axE.set_ylabel("U₁ axis alignment", fontsize=7.3)
    axE.spines[["top", "right"]].set_visible(False); _letter(axE, "e")

    # (f) minimal m-mechanism contrast: fixed-kick response norm per condition
    axF = fig.add_subplot(gs[1, 2])
    ctrl_path = os.path.join(out, "controls_summary.json")
    conds = cfg["m_controls"]["conditions"]
    cc = {"native_zm": "#555555", "m_reset": "#d95f0e", "m_uniform": "#8c96c6", "m_shuffle": "#41ab5d"}
    if os.path.exists(ctrl_path):
        cr = json.load(open(ctrl_path))["rows"]
        st_show = "settled_plateau"
        for ci, cond in enumerate(conds):
            vals = []
            for row in cr:
                s = row["states"].get(st_show, {})
                if isinstance(s, dict) and cond in s:
                    vals.append(s[cond].get("response_norm"))
            vals = [v for v in vals if v is not None]
            axF.scatter([ci] * len(vals), vals, color=cc.get(cond, "#666"), s=22, zorder=3)
            if vals:
                axF.plot([ci - 0.28, ci + 0.28], [np.mean(vals)] * 2, color=cc.get(cond, "#666"), lw=1.8)
        axF.set_xticks(range(len(conds)))
        axF.set_xticklabels([c.replace("_zm", "").replace("m_", "") for c in conds], fontsize=6.4, rotation=20, ha="right")
        axF.set_ylabel("fixed-kick response\nnorm — settled plateau", fontsize=7.0)
    else:
        _blank(axF, "m-controls\nnot run")
    axF.spines[["top", "right"]].set_visible(False); _letter(axF, "f")

    _save(fig, "figure5_mz_eigenmode_B_mode_tracking")


def _save(fig, stem):
    os.makedirs(CAND_DIR, exist_ok=True)
    p = os.path.join(CAND_DIR, stem)
    fig.savefig(p + ".png", dpi=300); fig.savefig(p + ".pdf"); plt.close(fig)
    print(f"[fig] wrote {p}.png/.pdf", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1, help="seed for the Figure-A trajectory + maps")
    args = ap.parse_args()
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import yaml
    cfg = yaml.safe_load(open(os.path.join(ROOT, "config", "topic4_mz_m_eigenmode_tracking.yaml")))
    figure_a(args.seed, cfg)
    figure_b(cfg)


if __name__ == "__main__":
    main()
