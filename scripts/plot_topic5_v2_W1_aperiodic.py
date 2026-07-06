#!/usr/bin/env python
"""Topic 5 V2 Phase-1-v2 — W1 "1/f as control" figure (one version per pool).

The scientific point (locked tier = exploratory candidate scaffold refinement): the interictal-HFO /
early-ictal field alignment is NOT ripple-specific and its apparent frequency-specific structure is
LARGELY absorbed by the 1/f (aperiodic) background — after removing 1/f, only gamma still beats the
weak spatial null in BOTH pools.

Panel A — how 1/f is measured (transparency): one representative contact's baseline PSD (log-log),
the actual OLS 1/f fit over [1,200] Hz, the excised mains-harmonic bins, and the shaded band-excess
(power above the 1/f floor); the gamma bump is highlighted.

Panel B — the collapse: per primary band, the cohort alignment ABOVE the spatial null
(Delta = observed - null median) at three stages raw -> minus-broadband(common_resid) ->
minus-1/f(aperiodic_resid). A filled+starred marker survives the max-over-bands FWER spatial null;
open does not. |maxAB| is deliberately NOT the y-axis: it is ~flat across bands (any residual field
retains some structure), so the collapse is only visible null-relative.

NOT a formal gate figure: "residual survival under weak spatial null", not "Gate B/C passed".

Usage:  python scripts/plot_topic5_v2_W1_aperiodic.py --substrate broad
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.build_topic5_ictal_field_long_cache import (  # noqa: E402
    iter_subject_seizure_windows, GUARD_SEC, MIN_BASELINE_SEC)
from src.topic5_ictal_recruitment import _spectrogram_on_hop  # noqa: E402
from src.ictal_onset_extraction import resolve_baseline_window  # noqa: E402
from src.topic5_v2_band_scan import (  # noqa: E402
    load_phase1_config, line_noise_bin_mask, aperiodic_corrected_excess_power)

V2_ROOT = _ROOT / "results/topic5_ictal_recruitment/v2_band_scan"
FIT_LO, FIT_HI, MIN_R2 = 1.0, 200.0, 0.5
FEATURES = ["raw", "common_resid", "aperiodic_resid"]
FEATURE_LABEL = {"raw": "raw", "common_resid": "− broadband\n(common-field)", "aperiodic_resid": "− 1/f\n(aperiodic)"}
FEATURE_COLOR = {"raw": "#9aa0a6", "common_resid": "#4c72b0", "aperiodic_resid": "#c44e52"}
# representative contact source (in both pools, fs=1024 -> full band range incl. ripple, clean fits)
REP_SUBJECT = {"broad": "epilepsiae_1077", "narrow": "epilepsiae_1077"}

BAND_SHORT = {"delta_HYP_slow": "δ", "theta_preictal_PAC": "θ", "alpha_sharp_leq13": "α",
              "beta_LVFA_low": "β", "gamma_LVFA": "γ", "hg_low_ripple": "R", "ripple_high": "FR"}


def _band_specs(cfg):
    return [(r[0], float(r[1]), float(r[2])) for r in cfg["bands"]["primary"]]


# --------------------------------------------------------------------------- Panel A
def panel_a_data(substrate, subject, cfg):
    """One representative contact's baseline-median PSD + its actual 1/f fit (mirrors the cache builder)."""
    harmonics = list(cfg["line_noise"]["harmonics_hz"]); halfwidth = float(cfg["line_noise"]["halfwidth_hz"])
    win = float(cfg["power"]["spectrogram_win_sec"]); hop = float(cfg["power"]["spectrogram_hop_sec"])
    specs = _band_specs(cfg)
    gamma = next((lo, hi) for (n, lo, hi) in specs if n == "gamma_LVFA")

    for idx, sw, eeg_rel in iter_subject_seizure_windows(subject, substrate, drops=[]):
        f, t, Sxx = _spectrogram_on_hop(sw.signal, sw.fs, win, hop)          # (n_ch, n_freq, n_time)
        line_mask = line_noise_bin_mask(f, harmonics, halfwidth)
        bl = resolve_baseline_window(Sxx.shape[2], hop_sec=hop, pre_sec=sw.pre_sec,
                                     buffer_sec=GUARD_SEC, eeg_onset_rel_sec=eeg_rel,
                                     min_baseline_valid_sec=MIN_BASELINE_SEC)
        psd_bl = np.nanmedian(Sxx[:, :, bl.start_idx:bl.end_idx], axis=2)    # (n_ch, n_freq) baseline PSD
        disp_hi = min(sw.fs / 2.0, 260)
        # representative channel: a clean-HUGGING 1/f fit (r2>=0.96 and the fitted line floats <=~0.6
        # decades above the PSD anywhere -> no ugly low-freq knee overshoot) with the largest visible
        # gamma bump; fall back to the lowest-overshoot contacts if none clear the bar.
        cands = []
        for c in range(psd_bl.shape[0]):
            r = aperiodic_corrected_excess_power(f, psd_bl[c], gamma[0], gamma[1], line_mask,
                                                 FIT_LO, FIT_HI, MIN_R2, half_open=True)
            if not (r["ok"] and np.isfinite(r["excess_power"]) and r["excess_power"] > 0):
                continue
            floor = 10.0 ** (r["slope"] * np.log10(np.where(f > 0, f, np.nan)) + r["offset"])
            m = (f >= FIT_LO) & (f <= disp_hi) & (psd_bl[c] > 0) & np.isfinite(psd_bl[c])
            overshoot = float(np.nanmax(np.log10(floor[m]) - np.log10(psd_bl[c][m]))) if m.any() else 9.9
            cands.append((c, r, overshoot))
        if not cands:
            continue
        good = [x for x in cands if x[1]["fit_r2"] >= 0.96 and x[2] <= 0.6]
        if not good:
            good = sorted(cands, key=lambda x: x[2])[:5]
        c, fit, _ov = max(good, key=lambda x: x[1]["excess_power"])
        ch = sw.ch_names[c] if c < len(sw.ch_names) else f"ch{c}"
        return {"f": f, "psd": psd_bl[c], "line_mask": line_mask, "slope": fit["slope"],
                "offset": fit["offset"], "r2": fit["fit_r2"], "specs": specs, "gamma": gamma,
                "ch": ch, "subject": subject, "fs": float(sw.fs),
                "harmonics": harmonics, "halfwidth": halfwidth}
    raise RuntimeError(f"no clean representative contact for {subject}/{substrate}")


def draw_panel_a(ax, d):
    f, psd = d["f"], d["psd"]
    pos = (f > 0) & np.isfinite(psd) & (psd > 0)
    floor = 10.0 ** (d["slope"] * np.log10(np.where(f > 0, f, np.nan)) + d["offset"])
    # PSD + 1/f fit
    ax.loglog(f[pos], psd[pos], color="#222222", lw=1.4, zorder=4, label="baseline PSD (one contact)")
    fitr = (f >= FIT_LO) & (f <= FIT_HI)
    ax.loglog(f[fitr], floor[fitr], color="#c44e52", lw=2.0, ls="--", zorder=5,
              label=f"1/f fit  (slope {d['slope']:.2f}, r²={d['r2']:.2f})")
    # shaded excess = power above the fitted 1/f floor (what the band-excess metric sums per band)
    exc = pos & (psd > floor) & (f >= FIT_LO)
    ax.fill_between(f, floor, psd, where=exc, interpolate=True, color="#c9c9c9", alpha=0.6, lw=0,
                    zorder=2, label="power above 1/f floor (band excess)")
    # excised mains-harmonic bins
    for h in d["harmonics"]:
        ax.axvspan(h - d["halfwidth"], h + d["halfwidth"], color="#bbbbbb", alpha=0.25, lw=0, zorder=1)
    ax.axvspan(np.nan, np.nan, color="#bbbbbb", alpha=0.25, lw=0, label="mains bins excised")
    # primary band dividers (so each shaded bump sits in a labelled band)
    for (nm, lo, hi) in d["specs"]:
        ax.axvline(lo, color="#d3d3d3", lw=0.7, zorder=0)
    ax.axvline(d["specs"][-1][2], color="#d3d3d3", lw=0.7, zorder=0)                       # close the last band
    ax.set_ylim(np.nanmin(psd[pos]) * 0.3, np.nanmax(psd[pos]) * 8)      # PSD range; don't let the fit extrapolation blow up the axis
    ax.set_xlim(FIT_LO, min(d["fs"] / 2.0, 260))
    ax.set_xlabel("frequency (Hz, log)", fontsize=13)
    ax.set_ylabel("power spectral density (log)", fontsize=13)
    ax.tick_params(labelsize=12)
    # band-name axis (top): map each frequency region to its named primary band (δ..FR)
    secax = ax.secondary_xaxis("top")
    secax.set_xticks([np.sqrt(lo * hi) for (nm, lo, hi) in d["specs"]])                    # geometric centres on the log axis
    secax.set_xticklabels([BAND_SHORT[nm] for (nm, lo, hi) in d["specs"]], fontsize=13, color="#444")
    secax.tick_params(length=0)
    ax.text(0.985, 0.96, f"contact {d['subject'].replace('epilepsiae_','E')} · {d['ch']}",
            transform=ax.transAxes, ha="right", va="top", fontsize=10, color="#666")
    ax.set_title("A  How 1/f is measured  (log-log OLS over 1–200 Hz; shaded = power above the 1/f floor)",
                 fontsize=11, loc="left", pad=18)
    ax.legend(fontsize=10, loc="lower left", framealpha=0.9)
    ax.grid(True, which="both", alpha=0.15)


# --------------------------------------------------------------------------- Panel B
def panel_b_data(substrate, cfg):
    specs = _band_specs(cfg)
    order = [n for (n, _, _) in specs]
    surv = pd.read_csv(V2_ROOT / "phase1_residual_survival_summary.csv")
    surv = surv[surv.substrate == substrate]
    rows = {}
    for feat in FEATURES:
        s = surv[surv.feature == feat].set_index("band")
        rows[feat] = {b: {"p": float(s.loc[b, "max_over_bands_p"]) if b in s.index else float("nan"),
                          "surv": int(s.loc[b, "survive"]) if b in s.index else 0} for b in order}
    n_surv = {feat: sum(v["surv"] for v in rows[feat].values()) for feat in FEATURES}
    return specs, order, rows, n_surv


def draw_panel_b(ax, specs, order, rows, n_surv, substrate):
    # Survival grid: rows = the two RESIDUAL controls (raw layer is F2's; not repeated here),
    # cols = 7 primary bands. Filled = the residual field still beats the weak spatial null (FWER);
    # cell value = max-over-bands p. Reading top→bottom shows the collapse to gamma.
    stages = ["common_resid", "aperiodic_resid"]
    slabel = {"common_resid": "− broadband", "aperiodic_resid": "− 1/f"}
    nb = len(order)
    gi = order.index("gamma_LVFA")
    ax.axvspan(gi - 0.5, gi + 0.5, color="#dd8452", alpha=0.12, zorder=0)                  # gamma column
    for ri, feat in enumerate(stages):
        for ci, b in enumerate(order):
            cell = rows[feat][b]; surv = cell["surv"]; p = cell["p"]
            robust = (feat == "aperiodic_resid" and b == "gamma_LVFA")                     # γ survives 1/f in BOTH pools
            fc = ("#d4a017" if robust else FEATURE_COLOR[feat]) if surv else "#f0f0f0"
            ax.add_patch(Rectangle((ci - 0.46, ri - 0.42), 0.92, 0.84, facecolor=fc,
                                   alpha=0.85 if surv else 1.0, edgecolor="#aaaaaa", lw=0.8, zorder=1))
            mark = "★ " if robust else ("✓ " if surv else "")
            ax.text(ci, ri, f"{mark}{p:.3f}", ha="center", va="center", fontsize=11,
                    color="black" if surv else "#999999", fontweight="bold" if surv else "normal", zorder=2)
    ax.set_xlim(-0.6, nb - 0.4)
    ax.set_ylim(1.75, -0.9)                                                                # 2 rows + room for band labels / title
    ax.set_xticks(range(nb))
    ax.set_xticklabels([f"{BAND_SHORT[b]}\n{lo:g}–{hi:g}" for (b, lo, hi) in specs], fontsize=12)
    ax.set_yticks(range(len(stages)))
    ax.set_yticklabels([f"{slabel[s]}\n{n_surv[s]}/7 survive" for s in stages], fontsize=12)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    title = ("B  Which bands still beat the spatial null after residual controls\n"
             f"filled ✓ = survives max-over-bands FWER;  raw baseline {n_surv['raw']}/7 → see F2")
    if substrate == "broad":
        title += "\nβ near-Nyquist-fragile · FR = family-ceiling artifact (§1.1) → only γ robust"
    ax.set_title(title, fontsize=11, loc="left")


# --------------------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--substrate", choices=["broad", "narrow"], required=True)
    ap.add_argument("--subject", default=None, help="representative contact source (default per-pool)")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    cfg = load_phase1_config()
    subject = args.subject or REP_SUBJECT[args.substrate]
    n = {"broad": 17, "narrow": 20}[args.substrate]

    da = panel_a_data(args.substrate, subject, cfg)
    specs, order, rows, n_surv = panel_b_data(args.substrate, cfg)

    tail = ("only gamma survives 1/f removal" if args.substrate == "narrow"
            else "only gamma is robust to 1/f removal")
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(14.5, 6.0), constrained_layout=True)
    fig.suptitle(f"Interictal-HFO / early-ictal alignment is largely a 1/f effect — {tail}"
                 f"   ({args.substrate} pool, n={n})", fontsize=13, fontweight="bold")
    draw_panel_a(axA, da)
    draw_panel_b(axB, specs, order, rows, n_surv, args.substrate)

    outdir = Path(args.outdir) if args.outdir else (V2_ROOT / "figures")
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"phase1_v2_W1_aperiodic_{args.substrate}.png"
    fig.savefig(out, dpi=150)
    print(f"[done] {out}  (rep contact {subject}/{da['ch']} r²={da['r2']:.2f}; survivors "
          f"raw {n_surv['raw']}/7 → −bb {n_surv['common_resid']}/7 → −1/f {n_surv['aperiodic_resid']}/7)")


if __name__ == "__main__":
    main()
