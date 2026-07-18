#!/usr/bin/env python3
"""Topic 5 · multi-band peri-onset field-similarity timecourse (V2 band-scan cache extension).

Exploratory, time-resolved extension of the accepted Fig3-Sup1 (multi-band early-ictal energy
field <-> interictal HFO geometry alignment) into the Fig3-B peri-onset window. For each subject /
seizure / config band, read the pre-computed masked band-power baseline-robust-z from the committed
v2 band-scan cache, slide a ``[-120,+20]s`` onset-aligned window, and score the signed energy field
against the subject's interictal template A/B propagation fields on the SAME formal normalized
contact plane + SAME mirror-invariant signed-corr metric that Fig3-B uses (reused verbatim, NOT
re-invented).

Scientific tier (LOCKED): exploratory candidate scaffold. This answers ONLY three descriptive
questions -- is the ictal-band <-> interictal-geometry similarity already present BEFORE onset, does
it LIFT near onset, and is it BAND-GENERIC or a few-subject band-leaning phenomenon. It is NOT a
formal Gate A/B/C, NOT HFO-/ripple-specific, NOT a mechanism / oscillation / propagation-replay
claim. The formal cohort shift stays Fig3-A (Data-vs-Null); this is per-subject material pool.

Data contract (verified against the cache producer before writing -- CLAUDE.md §6):
  * cache: results/topic5_ictal_recruitment/v2_band_scan/cache/{ds_sid}.{npz,json}
      - npz["{band}__zt__{idx}"]   (n_ch, n_bins) float32 baseline-robust-z of log band power
      - npz["{band}__relt__{idx}"] per-bin time rel clinical onset (= spectrogram t - pre_sec);
        spans ~[-129.5, offset + POST_PAD(90)] so [-120,+20] is ALWAYS covered.
      - npz["channels"] montage-order names; json["analysis_channels"] fixed valid mask
      - seizure_idx is the SPARSE inventory row index (json["seizure_idxs"]).
  * contact set (band-INDEPENDENT): matched_channels(t_a, analysis_channels) -- template-A anchored
    ∩ fixed valid mask; >= 6 required. ONE _scorer per subject reused across all bands so the
    band comparison is apples-to-apples (CLAUDE.md §7): only the ictal band-z changes, never the
    template geometry or the contact set.
  * metric: corr_pair_mirror_invariant_signed on the formal normalized plane (REUSED from
    compute_topic5_signed_broadband_similarity._scorer). The ictal field is SIGNED robust-z; the
    template field is typical_rank -- the two are smoothed on the same plane and correlated. We do
    NOT rank-normalize the ictal z (user contract: never mix rank-normalized energy with signed
    robust-z in one metric).
  * window grid: [start, min(offset, stop)] offset-truncated (Fig3-B semantics); offset via
    _offset_rel(inv_rows[idx]) -- inventory metadata only, no raw signal reload.
  * legacy_bb_1_45 is a 1-45 Hz legacy reproduction band -- NOT the 1-150 reference and NOT part of
    the v2 band family; excluded here by construction. The optional 1-150 Hz reference curve on the
    line plot comes ONLY from the Fig3-B raw-block aggregate CSV, clearly labelled as a different
    pipeline (different hop / seizure eligibility) -- never conflated with a cache band.

Runs OMP_NUM_THREADS=1 (parallel-numpy discipline). Fail-closed: one subject or one band failing
never aborts the batch; every drop carries a reason.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # parallel-numpy discipline (user contract)

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ---- reused metric machinery (do NOT re-invent -- CLAUDE.md §6 "re-use don't re-invent") --------
from scripts.compute_topic5_signed_broadband_similarity import (  # noqa: E402
    _load_axis,
    _nan,
    _scorer,
)
from scripts.plot_topic5_signed_broadband_movie import _offset_rel, _window_values  # noqa: E402
from scripts.run_topic5_t0_eligibility import _inventory_rows  # noqa: E402
from src.topic5_axis_alignment import matched_channels  # noqa: E402
from src.topic5_v2_band_scan import load_phase1_config  # noqa: E402

CACHE_DIR = _ROOT / "results/topic5_ictal_recruitment/v2_band_scan/cache"
MARKER_ROOT = _ROOT / "results/topic5_ictal_recruitment/v2_band_scan"  # {broad,narrow}/*.marker.json
FIG3B_DIR = _ROOT / "results/topic5_ictal_recruitment/field_dynamics_signed"
DEFAULT_OUT = _ROOT / "results/topic5_ictal_recruitment/v2_band_timecourse"

MIN_MATCHED = 6  # same threshold as the Fig3-B broadband path

# Self-contained, paper-grade band labels (symbol + Hz range) so figures need no external legend.
BAND_LABEL = {
    "delta_HYP_slow": "δ 1–4", "theta_preictal_PAC": "θ 4–8", "alpha_sharp_leq13": "α 8–13",
    "beta_LVFA_low": "β 13–30", "gamma_LVFA": "γ 30–80", "hg_low_ripple": "R 80–150",
    "ripple_high": "FR 150–250",
    "low_HYP_1_13": "1–13", "LVFA_13_80": "13–80", "ripple_safe_80_220": "80–220",
    "ripple_full_80_250": "80–250",
}
# Line-plot grouping (the user's low / LVFA / ripple split) to avoid 7 overlapping IQR bands.
LINE_GROUPS = [
    ("low (1–13 Hz)", ["delta_HYP_slow", "theta_preictal_PAC", "alpha_sharp_leq13"]),
    ("LVFA (13–80 Hz)", ["beta_LVFA_low", "gamma_LVFA"]),
    ("ripple (80–250 Hz)", ["hg_low_ripple", "ripple_high"]),
]


# --------------------------------------------------------------------------------------------------
# small nan-safe reducers
def _finite(a):
    return np.asarray([x for x in np.asarray(a, float).ravel() if np.isfinite(x)], float)


def _med(a):
    a = _finite(a)
    return float(np.median(a)) if a.size else float("nan")


def _mean(a):
    a = _finite(a)
    return float(np.mean(a)) if a.size else float("nan")


def _pct(a, p):
    a = _finite(a)
    return float(np.percentile(a, p)) if a.size else float("nan")


def _pretty(ds_sid: str) -> str:
    return ds_sid.replace("epilepsiae_", "E").replace("yuquan_", "Y-")


def _sec_tag(x: float) -> str:
    return f"{'m' if float(x) < 0 else 'p'}{abs(float(x)):g}".replace(".", "p")


def _window_tag(window_sec: float, step_sec: float) -> str:
    if abs(float(window_sec) - float(step_sec)) <= 1e-9:
        return f"{window_sec:g}s"
    return f"{window_sec:g}s_step{step_sec:g}s"


def _stem(ds_sid: str, start: float, stop: float, window: float, step: float) -> str:
    return (f"{ds_sid}_multiband_similarity_timecourse_"
            f"{_sec_tag(start)}_{_sec_tag(stop)}_{_window_tag(window, step)}")


# --------------------------------------------------------------------------------------------------
# band family from config (primary 7 half-open + composites 4 closed; legacy_bb NOT included)
def band_specs(which: str):
    cfg = load_phase1_config()
    prim = [(str(n), float(lo), float(hi)) for n, lo, hi in cfg["bands"]["primary"]]
    comp = [(str(n), float(lo), float(hi)) for n, lo, hi in cfg["bands"]["composites"]]
    primary_names = [n for n, _, _ in prim]
    if which == "primary":
        return prim, primary_names
    if which == "all":
        return prim + comp, primary_names
    raise ValueError(f"--bands must be 'primary' or 'all', got {which!r}")


# --------------------------------------------------------------------------------------------------
# subject setup: cache + band-independent scorer over the fixed matched∩analysis contact set
def load_subject_cache(ds_sid: str):
    npz_p = CACHE_DIR / f"{ds_sid}.npz"
    js_p = CACHE_DIR / f"{ds_sid}.json"
    if not (npz_p.exists() and js_p.exists()):
        raise FileNotFoundError(f"v2 band cache missing for {ds_sid}")
    meta = json.loads(js_p.read_text())
    npz = np.load(npz_p, allow_pickle=True)
    channels = [str(c) for c in npz["channels"]]
    analysis = [str(c) for c in meta.get("analysis_channels", [])]
    if not analysis:
        raise RuntimeError("empty analysis_channels in cache")
    return meta, npz, channels, analysis


def build_scorer(ds_sid: str, channels, analysis):
    """Return (matched, names, raw_idx, score) -- band-INDEPENDENT (template A/B + plane)."""
    axis_a = _load_axis(ds_sid, "t_a")
    if axis_a is None:
        raise RuntimeError("missing interictal template A (_t_a.json)")
    cache_index = {n: i for i, n in enumerate(channels)}
    valid = set(analysis) & set(channels)                 # analysis ⊆ channels by construction
    matched = matched_channels(axis_a, valid)             # template-A order ∩ fixed valid mask
    if len(matched) < MIN_MATCHED:
        raise RuntimeError(f"insufficient matched contacts ({len(matched)} < {MIN_MATCHED})")
    names = [c["name"] for c in matched]
    raw_idx = np.array([cache_index[n] for n in names], int)
    # §6 name-alignment guard: selecting cache rows by raw_idx MUST reproduce `names` exactly,
    # otherwise the band-z would be correlated against the wrong contacts (silent contamination).
    assert [channels[i] for i in raw_idx] == names, f"{ds_sid}: channel index/name misalignment"
    score = _scorer(ds_sid, matched)
    return matched, names, raw_idx, score


def window_starts(dataset, sid, idx, inv_rows, start_sec, stop_sec, window_sec, step_sec):
    """Offset-truncated peri-onset grid: windows in [start, min(offset, stop)] (Fig3-B semantics).

    offset (clinical offset rel onset) from inventory metadata; on any failure fall back to the
    fixed stop cap and record a note (fail-closed -- never drop the seizure for a metadata hiccup).
    """
    hi_cap, note = float(stop_sec), None
    if 0 <= idx < len(inv_rows):
        try:
            hi_cap = min(float(_offset_rel(dataset, inv_rows[idx])), float(stop_sec))
        except Exception as e:  # noqa: BLE001
            note = f"offset_unresolved:{type(e).__name__}"
    else:
        note = "idx_out_of_inventory_range"
    last_lo = hi_cap - float(window_sec)
    if last_lo < float(start_sec) - 1e-9:
        return np.array([], float), note
    return np.arange(float(start_sec), last_lo + 1e-9, float(step_sec)), note


def _row(ds_sid, idx, band, lo, hi, w0, w1, per_t, best, n_contacts):
    r = {
        "subject": ds_sid, "seizure_idx": int(idx), "band": band,
        "band_lo": float(lo), "band_hi": float(hi),
        "window_start_sec": float(w0), "window_end_sec": float(w1),
        "window_center_sec": float((w0 + w1) / 2.0),
        "n_contacts": int(n_contacts), "best_template": best,
    }
    for k in ("A", "B"):
        d = per_t.get(k, {}) or {}
        r[f"{k}_signed_corr"] = _nan(d.get("signed_corr"))
        r[f"{k}_abs_corr"] = _nan(d.get("abs_corr"))
    absv = np.array([r["A_abs_corr"], r["B_abs_corr"]], float)
    r["maxAB_abs_corr"] = float(np.nanmax(absv)) if np.isfinite(absv).any() else float("nan")
    r["maxAB_signed_corr"] = r[f"{best}_signed_corr"] if best in ("A", "B") else float("nan")
    return r


def compute_subject(ds_sid, specs, start_sec, stop_sec, window_sec, step_sec):
    meta, npz, channels, analysis = load_subject_cache(ds_sid)
    matched, names, raw_idx, score = build_scorer(ds_sid, channels, analysis)
    dataset, sid = ds_sid.split("_", 1)
    inv_rows, _ = _inventory_rows(dataset, sid)
    seizure_idxs = [int(i) for i in meta.get("seizure_idxs", [])]
    npz_keys = set(npz.files)

    rows = []
    band_drops = []          # [(band, reason)] -- band produced no scored window for this subject
    seizure_notes = set()    # {(idx, note)} -- offset fallbacks etc.
    for band, lo, hi in specs:
        scored = 0
        for idx in seizure_idxs:
            zk, rk = f"{band}__zt__{idx}", f"{band}__relt__{idx}"
            if zk not in npz_keys or rk not in npz_keys:
                continue     # band skipped for this seizure at build time (e.g. Nyquist) -- skip
            starts, note = window_starts(dataset, sid, idx, inv_rows,
                                         start_sec, stop_sec, window_sec, step_sec)
            if note is not None:
                seizure_notes.add((idx, note))
            if starts.size == 0:
                continue
            z_sel = np.asarray(npz[zk], float)[raw_idx]      # matched rows, `names` order
            relt = np.asarray(npz[rk], float)
            wv = _window_values(z_sel, relt, starts, float(window_sec))  # (n_win, n_matched)
            for w0, vals in zip(starts, wv):
                per_t, best = score(vals)
                rows.append(_row(ds_sid, idx, band, lo, hi, float(w0), float(w0 + window_sec),
                                 per_t, best, int(np.isfinite(vals).sum())))
                scored += 1
        if scored == 0:
            band_drops.append((band, "no_windows_scored"))
    if not rows:
        raise RuntimeError("no (band, seizure, window) produced a score")

    per = pd.DataFrame(rows)
    info = {
        "n_matched_contacts": len(names),
        "matched_names": names,
        "n_seizures_cache": len(seizure_idxs),
        "band_drops": band_drops,
        "seizure_notes": sorted(seizure_notes),
    }
    return per, info


def aggregate(per: pd.DataFrame) -> pd.DataFrame:
    """subject × band × window: median/IQR/mean/sd/var/n of maxAB_abs + median signed A/B/maxAB."""
    keys = ["band", "band_lo", "band_hi", "window_start_sec", "window_end_sec", "window_center_sec"]
    out = []
    for (band, lo, hi, w0, w1, wc), g in per.groupby(keys, sort=True):
        v = pd.to_numeric(g["maxAB_abs_corr"], errors="coerce").dropna().to_numpy(float)
        out.append({
            "band": band, "band_lo": lo, "band_hi": hi,
            "window_start_sec": w0, "window_end_sec": w1, "window_center_sec": wc,
            "n_seizures": int(v.size),
            "median_maxAB_abs": _med(v), "mean_maxAB_abs": _mean(v),
            "q25_maxAB_abs": _pct(v, 25), "q75_maxAB_abs": _pct(v, 75),
            "iqr_maxAB_abs": (_pct(v, 75) - _pct(v, 25)) if v.size else float("nan"),
            "sd_maxAB_abs": float(np.std(v, ddof=1)) if v.size >= 2 else float("nan"),
            "var_maxAB_abs": float(np.var(v, ddof=1)) if v.size >= 2 else float("nan"),
            "median_signed_A": _med(g["A_signed_corr"]),
            "median_signed_B": _med(g["B_signed_corr"]),
            "median_maxAB_signed": _med(g["maxAB_signed_corr"]),
        })
    return pd.DataFrame(out)


# --------------------------------------------------------------------------------------------------
# figures
def _pivot(agg, col, band_order, centers):
    M = np.full((len(band_order), len(centers)), np.nan)
    bi = {b: i for i, b in enumerate(band_order)}
    ci = {c: j for j, c in enumerate(centers)}
    for _, r in agg.iterrows():
        if r["band"] in bi and r["window_center_sec"] in ci:
            M[bi[r["band"]], ci[r["window_center_sec"]]] = r[col]
    return M


def fig_heatmaps(ds_sid, agg, band_order, out_png):
    centers = sorted(agg["window_center_sec"].unique())
    if not centers:
        return
    Mabs = _pivot(agg, "median_maxAB_abs", band_order, centers)
    MsA = _pivot(agg, "median_signed_A", band_order, centers)
    MsB = _pivot(agg, "median_signed_B", band_order, centers)
    labels = [BAND_LABEL.get(b, b) for b in band_order]
    x = np.asarray(centers, float)
    dx = (x[1] - x[0]) / 2.0 if x.size > 1 else 1.0
    extent = [x.min() - dx, x.max() + dx, len(band_order) - 0.5, -0.5]

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 0.42 * len(band_order) + 2.6),
                             constrained_layout=True, sharey=True)

    def _draw(ax, M, cmap, vmin, vmax, title):
        im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                       extent=extent, interpolation="nearest")
        ax.axvline(0, color="k", ls="--", lw=1.0)
        ax.set_xlabel("time from clinical onset (s)")
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    _draw(axes[0], Mabs, "viridis", 0.0, 1.0, "maxAB |r|  (sign-free scaffold)")
    _draw(axes[1], MsA, "RdBu_r", -1.0, 1.0, "signed r · template A")
    _draw(axes[2], MsB, "RdBu_r", -1.0, 1.0, "signed r · template B")
    axes[0].set_yticks(range(len(band_order)))
    axes[0].set_yticklabels(labels, fontsize=8)
    axes[0].set_ylabel("band")
    fig.suptitle(
        f"{_pretty(ds_sid)} · multi-band peri-onset field similarity  (median across seizures)\n"
        "exploratory candidate scaffold — not a formal gate, not ripple-specific",
        fontsize=12)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def fig_lines(ds_sid, agg, out_png, ref_df=None):
    fig, axes = plt.subplots(len(LINE_GROUPS), 1, figsize=(10.4, 9.2), sharex=True,
                             constrained_layout=True)
    for ax, (gname, bands) in zip(axes, LINE_GROUPS):
        colors = plt.cm.viridis(np.linspace(0.12, 0.86, max(2, len(bands))))
        drew = False
        for c, b in zip(colors, bands):
            sub = agg[agg["band"] == b].sort_values("window_center_sec")
            if sub.empty:
                continue
            xx = sub["window_center_sec"].to_numpy(float)
            med = sub["median_maxAB_abs"].to_numpy(float)
            q25 = sub["q25_maxAB_abs"].to_numpy(float)
            q75 = sub["q75_maxAB_abs"].to_numpy(float)
            ax.fill_between(xx, q25, q75, color=c, alpha=0.14, lw=0)
            ax.plot(xx, med, color=c, lw=1.9, label=BAND_LABEL.get(b, b))
            drew = True
        if ref_df is not None:
            ax.plot(ref_df["window_center_sec"], ref_df["median_maxAB_abs_corr"],
                    color="0.45", ls="--", lw=1.2, label="1–150 Hz ref (Fig3-B path)")
        ax.axvline(0, color="0.15", ls="--", lw=1.0)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("maxAB |r|")
        ax.set_title(gname, fontsize=10, loc="left")
        ax.grid(True, color="0.92", lw=0.6)
        if drew or ref_df is not None:
            ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=8)
    axes[-1].set_xlabel("time from clinical onset (s), window center")
    fig.suptitle(
        f"{_pretty(ds_sid)} · primary-band peri-onset maxAB |r|  (median line, IQR shaded)\n"
        "exploratory candidate scaffold — sign-free scaffold readout",
        fontsize=12)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def load_fig3b_reference(ds_sid, start, stop, window, step):
    """1-150 Hz Fig3-B aggregate curve if present (raw-block path; different hop/eligibility)."""
    fp = FIG3B_DIR / (f"{ds_sid}_signed_broadband_1_150Hz_similarity_timecourse_"
                      f"{_sec_tag(start)}_{_sec_tag(stop)}_{_window_tag(window, step)}_aggregate.csv")
    if not fp.exists():
        return None
    df = pd.read_csv(fp)
    if {"window_center_sec", "median_maxAB_abs_corr"} <= set(df.columns):
        return df.sort_values("window_center_sec")
    return None


# --------------------------------------------------------------------------------------------------
# cohort (never pool axis sets)
def cohort_members(axis_set: str):
    # accepted Phase-1 cohort membership = the per-subject *.marker.json written by the alignment
    # run. They live in feature subdirs (v2_band_scan/{axis_set}/_partial_*/) so rglob + dedupe by
    # subject name (broad=17, narrow=20; never pooled).
    d = MARKER_ROOT / axis_set
    if not d.exists():
        return []
    return sorted({p.name[: -len(".marker.json")] for p in d.rglob("*.marker.json")})


def build_cohort(axis_set, band_order, out_root, start, stop, window, step, fig_dir):
    members = cohort_members(axis_set)
    aggs = {}
    for s in members:
        ap = out_root / f"{_stem(s, start, stop, window, step)}_aggregate.csv"
        if ap.exists():
            aggs[s] = pd.read_csv(ap)
    if not aggs:
        return None
    centers = sorted({float(c) for df in aggs.values() for c in df["window_center_sec"].unique()})
    bi = {b: i for i, b in enumerate(band_order)}
    ci = {c: j for j, c in enumerate(centers)}
    stacks = {(b, c): [] for b in band_order for c in centers}
    for df in aggs.values():
        for _, r in df.iterrows():
            key = (r["band"], float(r["window_center_sec"]))
            if key in stacks and np.isfinite(r["median_maxAB_abs"]):
                stacks[key].append(float(r["median_maxAB_abs"]))
    M = np.full((len(band_order), len(centers)), np.nan)
    for (b, c), vals in stacks.items():
        if vals:
            M[bi[b], ci[c]] = float(np.median(vals))

    # cohort band × time heatmap (subject-median of per-subject median maxAB_abs)
    labels = [BAND_LABEL.get(b, b) for b in band_order]
    x = np.asarray(centers, float)
    dx = (x[1] - x[0]) / 2.0 if x.size > 1 else 1.0
    fig, ax = plt.subplots(figsize=(9.6, 0.42 * len(band_order) + 2.4), constrained_layout=True)
    im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0,
                   extent=[x.min() - dx, x.max() + dx, len(band_order) - 0.5, -0.5],
                   interpolation="nearest")
    ax.axvline(0, color="k", ls="--", lw=1.0)
    ax.set_yticks(range(len(band_order)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_ylabel("band")
    ax.set_xlabel("time from clinical onset (s)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="cohort median maxAB |r|")
    fig.suptitle(
        f"Cohort ({axis_set}, n={len(aggs)}) · multi-band peri-onset field similarity\n"
        "exploratory descriptive summary — not a formal cohort statistic",
        fontsize=12)
    fig_dir.mkdir(parents=True, exist_ok=True)
    heat_png = fig_dir / f"cohort_band_time_heatmap_{axis_set}.png"
    fig.savefig(heat_png, dpi=150)
    plt.close(fig)

    # per-band pre(-120,0) vs early(0,20) descriptive delta table (subject-level then cohort median)
    drows = []
    for b in band_order:
        pre_vals, early_vals, deltas = [], [], []
        for df in aggs.values():
            sub = df[df["band"] == b]
            pre = sub[sub["window_end_sec"] <= 0]["median_maxAB_abs"]
            early = sub[(sub["window_start_sec"] >= 0) & (sub["window_end_sec"] <= 20)]["median_maxAB_abs"]
            if len(pre) and len(early):
                pv, ev = _med(pre), _med(early)
                if np.isfinite(pv) and np.isfinite(ev):
                    pre_vals.append(pv)
                    early_vals.append(ev)
                    deltas.append(ev - pv)
        drows.append({
            "axis_set": axis_set, "band": b,
            "band_label": BAND_LABEL.get(b, b),
            "n_subjects": len(deltas),
            "cohort_median_pre": _med(pre_vals),
            "cohort_median_early": _med(early_vals),
            "cohort_median_delta_early_minus_pre": _med(deltas),
            "n_subjects_delta_positive": int(np.sum(np.asarray(deltas) > 0)) if deltas else 0,
        })
    delta_df = pd.DataFrame(drows)
    delta_csv = out_root / f"cohort_pre_vs_early_delta_{axis_set}.csv"
    delta_df.to_csv(delta_csv, index=False)
    return {"axis_set": axis_set, "n_subjects": len(aggs), "subjects": sorted(aggs),
            "heatmap": str(heat_png.relative_to(_ROOT)), "delta_csv": str(delta_csv.relative_to(_ROOT))}


# --------------------------------------------------------------------------------------------------
def resolve_subjects(arg):
    if not arg or (len(arg) == 1 and arg[0] == "all"):
        return sorted(p.stem for p in CACHE_DIR.glob("*.npz"))
    return list(dict.fromkeys(arg))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subjects", nargs="*", default=["all"],
                    help="'all' (default = every cached subject) or explicit ds_sid list")
    ap.add_argument("--start-sec", type=float, default=-120.0)
    ap.add_argument("--stop-sec", type=float, default=20.0)
    ap.add_argument("--window-sec", type=float, default=10.0)
    ap.add_argument("--step-sec", type=float, default=2.0)
    ap.add_argument("--bands", choices=["primary", "all"], default="primary")
    ap.add_argument("--axis-set", choices=["broad", "narrow", "both"], default="broad",
                    help="cohort membership for the cohort summary (never pooled)")
    ap.add_argument("--out-root", default=str(DEFAULT_OUT))
    ap.add_argument("--no-figures", action="store_true", help="CSV/index only, skip per-subject figs")
    ap.add_argument("--no-cohort", action="store_true", help="skip the cohort summary pass")
    ap.add_argument("--cohort-only", action="store_true",
                    help="skip per-subject recompute; rebuild cohort products from existing aggregates")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    fig_dir = out_root / "figures"
    out_root.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    specs, primary_names = band_specs(args.bands)
    band_order = [n for n, _, _ in specs]
    subjects = resolve_subjects(args.subjects)
    broad_set, narrow_set = set(cohort_members("broad")), set(cohort_members("narrow"))

    print(f"[multiband-timecourse] {len(subjects)} subjects | bands={args.bands}({len(specs)}) | "
          f"window {args.window_sec:g}s step {args.step_sec:g}s | "
          f"[{args.start_sec:g},{args.stop_sec:g}]s -> {out_root}", flush=True)

    index_rows = []
    for ds_sid in ([] if args.cohort_only else subjects):
        try:
            per, info = compute_subject(ds_sid, specs, args.start_sec, args.stop_sec,
                                        args.window_sec, args.step_sec)
            agg = aggregate(per)
            stem = _stem(ds_sid, args.start_sec, args.stop_sec, args.window_sec, args.step_sec)
            per_csv = out_root / f"{stem}_per_seizure.csv"
            agg_csv = out_root / f"{stem}_aggregate.csv"
            per.to_csv(per_csv, index=False)
            agg.to_csv(agg_csv, index=False)

            heat_png = fig_dir / f"{stem}_band_time_heatmap.png"
            line_png = fig_dir / f"{stem}_primary_band_lines.png"
            fig_err = None
            if not args.no_figures:
                try:
                    ref = load_fig3b_reference(ds_sid, args.start_sec, args.stop_sec,
                                               args.window_sec, args.step_sec)
                    fig_heatmaps(ds_sid, agg, band_order, heat_png)
                    fig_lines(ds_sid, agg[agg["band"].isin(primary_names)], line_png, ref)
                except Exception as e:  # noqa: BLE001  -- figure failure must not drop the data
                    fig_err = f"{type(e).__name__}: {e}"
                    print(f"  [{ds_sid}] FIGURE ERROR {fig_err}", flush=True)

            per_band_med = {b: _med(agg[agg["band"] == b]["median_maxAB_abs"]) for b in band_order}
            index_rows.append({
                "subject": ds_sid, "status": "ok", "drop_reason": "",
                "in_broad": ds_sid in broad_set, "in_narrow": ds_sid in narrow_set,
                "n_seizures": int(per["seizure_idx"].nunique()),
                "n_windows": int(per["window_center_sec"].nunique()),
                "n_bands_scored": int(per["band"].nunique()),
                "n_matched_contacts": info["n_matched_contacts"],
                "overall_median_maxAB_abs": _med(per["maxAB_abs_corr"]),
                "overall_median_signed_A": _med(per["A_signed_corr"]),
                "overall_median_signed_B": _med(per["B_signed_corr"]),
                "band_drops": ";".join(f"{b}:{r}" for b, r in info["band_drops"]),
                "offset_notes": ";".join(f"{i}:{n}" for i, n in info["seizure_notes"]),
                "per_band_median_maxAB_abs": json.dumps(per_band_med),
                "per_seizure_csv": str(per_csv.relative_to(_ROOT)),
                "aggregate_csv": str(agg_csv.relative_to(_ROOT)),
                "figure_heatmap": str(heat_png.relative_to(_ROOT)) if not args.no_figures and not fig_err else "",
                "figure_lines": str(line_png.relative_to(_ROOT)) if not args.no_figures and not fig_err else "",
                "figure_error": fig_err or "",
            })
            print(f"  [{ds_sid}] ok | {index_rows[-1]['n_seizures']} sz × "
                  f"{index_rows[-1]['n_windows']} win × {index_rows[-1]['n_bands_scored']} bands | "
                  f"median maxAB|r|={index_rows[-1]['overall_median_maxAB_abs']:.3f}", flush=True)
        except Exception as e:  # noqa: BLE001  -- fail-closed per subject
            reason = f"{type(e).__name__}: {e}"
            index_rows.append({
                "subject": ds_sid, "status": "drop", "drop_reason": reason,
                "in_broad": ds_sid in broad_set, "in_narrow": ds_sid in narrow_set,
                "n_seizures": 0, "n_windows": 0, "n_bands_scored": 0, "n_matched_contacts": 0,
                "overall_median_maxAB_abs": float("nan"),
                "overall_median_signed_A": float("nan"), "overall_median_signed_B": float("nan"),
                "band_drops": "", "offset_notes": "", "per_band_median_maxAB_abs": "{}",
                "per_seizure_csv": "", "aggregate_csv": "",
                "figure_heatmap": "", "figure_lines": "", "figure_error": "",
            })
            print(f"  [{ds_sid}] DROP {reason}", flush=True)
            traceback.print_exc()

    if args.cohort_only:
        print("[multiband-timecourse] --cohort-only: rebuilding cohort from existing aggregates",
              flush=True)
    else:
        idx_df = pd.DataFrame(index_rows)
        idx_csv = out_root / "multiband_timecourse_subject_index.csv"
        idx_json = out_root / "multiband_timecourse_subject_index.json"
        idx_df.to_csv(idx_csv, index=False)
        idx_json.write_text(json.dumps(index_rows, indent=2, ensure_ascii=False) + "\n")
        n_ok = int((idx_df["status"] == "ok").sum())
        print(f"[multiband-timecourse] {n_ok}/{len(subjects)} subjects ok -> {idx_csv}", flush=True)

    cohort_summaries = []
    if not args.no_cohort:
        axis_sets = ["broad", "narrow"] if args.axis_set == "both" else [args.axis_set]
        for ax in axis_sets:
            summ = build_cohort(ax, band_order, out_root, args.start_sec, args.stop_sec,
                                args.window_sec, args.step_sec, fig_dir)
            if summ:
                cohort_summaries.append(summ)
                print(f"  [cohort:{ax}] n={summ['n_subjects']} -> {summ['heatmap']}", flush=True)
            else:
                print(f"  [cohort:{ax}] no member aggregates found, skipped", flush=True)
    if cohort_summaries:
        (out_root / "cohort_summary.json").write_text(
            json.dumps(cohort_summaries, indent=2, ensure_ascii=False) + "\n")
    print("MULTIBAND TIMECOURSE DONE", flush=True)


if __name__ == "__main__":
    main()
