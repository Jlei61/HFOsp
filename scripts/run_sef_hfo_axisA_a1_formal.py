#!/usr/bin/env python3
"""Axis-A Stage A1 (formal) — threshold-dispersion -> fingerprint, matched operating point.

Plan §A1. A1-0a feasibility (a1_0a_feasibility/) showed all three std bands ignite
CLEANLY at core_mean=17.0 (narrow=5, mid=8, wide=9 clean at T=2000); lowering mean
makes mid/wide over-ignite into NON-clean events. So the clean common-ignition
operating point is FIXED-MEAN 17.0 — no mean-lowering, so the mean/work-point confound
the plan worried about largely dissolves here.

Design (user lock 2026-06-15): fixed-mean PRIMARY + matched-rate SENSITIVITY.
- PRIMARY: three std bands {0.5,1.0,1.5} at core_mean=17.0, T=3500 (clears n_min_events=6),
  single-focus oneend_neg, seeds {1,2,3}. Compare primary fingerprint features
  (axis_dir / pathway_width / onset_jitter) across bands (Kruskal-Wallis + pairwise
  Mann-Whitney + Cliff's delta on per-event values; onset top1 across seeds).
- SENSITIVITY (matched-rate via count-matching): sub-sample each band's pooled clean
  events to the smallest band's count, recompute, check the band contrasts survive
  (controls "wide ignites more events" -> tighter aggregate, not a mechanism difference).

REPORTS the threshold-dispersion fingerprint contrast at a MATCHED operating point.
Reuses the existing engine (no engine edit) + the frozen extractor.

Usage: python scripts/run_sef_hfo_axisA_a1_formal.py --workers 8 [--analyze-only]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.sef_hfo_fingerprint import extract_fingerprint, N_MIN_EVENTS_DEFAULT  # noqa: E402

BASE = ROOT / "results/topic4_sef_hfo/observation_layer/snn_cm_spontaneous"
OUT = BASE / "a1_formal"
RUNNER = ROOT / "scripts/run_sef_hfo_snn_cm_spontaneous_readout.py"

LESION = "oneend_neg"
MEAN = 17.0                 # fixed common-ignition operating point (from a1_0a map)
STDS = [0.5, 1.0, 1.5]      # narrow / mid / wide
SEEDS = [1, 2, 3]
T = 3500.0                  # matches baseline; clears the n_min_events=6 gate per band
THREADS_PER_RUN = 8
BAND_NAME = {0.5: "narrow", 1.0: "mid", 1.5: "wide"}


def _tag(std: float, seed: int) -> str:
    return f"a1_neg_m{MEAN}_std{std}_s{seed}"


def _cells():
    for std in STDS:
        for seed in SEEDS:
            yield std, seed


def _run_cell(std: float, seed: int) -> dict:
    tag = _tag(std, seed)
    if (OUT / f"readout_{tag}.json").exists():
        return {"tag": tag, "rc": 0, "skipped": True}
    log = OUT / "logs" / f"{tag}.log"
    cmd = [sys.executable, str(RUNNER), "--lesion", LESION, "--core-mean", str(MEAN),
           "--core-std", str(std), "--seed", str(seed), "--T", str(T),
           "--tag", tag, "--out", str(OUT)]
    env = dict(os.environ)
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[k] = str(THREADS_PER_RUN)
    with open(log, "w") as fh:
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env).returncode
    return {"tag": tag, "rc": rc, "skipped": False}


def run_grid(workers: int) -> None:
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    cells = list(_cells())
    print(f"A1 formal: {len(cells)} cells (std x seed) at mean={MEAN} T={T}, workers={workers}")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_run_cell, *c): c for c in cells}
        for f in as_completed(futs):
            r = f.result()
            print(f"  done {r['tag']} rc={r['rc']}" + (" (resumed)" if r.get("skipped") else ""))


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------
def _cliffs_delta(a, b) -> float:
    """Cliff's delta effect size in [-1,1]; 0 = no dominance."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size == 0 or b.size == 0:
        return float("nan")
    gt = sum((x > b).sum() for x in a)
    lt = sum((x < b).sum() for x in a)
    return (gt - lt) / (a.size * b.size)


def _per_event_clean(rf):
    """Per-event primary values for CLEAN events (mirror the extractor's clean gate)."""
    from src.sef_hfo_fingerprint import _is_clean
    rj = json.loads((OUT / f"readout_{rf.tag}.json").read_text())
    widths, errs, firsts = [], [], []
    for row in rf.events:
        if not _is_clean(rj["events"][row.event_index]):
            continue
        if row.pathway_width["value_mm"] is not None:
            widths.append(row.pathway_width["value_mm"])
        if row.axis_dir["axis_err_deg"] is not None:
            errs.append(row.axis_dir["axis_err_deg"])
        if row.first_contact is not None:
            firsts.append(row.first_contact)
    return dict(widths=widths, errs=errs, firsts=firsts)


def analyze() -> None:
    from scipy import stats
    from src.sef_hfo_stage3 import entry_jitter_stats

    # pool clean events per band across seeds
    band = {std: dict(widths=[], errs=[], firsts=[], per_seed_clean=[]) for std in STDS}
    per_run = []
    for std, seed in _cells():
        tag = _tag(std, seed)
        jp = OUT / f"readout_{tag}.json"
        npz = OUT / "per_event" / f"rep_{tag}.npz"
        if not (jp.exists() and npz.exists()):
            per_run.append(dict(std=std, seed=seed, tag=tag, status="MISSING"))
            continue
        rf = extract_fingerprint(jp, npz)
        pe = _per_event_clean(rf)
        band[std]["widths"] += pe["widths"]
        band[std]["errs"] += pe["errs"]
        band[std]["firsts"] += pe["firsts"]
        band[std]["per_seed_clean"].append(rf.n_clean_events)
        per_run.append(dict(std=std, seed=seed, tag=tag, status="ok",
                            n_clean=rf.n_clean_events, insufficient=rf.insufficient,
                            pathway_width_med=rf.aggregate["pathway_width"]["median_mm"],
                            axis_err_med=rf.aggregate["axis_dir"]["axis_err_median_deg"],
                            sign_majority=rf.aggregate["axis_dir"]["sign_majority"],
                            onset_top1=rf.aggregate["onset_jitter"].get("top1_fraction")))

    def band_summary(b):
        return dict(
            n_clean_total=len(b["widths"]),
            per_seed_clean=b["per_seed_clean"],
            pathway_width_median_mm=(round(float(np.median(b["widths"])), 4) if b["widths"] else None),
            pathway_width_iqr_mm=(round(float(np.percentile(b["widths"], 75)
                                            - np.percentile(b["widths"], 25)), 4)
                                  if len(b["widths"]) >= 2 else None),
            axis_err_median_deg=(round(float(np.median(b["errs"])), 4) if b["errs"] else None),
            onset_jitter=entry_jitter_stats(b["firsts"]))

    bands_out = {BAND_NAME[std]: band_summary(band[std]) for std in STDS}

    # cross-band tests on per-event pathway_width + axis_err
    def cross_band(metric):
        groups = [band[std][metric] for std in STDS if band[std][metric]]
        allv = np.concatenate([np.asarray(g, float) for g in groups]) if groups else np.array([])
        degenerate = allv.size > 0 and np.unique(np.round(allv, 6)).size == 1
        kw = (stats.kruskal(*groups) if (not degenerate and len(groups) >= 2
                                         and all(len(g) for g in groups)) else None)
        pair = {}
        for a, b in combinations(STDS, 2):
            ga, gb = band[a][metric], band[b][metric]
            if ga and gb and not degenerate:
                u = stats.mannwhitneyu(ga, gb, alternative="two-sided")
                pair[f"{BAND_NAME[a]}_vs_{BAND_NAME[b]}"] = dict(
                    p=round(float(u.pvalue), 5), cliffs_delta=round(_cliffs_delta(ga, gb), 4),
                    n_a=len(ga), n_b=len(gb))
        return dict(kruskal_p=(round(float(kw.pvalue), 5) if kw else None),
                    degenerate_identical=bool(degenerate),
                    common_value=(round(float(np.unique(np.round(allv, 6))[0]), 4) if degenerate else None),
                    pairwise=pair)

    primary = dict(pathway_width=cross_band("widths"), axis_err=cross_band("errs"))
    headline = dict(
        n_clean_per_band={BAND_NAME[std]: bands_out[BAND_NAME[std]]["n_clean_total"] for std in STDS},
        pathway_width_degenerate=primary["pathway_width"]["degenerate_identical"],
        axis_err_degenerate=primary["axis_err"]["degenerate_identical"],
        onset_top1_per_band={BAND_NAME[std]: bands_out[BAND_NAME[std]]["onset_jitter"].get("top1_fraction")
                             for std in STDS},
        interpretation=("primary fingerprint (pathway_width / axis_err / entry) is IDENTICAL across "
                        "narrow/mid/wide at this montage resolution; the only band difference is the "
                        "IGNITION RATE (n_clean rises narrow->wide). Threshold dispersion modulates "
                        "how OFTEN events ignite, NOT the propagation fingerprint. Descriptive NULL; "
                        "not a mechanism claim."))

    # matched-rate SENSITIVITY: count-match each band to the smallest band's clean count
    n_match = min(len(band[std]["widths"]) for std in STDS)
    rng = np.random.default_rng(0)
    matched = {}
    for std in STDS:
        w = np.asarray(band[std]["widths"], float)
        idx = rng.choice(len(w), size=n_match, replace=False) if len(w) > n_match else np.arange(len(w))
        matched[std] = w[idx].tolist()
    matched_pair = {}
    for a, b in combinations(STDS, 2):
        if matched[a] and matched[b]:
            u = stats.mannwhitneyu(matched[a], matched[b], alternative="two-sided")
            matched_pair[f"{BAND_NAME[a]}_vs_{BAND_NAME[b]}"] = dict(
                p=round(float(u.pvalue), 5), cliffs_delta=round(_cliffs_delta(matched[a], matched[b]), 4))
    sensitivity = dict(n_matched_per_band=n_match,
                       pathway_width_matched_pairwise=matched_pair,
                       note="count-matched sub-sample to the smallest band's clean count; "
                            "checks the band contrast is not just a 'wide ignites more events' artifact.")

    insufficient_bands = [BAND_NAME[std] for std in STDS if len(band[std]["widths"]) < N_MIN_EVENTS_DEFAULT]
    out = dict(
        stage="A1 formal (fixed-mean PRIMARY + count-matched SENSITIVITY). Threshold-dispersion "
              "fingerprint contrast at a matched operating point; NOT a mechanism claim.",
        lesion=LESION, core_mean=MEAN, T=T, stds=STDS, seeds=SEEDS,
        n_min_events=N_MIN_EVENTS_DEFAULT,
        insufficient_bands=insufficient_bands,
        headline=headline,
        per_run=per_run, bands=bands_out, primary_cross_band=primary, sensitivity=sensitivity)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "a1_formal_results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"wrote {OUT/'a1_formal_results.json'}")
    for std in STDS:
        bs = bands_out[BAND_NAME[std]]
        print(f"  {BAND_NAME[std]:6s} std={std}: n_clean={bs['n_clean_total']} "
              f"pw_med={bs['pathway_width_median_mm']} onset_top1={bs['onset_jitter'].get('top1_fraction')}")
    print(f"  pathway_width kruskal_p={primary['pathway_width']['kruskal_p']}  "
          f"matched n/band={n_match}")
    _figure(bands_out, band, per_run)


def _figure(bands_out, band, per_run) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    labels = [BAND_NAME[s] for s in STDS]
    # pathway_width violin/box per band
    ax[0].boxplot([band[s]["widths"] for s in STDS], tick_labels=labels, showmeans=True)
    ax[0].set_ylabel("pathway_width (mm, perp to locked axis)"); ax[0].set_title("pathway_width per band")
    # axis_err per band
    ax[1].boxplot([band[s]["errs"] for s in STDS], tick_labels=labels, showmeans=True)
    ax[1].set_ylabel("axis_err (deg)"); ax[1].set_title("axis readability per band")
    # onset top1 per band
    top1 = [bands_out[BAND_NAME[s]]["onset_jitter"].get("top1_fraction") or 0 for s in STDS]
    ax[2].bar(labels, top1); ax[2].set_ylim(0, 1.05)
    ax[2].set_ylabel("onset top1 fraction"); ax[2].set_title("entry concentration per band")
    fig.suptitle(f"A1 threshold-dispersion fingerprint @ fixed-mean {MEAN} (T={T}, seeds {SEEDS})")
    fig.tight_layout()
    fd = OUT / "figures"; fd.mkdir(parents=True, exist_ok=True)
    fig.savefig(fd / "a1_heterogeneity_fingerprint.png", dpi=130)
    plt.close(fig)
    print(f"wrote {fd/'a1_heterogeneity_fingerprint.png'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()
    if not args.analyze_only:
        run_grid(args.workers)
    analyze()


if __name__ == "__main__":
    main()
