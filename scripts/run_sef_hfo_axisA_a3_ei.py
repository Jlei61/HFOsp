#!/usr/bin/env python3
"""Axis-A Stage A3 — local E/I lesion -> fingerprint, event-rate matched to the V_th baseline.

Plan §A3 + spec. Single-focus E/I lesions (oneend_inhib w_EI down / oneend_recur w_EE up /
oneend_combined) at FLAT V_th=18.0; the excitability is in the weight lesion. Magnitudes are
tuned ONLY to land the per-run clean-event rate inside 0.8-1.25x of the V_th baseline (anti-
tuning: NEVER tuned to match the read-out answer). Then ask:
  (a) does the E/I lesion produce CLEAN directional templates (axis readable, sign consistent)?
  (b) is its fingerprint distinguishable from the V_th-down reference (same mean/T/geometry/seeds
      = the A1-formal std=1.5 'wide' band) and across E/I mechanisms?

Conclusion cap (spec §3): "在读出层面 E/I 病灶能/不能复现 V_th↓ 的双向模板(事件率匹配前提下)" —
never "E/I 机制被证实". Reuses the (re-blessed) engine + the frozen extractor.

Usage: python scripts/run_sef_hfo_axisA_a3_ei.py --workers 9 [--analyze-only]
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
from src.sef_hfo_fingerprint import extract_fingerprint, _is_clean, N_MIN_EVENTS_DEFAULT  # noqa: E402

BASE = ROOT / "results/topic4_sef_hfo/observation_layer/snn_cm_spontaneous"
OUT = BASE / "a3_ei"
A1 = BASE / "a1_formal"                 # V_th reference (std=1.5 wide band, same op point)
RUNNER = ROOT / "scripts/run_sef_hfo_snn_cm_spontaneous_readout.py"

# magnitudes — TUNED POST-PILOT for the 0.8-1.25x rate band (edit after a3_pilot).
MAG = {
    "oneend_inhib": dict(ei_scale=0.5),
    "oneend_recur": dict(ee_gain=1.5),
    "oneend_combined": dict(ei_scale=0.6, ee_gain=1.3),
}
SEEDS = [1, 2, 3]
T = 3500.0
THREADS_PER_RUN = 8
VTH_REF_STD = 1.5                       # the A1 'wide' V_th band used as the V_th-down reference
BASELINE_CLEAN = 15                     # V_th baseline oneend_neg clean count at this op point (T=3500)


def _tag(lesion: str, seed: int) -> str:
    return f"a3_{lesion}_s{seed}"


def _cells():
    for lesion in MAG:
        for seed in SEEDS:
            yield lesion, seed


def _run_cell(lesion: str, seed: int) -> dict:
    tag = _tag(lesion, seed)
    if (OUT / f"readout_{tag}.json").exists():
        return {"tag": tag, "rc": 0, "skipped": True}
    args = []
    for k, v in MAG[lesion].items():
        args += [f"--{k.replace('_', '-')}", str(v)]
    log = OUT / "logs" / f"{tag}.log"
    cmd = [sys.executable, str(RUNNER), "--lesion", lesion, "--seed", str(seed),
           "--T", str(T), "--tag", tag, "--out", str(OUT)] + args
    env = dict(os.environ)
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[k] = str(THREADS_PER_RUN)
    with open(log, "w") as fh:
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env).returncode
    return {"tag": tag, "rc": rc, "skipped": False}


def run_grid(workers: int) -> None:
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    cells = list(_cells())
    print(f"A3 E/I: {len(cells)} cells (lesion x seed) T={T}, workers={workers}, MAG={MAG}")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_run_cell, *c): c for c in cells}
        for f in as_completed(futs):
            r = f.result()
            print(f"  done {r['tag']} rc={r['rc']}" + (" (resumed)" if r.get("skipped") else ""))


# ---------------------------------------------------------------------------
def _cliffs_delta(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size == 0 or b.size == 0:
        return float("nan")
    gt = sum((x > b).sum() for x in a); lt = sum((x < b).sum() for x in a)
    return (gt - lt) / (a.size * b.size)


def _pool_clean(readout_dir: Path, tag: str):
    jp = readout_dir / f"readout_{tag}.json"
    npz = readout_dir / "per_event" / f"rep_{tag}.npz"
    if not (jp.exists() and npz.exists()):
        return None
    rf = extract_fingerprint(jp, npz)
    rj = json.loads(jp.read_text())
    widths, errs, firsts, signs = [], [], [], []
    for row in rf.events:
        if not _is_clean(rj["events"][row.event_index]):
            continue
        if row.pathway_width["value_mm"] is not None:
            widths.append(row.pathway_width["value_mm"])
        if row.axis_dir["axis_err_deg"] is not None:
            errs.append(row.axis_dir["axis_err_deg"])
        if row.first_contact is not None:
            firsts.append(row.first_contact)
        if row.direction_sign is not None:
            signs.append(row.direction_sign)
    return dict(rf=rf, widths=widths, errs=errs, firsts=firsts, signs=signs,
                n_clean=rf.n_clean_events, n_total=rf.n_events_total)


def analyze() -> None:
    from scipy import stats
    from src.sef_hfo_stage3 import entry_jitter_stats

    # V_th reference = pooled A1 'wide' std=1.5 runs (same op point/T/seeds)
    vth = dict(widths=[], errs=[], firsts=[], per_seed=[])
    for seed in SEEDS:
        pc = _pool_clean(A1, f"a1_neg_m17.0_std{VTH_REF_STD}_s{seed}")
        if pc:
            vth["widths"] += pc["widths"]; vth["errs"] += pc["errs"]; vth["firsts"] += pc["firsts"]
            vth["per_seed"].append(pc["n_clean"])

    lesions = {les: dict(widths=[], errs=[], firsts=[], signs=[], per_seed_clean=[], per_seed_total=[])
               for les in MAG}
    per_run = []
    for lesion, seed in _cells():
        pc = _pool_clean(OUT, _tag(lesion, seed))
        if pc is None:
            per_run.append(dict(lesion=lesion, seed=seed, status="MISSING")); continue
        L = lesions[lesion]
        L["widths"] += pc["widths"]; L["errs"] += pc["errs"]; L["firsts"] += pc["firsts"]
        L["signs"] += pc["signs"]; L["per_seed_clean"].append(pc["n_clean"])
        L["per_seed_total"].append(pc["n_total"])
        per_run.append(dict(lesion=lesion, seed=seed, status="ok", n_clean=pc["n_clean"],
                            n_total=pc["n_total"], insufficient=pc["rf"].insufficient,
                            pathway_width_med=pc["rf"].aggregate["pathway_width"]["median_mm"],
                            axis_err_med=pc["rf"].aggregate["axis_dir"]["axis_err_median_deg"],
                            sign_majority=pc["rf"].aggregate["axis_dir"]["sign_majority"]))

    def summ(d):
        w = d["widths"]; signs = d["signs"]
        sign_maj = float(np.sign(np.sum(signs))) if signs else None
        return dict(
            n_clean_total=len(w), per_seed_clean=d.get("per_seed_clean"),
            pathway_width_median_mm=(round(float(np.median(w)), 4) if w else None),
            axis_err_median_deg=(round(float(np.median(d["errs"])), 4) if d["errs"] else None),
            sign_majority=sign_maj,
            sign_consistency=(round(float(np.mean(np.array(signs) == sign_maj)), 4) if signs else None),
            onset_jitter=entry_jitter_stats(d["firsts"]),
            mean_clean_per_seed=(round(float(np.mean(d["per_seed_clean"])), 2) if d.get("per_seed_clean") else None))

    out_lesions = {}
    for les in MAG:
        s = summ(lesions[les])
        # rate-match: per-seed clean mean vs baseline band
        mc = s["mean_clean_per_seed"]
        s["rate_match"] = dict(
            baseline_clean=BASELINE_CLEAN,
            ratio=(round(mc / BASELINE_CLEAN, 3) if mc else None),
            in_band=(bool(0.8 * BASELINE_CLEAN <= (mc or 0) <= 1.25 * BASELINE_CLEAN) if mc else False),
            note="if NOT in band: re-tune MAG[lesion]; comparison is 'inconclusive' until matched.")
        # produces clean directional templates? (axis readable + sign consistent + enough events)
        s["clean_directional_templates"] = bool(
            s["n_clean_total"] >= N_MIN_EVENTS_DEFAULT and s["axis_err_median_deg"] is not None
            and (s["sign_consistency"] or 0) >= 0.8)
        # fingerprint vs V_th reference (pathway_width)
        if lesions[les]["widths"] and vth["widths"]:
            u = stats.mannwhitneyu(lesions[les]["widths"], vth["widths"], alternative="two-sided")
            s["pathway_width_vs_vth"] = dict(
                p=round(float(u.pvalue), 5),
                cliffs_delta=round(_cliffs_delta(lesions[les]["widths"], vth["widths"]), 4),
                vth_median_mm=round(float(np.median(vth["widths"])), 4))
        out_lesions[les] = s

    # cross-E/I pathway_width
    groups = [lesions[les]["widths"] for les in MAG if lesions[les]["widths"]]
    cross = (dict(kruskal_p=round(float(stats.kruskal(*groups).pvalue), 5))
             if len(groups) >= 2 and all(len(g) for g in groups) else dict(kruskal_p=None))
    cross["pairwise"] = {}
    for a, b in combinations(list(MAG), 2):
        ga, gb = lesions[a]["widths"], lesions[b]["widths"]
        if ga and gb:
            u = stats.mannwhitneyu(ga, gb, alternative="two-sided")
            cross["pairwise"][f"{a}_vs_{b}"] = dict(p=round(float(u.pvalue), 5),
                                                    cliffs_delta=round(_cliffs_delta(ga, gb), 4))

    out = dict(
        stage="A3 local E/I lesion -> fingerprint (event-rate matched). 在读出层面 E/I 病灶能/不能"
              "复现 V_th↓ 的双向模板(事件率匹配前提下);NOT 'E/I 机制被证实'.",
        T=T, seeds=SEEDS, magnitudes=MAG, baseline_clean=BASELINE_CLEAN,
        vth_reference=dict(std=VTH_REF_STD, n_clean=len(vth["widths"]), per_seed=vth["per_seed"],
                           pathway_width_median_mm=(round(float(np.median(vth["widths"])), 4)
                                                    if vth["widths"] else None)),
        per_run=per_run, lesions=out_lesions, cross_ei=cross)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "a3_ei_results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"wrote {OUT/'a3_ei_results.json'}")
    for les in MAG:
        s = out_lesions[les]
        print(f"  {les}: n_clean={s['n_clean_total']} mean/seed={s['mean_clean_per_seed']} "
              f"rate_in_band={s['rate_match']['in_band']} clean_templates={s['clean_directional_templates']} "
              f"pw_med={s['pathway_width_median_mm']} sign_cons={s['sign_consistency']}")
    _figure(out_lesions, lesions, vth)


def _figure(out_lesions, lesions, vth) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = list(MAG)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    data = [vth["widths"]] + [lesions[k]["widths"] for k in keys]
    labels = ["V_th↓ (wide)"] + [k.replace("oneend_", "") for k in keys]
    ax[0].boxplot(data, labels=labels, showmeans=True)
    ax[0].set_ylabel("pathway_width (mm, perp to locked axis)")
    ax[0].set_title("E/I lesion fingerprint vs V_th↓")
    top1 = [(out_lesions[k]["onset_jitter"].get("top1_fraction") or 0) for k in keys]
    ax[1].bar([k.replace("oneend_", "") for k in keys], top1)
    ax[1].set_ylim(0, 1.05); ax[1].set_ylabel("onset top1 fraction")
    ax[1].set_title("entry concentration per E/I lesion")
    fig.suptitle(f"A3 E/I-lesion fingerprint (rate-matched, T={T}, seeds {SEEDS})")
    fig.tight_layout()
    fd = OUT / "figures"; fd.mkdir(parents=True, exist_ok=True)
    fig.savefig(fd / "a3_ei_fingerprint.png", dpi=130); plt.close(fig)
    print(f"wrote {fd/'a3_ei_fingerprint.png'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()
    if not args.analyze_only:
        run_grid(args.workers)
    analyze()


if __name__ == "__main__":
    main()
