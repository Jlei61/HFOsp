"""Topic5 A-line §3.2 window-sensitivity aggregation.

Reads the 12 per-(window x metric) alignment outputs (win_{window}_{act}_B1000.json) and asks:
which post-onset window aligns best, and does the guard-OUTSIDE distal pre window [-120,-90]
(the load-bearing negative control) fall well below the ictal windows?

Per (window, metric): cohort n_pass + patient-level effect-size (real - null_median) + bootstrap
95% CI for the channel layer (broadband PRIMARY) and the joint layer (hfa carries this). The
expectation 5-10 >= 0-10 >= 0-5 is reported, NOT a gate. The gate is: distal [-120,-90] effect-size
CI must contain 0 / be clearly below the ictal windows (alignment is ictal-specific, not a pure
anatomical anchor). Proximal [-10,0] (inside the [-60,0] peri-onset guard) is descriptive only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

WIN = Path("results/topic5_ictal_recruitment/axis_alignment/window")
WINDOWS = ["post_0_5", "post_5_10", "post_0_10", "post_0_20", "pre_prox_m10_0", "pre_distal_m120_m90"]
RNG_SEED = 20260615
MIN_EFF = 4   # a null must shuffle >= this many channels to count (drops degenerate joint nulls)


def _load(window, act):
    f = WIN / f"win_{window}_{act}_B1000.json"
    if not f.exists():
        return []
    j = json.load(open(f))
    return [r for r in j["per_subject"] if r.get("dataset") == "epilepsiae" and r.get("status") == "ok"]


def _effect_ci(diffs, seed=RNG_SEED, n_boot=10000):
    d = np.asarray([x for x in diffs if np.isfinite(x)], float)
    if d.size == 0:
        return None, None, None
    rng = np.random.default_rng(seed)
    boot = [np.median(d[rng.integers(0, d.size, d.size)]) for _ in range(n_boot)]
    return float(np.median(d)), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def _cohort(rows, layer):
    passk, medk = f"pass_{layer}_null", f"{layer}_null_median"
    sub = [r for r in rows if r.get(passk) is not None and r.get(medk) is not None]
    # adequate-only: drop degenerate nulls (joint on small/few-shaft subjects), same as main agg
    sub = [r for r in sub
           if (r.get("effective_shuffle_n") or {}).get(layer) is not None
           and r["effective_shuffle_n"][layer] >= MIN_EFF]
    if not sub:
        return None
    npass = sum(bool(r[passk]) for r in sub)
    diffs = [r["real_median_abs_corr"] - r[medk] for r in sub]
    eff, lo, hi = _effect_ci(diffs)
    return {"n": len(sub), "n_pass": npass,
            "binom_p": round(float(binomtest(npass, len(sub), 0.05, alternative="greater").pvalue), 5),
            "effect_size": round(eff, 5), "effect_ci": [round(lo, 5), round(hi, 5)],
            "real_median": round(float(np.median([r["real_median_abs_corr"] for r in sub])), 4)}


def main():
    summary = {"note": "Topic5 A-line window sensitivity. Distal [-120,-90] = load-bearing negative "
                       "control. Proximal [-10,0] descriptive only (inside peri-onset guard).",
               "rows": []}
    md = ["| window | metric | layer | n_pass/n | real_med | effect | 95% CI |",
          "|---|---|---|---|---|---|---|"]
    for act, layer in [("broadband", "channel"), ("hfa", "joint"), ("hfa", "channel")]:
        for w in WINDOWS:
            rows = _load(w, act)
            c = _cohort(rows, layer)
            if c is None:
                print(f"[win-agg] MISSING/empty {w} {act} {layer}", flush=True)
                continue
            rec = {"window": w, "metric": act, "layer": layer, **c}
            summary["rows"].append(rec)
            md.append(f"| {w} | {act} | {layer} | {c['n_pass']}/{c['n']} | {c['real_median']} | "
                      f"{c['effect_size']} | [{c['effect_ci'][0]}, {c['effect_ci'][1]}] |")
            print(f"[{w:22s} {act:10s} {layer:8s}] pass {c['n_pass']}/{c['n']} "
                  f"real_med={c['real_median']} eff={c['effect_size']} CI={c['effect_ci']}", flush=True)
    # ICTAL-SPECIFICITY test (reviewer P1): per-subject paired post(0-10) - distal(pre) on the RAW
    # |alignment|. post >> distal => ictal-early-specific; post ~ distal => persistent scaffold (the
    # axis reads the same coarse structure pre-onset and at onset). This is THE quantity that decides
    # whether we may say "ictal-specific" or only "stable scaffold readout".
    from scipy.stats import wilcoxon as _wil
    spec = {}
    for act in ["broadband", "hfa"]:
        post = {r["subject_id"]: r for r in _load("post_0_10", act)}
        dist = {r["subject_id"]: r for r in _load("pre_distal_m120_m90", act)}
        common = sorted(set(post) & set(dist))
        diffs = [post[s]["real_median_abs_corr"] - dist[s]["real_median_abs_corr"] for s in common]
        if len(diffs) < 3:
            continue
        eff, lo, hi = _effect_ci(diffs)
        try:
            wp = float(_wil(diffs, alternative="greater").pvalue) if any(d != 0 for d in diffs) else 1.0
        except ValueError:
            wp = 1.0
        spec[act] = {"n_paired": len(common),
                     "median_post_minus_distal_abscorr": round(eff, 5), "ci": [round(lo, 5), round(hi, 5)],
                     "wilcoxon_post_gt_distal_p": round(wp, 5),
                     "ictal_specific": bool(lo is not None and lo > 0 and wp < 0.05)}
        print(f"[ictal-specificity {act}] post-distal |corr| median={eff:.4f} CI=[{lo:.4f},{hi:.4f}] "
              f"Wilcox(post>distal) p={wp:.4f} -> ictal_specific={spec[act]['ictal_specific']}", flush=True)
    summary["ictal_specificity"] = spec

    # Negative-control verdict, derived HONESTLY from the ictal-specificity paired test (NOT from a
    # lenient 'distal CI contains 0'): the alignment is ictal-early-specific ONLY if post(0-10) is
    # significantly stronger than distal pre. Both metrics fail this -> persistent scaffold.
    verdict = {}
    for act, layer in [("broadband", "channel"), ("hfa", "joint")]:
        d = next((r for r in summary["rows"] if r["window"] == "pre_distal_m120_m90"
                  and r["metric"] == act and r["layer"] == layer), None)
        i = next((r for r in summary["rows"] if r["window"] == "post_0_10"
                  and r["metric"] == act and r["layer"] == layer), None)
        sp = spec.get(act, {})
        if d and i:
            is_spec = bool(sp.get("ictal_specific"))
            verdict[f"{act}_{layer}"] = {
                "distal_effect": d["effect_size"], "ictal_0_10_effect": i["effect_size"],
                "distal_n_pass": d["n_pass"], "ictal_0_10_n_pass": i["n_pass"],
                "post_minus_distal_abscorr": sp.get("median_post_minus_distal_abscorr"),
                "wilcoxon_post_gt_distal_p": sp.get("wilcoxon_post_gt_distal_p"),
                "ictal_specific": is_spec,
                "conclusion": "ictal_early_specific" if is_spec else "persistent_scaffold_NOT_ictal_specific"}
    summary["negative_control_verdict"] = verdict
    md.append("")
    md.append("## Ictal-specificity (paired post[0-10] - distal[-120,-90] on raw |alignment|)")
    md.append("post >> distal => ictal-specific; post ~ distal => persistent scaffold (NOT ictal-specific).")
    md.append("")
    md.append("| metric | n | median(post-distal) | 95% CI | Wilcox(post>distal) p | ictal_specific |")
    md.append("|---|---|---|---|---|---|")
    for act, s in spec.items():
        md.append(f"| {act} | {s['n_paired']} | {s['median_post_minus_distal_abscorr']} | "
                  f"[{s['ci'][0]}, {s['ci'][1]}] | {s['wilcoxon_post_gt_distal_p']} | {s['ictal_specific']} |")
    out = WIN / "window_summary.json"
    json.dump(summary, open(out, "w"), indent=2, ensure_ascii=False)
    (WIN / "window_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nwrote {out}\nnegative-control verdict: {json.dumps(verdict, ensure_ascii=False)}")
    print(f"ictal-specificity: {json.dumps(spec, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
