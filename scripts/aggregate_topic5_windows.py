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
    # explicit negative-control verdict: distal vs 0-10 for broadband-channel + hfa-joint
    verdict = {}
    for act, layer in [("broadband", "channel"), ("hfa", "joint")]:
        d = next((r for r in summary["rows"] if r["window"] == "pre_distal_m120_m90"
                  and r["metric"] == act and r["layer"] == layer), None)
        i = next((r for r in summary["rows"] if r["window"] == "post_0_10"
                  and r["metric"] == act and r["layer"] == layer), None)
        if d and i:
            ci_contains_0 = d["effect_ci"][0] <= 0 <= d["effect_ci"][1]
            below_ictal = d["effect_size"] < i["effect_size"]
            verdict[f"{act}_{layer}"] = {"distal_effect": d["effect_size"], "ictal_0_10_effect": i["effect_size"],
                                         "distal_ci_contains_0": ci_contains_0, "distal_below_ictal": below_ictal,
                                         "negative_control_ok": bool(ci_contains_0 or below_ictal)}
    summary["negative_control_verdict"] = verdict
    out = WIN / "window_summary.json"
    json.dump(summary, open(out, "w"), indent=2, ensure_ascii=False)
    (WIN / "window_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nwrote {out}\nnegative-control verdict: {json.dumps(verdict, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
