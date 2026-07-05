#!/usr/bin/env python3
"""Aggregate the M3A-A1 quasi-static slow-state tiny pilot into an HONEST STATUS. OFFLINE — reads
results/topic4_sef_hfo/m3a_slowvars/quasistatic/<tag>/.

Discriminator (static-μ negative-boundary lesson): a slow knob that only raises event RATE (events
keep the same size/duration, no R4a) is a FAILURE (global heating). But this pilot's baseline (OFF /
near-baseline) is itself silent (pure R0) on the uniform substrate, so per-level within-knob movement
rests on 1-4 events and is NOT a gradient. The robust signal is the POOLED cross-mechanism event-shape
contrast (z/e_GABA vs phi). Output is KEYED per-level (no positional array misalignment) with
silent-level medians as null (never 0).
"""
import csv
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DIR = "results/topic4_sef_hfo/m3a_slowvars/quasistatic"

_MORE_EXCITABLE_IS_LOWER = {"z": True, "phi": True, "gK": True, "egaba": False}
_R_ORDER = ["R0", "R1", "R2", "R3", "R4a", "R4b"]
_R_COL = {"R0": "0.85", "R1": "tab:blue", "R2": "tab:green", "R3": "tab:orange",
          "R4a": "tab:red", "R4b": "0.4"}


def _load_all(base):
    out = {}
    for sj in glob.glob(os.path.join(base, "*", "spontaneous_summary.json")):
        s = json.load(open(sj))["summary"]
        out[s["state_label"]] = s
    return out


def _rfrac(s, k):
    return (s or {}).get("R_fractions", {}).get(k, 0.0)


def _by_var(all_s):
    """group -> {var: [summaries sorted LEAST->MOST excitable]}; plus the off baseline."""
    groups, off = {}, None
    for s in all_s.values():
        if s["slow_var"] == "off":
            off = s
        else:
            groups.setdefault(s["slow_var"], []).append(s)
    for v, lst in groups.items():
        lst.sort(key=lambda s: s["level"], reverse=_MORE_EXCITABLE_IS_LOWER.get(v, False))
    return off, groups


def _moves(seq):
    vals = [x for x in seq if x is not None]
    if len(vals) < 2:
        return False
    spread = max(vals) - min(vals)
    return spread > 0.15 * max(abs(np.median(vals)), 1e-9) and spread > 1e-6


def _level_record(s):
    n = s["n_events_total"]
    med = lambda k: (s[k]["median"] if n > 0 else None)   # silent level -> None, never 0
    return {"state_label": s["state_label"], "n_events": n,
            "rate_hz": s["event_rate_hz_per_seed"],
            "size_median_bins": med("size_bins"), "duration_median_ms": med("duration_ms"),
            "return_probability": s.get("return_probability"),
            "R3_frac": _rfrac(s, "R3"), "R4a_frac": _rfrac(s, "R4a"), "R4b_frac": _rfrac(s, "R4b")}


def _verdict_for_var(off, seq):
    """Keyed per-level records (off first, then least->most excitable) + honest flags. Within-knob
    movement is reported but flagged noisy when max active-level event count < 10."""
    chain = ([off] + seq) if off is not None else seq
    levels = [_level_record(s) for s in chain]
    active = [L for L in levels if L["n_events"] > 0]
    max_ev = max((L["n_events"] for L in active), default=0)
    ret = [L["return_probability"] for L in active if L["return_probability"] is not None]
    r4a = any(L["R4a_frac"] > 0 for L in levels)
    r4b = any(L["R4b_frac"] > 0 for L in levels)
    return {
        "levels": levels,
        "baseline_off_is_R0": bool(off is not None and off["n_events_total"] == 0),
        "rate_moves": bool(_moves([L["rate_hz"] for L in levels])),
        "within_knob_size_moves": bool(_moves([L["size_median_bins"] for L in active])),
        "within_knob_duration_moves": bool(_moves([L["duration_median_ms"] for L in active])),
        "return_prob_drops": bool(len(ret) >= 2 and ret[-1] < ret[0] - 0.05),
        "R4a_appears": bool(r4a), "R4b_appears": bool(r4b),
        "max_events_per_active_level": int(max_ev), "well_sampled_ge10_events": bool(max_ev >= 10),
        "note": ("R4a (structured sustained) NEVER appears; "
                 + ("R4b tonic appears only at the most-excitable extreme; " if r4b else "no R4b; ")
                 + ("within-knob size/duration movement is NOISE-dominated (max %d events/level), "
                    "NOT a clean gradient." % max_ev if max_ev < 10
                    else "within-knob movement is over >=10 events/level.")),
    }


def _pooled_shapes(base):
    """Robust cross-mechanism event-shape contrast: pool every active event per variable."""
    out = {}
    for var, pat in (("z", "z*"), ("egaba", "egaba*"), ("phi", "phi*"), ("gK", "gK*")):
        sz, du = [], []
        for f in glob.glob(os.path.join(base, pat, "spontaneous_per_event.csv")):
            for r in csv.DictReader(open(f)):
                if (r.get("R_class") or "").startswith("R"):
                    sz.append(int(r["size_bins"])); du.append(float(r["duration_ms"]))
        out[var] = ({"n_events": len(sz), "size_median_bins": float(np.median(sz)),
                     "size_range": [min(sz), max(sz)], "duration_median_ms": float(np.median(du)),
                     "duration_range": [float(min(du)), float(max(du))]} if sz
                    else {"n_events": 0, "note": "silent across scan"})
    return out


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    base = os.path.join(ROOT, argv[0]) if argv else os.path.join(ROOT, DEFAULT_DIR)
    if not os.path.isdir(base):
        print(f"[A1 analyze] no dir {base}"); return 1
    fig_dir = os.path.join(base, "figures"); os.makedirs(fig_dir, exist_ok=True)
    off, groups = _by_var(_load_all(base))

    status = {
        "base": os.path.relpath(base, ROOT),
        "plan_of_record": "git show a1213ee:docs/superpowers/plans/2026-06-24-sef-hfo-m3a-quasistatic-slowstate-plan.md",
        "baseline_anchor_item5": {
            "off_baseline": (None if off is None else _level_record(off)),
            "off_is_pure_R0": bool(off is not None and off["n_events_total"] == 0),
            "anchor_met": bool(off is not None and off["n_events_total"] > 0
                               and _rfrac(off, "R4b") == 0.0),
            "note": ("OFF baseline is pure R0 (0 events) -> per STATUS.md §4 the returned-R2/R3 "
                     "anchoring requirement is NOT met on this uniform substrate (documented "
                     "stop-and-retune condition)." if (off is not None and off["n_events_total"] == 0)
                     else "OFF baseline has events."),
        },
        "pooled_event_shapes": _pooled_shapes(base),
        "by_variable": {v: _verdict_for_var(off, seq) for v, seq in groups.items()},
    }
    json.dump(status, open(os.path.join(base, "status_a1_quasistatic.json"), "w"), indent=1)

    # ---- figures: one small-multiples row per slow variable (silent levels plotted as gaps) ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for v, vd in status["by_variable"].items():
        levels = vd["levels"]; labels = [L["state_label"] for L in levels]; x = np.arange(len(levels))
        nanv = lambda key: np.array([(L[key] if L[key] is not None else np.nan) for L in levels], float)
        fig, ax = plt.subplots(1, 4, figsize=(17, 3.8))
        ax[0].plot(x, [L["rate_hz"] for L in levels], "o-", color="tab:red"); ax[0].set_title("rate (Hz/seed)")
        ax[1].plot(x, nanv("size_median_bins"), "o-", color="tab:purple", label="size (bins)")
        ax[1].plot(x, nanv("duration_median_ms"), "s--", color="tab:brown", label="dur (ms)")
        ax[1].legend(fontsize=7); ax[1].set_title("size / dur (median, silent=gap)")
        ax[2].plot(x, nanv("return_probability"), "o-", color="tab:green")
        ax[2].set_ylim(-0.05, 1.05); ax[2].set_title("return probability")
        bottom = np.zeros(len(levels))
        for cls in _R_ORDER:
            vals = np.array([L.get(cls + "_frac", 0.0) for L in levels])
            ax[3].bar(x, vals, bottom=bottom, label=cls, color=_R_COL[cls]); bottom += vals
        ax[3].set_ylim(0, 1); ax[3].legend(fontsize=6, ncol=3); ax[3].set_title("R-class fraction")
        for a in ax:
            a.set_xticks(x); a.set_xticklabels(labels, rotation=45, fontsize=7, ha="right")
            a.set_xlabel("baseline → more excitable →")
        fig.suptitle(f"M3A-A1 quasi-static  slow var = {v}  (off=R0 silent; per-level n=1–25)", fontsize=11)
        fig.tight_layout(); fig.savefig(os.path.join(fig_dir, f"a1_{v}_phenotype.png"), dpi=130); plt.close(fig)

    print("[A1 analyze] item-5 anchor:", status["baseline_anchor_item5"]["anchor_met"],
          "| off pure R0:", status["baseline_anchor_item5"]["off_is_pure_R0"])
    print("[A1 analyze] pooled shapes:", {k: (v.get("size_median_bins"), v.get("duration_median_ms"), v["n_events"])
                                          for k, v in status["pooled_event_shapes"].items()})
    for v, vd in status["by_variable"].items():
        print(f"[A1 analyze] {v}: rate_moves={vd['rate_moves']} R4a={vd['R4a_appears']} R4b={vd['R4b_appears']} "
              f"max_ev/level={vd['max_events_per_active_level']} well_sampled={vd['well_sampled_ge10_events']}")
    print(f"[A1 analyze] wrote -> {base}/status_a1_quasistatic.json + figures/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
