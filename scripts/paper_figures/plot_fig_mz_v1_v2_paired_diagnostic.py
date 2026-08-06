"""V1 (z-only) vs V2 (z+m, tau_adp=500) paired comparison + compact 3-seed diagnostic (task §10/§11).

Reads the per-seed bridge_metrics.json of both bridges, pairs BY SEED (never treating the 3 seeds as 3
patients or V1+V2 as 6 samples), writes v1_vs_v2_comparison.{json,csv}, and draws a compact 2-panel
diagnostic (same-seed contact maxAB and operational-runaway onset latency, V1 vs V2). This diagnostic is
NOT Figure 5.

Usage:
  python scripts/paper_figures/plot_fig_mz_v1_v2_paired_diagnostic.py [--v1-root ...] [--v2-root ...]
"""
import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
V1_ROOT = os.path.join(ROOT, "results", "topic4_sef_hfo", "mz_early_field_bridge")
V2_ROOT = os.path.join(ROOT, "results", "topic4_sef_hfo", "mz_early_field_bridge_v2_zm_tau500")
FIG_DIR = os.path.join(ROOT, "results", "paper-ready-figure", "fig_mz_early_bridge_v2_zm_tau500", "figures")
WK = "early_0_50_ms"


def _seed_metrics(root, seed):
    """Extract the comparison fields from one bridge_metrics.json, or None if absent/incomplete."""
    p = os.path.join(root, "per_seed", f"seed{seed}", "bridge_metrics.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    if d.get("status") != "complete":
        return {"seed": seed, "status": d.get("status")}
    w = (d.get("by_window") or {}).get(WK, {})
    ca = (w.get("contact") or {}).get("all_support", {})
    mx = ca.get("maxab") or {}
    ws = ca.get("within_shaft_null") or {}
    un = ca.get("unrestricted_null") or {}
    fd = w.get("contact_field_diag") or {}
    lp = w.get("local_participation") or {}
    sa = (w.get("source") or {}).get("all_support", {})
    smx = sa.get("maxab") or {}
    tor = sa.get("toroidal_null") or {}
    rho_a, rho_b = mx.get("rho_a"), mx.get("rho_b")
    winner = None
    if isinstance(rho_a, (int, float)) and isinstance(rho_b, (int, float)):
        winner = "B_to_A" if rho_b >= rho_a else "A_to_B"
    return {
        "seed": seed, "status": "complete",
        "t120_ms": d.get("t120_ms"), "t_recruit_ms": (d.get("onset") or {}).get("t_recruit_ms"),
        "onset_diff_ms": (d.get("onset") or {}).get("onset_diff_ms"),
        "n_returning_events": d.get("n_returning_events"),
        "maxab_eligible": d.get("maxab_eligible"),
        "rho_maxab": mx.get("rho_maxab"), "rho_a": rho_a, "rho_b": rho_b,
        "winner_direction": winner, "single_direction": mx.get("single_direction"),
        "within_shaft_p": ws.get("p_one_sided"), "unrestricted_p": un.get("p_one_sided"),
        "energy_dynamic_range": fd.get("dynamic_range"), "recruited_contacts": fd.get("recruited"),
        "field_support": fd.get("support"),
        "local_participation_status": lp.get("status"), "local_participation_median": lp.get("median"),
        "source_rho_maxab": smx.get("rho_maxab"), "source_toroidal_p": tor.get("p_one_sided"),
        "preflight_gate": d.get("preflight_gate"),
    }


def _delta(a, b):
    return (b - a) if isinstance(a, (int, float)) and isinstance(b, (int, float)) else None


def build_comparison(v1_root, v2_root, seeds):
    rows = []
    for s in seeds:
        v1, v2 = _seed_metrics(v1_root, s), _seed_metrics(v2_root, s)
        rows.append({"seed": s, "v1_zonly": v1, "v2_zm": v2,
                     "delta": {"d_t120_ms": _delta((v1 or {}).get("t120_ms"), (v2 or {}).get("t120_ms")),
                               "d_maxab": _delta((v1 or {}).get("rho_maxab"), (v2 or {}).get("rho_maxab")),
                               "d_recruited": _delta((v1 or {}).get("recruited_contacts"), (v2 or {}).get("recruited_contacts")),
                               "d_source_maxab": _delta((v1 or {}).get("source_rho_maxab"), (v2 or {}).get("source_rho_maxab")),
                               "same_winner": ((v1 or {}).get("winner_direction") == (v2 or {}).get("winner_direction")
                                               if v1 and v2 else None)}})
    return rows


def _write_csv(rows, path):
    flat = []
    for r in rows:
        for tag in ("v1_zonly", "v2_zm"):
            m = r.get(tag) or {}
            if m.get("status") == "complete":
                flat.append({"seed": r["seed"], "variant": tag, **{k: v for k, v in m.items()
                             if k not in ("seed", "preflight_gate")}})
    if not flat:
        return
    keys = list(flat[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in flat:
            w.writerow(r)


def plot_diagnostic(rows, out_png):
    complete = [r for r in rows if (r.get("v1_zonly") or {}).get("status") == "complete"
                and (r.get("v2_zm") or {}).get("status") == "complete"]
    seeds = [r["seed"] for r in complete]
    if not complete:
        print("[diagnostic] no seed complete in BOTH V1 and V2 yet; skipping figure")
        return None
    x = np.arange(len(seeds)); w = 0.38
    fig, (axm, axo) = plt.subplots(1, 2, figsize=(9.0, 3.8))

    # panel A: same-seed contact maxAB, V1 vs V2, star when within-shaft p<0.05
    v1m = [(r["v1_zonly"]["rho_maxab"] or np.nan) for r in complete]
    v2m = [(r["v2_zm"]["rho_maxab"] or np.nan) for r in complete]
    axm.bar(x - w / 2, v1m, w, label="V1 z-only", color="#6c8ebf", edgecolor="black", linewidth=0.6)
    axm.bar(x + w / 2, v2m, w, label="V2 z+m", color="#b85450", edgecolor="black", linewidth=0.6)
    for i, r in enumerate(complete):
        for dx, m, pk in ((-w / 2, v1m[i], r["v1_zonly"]["within_shaft_p"]),
                          (w / 2, v2m[i], r["v2_zm"]["within_shaft_p"])):
            if isinstance(pk, (int, float)) and pk < 0.05 and np.isfinite(m):
                axm.text(i + dx, m + 0.02, "*", ha="center", va="bottom", fontsize=14, fontweight="bold")
    axm.set_xticks(x); axm.set_xticklabels([f"seed {s}" for s in seeds])
    axm.set_ylabel("same-seed contact maxAB")
    axm.set_ylim(0, 1.05); axm.set_title("Interictal-axis prediction of\npre-runaway energy (maxAB)", fontsize=10)
    axm.legend(fontsize=8, frameon=False, loc="lower right")

    # panel B: operational-runaway onset latency (t120), V1 vs V2
    v1t = [(r["v1_zonly"]["t120_ms"] or np.nan) / 1000.0 for r in complete]
    v2t = [(r["v2_zm"]["t120_ms"] or np.nan) / 1000.0 for r in complete]
    axo.bar(x - w / 2, v1t, w, label="V1 z-only", color="#6c8ebf", edgecolor="black", linewidth=0.6)
    axo.bar(x + w / 2, v2t, w, label="V2 z+m", color="#b85450", edgecolor="black", linewidth=0.6)
    axo.set_xticks(x); axo.set_xticklabels([f"seed {s}" for s in seeds])
    axo.set_ylabel("operational-runaway onset t120 (s)")
    axo.set_title("Onset latency: adding m\ndelays the transition", fontsize=10)
    axo.legend(fontsize=8, frameon=False, loc="lower right")

    fig.suptitle("E1146 z-only (V1) vs z+m tau500 (V2) — paired by noise seed (n=3 seeds, one substrate; not a cohort)",
                 fontsize=10, y=1.02)
    fig.text(0.008, 0.004, "* within-shaft p < 0.05", fontsize=8, color="0.4")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    fig.savefig(out_png.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[diagnostic] wrote {out_png} (+pdf) for seeds {seeds}")
    return out_png


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1-root", default=V1_ROOT)
    ap.add_argument("--v2-root", default=V2_ROOT)
    ap.add_argument("--out-dir", default=V2_ROOT)
    ap.add_argument("--fig-dir", default=FIG_DIR)
    ap.add_argument("--seeds", default="1,3,4")
    args = ap.parse_args(argv)
    seeds = [int(s) for s in args.seeds.split(",")]
    rows = build_comparison(args.v1_root, args.v2_root, seeds)
    os.makedirs(args.out_dir, exist_ok=True)
    json.dump({"experiment": "V1 z-only vs V2 z+m tau500 paired-by-seed comparison (task §10)",
               "note": "3 noise seeds on ONE E1146 substrate; NOT 3 patients and NOT 6 independent samples.",
               "primary_window": WK, "rows": rows},
              open(os.path.join(args.out_dir, "v1_vs_v2_comparison.json"), "w"), indent=2)
    _write_csv(rows, os.path.join(args.out_dir, "v1_vs_v2_comparison.csv"))
    plot_diagnostic(rows, os.path.join(args.fig_dir, "fig_mz_v1_v2_paired_diagnostic.png"))
    print(f"[compare] wrote v1_vs_v2_comparison.{{json,csv}} -> {args.out_dir}")


if __name__ == "__main__":
    main()
