# scripts/run_sef_hfo_step0b_pulse.py
"""Step 0b: finite-pulse response map over 0a's CANDIDATE operating points, recovery OFF
and ON (both reported, no auto-select), L/grid + dt sensitivity, plus the framework
Step-0b output contract: full response surface + margin waterfall + example snapshots."""
import argparse, json
from dataclasses import replace
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import sys; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root on path (run as `python scripts/X.py`)
from src.sef_hfo_field import SEFParams
from src.sef_hfo_stability import self_consistent_operating_point
from src.sef_hfo_pulse import PULSE_FAMILY, run_pulse, amplitude_thresholds, _centroid_x
OUT = Path("results/topic4_sef_hfo/finite_pulse")
A_LO, A_HI, A_SURF = 0.2, 3.0, 12
CLASSES = ["extinction", "local_bump", "self_limited_propagation", "global_synchronous", "runaway"]
CLS_CMAP = ListedColormap(["#dddddd", "#9ecae1", "#31a354", "#fdae6b", "#de2d26"])

def _memo_label_fn(p, op, I_E, I_I, r, T, dt, t_max):
    """A REAL pulse runner with a cache (fixes the nearest-label hack): every distinct A
    runs a fresh simulation; repeats are reused. amplitude_thresholds' refine therefore
    runs genuine pulses at fine amplitudes."""
    cache = {}
    def fn(A):
        key = round(float(A), 4)
        if key not in cache:
            cache[key] = run_pulse(p, op, I_E, I_I, r, T, A, dt, t_max)
        return cache[key]
    return fn

def scan_point(p, I_E, I_I, dt, t_max):
    op = self_consistent_operating_point(p, I_E, I_I)
    if not op.get("converged"): return None
    a_surf = list(np.linspace(A_LO, A_HI, A_SURF))
    cells = []
    for r in PULSE_FAMILY["radii"]:
        for T in PULSE_FAMILY["durations"]:
            lf = _memo_label_fn(p, op, I_E, I_I, r, T, dt, t_max)
            grid = [lf(A) for A in a_surf]                       # response-surface row (real sims)
            thr = amplitude_thresholds(lf, A_LO, A_HI)           # real refine reuses cache
            has_win = (np.isfinite(thr["A_self_limited"]) and np.isfinite(thr["safety_margin"])
                       and thr["safety_margin"] > 0)
            cells.append({"r": r, "T": T, **thr, "grid": grid, "has_self_limited_window": bool(has_win)})
    return {"I_E": I_E, "I_I": I_I, "op": op, "a_surf": a_surf, "cells": cells}

def family_summary(p, family, dt, t_max):
    pts = [sp for (I_E, I_I) in family if (sp := scan_point(p, I_E, I_I, dt, t_max)) is not None]
    for sp in pts:
        sp["has_window"] = any(c["has_self_limited_window"] for c in sp["cells"])
    n = len(pts); k = sum(sp["has_window"] for sp in pts)
    return {"n_candidates": n, "n_with_window": k, "fraction_with_window": (k / n if n else 0.0), "points": pts}

def _find_point(summ, I_E, I_I):
    return next((sp for sp in summ["points"] if sp["I_E"] == I_E and sp["I_I"] == I_I), None)

def plot_response_surface(off, on, out):
    rep = next((sp for sp in off["points"] if sp["has_window"]), off["points"][0] if off["points"] else None)
    if rep is None: return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, (name, summ) in zip(axes, [("recovery OFF", off), ("recovery ON", on)]):
        sp = _find_point(summ, rep["I_E"], rep["I_I"])
        if sp is None: continue
        mat = np.array([[CLASSES.index(l) for l in c["grid"]] for c in sp["cells"]])
        ax.imshow(mat, aspect="auto", cmap=CLS_CMAP, vmin=-0.5, vmax=4.5,
                  extent=[A_LO, A_HI, len(sp["cells"]) - 0.5, -0.5])
        ax.set_yticks(range(len(sp["cells"])))
        ax.set_yticklabels([f"r={c['r']:.0f} T={c['T']:.0f}" for c in sp["cells"]])
        ax.set_xlabel("pulse amplitude"); ax.set_title(f"{name}  (operating point I_E={rep['I_E']:.2f})")
    handles = [plt.Rectangle((0, 0), 1, 1, color=CLS_CMAP(i)) for i in range(5)]
    fig.legend(handles, [c.replace("_", " ") for c in CLASSES], loc="lower center", ncol=5, fontsize=8)
    fig.suptitle("Step 0b finite-pulse response surface (representative candidate point)")
    fig.tight_layout(rect=[0, 0.07, 1, 1]); fig.savefig(out / "figures" / "step0b_response_surface.png", dpi=140)
    plt.close(fig)

def _best_cell(sp):
    fin = [c for c in sp["cells"] if np.isfinite(c["safety_margin"])]
    pos = [c for c in fin if c["safety_margin"] > 0]
    return (max(pos, key=lambda c: c["safety_margin"]) if pos
            else (max(fin, key=lambda c: c["safety_margin"]) if fin else sp["cells"][0]))

def plot_margin_waterfall(off, on, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, (name, summ) in zip(axes, [("recovery OFF", off), ("recovery ON", on)]):
        pts = summ["points"]; x = np.arange(len(pts))
        a_sl = [c["A_self_limited"] if np.isfinite(c["A_self_limited"]) else np.nan
                for c in (_best_cell(sp) for sp in pts)]
        a_ru = [c["A_runaway"] if np.isfinite(c["A_runaway"]) else np.nan
                for c in (_best_cell(sp) for sp in pts)]
        ax.vlines(x, a_sl, a_ru, color="0.75", lw=6)             # safety-margin band
        ax.plot(x, a_sl, "o", color="#31a354", label="A_self_limited")
        ax.plot(x, a_ru, "s", color="#de2d26", label="A_runaway")
        ax.set_xticks(x); ax.set_xticklabels([f"{sp['I_E']:.2f}" for sp in pts], rotation=90, fontsize=7)
        ax.set_xlabel("candidate operating point (I_E)"); ax.set_title(name)
    axes[0].set_ylabel("pulse amplitude"); axes[0].legend(fontsize=8)
    fig.suptitle("Step 0b safety margin per candidate point  (band = A_runaway - A_self_limited)")
    fig.tight_layout(); fig.savefig(out / "figures" / "step0b_margin_waterfall.png", dpi=140); plt.close(fig)

def _find_example(summ, label):
    for sp in summ["points"]:
        for c in sp["cells"]:
            for A, lbl in zip(sp["a_surf"], c["grid"]):
                if lbl == label:
                    return (sp["I_E"], sp["I_I"], sp["op"], c["r"], c["T"], A)
    return None

def plot_example_snapshots(p, off, dt, t_max, out):
    rows = [(lbl, _find_example(off, lbl)) for lbl in
            ["self_limited_propagation", "global_synchronous", "runaway"]]
    rows = [(lbl, ex) for lbl, ex in rows if ex is not None]
    if not rows: return
    nt = 4; fig, axes = plt.subplots(len(rows), nt, figsize=(3 * nt, 3 * len(rows)))
    axes = np.atleast_2d(axes)
    for i, (lbl, (I_E, I_I, op, r, T, A)) in enumerate(rows):
        _, act = run_pulse(p, op, I_E, I_I, r, T, A, dt, t_max, return_activity=True)
        idxs = np.linspace(0, len(act) - 1, nt).astype(int)
        for j, ti in enumerate(idxs):
            axes[i, j].imshow(act[ti], cmap="magma")
            cx = _centroid_x(act[ti], op["r_E0"], 0.05)
            if not np.isnan(cx): axes[i, j].axvline(cx, color="cyan", lw=1.2)   # centroid -> moves if propagating
            axes[i, j].set_xticks([]); axes[i, j].set_yticks([]); axes[i, j].set_title(f"t={ti*dt:.0f}", fontsize=8)
        axes[i, 0].set_ylabel(lbl.replace("_", " "), fontsize=8)
    fig.suptitle("Step 0b example responses: snapshots + centroid (cyan) — propagation moves, flash does not")
    fig.tight_layout(); fig.savefig(out / "figures" / "step0b_example_snapshots.png", dpi=130); plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage0a", default="results/topic4_sef_hfo/linear_stability/step0a_stability.json")
    ap.add_argument("--dt", type=float, default=0.05); ap.add_argument("--t-max", type=float, default=40.0)
    ap.add_argument("--tau-a", type=float, default=15.0)
    args = ap.parse_args()
    a0 = json.loads(Path(args.stage0a).read_text())
    family = [(d["I_E"], d["I_I"]) for d in a0.get("candidate_operating_points", [])] \
             or [(d["I_E"], d["I_I"]) for d in a0["per_point"] if d.get("converged")]
    base = SEFParams()
    off = family_summary(replace(base, b_a=0.0), family, args.dt, args.t_max)
    on = family_summary(replace(base, b_a=1.0, tau_a=args.tau_a), family, args.dt, args.t_max)
    sens = {"half_dt": family_summary(replace(base, b_a=0.0), family, args.dt / 2, args.t_max)["fraction_with_window"],
            "smaller_L": family_summary(replace(base, b_a=0.0, n=48, L=48.0), family, args.dt, args.t_max)["fraction_with_window"]}
    OUT.mkdir(parents=True, exist_ok=True)
    def _strip(s):   # drop bulky 'op' before JSON (keep 'grid' for provenance of the surface)
        return {**s, "points": [{k: v for k, v in sp.items() if k != "op"} for sp in s["points"]]}
    (OUT / "step0b_pulse.json").write_text(json.dumps(
        {"family_size": len(family), "runs": {"recovery_off": _strip(off), "recovery_on": _strip(on)},
         "sensitivity": sens, "recovery_decision": "REPORT_BOTH_no_auto_select"}, indent=2, default=float))
    plot_response_surface(off, on, OUT)
    plot_margin_waterfall(off, on, OUT)
    plot_example_snapshots(replace(base, b_a=0.0), off, args.dt, args.t_max, OUT)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["off", "on"], [off["fraction_with_window"], on["fraction_with_window"]]); ax.set_ylim(0, 1)
    ax.set_ylabel("fraction of candidate points with self-limited window (+margin)")
    ax.set_title("Step 0b: recovery OFF vs ON (report both)")
    fig.tight_layout(); fig.savefig(OUT / "figures" / "step0b_window_fraction.png", dpi=140); plt.close(fig)
    print(f"[step0b] off={off['fraction_with_window']:.2f} on={on['fraction_with_window']:.2f} sens={sens}")

if __name__ == "__main__":
    main()
