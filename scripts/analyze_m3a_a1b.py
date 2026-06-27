#!/usr/bin/env python3
"""Aggregate the A1b 2D state-topography grid (local-loop x global-restraint on the Stage-3 two-focus
core) into the activity-readout panels + a 2D state-class surface. OFFLINE.

A1b is a STATE-TOPOGRAPHY MAP, not a dynamic causal chain: it asks where the (local:global) ratio
lands among protective/silent -> interictal-like axial self-limited -> seizure-like large
synchronized recruitment -> runaway. The state is read from MANY indicators (not just event-rate):
global E rate, core-vs-surround rate, tonic duty cycle, spatial extent, collision_rate_returned_sidecar,
return-to-baseline fraction. Thresholds are DESCRIPTIVE (the raw metrics are reported alongside).
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DIR = "results/topic4_sef_hfo/m3a_slowvars/a1b_grid"
LOCAL_OF = {1.0: 0, 0.85: 1, 0.70: 2}     # core_ei_scale -> local level


def _fullfield(base, tag, NE):
    p = os.path.join(base, f"fullfield_{tag}.json")
    if not os.path.exists(p):
        return None, None
    e = json.load(open(p))["events"]
    if not e:
        return 0.0, 0.0
    return float(np.median([x["n_fired_E"] for x in e]) / NE), float(np.median([x["r95_mm"] for x in e]))


def _load(base):
    rows = []
    for f in glob.glob(os.path.join(base, "readout_*.json")):
        s = json.load(open(f)); c = s["config"]; sc = s.get("stage3_source_counts") or {}
        act = s.get("activity") or {}
        ev = s.get("events") or []
        ret_frac = (float(np.mean([bool(e.get("returned")) for e in ev])) if ev else None)
        n_sidecar = (sum(sc.get(k, 0) for k in ("neg_clean", "pos_clean", "collision", "ambiguous"))
                     if sc else 0)
        amb_rate = (sc.get("ambiguous", 0) / n_sidecar if n_sidecar else None)
        nf_frac, r95 = _fullfield(base, s["tag"], int(c["NE"]))
        nf, nr = s["n_clean_forward"], s["n_clean_reverse"]
        is_pos = bool(c.get("shunt_gaba"))
        rows.append(dict(
            tag=s["tag"], is_posctrl=is_pos, seed=int(c["seed"]),
            local=(None if is_pos else LOCAL_OF.get(round(c["core_ei_scale"], 2))),
            glob=(None if is_pos else c["global_ei_scale"]),
            local_global_ratio=c.get("local_global_ratio"),
            event_rate_hz=round(s["n_events"] / (c["T"] / 1000.0), 4), n_events=s["n_events"],
            global_E_rate_mean_hz=act.get("global_E_rate_mean_hz"),
            global_E_rate_p95_hz=act.get("global_E_rate_p95_hz"),
            tonic_fraction=act.get("tonic_fraction"),
            active_E_fraction_peak=act.get("active_E_fraction_peak"),
            core_E_rate_mean_hz=act.get("core_E_rate_mean_hz"),
            surround_E_rate_mean_hz=act.get("surround_E_rate_mean_hz"),
            collision_rate_returned_sidecar=sc.get("collision_rate"), ambiguous_rate=amb_rate,
            n_sidecar_events=n_sidecar, n_total_events=s["n_events"],
            return_to_baseline_fraction=ret_frac, nf_frac=nf_frac, r95_mm=r95,
            direction_balance=(min(nf, nr) / max(nf, nr) if max(nf, nr) else None)))
    return rows


def _state(r):
    """Descriptive 4-state label from the indicators (raw metrics reported alongside). Thresholds
    calibrated to the A1b grid (2026-06-25): runaway = doesn't return / sustained core firing;
    silent = near-zero GLOBAL rate (the tiny-blip 'events' are not real); seizure-like = elevated
    two-foci collision on LARGE events that still return; else interictal axial self-limited."""
    gr = r["global_E_rate_mean_hz"] or 0; coreR = r["core_E_rate_mean_hz"] or 0
    coll = r["collision_rate_returned_sidecar"]; ret = r["return_to_baseline_fraction"]
    tonic = r["tonic_fraction"] or 0; r95 = r["r95_mm"] or 0
    if (ret is not None and ret < 0.4) or tonic > 0.4 or coreR > 100.0:
        return "runaway"                                   # non-returning / sustained core firing
    if gr < 0.3:
        return "silent"                                    # protective: near-zero global E rate
    if (coll is not None and coll > 0.3) and r95 > 8.0:
        return "seizure_like"                              # elevated foci synchronization, large, returns
    return "interictal_like"                               # axial self-limited


def _agg(rows):
    def m(k):
        v = [r[k] for r in rows if r[k] is not None]
        return round(float(np.mean(v)), 4) if v else None
    from collections import Counter
    states = [_state(r) for r in rows]
    keys = ["event_rate_hz", "global_E_rate_mean_hz", "global_E_rate_p95_hz", "tonic_fraction",
            "active_E_fraction_peak", "core_E_rate_mean_hz", "surround_E_rate_mean_hz",
            "collision_rate_returned_sidecar", "ambiguous_rate", "return_to_baseline_fraction",
            "nf_frac", "r95_mm", "direction_balance", "local_global_ratio"]
    out = {k: m(k) for k in keys}
    out.update(n_seeds=len(rows), state=Counter(states).most_common(1)[0][0], states=states)
    return out


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    base = os.path.join(ROOT, argv[0]) if argv else os.path.join(ROOT, DEFAULT_DIR)
    rows = _load(base)
    if not rows:
        print(f"[a1b] no readout in {base}"); return 1
    grid = [r for r in rows if not r["is_posctrl"]]
    pos = [r for r in rows if r["is_posctrl"]]
    cells = {}
    for r in grid:
        cells.setdefault((r["local"], r["glob"]), []).append(r)
    agg = {f"l{lo}_g{g}": _agg(v) for (lo, g), v in cells.items()}
    # Missing grid cells = the run timed out (the most-excitable corner self-sustains and never
    # finishes) -> mark RUNAWAY explicitly so the state surface shows it, not a blank.
    for lo in (0, 1, 2):
        for g in (0.7, 1.0, 1.3, 1.6):
            agg.setdefault(f"l{lo}_g{g}", {"state": "runaway", "n_seeds": 0,
                                           "note": "timeout (self-sustaining / runaway)"})
    posagg = _agg(pos) if pos else None
    status = {"base": os.path.relpath(base, ROOT), "n_runs": len(rows),
              "grid": agg, "posctrl_egaba16": posagg,
              "note": ("A1b = state topography map (NOT dynamic causal chain). collision renamed "
                       "collision_rate_returned_sidecar. local_global_ratio is a model coordinate, "
                       "not a physiological quantity.")}
    json.dump(status, open(os.path.join(base, "status_a1b.json"), "w"), indent=1)

    # ---- figures ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir = os.path.join(base, "figures"); os.makedirs(fig_dir, exist_ok=True)
    locals_, globals_ = [0, 1, 2], [0.7, 1.0, 1.3, 1.6]
    STATE_C = {"silent": 0, "interictal_like": 1, "seizure_like": 2, "runaway": 3}
    STATE_CMAP = plt.matplotlib.colors.ListedColormap(["#cfe8ff", "#9ad27f", "#e8a33d", "#c0392b"])

    def cell(lo, g, k):
        a = agg.get(f"l{lo}_g{g}")
        return (a.get(k) if a else None)

    # Fig 1: 2D local x global heatmaps of the 6 core indicators + the state class.
    panels = [("state", "state class"), ("event_rate_hz", "event rate (Hz)"),
              ("global_E_rate_mean_hz", "global E rate (Hz)"), ("tonic_fraction", "tonic fraction"),
              ("collision_rate_returned_sidecar", "collision (sidecar, returned)"),
              ("return_to_baseline_fraction", "return fraction"), ("r95_mm", "spatial r95 (mm)"),
              ("active_E_fraction_peak", "peak active frac")]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax, (k, title) in zip(axes.ravel(), panels):
        M = np.full((len(locals_), len(globals_)), np.nan)
        for i, lo in enumerate(locals_):
            for j, g in enumerate(globals_):
                a = agg.get(f"l{lo}_g{g}")
                if a:
                    M[i, j] = STATE_C.get(a["state"], np.nan) if k == "state" else (a.get(k) if a.get(k) is not None else np.nan)
        im = ax.imshow(M, origin="lower", aspect="auto",
                       cmap=(STATE_CMAP if k == "state" else "viridis"),
                       vmin=(0 if k == "state" else None), vmax=(3 if k == "state" else None))
        ax.set_xticks(range(len(globals_))); ax.set_xticklabels(globals_)
        ax.set_yticks(range(len(locals_))); ax.set_yticklabels([f"local{l}" for l in locals_])
        ax.set_xlabel("global_ei_scale (restraint →)"); ax.set_ylabel("local loop ↑")
        ax.set_title(title, fontsize=10)
        for i in range(len(locals_)):
            for j in range(len(globals_)):
                if not np.isnan(M[i, j]):
                    txt = (["silent", "inter", "seiz", "run"][int(M[i, j])] if k == "state" else f"{M[i,j]:.2f}")
                    ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                            color=("white" if k == "state" and M[i, j] >= 2 else "black"))
        if k != "state":
            fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("A1b state topography — local-loop × global-restraint on Stage-3 two-focus core "
                 f"(seeds {sorted(set(r['seed'] for r in grid))}; e_GABA16 posctrl state="
                 f"{posagg['state'] if posagg else 'NA'})", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "a1b_state_surface.png"), dpi=130); plt.close(fig)

    print("[a1b] grid states:")
    for lo in locals_:
        print("  local%d: " % lo + " | ".join(
            f"g{g}={(agg.get(f'l{lo}_g{g}') or {}).get('state','NA')}" for g in globals_))
    print(f"[a1b] posctrl e_GABA16 state: {posagg['state'] if posagg else 'NA'}")
    print(f"[a1b] wrote {base}/status_a1b.json + figures/a1b_state_surface.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
