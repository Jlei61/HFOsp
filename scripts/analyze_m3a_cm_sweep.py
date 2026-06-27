#!/usr/bin/env python3
"""Aggregate the M3A slow-variable sweep on the Stage-3 two-focus core (cm-spontaneous readout) into
an HONEST phenotype-vs-slow-level summary + figure. OFFLINE — reads
results/topic4_sef_hfo/m3a_slowvars/cm_sweep/readout_*.json.

Key discriminators (the Stage-3 two-focus core lets us read the INTERICTAL→synchronized transition):
  - collision_rate : fraction of events where the two foci co-ignite within delta_onset (30ms). LOW =
    two INDEPENDENT alternating foci (interictal); HIGH = the foci SYNCHRONIZE / co-fire (seizure-like).
  - direction balance: n_clean_forward vs n_clean_reverse. Balanced = two distinct templates; collapsed
    to one sign = unidirectional / merged.
  - event_rate (events/s) and activity bar.
A slow knob that moves collision_rate / direction-collapse (not just rate) is a real ≥2-state transition.
Sparse (per-combo events ~5-30, seeds 1-3) -> trends, not statistics.
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DIR = "results/topic4_sef_hfo/m3a_slowvars/cm_sweep"
_MORE_EXC_LOWER = {"z": True, "phi": True, "gK": True, "egaba": False}


def _load(base):
    out = []
    for f in glob.glob(os.path.join(base, "readout_*.json")):
        s = json.load(open(f)); c = s["config"]
        sc = s.get("stage3_source_counts") or {}
        if c.get("shunt_gaba"):
            mech, level = "egaba", float(c.get("e_gaba"))
        elif c.get("slow_var", "none") != "none":
            mech, level = c["slow_var"], float(c["slow_level"])
        else:
            mech, level = "off", None
        nf, nr = s["n_clean_forward"], s["n_clean_reverse"]
        out.append(dict(
            tag=s["tag"], mech=mech, level=level, seed=int(c["seed"]),
            g_gaba=c.get("g_gaba_scale"), n_events=s["n_events"],
            event_rate_hz=round(s["n_events"] / (c["T"] / 1000.0), 4),
            n_fwd=nf, n_rev=nr, dir_balance=(min(nf, nr) / max(nf, nr) if max(nf, nr) else None),
            # collision_rate_returned_sidecar = sidecar (stage3_source_counts) fraction of RETURNED
            # events where the two foci co-ignite within delta_onset (30ms). Reported with the
            # ambiguous rate + sidecar/total event counts so a low n_sidecar isn't read as "clean".
            collision_rate_returned_sidecar=sc.get("collision_rate"), neg_clean=sc.get("neg_clean"),
            pos_clean=sc.get("pos_clean"), bar=s["detector"]["bar"],
            ambiguous_rate=((sc.get("ambiguous", 0) / max(sum(sc.get(k, 0) for k in
                            ("neg_clean", "pos_clean", "collision", "ambiguous")), 1)) if sc else None),
            n_sidecar_events=(sum(sc.get(k, 0) for k in ("neg_clean", "pos_clean", "collision", "ambiguous"))
                              if sc else 0),
            n_total_events=s["n_events"],
            true_floor=s["detector"].get("true_inter_event_floor")))
    return out


def _agg_seed(rows):
    """mean over seeds for a (mech, level, g_gaba) cell."""
    def m(k):
        v = [r[k] for r in rows if r[k] is not None]
        return round(float(np.mean(v)), 4) if v else None
    r0 = rows[0]
    return dict(mech=r0["mech"], level=r0["level"], g_gaba=r0["g_gaba"], n_seeds=len(rows),
                event_rate_hz=m("event_rate_hz"),
                collision_rate_returned_sidecar=m("collision_rate_returned_sidecar"),
                ambiguous_rate=m("ambiguous_rate"), n_sidecar_events=m("n_sidecar_events"),
                n_total_events=m("n_total_events"),
                dir_balance=m("dir_balance"), n_events=m("n_events"), bar=m("bar"),
                n_fwd=m("n_fwd"), n_rev=m("n_rev"))


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    base = os.path.join(ROOT, argv[0]) if argv else os.path.join(ROOT, DEFAULT_DIR)
    rows = _load(base)
    if not rows:
        print(f"[m3a analyze] no readout_*.json in {base}"); return 1
    # group by (mech, level, g_gaba) over seeds
    cells = {}
    for r in rows:
        cells.setdefault((r["mech"], r["level"], r["g_gaba"]), []).append(r)
    agg = [_agg_seed(v) for v in cells.values()]
    off = next((a for a in agg if a["mech"] == "off"), None)

    by_mech = {}
    for a in agg:
        if a["mech"] == "off":
            continue
        by_mech.setdefault(a["mech"], []).append(a)
    for mech, lst in by_mech.items():
        # egaba: split by g_gaba; sort least->most excitable
        lst.sort(key=lambda a: (a["g_gaba"] or 0, a["level"]),
                 reverse=_MORE_EXC_LOWER.get(mech, False))

    status = {"base": os.path.relpath(base, ROOT), "off_baseline": off,
              "by_mechanism": by_mech, "n_runs": len(rows)}
    json.dump(status, open(os.path.join(base, "status_m3a_cm_sweep.json"), "w"), indent=1)

    # ---- figure: per-mechanism collision_rate / event_rate / direction-balance vs level ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir = os.path.join(base, "figures"); os.makedirs(fig_dir, exist_ok=True)
    mechs = [m for m in ("z", "egaba", "phi", "gK") if m in by_mech]
    fig, axes = plt.subplots(len(mechs), 3, figsize=(14, 3.3 * len(mechs)), squeeze=False)
    for i, mech in enumerate(mechs):
        lst = by_mech[mech]
        # for egaba, separate g lines
        groups = {}
        for a in lst:
            groups.setdefault(a["g_gaba"] if mech == "egaba" else "_", []).append(a)
        for gk, gl in groups.items():
            gl.sort(key=lambda a: a["level"], reverse=_MORE_EXC_LOWER.get(mech, False))
            x = [a["level"] for a in gl]
            lab = (f"g={gk}" if mech == "egaba" else None)
            axes[i][0].plot(x, [a["collision_rate_returned_sidecar"] for a in gl], "o-", label=lab)
            axes[i][1].plot(x, [a["event_rate_hz"] for a in gl], "o-", label=lab)
            axes[i][2].plot(x, [(a["dir_balance"] if a["dir_balance"] is not None else np.nan) for a in gl], "o-", label=lab)
        if off:
            for j, key in enumerate(("collision_rate_returned_sidecar", "event_rate_hz", "dir_balance")):
                if off[key] is not None:
                    axes[i][j].axhline(off[key], ls="--", color="0.6", lw=1, label="off")
        axes[i][0].set_ylabel(f"{mech}\n(← more exc)" if _MORE_EXC_LOWER.get(mech) else f"{mech}\n(more exc →)")
        axes[i][0].set_title("collision_rate_returned_sidecar (foci co-ignite)"); axes[i][0].set_ylim(-0.05, 1.05)
        axes[i][1].set_title("event_rate (Hz)")
        axes[i][2].set_title("direction balance min/max(fwd,rev)"); axes[i][2].set_ylim(-0.05, 1.05)
        if mech == "egaba":
            for a in axes[i]:
                a.legend(fontsize=7)
    fig.suptitle("M3A slow-variable sweep on Stage-3 two-focus core — interictal(独立交替) → synchronized(collision) transition",
                 fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "m3a_sweep_summary.png"), dpi=130); plt.close(fig)

    print("[m3a analyze] off baseline:", off)
    for mech, lst in by_mech.items():
        cr = [(a["level"], a["collision_rate_returned_sidecar"], a["event_rate_hz"], a["dir_balance"]) for a in lst]
        print(f"[m3a analyze] {mech}: (level, collision_returned_sidecar, rate, dir_bal) = {cr}")
    print(f"[m3a analyze] wrote {base}/status_m3a_cm_sweep.json + figures/m3a_sweep_summary.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
