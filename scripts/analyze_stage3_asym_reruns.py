"""Stage 3 source-asymmetry re-run analysis (2026-06-15). Reads the battery outputs
(asym_reruns/{readout,sidecar,fullfield}_{base,swap,mirror}_s*.json) and answers the 3 probes:

  SWAP   : per seed, baseline winner end (hidden neg_clean vs pos_clean) vs swap winner end.
           swap moves each core's threshold DRAW to the other end. If the winner FLIPS end under
           swap => the per-network winner is threshold-DRAW-driven (per-run luck), NOT a fixed
           neg/pos structural/geometric bias. Low flip rate => position-driven (geom/connectivity).
  MIRROR : identical threshold profile on both cores. If a source imbalance / direction bias
           PERSISTS (consistently same end) => geometry/connectivity/read-out artifact, not
           threshold heterogeneity. If it balances => threshold draw was the source.
  FULLFIELD : per-event FULL-neuron-field spread (std of fired-E positions) + duration + n_fired_E,
           local (n_part<7) vs global (n_part>=7), and split by read-out sign (source proxy) — does
           the duration/spread story (and the neg/pos asymmetry) hold on the real field, not just
           the 12 virtual contacts?
"""
import os, glob, json
import numpy as np

ROOT = "results/topic4_sef_hfo/observation_layer/snn_cm_spontaneous/asym_reruns"
PART_MIN = 7


def load(cond, seed):
    p = os.path.join(ROOT, f"readout_{cond}_s{seed}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def winner(sc):
    if not sc:
        return None
    n, p = sc.get("neg_clean", 0), sc.get("pos_clean", 0)
    return "tie" if n == p else ("neg" if n > p else "pos")


def seeds_present():
    ss = set()
    for f in glob.glob(os.path.join(ROOT, "readout_base_s*.json")):
        ss.add(int(f.split("_s")[-1].split(".")[0]))
    return sorted(ss)


def swap_mirror_table():
    rows, flips, mirror_imbalance = [], [], []
    for s in seeds_present():
        rb, rs, rm = load("base", s), load("swap", s), load("mirror", s)
        scb = rb.get("stage3_source_counts") if rb else None
        scs = rs.get("stage3_source_counts") if rs else None
        scm = rm.get("stage3_source_counts") if rm else None
        wb, ws, wm = winner(scb), winner(scs), winner(scm)
        flip = (wb in ("neg", "pos") and ws in ("neg", "pos") and wb != ws)
        if wb in ("neg", "pos") and ws in ("neg", "pos"):
            flips.append(flip)
        if scm:
            mirror_imbalance.append(abs(scm.get("neg_clean", 0) - scm.get("pos_clean", 0)))
        rows.append(dict(seed=s,
                         base=f"n{scb['neg_clean']}/p{scb['pos_clean']}(win {wb})" if scb else None,
                         swap=f"n{scs['neg_clean']}/p{scs['pos_clean']}(win {ws})" if scs else None,
                         mirror=f"n{scm['neg_clean']}/p{scm['pos_clean']}(win {wm})" if scm else None,
                         winner_flipped_under_swap=flip))
    MIN_JUDGEABLE = 4   # minimum power gate: fewer judgeable seeds => no causal verdict (P1 2026-06-15)
    if len(flips) < MIN_JUDGEABLE:
        sv = (f"UNDERPOWERED / inconclusive (n_seeds_judgeable={len(flips)} < {MIN_JUDGEABLE}; "
              "cold low-collision cell yields too few clean single-source events per run)")
    elif sum(flips) >= 0.6 * len(flips):
        sv = "threshold-DRAW-driven (winner flips with the draw => per-run luck)"
    elif sum(flips) <= 0.4 * len(flips):
        sv = "position-driven (winner stays => geometry/connectivity/readout)"
    else:
        sv = "MIXED / inconclusive"
    verdict = dict(
        n_seeds_judgeable=len(flips),
        swap_flip_rate=(round(sum(flips) / len(flips), 3) if flips else None),
        swap_verdict=sv,
        mirror_residual_imbalance_median=(round(float(np.median(mirror_imbalance)), 2)
                                          if mirror_imbalance else None),
        mirror_note="if residual imbalance ~0 across seeds => threshold heterogeneity WAS the source; "
                    "if it persists same-end => geometry/connectivity/readout artifact")
    return rows, verdict


def _hidden_source_by_ton(seed):
    """sidecar hidden_source_label + clean_for_timing keyed by rounded t_on (base condition).
    The read-out SIGN is unreliable on the pos end (Stage 3), so the source split MUST use the
    core-onset hidden label, NOT sign (P1 2026-06-15)."""
    p = os.path.join(ROOT, f"sidecar_base_s{seed}.json")
    if not os.path.exists(p):
        return {}
    return {round(e["t_on"], 1): (e.get("hidden_source_label"), bool(e.get("clean_for_timing")))
            for e in json.load(open(p)).get("events", [])}


def fullfield_table():
    """Pool baseline full-field events; local vs global overall, then CLEAN GLOBAL split by HIDDEN
    SOURCE (core-onset label joined from the sidecar by t_on), NOT read-out sign."""
    evs = []
    for f in glob.glob(os.path.join(ROOT, "fullfield_base_s*.json")):
        seed = int(f.split("_s")[-1].split(".")[0])
        hid = _hidden_source_by_ton(seed)
        for e in json.load(open(f)).get("events", []):
            lab, clean = hid.get(round(e["t_on"], 1), (None, False))
            evs.append({**e, "hidden_source": lab, "clean_for_timing": clean})
    if not evs:
        return {"note": "no fullfield_base_*.json yet"}
    def med(xs, k):
        v = [e[k] for e in xs if e.get(k) is not None]
        return round(float(np.median(v)), 3) if v else None
    def block(sub):
        return dict(n=len(sub), fullfield_extent_mm=med(sub, "fullfield_extent_mm"),
                    duration=med(sub, "duration"), n_fired_E=med(sub, "n_fired_E"),
                    n_part=med(sub, "n_part"))
    out = {"n_events": len(evs),
           "local": block([e for e in evs if (e.get("n_part") or 0) < PART_MIN]),
           "global": block([e for e in evs if (e.get("n_part") or 0) >= PART_MIN])}
    # CLEAN GLOBAL split by HIDDEN source (the load-bearing comparison)
    cg = [e for e in evs if (e.get("n_part") or 0) >= PART_MIN and e["clean_for_timing"]
          and e["hidden_source"] in ("neg", "pos")]
    out["clean_global_n"] = len(cg)
    for lab in ("neg", "pos"):
        out[f"clean_global_{lab}_src"] = block([e for e in cg if e["hidden_source"] == lab])
    return out


def main():
    rows, verdict = swap_mirror_table()
    ff = fullfield_table()
    summary = dict(swap_mirror_per_seed=rows, swap_mirror_verdict=verdict, fullfield=ff)
    json.dump(summary, open(os.path.join(ROOT, "asym_reruns_summary.json"), "w"), indent=2,
              default=lambda o: None)

    print("=== SWAP / MIRROR per seed (winner = hidden neg_clean vs pos_clean) ===")
    for r in rows:
        print(f"  seed {r['seed']}: base {r['base']} | swap {r['swap']} | mirror {r['mirror']} "
              f"| flip={r['winner_flipped_under_swap']}")
    print("\nVERDICT:", json.dumps(verdict, indent=2, ensure_ascii=False))
    print("\n=== FULL-FIELD local vs global (baseline runs) ===")
    print(json.dumps(ff, indent=2, ensure_ascii=False))
    print(f"\nwrote {os.path.join(ROOT, 'asym_reruns_summary.json')}")


if __name__ == "__main__":
    main()
