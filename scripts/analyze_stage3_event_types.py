"""Stage 3 event-typing table — classify every spontaneous event in the two-foci one-network runs,
to decide WHY most events stay local: weak (low-energy) nucleation vs refractory/collision truncation
of propagation. NOT a pass/fail gate — an exploratory mechanism characterization (user 2026-06-14).

Per event (from sidecar_*.json; cores DID fire even when the read-out can't see it):
  source (ground-truth nucleation, from core onsets, delta_onset=30ms):
    both : both cores crossed within delta            (co-ignition at the source)
    neg  : neg crossed and (pos None or neg+delta<pos)
    pos  : symmetric
    none : neither core crossed (no detected ignition)
  readout         : sign (+1 fwd / -1 rev / None), n_part (contacts lit, of 12), axis_err
  propagation_class:
    collision       : source == both
    readable_global : n_part>=7 AND axis readable (spread far enough for a clean direction)
    local           : otherwise (nucleated but did NOT spread to >=7 contacts)
  core_ignite_frac : max active-fraction the source core reached in its first bins  <-- ENERGY proxy:
                     HIGH on a local event => strong nucleation but contained (truncation);
                     LOW => weak nucleation. THIS is the low-energy-vs-truncation discriminator.
  time_since_prev  : t_on - prev t_on (within the run)
  previous_source  : source of the previous event
  prev_global      : previous event was readable_global
  within_recovery  : prev_global AND time_since_prev < RECOVERY_MS  (refractory-shadow flag)
  core_onset_diff  : |neg_onset - pos_onset| (None if a core didn't cross)
  size=n_part, duration=t_off-t_on
"""
import os
import re
import csv
import json
import glob
import argparse
import statistics as st
from collections import Counter, defaultdict

ROOT = "results/topic4_sef_hfo/observation_layer/snn_cm_spontaneous"
DELTA = 30.0
RECOVERY_MS = 250.0     # refractory-shadow window after a big (readable_global) event (reported, tunable)
READABLE_NPART = 7      # endpoint-centroid axis needs >= 2*k_dir+1 = 7 participating contacts


def classify_source(on_neg, on_pos):
    has_n, has_p = on_neg is not None, on_pos is not None
    if has_n and has_p:
        return "both" if abs(on_neg - on_pos) <= DELTA else ("neg" if on_neg < on_pos else "pos")
    if has_n:
        return "neg"
    if has_p:
        return "pos"
    return "none"


def core_ignite_frac(ev):
    fr = (ev.get("core_frac_neg_first_bins") or []) + (ev.get("core_frac_pos_first_bins") or [])
    return round(max(fr), 4) if fr else None


def prop_class(source, n_part, axis_err):
    if source == "both":
        return "collision"
    if (n_part or 0) >= READABLE_NPART and axis_err is not None and axis_err < 25:
        return "readable_global"
    return "local"


def cell_of(tag):
    m = re.search(r"sep([\d.]+)_std([\d.]+)_m([\d.]+)", tag)
    if m:
        return f"sep{m.group(1)}/std{m.group(2)}/m{m.group(3)}"
    return "conf:sep0.7/std1.0/m17.5" if tag.startswith("conf_") else tag


def build_rows():
    rows = []
    sides = sorted(glob.glob(os.path.join(ROOT, "**", "sidecar_*.json"), recursive=True))
    for sc in sides:
        tag = re.sub(r".*sidecar_(.*)\.json", r"\1", os.path.basename(sc))
        if not (tag.startswith("conf_") or "gs_te_" in tag):    # two-foci runs only
            continue
        cell = cell_of(tag)
        ev = json.load(open(sc)).get("events", [])
        ev = sorted(ev, key=lambda e: e["t_on"])
        prev = None
        for e in ev:
            src = classify_source(e.get("core_onset_neg"), e.get("core_onset_pos"))
            pc = prop_class(src, e.get("n_part"), e.get("axis_err"))
            tsp = None if prev is None else round(e["t_on"] - prev["t_on"], 1)
            prev_src = None if prev is None else prev["_src"]
            prev_global = bool(prev is not None and prev["_pc"] == "readable_global")
            within_rec = bool(prev_global and tsp is not None and tsp < RECOVERY_MS)
            od = (None if e.get("core_onset_neg") is None or e.get("core_onset_pos") is None
                  else round(abs(e["core_onset_neg"] - e["core_onset_pos"]), 1))
            rows.append(dict(cell=cell, tag=tag, event_id=e["event_id"], t_on=e["t_on"],
                             source=src, sign=e.get("sign"), n_part=e.get("n_part"),
                             axis_err=e.get("axis_err"), propagation_class=pc,
                             core_ignite_frac=core_ignite_frac(e),
                             time_since_prev=tsp, previous_source=prev_src,
                             prev_global=prev_global, within_recovery=within_rec,
                             core_onset_diff=od, duration=round(e["t_off"] - e["t_on"], 1)))
            e["_src"], e["_pc"] = src, pc
            prev = e
    return rows


def summarize(rows):
    print(f"\n=== {len(rows)} events across {len(set(r['tag'] for r in rows))} two-foci runs ===")
    pc = Counter(r["propagation_class"] for r in rows)
    print("propagation_class:", dict(pc))
    src = Counter(r["source"] for r in rows)
    print("source:", dict(src))

    def fr(cls):
        v = [r["core_ignite_frac"] for r in rows if r["propagation_class"] == cls and r["core_ignite_frac"] is not None]
        return v
    print("\n--- DISCRIMINATOR: core ignition strength by class (energy proxy) ---")
    for cls in ("local", "readable_global", "collision"):
        v = fr(cls)
        if v:
            print(f"  {cls:>16}: n={len(v):>3}  core_ignite_frac median={st.median(v):.3f}  "
                  f"[{min(v):.2f},{max(v):.2f}]  frac>=0.10: {sum(x>=0.10 for x in v)/len(v):.0%}")
    print("  => if LOCAL events have HIGH core_ignite_frac ~ readable_global's, locality = "
          "CONTAINED propagation (truncation), NOT weak nucleation.")

    loc = [r for r in rows if r["propagation_class"] == "local"]
    print("\n--- DISCRIMINATOR: do local events follow a big event (refractory shadow)? ---")
    if loc:
        nrec = sum(1 for r in loc if r["within_recovery"])
        print(f"  local events within {RECOVERY_MS:.0f}ms after a readable_global: {nrec}/{len(loc)} = {nrec/len(loc):.0%}")
        haveprev = [r for r in loc if r["time_since_prev"] is not None]
        if haveprev:
            print(f"  local time_since_prev median={st.median([r['time_since_prev'] for r in haveprev]):.0f}ms")
        glob_ev = [r for r in rows if r["propagation_class"] == "readable_global" and r["time_since_prev"] is not None]
        if glob_ev:
            print(f"  global time_since_prev median={st.median([r['time_since_prev'] for r in glob_ev]):.0f}ms")
    print("\n--- source of local events (did a core actually fire?) ---")
    print("  local source:", dict(Counter(r["source"] for r in loc)))
    print("  => local with source in {neg,pos} = a core fired but didn't spread (nucleated-then-contained).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "event_types.csv"))
    a = ap.parse_args()
    rows = build_rows()
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {a.out} ({len(rows)} rows)")
    summarize(rows)


if __name__ == "__main__":
    main()
