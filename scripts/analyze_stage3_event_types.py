"""Stage 3 event-typing — classify every spontaneous event in the two-foci one-network runs, to
explore the LOCAL-to-GLOBAL event hierarchy: why most events stay local (strong nucleation but
contained / relay-failure), some become readable global propagation templates, and others co-ignite.
NOT a pass/fail gate — exploratory mechanism characterization (user 2026-06-14, review-corrected).

Per event (from sidecar_*.json; cores DID fire even when the read-out can't see it):
  source (ground-truth nucleation, from core onsets, delta_onset=30ms):
    both : both cores crossed within delta            (co-ignition at the source)
    neg  : neg crossed and (pos None or neg+delta<pos)
    pos  : symmetric
    none : neither core crossed (no detected ignition)
  readout            : sign, n_part (contacts lit, of 12), axis_err
  propagation_class:
    collision              : source == both
    readable_global        : n_part>=7 AND axis readable AND source in {neg,pos}
    readable_unknown_source: n_part>=7 AND axis readable AND source == none  (clean readout, no detected core)
    local                  : otherwise (nucleated but did NOT spread to >=7 contacts)
  source_core_ignite_frac  : max active-fraction the SOURCE core reached (energy of the nucleating core)
  other_core_ignite_frac   : max active-fraction the OTHER core reached
  core_ignite_asymmetry    : source - other  (HIGH on a local => one core fired hard, other quiet =>
                             a strong ONE-SIDED nucleation that did not spread = CONTAINED, not weak)
  time_since_prev, previous_source, prev_global, within_recovery (prev_global & gap<RECOVERY_MS),
  core_onset_diff, size=n_part, duration=t_off-t_on

NOTE wording (review 2026-06-14): the evidence supports CONTAINED propagation / relay-failure (core
fires hard, activity does not spread to enough contacts). It does NOT (yet) prove "truncation" in the
sense of a wavefront that starts outward then stops — that needs a spatiotemporal snapshot (next step).
Source balance is reported PER CELL, never as a single pooled "balanced regime".
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
RECOVERY_MS = 250.0
READABLE_NPART = 7


def classify_source(on_neg, on_pos):
    has_n, has_p = on_neg is not None, on_pos is not None
    if has_n and has_p:
        return "both" if abs(on_neg - on_pos) <= DELTA else ("neg" if on_neg < on_pos else "pos")
    if has_n:
        return "neg"
    if has_p:
        return "pos"
    return "none"


def core_max(ev, side):
    fr = ev.get(f"core_frac_{side}_first_bins") or []
    return round(max(fr), 4) if fr else 0.0


def source_other_frac(ev, source):
    neg_f, pos_f = core_max(ev, "neg"), core_max(ev, "pos")
    if source == "neg":
        return neg_f, pos_f
    if source == "pos":
        return pos_f, neg_f
    # both / none: no single source core -> report stronger as 'source', weaker as 'other' (asym~0 expected)
    return max(neg_f, pos_f), min(neg_f, pos_f)


def prop_class(source, n_part, axis_err):
    if source == "both":
        return "collision"
    readable = (n_part or 0) >= READABLE_NPART and axis_err is not None
    if readable:
        return "readable_global" if source in ("neg", "pos") else "readable_unknown_source"
    return "local"


def cell_of(tag):
    m = re.search(r"sep([\d.]+)_std([\d.]+)_m([\d.]+)", tag)
    if m:
        return f"sep{m.group(1)}/std{m.group(2)}/m{m.group(3)}"
    return "conf:sep0.7/std1.0/m17.5" if tag.startswith("conf_") else tag


def seed_of(tag):
    m = re.search(r"_s(\d+)$", tag)
    return int(m.group(1)) if m else -1


def build_rows():
    rows = []
    for sc in sorted(glob.glob(os.path.join(ROOT, "**", "sidecar_*.json"), recursive=True)):
        tag = re.sub(r".*sidecar_(.*)\.json", r"\1", os.path.basename(sc))
        if not (tag.startswith("conf_") or "gs_te_" in tag):
            continue
        cell, seed = cell_of(tag), seed_of(tag)
        ev = sorted(json.load(open(sc)).get("events", []), key=lambda e: e["t_on"])
        prev = None
        for e in ev:
            src = classify_source(e.get("core_onset_neg"), e.get("core_onset_pos"))
            pc = prop_class(src, e.get("n_part"), e.get("axis_err"))
            sf, of = source_other_frac(e, src)
            tsp = None if prev is None else round(e["t_on"] - prev["t_on"], 1)
            prev_global = bool(prev is not None and prev["_pc"] in ("readable_global", "readable_unknown_source"))
            within_rec = bool(prev_global and tsp is not None and tsp < RECOVERY_MS)
            od = (None if e.get("core_onset_neg") is None or e.get("core_onset_pos") is None
                  else round(abs(e["core_onset_neg"] - e["core_onset_pos"]), 1))
            rows.append(dict(cell=cell, seed=seed, tag=tag, event_id=e["event_id"], t_on=e["t_on"],
                             source=src, sign=e.get("sign"), n_part=e.get("n_part"),
                             axis_err=e.get("axis_err"), propagation_class=pc,
                             source_core_ignite_frac=sf, other_core_ignite_frac=of,
                             core_ignite_asymmetry=round(sf - of, 4),
                             time_since_prev=tsp,
                             previous_source=(None if prev is None else prev["_src"]),
                             prev_global=prev_global, within_recovery=within_rec,
                             core_onset_diff=od, duration=round(e["t_off"] - e["t_on"], 1)))
            e["_src"], e["_pc"] = src, pc
            prev = e
    return rows


def _wcsv(path, rows, fields):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def by_group(rows, keyfn):
    out = []
    groups = defaultdict(list)
    for r in rows:
        groups[keyfn(r)].append(r)
    for key in sorted(groups, key=str):
        g = groups[key]
        pc = Counter(r["propagation_class"] for r in g)
        sc = Counter(r["source"] for r in g)
        neg = [r for r in g if r["source"] == "neg"]
        pos = [r for r in g if r["source"] == "pos"]
        pgn = sum(1 for r in neg if r["propagation_class"] == "readable_global") / len(neg) if neg else None
        pgp = sum(1 for r in pos if r["propagation_class"] == "readable_global") / len(pos) if pos else None
        out.append(dict(group=key, n=len(g),
                        local=pc.get("local", 0), readable_global=pc.get("readable_global", 0),
                        collision=pc.get("collision", 0),
                        readable_unknown_source=pc.get("readable_unknown_source", 0),
                        neg=sc.get("neg", 0), pos=sc.get("pos", 0), both=sc.get("both", 0), none=sc.get("none", 0),
                        P_global_given_neg=(None if pgn is None else round(pgn, 3)),
                        P_global_given_pos=(None if pgp is None else round(pgp, 3))))
    return out


def main():
    rows = build_rows()
    fields = list(rows[0].keys())
    _wcsv(os.path.join(ROOT, "event_types.csv"), rows, fields)

    cellrows = by_group(rows, lambda r: r["cell"])
    seedrows = by_group(rows, lambda r: f"{r['cell']}|s{r['seed']}")
    gfields = list(cellrows[0].keys())
    _wcsv(os.path.join(ROOT, "event_type_by_cell.csv"), cellrows, gfields)
    _wcsv(os.path.join(ROOT, "event_type_by_seed.csv"), seedrows, gfields)
    rg = [r for r in rows if r["propagation_class"] == "readable_global"]
    _wcsv(os.path.join(ROOT, "readable_global_events.csv"), rg, fields)

    print(f"wrote event_types.csv ({len(rows)}), event_type_by_cell.csv ({len(cellrows)}), "
          f"event_type_by_seed.csv ({len(seedrows)}), readable_global_events.csv ({len(rg)})")
    pc = Counter(r["propagation_class"] for r in rows)
    print("\npooled propagation_class:", dict(pc), " (POOLED across 19 cells — per-cell varies, see by_cell csv)")
    print("\n--- source-core vs other-core ignition (local events, source in {neg,pos}) ---")
    locns = [r for r in rows if r["propagation_class"] == "local" and r["source"] in ("neg", "pos")]
    if locns:
        print(f"  n={len(locns)}  source_core_frac median={st.median([r['source_core_ignite_frac'] for r in locns]):.3f}  "
              f"other_core_frac median={st.median([r['other_core_ignite_frac'] for r in locns]):.3f}  "
              f"asymmetry median={st.median([r['core_ignite_asymmetry'] for r in locns]):.3f}")
        print("  => high source_core + high asymmetry on local events = ONE-SIDED strong ignition that did "
              "NOT spread (contained / relay-failure), not weak nucleation.")
    print("\n--- per-cell snapshot (full table in event_type_by_cell.csv) ---")
    for c in cellrows:
        print(f"  {c['group']:>26}: n={c['n']:>3} local{c['local']} glob{c['readable_global']} coll{c['collision']} "
              f"unk{c['readable_unknown_source']} | src n{c['neg']}/p{c['pos']}/b{c['both']}/0{c['none']} | "
              f"P(glob|neg)={c['P_global_given_neg']} P(glob|pos)={c['P_global_given_pos']}")


if __name__ == "__main__":
    main()
